use axum::{
    Json,
    extract::{Path as AxumPath, Query, State},
    http::StatusCode,
};
use dataset_packer::macroxue::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::app::{AppState, PolicyKindLegend};
use crate::student::StudentBinsRecord;

#[derive(Clone, Serialize)]
struct AnnotationPayload {
    policy_kind_mask: u8,
    argmax_head: u8,
    argmax_prob: f32,
    policy_p1: [f32; 4],
    policy_logp: [f32; 4],
    policy_hard: [f32; 4],
}

#[derive(Clone, Serialize)]
struct StudentBinsPayload {
    num_bins: usize,
    heads: [Vec<f32>; 4],
}

#[derive(Clone, Serialize)]
struct StepResponse {
    step_index: u32,
    board: [u8; 16],
    board_value: f32,
    branch_evs: Vec<Option<f32>>,
    relative_branch_evs: Vec<Option<i32>>,
    advantage_branch: Vec<Option<i32>>,
    legal_mask: u8,
    teacher_move: u8,
    is_disagreement: bool,
    valuation_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    annotation: Option<AnnotationPayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    student_bins: Option<StudentBinsPayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens: Option<Vec<u16>>,
}

#[derive(Serialize)]
struct Pagination {
    offset: usize,
    limit: usize,
    total: usize,
}

#[derive(Serialize)]
struct RunSummaryResponse {
    run_id: u32,
    seed: u64,
    steps: usize,
    max_score: u64,
    highest_tile: u32,
    policy_kind_mask: u8,
    disagreement_count: usize,
    disagreement_percentage: f32,
}

#[derive(Serialize)]
pub(crate) struct RunsResponse {
    total: usize,
    page: usize,
    page_size: usize,
    runs: Vec<RunSummaryResponse>,
    policy_kind_legend: PolicyKindLegend,
    #[serde(skip_serializing_if = "Option::is_none")]
    student_bins: Option<StudentBinsMetadata>,
}

#[derive(Serialize)]
pub(crate) struct RunDetailResponse {
    run: RunSummaryResponse,
    pagination: Pagination,
    steps: Vec<StepResponse>,
    policy_kind_legend: PolicyKindLegend,
    #[serde(skip_serializing_if = "Option::is_none")]
    student_bins: Option<StudentBinsMetadata>,
}

#[derive(Serialize)]
pub(crate) struct DisagreementsResponse {
    disagreements: Vec<u32>,
    total: usize,
}

#[derive(Serialize)]
pub(crate) struct HealthResponse {
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokenizer: Option<HealthTokenizerInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    student_bins: Option<StudentBinsMetadata>,
}

#[derive(Serialize)]
struct HealthTokenizerInfo {
    tokenizer_type: String,
    num_bins: usize,
    vocab_order: Vec<String>,
    valuation_types: Vec<String>,
}

#[derive(Serialize)]
struct StudentBinsMetadata {
    num_bins: usize,
}

#[derive(Deserialize, Default)]
pub struct RunsQuery {
    page: Option<usize>,
    page_size: Option<usize>,
    min_score: Option<u64>,
    max_score: Option<u64>,
    min_highest_tile: Option<u32>,
    max_highest_tile: Option<u32>,
    min_steps: Option<usize>,
    max_steps: Option<usize>,
    sort: Option<String>,
}

#[derive(Deserialize, Default)]
pub struct StepsQuery {
    offset: Option<usize>,
    limit: Option<usize>,
    tokenize: Option<bool>,
}

pub async fn list_runs(
    State(state): State<AppState>,
    Query(query): Query<RunsQuery>,
) -> Json<RunsResponse> {
    let mut items: Vec<&crate::app::RunEntry> = state.runs.iter().collect();
    items.retain(|entry| {
        let s = &entry.summary;
        if let Some(min) = query.min_score {
            if s.max_score < min {
                return false;
            }
        }
        if let Some(max) = query.max_score {
            if s.max_score > max {
                return false;
            }
        }
        if let Some(min_tile) = query.min_highest_tile {
            if s.highest_tile < min_tile {
                return false;
            }
        }
        if let Some(max_tile) = query.max_highest_tile {
            if s.highest_tile > max_tile {
                return false;
            }
        }
        if let Some(min_steps) = query.min_steps {
            if s.steps < min_steps {
                return false;
            }
        }
        if let Some(max_steps) = query.max_steps {
            if s.steps > max_steps {
                return false;
            }
        }
        true
    });

    match query.sort.as_deref() {
        Some("score_asc") => items.sort_by_key(|entry| entry.summary.max_score),
        Some("tile_desc") => {
            items.sort_by(|a, b| b.summary.highest_tile.cmp(&a.summary.highest_tile))
        }
        Some("tile_asc") => items.sort_by_key(|entry| entry.summary.highest_tile),
        Some("steps_desc") => items.sort_by(|a, b| b.summary.steps.cmp(&a.summary.steps)),
        Some("steps_asc") => items.sort_by_key(|entry| entry.summary.steps),
        _ => items.sort_by(|a, b| b.summary.max_score.cmp(&a.summary.max_score)),
    }

    let total = items.len();
    let page_size = query.page_size.unwrap_or(25).clamp(1, 500);
    let page = query.page.unwrap_or(1).max(1);
    let start = (page - 1).saturating_mul(page_size);
    let end = (start + page_size).min(total);
    let slice = if start >= total {
        &[]
    } else {
        &items[start..end]
    };

    let runs = slice
        .iter()
        .map(|entry| RunSummaryResponse {
            run_id: entry.summary.run_id,
            seed: entry.summary.seed,
            steps: entry.summary.steps,
            max_score: entry.summary.max_score,
            highest_tile: entry.summary.highest_tile,
            policy_kind_mask: entry.policy_kind_mask,
            disagreement_count: entry.disagreement_count,
            disagreement_percentage: entry.disagreement_percentage,
        })
        .collect();

    Json(RunsResponse {
        total,
        page,
        page_size,
        runs,
        policy_kind_legend: (*state.policy_kind_legend).clone(),
        student_bins: state
            .student_bins
            .as_ref()
            .as_ref()
            .map(|info| StudentBinsMetadata {
                num_bins: info.num_bins,
            }),
    })
}

pub async fn get_run(
    State(state): State<AppState>,
    AxumPath(run_id): AxumPath<u32>,
    Query(query): Query<StepsQuery>,
) -> Result<Json<RunDetailResponse>, (StatusCode, String)> {
    let idx = state
        .run_index
        .get(&run_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("run {run_id} not found")))?;
    let entry = &state.runs[*idx];
    let total = entry.steps.len();
    let offset = query.offset.unwrap_or(0).min(total);
    let limit = query
        .limit
        .unwrap_or(200)
        .clamp(1, 1000)
        .min(total.saturating_sub(offset));
    let slice = &entry.steps[offset..offset + limit];

    let mut steps: Vec<StepResponse> = slice
        .iter()
        .map(|step| StepResponse {
            step_index: step.step_index,
            board: step.board,
            board_value: step.board_value,
            branch_evs: step.branch_evs.iter().copied().collect(),
            relative_branch_evs: step.relative_branch_evs.iter().copied().collect(),
            advantage_branch: step.advantage_branch.iter().copied().collect(),
            legal_mask: step.legal_mask,
            teacher_move: step.teacher_move,
            is_disagreement: step.is_disagreement,
            valuation_type: state
                .valuation_names
                .get(step.valuation_type as usize)
                .cloned()
                .unwrap_or_else(|| "search".to_string()),
            annotation: step.annotation.as_ref().map(|ann| AnnotationPayload {
                policy_kind_mask: ann.policy_kind_mask,
                argmax_head: ann.argmax_head,
                argmax_prob: ann.argmax_prob,
                policy_p1: ann.policy_p1,
                policy_logp: ann.policy_logp,
                policy_hard: ann.policy_hard,
            }),
            student_bins: step.student_bins.as_ref().map(|bins| StudentBinsPayload {
                num_bins: bins.probs[0].len(),
                heads: clone_heads(bins),
            }),
            tokens: None,
        })
        .collect();

    if query.tokenize.unwrap_or(false) {
        if let Some(tokenizer) = state.tokenizer.as_ref().as_ref() {
            for (step_resp, step_record) in steps.iter_mut().zip(slice.iter()) {
                let valuation_type = &step_resp.valuation_type;
                let branch_evs: [f32; 4] = step_record.branch_evs.map(|opt| opt.unwrap_or(0.0));
                let board_eval = if valuation_type == "search" {
                    Some(step_record.board_value as i32)
                } else {
                    None
                };

                match tokenizer.encode_row(
                    valuation_type,
                    &branch_evs,
                    step_record.teacher_move,
                    step_record.legal_mask,
                    board_eval,
                ) {
                    Ok(tokens) => {
                        step_resp.tokens = Some(tokens.to_vec());
                    }
                    Err(err) => {
                        warn!(
                            "failed to tokenize step {}: {}",
                            step_record.step_index, err
                        );
                    }
                }
            }
        } else {
            warn!("tokenization requested but no tokenizer loaded");
        }
    }

    let disagreement_count = entry.disagreement_count;
    let disagreement_percentage = entry.disagreement_percentage;
    let response = RunDetailResponse {
        run: RunSummaryResponse {
            run_id: entry.summary.run_id,
            seed: entry.summary.seed,
            steps: entry.summary.steps,
            max_score: entry.summary.max_score,
            highest_tile: entry.summary.highest_tile,
            policy_kind_mask: entry.policy_kind_mask,
            disagreement_count,
            disagreement_percentage,
        },
        pagination: Pagination {
            offset,
            limit,
            total,
        },
        steps,
        policy_kind_legend: (*state.policy_kind_legend).clone(),
        student_bins: state
            .student_bins
            .as_ref()
            .as_ref()
            .map(|info| StudentBinsMetadata {
                num_bins: info.num_bins,
            }),
    };
    Ok(Json(response))
}

pub async fn get_disagreements(
    State(state): State<AppState>,
    AxumPath(run_id): AxumPath<u32>,
    Query(query): Query<StepsQuery>,
) -> Result<Json<DisagreementsResponse>, (StatusCode, String)> {
    let idx = state
        .run_index
        .get(&run_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("run {run_id} not found")))?;
    let entry = &state.runs[*idx];
    let total = entry.disagreements.len();
    let offset = query.offset.unwrap_or(0).min(total);
    let limit = query
        .limit
        .unwrap_or(100)
        .clamp(1, 1000)
        .min(total.saturating_sub(offset));
    let slice = &entry.disagreements[offset..offset + limit];
    Ok(Json(DisagreementsResponse {
        disagreements: slice.to_vec(),
        total,
    }))
}

pub async fn get_health(State(state): State<AppState>) -> Json<HealthResponse> {
    let tokenizer_info = state
        .tokenizer
        .as_ref()
        .as_ref()
        .map(|tokenizer| HealthTokenizerInfo {
            tokenizer_type: tokenizer.tokenizer_type().to_string(),
            num_bins: tokenizer.num_bins(),
            vocab_order: tokenizer.vocab_order().to_vec(),
            valuation_types: tokenizer.valuation_types().to_vec(),
        });

    Json(HealthResponse {
        status: "ok".to_string(),
        tokenizer: tokenizer_info,
        student_bins: state
            .student_bins
            .as_ref()
            .as_ref()
            .map(|info| StudentBinsMetadata {
                num_bins: info.num_bins,
            }),
    })
}

fn clone_heads(record: &StudentBinsRecord) -> [Vec<f32>; 4] {
    std::array::from_fn(|idx| record.probs[idx].clone())
}
