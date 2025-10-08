use std::collections::HashMap;
use std::fs::File;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use axum::{
    Json, Router,
    extract::{Path as AxumPath, Query, State},
    http::StatusCode,
    routing::get,
};
use clap::Parser;
use dataset_packer::evaluate;
use dataset_packer::macroxue::{self, RunSummary};
use dataset_packer::macroxue::tokenizer::{MacroxueTokenizerV2, Tokenizer};
use dataset_packer::schema::{AnnotationRow, MacroxueStepRow, annotation_kinds};
use npyz::NpyFile;
use rayon::prelude::*;
use serde::Deserialize;
use serde::Serialize;
use serde_json;
use tokio::signal;
use tower_http::services::ServeDir;
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser, Debug)]
struct Args {
    /// Path to the base Macroxue dataset directory (steps-*.npy + metadata.db).
    #[arg(long)]
    dataset: PathBuf,
    /// Path to the annotation directory containing annotations-*.npy.
    #[arg(long)]
    annotations: PathBuf,
    /// Optional path to the tokenizer JSON file for tokenization support.
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    /// Optional path to the UI dist directory to serve static files.
    #[arg(long)]
    ui_path: Option<PathBuf>,
    /// Host interface to bind (default 0.0.0.0).
    #[arg(long, default_value = "0.0.0.0")]
    host: String,
    /// Port to bind (default 8080).
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// Optional tracing filter, e.g. "info", "debug".
    #[arg(long, default_value = "info")]
    log: String,
}

#[derive(Clone, Serialize)]
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
    tokens: Option<Vec<u16>>,
}

#[derive(Clone, Serialize, Deserialize)]
struct PolicyKindLegend {
    policy_p1: u8,
    policy_logprobs: u8,
    policy_hard: u8,
}

impl Default for PolicyKindLegend {
    fn default() -> Self {
        Self {
            policy_p1: annotation_kinds::POLICY_P1,
            policy_logprobs: annotation_kinds::POLICY_LOGPROBS,
            policy_hard: annotation_kinds::POLICY_HARD,
        }
    }
}

#[derive(Serialize)]
struct Pagination {
    offset: usize,
    limit: usize,
    total: usize,
}

#[derive(Serialize)]
struct RunsResponse {
    total: usize,
    page: usize,
    page_size: usize,
    runs: Vec<RunSummaryResponse>,
    policy_kind_legend: PolicyKindLegend,
}

#[derive(Serialize)]
struct RunDetailResponse {
    run: RunSummaryResponse,
    pagination: Pagination,
    steps: Vec<StepResponse>,
    policy_kind_legend: PolicyKindLegend,
}

#[derive(Serialize)]
struct DisagreementsResponse {
    disagreements: Vec<u32>,
    total: usize,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    tokenizer: Option<HealthTokenizerInfo>,
}

#[derive(Serialize)]
struct HealthTokenizerInfo {
    tokenizer_type: String,
    num_bins: usize,
    vocab_order: Vec<String>,
    valuation_types: Vec<String>,
}

struct StepRecord {
    step_index: u32,
    board: [u8; 16],
    board_value: f32,
    branch_evs: [Option<f32>; 4],
    relative_branch_evs: [Option<i32>; 4],
    advantage_branch: [Option<i32>; 4],
    legal_mask: u8,
    teacher_move: u8,
    is_disagreement: bool,
    valuation_type: u8,
    annotation: Option<AnnotationPayload>,
    annotation_mask: u8,
}

struct RunEntry {
    summary: RunSummary,
    steps: Vec<StepRecord>,
    policy_kind_mask: u8,
    disagreements: Vec<u32>,
    disagreement_count: usize,
    disagreement_percentage: f32,
}

#[derive(Clone)]
struct AppState {
    runs: Arc<Vec<RunEntry>>,
    run_index: Arc<HashMap<u32, usize>>,
    policy_kind_legend: Arc<PolicyKindLegend>,
    valuation_names: Arc<Vec<String>>,
    tokenizer: Arc<Option<MacroxueTokenizerV2>>,
}

struct DatasetLoad {
    runs: Vec<RunEntry>,
    policy_kind_legend: PolicyKindLegend,
    valuation_names: Vec<String>,
}

struct ManifestData {
    legend: PolicyKindLegend,
    run_masks: HashMap<u32, u8>,
}

impl Default for ManifestData {
    fn default() -> Self {
        Self {
            legend: PolicyKindLegend::default(),
            run_masks: HashMap::new(),
        }
    }
}

#[derive(Deserialize, Default)]
struct RunsQuery {
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
struct StepsQuery {
    offset: Option<usize>,
    limit: Option<usize>,
    tokenize: Option<bool>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(args.log.clone()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("loading dataset" = %args.dataset.display(), "annotations" = %args.annotations.display());
    let dataset = load_dataset(&args.dataset, &args.annotations)?;
    let run_index = dataset
        .runs
        .iter()
        .enumerate()
        .map(|(idx, entry)| (entry.summary.run_id, idx))
        .collect::<HashMap<_, _>>();

    let tokenizer = if let Some(tokenizer_path) = &args.tokenizer {
        info!("loading tokenizer" = %tokenizer_path.display());
        Some(MacroxueTokenizerV2::from_path(tokenizer_path).with_context(|| format!("failed to load tokenizer from {}", tokenizer_path.display()))?)
    } else {
        None
    };

    let state = AppState {
        runs: Arc::new(dataset.runs),
        run_index: Arc::new(run_index),
        policy_kind_legend: Arc::new(dataset.policy_kind_legend),
        valuation_names: Arc::new(dataset.valuation_names),
        tokenizer: Arc::new(tokenizer),
    };

    use tower_http::cors::CorsLayer;

    let router = Router::new()
        .route("/health", get(get_health))
        .route("/runs", get(list_runs))
        .route("/runs/:run_id", get(get_run))
        .route("/runs/:run_id/disagreements", get(get_disagreements))
        .layer(CorsLayer::permissive());

    let router = if let Some(ui_path) = &args.ui_path {
        router.fallback_service(ServeDir::new(ui_path).append_index_html_on_directories(true))
    } else {
        router
    };

    let router = router.with_state(state);

    let addr: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .context("invalid host/port combination")?;
    info!("listening" = %addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install CTRL+C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
}

async fn list_runs(
    State(state): State<AppState>,
    Query(query): Query<RunsQuery>,
) -> Json<RunsResponse> {
    let mut items: Vec<&RunEntry> = state.runs.iter().collect();
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
    })
}

async fn get_run(
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
            annotation: step.annotation.clone(),
            tokens: None,
        })
        .collect();

    // Tokenize if requested and tokenizer is available
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
                        warn!("failed to tokenize step {}: {}", step_record.step_index, err);
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
    };
    Ok(Json(response))
}

async fn get_disagreements(
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

async fn get_health(State(state): State<AppState>) -> Json<HealthResponse> {
    let tokenizer_info = state.tokenizer.as_ref().as_ref().map(|tokenizer| HealthTokenizerInfo {
        tokenizer_type: tokenizer.tokenizer_type().to_string(),
        num_bins: tokenizer.num_bins(),
        vocab_order: tokenizer.vocab_order().to_vec(),
        valuation_types: tokenizer.valuation_types().to_vec(),
    });

    Json(HealthResponse {
        status: "ok".to_string(),
        tokenizer: tokenizer_info,
    })
}

fn load_dataset(dataset_dir: &Path, annotations_dir: &Path) -> Result<DatasetLoad> {
    let runs = macroxue::load_runs(dataset_dir)?;
    let step_map = load_macro_steps(dataset_dir)?;
    let annotations = load_annotations(annotations_dir)?;
    let manifest = load_manifest(annotations_dir).unwrap_or_default();
    let valuation_names = macroxue::load_valuation_names(dataset_dir).unwrap_or_else(|err| {
        warn!("failed to load valuation_types.json: {err}");
        vec!["search".to_string()]
    });

    let mut entries = Vec::with_capacity(runs.len());
    for summary in runs {
        let steps = match step_map.get(&summary.run_id) {
            Some(s) => s,
            None => {
                error!("missing steps for run" = summary.run_id);
                continue;
            }
        };

        let records: Vec<StepRecord> = steps
            .par_iter()
            .map(|row| {
                let board = decode_board(row.board, row.tile_65536_mask);
                let board_i32 = board.map(|x| x as i32);
                let board_value = evaluate(&board_i32, false) as f32;

                let branch_evs = branch_evs_as_options(row);
                let valuation_idx = row.valuation_type as usize;
                let valuation_name = valuation_names
                    .get(valuation_idx)
                    .map(|s| s.as_str())
                    .unwrap_or("search");
                let is_search = valuation_name.eq_ignore_ascii_case("search");

                let mut relative_branch_evs = [None; 4];
                let mut advantage_branch = [None; 4];
                let mut max_rel: Option<i32> = None;
                let mut max_ev: Option<f32> = None;

                for (idx, maybe_ev) in branch_evs.iter().enumerate() {
                    if let Some(ev) = maybe_ev {
                        if is_search {
                            let rel = (ev * 1000.0 - board_value).round() as i32;
                            relative_branch_evs[idx] = Some(rel);
                            max_rel = Some(match max_rel {
                                Some(current) => current.max(rel),
                                None => rel,
                            });
                        } else {
                            max_ev = Some(match max_ev {
                                Some(current) => current.max(*ev),
                                None => *ev,
                            });
                        }
                    }
                }

                if is_search {
                    let best = max_rel.unwrap_or(0);
                    for idx in 0..4 {
                        if let Some(rel) = relative_branch_evs[idx] {
                            advantage_branch[idx] = Some(rel - best);
                        }
                    }
                } else if let Some(best_ev) = max_ev {
                    for idx in 0..4 {
                        if let Some(ev) = branch_evs[idx] {
                            let delta = (ev - best_ev) * 1000.0;
                            advantage_branch[idx] = Some(delta.round() as i32);
                        }
                    }
                }

                let annotation_row = annotations.get(&(row.run_id, row.step_index)).copied();
                let (annotation, annotation_mask, is_disagreement) =
                    if let Some(ann) = annotation_row {
                        let payload = AnnotationPayload {
                            policy_kind_mask: ann.policy_kind_mask,
                            argmax_head: ann.argmax_head,
                            argmax_prob: ann.argmax_prob,
                            policy_p1: ann.policy_p1,
                            policy_logp: ann.policy_logp,
                            policy_hard: ann.policy_hard,
                        };
                        let disagree = row.move_dir != 255 && row.move_dir != ann.argmax_head;
                        (Some(payload), ann.policy_kind_mask, disagree)
                    } else {
                        (None, 0, false)
                    };

                StepRecord {
                    step_index: row.step_index,
                    board,
                    board_value,
                    branch_evs,
                    relative_branch_evs,
                    advantage_branch,
                    legal_mask: row.ev_legal,
                    teacher_move: row.move_dir,
                    is_disagreement,
                    valuation_type: row.valuation_type,
                    annotation,
                    annotation_mask,
                }
            })
            .collect();

        let mut policy_kind_mask = manifest
            .run_masks
            .get(&summary.run_id)
            .copied()
            .unwrap_or_default();
        for record in &records {
            policy_kind_mask |= record.annotation_mask;
        }

        let disagreements: Vec<u32> = records
            .iter()
            .filter(|step| step.is_disagreement)
            .map(|step| step.step_index)
            .collect();
        let disagreement_count = disagreements.len();
        let disagreement_percentage = if records.is_empty() {
            0.0
        } else {
            disagreement_count as f32 / records.len() as f32
        };

        entries.push(RunEntry {
            summary,
            steps: records,
            policy_kind_mask,
            disagreements,
            disagreement_count,
            disagreement_percentage,
        });
    }

    Ok(DatasetLoad {
        runs: entries,
        policy_kind_legend: manifest.legend,
        valuation_names,
    })
}

#[derive(Deserialize)]
struct ManifestFile {
    #[serde(default)]
    kinds: Option<PolicyKindLegend>,
    #[serde(default)]
    runs: Vec<ManifestEntryFile>,
}

#[derive(Deserialize)]
struct ManifestEntryFile {
    run_id: u32,
    policy_kind_mask: u8,
}

fn load_manifest(dir: &Path) -> Result<ManifestData> {
    let path = dir.join("annotation_manifest.json");
    if !path.exists() {
        return Ok(ManifestData::default());
    }
    let file = File::open(&path).with_context(|| format!("open {}", path.display()))?;
    let manifest: ManifestFile = serde_json::from_reader(file)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    let legend = manifest.kinds.unwrap_or_default();
    let run_masks = manifest
        .runs
        .into_iter()
        .map(|entry| (entry.run_id, entry.policy_kind_mask))
        .collect();
    Ok(ManifestData { legend, run_masks })
}

fn load_macro_steps(dir: &Path) -> Result<HashMap<u32, Vec<MacroxueStepRow>>> {
    let mut map: HashMap<u32, Vec<MacroxueStepRow>> = HashMap::new();
    for shard in collect_shards(dir, "steps")? {
        let file = File::open(&shard).with_context(|| format!("open {}", shard.display()))?;
        let mut reader = std::io::BufReader::new(file);
        let npy = NpyFile::new(&mut reader)
            .with_context(|| format!("failed to read {}", shard.display()))?;
        let mut data = npy
            .data::<MacroxueStepRow>()
            .map_err(|err| anyhow!("{}: {err}", shard.display()))?;
        while let Some(row) = data.next() {
            let row =
                row.with_context(|| format!("failed to decode row in {}", shard.display()))?;
            map.entry(row.run_id).or_default().push(row);
        }
    }
    Ok(map)
}

fn load_annotations(dir: &Path) -> Result<HashMap<(u32, u32), AnnotationRow>> {
    let mut map = HashMap::new();
    for shard in collect_shards(dir, "annotations")? {
        let file = File::open(&shard).with_context(|| format!("open {}", shard.display()))?;
        let mut reader = std::io::BufReader::new(file);
        let npy = NpyFile::new(&mut reader)
            .with_context(|| format!("failed to read {}", shard.display()))?;
        let mut data = npy
            .data::<AnnotationRow>()
            .map_err(|err| anyhow!("{}: {err}", shard.display()))?;
        while let Some(row) = data.next() {
            let row =
                row.with_context(|| format!("failed to decode row in {}", shard.display()))?;
            map.insert((row.run_id, row.step_index), row);
        }
    }
    Ok(map)
}

fn collect_shards(dir: &Path, prefix: &str) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let single = dir.join(format!("{prefix}.npy"));
    if single.exists() {
        files.push(single);
    }
    let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("failed to read {}", dir.display()))?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let name = entry.file_name();
            let name = name.to_str()?;
            if name.starts_with(&format!("{prefix}-")) && name.ends_with(".npy") {
                Some(entry.path())
            } else {
                None
            }
        })
        .collect();
    shards.sort();
    files.extend(shards);
    if files.is_empty() {
        Err(anyhow!("no {prefix}*.npy files found in {}", dir.display()))
    } else {
        Ok(files)
    }
}

fn branch_evs_as_options(row: &MacroxueStepRow) -> [Option<f32>; 4] {
    let mut out = [None; 4];
    for idx in 0..4 {
        if row.ev_legal & (1 << idx) != 0 {
            out[idx] = Some(row.branch_evs[idx]);
        }
    }
    out
}

fn decode_board(packed: u64, mask: u16) -> [u8; 16] {
    let mut out = [0u8; 16];
    for i in 0..16 {
        let shift = (15 - i) * 4;
        let nib = ((packed >> shift) & 0xF) as u8;
        let mut val = nib;
        if (mask >> i) & 1 == 1 {
            val = 16;
        }
        out[i] = val;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disagreement_calculation() {
        // Mock a step with teacher_move = 0, argmax_head = 1 -> disagreement
        let step1 = StepRecord {
            step_index: 0,
            board: [0; 16],
            board_value: 0.0,
            branch_evs: [None; 4],
            relative_branch_evs: [None; 4],
            advantage_branch: [None; 4],
            legal_mask: 0,
            teacher_move: 0,
            is_disagreement: false,
            valuation_type: 0,
            annotation: Some(AnnotationPayload {
                policy_kind_mask: 0,
                argmax_head: 1,
                argmax_prob: 0.5,
                policy_p1: [0.0; 4],
                policy_logp: [0.0; 4],
                policy_hard: [0.0; 4],
            }),
            annotation_mask: 0,
        };

        // Mock a step with teacher_move = 1, argmax_head = 1 -> no disagreement
        let step2 = StepRecord {
            step_index: 1,
            board: [0; 16],
            board_value: 0.0,
            branch_evs: [None; 4],
            relative_branch_evs: [None; 4],
            advantage_branch: [None; 4],
            legal_mask: 0,
            teacher_move: 1,
            is_disagreement: false,
            valuation_type: 0,
            annotation: Some(AnnotationPayload {
                policy_kind_mask: 0,
                argmax_head: 1,
                argmax_prob: 0.5,
                policy_p1: [0.0; 4],
                policy_logp: [0.0; 4],
                policy_hard: [0.0; 4],
            }),
            annotation_mask: 0,
        };

        // Mock a step with teacher_move = 255 -> no disagreement
        let step3 = StepRecord {
            step_index: 2,
            board: [0; 16],
            board_value: 0.0,
            branch_evs: [None; 4],
            relative_branch_evs: [None; 4],
            advantage_branch: [None; 4],
            legal_mask: 0,
            teacher_move: 255,
            is_disagreement: false,
            valuation_type: 0,
            annotation: Some(AnnotationPayload {
                policy_kind_mask: 0,
                argmax_head: 0,
                argmax_prob: 0.5,
                policy_p1: [0.0; 4],
                policy_logp: [0.0; 4],
                policy_hard: [0.0; 4],
            }),
            annotation_mask: 0,
        };

        // Mock a step with no annotation -> no disagreement
        let step4 = StepRecord {
            step_index: 3,
            board: [0; 16],
            board_value: 0.0,
            branch_evs: [None; 4],
            relative_branch_evs: [None; 4],
            advantage_branch: [None; 4],
            legal_mask: 0,
            teacher_move: 0,
            is_disagreement: false,
            valuation_type: 0,
            annotation: None,
            annotation_mask: 0,
        };

        let steps = vec![step1, step2, step3, step4];
        let disagreement_count = steps
            .iter()
            .filter(|s| {
                if let Some(ann) = &s.annotation {
                    s.teacher_move != 255 && s.teacher_move != ann.argmax_head
                } else {
                    false
                }
            })
            .count();

        assert_eq!(disagreement_count, 1);
        let total_steps = steps.len();
        let percentage = disagreement_count as f32 / total_steps as f32;
        assert_eq!(percentage, 0.25);
    }
}
