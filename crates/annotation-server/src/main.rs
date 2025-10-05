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
use tower_http::services::ServeDir;
use clap::Parser;
use dataset_packer::macroxue::{self, RunSummary};
use dataset_packer::schema::{AnnotationRow, MacroxueStepRow, annotation_kinds};
use npyz::NpyFile;
use serde::Deserialize;
use serde::Serialize;
use serde_json;
use tokio::signal;
use tracing::{error, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser, Debug)]
struct Args {
    /// Path to the base Macroxue dataset directory (steps-*.npy + metadata.db).
    #[arg(long)]
    dataset: PathBuf,
    /// Path to the annotation directory containing annotations-*.npy.
    #[arg(long)]
    annotations: PathBuf,
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
    branch_evs: Vec<Option<f32>>,
    legal_mask: u8,
    teacher_move: u8,
    is_disagreement: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    annotation: Option<AnnotationPayload>,
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

struct StepRecord {
    step_index: u32,
    board: [u8; 16],
    branch_evs: [Option<f32>; 4],
    legal_mask: u8,
    teacher_move: u8,
    annotation: Option<AnnotationRow>,
}

struct RunEntry {
    summary: RunSummary,
    steps: Vec<StepRecord>,
    policy_kind_mask: u8,
}

#[derive(Clone)]
struct AppState {
    runs: Arc<Vec<RunEntry>>,
    run_index: Arc<HashMap<u32, usize>>,
    policy_kind_legend: Arc<PolicyKindLegend>,
}

struct DatasetLoad {
    runs: Vec<RunEntry>,
    policy_kind_legend: PolicyKindLegend,
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
    let state = AppState {
        runs: Arc::new(dataset.runs),
        run_index: Arc::new(run_index),
        policy_kind_legend: Arc::new(dataset.policy_kind_legend),
    };

    use tower_http::cors::CorsLayer;

    let router = Router::new()
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
        .map(|entry| {
            let total_steps = entry.steps.len();
            let disagreement_count = entry.steps.iter().filter(|s| {
                if let Some(ann) = s.annotation {
                    s.teacher_move != 255 && s.teacher_move != ann.argmax_head
                } else {
                    false
                }
            }).count();
            let disagreement_percentage = if total_steps > 0 { disagreement_count as f32 / total_steps as f32 } else { 0.0 };
            RunSummaryResponse {
                run_id: entry.summary.run_id,
                seed: entry.summary.seed,
                steps: entry.summary.steps,
                max_score: entry.summary.max_score,
                highest_tile: entry.summary.highest_tile,
                policy_kind_mask: entry.policy_kind_mask,
                disagreement_count,
                disagreement_percentage,
            }
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

    let steps = slice
        .iter()
        .map(|step| {
            let is_disagreement = if let Some(ann) = step.annotation {
                step.teacher_move != 255 && step.teacher_move != ann.argmax_head
            } else {
                false
            };
            StepResponse {
                step_index: step.step_index,
                board: step.board,
                branch_evs: step.branch_evs.iter().copied().collect(),
                legal_mask: step.legal_mask,
                teacher_move: step.teacher_move,
                is_disagreement,
                annotation: step.annotation.map(|ann| AnnotationPayload {
                    policy_kind_mask: ann.policy_kind_mask,
                    argmax_head: ann.argmax_head,
                    argmax_prob: ann.argmax_prob,
                    policy_p1: ann.policy_p1,
                    policy_logp: ann.policy_logp,
                    policy_hard: ann.policy_hard,
                }),
            }
        })
        .collect();

    let total_steps = entry.steps.len();
    let disagreement_count = entry.steps.iter().filter(|s| {
        if let Some(ann) = s.annotation {
            s.teacher_move != 255 && s.teacher_move != ann.argmax_head
        } else {
            false
        }
    }).count();
    let disagreement_percentage = if total_steps > 0 { disagreement_count as f32 / total_steps as f32 } else { 0.0 };
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
    let mut disagreements: Vec<u32> = entry.steps.iter().filter_map(|s| {
        if let Some(ann) = s.annotation {
            if s.teacher_move != 255 && s.teacher_move != ann.argmax_head {
                Some(s.step_index)
            } else {
                None
            }
        } else {
            None
        }
    }).collect();
    disagreements.sort();
    let total = disagreements.len();
    let offset = query.offset.unwrap_or(0).min(total);
    let limit = query
        .limit
        .unwrap_or(100)
        .clamp(1, 1000)
        .min(total.saturating_sub(offset));
    let slice = &disagreements[offset..offset + limit];
    Ok(Json(DisagreementsResponse {
        disagreements: slice.to_vec(),
        total,
    }))
}

fn load_dataset(dataset_dir: &Path, annotations_dir: &Path) -> Result<DatasetLoad> {
    let runs = macroxue::load_runs(dataset_dir)?;
    let step_map = load_macro_steps(dataset_dir)?;
    let annotations = load_annotations(annotations_dir)?;
    let manifest = load_manifest(annotations_dir).unwrap_or_default();

    let mut entries = Vec::with_capacity(runs.len());
    for summary in runs {
        let steps = match step_map.get(&summary.run_id) {
            Some(s) => s,
            None => {
                error!("missing steps for run" = summary.run_id);
                continue;
            }
        };
        let mut records = Vec::with_capacity(steps.len());
        let mut policy_kind_mask = manifest
            .run_masks
            .get(&summary.run_id)
            .copied()
            .unwrap_or_default();
        for row in steps {
            let board = decode_board(row.board, row.tile_65536_mask);
            let branch_evs = branch_evs_as_options(row);
            let annotation = annotations.get(&(row.run_id, row.step_index)).copied();
            if let Some(ann) = annotation {
                policy_kind_mask |= ann.policy_kind_mask;
            }
            let legal_mask = row.ev_legal;
            let teacher_move = row.move_dir;
            records.push(StepRecord {
                step_index: row.step_index,
                board,
                branch_evs,
                legal_mask,
                teacher_move,
                annotation,
            });
        }
        entries.push(RunEntry {
            summary,
            steps: records,
            policy_kind_mask,
        });
    }
    Ok(DatasetLoad {
        runs: entries,
        policy_kind_legend: manifest.legend,
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
            branch_evs: [None; 4],
            legal_mask: 0,
            teacher_move: 0,
            annotation: Some(AnnotationRow {
                run_id: 1,
                step_index: 0,
                teacher_move: 0,
                legal_mask: 0,
                policy_kind_mask: 0,
                argmax_head: 1,
                argmax_prob: 0.5,
                policy_p1: [0.0; 4],
                policy_logp: [0.0; 4],
                policy_hard: [0.0; 4],
            }),
        };

        // Mock a step with teacher_move = 1, argmax_head = 1 -> no disagreement
        let step2 = StepRecord {
            step_index: 1,
            board: [0; 16],
            branch_evs: [None; 4],
            legal_mask: 0,
            teacher_move: 1,
            annotation: Some(AnnotationRow {
                run_id: 1,
                step_index: 1,
                teacher_move: 1,
                legal_mask: 0,
                policy_kind_mask: 0,
                argmax_head: 1,
                argmax_prob: 0.5,
                policy_p1: [0.0; 4],
                policy_logp: [0.0; 4],
                policy_hard: [0.0; 4],
            }),
        };

        // Mock a step with teacher_move = 255 -> no disagreement
        let step3 = StepRecord {
            step_index: 2,
            board: [0; 16],
            branch_evs: [None; 4],
            legal_mask: 0,
            teacher_move: 255,
            annotation: Some(AnnotationRow {
                run_id: 1,
                step_index: 2,
                teacher_move: 255,
                legal_mask: 0,
                policy_kind_mask: 0,
                argmax_head: 0,
                argmax_prob: 0.5,
                policy_p1: [0.0; 4],
                policy_logp: [0.0; 4],
                policy_hard: [0.0; 4],
            }),
        };

        // Mock a step with no annotation -> no disagreement
        let step4 = StepRecord {
            step_index: 3,
            board: [0; 16],
            branch_evs: [None; 4],
            legal_mask: 0,
            teacher_move: 0,
            annotation: None,
        };

        let steps = vec![step1, step2, step3, step4];
        let disagreement_count = steps.iter().filter(|s| {
            if let Some(ann) = s.annotation {
                s.teacher_move != 255 && s.teacher_move != ann.argmax_head
            } else {
                false
            }
        }).count();

        assert_eq!(disagreement_count, 1);
        let total_steps = steps.len();
        let percentage = disagreement_count as f32 / total_steps as f32;
        assert_eq!(percentage, 0.25);
    }
}
