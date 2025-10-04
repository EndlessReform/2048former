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
use dataset_packer::macrosxue::{self, RunSummary};
use dataset_packer::schema::{AnnotationRow, MacroxueStepRow};
use npyz::NpyFile;
use serde::Deserialize;
use serde::Serialize;
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
}

#[derive(Clone, Serialize)]
struct AnnotationPayload {
    argmax_head: u8,
    argmax_prob: f32,
    policy_p1: [f32; 4],
}

#[derive(Clone, Serialize)]
struct StepResponse {
    step_index: u32,
    board: [u8; 16],
    branch_evs: Vec<Option<f32>>,
    legal_mask: u8,
    teacher_move: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    annotation: Option<AnnotationPayload>,
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
}

#[derive(Serialize)]
struct RunDetailResponse {
    run: RunSummaryResponse,
    pagination: Pagination,
    steps: Vec<StepResponse>,
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
}

#[derive(Clone)]
struct AppState {
    runs: Arc<Vec<RunEntry>>,
    run_index: Arc<HashMap<u32, usize>>,
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
    let runs = load_dataset(&args.dataset, &args.annotations)?;
    let run_index = runs
        .iter()
        .enumerate()
        .map(|(idx, entry)| (entry.summary.run_id, idx))
        .collect::<HashMap<_, _>>();
    let state = AppState {
        runs: Arc::new(runs),
        run_index: Arc::new(run_index),
    };

    let router = Router::new()
        .route("/runs", get(list_runs))
        .route("/runs/:run_id", get(get_run))
        .with_state(state);

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
        })
        .collect();

    Json(RunsResponse {
        total,
        page,
        page_size,
        runs,
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
        .map(|step| StepResponse {
            step_index: step.step_index,
            board: step.board,
            branch_evs: step.branch_evs.iter().copied().collect(),
            legal_mask: step.legal_mask,
            teacher_move: step.teacher_move,
            annotation: step.annotation.map(|ann| AnnotationPayload {
                argmax_head: ann.argmax_head,
                argmax_prob: ann.argmax_prob,
                policy_p1: ann.policy_p1,
            }),
        })
        .collect();

    let response = RunDetailResponse {
        run: RunSummaryResponse {
            run_id: entry.summary.run_id,
            seed: entry.summary.seed,
            steps: entry.summary.steps,
            max_score: entry.summary.max_score,
            highest_tile: entry.summary.highest_tile,
        },
        pagination: Pagination {
            offset,
            limit,
            total,
        },
        steps,
    };
    Ok(Json(response))
}

fn load_dataset(dataset_dir: &Path, annotations_dir: &Path) -> Result<Vec<RunEntry>> {
    let runs = macrosxue::load_runs(dataset_dir)?;
    let step_map = load_macro_steps(dataset_dir)?;
    let annotations = load_annotations(annotations_dir)?;

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
        for row in steps {
            let board = decode_board(row.board, row.tile_65536_mask);
            let branch_evs = branch_evs_as_options(row);
            let annotation = annotations.get(&(row.run_id, row.step_index)).copied();
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
        });
    }
    Ok(entries)
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
