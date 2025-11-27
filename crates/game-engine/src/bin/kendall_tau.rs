#![allow(unexpected_cfgs, non_local_definitions)]

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use dataset_packer::schema::{MacroxueStepRow, StructuredRow};
use dataset_packer::selfplay::collect_selfplay_step_files;
use dataset_packer::writer::StepsWriter;
use game_engine::{config, grpc};
use log::{info, warn};
use npyz::{DType, Field, NpyFile, TypeStr};
use tonic::Code;
use twenty48_utils::engine::{Board, Move};

#[derive(Parser, Debug)]
#[command(
    name = "kendall-tau",
    about = "Score Macroxue branches with the value head and compute Kendall's tau against branch_evs."
)]
struct Args {
    /// Path to the packed Macroxue dataset (containing steps.npy / steps-*.npy).
    #[arg(long)]
    dataset: PathBuf,

    /// Output directory for the Kendall sidecar (defaults to the dataset dir).
    #[arg(long)]
    output: Option<PathBuf>,

    /// Path to the inference server UDS socket.
    #[arg(long, conflicts_with = "tcp")]
    uds: Option<PathBuf>,

    /// TCP address for the inference server (e.g., http://127.0.0.1:50051).
    #[arg(long, conflicts_with = "uds")]
    tcp: Option<String>,

    /// Number of expanded boards to submit per inference request.
    #[arg(long, default_value_t = 2048)]
    batch_size: usize,

    /// Optional cap on the number of steps to process (for smoke tests).
    #[arg(long)]
    limit: Option<usize>,

    /// Overwrite existing kendall-*.npy shards if present.
    #[arg(long, default_value_t = false)]
    overwrite: bool,
}

#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, npyz::Serialize, npyz::Deserialize, npyz::AutoSerialize,
)]
struct KendallRow {
    run_id: u32,
    step_index: u32,
    legal_mask: u8,
    scored_mask: u8,
    teacher_best: u8,
    predicted_best: u8,
    kendall_tau: f32,
    value_scores: [f32; 4],
}

impl StructuredRow for KendallRow {
    fn dtype() -> DType {
        let u1: TypeStr = "<u1".parse().unwrap();
        let u4: TypeStr = "<u4".parse().unwrap();
        let f4: TypeStr = "<f4".parse().unwrap();
        DType::Record(vec![
            Field {
                name: "run_id".into(),
                dtype: DType::Plain(u4.clone()),
            },
            Field {
                name: "step_index".into(),
                dtype: DType::Plain(u4.clone()),
            },
            Field {
                name: "legal_mask".into(),
                dtype: DType::Plain(u1.clone()),
            },
            Field {
                name: "scored_mask".into(),
                dtype: DType::Plain(u1.clone()),
            },
            Field {
                name: "teacher_best".into(),
                dtype: DType::Plain(u1.clone()),
            },
            Field {
                name: "predicted_best".into(),
                dtype: DType::Plain(u1.clone()),
            },
            Field {
                name: "kendall_tau".into(),
                dtype: DType::Plain(f4.clone()),
            },
            Field {
                name: "value_scores".into(),
                dtype: DType::Array(4, Box::new(DType::Plain(f4))),
            },
        ])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct StepKey {
    run_id: u32,
    step_index: u32,
}

#[derive(Clone, Debug)]
struct StepAccumulator {
    branch_evs: [f32; 4],
    legal_mask: u8,
    scored_mask: u8,
    expected: usize,
    filled: usize,
    preds: [f32; 4],
}

impl StepAccumulator {
    fn new(branch_evs: [f32; 4], legal_mask: u8, scored_mask: u8) -> Self {
        Self {
            branch_evs,
            legal_mask,
            scored_mask,
            expected: scored_mask.count_ones() as usize,
            filled: 0,
            preds: [f32::NAN; 4],
        }
    }
}

#[derive(Clone, Debug)]
struct PendingItem {
    id: u64,
    key: StepKey,
    move_idx: usize,
    board: [u8; 16],
}

#[derive(Clone, Debug)]
struct PredictionResult {
    key: StepKey,
    move_idx: usize,
    value: f32,
}

#[derive(Default)]
struct Stats {
    total_steps: usize,
    total_moves: usize,
    scored_steps: usize,
    tau_steps: usize,
    tau_sum: f64,
    top_match: usize,
    illegal_filtered: usize,
}

impl Stats {
    fn record(&mut self, step: &StepAccumulator, row: &KendallRow, tau: Option<f32>) {
        self.total_steps += 1;
        self.total_moves += step.expected;
        if step.scored_mask != 0 {
            self.scored_steps += 1;
        }
        if let Some(t) = tau {
            self.tau_steps += 1;
            self.tau_sum += t as f64;
        }
        if row.teacher_best != u8::MAX && row.teacher_best == row.predicted_best {
            self.top_match += 1;
        }
    }

    fn mean_tau(&self) -> Option<f64> {
        if self.tau_steps == 0 {
            return None;
        }
        Some(self.tau_sum / self.tau_steps as f64)
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    if args.batch_size == 0 {
        bail!("batch_size must be positive");
    }

    twenty48_utils::engine::new();

    let connection = build_connection(&args)?;
    let dataset_dir = args.dataset.canonicalize().unwrap_or(args.dataset.clone());
    let output_dir = args.output.clone().unwrap_or_else(|| dataset_dir.clone());

    let mut client = game_engine::pipeline::connect_inference(&connection).await?;
    let mut describe_client = client.clone();
    let value_meta = describe_value_head(&mut describe_client).await?;

    let step_files = collect_selfplay_step_files(&dataset_dir)?;
    if step_files.is_empty() {
        bail!(
            "no steps.npy or steps-*.npy found under {}",
            dataset_dir.display()
        );
    }
    validate_macroxue_shard(&step_files[0])?;

    let shard_counts = step_shard_row_counts(&step_files)?;
    let planned_counts = plan_row_counts(&shard_counts, args.limit)?;
    let target_rows: usize = planned_counts.iter().sum();
    if target_rows == 0 {
        bail!("planned row count is zero; nothing to do");
    }

    info!(
        "Starting Kendall tau evaluation: dataset={} output={} batch_size={} target_steps={}",
        dataset_dir.display(),
        output_dir.display(),
        args.batch_size,
        target_rows
    );

    let mut writer = StepsWriter::<KendallRow>::with_shard_plan(
        &output_dir,
        "kendall",
        &planned_counts,
        args.overwrite,
    )?;

    let mut stats = Stats::default();
    let mut step_state: HashMap<StepKey, StepAccumulator> = HashMap::new();
    let mut step_order: VecDeque<StepKey> = VecDeque::new();
    let mut pending: Vec<PendingItem> = Vec::new();
    let mut next_id: u64 = 0;

    let mut rows_seen: usize = 0;
    for path in step_files {
        if rows_seen >= target_rows {
            break;
        }
        let file = File::open(&path)
            .with_context(|| format!("failed to open steps shard {}", path.display()))?;
        let mut reader = BufReader::new(file);
        let npy = NpyFile::new(&mut reader)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let mut rows = npy
            .data::<MacroxueStepRow>()
            .map_err(|err| anyhow!("{}: {err}", path.display()))?;
        while let Some(row) = rows.next() {
            if rows_seen >= target_rows {
                break;
            }
            let row = row.with_context(|| format!("failed to decode row in {}", path.display()))?;
            rows_seen += 1;
            let key = StepKey {
                run_id: row.run_id,
                step_index: row.step_index,
            };
            let legal_mask = row.ev_legal & 0x0F;
            let board = Board::from_raw(row.board);
            let mut scored_mask = 0u8;
            for (idx, mv) in [Move::Up, Move::Down, Move::Left, Move::Right]
                .into_iter()
                .enumerate()
            {
                let bit = 1u8 << idx;
                if legal_mask & bit == 0 {
                    continue;
                }
                let shifted = board.shift(mv);
                if shifted == board {
                    stats.illegal_filtered += 1;
                    continue;
                }
                let tiles = shifted.to_vec();
                let mut exps = [0u8; 16];
                exps.copy_from_slice(&tiles[..16]);
                pending.push(PendingItem {
                    id: next_id,
                    key,
                    move_idx: idx,
                    board: exps,
                });
                next_id = next_id.wrapping_add(1);
                scored_mask |= bit;
            }

            let acc = StepAccumulator::new(row.branch_evs, legal_mask, scored_mask);
            step_order.push_back(key);
            step_state.insert(key, acc);
            if pending.len() >= args.batch_size {
                flush_batch(
                    &mut client,
                    &mut pending,
                    &value_meta,
                    &mut step_state,
                    &mut step_order,
                    &mut writer,
                    &mut stats,
                )
                .await?;
            }
            flush_ready_steps(&mut step_order, &mut step_state, &mut writer, &mut stats)?;
        }
    }

    flush_batch(
        &mut client,
        &mut pending,
        &value_meta,
        &mut step_state,
        &mut step_order,
        &mut writer,
        &mut stats,
    )
    .await?;
    flush_ready_steps(&mut step_order, &mut step_state, &mut writer, &mut stats)?;

    if !step_order.is_empty() || !step_state.is_empty() {
        bail!(
            "incomplete write: {} steps remain in queue",
            step_order.len()
        );
    }

    let shards_written = writer.finish()?;
    if stats.total_steps != target_rows {
        bail!(
            "wrote {} rows but expected {} based on shard plan",
            stats.total_steps,
            target_rows
        );
    }
    let mean_tau = stats.mean_tau();
    println!("Kendall tau sidecar written to {}", output_dir.display());
    println!("- shards: {}", shards_written);
    println!(
        "- steps processed: {} (planned {})",
        stats.total_steps, target_rows
    );
    println!(
        "- moves scored: {} ({} steps with value predictions)",
        stats.total_moves, stats.scored_steps
    );
    println!(
        "- Kendall tau coverage: {} steps{}",
        stats.tau_steps,
        mean_tau
            .map(|m| format!(", mean {:.4}", m))
            .unwrap_or_else(|| String::from(""))
    );
    let top_pct = if stats.total_steps == 0 {
        0.0
    } else {
        (stats.top_match as f64 / stats.total_steps as f64) * 100.0
    };
    println!(
        "- predicted top = teacher top: {} ({:.2}%)",
        stats.top_match, top_pct
    );
    if stats.illegal_filtered > 0 {
        println!(
            "- skipped {} branch(es) marked legal but unchanged after shift",
            stats.illegal_filtered
        );
    }

    Ok(())
}

fn build_connection(args: &Args) -> Result<config::Connection> {
    if args.uds.is_none() && args.tcp.is_none() {
        bail!("either --uds or --tcp must be provided");
    }
    if args.uds.is_some() && args.tcp.is_some() {
        bail!("specify only one of --uds or --tcp");
    }
    Ok(config::Connection {
        uds_path: args.uds.clone(),
        tcp_addr: args.tcp.clone(),
    })
}

async fn describe_value_head(client: &mut grpc::Client) -> Result<Option<grpc::pb::ValueMetadata>> {
    match client.describe(grpc::pb::DescribeRequest {}).await {
        Ok(resp) => {
            let resp = resp.into_inner();
            if !resp.has_value_head {
                bail!("inference server reports no value head; cannot compute Kendall tau");
            }
            Ok(resp.model_metadata.and_then(|m| m.value))
        }
        Err(status) => {
            if status.code() != Code::Unimplemented {
                return Err(status.into());
            }
            warn!("Describe RPC not implemented; continuing without value metadata");
            Ok(None)
        }
    }
}

fn validate_macroxue_shard(path: &Path) -> Result<()> {
    let file = File::open(path)
        .with_context(|| format!("failed to open steps shard {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let npy =
        NpyFile::new(&mut reader).with_context(|| format!("failed to read {}", path.display()))?;
    let mut rows = npy
        .data::<MacroxueStepRow>()
        .map_err(|err| anyhow!("{}: {err}", path.display()))?;
    if let Some(row) = rows.next() {
        row.with_context(|| format!("failed to decode first row in {}", path.display()))?;
        Ok(())
    } else {
        bail!("{} is empty", path.display())
    }
}

fn step_shard_row_counts(step_paths: &[PathBuf]) -> Result<Vec<usize>> {
    let mut counts = Vec::with_capacity(step_paths.len());
    for path in step_paths {
        let file = File::open(path)
            .with_context(|| format!("failed to open steps shard {}", path.display()))?;
        let mut reader = BufReader::new(file);
        let npy = NpyFile::new(&mut reader)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let shape = npy.shape();
        if shape.is_empty() {
            bail!("{} has no dimensions", path.display());
        }
        let rows = usize::try_from(shape[0])
            .map_err(|_| anyhow!("{} has too many rows for this platform", path.display()))?;
        if rows == 0 {
            bail!("{} has zero rows", path.display());
        }
        counts.push(rows);
    }
    Ok(counts)
}

fn plan_row_counts(counts: &[usize], limit: Option<usize>) -> Result<Vec<usize>> {
    if counts.is_empty() {
        bail!("no step shards found");
    }
    if let Some(0) = limit {
        bail!("limit must be > 0 when provided");
    }
    if limit.is_none() {
        return Ok(counts.to_vec());
    }
    let mut remaining = limit.unwrap();
    let mut planned = Vec::new();
    for &rows in counts {
        if remaining == 0 {
            break;
        }
        let take = rows.min(remaining);
        planned.push(take);
        remaining = remaining.saturating_sub(take);
    }
    Ok(planned)
}

async fn flush_batch(
    client: &mut grpc::Client,
    pending: &mut Vec<PendingItem>,
    value_meta: &Option<grpc::pb::ValueMetadata>,
    step_state: &mut HashMap<StepKey, StepAccumulator>,
    step_order: &mut VecDeque<StepKey>,
    writer: &mut StepsWriter<KendallRow>,
    stats: &mut Stats,
) -> Result<()> {
    if pending.is_empty() {
        return Ok(());
    }
    let batch = std::mem::take(pending);
    let predictions = send_batch(client, batch, value_meta.as_ref()).await?;
    for pred in predictions {
        let step = step_state.get_mut(&pred.key).ok_or_else(|| {
            anyhow!(
                "received prediction for unknown step (run {}, step {})",
                pred.key.run_id,
                pred.key.step_index
            )
        })?;
        if step.preds.get(pred.move_idx).is_none() {
            bail!(
                "invalid move index {} for step {:?}",
                pred.move_idx,
                pred.key
            );
        }
        step.preds[pred.move_idx] = pred.value;
        step.filled += 1;
    }
    flush_ready_steps(step_order, step_state, writer, stats)
}

async fn send_batch(
    client: &mut grpc::Client,
    batch: Vec<PendingItem>,
    value_meta: Option<&grpc::pb::ValueMetadata>,
) -> Result<Vec<PredictionResult>> {
    if batch.is_empty() {
        return Ok(Vec::new());
    }
    let mut metas = Vec::with_capacity(batch.len());
    let items: Vec<grpc::pb::Item> = batch
        .into_iter()
        .map(|item| {
            metas.push((item.id, item.key, item.move_idx));
            grpc::pb::Item {
                id: item.id,
                board: item.board.to_vec(),
            }
        })
        .collect();
    let req = grpc::pb::InferRequest {
        model_id: String::new(),
        items,
        batch_id: 0,
        return_embedding: false,
        argmax_only: false,
        output_mode: grpc::pb::infer_request::OutputMode::ValueOnly as i32,
        include_value_probs: false,
    };
    let resp = client.infer(req).await?.into_inner();
    if resp.outputs.len() != metas.len() {
        bail!(
            "inference response length {} does not match request {}",
            resp.outputs.len(),
            metas.len()
        );
    }
    if !resp.item_ids.is_empty() && resp.item_ids.len() != metas.len() {
        bail!(
            "item_ids length {} does not match request {}",
            resp.item_ids.len(),
            metas.len()
        );
    }
    let mut results = Vec::with_capacity(metas.len());
    for (idx, (id, key, move_idx)) in metas.into_iter().enumerate() {
        if let Some(returned) = resp.item_ids.get(idx) {
            if *returned != id {
                bail!(
                    "item_id mismatch at index {}: expected {}, got {}",
                    idx,
                    id,
                    returned
                );
            }
        }
        let output = resp
            .outputs
            .get(idx)
            .ok_or_else(|| anyhow!("missing output at index {}", idx))?;
        let value = decode_value_output(output, value_meta)
            .ok_or_else(|| anyhow!("value output missing for item {}", id))?;
        results.push(PredictionResult {
            key,
            move_idx,
            value,
        });
    }
    Ok(results)
}

fn decode_value_output(
    output: &grpc::pb::Output,
    meta: Option<&grpc::pb::ValueMetadata>,
) -> Option<f32> {
    let value = output.value.as_ref()?;
    if let Some(v) = value.value {
        return Some(v);
    }
    if let Some(vx) = value.value_xform {
        return Some(vx);
    }
    if value.probs.is_empty() {
        return None;
    }
    let n = value.probs.len();
    let support_min = meta.and_then(|m| m.support_min).unwrap_or(0.0);
    let support_max = meta
        .and_then(|m| m.support_max)
        .unwrap_or_else(|| if n > 0 { (n - 1) as f32 } else { 0.0 });
    let step = if n > 1 {
        (support_max - support_min) / (n as f32 - 1.0)
    } else {
        0.0
    };
    let mut exp = 0f32;
    for (i, p) in value.probs.iter().copied().enumerate() {
        exp += (support_min + step * i as f32) * p;
    }
    Some(exp)
}

fn flush_ready_steps(
    order: &mut VecDeque<StepKey>,
    steps: &mut HashMap<StepKey, StepAccumulator>,
    writer: &mut StepsWriter<KendallRow>,
    stats: &mut Stats,
) -> Result<()> {
    loop {
        let key = match order.front().copied() {
            Some(k) => k,
            None => break,
        };
        let ready = match steps.get(&key) {
            Some(step) => step.filled == step.expected,
            None => true,
        };
        if !ready {
            break;
        }
        let step = steps
            .remove(&key)
            .ok_or_else(|| anyhow!("missing step state for {:?}", key))?;
        let (row, tau) = build_row(key, &step);
        writer.write(&[row])?;
        stats.record(&step, &row, tau);
        order.pop_front();
    }
    Ok(())
}

fn build_row(key: StepKey, step: &StepAccumulator) -> (KendallRow, Option<f32>) {
    let mask = step.scored_mask & step.legal_mask;
    let tau = kendall_tau(&step.preds, &step.branch_evs, mask);
    let teacher_best = best_move(&step.branch_evs, mask);
    let predicted_best = best_move(&step.preds, mask);
    (
        KendallRow {
            run_id: key.run_id,
            step_index: key.step_index,
            legal_mask: step.legal_mask,
            scored_mask: step.scored_mask,
            teacher_best,
            predicted_best,
            kendall_tau: tau.unwrap_or(f32::NAN),
            value_scores: step.preds,
        },
        tau,
    )
}

fn best_move(values: &[f32; 4], mask: u8) -> u8 {
    let mut best = u8::MAX;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, &v) in values.iter().enumerate() {
        if mask & (1u8 << idx) == 0 {
            continue;
        }
        if !v.is_finite() {
            continue;
        }
        if best == u8::MAX || v > best_val || (v == best_val && (idx as u8) < best) {
            best = idx as u8;
            best_val = v;
        }
    }
    best
}

fn kendall_tau(preds: &[f32; 4], teacher: &[f32; 4], mask: u8) -> Option<f32> {
    let mut indices = Vec::new();
    for idx in 0..4 {
        if mask & (1u8 << idx) != 0 && preds[idx].is_finite() && teacher[idx].is_finite() {
            indices.push(idx);
        }
    }
    let n = indices.len();
    if n < 2 {
        return None;
    }
    let mut concordant = 0i32;
    let mut discordant = 0i32;
    let mut tie_pred = 0i32;
    let mut tie_teacher = 0i32;
    for a in 0..n {
        for b in (a + 1)..n {
            let i = indices[a];
            let j = indices[b];
            let pred_cmp = preds[i]
                .partial_cmp(&preds[j])
                .unwrap_or(std::cmp::Ordering::Equal);
            let teacher_cmp = teacher[i]
                .partial_cmp(&teacher[j])
                .unwrap_or(std::cmp::Ordering::Equal);
            if pred_cmp == std::cmp::Ordering::Equal {
                tie_pred += 1;
            }
            if teacher_cmp == std::cmp::Ordering::Equal {
                tie_teacher += 1;
            }
            match (pred_cmp, teacher_cmp) {
                (std::cmp::Ordering::Greater, std::cmp::Ordering::Greater)
                | (std::cmp::Ordering::Less, std::cmp::Ordering::Less) => {
                    concordant += 1;
                }
                (std::cmp::Ordering::Greater, std::cmp::Ordering::Less)
                | (std::cmp::Ordering::Less, std::cmp::Ordering::Greater) => {
                    discordant += 1;
                }
                _ => {}
            }
        }
    }
    let pairs = (n * (n - 1) / 2) as f64;
    let denom = ((pairs - tie_pred as f64) * (pairs - tie_teacher as f64)).sqrt();
    if denom == 0.0 {
        return None;
    }
    Some(((concordant - discordant) as f64 / denom) as f32)
}
