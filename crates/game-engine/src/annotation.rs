use crate::{config, pipeline};
use anyhow::{Context, Result, anyhow, bail};
use dataset_packer;
use dataset_packer::schema::{AnnotationRow, MacroxueStepRow, SelfplayStepRow};
use dataset_packer::writer::StepsWriter;
use futures::stream::{FuturesUnordered, StreamExt};
use std::fs;
use std::path::{Path, PathBuf};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::feeder::{self, FeederHandle, InferenceOutput};

/// Configuration for running the annotation job.
pub struct JobConfig {
    pub dataset_dir: PathBuf,
    pub output_dir: PathBuf,
    pub overwrite: bool,
    pub limit: Option<usize>,
    pub config: config::Config,
}

/// Run the annotation job end-to-end.
pub async fn run(job: JobConfig) -> Result<()> {
    if job.config.orchestrator.argmax_only {
        bail!("annotation requires full distributions; set argmax_only = false in config");
    }

    fs::create_dir_all(&job.output_dir)
        .with_context(|| format!("failed to create output dir {}", job.output_dir.display()))?;

    let client = pipeline::connect_inference(&job.config.orchestrator.connection).await?;
    let (mut feeder, handle) = pipeline::build_feeder(job.config.orchestrator.batch.clone(), false);
    let cancel = CancellationToken::new();
    feeder.set_cancel_token(cancel.clone());
    let _emb_tx = if job.config.orchestrator.inline_embeddings {
        let (tx, mut rx) = mpsc::channel::<feeder::EmbeddingRow>(8);
        feeder.set_embeddings_channel(tx.clone());
        // Drain embeddings but discard for now
        tokio::spawn(async move { while rx.recv().await.is_some() {} });
        Some(tx)
    } else {
        None
    };
    let feeder_task = feeder.spawn(client);

    let mut writer = StepsWriter::<AnnotationRow>::with_prefix(
        &job.output_dir,
        "annotations",
        Some(job.config.orchestrator.report.shard_max_steps_or_default()),
        job.overwrite,
    )?;

    let step_files = dataset_packer::selfplay::collect_selfplay_step_files(&job.dataset_dir)?;
    if step_files.is_empty() {
        bail!(
            "no steps.npy files found under {}",
            job.dataset_dir.display()
        );
    }
    let kind = detect_kind(&step_files)?;

    copy_metadata(&job.dataset_dir, &job.output_dir, &kind, job.overwrite)?;

    let mut processed: usize = 0;
    let mut inflight = FuturesUnordered::new();
    let max_inflight = pipeline::max_inflight_items(&job.config.orchestrator.batch).max(1);

    for path in step_files {
        match kind {
            DatasetKind::Macroxue => {
                let rows = load_macro_shard(&path)?;
                for row in rows {
                    if job.limit.map_or(false, |lim| processed >= lim) {
                        break;
                    }
                    processed += 1;
                    enqueue_sample(
                        &handle,
                        &mut inflight,
                        max_inflight,
                        StepSample::from_macro(row),
                        &mut writer,
                    )
                    .await?;
                }
            }
            DatasetKind::Selfplay => {
                let rows = dataset_packer::selfplay::load_selfplay_shard(&path)?;
                for row in rows {
                    if job.limit.map_or(false, |lim| processed >= lim) {
                        break;
                    }
                    processed += 1;
                    enqueue_sample(
                        &handle,
                        &mut inflight,
                        max_inflight,
                        StepSample::from_selfplay(row),
                        &mut writer,
                    )
                    .await?;
                }
            }
        }
        if job.limit.map_or(false, |lim| processed >= lim) {
            break;
        }
    }

    drop(handle);
    cancel.cancel();

    while let Some(res) = inflight.next().await {
        let row = res??;
        write_result(row, &mut writer)?;
    }

    writer.finish()?;

    if let Err(e) = feeder_task.await {
        return Err(anyhow!("feeder task failed: {e}"));
    }

    Ok(())
}

async fn enqueue_sample(
    handle: &FeederHandle,
    inflight: &mut FuturesUnordered<tokio::task::JoinHandle<Result<AnnotationRow>>>,
    max_inflight: usize,
    sample: StepSample,
    writer: &mut StepsWriter<AnnotationRow>,
) -> Result<()> {
    while inflight.len() >= max_inflight {
        if let Some(join) = inflight.next().await {
            let row = join??;
            writer.write(std::slice::from_ref(&row))?;
        }
    }
    let id = sample.unique_id();
    let board = sample.board;
    let rx = handle.submit(id, sample.run_id as u32, board).await;
    inflight.push(tokio::spawn(
        async move { consume_result(sample, rx).await },
    ));
    Ok(())
}

async fn consume_result(
    sample: StepSample,
    rx: tokio::sync::oneshot::Receiver<Result<InferenceOutput, tonic::Status>>,
) -> Result<AnnotationRow> {
    let output = rx
        .await
        .map_err(|e| anyhow!("inference oneshot canceled: {e}"))??;
    match output {
        InferenceOutput::Bins(heads) => {
            let mut p1 = [0f32; 4];
            let mut argmax_head = 0u8;
            let mut argmax_p = -1f32;
            for (idx, head_probs) in heads.iter().enumerate() {
                if let Some(val) = head_probs.last() {
                    p1[idx] = *val;
                    if *val > argmax_p {
                        argmax_p = *val;
                        argmax_head = idx as u8;
                    }
                }
            }
            Ok(AnnotationRow {
                run_id: sample.run_id,
                step_index: sample.step_index,
                teacher_move: sample.teacher_move.unwrap_or(255),
                legal_mask: sample.legal_mask.unwrap_or(0),
                argmax_head,
                argmax_prob: argmax_p.max(0.0),
                policy_p1: p1,
            })
        }
        InferenceOutput::Argmax { head: _, _p1 } => Err(anyhow!(
            "inference server replied argmax-only for item {} -- disable argmax_only",
            sample.step_index
        )),
    }
}

fn write_result(row: AnnotationRow, writer: &mut StepsWriter<AnnotationRow>) -> Result<()> {
    writer.write(std::slice::from_ref(&row))?;
    Ok(())
}

fn load_macro_shard(path: &Path) -> Result<Vec<MacroxueStepRow>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("failed to open steps shard {}", path.display()))?;
    let mut reader = std::io::BufReader::new(file);
    let npy = npyz::NpyFile::new(&mut reader)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let rows: Vec<MacroxueStepRow> = npy
        .into_vec()
        .map_err(|err| anyhow!("{}: {err}", path.display()))?;
    Ok(rows)
}

fn detect_kind(step_files: &[PathBuf]) -> Result<DatasetKind> {
    let first = step_files
        .first()
        .ok_or_else(|| anyhow!("no step files available"))?;
    if try_read_macro(first).is_ok() {
        Ok(DatasetKind::Macroxue)
    } else if try_read_selfplay(first).is_ok() {
        Ok(DatasetKind::Selfplay)
    } else {
        Err(anyhow!(
            "unable to determine dataset type from {}",
            first.display()
        ))
    }
}

fn try_read_macro(path: &Path) -> Result<()> {
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    let npy = npyz::NpyFile::new(&mut reader)?;
    let mut data = npy.data::<MacroxueStepRow>()?;
    if data.next().is_some() {
        Ok(())
    } else {
        Err(anyhow!("empty shard"))
    }
}

fn try_read_selfplay(path: &Path) -> Result<()> {
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    let npy = npyz::NpyFile::new(&mut reader)?;
    let mut data = npy.data::<SelfplayStepRow>()?;
    if data.next().is_some() {
        Ok(())
    } else {
        Err(anyhow!("empty shard"))
    }
}

fn copy_metadata(src: &Path, dst: &Path, kind: &DatasetKind, overwrite: bool) -> Result<()> {
    let meta_src = src.join("metadata.db");
    if meta_src.exists() {
        let meta_dst = dst.join("metadata.db");
        if overwrite || !meta_dst.exists() {
            fs::copy(&meta_src, &meta_dst).with_context(|| {
                format!(
                    "failed to copy {} -> {}",
                    meta_src.display(),
                    meta_dst.display()
                )
            })?;
        }
    }
    if matches!(kind, DatasetKind::Macroxue) {
        let val_src = src.join("valuation_types.json");
        if val_src.exists() {
            let val_dst = dst.join("valuation_types.json");
            if overwrite || !val_dst.exists() {
                fs::copy(&val_src, &val_dst).with_context(|| {
                    format!(
                        "failed to copy {} -> {}",
                        val_src.display(),
                        val_dst.display()
                    )
                })?;
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct StepSample {
    run_id: u32,
    step_index: u32,
    teacher_move: Option<u8>,
    legal_mask: Option<u8>,
    board: [u8; 16],
}

impl StepSample {
    fn from_macro(row: MacroxueStepRow) -> Self {
        Self {
            run_id: row.run_id,
            step_index: row.step_index,
            teacher_move: Some(row.move_dir),
            legal_mask: Some(row.ev_legal),
            board: decode_macro_board(row.board, row.tile_65536_mask),
        }
    }

    fn from_selfplay(row: SelfplayStepRow) -> Self {
        Self {
            run_id: row.run_id as u32,
            step_index: row.step_idx,
            teacher_move: None,
            legal_mask: None,
            board: row.exps,
        }
    }

    fn unique_id(&self) -> u64 {
        ((self.run_id as u64) << 32) | (self.step_index as u64)
    }
}

fn decode_macro_board(packed: u64, mask: u16) -> [u8; 16] {
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

#[derive(Clone, Copy)]
enum DatasetKind {
    Macroxue,
    Selfplay,
}
