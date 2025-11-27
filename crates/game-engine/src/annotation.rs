use crate::{config, grpc, pipeline};
use anyhow::{Context, Result, anyhow, bail};
use dataset_packer;
use dataset_packer::macroxue;
use dataset_packer::schema::{
    AnnotationRow, MAX_STUDENT_BINS, MAX_VALUE_BINS, MacroxueStepRow, SelfplayStepRow,
    ValueAnnotationRow, annotation_kinds, value_annotation_kinds,
};
use dataset_packer::writer::StepsWriter;
use futures::stream::{FuturesUnordered, StreamExt};
use log::{info, warn};
use npyz::WriterBuilder;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use tokio::sync::mpsc;

use crate::feeder::{self, InferenceOutput};
use crate::grpc::pb;
use tonic::Code;

struct StudentBinsPayload {
    bin_count: usize,
    probs: [[f32; MAX_STUDENT_BINS]; 4],
}

struct ValueProbsPayload {
    bin_count: usize,
    probs: Vec<f32>,
}

struct ValuePayload {
    row: ValueAnnotationRow,
    probs: Option<ValueProbsPayload>,
}

struct AnnotatedOutput {
    seq: u64,
    annotation: AnnotationRow,
    bins: Option<StudentBinsPayload>,
    value: Option<ValuePayload>,
}

#[derive(Clone, Default)]
struct ServerCapabilities {
    has_value_head: bool,
    model_metadata: Option<pb::ModelMetadata>,
}

/// Configuration for running the annotation job.
pub struct JobConfig {
    pub dataset_dir: PathBuf,
    pub output_dir: PathBuf,
    pub overwrite: bool,
    pub limit: Option<usize>,
    pub config: config::AnnotationConfig,
    pub write_value_sidecar: bool,
}

/// Run the annotation job end-to-end.
pub async fn run(job: JobConfig) -> Result<()> {
    if job.config.orchestrator.argmax_only {
        bail!("annotation requires full distributions; set argmax_only = false in config");
    }

    fs::create_dir_all(&job.output_dir)
        .with_context(|| format!("failed to create output dir {}", job.output_dir.display()))?;

    let client = pipeline::connect_inference(&job.config.orchestrator.connection).await?;
    let mut describe_client = client.clone();
    let describe_caps = describe_once(&mut describe_client).await;
    let value_request_enabled = job.write_value_sidecar
        && describe_caps
            .as_ref()
            .map(|caps| {
                caps.has_value_head
                    || caps
                        .model_metadata
                        .as_ref()
                        .and_then(|m| m.value.as_ref())
                        .is_some()
            })
            .unwrap_or(true);
    let capabilities = describe_caps.unwrap_or_default();

    let (mut feeder, handle) = pipeline::build_feeder_with_value(
        job.config.orchestrator.batch.clone(),
        false,
        value_request_enabled,
    );
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

    info!(
        "Starting annotation: dataset={} output={} overwrite={} limit={:?}",
        job.dataset_dir.display(),
        job.output_dir.display(),
        job.overwrite,
        job.limit
    );

    let shard_max = job.config.orchestrator.report.shard_max_steps_or_default();
    let mut writer = StepsWriter::<AnnotationRow>::with_prefix(
        &job.output_dir,
        "annotations",
        Some(shard_max),
        job.overwrite,
    )?;
    let mut student_writer =
        StudentBinsWriter::new(&job.output_dir, Some(shard_max), job.overwrite)?;
    let mut value_writer = if value_request_enabled {
        Some(StepsWriter::<ValueAnnotationRow>::with_prefix(
            &job.output_dir,
            "annotations-value",
            Some(shard_max),
            job.overwrite,
        )?)
    } else {
        None
    };
    let mut value_probs_writer = if value_request_enabled {
        Some(ValueProbsWriter::new(
            &job.output_dir,
            Some(shard_max),
            job.overwrite,
        )?)
    } else {
        None
    };

    let mut run_masks: HashMap<u32, u8> = HashMap::new();
    let mut value_run_masks: HashMap<u32, u8> = HashMap::new();
    let value_metadata = capabilities
        .model_metadata
        .as_ref()
        .and_then(|m| m.value.clone());
    let mut value_seen = false;

    let step_files = dataset_packer::selfplay::collect_selfplay_step_files(&job.dataset_dir)?;
    if step_files.is_empty() {
        bail!(
            "no steps.npy files found under {}",
            job.dataset_dir.display()
        );
    }
    let kind = detect_kind(&step_files)?;

    info!(
        "Discovered {} step shards; dataset kind: {:?}",
        step_files.len(),
        kind
    );

    copy_metadata(&job.dataset_dir, &job.output_dir, &kind, job.overwrite)?;

    let max_inflight = pipeline::max_inflight_items(&job.config.orchestrator.batch).max(1);
    let mut inflight: FuturesUnordered<_> = FuturesUnordered::new();
    let mut buffer: BTreeMap<u64, AnnotatedOutput> = BTreeMap::new();
    let mut next_seq_to_write: u64 = 0;
    let mut next_seq: u64 = 0;
    let mut processed: usize = 0;

    for path in step_files {
        match kind {
            DatasetKind::Macroxue => {
                let rows = load_macro_shard(&path)?;
                for row in rows {
                    if job.limit.map_or(false, |lim| processed >= lim) {
                        break;
                    }
                    let seq = next_seq;
                    next_seq += 1;
                    processed += 1;
                    let sample = StepSample::from_macro(row, seq);
                    let rx = handle
                        .submit(sample.unique_id(), sample.run_id, sample.board)
                        .await;
                    inflight.push(tokio::spawn(
                        async move { consume_result(sample, rx).await },
                    ));
                    while inflight.len() >= max_inflight {
                        if let Some(join) = inflight.next().await {
                            let output = join??;
                            buffer.insert(output.seq, output);
                            flush_ready_outputs(
                                &mut buffer,
                                &mut next_seq_to_write,
                                &mut writer,
                                &mut student_writer,
                                &mut run_masks,
                                value_writer.as_mut(),
                                value_probs_writer.as_mut(),
                                &mut value_run_masks,
                                &mut value_seen,
                            )?;
                        }
                    }
                }
            }
            DatasetKind::Selfplay => {
                let rows = dataset_packer::selfplay::load_selfplay_shard(&path)?;
                for row in rows {
                    if job.limit.map_or(false, |lim| processed >= lim) {
                        break;
                    }
                    let seq = next_seq;
                    next_seq += 1;
                    processed += 1;
                    let sample = StepSample::from_selfplay(row, seq);
                    let rx = handle
                        .submit(sample.unique_id(), sample.run_id, sample.board)
                        .await;
                    inflight.push(tokio::spawn(
                        async move { consume_result(sample, rx).await },
                    ));
                    while inflight.len() >= max_inflight {
                        if let Some(join) = inflight.next().await {
                            let output = join??;
                            buffer.insert(output.seq, output);
                            flush_ready_outputs(
                                &mut buffer,
                                &mut next_seq_to_write,
                                &mut writer,
                                &mut student_writer,
                                &mut run_masks,
                                value_writer.as_mut(),
                                value_probs_writer.as_mut(),
                                &mut value_run_masks,
                                &mut value_seen,
                            )?;
                        }
                    }
                }
            }
        }
        if job.limit.map_or(false, |lim| processed >= lim) {
            break;
        }
    }

    drop(handle);

    while let Some(join) = inflight.next().await {
        let output = join??;
        buffer.insert(output.seq, output);
        flush_ready_outputs(
            &mut buffer,
            &mut next_seq_to_write,
            &mut writer,
            &mut student_writer,
            &mut run_masks,
            value_writer.as_mut(),
            value_probs_writer.as_mut(),
            &mut value_run_masks,
            &mut value_seen,
        )?;
    }

    flush_ready_outputs(
        &mut buffer,
        &mut next_seq_to_write,
        &mut writer,
        &mut student_writer,
        &mut run_masks,
        value_writer.as_mut(),
        value_probs_writer.as_mut(),
        &mut value_run_masks,
        &mut value_seen,
    )?;

    let value_bin_count = if let Some(writer) = value_probs_writer.as_mut() {
        let bins = writer.bin_count().unwrap_or(0);
        writer.finish()?;
        bins
    } else {
        0
    };
    if let Some(writer) = value_writer {
        writer.finish()?;
    }
    writer.finish()?;
    student_writer.finish()?;
    let student_bin_count = student_writer.bin_count().unwrap_or(0);

    write_manifest(
        &job.output_dir,
        &run_masks,
        student_bin_count,
        &value_run_masks,
        value_bin_count,
        value_seen && value_request_enabled,
        value_metadata.as_ref(),
        job.overwrite,
    )?;

    info!(
        "Annotation finished: wrote {} steps across {} runs into {}",
        processed,
        run_masks.len(),
        job.output_dir.display()
    );

    if let Err(e) = feeder_task.await {
        return Err(anyhow!("feeder task failed: {e}"));
    }

    Ok(())
}

async fn consume_result(
    sample: StepSample,
    rx: tokio::sync::oneshot::Receiver<Result<InferenceOutput, tonic::Status>>,
) -> Result<AnnotatedOutput> {
    let output = rx.await.map_err(|e| {
        anyhow!(
            "inference oneshot canceled for run {} step {} (id {}) : {}",
            sample.run_id,
            sample.step_index,
            sample.unique_id(),
            e
        )
    })??;
    match output {
        InferenceOutput::Bins { heads, value } => {
            let mut p1 = [0f32; 4];
            let mut logp = [f32::NEG_INFINITY; 4];
            let mut argmax_head = 0u8;
            let mut argmax_p = -1f32;
            let mut has_policy = false;
            let mut bins_payload = None;
            let mut bin_count = 0usize;
            if !heads.is_empty() {
                bin_count = heads[0].len();
                if bin_count > MAX_STUDENT_BINS {
                    return Err(anyhow!(
                        "student bin count {} exceeds MAX_STUDENT_BINS ({})",
                        bin_count,
                        MAX_STUDENT_BINS
                    ));
                }
                for head in &heads {
                    if head.len() != bin_count {
                        return Err(anyhow!(
                            "inconsistent bin counts: expected {}, got {}",
                            bin_count,
                            head.len()
                        ));
                    }
                }
            }
            let mut branch_probs = [[0f32; MAX_STUDENT_BINS]; 4];
            for (idx, head_probs) in heads.iter().enumerate() {
                if let Some(val) = head_probs.last() {
                    let prob = *val;
                    p1[idx] = prob;
                    logp[idx] = if prob > 0.0 {
                        prob.ln()
                    } else {
                        f32::NEG_INFINITY
                    };
                    has_policy = true;
                    if prob > argmax_p {
                        argmax_p = prob;
                        argmax_head = idx as u8;
                    }
                }
                for (bin_idx, &prob) in head_probs.iter().enumerate() {
                    branch_probs[idx][bin_idx] = prob;
                }
            }
            let mut policy_kind_mask = 0u8;
            if has_policy {
                policy_kind_mask |= annotation_kinds::POLICY_P1;
                policy_kind_mask |= annotation_kinds::POLICY_LOGPROBS;
            }
            let mut policy_hard = [0f32; 4];
            if let Some(move_dir) = sample.teacher_move {
                if move_dir < 4 {
                    policy_hard[move_dir as usize] = 1.0;
                    policy_kind_mask |= annotation_kinds::POLICY_HARD;
                }
            }
            if bin_count > 0 {
                bins_payload = Some(StudentBinsPayload {
                    bin_count,
                    probs: branch_probs,
                });
            }
            let mut value_payload = None;
            if let Some(v) = value {
                let probs_len = v.probs.len();
                if probs_len > MAX_VALUE_BINS {
                    return Err(anyhow!(
                        "value bin count {} exceeds MAX_VALUE_BINS ({})",
                        probs_len,
                        MAX_VALUE_BINS
                    ));
                }
                let row = ValueAnnotationRow {
                    run_id: sample.run_id,
                    step_index: sample.step_index,
                    value: v.value.unwrap_or(f32::NAN),
                    value_xform: v.value_xform.unwrap_or(f32::NAN),
                };
                let probs = if probs_len > 0 {
                    Some(ValueProbsPayload {
                        bin_count: probs_len,
                        probs: v.probs,
                    })
                } else {
                    None
                };
                value_payload = Some(ValuePayload { row, probs });
            }
            Ok(AnnotatedOutput {
                seq: sample.seq,
                annotation: AnnotationRow {
                    run_id: sample.run_id,
                    step_index: sample.step_index,
                    teacher_move: sample.teacher_move.unwrap_or(255),
                    legal_mask: sample.legal_mask.unwrap_or(0),
                    policy_kind_mask,
                    argmax_head,
                    argmax_prob: argmax_p.max(0.0),
                    policy_p1: p1,
                    policy_logp: logp,
                    policy_hard,
                },
                bins: bins_payload,
                value: value_payload,
            })
        }
        InferenceOutput::Argmax { head: _, _p1 } => Err(anyhow!(
            "inference server replied argmax-only for item {} -- disable argmax_only",
            sample.step_index
        )),
    }
}

fn write_result(
    output: AnnotatedOutput,
    writer: &mut StepsWriter<AnnotationRow>,
    bins_writer: &mut StudentBinsWriter,
    run_masks: &mut HashMap<u32, u8>,
    value_writer: Option<&mut StepsWriter<ValueAnnotationRow>>,
    value_bins_writer: Option<&mut ValueProbsWriter>,
    value_run_masks: &mut HashMap<u32, u8>,
    value_seen: &mut bool,
) -> Result<()> {
    let AnnotatedOutput {
        seq: _,
        annotation,
        bins,
        value,
    } = output;
    writer.write(std::slice::from_ref(&annotation))?;
    let entry = run_masks
        .entry(annotation.run_id)
        .and_modify(|mask| *mask |= annotation.policy_kind_mask)
        .or_insert(annotation.policy_kind_mask);
    if let Some(payload) = bins {
        bins_writer.write(payload.bin_count, &payload.probs)?;
        *entry |= annotation_kinds::POLICY_STUDENT_BINS;
    }
    if let Some(payload) = value {
        if let Some(writer) = value_writer {
            writer.write(std::slice::from_ref(&payload.row))?;
            let entry = value_run_masks.entry(payload.row.run_id).or_insert(0);
            *entry |= value_annotation_kinds::VALUE;
            *value_seen = true;
        }
        if let (Some(probs), Some(writer)) = (payload.probs.as_ref(), value_bins_writer) {
            writer.write(probs.bin_count, &probs.probs)?;
            let entry = value_run_masks.entry(payload.row.run_id).or_insert(0);
            *entry |= value_annotation_kinds::VALUE_PROBS;
            *value_seen = true;
        }
    }
    Ok(())
}

fn flush_ready_outputs(
    buffer: &mut BTreeMap<u64, AnnotatedOutput>,
    next_seq: &mut u64,
    writer: &mut StepsWriter<AnnotationRow>,
    bins_writer: &mut StudentBinsWriter,
    run_masks: &mut HashMap<u32, u8>,
    mut value_writer: Option<&mut StepsWriter<ValueAnnotationRow>>,
    mut value_bins_writer: Option<&mut ValueProbsWriter>,
    value_run_masks: &mut HashMap<u32, u8>,
    value_seen: &mut bool,
) -> Result<()> {
    while let Some(output) = buffer.remove(&*next_seq) {
        write_result(
            output,
            writer,
            bins_writer,
            run_masks,
            value_writer.as_deref_mut(),
            value_bins_writer.as_deref_mut(),
            value_run_masks,
            value_seen,
        )?;
        *next_seq += 1;
    }
    Ok(())
}

#[derive(Serialize)]
struct ManifestKinds {
    policy_p1: u8,
    policy_logprobs: u8,
    policy_hard: u8,
    policy_student_bins: u8,
}

#[derive(Serialize)]
struct ManifestEntry {
    run_id: u32,
    policy_kind_mask: u8,
}

#[derive(Serialize)]
struct StudentBinsManifest {
    dtype: String,
    num_bins: usize,
}

#[derive(Serialize)]
struct ValueKinds {
    value: u8,
    value_probs: u8,
}

#[derive(Serialize)]
struct ValueManifestEntry {
    run_id: u32,
    value_kind_mask: u8,
}

#[derive(Serialize)]
struct ValueMetadataSummary {
    #[serde(skip_serializing_if = "Option::is_none")]
    objective: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vocab_size: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vocab_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    support_min: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    support_max: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    transform_epsilon: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    apply_transform: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    target: Option<String>,
}

#[derive(Serialize)]
struct ValueManifest {
    kinds: ValueKinds,
    runs: Vec<ValueManifestEntry>,
    num_bins: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<ValueMetadataSummary>,
}

#[derive(Serialize)]
struct AnnotationManifest {
    kinds: ManifestKinds,
    runs: Vec<ManifestEntry>,
    student_bins: StudentBinsManifest,
    #[serde(skip_serializing_if = "Option::is_none")]
    value: Option<ValueManifest>,
}

fn write_manifest(
    output_dir: &Path,
    run_masks: &HashMap<u32, u8>,
    num_bins: usize,
    value_run_masks: &HashMap<u32, u8>,
    value_num_bins: usize,
    value_enabled: bool,
    value_metadata: Option<&pb::ValueMetadata>,
    overwrite: bool,
) -> Result<()> {
    let mut runs: Vec<ManifestEntry> = run_masks
        .iter()
        .map(|(run_id, mask)| ManifestEntry {
            run_id: *run_id,
            policy_kind_mask: *mask,
        })
        .collect();
    runs.sort_by_key(|entry| entry.run_id);

    let mut value_runs: Vec<ValueManifestEntry> = value_run_masks
        .iter()
        .map(|(run_id, mask)| ValueManifestEntry {
            run_id: *run_id,
            value_kind_mask: *mask,
        })
        .collect();
    value_runs.sort_by_key(|entry| entry.run_id);

    let value_manifest = if value_enabled {
        Some(ValueManifest {
            kinds: ValueKinds {
                value: value_annotation_kinds::VALUE,
                value_probs: value_annotation_kinds::VALUE_PROBS,
            },
            runs: value_runs,
            num_bins: value_num_bins,
            metadata: value_metadata.map(value_meta_summary_from_proto),
        })
    } else {
        None
    };

    let manifest = AnnotationManifest {
        kinds: ManifestKinds {
            policy_p1: annotation_kinds::POLICY_P1,
            policy_logprobs: annotation_kinds::POLICY_LOGPROBS,
            policy_hard: annotation_kinds::POLICY_HARD,
            policy_student_bins: annotation_kinds::POLICY_STUDENT_BINS,
        },
        runs,
        student_bins: StudentBinsManifest {
            dtype: "f32".to_string(),
            num_bins,
        },
        value: value_manifest,
    };

    let json = serde_json::to_vec_pretty(&manifest)?;
    let path = output_dir.join("annotation_manifest.json");
    if path.exists() && !overwrite {
        bail!(
            "annotation_manifest.json already exists in {} (run with --overwrite to replace)",
            output_dir.display()
        );
    }
    fs::write(&path, json)
        .with_context(|| format!("failed to write annotation manifest to {}", path.display()))?;
    Ok(())
}

fn value_meta_summary_from_proto(meta: &pb::ValueMetadata) -> ValueMetadataSummary {
    ValueMetadataSummary {
        objective: if meta.objective.is_empty() {
            None
        } else {
            Some(meta.objective.clone())
        },
        vocab_size: if meta.vocab_size == 0 {
            None
        } else {
            Some(meta.vocab_size)
        },
        vocab_type: if meta.vocab_type.is_empty() {
            None
        } else {
            Some(meta.vocab_type.clone())
        },
        support_min: meta.support_min,
        support_max: meta.support_max,
        transform_epsilon: meta.transform_epsilon,
        apply_transform: meta.apply_transform,
        target: if meta.target.is_empty() {
            None
        } else {
            Some(meta.target.clone())
        },
    }
}

struct StudentBinsWriter {
    output_dir: PathBuf,
    shard_max: Option<usize>,
    shard_idx: usize,
    steps_in_shard: usize,
    bin_count: Option<usize>,
    buffer: Vec<f32>,
}

impl StudentBinsWriter {
    fn new(output_dir: &Path, shard_max: Option<usize>, overwrite: bool) -> Result<Self> {
        let output_dir = output_dir.to_path_buf();
        let prefix = Self::prefix();
        if overwrite {
            Self::clean_existing(&output_dir, prefix)?;
        } else {
            Self::check_no_existing(&output_dir, prefix)?;
        }
        Ok(Self {
            output_dir,
            shard_max,
            shard_idx: 0,
            steps_in_shard: 0,
            bin_count: None,
            buffer: Vec::new(),
        })
    }

    fn write(&mut self, bin_count: usize, probs: &[[f32; MAX_STUDENT_BINS]; 4]) -> Result<()> {
        if bin_count > MAX_STUDENT_BINS {
            bail!(
                "bin count {} exceeds MAX_STUDENT_BINS ({})",
                bin_count,
                MAX_STUDENT_BINS
            );
        }
        match self.bin_count {
            Some(existing) if existing != bin_count => {
                bail!(
                    "student bin count mismatch: expected {}, got {}",
                    existing,
                    bin_count
                );
            }
            None => {
                self.bin_count = Some(bin_count);
            }
            _ => {}
        }

        self.buffer.reserve(4 * bin_count);
        for head in 0..4 {
            self.buffer.extend_from_slice(&probs[head][..bin_count]);
        }
        self.steps_in_shard += 1;

        if let Some(limit) = self.shard_max {
            if self.steps_in_shard >= limit {
                self.flush_current_shard()?;
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        self.flush_current_shard()?;
        Ok(())
    }

    fn bin_count(&self) -> Option<usize> {
        self.bin_count
    }

    fn flush_current_shard(&mut self) -> Result<()> {
        if self.steps_in_shard == 0 {
            return Ok(());
        }
        let bin_count = self
            .bin_count
            .ok_or_else(|| anyhow!("student bin writer flushed before any bins recorded"))?;
        let prefix = Self::prefix();
        let file_name = if self.shard_max.is_some() {
            format!("{prefix}-{idx:05}.npy", idx = self.shard_idx)
        } else {
            format!("{prefix}.npy")
        };
        let tmp_name = format!("{file_name}.tmp");
        let final_path = self.output_dir.join(&file_name);
        let tmp_path = self.output_dir.join(tmp_name);
        let file = std::io::BufWriter::new(
            std::fs::File::create(&tmp_path)
                .with_context(|| format!("failed to create {}", tmp_path.display()))?,
        );
        let mut writer = npyz::WriteOptions::new()
            .default_dtype()
            .shape(&[self.steps_in_shard as u64, 4, bin_count as u64])
            .writer(file)
            .begin_nd()?;
        writer.extend(self.buffer.iter().copied())?;
        writer.finish()?;
        std::fs::rename(&tmp_path, &final_path).with_context(|| {
            format!(
                "failed to rename {} to {}",
                tmp_path.display(),
                final_path.display()
            )
        })?;

        self.buffer.clear();
        self.steps_in_shard = 0;
        if self.shard_max.is_some() {
            self.shard_idx += 1;
        }
        Ok(())
    }

    fn clean_existing(dir: &Path, prefix: &str) -> Result<()> {
        let single = dir.join(format!("{prefix}.npy"));
        if single.exists() {
            std::fs::remove_file(&single)
                .with_context(|| format!("failed to remove {}", single.display()))?;
        }
        if dir.exists() {
            for entry in std::fs::read_dir(dir)
                .with_context(|| format!("failed to read {}", dir.display()))?
            {
                let entry = entry?;
                if let Some(name) = entry.file_name().to_str() {
                    if name.starts_with(prefix) && name.ends_with(".npy") {
                        std::fs::remove_file(entry.path()).with_context(|| {
                            format!("failed to remove {}", entry.path().display())
                        })?;
                    }
                }
            }
        }
        Ok(())
    }

    fn check_no_existing(dir: &Path, prefix: &str) -> Result<()> {
        let single = dir.join(format!("{prefix}.npy"));
        if single.exists() {
            bail!(
                "{} already exists in {} (use --overwrite)",
                single.file_name().unwrap().to_string_lossy(),
                dir.display()
            );
        }
        if dir.exists() {
            for entry in std::fs::read_dir(dir)
                .with_context(|| format!("failed to read {}", dir.display()))?
            {
                let entry = entry?;
                if let Some(name) = entry.file_name().to_str() {
                    if name.starts_with(prefix) && name.ends_with(".npy") {
                        bail!(
                            "found existing {} in {} (use --overwrite)",
                            name,
                            dir.display()
                        );
                    }
                }
            }
        }
        Ok(())
    }

    fn prefix() -> &'static str {
        "annotations-student"
    }
}

struct ValueProbsWriter {
    output_dir: PathBuf,
    shard_max: Option<usize>,
    shard_idx: usize,
    steps_in_shard: usize,
    bin_count: Option<usize>,
    buffer: Vec<f32>,
}

impl ValueProbsWriter {
    fn new(output_dir: &Path, shard_max: Option<usize>, overwrite: bool) -> Result<Self> {
        let output_dir = output_dir.to_path_buf();
        let prefix = Self::prefix();
        if overwrite {
            Self::clean_existing(&output_dir, prefix)?;
        } else {
            Self::check_no_existing(&output_dir, prefix)?;
        }
        Ok(Self {
            output_dir,
            shard_max,
            shard_idx: 0,
            steps_in_shard: 0,
            bin_count: None,
            buffer: Vec::new(),
        })
    }

    fn write(&mut self, bin_count: usize, probs: &[f32]) -> Result<()> {
        if bin_count == 0 {
            bail!("value bin count must be > 0");
        }
        if bin_count > MAX_VALUE_BINS {
            bail!(
                "value bin count {} exceeds MAX_VALUE_BINS ({})",
                bin_count,
                MAX_VALUE_BINS
            );
        }
        if probs.len() != bin_count {
            bail!(
                "value probs length {} does not match bin_count {}",
                probs.len(),
                bin_count
            );
        }
        match self.bin_count {
            Some(existing) if existing != bin_count => {
                bail!(
                    "value bin count mismatch: expected {}, got {}",
                    existing,
                    bin_count
                );
            }
            None => self.bin_count = Some(bin_count),
            _ => {}
        }

        self.buffer.extend_from_slice(probs);
        self.steps_in_shard += 1;

        if let Some(limit) = self.shard_max {
            if self.steps_in_shard >= limit {
                self.flush_current_shard()?;
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        self.flush_current_shard()?;
        Ok(())
    }

    fn bin_count(&self) -> Option<usize> {
        self.bin_count
    }

    fn flush_current_shard(&mut self) -> Result<()> {
        if self.steps_in_shard == 0 {
            return Ok(());
        }
        let bin_count = self
            .bin_count
            .ok_or_else(|| anyhow!("value writer flushed before any bins recorded"))?;
        let prefix = Self::prefix();
        let file_name = if self.shard_max.is_some() {
            format!("{prefix}-{idx:05}.npy", idx = self.shard_idx)
        } else {
            format!("{prefix}.npy")
        };
        let tmp_name = format!("{file_name}.tmp");
        let final_path = self.output_dir.join(&file_name);
        let tmp_path = self.output_dir.join(tmp_name);
        let file = std::io::BufWriter::new(
            std::fs::File::create(&tmp_path)
                .with_context(|| format!("failed to create {}", tmp_path.display()))?,
        );
        let mut writer = npyz::WriteOptions::new()
            .default_dtype()
            .shape(&[self.steps_in_shard as u64, bin_count as u64])
            .writer(file)
            .begin_nd()?;
        writer.extend(self.buffer.iter().copied())?;
        writer.finish()?;
        std::fs::rename(&tmp_path, &final_path).with_context(|| {
            format!(
                "failed to rename {} to {}",
                tmp_path.display(),
                final_path.display()
            )
        })?;

        self.buffer.clear();
        self.steps_in_shard = 0;
        if self.shard_max.is_some() {
            self.shard_idx += 1;
        }
        Ok(())
    }

    fn clean_existing(dir: &Path, prefix: &str) -> Result<()> {
        let single = dir.join(format!("{prefix}.npy"));
        if single.exists() {
            std::fs::remove_file(&single)
                .with_context(|| format!("failed to remove {}", single.display()))?;
        }
        if dir.exists() {
            for entry in std::fs::read_dir(dir)
                .with_context(|| format!("failed to read {}", dir.display()))?
            {
                let entry = entry?;
                if let Some(name) = entry.file_name().to_str() {
                    if name.starts_with(prefix) && name.ends_with(".npy") {
                        std::fs::remove_file(entry.path()).with_context(|| {
                            format!("failed to remove {}", entry.path().display())
                        })?;
                    }
                }
            }
        }
        Ok(())
    }

    fn check_no_existing(dir: &Path, prefix: &str) -> Result<()> {
        let single = dir.join(format!("{prefix}.npy"));
        if single.exists() {
            bail!(
                "{} already exists in {} (use --overwrite)",
                single.file_name().unwrap().to_string_lossy(),
                dir.display()
            );
        }
        if dir.exists() {
            for entry in std::fs::read_dir(dir)
                .with_context(|| format!("failed to read {}", dir.display()))?
            {
                let entry = entry?;
                if let Some(name) = entry.file_name().to_str() {
                    if name.starts_with(prefix) && name.ends_with(".npy") {
                        bail!(
                            "found existing {} in {} (use --overwrite)",
                            name,
                            dir.display()
                        );
                    }
                }
            }
        }
        Ok(())
    }

    fn prefix() -> &'static str {
        "annotations-value-bins"
    }
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

async fn describe_once(client: &mut grpc::Client) -> Option<ServerCapabilities> {
    match client.describe(pb::DescribeRequest {}).await {
        Ok(resp) => {
            let resp = resp.into_inner();
            Some(ServerCapabilities {
                has_value_head: resp.has_value_head,
                model_metadata: resp.model_metadata,
            })
        }
        Err(status) => {
            if status.code() != Code::Unimplemented {
                warn!("Describe RPC failed: {}", status);
            }
            None
        }
    }
}

#[derive(Clone, Copy)]
struct StepSample {
    seq: u64,
    run_id: u32,
    step_index: u32,
    teacher_move: Option<u8>,
    legal_mask: Option<u8>,
    board: [u8; 16],
}

impl StepSample {
    fn from_macro(row: MacroxueStepRow, seq: u64) -> Self {
        Self {
            seq,
            run_id: row.run_id,
            step_index: row.step_index,
            teacher_move: Some(row.move_dir),
            legal_mask: Some(row.ev_legal),
            board: macroxue::decode_board(row.board, row.tile_65536_mask),
        }
    }

    fn from_selfplay(row: SelfplayStepRow, seq: u64) -> Self {
        Self {
            seq,
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

#[derive(Clone, Copy, Debug)]
enum DatasetKind {
    Macroxue,
    Selfplay,
}
