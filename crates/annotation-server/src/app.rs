use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use dataset_packer::evaluate;
use dataset_packer::macroxue::{self, RunSummary};
use dataset_packer::schema::{AnnotationRow, MacroxueStepRow, annotation_kinds};
use npyz::NpyFile;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::{error, warn};

use crate::student::{StudentBinsRecord, load_student_bins_shards};

#[derive(Clone, Serialize, Deserialize)]
pub struct PolicyKindLegend {
    pub policy_p1: u8,
    pub policy_logprobs: u8,
    pub policy_hard: u8,
    pub policy_student_bins: u8,
}

impl Default for PolicyKindLegend {
    fn default() -> Self {
        Self {
            policy_p1: annotation_kinds::POLICY_P1,
            policy_logprobs: annotation_kinds::POLICY_LOGPROBS,
            policy_hard: annotation_kinds::POLICY_HARD,
            policy_student_bins: annotation_kinds::POLICY_STUDENT_BINS,
        }
    }
}

#[derive(Clone)]
pub struct StepRecord {
    pub step_index: u32,
    pub board: [u8; 16],
    pub board_value: f32,
    pub branch_evs: [Option<f32>; 4],
    pub relative_branch_evs: [Option<i32>; 4],
    pub advantage_branch: [Option<i32>; 4],
    pub legal_mask: u8,
    pub teacher_move: u8,
    pub is_disagreement: bool,
    pub valuation_type: u8,
    pub annotation: Option<AnnotationPayload>,
    pub student_bins: Option<StudentBinsRecord>,
    pub annotation_mask: u8,
}

#[derive(Clone)]
pub struct AnnotationPayload {
    pub policy_kind_mask: u8,
    pub argmax_head: u8,
    pub argmax_prob: f32,
    pub policy_p1: [f32; 4],
    pub policy_logp: [f32; 4],
    pub policy_hard: [f32; 4],
}

pub struct RunEntry {
    pub summary: RunSummary,
    pub steps: Vec<StepRecord>,
    pub policy_kind_mask: u8,
    pub disagreements: Vec<u32>,
    pub disagreement_count: usize,
    pub disagreement_percentage: f32,
}

#[derive(Clone)]
pub struct StudentBinsInfo {
    pub num_bins: usize,
}

#[derive(Clone)]
pub struct AppState {
    pub runs: Arc<Vec<RunEntry>>,
    pub run_index: Arc<HashMap<u32, usize>>,
    pub policy_kind_legend: Arc<PolicyKindLegend>,
    pub valuation_names: Arc<Vec<String>>,
    pub tokenizer: Arc<Option<macroxue::tokenizer::MacroxueTokenizerV2>>,
    pub student_bins: Arc<Option<StudentBinsInfo>>,
}

pub struct DatasetLoad {
    pub runs: Vec<RunEntry>,
    pub policy_kind_legend: PolicyKindLegend,
    pub valuation_names: Vec<String>,
    pub student_bins: Option<StudentBinsInfo>,
}

impl AppState {
    pub fn from_dataset(
        data: DatasetLoad,
        tokenizer: Option<macroxue::tokenizer::MacroxueTokenizerV2>,
    ) -> Self {
        let run_index = data
            .runs
            .iter()
            .enumerate()
            .map(|(idx, entry)| (entry.summary.run_id, idx))
            .collect::<HashMap<_, _>>();
        Self {
            runs: Arc::new(data.runs),
            run_index: Arc::new(run_index),
            policy_kind_legend: Arc::new(data.policy_kind_legend),
            valuation_names: Arc::new(data.valuation_names),
            tokenizer: Arc::new(tokenizer),
            student_bins: Arc::new(data.student_bins),
        }
    }
}

pub fn load_dataset(dataset_dir: &Path, annotations_dir: &Path) -> Result<DatasetLoad> {
    let runs = macroxue::load_runs(dataset_dir)?;
    let step_map = load_macro_steps(dataset_dir)?;
    let annotations = load_annotations(annotations_dir)?;
    let annotations_map: HashMap<(u32, u32), AnnotationRow> = annotations
        .iter()
        .map(|row| ((row.run_id, row.step_index), *row))
        .collect();
    let manifest = load_manifest(annotations_dir).unwrap_or_default();
    let valuation_names = macroxue::load_valuation_names(dataset_dir).unwrap_or_else(|err| {
        warn!("failed to load valuation_types.json: {err}");
        vec!["search".to_string()]
    });

    let student_bins_data = load_student_bins_shards(annotations_dir)?;
    let (student_bins_info, student_bins_map) = match student_bins_data {
        Some(data) => {
            let info = StudentBinsInfo {
                num_bins: data.num_bins,
            };
            let mut map = HashMap::new();
            let expected = annotations.len();
            let actual = data.records.len();
            if actual != expected {
                warn!(
                    "student sidecar length mismatch: annotations={} bins={} (truncating to min)",
                    expected, actual
                );
            }
            let count = expected.min(actual);
            for idx in 0..count {
                let row = annotations[idx];
                map.insert((row.run_id, row.step_index), data.records[idx].clone());
            }
            (Some(info), Some(map))
        }
        None => (None, None),
    };

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
                let board = macroxue::decode_board(row.board, row.tile_65536_mask);
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

                let annotation_row = annotations_map.get(&(row.run_id, row.step_index)).copied();
                let mut annotation_mask = 0;
                let mut annotation = None;
                if let Some(ann) = annotation_row {
                    annotation_mask = ann.policy_kind_mask;
                    annotation = Some(AnnotationPayload {
                        policy_kind_mask: ann.policy_kind_mask,
                        argmax_head: ann.argmax_head,
                        argmax_prob: ann.argmax_prob,
                        policy_p1: ann.policy_p1,
                        policy_logp: ann.policy_logp,
                        policy_hard: ann.policy_hard,
                    });
                }

                let student_bins = student_bins_map
                    .as_ref()
                    .and_then(|map| map.get(&(row.run_id, row.step_index)).cloned());
                if student_bins.is_some() {
                    annotation_mask |= annotation_kinds::POLICY_STUDENT_BINS;
                }
                if let Some(payload) = annotation.as_mut() {
                    payload.policy_kind_mask = annotation_mask;
                }

                let is_disagreement = annotation
                    .as_ref()
                    .map(|ann| row.move_dir != 255 && row.move_dir != ann.argmax_head)
                    .unwrap_or(false);

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
                    student_bins,
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
        policy_kind_legend: manifest.legend.clone(),
        valuation_names,
        student_bins: student_bins_info.or(manifest.student_bins),
    })
}

#[derive(Deserialize)]
struct ManifestFile {
    #[serde(default)]
    kinds: Option<PolicyKindLegend>,
    #[serde(default)]
    runs: Vec<ManifestEntryFile>,
    #[serde(default)]
    student_bins: Option<StudentBinsFile>,
}

#[derive(Deserialize)]
struct ManifestEntryFile {
    run_id: u32,
    policy_kind_mask: u8,
}

#[derive(Deserialize)]
struct StudentBinsFile {
    #[serde(default)]
    num_bins: Option<usize>,
}

struct ManifestData {
    legend: PolicyKindLegend,
    run_masks: HashMap<u32, u8>,
    student_bins: Option<StudentBinsInfo>,
}

impl Default for ManifestData {
    fn default() -> Self {
        Self {
            legend: PolicyKindLegend::default(),
            run_masks: HashMap::new(),
            student_bins: None,
        }
    }
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
    let student_bins = manifest
        .student_bins
        .and_then(|info| info.num_bins.map(|bins| StudentBinsInfo { num_bins: bins }));
    Ok(ManifestData {
        legend,
        run_masks,
        student_bins,
    })
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

fn load_annotations(dir: &Path) -> Result<Vec<AnnotationRow>> {
    let mut rows = Vec::new();
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
            rows.push(row);
        }
    }
    Ok(rows)
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
                let suffix = &name[prefix.len() + 1..name.len() - ".npy".len()];
                if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
                    return Some(entry.path());
                }
                return None;
            }
            None
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
