use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use rusqlite::{Connection, OptionalExtension, params};

/// Summary of a packed run extracted from the metadata sidecar.
#[derive(Debug, Clone)]
pub struct RunSummary {
    /// Contiguous run identifier assigned during packing.
    pub run_id: u32,
    /// PRNG seed recorded in the sidecar.
    pub seed: u64,
    /// Number of step rows emitted for this run.
    pub steps: usize,
    /// Final score reported by the Macroxue engine.
    pub max_score: u64,
    /// Highest tile (2^k) reached in the run.
    pub highest_tile: u32,
}

pub(crate) fn write_metadata(
    out_dir: &Path,
    runs: &[RunSummary],
    overwrite: bool,
    has_cumulative_reward: bool,
) -> Result<()> {
    let path = out_dir.join("metadata.db");
    if path.exists() {
        if overwrite {
            fs::remove_file(&path)
                .with_context(|| format!("failed to remove {}", path.display()))?;
        } else {
            bail!("metadata.db already exists (use --overwrite)");
        }
    }
    let mut conn =
        Connection::open(&path).with_context(|| format!("failed to open {}", path.display()))?;
    conn.pragma_update(None, "journal_mode", &"WAL")?;
    conn.pragma_update(None, "synchronous", &"NORMAL")?;
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY,
            seed BIGINT NOT NULL,
            steps INT NOT NULL,
            max_score INT NOT NULL,
            highest_tile INT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS session (
            meta_key TEXT PRIMARY KEY,
            meta_value TEXT NOT NULL
        );
        ",
    )?;
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare(
            "INSERT INTO runs (id, seed, steps, max_score, highest_tile) VALUES (?1, ?2, ?3, ?4, ?5)",
        )?;
        for run in runs {
            stmt.execute(params![
                run.run_id as i64,
                run.seed as i64,
                run.steps as i64,
                run.max_score as i64,
                run.highest_tile as i64
            ])?;
        }
    }
    {
        let mut stmt = tx.prepare(
            "INSERT OR REPLACE INTO session (meta_key, meta_value) VALUES (?1, ?2)",
        )?;
        let value = if has_cumulative_reward { "true" } else { "false" };
        stmt.execute(params!["cumulative_reward", value])?;
    }
    tx.commit()?;
    Ok(())
}

pub(crate) fn write_valuation_types(
    out_dir: &Path,
    names: &[String],
    overwrite: bool,
) -> Result<()> {
    let path = out_dir.join("valuation_types.json");
    if path.exists() {
        if overwrite {
            fs::remove_file(&path)
                .with_context(|| format!("failed to remove {}", path.display()))?;
        } else {
            bail!("valuation_types.json already exists (use --overwrite)");
        }
    }
    let mut file =
        File::create(&path).with_context(|| format!("failed to create {}", path.display()))?;
    let mapping: HashMap<&str, u8> = names
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.as_str(), idx as u8))
        .collect();
    let json = serde_json::to_string_pretty(&mapping)?;
    file.write_all(json.as_bytes())?;
    file.write_all(b"\n")?;
    Ok(())
}

pub fn load_runs(dir: &Path) -> Result<Vec<RunSummary>> {
    let path = dir.join("metadata.db");
    if !path.exists() {
        bail!("missing metadata.db in {}", dir.display());
    }
    let conn =
        Connection::open(&path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut stmt = conn
        .prepare("SELECT id, seed, steps, max_score, highest_tile FROM runs ORDER BY id ASC")
        .with_context(|| format!("failed to prepare query for {}", path.display()))?;
    let rows = stmt
        .query_map([], |row| {
            Ok(RunSummary {
                run_id: row.get::<_, i64>(0)? as u32,
                seed: row.get::<_, i64>(1)? as u64,
                steps: row.get::<_, i64>(2)? as usize,
                max_score: row.get::<_, i64>(3)? as u64,
                highest_tile: row.get::<_, i64>(4)? as u32,
            })
        })
        .with_context(|| format!("failed to query runs from {}", path.display()))?;
    let mut summaries = Vec::new();
    for row in rows {
        summaries.push(row?);
    }
    Ok(summaries)
}

pub fn load_valuation_names(dir: &Path) -> Result<Vec<String>> {
    let path = dir.join("valuation_types.json");
    if !path.exists() {
        bail!("missing valuation_types.json in {}", dir.display());
    }
    let text =
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let map: HashMap<String, u8> = serde_json::from_str(&text)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    if map.is_empty() {
        return Ok(Vec::new());
    }
    let max_id = map
        .values()
        .copied()
        .max()
        .ok_or_else(|| anyhow!("valuation_types.json has no entries"))?;
    let mut names = vec![String::new(); max_id as usize + 1];
    for (name, id) in map {
        let slot = names
            .get_mut(id as usize)
            .ok_or_else(|| anyhow!("valuation id {} out of range", id))?;
        *slot = name;
    }
    if names.iter().any(|s| s.is_empty()) {
        bail!("valuation_types.json is sparse or has missing indices");
    }
    Ok(names)
}

pub(crate) fn load_cumulative_reward_flag(dir: &Path) -> Result<Option<bool>> {
    let value = load_session_value(dir, "cumulative_reward")?;
    match value.as_deref() {
        None => Ok(None),
        Some("true") => Ok(Some(true)),
        Some("false") => Ok(Some(false)),
        Some(other) => bail!("invalid cumulative_reward session value '{other}'"),
    }
}

fn load_session_value(dir: &Path, key: &str) -> Result<Option<String>> {
    let path = dir.join("metadata.db");
    if !path.exists() {
        bail!("missing metadata.db in {}", dir.display());
    }
    let conn =
        Connection::open(&path).with_context(|| format!("failed to open {}", path.display()))?;
    let has_session: Option<i32> = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='session'",
            [],
            |row| row.get(0),
        )
        .optional()?;
    if has_session.is_none() {
        return Ok(None);
    }
    let value: Option<String> = conn
        .query_row(
            "SELECT meta_value FROM session WHERE meta_key = ?1",
            [key],
            |row| row.get(0),
        )
        .optional()?;
    Ok(value)
}
