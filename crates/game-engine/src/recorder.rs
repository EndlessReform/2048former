use std::path::{Path, PathBuf};

use rusqlite::{Connection, OptionalExtension, params};

/// Summary for a completed run/game.
#[derive(Debug, Clone, Copy)]
pub struct RunSummary {
    pub id: u64,
    pub seed: u64,
    pub steps: u64,
    pub max_score: u64,
    pub highest_tile: u32,
}

/// Minimal session recorder that writes `metadata.db` with `runs` and `session` tables.
///
/// Schema:
/// - runs(id INTEGER PRIMARY KEY, seed BIGINT, steps INT, max_score INT, highest_tile INT)
/// - session(meta_key TEXT PRIMARY KEY, meta_value TEXT)
pub struct SessionRecorder {
    session_dir: PathBuf,
    conn: Connection,
}

impl SessionRecorder {
    /// Create or open a session at `dir`, ensure schema exists.
    pub fn new<P: AsRef<Path>>(dir: P) -> Result<Self, rusqlite::Error> {
        let session_dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&session_dir)
            .map_err(|_e| rusqlite::Error::ExecuteReturnedResults)?;
        let db_path = session_dir.join("metadata.db");
        let conn = Connection::open(db_path)?;
        conn.pragma_update(None, "journal_mode", &"WAL")?;
        conn.pragma_update(None, "synchronous", &"NORMAL")?;
        // Create schema if missing
        conn.execute_batch(
            r#"
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
            "#,
        )?;
        Ok(Self { session_dir, conn })
    }

    /// Insert or update a run summary row.
    pub fn upsert_run(&mut self, r: RunSummary) -> Result<(), rusqlite::Error> {
        self.conn.execute(
            "INSERT INTO runs (id, seed, steps, max_score, highest_tile) VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(id) DO UPDATE SET seed=excluded.seed, steps=excluded.steps, max_score=excluded.max_score, highest_tile=excluded.highest_tile",
            params![r.id as i64, r.seed as i64, r.steps as i64, r.max_score as i64, r.highest_tile as i64],
        )?;
        Ok(())
    }

    /// Set a session meta value by key (stored as TEXT; put JSON if needed).
    pub fn set_meta<K: AsRef<str>, V: AsRef<str>>(
        &mut self,
        key: K,
        value: V,
    ) -> Result<(), rusqlite::Error> {
        self.conn.execute(
            "INSERT INTO session (meta_key, meta_value) VALUES (?1, ?2)
             ON CONFLICT(meta_key) DO UPDATE SET meta_value=excluded.meta_value",
            params![key.as_ref(), value.as_ref()],
        )?;
        Ok(())
    }

    /// Optional helper to read back a meta value in tests/tools.
    pub fn get_meta<K: AsRef<str>>(&self, key: K) -> Result<Option<String>, rusqlite::Error> {
        self.conn
            .query_row(
                "SELECT meta_value FROM session WHERE meta_key = ?1",
                params![key.as_ref()],
                |row| row.get::<_, String>(0),
            )
            .optional()
    }

    /// Optional helper to fetch a run back.
    pub fn get_run(&self, id: u64) -> Result<Option<RunSummary>, rusqlite::Error> {
        self.conn
            .query_row(
                "SELECT id, seed, steps, max_score, highest_tile FROM runs WHERE id = ?1",
                params![id as i64],
                |row| {
                    Ok(RunSummary {
                        id: row.get::<_, i64>(0)? as u64,
                        seed: row.get::<_, i64>(1)? as u64,
                        steps: row.get::<_, i64>(2)? as u64,
                        max_score: row.get::<_, i64>(3)? as u64,
                        highest_tile: row.get::<_, i64>(4)? as u32,
                    })
                },
            )
            .optional()
    }

    /// Absolute path to the session directory.
    pub fn session_dir(&self) -> &Path {
        &self.session_dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn basic_session_roundtrip() {
        let td = tempdir().unwrap();
        let path = td.path().join("session");
        let mut rec = SessionRecorder::new(&path).expect("open session");
        assert!(rec.session_dir().exists());
        // Set some meta
        rec.set_meta("ts", "2025-09-10T12:34:56Z").unwrap();
        rec.set_meta("model_id", "v1_pretrained_50m").unwrap();
        // Upsert two runs
        rec.upsert_run(RunSummary {
            id: 1,
            seed: 42,
            steps: 1000,
            max_score: 12345,
            highest_tile: 8192,
        })
        .unwrap();
        rec.upsert_run(RunSummary {
            id: 2,
            seed: 43,
            steps: 1500,
            max_score: 23456,
            highest_tile: 16384,
        })
        .unwrap();
        // Update one run
        rec.upsert_run(RunSummary {
            id: 1,
            seed: 42,
            steps: 1100,
            max_score: 13000,
            highest_tile: 8192,
        })
        .unwrap();

        // Read back
        let ts = rec.get_meta("ts").unwrap();
        assert_eq!(ts.as_deref(), Some("2025-09-10T12:34:56Z"));
        let r1 = rec.get_run(1).unwrap().expect("run 1");
        assert_eq!(r1.steps, 1100);
        assert_eq!(r1.max_score, 13000);
        let r2 = rec.get_run(2).unwrap().expect("run 2");
        assert_eq!(r2.highest_tile, 16384);
    }
}
