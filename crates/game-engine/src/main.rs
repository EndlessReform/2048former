mod actor;
mod config;
mod ds_writer;
mod feeder;
mod grpc;
mod recorder;

use clap::Parser;
use log::{debug, error, info};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use actor::GameActor;
use feeder::Feeder;
use rand::{RngCore, SeedableRng};
use recorder::{RunSummary, SessionRecorder};

use config::Config;

struct ShardedWriter {
    session_dir: PathBuf,
    shard_max_steps: usize,
    max_gb: Option<f64>,
    inline_embeddings: bool,
    shard_idx: u32,
    // Ready, paired rows (guaranteed to have both step and embedding when enabled)
    steps_buf: Vec<ds_writer::StepRow>,
    emb_buf: Vec<f32>,
    emb_dim: Option<usize>,
    // Pending unmatched items by global id = (run_id<<32)|step_idx
    pending_steps: HashMap<u64, ds_writer::StepRow>,
    pending_embs: HashMap<u64, Vec<f32>>,
    total_steps: u64,
    buffer_gauge: Option<Arc<AtomicUsize>>,
    data_mode: bool,
}

impl ShardedWriter {
    fn new(
        session_dir: PathBuf,
        shard_max_steps: usize,
        max_gb: Option<f64>,
        inline_embeddings: bool,
        buffer_gauge: Option<Arc<AtomicUsize>>,
        data_mode: bool,
    ) -> Self {
        if let Some(g) = &buffer_gauge {
            g.store(0, Ordering::Relaxed);
        }
        Self {
            session_dir,
            shard_max_steps,
            max_gb,
            inline_embeddings,
            shard_idx: 0,
            steps_buf: Vec::with_capacity(shard_max_steps),
            emb_buf: Vec::new(),
            emb_dim: None,
            pending_steps: HashMap::new(),
            pending_embs: HashMap::new(),
            total_steps: 0,
            buffer_gauge,
            data_mode,
        }
    }

    fn update_buffer_gauge(&self) {
        if let Some(g) = &self.buffer_gauge {
            g.store(self.steps_buf.len(), Ordering::Relaxed);
        }
    }

    fn push_step(&mut self, row: ds_writer::StepRow) {
        let id = ((row.run_id as u64) << 32) | (row.step_idx as u64);
        if self.inline_embeddings {
            if let Some(v) = self.pending_embs.remove(&id) {
                // Pair immediately
                if let Some(dim) = self.emb_dim {
                    if v.len() == dim {
                        self.steps_buf.push(row);
                        self.emb_buf.extend_from_slice(&v);
                    } else {
                        // Drop mismatched embedding silently
                        self.pending_steps.insert(id, row);
                    }
                } else {
                    // First embedding defines dim
                    self.emb_dim = Some(v.len());
                    self.steps_buf.push(row);
                    self.emb_buf.extend_from_slice(&v);
                }
            } else {
                self.pending_steps.insert(id, row);
            }
        } else {
            self.steps_buf.push(row);
        }
        self.total_steps += 1;
        self.update_buffer_gauge();
        if self.total_steps % 10_000 == 0 {
            debug!(
                "total steps: {}, paired buffered: {}",
                self.total_steps,
                self.steps_buf.len()
            );
        }
        if self.steps_buf.len() >= self.shard_max_steps {
            self.write_shard("shard_full");
        }
    }

    fn push_embedding(&mut self, er: feeder::EmbeddingRow) {
        if !self.inline_embeddings {
            return;
        }
        if self.emb_dim.is_none() {
            self.emb_dim = Some(er.dim);
        }
        if self.emb_dim != Some(er.dim) {
            return;
        }
        let id = er.id;
        if let Some(step) = self.pending_steps.remove(&id) {
            self.steps_buf.push(step);
            self.emb_buf.extend_from_slice(&er.values);
        } else {
            self.pending_embs.insert(id, er.values);
        }
        self.update_buffer_gauge();
        if self.steps_buf.len() >= self.shard_max_steps {
            self.write_shard("emb_full");
        }
    }

    fn write_shard(&mut self, reason: &str) {
        if self.steps_buf.is_empty() {
            return;
        }
        self.shard_idx += 1;
        let shard_id = format!("{:06}", self.shard_idx);

        // Write steps shard
        let steps_path = self.session_dir.join(format!("steps-{shard_id}.npy"));
        if self.data_mode {
            info!(
                "dataset flush reason={} steps={} shard={}",
                reason,
                self.steps_buf.len(),
                steps_path.display()
            );
        } else {
            info!(
                "flushing {} steps to {} (reason: {})",
                self.steps_buf.len(),
                steps_path.display(),
                reason
            );
        }
        let steps_bytes_est = self.steps_buf.len() * (8 + 4 + 16);
        if self
            .max_gb
            .map_or(false, |gb| (steps_bytes_est as f64) / 1e9 > gb)
        {
            eprintln!("Skipping steps shard write: est. size > cap");
        } else if let Err(e) = ds_writer::write_steps_npy(&self.steps_buf, &steps_path) {
            eprintln!("Failed to write steps shard {}: {}", shard_id, e);
        }

        // Write embeddings shard if enabled and consistent
        if self.inline_embeddings {
            if let Some(dim) = self.emb_dim {
                if !self.emb_buf.is_empty() {
                    let emb_path = self.session_dir.join(format!("embeddings-{shard_id}.npy"));
                    let emb_bytes_est = self.emb_buf.len() * 4;
                    if self
                        .max_gb
                        .map_or(false, |gb| (emb_bytes_est as f64) / 1e9 > gb)
                    {
                        eprintln!("Skipping embeddings shard write: est. size > cap");
                    } else if let Err(e) = ds_writer::write_embeddings_npy(
                        &self.emb_buf,
                        self.steps_buf.len(),
                        dim,
                        &emb_path,
                    ) {
                        eprintln!("Failed to write embeddings shard {}: {}", shard_id, e);
                    }
                }
            }
        }
        self.steps_buf.clear();
        self.emb_buf.clear();
        self.update_buffer_gauge();
    }

    fn flush(&mut self) {
        self.write_shard("flush");
    }
}

#[derive(Parser, Debug)]
struct Args {
    /// Path to configuration file
    #[arg(long, value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    config: PathBuf,
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    env_logger::init();
    let args = Args::parse();
    let config = match Config::from_toml(&args.config) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to read config: {e}");
            std::process::exit(2);
        }
    };
    let data_collection_mode = config.max_steps.is_some();
    if !data_collection_mode {
        println!("Using configuration file: {}", args.config.display());
    }
    debug!("Loaded config: {:#?}", config);

    // Initialize game engine tables if needed
    ai_2048::engine::new();

    // Establish gRPC connection (UDS preferred when set; else TCP)
    let conn = &config.orchestrator.connection;
    let client = if let Some(uds_path) = &conn.uds_path {
        match grpc::connect_uds(uds_path).await {
            Ok(c) => c,
            Err(e) => {
                eprintln!(
                    "Failed to connect to inference server over UDS {}: {e}",
                    uds_path.display()
                );
                std::process::exit(2);
            }
        }
    } else if let Some(tcp) = &conn.tcp_addr {
        match grpc::connect(tcp).await {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Failed to connect to inference server at {tcp}: {e}");
                std::process::exit(2);
            }
        }
    } else {
        eprintln!("Either uds_path or tcp_addr must be set under [orchestrator.connection]");
        std::process::exit(2);
    };
    if !data_collection_mode {
        println!("Connected to inference server");
    }

    // Shared cancellation to coordinate graceful shutdown
    let cancel = CancellationToken::new();

    // Start feeder
    let (mut feeder, handle) = Feeder::new(config.orchestrator.batch.clone());
    feeder.set_cancel_token(cancel.clone());
    // Optional embeddings pipeline
    let (emb_tx, mut emb_rx) = mpsc::channel::<feeder::EmbeddingRow>(65_536);
    if config.orchestrator.inline_embeddings {
        feeder.set_embeddings_channel(emb_tx.clone());
    }
    let feeder_task = feeder.spawn(client);

    let log_regular = !data_collection_mode;
    let buffer_gauge: Option<Arc<AtomicUsize>> = if data_collection_mode {
        Some(Arc::new(AtomicUsize::new(0)))
    } else {
        None
    };

    // Optional results writer and session recorder (metadata.db)
    let (res_tx, mut res_rx) = mpsc::channel::<actor::GameResult>(1024);
    // Optional step recorder channel (structured steps -> steps.npy)
    let (step_tx, mut step_rx) = mpsc::channel::<ds_writer::StepRow>(65_536);
    let writer_handle = {
        let results_path = config.orchestrator.report.results_file.clone();
        let session_dir = config.orchestrator.report.session_dir.clone();
        let max_ram_mb = config.orchestrator.report.max_ram_mb;
        let max_gb = config.orchestrator.report.max_gb;
        if log_regular {
            // Surface where results will be written (or disabled)
            match results_path.as_ref() {
                Some(p) => eprintln!("[client] results file: {}", p.display()),
                None => eprintln!("[client] results file: disabled"),
            }
        }
        Some(tokio::spawn(async move {
            // Results JSONL sink (optional)
            let mut file_opt = if let Some(path) = results_path.as_ref() {
                match tokio::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .await
                {
                    Ok(f) => Some(f),
                    Err(e) => {
                        eprintln!("Failed to open results file {}: {}", path.display(), e);
                        None
                    }
                }
            } else {
                None
            };

            // Session recorder (sync; lightweight ops per finished game)
            let mut rec_opt = match session_dir.as_ref() {
                Some(dir) => match SessionRecorder::new(dir) {
                    Ok(mut rec) => {
                        // Write a minimal meta record for provenance
                        let ts = match std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                        {
                            Ok(d) => format!("{}", d.as_secs_f64()),
                            Err(_) => String::from("0.0"),
                        };
                        let _ = rec.set_meta("ts", &ts);
                        Some(rec)
                    }
                    Err(e) => {
                        eprintln!(
                            "Failed to init session recorder at {}: {}",
                            dir.display(),
                            e
                        );
                        None
                    }
                },
                None => None,
            };

            while let Some(r) = res_rx.recv().await {
                // Emit JSONL if enabled
                if let Some(f) = file_opt.as_mut() {
                    #[derive(serde::Serialize)]
                    struct Rec {
                        game_id: u32,
                        seed: u64,
                        steps: u64,
                        score: u64,
                        highest_tile: u32,
                    }
                    let rec = Rec {
                        game_id: r.game_id,
                        seed: r.seed,
                        steps: r.steps,
                        score: r.score,
                        highest_tile: r.highest_tile,
                    };
                    if let Ok(line) = serde_json::to_string(&rec) {
                        let _ = tokio::io::AsyncWriteExt::write_all(f, line.as_bytes()).await;
                        let _ = tokio::io::AsyncWriteExt::write_all(f, b"\n").await;
                        let _ = tokio::io::AsyncWriteExt::flush(f).await;
                    }
                }
                // Record run into SQLite if enabled
                if let Some(rec) = rec_opt.as_mut() {
                    let _ = rec.upsert_run(RunSummary {
                        id: r.game_id as u64,
                        seed: r.seed,
                        steps: r.steps,
                        max_score: r.score,
                        highest_tile: r.highest_tile,
                    });
                }
            }
            // Disk failsafe info: we don't write here, only in the step/emb writers below.
        }))
    };

    // Optional step collector -> steps-*.npy, embeddings-*.npy
    let step_writer_handle = {
        let report_conf = config.orchestrator.report.clone();
        let inline_emb = config.orchestrator.inline_embeddings;
        let buffer_gauge = buffer_gauge.clone();
        let data_mode = data_collection_mode;
        Some(tokio::spawn(async move {
            if let Some(dir) = report_conf.session_dir.as_ref() {
                let mut writer = ShardedWriter::new(
                    dir.clone(),
                    report_conf.shard_max_steps_or_default(),
                    report_conf.max_gb,
                    inline_emb,
                    buffer_gauge,
                    data_mode,
                );

                let mut step_done = false;
                let mut emb_done = !inline_emb;
                loop {
                    tokio::select! {
                        biased;
                        maybe_row = step_rx.recv(), if !step_done => {
                            match maybe_row {
                                Some(row) => writer.push_step(row),
                                None => {
                                    if !data_mode {
                                        info!("step_rx closed");
                                    }
                                    step_done = true;
                                }
                            }
                        }
                        maybe_emb = emb_rx.recv(), if !emb_done => {
                            match maybe_emb {
                                Some(er) => writer.push_embedding(er),
                                None => {
                                    if !data_mode {
                                        info!("emb_rx closed");
                                    }
                                    emb_done = true;
                                }
                            }
                        }
                        else => { break; }
                    }
                }
                writer.flush();
            } else {
                // Drain and drop
                while step_rx.recv().await.is_some() {}
                while emb_rx.recv().await.is_some() {}
            }
        }))
    };

    // Spawn per-game actors up to max_concurrent_games
    let target_games: Option<usize> = config.num_seeds.map(|v| v as usize);
    let target_steps: Option<u64> = config.max_steps;
    // Shared global step budget across all actors (if max_steps is set)
    let step_budget = target_steps.map(actor::StepBudget::new);
    let max_conc = config.max_concurrent_games as usize;
    let mut started: usize = 0;
    let mut finished: usize = 0;
    let mut total_steps: u64 = 0;
    // Progress accounting
    let started_counter = Arc::new(AtomicUsize::new(0));
    let finished_counter = Arc::new(AtomicUsize::new(0));
    let mut set: JoinSet<actor::GameResult> = JoinSet::new();

    // Seed strategy (default reproducible):
    // - If random_seeds=true -> fully random per game
    // - Else use fixed base: `fixed_seed` if set, otherwise default constant
    let random_seeds = config.orchestrator.random_seeds;
    let base_seed = config.orchestrator.fixed_seed.unwrap_or(0x00C0FFEEu64);
    let mut seed_rng = rand::rngs::StdRng::from_entropy();
    let mut next_game_id: u32 = 0;

    let mut make_seed = |idx: usize| -> u64 {
        if random_seeds {
            seed_rng.next_u64()
        } else {
            base_seed.wrapping_add(idx as u64)
        }
    };

    // Bug fix: bound spawning by `started` (not `finished`) to avoid overshooting `num_seeds`.
    let can_spawn_more = |started_games: usize, total_steps_done: u64| -> bool {
        let games_ok = match target_games {
            Some(g) => started_games < g,
            None => true,
        };
        let steps_ok = match target_steps {
            Some(s) => total_steps_done < s,
            None => true,
        };
        games_ok && steps_ok
    };

    // Spawn a progress reporter appropriate for the current mode
    let progress_task: Option<tokio::task::JoinHandle<()>> = if data_collection_mode {
        if let Some(budget) = step_budget.clone() {
            let buffer_gauge = buffer_gauge.clone();
            let target_steps_val = target_steps;
            Some(tokio::spawn(async move {
                let mut next_bucket: u64 = 100_000;
                let mut prev_steps: u64 = 0;
                let mut prev_time = Instant::now();
                loop {
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    let steps = budget.used();
                    let now = Instant::now();
                    let dt = now.duration_since(prev_time).as_secs_f64();
                    let delta = steps.saturating_sub(prev_steps);
                    let rate = if dt > 0.0 { delta as f64 / dt } else { 0.0 };
                    let buffered = buffer_gauge
                        .as_ref()
                        .map(|g| g.load(Ordering::Relaxed))
                        .unwrap_or(0);
                    while steps >= next_bucket {
                        let milestone = next_bucket;
                        info!(
                            "dataset progress total_steps={} buffered={} steps_per_s={:.2}",
                            milestone, buffered, rate
                        );
                        next_bucket = next_bucket.saturating_add(100_000);
                    }
                    prev_steps = steps;
                    prev_time = now;
                    if let Some(target) = target_steps_val {
                        if steps >= target {
                            break;
                        }
                    }
                }
            }))
        } else {
            None
        }
    } else {
        let started_c = started_counter.clone();
        let finished_c = finished_counter.clone();
        Some(tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(2)).await;
                let s = started_c.load(Ordering::Relaxed);
                let f = finished_c.load(Ordering::Relaxed);
                let running = s.saturating_sub(f);
                eprintln!(
                    "[client] progress: started={} running={} finished={}",
                    s, running, f
                );
                if running == 0 {
                    break;
                }
            }
        }))
    };

    // Main game loop
    loop {
        // Spawn new actors if there is capacity and we should continue
        while set.len() < max_conc && can_spawn_more(started, total_steps) {
            let game_id = next_game_id;
            let actor = GameActor::new(
                game_id,
                handle.clone(),
                make_seed(started),
                config.sampling.clone(),
                Some(step_tx.clone()),
                cancel.clone(),
                step_budget.clone(),
            );
            set.spawn(actor.run());
            started += 1;
            started_counter.fetch_add(1, Ordering::Relaxed);
            next_game_id = next_game_id.wrapping_add(1);
        }

        // If the set is empty, we're done
        if set.is_empty() {
            break;
        }

        // Wait for a game to finish
        if let Some(res) = set.join_next().await {
            match res {
                Ok(r) => {
                    total_steps = total_steps.saturating_add(r.steps);
                    if res_tx.capacity() > 0 {
                        let _ = res_tx.send(r).await;
                    } else { /* drop if backpressured */
                    }
                }
                Err(e) => {
                    error!("actor task failed: {e}");
                }
            }
            finished += 1;
            finished_counter.fetch_add(1, Ordering::Relaxed);
        } else {
            // This should not be reached if !set.is_empty()
            break;
        }

        // If we've reached the limit, decide whether to cancel immediately
        // or allow all in-flight games to finish:
        // - If we are running with a global step budget AND saving embeddings,
        //   we cancel to respect the exact budget for dataset collection.
        // - Otherwise (typical benchmark/results mode), do NOT cancel; stop
        //   spawning new games and let existing ones finish naturally.
        if !can_spawn_more(started, total_steps) {
            if step_budget.is_some() && config.orchestrator.inline_embeddings {
                cancel.cancel();
            }
        }
    }
    if log_regular {
        info!("Draining pipeline...");
    }
    drop(set);

    // All actors done; drop the handle so feeder can exit, then wait for it
    if log_regular {
        info!("Draining pipeline...");
    }
    // Signal end of submissions and step collection
    drop(handle);
    drop(step_tx);
    // Ensure embeddings channel closes for step writer termination
    drop(emb_tx);

    if log_regular {
        info!("Waiting for feeder task...");
    }
    let _ = feeder_task.await;
    if log_regular {
        info!("Feeder task finished.");
    }

    drop(res_tx);
    if let Some(h) = writer_handle {
        if log_regular {
            info!("Waiting for writer task...");
        }
        let _ = h.await;
        if log_regular {
            info!("Writer task finished.");
        }
    }
    if let Some(h) = step_writer_handle {
        if log_regular {
            info!("Waiting for step writer task...");
        }
        let _ = h.await;
        if log_regular {
            info!("Step writer task finished.");
        }
    }
    if let Some(task) = progress_task {
        let _ = task.await;
    }
    if log_regular {
        info!("All tasks finished.");
    }
}
