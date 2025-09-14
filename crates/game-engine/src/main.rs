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
    steps: Vec<ds_writer::StepRow>,
    emb_map: HashMap<u64, (usize, Vec<f32>)>,
    emb_dim: Option<usize>,
    total_steps: u64,
}

impl ShardedWriter {
    fn new(
        session_dir: PathBuf,
        shard_max_steps: usize,
        max_gb: Option<f64>,
        inline_embeddings: bool,
    ) -> Self {
        Self {
            session_dir,
            shard_max_steps,
            max_gb,
            inline_embeddings,
            shard_idx: 0,
            steps: Vec::with_capacity(shard_max_steps),
            emb_map: HashMap::new(),
            emb_dim: None,
            total_steps: 0,
        }
    }

    fn push_step(&mut self, row: ds_writer::StepRow) {
        self.steps.push(row);
        self.total_steps += 1;
        if self.total_steps % 10_000 == 0 {
            debug!(
                "total steps: {}, buffered: {}",
                self.total_steps,
                self.steps.len()
            );
        }
        if self.steps.len() >= self.shard_max_steps {
            self.write_shard("shard_full");
        }
    }

    fn push_embedding(&mut self, er: feeder::EmbeddingRow) {
        if self.emb_dim.is_none() {
            self.emb_dim = Some(er.dim);
        }
        if self.emb_dim == Some(er.dim) {
            self.emb_map.insert(er.id, (er.dim, er.values));
        }
        if self.emb_map.len() >= self.shard_max_steps {
            self.write_shard("emb_full");
        }
    }

    fn write_shard(&mut self, reason: &str) {
        if self.steps.is_empty() {
            return;
        }
        self.shard_idx += 1;
        let shard_id = format!("{:06}", self.shard_idx);

        // Write steps shard
        let steps_path = self.session_dir.join(format!("steps-{shard_id}.npy"));
        info!(
            "flushing {} steps to {} (reason: {})",
            self.steps.len(),
            steps_path.display(),
            reason
        );
        let steps_bytes_est = self.steps.len() * (8 + 4 + 16);
        if self
            .max_gb
            .map_or(false, |gb| (steps_bytes_est as f64) / 1e9 > gb)
        {
            eprintln!("Skipping steps shard write: est. size > cap");
        } else if let Err(e) = ds_writer::write_steps_npy(&self.steps, &steps_path) {
            eprintln!("Failed to write steps shard {}: {}", shard_id, e);
        }

        // Write embeddings shard if enabled and consistent
        if self.inline_embeddings {
            if let Some(dim) = self.emb_dim {
                let mut floats = Vec::with_capacity(self.steps.len() * dim);
                let mut consistent = true;
                for row in &self.steps {
                    let id = ((row.run_id as u64) << 32) | (row.step_idx as u64);
                    if let Some((_d, v)) = self.emb_map.remove(&id) {
                        floats.extend_from_slice(&v);
                    } else {
                        consistent = false;
                        break;
                    }
                }

                if consistent && !floats.is_empty() {
                    let emb_path = self.session_dir.join(format!("embeddings-{shard_id}.npy"));
                    let emb_bytes_est = floats.len() * 4;
                    if self
                        .max_gb
                        .map_or(false, |gb| (emb_bytes_est as f64) / 1e9 > gb)
                    {
                        eprintln!("Skipping embeddings shard write: est. size > cap");
                    } else if let Err(e) =
                        ds_writer::write_embeddings_npy(&floats, self.steps.len(), dim, &emb_path)
                    {
                        eprintln!("Failed to write embeddings shard {}: {}", shard_id, e);
                    }
                }
            }
        }
        self.steps.clear();
        self.emb_map.clear();
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
    println!("Using configuration file: {}", args.config.display());
    let config = match Config::from_toml(&args.config) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to read config: {e}");
            std::process::exit(2);
        }
    };
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
    println!("Connected to inference server");

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

    // Optional results writer and session recorder (metadata.db)
    let (res_tx, mut res_rx) = mpsc::channel::<actor::GameResult>(1024);
    // Optional step recorder channel (structured steps -> steps.npy)
    let (step_tx, mut step_rx) = mpsc::channel::<ds_writer::StepRow>(65_536);
    let writer_handle = {
        let results_path = config.orchestrator.report.results_file.clone();
        let session_dir = config.orchestrator.report.session_dir.clone();
        let max_ram_mb = config.orchestrator.report.max_ram_mb;
        let max_gb = config.orchestrator.report.max_gb;
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
        Some(tokio::spawn(async move {
            if let Some(dir) = report_conf.session_dir.as_ref() {
                let mut writer = ShardedWriter::new(
                    dir.clone(),
                    report_conf.shard_max_steps_or_default(),
                    report_conf.max_gb,
                    inline_emb,
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
                                    info!("step_rx closed");
                                    step_done = true;
                                }
                            }
                        }
                        maybe_emb = emb_rx.recv(), if !emb_done => {
                            match maybe_emb {
                                Some(er) => writer.push_embedding(er),
                                None => {
                                    info!("emb_rx closed");
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
    let max_conc = config.max_concurrent_games as usize;
    let mut started: usize = 0;
    let mut finished: usize = 0;
    let mut total_steps: u64 = 0;
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

    let should_continue = |finished_games: usize, total_steps_done: u64| -> bool {
        let games_ok = match target_games {
            Some(g) => finished_games < g,
            None => true,
        };
        let steps_ok = match target_steps {
            Some(s) => total_steps_done < s,
            None => true,
        };
        games_ok && steps_ok
    };

    // Main game loop
    loop {
        // Spawn new actors if there is capacity and we should continue
        while set.len() < max_conc && should_continue(finished, total_steps) {
            let game_id = next_game_id;
            let actor = GameActor::new(
                game_id,
                handle.clone(),
                make_seed(started),
                config.sampling.clone(),
                Some(step_tx.clone()),
                cancel.clone(),
                target_steps.map(actor::StepBudget::new),
            );
            set.spawn(actor.run());
            started += 1;
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
        } else {
            // This should not be reached if !set.is_empty()
            break;
        }

        // If we've reached the limit, request graceful cancellation.
        if !should_continue(finished, total_steps) {
            cancel.cancel();
        }
    }

    info!("Draining pipeline...");
    drop(set);

    // All actors done; drop the handle so feeder can exit, then wait for it
    info!("Draining pipeline...");
    // Signal end of submissions and step collection
    drop(handle);
    drop(step_tx);
    // Ensure embeddings channel closes for step writer termination
    drop(emb_tx);

    info!("Waiting for feeder task...");
    let _ = feeder_task.await;
    info!("Feeder task finished.");

    drop(res_tx);
    if let Some(h) = writer_handle {
        info!("Waiting for writer task...");
        let _ = h.await;
        info!("Writer task finished.");
    }
    if let Some(h) = step_writer_handle {
        info!("Waiting for step writer task...");
        let _ = h.await;
        info!("Step writer task finished.");
    }
    info!("All tasks finished.");
}
