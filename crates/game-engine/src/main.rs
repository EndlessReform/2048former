mod config;
mod grpc;
mod feeder;
mod actor;
mod recorder;
mod ds_writer;

use clap::Parser;
use std::path::PathBuf;
use tokio::task::JoinSet;
use tokio::sync::mpsc;

use actor::GameActor;
use feeder::Feeder;
use recorder::{RunSummary, SessionRecorder};

use config::Config;

#[derive(Parser, Debug)]
struct Args {
    /// Path to configuration file
    #[arg(long, value_name = "FILE", value_parser = clap::value_parser!(PathBuf))]
    config: PathBuf,
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let args = Args::parse();
    println!("Using configuration file: {}", args.config.display());
    let config = match Config::from_toml(&args.config) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to read config: {e}");
            std::process::exit(2);
        }
    };

    // Initialize game engine tables if needed
    ai_2048::engine::new();

    // Establish gRPC connection (UDS preferred when set; else TCP)
    let conn = &config.orchestrator.connection;
    let client = if let Some(uds_path) = &conn.uds_path {
        match grpc::connect_uds(uds_path).await {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Failed to connect to inference server over UDS {}: {e}", uds_path.display());
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

    // Start feeder
    let (mut feeder, handle) = Feeder::new(config.orchestrator.batch.clone());
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
        Some(tokio::spawn(async move {
            // Results JSONL sink (optional)
            let mut file_opt = if let Some(path) = results_path.as_ref() {
                match tokio::fs::OpenOptions::new().create(true).append(true).open(path).await {
                    Ok(f) => Some(f),
                    Err(e) => {
                        eprintln!("Failed to open results file {}: {}", path.display(), e);
                        None
                    }
                }
            } else { None };

            // Session recorder (sync; lightweight ops per finished game)
            let mut rec_opt = match session_dir.as_ref() {
                Some(dir) => match SessionRecorder::new(dir) {
                    Ok(mut rec) => {
                        // Write a minimal meta record for provenance
                        let ts = match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
                            Ok(d) => format!("{}", d.as_secs_f64()),
                            Err(_) => String::from("0.0"),
                        };
                        let _ = rec.set_meta("ts", &ts);
                        Some(rec)
                    }
                    Err(e) => {
                        eprintln!("Failed to init session recorder at {}: {}", dir.display(), e);
                        None
                    }
                },
                None => None,
            };

            while let Some(r) = res_rx.recv().await {
                // Emit JSONL if enabled
                if let Some(f) = file_opt.as_mut() {
                    #[derive(serde::Serialize)]
                    struct Rec { game_id: u32, seed: u64, steps: u64, score: u64, highest_tile: u32 }
                    let rec = Rec { game_id: r.game_id, seed: r.seed, steps: r.steps, score: r.score, highest_tile: r.highest_tile };
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
        }))
    };

    // Optional step collector -> steps.npy
    let step_writer_handle = {
        let session_dir = config.orchestrator.report.session_dir.clone();
        let inline_emb = config.orchestrator.inline_embeddings;
        Some(tokio::spawn(async move {
            if let Some(dir) = session_dir.as_ref() {
                use std::collections::HashMap;
                let mut steps: Vec<ds_writer::StepRow> = Vec::new();
                let mut id_to_idx: HashMap<u64, usize> = HashMap::new();
                let mut emb_map: HashMap<u64, (usize, Vec<f32>)> = HashMap::new();
                let mut emb_dim_opt: Option<usize> = None;

                // Drain both channels
                let mut step_done = false;
                let mut emb_done = !inline_emb; // if not enabled, consider done
                loop {
                    tokio::select! {
                        biased;
                        maybe_row = step_rx.recv(), if !step_done => {
                            match maybe_row {
                                Some(row) => {
                                    let idx = steps.len();
                                    let id = ((row.run_id as u64) << 32) | (row.step_idx as u64);
                                    id_to_idx.insert(id, idx);
                                    steps.push(row);
                                }
                                None => { step_done = true; }
                            }
                        }
                        maybe_emb = emb_rx.recv(), if !emb_done => {
                            match maybe_emb {
                                Some(er) => {
                                    if emb_dim_opt.is_none() { emb_dim_opt = Some(er.dim); }
                                    if emb_dim_opt == Some(er.dim) {
                                        emb_map.insert(er.id, (er.dim, er.values));
                                    }
                                }
                                None => { emb_done = true; }
                            }
                        }
                        else => { break; }
                    }
                }

                // Write steps
                if !steps.is_empty() {
                    let path = dir.join("steps.npy");
                    if let Err(e) = ds_writer::write_steps_npy(&steps, &path) {
                        eprintln!("Failed to write steps.npy at {}: {}", path.display(), e);
                    }
                }

                // If embeddings present and complete, write shard 000001
                if inline_emb {
                    if let Some(dim) = emb_dim_opt {
                        if steps.len() == emb_map.len() {
                            let mut floats = Vec::with_capacity(steps.len() * dim);
                            for row in &steps {
                                let id = ((row.run_id as u64) << 32) | (row.step_idx as u64);
                                if let Some((_d, v)) = emb_map.remove(&id) {
                                    floats.extend_from_slice(&v);
                                } else {
                                    // missing embedding; abort writing
                                    floats.clear();
                                    break;
                                }
                            }
                            if floats.len() == steps.len() * dim {
                                let path = dir.join("embeddings-000001.npy");
                                if let Err(e) = ds_writer::write_embeddings_npy(&floats, steps.len(), dim, &path) {
                                    eprintln!("Failed to write embeddings shard at {}: {}", path.display(), e);
                                }
                            }
                        }
                    }
                }
            } else {
                // Drain and drop
                while step_rx.recv().await.is_some() {}
                while emb_rx.recv().await.is_some() {}
            }
        }))
    };

    // Spawn per-game actors up to max_concurrent_games, total num_seeds games
    let total = config.num_seeds as usize;
    let max_conc = config.max_concurrent_games as usize;
    let mut started: usize = 0;
    let mut finished: usize = 0;
    let mut set: JoinSet<actor::GameResult> = JoinSet::new();

    // Seed base for reproducibility across runs
    let base_seed = 0xC0FFEEu64;

    // Prime concurrent actors
    while started < total && set.len() < max_conc {
        let game_id = started as u32;
        let actor = GameActor::new(
            game_id,
            handle.clone(),
            base_seed.wrapping_add(started as u64),
            config.sampling.clone(),
            Some(step_tx.clone()),
        );
        set.spawn(actor.run());
        started += 1;
    }

    // Keep the pipeline full until all games complete
    while finished < total {
        if let Some(res) = set.join_next().await {
            match res {
                Ok(r) => {
                    if res_tx.capacity() > 0 { let _ = res_tx.send(r).await; } else { /* drop if backpressured */ }
                }
                Err(e) => {
                    eprintln!("actor task failed: {e}");
                }
            }
            finished += 1;
            if started < total {
                let game_id = started as u32;
                let actor = GameActor::new(
                    game_id,
                    handle.clone(),
                    base_seed.wrapping_add(started as u64),
                    config.sampling.clone(),
                    Some(step_tx.clone()),
                );
                set.spawn(actor.run());
                started += 1;
            }
        }
    }

    // All actors done; drop the handle so feeder can exit, then wait for it
    drop(handle);
    let _ = feeder_task.await;
    drop(res_tx);
    drop(step_tx);
    if let Some(h) = writer_handle { let _ = h.await; }
    if let Some(h) = step_writer_handle { let _ = h.await; }
}
