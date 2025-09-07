mod config;
mod grpc;
mod feeder;
mod actor;

use clap::Parser;
use std::path::PathBuf;
use tokio::task::JoinSet;

use actor::GameActor;
use feeder::Feeder;

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

    // Start feeder
    let (feeder, handle) = Feeder::new(config.orchestrator.batch.clone());
    let feeder_task = feeder.spawn(client);

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
        let actor = GameActor::new(game_id, handle.clone(), base_seed.wrapping_add(started as u64));
        set.spawn(actor.run());
        started += 1;
    }

    // Keep the pipeline full until all games complete
    while finished < total {
        if let Some(res) = set.join_next().await {
            match res {
                Ok(r) => {
                    println!(
                        "game {} done | steps={} score={} top_tile={}",
                        r.game_id, r.steps, r.score, r.highest_tile
                    );
                }
                Err(e) => {
                    eprintln!("actor task failed: {e}");
                }
            }
            finished += 1;
            if started < total {
                let game_id = started as u32;
                let actor = GameActor::new(game_id, handle.clone(), base_seed.wrapping_add(started as u64));
                set.spawn(actor.run());
                started += 1;
            }
        }
    }

    // All actors done; drop the handle so feeder can exit, then wait for it
    drop(handle);
    let _ = feeder_task.await;
}
