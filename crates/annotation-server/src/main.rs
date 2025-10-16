mod app;
mod args;
mod routes;
mod student;

use std::net::SocketAddr;

use anyhow::{Context, Result};
use args::Args;
use axum::{Router, routing::get};
use clap::Parser;
use dataset_packer::macroxue::tokenizer::MacroxueTokenizerV2;
use routes::{get_disagreements, get_health, get_run, list_runs};
use tokio::signal;
use tower_http::{cors::CorsLayer, services::ServeDir};
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(args.log.clone()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("loading dataset" = %args.dataset.display(), "annotations" = %args.annotations.display());
    let dataset = app::load_dataset(&args.dataset, &args.annotations)?;

    let tokenizer = if let Some(tokenizer_path) = &args.tokenizer {
        info!("loading tokenizer" = %tokenizer_path.display());
        Some(
            MacroxueTokenizerV2::from_path(tokenizer_path).with_context(|| {
                format!("failed to load tokenizer from {}", tokenizer_path.display())
            })?,
        )
    } else {
        None
    };

    let state = app::AppState::from_dataset(dataset, tokenizer);

    let router = Router::new()
        .route("/health", get(get_health))
        .route("/runs", get(list_runs))
        .route("/runs/:run_id", get(get_run))
        .route("/runs/:run_id/disagreements", get(get_disagreements))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let router = if let Some(ui_path) = &args.ui_path {
        router.fallback_service(ServeDir::new(ui_path).append_index_html_on_directories(true))
    } else {
        router
    };

    let addr: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .context("invalid host/port combination")?;
    info!("listening" = %addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install CTRL+C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
}
