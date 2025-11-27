use crate::{config, feeder, grpc};
use anyhow::{Context, Result, bail};

/// Establish a gRPC inference client using either UDS or TCP configuration.
pub async fn connect_inference(conn: &config::Connection) -> Result<grpc::Client> {
    if let Some(uds_path) = &conn.uds_path {
        let client = grpc::connect_uds(uds_path)
            .await
            .with_context(|| format!("failed to connect over UDS {}", uds_path.display()))?;
        Ok(client)
    } else if let Some(tcp) = &conn.tcp_addr {
        let client = grpc::connect(tcp)
            .await
            .with_context(|| format!("failed to connect to inference server at {tcp}"))?;
        Ok(client)
    } else {
        bail!("either uds_path or tcp_addr must be specified in connection config")
    }
}

/// Convenience wrapper around [`feeder::Feeder::new`] for callers that only
/// need the handle-task pair.
pub fn build_feeder(
    batch_cfg: config::Batch,
    argmax_only: bool,
) -> (feeder::Feeder, feeder::FeederHandle) {
    build_feeder_with_value(batch_cfg, argmax_only, true)
}

/// Construct a feeder while controlling whether value outputs are requested.
pub fn build_feeder_with_value(
    batch_cfg: config::Batch,
    argmax_only: bool,
    value_outputs: bool,
) -> (feeder::Feeder, feeder::FeederHandle) {
    feeder::Feeder::new(batch_cfg, argmax_only, value_outputs)
}

/// Maximum number of inflight items based on batch configuration. Useful for
/// sizing client-side buffers.
pub fn max_inflight_items(batch_cfg: &config::Batch) -> usize {
    batch_cfg.inflight_batches * batch_cfg.max_batch
}
