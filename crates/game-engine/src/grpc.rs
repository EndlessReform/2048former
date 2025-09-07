// Mirror the proto package path for include_proto!.
pub mod api {
    pub mod train_2048 {
        pub mod inference {
            pub mod v1 {
                tonic::include_proto!("train_2048.inference.v1");
            }
        }
    }
}

// Re-export for concise paths: pb::inference_client::InferenceClient, messages, etc.
pub use api::train_2048::inference::v1 as pb;

use pb::inference_client::InferenceClient;
use tonic::transport::Channel;

// Public alias for the generated client type
pub type Client = InferenceClient<Channel>;

/// Connect to the inference endpoint over TCP, e.g. "http://127.0.0.1:50051".
/// For UDS support, add a separate helper using `connect_with_connector`.
pub async fn connect<D: AsRef<str>>(dst: D) -> Result<Client, tonic::transport::Error> {
    InferenceClient::connect(dst.as_ref().to_string()).await
}

/// Convenience call for the bins API (probabilities over bins).
/// Use this for quick tests; production code should batch via the Feeder.
pub async fn infer_once(
    client: &mut Client,
    batch_id: Option<u64>,
    model_id: Option<String>,
    items: impl IntoIterator<Item = (u64, [u8; 16])>,
) -> Result<pb::InferResponse, tonic::Status> {
    let items_pb: Vec<pb::Item> = items
        .into_iter()
        .map(|(id, board)| pb::Item { id, board: board.to_vec() })
        .collect();
    let req = pb::InferRequest {
        model_id: model_id.unwrap_or_default(),
        items: items_pb,
        batch_id: batch_id.unwrap_or_default(),
    };
    let resp = client.infer(req).await?.into_inner();
    Ok(resp)
}
