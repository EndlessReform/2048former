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

/// Connect to the inference endpoint, e.g. "http://127.0.0.1:50051" or "https://…".
pub async fn connect<D: AsRef<str>>(dst: D) -> Result<Client, tonic::transport::Error> {
    InferenceClient::connect(dst.as_ref().to_string()).await
}

// Example convenience call
pub async fn infer_once(
    client: &mut Client,
    items: Vec<Vec<u32>>, // each is a flattened 4x4 board (len 16)
    model_id: impl Into<String>,
    request_id: impl Into<String>,
) -> Result<pb::InferResponse, tonic::Status> {
    let req = pb::InferRequest {
        model_id: model_id.into(),
        items: items.into_iter().map(|ids| pb::Tokens { ids }).collect(),
        request_id: request_id.into(),
    };
    let resp = client.infer(req).await?.into_inner();
    Ok(resp)
}
