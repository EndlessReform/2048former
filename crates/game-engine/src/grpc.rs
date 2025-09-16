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

use hyper_util::rt::tokio::TokioIo;
use pb::inference_client::InferenceClient;
use std::io;
use tokio::net::UnixStream;
use tonic::transport::{Channel, Endpoint, Uri};
use tower::service_fn;

// Public alias for the generated client type
pub type Client = InferenceClient<Channel>;

/// Connect to the inference endpoint over TCP, e.g. "http://127.0.0.1:50051".
/// For UDS support, add a separate helper using `connect_with_connector`.
pub async fn connect<D: AsRef<str>>(dst: D) -> Result<Client, tonic::transport::Error> {
    let ep = Endpoint::try_from(dst.as_ref().to_string())?;
    let channel = ep.connect().await?;
    Ok(InferenceClient::new(channel))
}

/// Connect to the inference endpoint over a Unix Domain Socket at `path`.
/// Example path: "/tmp/2048_infer.sock".
pub async fn connect_uds<P: AsRef<std::path::Path>>(
    path: P,
) -> Result<Client, tonic::transport::Error> {
    // Hyper requires a valid URI even when using a custom connector; use a dummy.
    let ep = Endpoint::try_from("http://[::]:50051")?;
    let path_string = path.as_ref().to_path_buf();
    let channel = ep
        .connect_with_connector(service_fn(move |_uri: Uri| {
            let p = path_string.clone();
            async move {
                let stream = UnixStream::connect(p).await?;
                Ok::<_, io::Error>(TokioIo::new(stream))
            }
        }))
        .await?;
    Ok(InferenceClient::new(channel))
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
        .map(|(id, board)| pb::Item {
            id,
            board: board.to_vec(),
        })
        .collect();
    let req = pb::InferRequest {
        model_id: model_id.unwrap_or_default(),
        items: items_pb,
        batch_id: batch_id.unwrap_or_default(),
        return_embedding: false,
        argmax_only: false,
    };
    let resp = client.infer(req).await?.into_inner();
    Ok(resp)
}
