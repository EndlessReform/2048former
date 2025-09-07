fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile protobuf definitions for the inference API.
    // Paths are resolved relative to the workspace: proto/...
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")?;
    let proto_root = std::path::Path::new(&manifest_dir).join("../../proto");
    let api_proto = proto_root.join("train_2048/inference/v1/inference.proto");

    // Generate both prost message types and tonic service stubs
    tonic_prost_build::configure()
        .build_server(false)
        .compile_protos(&[api_proto], &[proto_root.clone()])?;

    println!("cargo:rerun-if-changed={}", proto_root.display());
    Ok(())
}
