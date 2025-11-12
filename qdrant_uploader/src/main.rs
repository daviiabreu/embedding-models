use std::{
    convert::TryInto,
    env,
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use qdrant_client::{
    client::Payload,
    qdrant::{
        CreateCollectionBuilder,
        Distance,
        PointStruct,
        UpsertPointsBuilder,
        VectorParamsBuilder,
    },
    Qdrant,
};
use serde::Deserialize;
use serde_json::Value;

mod dotenv_helper;

const DEFAULT_COLLECTION_NAME: &str = "inteli_admission_chunks";
const DEFAULT_EMBEDDINGS_PATH: &str =
    "documents/Edital-Processo-Seletivo-Inteli_-Graduacao-2026_AJUSTADO-embeddings.json";

#[derive(Debug, Deserialize)]
struct ChunkRecord {
    id: String,
    content: String,
    metadata: Value,
    embedding: Vec<f32>,
}

#[tokio::main]
async fn main() -> Result<()> {
    if let Err(err) = dotenv_helper::load_env() {
        eprintln!("⚠️  Unable to load .env: {err}");
    }

    let api_key = env::var("QDRANT_API_KEY")
        .context("Set the QDRANT_API_KEY environment variable with your Qdrant API key")?;
    let url = env::var("QDRANT_URL")
        .unwrap_or_else(|_| "https://e44c9510-8af0-4f60-9cdb-e7bd1fe6cd7c.us-west-2-0.aws.cloud.qdrant.io:6334".into());
    let collection_name =
        env::var("QDRANT_COLLECTION").unwrap_or_else(|_| DEFAULT_COLLECTION_NAME.to_string());
    let embeddings_path = env::var("EMBEDDINGS_PATH")
        .unwrap_or_else(|_| DEFAULT_EMBEDDINGS_PATH.to_string());
    let embeddings_path = resolve_embeddings_path(&embeddings_path);
    println!("Using embeddings file: {}", embeddings_path.display());

    let client = Qdrant::from_url(&url)
        .api_key(api_key)
        .build()
        .context("Failed to build Qdrant client")?;

    let records = load_embeddings(&embeddings_path).with_context(|| {
        format!(
            "Failed to load embeddings from {}",
            embeddings_path.display()
        )
    })?;

    if records.is_empty() {
        println!("No embedding records found, nothing to upload.");
        return Ok(());
    }

    let vector_size = records[0].embedding.len() as u64;
    ensure_collection(&client, &collection_name, vector_size).await?;

    let mut points = Vec::with_capacity(records.len());
    for (idx, record) in records.into_iter().enumerate() {
        let point_id = extract_point_id(&record.id, idx as u64);

        let payload: Payload = serde_json::json!({
            "chunk_id": record.id,
            "content": record.content,
            "metadata": record.metadata,
        })
        .try_into()
        .context("Failed to convert payload to Qdrant format")?;

        points.push(PointStruct::new(
            point_id,
            record.embedding,
            payload,
        ));
    }

    let upsert_request =
        UpsertPointsBuilder::new(collection_name.clone(), points).wait(true);

    client
        .upsert_points(upsert_request)
        .await
        .context("Failed to upsert points into Qdrant")?;

    println!("✅ Uploaded embeddings to collection `{}`", collection_name);
    Ok(())
}

fn load_embeddings(path: &Path) -> Result<Vec<ChunkRecord>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let records: Vec<ChunkRecord> = serde_json::from_reader(reader)?;
    Ok(records)
}

async fn ensure_collection(client: &Qdrant, collection_name: &str, vector_size: u64) -> Result<()> {
    let create_collection = CreateCollectionBuilder::new(collection_name.to_string())
        .vectors_config(VectorParamsBuilder::new(vector_size, Distance::Cosine));

    match client.create_collection(create_collection).await {
        Ok(_) => {
            println!(
                "Created collection `{}` (dim: {}, distance: Cosine)",
                collection_name, vector_size
            );
        }
        Err(err) => {
            // Qdrant returns a tonic Status when the collection already exists.
            if err.to_string().contains("already exists") {
                println!("Collection `{}` already exists. Skipping creation.", collection_name);
            } else {
                return Err(err).context("Failed to create collection in Qdrant");
            }
        }
    }

    Ok(())
}

fn resolve_embeddings_path(path_str: &str) -> PathBuf {
    let path = PathBuf::from(path_str);
    if path.exists() {
        return path;
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let manifest_candidate = manifest_dir.join(path_str);
    if manifest_candidate.exists() {
        return manifest_candidate;
    }

    if let Some(parent) = manifest_dir.parent() {
        let parent_candidate = parent.join(path_str);
        if parent_candidate.exists() {
            return parent_candidate;
        }
    }

    path
}

fn extract_point_id(raw_id: &str, fallback: u64) -> u64 {
    raw_id
        .rsplit('_')
        .next()
        .and_then(|part| part.parse::<u64>().ok())
        .unwrap_or(fallback)
}
