use std::path::Path;

use anyhow::{Context, Result};

/// Try to load environment variables from a `.env` file.
/// Looks in the crate directory first, then the workspace root.
pub fn load_env() -> Result<()> {
    if let Ok(path) = dotenvy::dotenv() {
        log_loaded(&path);
        return Ok(());
    }

    let workspace_env = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|root| root.join(".env"));

    if let Some(path) = workspace_env {
        if path.exists() {
            dotenvy::from_path(&path)
                .with_context(|| format!("Failed to load environment from {}", path.display()))?;
            log_loaded(&path);
        }
    }

    Ok(())
}

fn log_loaded(path: &Path) {
    println!("Loaded environment variables from {}", path.display());
}
