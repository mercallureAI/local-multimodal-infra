use chrono::{DateTime, Utc};
use local_core::{AssetKind, AssetListQuery, AssetListResponse, AssetRecord, FileRef};
use local_error::{InfraError, Result};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
};

pub fn local_path(file: &FileRef) -> Result<PathBuf> {
    file.path.clone().ok_or_else(|| {
        InfraError::BadRequest(
            "FileRef.path is required for local MVP; JSON base64 media is not accepted".to_string(),
        )
    })
}

pub fn sha256_file(path: impl AsRef<Path>) -> Result<String> {
    let path = path.as_ref();
    let mut file = File::open(path).map_err(|e| InfraError::io(Some(path.to_path_buf()), e))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 16 * 1024];
    loop {
        let read = file
            .read(&mut buf)
            .map_err(|e| InfraError::io(Some(path.to_path_buf()), e))?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[derive(Debug, Clone)]
pub struct AssetsStore {
    root: PathBuf,
}

impl AssetsStore {
    pub fn new(data_dir: impl Into<PathBuf>) -> Self {
        Self {
            root: data_dir.into().join("assets"),
        }
    }

    pub fn put(
        &self,
        kind: AssetKind,
        path: &str,
        content_type: Option<String>,
        expires_at: Option<DateTime<Utc>>,
        bytes: &[u8],
    ) -> Result<AssetRecord> {
        self.cleanup_expired_before_access("put");
        let relative = normalize_asset_path(path)?;
        let asset_path = self.asset_path(kind, &relative);
        if let Some(parent) = asset_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| InfraError::io(Some(parent.to_path_buf()), e))?;
        }
        let mut file =
            File::create(&asset_path).map_err(|e| InfraError::io(Some(asset_path.clone()), e))?;
        file.write_all(bytes)
            .map_err(|e| InfraError::io(Some(asset_path.clone()), e))?;
        let sha256 = sha256_bytes(bytes);
        let record = AssetRecord {
            uri: asset_uri(kind, &relative),
            kind,
            path: relative,
            content_type,
            size: bytes.len() as u64,
            sha256: Some(sha256),
            created_at: Utc::now(),
            expires_at,
            download_url: None,
            upload_url: None,
        };
        self.write_record(&record)?;
        Ok(record)
    }

    pub fn get(&self, uri_or_kind_path: &str) -> Result<AssetRecord> {
        self.cleanup_expired_before_access("get");
        let (kind, relative) = parse_asset_ref(uri_or_kind_path)?;
        let record = self.read_record(kind, &relative)?;
        if is_expired(&record) {
            self.delete(&record.uri)?;
            return Err(InfraError::NotFound(format!("asset `{}`", record.uri)));
        }
        Ok(record)
    }

    pub fn local_path(&self, uri_or_kind_path: &str) -> Result<PathBuf> {
        let record = self.get(uri_or_kind_path)?;
        Ok(self.asset_path(record.kind, &record.path))
    }

    pub fn read_bytes(&self, uri_or_kind_path: &str) -> Result<(AssetRecord, Vec<u8>)> {
        let record = self.get(uri_or_kind_path)?;
        let path = self.asset_path(record.kind, &record.path);
        let bytes = fs::read(&path).map_err(|e| InfraError::io(Some(path), e))?;
        Ok((record, bytes))
    }

    pub fn list(&self, query: &AssetListQuery) -> Result<AssetListResponse> {
        let mut assets = self.list_without_cleanup(query)?.assets;
        assets.retain(|record| {
            (query.include_expired || !is_expired(record))
                && query
                    .prefix
                    .as_ref()
                    .is_none_or(|prefix| record.path.starts_with(prefix))
                && query.contains.as_ref().is_none_or(|needle| {
                    record.path.contains(needle) || record.uri.contains(needle)
                })
        });
        assets.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(AssetListResponse { assets })
    }

    pub fn delete(&self, uri_or_kind_path: &str) -> Result<()> {
        let (kind, relative) = parse_asset_ref(uri_or_kind_path)?;
        let asset_path = self.asset_path(kind, &relative);
        let meta_path = self.meta_path(kind, &relative);
        remove_file_if_exists(&asset_path)?;
        remove_file_if_exists(&meta_path)?;
        Ok(())
    }

    pub fn record_for_existing(
        &self,
        kind: AssetKind,
        asset_path: &str,
        source: &Path,
        content_type: Option<String>,
        expires_at: Option<DateTime<Utc>>,
    ) -> Result<AssetRecord> {
        let bytes = fs::read(source).map_err(|e| InfraError::io(Some(source.to_path_buf()), e))?;
        self.put(kind, asset_path, content_type, expires_at, &bytes)
    }

    fn asset_path(&self, kind: AssetKind, relative: &str) -> PathBuf {
        self.root.join(kind.as_str()).join(relative)
    }

    fn meta_root(&self, kind: AssetKind) -> PathBuf {
        self.root.join(".metadata").join(kind.as_str())
    }

    fn meta_path(&self, kind: AssetKind, relative: &str) -> PathBuf {
        self.meta_root(kind).join(format!("{relative}.json"))
    }

    fn write_record(&self, record: &AssetRecord) -> Result<()> {
        let path = self.meta_path(record.kind, &record.path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| InfraError::io(Some(parent.to_path_buf()), e))?;
        }
        let bytes = serde_json::to_vec_pretty(record)?;
        fs::write(&path, bytes).map_err(|e| InfraError::io(Some(path), e))
    }

    fn read_record(&self, kind: AssetKind, relative: &str) -> Result<AssetRecord> {
        let path = self.meta_path(kind, relative);
        let bytes = fs::read(&path).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => {
                InfraError::NotFound(format!("asset `{}`", asset_uri(kind, relative)))
            }
            _ => InfraError::io(Some(path.clone()), e),
        })?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    fn collect_records(
        &self,
        kind: AssetKind,
        dir: &Path,
        out: &mut Vec<AssetRecord>,
    ) -> Result<()> {
        for entry in fs::read_dir(dir).map_err(|e| InfraError::io(Some(dir.to_path_buf()), e))? {
            let entry = entry.map_err(|e| InfraError::io(Some(dir.to_path_buf()), e))?;
            let path = entry.path();
            if path.is_dir() {
                self.collect_records(kind, &path, out)?;
            } else if path.extension().and_then(|value| value.to_str()) == Some("json") {
                let bytes = fs::read(&path).map_err(|e| InfraError::io(Some(path.clone()), e))?;
                let record: AssetRecord = serde_json::from_slice(&bytes)?;
                if record.kind == kind {
                    out.push(record);
                }
            }
        }
        Ok(())
    }

    fn cleanup_expired_before_access(&self, operation: &str) {
        if let Err(err) = self.cleanup_expired() {
            tracing::warn!(operation, error = %err, "asset cleanup failed before store access");
        }
    }

    pub fn cleanup_expired(&self) -> Result<()> {
        let query = AssetListQuery {
            include_expired: true,
            ..AssetListQuery::default()
        };
        let response = self.list_without_cleanup(&query)?;
        for record in response.assets.into_iter().filter(is_expired) {
            self.delete(&record.uri)?;
        }
        Ok(())
    }

    fn list_without_cleanup(&self, query: &AssetListQuery) -> Result<AssetListResponse> {
        let mut assets = Vec::new();
        for kind in [AssetKind::Material, AssetKind::Artifact] {
            let meta_root = self.meta_root(kind);
            if meta_root.exists() {
                self.collect_records(kind, &meta_root, &mut assets)?;
            }
        }
        assets.retain(|record| {
            query.kind.is_none_or(|kind| record.kind == kind)
                && query
                    .prefix
                    .as_ref()
                    .is_none_or(|prefix| record.path.starts_with(prefix))
                && query.contains.as_ref().is_none_or(|needle| {
                    record.path.contains(needle) || record.uri.contains(needle)
                })
        });
        Ok(AssetListResponse { assets })
    }
}

pub fn normalize_asset_path(path: &str) -> Result<String> {
    let raw = path.replace('\\', "/");
    if raw.trim().is_empty() || raw.starts_with('/') || raw.contains(':') {
        return Err(InfraError::BadRequest(
            "asset path must be a relative folder/file path".to_string(),
        ));
    }
    let mut parts = Vec::new();
    for component in Path::new(&raw).components() {
        match component {
            Component::Normal(value) => {
                let part = value.to_str().ok_or_else(|| {
                    InfraError::BadRequest("asset path must be valid UTF-8".to_string())
                })?;
                if part.is_empty() || part == "." || part == ".." {
                    return Err(InfraError::BadRequest(
                        "asset path contains an invalid segment".to_string(),
                    ));
                }
                parts.push(part.to_string());
            }
            Component::CurDir => {}
            _ => {
                return Err(InfraError::BadRequest(
                    "asset path may not escape the assets root".to_string(),
                ))
            }
        }
    }
    if parts.is_empty() {
        return Err(InfraError::BadRequest(
            "asset path must include a file name".to_string(),
        ));
    }
    Ok(parts.join("/"))
}

pub fn asset_uri(kind: AssetKind, path: &str) -> String {
    format!("assets://{}/{}", kind.as_str(), path)
}

pub fn parse_asset_uri(uri: &str) -> Result<(AssetKind, String)> {
    let rest = uri
        .strip_prefix("assets://")
        .ok_or_else(|| InfraError::BadRequest("asset URI must start with assets://".to_string()))?;
    let (kind, path) = rest.split_once('/').ok_or_else(|| {
        InfraError::BadRequest("asset URI must be assets://<kind>/<path>".to_string())
    })?;
    let kind = kind.parse::<AssetKind>().map_err(InfraError::BadRequest)?;
    let path = normalize_asset_path(path)?;
    Ok((kind, path))
}

fn parse_asset_ref(value: &str) -> Result<(AssetKind, String)> {
    if value.starts_with("assets://") {
        parse_asset_uri(value)
    } else {
        Err(InfraError::BadRequest(
            "asset reference must be an assets:// URI".to_string(),
        ))
    }
}

fn is_expired(record: &AssetRecord) -> bool {
    record
        .expires_at
        .is_some_and(|expires| expires <= Utc::now())
}

fn remove_file_if_exists(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(InfraError::io(Some(path.to_path_buf()), err)),
    }
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn rejects_escaping_asset_paths() {
        assert!(normalize_asset_path("../x").is_err());
        assert!(normalize_asset_path("/x").is_err());
        assert!(normalize_asset_path("C:/x").is_err());
        assert_eq!(
            normalize_asset_path("folder/file.txt").unwrap(),
            "folder/file.txt"
        );
    }

    #[test]
    fn list_include_expired_can_observe_records_until_cleanup_runs() {
        let root = test_data_dir();
        let store = AssetsStore::new(&root);
        let expired_at = Utc::now() - Duration::seconds(1);
        let record = store
            .put(
                AssetKind::Material,
                "user/expired.txt",
                Some("text/plain".to_string()),
                Some(expired_at),
                b"expired",
            )
            .expect("put expired asset");

        let default_list = store
            .list(&AssetListQuery::default())
            .expect("list non-expired");
        assert!(default_list.assets.is_empty());

        let include_expired = store
            .list(&AssetListQuery {
                include_expired: true,
                ..AssetListQuery::default()
            })
            .expect("list including expired");
        assert_eq!(include_expired.assets.len(), 1);
        assert_eq!(include_expired.assets[0].uri, record.uri);

        store.cleanup_expired().expect("cleanup expired assets");
        let after_cleanup = store
            .list(&AssetListQuery {
                include_expired: true,
                ..AssetListQuery::default()
            })
            .expect("list after cleanup");
        assert!(after_cleanup.assets.is_empty());

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn get_deletes_expired_asset_and_returns_not_found() {
        let root = test_data_dir();
        let store = AssetsStore::new(&root);
        let record = store
            .put(
                AssetKind::Artifact,
                "tasks/task/output.wav",
                Some("audio/wav".to_string()),
                Some(Utc::now() - Duration::seconds(1)),
                b"RIFF",
            )
            .expect("put expired artifact");

        let err = store
            .get(&record.uri)
            .expect_err("expired asset should not be returned by get");
        assert!(matches!(err, InfraError::NotFound(_)), "{err}");
        assert!(!store.asset_path(record.kind, &record.path).exists());
        assert!(!store.meta_path(record.kind, &record.path).exists());

        let _ = fs::remove_dir_all(root);
    }

    fn test_data_dir() -> PathBuf {
        std::env::temp_dir().join(format!(
            "local-files-test-{}",
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ))
    }
}
