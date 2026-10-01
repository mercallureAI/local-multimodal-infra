use chrono::Utc;
use local_core::{
    ArtifactKind, DownloadState, DownloadStatus, InferenceOutput, InferenceTask, JobState,
    ModelArtifact, ModelDownloadStatus, ModelSpec, NodeStatus, TaskStatus,
};
use local_error::{InfraError, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Component, Path, PathBuf},
};

#[derive(Clone)]
pub struct SqliteModelStore {
    database_path: PathBuf,
    model_dir: PathBuf,
    http: reqwest::Client,
}

impl SqliteModelStore {
    pub fn new(database_path: impl Into<PathBuf>, model_dir: impl Into<PathBuf>) -> Result<Self> {
        let store = Self {
            database_path: database_path.into(),
            model_dir: model_dir.into(),
            http: reqwest::Client::new(),
        };
        store.init()?;
        Ok(store)
    }

    pub fn database_path(&self) -> &Path {
        &self.database_path
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    fn init(&self) -> Result<()> {
        if let Some(parent) = self.database_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| InfraError::io(Some(parent.to_path_buf()), e))?;
        }
        fs::create_dir_all(&self.model_dir)
            .map_err(|e| InfraError::io(Some(self.model_dir.clone()), e))?;
        let conn = self.conn()?;
        conn.execute_batch(
            r#"
            PRAGMA journal_mode = WAL;
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                enabled INTEGER NOT NULL,
                spec_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS artifact_downloads (
                model_id TEXT NOT NULL,
                artifact TEXT NOT NULL,
                state TEXT NOT NULL,
                path TEXT,
                sha256 TEXT,
                message TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (model_id, artifact)
            );
            CREATE TABLE IF NOT EXISTS workers (
                node_id TEXT PRIMARY KEY,
                status_json TEXT NOT NULL,
                session_token_hash TEXT,
                last_heartbeat_at TEXT,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                task_json TEXT NOT NULL,
                output_json TEXT,
                error TEXT,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS generic_tasks (
                task_id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                status_json TEXT NOT NULL,
                error TEXT,
                updated_at TEXT NOT NULL
            );
            "#,
        )
        .map_err(sql_err)?;
        ensure_column(&conn, "workers", "session_token_hash", "TEXT")?;
        ensure_column(&conn, "workers", "last_heartbeat_at", "TEXT")?;
        Ok(())
    }

    fn conn(&self) -> Result<Connection> {
        Connection::open(&self.database_path).map_err(sql_err)
    }

    pub fn delete_models<const N: usize>(&self, ids: [&str; N]) -> Result<()> {
        let conn = self.conn()?;
        for id in ids {
            conn.execute("DELETE FROM models WHERE id = ?1", params![id])
                .map_err(sql_err)?;
            conn.execute(
                "DELETE FROM artifact_downloads WHERE model_id = ?1",
                params![id],
            )
            .map_err(sql_err)?;
        }
        Ok(())
    }

    pub fn seed_models(&self, specs: Vec<ModelSpec>) -> Result<()> {
        for spec in specs {
            self.upsert_model_preserving_enabled(spec)?;
        }
        Ok(())
    }

    pub fn upsert_model(&self, spec: ModelSpec) -> Result<ModelSpec> {
        self.write_model(spec)
    }

    fn upsert_model_preserving_enabled(&self, mut spec: ModelSpec) -> Result<ModelSpec> {
        if let Some(existing) = self.get_model(&spec.id)? {
            spec.enabled = existing.enabled;
        }
        self.write_model(spec)
    }

    fn write_model(&self, spec: ModelSpec) -> Result<ModelSpec> {
        let spec = self.normalize_model_spec(spec)?;
        let mut conn = self.conn()?;
        let json = serde_json::to_string(&spec)?;
        let transaction = conn.transaction().map_err(sql_err)?;
        let existing_json = transaction
            .query_row(
                "SELECT spec_json FROM models WHERE id = ?1",
                params![spec.id],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(sql_err)?;
        let artifacts_changed = existing_json
            .map(|existing_json| -> Result<bool> {
                let existing: ModelSpec = serde_json::from_str(&existing_json)?;
                Ok(serde_json::to_value(existing.artifacts)?
                    != serde_json::to_value(&spec.artifacts)?)
            })
            .transpose()?
            .unwrap_or(false);
        if artifacts_changed {
            transaction
                .execute(
                    "DELETE FROM artifact_downloads WHERE model_id = ?1",
                    params![spec.id],
                )
                .map_err(sql_err)?;
        }
        transaction.execute(
            "INSERT INTO models (id, enabled, spec_json, updated_at) VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(id) DO UPDATE SET enabled=excluded.enabled, spec_json=excluded.spec_json, updated_at=excluded.updated_at",
            params![spec.id, bool_to_i64(spec.enabled), json, Utc::now().to_rfc3339()],
        )
        .map_err(sql_err)?;
        transaction.commit().map_err(sql_err)?;
        Ok(spec)
    }

    pub fn normalize_model_spec(&self, mut spec: ModelSpec) -> Result<ModelSpec> {
        let base = stable_model_base(&self.model_dir, &spec.id)?;
        for artifact in &mut spec.artifacts {
            normalize_artifact_path(artifact, &base)?;
        }
        Ok(spec)
    }

    pub fn list_models(&self) -> Result<Vec<ModelSpec>> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare("SELECT enabled, spec_json FROM models ORDER BY id")
            .map_err(sql_err)?;
        let rows = stmt
            .query_map([], |row| {
                let enabled: i64 = row.get(0)?;
                let spec_json: String = row.get(1)?;
                Ok((enabled, spec_json))
            })
            .map_err(sql_err)?;
        let mut models = Vec::new();
        for row in rows {
            let (enabled, json) = row.map_err(sql_err)?;
            let mut spec: ModelSpec = serde_json::from_str(&json)?;
            spec.enabled = enabled != 0;
            models.push(spec);
        }
        Ok(models)
    }

    pub fn get_model(&self, id: &str) -> Result<Option<ModelSpec>> {
        let conn = self.conn()?;
        conn.query_row(
            "SELECT enabled, spec_json FROM models WHERE id = ?1",
            params![id],
            |row| {
                let enabled: i64 = row.get(0)?;
                let spec_json: String = row.get(1)?;
                Ok((enabled, spec_json))
            },
        )
        .optional()
        .map_err(sql_err)?
        .map(|(enabled, json)| {
            let mut spec: ModelSpec = serde_json::from_str(&json)?;
            spec.enabled = enabled != 0;
            Ok(spec)
        })
        .transpose()
    }

    pub fn set_enabled(&self, id: &str, enabled: bool) -> Result<ModelSpec> {
        let mut spec = self
            .get_model(id)?
            .ok_or_else(|| InfraError::NotFound(format!("model `{id}`")))?;
        spec.enabled = enabled;
        self.write_model(spec)
    }

    pub fn record_worker(&self, status: &NodeStatus) -> Result<()> {
        self.record_worker_auth(status, None)
    }

    pub fn record_worker_auth(
        &self,
        status: &NodeStatus,
        session_token_hash: Option<&str>,
    ) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            "INSERT INTO workers (node_id, status_json, session_token_hash, last_heartbeat_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(node_id) DO UPDATE SET status_json=excluded.status_json,
                session_token_hash=COALESCE(excluded.session_token_hash, workers.session_token_hash),
                last_heartbeat_at=excluded.last_heartbeat_at,
                updated_at=excluded.updated_at",
            params![
                status.registration.node_id,
                serde_json::to_string(status)?,
                session_token_hash,
                status.last_heartbeat_at.to_rfc3339(),
                Utc::now().to_rfc3339()
            ],
        )
        .map_err(sql_err)?;
        Ok(())
    }

    pub fn record_generic_task(&self, status: &TaskStatus) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            "INSERT INTO generic_tasks (task_id, state, status_json, error, updated_at) VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(task_id) DO UPDATE SET state=excluded.state, status_json=excluded.status_json, error=excluded.error, updated_at=excluded.updated_at",
            params![
                status.task_id,
                serde_json::to_string(&status.state)?,
                serde_json::to_string(status)?,
                status.error,
                status.updated_at.to_rfc3339()
            ],
        )
        .map_err(sql_err)?;
        Ok(())
    }

    pub fn get_generic_task(&self, task_id: &str) -> Result<Option<TaskStatus>> {
        let conn = self.conn()?;
        conn.query_row(
            "SELECT status_json FROM generic_tasks WHERE task_id = ?1",
            params![task_id],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(sql_err)?
        .map(|json| serde_json::from_str(&json).map_err(InfraError::from))
        .transpose()
    }

    pub fn record_job_state(
        &self,
        task: &InferenceTask,
        state: JobState,
        output: Option<&InferenceOutput>,
        error: Option<&str>,
    ) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            "INSERT INTO jobs (job_id, state, task_json, output_json, error, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(job_id) DO UPDATE SET state=excluded.state, task_json=excluded.task_json, output_json=excluded.output_json, error=excluded.error, updated_at=excluded.updated_at",
            params![
                task.id.to_string(),
                serde_json::to_string(&state)?,
                serde_json::to_string(task)?,
                output.map(serde_json::to_string).transpose()?,
                error,
                Utc::now().to_rfc3339()
            ],
        )
        .map_err(sql_err)?;
        Ok(())
    }

    pub async fn download_model(&self, spec: &ModelSpec) -> Result<Vec<DownloadStatus>> {
        let spec = self.write_model(spec.clone())?;
        self.download_model_artifacts(&spec).await
    }

    pub async fn download_model_artifacts(&self, spec: &ModelSpec) -> Result<Vec<DownloadStatus>> {
        let mut statuses = Vec::new();
        for (index, artifact) in spec.artifacts.iter().enumerate() {
            match artifact.kind {
                ArtifactKind::Local => {
                    let status = self.import_local_artifact(&spec.id, index, artifact)?;
                    statuses.push(status);
                }
                ArtifactKind::Url => {
                    let url = artifact.url.as_deref().ok_or_else(|| {
                        InfraError::Config(format!(
                            "URL artifact for model `{}` is missing url",
                            spec.id
                        ))
                    })?;
                    let key = format!("url:{index}");
                    if let Some(status) = self.reusable_download_status(
                        &spec.id,
                        &key,
                        &artifact.path,
                        artifact.sha256.as_deref(),
                    )? {
                        statuses.push(status);
                        continue;
                    }
                    self.status(
                        &spec.id,
                        key.clone(),
                        DownloadState::Downloading,
                        Some(artifact.path.clone()),
                        artifact.sha256.clone(),
                        Some("download started".to_string()),
                    )?;
                    let status = self
                        .download_url(
                            &spec.id,
                            key,
                            url,
                            &artifact.path,
                            artifact.sha256.as_deref(),
                        )
                        .await?;
                    statuses.push(status);
                }
                ArtifactKind::HuggingFace => {
                    let repo_id = artifact.repo_id.as_deref().ok_or_else(|| {
                        InfraError::Config(format!(
                            "Hugging Face artifact for model `{}` is missing repo_id",
                            spec.id
                        ))
                    })?;
                    let revision = artifact.revision.as_deref().unwrap_or("main");
                    let files = if artifact.files.is_empty() && !artifact.allow_patterns.is_empty()
                    {
                        let files = self
                            .hf_files_for_patterns(repo_id, revision, &artifact.allow_patterns)
                            .await?;
                        self.delete_download_status(&spec.id, &hf_metadata_key(index))?;
                        files
                    } else {
                        artifact.files.clone()
                    };
                    if files.is_empty() {
                        return Err(InfraError::Config(format!(
                            "Hugging Face artifact for model `{}` has no files or allow_patterns",
                            spec.id
                        )));
                    }
                    for file in files {
                        let target = hf_target_path(artifact, &file);
                        let key = format!("hf:{index}:{file}");
                        if let Some(status) = self.reusable_download_status(
                            &spec.id,
                            &key,
                            &target,
                            artifact.sha256.as_deref(),
                        )? {
                            statuses.push(status);
                            continue;
                        }
                        self.status(
                            &spec.id,
                            key.clone(),
                            DownloadState::Downloading,
                            Some(target.clone()),
                            artifact.sha256.clone(),
                            Some("download started".to_string()),
                        )?;
                        let url = huggingface_resolve_url(repo_id, revision, &file);
                        let status = self
                            .download_url(&spec.id, key, &url, &target, artifact.sha256.as_deref())
                            .await?;
                        statuses.push(status);
                    }
                }
            }
        }
        Ok(statuses)
    }

    pub fn prepare_model_download(&self, spec: &ModelSpec) -> Result<ModelDownloadStatus> {
        let spec = self.write_model(spec.clone())?;
        for expected in expected_downloads(&spec) {
            if self
                .reusable_download_status(
                    &spec.id,
                    &expected.artifact,
                    &expected.path,
                    expected.sha256.as_deref(),
                )?
                .is_some()
            {
                continue;
            }
            self.status(
                &spec.id,
                expected.artifact,
                DownloadState::Downloading,
                Some(expected.path),
                expected.sha256,
                Some("download queued".to_string()),
            )?;
        }
        for (index, artifact) in spec.artifacts.iter().enumerate() {
            if matches!(artifact.kind, ArtifactKind::HuggingFace)
                && artifact.files.is_empty()
                && !artifact.allow_patterns.is_empty()
            {
                self.status(
                    &spec.id,
                    hf_metadata_key(index),
                    DownloadState::Downloading,
                    Some(artifact.path.clone()),
                    artifact.sha256.clone(),
                    Some("resolving Hugging Face allow_patterns".to_string()),
                )?;
            }
        }
        self.model_download_status(&spec)
    }

    pub fn fail_active_model_download(&self, model_id: &str, error: &str) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            "UPDATE artifact_downloads SET state = 'failed', message = ?2, updated_at = ?3 WHERE model_id = ?1 AND state = 'downloading'",
            params![model_id, error, Utc::now().to_rfc3339()],
        )
        .map_err(sql_err)?;
        Ok(())
    }

    pub fn model_download_status(&self, spec: &ModelSpec) -> Result<ModelDownloadStatus> {
        let mut recorded = self
            .download_statuses(&spec.id)?
            .into_iter()
            .map(|status| (status.artifact.clone(), status))
            .collect::<BTreeMap<_, _>>();
        let mut artifacts = Vec::new();

        for expected in expected_downloads(spec) {
            let recorded_status = recorded.remove(&expected.artifact).filter(|status| {
                status.path.as_ref() == Some(&expected.path) && status.sha256 == expected.sha256
            });
            let status = match recorded_status {
                Some(status) => refresh_recorded_status(status),
                None if expected.path.is_file() && expected.sha256.is_some() => {
                    let expected_sha256 = expected.sha256.as_deref().expect("sha256 checked");
                    let actual = sha256_file(&expected.path)?;
                    let (state, message) = if actual.eq_ignore_ascii_case(expected_sha256) {
                        (
                            DownloadState::Downloaded,
                            format!("existing artifact verified with sha256 {actual}"),
                        )
                    } else {
                        (
                            DownloadState::Failed,
                            format!(
                                "artifact sha256 mismatch: expected {expected_sha256}, got {actual}"
                            ),
                        )
                    };
                    DownloadStatus {
                        model_id: spec.id.clone(),
                        artifact: expected.artifact,
                        state,
                        path: Some(expected.path),
                        sha256: expected.sha256,
                        message: Some(message),
                        updated_at: Utc::now(),
                    }
                }
                None if matches!(expected.kind, ArtifactKind::Local) && expected.path.exists() => {
                    let (state, message) = match expected.sha256.as_deref() {
                        Some(_) => (
                            DownloadState::Failed,
                            "cannot verify sha256 for a local directory artifact".to_string(),
                        ),
                        None => (
                            DownloadState::Downloaded,
                            "local artifact already exists".to_string(),
                        ),
                    };
                    DownloadStatus {
                        model_id: spec.id.clone(),
                        artifact: expected.artifact,
                        state,
                        path: Some(expected.path),
                        sha256: expected.sha256,
                        message: Some(message),
                        updated_at: Utc::now(),
                    }
                }
                None => DownloadStatus {
                    model_id: spec.id.clone(),
                    artifact: expected.artifact,
                    state: DownloadState::NotStarted,
                    path: Some(expected.path),
                    sha256: expected.sha256,
                    message: Some("artifact has no matching completed download record".to_string()),
                    updated_at: Utc::now(),
                },
            };
            artifacts.push(status);
        }
        for (prefix, sha256) in dynamic_hf_status_prefixes(spec) {
            artifacts.extend(
                recorded
                    .iter()
                    .filter(|(artifact, status)| {
                        artifact.starts_with(&prefix) && status.sha256 == sha256
                    })
                    .map(|(_, status)| refresh_recorded_status(status.clone())),
            );
        }
        artifacts.sort_by(|left, right| left.artifact.cmp(&right.artifact));
        artifacts.dedup_by(|left, right| left.artifact == right.artifact);

        let downloaded = !artifacts.is_empty()
            && artifacts
                .iter()
                .all(|status| matches!(status.state, DownloadState::Downloaded));
        let state = if downloaded {
            DownloadState::Downloaded
        } else if artifacts
            .iter()
            .any(|status| matches!(status.state, DownloadState::Downloading))
        {
            DownloadState::Downloading
        } else if artifacts
            .iter()
            .any(|status| matches!(status.state, DownloadState::Failed))
        {
            DownloadState::Failed
        } else {
            DownloadState::NotStarted
        };
        let updated_at = artifacts.iter().map(|status| status.updated_at).max();
        Ok(ModelDownloadStatus {
            model_id: spec.id.clone(),
            downloaded,
            state,
            artifacts,
            updated_at,
        })
    }

    fn download_statuses(&self, model_id: &str) -> Result<Vec<DownloadStatus>> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT artifact, state, path, sha256, message, updated_at FROM artifact_downloads WHERE model_id = ?1 ORDER BY artifact",
            )
            .map_err(sql_err)?;
        let rows = stmt
            .query_map(params![model_id], |row| {
                let state: String = row.get(1)?;
                let updated_at: String = row.get(5)?;
                Ok((
                    row.get::<_, String>(0)?,
                    state,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, Option<String>>(4)?,
                    updated_at,
                ))
            })
            .map_err(sql_err)?;
        let mut statuses = Vec::new();
        for row in rows {
            let (artifact, state, path, sha256, message, updated_at) = row.map_err(sql_err)?;
            statuses.push(DownloadStatus {
                model_id: model_id.to_string(),
                artifact,
                state: parse_download_state(&state)?,
                path: path.map(PathBuf::from),
                sha256,
                message,
                updated_at: chrono::DateTime::parse_from_rfc3339(&updated_at)
                    .map_err(|error| {
                        InfraError::Backend(format!("invalid download timestamp: {error}"))
                    })?
                    .with_timezone(&Utc),
            });
        }
        Ok(statuses)
    }

    fn delete_download_status(&self, model_id: &str, artifact: &str) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            "DELETE FROM artifact_downloads WHERE model_id = ?1 AND artifact = ?2",
            params![model_id, artifact],
        )
        .map_err(sql_err)?;
        Ok(())
    }

    fn reusable_download_status(
        &self,
        model_id: &str,
        artifact: &str,
        target: &Path,
        expected_sha256: Option<&str>,
    ) -> Result<Option<DownloadStatus>> {
        if !target.is_file() {
            return Ok(None);
        }
        if let Some(expected) = expected_sha256 {
            let actual = sha256_file(target)?;
            if actual.eq_ignore_ascii_case(expected) {
                return self
                    .status(
                        model_id,
                        artifact.to_string(),
                        DownloadState::Downloaded,
                        Some(target.to_path_buf()),
                        Some(expected.to_string()),
                        Some(format!("existing artifact verified with sha256 {actual}")),
                    )
                    .map(Some);
            }
            return Ok(None);
        }
        Ok(self
            .download_statuses(model_id)?
            .into_iter()
            .find(|status| {
                status.artifact == artifact
                    && matches!(status.state, DownloadState::Downloaded)
                    && status.path.as_deref() == Some(target)
                    && status.sha256.is_none()
            }))
    }

    fn import_local_artifact(
        &self,
        model_id: &str,
        index: usize,
        artifact: &ModelArtifact,
    ) -> Result<DownloadStatus> {
        let key = format!("local:{index}");
        let source = artifact.source_path.as_ref().unwrap_or(&artifact.path);
        if artifact.path.exists() {
            if artifact.path.is_file() {
                if let Some(expected) = artifact.sha256.as_deref() {
                    let actual = sha256_file(&artifact.path)?;
                    if !actual.eq_ignore_ascii_case(expected) {
                        return self.status(
                            model_id,
                            key,
                            DownloadState::Failed,
                            Some(artifact.path.clone()),
                            artifact.sha256.clone(),
                            Some(format!(
                                "sha256 mismatch: expected {expected}, got {actual}"
                            )),
                        );
                    }
                }
            }
            let message = if same_path(source, &artifact.path) {
                "stable local artifact exists; source and destination are the same".to_string()
            } else if source.exists() {
                "stable local artifact already exists; no copy needed".to_string()
            } else {
                format!(
                    "stable local artifact exists at {}; source {} is not present",
                    artifact.path.display(),
                    source.display()
                )
            };
            return self.status(
                model_id,
                key,
                DownloadState::Downloaded,
                Some(artifact.path.clone()),
                artifact.sha256.clone(),
                Some(message),
            );
        }
        if !source.exists() {
            return self.status(
                model_id,
                key,
                DownloadState::Failed,
                Some(artifact.path.clone()),
                artifact.sha256.clone(),
                Some(format!(
                    "local source artifact is missing: {}; stable destination is {}",
                    source.display(),
                    artifact.path.display()
                )),
            );
        }
        if source.is_dir() {
            copy_dir_recursive(source, &artifact.path)?;
            self.status(
                model_id,
                key,
                DownloadState::Downloaded,
                Some(artifact.path.clone()),
                artifact.sha256.clone(),
                Some(format!(
                    "copied local directory {} into stable destination {}",
                    source.display(),
                    artifact.path.display()
                )),
            )
        } else {
            if let Some(parent) = artifact.path.parent() {
                fs::create_dir_all(parent)
                    .map_err(|e| InfraError::io(Some(parent.to_path_buf()), e))?;
            }
            fs::copy(source, &artifact.path)
                .map_err(|e| InfraError::io(Some(source.clone()), e))?;
            if let Some(expected) = artifact.sha256.as_deref() {
                let actual = sha256_file(&artifact.path)?;
                if !actual.eq_ignore_ascii_case(expected) {
                    return self.status(
                        model_id,
                        key,
                        DownloadState::Failed,
                        Some(artifact.path.clone()),
                        artifact.sha256.clone(),
                        Some(format!(
                            "sha256 mismatch: expected {expected}, got {actual}"
                        )),
                    );
                }
            }
            self.status(
                model_id,
                key,
                DownloadState::Downloaded,
                Some(artifact.path.clone()),
                artifact.sha256.clone(),
                Some(format!(
                    "copied local file {} into stable destination {}",
                    source.display(),
                    artifact.path.display()
                )),
            )
        }
    }

    async fn hf_files_for_patterns(
        &self,
        repo_id: &str,
        revision: &str,
        patterns: &[String],
    ) -> Result<Vec<String>> {
        for pattern in patterns {
            validate_relative_component_path(pattern, "Hugging Face allow_pattern")?;
        }
        let url = format!("https://huggingface.co/api/models/{repo_id}/revision/{revision}");
        let mut request = self.http.get(&url);
        if let Ok(token) =
            std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGINGFACE_HUB_TOKEN"))
        {
            request = request.bearer_auth(token);
        }
        let response = request
            .send()
            .await
            .map_err(|e| InfraError::Backend(format!("fetch Hugging Face metadata {url}: {e}")))?;
        if !response.status().is_success() {
            let status = response.status();
            return Err(InfraError::Backend(format!(
                "Hugging Face metadata request failed with HTTP status {status}"
            )));
        }
        let metadata = response
            .json::<HfModelInfo>()
            .await
            .map_err(|e| InfraError::Backend(format!("decode Hugging Face metadata: {e}")))?;
        let mut files = metadata
            .siblings
            .into_iter()
            .map(|s| s.rfilename)
            .filter(|file| patterns.iter().any(|pattern| glob_match(pattern, file)))
            .collect::<Vec<_>>();
        files.sort();
        files.dedup();
        if files.is_empty() {
            return Err(InfraError::Config(format!(
                "Hugging Face allow_patterns {:?} matched no files in {repo_id}@{revision}",
                patterns
            )));
        }
        for file in &files {
            validate_relative_component_path(file, "Hugging Face sibling file")?;
        }
        Ok(files)
    }

    async fn download_url(
        &self,
        model_id: &str,
        key: String,
        url: &str,
        target: &Path,
        expected_sha256: Option<&str>,
    ) -> Result<DownloadStatus> {
        if target.exists() {
            let matches = match expected_sha256 {
                Some(expected) if target.is_file() => {
                    sha256_file(target)?.eq_ignore_ascii_case(expected)
                }
                Some(_) | None => false,
            };
            if matches {
                return self.status(
                    model_id,
                    key,
                    DownloadState::Downloaded,
                    Some(target.to_path_buf()),
                    expected_sha256.map(str::to_string),
                    Some("artifact already exists locally; download skipped".to_string()),
                );
            }
        }
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| InfraError::io(Some(parent.to_path_buf()), e))?;
        }
        let mut request = self.http.get(url);
        if let Ok(token) =
            std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGINGFACE_HUB_TOKEN"))
        {
            if url.contains("huggingface.co/") {
                request = request.bearer_auth(token);
            }
        }
        let response = request
            .send()
            .await
            .map_err(|e| InfraError::Backend(format!("download {url}: {e}")))?;
        if !response.status().is_success() {
            let status = response.status();
            return self.status(
                model_id,
                key,
                DownloadState::Failed,
                Some(target.to_path_buf()),
                expected_sha256.map(str::to_string),
                Some(format!("download failed with HTTP status {status}")),
            );
        }
        let bytes = response
            .bytes()
            .await
            .map_err(|e| InfraError::Backend(format!("read downloaded bytes from {url}: {e}")))?;
        let actual = sha256_hex(&bytes);
        if let Some(expected) = expected_sha256 {
            if !actual.eq_ignore_ascii_case(expected) {
                return self.status(
                    model_id,
                    key,
                    DownloadState::Failed,
                    Some(target.to_path_buf()),
                    Some(expected.to_string()),
                    Some(format!(
                        "sha256 mismatch: expected {expected}, got {actual}"
                    )),
                );
            }
        }
        let partial = partial_download_path(target)?;
        let write_result = (|| -> Result<()> {
            let mut file =
                fs::File::create(&partial).map_err(|e| InfraError::io(Some(partial.clone()), e))?;
            file.write_all(&bytes)
                .map_err(|e| InfraError::io(Some(partial.clone()), e))?;
            file.sync_all()
                .map_err(|e| InfraError::io(Some(partial.clone()), e))?;
            replace_download_target(&partial, target)?;
            Ok(())
        })();
        if write_result.is_err() && partial.exists() {
            let _ = fs::remove_file(&partial);
        }
        write_result?;
        let message = if expected_sha256.is_some() {
            Some(format!("downloaded and verified sha256 {actual}"))
        } else {
            Some(format!(
                "downloaded; sha256 not configured, verification skipped (actual {actual})"
            ))
        };
        self.status(
            model_id,
            key,
            DownloadState::Downloaded,
            Some(target.to_path_buf()),
            expected_sha256.map(str::to_string),
            message,
        )
    }

    fn status(
        &self,
        model_id: &str,
        artifact: String,
        state: DownloadState,
        path: Option<PathBuf>,
        sha256: Option<String>,
        message: Option<String>,
    ) -> Result<DownloadStatus> {
        let status = DownloadStatus {
            model_id: model_id.to_string(),
            artifact,
            state,
            path,
            sha256,
            message,
            updated_at: Utc::now(),
        };
        self.record_download_status(&status)?;
        Ok(status)
    }

    fn record_download_status(&self, status: &DownloadStatus) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            "INSERT INTO artifact_downloads (model_id, artifact, state, path, sha256, message, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(model_id, artifact) DO UPDATE SET state=excluded.state, path=excluded.path, sha256=excluded.sha256, message=excluded.message, updated_at=excluded.updated_at",
            params![
                status.model_id,
                status.artifact,
                state_name(&status.state),
                status.path.as_ref().map(|p| p.to_string_lossy().to_string()),
                status.sha256,
                status.message,
                status.updated_at.to_rfc3339(),
            ],
        )
        .map_err(sql_err)?;
        Ok(())
    }
}

pub fn validate_local_artifact(artifact: &ModelArtifact) -> Result<()> {
    match artifact.kind {
        ArtifactKind::Local => {
            if artifact.path.exists() {
                Ok(())
            } else {
                Err(InfraError::ModelNotConfigured {
                    model_id: "unknown".to_string(),
                    reason: format!("local artifact does not exist: {}", artifact.path.display()),
                })
            }
        }
        ArtifactKind::HuggingFace | ArtifactKind::Url => Ok(()),
    }
}

fn normalize_artifact_path(artifact: &mut ModelArtifact, base: &Path) -> Result<()> {
    match artifact.kind {
        ArtifactKind::Local => {
            let original = if artifact.source_path.is_some() {
                artifact.source_path.clone()
            } else if artifact.path.as_os_str().is_empty() {
                None
            } else {
                Some(artifact.path.clone())
            };
            let destination = local_destination(base, original.as_deref())?;
            if let Some(source) = original {
                if !same_path(&source, &destination) {
                    artifact.source_path = Some(source);
                } else {
                    artifact.source_path = None;
                }
            }
            artifact.path = destination;
        }
        ArtifactKind::HuggingFace => {
            for file in &artifact.files {
                validate_relative_component_path(file, "Hugging Face file")?;
            }
            for pattern in &artifact.allow_patterns {
                validate_relative_component_path(pattern, "Hugging Face allow_pattern")?;
            }
            artifact.source_path = None;
            artifact.path = if artifact.files.len() == 1 && artifact.allow_patterns.is_empty() {
                base.join(&artifact.files[0])
            } else {
                base.to_path_buf()
            };
        }
        ArtifactKind::Url => {
            let relative = url_destination_relative(artifact, base)?;
            artifact.source_path = None;
            artifact.path = base.join(relative);
        }
    }
    Ok(())
}

fn stable_model_base(model_dir: &Path, model_id: &str) -> Result<PathBuf> {
    let id_path = Path::new(model_id);
    validate_relative_component_path(id_path, "model_id")?;
    Ok(model_dir.join(id_path))
}

fn local_destination(base: &Path, source: Option<&Path>) -> Result<PathBuf> {
    let Some(source) = source else {
        return Ok(base.to_path_buf());
    };
    if source.starts_with(base) {
        return stable_destination_from_existing(base, source, false, "local artifact path");
    }
    if source.exists() && source.is_dir() {
        return Ok(base.to_path_buf());
    }
    if source.extension().is_none() {
        return Ok(base.to_path_buf());
    }
    let filename = source.file_name().ok_or_else(|| {
        InfraError::Config(format!(
            "local source path {} has no filename for stable artifact destination",
            source.display()
        ))
    })?;
    validate_relative_component_path(Path::new(filename), "local artifact filename")?;
    Ok(base.join(filename))
}

fn url_destination_relative(artifact: &ModelArtifact, base: &Path) -> Result<PathBuf> {
    if !artifact.path.as_os_str().is_empty() {
        if artifact.path.starts_with(base) {
            let destination =
                stable_destination_from_existing(base, &artifact.path, true, "URL artifact path")?;
            return destination
                .strip_prefix(base)
                .map(Path::to_path_buf)
                .map_err(|e| InfraError::Config(format!("strip stable URL path prefix: {e}")));
        }
        if artifact.path.is_absolute() {
            return Err(InfraError::Config(format!(
                "URL artifact destination path must be relative or empty, got {}",
                artifact.path.display()
            )));
        }
        validate_relative_path(&artifact.path, "URL artifact path")?;
        return Ok(artifact.path.clone());
    }
    let url = artifact.url.as_deref().ok_or_else(|| {
        InfraError::Config("URL artifact requires url before path can be materialized".to_string())
    })?;
    let filename = url_basename(url)?;
    validate_relative_component_path(filename, "URL artifact URL basename")?;
    Ok(PathBuf::from(filename))
}

fn stable_destination_from_existing(
    base: &Path,
    path: &Path,
    require_suffix: bool,
    label: &str,
) -> Result<PathBuf> {
    let suffix = path
        .strip_prefix(base)
        .map_err(|e| InfraError::Config(format!("strip stable {label} prefix: {e}")))?;
    if suffix.as_os_str().is_empty() {
        if require_suffix {
            return Err(InfraError::Config(format!(
                "{label} must name a file below stable model root, got {}",
                path.display()
            )));
        }
        return Ok(base.to_path_buf());
    }
    validate_relative_path(suffix, label)?;
    Ok(base.join(suffix))
}

fn validate_relative_component_path(path: impl AsRef<Path>, label: &str) -> Result<()> {
    validate_relative_path(path.as_ref(), label)
}

fn validate_relative_path(path: &Path, label: &str) -> Result<()> {
    if path.as_os_str().is_empty() || path.is_absolute() {
        return Err(InfraError::Config(format!(
            "{label} must be a non-empty relative path, got {}",
            path.display()
        )));
    }
    for component in path.components() {
        match component {
            Component::Normal(_) => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) | Component::CurDir => {
                return Err(InfraError::Config(format!(
                    "{label} must not contain absolute, current-dir, or parent traversal components: {}",
                    path.display()
                )))
            }
        }
    }
    Ok(())
}

fn url_basename(url: &str) -> Result<&str> {
    let without_fragment = url.split('#').next().unwrap_or(url);
    let without_query = without_fragment
        .split('?')
        .next()
        .unwrap_or(without_fragment);
    without_query
        .rsplit('/')
        .next()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| InfraError::Config(format!("URL has no basename: {url}")))
}

fn copy_dir_recursive(source: &Path, destination: &Path) -> Result<()> {
    fs::create_dir_all(destination)
        .map_err(|e| InfraError::io(Some(destination.to_path_buf()), e))?;
    for entry in fs::read_dir(source).map_err(|e| InfraError::io(Some(source.to_path_buf()), e))? {
        let entry = entry.map_err(|e| InfraError::io(Some(source.to_path_buf()), e))?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        if source_path.is_dir() {
            copy_dir_recursive(&source_path, &destination_path)?;
        } else {
            if let Some(parent) = destination_path.parent() {
                fs::create_dir_all(parent)
                    .map_err(|e| InfraError::io(Some(parent.to_path_buf()), e))?;
            }
            fs::copy(&source_path, &destination_path)
                .map_err(|e| InfraError::io(Some(source_path.clone()), e))?;
        }
    }
    Ok(())
}

fn same_path(a: &Path, b: &Path) -> bool {
    a == b
}

#[derive(Debug, Deserialize)]
struct HfModelInfo {
    #[serde(default)]
    siblings: Vec<HfSibling>,
}

#[derive(Debug, Deserialize)]
struct HfSibling {
    rfilename: String,
}

fn glob_match(pattern: &str, value: &str) -> bool {
    glob_match_bytes(pattern.as_bytes(), value.as_bytes())
}

fn glob_match_bytes(pattern: &[u8], value: &[u8]) -> bool {
    if pattern.is_empty() {
        return value.is_empty();
    }
    match pattern[0] {
        b'*' => {
            glob_match_bytes(&pattern[1..], value)
                || (!value.is_empty() && glob_match_bytes(pattern, &value[1..]))
        }
        b'?' => !value.is_empty() && glob_match_bytes(&pattern[1..], &value[1..]),
        byte => {
            !value.is_empty() && byte == value[0] && glob_match_bytes(&pattern[1..], &value[1..])
        }
    }
}

fn hf_target_path(artifact: &ModelArtifact, file: &str) -> PathBuf {
    if artifact.files.len() == 1 {
        artifact.path.clone()
    } else {
        artifact.path.join(file)
    }
}

fn huggingface_resolve_url(repo_id: &str, revision: &str, file: &str) -> String {
    format!("https://huggingface.co/{repo_id}/resolve/{revision}/{file}")
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).map_err(|e| InfraError::io(Some(path.to_path_buf()), e))?;
    Ok(sha256_hex(&bytes))
}

fn bool_to_i64(value: bool) -> i64 {
    if value {
        1
    } else {
        0
    }
}

fn ensure_column(conn: &Connection, table: &str, column: &str, ty: &str) -> Result<()> {
    let mut stmt = conn
        .prepare(&format!("PRAGMA table_info({table})"))
        .map_err(sql_err)?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(sql_err)?;
    for row in rows {
        if row.map_err(sql_err)? == column {
            return Ok(());
        }
    }
    conn.execute(&format!("ALTER TABLE {table} ADD COLUMN {column} {ty}"), [])
        .map_err(sql_err)?;
    Ok(())
}

fn state_name(state: &DownloadState) -> &'static str {
    match state {
        DownloadState::NotStarted => "not_started",
        DownloadState::Downloading => "downloading",
        DownloadState::Downloaded => "downloaded",
        DownloadState::Failed => "failed",
        DownloadState::Skipped => "skipped",
    }
}

fn parse_download_state(state: &str) -> Result<DownloadState> {
    match state {
        "not_started" => Ok(DownloadState::NotStarted),
        "downloading" => Ok(DownloadState::Downloading),
        "downloaded" => Ok(DownloadState::Downloaded),
        "failed" => Ok(DownloadState::Failed),
        "skipped" => Ok(DownloadState::Skipped),
        other => Err(InfraError::Backend(format!(
            "unknown artifact download state `{other}`"
        ))),
    }
}

struct ExpectedDownload {
    artifact: String,
    kind: ArtifactKind,
    path: PathBuf,
    sha256: Option<String>,
}

fn expected_downloads(spec: &ModelSpec) -> Vec<ExpectedDownload> {
    let mut expected = Vec::new();
    for (index, artifact) in spec.artifacts.iter().enumerate() {
        match artifact.kind {
            ArtifactKind::Local => expected.push(ExpectedDownload {
                artifact: format!("local:{index}"),
                kind: artifact.kind,
                path: artifact.path.clone(),
                sha256: artifact.sha256.clone(),
            }),
            ArtifactKind::Url => expected.push(ExpectedDownload {
                artifact: format!("url:{index}"),
                kind: artifact.kind,
                path: artifact.path.clone(),
                sha256: artifact.sha256.clone(),
            }),
            ArtifactKind::HuggingFace => {
                expected.extend(artifact.files.iter().map(|file| ExpectedDownload {
                    artifact: format!("hf:{index}:{file}"),
                    kind: artifact.kind,
                    path: hf_target_path(artifact, file),
                    sha256: artifact.sha256.clone(),
                }));
            }
        }
    }
    expected
}

fn dynamic_hf_status_prefixes(spec: &ModelSpec) -> Vec<(String, Option<String>)> {
    spec.artifacts
        .iter()
        .enumerate()
        .filter(|(_, artifact)| {
            matches!(artifact.kind, ArtifactKind::HuggingFace)
                && artifact.files.is_empty()
                && !artifact.allow_patterns.is_empty()
        })
        .map(|(index, artifact)| (format!("hf:{index}:"), artifact.sha256.clone()))
        .collect()
}

fn hf_metadata_key(index: usize) -> String {
    format!("hf:{index}:__metadata__")
}

fn partial_download_path(target: &Path) -> Result<PathBuf> {
    let file_name = target.file_name().ok_or_else(|| {
        InfraError::Config(format!(
            "download target has no file name: {}",
            target.display()
        ))
    })?;
    let mut partial_name = file_name.to_os_string();
    partial_name.push(".part");
    Ok(target.with_file_name(partial_name))
}

fn replace_download_target(partial: &Path, target: &Path) -> Result<()> {
    let backup = target.with_file_name({
        let mut name = target
            .file_name()
            .ok_or_else(|| {
                InfraError::Config(format!(
                    "download target has no file name: {}",
                    target.display()
                ))
            })?
            .to_os_string();
        name.push(".previous");
        name
    });
    if !target.exists() {
        fs::rename(partial, target)
            .map_err(|error| InfraError::io(Some(target.to_path_buf()), error))?;
        if backup.exists() {
            let _ = fs::remove_file(backup);
        }
        return Ok(());
    }
    if backup.exists() {
        fs::remove_file(&backup).map_err(|error| InfraError::io(Some(backup.clone()), error))?;
    }
    fs::rename(target, &backup)
        .map_err(|error| InfraError::io(Some(target.to_path_buf()), error))?;
    if let Err(error) = fs::rename(partial, target) {
        let restore = fs::rename(&backup, target);
        return match restore {
            Ok(()) => Err(InfraError::io(Some(target.to_path_buf()), error)),
            Err(restore_error) => Err(InfraError::Backend(format!(
                "replace download target {} failed: {error}; restoring {} also failed: {restore_error}",
                target.display(),
                backup.display()
            ))),
        };
    }
    let _ = fs::remove_file(backup);
    Ok(())
}

fn refresh_recorded_status(mut status: DownloadStatus) -> DownloadStatus {
    if matches!(status.state, DownloadState::Downloaded)
        && !status.path.as_ref().is_some_and(|path| path.exists())
    {
        status.state = DownloadState::NotStarted;
        status.message = Some("recorded artifact is missing locally".to_string());
        status.updated_at = Utc::now();
    }
    status
}

fn sql_err(error: rusqlite::Error) -> InfraError {
    InfraError::Backend(format!("sqlite: {error}"))
}

#[cfg(test)]
mod tests;
