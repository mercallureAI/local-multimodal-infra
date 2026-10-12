//! Task files for a worker that may not share the controller's file system.
//!
//! The controller names input files by its own paths and reads output files
//! by the worker's. On the same machine or volume (the worker finds the
//! controller's instance file at the path the controller gave) nothing
//! moves; otherwise the inputs are fetched from the controller's
//! `GET /internal/files` before the inference and the outputs sent to its
//! `POST /internal/files` after, the paths in the task and the output
//! rewritten to match.

use crate::{WorkerState, WORKER_TOKEN_HEADER};
use local_core::{
    InferenceOutput, InferenceTask, WorkerRegistrationResponse, CONTROLLER_INSTANCE_FILE,
};
use local_error::{InfraError, Result};
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

/// The controller this worker registered with.
#[derive(Debug, Clone)]
pub(crate) struct ControllerLink {
    pub url: String,
    /// Whether the controller's files are readable here by their paths.
    pub shared_files: bool,
}

impl ControllerLink {
    pub fn new(url: &str, registration: &WorkerRegistrationResponse) -> Self {
        let shared_files = match (
            &registration.controller_data_dir,
            &registration.controller_instance,
        ) {
            (Some(dir), Some(instance)) => {
                std::fs::read_to_string(dir.join(CONTROLLER_INSTANCE_FILE))
                    .is_ok_and(|found| found.trim() == instance)
            }
            // A controller that does not say: files shared, as before.
            _ => true,
        };
        Self {
            url: url.trim_end_matches('/').to_string(),
            shared_files,
        }
    }
}

/// Input files fetched for one task, removed on drop.
#[derive(Debug, Default)]
pub(crate) struct StagedInputs(Option<PathBuf>);

impl Drop for StagedInputs {
    fn drop(&mut self) {
        if let Some(dir) = &self.0 {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

impl WorkerState {
    async fn remote_controller(&self) -> Option<(ControllerLink, String)> {
        let link = self.controller.read().await.clone()?;
        if link.shared_files {
            return None;
        }
        Some((link, self.session_token().await?))
    }

    /// Fetches the task's input files when the controller's are not shared,
    /// pointing the task at the copies.
    pub(crate) async fn stage_inputs(&self, task: &mut InferenceTask) -> Result<StagedInputs> {
        let Some((link, token)) = self.remote_controller().await else {
            return Ok(StagedInputs::default());
        };
        let dir = self
            .data_dir
            .join("tmp")
            .join("inputs")
            .join(task.id.as_simple().to_string());
        let mut staged = StagedInputs::default();
        for (index, file) in task.input.files_mut().into_iter().enumerate() {
            let Some(path) = file.path.clone() else {
                continue;
            };
            let name = path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| "input.bin".to_string());
            let target = dir.join(format!("{index}-{name}"));
            if staged.0.is_none() {
                tokio::fs::create_dir_all(&dir)
                    .await
                    .map_err(|e| InfraError::io(Some(dir.clone()), e))?;
                staged.0 = Some(dir.clone());
            }
            if self.fetch(&link, &token, &path, &target).await? {
                file.path = Some(target);
            } else if !path.is_file() {
                // Neither the controller's nor this machine's.
                return Err(InfraError::BadRequest(format!(
                    "input file {} is not served by the controller nor present on worker {}",
                    path.display(),
                    self.node_id
                )));
            }
        }
        Ok(staged)
    }

    /// Downloads the controller's `path` to `target`; `false` when the
    /// controller does not serve it (outside its data directory, missing).
    async fn fetch(
        &self,
        link: &ControllerLink,
        token: &str,
        path: &Path,
        target: &Path,
    ) -> Result<bool> {
        let mut response = self
            .http
            .get(format!("{}/internal/files", link.url))
            .header(WORKER_TOKEN_HEADER, token)
            .query(&[("path", path.to_string_lossy())])
            .send()
            .await
            .map_err(|e| {
                InfraError::Backend(format!("fetch {} from the controller: {e}", path.display()))
            })?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(false);
        }
        if !response.status().is_success() {
            return Err(InfraError::Backend(format!(
                "controller refused {}: {}",
                path.display(),
                response.status()
            )));
        }
        let mut file = tokio::fs::File::create(target)
            .await
            .map_err(|e| InfraError::io(Some(target.to_path_buf()), e))?;
        while let Some(chunk) = response.chunk().await.map_err(|e| {
            InfraError::Backend(format!("fetch {} from the controller: {e}", path.display()))
        })? {
            file.write_all(&chunk)
                .await
                .map_err(|e| InfraError::io(Some(target.to_path_buf()), e))?;
        }
        file.flush()
            .await
            .map_err(|e| InfraError::io(Some(target.to_path_buf()), e))?;
        Ok(true)
    }

    /// Sends the output's files to the controller when its files are not
    /// shared, pointing the output at the controller's copies.
    pub(crate) async fn publish_outputs(&self, output: &mut InferenceOutput) -> Result<()> {
        let Some((link, token)) = self.remote_controller().await else {
            return Ok(());
        };
        for file in output.files_mut() {
            let Some(path) = file.path.clone() else {
                continue;
            };
            let bytes = tokio::fs::read(&path)
                .await
                .map_err(|e| InfraError::io(Some(path.clone()), e))?;
            let name = path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| "output.bin".to_string());
            let response = self
                .http
                .post(format!("{}/internal/files", link.url))
                .header(WORKER_TOKEN_HEADER, &token)
                .query(&[("name", name)])
                .body(bytes)
                .send()
                .await
                .map_err(|e| {
                    InfraError::Backend(format!("send {} to the controller: {e}", path.display()))
                })?;
            if !response.status().is_success() {
                return Err(InfraError::Backend(format!(
                    "controller refused output {}: {}",
                    path.display(),
                    response.status()
                )));
            }
            let stored: serde_json::Value = response.json().await.map_err(|e| {
                InfraError::Backend(format!("decode the controller's output path: {e}"))
            })?;
            let stored = stored["path"].as_str().ok_or_else(|| {
                InfraError::Backend("the controller returned no output path".to_string())
            })?;
            file.path = Some(PathBuf::from(stored));
            let _ = tokio::fs::remove_file(&path).await;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use local_controller::{ControllerOptions, ControllerState};
    use local_core::{FileRef, InferenceInput, TaskKind};
    use local_registry::ModelRegistry;
    use local_runtime::RuntimeManagerConfig;

    #[tokio::test]
    async fn a_worker_without_the_controllers_files_fetches_inputs_and_sends_outputs() {
        let controller_dir = tempfile::tempdir().expect("controller dir");
        // A running controller's data directory exists (its database is there).
        std::fs::create_dir_all(controller_dir.path().join("data")).expect("data dir");
        let worker_dir = tempfile::tempdir().expect("worker dir");
        let controller = ControllerState::new_with_options(
            ModelRegistry::from_models(Vec::new()),
            None,
            ControllerOptions {
                data_dir: controller_dir.path().join("data"),
                asset_cleanup_interval: None,
                ..ControllerOptions::default()
            },
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let url = format!("http://{}", listener.local_addr().expect("addr"));
        tokio::spawn(async move { axum::serve(listener, controller.app()).await });

        let worker = WorkerState::from_registry(
            "remote-worker".to_string(),
            "http://127.0.0.1:0".to_string(),
            None,
            ModelRegistry::from_models(Vec::new()),
            RuntimeManagerConfig::default(),
        )
        .await
        .with_data_dir(worker_dir.path().to_path_buf());
        worker
            .register_with_controller(&url)
            .await
            .expect("register");
        // Same machine: the instance file is found, nothing is copied.
        assert!(
            worker
                .controller
                .read()
                .await
                .as_ref()
                .expect("link")
                .shared_files
        );
        let audio = controller_dir.path().join("data/uploads/task/audio.m4a");
        std::fs::create_dir_all(audio.parent().expect("parent")).expect("mkdir");
        std::fs::write(&audio, b"compressed audio").expect("write input");
        let mut task = InferenceTask::new(
            TaskKind::AsrTranscribe,
            None,
            InferenceInput::AsrTranscribe {
                audio: FileRef::local(&audio),
            },
        );
        let staged = worker.stage_inputs(&mut task).await.expect("stage shared");
        assert!(staged.0.is_none());
        assert_eq!(
            task.input.files_mut()[0].path.as_deref(),
            Some(audio.as_path())
        );

        // Another machine.
        worker
            .controller
            .write()
            .await
            .as_mut()
            .expect("link")
            .shared_files = false;
        let staged = worker.stage_inputs(&mut task).await.expect("stage remote");
        let fetched = task.input.files_mut()[0]
            .path
            .clone()
            .expect("fetched path");
        assert!(
            fetched.starts_with(worker_dir.path()),
            "{}",
            fetched.display()
        );
        assert_eq!(
            std::fs::read(&fetched).expect("fetched"),
            b"compressed audio"
        );
        drop(staged);
        assert!(
            !fetched.exists(),
            "staged inputs are removed after the task"
        );

        // Not the controller's (outside its data directory) but the worker's.
        let local = worker_dir.path().join("own.wav");
        std::fs::write(&local, b"worker file").expect("write local");
        let mut task = InferenceTask::new(
            TaskKind::AsrTranscribe,
            None,
            InferenceInput::AsrTranscribe {
                audio: FileRef::local(&local),
            },
        );
        let _staged = worker.stage_inputs(&mut task).await.expect("stage local");
        assert_eq!(
            task.input.files_mut()[0].path.as_deref(),
            Some(local.as_path())
        );
        // Nowhere.
        let mut task = InferenceTask::new(
            TaskKind::AsrTranscribe,
            None,
            InferenceInput::AsrTranscribe {
                audio: FileRef::local(worker_dir.path().join("missing.wav")),
            },
        );
        assert!(worker.stage_inputs(&mut task).await.is_err());

        // Outputs go to the controller.
        let speech = worker_dir.path().join("speech.wav");
        std::fs::write(&speech, b"synthesized").expect("write output");
        let mut output = InferenceOutput::TtsAudio {
            audio: FileRef::local(&speech),
        };
        worker.publish_outputs(&mut output).await.expect("publish");
        let InferenceOutput::TtsAudio { audio: published } = output else {
            unreachable!()
        };
        let stored = published.path.expect("stored path");
        assert!(
            stored.starts_with(std::path::absolute(controller_dir.path()).expect("absolute")),
            "{}",
            stored.display()
        );
        assert_eq!(std::fs::read(&stored).expect("stored"), b"synthesized");
        assert!(!speech.exists());

        // Only registered workers, only the data directory.
        let client = reqwest::Client::new();
        let anonymous = client
            .get(format!("{url}/internal/files"))
            .query(&[("path", audio.to_string_lossy())])
            .send()
            .await
            .expect("anonymous request");
        assert_eq!(anonymous.status(), reqwest::StatusCode::UNAUTHORIZED);
        let token = worker.session_token().await.expect("token");
        let outside = client
            .get(format!("{url}/internal/files"))
            .header(WORKER_TOKEN_HEADER, &token)
            .query(&[("path", local.to_string_lossy())])
            .send()
            .await
            .expect("outside request");
        assert_eq!(outside.status(), reqwest::StatusCode::NOT_FOUND);
        let escape = controller_dir.path().join("data/../../escape.txt");
        let escaping = client
            .get(format!("{url}/internal/files"))
            .header(WORKER_TOKEN_HEADER, &token)
            .query(&[("path", escape.to_string_lossy())])
            .send()
            .await
            .expect("escaping request");
        assert_eq!(escaping.status(), reqwest::StatusCode::NOT_FOUND);
    }
}
