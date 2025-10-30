use std::collections::HashMap;
use std::fs;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use chrono::Local;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RunnerError {
    #[error("failed to prepare output directory: {0}")]
    Io(#[from] std::io::Error),

    #[error("checkpoint error: {0}")]
    Checkpoint(String),

    #[error("onnx export error: {0}")]
    Export(String),
}

pub trait MetricsSink {
    fn record(&mut self, step: usize, name: &str, value: f64);
}

pub trait CheckpointSink<P> {
    fn save(&mut self, path: &Path, params: &P) -> Result<(), String>;
}

pub trait OnnxExporter<P> {
    fn export(
        &self,
        params: &P,
        action_size: usize,
        obs_size: usize,
        path: &Path,
    ) -> Result<(), String>;
}

#[derive(Debug, Clone)]
pub struct RunnerConfig {
    pub output_dir: PathBuf,
    pub num_timesteps: usize,
    pub action_size: usize,
    pub obs_size: usize,
}

impl RunnerConfig {
    pub fn new(
        output_dir: impl Into<PathBuf>,
        num_timesteps: usize,
        action_size: usize,
        obs_size: usize,
    ) -> Self {
        Self {
            output_dir: output_dir.into(),
            num_timesteps,
            action_size,
            obs_size,
        }
    }
}

pub struct Runner<P, M, C, E>
where
    M: MetricsSink,
    C: CheckpointSink<P>,
    E: OnnxExporter<P>,
{
    config: RunnerConfig,
    metrics_sink: M,
    checkpoint_sink: C,
    exporter: E,
    restore_checkpoint_path: Option<PathBuf>,
    _marker: PhantomData<P>,
}

impl<P, M, C, E> Runner<P, M, C, E>
where
    M: MetricsSink,
    C: CheckpointSink<P>,
    E: OnnxExporter<P>,
{
    pub fn new(
        mut config: RunnerConfig,
        metrics_sink: M,
        checkpoint_sink: C,
        exporter: E,
    ) -> Result<Self, RunnerError> {
        if config.output_dir.is_relative() {
            config.output_dir = std::env::current_dir()?.join(&config.output_dir);
        }
        fs::create_dir_all(&config.output_dir)?;
        prepare_cache()?;

        Ok(Self {
            config,
            metrics_sink,
            checkpoint_sink,
            exporter,
            restore_checkpoint_path: None,
            _marker: PhantomData,
        })
    }

    pub fn config(&self) -> &RunnerConfig {
        &self.config
    }

    pub fn set_restore_checkpoint_path(&mut self, path: impl Into<PathBuf>) {
        self.restore_checkpoint_path = Some(path.into());
    }

    pub fn restore_checkpoint_path(&self) -> Option<&Path> {
        self.restore_checkpoint_path.as_deref()
    }

    pub fn progress_callback(&mut self, num_steps: usize, metrics: &HashMap<String, f64>) {
        for (name, value) in metrics {
            self.metrics_sink.record(num_steps, name, *value);
        }

        if let (Some(reward), Some(std_dev)) = (
            metrics.get("eval/episode_reward"),
            metrics.get("eval/episode_reward_std"),
        ) {
            println!(
                "-----------\nSTEP: {num_steps} reward: {reward} reward_std: {std_dev}\n-----------"
            );
        }
    }

    pub fn policy_params_callback(
        &mut self,
        current_step: usize,
        params: &P,
    ) -> Result<(), RunnerError> {
        let timestamp = Local::now().format("%Y_%m_%d_%H%M%S").to_string();
        let checkpoint_dir = self
            .config
            .output_dir
            .join(format!("{timestamp}_{current_step}"));
        fs::create_dir_all(&checkpoint_dir)?;
        self.checkpoint_sink
            .save(&checkpoint_dir, params)
            .map_err(RunnerError::Checkpoint)?;

        let onnx_path = self
            .config
            .output_dir
            .join(format!("{timestamp}_{current_step}.onnx"));
        self.exporter
            .export(
                params,
                self.config.action_size,
                self.config.obs_size,
                &onnx_path,
            )
            .map_err(RunnerError::Export)?;

        println!(
            "Saved checkpoint (step: {current_step}) to {}",
            checkpoint_dir.display()
        );
        println!("Exported ONNX policy to {}", onnx_path.display());
        Ok(())
    }
}

fn prepare_cache() -> Result<(), RunnerError> {
    fs::create_dir_all(".tmp/jax_cache")?;
    // SAFETY: Environment variables are process-wide but setting them mirrors the
    // Python runner behaviour. The strings are valid UTF-8 and the cache
    // directory exists, so the calls are safe.
    unsafe {
        std::env::set_var("JAX_COMPILATION_CACHE_DIR", ".tmp/jax_cache");
        std::env::set_var("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "-1");
        std::env::set_var("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{CheckpointSink, MetricsSink, OnnxExporter, Runner, RunnerConfig, RunnerError};
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};

    #[derive(Default)]
    struct TestMetricsSink {
        entries: RefCell<Vec<(usize, String, f64)>>,
    }

    impl MetricsSink for TestMetricsSink {
        fn record(&mut self, step: usize, name: &str, value: f64) {
            self.entries
                .borrow_mut()
                .push((step, name.to_string(), value));
        }
    }

    #[derive(Default)]
    struct TestCheckpointSink {
        saved: RefCell<Vec<PathBuf>>,
    }

    impl CheckpointSink<String> for TestCheckpointSink {
        fn save(&mut self, path: &Path, params: &String) -> Result<(), String> {
            let mut entry = path.to_path_buf();
            entry.push(params);
            self.saved.borrow_mut().push(entry);
            Ok(())
        }
    }

    struct TestExporter {
        exported: RefCell<Vec<PathBuf>>,
    }

    impl OnnxExporter<String> for TestExporter {
        fn export(
            &self,
            _params: &String,
            _action_size: usize,
            _obs_size: usize,
            path: &Path,
        ) -> Result<(), String> {
            self.exported.borrow_mut().push(path.to_path_buf());
            Ok(())
        }
    }

    #[test]
    fn records_metrics_and_exports() -> Result<(), RunnerError> {
        let config = RunnerConfig::new("./test_output", 128, 12, 24);
        let metrics = TestMetricsSink::default();
        let checkpoint = TestCheckpointSink::default();
        let exporter = TestExporter {
            exported: RefCell::new(Vec::new()),
        };
        let mut runner = Runner::new(config, metrics, checkpoint, exporter)?;
        let mut metrics_map = HashMap::new();
        metrics_map.insert("eval/episode_reward".to_string(), 42.0);
        metrics_map.insert("eval/episode_reward_std".to_string(), 4.2);
        runner.progress_callback(10, &metrics_map);
        runner.policy_params_callback(10, &"params".to_string())?;
        Ok(())
    }
}
