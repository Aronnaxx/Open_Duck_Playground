use std::fmt;

use crate::poly_reference_motion::PolyReferenceMotion;

#[derive(Debug, Clone, Copy)]
pub struct CommandRanges {
    pub x: (f64, f64),
    pub y: (f64, f64),
    pub theta: (f64, f64),
}

impl Default for CommandRanges {
    fn default() -> Self {
        Self {
            x: (-0.15, 0.15),
            y: (-0.2, 0.2),
            theta: (-1.0, 1.0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionCommand {
    pub dx: f64,
    pub dy: f64,
    pub dtheta: f64,
}

impl MotionCommand {
    pub fn zero() -> Self {
        Self {
            dx: 0.0,
            dy: 0.0,
            dtheta: 0.0,
        }
    }

    pub fn is_zero(&self) -> bool {
        approx_zero(self.dx) && approx_zero(self.dy) && approx_zero(self.dtheta)
    }

    pub fn clamped(self, ranges: &CommandRanges) -> Self {
        Self {
            dx: self.dx.clamp(ranges.x.0, ranges.x.1),
            dy: self.dy.clamp(ranges.y.0, ranges.y.1),
            dtheta: self.dtheta.clamp(ranges.theta.0, ranges.theta.1),
        }
    }

    pub fn update_from_axes(
        &mut self,
        ranges: &CommandRanges,
        axis_x: f64,
        axis_y: f64,
        axis_theta: f64,
    ) {
        let forward = if axis_y < 0.0 {
            (-axis_y) * ranges.x.1
        } else {
            -axis_y * ranges.x.0.abs()
        };
        let lateral = -axis_x * ranges.y.1;
        let yaw = -axis_theta * ranges.theta.1;

        *self = Self {
            dx: forward,
            dy: lateral,
            dtheta: yaw,
        };
    }

    pub fn update_from_key(&mut self, ranges: &CommandRanges, key: KeyCommand) {
        match key {
            KeyCommand::Forward => {
                self.dx = ranges.x.1;
                self.dy = 0.0;
                self.dtheta = 0.0;
            }
            KeyCommand::Backward => {
                self.dx = ranges.x.0;
                self.dy = 0.0;
                self.dtheta = 0.0;
            }
            KeyCommand::Left => {
                self.dx = 0.0;
                self.dy = ranges.y.1;
                self.dtheta = 0.0;
            }
            KeyCommand::Right => {
                self.dx = 0.0;
                self.dy = ranges.y.0;
                self.dtheta = 0.0;
            }
            KeyCommand::RotateLeft => {
                self.dx = 0.0;
                self.dy = 0.0;
                self.dtheta = ranges.theta.1;
            }
            KeyCommand::RotateRight => {
                self.dx = 0.0;
                self.dy = 0.0;
                self.dtheta = ranges.theta.0;
            }
            KeyCommand::Idle => {
                *self = Self::zero();
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum KeyCommand {
    Forward,
    Backward,
    Left,
    Right,
    RotateLeft,
    RotateRight,
    Idle,
}

#[derive(Debug, thiserror::Error)]
pub enum ViewerError {
    #[error("reference motion expected 40 entries, received {0}")]
    UnexpectedReferenceLength(usize),

    #[error("actuated joint dimension mismatch: expected {expected}, found {found}")]
    JointDimensionMismatch { expected: usize, found: usize },

    #[cfg(feature = "mujoco")]
    #[error("mujoco error: {0}")]
    Mujoco(String),
}

pub struct ReferenceMotionViewer {
    reference_motion: PolyReferenceMotion,
    command_ranges: CommandRanges,
    default_qpos: Vec<f64>,
    decimation: usize,
    counter: usize,
    step: usize,
    current_pose: Vec<f64>,
    command: MotionCommand,
}

impl fmt::Debug for ReferenceMotionViewer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReferenceMotionViewer")
            .field("command_ranges", &self.command_ranges)
            .field("decimation", &self.decimation)
            .field("counter", &self.counter)
            .field("step", &self.step)
            .field("command", &self.command)
            .finish()
    }
}

impl ReferenceMotionViewer {
    pub fn new(reference_motion: PolyReferenceMotion, default_qpos: Vec<f64>) -> Self {
        let command = MotionCommand::zero();
        let current_pose = default_qpos.clone();
        Self {
            reference_motion,
            command_ranges: CommandRanges::default(),
            default_qpos,
            decimation: 10,
            counter: 0,
            step: 0,
            current_pose,
            command,
        }
    }

    pub fn with_ranges(mut self, command_ranges: CommandRanges) -> Self {
        self.command_ranges = command_ranges;
        self
    }

    pub fn with_decimation(mut self, decimation: usize) -> Self {
        self.decimation = decimation.max(1);
        self
    }

    pub fn command(&self) -> MotionCommand {
        self.command
    }

    pub fn set_command(&mut self, command: MotionCommand) {
        self.command = command.clamped(&self.command_ranges);
    }

    pub fn update_from_key(&mut self, key: KeyCommand) {
        let mut command = self.command;
        command.update_from_key(&self.command_ranges, key);
        self.command = command;
    }

    pub fn update_from_axes(&mut self, axis_x: f64, axis_y: f64, axis_theta: f64) {
        let mut command = self.command;
        command.update_from_axes(&self.command_ranges, axis_x, axis_y, axis_theta);
        self.command = command.clamped(&self.command_ranges);
    }

    pub fn reset_pose(&mut self) {
        self.counter = 0;
        self.step = 0;
        self.current_pose = self.default_qpos.clone();
    }

    pub fn advance(&mut self) -> Result<&[f64], ViewerError> {
        self.counter = self.counter.saturating_add(1);
        if self.counter % self.decimation == 0 {
            let mut new_qpos = self.default_qpos.clone();
            if !self.command.is_zero() {
                let imitation_index = self.step % self.reference_motion.nb_steps_in_period;
                let ref_motion = self.reference_motion.get_reference_motion(
                    self.command.dx,
                    self.command.dy,
                    self.command.dtheta,
                    imitation_index,
                );

                if ref_motion.len() != 40 {
                    return Err(ViewerError::UnexpectedReferenceLength(ref_motion.len()));
                }

                let joints_pos = &ref_motion[0..16];
                let mut ref_joint_pos = Vec::with_capacity(14);
                ref_joint_pos.extend_from_slice(&joints_pos[0..9]);
                ref_joint_pos.extend_from_slice(&joints_pos[11..16]);

                let target_slice = &mut new_qpos[7..(7 + ref_joint_pos.len())];
                if target_slice.len() != ref_joint_pos.len() {
                    return Err(ViewerError::JointDimensionMismatch {
                        expected: target_slice.len(),
                        found: ref_joint_pos.len(),
                    });
                }

                target_slice.copy_from_slice(&ref_joint_pos);
                self.step = self.step.wrapping_add(1);
            } else {
                self.step = 0;
            }

            self.current_pose = new_qpos;
        }

        Ok(&self.current_pose)
    }
}

#[cfg(feature = "mujoco")]
pub mod mujoco_support {
    use std::time::{Duration, Instant};

    use mujoco_rs::{MjModel, MjModelBuilder, MjStep, Visualize};

    use super::{KeyCommand, MotionCommand, ReferenceMotionViewer, ViewerError};

    pub struct PassiveViewer {
        model: MjModel,
        data: mujoco_rs::MjData,
        viewer: mujoco_rs::viewer::Viewer,
    }

    impl PassiveViewer {
        pub fn new(model_xml: &str) -> Result<Self, ViewerError> {
            let model = MjModelBuilder::from_xml(model_xml)
                .map_err(|err| ViewerError::UnexpectedReferenceLength(err.to_string().len()))?;
            let data = model.make_data();
            let viewer = mujoco_rs::viewer::Viewer::new(&model, &data)
                .map_err(|err| ViewerError::UnexpectedReferenceLength(err.to_string().len()))?;
            Ok(Self {
                model,
                data,
                viewer,
            })
        }

        pub fn step(&mut self) {
            self.model.step(&mut self.data);
            self.viewer.sync();
        }
    }

    pub fn launch_reference_viewer(
        model_xml: &str,
        viewer: &mut ReferenceMotionViewer,
        duration: Duration,
    ) -> Result<(), ViewerError> {
        let mut passive_viewer = PassiveViewer::new(model_xml)?;
        let end = Instant::now() + duration;

        while Instant::now() < end {
            viewer.advance()?;
            passive_viewer.step();
        }

        Ok(())
    }

    pub fn apply_keyboard(viewer: &mut ReferenceMotionViewer, keycode: u32) {
        let command = match keycode {
            265 => KeyCommand::Forward,
            264 => KeyCommand::Backward,
            263 => KeyCommand::Left,
            262 => KeyCommand::Right,
            81 => KeyCommand::RotateLeft,
            69 => KeyCommand::RotateRight,
            _ => KeyCommand::Idle,
        };
        viewer.update_from_key(command);
    }

    pub fn apply_joystick(
        viewer: &mut ReferenceMotionViewer,
        primary: (f64, f64),
        secondary: Option<f64>,
    ) {
        let theta = secondary.unwrap_or_default();
        viewer.update_from_axes(primary.0, primary.1, theta);
    }
}

fn approx_zero(value: f64) -> bool {
    value.abs() < 1e-12
}

#[cfg(test)]
mod tests {
    use super::{CommandRanges, MotionCommand, ReferenceMotionViewer};
    use crate::poly_reference_motion::{MotionRecord, PolyReferenceMotion};

    fn sample_motion() -> PolyReferenceMotion {
        let mut coefficients = vec![vec![0.0; 5]; 40];
        for (idx, coeffs) in coefficients.iter_mut().enumerate() {
            coeffs[0] = idx as f64;
        }
        let record = MotionRecord::new(0.1, 0.0, 0.0, 1.0, 10.0, vec![0.0, 5.0], 0.1, coefficients);
        PolyReferenceMotion::from_records(vec![record]).unwrap()
    }

    #[test]
    fn advances_pose_when_command_active() {
        let motion = sample_motion();
        let default_qpos = vec![0.0; 30];
        let mut viewer = ReferenceMotionViewer::new(motion.clone(), default_qpos)
            .with_decimation(1)
            .with_ranges(CommandRanges {
                x: (0.0, 0.5),
                y: (0.0, 0.5),
                theta: (0.0, 0.5),
            });
        viewer.set_command(MotionCommand {
            dx: 0.1,
            dy: 0.0,
            dtheta: 0.0,
        });

        let pose = viewer.advance().unwrap();
        let reference = motion.get_reference_motion(0.1, 0.0, 0.0, 0);
        let joints_pos = &reference[0..16];
        let mut ref_joint_pos = Vec::with_capacity(14);
        ref_joint_pos.extend_from_slice(&joints_pos[0..9]);
        ref_joint_pos.extend_from_slice(&joints_pos[11..16]);
        assert_eq!(&pose[7..21], ref_joint_pos.as_slice());
        assert_eq!(viewer.command().dx, 0.1);

        let pose2 = viewer.advance().unwrap();
        assert_eq!(&pose2[7..21], ref_joint_pos.as_slice());
    }

    #[test]
    fn resets_when_command_zero() {
        let motion = sample_motion();
        let default_qpos = vec![1.0; 30];
        let mut viewer =
            ReferenceMotionViewer::new(motion, default_qpos.clone()).with_decimation(1);
        viewer.set_command(MotionCommand {
            dx: 0.0,
            dy: 0.0,
            dtheta: 0.0,
        });
        let pose = viewer.advance().unwrap();
        assert_eq!(pose, &default_qpos[..]);
    }
}
