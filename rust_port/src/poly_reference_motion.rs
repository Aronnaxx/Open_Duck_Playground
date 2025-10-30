use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::Path;

use itertools::Itertools;
use ordered_float::OrderedFloat;
use serde::Deserialize;
use thiserror::Error;

type Polynomial = Vec<f64>;
type PolynomialSet = Vec<Polynomial>;
type MotionGrid = Vec<Vec<Vec<PolynomialSet>>>;
type MotionLookup = BTreeMap<
    OrderedFloat<f64>,
    BTreeMap<OrderedFloat<f64>, BTreeMap<OrderedFloat<f64>, PolynomialSet>>,
>;

#[derive(Debug, Error)]
pub enum MotionError {
    #[error("no motion records were provided")]
    EmptyDataset,

    #[error("period mismatch between records")]
    PeriodMismatch,

    #[error("fps mismatch between records")]
    FpsMismatch,

    #[error("frame offsets mismatch between records")]
    FrameOffsetsMismatch,

    #[error("support ratio mismatch between records")]
    SupportRatioMismatch,

    #[error("duplicate entry for velocity triple ({dx}, {dy}, {dtheta})")]
    DuplicateEntry { dx: f64, dy: f64, dtheta: f64 },

    #[error("failed to read JSON definition: {0}")]
    Json(#[from] serde_json::Error),

    #[error("i/o error while loading reference motion: {0}")]
    Io(#[from] std::io::Error),
}

/// Single record describing a polynomial reference motion for a specific velocity triple.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct MotionRecord {
    pub dx: f64,
    pub dy: f64,
    pub dtheta: f64,
    pub period: f64,
    pub fps: f64,
    #[serde(default)]
    pub frame_offsets: Vec<f64>,
    pub startend_double_support_ratio: f64,
    pub coefficients: Vec<Vec<f64>>,
}

impl MotionRecord {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dx: f64,
        dy: f64,
        dtheta: f64,
        period: f64,
        fps: f64,
        frame_offsets: Vec<f64>,
        startend_double_support_ratio: f64,
        coefficients: Vec<Vec<f64>>,
    ) -> Self {
        Self {
            dx,
            dy,
            dtheta,
            period,
            fps,
            frame_offsets,
            startend_double_support_ratio,
            coefficients,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PolyReferenceMotion {
    dx_range: (f64, f64),
    dy_range: (f64, f64),
    dtheta_range: (f64, f64),
    dxs: Vec<f64>,
    dys: Vec<f64>,
    dthetas: Vec<f64>,
    data_array: Vec<Vec<Vec<Vec<Vec<f64>>>>>,
    pub period: f64,
    pub fps: f64,
    pub nb_steps_in_period: usize,
    pub frame_offsets: Vec<usize>,
    pub startend_double_support_ratio: f64,
    pub start_offset: usize,
}

impl PolyReferenceMotion {
    pub fn from_records<I>(records: I) -> Result<Self, MotionError>
    where
        I: IntoIterator<Item = MotionRecord>,
    {
        let mut records_vec: Vec<MotionRecord> = records.into_iter().collect();
        if records_vec.is_empty() {
            return Err(MotionError::EmptyDataset);
        }

        records_vec.sort_by(|a, b| {
            OrderedFloat(a.dx)
                .cmp(&OrderedFloat(b.dx))
                .then_with(|| OrderedFloat(a.dy).cmp(&OrderedFloat(b.dy)))
                .then_with(|| OrderedFloat(a.dtheta).cmp(&OrderedFloat(b.dtheta)))
        });

        let base = &records_vec[0];
        let mut dx_range = (base.dx, base.dx);
        let mut dy_range = (base.dy, base.dy);
        let mut dtheta_range = (base.dtheta, base.dtheta);

        let mut dxs = BTreeSet::new();
        let mut dys = BTreeSet::new();
        let mut dthetas = BTreeSet::new();

        let mut map: MotionLookup = BTreeMap::new();

        let base_frame_offsets = base
            .frame_offsets
            .iter()
            .map(|v| v.round() as usize)
            .collect_vec();
        let base_period = base.period;
        let base_fps = base.fps;
        let base_support_ratio = base.startend_double_support_ratio;

        for record in &records_vec {
            if !approx_eq(record.period, base_period) {
                return Err(MotionError::PeriodMismatch);
            }
            if !approx_eq(record.fps, base_fps) {
                return Err(MotionError::FpsMismatch);
            }
            if !approx_eq(record.startend_double_support_ratio, base_support_ratio) {
                return Err(MotionError::SupportRatioMismatch);
            }
            let record_offsets = record
                .frame_offsets
                .iter()
                .map(|v| v.round() as usize)
                .collect_vec();
            if record_offsets != base_frame_offsets {
                return Err(MotionError::FrameOffsetsMismatch);
            }

            dx_range.0 = dx_range.0.min(record.dx);
            dx_range.1 = dx_range.1.max(record.dx);
            dy_range.0 = dy_range.0.min(record.dy);
            dy_range.1 = dy_range.1.max(record.dy);
            dtheta_range.0 = dtheta_range.0.min(record.dtheta);
            dtheta_range.1 = dtheta_range.1.max(record.dtheta);

            dxs.insert(OrderedFloat(record.dx));
            dys.insert(OrderedFloat(record.dy));
            dthetas.insert(OrderedFloat(record.dtheta));

            let coeffs = record
                .coefficients
                .iter()
                .map(|coeffs| coeffs.iter().rev().cloned().collect_vec())
                .collect_vec();

            let dy_map = map.entry(OrderedFloat(record.dx)).or_default();
            let dtheta_map = dy_map.entry(OrderedFloat(record.dy)).or_default();
            if dtheta_map
                .insert(OrderedFloat(record.dtheta), coeffs)
                .is_some()
            {
                return Err(MotionError::DuplicateEntry {
                    dx: record.dx,
                    dy: record.dy,
                    dtheta: record.dtheta,
                });
            }
        }

        let dxs_vec = dxs.into_iter().map(|v| v.into_inner()).collect_vec();
        let dys_vec = dys.into_iter().map(|v| v.into_inner()).collect_vec();
        let dthetas_vec = dthetas.into_iter().map(|v| v.into_inner()).collect_vec();

        let mut data_array: MotionGrid =
            vec![
                vec![vec![Vec::<Polynomial>::new(); dthetas_vec.len()]; dys_vec.len()];
                dxs_vec.len()
            ];

        for (ix, dx) in dxs_vec.iter().enumerate() {
            for (iy, dy) in dys_vec.iter().enumerate() {
                for (itheta, dtheta) in dthetas_vec.iter().enumerate() {
                    let coeffs = map
                        .get(&OrderedFloat(*dx))
                        .and_then(|dy_map| dy_map.get(&OrderedFloat(*dy)))
                        .and_then(|theta_map| theta_map.get(&OrderedFloat(*dtheta)))
                        .cloned()
                        .unwrap_or_default();
                    data_array[ix][iy][itheta] = coeffs;
                }
            }
        }

        let nb_steps_in_period = (base_period * base_fps).round() as usize;
        let start_offset = (base_support_ratio * base_fps).round() as usize;

        Ok(Self {
            dx_range,
            dy_range,
            dtheta_range,
            dxs: dxs_vec,
            dys: dys_vec,
            dthetas: dthetas_vec,
            data_array,
            period: base_period,
            fps: base_fps,
            nb_steps_in_period,
            frame_offsets: base_frame_offsets,
            startend_double_support_ratio: base_support_ratio,
            start_offset,
        })
    }

    pub fn from_json_reader<R: Read>(reader: R) -> Result<Self, MotionError> {
        let records: Vec<MotionRecord> = serde_json::from_reader(reader)?;
        Self::from_records(records)
    }

    pub fn from_json_path(path: impl AsRef<Path>) -> Result<Self, MotionError> {
        let file = File::open(path)?;
        Self::from_json_reader(file)
    }

    pub fn vel_to_index(&self, dx: f64, dy: f64, dtheta: f64) -> (usize, usize, usize) {
        let dx = clip(dx, self.dx_range.0, self.dx_range.1);
        let dy = clip(dy, self.dy_range.0, self.dy_range.1);
        let dtheta = clip(dtheta, self.dtheta_range.0, self.dtheta_range.1);

        let ix = nearest_index(&self.dxs, dx);
        let iy = nearest_index(&self.dys, dy);
        let itheta = nearest_index(&self.dthetas, dtheta);
        (ix, iy, itheta)
    }

    pub fn get_reference_motion(&self, dx: f64, dy: f64, dtheta: f64, step: usize) -> Vec<f64> {
        let (ix, iy, itheta) = self.vel_to_index(dx, dy, dtheta);
        let t = (step % self.nb_steps_in_period) as f64 / self.nb_steps_in_period as f64;
        let t = t.clamp(0.0, 1.0);
        let coeffs = &self.data_array[ix][iy][itheta];
        coeffs.iter().map(|poly| polyval(poly, t)).collect()
    }

    pub fn sample_polynomial(coeffs: &[f64], t: f64) -> f64 {
        polyval(coeffs, t)
    }
}

fn polyval(coeffs: &[f64], t: f64) -> f64 {
    coeffs.iter().fold(0.0, |acc, coeff| acc * t + coeff)
}

fn clip(value: f64, min: f64, max: f64) -> f64 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

fn nearest_index(values: &[f64], target: f64) -> usize {
    values
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let da = (*a - target).abs();
            let db = (*b - target).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

#[cfg(test)]
mod tests {
    use super::{MotionRecord, PolyReferenceMotion};

    #[test]
    fn builds_and_queries_motion() {
        let records = vec![
            MotionRecord::new(
                0.0,
                0.0,
                0.0,
                1.0,
                10.0,
                vec![0.0, 5.0],
                0.1,
                vec![vec![0.0, 1.0], vec![1.0, 0.0]],
            ),
            MotionRecord::new(
                1.0,
                0.0,
                0.0,
                1.0,
                10.0,
                vec![0.0, 5.0],
                0.1,
                vec![vec![0.0, 2.0], vec![1.0, 1.0]],
            ),
        ];

        let motion = PolyReferenceMotion::from_records(records).unwrap();
        let values = motion.get_reference_motion(0.2, 0.0, 0.0, 3);
        assert_eq!(values.len(), 2);
        assert!(values[0] >= 0.0);
    }
}
