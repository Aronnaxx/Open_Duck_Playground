use std::iter::zip;

/// Implements a first-order low-pass filter identical to the original Python
/// implementation that was based on JAX arrays. The filter keeps track of the
/// previous action vector and gradually blends it with the most recently pushed
/// action using a configurable cut-off frequency.
#[derive(Debug, Clone)]
pub struct LowPassActionFilter {
    control_freq: f64,
    cutoff_frequency: f64,
    alpha: f64,
    last_action: Vec<f64>,
    current_action: Vec<f64>,
}

impl LowPassActionFilter {
    /// Creates a new filter instance.
    ///
    /// * `control_freq` - The control frequency in Hertz.
    /// * `cutoff_frequency` - The filter cut-off frequency in Hertz (defaults to 30 Hz in the
    ///   original implementation).
    pub fn new(control_freq: f64, cutoff_frequency: f64) -> Self {
        assert!(
            control_freq.is_finite() && control_freq > 0.0,
            "control_freq must be > 0"
        );
        assert!(
            cutoff_frequency.is_finite() && cutoff_frequency > 0.0,
            "cutoff_frequency must be > 0"
        );

        let alpha = Self::compute_alpha(control_freq, cutoff_frequency);

        Self {
            control_freq,
            cutoff_frequency,
            alpha,
            last_action: Vec::new(),
            current_action: Vec::new(),
        }
    }

    fn compute_alpha(control_freq: f64, cutoff_frequency: f64) -> f64 {
        (1.0 / cutoff_frequency) / ((1.0 / control_freq) + (1.0 / cutoff_frequency))
    }

    /// Pushes a new action vector into the filter.
    pub fn push(&mut self, action: &[f64]) {
        if self.last_action.is_empty() {
            self.last_action = action.to_vec();
        }
        self.current_action = action.to_vec();
    }

    /// Returns the filtered action. The call updates the internal state so repeated calls will
    /// keep the exponentially-smoothed value in sync with the Python behaviour.
    pub fn get_filtered_action(&mut self) -> Vec<f64> {
        if self.current_action.is_empty() {
            return self.last_action.clone();
        }

        if self.last_action.len() != self.current_action.len() {
            self.last_action.resize(self.current_action.len(), 0.0);
        }

        for (last, current) in zip(&mut self.last_action, &self.current_action) {
            *last = self.alpha * *last + (1.0 - self.alpha) * *current;
        }

        self.last_action.clone()
    }

    /// Returns the currently configured control frequency.
    pub fn control_frequency(&self) -> f64 {
        self.control_freq
    }

    /// Returns the currently configured cut-off frequency.
    pub fn cutoff_frequency(&self) -> f64 {
        self.cutoff_frequency
    }

    /// Returns the smoothing factor computed from the configuration values.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
}

#[cfg(test)]
mod tests {
    use super::LowPassActionFilter;

    #[test]
    fn filters_values() {
        let mut filter = LowPassActionFilter::new(100.0, 30.0);

        filter.push(&[0.0, 0.0]);
        assert_eq!(filter.get_filtered_action(), vec![0.0, 0.0]);

        filter.push(&[1.0, -1.0]);
        let values = filter.get_filtered_action();
        assert!(values[0] > 0.0 && values[0] < 1.0);
        assert!(values[1] < 0.0 && values[1] > -1.0);
    }
}
