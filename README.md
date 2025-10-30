# Open Duck Playground

# Installation 

Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

# Training

If you want to use the [imitation reward](https://la.disneyresearch.com/wp-content/uploads/BD_X_paper.pdf), you can generate reference motion with [this repo](https://github.com/apirrone/Open_Duck_reference_motion_generator)

Then copy `polynomial_coefficients.pkl` in `playground/<robot>/data/`

You'll also have to set `USE_IMITATION_REWARD=True` in it's `joystick.py` file

Run: 

```bash
uv run playground/<robot>/runner.py 
```

## Tensorboard

```bash
uv run tensorboard --logdir=<yourlogdir>
```

# Inference 

Infer mujoco

(for now this is specific to open_duck_mini_v2)

```bash
uv run playground/open_duck_mini_v2/mujoco_infer.py -o <path_to_.onnx>
```

## Rust port

An initial Rust translation of the playground lives in [`rust_port/`](rust_port). The crate implements the polynomial reference motion loader, low-pass action filters, and now includes Rust-native facsimiles of the MuJoCo viewer and training runner utilities so they can be orchestrated from strongly typed code.

```
cargo test --manifest-path rust_port/Cargo.toml
```

Reference motions can be instantiated from JSON definitions that mirror the structure of the original polynomial coefficients.
Each record in the JSON array must provide the velocity triple and the polynomial coefficients expressed in ascending order:

```json
[
  {
    "dx": 0.0,
    "dy": 0.0,
    "dtheta": 0.0,
    "period": 1.0,
    "fps": 60.0,
    "frame_offsets": [0, 30],
    "startend_double_support_ratio": 0.25,
    "coefficients": [[0.0, 0.5, -0.1], [1.2, 0.0]]
  }
]
```

The loader mirrors the selection logic from the Python version by clamping velocities to the available ranges and sampling the polynomials via Horner's method. Additional records can be appended to cover new gait targets. The viewer module reproduces the keyboard and joystick command mapping and generates target joint poses directly from the Rust motion loader. The runner module provides generic hooks for PPO-style trainers by abstracting metric logging, checkpointing and ONNX exports behind traits, allowing consumers to plug in custom backends while reusing the original workflow structure.

The viewer integrates with [`mujoco-rs`](https://crates.io/crates/mujoco-rs) behind an optional `mujoco` Cargo feature. Enabling the feature allows the passive viewer to stream reference poses to MuJoCo's Rust-native renderer while leaving the remainder of the API available without linking to the MuJoCo binaries.

# Documentation

## Project structure : 

```
.
├── pyproject.toml
├── README.md
├── playground
│   ├── common
│   │   ├── export_onnx.py
│   │   ├── onnx_infer.py
│   │   ├── poly_reference_motion.py
│   │   ├── randomize.py
│   │   ├── rewards.py
│   │   └── runner.py
│   ├── open_duck_mini_v2
│   │   ├── base.py
│   │   ├── data
│   │   │   └── polynomial_coefficients.pkl
│   │   ├── joystick.py
│   │   ├── mujoco_infer.py
│   │   ├── constants.py
│   │   ├── runner.py
│   │   └── xmls
│   │       ├── assets
│   │       ├── open_duck_mini_v2_no_head.xml
│   │       ├── open_duck_mini_v2.xml
│   │       ├── scene_mjx_flat_terrain.xml
│   │       ├── scene_mjx_rough_terrain.xml
│   │       └── scene.xml
```

## Adding a new robot

Create a new directory in `playground` named after `<your robot>`. You can copy the `open_duck_mini_v2` directory as a starting point.

You will need to:
- Edit `base.py`: Mainly renaming stuff to match you robot's name
- Edit `constants.py`: specify the names of some important geoms, sensors etc
  - In your `mjcf`, you'll probably have to add some sites, name some bodies/geoms and add the sensors. Look at how we did it for `open_duck_mini_v2`
- Add your `mjcf` assets in `xmls`. 
- Edit `joystick.py` : to choose the rewards you are interested in
  - Note: for now there is still some hard coded values etc. We'll improve things on the way
- Edit `runner.py`



# Notes

Inspired from https://github.com/kscalelabs/mujoco_playground


## Current win

```bash
uv run playground/open_duck_mini_v2/runner.py --task flat_terrain_backlash --num_timesteps 300000000
```
