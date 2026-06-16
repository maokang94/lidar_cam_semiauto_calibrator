# LidarCam Semi-Automatic Calibrator

A semi-automatic extrinsic calibration toolkit for LiDAR-camera systems.

This open-source release provides the offline calibration application only and does not include Fast DDS integration. The application loads images, point clouds, camera intrinsics, initial extrinsics, and output paths from `config/calib_config.yaml`.

## Features

- Load PNG/JPG images and PCD point clouds
- Project LiDAR point clouds onto camera images in real time
- Manually refine the initial extrinsic parameters using keyboard controls
- Manually pair LiDAR 3D lines with image 2D lines in G mode
- Automatically snap selected image points to nearby corners or strong intensity features
- Optimize LiDAR-camera extrinsics using Ceres Solver
- Export both LiDAR-to-camera and camera-to-LiDAR transformations

## Dependencies

The project has been tested on Ubuntu with the following dependencies:

- CMake >= 3.10
- A C++17-compatible compiler
- OpenCV
- PCL
- Eigen3
- Ceres Solver

Example installation command:

```bash
sudo apt install cmake build-essential libopencv-dev libpcl-dev libeigen3-dev libceres-dev
```

## Build

```bash
cd lidar_cam_semiauto_calibrator
cmake -S . -B build
cmake --build build -j4
```

After compilation, the executable is located at:

```bash
build/lidar_cam_semiauto_calibrator
```

## Run With Sample Data

A single-frame sample dataset is included in:

```bash
sample_data/
```

Run the front-camera example:

```bash
./build/lidar_cam_semiauto_calibrator --config config/calib_config.yaml --pair Front
```

Supported `pair` values are:

```text
Front
Fish
Left
Right
Back
```

For example:

```bash
./build/lidar_cam_semiauto_calibrator --config config/calib_config.yaml --pair Left
```

## Use Your Own Data

Update the data paths for the corresponding sensor pair in `config/calib_config.yaml`:

```yaml
sensor_pairs:
  Front:
    data:
      image_dir: "../sample_data/camera_front/data_undistorted"
      pcd_dir: "../sample_data/lidar_main/data"
      result_file: "../result/extrinsic_result.toml"
      static_id: 0
      static_pick_idx: 0
```

Both absolute and relative paths are supported. Relative paths are resolved against the directory containing `calib_config.yaml`.

The recommended filename format is:

```text
static_006_mid_00_xxx.png
static_006_mid_00_xxx.pcd
```

Where:

- `static_006` corresponds to `calib_data.latest_static_id: 6`
- `mid_00` corresponds to `static_pick_idx: 0`

To disable the `static_XXX` naming convention, set `calib_data.latest_static_id` to `0`. The application will then fall back to loading the first image and the first PCD file found in the configured directories.

## Basic Operation

1. Launch the application and specify a sensor pair with `--pair`.
2. Check whether the LiDAR point cloud is projected near the corresponding image structures.
3. Use the keyboard controls to coarsely adjust the extrinsic parameters.
4. Press `G`, or select `Enter G`, to enter manual line-pairing mode.
5. Follow the on-screen instructions and select: LiDAR point 1, image point 1, LiDAR point 2, and image point 2.
6. Select `Accept` to confirm the current line pair.
7. Repeat the process to add multiple line pairs.
8. Select `Optimize` to refine the extrinsic parameters.
9. When the result is satisfactory, select `Save Result`.

For a more detailed workflow, see:

```text
OPERATION_MANUAL.md
```

## Output

Selecting `Save Result` writes the following files:

| File | Description |
| --- | --- |
| `result/extrinsic_result.toml` | LiDAR-to-camera extrinsic parameters |
| `result/extrinsic_result_camera_to_lidar.toml` | Camera-to-LiDAR extrinsic parameters |
| `config/calib_config.yaml` | Updated initial extrinsic parameters |

> **Note:** `Optimize` updates only the current extrinsic parameters shown in the application. No result files are written until `Save Result` is selected.

## Keyboard Shortcuts

| Key | Action |
| --- | --- |
| `J` / `L` | Adjust yaw |
| `I` / `K` | Adjust pitch |
| `U` / `O` | Adjust roll |
| `A` / `D` | Adjust X translation |
| `Q` / `E` | Adjust Y translation |
| `W` / `S` | Adjust Z translation |
| `G` | Enter or leave G mode |
| `Enter` | Run optimization |
| `P` | Save the calibration result |
| `Esc` | Exit the current mode or close the application |

## Project Layout

```text
.
├── CMakeLists.txt
├── README.md
├── OPERATION_MANUAL.md
├── config/
│   └── calib_config.yaml
├── include/
├── src/
├── sample_data/
└── result/
```

## Notes

- This release does not include the Fast DDS data acquisition application.
- Calibration accuracy depends on the initial extrinsic estimate, image quality, point-cloud quality, and the quality of the selected line correspondences.
- For better observability, select multiple clear line features with different orientations and spatial locations.
