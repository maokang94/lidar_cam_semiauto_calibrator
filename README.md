# lidar_cam_semiauto_calibrator
Semi-automatic LiDAR-camera calibration toolkit for multi-camera and multi-LiDAR systems. Supports point cloud clustering, contour extraction, interactive feature annotation, 3D-2D line optimization and real-time calibration visualization.
# Calibration Operation Guide

This guide describes the offline LiDAR-camera extrinsic calibration workflow. It starts directly from calibration setup and does not include Fast DDS data acquisition or playback procedures.

## 1. Calibration Scene and Target Placement

For reliable calibration, place large and clearly visible objects within the overlapping field of view of the LiDAR and camera.

Recommended practices:

- Prefer large targets with clear geometric edges.
- If no fixed structures are available, place three movable targets in each viewing direction and arrange them in a roughly triangular layout.
- Suitable targets include plastic stools, bins, calibration boards, boxes, and other rigid objects with distinct corners.
- Larger targets are easier to observe in both images and point clouds.
- Small targets may produce too few returns on sparse BP LiDARs.
- Existing vehicles, retaining walls, and mounted calibration boards may also be used when they provide clear and repeatable edges.
- Select targets at different image locations and, when possible, at different depths.

The objective is to identify corresponding LiDAR and image features that represent the same physical edges or corners.

## 2. Start the Calibration Application

Run the calibration application with the required sensor pair:

```bash
./build/lidar_cam_semiauto_calibrator \
  --config config/calib_config.yaml \
  --pair Front
```

Supported sensor pairs:

| Pair | Description |
| --- | --- |
| `Front` | Front telephoto camera and main LiDAR |
| `Fish` | Fisheye camera and main LiDAR |
| `Left` | Left camera and left BP LiDAR |
| `Right` | Right camera and right BP LiDAR |
| `Back` | Rear camera and rear BP LiDAR |

After startup, the application displays the camera image, projected LiDAR points and boundaries, live parameters, the current extrinsic estimate, and calibration controls.

![Calibration main window](docs/images/calibration_main_window.png)

## 3. Inspect Point-Cloud Preprocessing

Before selecting calibration features, inspect the ground removal and clustering results.

The parameters in the left panel can be adjusted at runtime:

| Parameter | Description |
| --- | --- |
| `Cluster Tol` | Euclidean clustering distance threshold |
| `Min Points` | Minimum number of points required for a cluster |
| `RANSAC` | Ground-plane removal distance threshold |
| `Canny Low` | Lower Canny threshold for image-edge visualization |
| `Canny High` | Upper Canny threshold for image-edge visualization |
| `Edge Grad` | Minimum image-gradient magnitude |
| `Boundary Close` | Morphological closing size used for boundary construction |

In most cases, only the LiDAR preprocessing parameters need adjustment. The goal is to preserve the target points, suppress the ground, and obtain clear LiDAR boundaries or selectable LiDAR points.

For sparse BP LiDARs, use a larger clustering tolerance and a smaller minimum cluster size than for the main AT128 LiDAR.

## 4. Coarsely Adjust the Extrinsic Parameters

Check whether the projected LiDAR points appear near the corresponding objects in the image.

If the initial projection is significantly displaced, use the keyboard controls:

| Key | Action |
| --- | --- |
| `J` / `L` | Adjust yaw |
| `I` / `K` | Adjust pitch |
| `U` / `O` | Adjust roll |
| `A` / `D` | Adjust X translation in the camera coordinate system |
| `Q` / `E` | Adjust Y translation in the camera coordinate system |
| `W` / `S` | Adjust Z translation in the camera coordinate system |

A precise manual alignment is not required. The LiDAR and image features only need to be close enough for reliable visual correspondence and point selection.

## 5. Enter G Mode and Add 3D-2D Line Correspondences

Press `G` or select **Enter G** to enter manual line-pairing mode.

For each line pair, select four points in the following order:

1. LiDAR point 1
2. Corresponding image point 1
3. LiDAR point 2
4. Corresponding image point 2

The two selected LiDAR points define a 3D line segment. The two selected image points define the corresponding 2D image line segment.

### Point snapping

- A selected LiDAR point is snapped to the nearest valid LiDAR boundary or projected LiDAR point.
- An image point may be snapped to a nearby corner or strong grayscale feature.
- If no suitable image feature is found, the original mouse position is used.

After selecting all four points, the application displays the LiDAR line, the image line, and their endpoint correspondence.

![Manual line-pair selection](docs/images/manual_line_pairing.png)

Review the pair before accepting it:

![Line-pair review](docs/images/line_pair_review.png)

Select **Accept** or press `A` to add the current pair.

### Recommended feature selection

- Select multiple line pairs.
- Prefer clear physical edges and corners.
- Select lines with different orientations.
- Distribute selected features across different image regions.
- Use targets at different depths when possible.
- Avoid selecting all lines from one small image region.
- Ensure each LiDAR line and image line represent the same physical edge.
- Avoid shadow boundaries, moving objects, occlusions, and ambiguous geometry.

### Pair-management controls

| Control | Action |
| --- | --- |
| `Accept` or `A` | Accept the current pair |
| `Redo` or `R` | Discard and reselect the current pair |
| `Undo Pair` or `U` | Remove the most recently accepted pair |
| `Clear Pairs` or `X` | Remove all accepted pairs |
| `More Lines` | Continue adding line pairs |
| `G` | Enter or leave G mode |

## 6. Optimize the Extrinsic Parameters

After adding enough line correspondences, select **Optimize** or press `Enter`.

The optimizer refines the six-degree-of-freedom LiDAR-to-camera extrinsic transformation:

```text
[yaw, roll, pitch, x, y, z]
```

During optimization, verify that the projected LiDAR features move toward the corresponding image features.

A good result should improve the consistency of multiple selected targets simultaneously. Do not judge the result from only one line pair.

![Optimized projection](docs/images/optimized_projection.png)

If the result is unsatisfactory:

- Check whether every line pair represents the same physical edge.
- Remove ambiguous or incorrectly ordered pairs.
- Add more non-parallel line pairs.
- Add features from different image regions or target depths.
- Improve the coarse initial extrinsic estimate.
- Recheck ground removal and clustering if target LiDAR points are missing.

> **Important:** Optimization updates only the current in-memory extrinsic estimate. It does not write result files automatically.

## 7. Save the Calibration Result

When the optimized projection is satisfactory, select **Save Result** or press `P`.

The application writes:

| File | Description |
| --- | --- |
| `result/extrinsic_result.toml` | LiDAR-to-camera extrinsic parameters |
| `result/extrinsic_result_camera_to_lidar.toml` | Camera-to-LiDAR extrinsic parameters |
| `config/calib_config.yaml` | Updated initial extrinsic parameters for future sessions |

After saving, exit the application and continue with the next camera-LiDAR pair.

## 8. Recommended Calibration Sequence

1. Place large targets with clear edges in the shared LiDAR-camera field of view.
2. Start the application with the correct `--pair`.
3. Inspect ground removal, clustering, and LiDAR projection.
4. Coarsely adjust the initial extrinsic parameters.
5. Enter G mode.
6. Select one 3D LiDAR line and its corresponding 2D image line.
7. Accept the pair.
8. Repeat for multiple non-parallel and spatially distributed line pairs.
9. Run optimization.
10. Inspect the projection on all selected targets.
11. Refine the selected pairs if necessary and optimize again.
12. Save the final result.
13. Verify the generated files under `result/` and the updated YAML configuration.

## 9. Troubleshooting

### The projected point cloud is completely outside the image

Check:

- The selected `--pair`
- Camera intrinsics
- Initial extrinsic parameters
- Image resolution and undistortion settings
- Whether the LiDAR point cloud belongs to the selected camera timestamp

### LiDAR points are difficult to select in G mode

Possible causes:

- The target is too small or too far away.
- The LiDAR returns are too sparse.
- Ground removal incorrectly removes target points.
- The clustering tolerance is too small.
- The minimum cluster size is too large.
- The initial extrinsic estimate is too far from the correct solution.

Use larger targets, adjust preprocessing parameters, or coarsely refine the extrinsic estimate before selecting lines.

### Image-point snapping selects an incorrect location

Adjust:

```yaml
manual_feature_pairing:
  image_corner_snap_enable: 1
  image_corner_snap_radius_px: 12
```

To disable automatic image-point snapping:

```yaml
manual_feature_pairing:
  image_corner_snap_enable: 0
```

### Optimization finishes but no result file is created

Select **Save Result** or press `P`. The **Optimize** action does not save files.
