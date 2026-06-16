# LidarCam 半自动标定工具

一个面向 LiDAR–相机系统的半自动外参标定工具包。

本开源版本仅提供离线标定程序，不包含 Fast DDS 数据采集与通信模块。程序从 `config/calib_config.yaml` 中读取图像、点云、相机内参、初始外参以及结果保存路径。

## 功能特性

- 支持读取 PNG/JPG 图像与 PCD 点云
- 支持将 LiDAR 点云实时投影到相机图像
- 支持通过键盘对初始外参进行粗调
- 支持在 G 模式下手动配对 LiDAR 三维直线与图像二维直线
- 支持将图像选点自动吸附到附近角点或强灰度特征
- 使用 Ceres Solver 优化 LiDAR–相机外参
- 支持导出 LiDAR 到相机以及相机到 LiDAR 的双向外参

## 依赖环境

项目已在 Ubuntu 环境下使用以下依赖进行测试：

- CMake >= 3.10
- 支持 C++17 的编译器
- OpenCV
- PCL
- Eigen3
- Ceres Solver

示例安装命令：

```bash
sudo apt install cmake build-essential libopencv-dev libpcl-dev libeigen3-dev libceres-dev
```

## 编译

```bash
cd lidar_cam_semiauto_calibrator
cmake -S . -B build
cmake --build build -j4
```

编译完成后，可执行文件位于：

```bash
build/lidar_cam_semiauto_calibrator
```

## 使用示例数据运行

仓库内提供了一帧示例数据，位于：

```bash
sample_data/
```

运行前视相机示例：

```bash
./build/lidar_cam_semiauto_calibrator --config config/calib_config.yaml --pair Front
```

支持的 `pair` 取值包括：

```text
Front
Fish
Left
Right
Back
```

例如：

```bash
./build/lidar_cam_semiauto_calibrator --config config/calib_config.yaml --pair Left
```

## 使用自定义数据

在 `config/calib_config.yaml` 中修改对应传感器组合的数据路径：

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

支持绝对路径和相对路径。相对路径会以 `calib_config.yaml` 所在目录为基准进行解析。

推荐使用以下文件命名格式：

```text
static_006_mid_00_xxx.png
static_006_mid_00_xxx.pcd
```

其中：

- `static_006` 对应 `calib_data.latest_static_id: 6`
- `mid_00` 对应 `static_pick_idx: 0`

如不使用 `static_XXX` 命名规则，可将 `calib_data.latest_static_id` 设置为 `0`。此时程序会退回到读取配置目录中第一张图像和第一帧 PCD 的逻辑。

## 基本操作流程

1. 启动程序，并通过 `--pair` 指定传感器组合。
2. 检查 LiDAR 点云是否大致投影到对应的图像结构附近。
3. 使用键盘对外参进行粗调。
4. 按 `G`，或选择 `Enter G`，进入手动线特征配对模式。
5. 按照屏幕提示依次选择：LiDAR 点 1、图像点 1、LiDAR 点 2、图像点 2。
6. 选择 `Accept`，确认当前线对。
7. 重复上述步骤，添加多组线特征。
8. 选择 `Optimize`，对外参进行优化。
9. 当标定效果满足要求后，选择 `Save Result` 保存结果。

更详细的操作说明请参见：

```text
OPERATION_MANUAL.md
```

## 输出结果

选择 `Save Result` 后，会写入以下文件：

| 文件 | 说明 |
| --- | --- |
| `result/extrinsic_result.toml` | LiDAR 到相机的外参 |
| `result/extrinsic_result_camera_to_lidar.toml` | 相机到 LiDAR 的外参 |
| `config/calib_config.yaml` | 更新后的初始外参 |

> **注意：** `Optimize` 只更新程序界面中的当前外参，不会自动写入结果文件。只有选择 `Save Result` 后，标定结果才会被保存。

## 键盘快捷键

| 按键 | 功能 |
| --- | --- |
| `J` / `L` | 调整 yaw |
| `I` / `K` | 调整 pitch |
| `U` / `O` | 调整 roll |
| `A` / `D` | 调整 X 方向平移 |
| `Q` / `E` | 调整 Y 方向平移 |
| `W` / `S` | 调整 Z 方向平移 |
| `G` | 进入或退出 G 模式 |
| `Enter` | 执行优化 |
| `P` | 保存标定结果 |
| `Esc` | 退出当前模式或关闭程序 |

## 项目结构

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

## 注意事项

- 本开源版本不包含 Fast DDS 数据采集程序。
- 标定精度取决于初始外参、图像质量、点云质量以及人工选择线特征的质量。
- 为提高外参的可观测性，建议选择多组方向不同、位置不同且边缘清晰的线特征。
