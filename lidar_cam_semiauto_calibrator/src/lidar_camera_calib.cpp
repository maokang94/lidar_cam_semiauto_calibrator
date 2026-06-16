#include "include/calib_app_helpers.hpp"

Eigen::Matrix3d inner;
bool use_inverse_extrinsic = false;

int main(int argc, char **argv) {
    CalibAppOptions options;
    if (!parseCalibAppArgs(argc, argv, options)) {
        return -1;
    }

    CalibSensorConfig sensor_cfg;
    if (!loadCalibSensorConfig(options, sensor_cfg)) {
        return -1;
    }
    initGlobalCameraMatrix(sensor_cfg);

    CalibInputFiles input_files;
    if (!selectCalibInputFiles(options, sensor_cfg, input_files)) {
        return -1;
    }
    // 应用版把去地面放到交互会话内执行，便于实时调 RANSAC 阈值并刷新显示。
    Calibration calibra(input_files.image_file, input_files.final_pcd, "");
    applySensorConfigToCalibration(calibra, sensor_cfg);

    return runInteractiveCalibrationSession(options, sensor_cfg, calibra);
}
