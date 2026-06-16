#ifndef CALIB_UTILS_HPP
#define CALIB_UTILS_HPP

#include <dirent.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <opencv2/core/persistence.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include "common.h"

inline bool isAbsolutePath(const std::string& path) {
    return !path.empty() && path[0] == '/';
}

inline std::string parentDirectory(const std::string& path) {
    const size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return ".";
    if (pos == 0) return "/";
    return path.substr(0, pos);
}

inline std::string resolvePathFromConfig(const std::string& yaml_path,
                                         const std::string& raw_path) {
    if (raw_path.empty() || isAbsolutePath(raw_path)) return raw_path;
    return parentDirectory(yaml_path) + "/" + raw_path;
}

// 从目录获取第一个图片路径
inline std::string getFirstImage(const std::string& dir_path) {
    DIR *dir; struct dirent *ent;
    if ((dir = opendir(dir_path.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string fname = ent->d_name;
            if (fname.length() >= 4 && fname.substr(fname.length()-4) == ".png") {
                closedir(dir);
                return dir_path + "/" + fname;
            }
        }
        closedir(dir);
    }
    return "";
}

// 合并多帧雷达点云到一个文件
inline bool mergeLidarFrames(const std::string& dir_path, int num_frames, const std::string& output_file) {
    DIR *dir; struct dirent *ent;
    std::vector<std::string> pcd_files;
    if ((dir = opendir(dir_path.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string fname = ent->d_name;
            if (fname.length() >= 4 && fname.substr(fname.length()-4) == ".pcd") {
                pcd_files.push_back(dir_path + "/" + fname);
            }
        }
        closedir(dir);
    }
    if (pcd_files.empty()) return false;
    std::sort(pcd_files.begin(), pcd_files.end());
    pcl::PointCloud<pcl::PointXYZI> merged_cloud;
    int frames_to_merge = std::min((int)pcd_files.size(), num_frames);
    for (int i = 0; i < frames_to_merge; ++i) {
        pcl::PointCloud<pcl::PointXYZI> temp_cloud;
        if (pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_files[i], temp_cloud) != -1)
            merged_cloud += temp_cloud;
    }
    merged_cloud.is_dense = false;
    pcl::io::savePCDFileBinary(output_file, merged_cloud);
    return true;
}

// 从 YAML 配置文件读取指定传感器对的所有参数
inline bool loadSensorConfig(const std::string &yaml_path,
                             const std::string &pair_name,
                             std::vector<double> &intrinsic_mat,
                             std::vector<double> &dist_coeffs,
                             std::vector<double> &extrinsic_init,
                             std::string &img_dir,
                             std::string &pcd_dir,
                             std::string &result_file,
                             std::vector<cv::Rect> &camera_rois,
                             std::vector<double> &lidar_roi) {
    cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open config file: " << yaml_path << std::endl;
        return false;
    }
    cv::FileNode sensor_pairs = fs["sensor_pairs"];
    cv::FileNode sensor_node = sensor_pairs[pair_name];
    if (sensor_node.empty()) {
        std::cerr << "Pair '" << pair_name << "' not found in config." << std::endl;
        return false;
    }

    // 内参矩阵 (9 个 double)
    cv::FileNode intrinsic_node = sensor_node["camera"]["intrinsic_matrix"];
    for (size_t i = 0; i < 9; ++i)
        intrinsic_mat.push_back(intrinsic_node[i]);

    // 畸变系数 (取前8个，不足补0)
    cv::FileNode distort_node = sensor_node["camera"]["distortion_coeffs"];
    for (size_t i = 0; i < distort_node.size(); ++i)
        dist_coeffs.push_back(distort_node[i]);
    if (dist_coeffs.size() < 5) dist_coeffs.resize(5, 0.0);

    // 初始外参 (yaw, roll, pitch, x, y, z)
    cv::FileNode ext_node = sensor_node["extrinsic_initial"];
    extrinsic_init.push_back(ext_node["yaw_deg"]);
    extrinsic_init.push_back(ext_node["roll_deg"]);
    extrinsic_init.push_back(ext_node["pitch_deg"]);
    extrinsic_init.push_back(ext_node["x"]);
    extrinsic_init.push_back(ext_node["y"]);
    extrinsic_init.push_back(ext_node["z"]);

    // 数据路径
    img_dir = resolvePathFromConfig(yaml_path, (std::string)sensor_node["data"]["image_dir"]);
    pcd_dir = resolvePathFromConfig(yaml_path, (std::string)sensor_node["data"]["pcd_dir"]);
    result_file = resolvePathFromConfig(yaml_path, (std::string)sensor_node["data"]["result_file"]);

    // ROI 配置：camera ROI 用像素坐标；lidar ROI 用雷达坐标系 xyz 范围。
    // 兼容两种相机 ROI 写法：
    //   roi.camera.rect: [x, y, width, height]
    //   roi.camera.rects: [[x, y, width, height], ...]
    camera_rois.clear();
    lidar_roi.clear();
    cv::FileNode roi_node = sensor_node["cluster_debug"]["roi"];
    if (roi_node.empty()) roi_node = sensor_node["roi"];
    if (!roi_node.empty()) {
        cv::FileNode cam_roi_node = roi_node["camera"];
        if (!cam_roi_node.empty()) {
            cv::FileNode rects_node = cam_roi_node["rects"];
            if (!rects_node.empty()) {
                for (auto it = rects_node.begin(); it != rects_node.end(); ++it) {
                    if ((*it).size() >= 4) {
                        int x = (int)(*it)[0];
                        int y = (int)(*it)[1];
                        int w = (int)(*it)[2];
                        int h = (int)(*it)[3];
                        if (w > 0 && h > 0) camera_rois.emplace_back(x, y, w, h);
                    }
                }
            } else {
                cv::FileNode rect_node = cam_roi_node["rect"];
                if (!rect_node.empty() && rect_node.size() >= 4) {
                    int x = (int)rect_node[0];
                    int y = (int)rect_node[1];
                    int w = (int)rect_node[2];
                    int h = (int)rect_node[3];
                    if (w > 0 && h > 0) camera_rois.emplace_back(x, y, w, h);
                }
            }
        }

        cv::FileNode lidar_roi_node = roi_node["lidar"];
        cv::FileNode xyz_min_node = lidar_roi_node["xyz_min"];
        cv::FileNode xyz_max_node = lidar_roi_node["xyz_max"];
        if (!lidar_roi_node.empty() && !xyz_min_node.empty() && !xyz_max_node.empty() &&
            xyz_min_node.size() >= 3 && xyz_max_node.size() >= 3) {
            lidar_roi.resize(6);
            lidar_roi[0] = (double)xyz_min_node[0];
            lidar_roi[1] = (double)xyz_max_node[0];
            lidar_roi[2] = (double)xyz_min_node[1];
            lidar_roi[3] = (double)xyz_max_node[1];
            lidar_roi[4] = (double)xyz_min_node[2];
            lidar_roi[5] = (double)xyz_max_node[2];
        }
    }

    std::cout << "✅ ROI配置: camera_rois=" << camera_rois.size();
    if (lidar_roi.size() == 6) {
        std::cout << ", lidar_roi=[x:" << lidar_roi[0] << "~" << lidar_roi[1]
                  << ", y:" << lidar_roi[2] << "~" << lidar_roi[3]
                  << ", z:" << lidar_roi[4] << "~" << lidar_roi[5] << "]";
    } else {
        std::cout << ", lidar_roi=未配置";
    }
    std::cout << std::endl;

    fs.release();
    return true;
}

// 地面滤除：使用RANSAC平面拟合，提取非地面点
inline pcl::PointCloud<pcl::PointXYZI>::Ptr removeGroundPlane(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
    double distance_threshold = 0.2)   // 可根据实际地面起伏调整
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>);
    if (cloud->empty()) return cloud_filtered;

    pcl::SampleConsensusModelPlane<pcl::PointXYZI>::Ptr model_plane(
        new pcl::SampleConsensusModelPlane<pcl::PointXYZI>(cloud));
    pcl::RandomSampleConsensus<pcl::PointXYZI> ransac(model_plane);
    ransac.setDistanceThreshold(distance_threshold);
    ransac.computeModel();

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    ransac.getInliers(inliers->indices);

    // 创建非地面索引
    pcl::PointIndices::Ptr outliers(new pcl::PointIndices);
    std::vector<bool> inlier_mask(cloud->size(), false);
    for (int idx : inliers->indices) inlier_mask[idx] = true;
    for (size_t i = 0; i < cloud->size(); ++i)
        if (!inlier_mask[i]) outliers->indices.push_back(i);

    pcl::ExtractIndices<pcl::PointXYZI> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(outliers);
    extract.setNegative(false);
    extract.filter(*cloud_filtered);

    return cloud_filtered;
}
// ==========================================
// 🌟 核心投影函数：点云加畸变投影到图像
// ==========================================
int projectCloudWithDistortion(
    cv::Mat& show_img,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    const Eigen::Matrix3d& R_ext, const Eigen::Vector3d& T_ext,
    double fx, double fy, double cx, double cy,
    double k1, double k2, double p1, double p2)
{
    int pts = 0;
    for (auto p : cloud->points) {
        Eigen::Vector3d pt_raw(p.x, p.y, p.z);

        // 1. 应用外参 (旋转 + 平移)
        Eigen::Vector3d pt_cam = R_ext * pt_raw + T_ext;

        // 2. 深度过滤 (剔除相机背后和极近的点)
        if (pt_cam.z() <= 0.1) continue;

        // 3. 归一化平面坐标
        double x = pt_cam.x() / pt_cam.z();
        double y = pt_cam.y() / pt_cam.z();

        // 4. 计算畸变系数 r^2 和 r^4
        double r2 = x * x + y * y;
        double r4 = r2 * r2;

        // 5. 应用径向畸变 (k1, k2)
        double radial_distortion = 1.0 + k1 * r2 + k2 * r4;

        // 6. 应用切向畸变 (p1, p2) 并计算最终的畸变坐标 (xd, yd)
        double xd = x * radial_distortion + 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
        double yd = y * radial_distortion + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;

        // 7. 应用内参矩阵，转换为像素坐标 (u, v)
        int u = cvRound(fx * xd + cx);
        int v = cvRound(fy * yd + cy);

        // 8. 绘制点云
        if (u >= 0 && u < show_img.cols && v >= 0 && v < show_img.rows) {
            cv::circle(show_img, cv::Point(u, v), 1, cv::Scalar(0, 255, 0), -1);
            pts++;
        }
    }
    return pts; // 返回投影在图像内的有效点数
}

// ==========================================
// 🌟 独立函数：将标定结果更新或追加到 TOML 文件
// ==========================================
// ==========================================
// 🌟 独立函数：将标定结果更新或追加到 TOML 文件
// ==========================================
bool updateOrAppendToml(const std::string& result_file,
                        const std::string& pair_name,
                        const Eigen::Matrix3d& R_ext,
                        const Eigen::Vector3d& T_ext,
                        double cur_yaw, double cur_pitch, double cur_roll) // 新增锚点参数
{
    // =======================================================
    // 1. 反解当前实时使用的旋转矩阵。
    // 约定：R_lidar_to_cam = Ry(yaw) * Rx(pitch) * Rz(roll)
    // yaw 绕相机 Y 轴，pitch 绕相机 X 轴，roll 绕相机 Z 轴。
    // =======================================================
    Eigen::Vector3d final_euler =
        lidarToCameraEulerDegrees(R_ext, cur_yaw, cur_roll, cur_pitch);
    double final_yaw = final_euler[0];
    double final_roll = final_euler[1];
    double final_pitch = final_euler[2];

    double final_x = T_ext[0];
    double final_y = T_ext[1];
    double final_z = T_ext[2];

    // =======================================================
    // 2. 确定 Section 名称和描述
    // =======================================================
    std::string section_name = "";
    std::string desc = "";
    if (pair_name == "Front" || pair_name == "FRONT") {
        section_name = "SURROUD_FRONT_CAMERA";
        desc = "长焦相机外参标定(与主激光雷达)参数(yaw,roll,pitch,x,y,z)";
    } else if (pair_name == "Fish" || pair_name == "FISH") {
        section_name = "SURROUD_FISH_CAMERA";
        desc = "鱼眼相机外参标定(与主激光雷达)参数(yaw,roll,pitch,x,y,z)";
    } else if (pair_name == "Left" || pair_name == "LEFT") {
        section_name = "SURROUD_LEFT_CAMERA";
        desc = "左相机外参标定(与主激光雷达)参数(yaw,roll,pitch,x,y,z)";
    } else if (pair_name == "Right" || pair_name == "RIGHT") {
        section_name = "SURROUD_RIGHT_CAMERA";
        desc = "右相机外参标定(与主激光雷达)参数(yaw,roll,pitch,x,y,z)";
    } else if (pair_name == "Back" || pair_name == "BACK") {
        section_name = "SURROUD_BACK_CAMERA";
        desc = "后相机外参标定(与主激光雷达)参数(yaw,roll,pitch,x,y,z)";
    } else {
        section_name = "UNKNOWN_CAMERA_" + pair_name;
        desc = "未知相机外参标定参数(yaw,roll,pitch,x,y,z)";
    }

    std::string target_section = "[epu.sensor_calibration_config." + section_name + ".extrinsic_params]";

    // 🌟 修复关键点：严格按照 [yaw, roll, pitch, x, y, z] 的顺序装填
    char value_buf[256];
    sprintf(value_buf, "value = [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
            final_yaw, final_roll, final_pitch, final_x, final_y, final_z);
    std::string new_value_str = value_buf;
    std::string desc_str = "value_des = \"" + desc + "\"";

    // =======================================================
    // 3. TOML 文本解析与覆盖/追加
    // =======================================================
    std::vector<std::string> lines;
    std::ifstream infile(result_file);
    bool section_exists = false;

    if (infile.is_open()) {
        std::string line;
        bool in_target_section = false;
        while (std::getline(infile, line)) {
            if (line.find(target_section) != std::string::npos) {
                section_exists = true;
                in_target_section = true;
                lines.push_back(line);
                continue;
            }
            if (in_target_section && !line.empty() && line[0] == '[') {
                in_target_section = false;
            }
            if (in_target_section) {
                if (line.find("value_des") != std::string::npos) {
                    lines.push_back(desc_str);
                } else if (line.find("value") != std::string::npos && line.find("value_des") == std::string::npos) {
                    lines.push_back(new_value_str);
                } else {
                    lines.push_back(line);
                }
            } else {
                lines.push_back(line);
            }
        }
        infile.close();
    }

    std::ofstream outfile(result_file, std::ios::trunc);
    if (!outfile.is_open()) {
        std::cerr << "❌ 无法打开输出文件进行写入: " << result_file << std::endl;
        return false;
    }
    for (const auto& l : lines) outfile << l << "\n";
    if (!section_exists) {
        if (!lines.empty() && !lines.back().empty()) outfile << "\n";
        outfile << target_section << "\n" << desc_str << "\n" << new_value_str << "\n\n";
    }
    outfile.close();
    return true;
}
#endif // CALIB_UTILS_HPP
