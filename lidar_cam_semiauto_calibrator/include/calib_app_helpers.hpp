#ifndef CALIB_APP_HELPERS_HPP
#define CALIB_APP_HELPERS_HPP

// Helper implementations used by src/lidar_camera_calib.cpp.
// Kept header-only because the current target builds a single translation unit.

#include "calib_utils.hpp"
#include "common.h"
#include "lidar_camera_calib.hpp"
#include "ceres/ceres.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core/eigen.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

extern bool use_inverse_extrinsic;
extern Eigen::Matrix3d inner;

static bool g_verbose_log = false;
static bool g_show_optimization_windows = false;

// ============================================================
// 人工红蓝特征配对：用户显式指定 LiDAR 红边与图像蓝边的对应关系
// ============================================================
struct LidarBoundaryCandidate {
    int id = -1;
    int cluster_id = -1;
    bool closed = true;
    std::vector<cv::Point> uv_pts;
    std::vector<Eigen::Vector3d> lidar_pts;
};

// F 模式：吸附到已提取的红/蓝边界，并只选择两次点击之间的局部边界段。
// G 模式：图像侧完全不依赖 Canny/分割结果；人工点击两条直线的端点，
//          LiDAR 两点保留对应 3D 坐标，构成真正的 3D 线 -> 2D 图像线约束。
enum class ManualFeatureMode {
    BOUNDARY_SEGMENT = 0,
    FREE_3D2D_LINE
};

enum class ManualFeaturePickStage {
    LIDAR_START = 0,
    LIDAR_END,
    IMAGE_START,
    IMAGE_END,
    REVIEW
};

struct ManualFeaturePair {
    int id = -1;
    ManualFeatureMode mode = ManualFeatureMode::BOUNDARY_SEGMENT;
    int lidar_boundary_id = -1;
    int image_contour_id = -1;
    // F 模式：局部 LiDAR 边界段及其逐点 3D 关联。
    // G 模式：仅保存人工选择的 3D 直线两个端点。
    std::vector<cv::Point> lidar_uv_initial;
    std::vector<Eigen::Vector3d> lidar_pts;
    // F 模式：两次点击之间的局部蓝色折线段。
    // G 模式：人工点击的两个图像端点，表示一条 2D 直线段。
    std::vector<cv::Point> image_uv;
    double angle_diff_deg = 0.0;
    double length_ratio = 1.0;
};

struct ManualFeaturePickState {
    bool enabled_by_config = true;
    bool free_line_enabled_by_config = true;
    bool active = false;
    bool guide_window_created = false;
    ManualFeatureMode mode = ManualFeatureMode::BOUNDARY_SEGMENT;
    ManualFeaturePickStage stage = ManualFeaturePickStage::LIDAR_START;

    double snap_radius_px = 18.0;
    int min_segment_points = 4;
    double min_segment_length_px = 12.0;
    double image_densify_step_px = 1.0;
    bool image_corner_snap_enabled = true;
    int image_corner_snap_radius_px = 12;

    std::vector<LidarBoundaryCandidate> lidar_candidates;
    std::vector<std::vector<cv::Point>> image_candidates;

    int lidar_candidate_id = -1;
    int lidar_end_candidate_id = -1;
    int lidar_start_idx = -1;
    int lidar_end_idx = -1;
    int image_candidate_id = -1;
    int image_start_idx = -1;
    int image_end_idx = -1;

    std::vector<cv::Point> current_lidar_uv;
    std::vector<Eigen::Vector3d> current_lidar_pts;
    std::vector<cv::Point> current_image_uv;

    ManualFeaturePair pending_pair;
    bool pending_valid = false;
    std::vector<ManualFeaturePair> accepted_pairs;

    std::string message = "Click Enter G Mode, or press G.";
};

static ManualFeaturePickState g_manual_feature;
static cv::Mat g_manual_feature_gray_image;

static void handleManualFeatureClick(const cv::Point& click);
static void resetCurrentManualFeatureSelection(const std::string& reason = std::string());

enum class AppUiCommand {
    NONE = 0,
    ENTER_G,
    ACCEPT_PAIR,
    REDO_PAIR,
    UNDO_PAIR,
    CLEAR_PAIRS,
    CONTINUE_PICK,
    OPTIMIZE_ONLY,
    SAVE_RESULT,
    EXIT_APP
};

enum class AppParamId {
    NONE = 0,
    CLUSTER_TOL,
    CLUSTER_MIN_POINTS,
    RANSAC,
    CANNY_LOW,
    CANNY_HIGH,
    EDGE_GRAD,
    BOUNDARY_CLOSE
};

struct AppButton {
    cv::Rect rect;
    AppUiCommand command = AppUiCommand::NONE;
    std::string label;
    bool active = true;
};

struct AppSlider {
    cv::Rect track;
    AppParamId id = AppParamId::NONE;
    int min_value = 0;
    int max_value = 100;
};

struct AppUiState {
    int image_width = 0;
    int image_height = 0;
    int image_offset_x = 0;
    int image_offset_y = 0;
    int left_panel_width = 260;
    int panel_width = 360;
    AppUiCommand pending_command = AppUiCommand::NONE;
    std::vector<AppButton> buttons;
    std::vector<AppSlider> sliders;
    AppParamId active_slider = AppParamId::NONE;
    bool params_dirty = false;
    bool params_window_ready = false;

    int cluster_tol_x100 = 80;
    int cluster_min_points = 80;
    int ransac_x1000 = 120;
    int canny_low = 100;
    int canny_high = 300;
    int edge_grad = 90;
    int boundary_close = 8;
};

static AppUiState g_app_ui;

static void onAppParamTrackbar(int, void*) {
    g_app_ui.params_dirty = true;
}

static AppUiCommand consumeAppUiCommand() {
    AppUiCommand cmd = g_app_ui.pending_command;
    g_app_ui.pending_command = AppUiCommand::NONE;
    return cmd;
}

// ============================================================
// 鼠标 UV / ROI 交互工具：只用于调试图像 ROI
// ============================================================
struct MouseRoiState {
    cv::Point uv = cv::Point(-1, -1);
    bool uv_valid = false;

    bool enabled = false;
    bool dragging = false;
    bool has_selection = false;
    cv::Point start = cv::Point(-1, -1);
    cv::Point current = cv::Point(-1, -1);
    cv::Rect selected_roi;

    // 本次调试会话中已经框选、但尚未 Shift+S 保存的一组 ROI。
    std::vector<cv::Rect> pending_rois;
};

static MouseRoiState g_mouse_roi;

static cv::Rect normalizedRectFromPoints(const cv::Point& a, const cv::Point& b) {
    int x0 = std::min(a.x, b.x);
    int y0 = std::min(a.y, b.y);
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    return cv::Rect(x0, y0, x1 - x0, y1 - y0);
}

static int* appParamStorage(AppParamId id) {
    switch (id) {
        case AppParamId::CLUSTER_TOL: return &g_app_ui.cluster_tol_x100;
        case AppParamId::CLUSTER_MIN_POINTS: return &g_app_ui.cluster_min_points;
        case AppParamId::RANSAC: return &g_app_ui.ransac_x1000;
        case AppParamId::CANNY_LOW: return &g_app_ui.canny_low;
        case AppParamId::CANNY_HIGH: return &g_app_ui.canny_high;
        case AppParamId::EDGE_GRAD: return &g_app_ui.edge_grad;
        case AppParamId::BOUNDARY_CLOSE: return &g_app_ui.boundary_close;
        default: return nullptr;
    }
}

static const AppSlider* findAppSlider(AppParamId id) {
    for (const auto& s : g_app_ui.sliders) {
        if (s.id == id) return &s;
    }
    return nullptr;
}

static void updateAppSliderFromMouse(const AppSlider& slider, int panel_x) {
    int* value = appParamStorage(slider.id);
    if (!value) return;
    const int x0 = slider.track.x;
    const int x1 = slider.track.x + std::max(1, slider.track.width);
    const double t = std::max(0.0, std::min(1.0,
        static_cast<double>(panel_x - x0) / static_cast<double>(x1 - x0)));
    const int next = static_cast<int>(std::round(
        slider.min_value + t * (slider.max_value - slider.min_value)));
    const int clamped = std::max(slider.min_value, std::min(slider.max_value, next));
    if (*value != clamped) {
        *value = clamped;
        g_app_ui.params_dirty = true;
    }
}

static bool handleAppSliderMouse(int event, int x, int y) {
    if (g_app_ui.left_panel_width <= 0 || x >= g_app_ui.left_panel_width) return false;
    cv::Point panel_pt(x, y);
    if (event == cv::EVENT_LBUTTONDOWN) {
        for (const auto& s : g_app_ui.sliders) {
            cv::Rect hit(s.track.x, s.track.y - 10, s.track.width, s.track.height + 20);
            if (hit.contains(panel_pt)) {
                g_app_ui.active_slider = s.id;
                updateAppSliderFromMouse(s, panel_pt.x);
                return true;
            }
        }
        return true;
    }
    if (event == cv::EVENT_MOUSEMOVE && g_app_ui.active_slider != AppParamId::NONE) {
        if (const AppSlider* s = findAppSlider(g_app_ui.active_slider)) {
            updateAppSliderFromMouse(*s, panel_pt.x);
        }
        return true;
    }
    if (event == cv::EVENT_LBUTTONUP) {
        if (g_app_ui.active_slider != AppParamId::NONE) {
            if (const AppSlider* s = findAppSlider(g_app_ui.active_slider)) {
                updateAppSliderFromMouse(*s, panel_pt.x);
            }
            g_app_ui.active_slider = AppParamId::NONE;
        }
        return true;
    }
    return true;
}

static void onMouseRoiTool(int event, int x, int y, int /*flags*/, void* /*userdata*/) {
    if (handleAppSliderMouse(event, x, y)) return;

    const int image_x0 = g_app_ui.image_offset_x;
    const int image_x1 = image_x0 + g_app_ui.image_width;
    const int image_y0 = g_app_ui.image_offset_y;
    const int image_y1 = image_y0 + g_app_ui.image_height;
    if (event == cv::EVENT_LBUTTONDOWN && g_app_ui.image_width > 0 && x >= image_x1) {
        cv::Point panel_pt(x - image_x1, y);
        for (const auto& b : g_app_ui.buttons) {
            if (b.rect.contains(panel_pt)) {
                if (b.active) g_app_ui.pending_command = b.command;
                return;
            }
        }
        return;
    }
    if (g_app_ui.image_width > 0 &&
        (x < image_x0 || x >= image_x1 || y < image_y0 || y >= image_y1)) {
        return;
    }

    const int img_x = x - image_x0;
    const int img_y = y - image_y0;

    // UV 坐标在 ROI 模式和人工特征模式下都实时更新。
    if (!g_mouse_roi.enabled && !g_manual_feature.active) return;

    g_mouse_roi.uv = cv::Point(img_x, img_y);
    g_mouse_roi.uv_valid = true;

    // 人工特征配对模式优先接管左键；此时不会误触发 ROI 拖拽。
    if (g_manual_feature.active) {
        if (event == cv::EVENT_LBUTTONDOWN) {
            handleManualFeatureClick(cv::Point(img_x, img_y));
        }
        return;
    }

    if (event == cv::EVENT_LBUTTONDOWN) {
        g_mouse_roi.dragging = true;
        g_mouse_roi.has_selection = false;
        g_mouse_roi.start = cv::Point(img_x, img_y);
        g_mouse_roi.current = g_mouse_roi.start;
    } else if (event == cv::EVENT_MOUSEMOVE) {
        if (g_mouse_roi.dragging) {
            g_mouse_roi.current = cv::Point(img_x, img_y);
        }
    } else if (event == cv::EVENT_LBUTTONUP) {
        g_mouse_roi.dragging = false;
        g_mouse_roi.current = cv::Point(img_x, img_y);
        cv::Rect r = normalizedRectFromPoints(g_mouse_roi.start, g_mouse_roi.current);
        if (r.width >= 4 && r.height >= 4) {
            g_mouse_roi.selected_roi = r;
            g_mouse_roi.has_selection = true;
            g_mouse_roi.pending_rois.push_back(r);
            if (g_verbose_log) {
                std::cout << "[MouseROI] added pending ROI #" << g_mouse_roi.pending_rois.size() - 1
                          << ": [" << r.x << ", " << r.y << ", " << r.width << ", " << r.height
                          << "]  pending=" << g_mouse_roi.pending_rois.size() << std::endl;
            }
        }
    }
}

static void setupMouseRoiToolWindow(const std::string& win_name,
                                    bool enable_roi,
                                    bool enable_manual_feature = false) {
    cv::namedWindow(win_name, cv::WINDOW_AUTOSIZE);
    g_mouse_roi.enabled = enable_roi;
    g_mouse_roi.uv_valid = false;
    g_mouse_roi.dragging = false;
    g_mouse_roi.has_selection = false;
    g_mouse_roi.pending_rois.clear();
    if (enable_roi || enable_manual_feature) {
        cv::setMouseCallback(win_name, onMouseRoiTool, nullptr);
    } else {
        cv::setMouseCallback(win_name, nullptr, nullptr);
    }
}

// ============================================================
// 路径 / 静止段文件选择
// ============================================================
static std::string joinPath(const std::string &dir, const std::string &name) {
    if (dir.empty()) return name;
    if (dir.back() == '/') return dir + name;
    return dir + "/" + name;
}

static std::string formatStaticPrefix(int static_id) {
    std::ostringstream oss;
    oss << "static_" << std::setw(3) << std::setfill('0') << static_id << "_";
    return oss.str();
}

static std::string formatPickToken(int pick_idx) {
    std::ostringstream oss;
    oss << "mid_" << std::setw(2) << std::setfill('0') << pick_idx << "_";
    return oss.str();
}

static bool hasSuffix(const std::string &s, const std::string &suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static std::vector<std::string> listStaticFiles(const std::string &dir,
                                                int static_id,
                                                int pick_idx,
                                                const std::vector<std::string> &suffixes) {
    std::vector<std::string> files;
    DIR *dp = opendir(dir.c_str());
    if (!dp) return files;

    const std::string static_token = formatStaticPrefix(static_id);
    const std::string pick_token = formatPickToken(pick_idx);

    struct dirent *entry = nullptr;
    while ((entry = readdir(dp)) != nullptr) {
        std::string name(entry->d_name);
        if (name == "." || name == "..") continue;
        if (name.find(static_token) == std::string::npos) continue;
        if (pick_idx >= 0 && name.find(pick_token) == std::string::npos) continue;

        bool ext_ok = false;
        for (const auto &suffix : suffixes) {
            if (hasSuffix(name, suffix)) { ext_ok = true; break; }
        }
        if (!ext_ok) continue;
        files.push_back(joinPath(dir, name));
    }
    closedir(dp);
    std::sort(files.begin(), files.end());
    return files;
}

static std::string selectStaticImageFile(const std::string &img_dir,
                                         int static_id,
                                         int pick_idx) {
    if (static_id <= 0) {
        std::cout << "[StaticSelect] static_id <= 0，使用旧逻辑 getFirstImage(): "
                  << img_dir << std::endl;
        return getFirstImage(img_dir);
    }

    const std::vector<std::string> img_suffixes = {".png", ".jpg", ".jpeg", ".bmp"};
    auto exact = listStaticFiles(img_dir, static_id, pick_idx, img_suffixes);
    if (!exact.empty()) return exact.front();

    std::cerr << "⚠️ 未找到指定 pick_idx 的静止图像: dir=" << img_dir
              << ", static_id=" << static_id
              << ", pick_idx=" << pick_idx
              << "，尝试使用该静止段任意 mid 帧。" << std::endl;
    auto any_pick = listStaticFiles(img_dir, static_id, -1, img_suffixes);
    if (!any_pick.empty()) return any_pick.front();

    return "";
}

static std::string selectStaticPcdFile(const std::string &pcd_dir,
                                       int static_id,
                                       int pick_idx) {
    if (static_id <= 0) return "";

    const std::vector<std::string> pcd_suffixes = {".pcd"};
    auto exact = listStaticFiles(pcd_dir, static_id, pick_idx, pcd_suffixes);
    if (!exact.empty()) return exact.front();

    std::cerr << "⚠️ 未找到指定 pick_idx 的静止 PCD: dir=" << pcd_dir
              << ", static_id=" << static_id
              << ", pick_idx=" << pick_idx
              << "，尝试使用该静止段任意 mid 帧。" << std::endl;
    auto any_pick = listStaticFiles(pcd_dir, static_id, -1, pcd_suffixes);
    if (!any_pick.empty()) return any_pick.front();

    return "";
}

static std::string canonicalPairName(std::string name) {
    if (name.empty()) return name;
    for (auto &c : name) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    name[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(name[0])));
    return name;
}

static cv::FileNode getPairNode(cv::FileStorage &fs, const std::string &pair_name) {
    cv::FileNode pairs = fs["sensor_pairs"];
    if (pairs.empty()) return cv::FileNode();

    std::vector<std::string> candidates = {
        pair_name,
        canonicalPairName(pair_name),
        std::string(pair_name)
    };

    for (auto &c : candidates) {
        cv::FileNode n = pairs[c];
        if (!n.empty()) return n;
    }
    return cv::FileNode();
}

static void readStaticSelectionConfig(const std::string &config_file,
                                      const std::string &pair_name,
                                      int &static_id,
                                      int &pick_idx) {
    static_id = 0;
    pick_idx = 0;

    cv::FileStorage fs(config_file, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "⚠️ 无法打开 YAML 读取 static_id，继续使用旧逻辑。" << std::endl;
        return;
    }

    cv::FileNode pair = getPairNode(fs, pair_name);
    if (pair.empty()) {
        std::cerr << "⚠️ YAML 中找不到 pair=" << pair_name
                  << " 的 static_id，继续使用旧逻辑。" << std::endl;
        return;
    }

    cv::FileNode data = pair["data"];
    auto readIntIfExists = [](const cv::FileNode &node, const std::vector<std::string> &keys, int &value) {
        for (const auto &k : keys) {
            cv::FileNode v = node[k];
            if (!v.empty()) {
                value = static_cast<int>(v);
                return true;
            }
        }
        return false;
    };

    if (!data.empty()) {
        readIntIfExists(data, {"static_id", "static_index", "static_session", "static_seq"}, static_id);
        readIntIfExists(data, {"static_pick_idx", "static_pick_index", "pick_idx", "mid_idx"}, pick_idx);
    }
    readIntIfExists(pair, {"static_id", "static_index", "static_session", "static_seq"}, static_id);
    readIntIfExists(pair, {"static_pick_idx", "static_pick_index", "pick_idx", "mid_idx"}, pick_idx);

    if (static_id <= 0) {
        cv::FileNode calib_data = fs["calib_data"];
        int latest_static_id = 0;
        int latest_pick_idx = 0;
        if (!calib_data.empty()) {
            readIntIfExists(calib_data, {"latest_static_id", "static_id"}, latest_static_id);
            readIntIfExists(calib_data, {"latest_static_pick_idx", "latest_pick_idx", "static_pick_idx"}, latest_pick_idx);
        }
        if (latest_static_id > 0) {
            static_id = latest_static_id;
            pick_idx = latest_pick_idx;
            std::cout << "[StaticSelect] pair static_id<=0，使用 calib_data.latest_static_id="
                      << static_id << ", latest_static_pick_idx=" << pick_idx << std::endl;
        }
    }

    std::cout << "[StaticSelect] pair=" << pair_name
              << ", static_id=" << static_id
              << ", pick_idx=" << pick_idx << std::endl;
}

static void printCopyToYamlExtrinsic(const std::string &tag,
                                     double yaw_deg,
                                     double roll_deg,
                                     double pitch_deg,
                                     const Eigen::Vector3d &t_ext) {
    if (!g_verbose_log && tag.find("SAVE") == std::string::npos) return;
    std::cout << std::fixed << std::setprecision(6)
              << tag << " extrinsic_initial: ["
              << yaw_deg << ", "
              << roll_deg << ", "
              << pitch_deg << ", "
              << t_ext.x() << ", "
              << t_ext.y() << ", "
              << t_ext.z() << "]"
              << std::endl;
}

// ============================================================
// 聚类调试配置
// ============================================================
struct ClusterDebugConfig {
    bool enable = true;

    bool remove_ground_by_default = true;
    double ground_distance_threshold = 0.12;

    double min_camera_depth = 0.8;
    double max_camera_depth = 120.0;
    double min_lidar_range = 1.0;
    double max_lidar_range = 120.0;

    // LiDAR 原始坐标系下的 XYZ 过滤范围。
    // 之前只有 range 和 z 过滤；这里补上 x/y，默认 x/y 很宽，不影响旧配置。
    // 推荐在 YAML 的 cluster_debug.roi.lidar.xyz_min / xyz_max 里统一配置。
    bool use_lidar_xyz_roi = false;
    double lidar_x_min = -1.0e9;
    double lidar_x_max =  1.0e9;
    double lidar_y_min = -1.0e9;
    double lidar_y_max =  1.0e9;
    double lidar_z_min = -3.0;
    double lidar_z_max = 10.0;

    // 3D 欧式聚类。当前只用于看“去地面后点云分成了哪些类”。
    double cluster_tolerance_m = 0.8;
    int cluster_min_points = 80;
    int cluster_max_points = 1000000;
    int max_display_clusters = 80;

    // 图像 ROI：用于观察/后续限制目标区域。默认只绘制 ROI，不强制裁掉 ROI 外点云。
    bool use_image_rois = false;
    bool restrict_projection_to_rois = false;
    std::vector<cv::Rect> image_rois;

    // 鼠标 ROI 调试工具：只影响主调试窗口，不参与标定模型。
    bool enable_mouse_roi_tool = false;
    bool mouse_roi_replace_existing = true;

    // 可视化
    int draw_cluster_step = 1;
    int point_radius = 1;
    // 点云投影颜色弱化系数，避免绿色/彩色点云遮挡红蓝边界特征。
    // 1.0=原始颜色；0.25~0.45=更适合看边界和 ROI 特征。
    double cluster_point_color_scale = 0.35;
    bool draw_cluster_labels = true;

    // 显示阶段控制：主窗口先只显示细红/细蓝候选边界；
    // 进入优化预览、实时迭代、最终确认时，再把真正匹配/参与优化的红蓝线对加粗。
    bool main_window_highlight_matches = false;
    int matched_boundary_line_thickness = 4;
    // ROI 框选预览阶段：第一层仍显示全图细红/细蓝；框选 ROI 后，ROI 内候选边界加粗，
    // 这样既方便选区域，也能在保存前确认 ROI 内特征是否正确。
    bool roi_preview_thick_candidates = false;
    int roi_preview_boundary_line_thickness = 3;

    // 最终保存确认后，额外显示一次“去地面后整体点云投影”效果图，便于检查全局外参效果。
    bool show_final_full_projection = true;
    int final_projection_point_step = 1;
    int final_projection_point_radius = 1;
    double final_projection_point_color_scale = 0.45;

    // 聚类投影边界：先按 3D cluster 聚类，再从“整个 cluster 的投影点云足迹”提边界。
    // lidar_boundary_mode:
    //   0 = 旧模式：稀疏投影点 mask 直接 findContours，容易把扫描线当边界。
    //   1 = 凸包模式：整个 cluster 投影点做 2D convex hull；整体包络稳定，但会跨过凹区/空区。
    //   2 = 推荐：投影点云足迹 concave/occupancy 外轮廓；贴合点云外边界，不再用凸包跨空区。
    bool draw_cluster_boundaries = true;
    int lidar_boundary_mode = 2;
    int boundary_mask_dilate_px = 3;
    int boundary_fill_radius_px = 2;
    int boundary_close_px = 8;
    int lidar_boundary_sample_step_px = 3;
    bool boundary_keep_largest_only = false;
    double boundary_hull_approx_epsilon_px = 1.5;
    int boundary_line_thickness = 2;
    double boundary_min_area_px = 20.0;
    double boundary_min_arc_px = 20.0;
    int boundary_min_projected_points = 10;

    // 图像边缘调试：只在 LiDAR cluster 投影边界附近搜索图像 Canny/Scharr 边缘。
    // 本版只显示结果，不做匹配优化。
    bool draw_nearby_image_edges = true;
    bool draw_all_canny_edges = false;
    int canny_low = 60;
    int canny_high = 180;
    double edge_gradient_min = 45.0;
    double edge_search_radius_px = 8.0;
    int edge_sample_step = 3;
    int image_edge_point_radius = 2;
    bool draw_edge_match_lines = false;
    bool restrict_image_edges_to_rois = false;

    // 图像 ROI 轮廓特征：不再只在 LiDAR 边界周围逐点找边缘，
    // 而是在图像 ROI 内先独立提取连续 Canny 轮廓/折线，再和 LiDAR 红色轮廓做结构匹配。
    bool use_roi_image_contours = true;
    bool draw_roi_image_contours = true;
    bool restrict_image_contours_to_rois = true;
    int image_contour_min_points = 8;
    double image_contour_min_length_px = 30.0;
    double image_contour_min_area_px = 0.0;
    double image_contour_approx_epsilon_px = 1.5;
    double image_contour_match_radius_px = 25.0;

    // 图像边缘线可视化：把 LiDAR 边界附近搜到的蓝色图像边缘点
    // 按 LiDAR contour 顺序连接成短折线。只用于调试显示，不参与优化。
    bool draw_nearby_image_edge_lines = true;
    bool draw_nearby_image_edge_points = false;
    int image_edge_line_thickness = 2;
    int image_edge_line_min_points = 4;
    double image_edge_line_max_gap_px = 18.0;
    double image_edge_line_min_length_px = 25.0;

    // 红色 LiDAR contour 与蓝色 image edge polyline 的结构匹配调试。
    // 普通模式仍按距离/角度筛选；trusted ROI 模式下，认为人工 ROI 内红蓝轮廓可信，
    // 只做较宽松的几何门限，把更多线对直接送入优化。
    bool draw_line_pair_matches = true;
    bool draw_unmatched_lidar_contours = true;
    int line_pair_min_points = 5;
    double line_pair_min_red_length_px = 25.0;
    double line_pair_min_blue_length_px = 20.0;
    double line_pair_max_mean_dist_px = 18.0;
    double line_pair_max_angle_diff_deg = 35.0;
    double line_pair_max_gap_px = 22.0;
    int line_pair_max_pairs = 30;
    bool draw_line_pair_center_links = true;
    bool draw_line_pair_labels = false;

    // 人工 ROI 高信任模式：用于固定车位/标志物场景。
    // 开启后认为 ROI 内红色 LiDAR 边界和蓝色图像边界均可信，
    // 放宽线对接收条件，并降低“距离大就自动降权”的影响。
    bool trust_roi_line_pairs = false;
    double trusted_line_pair_max_mean_dist_px = 80.0;
    double trusted_line_pair_max_angle_diff_deg = 180.0;
    double trusted_line_pair_weight = 1.0;
    int trusted_min_residual_count = 1;

    // 人工特征配对模式：用户依次点击红色 LiDAR 边界的起点/终点，
    // 再点击对应蓝色图像边界的起点/终点；确认后形成一组可靠约束。
    bool enable_manual_feature_pairing = true;
    double manual_feature_snap_radius_px = 18.0;
    int manual_feature_min_segment_points = 4;
    double manual_feature_min_segment_length_px = 12.0;
    bool manual_image_corner_snap_enabled = true;
    int manual_image_corner_snap_radius_px = 12;
    int manual_feature_residual_sample_step = 1;
    double manual_feature_normal_weight = 1.0;
    double manual_feature_endpoint_weight = 0.45;
    double manual_feature_center_weight = 0.20;
    double manual_feature_huber_loss_px = 5.0;
    int manual_feature_max_iterations = 100;
    double manual_feature_max_delta_angle_deg = 6.0;
    double manual_feature_max_delta_translation_m = 0.35;
    double manual_feature_prior_sigma_angle_deg = 4.0;
    double manual_feature_prior_sigma_translation_m = 0.20;
    double manual_feature_validation_mean_px = 8.0;
    double manual_feature_validation_max_px = 18.0;
    int manual_feature_validation_max_rounds = 3;
    double manual_feature_validation_min_improvement_px = 0.25;
    double manual_feature_near_priority_range_m = 25.0;
    double manual_feature_near_priority_weight = 2.0;

    // G：自由人工 3D LiDAR 直线 -> 2D 图像直线模式。
    // 图像端点使用鼠标原始点击位置，不依赖 ROI、Canny、颜色分割或蓝色候选线。
    bool enable_manual_free_line_pairing = true;
    double manual_feature_image_densify_step_px = 1.0;
    int manual_free_line_sample_count = 16;
    double manual_free_line_distance_weight = 1.0;
    double manual_free_line_direction_weight = 0.30;
    double manual_free_line_endpoint_weight = 0.15;
    double manual_free_line_length_weight = 0.08;

    // 线对优化：只使用已经匹配成功的红/蓝线对做小范围 Ceres 微调。
    bool enable_line_pair_optimization = true;
    int min_line_pairs_for_optimization = 1;
    int max_line_pairs_for_optimization = 60;
    int line_pair_residual_sample_step = 1;
    double line_pair_lidar_assoc_radius_px = 18.0;
    double max_delta_yaw_deg = 6.0;
    double max_delta_roll_deg = 6.0;
    double max_delta_pitch_deg = 6.0;
    double max_delta_translation_m = 0.35;
    double prior_sigma_angle_deg = 5.0;
    double prior_sigma_translation_m = 0.25;
    int optimization_max_iterations = 120;
    double optimization_huber_loss_px = 12.0;
};

static ClusterDebugConfig readClusterDebugConfig(const std::string& config_file,
                                                 const std::string& pair_name) {
    ClusterDebugConfig cfg;
    cv::FileStorage fs(config_file, cv::FileStorage::READ);
    if (!fs.isOpened()) return cfg;

    cv::FileNode pair = getPairNode(fs, pair_name);
    if (pair.empty()) return cfg;

    auto readBool = [](const cv::FileNode& node, const std::string& k, bool& v) {
        cv::FileNode n = node[k];
        if (!n.empty()) v = ((int)n != 0);
    };
    auto readInt = [](const cv::FileNode& node, const std::string& k, int& v) {
        cv::FileNode n = node[k];
        if (!n.empty()) v = (int)n;
    };
    auto readDouble = [](const cv::FileNode& node, const std::string& k, double& v) {
        cv::FileNode n = node[k];
        if (!n.empty()) v = (double)n;
    };

    auto applyScalarConfig = [&](const cv::FileNode& node) {
        if (node.empty()) return;

        readBool(node, "enable", cfg.enable);
        readBool(node, "remove_ground_by_default", cfg.remove_ground_by_default);
        readDouble(node, "ground_distance_threshold", cfg.ground_distance_threshold);

        readDouble(node, "min_camera_depth", cfg.min_camera_depth);
        readDouble(node, "max_camera_depth", cfg.max_camera_depth);
        readDouble(node, "min_lidar_range", cfg.min_lidar_range);
        readDouble(node, "max_lidar_range", cfg.max_lidar_range);
        readBool(node, "use_lidar_xyz_roi", cfg.use_lidar_xyz_roi);
        readDouble(node, "lidar_x_min", cfg.lidar_x_min);
        readDouble(node, "lidar_x_max", cfg.lidar_x_max);
        readDouble(node, "lidar_y_min", cfg.lidar_y_min);
        readDouble(node, "lidar_y_max", cfg.lidar_y_max);
        readDouble(node, "lidar_z_min", cfg.lidar_z_min);
        readDouble(node, "lidar_z_max", cfg.lidar_z_max);

        readDouble(node, "cluster_tolerance_m", cfg.cluster_tolerance_m);
        readInt(node, "cluster_min_points", cfg.cluster_min_points);
        readInt(node, "cluster_max_points", cfg.cluster_max_points);
        readInt(node, "max_display_clusters", cfg.max_display_clusters);

        readInt(node, "draw_cluster_step", cfg.draw_cluster_step);
        readInt(node, "point_radius", cfg.point_radius);
        readDouble(node, "cluster_point_color_scale", cfg.cluster_point_color_scale);
        readBool(node, "draw_cluster_labels", cfg.draw_cluster_labels);
        readBool(node, "main_window_highlight_matches", cfg.main_window_highlight_matches);
        readInt(node, "matched_boundary_line_thickness", cfg.matched_boundary_line_thickness);
        readBool(node, "roi_preview_thick_candidates", cfg.roi_preview_thick_candidates);
        readInt(node, "roi_preview_boundary_line_thickness", cfg.roi_preview_boundary_line_thickness);
        readBool(node, "show_final_full_projection", cfg.show_final_full_projection);
        readInt(node, "final_projection_point_step", cfg.final_projection_point_step);
        readInt(node, "final_projection_point_radius", cfg.final_projection_point_radius);
        readDouble(node, "final_projection_point_color_scale", cfg.final_projection_point_color_scale);

        readBool(node, "use_image_rois", cfg.use_image_rois);
        readBool(node, "restrict_projection_to_rois", cfg.restrict_projection_to_rois);
        readBool(node, "draw_cluster_boundaries", cfg.draw_cluster_boundaries);
        readInt(node, "lidar_boundary_mode", cfg.lidar_boundary_mode);
        readInt(node, "boundary_mask_dilate_px", cfg.boundary_mask_dilate_px);
        readInt(node, "boundary_fill_radius_px", cfg.boundary_fill_radius_px);
        readInt(node, "boundary_close_px", cfg.boundary_close_px);
        readInt(node, "lidar_boundary_sample_step_px", cfg.lidar_boundary_sample_step_px);
        readBool(node, "boundary_keep_largest_only", cfg.boundary_keep_largest_only);
        readDouble(node, "boundary_hull_approx_epsilon_px", cfg.boundary_hull_approx_epsilon_px);
        readInt(node, "boundary_line_thickness", cfg.boundary_line_thickness);
        readDouble(node, "boundary_min_area_px", cfg.boundary_min_area_px);
        readDouble(node, "boundary_min_arc_px", cfg.boundary_min_arc_px);
        readInt(node, "boundary_min_projected_points", cfg.boundary_min_projected_points);

        readBool(node, "draw_nearby_image_edges", cfg.draw_nearby_image_edges);
        readBool(node, "draw_all_canny_edges", cfg.draw_all_canny_edges);
        readInt(node, "canny_low", cfg.canny_low);
        readInt(node, "canny_high", cfg.canny_high);
        readDouble(node, "edge_gradient_min", cfg.edge_gradient_min);
        readDouble(node, "edge_search_radius_px", cfg.edge_search_radius_px);
        readInt(node, "edge_sample_step", cfg.edge_sample_step);
        readInt(node, "image_edge_point_radius", cfg.image_edge_point_radius);
        readBool(node, "draw_edge_match_lines", cfg.draw_edge_match_lines);
        readBool(node, "restrict_image_edges_to_rois", cfg.restrict_image_edges_to_rois);
        readBool(node, "use_roi_image_contours", cfg.use_roi_image_contours);
        readBool(node, "draw_roi_image_contours", cfg.draw_roi_image_contours);
        readBool(node, "restrict_image_contours_to_rois", cfg.restrict_image_contours_to_rois);
        readInt(node, "image_contour_min_points", cfg.image_contour_min_points);
        readDouble(node, "image_contour_min_length_px", cfg.image_contour_min_length_px);
        readDouble(node, "image_contour_min_area_px", cfg.image_contour_min_area_px);
        readDouble(node, "image_contour_approx_epsilon_px", cfg.image_contour_approx_epsilon_px);
        readDouble(node, "image_contour_match_radius_px", cfg.image_contour_match_radius_px);
        readBool(node, "draw_nearby_image_edge_lines", cfg.draw_nearby_image_edge_lines);
        readBool(node, "draw_nearby_image_edge_points", cfg.draw_nearby_image_edge_points);
        readInt(node, "image_edge_line_thickness", cfg.image_edge_line_thickness);
        readInt(node, "image_edge_line_min_points", cfg.image_edge_line_min_points);
        readDouble(node, "image_edge_line_max_gap_px", cfg.image_edge_line_max_gap_px);
        readDouble(node, "image_edge_line_min_length_px", cfg.image_edge_line_min_length_px);

        readBool(node, "draw_line_pair_matches", cfg.draw_line_pair_matches);
        readBool(node, "draw_unmatched_lidar_contours", cfg.draw_unmatched_lidar_contours);
        readInt(node, "line_pair_min_points", cfg.line_pair_min_points);
        readDouble(node, "line_pair_min_red_length_px", cfg.line_pair_min_red_length_px);
        readDouble(node, "line_pair_min_blue_length_px", cfg.line_pair_min_blue_length_px);
        readDouble(node, "line_pair_max_mean_dist_px", cfg.line_pair_max_mean_dist_px);
        readDouble(node, "line_pair_max_angle_diff_deg", cfg.line_pair_max_angle_diff_deg);
        readDouble(node, "line_pair_max_gap_px", cfg.line_pair_max_gap_px);
        readInt(node, "line_pair_max_pairs", cfg.line_pair_max_pairs);
        readBool(node, "draw_line_pair_center_links", cfg.draw_line_pair_center_links);
        readBool(node, "draw_line_pair_labels", cfg.draw_line_pair_labels);
        readBool(node, "trust_roi_line_pairs", cfg.trust_roi_line_pairs);
        readDouble(node, "trusted_line_pair_max_mean_dist_px", cfg.trusted_line_pair_max_mean_dist_px);
        readDouble(node, "trusted_line_pair_max_angle_diff_deg", cfg.trusted_line_pair_max_angle_diff_deg);
        readDouble(node, "trusted_line_pair_weight", cfg.trusted_line_pair_weight);
        readInt(node, "trusted_min_residual_count", cfg.trusted_min_residual_count);

        readBool(node, "enable_manual_feature_pairing", cfg.enable_manual_feature_pairing);
        readDouble(node, "manual_feature_snap_radius_px", cfg.manual_feature_snap_radius_px);
        readInt(node, "manual_feature_min_segment_points", cfg.manual_feature_min_segment_points);
        readDouble(node, "manual_feature_min_segment_length_px", cfg.manual_feature_min_segment_length_px);
        readBool(node, "manual_image_corner_snap_enabled", cfg.manual_image_corner_snap_enabled);
        readInt(node, "manual_image_corner_snap_radius_px", cfg.manual_image_corner_snap_radius_px);
        readInt(node, "manual_feature_residual_sample_step", cfg.manual_feature_residual_sample_step);
        readDouble(node, "manual_feature_normal_weight", cfg.manual_feature_normal_weight);
        readDouble(node, "manual_feature_endpoint_weight", cfg.manual_feature_endpoint_weight);
        readDouble(node, "manual_feature_center_weight", cfg.manual_feature_center_weight);
        readDouble(node, "manual_feature_huber_loss_px", cfg.manual_feature_huber_loss_px);
        readInt(node, "manual_feature_max_iterations", cfg.manual_feature_max_iterations);
        readDouble(node, "manual_feature_max_delta_angle_deg", cfg.manual_feature_max_delta_angle_deg);
        readDouble(node, "manual_feature_max_delta_translation_m", cfg.manual_feature_max_delta_translation_m);
        readDouble(node, "manual_feature_prior_sigma_angle_deg", cfg.manual_feature_prior_sigma_angle_deg);
        readDouble(node, "manual_feature_prior_sigma_translation_m", cfg.manual_feature_prior_sigma_translation_m);
        readDouble(node, "manual_feature_validation_mean_px", cfg.manual_feature_validation_mean_px);
        readDouble(node, "manual_feature_validation_max_px", cfg.manual_feature_validation_max_px);
        readInt(node, "manual_feature_validation_max_rounds", cfg.manual_feature_validation_max_rounds);
        readDouble(node, "manual_feature_validation_min_improvement_px", cfg.manual_feature_validation_min_improvement_px);
        readDouble(node, "manual_feature_near_priority_range_m", cfg.manual_feature_near_priority_range_m);
        readDouble(node, "manual_feature_near_priority_weight", cfg.manual_feature_near_priority_weight);
        readBool(node, "enable_manual_free_line_pairing", cfg.enable_manual_free_line_pairing);
        readDouble(node, "manual_feature_image_densify_step_px", cfg.manual_feature_image_densify_step_px);
        readInt(node, "manual_free_line_sample_count", cfg.manual_free_line_sample_count);
        readDouble(node, "manual_free_line_distance_weight", cfg.manual_free_line_distance_weight);
        readDouble(node, "manual_free_line_direction_weight", cfg.manual_free_line_direction_weight);
        readDouble(node, "manual_free_line_endpoint_weight", cfg.manual_free_line_endpoint_weight);
        readDouble(node, "manual_free_line_length_weight", cfg.manual_free_line_length_weight);

        cv::FileNode manual_feature_node = node["manual_feature_pairing"];
        if (!manual_feature_node.empty()) {
            readBool(manual_feature_node, "enable", cfg.enable_manual_feature_pairing);
            readDouble(manual_feature_node, "snap_radius_px", cfg.manual_feature_snap_radius_px);
            readInt(manual_feature_node, "min_segment_points", cfg.manual_feature_min_segment_points);
            readDouble(manual_feature_node, "min_segment_length_px", cfg.manual_feature_min_segment_length_px);
            readBool(manual_feature_node, "image_corner_snap_enable", cfg.manual_image_corner_snap_enabled);
            readBool(manual_feature_node, "image_corner_snap_enabled", cfg.manual_image_corner_snap_enabled);
            readInt(manual_feature_node, "image_corner_snap_radius_px", cfg.manual_image_corner_snap_radius_px);
            readInt(manual_feature_node, "residual_sample_step", cfg.manual_feature_residual_sample_step);
            readDouble(manual_feature_node, "normal_weight", cfg.manual_feature_normal_weight);
            readDouble(manual_feature_node, "endpoint_weight", cfg.manual_feature_endpoint_weight);
            readDouble(manual_feature_node, "center_weight", cfg.manual_feature_center_weight);
            readDouble(manual_feature_node, "huber_loss_px", cfg.manual_feature_huber_loss_px);
            readInt(manual_feature_node, "max_iterations", cfg.manual_feature_max_iterations);
            readDouble(manual_feature_node, "max_delta_angle_deg", cfg.manual_feature_max_delta_angle_deg);
            readDouble(manual_feature_node, "max_delta_translation_m", cfg.manual_feature_max_delta_translation_m);
            readDouble(manual_feature_node, "prior_sigma_angle_deg", cfg.manual_feature_prior_sigma_angle_deg);
            readDouble(manual_feature_node, "prior_sigma_translation_m", cfg.manual_feature_prior_sigma_translation_m);
            readDouble(manual_feature_node, "validation_mean_px", cfg.manual_feature_validation_mean_px);
            readDouble(manual_feature_node, "validation_max_px", cfg.manual_feature_validation_max_px);
            readInt(manual_feature_node, "validation_max_rounds", cfg.manual_feature_validation_max_rounds);
            readDouble(manual_feature_node, "validation_min_improvement_px", cfg.manual_feature_validation_min_improvement_px);
            readDouble(manual_feature_node, "near_priority_range_m", cfg.manual_feature_near_priority_range_m);
            readDouble(manual_feature_node, "near_priority_weight", cfg.manual_feature_near_priority_weight);
            readBool(manual_feature_node, "enable_free_line_mode", cfg.enable_manual_free_line_pairing);
            readDouble(manual_feature_node, "image_densify_step_px", cfg.manual_feature_image_densify_step_px);
            readInt(manual_feature_node, "free_line_sample_count", cfg.manual_free_line_sample_count);
            readDouble(manual_feature_node, "free_line_distance_weight", cfg.manual_free_line_distance_weight);
            readDouble(manual_feature_node, "free_line_direction_weight", cfg.manual_free_line_direction_weight);
            readDouble(manual_feature_node, "free_line_endpoint_weight", cfg.manual_free_line_endpoint_weight);
            readDouble(manual_feature_node, "free_line_length_weight", cfg.manual_free_line_length_weight);
        }

        readBool(node, "enable_line_pair_optimization", cfg.enable_line_pair_optimization);
        readInt(node, "min_line_pairs_for_optimization", cfg.min_line_pairs_for_optimization);
        readInt(node, "max_line_pairs_for_optimization", cfg.max_line_pairs_for_optimization);
        readInt(node, "line_pair_residual_sample_step", cfg.line_pair_residual_sample_step);
        readDouble(node, "line_pair_lidar_assoc_radius_px", cfg.line_pair_lidar_assoc_radius_px);
        readDouble(node, "max_delta_yaw_deg", cfg.max_delta_yaw_deg);
        readDouble(node, "max_delta_roll_deg", cfg.max_delta_roll_deg);
        readDouble(node, "max_delta_pitch_deg", cfg.max_delta_pitch_deg);
        readDouble(node, "max_delta_translation_m", cfg.max_delta_translation_m);
        readDouble(node, "prior_sigma_angle_deg", cfg.prior_sigma_angle_deg);
        readDouble(node, "prior_sigma_translation_m", cfg.prior_sigma_translation_m);
        readInt(node, "optimization_max_iterations", cfg.optimization_max_iterations);
        readDouble(node, "optimization_huber_loss_px", cfg.optimization_huber_loss_px);

        cv::FileNode mouse_roi_tool = node["mouse_roi_tool"];
        if (!mouse_roi_tool.empty()) {
            cv::FileNode n_enable = mouse_roi_tool["enable"];
            if (!n_enable.empty()) cfg.enable_mouse_roi_tool = ((int)n_enable != 0);
            cv::FileNode n_replace = mouse_roi_tool["replace_existing"];
            if (!n_replace.empty()) cfg.mouse_roi_replace_existing = ((int)n_replace != 0);
        }
        readBool(node, "enable_mouse_roi_tool", cfg.enable_mouse_roi_tool);
        readBool(node, "mouse_roi_replace_existing", cfg.mouse_roi_replace_existing);
    };

    cv::FileNode defaults_node = fs["cluster_debug_defaults"];
    cv::FileNode node = pair["cluster_debug"];
    applyScalarConfig(defaults_node);
    applyScalarConfig(node);

    auto appendRectFromNode = [](const cv::FileNode& rnode, std::vector<cv::Rect>& out) {
        if (rnode.empty() || rnode.size() < 4) return;
        int x = static_cast<int>(rnode[0]);
        int y = static_cast<int>(rnode[1]);
        int w = static_cast<int>(rnode[2]);
        int h = static_cast<int>(rnode[3]);
        if (w > 0 && h > 0) out.emplace_back(x, y, w, h);
    };

    auto appendRectsFromNode = [&](const cv::FileNode& rects_node) {
        if (rects_node.empty() || !rects_node.isSeq()) return;
        for (auto it = rects_node.begin(); it != rects_node.end(); ++it) {
            appendRectFromNode(*it, cfg.image_rois);
        }
    };

    auto readVec3 = [](const cv::FileNode& node, Eigen::Vector3d& out) -> bool {
        if (node.empty() || node.size() < 3) return false;
        out.x() = static_cast<double>(node[0]);
        out.y() = static_cast<double>(node[1]);
        out.z() = static_cast<double>(node[2]);
        return true;
    };

    // =========================================================
    // ROI 读取规则：
    //   1) 新推荐：sensor_pairs.<pair>.cluster_debug.roi.camera / roi.lidar
    //   2) 兼容旧版：cluster_debug.image_rois
    //   3) 兼容旧版：sensor_pairs.<pair>.roi.camera / roi.lidar
    // 新配置中已经把两个 ROI 合并到 cluster_debug.roi 下，避免一份配置被读两次。
    // =========================================================
    cv::FileNode debug_roi = node["roi"];
    cv::FileNode debug_cam_roi;
    cv::FileNode debug_lidar_roi;
    if (!debug_roi.empty()) {
        debug_cam_roi = debug_roi["camera"];
        debug_lidar_roi = debug_roi["lidar"];
    }

    if (!debug_cam_roi.empty()) {
        cv::FileNode n_enable = debug_cam_roi["enable"];
        if (!n_enable.empty()) cfg.use_image_rois = ((int)n_enable != 0);

        cv::FileNode n_restrict_proj = debug_cam_roi["restrict_projection"];
        if (!n_restrict_proj.empty()) cfg.restrict_projection_to_rois = ((int)n_restrict_proj != 0);

        cv::FileNode n_restrict_edge = debug_cam_roi["restrict_image_edges"];
        if (!n_restrict_edge.empty()) cfg.restrict_image_edges_to_rois = ((int)n_restrict_edge != 0);

        appendRectFromNode(debug_cam_roi["rect"], cfg.image_rois);
        appendRectsFromNode(debug_cam_roi["rects"]);
    }

    if (!debug_lidar_roi.empty()) {
        cv::FileNode n_enable = debug_lidar_roi["enable"];
        const bool has_enable = !n_enable.empty();
        if (has_enable) cfg.use_lidar_xyz_roi = ((int)n_enable != 0);

        Eigen::Vector3d xyz_min, xyz_max;
        bool has_min = readVec3(debug_lidar_roi["xyz_min"], xyz_min);
        bool has_max = readVec3(debug_lidar_roi["xyz_max"], xyz_max);
        if (has_min && has_max) {
            // 如果 YAML 没有显式写 enable，则有 xyz_min/xyz_max 就默认启用；
            // 如果显式 enable: 0，则只记录范围但不启用 XYZ ROI。
            if (!has_enable) cfg.use_lidar_xyz_roi = true;
            cfg.lidar_x_min = xyz_min.x(); cfg.lidar_y_min = xyz_min.y(); cfg.lidar_z_min = xyz_min.z();
            cfg.lidar_x_max = xyz_max.x(); cfg.lidar_y_max = xyz_max.y(); cfg.lidar_z_max = xyz_max.z();
        }
    }

    // 兼容 cluster_debug.image_rois。
    cv::FileNode debug_rois = node["image_rois"];
    if (cfg.image_rois.empty() && !debug_rois.empty()) {
        appendRectsFromNode(debug_rois);
    }

    // 兼容已有 pair.roi.camera / pair.roi.lidar；新配置不建议再使用这个块。
    cv::FileNode old_roi = pair["roi"];
    if (!old_roi.empty()) {
        if (cfg.image_rois.empty()) {
            cv::FileNode cam = old_roi["camera"];
            appendRectFromNode(cam["rect"], cfg.image_rois);
            appendRectsFromNode(cam["rects"]);
        }
        if (!cfg.use_lidar_xyz_roi) {
            cv::FileNode lidar = old_roi["lidar"];
            Eigen::Vector3d xyz_min, xyz_max;
            bool has_min = readVec3(lidar["xyz_min"], xyz_min);
            bool has_max = readVec3(lidar["xyz_max"], xyz_max);
            if (has_min && has_max) {
                cfg.use_lidar_xyz_roi = true;
                cfg.lidar_x_min = xyz_min.x(); cfg.lidar_y_min = xyz_min.y(); cfg.lidar_z_min = xyz_min.z();
                cfg.lidar_x_max = xyz_max.x(); cfg.lidar_y_max = xyz_max.y(); cfg.lidar_z_max = xyz_max.z();
            }
        }
    }

    if (g_verbose_log) std::cout << "[ClusterDebugConfig] remove_ground_by_default=" << cfg.remove_ground_by_default
              << ", ground_distance_threshold=" << cfg.ground_distance_threshold
              << ", cluster_tolerance_m=" << cfg.cluster_tolerance_m
              << ", cluster_min_points=" << cfg.cluster_min_points
              << ", cluster_max_points=" << cfg.cluster_max_points
              << ", use_image_rois=" << cfg.use_image_rois
              << ", image_roi_count=" << cfg.image_rois.size()
              << ", use_lidar_xyz_roi=" << cfg.use_lidar_xyz_roi
              << ", lidar_xyz_min=[" << cfg.lidar_x_min << ", " << cfg.lidar_y_min << ", " << cfg.lidar_z_min << "]"
              << ", lidar_xyz_max=[" << cfg.lidar_x_max << ", " << cfg.lidar_y_max << ", " << cfg.lidar_z_max << "]"
              << ", draw_cluster_boundaries=" << cfg.draw_cluster_boundaries
              << ", lidar_boundary_mode=" << cfg.lidar_boundary_mode
              << ", boundary_fill_radius_px=" << cfg.boundary_fill_radius_px
              << ", boundary_close_px=" << cfg.boundary_close_px
              << ", draw_nearby_image_edges=" << cfg.draw_nearby_image_edges
              << ", edge_search_radius_px=" << cfg.edge_search_radius_px
              << std::endl;
    return cfg;
}

static bool passLidarFilter(const Eigen::Vector3d& p, const ClusterDebugConfig& cfg) {
    if (!std::isfinite(p.x()) || !std::isfinite(p.y()) || !std::isfinite(p.z())) return false;
    const double r = p.norm();
    if (r < cfg.min_lidar_range || r > cfg.max_lidar_range) return false;
    // 注意：这里的 x/y/z 都是 LiDAR 自身坐标系，不是车体坐标系。
    // use_lidar_xyz_roi=1 时按完整 XYZ 范围裁剪；否则只保留旧版 z 过滤。
    if (cfg.use_lidar_xyz_roi) {
        if (p.x() < cfg.lidar_x_min || p.x() > cfg.lidar_x_max) return false;
        if (p.y() < cfg.lidar_y_min || p.y() > cfg.lidar_y_max) return false;
        if (p.z() < cfg.lidar_z_min || p.z() > cfg.lidar_z_max) return false;
    } else {
        if (p.z() < cfg.lidar_z_min || p.z() > cfg.lidar_z_max) return false;
    }
    return true;
}

static cv::Scalar clusterColor(int idx) {
    static const std::vector<cv::Scalar> palette = {
        cv::Scalar(0, 255, 0),      // green
        cv::Scalar(255, 0, 0),      // blue
        cv::Scalar(0, 0, 255),      // red
        cv::Scalar(0, 255, 255),    // yellow
        cv::Scalar(255, 0, 255),    // magenta
        cv::Scalar(255, 255, 0),    // cyan
        cv::Scalar(0, 128, 255),
        cv::Scalar(255, 128, 0),
        cv::Scalar(128, 0, 255),
        cv::Scalar(0, 255, 128),
        cv::Scalar(128, 255, 0),
        cv::Scalar(255, 0, 128),
        cv::Scalar(80, 180, 255),
        cv::Scalar(180, 80, 255),
        cv::Scalar(255, 180, 80),
        cv::Scalar(80, 255, 180)
    };
    return palette[idx % palette.size()];
}


static cv::Scalar scaleBgrColor(const cv::Scalar& c, double scale) {
    scale = std::max(0.0, std::min(1.0, scale));
    return cv::Scalar(c[0] * scale, c[1] * scale, c[2] * scale);
}

// 轻量文本绘制：不用大块黑色背景，改为细描边文字，尽量不遮挡图像。
static void putTextReadable(cv::Mat& img,
                            const std::string& text,
                            const cv::Point& org,
                            double scale,
                            const cv::Scalar& color,
                            int thickness = 1) {
    cv::putText(img, text, org,
                cv::FONT_HERSHEY_SIMPLEX,
                scale,
                cv::Scalar(35, 35, 35),
                thickness + 2,
                cv::LINE_AA);
    cv::putText(img, text, org,
                cv::FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thickness,
                cv::LINE_AA);
}

// 可选的半透明小背景条。默认只用于最终确认/优化窗口，避免纯黑块遮挡画面。
static void drawTransparentTextBand(cv::Mat& img,
                                    const cv::Point& p0,
                                    const cv::Point& p1,
                                    double alpha = 0.18) {
    cv::Mat overlay = img.clone();
    cv::rectangle(overlay, p0, p1, cv::Scalar(0, 0, 0), -1);
    cv::addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, img);
}


static void drawMouseRoiToolOverlay(cv::Mat& img, const ClusterDebugConfig& cfg) {
    if (g_manual_feature.active) return;
    if (!cfg.enable_mouse_roi_tool || !g_mouse_roi.enabled) return;

    const cv::Scalar uv_color(0, 255, 255);
    const cv::Scalar drag_color(255, 0, 255);
    const cv::Scalar selected_color(0, 255, 0);

    if (g_mouse_roi.uv_valid &&
        g_mouse_roi.uv.x >= 0 && g_mouse_roi.uv.x < img.cols &&
        g_mouse_roi.uv.y >= 0 && g_mouse_roi.uv.y < img.rows) {
        cv::line(img,
                 cv::Point(std::max(0, g_mouse_roi.uv.x - 8), g_mouse_roi.uv.y),
                 cv::Point(std::min(img.cols - 1, g_mouse_roi.uv.x + 8), g_mouse_roi.uv.y),
                 uv_color, 1, cv::LINE_AA);
        cv::line(img,
                 cv::Point(g_mouse_roi.uv.x, std::max(0, g_mouse_roi.uv.y - 8)),
                 cv::Point(g_mouse_roi.uv.x, std::min(img.rows - 1, g_mouse_roi.uv.y + 8)),
                 uv_color, 1, cv::LINE_AA);

        char uv_buf[96];
        std::snprintf(uv_buf, sizeof(uv_buf), "UV u=%d v=%d", g_mouse_roi.uv.x, g_mouse_roi.uv.y);
        cv::Point org(g_mouse_roi.uv.x + 12, g_mouse_roi.uv.y - 12);
        if (org.x > img.cols - 170) org.x = g_mouse_roi.uv.x - 160;
        if (org.x < 8) org.x = 8;
        if (org.y < 22) org.y = g_mouse_roi.uv.y + 28;
        if (org.y > img.rows - 8) org.y = img.rows - 8;
        putTextReadable(img, uv_buf, org, 0.48, uv_color, 1);
    }

    // 已经框选但尚未保存的一组 ROI。
    for (size_t i = 0; i < g_mouse_roi.pending_rois.size(); ++i) {
        cv::Rect r = g_mouse_roi.pending_rois[i] & cv::Rect(0, 0, img.cols, img.rows);
        if (r.area() <= 0) continue;
        cv::rectangle(img, r, selected_color, 2, cv::LINE_AA);
        char roi_buf[160];
        std::snprintf(roi_buf, sizeof(roi_buf), "ROI%zu [%d,%d,%d,%d]",
                      i, r.x, r.y, r.width, r.height);
        cv::Point org(r.x + 4, std::max(22, r.y - 6));
        putTextReadable(img, roi_buf, org, 0.46, selected_color, 1);
    }

    // 当前正在拖拽的 ROI，用紫色显示，不进入 pending_rois，直到鼠标左键松开。
    if (g_mouse_roi.dragging) {
        cv::Rect active_rect = normalizedRectFromPoints(g_mouse_roi.start, g_mouse_roi.current);
        active_rect = active_rect & cv::Rect(0, 0, img.cols, img.rows);
        if (active_rect.area() > 0) {
            cv::rectangle(img, active_rect, drag_color, 2, cv::LINE_AA);
            char roi_buf[160];
            std::snprintf(roi_buf, sizeof(roi_buf), "drag ROI [%d,%d,%d,%d]",
                          active_rect.x, active_rect.y, active_rect.width, active_rect.height);
            cv::Point org(active_rect.x + 4, std::max(22, active_rect.y - 6));
            putTextReadable(img, roi_buf, org, 0.48, drag_color, 1);
        }
    }

    // 底部轻量提示，避免遮挡主要标定区域。
    char help[256];
    std::snprintf(help, sizeof(help),
                  "Mouse ROI: move=UV, drag=add ROI, pending=%zu, Shift+S=save all, C=clear pending",
                  g_mouse_roi.pending_rois.size());
    putTextReadable(img, help, cv::Point(18, img.rows - 18), 0.45, cv::Scalar(0, 255, 255), 1);
}

static int leadingSpaces(const std::string& line) {
    int n = 0;
    while (n < static_cast<int>(line.size()) && line[n] == ' ') ++n;
    return n;
}

static std::string trimString(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

static bool startsWithString(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

static int findYamlChildLine(const std::vector<std::string>& lines,
                             int start,
                             int end,
                             int indent,
                             const std::string& key) {
    const std::string target = key + ":";
    for (int i = start; i < end; ++i) {
        const std::string t = trimString(lines[i]);
        if (t.empty() || startsWithString(t, "#")) continue;
        if (leadingSpaces(lines[i]) == indent && startsWithString(t, target)) return i;
    }
    return -1;
}

static int yamlBlockEnd(const std::vector<std::string>& lines, int start, int parent_indent) {
    for (int i = start + 1; i < static_cast<int>(lines.size()); ++i) {
        const std::string t = trimString(lines[i]);
        if (t.empty() || startsWithString(t, "#")) continue;
        if (leadingSpaces(lines[i]) <= parent_indent) return i;
    }
    return static_cast<int>(lines.size());
}

static std::vector<std::string> makeRoiYamlLines(const std::vector<cv::Rect>& rois) {
    std::vector<std::string> out;
    out.reserve(rois.size());
    for (const auto& roi : rois) {
        out.push_back("            - [" + std::to_string(roi.x) + ", " + std::to_string(roi.y) + ", " +
                      std::to_string(roi.width) + ", " + std::to_string(roi.height) +
                      "]  # [x, y, w, h] 鼠标批量框选保存的图像 ROI");
    }
    return out;
}

static bool saveMouseRoisToYaml(const std::string& config_file,
                                const std::string& pair_name,
                                const std::vector<cv::Rect>& rois,
                                bool replace_existing) {
    if (rois.empty()) {
        std::cerr << "⚠️ [MouseROI] 没有待保存 ROI。" << std::endl;
        return false;
    }

    std::ifstream in(config_file);
    if (!in.is_open()) {
        std::cerr << "❌ [MouseROI] 无法打开 YAML: " << config_file << std::endl;
        return false;
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) lines.push_back(line);
    in.close();

    const std::vector<std::string> new_roi_lines = makeRoiYamlLines(rois);
    const std::string canonical_pair = canonicalPairName(pair_name);
    int pair_line = -1;
    for (int i = 0; i < static_cast<int>(lines.size()); ++i) {
        const std::string t = trimString(lines[i]);
        if (leadingSpaces(lines[i]) == 2 && (t == pair_name + ":" || t == canonical_pair + ":")) {
            pair_line = i;
            break;
        }
    }
    if (pair_line < 0) {
        std::cerr << "❌ [MouseROI] YAML 中找不到 sensor_pairs." << pair_name << std::endl;
        return false;
    }

    const int pair_end = yamlBlockEnd(lines, pair_line, 2);
    int cluster_line = findYamlChildLine(lines, pair_line + 1, pair_end, 4, "cluster_debug");
    if (cluster_line < 0) {
        std::cerr << "❌ [MouseROI] YAML 中找不到 cluster_debug。" << std::endl;
        return false;
    }

    const int cluster_end = yamlBlockEnd(lines, cluster_line, 4);
    int roi_line = findYamlChildLine(lines, cluster_line + 1, cluster_end, 6, "roi");
    if (roi_line < 0) {
        // 插到 cluster_debug 末尾。
        std::vector<std::string> block = {
            "      # ROI 合并配置：鼠标工具自动创建",
            "      roi:",
            "        camera:",
            "          enable: 1  # 1=启用图像 ROI 显示/限制",
            "          restrict_projection: 0  # 1=只显示/使用 ROI 内 LiDAR 投影点",
            "          restrict_image_edges: 0  # 1=图像边缘搜索结果也必须落在 ROI 内",
            "          rects:"
        };
        block.insert(block.end(), new_roi_lines.begin(), new_roi_lines.end());
        lines.insert(lines.begin() + cluster_end, block.begin(), block.end());
    } else {
        const int roi_end = yamlBlockEnd(lines, roi_line, 6);
        int camera_line = findYamlChildLine(lines, roi_line + 1, roi_end, 8, "camera");
        if (camera_line < 0) {
            std::vector<std::string> block = {
                "        camera:",
                "          enable: 1  # 1=启用图像 ROI 显示/限制",
                "          restrict_projection: 0  # 1=只显示/使用 ROI 内 LiDAR 投影点",
                "          restrict_image_edges: 0  # 1=图像边缘搜索结果也必须落在 ROI 内",
                "          rects:"
            };
            block.insert(block.end(), new_roi_lines.begin(), new_roi_lines.end());
            lines.insert(lines.begin() + roi_line + 1, block.begin(), block.end());
        } else {
            const int camera_end = yamlBlockEnd(lines, camera_line, 8);
            int rects_line = findYamlChildLine(lines, camera_line + 1, camera_end, 10, "rects");
            if (rects_line >= 0) {
                int rects_end = rects_line + 1;
                while (rects_end < camera_end) {
                    const std::string t = trimString(lines[rects_end]);
                    if (!t.empty() && !startsWithString(t, "#") && leadingSpaces(lines[rects_end]) <= 10) break;
                    ++rects_end;
                }
                if (replace_existing) {
                    lines.erase(lines.begin() + rects_line + 1, lines.begin() + rects_end);
                    lines.insert(lines.begin() + rects_line + 1, new_roi_lines.begin(), new_roi_lines.end());
                } else {
                    lines.insert(lines.begin() + rects_end, new_roi_lines.begin(), new_roi_lines.end());
                }
            } else {
                std::vector<std::string> block = {"          rects:"};
                block.insert(block.end(), new_roi_lines.begin(), new_roi_lines.end());
                lines.insert(lines.begin() + camera_end, block.begin(), block.end());
            }
        }
    }

    std::ofstream out(config_file, std::ios::trunc);
    if (!out.is_open()) {
        std::cerr << "❌ [MouseROI] 无法写入 YAML: " << config_file << std::endl;
        return false;
    }
    for (const auto& l : lines) out << l << "\n";

    std::cout << "✅ [MouseROI] 已保存 " << pair_name << " 的 " << rois.size()
              << " 个 ROI 到 YAML" << (replace_existing ? "（替换原 rects）" : "（追加到原 rects）")
              << std::endl;
    for (size_t i = 0; i < rois.size(); ++i) {
        const auto& r = rois[i];
        std::cout << "   ROI" << i << ": [" << r.x << ", " << r.y << ", "
                  << r.width << ", " << r.height << "]" << std::endl;
    }
    return true;
}

static std::string formatYamlDouble(double v, int precision = 6) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << v;
    std::string out = oss.str();
    // 去掉多余的 0，YAML 更清爽；保留至少一位小数，避免整数外参看起来像 int。
    while (out.size() > 2 && out.back() == '0') out.pop_back();
    if (!out.empty() && out.back() == '.') out.push_back('0');
    return out;
}

static std::string replaceYamlScalarValuePreserveComment(const std::string& old_line,
                                                         int indent,
                                                         const std::string& key,
                                                         const std::string& value,
                                                         const std::string& default_comment = "") {
    std::string comment;
    const size_t hash_pos = old_line.find('#');
    if (hash_pos != std::string::npos) {
        comment = old_line.substr(hash_pos);
    } else if (!default_comment.empty()) {
        comment = "# " + default_comment;
    }

    std::string newline(indent, ' ');
    newline += key + ": " + value;
    if (!comment.empty()) newline += "  " + comment;
    return newline;
}

static bool updateYamlExtrinsicInitial(const std::string& config_file,
                                       const std::string& pair_name,
                                       double yaw_deg,
                                       double roll_deg,
                                       double pitch_deg,
                                       const Eigen::Vector3d& t_ext) {
    std::ifstream in(config_file);
    if (!in.is_open()) {
        std::cerr << "❌ [YAML_SAVE] 无法打开 YAML: " << config_file << std::endl;
        return false;
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) lines.push_back(line);
    in.close();

    const std::string canonical_pair = canonicalPairName(pair_name);
    int pair_line = -1;
    for (int i = 0; i < static_cast<int>(lines.size()); ++i) {
        const std::string t = trimString(lines[i]);
        if (leadingSpaces(lines[i]) == 2 && (t == pair_name + ":" || t == canonical_pair + ":")) {
            pair_line = i;
            break;
        }
    }
    if (pair_line < 0) {
        std::cerr << "❌ [YAML_SAVE] YAML 中找不到 sensor_pairs." << pair_name << std::endl;
        return false;
    }

    const int pair_end = yamlBlockEnd(lines, pair_line, 2);
    int ext_line = findYamlChildLine(lines, pair_line + 1, pair_end, 4, "extrinsic_initial");

    const std::string yaw_s   = formatYamlDouble(yaw_deg, 6);
    const std::string roll_s  = formatYamlDouble(roll_deg, 6);
    const std::string pitch_s = formatYamlDouble(pitch_deg, 6);
    const std::string x_s     = formatYamlDouble(t_ext.x(), 6);
    const std::string y_s     = formatYamlDouble(t_ext.y(), 6);
    const std::string z_s     = formatYamlDouble(t_ext.z(), 6);

    std::vector<std::string> new_block = {
        "    extrinsic_initial:",
        "      yaw_deg: " + yaw_s + "  # yaw，单位 deg；保存顺序对应 [yaw, roll, pitch, x, y, z]",
        "      roll_deg: " + roll_s + "  # roll，单位 deg；绕相机 Z 轴",
        "      pitch_deg: " + pitch_s + "  # pitch，单位 deg；绕相机 X 轴",
        "      x: " + x_s + "  # T_ext.x，单位 m，相机坐标系下 LiDAR 原点位置",
        "      y: " + y_s + "  # T_ext.y，单位 m，相机坐标系下 LiDAR 原点位置",
        "      z: " + z_s + "  # T_ext.z，单位 m，相机坐标系下 LiDAR 原点位置"
    };

    if (ext_line < 0) {
        // 没有 extrinsic_initial 时，插到 camera block 后面；找不到 camera 就插到 pair 开头后。
        int insert_pos = pair_line + 1;
        int camera_line = findYamlChildLine(lines, pair_line + 1, pair_end, 4, "camera");
        if (camera_line >= 0) insert_pos = yamlBlockEnd(lines, camera_line, 4);
        lines.insert(lines.begin() + insert_pos, new_block.begin(), new_block.end());
    } else {
        const std::string ext_trim = trimString(lines[ext_line]);
        const bool inline_seq = (ext_trim.find('[') != std::string::npos);
        const int ext_end = inline_seq ? (ext_line + 1) : yamlBlockEnd(lines, ext_line, 4);

        if (!inline_seq) {
            // 优先保留现有 map 风格和每行注释。
            int yaw_line = findYamlChildLine(lines, ext_line + 1, ext_end, 6, "yaw_deg");
            int roll_line = findYamlChildLine(lines, ext_line + 1, ext_end, 6, "roll_deg");
            int pitch_line = findYamlChildLine(lines, ext_line + 1, ext_end, 6, "pitch_deg");
            int x_line = findYamlChildLine(lines, ext_line + 1, ext_end, 6, "x");
            int y_line = findYamlChildLine(lines, ext_line + 1, ext_end, 6, "y");
            int z_line = findYamlChildLine(lines, ext_line + 1, ext_end, 6, "z");

            if (yaw_line >= 0 && roll_line >= 0 && pitch_line >= 0 &&
                x_line >= 0 && y_line >= 0 && z_line >= 0) {
                lines[yaw_line] = replaceYamlScalarValuePreserveComment(lines[yaw_line], 6, "yaw_deg", yaw_s, "yaw，单位 deg");
                lines[roll_line] = replaceYamlScalarValuePreserveComment(lines[roll_line], 6, "roll_deg", roll_s, "roll，单位 deg");
                lines[pitch_line] = replaceYamlScalarValuePreserveComment(lines[pitch_line], 6, "pitch_deg", pitch_s, "pitch，单位 deg");
                lines[x_line] = replaceYamlScalarValuePreserveComment(lines[x_line], 6, "x", x_s, "T_ext.x，单位 m");
                lines[y_line] = replaceYamlScalarValuePreserveComment(lines[y_line], 6, "y", y_s, "T_ext.y，单位 m");
                lines[z_line] = replaceYamlScalarValuePreserveComment(lines[z_line], 6, "z", z_s, "T_ext.z，单位 m");
            } else {
                // 结构不完整时，直接替换整个 extrinsic_initial block。
                lines.erase(lines.begin() + ext_line, lines.begin() + ext_end);
                lines.insert(lines.begin() + ext_line, new_block.begin(), new_block.end());
            }
        } else {
            // 兼容旧格式 extrinsic_initial: [yaw, roll, pitch, x, y, z]，替换为更清晰的 map 格式。
            lines.erase(lines.begin() + ext_line, lines.begin() + ext_end);
            lines.insert(lines.begin() + ext_line, new_block.begin(), new_block.end());
        }
    }

    std::ofstream out(config_file, std::ios::trunc);
    if (!out.is_open()) {
        std::cerr << "❌ [YAML_SAVE] 无法写入 YAML: " << config_file << std::endl;
        return false;
    }
    for (const auto& l : lines) out << l << "\n";

    std::cout << "✅ [YAML_SAVE] 已同步更新 " << pair_name
              << " 的 extrinsic_initial 到 YAML: " << config_file << std::endl;
    std::cout << std::fixed << std::setprecision(6)
              << "   extrinsic_initial: [" << yaw_deg << ", " << roll_deg << ", " << pitch_deg
              << ", " << t_ext.x() << ", " << t_ext.y() << ", " << t_ext.z() << "]" << std::endl;
    return true;
}


struct ClusterInfo {
    int id = -1;
    std::vector<Eigen::Vector3d> points;
    Eigen::Vector3d min_pt = Eigen::Vector3d::Zero();
    Eigen::Vector3d max_pt = Eigen::Vector3d::Zero();
    cv::Scalar color = cv::Scalar(255, 255, 255);
};

struct ClusterDebugResult {
    std::vector<ClusterInfo> clusters;
    int raw_points = 0;
    int filtered_points = 0;
    int raw_cluster_count = 0;
    int kept_cluster_points = 0;
};

static ClusterDebugResult buildEuclideanClusters(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    const ClusterDebugConfig& cfg) {

    ClusterDebugResult out;
    if (!cloud) return out;
    out.raw_points = static_cast<int>(cloud->points.size());

    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZI>);
    filtered->reserve(cloud->points.size());
    for (const auto& p : cloud->points) {
        Eigen::Vector3d ep(p.x, p.y, p.z);
        if (!passLidarFilter(ep, cfg)) continue;
        filtered->push_back(p);
    }
    filtered->width = static_cast<uint32_t>(filtered->size());
    filtered->height = 1;
    filtered->is_dense = false;
    out.filtered_points = static_cast<int>(filtered->size());

    if (filtered->empty()) {
        std::cerr << "⚠️ [Cluster] 过滤后点云为空。" << std::endl;
        return out;
    }

    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(filtered);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance(cfg.cluster_tolerance_m);
    ec.setMinClusterSize(std::max(1, cfg.cluster_min_points));
    ec.setMaxClusterSize(std::max(cfg.cluster_min_points, cfg.cluster_max_points));
    ec.setSearchMethod(tree);
    ec.setInputCloud(filtered);
    ec.extract(cluster_indices);
    out.raw_cluster_count = static_cast<int>(cluster_indices.size());

    std::sort(cluster_indices.begin(), cluster_indices.end(),
              [](const pcl::PointIndices& a, const pcl::PointIndices& b) {
                  return a.indices.size() > b.indices.size();
              });

    int keep_num = std::min(static_cast<int>(cluster_indices.size()), std::max(1, cfg.max_display_clusters));
    out.clusters.reserve(keep_num);

    for (int ci = 0; ci < keep_num; ++ci) {
        ClusterInfo info;
        info.id = ci;
        info.color = clusterColor(ci);
        info.points.reserve(cluster_indices[ci].indices.size());

        Eigen::Vector3d min_p(std::numeric_limits<double>::max(),
                              std::numeric_limits<double>::max(),
                              std::numeric_limits<double>::max());
        Eigen::Vector3d max_p(-std::numeric_limits<double>::max(),
                              -std::numeric_limits<double>::max(),
                              -std::numeric_limits<double>::max());

        for (int idx : cluster_indices[ci].indices) {
            const auto& pp = filtered->points[idx];
            Eigen::Vector3d p(pp.x, pp.y, pp.z);
            info.points.push_back(p);
            min_p = min_p.cwiseMin(p);
            max_p = max_p.cwiseMax(p);
        }
        info.min_pt = min_p;
        info.max_pt = max_p;
        out.kept_cluster_points += static_cast<int>(info.points.size());
        out.clusters.push_back(std::move(info));
    }

    if (g_verbose_log) std::cout << "✅ [Cluster] raw_points=" << out.raw_points
              << ", filtered_points=" << out.filtered_points
              << ", raw_clusters=" << out.raw_cluster_count
              << ", kept_clusters=" << out.clusters.size()
              << ", kept_points=" << out.kept_cluster_points
              << ", tolerance=" << cfg.cluster_tolerance_m
              << ", min_points=" << cfg.cluster_min_points << std::endl;

    for (const auto& c : out.clusters) {
        Eigen::Vector3d extent = c.max_pt - c.min_pt;
        if (g_verbose_log) std::cout << "   cluster #" << c.id
                  << " pts=" << c.points.size()
                  << " extent=[" << extent.x() << ", " << extent.y() << ", " << extent.z() << "]"
                  << " min=[" << c.min_pt.x() << ", " << c.min_pt.y() << ", " << c.min_pt.z() << "]"
                  << " max=[" << c.max_pt.x() << ", " << c.max_pt.y() << ", " << c.max_pt.z() << "]"
                  << std::endl;
    }

    return out;
}

static cv::Rect clampRectToImage(const cv::Rect& r, int width, int height) {
    cv::Rect img_rect(0, 0, width, height);
    return r & img_rect;
}

static bool pointInAnyImageROI(const cv::Point& uv, const ClusterDebugConfig& cfg, int width, int height) {
    if (!cfg.use_image_rois || cfg.image_rois.empty()) return true;
    for (const auto& r0 : cfg.image_rois) {
        cv::Rect r = clampRectToImage(r0, width, height);
        if (r.area() > 0 && r.contains(uv)) return true;
    }
    return false;
}

static void drawImageROIs(cv::Mat& show, const ClusterDebugConfig& cfg) {
    if (!cfg.use_image_rois || cfg.image_rois.empty()) return;
    for (size_t i = 0; i < cfg.image_rois.size(); ++i) {
        cv::Rect r = clampRectToImage(cfg.image_rois[i], show.cols, show.rows);
        if (r.area() <= 0) continue;
        cv::rectangle(show, r, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        std::string label = "ROI" + std::to_string(i);
        putTextReadable(show, label, r.tl() + cv::Point(4, 18),
                        0.50, cv::Scalar(0, 255, 255), 1);
    }
}

// F/G 人工配对专用配置：完全忽略所有相机 ROI 和 LiDAR XYZ ROI。
// 人工模式只认鼠标实际选中的 3D LiDAR 特征与 2D 图像特征；
// YAML 中已有的 ROI 只保留给旧的自动匹配/ROI 调试流程，不参与人工选点和人工优化。
static ClusterDebugConfig makeManualFeatureNoRoiConfig(const ClusterDebugConfig& base) {
    ClusterDebugConfig cfg = base;

    // 图像侧：全图提取候选边界，禁止任何 ROI 裁剪。
    cfg.use_image_rois = false;
    cfg.image_rois.clear();
    cfg.restrict_projection_to_rois = false;
    cfg.restrict_image_edges_to_rois = false;
    cfg.restrict_image_contours_to_rois = false;

    // LiDAR 侧：人工候选点云完全不使用 cluster_debug.roi.lidar.xyz_min/xyz_max，
    // 包括其中的 z 范围。仍保留 min/max_lidar_range 和去地面等非 ROI 通用预处理。
    cfg.use_lidar_xyz_roi = false;
    cfg.lidar_x_min = -1.0e9;
    cfg.lidar_x_max =  1.0e9;
    cfg.lidar_y_min = -1.0e9;
    cfg.lidar_y_max =  1.0e9;
    cfg.lidar_z_min = -1.0e9;
    cfg.lidar_z_max =  1.0e9;

    cfg.roi_preview_thick_candidates = false;
    // 人工选点阶段只看完整候选边界，不运行自动红蓝匹配，避免视觉干扰。
    cfg.draw_nearby_image_edges = false;
    cfg.draw_line_pair_center_links = false;
    cfg.draw_line_pair_labels = false;
    cfg.cluster_point_color_scale = std::min(cfg.cluster_point_color_scale, 0.30);
    return cfg;
}

// 兼容旧调用名。当前全图候选显示与 F/G 人工配对共用同一套“无 ROI”配置。
static ClusterDebugConfig makeGlobalCandidateDisplayConfig(const ClusterDebugConfig& base) {
    return makeManualFeatureNoRoiConfig(base);
}

// ROI 框选后用于预览：
// 关键原则：红色 LiDAR 边界仍然来自“完整 cluster 的原始投影边界”，
// 不在 ROI 内重新截断点云/重新提边界；ROI 只用于从这些原始细红边界中筛出可匹配段并加粗显示。
static ClusterDebugConfig makeRoiFeaturePreviewConfig(const ClusterDebugConfig& base,
                                                      const std::vector<cv::Rect>& rois) {
    ClusterDebugConfig cfg = base;
    cfg.use_image_rois = true;
    cfg.image_rois = rois;

    // 这里必须保持 false。
    // 如果设为 true，projectPoint() 会先把 ROI 外 LiDAR 投影点裁掉，
    // 后续 buildClusterBoundaryContoursFromProjection() 会基于“被 ROI 截断后的点云”重新生成一条新的红边界。
    // 这不是我们想要的；我们只想把原始细红边界落在 ROI 内的部分加粗。
    cfg.restrict_projection_to_rois = false;

    // 图像轮廓和红蓝匹配仍然限制在 ROI 内。
    cfg.restrict_image_edges_to_rois = true;
    cfg.restrict_image_contours_to_rois = true;

    // 不把整条候选红边界直接加粗，只加粗匹配成功的红/蓝局部线段。
    cfg.roi_preview_thick_candidates = false;
    cfg.draw_line_pair_matches = true;
    cfg.draw_line_pair_center_links = false;
    cfg.draw_line_pair_labels = false;
    cfg.cluster_point_color_scale = std::min(cfg.cluster_point_color_scale, 0.20);
    return cfg;
}

static bool sameRectVector(const std::vector<cv::Rect>& a, const std::vector<cv::Rect>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].x != b[i].x || a[i].y != b[i].y ||
            a[i].width != b[i].width || a[i].height != b[i].height) {
            return false;
        }
    }
    return true;
}

static std::vector<cv::Rect> currentPreviewRois(const ClusterDebugConfig& cfg, bool roi_saved_once) {
    if (!g_mouse_roi.pending_rois.empty()) return g_mouse_roi.pending_rois;
    if (roi_saved_once && cfg.use_image_rois && !cfg.image_rois.empty()) return cfg.image_rois;
    return {};
}

struct ImageEdgeData {
    cv::Mat gray;
    cv::Mat edges;     // CV_8U Canny edges
    cv::Mat grad_x;    // CV_32F Scharr x
    cv::Mat grad_y;    // CV_32F Scharr y
    cv::Mat grad_mag;  // CV_32F gradient magnitude
    int edge_count = 0;

    // 图像 ROI 内独立提取的连续边缘轮廓/折线。
    // 这些蓝色线是图像自己的特征，不依赖 LiDAR 红线逐点搜索。
    std::vector<std::vector<cv::Point>> roi_edge_contours;
};

struct ProjectedLidarPoint {
    cv::Point uv;
    Eigen::Vector3d pt_lidar;
};

static std::vector<cv::Point> densifyClosedPolyline(const std::vector<cv::Point>& poly, int step_px) {
    std::vector<cv::Point> out;
    if (poly.size() < 2) return out;
    const int step = std::max(1, step_px);
    for (size_t i = 0; i < poly.size(); ++i) {
        const cv::Point2d a = poly[i];
        const cv::Point2d b = poly[(i + 1) % poly.size()];
        const double len = cv::norm(b - a);
        const int n = std::max(1, static_cast<int>(std::ceil(len / static_cast<double>(step))));
        for (int k = 0; k < n; ++k) {
            const double t = static_cast<double>(k) / static_cast<double>(n);
            cv::Point q(static_cast<int>(std::round(a.x + (b.x - a.x) * t)),
                        static_cast<int>(std::round(a.y + (b.y - a.y) * t)));
            if (out.empty() || out.back() != q) out.push_back(q);
        }
    }
    if (out.size() >= 2 && out.front() == out.back()) out.pop_back();
    return out;
}

static std::vector<cv::Point> projectedPointsToClusterHullContour(
    const std::vector<ProjectedLidarPoint>& projected_points,
    const ClusterDebugConfig& cfg) {
    std::vector<cv::Point> pts;
    pts.reserve(projected_points.size());
    for (const auto& p : projected_points) pts.push_back(p.uv);
    if (pts.size() < 3) return {};

    std::vector<cv::Point> hull;
    cv::convexHull(pts, hull, true);
    if (hull.size() < 3) return {};

    if (cfg.boundary_hull_approx_epsilon_px > 0.0 && hull.size() >= 3) {
        std::vector<cv::Point> approx;
        cv::approxPolyDP(hull, approx, cfg.boundary_hull_approx_epsilon_px, true);
        if (approx.size() >= 3) hull.swap(approx);
    }

    std::vector<cv::Point> dense = densifyClosedPolyline(hull, cfg.lidar_boundary_sample_step_px);
    if (dense.size() >= 3) return dense;
    return hull;
}

static std::vector<std::vector<cv::Point>> buildClusterBoundaryContoursFromProjection(
    const cv::Size& image_size,
    const cv::Mat& sparse_mask,
    const std::vector<ProjectedLidarPoint>& projected_points,
    const ClusterDebugConfig& cfg) {

    std::vector<std::vector<cv::Point>> contours;

    if (cfg.lidar_boundary_mode == 1) {
        // 凸包模式：只作为对比调试。它会把凹区/空区一起包进去，不适合精确匹配图像边缘。
        auto hull_contour = projectedPointsToClusterHullContour(projected_points, cfg);
        if (!hull_contour.empty()) contours.push_back(std::move(hull_contour));
        return contours;
    }

    cv::Mat boundary_mask = cv::Mat::zeros(image_size, CV_8U);
    if (cfg.lidar_boundary_mode == 2) {
        // 推荐模式：投影点云足迹外轮廓。
        // 不是凸包：先把同一个 3D cluster 的投影点画成小圆，再用较小 close 连接相邻扫描线，
        // 最后只取外部 contour。这样红线跟着点云外边界走，不会跨过大块空白区域。
        const int radius = std::max(1, cfg.boundary_fill_radius_px);
        for (const auto& p : projected_points) {
            cv::circle(boundary_mask, p.uv, radius, cv::Scalar(255), -1, cv::LINE_AA);
        }
        const int close_px = std::max(0, cfg.boundary_close_px);
        if (close_px > 0) {
            cv::Mat kernel = cv::getStructuringElement(
                cv::MORPH_ELLIPSE,
                cv::Size(2 * close_px + 1, 2 * close_px + 1));
            cv::morphologyEx(boundary_mask, boundary_mask, cv::MORPH_CLOSE, kernel);
        }
    } else {
        // 旧模式：稀疏投影点 mask + 小膨胀。保留用于对比调试。
        boundary_mask = sparse_mask.clone();
        const int dilate_px = std::max(0, cfg.boundary_mask_dilate_px);
        if (dilate_px > 0) {
            cv::Mat kernel = cv::getStructuringElement(
                cv::MORPH_ELLIPSE,
                cv::Size(2 * dilate_px + 1, 2 * dilate_px + 1));
            cv::dilate(sparse_mask, boundary_mask, kernel);
            cv::morphologyEx(boundary_mask, boundary_mask, cv::MORPH_CLOSE, kernel);
        }
    }

    cv::findContours(boundary_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    if (cfg.lidar_boundary_mode == 2) {
        // occupancy 模式下投影点可能分成多个断开的有效目标，例如两个椅子。
        // 旧逻辑只保留最大轮廓，会把较小但可点击的目标轮廓丢掉；默认保留全部，后续再按面积/周长阈值过滤。
        if (contours.size() > 1) {
            std::sort(contours.begin(), contours.end(),
                [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                    return std::abs(cv::contourArea(a)) > std::abs(cv::contourArea(b));
                });
            if (cfg.boundary_keep_largest_only) {
                contours.resize(1);
            }
        }

        // 轮廓轻微简化再重新加密采样：减少锯齿，但保留贴合外边界的形状。
        if (!contours.empty() && cfg.boundary_hull_approx_epsilon_px > 0.0) {
            std::vector<cv::Point> approx;
            cv::approxPolyDP(contours.front(), approx, cfg.boundary_hull_approx_epsilon_px, true);
            if (approx.size() >= 3) {
                contours.front() = densifyClosedPolyline(approx, cfg.lidar_boundary_sample_step_px);
            } else {
                contours.front() = densifyClosedPolyline(contours.front(), cfg.lidar_boundary_sample_step_px);
            }
        }
    }

    return contours;
}

static double contourLengthPxLocal(const std::vector<cv::Point>& pts) {
    if (pts.size() < 2) return 0.0;
    double len = 0.0;
    for (size_t i = 1; i < pts.size(); ++i) len += cv::norm(pts[i] - pts[i - 1]);
    return len;
}

static ImageEdgeData buildImageEdgeData(const cv::Mat& bgr_or_gray,
                                        const ClusterDebugConfig& cfg) {
    ImageEdgeData out;
    if (bgr_or_gray.channels() == 1) out.gray = bgr_or_gray.clone();
    else cv::cvtColor(bgr_or_gray, out.gray, cv::COLOR_BGR2GRAY);

    cv::Mat blur;
    cv::GaussianBlur(out.gray, blur, cv::Size(5, 5), 0.0);

    int high = cfg.canny_high > cfg.canny_low ? cfg.canny_high : cfg.canny_low * 3;
    cv::Canny(blur, out.edges, cfg.canny_low, high, 3, true);
    cv::Scharr(blur, out.grad_x, CV_32F, 1, 0);
    cv::Scharr(blur, out.grad_y, CV_32F, 0, 1);
    cv::magnitude(out.grad_x, out.grad_y, out.grad_mag);
    out.edge_count = cv::countNonZero(out.edges);

    // 在图像 ROI 内独立提取连续边缘轮廓。
    // 后续红蓝线匹配时，LiDAR 红色轮廓会和这些蓝色图像轮廓做结构匹配，
    // 不再只是在红色点周围找最近 Canny px. 
    if (cfg.use_roi_image_contours) {
        cv::Mat edge_mask = cv::Mat::zeros(out.edges.size(), CV_8U);
        for (int y = 0; y < out.edges.rows; ++y) {
            const uchar* ep = out.edges.ptr<uchar>(y);
            const float* gp = out.grad_mag.ptr<float>(y);
            uchar* mp = edge_mask.ptr<uchar>(y);
            for (int x = 0; x < out.edges.cols; ++x) {
                if (ep[x] == 0 || gp[x] < cfg.edge_gradient_min) continue;
                cv::Point p(x, y);
                if (cfg.restrict_image_contours_to_rois &&
                    !pointInAnyImageROI(p, cfg, out.edges.cols, out.edges.rows)) {
                    continue;
                }
                mp[x] = 255;
            }
        }

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edge_mask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
        for (const auto& c0 : contours) {
            if (static_cast<int>(c0.size()) < cfg.image_contour_min_points) continue;
            const double arc = cv::arcLength(c0, false);
            if (arc < cfg.image_contour_min_length_px) continue;
            const double area = std::abs(cv::contourArea(c0));
            if (area < cfg.image_contour_min_area_px) continue;

            std::vector<cv::Point> c = c0;
            if (cfg.image_contour_approx_epsilon_px > 0.0 && c0.size() >= 3) {
                cv::approxPolyDP(c0, c, cfg.image_contour_approx_epsilon_px, false);
            }
            if (static_cast<int>(c.size()) < 2) continue;
            if (contourLengthPxLocal(c) < cfg.image_contour_min_length_px) continue;
            out.roi_edge_contours.push_back(std::move(c));
        }
    }

    if (g_verbose_log) std::cout << "✅ [ImageEdge] Canny edges=" << out.edge_count
              << ", roi_contours=" << out.roi_edge_contours.size()
              << ", canny=[" << cfg.canny_low << ", " << high << "]"
              << ", edge_gradient_min=" << cfg.edge_gradient_min
              << ", search_radius=" << cfg.edge_search_radius_px << " px" << std::endl;
    return out;
}

static void drawAllCannyEdges(cv::Mat& show,
                              const ImageEdgeData& edge,
                              const ClusterDebugConfig& cfg) {
    if (!cfg.draw_all_canny_edges || edge.edges.empty()) return;
    for (int y = 0; y < edge.edges.rows; ++y) {
        const uchar* ep = edge.edges.ptr<uchar>(y);
        const float* gp = edge.grad_mag.ptr<float>(y);
        for (int x = 0; x < edge.edges.cols; ++x) {
            if (ep[x] == 0 || gp[x] < cfg.edge_gradient_min) continue;
            cv::Point p(x, y);
            if (cfg.restrict_image_edges_to_rois && !pointInAnyImageROI(p, cfg, show.cols, show.rows)) continue;
            // 深灰蓝色：全部图像 Canny 边缘，仅 debug 使用。
            cv::circle(show, p, 1, cv::Scalar(90, 90, 40), -1, cv::LINE_AA);
        }
    }
}

static bool findNearestImageEdgeAroundBoundary(const ImageEdgeData& edge,
                                               const cv::Point& query,
                                               const ClusterDebugConfig& cfg,
                                               int width,
                                               int height,
                                               cv::Point& edge_uv) {
    if (edge.edges.empty()) return false;
    const int r = std::max(1, static_cast<int>(std::round(cfg.edge_search_radius_px)));
    double best_d2 = std::numeric_limits<double>::max();
    bool found = false;

    for (int y = std::max(0, query.y - r); y <= std::min(height - 1, query.y + r); ++y) {
        const uchar* ep = edge.edges.ptr<uchar>(y);
        const float* gp = edge.grad_mag.ptr<float>(y);
        for (int x = std::max(0, query.x - r); x <= std::min(width - 1, query.x + r); ++x) {
            if (ep[x] == 0) continue;
            if (gp[x] < cfg.edge_gradient_min) continue;
            cv::Point p(x, y);
            if (cfg.restrict_image_edges_to_rois && !pointInAnyImageROI(p, cfg, width, height)) continue;
            const double dx = static_cast<double>(x - query.x);
            const double dy = static_cast<double>(y - query.y);
            const double d2 = dx * dx + dy * dy;
            if (d2 > cfg.edge_search_radius_px * cfg.edge_search_radius_px) continue;
            if (d2 < best_d2) {
                best_d2 = d2;
                edge_uv = p;
                found = true;
            }
        }
    }
    return found;
}


static void drawRoiImageContours(cv::Mat& show,
                                 const ImageEdgeData& edge,
                                 const ClusterDebugConfig& cfg) {
    if (!cfg.draw_roi_image_contours) return;
    const bool thick_preview = cfg.roi_preview_thick_candidates;
    const int thick = thick_preview ? std::max(2, cfg.roi_preview_boundary_line_thickness) : 1;
    const cv::Scalar color = thick_preview
        ? cv::Scalar(255, 0, 0)      // ROI 预览阶段：粗亮蓝，表示 ROI 内图像边界候选
        : cv::Scalar(150, 70, 0);    // 全图/普通阶段：细暗蓝，表示图像边界候选
    for (const auto& c : edge.roi_edge_contours) {
        if (c.size() < 2) continue;
        std::vector<std::vector<cv::Point>> vv{c};
        cv::polylines(show, vv, false, color, thick, cv::LINE_AA);
    }
}

static bool nearestPointOnPolyline(const std::vector<cv::Point>& poly,
                                   const cv::Point& q,
                                   cv::Point& nearest,
                                   double& best_dist) {
    if (poly.empty()) return false;
    best_dist = std::numeric_limits<double>::max();
    bool found = false;
    for (const auto& p : poly) {
        const double d = cv::norm(p - q);
        if (d < best_dist) {
            best_dist = d;
            nearest = p;
            found = true;
        }
    }
    return found;
}

static bool findNearestRoiImageContourPoint(const ImageEdgeData& edge,
                                            const cv::Point& query,
                                            const ClusterDebugConfig& cfg,
                                            int width,
                                            int height,
                                            cv::Point& edge_uv,
                                            int& contour_id) {
    contour_id = -1;
    if (edge.roi_edge_contours.empty()) return false;
    if (cfg.restrict_image_edges_to_rois && !pointInAnyImageROI(query, cfg, width, height)) return false;

    const double max_dist = std::max(1.0, cfg.image_contour_match_radius_px);
    double best_d = std::numeric_limits<double>::max();
    cv::Point best_p;
    int best_id = -1;

    for (int i = 0; i < static_cast<int>(edge.roi_edge_contours.size()); ++i) {
        cv::Point p;
        double d = 0.0;
        if (!nearestPointOnPolyline(edge.roi_edge_contours[i], query, p, d)) continue;
        if (d < best_d) {
            best_d = d;
            best_p = p;
            best_id = i;
        }
    }

    if (best_id < 0 || best_d > max_dist) return false;
    edge_uv = best_p;
    contour_id = best_id;
    return true;
}

static double polylineLengthPx(const std::vector<cv::Point>& pts) {
    if (pts.size() < 2) return 0.0;
    double len = 0.0;
    for (size_t i = 1; i < pts.size(); ++i) {
        len += cv::norm(pts[i] - pts[i - 1]);
    }
    return len;
}

static void drawImageEdgePolylineSegment(cv::Mat& show,
                                         const std::vector<cv::Point>& seg,
                                         const ClusterDebugConfig& cfg) {
    if (static_cast<int>(seg.size()) < std::max(2, cfg.image_edge_line_min_points)) return;
    if (polylineLengthPx(seg) < cfg.image_edge_line_min_length_px) return;

    std::vector<std::vector<cv::Point>> vv;
    vv.push_back(seg);
    cv::polylines(show,
                  vv,
                  false,
                  cv::Scalar(255, 0, 0),  // 蓝色：图像边缘折线
                  std::max(1, cfg.image_edge_line_thickness),
                  cv::LINE_AA);
}

static void drawMatchedImageEdgePolylines(cv::Mat& show,
                                          const std::vector<cv::Point>& matched_edge_pts,
                                          const ClusterDebugConfig& cfg) {
    if (matched_edge_pts.empty()) return;

    if (cfg.draw_nearby_image_edge_points) {
        for (const auto& p : matched_edge_pts) {
            cv::circle(show,
                       p,
                       std::max(1, cfg.image_edge_point_radius),
                       cv::Scalar(255, 0, 0),
                       -1,
                       cv::LINE_AA);
        }
    }

    if (!cfg.draw_nearby_image_edge_lines) return;

    const double max_gap = std::max(1.0, cfg.image_edge_line_max_gap_px);
    std::vector<cv::Point> seg;
    seg.reserve(matched_edge_pts.size());

    for (size_t i = 0; i < matched_edge_pts.size(); ++i) {
        if (seg.empty()) {
            seg.push_back(matched_edge_pts[i]);
            continue;
        }

        const double d = cv::norm(matched_edge_pts[i] - seg.back());
        if (d <= max_gap) {
            seg.push_back(matched_edge_pts[i]);
        } else {
            drawImageEdgePolylineSegment(show, seg, cfg);
            seg.clear();
            seg.push_back(matched_edge_pts[i]);
        }
    }

    drawImageEdgePolylineSegment(show, seg, cfg);
}


struct RedBlueLinePair {
    int id = -1;
    std::vector<cv::Point> red_pts;
    std::vector<cv::Point> blue_pts;
    std::vector<Eigen::Vector3d> lidar_pts;
    double red_len = 0.0;
    double blue_len = 0.0;
    double mean_dist = 0.0;
    double max_dist = 0.0;
    double angle_diff_deg = 0.0;
    cv::Point2d red_center = cv::Point2d(0, 0);
    cv::Point2d blue_center = cv::Point2d(0, 0);
    bool accepted = false;
};

static cv::Point2d meanPoint(const std::vector<cv::Point>& pts) {
    if (pts.empty()) return cv::Point2d(0, 0);
    cv::Point2d c(0, 0);
    for (const auto& p : pts) {
        c.x += p.x;
        c.y += p.y;
    }
    c.x /= static_cast<double>(pts.size());
    c.y /= static_cast<double>(pts.size());
    return c;
}

static Eigen::Vector2d principalDirection2D(const std::vector<cv::Point>& pts) {
    if (pts.size() < 2) return Eigen::Vector2d(1.0, 0.0);

    cv::Point2d c = meanPoint(pts);
    double sxx = 0.0, sxy = 0.0, syy = 0.0;
    for (const auto& p : pts) {
        double x = p.x - c.x;
        double y = p.y - c.y;
        sxx += x * x;
        sxy += x * y;
        syy += y * y;
    }

    // 2x2 covariance principal eigenvector.
    double trace = sxx + syy;
    double det = sxx * syy - sxy * sxy;
    double disc = std::max(0.0, trace * trace * 0.25 - det);
    double lambda = trace * 0.5 + std::sqrt(disc);

    Eigen::Vector2d dir;
    if (std::abs(sxy) > 1e-9 || std::abs(lambda - sxx) > 1e-9) {
        dir = Eigen::Vector2d(sxy, lambda - sxx);
    } else {
        dir = Eigen::Vector2d(1.0, 0.0);
    }

    if (dir.norm() < 1e-9) {
        const cv::Point& a = pts.front();
        const cv::Point& b = pts.back();
        dir = Eigen::Vector2d(b.x - a.x, b.y - a.y);
    }
    if (dir.norm() < 1e-9) return Eigen::Vector2d(1.0, 0.0);
    dir.normalize();
    return dir;
}

static double directionAngleDiffDeg(const std::vector<cv::Point>& a,
                                    const std::vector<cv::Point>& b) {
    Eigen::Vector2d da = principalDirection2D(a);
    Eigen::Vector2d db = principalDirection2D(b);
    double dot = std::abs(da.dot(db));
    dot = std::min(1.0, std::max(0.0, dot));
    return std::acos(dot) * 180.0 / M_PI;
}

static RedBlueLinePair evaluateLinePair(const std::vector<cv::Point>& red_pts,
                                        const std::vector<cv::Point>& blue_pts,
                                        const std::vector<Eigen::Vector3d>& lidar_pts,
                                        const ClusterDebugConfig& cfg) {
    RedBlueLinePair pair;
    pair.red_pts = red_pts;
    pair.blue_pts = blue_pts;
    pair.lidar_pts = lidar_pts;
    pair.red_len = polylineLengthPx(red_pts);
    pair.blue_len = polylineLengthPx(blue_pts);
    pair.red_center = meanPoint(red_pts);
    pair.blue_center = meanPoint(blue_pts);

    const size_t n = std::min(red_pts.size(), blue_pts.size());
    if (n == 0) return pair;

    double sum = 0.0;
    double max_d = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = cv::norm(red_pts[i] - blue_pts[i]);
        sum += d;
        max_d = std::max(max_d, d);
    }
    pair.mean_dist = sum / static_cast<double>(n);
    pair.max_dist = max_d;
    pair.angle_diff_deg = directionAngleDiffDeg(red_pts, blue_pts);

    const double max_mean_dist = cfg.trust_roi_line_pairs
        ? cfg.trusted_line_pair_max_mean_dist_px
        : cfg.line_pair_max_mean_dist_px;
    const double max_angle_diff = cfg.trust_roi_line_pairs
        ? cfg.trusted_line_pair_max_angle_diff_deg
        : cfg.line_pair_max_angle_diff_deg;

    const int min_pair_points = cfg.trust_roi_line_pairs ? 2 : std::max(2, cfg.line_pair_min_points);
    const double min_red_len = cfg.trust_roi_line_pairs ? 0.0 : cfg.line_pair_min_red_length_px;
    const double min_blue_len = cfg.trust_roi_line_pairs ? 0.0 : cfg.line_pair_min_blue_length_px;

    pair.accepted =
        static_cast<int>(n) >= min_pair_points &&
        lidar_pts.size() >= n &&
        pair.red_len >= min_red_len &&
        pair.blue_len >= min_blue_len &&
        pair.mean_dist <= max_mean_dist &&
        pair.angle_diff_deg <= max_angle_diff;

    return pair;
}

static void drawAcceptedLinePair(cv::Mat& show,
                                 const RedBlueLinePair& pair,
                                 const ClusterDebugConfig& cfg,
                                 bool highlight_pair = true) {
    const int thick = highlight_pair
        ? std::max(2, cfg.matched_boundary_line_thickness)
        : 1;
    const cv::Scalar red_color = highlight_pair ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 0, 220);
    const cv::Scalar blue_color = highlight_pair ? cv::Scalar(255, 0, 0) : cv::Scalar(180, 80, 0);

    if (pair.red_pts.size() >= 2) {
        std::vector<std::vector<cv::Point>> vv{pair.red_pts};
        cv::polylines(show, vv, false, red_color, thick, cv::LINE_AA);
    }
    if (pair.blue_pts.size() >= 2) {
        std::vector<std::vector<cv::Point>> vv{pair.blue_pts};
        cv::polylines(show, vv, false, blue_color, thick, cv::LINE_AA);
    }

    cv::Point rc(static_cast<int>(std::round(pair.red_center.x)),
                 static_cast<int>(std::round(pair.red_center.y)));
    cv::Point bc(static_cast<int>(std::round(pair.blue_center.x)),
                 static_cast<int>(std::round(pair.blue_center.y)));
    if (highlight_pair && cfg.draw_line_pair_center_links) {
        cv::line(show, rc, bc, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        cv::circle(show, rc, 3, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
        cv::circle(show, bc, 3, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
    }

    if (highlight_pair && cfg.draw_line_pair_labels) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "P%d d=%.1f a=%.1f",
                      pair.id, pair.mean_dist, pair.angle_diff_deg);
        cv::Point label((rc.x + bc.x) / 2 + 4, (rc.y + bc.y) / 2 - 4);
        putTextReadable(show, buf, label,
                        0.34, cv::Scalar(230, 230, 230), 1);
    }
}

static void finalizeAndDrawLinePairSegment(cv::Mat& show,
                                           std::vector<cv::Point>& red_seg,
                                           std::vector<cv::Point>& blue_seg,
                                           std::vector<Eigen::Vector3d>& lidar_seg,
                                           const ClusterDebugConfig& cfg,
                                           int& raw_pair_count,
                                           int& accepted_pair_count,
                                           std::vector<RedBlueLinePair>* accepted_pairs_out = nullptr,
                                           bool highlight_accepted_pair = true) {
    if (red_seg.empty() || blue_seg.empty()) {
        red_seg.clear();
        blue_seg.clear();
        lidar_seg.clear();
        return;
    }

    RedBlueLinePair pair = evaluateLinePair(red_seg, blue_seg, lidar_seg, cfg);
    raw_pair_count++;

    if (pair.accepted && accepted_pair_count < std::max(1, cfg.line_pair_max_pairs)) {
        pair.id = accepted_pair_count;
        drawAcceptedLinePair(show, pair, cfg, highlight_accepted_pair);
        if (accepted_pairs_out) accepted_pairs_out->push_back(pair);
        accepted_pair_count++;
    } else {
        // 调试时仍可显示未通过筛选的蓝线，颜色更暗，避免误以为没有图像边缘。
        if (cfg.draw_nearby_image_edge_lines && pair.blue_pts.size() >= 2) {
            std::vector<std::vector<cv::Point>> vv{pair.blue_pts};
            cv::polylines(show, vv, false, cv::Scalar(130, 60, 0), 1, cv::LINE_AA);
        }
        if (cfg.draw_nearby_image_edge_points) {
            for (const auto& p : pair.blue_pts) {
                cv::circle(show, p, std::max(1, cfg.image_edge_point_radius),
                           cv::Scalar(130, 60, 0), -1, cv::LINE_AA);
            }
        }
    }

    red_seg.clear();
    blue_seg.clear();
    lidar_seg.clear();
}

static bool findNearestProjectedLidarPoint(const std::vector<ProjectedLidarPoint>& projected,
                                           const cv::Point& query,
                                           const ClusterDebugConfig& cfg,
                                           Eigen::Vector3d& pt_out) {
    if (projected.empty()) return false;
    const double max_r2 = cfg.line_pair_lidar_assoc_radius_px * cfg.line_pair_lidar_assoc_radius_px;
    double best_d2 = std::numeric_limits<double>::max();
    int best_idx = -1;
    for (int i = 0; i < static_cast<int>(projected.size()); ++i) {
        double dx = projected[i].uv.x - query.x;
        double dy = projected[i].uv.y - query.y;
        double d2 = dx * dx + dy * dy;
        if (d2 < best_d2) { best_d2 = d2; best_idx = i; }
    }
    if (best_idx < 0 || best_d2 > max_r2) return false;
    pt_out = projected[best_idx].pt_lidar;
    return true;
}

static bool findNearestProjectedLidarPointWithRadius(
    const std::vector<ProjectedLidarPoint>& projected,
    const cv::Point& query,
    double max_radius_px,
    Eigen::Vector3d& pt_out) {
    if (projected.empty()) return false;
    const double max_r2 = std::max(1.0, max_radius_px) * std::max(1.0, max_radius_px);
    double best_d2 = std::numeric_limits<double>::max();
    int best_idx = -1;
    for (int i = 0; i < static_cast<int>(projected.size()); ++i) {
        const double dx = static_cast<double>(projected[i].uv.x - query.x);
        const double dy = static_cast<double>(projected[i].uv.y - query.y);
        const double d2 = dx * dx + dy * dy;
        if (d2 < best_d2) {
            best_d2 = d2;
            best_idx = i;
        }
    }
    if (best_idx < 0 || best_d2 > max_r2) return false;
    pt_out = projected[best_idx].pt_lidar;
    return true;
}

static bool projectPoint(const Eigen::Vector3d& pt_l,
                         const Eigen::Matrix3d& R_ext,
                         const Eigen::Vector3d& T_ext,
                         const Eigen::Matrix3d& K,
                         int width,
                         int height,
                         const ClusterDebugConfig& cfg,
                         cv::Point& uv_out) {
    Eigen::Vector3d pt_cam;
    if (!use_inverse_extrinsic) {
        pt_cam = R_ext * pt_l + T_ext;
    } else {
        pt_cam = R_ext.transpose() * (pt_l - T_ext);
    }

    if (pt_cam.z() < cfg.min_camera_depth || pt_cam.z() > cfg.max_camera_depth) return false;
    Eigen::Vector3d uvw = K * pt_cam;
    int u = static_cast<int>(std::round(uvw.x() / uvw.z()));
    int v = static_cast<int>(std::round(uvw.y() / uvw.z()));
    if (u < 0 || u >= width || v < 0 || v >= height) return false;
    uv_out = cv::Point(u, v);
    if (cfg.restrict_projection_to_rois && !pointInAnyImageROI(uv_out, cfg, width, height)) return false;
    return true;
}

struct ProjectionDrawStats {
    int projected_points = 0;
    int boundary_contours = 0;
    int nearby_image_edges = 0;
    int raw_line_pairs = 0;
    int accepted_line_pairs = 0;
};

static ProjectionDrawStats drawClusterProjection(cv::Mat& show,
                                                 const std::vector<ClusterInfo>& clusters,
                                                 const Eigen::Matrix3d& R_ext,
                                                 const Eigen::Vector3d& T_ext,
                                                 const Eigen::Matrix3d& K,
                                                 const ClusterDebugConfig& cfg,
                                                 const ImageEdgeData* image_edge = nullptr,
                                                 std::vector<RedBlueLinePair>* accepted_pairs_out = nullptr,
                                                 bool highlight_matches = true,
                                                 std::vector<LidarBoundaryCandidate>* lidar_boundaries_out = nullptr) {
    ProjectionDrawStats stats;
    if (lidar_boundaries_out) lidar_boundaries_out->clear();
    const int step = std::max(1, cfg.draw_cluster_step);

    if (image_edge && cfg.use_roi_image_contours) {
        drawRoiImageContours(show, *image_edge, cfg);
    }

    for (const auto& c : clusters) {
        int local_draw = 0;
        cv::Point label_uv(0, 0);
        bool label_valid = false;
        cv::Mat mask;
        std::vector<ProjectedLidarPoint> projected_points;
        if (cfg.draw_cluster_boundaries) {
            mask = cv::Mat::zeros(show.rows, show.cols, CV_8U);
            projected_points.reserve(c.points.size() / step + 1);
        }

        // 投影并绘制 cluster 点，同时建立该 cluster 的 2D mask。
        for (size_t i = 0; i < c.points.size(); i += step) {
            cv::Point uv;
            if (!projectPoint(c.points[i], R_ext, T_ext, K, show.cols, show.rows, cfg, uv)) continue;
            cv::circle(show, uv, std::max(1, cfg.point_radius),
                       scaleBgrColor(c.color, cfg.cluster_point_color_scale),
                       -1, cv::LINE_AA);
            stats.projected_points++;
            local_draw++;
            if (cfg.draw_cluster_boundaries) {
                mask.at<uchar>(uv.y, uv.x) = 255;
                projected_points.push_back(ProjectedLidarPoint{uv, c.points[i]});
            }
            if (!label_valid) { label_uv = uv; label_valid = true; }
        }

        // 该 cluster 投影点太少时，不画边界，避免噪声小类形成假边界。
        if (cfg.draw_cluster_boundaries && local_draw >= cfg.boundary_min_projected_points) {
            std::vector<std::vector<cv::Point>> contours =
                buildClusterBoundaryContoursFromProjection(show.size(), mask, projected_points, cfg);
            for (const auto& contour : contours) {
                if (contour.size() < 3) continue;
                const double area = std::abs(cv::contourArea(contour));
                const double arc = cv::arcLength(contour, true);
                if (area < cfg.boundary_min_area_px || arc < cfg.boundary_min_arc_px) continue;

                // 红色边界线：表示该 cluster 投影后的 2D 外轮廓。
                // 如果开启线对匹配，则先用较细的暗红画完整 contour，
                // 匹配成功的局部红线会在后面用更粗的亮红覆盖。
                const bool highlight_pair_segments = highlight_matches && cfg.draw_line_pair_matches;
                const bool thick_roi_preview = cfg.roi_preview_thick_candidates;
                const int base_red_thick = thick_roi_preview
                    ? std::max(2, cfg.roi_preview_boundary_line_thickness)
                    : 1;
                const cv::Scalar base_red_color = thick_roi_preview
                    ? cv::Scalar(0, 0, 255)
                    : (highlight_pair_segments
                        ? cv::Scalar(0, 0, 140)   // 优化/确认阶段：未匹配候选边界用暗红细线
                        : cv::Scalar(0, 0, 230)); // 主窗口阶段：候选 LiDAR 边界统一用细红线
                if (cfg.draw_unmatched_lidar_contours || !cfg.draw_line_pair_matches || !highlight_pair_segments) {
                    cv::polylines(show, contour, true, base_red_color, base_red_thick, cv::LINE_AA);
                }
                stats.boundary_contours++;

                // 为人工特征配对保留“原始完整红色边界”及其最近 3D LiDAR 点。
                // 这里只导出已有 contour，不按 ROI 重新截断或重新提取边界。
                if (lidar_boundaries_out) {
                    LidarBoundaryCandidate candidate;
                    candidate.id = static_cast<int>(lidar_boundaries_out->size());
                    candidate.cluster_id = c.id;
                    candidate.closed = true;
                    candidate.uv_pts.reserve(contour.size());
                    candidate.lidar_pts.reserve(contour.size());
                    const double assoc_radius = std::max(cfg.line_pair_lidar_assoc_radius_px,
                                                         cfg.manual_feature_snap_radius_px);
                    for (const auto& q : contour) {
                        Eigen::Vector3d p_l;
                        if (!findNearestProjectedLidarPointWithRadius(projected_points, q,
                                                                      assoc_radius, p_l)) {
                            continue;
                        }
                        candidate.uv_pts.push_back(q);
                        candidate.lidar_pts.push_back(p_l);
                    }
                    if (static_cast<int>(candidate.uv_pts.size()) >=
                        std::max(2, cfg.manual_feature_min_segment_points)) {
                        lidar_boundaries_out->push_back(std::move(candidate));
                    }
                }

                // 红-蓝线对匹配：
                // 新模式：图像蓝线先在 YAML 图像 ROI 内独立提取，
                // 再把红色 LiDAR contour 与 ROI 内蓝色图像 contour 做结构匹配。
                // 旧模式仍保留：如果 use_roi_image_contours=0，则退回红线附近逐点找 Canny 边缘。
                if (image_edge && cfg.draw_nearby_image_edges) {
                    const int sample_step = std::max(1, cfg.edge_sample_step);
                    const double max_pair_gap = std::max(1.0, cfg.line_pair_max_gap_px);
                    std::vector<cv::Point> red_seg;
                    std::vector<cv::Point> blue_seg;
                    std::vector<Eigen::Vector3d> lidar_seg;
                    red_seg.reserve(contour.size() / sample_step + 1);
                    blue_seg.reserve(contour.size() / sample_step + 1);
                    lidar_seg.reserve(contour.size() / sample_step + 1);
                    int last_blue_contour_id = -1;

                    for (size_t pi = 0; pi < contour.size(); pi += sample_step) {
                        cv::Point q = contour[pi];
                        cv::Point img_edge_uv;
                        int blue_contour_id = -1;
                        bool ok = false;

                        if (cfg.use_roi_image_contours) {
                            ok = findNearestRoiImageContourPoint(*image_edge, q, cfg, show.cols, show.rows,
                                                                 img_edge_uv, blue_contour_id);
                        } else {
                            ok = findNearestImageEdgeAroundBoundary(*image_edge, q, cfg, show.cols, show.rows, img_edge_uv);
                            blue_contour_id = 0;
                        }

                        if (!ok) {
                            finalizeAndDrawLinePairSegment(show, red_seg, blue_seg, lidar_seg, cfg,
                                                           stats.raw_line_pairs, stats.accepted_line_pairs, accepted_pairs_out, highlight_pair_segments);
                            last_blue_contour_id = -1;
                            continue;
                        }

                        Eigen::Vector3d assoc_lidar;
                        if (!findNearestProjectedLidarPoint(projected_points, q, cfg, assoc_lidar)) {
                            finalizeAndDrawLinePairSegment(show, red_seg, blue_seg, lidar_seg, cfg,
                                                           stats.raw_line_pairs, stats.accepted_line_pairs, accepted_pairs_out, highlight_pair_segments);
                            last_blue_contour_id = -1;
                            continue;
                        }

                        if (!red_seg.empty()) {
                            double red_gap = cv::norm(q - red_seg.back());
                            double blue_gap = cv::norm(img_edge_uv - blue_seg.back());
                            if (blue_contour_id != last_blue_contour_id ||
                                red_gap > max_pair_gap * 1.5 || blue_gap > max_pair_gap) {
                                finalizeAndDrawLinePairSegment(show, red_seg, blue_seg, lidar_seg, cfg,
                                                               stats.raw_line_pairs, stats.accepted_line_pairs, accepted_pairs_out, highlight_pair_segments);
                            }
                        }

                        red_seg.push_back(q);
                        blue_seg.push_back(img_edge_uv);
                        lidar_seg.push_back(assoc_lidar);
                        last_blue_contour_id = blue_contour_id;

                        if (cfg.draw_edge_match_lines) {
                            cv::line(show, q, img_edge_uv, cv::Scalar(255, 180, 0), 1, cv::LINE_AA);
                        }
                        stats.nearby_image_edges++;
                    }

                    finalizeAndDrawLinePairSegment(show, red_seg, blue_seg, lidar_seg, cfg,
                                                   stats.raw_line_pairs, stats.accepted_line_pairs, accepted_pairs_out, highlight_pair_segments);
                }
            }
        }

        if (cfg.draw_cluster_labels && label_valid) {
            char buf[128];
            std::snprintf(buf, sizeof(buf), "#%d n=%zu", c.id, c.points.size());
            putTextReadable(show, buf, label_uv + cv::Point(4, -4),
                            0.36, c.color, 1);
        }
    }
    return stats;
}



// ============================================================
// 人工特征配对交互与显示
// ============================================================
static const char* manualFeatureModeText(ManualFeatureMode mode) {
    return mode == ManualFeatureMode::FREE_3D2D_LINE
        ? "G Mode: 3D LiDAR line -> 2D image line"
        : "F Debug: boundary segment pairing";
}

static const char* manualFeatureStageText(ManualFeaturePickStage stage,
                                          ManualFeatureMode mode) {
    const bool free_mode = (mode == ManualFeatureMode::FREE_3D2D_LINE);
    switch (stage) {
        case ManualFeaturePickStage::LIDAR_START:
            return free_mode ? "Step 1/4: click RED LiDAR point 1" : "Click RED boundary start";
        case ManualFeaturePickStage::LIDAR_END:
            return free_mode ? "Step 3/4: click RED LiDAR point 2" : "Click RED boundary end";
        case ManualFeaturePickStage::IMAGE_START:
            return free_mode ? "Step 2/4: click matching IMAGE point 1" : "Click BLUE image segment start";
        case ManualFeaturePickStage::IMAGE_END:
            return free_mode ? "Step 4/4: click matching IMAGE point 2" : "Click BLUE image segment end";
        case ManualFeaturePickStage::REVIEW:
            return "Review pair: Accept or Redo";
    }
    return "Unknown step";
}

static cv::Point snapImageClickToCornerOrGradient(const cv::Point& click,
                                                  bool& snapped,
                                                  double& snap_distance) {
    snapped = false;
    snap_distance = 0.0;
    if (!g_manual_feature.image_corner_snap_enabled ||
        g_manual_feature_gray_image.empty()) {
        return click;
    }

    const int radius = std::max(3, g_manual_feature.image_corner_snap_radius_px);
    const cv::Rect image_rect(0, 0, g_manual_feature_gray_image.cols,
                              g_manual_feature_gray_image.rows);
    cv::Rect roi(click.x - radius, click.y - radius, radius * 2 + 1, radius * 2 + 1);
    roi &= image_rect;
    if (roi.width < 3 || roi.height < 3) return click;

    cv::Mat patch = g_manual_feature_gray_image(roi);
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(patch, corners, 12, 0.01, 4.0, cv::Mat(), 3, true, 0.04);

    cv::Point best = click;
    double best_d2 = std::numeric_limits<double>::max();
    for (const auto& c : corners) {
        cv::Point p(roi.x + static_cast<int>(std::round(c.x)),
                    roi.y + static_cast<int>(std::round(c.y)));
        const double dx = static_cast<double>(p.x - click.x);
        const double dy = static_cast<double>(p.y - click.y);
        const double d2 = dx * dx + dy * dy;
        if (d2 < best_d2) {
            best_d2 = d2;
            best = p;
        }
    }
    if (!corners.empty() && best_d2 <= radius * radius) {
        snapped = (best != click);
        snap_distance = std::sqrt(best_d2);
        return best;
    }

    cv::Mat gx, gy, mag;
    cv::Scharr(patch, gx, CV_32F, 1, 0);
    cv::Scharr(patch, gy, CV_32F, 0, 1);
    cv::magnitude(gx, gy, mag);
    double max_val = 0.0;
    cv::Point max_loc;
    cv::minMaxLoc(mag, nullptr, &max_val, nullptr, &max_loc);
    if (max_val >= 30.0) {
        cv::Point p(roi.x + max_loc.x, roi.y + max_loc.y);
        const double dx = static_cast<double>(p.x - click.x);
        const double dy = static_cast<double>(p.y - click.y);
        if (dx * dx + dy * dy <= radius * radius) {
            snapped = (p != click);
            snap_distance = std::sqrt(dx * dx + dy * dy);
            return p;
        }
    }

    return click;
}

static void resetCurrentManualFeatureSelection(const std::string& reason) {
    g_manual_feature.stage = ManualFeaturePickStage::LIDAR_START;
    g_manual_feature.lidar_candidate_id = -1;
    g_manual_feature.lidar_end_candidate_id = -1;
    g_manual_feature.lidar_start_idx = -1;
    g_manual_feature.lidar_end_idx = -1;
    g_manual_feature.image_candidate_id = -1;
    g_manual_feature.image_start_idx = -1;
    g_manual_feature.image_end_idx = -1;
    g_manual_feature.current_lidar_uv.clear();
    g_manual_feature.current_lidar_pts.clear();
    g_manual_feature.current_image_uv.clear();
    g_manual_feature.pending_pair = ManualFeaturePair();
    g_manual_feature.pending_valid = false;
    if (!reason.empty()) g_manual_feature.message = reason;
}

static void setManualFeatureMode(bool active,
                                 const ClusterDebugConfig& cfg,
                                 ManualFeatureMode mode = ManualFeatureMode::BOUNDARY_SEGMENT) {
    g_manual_feature.enabled_by_config = cfg.enable_manual_feature_pairing;
    g_manual_feature.free_line_enabled_by_config = cfg.enable_manual_free_line_pairing;
    if (active && !cfg.enable_manual_feature_pairing) {
        g_manual_feature.message = "Manual pairing is disabled by config.";
        return;
    }
    if (active && mode == ManualFeatureMode::FREE_3D2D_LINE &&
        !cfg.enable_manual_free_line_pairing) {
        g_manual_feature.message = "G mode is disabled by config.";
        return;
    }
    g_manual_feature.active = active;
    if (active) g_manual_feature.mode = mode;
    g_manual_feature.snap_radius_px = std::max(2.0, cfg.manual_feature_snap_radius_px);
    g_manual_feature.min_segment_points = std::max(2, cfg.manual_feature_min_segment_points);
    g_manual_feature.min_segment_length_px = std::max(2.0, cfg.manual_feature_min_segment_length_px);
    g_manual_feature.image_densify_step_px = std::max(0.5, cfg.manual_feature_image_densify_step_px);
    g_manual_feature.image_corner_snap_enabled = cfg.manual_image_corner_snap_enabled;
    g_manual_feature.image_corner_snap_radius_px = std::max(3, cfg.manual_image_corner_snap_radius_px);
    resetCurrentManualFeatureSelection(active
        ? (mode == ManualFeatureMode::FREE_3D2D_LINE
            ? "G mode: click LiDAR point 1, image point 1, LiDAR point 2, image point 2."
            : "F mode: select local red/blue boundary segments.")
        : "G mode closed. Accepted pairs are kept.");
    if (!active && g_manual_feature.guide_window_created) {
        cv::destroyWindow("Manual Pairing Guide");
        g_manual_feature.guide_window_created = false;
    }
}

static bool nearestPointInPolyline(const std::vector<cv::Point>& pts,
                                   const cv::Point& click,
                                   double max_radius,
                                   int& index_out,
                                   double& distance_out) {
    index_out = -1;
    distance_out = std::numeric_limits<double>::max();
    const double max_d2 = max_radius * max_radius;
    for (int i = 0; i < static_cast<int>(pts.size()); ++i) {
        const double dx = static_cast<double>(pts[i].x - click.x);
        const double dy = static_cast<double>(pts[i].y - click.y);
        const double d2 = dx * dx + dy * dy;
        if (d2 < distance_out * distance_out) {
            distance_out = std::sqrt(d2);
            index_out = i;
        }
    }
    if (index_out < 0 || distance_out * distance_out > max_d2) {
        index_out = -1;
        return false;
    }
    return true;
}

static bool nearestLidarCandidatePoint(const cv::Point& click,
                                       int& candidate_id,
                                       int& point_idx,
                                       double& distance_out) {
    candidate_id = -1;
    point_idx = -1;
    distance_out = std::numeric_limits<double>::max();
    for (int i = 0; i < static_cast<int>(g_manual_feature.lidar_candidates.size()); ++i) {
        int idx = -1;
        double d = 0.0;
        if (!nearestPointInPolyline(g_manual_feature.lidar_candidates[i].uv_pts,
                                    click, g_manual_feature.snap_radius_px, idx, d)) {
            continue;
        }
        if (d < distance_out) {
            distance_out = d;
            candidate_id = i;
            point_idx = idx;
        }
    }
    return candidate_id >= 0;
}

static bool nearestImageCandidatePoint(const cv::Point& click,
                                       int& candidate_id,
                                       int& point_idx,
                                       double& distance_out) {
    candidate_id = -1;
    point_idx = -1;
    distance_out = std::numeric_limits<double>::max();
    for (int i = 0; i < static_cast<int>(g_manual_feature.image_candidates.size()); ++i) {
        int idx = -1;
        double d = 0.0;
        if (!nearestPointInPolyline(g_manual_feature.image_candidates[i],
                                    click, g_manual_feature.snap_radius_px, idx, d)) {
            continue;
        }
        if (d < distance_out) {
            distance_out = d;
            candidate_id = i;
            point_idx = idx;
        }
    }
    return candidate_id >= 0;
}

template <typename T>
static std::vector<T> extractPolylineSegment(const std::vector<T>& pts,
                                             int start_idx,
                                             int end_idx,
                                             bool closed) {
    std::vector<T> out;
    const int n = static_cast<int>(pts.size());
    if (n == 0 || start_idx < 0 || end_idx < 0 || start_idx >= n || end_idx >= n) return out;
    if (!closed) {
        if (start_idx > end_idx) std::swap(start_idx, end_idx);
        out.insert(out.end(), pts.begin() + start_idx, pts.begin() + end_idx + 1);
        return out;
    }

    std::vector<T> forward;
    for (int i = start_idx;; i = (i + 1) % n) {
        forward.push_back(pts[i]);
        if (i == end_idx) break;
    }
    std::vector<T> backward;
    for (int i = start_idx;; i = (i - 1 + n) % n) {
        backward.push_back(pts[i]);
        if (i == end_idx) break;
    }
    return (forward.size() <= backward.size()) ? forward : backward;
}

static std::vector<cv::Point> extractLocalImagePolylineSegment(
    const std::vector<cv::Point>& pts,
    int start_idx,
    int end_idx,
    double densify_step_px) {
    if (pts.size() < 2) return {};
    // Canny/findContours 可能产生闭合轮廓，也可能产生开放折线。
    // 若首尾在空间上相邻，则按闭合轮廓取两次点击之间较短的一侧；
    // 否则按开放折线只取索引区间，绝不返回整条候选线。
    const double close_threshold = std::max(3.0, 2.5 * std::max(0.5, densify_step_px));
    const bool likely_closed = cv::norm(pts.front() - pts.back()) <= close_threshold;
    return extractPolylineSegment(pts, start_idx, end_idx, likely_closed);
}

static void orientImageSegmentLikeLidar(const std::vector<cv::Point>& lidar_uv,
                                        std::vector<cv::Point>& image_uv) {
    if (lidar_uv.size() < 2 || image_uv.size() < 2) return;
    const double same = cv::norm(lidar_uv.front() - image_uv.front()) +
                        cv::norm(lidar_uv.back() - image_uv.back());
    const double reversed = cv::norm(lidar_uv.front() - image_uv.back()) +
                            cv::norm(lidar_uv.back() - image_uv.front());
    if (reversed < same) std::reverse(image_uv.begin(), image_uv.end());
}

static double manualSegmentAngleDiffDeg(const std::vector<cv::Point>& a,
                                        const std::vector<cv::Point>& b) {
    if (a.size() < 2 || b.size() < 2) return 180.0;
    Eigen::Vector2d da(a.back().x - a.front().x, a.back().y - a.front().y);
    Eigen::Vector2d db(b.back().x - b.front().x, b.back().y - b.front().y);
    if (da.norm() < 1e-6 || db.norm() < 1e-6) return 180.0;
    da.normalize();
    db.normalize();
    double dot = std::abs(da.dot(db));
    dot = std::max(0.0, std::min(1.0, dot));
    return std::acos(dot) * 180.0 / M_PI;
}

static void buildPendingManualPair() {
    if (g_manual_feature.current_lidar_uv.size() < 2 ||
        g_manual_feature.current_lidar_pts.size() != g_manual_feature.current_lidar_uv.size() ||
        g_manual_feature.current_image_uv.size() < 2) {
        g_manual_feature.pending_valid = false;
        g_manual_feature.message = "Selected segment is too short. Redo and select again.";
        return;
    }
    if (g_manual_feature.mode != ManualFeatureMode::FREE_3D2D_LINE) {
        orientImageSegmentLikeLidar(g_manual_feature.current_lidar_uv,
                                    g_manual_feature.current_image_uv);
    }
    ManualFeaturePair pair;
    pair.id = static_cast<int>(g_manual_feature.accepted_pairs.size());
    pair.mode = g_manual_feature.mode;
    pair.lidar_boundary_id = g_manual_feature.lidar_candidate_id;
    pair.image_contour_id = g_manual_feature.image_candidate_id;
    pair.lidar_uv_initial = g_manual_feature.current_lidar_uv;
    pair.lidar_pts = g_manual_feature.current_lidar_pts;
    pair.image_uv = g_manual_feature.current_image_uv;
    pair.angle_diff_deg = manualSegmentAngleDiffDeg(pair.lidar_uv_initial, pair.image_uv);
    const double red_len = contourLengthPxLocal(pair.lidar_uv_initial);
    const double blue_len = contourLengthPxLocal(pair.image_uv);
    pair.length_ratio = (std::max(red_len, blue_len) > 1e-9)
        ? std::min(red_len, blue_len) / std::max(red_len, blue_len)
        : 0.0;
    g_manual_feature.pending_pair = std::move(pair);
    g_manual_feature.pending_valid = true;
    g_manual_feature.stage = ManualFeaturePickStage::REVIEW;
    std::ostringstream oss;
    oss << "Pair ready. angle_diff=" << std::fixed << std::setprecision(1)
        << g_manual_feature.pending_pair.angle_diff_deg
        << " deg, red=" << contourLengthPxLocal(g_manual_feature.pending_pair.lidar_uv_initial)
        << " px, image=" << contourLengthPxLocal(g_manual_feature.pending_pair.image_uv)
        << " px, ratio=" << g_manual_feature.pending_pair.length_ratio
        << ". Press A to accept or R to redo.";
    g_manual_feature.message = oss.str();
}

static void handleFree3D2DLineClick(const cv::Point& click) {
    double d = 0.0;
    if (g_manual_feature.stage == ManualFeaturePickStage::LIDAR_START) {
        int cid = -1, idx = -1;
        if (!nearestLidarCandidatePoint(click, cid, idx, d)) {
            g_manual_feature.message = "No RED LiDAR point near click.";
            return;
        }
        const auto& c = g_manual_feature.lidar_candidates[cid];
        if (idx < 0 || idx >= static_cast<int>(c.lidar_pts.size())) return;
        g_manual_feature.lidar_candidate_id = cid;
        g_manual_feature.lidar_start_idx = idx;
        g_manual_feature.current_lidar_uv = {c.uv_pts[idx]};
        g_manual_feature.current_lidar_pts = {c.lidar_pts[idx]};
        g_manual_feature.stage = ManualFeaturePickStage::IMAGE_START;
        g_manual_feature.message = "LiDAR point 1 selected. Click the matching image point 1.";
        return;
    }

    if (g_manual_feature.stage == ManualFeaturePickStage::IMAGE_START) {
        bool snapped = false;
        double snap_distance = 0.0;
        const cv::Point image_pt = snapImageClickToCornerOrGradient(click, snapped, snap_distance);
        g_manual_feature.current_image_uv = {image_pt};
        g_manual_feature.image_candidate_id = -1;
        g_manual_feature.image_start_idx = -1;
        g_manual_feature.stage = ManualFeaturePickStage::LIDAR_END;
        std::ostringstream oss;
        oss << "Image point 1 selected";
        if (snapped) {
            oss << " (auto-snapped " << std::fixed << std::setprecision(1)
                << snap_distance << " px).";
        } else {
            oss << ".";
        }
        oss << " Click RED LiDAR point 2.";
        g_manual_feature.message = oss.str();
        return;
    }

    if (g_manual_feature.stage == ManualFeaturePickStage::LIDAR_END) {
        int cid = -1, idx = -1;
        if (!nearestLidarCandidatePoint(click, cid, idx, d)) {
            g_manual_feature.message = "No RED LiDAR point near click.";
            return;
        }
        const auto& c = g_manual_feature.lidar_candidates[cid];
        if (idx < 0 || idx >= static_cast<int>(c.lidar_pts.size())) return;
        const cv::Point uv1 = c.uv_pts[idx];
        const Eigen::Vector3d p1 = c.lidar_pts[idx];
        const double uv_len = cv::norm(uv1 - g_manual_feature.current_lidar_uv.front());
        const double xyz_len = (p1 - g_manual_feature.current_lidar_pts.front()).norm();
        if (uv_len < g_manual_feature.min_segment_length_px || xyz_len < 0.02) {
            std::ostringstream oss;
            oss << "LiDAR 3D line too short: " << std::fixed << std::setprecision(1)
                << uv_len << " px / " << std::setprecision(3) << xyz_len
                << " m. Click a farther point 2.";
            g_manual_feature.message = oss.str();
            return;
        }
        g_manual_feature.lidar_end_candidate_id = cid;
        g_manual_feature.lidar_end_idx = idx;
        g_manual_feature.current_lidar_uv.push_back(uv1);
        g_manual_feature.current_lidar_pts.push_back(p1);
        g_manual_feature.stage = ManualFeaturePickStage::IMAGE_END;
        g_manual_feature.message = "LiDAR point 2 selected. Click the matching image point 2.";
        return;
    }

    if (g_manual_feature.stage == ManualFeaturePickStage::IMAGE_END) {
        bool snapped = false;
        double snap_distance = 0.0;
        const cv::Point image_pt = snapImageClickToCornerOrGradient(click, snapped, snap_distance);
        const double len = cv::norm(image_pt - g_manual_feature.current_image_uv.front());
        if (len < g_manual_feature.min_segment_length_px) {
            std::ostringstream oss;
            oss << "Image line too short: " << std::fixed << std::setprecision(1)
                << len << " px. Click a farther end point.";
            g_manual_feature.message = oss.str();
            return;
        }
        g_manual_feature.current_image_uv.push_back(image_pt);
        if (snapped) {
            std::ostringstream oss;
            oss << "Image point 2 auto-snapped " << std::fixed << std::setprecision(1)
                << snap_distance << " px. ";
            g_manual_feature.message = oss.str();
        }
        buildPendingManualPair();
        return;
    }

    g_manual_feature.message = "Pair ready. Click Accept or Redo.";
}

static void handleManualFeatureClick(const cv::Point& click) {
    if (!g_manual_feature.active) return;
    if (g_manual_feature.mode == ManualFeatureMode::FREE_3D2D_LINE) {
        handleFree3D2DLineClick(click);
        return;
    }
    double d = 0.0;
    if (g_manual_feature.stage == ManualFeaturePickStage::LIDAR_START) {
        int cid = -1, idx = -1;
        if (!nearestLidarCandidatePoint(click, cid, idx, d)) {
            g_manual_feature.message = "No RED LiDAR boundary near click. Click closer to a red line.";
            return;
        }
        g_manual_feature.lidar_candidate_id = cid;
        g_manual_feature.lidar_start_idx = idx;
        g_manual_feature.stage = ManualFeaturePickStage::LIDAR_END;
        g_manual_feature.message = "LiDAR START selected. Click END on the SAME red boundary.";
        return;
    }

    if (g_manual_feature.stage == ManualFeaturePickStage::LIDAR_END) {
        const int cid = g_manual_feature.lidar_candidate_id;
        if (cid < 0 || cid >= static_cast<int>(g_manual_feature.lidar_candidates.size())) {
            resetCurrentManualFeatureSelection("LiDAR candidate changed. Select RED start again.");
            return;
        }
        int idx = -1;
        if (!nearestPointInPolyline(g_manual_feature.lidar_candidates[cid].uv_pts,
                                    click, g_manual_feature.snap_radius_px, idx, d)) {
            g_manual_feature.message = "Click END on the SAME red boundary.";
            return;
        }
        g_manual_feature.lidar_end_candidate_id = cid;
        g_manual_feature.lidar_end_idx = idx;
        const auto& candidate = g_manual_feature.lidar_candidates[cid];
        g_manual_feature.current_lidar_uv = extractPolylineSegment(
            candidate.uv_pts, g_manual_feature.lidar_start_idx, idx, candidate.closed);
        g_manual_feature.current_lidar_pts = extractPolylineSegment(
            candidate.lidar_pts, g_manual_feature.lidar_start_idx, idx, candidate.closed);
        const double red_length_px = contourLengthPxLocal(g_manual_feature.current_lidar_uv);
        if (static_cast<int>(g_manual_feature.current_lidar_uv.size()) <
                g_manual_feature.min_segment_points ||
            red_length_px < g_manual_feature.min_segment_length_px) {
            std::ostringstream oss;
            oss << "RED segment too short: " << std::fixed << std::setprecision(1)
                << red_length_px << " px. Click a farther end point.";
            g_manual_feature.message = oss.str();
            g_manual_feature.current_lidar_uv.clear();
            g_manual_feature.current_lidar_pts.clear();
            return;
        }
        g_manual_feature.stage = ManualFeaturePickStage::IMAGE_START;
        g_manual_feature.message = "RED segment selected. Click matching BLUE segment START.";
        return;
    }

    if (g_manual_feature.stage == ManualFeaturePickStage::IMAGE_START) {
        int cid = -1, idx = -1;
        if (!nearestImageCandidatePoint(click, cid, idx, d)) {
            g_manual_feature.message = "No BLUE image contour near click. Click closer to a blue line.";
            return;
        }
        g_manual_feature.image_candidate_id = cid;
        g_manual_feature.image_start_idx = idx;
        g_manual_feature.stage = ManualFeaturePickStage::IMAGE_END;
        g_manual_feature.message = "Image START selected. Click END on the SAME blue contour.";
        return;
    }

    if (g_manual_feature.stage == ManualFeaturePickStage::IMAGE_END) {
        const int cid = g_manual_feature.image_candidate_id;
        if (cid < 0 || cid >= static_cast<int>(g_manual_feature.image_candidates.size())) {
            g_manual_feature.stage = ManualFeaturePickStage::IMAGE_START;
            g_manual_feature.message = "Image candidates changed. Select BLUE start again.";
            return;
        }
        int idx = -1;
        if (!nearestPointInPolyline(g_manual_feature.image_candidates[cid], click,
                                    g_manual_feature.snap_radius_px, idx, d)) {
            g_manual_feature.message = "Click END on the SAME blue contour.";
            return;
        }
        g_manual_feature.image_end_idx = idx;
        // 图像候选在 buildImageEdgeData() 中按开放折线保存：
        // arcLength(..., false) / approxPolyDP(..., false)。这里不能按闭合轮廓取最短环路，
        // 否则一条很长的直线经过简化后只有 2 个端点时，会被误判为“too short”。
        g_manual_feature.current_image_uv = extractLocalImagePolylineSegment(
            g_manual_feature.image_candidates[cid], g_manual_feature.image_start_idx, idx,
            g_manual_feature.image_densify_step_px);
        const double blue_length_px = contourLengthPxLocal(g_manual_feature.current_image_uv);
        // 蓝线允许只有两个折线端点；长度使用实际像素弧长判断，而不是简化后的顶点数量。
        if (g_manual_feature.current_image_uv.size() < 2 ||
            blue_length_px < g_manual_feature.min_segment_length_px) {
            std::ostringstream oss;
            oss << "BLUE segment too short: " << std::fixed << std::setprecision(1)
                << blue_length_px << " px. Click a farther end point.";
            g_manual_feature.message = oss.str();
            g_manual_feature.current_image_uv.clear();
            return;
        }
        buildPendingManualPair();
        return;
    }

    g_manual_feature.message = "Pair ready. Click Accept or Redo.";
}

static std::vector<cv::Point> densifyOpenPolylineForManualSelection(
    const std::vector<cv::Point>& pts,
    double step_px) {
    std::vector<cv::Point> out;
    if (pts.empty()) return out;
    out.push_back(pts.front());
    const double step = std::max(0.5, step_px);
    for (size_t i = 1; i < pts.size(); ++i) {
        const cv::Point2d a(pts[i - 1].x, pts[i - 1].y);
        const cv::Point2d b(pts[i].x, pts[i].y);
        const double len = cv::norm(b - a);
        const int pieces = std::max(1, static_cast<int>(std::ceil(len / step)));
        for (int k = 1; k <= pieces; ++k) {
            const double t = static_cast<double>(k) / pieces;
            cv::Point p(static_cast<int>(std::round((1.0 - t) * a.x + t * b.x)),
                        static_cast<int>(std::round((1.0 - t) * a.y + t * b.y)));
            if (out.empty() || p != out.back()) out.push_back(p);
        }
    }
    return out;
}

static void updateManualFeatureCandidates(
    const std::vector<LidarBoundaryCandidate>& lidar_candidates,
    const std::vector<std::vector<cv::Point>>& image_candidates,
    const ClusterDebugConfig& cfg) {
    g_manual_feature.enabled_by_config = cfg.enable_manual_feature_pairing;
    g_manual_feature.free_line_enabled_by_config = cfg.enable_manual_free_line_pairing;
    g_manual_feature.snap_radius_px = std::max(2.0, cfg.manual_feature_snap_radius_px);
    g_manual_feature.min_segment_points = std::max(2, cfg.manual_feature_min_segment_points);
    g_manual_feature.min_segment_length_px = std::max(2.0, cfg.manual_feature_min_segment_length_px);
    g_manual_feature.image_densify_step_px = std::max(0.5, cfg.manual_feature_image_densify_step_px);
    g_manual_feature.image_corner_snap_enabled = cfg.manual_image_corner_snap_enabled;
    g_manual_feature.image_corner_snap_radius_px = std::max(3, cfg.manual_image_corner_snap_radius_px);
    g_manual_feature.lidar_candidates = lidar_candidates;
    // F 模式候选线先按约 1 px 加密。这样即使 approxPolyDP 把长直线简化为两个端点，
    // 鼠标点击中间位置仍可落到对应的弧长位置，最终只截取两次点击之间的局部蓝线段。
    g_manual_feature.image_candidates.clear();
    g_manual_feature.image_candidates.reserve(image_candidates.size());
    for (const auto& c : image_candidates) {
        auto dense = densifyOpenPolylineForManualSelection(c, g_manual_feature.image_densify_step_px);
        if (dense.size() >= 2) g_manual_feature.image_candidates.push_back(std::move(dense));
    }
}

static bool projectManualLidarPoint(const Eigen::Vector3d& p_l,
                                    const Eigen::Matrix3d& R_ext,
                                    const Eigen::Vector3d& T_ext,
                                    const Eigen::Matrix3d& K,
                                    cv::Point& uv) {
    Eigen::Vector3d p_c;
    if (use_inverse_extrinsic) p_c = R_ext.transpose() * (p_l - T_ext);
    else p_c = R_ext * p_l + T_ext;
    if (p_c.z() <= 0.2) return false;
    const Eigen::Vector3d q = K * p_c;
    uv.x = static_cast<int>(std::round(q.x() / q.z()));
    uv.y = static_cast<int>(std::round(q.y() / q.z()));
    return std::isfinite(static_cast<double>(uv.x)) && std::isfinite(static_cast<double>(uv.y));
}

static std::vector<cv::Point> projectManualLidarSegment(
    const ManualFeaturePair& pair,
    const Eigen::Matrix3d& R_ext,
    const Eigen::Vector3d& T_ext,
    const Eigen::Matrix3d& K) {
    std::vector<cv::Point> out;
    out.reserve(pair.lidar_pts.size());
    for (const auto& p_l : pair.lidar_pts) {
        cv::Point uv;
        if (projectManualLidarPoint(p_l, R_ext, T_ext, K, uv)) out.push_back(uv);
    }
    return out;
}


struct ManualFeatureValidationStats {
    bool ok = false;
    int total_pairs = 0;
    int valid_pairs = 0;
    int near_pairs = 0;
    int worst_pair_id = -1;
    double mean_px = 0.0;
    double max_px = 0.0;
    double worst_pair_mean_px = 0.0;
    double near_mean_px = 0.0;
    double near_max_px = 0.0;
};

static double manualPairAverageRangeM(const ManualFeaturePair& pair) {
    if (pair.lidar_pts.empty()) return std::numeric_limits<double>::infinity();
    double sum = 0.0;
    for (const auto& p : pair.lidar_pts) sum += p.norm();
    return sum / static_cast<double>(pair.lidar_pts.size());
}

static double manualFeaturePairRangeWeight(const ManualFeaturePair& pair,
                                           const ClusterDebugConfig& cfg) {
    const double near_range = cfg.manual_feature_near_priority_range_m;
    const double near_weight = std::max(1.0, cfg.manual_feature_near_priority_weight);
    if (near_range <= 0.0 || near_weight <= 1.0) return 1.0;
    const double r = manualPairAverageRangeM(pair);
    return (std::isfinite(r) && r <= near_range) ? near_weight : 1.0;
}

static bool projectManualLidarPoint2d(const Eigen::Vector3d& p_l,
                                      const Eigen::Matrix3d& R_ext,
                                      const Eigen::Vector3d& T_ext,
                                      const Eigen::Matrix3d& K,
                                      cv::Point2d& uv) {
    Eigen::Vector3d p_c;
    if (use_inverse_extrinsic) p_c = R_ext.transpose() * (p_l - T_ext);
    else p_c = R_ext * p_l + T_ext;
    if (p_c.z() <= 0.2) return false;
    const Eigen::Vector3d q = K * p_c;
    uv.x = q.x() / q.z();
    uv.y = q.y() / q.z();
    return std::isfinite(uv.x) && std::isfinite(uv.y);
}

static double pointToSegmentDistancePx(const cv::Point2d& p,
                                       const cv::Point2d& a,
                                       const cv::Point2d& b) {
    const cv::Point2d ab = b - a;
    const double len2 = ab.x * ab.x + ab.y * ab.y;
    if (len2 < 1e-9) return cv::norm(p - a);
    double t = ((p.x - a.x) * ab.x + (p.y - a.y) * ab.y) / len2;
    t = std::max(0.0, std::min(1.0, t));
    return cv::norm(p - (a + t * ab));
}

static double pointToLineDistancePx(const cv::Point2d& p,
                                    const cv::Point2d& a,
                                    const cv::Point2d& b) {
    const cv::Point2d ab = b - a;
    const double len = std::sqrt(ab.x * ab.x + ab.y * ab.y);
    if (len < 1e-9) return cv::norm(p - a);
    return std::abs(ab.x * (p.y - a.y) - ab.y * (p.x - a.x)) / len;
}

static double pointToPolylineDistancePx(const cv::Point2d& p,
                                        const std::vector<cv::Point>& polyline) {
    if (polyline.empty()) return 1.0e6;
    if (polyline.size() == 1) return cv::norm(p - cv::Point2d(polyline.front().x, polyline.front().y));
    double best = std::numeric_limits<double>::max();
    for (size_t i = 1; i < polyline.size(); ++i) {
        const cv::Point2d a(polyline[i - 1].x, polyline[i - 1].y);
        const cv::Point2d b(polyline[i].x, polyline[i].y);
        best = std::min(best, pointToSegmentDistancePx(p, a, b));
    }
    return best;
}

static bool evaluateManualPairResidualPx(const ManualFeaturePair& pair,
                                         const Eigen::Matrix3d& R_ext,
                                         const Eigen::Vector3d& T_ext,
                                         const Eigen::Matrix3d& K,
                                         const ClusterDebugConfig& cfg,
                                         double& mean_px,
                                         double& max_px) {
    mean_px = 0.0;
    max_px = 0.0;
    double sum = 0.0;
    int count = 0;
    auto add_distance = [&](double d) {
        if (!std::isfinite(d)) d = 1.0e6;
        sum += d;
        max_px = std::max(max_px, d);
        count++;
    };

    if (pair.mode == ManualFeatureMode::FREE_3D2D_LINE) {
        if (pair.lidar_pts.size() < 2 || pair.image_uv.size() < 2) return false;
        const Eigen::Vector3d p0 = pair.lidar_pts.front();
        const Eigen::Vector3d p1 = pair.lidar_pts.back();
        const cv::Point2d q0(pair.image_uv.front().x, pair.image_uv.front().y);
        const cv::Point2d q1(pair.image_uv.back().x, pair.image_uv.back().y);
        const int sample_count = std::max(2, cfg.manual_free_line_sample_count);
        for (int i = 0; i < sample_count; ++i) {
            const double t = sample_count == 1 ? 0.0 : static_cast<double>(i) / (sample_count - 1);
            const Eigen::Vector3d p = (1.0 - t) * p0 + t * p1;
            cv::Point2d uv;
            if (!projectManualLidarPoint2d(p, R_ext, T_ext, K, uv)) {
                add_distance(1.0e6);
                continue;
            }
            add_distance(pointToLineDistancePx(uv, q0, q1));
        }
        cv::Point2d uv0, uv1;
        if (projectManualLidarPoint2d(p0, R_ext, T_ext, K, uv0)) add_distance(cv::norm(uv0 - q0));
        else add_distance(1.0e6);
        if (projectManualLidarPoint2d(p1, R_ext, T_ext, K, uv1)) add_distance(cv::norm(uv1 - q1));
        else add_distance(1.0e6);
    } else {
        if (pair.lidar_pts.empty() || pair.image_uv.size() < 2) return false;
        const int step = std::max(1, cfg.manual_feature_residual_sample_step);
        for (size_t i = 0; i < pair.lidar_pts.size(); i += static_cast<size_t>(step)) {
            cv::Point2d uv;
            if (!projectManualLidarPoint2d(pair.lidar_pts[i], R_ext, T_ext, K, uv)) {
                add_distance(1.0e6);
                continue;
            }
            add_distance(pointToPolylineDistancePx(uv, pair.image_uv));
        }
    }

    if (count <= 0) return false;
    mean_px = sum / static_cast<double>(count);
    return true;
}

static ManualFeatureValidationStats validateManualFeatureAlignment(
    const std::vector<ManualFeaturePair>& manual_pairs,
    const Eigen::Matrix3d& R_ext,
    const Eigen::Vector3d& T_ext,
    const Eigen::Matrix3d& K,
    const ClusterDebugConfig& cfg) {
    ManualFeatureValidationStats stats;
    stats.total_pairs = static_cast<int>(manual_pairs.size());
    const double mean_threshold = std::max(0.5, cfg.manual_feature_validation_mean_px);
    const double max_threshold = std::max(mean_threshold, cfg.manual_feature_validation_max_px);
    double sum_mean = 0.0;
    double near_sum_mean = 0.0;

    for (const auto& pair : manual_pairs) {
        double pair_mean = 0.0, pair_max = 0.0;
        const bool valid = evaluateManualPairResidualPx(pair, R_ext, T_ext, K, cfg, pair_mean, pair_max);
        if (!valid) {
            pair_mean = pair_max = 1.0e6;
        }
        stats.valid_pairs += valid ? 1 : 0;
        sum_mean += pair_mean;
        stats.max_px = std::max(stats.max_px, pair_max);
        if (pair_mean > stats.worst_pair_mean_px) {
            stats.worst_pair_mean_px = pair_mean;
            stats.worst_pair_id = pair.id;
        }
        const double range_m = manualPairAverageRangeM(pair);
        if (cfg.manual_feature_near_priority_range_m > 0.0 &&
            std::isfinite(range_m) && range_m <= cfg.manual_feature_near_priority_range_m) {
            stats.near_pairs++;
            near_sum_mean += pair_mean;
            stats.near_max_px = std::max(stats.near_max_px, pair_max);
        }
    }

    if (stats.total_pairs > 0) stats.mean_px = sum_mean / static_cast<double>(stats.total_pairs);
    if (stats.near_pairs > 0) stats.near_mean_px = near_sum_mean / static_cast<double>(stats.near_pairs);
    stats.ok = stats.total_pairs > 0 && stats.valid_pairs == stats.total_pairs &&
               stats.mean_px <= mean_threshold && stats.max_px <= max_threshold;
    return stats;
}

static std::string formatManualValidationSummary(const ManualFeatureValidationStats& stats) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1)
        << "mean=" << stats.mean_px << "px max=" << stats.max_px << "px";
    if (stats.worst_pair_id >= 0) {
        oss << " worst=#" << stats.worst_pair_id << "(" << stats.worst_pair_mean_px << "px)";
    }
    if (stats.near_pairs > 0) {
        oss << " near_mean=" << stats.near_mean_px << "px";
    }
    return oss.str();
}

static double manualValidationScoreForSelection(const ManualFeatureValidationStats& stats) {
    if (stats.near_pairs > 0) return stats.near_mean_px + 0.10 * stats.near_max_px;
    return stats.mean_px + 0.10 * stats.max_px;
}

static void drawManualPair(cv::Mat& show,
                           const ManualFeaturePair& pair,
                           const Eigen::Matrix3d& R_ext,
                           const Eigen::Vector3d& T_ext,
                           const Eigen::Matrix3d& K,
                           int thickness,
                           const cv::Scalar& red_color = cv::Scalar(0, 0, 255),
                           const cv::Scalar& blue_color = cv::Scalar(255, 0, 0)) {
    std::vector<cv::Point> red_now = projectManualLidarSegment(pair, R_ext, T_ext, K);
    if (red_now.size() >= 2) cv::polylines(show, red_now, false, red_color, thickness, cv::LINE_AA);
    if (pair.image_uv.size() >= 2) cv::polylines(show, pair.image_uv, false, blue_color, thickness, cv::LINE_AA);
    if (red_now.size() >= 2 && pair.image_uv.size() >= 2) {
        const cv::Scalar link_color(0, 255, 255);
        cv::line(show, red_now.front(), pair.image_uv.front(), link_color, 1, cv::LINE_AA);
        cv::line(show, red_now.back(), pair.image_uv.back(), link_color, 1, cv::LINE_AA);
        cv::circle(show, red_now.front(), 5, red_color, 2, cv::LINE_AA);
        cv::circle(show, red_now.back(), 5, red_color, 2, cv::LINE_AA);
        cv::circle(show, pair.image_uv.front(), 5, blue_color, 2, cv::LINE_AA);
        cv::circle(show, pair.image_uv.back(), 5, blue_color, 2, cv::LINE_AA);
        putTextReadable(show, "1", red_now.front() + cv::Point(6, -6), 0.42, link_color, 1);
        putTextReadable(show, "1", pair.image_uv.front() + cv::Point(6, -6), 0.42, link_color, 1);
        putTextReadable(show, "2", red_now.back() + cv::Point(6, -6), 0.42, link_color, 1);
        putTextReadable(show, "2", pair.image_uv.back() + cv::Point(6, -6), 0.42, link_color, 1);
    }
    if (!red_now.empty() && !pair.image_uv.empty()) {
        cv::Point label = pair.image_uv[pair.image_uv.size() / 2] + cv::Point(5, -5);
        const std::string prefix = pair.mode == ManualFeatureMode::FREE_3D2D_LINE ? "G" : "F";
        putTextReadable(show, prefix + std::to_string(pair.id), label, 0.45,
                        cv::Scalar(255, 255, 255), 1);
    }
}

static void drawManualFeatureOverlay(cv::Mat& show,
                                     const Eigen::Matrix3d& R_ext,
                                     const Eigen::Vector3d& T_ext,
                                     const Eigen::Matrix3d& K,
                                     const ClusterDebugConfig& cfg) {
    if (!g_manual_feature.active && g_manual_feature.accepted_pairs.empty()) return;

    for (const auto& pair : g_manual_feature.accepted_pairs) {
        drawManualPair(show, pair, R_ext, T_ext, K,
                       std::max(3, cfg.matched_boundary_line_thickness),
                       cv::Scalar(0, 0, 255), cv::Scalar(255, 0, 0));
    }
    if (g_manual_feature.pending_valid) {
        drawManualPair(show, g_manual_feature.pending_pair, R_ext, T_ext, K,
                       std::max(4, cfg.matched_boundary_line_thickness + 1),
                       cv::Scalar(0, 128, 255), cv::Scalar(255, 255, 0));
    } else {
        if (g_manual_feature.current_lidar_uv.size() >= 2) {
            cv::polylines(show, g_manual_feature.current_lidar_uv, false,
                          cv::Scalar(0, 128, 255), 4, cv::LINE_AA);
        }
        if (g_manual_feature.current_image_uv.size() >= 2) {
            cv::polylines(show, g_manual_feature.current_image_uv, false,
                          cv::Scalar(255, 255, 0), 4, cv::LINE_AA);
        }
    }

    auto drawSelectedPoint = [&](const std::vector<cv::Point>& pts, int idx, const cv::Scalar& color) {
        if (idx >= 0 && idx < static_cast<int>(pts.size())) {
            cv::circle(show, pts[idx], 6, color, 2, cv::LINE_AA);
        }
    };
    if (g_manual_feature.lidar_candidate_id >= 0 &&
        g_manual_feature.lidar_candidate_id < static_cast<int>(g_manual_feature.lidar_candidates.size())) {
        const auto& pts = g_manual_feature.lidar_candidates[g_manual_feature.lidar_candidate_id].uv_pts;
        drawSelectedPoint(pts, g_manual_feature.lidar_start_idx, cv::Scalar(0, 255, 255));
        if (g_manual_feature.lidar_end_candidate_id == g_manual_feature.lidar_candidate_id) {
            drawSelectedPoint(pts, g_manual_feature.lidar_end_idx, cv::Scalar(0, 255, 255));
        }
    }
    if (g_manual_feature.lidar_end_candidate_id >= 0 &&
        g_manual_feature.lidar_end_candidate_id != g_manual_feature.lidar_candidate_id &&
        g_manual_feature.lidar_end_candidate_id < static_cast<int>(g_manual_feature.lidar_candidates.size())) {
        const auto& pts = g_manual_feature.lidar_candidates[g_manual_feature.lidar_end_candidate_id].uv_pts;
        drawSelectedPoint(pts, g_manual_feature.lidar_end_idx, cv::Scalar(0, 255, 255));
    }
    if (g_manual_feature.mode == ManualFeatureMode::BOUNDARY_SEGMENT &&
        g_manual_feature.image_candidate_id >= 0 &&
        g_manual_feature.image_candidate_id < static_cast<int>(g_manual_feature.image_candidates.size())) {
        const auto& pts = g_manual_feature.image_candidates[g_manual_feature.image_candidate_id];
        drawSelectedPoint(pts, g_manual_feature.image_start_idx, cv::Scalar(255, 255, 0));
        drawSelectedPoint(pts, g_manual_feature.image_end_idx, cv::Scalar(255, 255, 0));
    } else if (g_manual_feature.mode == ManualFeatureMode::FREE_3D2D_LINE) {
        for (const auto& p : g_manual_feature.current_image_uv) {
            cv::circle(show, p, 6, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
        }
    }

    // Step text is drawn in the dedicated bottom bar of the main canvas.
}

static cv::Mat makeManualFeatureGuideImage() {
    cv::Mat guide(500, 760, CV_8UC3, cv::Scalar(28, 28, 28));
    putTextReadable(guide, "Manual LiDAR-Image Pairing", cv::Point(24, 42),
                    0.72, cv::Scalar(0,255,255), 1);
    putTextReadable(guide, manualFeatureModeText(g_manual_feature.mode), cv::Point(24, 78),
                    0.52, g_manual_feature.mode == ManualFeatureMode::FREE_3D2D_LINE
                        ? cv::Scalar(0,180,255) : cv::Scalar(255,180,0), 1);
    putTextReadable(guide, "Current step:", cv::Point(24, 116), 0.48,
                    cv::Scalar(220,220,220), 1);
    putTextReadable(guide, manualFeatureStageText(g_manual_feature.stage, g_manual_feature.mode),
                    cv::Point(180, 116), 0.50, cv::Scalar(0,255,255), 1);

    int y = 158;
    if (g_manual_feature.mode == ManualFeatureMode::BOUNDARY_SEGMENT) {
        putTextReadable(guide, "1. Click RED boundary START and END.", cv::Point(24,y), 0.43, cv::Scalar(220,220,220), 1); y += 34;
        putTextReadable(guide, "2. Click BLUE contour START and END.", cv::Point(24,y), 0.43, cv::Scalar(220,220,220), 1); y += 34;
        putTextReadable(guide, "   Only the local contour between the two clicks is selected.", cv::Point(24,y), 0.40, cv::Scalar(180,220,255), 1); y += 34;
        putTextReadable(guide, "   F mode uses full-image candidates; ROI is not required.", cv::Point(24,y), 0.40, cv::Scalar(180,220,255), 1); y += 42;
    } else {
        putTextReadable(guide, "1. Click RED LiDAR point 1, then its matching IMAGE point 1.", cv::Point(24,y), 0.42, cv::Scalar(220,220,220), 1); y += 34;
        putTextReadable(guide, "2. Click RED LiDAR point 2, then its matching IMAGE point 2.", cv::Point(24,y), 0.42, cv::Scalar(220,220,220), 1); y += 34;
        putTextReadable(guide, "   Image clicks auto-snap to a nearby corner/strong gray edge when found.", cv::Point(24,y), 0.40, cv::Scalar(180,220,255), 1); y += 42;
    }
    putTextReadable(guide, "3. A=accept, R/C=redo, U=undo last, X=clear all.", cv::Point(24,y), 0.42, cv::Scalar(220,220,220), 1); y += 34;
    putTextReadable(guide, "4. Repeat several non-parallel lines, then Optimize/ENTER.", cv::Point(24,y), 0.42, cv::Scalar(220,220,220), 1); y += 42;
    putTextReadable(guide, "F/G manual pairing: CAMERA ROI and LiDAR XYZ ROI are ignored", cv::Point(24,y), 0.43, cv::Scalar(0,255,255), 1); y += 28;
    putTextReadable(guide, "F: extracted-boundary mode    G: free 3D-to-2D line mode", cv::Point(24,y), 0.43, cv::Scalar(0,255,255), 1); y += 32;
    putTextReadable(guide, "ESC: leave mode; accepted groups are kept", cv::Point(24,y), 0.42, cv::Scalar(220,220,220), 1);

    putTextReadable(guide, "Accepted pairs: " + std::to_string(g_manual_feature.accepted_pairs.size()),
                    cv::Point(24, 462), 0.48, cv::Scalar(0,255,0), 1);
    if (!g_manual_feature.message.empty()) {
        putTextReadable(guide, g_manual_feature.message, cv::Point(230, 462),
                        0.38, cv::Scalar(0,200,255), 1);
    }
    return guide;
}

static void showManualFeatureGuide() {
    if (!g_manual_feature.active) return;
    cv::imshow("Manual Pairing Guide", makeManualFeatureGuideImage());
    g_manual_feature.guide_window_created = true;
}

static inline double radToDegLocal(double rad) { return rad * 180.0 / M_PI; }

static Eigen::Matrix3d rotationFromEulerDegreesLocal(double yaw_deg,
                                                     double roll_deg,
                                                     double pitch_deg) {
    return lidarToCameraRotation(degToRad(yaw_deg), degToRad(roll_deg), degToRad(pitch_deg));
}

static Eigen::Vector2d edgeNormalFromBluePolyline(const std::vector<cv::Point>& pts, size_t idx) {
    if (pts.size() < 2) return Eigen::Vector2d(1.0, 0.0);
    size_t i0 = (idx == 0) ? 0 : idx - 1;
    size_t i1 = std::min(pts.size() - 1, idx + 1);
    if (i0 == i1) {
        i0 = 0;
        i1 = pts.size() - 1;
    }
    Eigen::Vector2d tangent(pts[i1].x - pts[i0].x, pts[i1].y - pts[i0].y);
    if (tangent.norm() < 1e-9) tangent = Eigen::Vector2d(1.0, 0.0);
    tangent.normalize();
    Eigen::Vector2d normal(-tangent.y(), tangent.x());
    if (normal.norm() < 1e-9) normal = Eigen::Vector2d(1.0, 0.0);
    normal.normalize();
    return normal;
}

class LinePairNormalCost {
public:
    LinePairNormalCost(const Eigen::Vector3d& pt_lidar,
                       const cv::Point2d& blue_uv,
                       const Eigen::Vector2d& normal,
                       const Eigen::Matrix3d& K,
                       double base_yaw_deg,
                       double base_roll_deg,
                       double base_pitch_deg,
                       const Eigen::Vector3d& base_T,
                       double weight)
        : pt_(pt_lidar), blue_uv_(blue_uv), normal_(normal), K_(K),
          base_yaw_rad_(base_yaw_deg * M_PI / 180.0),
          base_roll_rad_(base_roll_deg * M_PI / 180.0),
          base_pitch_rad_(base_pitch_deg * M_PI / 180.0),
          base_T_(base_T), weight_(weight) {}

    template <typename T>
    bool operator()(const T* const delta, T* residual) const {
        // delta 顺序保持和 YAML/按键一致：[d_yaw, d_roll, d_pitch, dx, dy, dz]
        const T yaw   = T(base_yaw_rad_)   + delta[0];
        const T roll  = T(base_roll_rad_)  + delta[1];
        const T pitch = T(base_pitch_rad_) + delta[2];

        Eigen::AngleAxis<T> ry(yaw,   Eigen::Matrix<T, 3, 1>::UnitY());
        Eigen::AngleAxis<T> rx(pitch, Eigen::Matrix<T, 3, 1>::UnitX());
        Eigen::AngleAxis<T> rz(roll,  Eigen::Matrix<T, 3, 1>::UnitZ());
        Eigen::Matrix<T, 3, 3> R_now = (ry * rx * rz).toRotationMatrix();

        Eigen::Matrix<T, 3, 1> p_l(T(pt_.x()), T(pt_.y()), T(pt_.z()));
        Eigen::Matrix<T, 3, 1> p_c = R_now * p_l + base_T_.cast<T>() +
            Eigen::Matrix<T, 3, 1>(delta[3], delta[4], delta[5]);

        if (p_c.z() < T(0.2)) {
            residual[0] = T(1000.0);
            return true;
        }

        Eigen::Matrix<T, 3, 1> uvw = K_.cast<T>() * p_c;
        T u = uvw.x() / uvw.z();
        T v = uvw.y() / uvw.z();

        // 点到匹配蓝色图像折线局部法线的距离。沿边缘方向不强行约束，避免点到点抖动。
        T du = u - T(blue_uv_.x);
        T dv = v - T(blue_uv_.y);
        residual[0] = T(weight_) * (T(normal_.x()) * du + T(normal_.y()) * dv);
        return true;
    }

private:
    Eigen::Vector3d pt_;
    cv::Point2d blue_uv_;
    Eigen::Vector2d normal_;
    Eigen::Matrix3d K_;
    double base_yaw_rad_;
    double base_roll_rad_;
    double base_pitch_rad_;
    Eigen::Vector3d base_T_;
    double weight_;
};

class DeltaPriorCost {
public:
    DeltaPriorCost(int idx, double sigma) : idx_(idx), sigma_(std::max(1e-9, sigma)) {}
    template <typename T>
    bool operator()(const T* const delta, T* residual) const {
        residual[0] = delta[idx_] / T(sigma_);
        return true;
    }
private:
    int idx_;
    double sigma_;
};


class ManualPoint2DCost {
public:
    ManualPoint2DCost(const Eigen::Vector3d& pt_lidar,
                      const cv::Point2d& target_uv,
                      const Eigen::Matrix3d& K,
                      double base_yaw_deg,
                      double base_roll_deg,
                      double base_pitch_deg,
                      const Eigen::Vector3d& base_T,
                      double weight)
        : pt_(pt_lidar), target_uv_(target_uv), K_(K),
          base_yaw_rad_(base_yaw_deg * M_PI / 180.0),
          base_roll_rad_(base_roll_deg * M_PI / 180.0),
          base_pitch_rad_(base_pitch_deg * M_PI / 180.0),
          base_T_(base_T), weight_(weight) {}

    template <typename T>
    bool operator()(const T* const delta, T* residual) const {
        const T yaw   = T(base_yaw_rad_)   + delta[0];
        const T roll  = T(base_roll_rad_)  + delta[1];
        const T pitch = T(base_pitch_rad_) + delta[2];
        Eigen::AngleAxis<T> ry(yaw,   Eigen::Matrix<T, 3, 1>::UnitY());
        Eigen::AngleAxis<T> rx(pitch, Eigen::Matrix<T, 3, 1>::UnitX());
        Eigen::AngleAxis<T> rz(roll,  Eigen::Matrix<T, 3, 1>::UnitZ());
        Eigen::Matrix<T, 3, 3> R_now = (ry * rx * rz).toRotationMatrix();

        Eigen::Matrix<T, 3, 1> p_l(T(pt_.x()), T(pt_.y()), T(pt_.z()));
        Eigen::Matrix<T, 3, 1> p_c = R_now * p_l + base_T_.cast<T>() +
            Eigen::Matrix<T, 3, 1>(delta[3], delta[4], delta[5]);
        if (p_c.z() < T(0.2)) {
            residual[0] = T(1000.0);
            residual[1] = T(1000.0);
            return true;
        }
        Eigen::Matrix<T, 3, 1> uvw = K_.cast<T>() * p_c;
        const T u = uvw.x() / uvw.z();
        const T v = uvw.y() / uvw.z();
        residual[0] = T(weight_) * (u - T(target_uv_.x));
        residual[1] = T(weight_) * (v - T(target_uv_.y));
        return true;
    }

private:
    Eigen::Vector3d pt_;
    cv::Point2d target_uv_;
    Eigen::Matrix3d K_;
    double base_yaw_rad_;
    double base_roll_rad_;
    double base_pitch_rad_;
    Eigen::Vector3d base_T_;
    double weight_;
};

class ManualImageLineDistanceCost {
public:
    ManualImageLineDistanceCost(const Eigen::Vector3d& pt_lidar,
                                const Eigen::Vector2d& image_normal,
                                double image_c,
                                const Eigen::Matrix3d& K,
                                double base_yaw_deg,
                                double base_roll_deg,
                                double base_pitch_deg,
                                const Eigen::Vector3d& base_T,
                                double weight)
        : pt_(pt_lidar), n_(image_normal), c_(image_c), K_(K),
          base_yaw_rad_(base_yaw_deg * M_PI / 180.0),
          base_roll_rad_(base_roll_deg * M_PI / 180.0),
          base_pitch_rad_(base_pitch_deg * M_PI / 180.0),
          base_T_(base_T), weight_(weight) {}

    template <typename T>
    bool operator()(const T* const delta, T* residual) const {
        const T yaw = T(base_yaw_rad_) + delta[0];
        const T roll = T(base_roll_rad_) + delta[1];
        const T pitch = T(base_pitch_rad_) + delta[2];
        Eigen::AngleAxis<T> ry(yaw, Eigen::Matrix<T,3,1>::UnitY());
        Eigen::AngleAxis<T> rx(pitch, Eigen::Matrix<T,3,1>::UnitX());
        Eigen::AngleAxis<T> rz(roll, Eigen::Matrix<T,3,1>::UnitZ());
        const Eigen::Matrix<T,3,3> R = (ry * rx * rz).toRotationMatrix();
        const Eigen::Matrix<T,3,1> p(T(pt_.x()), T(pt_.y()), T(pt_.z()));
        const Eigen::Matrix<T,3,1> pc = R * p + base_T_.cast<T>() +
            Eigen::Matrix<T,3,1>(delta[3], delta[4], delta[5]);
        if (pc.z() < T(0.2)) { residual[0] = T(1000.0); return true; }
        const Eigen::Matrix<T,3,1> q = K_.cast<T>() * pc;
        const T u = q.x() / q.z();
        const T v = q.y() / q.z();
        residual[0] = T(weight_) * (T(n_.x()) * u + T(n_.y()) * v + T(c_));
        return true;
    }
private:
    Eigen::Vector3d pt_;
    Eigen::Vector2d n_;
    double c_;
    Eigen::Matrix3d K_;
    double base_yaw_rad_, base_roll_rad_, base_pitch_rad_;
    Eigen::Vector3d base_T_;
    double weight_;
};

class ManualProjectedLineDirectionCost {
public:
    ManualProjectedLineDirectionCost(const Eigen::Vector3d& p0,
                                     const Eigen::Vector3d& p1,
                                     const Eigen::Vector2d& target_dir,
                                     const Eigen::Matrix3d& K,
                                     double base_yaw_deg,
                                     double base_roll_deg,
                                     double base_pitch_deg,
                                     const Eigen::Vector3d& base_T,
                                     double weight)
        : p0_(p0), p1_(p1), target_dir_(target_dir), K_(K),
          base_yaw_rad_(base_yaw_deg * M_PI / 180.0),
          base_roll_rad_(base_roll_deg * M_PI / 180.0),
          base_pitch_rad_(base_pitch_deg * M_PI / 180.0),
          base_T_(base_T), weight_(weight) {}

    template <typename T>
    bool operator()(const T* const delta, T* residual) const {
        const T yaw = T(base_yaw_rad_) + delta[0];
        const T roll = T(base_roll_rad_) + delta[1];
        const T pitch = T(base_pitch_rad_) + delta[2];
        Eigen::AngleAxis<T> ry(yaw, Eigen::Matrix<T,3,1>::UnitY());
        Eigen::AngleAxis<T> rx(pitch, Eigen::Matrix<T,3,1>::UnitX());
        Eigen::AngleAxis<T> rz(roll, Eigen::Matrix<T,3,1>::UnitZ());
        const Eigen::Matrix<T,3,3> R = (ry * rx * rz).toRotationMatrix();
        auto project = [&](const Eigen::Vector3d& p, T& u, T& v) {
            const Eigen::Matrix<T,3,1> pp(T(p.x()), T(p.y()), T(p.z()));
            const Eigen::Matrix<T,3,1> pc = R * pp + base_T_.cast<T>() +
                Eigen::Matrix<T,3,1>(delta[3], delta[4], delta[5]);
            const Eigen::Matrix<T,3,1> q = K_.cast<T>() * pc;
            u = q.x() / q.z(); v = q.y() / q.z();
        };
        T u0, v0, u1, v1; project(p0_, u0, v0); project(p1_, u1, v1);
        const T dx = u1 - u0, dy = v1 - v0;
        const T len = ceres::sqrt(dx * dx + dy * dy + T(1e-12));
        const T px = dx / len, py = dy / len;
        residual[0] = T(weight_) * (px * T(target_dir_.y()) - py * T(target_dir_.x()));
        return true;
    }
private:
    Eigen::Vector3d p0_, p1_;
    Eigen::Vector2d target_dir_;
    Eigen::Matrix3d K_;
    double base_yaw_rad_, base_roll_rad_, base_pitch_rad_;
    Eigen::Vector3d base_T_;
    double weight_;
};

class ManualProjectedLineLengthCost {
public:
    ManualProjectedLineLengthCost(const Eigen::Vector3d& p0,
                                  const Eigen::Vector3d& p1,
                                  double target_length_px,
                                  const Eigen::Matrix3d& K,
                                  double base_yaw_deg,
                                  double base_roll_deg,
                                  double base_pitch_deg,
                                  const Eigen::Vector3d& base_T,
                                  double weight)
        : p0_(p0), p1_(p1), target_length_px_(std::max(1.0, target_length_px)), K_(K),
          base_yaw_rad_(base_yaw_deg * M_PI / 180.0),
          base_roll_rad_(base_roll_deg * M_PI / 180.0),
          base_pitch_rad_(base_pitch_deg * M_PI / 180.0),
          base_T_(base_T), weight_(weight) {}

    template <typename T>
    bool operator()(const T* const delta, T* residual) const {
        const T yaw = T(base_yaw_rad_) + delta[0];
        const T roll = T(base_roll_rad_) + delta[1];
        const T pitch = T(base_pitch_rad_) + delta[2];
        Eigen::AngleAxis<T> ry(yaw, Eigen::Matrix<T,3,1>::UnitY());
        Eigen::AngleAxis<T> rx(pitch, Eigen::Matrix<T,3,1>::UnitX());
        Eigen::AngleAxis<T> rz(roll, Eigen::Matrix<T,3,1>::UnitZ());
        const Eigen::Matrix<T,3,3> R = (ry * rx * rz).toRotationMatrix();
        auto project = [&](const Eigen::Vector3d& p, T& u, T& v) {
            const Eigen::Matrix<T,3,1> pp(T(p.x()), T(p.y()), T(p.z()));
            const Eigen::Matrix<T,3,1> pc = R * pp + base_T_.cast<T>() +
                Eigen::Matrix<T,3,1>(delta[3], delta[4], delta[5]);
            const Eigen::Matrix<T,3,1> q = K_.cast<T>() * pc;
            u = q.x() / q.z(); v = q.y() / q.z();
        };
        T u0, v0, u1, v1; project(p0_, u0, v0); project(p1_, u1, v1);
        const T dx = u1 - u0, dy = v1 - v0;
        const T len = ceres::sqrt(dx * dx + dy * dy + T(1e-12));
        residual[0] = T(weight_) * (len - T(target_length_px_)) / T(target_length_px_);
        return true;
    }
private:
    Eigen::Vector3d p0_, p1_;
    double target_length_px_;
    Eigen::Matrix3d K_;
    double base_yaw_rad_, base_roll_rad_, base_pitch_rad_;
    Eigen::Vector3d base_T_;
    double weight_;
};

static std::vector<cv::Point> resamplePolylineByArcLength(
    const std::vector<cv::Point>& pts, int target_count) {
    std::vector<cv::Point> out;
    if (pts.empty() || target_count <= 0) return out;
    if (pts.size() == 1 || target_count == 1) {
        out.assign(std::max(1, target_count), pts.front());
        return out;
    }

    std::vector<double> cumulative(pts.size(), 0.0);
    for (size_t i = 1; i < pts.size(); ++i) {
        cumulative[i] = cumulative[i - 1] + cv::norm(pts[i] - pts[i - 1]);
    }
    const double total = cumulative.back();
    if (total < 1e-6) {
        out.assign(target_count, pts.front());
        return out;
    }

    out.reserve(target_count);
    for (int k = 0; k < target_count; ++k) {
        const double t = (target_count == 1) ? 0.0 :
            total * static_cast<double>(k) / static_cast<double>(target_count - 1);
        auto it = std::lower_bound(cumulative.begin(), cumulative.end(), t);
        if (it == cumulative.begin()) {
            out.push_back(pts.front());
            continue;
        }
        if (it == cumulative.end()) {
            out.push_back(pts.back());
            continue;
        }
        const size_t i1 = static_cast<size_t>(it - cumulative.begin());
        const size_t i0 = i1 - 1;
        const double seg_len = cumulative[i1] - cumulative[i0];
        const double alpha = seg_len > 1e-9 ? (t - cumulative[i0]) / seg_len : 0.0;
        const double x = (1.0 - alpha) * pts[i0].x + alpha * pts[i1].x;
        const double y = (1.0 - alpha) * pts[i0].y + alpha * pts[i1].y;
        out.emplace_back(static_cast<int>(std::round(x)), static_cast<int>(std::round(y)));
    }
    return out;
}

class ManualFeatureIterationCallback : public ceres::IterationCallback {
public:
    ManualFeatureIterationCallback(const cv::Mat& image,
                                   const std::vector<ClusterInfo>& clusters,
                                   const ImageEdgeData& image_edge,
                                   const std::vector<ManualFeaturePair>& pairs,
                                   const Eigen::Matrix3d& K,
                                   const ClusterDebugConfig& cfg,
                                   double* delta,
                                   double base_yaw,
                                   double base_roll,
                                   double base_pitch,
                                   const Eigen::Vector3d& base_T)
        : image_(image), clusters_(clusters), image_edge_(image_edge), pairs_(pairs),
          K_(K), cfg_(cfg), delta_(delta), base_yaw_(base_yaw), base_roll_(base_roll),
          base_pitch_(base_pitch), base_T_(base_T) {}

    ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) override {
        cv::Mat show = image_.clone();
        const double yaw = base_yaw_ + radToDegLocal(delta_[0]);
        const double roll = base_roll_ + radToDegLocal(delta_[1]);
        const double pitch = base_pitch_ + radToDegLocal(delta_[2]);
        const Eigen::Matrix3d R_now = rotationFromEulerDegreesLocal(yaw, roll, pitch);
        const Eigen::Vector3d T_now = base_T_ + Eigen::Vector3d(delta_[3], delta_[4], delta_[5]);

        ClusterDebugConfig vis_cfg = cfg_;
        vis_cfg.use_image_rois = false;
        vis_cfg.image_rois.clear();
        vis_cfg.use_lidar_xyz_roi = false;
        vis_cfg.restrict_projection_to_rois = false;
        vis_cfg.restrict_image_edges_to_rois = false;
        vis_cfg.restrict_image_contours_to_rois = false;
        vis_cfg.draw_nearby_image_edges = false;
        vis_cfg.draw_line_pair_matches = false;
        vis_cfg.roi_preview_thick_candidates = false;
        vis_cfg.cluster_point_color_scale = std::min(0.20, cfg_.cluster_point_color_scale);
        drawClusterProjection(show, clusters_, R_now, T_now, K_, vis_cfg,
                              &image_edge_, nullptr, false, nullptr);
        // F/G 实时优化窗口不显示也不使用任何 ROI。
        for (const auto& pair : pairs_) {
            drawManualPair(show, pair, R_now, T_now, K_,
                           std::max(4, cfg_.matched_boundary_line_thickness));
        }

        char b1[512], b2[512];
        std::snprintf(b1, sizeof(b1),
                      "ManualFeatureOpt iter %d | cost %.3f | groups %zu",
                      summary.iteration, summary.cost, pairs_.size());
        std::snprintf(b2, sizeof(b2),
                      "Euler yaw %.3f roll %.3f pitch %.3f | dxyz %.3f %.3f %.3f",
                      yaw, roll, pitch, delta_[3], delta_[4], delta_[5]);
        drawTransparentTextBand(show, cv::Point(10, 10),
                                cv::Point(std::min(show.cols - 1, 1220), 74), 0.18);
        putTextReadable(show, b1, cv::Point(20, 35), 0.54, cv::Scalar(0,255,255), 1);
        putTextReadable(show, b2, cv::Point(20, 62), 0.50, cv::Scalar(0,255,255), 1);
        if (g_show_optimization_windows) {
            cv::imshow("Manual Feature Optimization - live", show);
            cv::waitKey(20);
        }
        return ceres::SOLVER_CONTINUE;
    }

private:
    cv::Mat image_;
    std::vector<ClusterInfo> clusters_;
    ImageEdgeData image_edge_;
    std::vector<ManualFeaturePair> pairs_;
    Eigen::Matrix3d K_;
    ClusterDebugConfig cfg_;
    double* delta_;
    double base_yaw_, base_roll_, base_pitch_;
    Eigen::Vector3d base_T_;
};

class LinePairIterationCallback : public ceres::IterationCallback {
public:
    LinePairIterationCallback(const cv::Mat& image,
                              const std::vector<ClusterInfo>& clusters,
                              const ImageEdgeData& image_edge,
                              const Eigen::Matrix3d& K,
                              const ClusterDebugConfig& cfg,
                              double* delta,
                              double base_yaw,
                              double base_roll,
                              double base_pitch,
                              const Eigen::Vector3d& base_T)
        : image_(image), clusters_(clusters), image_edge_(image_edge), K_(K), cfg_(cfg), delta_(delta),
          base_yaw_(base_yaw), base_roll_(base_roll), base_pitch_(base_pitch), base_T_(base_T) {}

    ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) override {
        cv::Mat show = image_.clone();
        Eigen::Matrix3d R_now = rotationFromEulerDegreesLocal(base_yaw_ + radToDegLocal(delta_[0]),
                                                              base_roll_ + radToDegLocal(delta_[1]),
                                                              base_pitch_ + radToDegLocal(delta_[2]));
        Eigen::Vector3d T_now = base_T_ + Eigen::Vector3d(delta_[3], delta_[4], delta_[5]);
        std::vector<RedBlueLinePair> pairs;
        ProjectionDrawStats st = drawClusterProjection(show, clusters_, R_now, T_now, K_, cfg_, &image_edge_, &pairs);
        drawImageROIs(show, cfg_);

        char b1[512], b2[512];
        std::snprintf(b1, sizeof(b1),
                      "LinePairOpt iter %d | cost %.2f | pairs %d/%d | residual pairs %zu",
                      summary.iteration, summary.cost, st.accepted_line_pairs, st.raw_line_pairs, pairs.size());
        std::snprintf(b2, sizeof(b2),
                      "Euler now yaw %.3f roll %.3f pitch %.3f | dxyz %.3f %.3f %.3f",
                      base_yaw_ + radToDegLocal(delta_[0]),
                      base_roll_ + radToDegLocal(delta_[1]),
                      base_pitch_ + radToDegLocal(delta_[2]),
                      delta_[3], delta_[4], delta_[5]);
        drawTransparentTextBand(show, cv::Point(10, 10), cv::Point(std::min(show.cols - 1, 1220), 72), 0.16);
        putTextReadable(show, b1, cv::Point(20, 34), 0.54, cv::Scalar(0,255,255), 1);
        putTextReadable(show, b2, cv::Point(20, 60), 0.50, cv::Scalar(0,255,255), 1);
        if (g_show_optimization_windows) {
            cv::imshow("Line Pair Optimization - live", show);
            cv::waitKey(20);
        }
        return ceres::SOLVER_CONTINUE;
    }

private:
    cv::Mat image_;
    std::vector<ClusterInfo> clusters_;
    ImageEdgeData image_edge_;
    Eigen::Matrix3d K_;
    ClusterDebugConfig cfg_;
    double* delta_;
    double base_yaw_, base_roll_, base_pitch_;
    Eigen::Vector3d base_T_;
};

static bool updateOrAppendTomlDirectEuler(const std::string& result_file,
                                          const std::string& pair_name,
                                          const Eigen::Vector3d& T_ext,
                                          double yaw_deg,
                                          double roll_deg,
                                          double pitch_deg,
                                          bool camera_to_lidar = false) {
    std::string section_name;
    std::string desc;
    if (pair_name == "Front" || pair_name == "FRONT") {
        section_name = "SURROUD_FRONT_CAMERA";
        desc = camera_to_lidar
            ? "长焦相机/图像坐标系到主激光雷达外参参数(yaw,roll,pitch,x,y,z)"
            : "长焦相机外参标定(主激光雷达到相机/图像坐标系)参数(yaw,roll,pitch,x,y,z)";
    } else if (pair_name == "Fish" || pair_name == "FISH") {
        section_name = "SURROUD_FISH_CAMERA";
        desc = camera_to_lidar
            ? "鱼眼相机/图像坐标系到主激光雷达外参参数(yaw,roll,pitch,x,y,z)"
            : "鱼眼相机外参标定(主激光雷达到相机/图像坐标系)参数(yaw,roll,pitch,x,y,z)";
    } else if (pair_name == "Left" || pair_name == "LEFT") {
        section_name = "SURROUD_LEFT_CAMERA";
        desc = camera_to_lidar
            ? "左相机/图像坐标系到对应激光雷达外参参数(yaw,roll,pitch,x,y,z)"
            : "左相机外参标定(激光雷达到相机/图像坐标系)参数(yaw,roll,pitch,x,y,z)";
    } else if (pair_name == "Right" || pair_name == "RIGHT") {
        section_name = "SURROUD_RIGHT_CAMERA";
        desc = camera_to_lidar
            ? "右相机/图像坐标系到对应激光雷达外参参数(yaw,roll,pitch,x,y,z)"
            : "右相机外参标定(激光雷达到相机/图像坐标系)参数(yaw,roll,pitch,x,y,z)";
    } else if (pair_name == "Back" || pair_name == "BACK") {
        section_name = "SURROUD_BACK_CAMERA";
        desc = camera_to_lidar
            ? "后相机/图像坐标系到对应激光雷达外参参数(yaw,roll,pitch,x,y,z)"
            : "后相机外参标定(激光雷达到相机/图像坐标系)参数(yaw,roll,pitch,x,y,z)";
    } else {
        section_name = "UNKNOWN_CAMERA_" + pair_name;
        desc = camera_to_lidar
            ? "未知相机/图像坐标系到LiDAR外参参数(yaw,roll,pitch,x,y,z)"
            : "未知LiDAR到相机/图像坐标系外参参数(yaw,roll,pitch,x,y,z)";
    }

    const std::string target_section = "[epu.sensor_calibration_config." + section_name + ".extrinsic_params]";
    char value_buf[256];
    std::snprintf(value_buf, sizeof(value_buf),
                  "value = [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
                  yaw_deg, roll_deg, pitch_deg, T_ext.x(), T_ext.y(), T_ext.z());
    const std::string new_value_str(value_buf);
    const std::string desc_str = "value_des = \"" + desc + "\"";

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
                if (line.find("value_des") != std::string::npos) lines.push_back(desc_str);
                else if (line.find("value") != std::string::npos && line.find("value_des") == std::string::npos) lines.push_back(new_value_str);
                else lines.push_back(line);
            } else {
                lines.push_back(line);
            }
        }
    }

    std::ofstream outfile(result_file, std::ios::trunc);
    if (!outfile.is_open()) {
        std::cerr << "❌ 无法打开输出文件: " << result_file << std::endl;
        return false;
    }
    for (const auto& l : lines) outfile << l << "\n";
    if (!section_exists) {
        if (!lines.empty() && !lines.back().empty()) outfile << "\n";
        outfile << target_section << "\n" << desc_str << "\n" << new_value_str << "\n\n";
    }
    return true;
}

static std::string cameraToLidarResultFilePath(const std::string& result_file) {
    const std::string suffix = ".toml";
    if (result_file.size() >= suffix.size() &&
        result_file.compare(result_file.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return result_file.substr(0, result_file.size() - suffix.size()) + "_camera_to_lidar.toml";
    }
    return result_file + "_camera_to_lidar.toml";
}

static bool saveCalibrationResult(const std::string& config_file,
                                  const std::string& result_file,
                                  const std::string& pair_name,
                                  const Eigen::Vector3d& T_ext,
                                  double yaw_deg,
                                  double roll_deg,
                                  double pitch_deg,
                                  const std::string& tag) {
    printCopyToYamlExtrinsic(tag, yaw_deg, roll_deg, pitch_deg, T_ext);

    const bool yaml_ok = updateYamlExtrinsicInitial(config_file, pair_name,
                                                   yaw_deg, roll_deg, pitch_deg, T_ext);
    const bool toml_ok = updateOrAppendTomlDirectEuler(result_file, pair_name,
                                                       T_ext, yaw_deg, roll_deg, pitch_deg);
    const Eigen::Matrix3d R_lidar_to_camera =
        lidarToCameraRotation(degToRad(yaw_deg), degToRad(roll_deg), degToRad(pitch_deg));
    const Eigen::Matrix3d R_camera_to_lidar = R_lidar_to_camera.transpose();
    const Eigen::Vector3d T_camera_to_lidar = -R_camera_to_lidar * T_ext;
    const Eigen::Vector3d inv_euler = lidarToCameraEulerDegrees(
        R_camera_to_lidar, -yaw_deg, -roll_deg, -pitch_deg);
    const std::string inverse_result_file = cameraToLidarResultFilePath(result_file);
    const bool inverse_toml_ok = updateOrAppendTomlDirectEuler(
        inverse_result_file, pair_name, T_camera_to_lidar,
        inv_euler[0], inv_euler[1], inv_euler[2], true);

    if (yaml_ok && toml_ok && inverse_toml_ok) {
        std::cout << "✅ 已自动保存标定结果:" << std::endl;
        std::cout << "   YAML: " << config_file << std::endl;
        std::cout << "   TOML(LiDAR->Camera): " << result_file << std::endl;
        std::cout << "   TOML(Camera->LiDAR): " << inverse_result_file << std::endl;
        return true;
    }

    std::cerr << "❌ 标定结果保存不完整: yaml_ok=" << yaml_ok
              << ", toml_ok=" << toml_ok
              << ", inverse_toml_ok=" << inverse_toml_ok << std::endl;
    return false;
}


static bool runManualFeatureOptimization(
    const cv::Mat& base_img,
    const std::vector<ClusterInfo>& clusters,
    const ImageEdgeData& image_edge,
    const std::vector<ManualFeaturePair>& manual_pairs,
    const Eigen::Matrix3d& K,
    const ClusterDebugConfig& cfg,
    double& cur_yaw,
    double& cur_roll,
    double& cur_pitch,
    Eigen::Matrix3d& R_ext,
    Eigen::Vector3d& T_ext,
    double& cost_before,
    double& cost_after) {

    cost_before = cost_after = 0.0;
    if (manual_pairs.empty()) {
        std::cout << "⚠️ 没有人工确认的红蓝特征组，不能执行人工特征优化。" << std::endl;
        return false;
    }

    cv::Mat preview = base_img.clone();
    // 防御性处理：人工优化无条件关闭全部 ROI。即便调用者误传了带 ROI 的 cfg，
    // 已确认的人工 3D-2D 特征也不会被 ROI 再过滤。
    ClusterDebugConfig vis_cfg = cfg;
    vis_cfg.use_image_rois = false;
    vis_cfg.image_rois.clear();
    vis_cfg.use_lidar_xyz_roi = false;
    vis_cfg.restrict_projection_to_rois = false;
    vis_cfg.restrict_image_edges_to_rois = false;
    vis_cfg.restrict_image_contours_to_rois = false;
    vis_cfg.draw_nearby_image_edges = false;
    vis_cfg.draw_line_pair_matches = false;
    vis_cfg.cluster_point_color_scale = std::min(0.20, cfg.cluster_point_color_scale);
    drawClusterProjection(preview, clusters, R_ext, T_ext, K, vis_cfg,
                          &image_edge, nullptr, false, nullptr);
    // 人工优化预览不绘制 ROI：优化约束只来自 accepted manual pairs。
    for (const auto& pair : manual_pairs) {
        drawManualPair(preview, pair, R_ext, T_ext, K,
                       std::max(4, cfg.matched_boundary_line_thickness));
    }
    drawTransparentTextBand(preview, cv::Point(10, 10),
                            cv::Point(std::min(preview.cols - 1, 1160), 62), 0.18);
    putTextReadable(preview,
                    "Manual pairs selected: " + std::to_string(manual_pairs.size()) +
                    " | starting Ceres optimization",
                    cv::Point(20, 40), 0.55, cv::Scalar(0,255,255), 1);
    if (g_show_optimization_windows) {
        cv::imshow("Manual Feature Optimization - selected pairs", preview);
        cv::waitKey(250);
    }

    const double base_yaw = cur_yaw;
    const double base_roll = cur_roll;
    const double base_pitch = cur_pitch;
    const Eigen::Vector3d base_T = T_ext;
    double delta[6] = {0, 0, 0, 0, 0, 0};

    ceres::Problem problem;
    int normal_residuals = 0;
    int point_residuals = 0;
    const int sample_step = std::max(1, cfg.manual_feature_residual_sample_step);

    for (size_t group_idx = 0; group_idx < manual_pairs.size(); ++group_idx) {
        const auto& pair = manual_pairs[group_idx];
        if (pair.lidar_pts.size() < 2 || pair.image_uv.size() < 2) continue;
        const double pair_weight = manualFeaturePairRangeWeight(pair, cfg);
        const double pair_range_m = manualPairAverageRangeM(pair);

        if (pair.mode == ManualFeatureMode::FREE_3D2D_LINE) {
            const Eigen::Vector3d p0 = pair.lidar_pts.front();
            const Eigen::Vector3d p1 = pair.lidar_pts.back();
            cv::Point2d q0(pair.image_uv.front().x, pair.image_uv.front().y);
            cv::Point2d q1(pair.image_uv.back().x, pair.image_uv.back().y);
            Eigen::Vector2d image_dir(q1.x - q0.x, q1.y - q0.y);
            const double image_len = image_dir.norm();
            if (image_len < 1e-6 || (p1 - p0).norm() < 1e-6) continue;
            image_dir /= image_len;
            const Eigen::Vector2d image_normal(-image_dir.y(), image_dir.x());
            const double image_c = -(image_normal.x() * q0.x + image_normal.y() * q0.y);

            const int sample_count = std::max(2, cfg.manual_free_line_sample_count);
            const double group_norm = 1.0 / std::sqrt(static_cast<double>(sample_count));
            const double line_weight = std::max(0.001, cfg.manual_free_line_distance_weight) * group_norm * pair_weight;
            for (int i = 0; i < sample_count; ++i) {
                const double t = sample_count == 1 ? 0.0 : static_cast<double>(i) / (sample_count - 1);
                const Eigen::Vector3d p = (1.0 - t) * p0 + t * p1;
                ceres::CostFunction* cost =
                    new ceres::AutoDiffCostFunction<ManualImageLineDistanceCost, 1, 6>(
                        new ManualImageLineDistanceCost(p, image_normal, image_c, K,
                            base_yaw, base_roll, base_pitch, base_T, line_weight));
                problem.AddResidualBlock(cost,
                    new ceres::HuberLoss(std::max(0.1, cfg.manual_feature_huber_loss_px)), delta);
                normal_residuals++;
            }

            if (cfg.manual_free_line_direction_weight > 0.0) {
                ceres::CostFunction* cost =
                    new ceres::AutoDiffCostFunction<ManualProjectedLineDirectionCost, 1, 6>(
                        new ManualProjectedLineDirectionCost(p0, p1, image_dir, K,
                            base_yaw, base_roll, base_pitch, base_T,
                            cfg.manual_free_line_direction_weight * pair_weight));
                problem.AddResidualBlock(cost, nullptr, delta);
                point_residuals++;
            }
            if (cfg.manual_free_line_length_weight > 0.0) {
                ceres::CostFunction* cost =
                    new ceres::AutoDiffCostFunction<ManualProjectedLineLengthCost, 1, 6>(
                        new ManualProjectedLineLengthCost(p0, p1, image_len, K,
                            base_yaw, base_roll, base_pitch, base_T,
                            cfg.manual_free_line_length_weight * pair_weight));
                problem.AddResidualBlock(cost, nullptr, delta);
                point_residuals++;
            }
            if (cfg.manual_free_line_endpoint_weight > 0.0) {
                for (int e = 0; e < 2; ++e) {
                    const Eigen::Vector3d& p = (e == 0 ? p0 : p1);
                    const cv::Point2d& q = (e == 0 ? q0 : q1);
                    ceres::CostFunction* cost =
                        new ceres::AutoDiffCostFunction<ManualPoint2DCost, 2, 6>(
                            new ManualPoint2DCost(p, q, K, base_yaw, base_roll,
                                base_pitch, base_T, cfg.manual_free_line_endpoint_weight * pair_weight));
                    problem.AddResidualBlock(cost,
                        new ceres::HuberLoss(std::max(0.1, cfg.manual_feature_huber_loss_px)), delta);
                    point_residuals++;
                }
            }
            if (g_verbose_log) std::cout << "   free 3D-2D line group " << group_idx
                      << " lidar_length_m=" << (p1 - p0).norm()
                      << " image_length_px=" << image_len
                      << " initial_angle_diff=" << pair.angle_diff_deg
                      << " avg_range_m=" << pair_range_m
                      << " weight=" << pair_weight << std::endl;
            continue;
        }

        const int target_count = static_cast<int>(pair.lidar_pts.size());
        std::vector<cv::Point> image_samples = resamplePolylineByArcLength(pair.image_uv, target_count);
        if (image_samples.size() != pair.lidar_pts.size()) continue;

        const int sampled_count = std::max(1, (target_count + sample_step - 1) / sample_step);
        const double group_norm = 1.0 / std::sqrt(static_cast<double>(sampled_count));
        const double normal_weight = std::max(0.001, cfg.manual_feature_normal_weight) * group_norm * pair_weight;

        for (int i = 0; i < target_count; i += sample_step) {
            const Eigen::Vector2d normal = edgeNormalFromBluePolyline(image_samples, static_cast<size_t>(i));
            ceres::CostFunction* cost =
                new ceres::AutoDiffCostFunction<LinePairNormalCost, 1, 6>(
                    new LinePairNormalCost(pair.lidar_pts[i], image_samples[i], normal,
                                           K, base_yaw, base_roll, base_pitch, base_T,
                                           normal_weight));
            problem.AddResidualBlock(cost,
                                     new ceres::HuberLoss(std::max(0.1, cfg.manual_feature_huber_loss_px)),
                                     delta);
            normal_residuals++;
        }

        // 端点与中心点使用完整二维残差，防止只有法向约束时线段沿切向滑动。
        std::vector<int> anchor_indices = {0, target_count / 2, target_count - 1};
        std::sort(anchor_indices.begin(), anchor_indices.end());
        anchor_indices.erase(std::unique(anchor_indices.begin(), anchor_indices.end()), anchor_indices.end());
        for (int idx : anchor_indices) {
            const bool is_center = (idx == target_count / 2 && idx != 0 && idx != target_count - 1);
            const double anchor_weight = is_center
                ? std::max(0.0, cfg.manual_feature_center_weight) * pair_weight
                : std::max(0.0, cfg.manual_feature_endpoint_weight) * pair_weight;
            if (anchor_weight <= 0.0) continue;
            ceres::CostFunction* cost =
                new ceres::AutoDiffCostFunction<ManualPoint2DCost, 2, 6>(
                    new ManualPoint2DCost(pair.lidar_pts[idx], image_samples[idx], K,
                                          base_yaw, base_roll, base_pitch, base_T,
                                          anchor_weight));
            problem.AddResidualBlock(cost,
                                     new ceres::HuberLoss(std::max(0.1, cfg.manual_feature_huber_loss_px)),
                                     delta);
            point_residuals++;
        }

        if (g_verbose_log) std::cout << "   manual group " << group_idx
                  << " lidar_pts=" << pair.lidar_pts.size()
                  << " image_pts=" << pair.image_uv.size()
                  << " initial_angle_diff=" << pair.angle_diff_deg
                  << " avg_range_m=" << pair_range_m
                  << " weight=" << pair_weight
                  << std::endl;
    }

    if (normal_residuals + point_residuals <= 0) {
        std::cout << "⚠️ 人工特征组无法构建任何有效残差。" << std::endl;
        return false;
    }
    if (manual_pairs.size() == 1) {
        std::cout << "⚠️ 只有 1 组线特征，完整 6DoF 约束不足；仍按要求优化，但结果会更依赖先验。" << std::endl;
    }

    const double sigma_angle = std::max(1e-9,
        cfg.manual_feature_prior_sigma_angle_deg * M_PI / 180.0);
    const double sigma_trans = std::max(1e-9,
        cfg.manual_feature_prior_sigma_translation_m);
    for (int i = 0; i < 3; ++i) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<DeltaPriorCost, 1, 6>(
                new DeltaPriorCost(i, sigma_angle)), nullptr, delta);
    }
    for (int i = 3; i < 6; ++i) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<DeltaPriorCost, 1, 6>(
                new DeltaPriorCost(i, sigma_trans)), nullptr, delta);
    }

    const double max_angle_rad = std::max(0.01, cfg.manual_feature_max_delta_angle_deg) * M_PI / 180.0;
    for (int i = 0; i < 3; ++i) {
        problem.SetParameterLowerBound(delta, i, -max_angle_rad);
        problem.SetParameterUpperBound(delta, i,  max_angle_rad);
    }
    const double max_trans = std::max(0.001, cfg.manual_feature_max_delta_translation_m);
    for (int i = 3; i < 6; ++i) {
        problem.SetParameterLowerBound(delta, i, -max_trans);
        problem.SetParameterUpperBound(delta, i,  max_trans);
    }

    ManualFeatureIterationCallback cb(base_img, clusters, image_edge, manual_pairs, K,
                                      cfg, delta, base_yaw, base_roll, base_pitch, base_T);
    ceres::Solver::Options options;
    options.max_num_iterations = std::max(1, cfg.manual_feature_max_iterations);
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.update_state_every_iteration = g_show_optimization_windows;
    if (g_show_optimization_windows) options.callbacks.push_back(&cb);

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (g_verbose_log) std::cout << summary.BriefReport() << std::endl;

    cost_before = summary.initial_cost;
    cost_after = summary.final_cost;
    cur_yaw = base_yaw + radToDegLocal(delta[0]);
    cur_roll = base_roll + radToDegLocal(delta[1]);
    cur_pitch = base_pitch + radToDegLocal(delta[2]);
    T_ext = base_T + Eigen::Vector3d(delta[3], delta[4], delta[5]);
    R_ext = rotationFromEulerDegreesLocal(cur_yaw, cur_roll, cur_pitch);

    std::cout << "✅ [ManualFeatureOpt] 完成：groups=" << manual_pairs.size()
              << ", normal_residuals=" << normal_residuals
              << ", point_residuals=" << point_residuals
              << ", cost " << cost_before << " -> " << cost_after
              << std::endl;
    printCopyToYamlExtrinsic("[MANUAL_FEATURE_AUTO_COPY_TO_YAML]",
                             cur_yaw, cur_roll, cur_pitch, T_ext);
    return true;
}

static bool runLinePairOptimization(const cv::Mat& base_img,
                                    const std::vector<ClusterInfo>& clusters,
                                    const ImageEdgeData& image_edge,
                                    const Eigen::Matrix3d& K,
                                    const ClusterDebugConfig& cfg,
                                    double& cur_yaw,
                                    double& cur_roll,
                                    double& cur_pitch,
                                    Eigen::Matrix3d& R_ext,
                                    Eigen::Vector3d& T_ext,
                                    double& cost_before,
                                    double& cost_after) {
    cost_before = cost_after = 0.0;
    if (!cfg.enable_line_pair_optimization) {
        std::cout << "ℹ️ enable_line_pair_optimization=0，跳过线对优化。" << std::endl;
        return false;
    }

    // 优化阶段也必须保持“原始 LiDAR 边界”原则：
    // 不能因为 YAML/ROI 开启 restrict_projection，就先裁掉 ROI 外点云再重新生成红边界。
    // ROI 只用于限制蓝色图像轮廓和红蓝线对匹配；红色 LiDAR contour 始终来自完整 cluster 投影边界。
    ClusterDebugConfig opt_cfg = cfg;
    opt_cfg.restrict_projection_to_rois = false;
    opt_cfg.roi_preview_thick_candidates = false;

    cv::Mat preview = base_img.clone();
    std::vector<RedBlueLinePair> pairs;
    ProjectionDrawStats st = drawClusterProjection(preview, clusters, R_ext, T_ext, K, opt_cfg, &image_edge, &pairs);
    drawImageROIs(preview, opt_cfg);
    if (g_show_optimization_windows) {
        cv::imshow("Line Pair Optimization - selected pairs", preview);
        cv::waitKey(200);
    }

    std::cout << "✅ [LinePairOpt] matched pairs=" << pairs.size()
              << " / raw=" << st.raw_line_pairs << std::endl;
    if (pairs.empty()) {
        std::cout << "⚠️ 没有任何红蓝匹配线对，无法构建 Ceres 残差，不能优化。" << std::endl;
        return false;
    }
    if (!cfg.trust_roi_line_pairs &&
        static_cast<int>(pairs.size()) < cfg.min_line_pairs_for_optimization) {
        std::cout << "⚠️ 普通模式匹配线对太少: " << pairs.size() << " < "
                  << cfg.min_line_pairs_for_optimization << "，不执行优化。" << std::endl;
        return false;
    }
    if (cfg.trust_roi_line_pairs &&
        static_cast<int>(pairs.size()) < cfg.min_line_pairs_for_optimization) {
        std::cout << "ℹ️ trusted ROI 模式：匹配线对只有 " << pairs.size()
                  << " 对，仍然强制进入 Ceres 优化。" << std::endl;
    }

    std::sort(pairs.begin(), pairs.end(), [](const RedBlueLinePair& a, const RedBlueLinePair& b) {
        if (std::abs(a.mean_dist - b.mean_dist) > 1e-6) return a.mean_dist < b.mean_dist;
        return a.angle_diff_deg < b.angle_diff_deg;
    });
    const int max_pairs_for_opt = cfg.trust_roi_line_pairs
        ? std::max(cfg.max_line_pairs_for_optimization, std::min(cfg.line_pair_max_pairs, 200))
        : cfg.max_line_pairs_for_optimization;
    if (static_cast<int>(pairs.size()) > max_pairs_for_opt) {
        pairs.resize(max_pairs_for_opt);
    }
    if (cfg.trust_roi_line_pairs) {
        std::cout << "✅ [LinePairOpt] trusted ROI mode: using " << pairs.size()
                  << " line pairs for optimization, max_pairs_for_opt=" << max_pairs_for_opt
                  << "，少量线对也会强制解算"
                  << std::endl;
    }

    const double base_yaw = cur_yaw;
    const double base_roll = cur_roll;
    const double base_pitch = cur_pitch;
    const Eigen::Vector3d base_T = T_ext;
    double delta[6] = {0, 0, 0, 0, 0, 0};

    ceres::Problem problem;
    int residual_count = 0;
    const int sample_step = std::max(1, cfg.line_pair_residual_sample_step);
    for (const auto& pair : pairs) {
        const size_t n = std::min(pair.lidar_pts.size(), pair.blue_pts.size());
        if (n < 2) continue;
        int used_in_pair = 0;
        for (size_t i = 0; i < n; i += sample_step) {
            Eigen::Vector2d normal = edgeNormalFromBluePolyline(pair.blue_pts, i);
            const double pair_norm_weight = 1.0 / std::sqrt(std::max(1.0, static_cast<double>(n / sample_step + 1)));
            const double quality_weight = cfg.trust_roi_line_pairs
                ? std::max(0.01, cfg.trusted_line_pair_weight)
                : 1.0 / std::sqrt(std::max(1.0, pair.mean_dist));
            const double w = pair_norm_weight * quality_weight;
            ceres::CostFunction* cost =
                new ceres::AutoDiffCostFunction<LinePairNormalCost, 1, 6>(
                    new LinePairNormalCost(pair.lidar_pts[i], pair.blue_pts[i], normal,
                                           K, base_yaw, base_roll, base_pitch, base_T, w));
            problem.AddResidualBlock(cost, new ceres::HuberLoss(cfg.optimization_huber_loss_px), delta);
            residual_count++;
            used_in_pair++;
        }
        std::cout << "   use pair mean_d=" << pair.mean_dist
                  << " angle=" << pair.angle_diff_deg
                  << " samples=" << used_in_pair << std::endl;
    }

    const int min_residual_count = cfg.trust_roi_line_pairs
        ? 1
        : 10;
    if (residual_count < min_residual_count) {
        std::cout << "⚠️ 有效残差为 " << residual_count
                  << "，无法构建优化问题，不执行优化。" << std::endl;
        return false;
    }
    if (cfg.trust_roi_line_pairs && residual_count < cfg.trusted_min_residual_count) {
        std::cout << "ℹ️ trusted ROI 模式：有效残差只有 " << residual_count
                  << " 个，小于配置 trusted_min_residual_count="
                  << cfg.trusted_min_residual_count
                  << "，但按调试策略仍然继续优化。" << std::endl;
    }

    const double sigma_angle = std::max(1e-9, cfg.prior_sigma_angle_deg * M_PI / 180.0);
    const double sigma_trans = std::max(1e-9, cfg.prior_sigma_translation_m);
    for (int i = 0; i < 3; ++i) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<DeltaPriorCost, 1, 6>(new DeltaPriorCost(i, sigma_angle)),
            nullptr, delta);
    }
    for (int i = 3; i < 6; ++i) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<DeltaPriorCost, 1, 6>(new DeltaPriorCost(i, sigma_trans)),
            nullptr, delta);
    }

    problem.SetParameterLowerBound(delta, 0, -cfg.max_delta_yaw_deg * M_PI / 180.0);
    problem.SetParameterUpperBound(delta, 0,  cfg.max_delta_yaw_deg * M_PI / 180.0);
    problem.SetParameterLowerBound(delta, 1, -cfg.max_delta_roll_deg * M_PI / 180.0);
    problem.SetParameterUpperBound(delta, 1,  cfg.max_delta_roll_deg * M_PI / 180.0);
    problem.SetParameterLowerBound(delta, 2, -cfg.max_delta_pitch_deg * M_PI / 180.0);
    problem.SetParameterUpperBound(delta, 2,  cfg.max_delta_pitch_deg * M_PI / 180.0);
    for (int i = 3; i < 6; ++i) {
        problem.SetParameterLowerBound(delta, i, -cfg.max_delta_translation_m);
        problem.SetParameterUpperBound(delta, i,  cfg.max_delta_translation_m);
    }

    LinePairIterationCallback cb(base_img, clusters, image_edge, K, opt_cfg, delta,
                                 base_yaw, base_roll, base_pitch, base_T);
    ceres::Solver::Options options;
    options.max_num_iterations = std::max(1, cfg.optimization_max_iterations);
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.update_state_every_iteration = g_show_optimization_windows;
    if (g_show_optimization_windows) options.callbacks.push_back(&cb);

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (g_verbose_log) std::cout << summary.BriefReport() << std::endl;

    cost_before = summary.initial_cost;
    cost_after = summary.final_cost;
    cur_yaw   = base_yaw   + radToDegLocal(delta[0]);
    cur_roll  = base_roll  + radToDegLocal(delta[1]);
    cur_pitch = base_pitch + radToDegLocal(delta[2]);
    T_ext = base_T + Eigen::Vector3d(delta[3], delta[4], delta[5]);
    R_ext = rotationFromEulerDegreesLocal(cur_yaw, cur_roll, cur_pitch);

    std::cout << "✅ [LinePairOpt] 自动优化完成：yaw=" << cur_yaw
              << ", roll=" << cur_roll
              << ", pitch=" << cur_pitch
              << ", T=[" << T_ext.x() << ", " << T_ext.y() << ", " << T_ext.z() << "]"
              << ", cost " << cost_before << " -> " << cost_after << std::endl;
    printCopyToYamlExtrinsic("[AUTO_COPY_TO_YAML]", cur_yaw, cur_roll, cur_pitch, T_ext);
    return true;
}

static cv::Mat makeReviewImage(const cv::Mat& base_img,
                               const std::vector<ClusterInfo>& clusters,
                               const ImageEdgeData& image_edge,
                               const Eigen::Matrix3d& R_ext,
                               const Eigen::Vector3d& T_ext,
                               const Eigen::Matrix3d& K,
                               const ClusterDebugConfig& cfg,
                               const std::string& title,
                               double yaw,
                               double roll,
                               double pitch) {
    cv::Mat show = base_img.clone();
    ClusterDebugConfig review_cfg = cfg;
    review_cfg.restrict_projection_to_rois = false;
    review_cfg.roi_preview_thick_candidates = false;
    ProjectionDrawStats st = drawClusterProjection(show, clusters, R_ext, T_ext, K, review_cfg, &image_edge, nullptr);
    drawImageROIs(show, review_cfg);
    char b1[512], b2[512], b3[512];
    std::snprintf(b1, sizeof(b1), "%s", title.c_str());
    std::snprintf(b2, sizeof(b2), "Yaw %.3f | Roll %.3f | Pitch %.3f", yaw, roll, pitch);
    std::snprintf(b3, sizeof(b3), "Cam-T X %.3f | Y %.3f | Z %.3f | pairs %d/%d",
                  T_ext.x(), T_ext.y(), T_ext.z(), st.accepted_line_pairs, st.raw_line_pairs);
    drawTransparentTextBand(show, cv::Point(10, 10), cv::Point(std::min(show.cols - 1, 1180), 92), 0.16);
    putTextReadable(show, b1, cv::Point(20, 33), 0.56, cv::Scalar(0,255,255), 1);
    putTextReadable(show, b2, cv::Point(20, 59), 0.52, cv::Scalar(0,255,255), 1);
    putTextReadable(show, b3, cv::Point(20, 83), 0.48, cv::Scalar(245,245,245), 1);
    return show;
}


static cv::Mat makeFinalFullPointCloudProjectionImage(
    const cv::Mat& base_img,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    const Eigen::Matrix3d& R_ext,
    const Eigen::Vector3d& T_ext,
    const Eigen::Matrix3d& K,
    const ClusterDebugConfig& cfg,
    const std::string& title,
    double yaw,
    double roll,
    double pitch) {

    cv::Mat show = base_img.clone();
    int projected = 0;
    int considered = 0;
    const int step = std::max(1, cfg.final_projection_point_step);
    const int radius = std::max(1, cfg.final_projection_point_radius);
    const cv::Scalar pt_color = scaleBgrColor(cv::Scalar(0, 255, 0), cfg.final_projection_point_color_scale);

    if (cloud) {
        for (size_t i = 0; i < cloud->points.size(); i += static_cast<size_t>(step)) {
            const auto& pp = cloud->points[i];
            if (!std::isfinite(pp.x) || !std::isfinite(pp.y) || !std::isfinite(pp.z)) continue;
            considered++;
            Eigen::Vector3d p_l(pp.x, pp.y, pp.z);
            Eigen::Vector3d p_c;
            if (use_inverse_extrinsic) {
                p_c = R_ext.transpose() * (p_l - T_ext);
            } else {
                p_c = R_ext * p_l + T_ext;
            }
            if (p_c.z() < cfg.min_camera_depth || p_c.z() > cfg.max_camera_depth) continue;
            Eigen::Vector3d uvw = K * p_c;
            int u = static_cast<int>(std::round(uvw.x() / uvw.z()));
            int v = static_cast<int>(std::round(uvw.y() / uvw.z()));
            if (u < 0 || u >= show.cols || v < 0 || v >= show.rows) continue;
            cv::circle(show, cv::Point(u, v), radius, pt_color, -1, cv::LINE_AA);
            projected++;
        }
    }

    char b1[512], b2[512], b3[512];
    std::snprintf(b1, sizeof(b1), "%s", title.c_str());
    std::snprintf(b2, sizeof(b2), "Yaw %.3f | Roll %.3f | Pitch %.3f | Cam-T %.3f %.3f %.3f",
                  yaw, roll, pitch, T_ext.x(), T_ext.y(), T_ext.z());
    std::snprintf(b3, sizeof(b3), "Full loaded LiDAR projection after ground filtering | projected %d / sampled %d | step=%d",
                  projected, considered, step);
    drawTransparentTextBand(show, cv::Point(10, 10), cv::Point(std::min(show.cols - 1, 1220), 92), 0.16);
    putTextReadable(show, b1, cv::Point(20, 33), 0.56, cv::Scalar(0,255,255), 1);
    putTextReadable(show, b2, cv::Point(20, 59), 0.50, cv::Scalar(245,245,245), 1);
    putTextReadable(show, b3, cv::Point(20, 83), 0.46, cv::Scalar(0,255,255), 1);
    return show;
}



struct CalibAppOptions {
    string config_file;
    string pair_name = "FRONT";
    bool remove_ground = false;
};

struct CalibSensorConfig {
    vector<double> intrinsic_mat;
    vector<double> distort_coeffs;
    vector<double> extrinsic_init;
    vector<cv::Rect> camera_rois;
    vector<double> lidar_roi;
    string img_dir;
    string pcd_dir;
    string result_file;
    ClusterDebugConfig cluster_cfg;
};

struct CalibInputFiles {
    string image_file;
    string final_pcd;
    int static_id = 0;
    int static_pick_idx = 0;
};

inline bool parseCalibAppArgs(int argc, char **argv, CalibAppOptions& options) {
    for (int i = 1; i < argc; ++i) {
        string arg(argv[i]);
        if (arg == "--config" && i + 1 < argc) {
            options.config_file = argv[++i];
        } else if (arg == "--pair" && i + 1 < argc) {
            options.pair_name = argv[++i];
        } else if (arg == "--remove_ground") {
            options.remove_ground = true;
        } else if (arg == "--inverse") {
            use_inverse_extrinsic = true;
        }
    }
    if (options.config_file.empty()) {
        cerr << "Usage: " << argv[0] << " --config <yaml_file> --pair <name> [--remove_ground] [--inverse]" << endl;
        return false;
    }
    return true;
}

inline bool loadCalibSensorConfig(const CalibAppOptions& options, CalibSensorConfig& sensor_cfg) {
    if (!loadSensorConfig(options.config_file, options.pair_name,
                          sensor_cfg.intrinsic_mat,
                          sensor_cfg.distort_coeffs,
                          sensor_cfg.extrinsic_init,
                          sensor_cfg.img_dir,
                          sensor_cfg.pcd_dir,
                          sensor_cfg.result_file,
                          sensor_cfg.camera_rois,
                          sensor_cfg.lidar_roi)) {
        cerr << "❌ 解析 YAML 配置文件失败！" << endl;
        return false;
    }
    sensor_cfg.cluster_cfg = readClusterDebugConfig(options.config_file, options.pair_name);
    return true;
}

inline void initGlobalCameraMatrix(const CalibSensorConfig& sensor_cfg) {
    double fx = sensor_cfg.intrinsic_mat[0], cx = sensor_cfg.intrinsic_mat[2];
    double fy = sensor_cfg.intrinsic_mat[4], cy = sensor_cfg.intrinsic_mat[5];
    inner << fx, 0, cx, 0, fy, cy, 0, 0, 1;
}

inline bool selectCalibInputFiles(const CalibAppOptions& options,
                                  const CalibSensorConfig& sensor_cfg,
                                  CalibInputFiles& input_files) {
    readStaticSelectionConfig(options.config_file, options.pair_name,
                              input_files.static_id, input_files.static_pick_idx);

    input_files.image_file = selectStaticImageFile(sensor_cfg.img_dir,
                                                   input_files.static_id,
                                                   input_files.static_pick_idx);
    string selected_pcd = selectStaticPcdFile(sensor_cfg.pcd_dir,
                                             input_files.static_id,
                                             input_files.static_pick_idx);
    string temp_merged_pcd = "/tmp/merged_10frames.pcd";

    if (input_files.static_id > 0) {
        if (input_files.image_file.empty() || selected_pcd.empty()) {
            cerr << "❌ 加载静止段图像或雷达点云失败！" << endl;
            cerr << "   image_dir=" << sensor_cfg.img_dir << endl;
            cerr << "   pcd_dir=" << sensor_cfg.pcd_dir << endl;
            cerr << "   static_id=" << input_files.static_id
                 << ", static_pick_idx=" << input_files.static_pick_idx << endl;
            return false;
        }
        input_files.final_pcd = selected_pcd;
        cout << "✅ 已选择静止段数据:" << endl;
        cout << "   image_file: " << input_files.image_file << endl;
        cout << "   pcd_file:   " << input_files.final_pcd << endl;
    } else {
        if (input_files.image_file.empty() || !mergeLidarFrames(sensor_cfg.pcd_dir, 1, temp_merged_pcd)) {
            cerr << "❌ 加载图像或雷达点云数据失败！" << endl;
            return false;
        }
        input_files.final_pcd = temp_merged_pcd;
        cout << "✅ 已使用旧逻辑加载数据:" << endl;
        cout << "   image_file: " << input_files.image_file << endl;
        cout << "   pcd_file:   " << input_files.final_pcd << endl;
    }
    return true;
}

inline bool prepareGroundFilteredPcd(const CalibAppOptions& options,
                                     const ClusterDebugConfig& cluster_cfg,
                                     CalibInputFiles& input_files) {
    bool remove_ground = options.remove_ground || cluster_cfg.remove_ground_by_default;
    if (!remove_ground) return true;

    const std::string ground_input_pcd = input_files.final_pcd;
    pcl::PointCloud<pcl::PointXYZI>::Ptr raw_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(ground_input_pcd, *raw_cloud) == -1) {
        cerr << "❌ 读取 PCD 文件以去除地面失败！" << endl;
        cerr << "   ground_input_pcd=" << ground_input_pcd << endl;
        return false;
    }

    std::cout << "✅ [GroundFilter] RANSAC distance threshold = "
              << cluster_cfg.ground_distance_threshold
              << " m（点到拟合地面平面的距离阈值，不是 z 高度）" << std::endl;
    pcl::PointCloud<pcl::PointXYZI>::Ptr no_ground =
        removeGroundPlane(raw_cloud, cluster_cfg.ground_distance_threshold);

    std::ostringstream ng_name;
    ng_name << "/tmp/no_ground_cluster_debug_" << options.pair_name;
    if (input_files.static_id > 0) {
        ng_name << "_static_" << std::setw(3) << std::setfill('0') << input_files.static_id
                << "_mid_" << std::setw(2) << std::setfill('0') << input_files.static_pick_idx;
    }
    ng_name << ".pcd";
    input_files.final_pcd = ng_name.str();

    pcl::io::savePCDFileBinary(input_files.final_pcd, *no_ground);
    cout << "✅ 已滤除地面点，剩余点数: " << no_ground->size() << endl;
    cout << "   ground_input_pcd:  " << ground_input_pcd << endl;
    cout << "   ground_output_pcd: " << input_files.final_pcd << endl;
    return true;
}

inline void applySensorConfigToCalibration(Calibration& calibra,
                                           const CalibSensorConfig& sensor_cfg) {
    calibra.setCalibROIs(sensor_cfg.camera_rois);
    calibra.setLidarROI(sensor_cfg.lidar_roi);

    calibra.fx_ = sensor_cfg.intrinsic_mat[0];
    calibra.cx_ = sensor_cfg.intrinsic_mat[2];
    calibra.fy_ = sensor_cfg.intrinsic_mat[4];
    calibra.cy_ = sensor_cfg.intrinsic_mat[5];
    calibra.k1_ = 0.0;
    calibra.k2_ = 0.0;
    calibra.p1_ = 0.0;
    calibra.p2_ = 0.0;
    calibra.k3_ = 0.0;
    calibra.s_  = 0.0;
}


static void initAppParameterPanel(const ClusterDebugConfig& cfg, const std::string& window_name) {
    if (g_app_ui.params_window_ready) return;
    g_app_ui.cluster_tol_x100 = std::max(1, std::min(300, static_cast<int>(std::round(cfg.cluster_tolerance_m * 100.0))));
    g_app_ui.cluster_min_points = std::max(1, std::min(1000, cfg.cluster_min_points));
    g_app_ui.ransac_x1000 = std::max(1, std::min(1000, static_cast<int>(std::round(cfg.ground_distance_threshold * 1000.0))));
    g_app_ui.canny_low = std::max(0, std::min(500, cfg.canny_low));
    g_app_ui.canny_high = std::max(1, std::min(800, cfg.canny_high));
    g_app_ui.edge_grad = std::max(0, std::min(500, static_cast<int>(std::round(cfg.edge_gradient_min))));
    g_app_ui.boundary_close = std::max(0, std::min(60, cfg.boundary_close_px));

    g_app_ui.params_window_ready = true;
    g_app_ui.params_dirty = false;
}

static bool applyAppParameterPanel(ClusterDebugConfig& cfg) {
    if (!g_app_ui.params_window_ready) return false;
    bool changed = false;
    auto setDouble = [&](double& dst, double value) {
        if (std::abs(dst - value) > 1e-9) { dst = value; changed = true; }
    };
    auto setInt = [&](int& dst, int value) {
        if (dst != value) { dst = value; changed = true; }
    };

    const double cluster_tol = std::max(1, g_app_ui.cluster_tol_x100) / 100.0;
    const int cluster_min = std::max(1, g_app_ui.cluster_min_points);
    const double ransac = std::max(1, g_app_ui.ransac_x1000) / 1000.0;
    const int canny_low = std::max(0, g_app_ui.canny_low);
    const int canny_high = std::max(canny_low + 1, g_app_ui.canny_high);
    const double edge_grad = std::max(0, g_app_ui.edge_grad);
    const int boundary_close = std::max(0, g_app_ui.boundary_close);

    setDouble(cfg.cluster_tolerance_m, cluster_tol);
    setInt(cfg.cluster_min_points, cluster_min);
    setDouble(cfg.ground_distance_threshold, ransac);
    setInt(cfg.canny_low, canny_low);
    setInt(cfg.canny_high, canny_high);
    setDouble(cfg.edge_gradient_min, edge_grad);
    setInt(cfg.boundary_close_px, boundary_close);
    return changed;
}

static void writeRuntimeParameterText(const std::string& config_file,
                                      const ClusterDebugConfig& cfg) {
    std::ofstream out(config_file + ".runtime_params.txt", std::ios::trunc);
    if (!out.is_open()) return;
    out << "# Runtime parameter snapshot. Trackbar changes update this file.\n";
    out << "cluster_tolerance_m=" << cfg.cluster_tolerance_m << "\n";
    out << "cluster_min_points=" << cfg.cluster_min_points << "\n";
    out << "ground_distance_threshold=" << cfg.ground_distance_threshold << "\n";
    out << "canny_low=" << cfg.canny_low << "\n";
    out << "canny_high=" << cfg.canny_high << "\n";
    out << "edge_gradient_min=" << cfg.edge_gradient_min << "\n";
    out << "boundary_close_px=" << cfg.boundary_close_px << "\n";
    out << "manual_feature_validation_mean_px=" << cfg.manual_feature_validation_mean_px << "\n";
    out << "manual_feature_validation_max_px=" << cfg.manual_feature_validation_max_px << "\n";
    out << "manual_feature_validation_max_rounds=" << cfg.manual_feature_validation_max_rounds << "\n";
    out << "manual_feature_near_priority_range_m=" << cfg.manual_feature_near_priority_range_m << "\n";
    out << "manual_feature_near_priority_weight=" << cfg.manual_feature_near_priority_weight << "\n";
    out << "manual_image_corner_snap_enabled=" << cfg.manual_image_corner_snap_enabled << "\n";
    out << "manual_image_corner_snap_radius_px=" << cfg.manual_image_corner_snap_radius_px << "\n";
}

static void drawAppButton(cv::Mat& panel,
                          const cv::Rect& rect,
                          const std::string& label,
                          const cv::Scalar& color,
                          bool active = true) {
    cv::Scalar fill = active ? color : cv::Scalar(66, 66, 66);
    cv::rectangle(panel, rect, fill, -1, cv::LINE_AA);
    cv::rectangle(panel, rect, active ? cv::Scalar(225, 225, 225) : cv::Scalar(125, 125, 125),
                  1, cv::LINE_AA);
    const double scale = label.size() > 12 ? 0.40 : 0.44;
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, scale, 1, &baseline);
    cv::Point org(rect.x + std::max(6, (rect.width - text_size.width) / 2),
                  rect.y + rect.height / 2 + text_size.height / 2 - 2);
    cv::putText(panel, label, org, cv::FONT_HERSHEY_SIMPLEX, scale,
                active ? cv::Scalar(255, 255, 255) : cv::Scalar(165, 165, 165), 1, cv::LINE_AA);
}

static cv::Mat composeAppCanvas(const cv::Mat& image,
                                const std::string& pair_name,
                                const ClusterDebugConfig& cfg,
                                double yaw,
                                double roll,
                                double pitch,
                                const Eigen::Vector3d& T_ext,
                                int accepted_pairs,
                                bool g_mode_active,
                                const std::string& workflow_status) {
    const int left_w = g_app_ui.left_panel_width;
    const int panel_w = g_app_ui.panel_width;
    const int step_h = 76;
    const int canvas_h = image.rows + step_h;
    cv::Mat canvas(canvas_h, left_w + image.cols + panel_w, image.type(), cv::Scalar(35, 35, 35));
    cv::Mat left_panel = canvas(cv::Rect(0, 0, left_w, canvas_h));
    left_panel.setTo(cv::Scalar(34, 36, 38));
    image.copyTo(canvas(cv::Rect(left_w, 0, image.cols, image.rows)));
    cv::Mat panel = canvas(cv::Rect(left_w + image.cols, 0, panel_w, canvas_h));
    panel.setTo(cv::Scalar(38, 40, 42));

    g_app_ui.image_width = image.cols;
    g_app_ui.image_height = image.rows;
    g_app_ui.image_offset_x = left_w;
    g_app_ui.image_offset_y = 0;
    g_app_ui.buttons.clear();
    g_app_ui.sliders.clear();

    cv::Mat step_panel = canvas(cv::Rect(left_w, image.rows, image.cols, step_h));
    step_panel.setTo(g_mode_active ? cv::Scalar(34, 45, 48) : cv::Scalar(32, 34, 36));
    cv::line(canvas, cv::Point(left_w, image.rows), cv::Point(left_w + image.cols, image.rows),
             cv::Scalar(85, 95, 100), 1, cv::LINE_AA);
    std::string step_title = g_mode_active
        ? manualFeatureStageText(g_manual_feature.stage, g_manual_feature.mode)
        : "Enter G to add LiDAR-image line pairs";
    std::string step_detail = g_mode_active
        ? g_manual_feature.message
        : workflow_status;
    if (step_detail.size() > 118) step_detail = step_detail.substr(0, 117) + ".";
    std::string order_hint = g_mode_active
        ? "Order: LiDAR point 1 -> image point 1 -> LiDAR point 2 -> image point 2. A=Accept, R=Redo."
        : "Adjust pose with keyboard or enter G mode for manual line constraints.";
    putTextReadable(canvas, step_title, cv::Point(left_w + 18, image.rows + 25),
                    0.54, g_mode_active ? cv::Scalar(0, 255, 255) : cv::Scalar(180, 220, 255), 1);
    putTextReadable(canvas, step_detail, cv::Point(left_w + 18, image.rows + 48),
                    0.42, cv::Scalar(245, 245, 245), 1);
    putTextReadable(canvas, order_hint, cv::Point(left_w + 18, image.rows + 68),
                    0.36, cv::Scalar(185, 220, 255), 1);

    auto put = [&](const std::string& s, int x, int y, double scale,
                   const cv::Scalar& c = cv::Scalar(235,235,235), int thick = 1) {
        cv::putText(panel, s, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, scale, c, thick, cv::LINE_AA);
    };
    auto line = [&](int y, const cv::Scalar& c = cv::Scalar(82, 86, 90)) {
        cv::line(panel, cv::Point(14, y), cv::Point(panel_w - 14, y), c, 1, cv::LINE_AA);
    };

    auto left_put = [&](const std::string& s, int x, int y, double scale,
                        const cv::Scalar& c = cv::Scalar(235,235,235), int thick = 1) {
        cv::putText(left_panel, s, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, scale, c, thick, cv::LINE_AA);
    };
    auto draw_slider = [&](const std::string& label, AppParamId id, int value,
                           int min_v, int max_v, const std::string& display, int& sy) {
        left_put(label, 14, sy, 0.39, cv::Scalar(0, 255, 255));
        left_put(display, 148, sy, 0.34, cv::Scalar(230, 230, 230));
        sy += 15;
        cv::Rect track(18, sy, left_w - 36, 6);
        cv::rectangle(left_panel, track, cv::Scalar(76, 80, 84), -1, cv::LINE_AA);
        const double t = (max_v > min_v)
            ? (static_cast<double>(value - min_v) / static_cast<double>(max_v - min_v))
            : 0.0;
        const int knob_x = track.x + static_cast<int>(std::round(std::max(0.0, std::min(1.0, t)) * track.width));
        cv::rectangle(left_panel, cv::Rect(track.x, track.y, std::max(1, knob_x - track.x), track.height),
                      cv::Scalar(45, 155, 190), -1, cv::LINE_AA);
        cv::circle(left_panel, cv::Point(knob_x, track.y + track.height / 2), 7,
                   cv::Scalar(235, 235, 235), -1, cv::LINE_AA);
        g_app_ui.sliders.push_back({track, id, min_v, max_v});
        sy += 30;
    };

    int sy = 28;
    left_put("Live Parameters", 14, sy, 0.52, cv::Scalar(0, 255, 255));
    sy += 28;
    char pbuf[128];
    std::snprintf(pbuf, sizeof(pbuf), "%.2fm", cfg.cluster_tolerance_m);
    draw_slider("Cluster Tol", AppParamId::CLUSTER_TOL, g_app_ui.cluster_tol_x100, 1, 300, pbuf, sy);
    std::snprintf(pbuf, sizeof(pbuf), "%d", cfg.cluster_min_points);
    draw_slider("Min Points", AppParamId::CLUSTER_MIN_POINTS, g_app_ui.cluster_min_points, 1, 1000, pbuf, sy);
    std::snprintf(pbuf, sizeof(pbuf), "%.3fm", cfg.ground_distance_threshold);
    draw_slider("RANSAC", AppParamId::RANSAC, g_app_ui.ransac_x1000, 1, 1000, pbuf, sy);
    std::snprintf(pbuf, sizeof(pbuf), "%d", cfg.canny_low);
    draw_slider("Canny Low", AppParamId::CANNY_LOW, g_app_ui.canny_low, 0, 500, pbuf, sy);
    std::snprintf(pbuf, sizeof(pbuf), "%d", cfg.canny_high);
    draw_slider("Canny High", AppParamId::CANNY_HIGH, g_app_ui.canny_high, 1, 800, pbuf, sy);
    std::snprintf(pbuf, sizeof(pbuf), "%.0f", cfg.edge_gradient_min);
    draw_slider("Edge Grad", AppParamId::EDGE_GRAD, g_app_ui.edge_grad, 0, 500, pbuf, sy);
    std::snprintf(pbuf, sizeof(pbuf), "%dpx", cfg.boundary_close_px);
    draw_slider("Boundary Close", AppParamId::BOUNDARY_CLOSE, g_app_ui.boundary_close, 0, 60, pbuf, sy);
    left_put("Drag sliders to update", 14, std::min(canvas_h - 50, sy + 10), 0.33, cv::Scalar(185, 220, 255));
    left_put("Runtime params saved", 14, std::min(canvas_h - 28, sy + 28), 0.32, cv::Scalar(160, 190, 215));

    int y = 26;
    put("LiDAR-Camera Calibration", 16, y, 0.55, cv::Scalar(0, 255, 255));
    y += 24;
    put("Pair: " + pair_name, 16, y, 0.42, cv::Scalar(180, 220, 255));
    y += 21;
    put(g_mode_active ? "G Mode: add line pairs" : "Manual Adjust Mode", 16, y, 0.43,
        g_mode_active ? cv::Scalar(0, 190, 255) : cv::Scalar(0, 255, 0));
    y += 10;
    line(y); y += 20;

    put("Current Extrinsic", 16, y, 0.46, cv::Scalar(0, 255, 255));
    y += 18;
    cv::Rect ext_box(14, y, panel_w - 28, 92);
    cv::rectangle(panel, ext_box, cv::Scalar(30, 32, 34), -1, cv::LINE_AA);
    cv::rectangle(panel, ext_box, cv::Scalar(85, 90, 95), 1, cv::LINE_AA);
    char buf[256];
    std::snprintf(buf, sizeof(buf), "Yaw   %+8.3f deg", yaw);
    put(buf, 26, y + 23, 0.42, cv::Scalar(245, 245, 245));
    std::snprintf(buf, sizeof(buf), "Roll  %+8.3f deg", roll);
    put(buf, 26, y + 47, 0.42, cv::Scalar(245, 245, 245));
    std::snprintf(buf, sizeof(buf), "Pitch %+8.3f deg", pitch);
    put(buf, 26, y + 71, 0.42, cv::Scalar(245, 245, 245));
    std::snprintf(buf, sizeof(buf), "T  X:%+.3f  Y:%+.3f  Z:%+.3f", T_ext.x(), T_ext.y(), T_ext.z());
    put(buf, 26, y + 91, 0.36, cv::Scalar(180, 220, 255));
    y += 108;

    put("Keys", 16, y, 0.43, cv::Scalar(0, 255, 255));
    y += 18;
    put("J/L yaw left/right  U/O roll", 18, y, 0.34, cv::Scalar(210, 210, 210)); y += 17;
    put("A/D X  Q/E Y  W/S Z", 18, y, 0.34, cv::Scalar(210, 210, 210)); y += 20;
    std::snprintf(buf, sizeof(buf), "Accepted G pairs: %d", accepted_pairs);
    put(buf, 18, y, 0.39, cv::Scalar(0, 255, 0)); y += 17;
    if (!workflow_status.empty()) {
        std::string status = workflow_status;
        if (status.size() > 42) status = status.substr(0, 41) + ".";
        put(status, 18, y, 0.32, cv::Scalar(180, 220, 255)); y += 18;
    }
    line(y); y += 12;

    auto add_button = [&](const std::string& label, AppUiCommand cmd,
                          const cv::Scalar& color, int col, bool active = true) {
        const int gap = 10;
        const int x0 = 18 + col * ((panel_w - 36 + gap) / 2);
        const int w = (panel_w - 36 - gap) / 2;
        cv::Rect r(x0, y, w, 30);
        drawAppButton(panel, r, label, color, active);
        g_app_ui.buttons.push_back({r, cmd, label, active});
    };
    auto add_full_button = [&](const std::string& label, AppUiCommand cmd,
                               const cv::Scalar& color, int by, bool active = true) {
        cv::Rect r(18, by, panel_w - 36, 32);
        drawAppButton(panel, r, label, color, active);
        g_app_ui.buttons.push_back({r, cmd, label, active});
    };

    add_button(g_mode_active ? "Exit G" : "Enter G", AppUiCommand::ENTER_G, cv::Scalar(160, 95, 20), 0);
    add_button("More Lines", AppUiCommand::CONTINUE_PICK, cv::Scalar(120, 100, 30), 1);
    y += 38;
    add_button("Accept", AppUiCommand::ACCEPT_PAIR, cv::Scalar(45, 145, 45), 0, g_mode_active);
    add_button("Redo", AppUiCommand::REDO_PAIR, cv::Scalar(80, 110, 175), 1, g_mode_active);
    y += 38;
    add_button("Undo Pair", AppUiCommand::UNDO_PAIR, cv::Scalar(105, 95, 170), 0);
    add_button("Clear Pairs", AppUiCommand::CLEAR_PAIRS, cv::Scalar(80, 80, 140), 1);
    y += 38;
    add_button("Optimize", AppUiCommand::OPTIMIZE_ONLY, cv::Scalar(20, 135, 150), 0);
    add_button("Save Result", AppUiCommand::SAVE_RESULT, cv::Scalar(25, 120, 80), 1);
    y += 44;

    put("Parameters are on the left", 18, y, 0.33, cv::Scalar(180, 220, 255));


    const int exit_y = std::max(8, canvas_h - 44);
    line(exit_y - 10);
    add_full_button("Exit Program", AppUiCommand::EXIT_APP, cv::Scalar(80, 80, 80), exit_y);
    return canvas;
}


inline int runInteractiveCalibrationSession(const CalibAppOptions& options,
                                            const CalibSensorConfig& sensor_cfg,
                                            Calibration& calibra) {
    const string& config_file = options.config_file;
    const string& pair_name = options.pair_name;
    const string& result_file = sensor_cfg.result_file;
    ClusterDebugConfig cluster_cfg = sensor_cfg.cluster_cfg;
    g_manual_feature = ManualFeaturePickState();
    g_manual_feature.enabled_by_config = cluster_cfg.enable_manual_feature_pairing;
    g_manual_feature.free_line_enabled_by_config = cluster_cfg.enable_manual_free_line_pairing;
    g_manual_feature.snap_radius_px = std::max(2.0, cluster_cfg.manual_feature_snap_radius_px);
    g_manual_feature.min_segment_points = std::max(2, cluster_cfg.manual_feature_min_segment_points);
    g_manual_feature.min_segment_length_px = std::max(2.0, cluster_cfg.manual_feature_min_segment_length_px);
    g_manual_feature.image_densify_step_px = std::max(0.5, cluster_cfg.manual_feature_image_densify_step_px);
    g_manual_feature.image_corner_snap_enabled = cluster_cfg.manual_image_corner_snap_enabled;
    g_manual_feature.image_corner_snap_radius_px = std::max(3, cluster_cfg.manual_image_corner_snap_radius_px);
    double cur_yaw = sensor_cfg.extrinsic_init[0];
    double cur_roll = sensor_cfg.extrinsic_init[1];
    double cur_pitch = sensor_cfg.extrinsic_init[2];
    Eigen::Vector3d T_ext(sensor_cfg.extrinsic_init[3],
                          sensor_cfg.extrinsic_init[4],
                          sensor_cfg.extrinsic_init[5]);

    cv::Mat base_img = calibra.image_.clone();
    if (base_img.channels() == 1) cv::cvtColor(base_img, base_img, cv::COLOR_GRAY2BGR);
    cv::cvtColor(base_img, g_manual_feature_gray_image, cv::COLOR_BGR2GRAY);

    writeRuntimeParameterText(config_file, cluster_cfg);

    bool roi_saved_once = false;
    std::vector<cv::Rect> preview_rois_cache;
    ClusterDebugConfig preview_cfg;
    ImageEdgeData preview_image_edge;
    bool preview_valid = false;
    std::string workflow_status = "Optimize does not save. Save Result writes YAML and two TOML files.";
    double undo_yaw = cur_yaw;
    double undo_roll = cur_roll;
    double undo_pitch = cur_pitch;
    Eigen::Vector3d undo_T = T_ext;
    Eigen::Matrix3d undo_R;

    Eigen::Matrix3d R_ext = lidarToCameraRotation(
        degToRad(cur_yaw), degToRad(cur_roll), degToRad(cur_pitch));
    undo_R = R_ext;

    pcl::PointCloud<pcl::PointXYZI>::Ptr source_cloud = calibra.raw_lidar_cloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr active_cloud = source_cloud;
    ClusterDebugConfig manual_no_roi_cfg = makeManualFeatureNoRoiConfig(cluster_cfg);
    ImageEdgeData manual_image_edge;
    ImageEdgeData image_edge;
    ClusterDebugResult cluster_res;
    ClusterDebugResult manual_cluster_res;

    auto rebuildRuntimeData = [&]() {
        if (cluster_cfg.remove_ground_by_default) {
            active_cloud = removeGroundPlane(source_cloud, cluster_cfg.ground_distance_threshold);
            if (g_verbose_log) {
                std::cout << "[Refresh] RANSAC=" << cluster_cfg.ground_distance_threshold
                          << " m, points=" << active_cloud->size() << std::endl;
            }
        } else {
            active_cloud = source_cloud;
        }
        manual_no_roi_cfg = makeManualFeatureNoRoiConfig(cluster_cfg);
        image_edge = buildImageEdgeData(base_img, cluster_cfg);
        manual_image_edge = buildImageEdgeData(base_img, manual_no_roi_cfg);
        cluster_res = buildEuclideanClusters(active_cloud, cluster_cfg);
        manual_cluster_res = buildEuclideanClusters(active_cloud, manual_no_roi_cfg);
        updateManualFeatureCandidates({}, {}, manual_no_roi_cfg);
        preview_valid = false;
        preview_rois_cache.clear();
        writeRuntimeParameterText(config_file, cluster_cfg);
    };

    debugProjectionSamples(
        source_cloud,
        R_ext,
        T_ext,
        inner,
        base_img.cols,
        base_img.rows
    );
    rebuildRuntimeData();

    std::cout << "[G人工选线] G 选点与人工优化已忽略 camera ROI 和 lidar XYZ ROI。"
              << " filtered_pts=" << manual_cluster_res.filtered_points
              << ", clusters=" << manual_cluster_res.clusters.size() << std::endl;

    printCopyToYamlExtrinsic("[INIT_COPY_TO_YAML]", cur_yaw, cur_roll, cur_pitch, T_ext);
    cout << "\n=========================================" << endl;
    cout << "[Calibration App] Manual adjust + G free-line pairing" << endl;
    cout << "Keys: J/L yaw, U/O roll, I/K pitch, A/D/Q/E/W/S translation" << endl;
    cout << "Buttons: Enter G, More Lines, Accept Pair, Optimize, Save Result, Exit Program" << endl;
    cout << "Left sliders: cluster/RANSAC/Canny/edge parameters update live" << endl;
    cout << "=========================================\n" << endl;

    const std::string main_window = "LiDAR Cluster Debug - colored clusters";
    setupMouseRoiToolWindow(main_window,
                            cluster_cfg.enable_mouse_roi_tool,
                            cluster_cfg.enable_manual_feature_pairing);
    initAppParameterPanel(cluster_cfg, main_window);

    while (true) {
        cv::Mat show = base_img.clone();

        // 第一层：始终显示全图细红 LiDAR 边界 + 细蓝图像边界，方便 ROI 和人工特征选择。
        drawAllCannyEdges(show, manual_image_edge, manual_no_roi_cfg);
        std::vector<LidarBoundaryCandidate> manual_lidar_candidates;
        ProjectionDrawStats draw_stats = drawClusterProjection(show, manual_cluster_res.clusters, R_ext, T_ext, inner, manual_no_roi_cfg,
                                                               &manual_image_edge,
                                                               nullptr,
                                                               false,
                                                               &manual_lidar_candidates);
        updateManualFeatureCandidates(manual_lidar_candidates,
                                      manual_image_edge.roi_edge_contours,
                                      manual_no_roi_cfg);

        // 第二层：如果本轮已经拖拽了未保存 ROI，或刚 Shift+S 保存过 ROI，
        // 则只在这些 ROI 内把红/蓝边界加粗显示，确认哪些特征会进入后续优化。
        std::vector<cv::Rect> preview_rois = currentPreviewRois(cluster_cfg, roi_saved_once);
        if (!g_manual_feature.active && !preview_rois.empty()) {
            std::vector<cv::Rect> valid_preview_rois;
            for (const auto& raw_r : preview_rois) {
                cv::Rect r = clampRectToImage(raw_r, base_img.cols, base_img.rows);
                if (r.area() > 0) valid_preview_rois.push_back(r);
            }
            if (!valid_preview_rois.empty()) {
                if (!preview_valid || !sameRectVector(valid_preview_rois, preview_rois_cache)) {
                    preview_cfg = makeRoiFeaturePreviewConfig(cluster_cfg, valid_preview_rois);
                    preview_image_edge = buildImageEdgeData(base_img, preview_cfg);
                    preview_rois_cache = valid_preview_rois;
                    preview_valid = true;
                }
                ProjectionDrawStats roi_stats = drawClusterProjection(show, cluster_res.clusters, R_ext, T_ext, inner, preview_cfg,
                                                                      preview_cfg.draw_nearby_image_edges ? &preview_image_edge : nullptr,
                                                                      nullptr,
                                                                      true);
                drawImageROIs(show, preview_cfg);
                draw_stats.accepted_line_pairs = roi_stats.accepted_line_pairs;
                draw_stats.raw_line_pairs = roi_stats.raw_line_pairs;
                draw_stats.nearby_image_edges = roi_stats.nearby_image_edges;
            }
        }

        // ROI 仅属于旧自动/调试流程。进入 F/G 人工模式后不再绘制黄色 ROI，
        // 避免误以为人工选点或人工优化受到 ROI 限制。
        if (!g_manual_feature.active) {
            drawImageROIs(show, cluster_cfg);
        }

        char text_buf_angle[256];
        char text_buf_trans[256];
        char text_buf_cluster[512];
        char text_buf_camera[256];
        std::snprintf(text_buf_camera, sizeof(text_buf_camera), "Camera: %s", pair_name.c_str());
        std::snprintf(text_buf_angle, sizeof(text_buf_angle),
                      "Yaw: %.2f | Roll: %.2f | Pitch: %.2f", cur_yaw, cur_roll, cur_pitch);
        std::snprintf(text_buf_trans, sizeof(text_buf_trans),
                      "Cam-T X: %.3f | Y: %.3f | Z: %.3f  (Pts: %d | contours: %d | imgPts: %d | pairs: %d/%d)",
                      T_ext[0], T_ext[1], T_ext[2],
                      draw_stats.projected_points, draw_stats.boundary_contours, draw_stats.nearby_image_edges,
                      draw_stats.accepted_line_pairs, draw_stats.raw_line_pairs);
        std::snprintf(text_buf_cluster, sizeof(text_buf_cluster),
                      "Clusters: raw=%d kept=%zu | filtered_pts=%d | tol=%.2fm | min_pts=%d",
                      cluster_res.raw_cluster_count, cluster_res.clusters.size(),
                      cluster_res.filtered_points, cluster_cfg.cluster_tolerance_m,
                      cluster_cfg.cluster_min_points);

        // 不再绘制大块黑色背景，避免遮挡图像细节；使用描边文字保证可读。
        putTextReadable(show, text_buf_camera,
                        cv::Point(std::min(show.cols - 360, 920), 32),
                        0.54, cv::Scalar(255, 130, 80), 1);
        putTextReadable(show, text_buf_angle,
                        cv::Point(18, 32),
                        0.54, cv::Scalar(0, 255, 255), 1);
        putTextReadable(show, text_buf_trans,
                        cv::Point(18, 58),
                        0.48, cv::Scalar(245, 245, 245), 1);
        putTextReadable(show, text_buf_cluster,
                        cv::Point(18, 82),
                        0.44, cv::Scalar(0, 255, 255), 1);

        drawManualFeatureOverlay(show, R_ext, T_ext, inner, cluster_cfg);
        cv::Mat canvas = composeAppCanvas(show, pair_name, cluster_cfg,
                                          cur_yaw, cur_roll, cur_pitch, T_ext,
                                          static_cast<int>(g_manual_feature.accepted_pairs.size()),
                                          g_manual_feature.active,
                                          workflow_status);
        cv::imshow(main_window, canvas);
        int k = cv::waitKey(20);
        AppUiCommand ui_cmd = consumeAppUiCommand();
        if (ui_cmd == AppUiCommand::ENTER_G) k = 'g';
        else if (ui_cmd == AppUiCommand::ACCEPT_PAIR) k = 'a';
        else if (ui_cmd == AppUiCommand::REDO_PAIR) k = 'r';
        else if (ui_cmd == AppUiCommand::UNDO_PAIR) k = 'u';
        else if (ui_cmd == AppUiCommand::CLEAR_PAIRS) k = 'x';
        else if (ui_cmd == AppUiCommand::CONTINUE_PICK) k = 'm';
        else if (ui_cmd == AppUiCommand::OPTIMIZE_ONLY) k = 13;
        else if (ui_cmd == AppUiCommand::SAVE_RESULT) k = 'p';
        else if (ui_cmd == AppUiCommand::EXIT_APP) {
            std::cout << "Exit Program." << std::endl;
            cv::destroyWindow(main_window);
            if (g_manual_feature.guide_window_created) {
                cv::destroyWindow("Manual Pairing Guide");
                g_manual_feature.guide_window_created = false;
            }
            return 0;
        }
        if (k < 0) {
            if (g_app_ui.params_dirty) {
                const bool changed = applyAppParameterPanel(cluster_cfg);
                g_app_ui.params_dirty = false;
                if (changed) rebuildRuntimeData();
            }
            continue;
        }
        if (g_app_ui.params_dirty) {
            const bool changed = applyAppParameterPanel(cluster_cfg);
            g_app_ui.params_dirty = false;
            if (changed) rebuildRuntimeData();
        }

        // 应用版只保留 G 自由 3D-2D 选线模式；F 调试模式不再开放。
        if (k == 'g' || k == 'G') {
            const bool close_same = g_manual_feature.active &&
                g_manual_feature.mode == ManualFeatureMode::FREE_3D2D_LINE;
            setManualFeatureMode(!close_same, cluster_cfg, ManualFeatureMode::FREE_3D2D_LINE);
            if (g_verbose_log) {
                std::cout << (g_manual_feature.active
                    ? "[ManualPair-G] entered G mode."
                    : "[ManualPair] left G mode; accepted pairs kept.") << std::endl;
            }
            continue;
        }

        if (k == 'm' || k == 'M') {
            if (!g_manual_feature.active ||
                g_manual_feature.mode != ManualFeatureMode::FREE_3D2D_LINE) {
                setManualFeatureMode(true, cluster_cfg, ManualFeatureMode::FREE_3D2D_LINE);
            } else if (g_manual_feature.stage == ManualFeaturePickStage::REVIEW) {
                g_manual_feature.message = "Accept or Redo current pair before adding another.";
                continue;
            } else {
                resetCurrentManualFeatureSelection("Continue on current optimized pose. Click RED LiDAR point 1.");
            }
            workflow_status = "Continue picking on current optimized pose.";
            g_manual_feature.message = "Add more G line pairs, then Optimize again.";
            continue;
        }

        if (k == 27) {
            if (g_manual_feature.active) {
                setManualFeatureMode(false, cluster_cfg, g_manual_feature.mode);
                if (g_verbose_log) std::cout << "[ManualPair] left G mode." << std::endl;
                continue;
            }
            std::cout << "用户退出，不保存。" << std::endl;
            cv::destroyWindow(main_window);
            if (g_manual_feature.guide_window_created) {
                cv::destroyWindow("Manual Pairing Guide");
                g_manual_feature.guide_window_created = false;
            }
            return 0;
        }

        if (g_manual_feature.active) {
            if (k == 'a' || k == 'A') {
                if (!g_manual_feature.pending_valid) {
                    g_manual_feature.message = "No pending pair. Complete four clicks first.";
                } else {
                    ManualFeaturePair accepted = g_manual_feature.pending_pair;
                    accepted.id = static_cast<int>(g_manual_feature.accepted_pairs.size());
                    g_manual_feature.accepted_pairs.push_back(std::move(accepted));
                    resetCurrentManualFeatureSelection(
                        g_manual_feature.mode == ManualFeatureMode::FREE_3D2D_LINE
                            ? "Pair accepted. Select next RED LiDAR point 1 or Optimize."
                            : "Pair accepted. Select next pair or Optimize.");
                    if (g_verbose_log) {
                        std::cout << "[ManualPair] accepted pair #"
                                  << g_manual_feature.accepted_pairs.size() - 1 << std::endl;
                    }
                }
                continue;
            }
            if (k == 'r' || k == 'R' || k == 'c' || k == 'C') {
                resetCurrentManualFeatureSelection("Current pair cleared. Click RED LiDAR point 1 again.");
                continue;
            }
            if (k == 'u' || k == 'U') {
                if (!g_manual_feature.accepted_pairs.empty()) {
                    g_manual_feature.accepted_pairs.pop_back();
                    resetCurrentManualFeatureSelection("Last accepted pair removed.");
                } else {
                    g_manual_feature.message = "No accepted pair to undo.";
                }
                continue;
            }
            if (k == 'x' || k == 'X') {
                g_manual_feature.accepted_pairs.clear();
                resetCurrentManualFeatureSelection("All manual pairs cleared.");
                workflow_status = "Pairs cleared. Select new G pairs and optimize again.";
                continue;
            }
            // 人工模式下其余字母键不再触发外参微调或 ROI 操作。
            if (k != 13 && k != 10 && k != 'p' && k != 'P' && k != 'm' && k != 'M') continue;
        }
        if (cluster_cfg.enable_mouse_roi_tool && (k == 'c' || k == 'C')) {
            g_mouse_roi.has_selection = false;
            g_mouse_roi.dragging = false;
            g_mouse_roi.pending_rois.clear();
            preview_valid = false;
            preview_rois_cache.clear();
            if (g_verbose_log) std::cout << "[MouseROI] cleared pending ROIs." << std::endl;
            continue;
        }
        if (cluster_cfg.enable_mouse_roi_tool && k == 'S') {
            if (g_mouse_roi.pending_rois.empty()) {
                std::cout << "⚠️ [MouseROI] 还没有拖拽添加 ROI，不能保存。" << std::endl;
                continue;
            }

            std::vector<cv::Rect> valid_rois;
            valid_rois.reserve(g_mouse_roi.pending_rois.size());
            for (const auto& raw_r : g_mouse_roi.pending_rois) {
                cv::Rect r = clampRectToImage(raw_r, base_img.cols, base_img.rows);
                if (r.area() > 0) valid_rois.push_back(r);
            }
            if (valid_rois.empty()) {
                std::cout << "⚠️ [MouseROI] 当前未保存 ROI 全部无效。" << std::endl;
                continue;
            }

            if (saveMouseRoisToYaml(config_file, pair_name, valid_rois, cluster_cfg.mouse_roi_replace_existing)) {
                cluster_cfg.use_image_rois = true;
                if (cluster_cfg.mouse_roi_replace_existing) cluster_cfg.image_rois.clear();
                for (const auto& r : valid_rois) cluster_cfg.image_rois.push_back(r);

                // ROI 改变后，ROI 内图像轮廓要重新提取，主窗口和优化马上生效。
                image_edge = buildImageEdgeData(base_img, cluster_cfg);
                roi_saved_once = true;
                preview_valid = false;
                preview_rois_cache.clear();

                // 保存后清空 pending，避免下一次 Shift+S 重复追加。保存后的 ROI 会作为黄色 YAML ROI 显示，
                // 同时 ROI 内红/蓝边界继续加粗预览，方便直接按 Enter 优化。
                g_mouse_roi.pending_rois.clear();
                g_mouse_roi.has_selection = false;
                g_mouse_roi.dragging = false;
            }
            continue;
        }
        if (k == 'p' || k == 'P') {
            const bool saved = saveCalibrationResult(config_file, result_file, pair_name, T_ext,
                                                     cur_yaw, cur_roll, cur_pitch,
                                                     "[SAVE_RESULT_COPY_TO_YAML]");
            workflow_status = saved
                ? "Saved current result to YAML, LiDAR->Camera TOML and Camera->LiDAR TOML."
                : "Save failed. Check terminal logs.";
            continue;
        }
        if (k == 13 || k == 10) {
            if (g_manual_feature.accepted_pairs.empty()) {
                workflow_status = "No accepted G pairs. Accept pairs before optimizing.";
                g_manual_feature.message = workflow_status;
                std::cout << "No accepted G pairs. Optimization skipped; result was not saved." << std::endl;
                continue;
            }

            undo_yaw = cur_yaw;
            undo_roll = cur_roll;
            undo_pitch = cur_pitch;
            undo_T = T_ext;
            undo_R = R_ext;

            double best_yaw = cur_yaw;
            double best_roll = cur_roll;
            double best_pitch = cur_pitch;
            Eigen::Vector3d best_T = T_ext;
            Eigen::Matrix3d best_R = R_ext;
            ManualFeatureValidationStats best_stats = validateManualFeatureAlignment(
                g_manual_feature.accepted_pairs, R_ext, T_ext, inner, manual_no_roi_cfg);
            double best_score = manualValidationScoreForSelection(best_stats);

            bool opt_ok = false;
            bool validation_ok = false;
            double last_score = best_score;
            double last_cost_before = 0.0;
            double last_cost_after = 0.0;
            const int max_rounds = std::max(1, manual_no_roi_cfg.manual_feature_validation_max_rounds);
            const double min_improve = std::max(0.0, manual_no_roi_cfg.manual_feature_validation_min_improvement_px);

            for (int round = 1; round <= max_rounds; ++round) {
                double cost_before = 0.0, cost_after = 0.0;
                if (round == 1) {
                    std::cout << "[Optimize] pairs=" << g_manual_feature.accepted_pairs.size()
                              << ", max_rounds=" << max_rounds << std::endl;
                }
                const bool round_ok = runManualFeatureOptimization(
                    base_img, manual_cluster_res.clusters, manual_image_edge,
                    g_manual_feature.accepted_pairs, inner, manual_no_roi_cfg,
                    cur_yaw, cur_roll, cur_pitch, R_ext, T_ext,
                    cost_before, cost_after);
                last_cost_before = cost_before;
                last_cost_after = cost_after;
                if (!round_ok) {
                    std::cout << "Optimization round failed; result was not saved." << std::endl;
                    break;
                }
                opt_ok = true;

                ManualFeatureValidationStats stats = validateManualFeatureAlignment(
                    g_manual_feature.accepted_pairs, R_ext, T_ext, inner, manual_no_roi_cfg);
                const double score = manualValidationScoreForSelection(stats);
                std::cout << "[ManualValidation] round " << round << ": "
                          << formatManualValidationSummary(stats)
                          << " threshold mean<=" << manual_no_roi_cfg.manual_feature_validation_mean_px
                          << " max<=" << manual_no_roi_cfg.manual_feature_validation_max_px
                          << (stats.ok ? " OK" : " NEED_MORE") << std::endl;

                if (score < best_score) {
                    best_score = score;
                    best_stats = stats;
                    best_yaw = cur_yaw;
                    best_roll = cur_roll;
                    best_pitch = cur_pitch;
                    best_T = T_ext;
                    best_R = R_ext;
                }
                validation_ok = stats.ok;
                if (validation_ok) {
                    best_stats = stats;
                    best_yaw = cur_yaw;
                    best_roll = cur_roll;
                    best_pitch = cur_pitch;
                    best_T = T_ext;
                    best_R = R_ext;
                    break;
                }
                if (round > 1 && (last_score - score) < min_improve) {
                    std::cout << "[ManualValidation] improvement below " << min_improve
                              << " px; stop retrying and keep near-priority best result." << std::endl;
                    break;
                }
                last_score = score;
            }

            if (!opt_ok) {
                cur_yaw = undo_yaw;
                cur_roll = undo_roll;
                cur_pitch = undo_pitch;
                T_ext = undo_T;
                R_ext = undo_R;
                workflow_status = "Optimization failed. Current pose restored; result was not saved.";
                g_manual_feature.message = workflow_status;
                std::cout << workflow_status << std::endl;
                continue;
            }

            cur_yaw = best_yaw;
            cur_roll = best_roll;
            cur_pitch = best_pitch;
            T_ext = best_T;
            R_ext = best_R;
            setManualFeatureMode(true, cluster_cfg, ManualFeatureMode::FREE_3D2D_LINE);

            std::ostringstream status;
            status << (validation_ok ? "Optimized OK: " : "Optimized, check: ")
                   << formatManualValidationSummary(best_stats)
                   << ". Add more lines or save.";
            workflow_status = status.str();
            g_manual_feature.message = validation_ok
                ? "Optimization passed validation. Continue picking more lines or Save Result."
                : "Residuals exceed threshold. Add more lines on current pose, then Optimize again.";
            std::cout << "[ManualValidation] selected result: "
                      << formatManualValidationSummary(best_stats)
                      << (validation_ok ? " OK" : " NOT_FULLY_OK") << std::endl;
            printCopyToYamlExtrinsic("[OPTIMIZED_NOT_SAVED_COPY_TO_YAML]",
                                     cur_yaw, cur_roll, cur_pitch, T_ext);
            (void)last_cost_before;
            (void)last_cost_after;
            continue;
        }

        double st = 0.05;
        double sr_deg = 0.2;

        bool angle_changed = true;
        if      (k=='j') cur_yaw   -= sr_deg;
        else if (k=='l') cur_yaw   += sr_deg;
        else if (k=='i') cur_pitch += sr_deg;
        else if (k=='k') cur_pitch -= sr_deg;
        else if (k=='u') cur_roll  -= sr_deg;
        else if (k=='o') cur_roll  += sr_deg;
        else angle_changed = false;

        if (angle_changed) {
            R_ext = lidarToCameraRotation(
                degToRad(cur_yaw), degToRad(cur_roll), degToRad(cur_pitch));
        }

        if (!angle_changed && (k=='a' || k=='d' || k=='w' || k=='s' || k=='q' || k=='e')) {
            if      (k=='a') T_ext.x() -= st;
            else if (k=='d') T_ext.x() += st;
            else if (k=='q') T_ext.y() -= st;
            else if (k=='e') T_ext.y() += st;
            else if (k=='w') T_ext.z() += st;
            else if (k=='s') T_ext.z() -= st;
        }

        if (angle_changed || k=='a' || k=='d' || k=='w' || k=='s' || k=='q' || k=='e') {
            printCopyToYamlExtrinsic("[CURRENT_COPY_TO_YAML]",
                                     cur_yaw, cur_roll, cur_pitch, T_ext);
        }
    }

    cv::destroyWindow(main_window);
    if (g_manual_feature.guide_window_created) {
        cv::destroyWindow("Manual Pairing Guide");
        g_manual_feature.guide_window_created = false;
    }
    std::cout << "ℹ️ 本版本支持人工多组红蓝特征配对优化；无人工组时保留原自动匹配优化。" << std::endl;
    return 0;
}


#endif // CALIB_APP_HELPERS_HPP
