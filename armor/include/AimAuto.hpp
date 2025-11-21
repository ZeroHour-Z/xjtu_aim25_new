#ifndef AIMAUTO
#define AIMAUTO
#include "globalParam.hpp"
#include "opencv2/core/mat.hpp"
#include "tracker.hpp"
#include <camera.hpp>
#include <chrono>
#include <cstdint>
#include <memory>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/core/eigen.hpp>
#include <vector>
#include<filter.hpp>
#include "OpenvinoInfer.hpp"  // 添加这个包含以定义Object类型
#include <memory>  // 添加这个包含以支持 std::unique_ptr

class AimAuto
{
private:
    GlobalParam *gp;
    Tracker *tracker;
    Armor last_armor;
    std::unique_ptr<OpenvinoInfer> inferer;  // 添加推理器作为成员变量
    void pnp_solve(Object &armor, Translator &ts, cv::Mat &src, Armor &tar, int number);
    void draw_armor_back(cv::Mat &src, Armor &armor, int number, cv::Scalar color = cv::Scalar(255, 255, 255));
    Filter vx_filter;
    Filter vy_filter;
    Filter vyaw_filter;
    void optimizeYawZ(
        const std::vector<cv::Point3f>& objPoints,
        const std ::vector<cv::Point2f>& imgPoints,
        double known_x,
        double known_y,
        double known_z,
        double camera_yaw,
        double camera_pitch,
        double &yaw,
        const cv::Mat& K,
        const cv::Mat& dist
    );
public:
    AimAuto(GlobalParam *gp);
    ~AimAuto();
    void auto_aim(GlobalParam &gp, cv::Mat &src, Translator &ts, double dt);
};


#endif // AIMAUTO