#ifndef TRADITIONAL_DETECTION_HPP
#define TRADITIONAL_DETECTION_HPP

#include "WMIdentify.hpp"
#include "globalParam.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

struct DetectionResult {
  std::vector<cv::Point> intersections;
  std::vector<cv::Point> circlePoints;
  cv::Mat processedImage;
  double processingTime;
};

struct KeyPoints {
  std::vector<std::vector<cv::Point>> circleContours;
  std::vector<double> circleAreas;
  std::vector<double> circularities;
  std::vector<cv::Point2f> rectCenters;
  std::vector<cv::Point> circlePoints;
  // 定义面积范围常量
  static constexpr double min_low = 150.0;
  static constexpr double min_high = 1800.0;
  static constexpr double max_low = 2500.0;
  static constexpr double max_high = 15000.0;

  bool isValid() const {
    // 检查基本条件
    if (circleContours.size() != 2 || rectCenters.size() != 1) {
      return false;
    }

    // 检查面积条件
    bool hasSmallCircle = false;
    bool hasLargeCircle = false;

    for (size_t i = 0; i < circleAreas.size(); ++i) {
      double area = circleAreas[i];
      if (area >= min_low && area <= min_high) {
        hasSmallCircle = true;
      } else if (area >= max_low && area <= max_high) {
        hasLargeCircle = true;
      }
    }

    return hasSmallCircle && hasLargeCircle;
  }
};

DetectionResult detect(const cv::Mat &inputImage, WMBlade &blade,
                       GlobalParam &gp, int is_blue, Translator &translator);

KeyPoints detect_key_points(const std::vector<std::vector<cv::Point>> &contours,
                            const std::vector<cv::Vec4i> &hierarchy,
                            cv::Mat &processedImage, WMBlade &blade,
                            GlobalParam &gp);

std::vector<cv::Point>
findIntersectionsByEquation(const cv::Point &center1, const cv::Point &center2,
                            double radius, const cv::RotatedRect &ellipse,
                            cv::Mat &pic, GlobalParam &gp, WMBlade &blade);

#endif // TRADITIONAL_DETECTION_HPP
