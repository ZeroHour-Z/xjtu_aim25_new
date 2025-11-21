/**
 * @file WMIdentify.cpp
 * @author Clarence Stark (3038736583@qq.com)
 * @brief 任意点位打符识别类实现
 * @version 0.1
 * @date 2024-12-08
 *
 * @copyright Copyright (c) 2024
 */

#include "WMIdentify.hpp"
#include "globalParam.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/types.hpp"
#include "traditional_detection.hpp"
#include <algorithm>
#include <chrono>
#include <deque>
#include <iostream>
#include <iterator>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <ostream>
// #define Pi 3.1415926
const double Pi = 3.1415926;

#ifndef ON
#define ON 1
#endif

#ifndef OFF
#define OFF 0
#endif

/**
 * @brief WMIdentify类构造函数
 * @param[in] gp     全局参数
 * @return void
 */
WMIdentify::WMIdentify(GlobalParam &gp) {
  this->gp = &gp;
  // this->gp->list_size = 150;
  this->list_stat = 0;
  this->t_start =
      std::chrono::system_clock::now().time_since_epoch().count() / 1e9;
  // 从gp中读取一些数据
  this->switch_INFO = this->gp->switch_INFO;
  this->switch_ERROR = this->gp->switch_ERROR;
  // this->get_armor_mode = this->gp->get_armor_mode;
  // 将R与Wing的状态设置为没有读取到内容
  this->R_stat = 0;
  this->Wing_stat = 0;
  this->Winghat_stat = 0;
  this->R_estimate.x = 0;
  this->R_estimate.y = 0;
  // this->d_RP2 = 0.7; // ❗️大符半径？
  // this->d_RP1P3 = 0.6;
  // this->d_P1P3 = 0.2;
  this->camera_matrix = (cv::Mat_<double>(3, 3) << this->gp->fx, 0,
                         this->gp->cx, 0, this->gp->fy, this->gp->cy, 0, 0, 1);
  this->dist_coeffs = (cv::Mat_<double>(1, 5) << this->gp->k1, this->gp->k2,
                       this->gp->p1, this->gp->p2, this->gp->k3);
  this->data_img = cv::Mat::zeros(400, 800, CV_8UC3);
  world_points = {cv::Point3f(0, 0, 0), cv::Point3f(this->gp->d_RP2, 0, 0),
                  cv::Point3f(this->gp->d_Radius, this->gp->d_P1P3 / 2, 0),
                  cv::Point3f(this->gp->d_Radius, -this->gp->d_P1P3 / 2, 0)};
  // 输出日志，初始化成功
  // //LOG_IF(INFO, this->switch_INFO) << "WMIdentify Successful";
  // this->starting_time =
  // std::chrono::duration_cast<std::chrono::milliseconds>(
  //     std::chrono::system_clock::now().time_since_epoch());
}

/**
 * @brief WMIdentify类析构函数
 * @return void
 */
WMIdentify::~WMIdentify() {
  // WMIdentify之中的内容都会自动析构
  // std::cout << "析构中，下次再见喵～" << std::endl;
  // 输出日志，析构成功
  // //LOG_IF(INFO, this->switch_INFO) << "~WMIdentify Successful";
}

void drawFixedWorldAxes(cv::Mat &frame, const cv::Mat &cameraMatrix,
                        const cv::Mat &distCoeffs, const cv::Mat &R_init,
                        const cv::Mat &t_init) {
  std::vector<cv::Point3f> axisPoints;
  axisPoints.push_back(cv::Point3f(0, 0, 0));
  axisPoints.push_back(cv::Point3f(1, 0, 0));
  axisPoints.push_back(cv::Point3f(0, 1, 0));
  axisPoints.push_back(cv::Point3f(0, 0, 1));

  cv::Mat rvec, tvec;
  cv::Rodrigues(R_init, rvec);
  tvec = t_init.clone();

  std::vector<cv::Point2f> imagePoints;
  cv::projectPoints(axisPoints, rvec, tvec, cameraMatrix, distCoeffs,
                    imagePoints);

  cv::line(frame, imagePoints[0], imagePoints[1], cv::Scalar(0, 0, 255), 2);
  cv::line(frame, imagePoints[0], imagePoints[2], cv::Scalar(0, 255, 0), 2);
  // cv::line(frame, imagePoints[0], imagePoints[3], cv::Scalar(255, 0, 0),
  //          2);

  cv::putText(frame, "X", imagePoints[1], cv::FONT_HERSHEY_SIMPLEX, 1.0,
              cv::Scalar(0, 0, 255), 2);
  cv::putText(frame, "Y", imagePoints[2], cv::FONT_HERSHEY_SIMPLEX, 1.0,
              cv::Scalar(0, 255, 0), 2);
  // cv::putText(frame, "Z", imagePoints[3], cv::FONT_HERSHEY_SIMPLEX, 1.0,
  //             cv::Scalar(255, 0, 0), 2);
}
void visualizeCameraViewpointApproximation(
    const cv::Mat &image, const cv::Mat &cameraMatrix,
    const cv::Mat &distCoeffs, const cv::Mat &rvec, const cv::Mat &tvec,
    const std::vector<cv::Point3f> &objectPoints) {

  cv::Point2f imageCenter(image.cols / 2.0f, image.rows / 2.0f);

  cv::circle(image, imageCenter, 5, cv::Scalar(255, 255, 255), -1);
  cv::Point2f imageEnd(image.cols / 2.0f, image.rows / 2.0f + 300);
  cv::line(image, imageCenter, imageEnd, cv::Scalar(255, 255, 255));
}

void WMIdentify::visualizeCameraViewpoint(
    const cv::Mat &image, const cv::Mat &cameraMatrix,
    const cv::Mat &distCoeffs, const cv::Mat &rvec, const cv::Mat &tvec,
    const std::vector<cv::Point3f> &objectPoints) {
  cv::Point3f vec1 = objectPoints[1] - objectPoints[0];
  cv::Point3f vec2 = objectPoints[2] - objectPoints[1];
  cv::Point3f normal = vec1.cross(vec2);
  normal /= cv::norm(normal);

  float d = -(normal.x * objectPoints[0].x + normal.y * objectPoints[0].y +
              normal.z * objectPoints[0].z);

  cv::Mat R;
  cv::Rodrigues(rvec, R);

  cv::Mat cameraPosition = -R.t() * tvec;

  cv::Mat zAxis = (cv::Mat_<double>(3, 1) << 0, 0, 1);
  cv::Mat zAxisWorld = R.t() * zAxis;

  double t = -(normal.x * cameraPosition.at<double>(0) +
               normal.y * cameraPosition.at<double>(1) +
               normal.z * cameraPosition.at<double>(2) + d) /
             (normal.x * zAxisWorld.at<double>(0) +
              normal.y * zAxisWorld.at<double>(1) +
              normal.z * zAxisWorld.at<double>(2));

  cv::Point3f intersection(
      cameraPosition.at<double>(0) + t * zAxisWorld.at<double>(0),
      cameraPosition.at<double>(1) + t * zAxisWorld.at<double>(1),
      cameraPosition.at<double>(2) + t * zAxisWorld.at<double>(2));

  std::vector<cv::Point3f> worldPoints = {intersection};
  std::vector<cv::Point2f> imagePoints;
  cv::projectPoints(worldPoints, rvec, tvec, cameraMatrix, distCoeffs,
                    imagePoints);

  // cv::Mat resultImage = image.clone();
  if (!imagePoints.empty()) {
    cv::circle(image, imagePoints[0], 5, cv::Scalar(0, 0, 255),
               -1); // 红点标记交点
                    // cv::imshow("Camera Viewpoint", image);
    // cv::waitKey(0);
  } else {
    std::cerr << "Error: Intersection point is not visible in the image."
              << std::endl;
  }
}

/**
 * @brief 清空所有数据列表
 * @return void
 */
void WMIdentify::clear() {
  // 清空队列中的内容
  this->blade_tip_list.clear();
  this->wing_idx.clear();
  this->R_center_list.clear();
  this->R_idx.clear();
  this->time_list.clear();
  this->angle_list.clear();

  this->angle_velocity_list.clear();
  this->angle_velocity_list.emplace_back(
      0); // 先填充一个0，方便之后UpdateList中的数据对齐
  // 输出日志，清空成
  this->list_stat = 1;
  // //LOG_IF(INFO, this->switch_INFO) << "clear Successful";
}

/**
 * @brief 任意点位能量机关识别,角度收集和预测所需参数计算
 * @param[in] input_img     输入图像
 * @param[in] ts            串口数据
 * @return void
 */
void WMIdentify::identifyWM(cv::Mat &input_img, Translator &translator) {

  // 输入图片
  this->receive_pic(input_img);
  // 调用网络识别扇叶，结果存入fans中

  WMBlade blade;
  DetectionResult result;
  result = detect(this->img, blade, *(this->gp), translator.message.status / 5, translator);

  // cv::imshow("Processed Image", result.processedImage);
  // !这个延迟是为了拖慢每一帧的处理速度，不然太快了
  // cv::waitKey(16);

  if (blade.apex.size() == 6) {

    translator.message.armor_flag = 12;

    // //LOG_IF(INFO, this->switch_INFO) << "blade.apex.size() :" <<
    // blade.apex.size();
    //  translator.message.armor_flag = 11;
    this->R_center = blade.apex[0];
    if (firstFrame) {
      image_points.clear();
      image_points.push_back(blade.apex[0]);
      // image_points.push_back(blade.apex[1]);
      image_points.push_back(blade.apex[2]);
      image_points.push_back(blade.apex[3]);
      image_points.push_back(blade.apex[4]);
      cv::solvePnP(world_points, image_points, camera_matrix, dist_coeffs,
                   first_rvec, first_tvec);

      cv::Rodrigues(first_rvec, first_rotation_matrix);
      this->world2car = calculateTransformationMatrix(first_rotation_matrix,
                                                      first_tvec, translator);
      this->starting_time =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::system_clock::now().time_since_epoch());

      firstFrame = false;
    }
    // drawFixedWorldAxes(result.processedImage, this->camera_matrix,
    //                    this->dist_coeffs, first_rotation_matrix, first_tvec);

    // 图像坐标系中的点
    image_points.clear();
    image_points.push_back(blade.apex[0]);
    // image_points.push_back(blade.apex[1]);
    image_points.push_back(blade.apex[2]);
    image_points.push_back(blade.apex[3]);
    image_points.push_back(blade.apex[4]);

    // 在图像上绘制blade.apex中的6个点，每个点使用不同颜色，并标注序号
    // if (blade.apex.size() == 6) {
    //   // 定义6种不同的颜色
    //   std::vector<cv::Scalar> colors = {
    //     cv::Scalar(255, 0, 0),     // 蓝色
    //     cv::Scalar(0, 255, 0),     // 绿色
    //     cv::Scalar(0, 0, 255),     // 红色
    //     cv::Scalar(255, 255, 0),   // 青色
    //     cv::Scalar(255, 0, 255),   // 洋红色
    //     cv::Scalar(0, 255, 255)    // 黄色
    //   };

    //   // 绘制每个点及其序号
    //   for (int i = 0; i < blade.apex.size(); i++) {
    //     // 绘制点
    //     cv::circle(result.processedImage, blade.apex[i], 8, colors[i], -1);

    //     // 绘制序号，使用与点相同的颜色
    //     cv::putText(result.processedImage,
    //                 std::to_string(i),
    //                 blade.apex[i] + cv::Point2f(15, 5),
    //                 cv::FONT_HERSHEY_SIMPLEX,
    //                 1.0,
    //                 colors[i],
    //                 2);
    //   }
    // }

    // cv::circle(result.processedImage, blade.apex[0], 10, cv::Scalar(0, 255,
    // 0), 10);

    // 进行PnP解算获取旋转矩阵
    cv::solvePnP(world_points, image_points, camera_matrix, dist_coeffs, rvec,
                 tvec);
    // 可视化光心（好像不太准？）
    visualizeCameraViewpointApproximation(
        result.processedImage, this->camera_matrix, this->dist_coeffs, rvec,
        tvec, world_points);

    cv::Rodrigues(rvec, rotation_matrix);
    this->world2car =
        calculateTransformationMatrix(rotation_matrix, tvec, translator);
    cv::Mat R_world_coordinate = (cv::Mat_<double>(4, 1) << 0, 0, 0, 1.0);
    cv::Mat R_world_coordinate_car = this->world2car * R_world_coordinate;
    cv::Point3f R_car;
    if (R_world_coordinate_car.at<double>(3, 0) != 0) { // 检查齐次坐标w分量
      R_car = cv::Point3f(R_world_coordinate_car.at<double>(0, 0) /
                              R_world_coordinate_car.at<double>(3, 0),
                          R_world_coordinate_car.at<double>(1, 0) /
                              R_world_coordinate_car.at<double>(3, 0),
                          R_world_coordinate_car.at<double>(2, 0) /
                              R_world_coordinate_car.at<double>(3, 0));
    }
    cv::putText(result.processedImage, "R_x : " + std::to_string(R_car.x),
                cv::Point(800, 400), cv::FONT_HERSHEY_SIMPLEX, 2,
                cv::Scalar(0, 255, 0), 2);
    cv::putText(result.processedImage, "R_y : " + std::to_string(R_car.y),
                cv::Point(800, 500), cv::FONT_HERSHEY_SIMPLEX, 2,
                cv::Scalar(0, 255, 0), 2);
    cv::putText(result.processedImage, "R_z : " + std::to_string(R_car.z),
                cv::Point(800, 600), cv::FONT_HERSHEY_SIMPLEX, 2,
                cv::Scalar(0, 255, 0), 2);
    cv::putText(result.processedImage,
                "R_z : " + std::to_string(std::sqrt(R_car.z * R_car.z +
                                                    R_car.x * R_car.x)),
                cv::Point(800, 700), cv::FONT_HERSHEY_SIMPLEX, 2,
                cv::Scalar(0, 255, 0), 2);
    this->distance = sqrt(R_world_coordinate_car.at<double>(0, 0) *
                              R_world_coordinate_car.at<double>(0, 0) +
                          R_world_coordinate_car.at<double>(1, 0) *
                              R_world_coordinate_car.at<double>(1, 0) +
                          R_world_coordinate_car.at<double>(2, 0) *
                              R_world_coordinate_car.at<double>(2, 0));
    if (this->distance < 4 || this->distance > 12) {
      translator.message.armor_flag = 10;
      std::cout << "distance wrong!" << std::endl;
      return;
    }

    cv::Point3f Top_world = cv::Point3f(this->gp->d_RP2, 0, 0);
    // 将Point3f转换为Mat格式(4x1矩阵,齐次坐标)
    cv::Mat Top_world_mat =
        (cv::Mat_<double>(4, 1) << Top_world.x, Top_world.y, Top_world.z, 1.0);

    // 进行矩阵乘法
    cv::Mat Top_car_mat = this->world2car * Top_world_mat;

    // 将结果转换回Point3f
    cv::Point3f Top_car;
    if (Top_car_mat.at<double>(3, 0) != 0) { // 检查齐次坐标w分量
      Top_car = cv::Point3f(
          Top_car_mat.at<double>(0, 0) / Top_car_mat.at<double>(3, 0),
          Top_car_mat.at<double>(1, 0) / Top_car_mat.at<double>(3, 0),
          Top_car_mat.at<double>(2, 0) / Top_car_mat.at<double>(3, 0));
    }
    cv::putText(result.processedImage, "Top_x : " + std::to_string(Top_car.x),
                cv::Point(800, 100), cv::FONT_HERSHEY_SIMPLEX, 2,
                cv::Scalar(0, 255, 0), 2);
    cv::putText(result.processedImage, "Top_y : " + std::to_string(Top_car.y),
                cv::Point(800, 200), cv::FONT_HERSHEY_SIMPLEX, 2,
                cv::Scalar(0, 255, 0), 2);
    cv::putText(result.processedImage, "Top_z : " + std::to_string(Top_car.z),
                cv::Point(800, 300), cv::FONT_HERSHEY_SIMPLEX, 2,
                cv::Scalar(0, 255, 0), 2);

    //  std::cout << "Top_X:" << Top_car.x << std::endl;
    //  std::cout << "Top_Y:" << Top_car.y << std::endl;
    //  std::cout << "Top_Z:" << Top_car.z << std::endl;

    cv::Mat relativeRotMat = rotation_matrix.t() * first_rotation_matrix;

    double theta =
        atan2(relativeRotMat.at<double>(1, 0), relativeRotMat.at<double>(0, 0));
    this->angle = theta;
    // drawFixedWorldAxes(result.processedImage, this->camera_matrix,
    //                    this->dist_coeffs, rotation_matrix, tvec);
    // cv::putText(result.processedImage,
    //             "theta : " + std::to_string(theta * 180 / CV_PI),
    //             cv::Point(100, 100), cv::FONT_HERSHEY_SIMPLEX, 2,
    //             cv::Scalar(0, 255, 0), 2);
    cv::circle(result.processedImage, blade.apex[0], 4, cv::Scalar(0, 0, 255),
               -1);

    double rot_angle = atan2(blade.apex[2].y - blade.apex[0].y,
                             blade.apex[2].x - blade.apex[0].x);
    this->rot_angle = rot_angle;

    // cv::imshow("Processed Image", result.processedImage);

    // !这个延迟是为了拖慢每一帧的处理速度，不然太快了
    // cv::waitKey(50);

    // TODO:START 计算当前帧的能量机关中心点在相机坐标系中的位置
    std::vector<cv::Point3f> world_points_single;
    world_points_single.push_back(cv::Point3f(this->gp->d_Radius, 0.0, 0.0));
    std::vector<cv::Point2f> image_points_single;
    cv::projectPoints(world_points_single, rvec, tvec, camera_matrix,
                      dist_coeffs, image_points_single);
    // cv::circle(result.processedImage, image_points_single[0], 5,
    //            cv::Scalar(255, 255, 255), -1);
    // cv::putText(result.processedImage, "Predicted Point",
    //             image_points_single[0] + cv::Point2f(10, 10),
    //             cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    // cv::imshow("Processed Image", result.processedImage);
    this->img_0 = result.processedImage;

    // cv::waitKey(1);

    // // 创建4x4的变换矩阵
    // cv::Mat transform_matrix = cv::Mat::eye(4, 4, CV_64F);
    // // 将3x3旋转矩阵复制到左上角
    // rotation_matrix.copyTo(transform_matrix(cv::Rect(0, 0, 3, 3)));
    // // 将平移向量复制到最右列的前三行
    // tvec.copyTo(transform_matrix(cv::Rect(3, 0, 1, 3)));

    // cv::Point3f world_point(this->gp->d_Radius * std::cos(theta),
    //                         this->gp->d_Radius * std::sin(theta), 0.0);
    // // 右乘形式: P * R = P * R^T
    // // world_point =
    // //     cv::Point3f(world_point.x * relativeRotMat.at<double>(0, 0) +
    // //                     world_point.y * relativeRotMat.at<double>(1, 0) +
    // //                     world_point.z * relativeRotMat.at<double>(2, 0),
    // //                 world_point.x * relativeRotMat.at<double>(0, 1) +
    // //                     world_point.y * relativeRotMat.at<double>(1, 1) +
    // //                     world_point.z * relativeRotMat.at<double>(2, 1),
    // //                 world_point.x * relativeRotMat.at<double>(0, 2) +
    // //                     world_point.y * relativeRotMat.at<double>(1, 2) +
    // //                     world_point.z * relativeRotMat.at<double>(2, 2));

    // std::cout << "world_point : " << world_point << std::endl;
    // cv::Mat matrixWorld = (cv::Mat_<double>(4, 1) << world_point.x,
    //                        world_point.y, world_point.z, 1.0);
    // // cv::Mat matrixWorld = (cv::Mat_<double>(4, 1) << 0, 0, 0, 1.0);
    // cv::Mat matrixCamera = transform_matrix * matrixWorld;
    // m_predictCamera = {(float)(matrixCamera.at<double>(0, 0)),
    //                    (float)(matrixCamera.at<double>(1, 0)),
    //                    (float)(matrixCamera.at<double>(2, 0))};

    // // 将相机坐标系下的点转换到像素坐标系
    // cv::Point2f m_predictPixel;
    // m_predictPixel.x =
    //     camera_matrix.at<double>(0, 0) * m_predictCamera.x /
    //     m_predictCamera.z + camera_matrix.at<double>(0, 2);
    // m_predictPixel.y =
    //     camera_matrix.at<double>(1, 1) * m_predictCamera.y /
    //     m_predictCamera.z + camera_matrix.at<double>(1, 2);

    // cv::circle(result.processedImage, m_predictPixel, 5,
    //            cv::Scalar(255, 255, 255), -1);
    // cv::putText(result.processedImage, "Predicted Point",
    //             m_predictPixel + cv::Point2f(10, 10),
    //             cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

    // if (!image_points.empty()) {
    //   cv::line(result.processedImage, image_points[0], m_predictPixel,
    //            cv::Scalar(0, 255, 0), 2);
    // }
    // cv::imshow("Processed Image", result.processedImage);
    // cv::waitKey(0);
    // TODO:END;

    // ?START：以下是使用相邻两帧旋转矩阵结算收集角度的code部分，暂且注释，开始设计使用当前帧与第一帧间旋转矩阵的办法。
    // // 计算相对于上一帧的旋转角度
    // if (!last_rotation_matrix.empty()) {
    //   cv::Mat R_w1_to_w2 = rotation_matrix.t() * last_rotation_matrix;

    //   // 提取绕z轴的旋转分量，作为delta theta帧间旋转角度
    //   double dtheta =
    //       atan2(R_w1_to_w2.at<double>(1, 0), R_w1_to_w2.at<double>(0, 0));
    //   // dtheta = dtheta * 180 / CV_PI;
    //   if (fabs(dtheta) > 30) {
    //     // 此处添加扇叶切换的跳变处理逻辑
    //     // 如果dtheta大于30度，则认为发生了扇叶切换
    //     // 将angle_list清空，并重新开始收集角度
    //     // 将angle_velocity_list清空
    //   }

    //   // 累加角度
    //   this->angle -= dtheta;
    //   //LOG_IF(INFO, this->switch_INFO) << "sita :" << 180 * this->angle
    //   / 3.14;

    //   // //LOG_IF(INFO, this->switch_INFO)
    //   //     << "angle_list.size()  = " << angle_list.size();
    //   // //LOG_IF(INFO, this->switch_INFO) << "angle: " << this->angle;
    //   // this->angle_list.emplace_back(this->angle);
    // }

    // // 保存当前旋转矩阵作为下一帧的last_rotation_matrix
    // last_rotation_matrix = rotation_matrix.clone();
    // ?END：以上是使用相邻两帧旋转矩阵结算收集角度的code部分，暂且注释，开始设计使用当前帧与第一帧间旋转矩阵的办法。

    // cv::circle(result.processedImage, blade.apex[1], 5, cv::Scalar(0, 0,
    // 255),
    //            -1);

    this->ready_to_update = true;

    if (angle_list.size() >= gp->list_size - 2) {
      // 如果角度收集达到阈值，那么需要开始预测了，要给Predict发alpha和phi角度，以及距离s

      //  this->phi = this->calculatePhi(rotation_matrix, tvec);
      //  this->alpha = this->calculateAlpha(rotation_matrix, tvec, translator);
      //  this->s = 0;
      // //LOG_IF(INFO, this->switch_INFO) << "phi = " << phi * 180 / CV_PI;
      // //LOG_IF(INFO, this->switch_INFO) << "alpha = " << alpha * 180 / CV_PI;

      this->list_stat = 1;
    } else {
      // 如果角度收集未达到阈值，则不执行updateList
      // LOG_IF(ERROR, this->switch_ERROR)
      //     << "didn't execute updateList, lack data";
    }
    // //LOG_IF(INFO, this->switch_INFO) << "进入 updateList";
    // this->updateList((double)ts.messageWM.predict_time / 1000);
    // 获取当前时间与程序启动时间的差值
    double current_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
                .count() /
            1000.0 -
        this->starting_time.count() / 1000.0;
    this->updateList(current_time);
    // //LOG_IF(INFO, this->switch_INFO) << "完成 updateList";

    // cv::imshow("Processed Image", result.processedImage);
    // this->updateList((double)ts.messageWM.predict_time / 1000);

    // 如果已经解算过（有固定世界系了）
    // 那么收集角度
  } else {
    translator.message.armor_flag = 10;
    // translator.message.v_z = 10.0;
    std::cout << translator.message.armor_flag << std::endl;
    std::cout << "识别失败，不预测" << std::endl;
  }
}

/**
 * @brief 角度解算和收集函数（像素点反投影回世界系）
 * @param[in] blade_tip     扇叶顶点
 * @param[in] rotation_matrix     旋转矩阵
 * @param[in] tvec            平移向量
 * @return void
 */
void WMIdentify::calculateAngle(cv::Point2f blade_tip, cv::Mat rotation_matrix,
                                cv::Mat tvec) {
  // 计算相机光心在世界坐标系中的位置
  cv::Mat camera_in_world = -rotation_matrix.t() * tvec;
  // 计算图像点在相机坐标系和世界坐标系中的向量
  double u = blade_tip.x;
  double v = blade_tip.y;

  cv::Mat direction_camera =
      (cv::Mat_<double>(3, 1) << (u - camera_matrix.at<double>(0, 2)) /
                                     camera_matrix.at<double>(0, 0),
       (v - camera_matrix.at<double>(1, 2)) / camera_matrix.at<double>(1, 1),
       1.0);

  cv::Mat direction_world = rotation_matrix.t() * direction_camera;

  // 计算射线与z=0平面的交点（反投影回世界系）
  double s =
      -camera_in_world.at<double>(2, 0) / direction_world.at<double>(2, 0);
  double X =
      camera_in_world.at<double>(0, 0) + s * direction_world.at<double>(0, 0);
  double Y =
      camera_in_world.at<double>(1, 0) + s * direction_world.at<double>(1, 0);

  this->angle = atan2(Y, X);

  // std::cout << "this->gp->list_size : " << this->gp->list_size << std::endl;
  // // list_size = 120 std::cout << "gp->gap % gp->gap_control : " << gp->gap %
  // gp->gap_control << std::endl;

  // std::cout << "angle_list.size() : " << this->angle_list.size() <<
  // std::endl; //LOG_IF(INFO, this->switch_INFO) << "angle: " << this->angle;
  // //LOG_IF(INFO, this->switch_INFO) << "length: " << sqrt(X * X + Y * Y);

  // std::cout << "length: " << sqrt(X * X + Y * Y) << std::endl;

  // std::cout << "X: " << X << std::endl;
  // std::cout << "Y: " << Y << std::endl;
}

cv::Mat WMIdentify::calculateTransformationMatrix(cv::Mat R_world2cam,
                                                  cv::Mat tvec,
                                                  Translator &translator) {
  cv::Mat T_world2cam =
      (cv::Mat_<double>(4, 4) << R_world2cam.at<double>(0, 0),
       R_world2cam.at<double>(0, 1), R_world2cam.at<double>(0, 2),
       tvec.at<double>(0), R_world2cam.at<double>(1, 0),
       R_world2cam.at<double>(1, 1), R_world2cam.at<double>(1, 2),
       tvec.at<double>(1), R_world2cam.at<double>(2, 0),
       R_world2cam.at<double>(2, 1), R_world2cam.at<double>(2, 2),
       tvec.at<double>(2), 0, 0, 0, 1);

  double tx_cam2cloud = -0.00;
  double ty_cam2cloud = -0.54;
  double tz_cam2cloud = -0.16;
  if (!translator.message.is_far) {
    tx_cam2cloud = this->gp->tx_cam2cloud;
    ty_cam2cloud = this->gp->ty_cam2cloud;
    tz_cam2cloud = this->gp->tz_cam2cloud;
  } else {
    tx_cam2cloud = this->gp->tx_cam2cloud_1;
    ty_cam2cloud = this->gp->ty_cam2cloud_1;
    tz_cam2cloud = this->gp->tz_cam2cloud_1;
  }
  cv::Mat T_cam2cloud = (cv::Mat_<double>(4, 4) << 1, 0, 0, tx_cam2cloud, 0, 1,
                         0, ty_cam2cloud, 0, 0, 1, tz_cam2cloud, 0, 0, 0, 1);

  double yaw = translator.message.yaw;
  double pitch = translator.message.pitch;
  // yaw = 0.5;
  // pitch = 0.2;

  double cy = std::cos(yaw);
  double sy = std::sin(yaw);
  double cp = std::cos(pitch);
  double sp = std::sin(pitch);
  //   // 方案1: 先pitch后yaw
  // cv::Mat T_cloud2car = (cv::Mat_<double>(4, 4) <<
  //     cy, -sy*cp, sy*sp, 0,
  //     sy, cy*cp, -cy*sp, 0,
  //     0, sp, cp, 0,
  //     0, 0, 0, 1);

  cv::Mat R_y = (cv::Mat_<double>(3, 3) << cy, 0, sy, 0, 1, 0, -sy, 0, cy);
  cv::Mat R_x = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, cp, -sp, 0, sp, cp);
  // 注意旋转的先后顺序：R_car2cam = R_y * R_x
  cv::Mat T_cloud2car = (cv::Mat_<double>(4, 4) << cy, -sy * sp, -sy * cp, 0, 0,
                         cp, -sp, 0, sy, cy * sp, cy * cp, 0, 0, 0, 0, 1);

  // 组合旋转矩阵 R = R_z(yaw) * R_x(pitch)
  // 计算结果：
  // [ cos(yaw)       , -sin(yaw)*cos(pitch),  sin(yaw)*sin(pitch) ]
  // [ sin(yaw)       ,  cos(yaw)*cos(pitch), -cos(yaw)*sin(pitch) ]
  // [      0       ,         sin(pitch),          cos(pitch)    ]
  // cv::Mat T_cloud2car = (cv::Mat_<double>(4, 4) << cy, -sy * cp, sy * sp, 0,
  // sy, cy * cp, -cy * sp, 0, 0, sp, cp, 0, 0, 0, 0, 1); cv::Mat T_cloud2car =
  // (cv::Mat_<double>(4, 4) << cy, -sy * sp, -sy * cp, 0, 0, cp, -sp, 0, sy, cy
  // * sp, cy * cp, 0, 0, 0, 0, 1);

  //    T_world2car = T_cloud2car * T_cam2cloud * T_world2cam
  cv::Mat T_world2car = T_cloud2car * T_cam2cloud * T_world2cam;

  return T_world2car;
}

cv::Mat WMIdentify::getTransformationMatrix() { return this->world2car; }
/**
 * @brief 计算phi
 * @param[in] rotation_matrix     旋转矩阵
 * @param[in] tvec            平移向量
 * @return void
 */
double WMIdentify::calculatePhi(cv::Mat rotation_matrix, cv::Mat tvec) {
  cv::Mat camera_in_world = -rotation_matrix.t() * tvec;
  cv::Mat Z_camera_in_world = rotation_matrix.t().col(2);
  // cv::Mat Z_camera_in_camera = (cv::Mat_<double>(3,1) << 0,0,1);
  // cv::Mat Z_camera_in_world = rotation_matrix.t() * (Z_camera_in_camera -
  // tvec);
  double Vx = camera_in_world.at<double>(0, 0);
  double Vz = camera_in_world.at<double>(2, 0);
  double Zx = Z_camera_in_world.at<double>(0, 0);
  double Zz = Z_camera_in_world.at<double>(2, 0);
  double phi = acos((Vx * Zx + Vz * Zz) /
                    (sqrt(Vx * Vx + Vz * Vz) * sqrt(Zx * Zx + Zz * Zz)));
  // //LOG_IF(INFO, this->switch_INFO) <<"Vx * Zx + Vz * Zz / (sqrt(Vx * Vx + Vz
  // * Vz) * sqrt(Zx * Zx + Zz * Zz)) = "<< Vx * Zx + Vz * Zz / (sqrt(Vx * Vx +
  // Vz
  // * Vz) * sqrt(Zx * Zx + Zz * Zz));
  return Pi - phi;
}
/**
 * @brief 计算alpha
 * @param[in] rotation_matrix     旋转矩阵
 * @param[in] tvec            平移向量
 * @return void
 */
// double WMIdentify::calculateAlpha(cv::Mat rotation_matrix, cv::Mat tvec,
//                                   Translator &translator) {
//   // 保持现有实现不变
//   cv::Mat camera_in_world = -rotation_matrix.t() * tvec;
//   // 相机z轴在世界坐标系中的向量
//   cv::Mat Z_camera_in_world = rotation_matrix.t().col(2);
//   double Vx = camera_in_world.at<double>(0, 0);
//   double Vz = camera_in_world.at<double>(2, 0);
//   double Zx = Z_camera_in_world.at<double>(0, 0);
//   double Zz = Z_camera_in_world.at<double>(2, 0);
//   // phi的值为二者点积除以二者模的乘积
//   double phi = acos((Vx * Zx + Vz * Zz) /
//                     (sqrt(Vx * Vx + Vz * Vz) * sqrt(Zx * Zx + Zz * Zz)));
//   return Pi - phi;
// }
double WMIdentify::calculateAlpha(cv::Mat R_world2cam, cv::Mat tvec,
                                  Translator &translator) {
  cv::Mat C_world = -R_world2cam.t() * tvec; // 3x1向量

  double cy = std::cos(-translator.message.yaw),
         sy = std::sin(-translator.message.yaw);
  double cp = std::cos(-translator.message.pitch),
         sp = std::sin(-translator.message.pitch);
  cv::Mat R_y = (cv::Mat_<double>(3, 3) << cy, 0, sy, 0, 1, 0, -sy, 0, cy);
  cv::Mat R_x = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, cp, -sp, 0, sp, cp);
  cv::Mat R_car2cam = R_y * R_x;

  cv::Mat R_world2car = R_car2cam.t() * R_world2cam;

  cv::Mat C_car = R_world2car * C_world; // 3x1向量

  cv::Mat z_world = (cv::Mat_<double>(3, 1) << 0, 0, 1);
  cv::Mat z_car = R_world2car * z_world; // 3x1向量

  cv::Point2d v_proj(C_car.at<double>(0, 0), C_car.at<double>(2, 0));
  cv::Point2d z_proj(z_car.at<double>(0, 0), z_car.at<double>(2, 0));

  double dot = v_proj.x * z_proj.x + v_proj.y * z_proj.y;
  double det = v_proj.x * z_proj.y - v_proj.y * z_proj.x; // 叉积只看 y 分量
  double alpha = std::atan2(det, dot);
  alpha = Pi - std::abs(alpha);

  return alpha;
}

/**
 * @brief 接收输入图像
 * @param[in] input_img     输入图像
 * @return void
 */
void WMIdentify::receive_pic(cv::Mat &input_img) {
  this->img_0 = input_img.clone();
  this->img = input_img.clone();
  // //LOG_IF(INFO, this->switch_INFO) << "receive_pic Successful";
}

/**
 * @brief 更新数据列表
 * @param[in] time     当前时间
 * @return void
 */
void WMIdentify::updateList(double time) {
  //  this->gp->list_size = 260;
  //  this->gp->gap_control = 3;

#ifdef DEBUGMODE
  if (this->R_center_list.size() > 0 && ready_to_update)
    cv::circle(this->img, this->R_center_list[this->R_center_list.size() - 1],
               10, cv::Scalar(0, 255, 255), -1);
  if (this->blade_tip_list.size() > 0 && ready_to_update)
    cv::circle(this->img, this->blade_tip_list[this->blade_tip_list.size() - 1],
               10, cv::Scalar(0, 0, 255), -1);
#endif
  // 更新时间队列
  if (this->time_list.size() >= this->gp->list_size &&
      gp->gap % gp->gap_control == 0) {
    this->time_list.pop_front();
  }
  if (ready_to_update && gp->gap % gp->gap_control == 0) {
    this->time_list.push_back(time);
    //  std::cout << "time: " << time << std::endl;
  }
  // 更新角度队列
  if (this->angle_list.size() >= this->gp->list_size &&
      gp->gap % gp->gap_control == 0) {
    this->angle_list.pop_front();
    // //LOG_IF(INFO, gp->switch_INFO) << "完成 angle_list pop_front";
  }
  if (ready_to_update && gp->gap % gp->gap_control == 0) {
    this->angle_list.push_back(-angle);
  }
  // 更新旋转角度队列
  if (this->rot_angle_list.size() >= this->gp->list_size &&
      gp->gap % gp->gap_control == 0) {
    this->rot_angle_list.pop_front();
    // //LOG_IF(INFO, gp->switch_INFO) << "完成 angle_list pop_front";
  }
  if (ready_to_update && gp->gap % gp->gap_control == 0) {
    this->rot_angle_list.push_back(rot_angle);
  }

  // 更新R点中心坐标队列
  if (this->R_center_list.size() >= this->gp->list_size &&
      gp->gap % gp->gap_control == 0) {
    this->R_center_list.pop_front();
    // //LOG_IF(INFO, gp->switch_INFO) << "完成 angle_list pop_front";
  }
  if (ready_to_update && gp->gap % gp->gap_control == 0) {
    this->R_center_list.push_back(R_center);
  }

  // 更新枪管偏移yaw角度队列
  if (this->R_yaw_list.size() >= this->gp->list_size) {
    this->R_yaw_list.pop_front();
  }
  if (ready_to_update && gp->gap % gp->gap_control == 0) {
    float u = R_center_list[R_center_list.size() - 1].x;
    float v = R_center_list[R_center_list.size() - 1].y;
    float tan_x = (u - (double)this->img.cols / 2) / 2580.88644;
    // std::cout<<tan_x<<std::endl;
    float yaw = 0;
    if (abs(tan_x) > 0.0001)
      yaw = atan(tan_x);
    this->R_yaw_list.emplace_back(yaw);
    //  std::cout << "R_yaw : " << yaw * 180 / Pi << std::endl;
  }

  // 更新phi队列
  if (this->phi_list.size() >= this->gp->list_size) {
    this->phi_list.pop_front();
  }
  if (ready_to_update) {
    this->phi_list.push_back(phi);
  }
  // 更新alpha队列
  if (this->alpha_list.size() >= this->gp->list_size) {
    this->alpha_list.pop_front();
  }
  if (ready_to_update) {
    this->alpha_list.push_back(alpha);
  }

  // 更新角速度队列
  if (this->angle_velocity_list.size() > this->gp->list_size &&
      gp->gap % gp->gap_control == 0) {
    this->angle_velocity_list.pop_front();
  }
  if (rot_angle_list.size() > 1 && time_list.size() > 1 && ready_to_update &&
      gp->gap % gp->gap_control == 0) {
    // 计算角度变化量
    double dangle = this->rot_angle_list[this->rot_angle_list.size() - 1] -
                    this->rot_angle_list[this->rot_angle_list.size() - 2];
    // 防止数据跳变
    if (dangle < 0) {
      dangle = dangle + 2 * Pi;
    }
    int shift = std::round(dangle / (0.4 * Pi));
    dangle = dangle - shift * 0.4 * Pi;
    // dangle += (abs(dangle) > Pi) ? 2 * Pi * (-dangle / abs(dangle)) : 0;
    // 计算时间变化量
    double dtime = (this->time_list[this->time_list.size() - 1] -
                    this->time_list[this->time_list.size() - 2]);
    // 更新角速度队列,简单检验一下数据，获得扇叶切换时间
    // if (abs(dangle / dtime) > 5) {
    //   this->FanChangeTime = time_list.back() * 1000;
    //   this->time_list.pop_front();
    //   this->angle_list.pop_front();
    //   gp->gap--;
    // } else {
    this->angle_velocity_list.emplace_back(dangle / dtime);
    // }

    // 更新旋转方向
    // std::cout<<this->time_list.back()<<std::endl;
    // std::cout<<" "<<this->angle_velocity_list.back()<<std::endl;
    this->direction = 0;
    for (int i = 0; i < angle_velocity_list.size(); i++) {
      this->direction += this->angle_velocity_list[i];
    }
  }

  // if (this->angle_velocity_list.size() > this->gp->list_size &&
  //   gp->gap % gp->gap_control == 0) {
  // this->angle_velocity_list.pop_front();
  // }
  // if (angle_list.size() > 1 && time_list.size() > 1 && ready_to_update &&
  //   gp->gap % gp->gap_control == 0) {
  //   // 计算角度变化量
  //   double dangle = this->angle_list[this->angle_list.size() - 1] -
  //                   this->angle_list[this->angle_list.size() - 2];
  //   // 防止数据跳变
  //   if (dangle < 0) {
  //     dangle = dangle + 2* Pi;
  //   }
  //   int shift = std::round(dangle / (0.4 * Pi));
  //   dangle = dangle - shift * 0.4 * Pi;
  //   // dangle += (abs(dangle) > Pi) ? 2 * Pi * (-dangle / abs(dangle)) : 0;
  //   // 计算时间变化量
  //   double dtime = (this->time_list[this->time_list.size() - 1] -
  //                   this->time_list[this->time_list.size() - 2]);
  //   // 更新角速度队列,简单检验一下数据，获得扇叶切换时间
  //   // if (abs(dangle / dtime) > 5) {
  //   //   this->FanChangeTime = time_list.back() * 1000;
  //   //   this->time_list.pop_front();
  //   //   this->angle_list.pop_front();
  //   //   gp->gap--;
  //   // } else {
  //   this->angle_velocity_list.emplace_back(dangle / dtime);
  //   // }

  //   // 更新旋转方向
  //   // std::cout<<this->time_list.back()<<std::endl;
  //   // std::cout<<" "<<this->angle_velocity_list.back()<<std::endl;
  //   this->direction = 0;
  //   for (int i = 0; i < angle_velocity_list.size(); i++) {
  //     this->direction += this->angle_velocity_list[i];
  //   }
  // }

  // 更新gap
  if (this->ready_to_update) {
    gp->gap++;
    std::cout << gp->gap << std::endl;
    std::cout << gp->gap_control << std::endl;
    std::cout << (gp->gap % gp->gap_control) << std::endl;
  }
  // 输出日志，更新队列成功
  // //LOG_IF(INFO, this->switch_INFO == ON) << "updateList successful";
}

/**
 * @brief 获取时间列表
 * @return std::deque<double> 时间列表
 */
std::deque<double> WMIdentify::getTimeList() { return this->time_list; }

cv::Mat WMIdentify::getRvec() { return this->rvec; }
cv::Mat WMIdentify::getTvec() { return this->tvec; }

cv::Mat WMIdentify::getDist_coeffs() { return this->dist_coeffs; }

cv::Mat WMIdentify::getCamera_matrix() { return this->camera_matrix; }

/**
 * @brief 获取角速度列表
 * @return std::deque<double> 角速度列表
 */
std::deque<double> WMIdentify::getAngleVelocityList() {
  return this->angle_velocity_list;
}

/**
 * @brief 获取旋转方向
 * @return double 旋转方向
 */
double WMIdentify::getDirection() { return this->direction; }

/**
 * @brief 获取最新角度
 * @return double 最新角度值
 */
std::deque<double> WMIdentify::getAngleList() { return this->angle_list; }

double WMIdentify::getLastAngle() {
  return this->angle_list[angle_list.size() - 1];
}

double WMIdentify::getLastRotAngle() {
  return this->rot_angle_list[rot_angle_list.size() - 1];
}

double WMIdentify::getR_yaw() {
  return this->R_yaw_list[R_yaw_list.size() - 1];
}

/**
 * @brief 获取R点中心坐标
 * @return cv::Point2d R点中心坐标
 */
cv::Point2d WMIdentify::getR_center() {
  return this->R_center_list[R_center_list.size() - 1];
}

/**
 * @brief 获取半径
 * @return double 半径值
 */
double WMIdentify::getRadius() {

  // //LOG_IF(INFO, this->switch_INFO == ON)
  //       << "R_center_list.size() : " << this->R_center_list.size();
  // //LOG_IF(INFO, this->switch_INFO == ON)
  //       << "blade_tip_list.size() : " << this->blade_tip_list.size();

  return sqrt(
      calculateDistanceSquare(this->R_center_list[R_center_list.size() - 1],
                              this->blade_tip_list[blade_tip_list.size() - 1]));
}

double WMIdentify::getPhi() { return this->phi; }

double WMIdentify::getAlpha() { return this->alpha; }

double WMIdentify::getRdistance() { return this->distance; }

/**
 * @brief 获取列表状态
 * @return int 列表状态
 */
int WMIdentify::getListStat() { return this->list_stat; }

/**
 * @brief 获取原始图像
 * @return cv::Mat 原始图像
 */
cv::Mat WMIdentify::getImg0() { return this->img_0; }

/**
 * @brief 获取数据图像
 * @return cv::Mat 数据图像
 */
cv::Mat WMIdentify::getData_img() { return this->data_img; }

/**
 * @brief 清空速度相关数据
 * @return void
 */
void WMIdentify::ClearSpeed() {
  this->angle_velocity_list.clear();
  this->time_list.clear();
}

/**
 * @brief 获取扇叶切换时间
 * @return uint32_t 扇叶切换时间
 */
uint32_t WMIdentify::GetFanChangeTime() { return this->FanChangeTime; }

/**
 * @brief 根据翻译器状态判断是否需要清空数据
 * @param[in] translator     翻译器对象
 * @return void
 */
void WMIdentify::JudgeClear(Translator translator) {
  if (translator.message.status % 5 == 0) // 进入自瞄便清空识别的所有数据
    this->clear();
}

/**
 * @brief 计算两点之间的距离平方
 * @param[in] p1 点1
 * @param[in] p2 点2
 * @return double 距离平方
 */
double WMIdentify::calculateDistanceSquare(cv::Point2f p1, cv::Point2f p2) {
  //    //LOG_IF(INFO, this->switch_INFO == ON) << "calculateDistanceSquare";

  return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}
