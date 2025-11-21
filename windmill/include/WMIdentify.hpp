#include <cmath>
#include <opencv2/core/types.hpp>
#if !defined(__WMIDENTIFY_HPP)
#define __WMIDENTIFY_HPP
#include "globalParam.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "traditional_detection.hpp"
#include <deque>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/video/video.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

class WMIdentify {
private:
  int switch_INFO;  //<! 是否开启INFO级别的日志
  int switch_ERROR; //<! 是否开启ERROR级别的日志
  GlobalParam *gp;
  int t_start;
  cv::Mat img; //<! 获取的原图，以及在调试过程中最终显示结果的背景图
  cv::Mat img_0; //<! 获取的原图（仅对其用神经网络画框，不做其余操作）
  cv::Mat data_img; //<! 转速与时间的数据图
  cv::Point2f R_estimate;
  cv::Mat binary;                   //<! 经过预处理得到的图像
  std::vector<cv::Vec4i> hierarchy; //<! 获得的hierarchy关系，用于寻找装甲板
  std::vector<std::vector<cv::Point>>
      wing_contours; //<! 扇页轮廓信息，用于寻找扇叶下半部分
  std::vector<cv::Vec4i>
      floodfill_hierarchy; //<!
                           // 使用漫水处理方法时的hierarchy关系，暂时该方法被弃用
  std::vector<std::vector<cv::Point>>
      floodfill_contours; //<!
                          // 使用漫水处理方法时的轮廓信息，只能用来寻找装甲板，暂时该方法被弃用
  std::vector<std::vector<cv::Point>> R_contours; //<! 寻找R使用的轮廓信息
  std::deque<cv::Point2d> blade_tip_list; //<! 装甲板中心点的队列
  std::deque<cv::Point2d> R_center_list;  //<! R中心点的队列
  std::deque<int>
      wing_idx; //<! 扇叶的下半部分在wing_contours中的索引，返回并用作更新队列
  std::deque<int>
      winghat_idx; //<! 扇叶的上半部分在R_contours中的索引，返回并用作更新队列
  std::deque<int> R_idx; //<! R的轮廓在R_contours中的索引，返回并用作更新队列
  int Wing_stat; //<!
                 // 识别Wing的时候当前的状态，0为没有识别到Wing，1为识别到Wing，2为识别到多个Wing
  int Winghat_stat; //<!
                    // 识别Winghat的时候当前的状态，0为没有识别到Wing，1为识别到Wing，2为识别到多个Wing
  int R_stat;       //<!
              // 识别R的时候当前的状态，0为没有识别到R，1为识别到R，2为识别到多个R
  int list_stat = 0;
  bool pnp_solved = false; // 标记是否已经求解pnp
  bool firstFrame = true;
  bool ready_to_update = false;        // 标记是否可以更新
  cv::Mat camera_matrix;               // 相机内参矩阵
  cv::Mat dist_coeffs;                 // 畸变系数
  cv::Mat rvec;                        // 旋转向量
  cv::Mat tvec;                        // 平移向量
  cv::Mat rotation_matrix;             // 旋转矩阵
  cv::Mat first_rvec;                  // 第一帧旋转向量
  cv::Mat first_tvec;                  // 第一帧平移向量
  cv::Mat first_rotation_matrix;       // 第一帧旋转矩阵
  cv::Mat rvec_for_predict;            // 旋转向量
  cv::Mat tvec_for_predict;            // 平移向量
  cv::Mat rotation_matrix_for_predict; // 旋转矩阵
  cv::Mat world2car; // 世界系到车身系的变换矩阵(4*4)
  std::vector<cv::Point3f> world_points; // 世界坐标系中的点
  std::vector<cv::Point2f> image_points; // 图像坐标系中的点
  double phi; // 相机系原点与世界系原点连线相对于视线偏移的水平角度
  double distance;
  double alpha; // 相机系原点与世界系原点连线相对于世界系z轴偏移的水平角度
  double angle; // 扇叶的旋转角度（相对于第一帧建立的固定世界系）
  double rot_angle;     // 旋转角度(相对于水平绝对世界系)
  cv::Point2d R_center; // R的中心点
  double s; // 相机系原点与世界系原点之间的水平距离
  double direction;
  std::deque<double> time_list;           //<! 时间队列
  std::deque<double> angle_list;          //<! 角度队列
  std::deque<double> rot_angle_list;      //<! 旋转角度队列
  std::deque<double> R_yaw_list;          //<! 枪管偏移yaw角队列
  std::deque<double> angle_velocity_list; //<! 角速度队列
  std::deque<double> phi_list;
  std::deque<double> alpha_list;

  uint32_t FanChangeTime;
  static constexpr double MAX_VELOCITY = 15.0;   // 最大允许角速度
  static const int DIRECTION_WINDOW;             // 方向计算窗口大小
  static constexpr double MIN_TIME_DELTA = 1e-6; // 最小时间差
  cv::Mat last_rotation_matrix;            // 存储上一帧的旋转矩阵
  std::chrono::milliseconds starting_time; // 计时器
  cv::Point3f m_predictCamera;             // 相机坐标系下的预测点

public:
  /**
   * @brief WMIdentify的构造函数，初始化一些参数
   *
   * @param gp 全局参数结构体，通过引用输入
   */
  WMIdentify(GlobalParam &);
  /**
   * @brief WMIdentify的析构函数，一般来说程序会自行销毁那些需要销毁的东西
   *
   */
  ~WMIdentify();
  /**
   * @brief 清空装甲板、R的中心点坐标以及索引的队列
   *
   */
  void clear();
  void identifyWM(cv::Mat &, Translator &);
  /**
   * @brief 输入图像的接口
   *
   * @param input_img 输入的图像
   */
  void receive_pic(cv::Mat &);
  /**
   * @brief 对图像进行预处理：蒙板，hsv二值化等操作
   *
   * @param gp 全局参数结构体，通过引用输入
   */
  void preprocess();
  /**
   * @brief
   * 通过预处理后得到的binary图像进行膨胀腐蚀获得需要的图像，获得轮廓信息，这些轮廓信息将用于后续的寻找装甲板及R
   *
   * @param gp 全局参数结构体，通过引用输入
   */
  void getContours();

  /**
   * @brief 通过判断R的图形特征，获取若干R，并储存其索引在队列中。
   *
   * @param gp 全局参数结构体，通过引用输入
   * @return int 若没有识别到R，返回0；识别到1个R，返回1；识别到多于一个R，返回2
   */
  int getValidR();
  /**
   * @brief
   * 在R_idx中若干索引中挑选真正的R的索引，通过的方法为判断R的坐标是否在中心区域
   *
   * @param gp 全局参数结构体，通过引用输入
   * @return int 0为失败，1为成功
   */
  int selectR();
  bool IsValidR(std::vector<cv::Point> contour, GlobalParam &gp);
  /**
   * @brief 通过索引，更新装甲板中心点与R中心点的队列
   *
   * @param gp 全局参数结构体，通过引用输入
   */
  void updateList(double);
  /**
   * @brief 获取时间队列
   *
   * @return std::deque<double> 时间队列
   */
  std::deque<double> getTimeList();
  /**
   * @brief 获取角速度队列
   *
   * @return std::deque<double> 角速度队列
   */
  std::deque<double> getAngleVelocityList();
  /**
   * @brief 清空数据队列
   *
   * @param translator 状态位，每回到自瞄一次就清理一次
   */
  void JudgeClear(Translator translator);
  double getDirection();
  std::deque<double> getAngleList();
  double getLastAngle();
  double getLastRotAngle();
  double getYaw();
  double getPhi();
  double getR_yaw();
  double getAlpha();
  double getRdistance();
  int getListStat();
  cv::Mat getImg0();
  cv::Mat getData_img();
  cv::Point2d getR_center();
  double getRadius();
  void ClearSpeed();
  uint32_t GetFanChangeTime();
  void calculateAngle(cv::Point2f blade_tip, cv::Mat rotation_matrix,
                      cv::Mat tvec);
  double calculatePhi(cv::Mat rotation_matrix, cv::Mat tvec);
  double calculateAlpha(cv::Mat rotation_matrix, cv::Mat tvec,
                        Translator &translator);
  bool update_R_list();
  double calculateDistanceSquare(cv::Point2f p1, cv::Point2f p2);
  cv::Point3f getPredictCamera() const { return m_predictCamera; }
  cv::Mat getRvec();
  cv::Mat getTvec();
  cv::Mat getDist_coeffs();
  cv::Mat getCamera_matrix();
  void visualizeCameraViewpoint(const cv::Mat &image,
                                const cv::Mat &cameraMatrix,
                                const cv::Mat &distCoeffs, const cv::Mat &rvec,
                                const cv::Mat &tvec,
                                const std::vector<cv::Point3f> &objectPoints);
  cv::Mat calculateTransformationMatrix(cv::Mat rotation_matrix, cv::Mat tvec,
                                        Translator &translator);
  cv::Mat getTransformationMatrix();
};

#endif // __WMIDENTIFY_HPP
