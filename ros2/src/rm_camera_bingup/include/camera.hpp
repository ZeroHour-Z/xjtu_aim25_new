/**
 * @file camera.hpp
 * @author riverflows2333 (liuyuzheng0801@163.com)
 * @brief 相机类声明文件，提供初始化相机、开始取流、设置参数、图像转化并获得图像、改变参数的方法
 * @version 0.1
 * @date 2022-12-25
 *
 * @copyright Copyright (c) 2022
 *
 */

#if !defined(__CAMERA_HPP)
#define __CAMERA_HPP

#include "MvCameraControl.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <glog/logging.h>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/video.hpp>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 基本模式与颜色枚举（与原工程保持一致的数值）
enum COLOR { RED = 0, BLUE = 1 };
enum ATTACK_MODE { ENERGY = 0, ARMOR = 1 };
enum SWITCH { OFF = 0, ON = 1 };

struct CameraConfig {
    int cam_index = 0;
    int color = BLUE;
    int attack_mode = ARMOR;
    int switch_INFO = ON;

    double height = 1080.0;
    double width = 1440.0;

    int enable_auto_exp = MV_EXPOSURE_AUTO_MODE_OFF; // int 表示，内部转换
    double energy_exp_time = 200.0;
    double armor_exp_time = 290.0;

    int enable_auto_gain = MV_GAIN_MODE_OFF; // int 表示，内部转换
    double gain = 17.0;
    double gamma_value = 0.7;

    unsigned int pixel_format = PixelType_Gvsp_BayerRG8;
    double frame_rate = 180.0;
};

class Camera {
public:
    void *handle;
    int nRet;
    int color;
    int attack_mode;
    unsigned char *pDataForRGB;
    MV_CC_DEVICE_INFO_LIST stDeviceList;
    MV_FRAME_OUT stOutFrame;
    MV_CC_PIXEL_CONVERT_PARAM CvtParam;
    MVCC_FLOATVALUE frame_rate;
    bool switch_INFO;

public:
    Camera(const CameraConfig &config);
    ~Camera();

    int start_camera();
    int set_param_once();
    int set_param_mult();
    int get_pic(cv::Mat *srcimg);
    int change_color(int input_color);
    int change_attack_mode(int input_attack_mode);
    int request_frame_rate();

    void init();
    void getFrame(cv::Mat &pic);

private:
    CameraConfig config_;
};

#endif // __CAMERA_HPP