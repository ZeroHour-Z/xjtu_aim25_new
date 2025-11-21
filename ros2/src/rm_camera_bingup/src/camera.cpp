#include "camera.hpp"

Camera::Camera(const CameraConfig &config)
    : handle(NULL), nRet(MV_OK), color(config.color), attack_mode(config.attack_mode),
      pDataForRGB(nullptr), CvtParam{0}, stOutFrame{0}, frame_rate{0}, switch_INFO(config.switch_INFO),
      config_(config) {
    memset(&this->stOutFrame, 0, sizeof(MV_FRAME_OUT));
    memset(&this->stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));

    // 枚举设备
    this->nRet = MV_CC_EnumDevices(MV_USB_DEVICE, &stDeviceList);
    if (stDeviceList.nDeviceNum == 0) {
        printf("nDeviceNum == 0\n");
        exit(-1);
    }
    // 创建句柄
    this->nRet = MV_CC_CreateHandle(&handle, stDeviceList.pDeviceInfo[config_.cam_index]);
    if (this->nRet != MV_OK) {
#ifdef THREADANALYSIS
        printf("CreateHandle ERROR\n");
#endif
        exit(-1);
    }

    // 开启设备
    this->nRet = MV_CC_OpenDevice(handle);
    if (this->nRet != MV_OK) {
#ifdef THREADANALYSIS
        printf("OpenDevice ERROR\n");
#endif
        exit(-1);
    }
    printf("cam%d初始化完毕，欢迎使用喵～\n", config_.cam_index);
}

Camera::~Camera() {
    // 停止取流
    this->nRet = MV_CC_StopGrabbing(this->handle);
    // 关闭设备
    this->nRet = MV_CC_CloseDevice(this->handle);
    // 销毁句柄
    this->nRet = MV_CC_DestroyHandle(this->handle);
    printf("析构中，再见喵～\n");
}

int Camera::start_camera() {
    this->nRet = MV_CC_StartGrabbing(this->handle);
    if (this->nRet != MV_OK) {
#ifdef THREADANALYSIS
        printf("StartGrabbing ERROR\n");
#endif
        exit(-1);
    }
    return 0;
}

int Camera::get_pic(cv::Mat *srcimg) {
    this->nRet = MV_CC_GetImageBuffer(handle, &stOutFrame, 400);
    if (this->nRet != MV_OK) {
        if (nRet != -2147483641) {
            exit(-1);
        }
    }
    cv::Mat temp;
    temp = cv::Mat(stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth, CV_8UC1, CvtParam.pSrcData = stOutFrame.pBufAddr);
    if (temp.empty() == 1) return -1;
    cv::cvtColor(temp, *srcimg, cv::COLOR_BayerRG2RGB);
    if (NULL != stOutFrame.pBufAddr) {
        this->nRet = MV_CC_FreeImageBuffer(handle, &stOutFrame);
        if (this->nRet != MV_OK) {
#ifdef THREADANALYSIS
            printf("FreeImageBuffer ERROR\n");
#endif
        }
    }
    return 0;
}

int Camera::set_param_once() {
    this->nRet = MV_CC_SetHeight(this->handle, static_cast<unsigned int>(config_.height));
    this->nRet = MV_CC_SetWidth(this->handle, static_cast<unsigned int>(config_.width));
#ifdef USETRIGGERMODE
    this->nRet = MV_CC_riggerMode(this->handle, MV_TRIGGER_MODE_ON);
#else
    this->nRet = MV_CC_SetTriggerMode(this->handle, MV_TRIGGER_MODE_OFF);
#endif
    this->nRet = MV_CC_SetBalanceWhiteAuto(this->handle, ON);
    this->nRet = MV_CC_SetGamma(this->handle, static_cast<float>(config_.gamma_value));
    this->nRet = MV_CC_SetExposureAutoMode(this->handle, static_cast<MV_CAM_EXPOSURE_AUTO_MODE>(config_.enable_auto_exp));
    this->nRet = MV_CC_SetExposureTime(this->handle, static_cast<float>(config_.armor_exp_time));
    this->nRet = MV_CC_SetGainMode(this->handle, static_cast<MV_CAM_GAIN_MODE>(config_.enable_auto_gain));
    this->nRet = MV_CC_SetGain(this->handle, static_cast<float>(config_.gain));
    this->nRet = MV_CC_SetPixelFormat(this->handle, config_.pixel_format);
    this->nRet = MV_CC_SetFrameRate(this->handle, static_cast<float>(config_.frame_rate));
    return 0;
}

int Camera::set_param_mult() {
    if (this->attack_mode == ENERGY)
        this->nRet = MV_CC_SetExposureTime(this->handle, static_cast<float>(config_.energy_exp_time));
    else if (this->attack_mode == ARMOR)
        this->nRet = MV_CC_SetExposureTime(this->handle, static_cast<float>(config_.armor_exp_time));
    return 0;
}

int Camera::change_color(int input_color) {
    this->color = input_color;
    return 0;
}

int Camera::change_attack_mode(int input_attack_mode) {
    this->attack_mode = input_attack_mode;
    return 0;
}

int Camera::request_frame_rate() {
    this->nRet = MV_CC_GetFrameRate(handle, &frame_rate);
    printf("当前帧率:%f\n", frame_rate.fCurValue);
    return 0;
}

void Camera::init() {
    this->set_param_once();
    this->start_camera();
}

void Camera::getFrame(cv::Mat &pic) {
    this->set_param_mult();
    this->get_pic(&pic);
}