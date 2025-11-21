#ifndef MVCAMERACONTROL_H_STUB
#define MVCAMERACONTROL_H_STUB

// Minimal type stubs to satisfy includes

typedef struct _MV_CC_DEVICE_INFO_LIST
{
    unsigned int nDeviceNum;
    void* pDeviceInfo[8];
} MV_CC_DEVICE_INFO_LIST;

typedef struct _MV_FRAME_OUT
{
    void* pBufAddr;
    unsigned int nFrameLen;
    unsigned int nWidth;
    unsigned int nHeight;
} MV_FRAME_OUT;

typedef struct _MV_CC_PIXEL_CONVERT_PARAM
{
    unsigned int nWidth;
    unsigned int nHeight;
} MV_CC_PIXEL_CONVERT_PARAM;

typedef struct _MVCC_FLOATVALUE
{
    float CurValue;
} MVCC_FLOATVALUE;

// Enums used in globalParam.hpp
typedef enum _MV_CAM_EXPOSURE_AUTO_MODE
{
    MV_EXPOSURE_AUTO_MODE_OFF = 0,
} MV_CAM_EXPOSURE_AUTO_MODE;

typedef enum _MV_CAM_GAIN_MODE
{
    MV_GAIN_MODE_OFF = 0,
} MV_CAM_GAIN_MODE;

// Added: trigger mode and trigger source enums
typedef enum _MV_CAM_TRIGGER_MODE
{
    MV_TRIGGER_MODE_OFF = 0,
    MV_TRIGGER_MODE_ON  = 1,
} MV_CAM_TRIGGER_MODE;

typedef enum _MV_CAM_TRIGGER_SOURCE
{
    MV_TRIGGER_SOURCE_LINE0 = 0,
    MV_TRIGGER_SOURCE_LINE1 = 1,
    MV_TRIGGER_SOURCE_LINE2 = 2,
    MV_TRIGGER_SOURCE_SOFTWARE = 7,
} MV_CAM_TRIGGER_SOURCE;

// Pixel type placeholder
#define PixelType_Gvsp_BayerRG8 0

#endif // MVCAMERACONTROL_H_STUB 