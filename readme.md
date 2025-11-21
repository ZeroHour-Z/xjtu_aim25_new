# 2025年笃行战队视觉组代码文档

## 简介

本文档主要介绍代码结构、代码流程、代码内容、算法原理、调参细节与其他内容。

## 环境依赖

1. MVS，可以前往[HIKROBOT官网](https://www.hikrobotics.com/cn/machinevision/service/download?module=0)找到Linux版本的MVS进行下载，之后解压压缩包，运行压缩包中的setup.sh即可。
2. glog，`sudo apt-get install libgoogle-glog-dev`。
3. Eigen3，`sudo apt-get install libeigen3-dev`。
4. ceres，通过 `sudo apt-get install`确保已经下载依赖 `libgoogle-glog-dev`、`libgflags-dev`、`libatlas-base-dev`、`libeigen3-dev`、`libsuitesparse-dev`，之后前往github中[ceres主页](https://github.com/ceres-solver/ceres-solver/tags)下载ceres1.1.4，使用cmake编译的方式安装。
5. openvino，进入官网[Openvino官网下载](https://www.intel.cn/content/www/cn/zh/developer/tools/openvino-toolkit/download.html)，其中有很详细的步骤说明。
    
    在此给出openvino 2024的一个安装方法:
    ```bash
    # 第 1 步：下载 GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB。您也可以使用以下命令
    wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
    ```
    ```bash
    # 第 2 步：将此密钥添加到系统密钥环。
    sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUBPUB
    ```
    ```bash
    # 第 3 步：通过以下命令添加存储库。
    # Ubuntu 22
    echo "deb https://apt.repos.intel.com/openvino/2024 ubuntu22 main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2024.list
    ```
    ```bash
    # 第 4 步：使用更新命令更新软件包列表并使用apt进行安装。
    sudo apt update
    sudo apt install openvino-2024.6.0
    ```

6. opencv，使用openvino自带的opencv。为了播放视频，需要安装依赖：`sudo apt-get install gstreamer1.0-libav`

