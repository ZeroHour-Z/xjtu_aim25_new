#include "AimAuto.hpp"
#include "WMIdentify.hpp"
#include "WMPredict.hpp"
#include "globalParam.hpp"

#include <atomic>
#include <chrono>
#include <cstring>
#include <cv_bridge/cv_bridge.h>
#include <errno.h>
#include <fcntl.h>
#include <functional>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <semaphore.h>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

class ArmorNode : public rclcpp::Node {
public:
  ArmorNode() : Node("armor_node") {
    // ROS2 参数管理，替代 GlobalParam::initGlobalParam 的文件读取
    this->declare_parameter<std::string>("image_topic", "/image_raw");
    this->declare_parameter<std::string>("rx_topic", "/autoaim/rx");
    this->declare_parameter<std::string>("tx_topic", "/autoaim/tx");
    this->declare_parameter<int>("target_color", 0);
    this->declare_parameter<double>("dt", 0.01);
    this->declare_parameter<double>("blue_exp_time", gp.blue_exp_time);
    this->declare_parameter<double>("red_exp_time", gp.red_exp_time);
    this->declare_parameter<int>("switch_INFO", gp.switch_INFO);
    this->declare_parameter<int>("switch_ERROR", gp.switch_ERROR);
    this->declare_parameter<double>("resize", gp.resize);
    // 纯视觉模式（不收发串口/通信）
    this->declare_parameter<bool>("vision_only", false);

    // 共享内存可选项
    this->declare_parameter<bool>("use_comm_shm", false);
    this->declare_parameter<std::string>("comm_shm_name", "/rm_comm_shm");
    this->declare_parameter<bool>("use_image_shm", false);
    this->declare_parameter<std::string>("image_shm_name", "/image_raw_shm");
    this->declare_parameter<int>("image_shm_size", 8388672); // 约 8MB + 头部

    const std::string image_topic = this->get_parameter("image_topic").as_string();
    const std::string rx_topic    = this->get_parameter("rx_topic").as_string();
    const std::string tx_topic    = this->get_parameter("tx_topic").as_string();
    gp.color                      = this->get_parameter("target_color").as_int();
    dt_                           = this->get_parameter("dt").as_double();
    gp.blue_exp_time              = this->get_parameter("blue_exp_time").as_double();
    gp.red_exp_time               = this->get_parameter("red_exp_time").as_double();
    gp.switch_INFO                = this->get_parameter("switch_INFO").as_int();
    gp.switch_ERROR               = this->get_parameter("switch_ERROR").as_int();
    gp.resize                     = this->get_parameter("resize").as_double();

    vision_only_ = this->get_parameter("vision_only").as_bool();

    use_comm_shm_   = this->get_parameter("use_comm_shm").as_bool();
    comm_shm_name_  = this->get_parameter("comm_shm_name").as_string();
    use_image_shm_  = this->get_parameter("use_image_shm").as_bool();
    image_shm_name_ = this->get_parameter("image_shm_name").as_string();
    image_shm_size_ = static_cast<size_t>(this->get_parameter("image_shm_size").as_int());

    aim_auto = std::make_unique<AimAuto>(&gp);
    wmi_ = std::make_unique<WMIdentify>(gp);
    wmp_ = std::make_unique<WMPredict>(gp);

    // 订阅图像（或启用共享内存模式）
    if (!use_image_shm_) {
      image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
          image_topic, rclcpp::SensorDataQoS(),
          std::bind(&ArmorNode::imageCallback, this, std::placeholders::_1));
    } else {
      setupImageShm();
      if (image_sem_) {
        running_.store(true);
        image_rx_thread_ = std::thread(&ArmorNode::imageRxLoop, this);
      }
    }

    // 串口RX/TX：topic 或共享内存；vision_only_ 时跳过
    if (!vision_only_) {
      if (!use_comm_shm_) {
        rx_sub_ = this->create_subscription<std_msgs::msg::String>(
            rx_topic, rclcpp::SystemDefaultsQoS(),
            std::bind(&ArmorNode::rxCallback, this, std::placeholders::_1));

        tx_pub_ =
            this->create_publisher<std_msgs::msg::String>(tx_topic, rclcpp::SystemDefaultsQoS());
      } else {
        setupCommShm();
        if (rx_sem_) {
          running_.store(true);
          comm_rx_thread_ = std::thread(&ArmorNode::commRxLoop, this);
        }
      }
    } else {
      // 纯视觉模式：初始化一次假输入，基于 target_color 设置颜色（0:RED,
      // 1:BLUE）
      translator_.message.status = static_cast<uint8_t>(gp.color * 5);
    }

    // 初始化自瞄模块
  }

  ~ArmorNode() override {
    running_.store(false);
    if (comm_rx_thread_.joinable()) {
      // 解除阻塞
      if (rx_sem_)
        sem_post(rx_sem_);
      comm_rx_thread_.join();
    }
    if (image_rx_thread_.joinable()) {
      if (image_sem_)
        sem_post(image_sem_);
      image_rx_thread_.join();
    }
    closeCommShm();
    closeImageShm();
  }

private:
  // 简单的HEX编解码，默认两字符一字节
  static bool hexToBytes(const std::string& hex, uint8_t* out64) {
    if (hex.size() != 128)
      return false;
    auto hexVal = [](char c) -> int {
      if (c >= '0' && c <= '9')
        return c - '0';
      if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
      if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;
      return -1;
    };
    for (size_t i = 0; i < 64; ++i) {
      int hi = hexVal(hex[2 * i]);
      int lo = hexVal(hex[2 * i + 1]);
      if (hi < 0 || lo < 0)
        return false;
      out64[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return true;
  }
  static std::string bytesToHex(const uint8_t* in64) {
    static const char* digits = "0123456789ABCDEF";
    std::string        s;
    s.resize(128);
    for (size_t i = 0; i < 64; ++i) {
      s[2 * i]     = digits[(in64[i] >> 4) & 0xF];
      s[2 * i + 1] = digits[in64[i] & 0xF];
    }
    return s;
  }
  // CRC16-CCITT，与原有 MessageManager 保持一致
  static void crc16_update(uint16_t* currectCrc, char* src, uint32_t lengthInBytes) {
    uint32_t crc = *currectCrc;
    for (uint32_t j = 0; j < lengthInBytes; ++j) {
      uint32_t byte = static_cast<uint8_t>(src[j]);
      crc ^= byte << 8;
      for (int i = 0; i < 8; ++i) {
        uint32_t tmp = crc << 1;
        if (crc & 0x8000)
          tmp ^= 0x1021;
        crc = tmp;
      }
    }
    *currectCrc = static_cast<uint16_t>(crc);
  }
  static void updateCrc(Translator& ts, int len) {
    ts.message.crc = 0;
    crc16_update(&ts.message.crc, ts.data, static_cast<uint32_t>(len));
  }

  void rxCallback(const std_msgs::msg::String::SharedPtr msg) {
    Translator t{};
    if (!hexToBytes(msg->data, reinterpret_cast<uint8_t*>(t.data))) {
      RCLCPP_WARN(this->get_logger(), "RX string invalid hex length=%zu", msg->data.size());
      return;
    }
    // 简单拷贝，复用与 main.cpp 类似的共享消息语义
    temp_       = t;
    translator_ = t;
  }

  void publishTranslator() {
    translator_.message.head = 0x71;
    translator_.message.tail = 0x4C;
    // 与原工程保持：以61字节参与CRC
    updateCrc(translator_, 61);
    if (vision_only_) {
      // 纯视觉模式不发布
      return;
    }
    if (!use_comm_shm_) {
      std_msgs::msg::String out;
      out.data = bytesToHex(reinterpret_cast<uint8_t*>(translator_.data));
      tx_pub_->publish(out);
      return;
    }
    // 共享内存二进制直写（64 字节）
    if (tx_shm_ && tx_sem_) {
      std::memcpy(tx_shm_->data, translator_.data, 64);
      tx_shm_->len = 64;
      sem_post(tx_sem_);
    }
  }

  void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "imageCallback: %s",
                           msg->header.frame_id.c_str());
    } catch (const cv_bridge::Exception& e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }

    cv::Mat frame = cv_ptr->image;
    processFrame(frame);
  }

  void processFrame(cv::Mat& frame) {
    // 纯视觉模式：在每帧开始构造一次本地输入（仿照 FakeMessage）
    if (vision_only_) {
      translator_.message.status = static_cast<uint8_t>(gp.color * 5);
    }
    // 依据状态位切换模式与颜色（移植自 main.cpp）
    if (translator_.message.status % 5 != 0 && translator_.message.status % 5 != 2) {
      gp.attack_mode = ENERGY;
    } else {
      gp.attack_mode = ARMOR;
    }
    gp.armor_exp_time = (translator_.message.status / 5) ? gp.red_exp_time : gp.blue_exp_time;
    if (translator_.message.status / 5 != gp.color) {
      gp.color = translator_.message.status / 5;
    }

    // 计算 dt 与延迟
    double time_stamp =
        std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();
    double dt                   = (last_time_stamp_ > 0.0) ? (time_stamp - last_time_stamp_) : dt_;
    translator_.message.latency = static_cast<float>(dt * 1000.0);
    last_time_stamp_            = time_stamp;

    try {
      if (gp.attack_mode == ARMOR) {
        // 自瞄
        RCLCPP_INFO(this->get_logger(), "auto aim");
        aim_auto->auto_aim(gp, frame, translator_, dt);

        publishTranslator();
      } else {
        // 能量机关
        wmi_->identifyWM(frame, translator_);
        wmp_->StartPredict(translator_, gp, *wmi_);
        publishTranslator();
      }
    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "processing exception: %s", e.what());
    }
#ifdef DEBUGMODE
    cv::imshow("src", frame);
    cv::waitKey(1);
#endif
  }

  // 共享内存——串口
  struct ShmTx {
    uint32_t len;
    uint8_t  data[64];
  };
  struct ShmRx {
    uint32_t len;
    uint8_t  data[512];
  };

  void setupCommShm() {
    std::string tx_name     = comm_shm_name_ + std::string("_tx");
    std::string rx_name     = comm_shm_name_ + std::string("_rx");
    std::string tx_sem_name = comm_shm_name_ + std::string("_tx_sem");
    std::string rx_sem_name = comm_shm_name_ + std::string("_rx_sem");

    tx_fd_ = shm_open(tx_name.c_str(), O_RDWR, 0666);
    rx_fd_ = shm_open(rx_name.c_str(), O_RDWR, 0666);
    if (tx_fd_ < 0 || rx_fd_ < 0) {
      RCLCPP_WARN(this->get_logger(), "comm shm_open failed: %s", std::strerror(errno));
      return;
    }
    tx_shm_ = static_cast<ShmTx*>(
        mmap(nullptr, sizeof(ShmTx), PROT_READ | PROT_WRITE, MAP_SHARED, tx_fd_, 0));
    rx_shm_ = static_cast<ShmRx*>(
        mmap(nullptr, sizeof(ShmRx), PROT_READ | PROT_WRITE, MAP_SHARED, rx_fd_, 0));
    if (tx_shm_ == MAP_FAILED || rx_shm_ == MAP_FAILED) {
      RCLCPP_WARN(this->get_logger(), "comm mmap failed");
      tx_shm_ = nullptr;
      rx_shm_ = nullptr;
      return;
    }
    tx_sem_ = sem_open(tx_sem_name.c_str(), 0);
    rx_sem_ = sem_open(rx_sem_name.c_str(), 0);
    if (tx_sem_ == SEM_FAILED || rx_sem_ == SEM_FAILED) {
      RCLCPP_WARN(this->get_logger(), "comm sem_open failed");
      if (tx_sem_ != SEM_FAILED) {
        sem_close(tx_sem_);
        tx_sem_ = nullptr;
      }
      if (rx_sem_ != SEM_FAILED) {
        sem_close(rx_sem_);
        rx_sem_ = nullptr;
      }
    }
  }

  void closeCommShm() {
    if (tx_shm_ && tx_shm_ != MAP_FAILED) {
      munmap(tx_shm_, sizeof(ShmTx));
      tx_shm_ = nullptr;
    }
    if (rx_shm_ && rx_shm_ != MAP_FAILED) {
      munmap(rx_shm_, sizeof(ShmRx));
      rx_shm_ = nullptr;
    }
    if (tx_fd_ >= 0) {
      close(tx_fd_);
      tx_fd_ = -1;
    }
    if (rx_fd_ >= 0) {
      close(rx_fd_);
      rx_fd_ = -1;
    }
    if (tx_sem_ && tx_sem_ != SEM_FAILED) {
      sem_close(tx_sem_);
      tx_sem_ = nullptr;
    }
    if (rx_sem_ && rx_sem_ != SEM_FAILED) {
      sem_close(rx_sem_);
      rx_sem_ = nullptr;
    }
  }

  void commRxLoop() {
    while (running_.load()) {
      if (!rx_sem_ || !rx_shm_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        continue;
      }
      // 等待数据到达
      if (sem_wait(rx_sem_) == -1) {
        if (errno == EINTR)
          continue;
        break;
      }
      if (!running_.load())
        break;
      uint32_t n = rx_shm_->len;
      if (n == 64) {
        Translator t{};
        std::memcpy(t.data, rx_shm_->data, 64);
        temp_       = t;
        translator_ = t;
      } else {
        // 未对齐到协议长度，忽略
      }
    }
  }

  // 共享内存——图像
  struct ImageShmHeader {
    uint32_t width;
    uint32_t height;
    uint32_t channels; // 期望 3
    uint32_t step;     // 每行字节数
    uint32_t data_len;
  };

  void setupImageShm() {
    std::string img_name     = image_shm_name_;
    std::string img_sem_name = image_shm_name_ + std::string("_sem");
    image_fd_                = shm_open(img_name.c_str(), O_RDWR, 0666);
    if (image_fd_ < 0) {
      RCLCPP_WARN(this->get_logger(), "image shm_open failed: %s", std::strerror(errno));
      return;
    }
    image_map_ = mmap(nullptr, image_shm_size_, PROT_READ | PROT_WRITE, MAP_SHARED, image_fd_, 0);
    if (image_map_ == MAP_FAILED) {
      RCLCPP_WARN(this->get_logger(), "image mmap failed");
      image_map_ = nullptr;
      return;
    }
    image_hdr_ = reinterpret_cast<ImageShmHeader*>(image_map_);
    image_data_ =
        reinterpret_cast<uint8_t*>(reinterpret_cast<char*>(image_map_) + sizeof(ImageShmHeader));
    image_sem_ = sem_open(img_sem_name.c_str(), 0);
    if (image_sem_ == SEM_FAILED) {
      RCLCPP_WARN(this->get_logger(), "image sem_open failed");
      image_sem_ = nullptr;
    }
  }

  void closeImageShm() {
    if (image_map_) {
      munmap(image_map_, image_shm_size_);
      image_map_ = nullptr;
    }
    if (image_fd_ >= 0) {
      close(image_fd_);
      image_fd_ = -1;
    }
    if (image_sem_ && image_sem_ != SEM_FAILED) {
      sem_close(image_sem_);
      image_sem_ = nullptr;
    }
    image_hdr_  = nullptr;
    image_data_ = nullptr;
  }

  void imageRxLoop() {
    while (running_.load()) {
      if (!image_sem_ || !image_hdr_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        continue;
      }
      if (sem_wait(image_sem_) == -1) {
        if (errno == EINTR)
          continue;
        break;
      }
      if (!running_.load())
        break;
      uint32_t w    = image_hdr_->width;
      uint32_t h    = image_hdr_->height;
      uint32_t step = image_hdr_->step;
      uint32_t ch   = image_hdr_->channels;
      uint32_t len  = image_hdr_->data_len;
      if (w == 0 || h == 0 || ch == 0 || step == 0 || len == 0)
        continue;
      // 只支持 BGR8
      if (ch != 3)
        continue;
      cv::Mat frame(h, w, CV_8UC3);
      // 按 step 拷贝每行
      for (uint32_t r = 0; r < h; ++r) {
        std::memcpy(frame.ptr(r), image_data_ + r * step, static_cast<size_t>(w) * ch);
      }
      processFrame(frame);
    }
  }

  GlobalParam                                              gp;
  double                                                   dt_{0.01};
  double                                                   last_time_stamp_{0.0};
  std::unique_ptr<AimAuto>                                 aim_auto;
  std::unique_ptr<WMIdentify>                              wmi_;
  std::unique_ptr<WMPredict>                               wmp_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr   rx_sub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr      tx_pub_;
  Translator                                               temp_{};
  Translator                                               translator_{};

  // 共享内存配置与资源
  bool        use_comm_shm_{false};
  std::string comm_shm_name_;
  int         tx_fd_{-1};
  int         rx_fd_{-1};
  ShmTx*      tx_shm_{nullptr};
  ShmRx*      rx_shm_{nullptr};
  sem_t*      tx_sem_{nullptr};
  sem_t*      rx_sem_{nullptr};
  std::thread comm_rx_thread_;

  bool            use_image_shm_{false};
  std::string     image_shm_name_;
  size_t          image_shm_size_{0};
  int             image_fd_{-1};
  void*           image_map_{nullptr};
  ImageShmHeader* image_hdr_{nullptr};
  uint8_t*        image_data_{nullptr};
  sem_t*          image_sem_{nullptr};
  std::thread     image_rx_thread_;

  bool              vision_only_{false};
  std::atomic<bool> running_{false};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ArmorNode>());
  rclcpp::shutdown();
  return 0;
}