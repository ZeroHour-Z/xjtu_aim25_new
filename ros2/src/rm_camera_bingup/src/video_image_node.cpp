#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"
#include "std_msgs/msg/header.hpp"

#include <string>
#include <algorithm>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <semaphore.h>
#include <unistd.h>
#include <errno.h>

class VideoImageNode : public rclcpp::Node {
public:
  VideoImageNode() : rclcpp::Node("video_image_node") {
    topic_name_ = this->declare_parameter<std::string>("topic_name", "/image_raw");
    video_path_ = this->declare_parameter<std::string>("video_path", "");
    publish_rate_hz_ = this->declare_parameter<double>("publish_rate", 0.0);
    loop_ = this->declare_parameter<bool>("loop", true);

    use_image_shm_ = this->declare_parameter<bool>("use_image_shm", false);
    image_shm_name_ = this->declare_parameter<std::string>("image_shm_name", "/image_raw_shm");
    image_shm_size_ = static_cast<size_t>(this->declare_parameter<int>("image_shm_size", 8388672));

    if (video_path_.empty()) {
      RCLCPP_FATAL(this->get_logger(), "video_path is empty");
      throw std::runtime_error("video_path is required");
    }
    if (!cap_.open(video_path_)) {
      RCLCPP_FATAL(this->get_logger(), "failed to open video: %s", video_path_.c_str());
      throw std::runtime_error("open video failed");
    }

    double fps = publish_rate_hz_ > 0.0 ? publish_rate_hz_ : 60.0;
    
    const auto period = std::chrono::duration<double>(1.0 / std::max(1.0, fps));

    if (use_image_shm_) {
      setupImageShm();
    } else {
      publisher_ = this->create_publisher<sensor_msgs::msg::Image>(topic_name_, rclcpp::SensorDataQoS());
    }

    timer_ = this->create_wall_timer(std::chrono::duration_cast<std::chrono::milliseconds>(period), std::bind(&VideoImageNode::onTimer, this));

    RCLCPP_INFO(this->get_logger(), "video_image_node started. video='%s', fps=%.2f, shm=%s", video_path_.c_str(), fps, use_image_shm_?"true":"false");
  }

  ~VideoImageNode() override {
    closeImageShm();
  }

private:
  void onTimer() {
    cv::Mat frame;
    if (!cap_.read(frame) || frame.empty()) {
      if (loop_) {
        cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
        if (!cap_.read(frame) || frame.empty()) {
          RCLCPP_WARN(this->get_logger(), "loop rewind failed or empty frame");
          return;
        }
      } else {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "end of video reached");
        return;
      }
    }

    if (frame.channels() == 1) {
      cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
    } else if (frame.channels() == 4) {
      cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
    }

    if (use_image_shm_ && image_hdr_ && image_data_) {
      const uint32_t w = static_cast<uint32_t>(frame.cols);
      const uint32_t h = static_cast<uint32_t>(frame.rows);
      const uint32_t ch = 3;
      const uint32_t step = static_cast<uint32_t>(frame.step);
      const uint32_t data_len = step * h;
      if (sizeof(ImageShmHeader) + data_len <= image_shm_size_) {
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "publish image to shm: %d, %d, %d, %d, %d at %.2f Hz", w, h, ch, step, data_len, publish_rate_hz_); 
        image_hdr_->height = h;
        image_hdr_->channels = ch;
        image_hdr_->width = w;
        image_hdr_->step = step;
        image_hdr_->data_len = data_len;
        if (frame.isContinuous() && step == w * ch) {
          std::memcpy(image_data_, frame.data, data_len);
        } else {
          for (uint32_t r = 0; r < h; ++r) {
            std::memcpy(image_data_ + r * step, frame.ptr(r), w * ch);
          }
        }
        if (image_sem_) sem_post(image_sem_);
      }
      return;
    }

    auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
    msg->header.stamp = this->now();
    msg->header.frame_id = "camera_optical_frame";
    // printf("publish image\n");
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "publish image at topic: %s with frequency: %f", topic_name_.c_str(), publish_rate_hz_);
    publisher_->publish(*msg);
  }

  struct ImageShmHeader {
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t step;
    uint32_t data_len;
  };

  void setupImageShm() {
    std::string name = image_shm_name_;
    std::string sem_name = image_shm_name_ + std::string("_sem");
    shm_fd_ = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
    if (shm_fd_ < 0) {
      RCLCPP_WARN(this->get_logger(), "image shm_open failed: %s", std::strerror(errno));
      return;
    }
    if (ftruncate(shm_fd_, static_cast<off_t>(image_shm_size_)) != 0) {
      RCLCPP_WARN(this->get_logger(), "image ftruncate failed: %s", std::strerror(errno));
    }
    shm_map_ = mmap(nullptr, image_shm_size_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
    if (shm_map_ == MAP_FAILED) {
      RCLCPP_WARN(this->get_logger(), "image mmap failed");
      shm_map_ = nullptr;
      return;
    }
    image_hdr_ = reinterpret_cast<ImageShmHeader*>(shm_map_);
    image_data_ = reinterpret_cast<uint8_t*>(reinterpret_cast<char*>(shm_map_) + sizeof(ImageShmHeader));
    image_hdr_->width = 0;
    image_hdr_->height = 0;
    image_hdr_->channels = 0;
    image_hdr_->step = 0;
    image_hdr_->data_len = 0;
    image_sem_ = sem_open(sem_name.c_str(), O_CREAT, 0666, 0);
    if (image_sem_ == SEM_FAILED) {
      RCLCPP_WARN(this->get_logger(), "image sem_open failed");
      image_sem_ = nullptr;
    }
  }

  void closeImageShm() {
    if (shm_map_) { munmap(shm_map_, image_shm_size_); shm_map_ = nullptr; }
    if (shm_fd_ >= 0) { close(shm_fd_); shm_fd_ = -1; }
    if (image_sem_ && image_sem_ != SEM_FAILED) { sem_close(image_sem_); image_sem_ = nullptr; }
    image_hdr_ = nullptr;
    image_data_ = nullptr;
  }

  std::string topic_name_;
  std::string video_path_;
  double publish_rate_hz_ {0.0};
  bool loop_ {true};

  cv::VideoCapture cap_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;

  bool use_image_shm_ {false};
  std::string image_shm_name_;
  size_t image_shm_size_ {0};
  int shm_fd_ {-1};
  void *shm_map_ {nullptr};
  ImageShmHeader *image_hdr_ {nullptr};
  uint8_t *image_data_ {nullptr};
  sem_t *image_sem_ {nullptr};
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<VideoImageNode>());
  rclcpp::shutdown();
  return 0;
} 