#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"
#include "std_msgs/msg/header.hpp"
#include <algorithm>

#include "camera.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <semaphore.h>
#include <unistd.h>
#include <errno.h>

class HKCameraNode : public rclcpp::Node {
public:
    HKCameraNode() : rclcpp::Node("hk_camera_node") {
        topic_name_ = this->declare_parameter<std::string>("topic_name", "/image_raw");
        publish_rate_hz_ = this->declare_parameter<double>("publish_rate", 200.0);

        CameraConfig cfg;
        cfg.color = this->declare_parameter<int>("color", cfg.color);
        cfg.attack_mode = this->declare_parameter<int>("attack_mode", cfg.attack_mode);
        cfg.cam_index = this->declare_parameter<int>("cam_index", cfg.cam_index);
        cfg.height = this->declare_parameter<double>("height", cfg.height);
        cfg.width = this->declare_parameter<double>("width", cfg.width);
        cfg.enable_auto_exp = this->declare_parameter<int>("enable_auto_exp", cfg.enable_auto_exp);
        cfg.energy_exp_time = this->declare_parameter<double>("energy_exp_time", cfg.energy_exp_time);
        cfg.armor_exp_time = this->declare_parameter<double>("armor_exp_time", cfg.armor_exp_time);
        cfg.enable_auto_gain = this->declare_parameter<int>("enable_auto_gain", cfg.enable_auto_gain);
        cfg.gain = this->declare_parameter<double>("gain", cfg.gain);
        cfg.gamma_value = this->declare_parameter<double>("gamma_value", cfg.gamma_value);
        cfg.frame_rate = this->declare_parameter<double>("frame_rate", cfg.frame_rate);
        cfg.pixel_format = static_cast<unsigned int>(this->declare_parameter<int>("pixel_format", static_cast<int>(cfg.pixel_format)));

        // 共享内存参数
        use_image_shm_ = this->declare_parameter<bool>("use_image_shm", false);
        image_shm_name_ = this->declare_parameter<std::string>("image_shm_name", "/image_raw_shm");
        image_shm_size_ = static_cast<size_t>(this->declare_parameter<int>("image_shm_size", 8388672));

        publisher_ = this->create_publisher<sensor_msgs::msg::Image>(topic_name_, rclcpp::SensorDataQoS());

        try {
            camera_ = std::make_unique<Camera>(cfg);
            camera_->init();
        } catch (const std::exception &e) {
            RCLCPP_FATAL(this->get_logger(), "Camera init failed: %s", e.what());
            throw;
        }

        if (use_image_shm_) {
            setupImageShm();
        }

        using namespace std::chrono_literals;
        auto period = std::chrono::duration<double>(1.0 / std::max(1.0, publish_rate_hz_));
        timer_ = this->create_wall_timer(std::chrono::duration_cast<std::chrono::milliseconds>(period), std::bind(&HKCameraNode::onTimer, this));

        RCLCPP_INFO(this->get_logger(), "hk_camera_node started. Publishing on '%s' at %.1f Hz (image_shm=%s)", topic_name_.c_str(), publish_rate_hz_, use_image_shm_?"true":"false");
    }

    ~HKCameraNode() override {
        closeImageShm();
    }

private:
    void onTimer() {
        cv::Mat frame_bgr;
        camera_->getFrame(frame_bgr);
        if (frame_bgr.empty()) {
            RCLCPP_WARN(this->get_logger(), "Failed to get image frame");
            return;
        }
        if (use_image_shm_ && image_hdr_ && image_data_) {
            const uint32_t w = static_cast<uint32_t>(frame_bgr.cols);
            const uint32_t h = static_cast<uint32_t>(frame_bgr.rows);
            const uint32_t ch = 3;
            const uint32_t step = static_cast<uint32_t>(frame_bgr.step);
            const uint32_t data_len = step * h;
            // 检查容量
            if (sizeof(ImageShmHeader) + data_len <= image_shm_size_) {
                image_hdr_->width = w;
                image_hdr_->height = h;
                image_hdr_->channels = ch;
                image_hdr_->step = step;
                image_hdr_->data_len = data_len;
                if (frame_bgr.isContinuous() && step == w * ch) {
                    std::memcpy(image_data_, frame_bgr.data, data_len);
                } else {
                    for (uint32_t r = 0; r < h; ++r) {
                        std::memcpy(image_data_ + r * step, frame_bgr.ptr(r), w * ch);
                    }
                }
                if (image_sem_) sem_post(image_sem_);
            }
            return; // 共享内存模式下不再发布话题
        }
        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame_bgr).toImageMsg();
        msg->header.stamp = this->now();
        msg->header.frame_id = "camera_optical_frame";
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
        // 初始化头
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
    double publish_rate_hz_ {200.0};
    std::unique_ptr<Camera> camera_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;

    // 共享内存资源
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
    rclcpp::spin(std::make_shared<HKCameraNode>());
    rclcpp::shutdown();
    return 0;
} 