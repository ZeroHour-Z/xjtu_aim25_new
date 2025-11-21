#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <semaphore.h>
#include <unistd.h>
#include <errno.h>

#include <opencv2/opencv.hpp>

struct ImageShmHeader {
  uint32_t width;
  uint32_t height;
  uint32_t channels;
  uint32_t step;
  uint32_t data_len;
};

int main(int argc, char** argv) {
  std::string shm_name   = (argc > 1) ? argv[1] : "/image_raw_shm";
  std::string sem_name   = shm_name + "_sem";
  size_t      shm_size   = (argc > 2) ? std::stoul(argv[2]) : 8388672; // 和发布端一致

  // 1) 打开 SHM
  int fd = shm_open(shm_name.c_str(), O_RDWR, 0666);
  if (fd < 0) {
    std::cerr << "shm_open failed: " << strerror(errno) << "\n";
    return 1;
  }
  void* map = mmap(nullptr, shm_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  if (map == MAP_FAILED) {
    std::cerr << "mmap failed: " << strerror(errno) << "\n";
    return 1;
  }

  // 2) 打开信号量
  sem_t* sem = sem_open(sem_name.c_str(), 0);
  if (sem == SEM_FAILED) {
    std::cerr << "sem_open failed: " << strerror(errno) << "\n";
    return 1;
  }

  auto* hdr  = reinterpret_cast<ImageShmHeader*>(map);
  auto* data = reinterpret_cast<uint8_t*>(reinterpret_cast<char*>(map) + sizeof(ImageShmHeader));

  // 3) 等一帧，然后保存到本地
  std::cout << "Waiting for a frame on semaphore '" << sem_name << "'...\n";
  if (sem_wait(sem) != 0) {
    std::cerr << "sem_wait failed: " << strerror(errno) << "\n";
    return 1;
  }

  // 简单的健壮性检查
  if (hdr->width == 0 || hdr->height == 0 || hdr->channels != 3 || hdr->step < hdr->width * 3) {
    std::cerr << "Invalid header: w=" << hdr->width << " h=" << hdr->height
              << " ch=" << hdr->channels << " step=" << hdr->step
              << " data_len=" << hdr->data_len << "\n";
    return 1;
  }
  if (sizeof(ImageShmHeader) + hdr->data_len > shm_size) {
    std::cerr << "data_len exceeds shm_size\n";
    return 1;
  }

  // 构造 Mat（注意 step）
  cv::Mat view(hdr->height, hdr->width, CV_8UC3, data, hdr->step);
  // 复制一份，避免生产者随后覆盖
  cv::Mat frame = view.clone();

  // 保存、显示或两者
  if (!cv::imwrite("frame_from_shm.png", frame)) {
    std::cerr << "imwrite failed\n";
  } else {
    std::cout << "Saved: frame_from_shm.png\n";
  }

  // 可选：循环读取多帧
  for (int i = 0; i < 50; ++i) {
    if (sem_wait(sem) != 0 || i % 10 != 0) break;
    cv::Mat v2(hdr->height, hdr->width, CV_8UC3, data, hdr->step);
    cv::imwrite("frame_" + std::to_string(i) + ".png", v2);
  }

  sem_close(sem);
  munmap(map, shm_size);
  close(fd);
  return 0;
}
/* 使用方法
g++ -O2 reader.cpp -o reader `pkg-config --cflags --libs opencv4` -lrt -lpthread
./reader /image_raw_shm 8388672
用来检测共享内存的发布情况，如果发布端没有发布，则不会保存图片
*/