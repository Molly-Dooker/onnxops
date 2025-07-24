#pragma once

#include <string>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <limits>
#include <vector>

namespace MyQuantLib {

// (MovingAverage  Histogram 용 통합)
struct ObserverState {
    // MovingAverage 용
    float min;
    float max;
    // Histogram 용
    int64_t             bins; // bin 개수
    std::vector<int64_t> hist; // 길이 = bins

    ObserverState() = default;
    // MA 전용 등록
    ObserverState(float init_min, float init_max)
        : min(init_min), max(init_max), bins(0), hist() {}
    // Histogram 전용 등록
    ObserverState(int64_t bins_)
        : min(std::numeric_limits<float>::max()),
          max(std::numeric_limits<float>::lowest()),
          bins(bins_),
          hist(bins_, 0) {}
};

// 상태를 관리하는 스레드 안전 싱글톤 클래스
class StateManager {
public:
    // 싱글톤 인스턴스 접근
    static StateManager& get_instance();

    // MovingAverage 옵저버 등록
    void register_moving_average(const std::string& id);
    // Histogram 옵저버 등록 (bins 개수 지정)
    void register_histogram(const std::string& id, int64_t bins);

    // 옵저버의 상태 값을 복사 반환
    ObserverState get_state(const std::string& id);

    // 옵저버의 상태에 대한 포인터를 반환 (CUDA 커널 등에서 직접 접근 용도)
    ObserverState* get_state_ptr(const std::string& id);

    // 복사 및 대입 금지
    StateManager(const StateManager&) = delete;
    StateManager& operator=(const StateManager&) = delete;

private:
    StateManager() = default;
    ~StateManager() = default;

    std::unordered_map<std::string, ObserverState> state_map_;
    std::mutex mutex_;
};

} // namespace MyQuantLib
