#pragma once

#include <string>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <limits>

namespace MyQuantLib {

// 옵저버의 상태를 저장하는 구조체
struct ObserverState {
    float min;
    float max;
};

// 상태를 관리하는 스레드 안전 싱글톤 클래스
class StateManager {
public:
    // 싱글톤 인스턴스 접근
    static StateManager& get_instance();

    // 새로운 옵저버를 등록 (초기 min=+∞, max=-∞)
    void register_observer(const std::string& id);

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
