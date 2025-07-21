#pragma once

#include <string>
#include <unordered_map>
#include <mutex>
#include <memory>

namespace MyQuantLib {

// 각 옵저버의 상태를 저장하는 구조체
struct ObserverState {
    float min;
    float max;
};

// 상태를 관리하는 스레드 안전한 싱글톤 클래스
class StateManager {
public:
    // 싱글톤 인스턴스를 가져옵니다.
    static StateManager& get_instance();

    // 새로운 옵저버를 등록합니다.
    void register_observer(const std::string& id);

    // 옵저버의 상태를 가져옵니다.
    ObserverState get_state(const std::string& id);

    // 옵저버의 상태에 대한 포인터를 가져옵니다. (CUDA 커널에서 직접 접근용)
    ObserverState* get_state_ptr(const std::string& id);

    // 복사 및 대입을 금지합니다.
    StateManager(const StateManager&) = delete;
    StateManager& operator=(const StateManager&) = delete;

private:
    StateManager() = default;
    ~StateManager() = default;

    std::unordered_map<std::string, ObserverState> state_map_;
    std::mutex mutex_;
};

} // namespace MyQuantLib