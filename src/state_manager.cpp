#include "state_manager.h"
#include <stdexcept>
#include <limits>

namespace MyQuantLib {

StateManager& StateManager::get_instance() {
    static StateManager instance;
    return instance;
}

void StateManager::register_observer(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_map_.find(id) == state_map_.end()) {
        // 초기 상태: min을 +∞, max를 -∞로 설정
        state_map_[id] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest()};
    }
}

ObserverState StateManager::get_state(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = state_map_.find(id);
    if (it == state_map_.end()) {
        throw std::runtime_error("Observer with id '" + id + "' not found.");
    }
    return it->second; // 구조체 복사 반환
}

ObserverState* StateManager::get_state_ptr(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = state_map_.find(id);
    if (it == state_map_.end()) {
        throw std::runtime_error("Attempted to access unregistered observer '" + id + "'.");
    }
    return &(it->second);
}

} // namespace MyQuantLib
