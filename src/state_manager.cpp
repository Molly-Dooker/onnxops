#include "state_manager.h"
#include <stdexcept>
#include <limits>

namespace MyQuantLib {

StateManager& StateManager::get_instance() {
    static StateManager instance;
    return instance;
}

void StateManager::register_moving_average(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!state_map_.count(id)) {
        state_map_.emplace(id,
            ObserverState(std::numeric_limits<float>::max(),
                          std::numeric_limits<float>::lowest()));
    }
}

void StateManager::register_histogram(const std::string& id, int64_t bins) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!state_map_.count(id)) {
        state_map_.emplace(id, ObserverState(bins));
    }
}

ObserverState StateManager::get_state(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = state_map_.find(id);
    if (it == state_map_.end()) {
        throw std::runtime_error("Observer with id '" + id + "' not found.");
    }
    return it->second;
}

ObserverState* StateManager::get_state_ptr(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = state_map_.find(id);
    if (it == state_map_.end()) {
        throw std::runtime_error("Attempted to access unregistered observer '" + id + "'.");
    }
    return &it->second;
}

}  // namespace MyQuantLib
