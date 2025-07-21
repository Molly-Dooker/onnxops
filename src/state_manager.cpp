#include "state_manager.h"
#include <limits>
#include <stdexcept>

namespace MyQuantLib {

StateManager& StateManager::get_instance() {
    static StateManager instance;
    return instance;
}

void StateManager::register_observer(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_map_.find(id) == state_map_.end()) {
        state_map_[id] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest()};
    }
}

ObserverState StateManager::get_state(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_map_.find(id) == state_map_.end()) {
        throw std::runtime_error("Observer with id '" + id + "' not found.");
    }
    return state_map_.at(id);
}

ObserverState* StateManager::get_state_ptr(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_map_.find(id) == state_map_.end()) {
        throw std::runtime_error("Attempted to access unregistered observer '" + id + "'.");
    }
    return &state_map_.at(id);
}

} // namespace MyQuantLib