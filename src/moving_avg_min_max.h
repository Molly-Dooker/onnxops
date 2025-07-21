#pragma once
#include <string>
#include <unordered_map>
#include <mutex>

struct ObserverState {
    double min = 0.0;
    double max = 0.0;
    bool is_initialized = false;
};

class ObserverManager {
public:
    static void update_stats(const std::string& name, float current_min, float current_max);
    static ObserverState get_stats(const std::string& name);
private:
    static std::unordered_map<std::string, ObserverState> all_observers_;
    static std::mutex mtx_;
    static constexpr double momentum_ = 0.1;
};
