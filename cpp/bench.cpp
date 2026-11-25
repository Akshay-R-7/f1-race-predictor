// Time 100,000 simulated races with 1 thread and with all cores.
#include <chrono>
#include <cstdio>
#include <thread>

#include "racesim.hpp"

int main() {
    std::vector<racesim::Driver> d(20);
    for (int i = 0; i < 20; ++i) { d[i].pace = i * 0.08; d[i].grid = i + 1; }
    racesim::Race r;
    for (int threads : {1, 0}) {
        auto t0 = std::chrono::steady_clock::now();
        auto s = racesim::simulate(d, r, 100000, threads, 1);
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("threads=%u  100000 races  %.2f s  (P1 win %.3f)\n",
                    threads ? threads : std::thread::hardware_concurrency(), sec, s.win_prob[0]);
    }
}
