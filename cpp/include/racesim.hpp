// Monte Carlo race simulator: run a race lap by lap many times and count outcomes.
#pragma once
#include <cstdint>
#include <vector>

namespace racesim {

struct Driver {
    double pace = 0.0;      // expected lap time vs. field median, seconds (lower = faster)
    double pace_sd = 0.15;  // day-to-day variation in that pace
    double dnf_prob = 0.05; // chance of retiring during the race
    double deg = 0.06;      // tyre wear: extra seconds per lap of tyre age
    int grid = 1;           // starting position, 1 = pole
};

struct Race {
    int laps = 57;
    double pit_loss = 22.0;            // seconds lost for a pit stop
    double sc_prob_per_lap = 0.004;    // chance a safety car starts on a given lap
    double overtake_difficulty = 0.4;  // 0 = passing is easy, 1 = almost impossible
    double grid_gap = 0.25;            // seconds between grid slots at the start
    double lap_noise = 0.3;            // random variation of a single lap
};

struct Summary {
    std::vector<double> win_prob, podium_prob, mean_finish, dnf_rate;
    std::int64_t n_sims = 0;
};

// One simulated race. Returns finishing order as driver indices; retired drivers last.
// `retired` is filled with 1 for drivers who did not finish.
template <class Rng>
std::vector<int> simulate_once(const std::vector<Driver>& d, const Race& r, Rng& rng,
                               std::vector<char>& retired);

// Run n_sims races on n_threads threads (0 = use all cores). Deterministic for a
// given (seed, n_threads).
Summary simulate(const std::vector<Driver>& drivers, const Race& race,
                 std::int64_t n_sims, int n_threads = 0, std::uint64_t seed = 42);

}  // namespace racesim
