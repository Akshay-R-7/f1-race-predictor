#include "racesim.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>

namespace racesim {

template <class Rng>
std::vector<int> simulate_once(const std::vector<Driver>& d, const Race& r, Rng& rng,
                               std::vector<char>& retired) {
    const int n = static_cast<int>(d.size());
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    std::vector<double> time(n), pace(n), hazard(n);
    std::vector<int> tyre_age(n, 0), stint(n), laps_done(n, 0);
    std::vector<char> pitted(n, 0);
    retired.assign(n, 0);

    for (int i = 0; i < n; ++i) {
        time[i] = (d[i].grid - 1) * r.grid_gap;
        pace[i] = d[i].pace + d[i].pace_sd * gauss(rng);   // today's form
        // Per-lap retirement chance that adds up to dnf_prob over the whole race.
        hazard[i] = 1.0 - std::pow(1.0 - d[i].dnf_prob, 1.0 / r.laps);
        // Pit when the time lost to old tyres outweighs a stop: optimal stint ~ sqrt(2*loss/deg).
        double best = d[i].deg > 1e-6 ? std::sqrt(2.0 * r.pit_loss / d[i].deg) : r.laps;
        stint[i] = std::clamp(static_cast<int>(best), 10, std::max(10, r.laps / 2));
    }

    std::vector<int> order(n), running, out_cars;   // reused every lap: no allocations inside the loop
    running.reserve(n);
    out_cars.reserve(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return d[a].grid < d[b].grid; });

    for (int lap = 1; lap <= r.laps; ++lap) {
        const bool sc = unif(rng) < r.sc_prob_per_lap;
        if (sc) {  // safety car bunches the field up behind the leader, order unchanged
            double lead = -1;
            int k = 0;
            for (int i : order) {
                if (retired[i]) continue;
                if (lead < 0) lead = time[i];
                time[i] = lead + 0.8 * k++;
            }
        }
        for (int i = 0; i < n; ++i) {
            pitted[i] = 0;
            if (retired[i]) continue;
            if (unif(rng) < hazard[i]) { retired[i] = 1; continue; }
            double lt = pace[i] + d[i].deg * tyre_age[i] + r.lap_noise * gauss(rng);
            const bool due = tyre_age[i] >= stint[i] && lap < r.laps - 3;
            const bool cheap_sc_stop = sc && tyre_age[i] > stint[i] * 0.6 && lap < r.laps - 3;
            if (due || cheap_sc_stop) {
                lt += sc ? r.pit_loss * 0.5 : r.pit_loss;   // pitting under a safety car is cheaper
                tyre_age[i] = 0;
                pitted[i] = 1;
            } else {
                ++tyre_age[i];
            }
            time[i] += lt;
            laps_done[i] = lap;
        }
        // Overtaking: walk the running order front to back. A faster car behind only gets
        // past with some probability; otherwise it is held up just behind the car ahead.
        running.clear();
        out_cars.clear();
        for (int i : order) (retired[i] ? out_cars : running).push_back(i);
        for (size_t k = 1; k < running.size(); ++k) {
            int c = running[k], a = running[k - 1];
            if (time[c] < time[a]) {
                bool passes = pitted[a] || unif(rng) > r.overtake_difficulty;
                if (passes) std::swap(running[k], running[k - 1]);
                else time[c] = time[a] + 0.2;
            }
        }
        // The order barely changes from lap to lap, so insertion sort is close to O(n) here,
        // much cheaper than a full sort.
        for (size_t k = 1; k < running.size(); ++k) {
            int c = running[k];
            size_t j = k;
            while (j > 0 && time[running[j - 1]] > time[c]) { running[j] = running[j - 1]; --j; }
            running[j] = c;
        }
        order.assign(running.begin(), running.end());
        order.insert(order.end(), out_cars.begin(), out_cars.end());
    }
    // Retired cars are classified behind finishers, the one that lasted longest first.
    std::stable_partition(order.begin(), order.end(), [&](int i) { return !retired[i]; });
    std::stable_sort(std::find_if(order.begin(), order.end(), [&](int i) { return retired[i]; }),
                     order.end(), [&](int a, int b) { return laps_done[a] > laps_done[b]; });
    return order;
}

template std::vector<int> simulate_once<std::mt19937_64>(const std::vector<Driver>&, const Race&,
                                                         std::mt19937_64&, std::vector<char>&);

Summary simulate(const std::vector<Driver>& drivers, const Race& race, std::int64_t n_sims,
                 int n_threads, std::uint64_t seed) {
    const int n = static_cast<int>(drivers.size());
    if (n < 2) throw std::invalid_argument("need at least two drivers");
    if (race.laps < 1) throw std::invalid_argument("laps must be positive");
    if (n_threads <= 0) n_threads = std::max(1u, std::thread::hardware_concurrency());

    struct Acc { std::vector<std::int64_t> wins, podiums, dnfs, finishes; std::vector<double> pos_sum; };
    std::vector<Acc> acc(n_threads, Acc{std::vector<std::int64_t>(n), std::vector<std::int64_t>(n),
                                        std::vector<std::int64_t>(n), std::vector<std::int64_t>(n),
                                        std::vector<double>(n)});
    std::vector<std::thread> pool;
    for (int t = 0; t < n_threads; ++t) {
        // Split the work evenly; each thread has its own random generator and counters,
        // so threads never share writable memory and need no locks.
        std::int64_t begin = n_sims * t / n_threads, end = n_sims * (t + 1) / n_threads;
        pool.emplace_back([&, t, begin, end] {
            std::mt19937_64 rng(seed + 0x9E3779B97F4A7C15ULL * (t + 1));
            std::vector<char> retired;
            Acc& a = acc[t];
            for (std::int64_t s = begin; s < end; ++s) {
                auto order = simulate_once(drivers, race, rng, retired);
                if (!retired[order[0]]) ++a.wins[order[0]];
                for (int p = 0; p < n; ++p) {
                    int i = order[p];
                    if (retired[i]) { ++a.dnfs[i]; continue; }
                    if (p < 3) ++a.podiums[i];
                    ++a.finishes[i];
                    a.pos_sum[i] += p + 1;
                }
            }
        });
    }
    for (auto& th : pool) th.join();

    Summary s;
    s.n_sims = n_sims;
    s.win_prob.assign(n, 0); s.podium_prob.assign(n, 0); s.mean_finish.assign(n, 0); s.dnf_rate.assign(n, 0);
    for (int i = 0; i < n; ++i) {
        std::int64_t w = 0, p = 0, f = 0, x = 0; double ps = 0;
        for (auto& a : acc) { w += a.wins[i]; p += a.podiums[i]; f += a.finishes[i]; x += a.dnfs[i]; ps += a.pos_sum[i]; }
        s.win_prob[i] = double(w) / n_sims;
        s.podium_prob[i] = double(p) / n_sims;
        s.dnf_rate[i] = double(x) / n_sims;
        s.mean_finish[i] = f ? ps / f : 0.0;
    }
    return s;
}

}  // namespace racesim
