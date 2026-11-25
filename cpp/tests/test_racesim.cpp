#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>

#include "racesim.hpp"

using namespace racesim;

static std::vector<Driver> field(int n, double spread = 0.1) {
    std::vector<Driver> d(n);
    for (int i = 0; i < n; ++i) { d[i].pace = i * spread; d[i].grid = i + 1; d[i].dnf_prob = 0.05; }
    return d;
}

TEST(Simulator, WinProbabilitiesSumToAtMostOne) {
    auto s = simulate(field(20), Race{}, 20000, 2, 1);
    double total = std::accumulate(s.win_prob.begin(), s.win_prob.end(), 0.0);
    EXPECT_LE(total, 1.0 + 1e-12);
    EXPECT_GT(total, 0.99);  // only fails to be 1 if every car retires, which is very rare
}

TEST(Simulator, SameSeedSameThreadsIsDeterministic) {
    auto a = simulate(field(10), Race{}, 5000, 3, 123);
    auto b = simulate(field(10), Race{}, 5000, 3, 123);
    EXPECT_EQ(a.win_prob, b.win_prob);
    EXPECT_EQ(a.mean_finish, b.mean_finish);
}

TEST(Simulator, MuchFasterDriverAlmostAlwaysWins) {
    auto d = field(10);
    d[5].pace = -1.5;  // 1.5 s per lap faster than everyone, starting 6th
    d[5].dnf_prob = 0.0;
    auto s = simulate(d, Race{}, 20000, 2, 7);
    EXPECT_GT(s.win_prob[5], 0.95);
}

TEST(Simulator, CertainRetirementNeverWins) {
    auto d = field(10);
    d[0].dnf_prob = 1.0;
    auto s = simulate(d, Race{}, 5000, 2, 3);
    EXPECT_EQ(s.win_prob[0], 0.0);
    EXPECT_NEAR(s.dnf_rate[0], 1.0, 1e-12);
}

TEST(Simulator, RetirementRateMatchesInput) {
    auto d = field(10);
    for (auto& x : d) x.dnf_prob = 0.2;
    auto s = simulate(d, Race{}, 40000, 4, 11);
    for (double r : s.dnf_rate) EXPECT_NEAR(r, 0.2, 0.015);
}

TEST(Simulator, HarderOvertakingHelpsThePoleSitter) {
    auto d = field(20, 0.02);       // a close field
    d[0].pace = 0.15;               // pole sitter is a bit slower than P2..P4
    Race easy; easy.overtake_difficulty = 0.05;
    Race monaco; monaco.overtake_difficulty = 0.95;
    auto e = simulate(d, easy, 20000, 2, 5);
    auto m = simulate(d, monaco, 20000, 2, 5);
    EXPECT_GT(m.win_prob[0], e.win_prob[0]);
}

TEST(Simulator, ThreadCountDoesNotChangeTheAnswer) {
    auto d = field(10);
    auto one = simulate(d, Race{}, 40000, 1, 9);
    auto four = simulate(d, Race{}, 40000, 4, 9);
    for (size_t i = 0; i < d.size(); ++i) EXPECT_NEAR(one.win_prob[i], four.win_prob[i], 0.015);
}

TEST(Simulator, RejectsBadInput) {
    EXPECT_THROW(simulate(field(1), Race{}, 10), std::invalid_argument);
    Race r; r.laps = 0;
    EXPECT_THROW(simulate(field(5), r, 10), std::invalid_argument);
}

TEST(Simulator, FinishingOrderIsAPermutation) {
    std::mt19937_64 rng(1);
    std::vector<char> retired;
    auto order = simulate_once(field(20), Race{}, rng, retired);
    std::vector<int> sorted = order;
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < 20; ++i) EXPECT_EQ(sorted[i], i);
}
