// Python bindings: `from f1sim import _racesim; _racesim.simulate(...)`
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "racesim.hpp"

namespace py = pybind11;
using namespace racesim;

PYBIND11_MODULE(_racesim, m) {
    m.doc() = "Fast multithreaded Monte Carlo F1 race simulator (C++)";
    m.def("simulate",
          [](const std::vector<double>& pace, const std::vector<double>& pace_sd,
             const std::vector<double>& dnf_prob, const std::vector<double>& deg,
             const std::vector<int>& grid, int laps, double pit_loss, double sc_prob_per_lap,
             double overtake_difficulty, double grid_gap, double lap_noise, std::int64_t n_sims,
             int n_threads, std::uint64_t seed) {
              size_t n = pace.size();
              if (pace_sd.size() != n || dnf_prob.size() != n || deg.size() != n || grid.size() != n)
                  throw std::invalid_argument("all driver lists must have the same length");
              std::vector<Driver> d(n);
              for (size_t i = 0; i < n; ++i) d[i] = Driver{pace[i], pace_sd[i], dnf_prob[i], deg[i], grid[i]};
              Race r; r.laps = laps; r.pit_loss = pit_loss; r.sc_prob_per_lap = sc_prob_per_lap;
              r.overtake_difficulty = overtake_difficulty;
              r.grid_gap = grid_gap; r.lap_noise = lap_noise;
              Summary s;
              {
                  py::gil_scoped_release release;  // let Python do other work while C++ runs
                  s = simulate(d, r, n_sims, n_threads, seed);
              }
              py::dict out;
              out["win_prob"] = s.win_prob; out["podium_prob"] = s.podium_prob;
              out["mean_finish"] = s.mean_finish; out["dnf_rate"] = s.dnf_rate;
              out["n_sims"] = s.n_sims;
              return out;
          },
          py::arg("pace"), py::arg("pace_sd"), py::arg("dnf_prob"), py::arg("deg"), py::arg("grid"),
          py::arg("laps"), py::arg("pit_loss") = 22.0, py::arg("sc_prob_per_lap") = 0.004,
          py::arg("overtake_difficulty") = 0.4, py::arg("grid_gap") = 0.25,
          py::arg("lap_noise") = 0.3, py::arg("n_sims") = 100000,
          py::arg("n_threads") = 0, py::arg("seed") = 42);
}
