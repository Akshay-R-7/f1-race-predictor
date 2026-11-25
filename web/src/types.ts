export interface DriverPrediction {
  driver: string;
  team: string;
  grid: number;
  win_prob: number;
  sim_win_prob?: number;
  podium_prob: number;
  mean_finish: number;
  dnf_prob: number;
  actual_finish: number | null;
}

export interface RaceListItem {
  race_id: string;
  season: number;
  round: number;
  name: string;
  winner: string;
  predicted_favourite: string;
}

export interface RaceDetail extends RaceListItem {
  drivers: DriverPrediction[];
}

export interface Scores { top1: number; log_loss: number; brier: number }

export interface Summary {
  test_seasons: string;
  test_races: number;
  model: Scores;
  simulator_only?: Scores;
  baseline_pole_wins: Scores;
  baseline_grid_prior: Scores;
}
