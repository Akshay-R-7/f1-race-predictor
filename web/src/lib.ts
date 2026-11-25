import type { DriverPrediction, RaceListItem } from "./types";

/** 0.4567 -> "45.7%"; tiny values show as "<0.1%" so they don't look like zero. */
export function formatPct(p: number): string {
  if (!Number.isFinite(p) || p < 0) return "–";
  if (p > 0 && p < 0.001) return "<0.1%";
  return `${(p * 100).toFixed(1)}%`;
}

/** Highest win probability first; ties broken by better grid slot. */
export function sortByWinProb(drivers: DriverPrediction[]): DriverPrediction[] {
  return [...drivers].sort((a, b) => b.win_prob - a.win_prob || a.grid - b.grid);
}

/** Share of races where the favourite actually won. */
export function favouriteHitRate(races: RaceListItem[]): number {
  if (races.length === 0) return 0;
  return races.filter((r) => r.winner === r.predicted_favourite).length / races.length;
}

export function finishLabel(pos: number | null): string {
  return pos === null ? "DNF" : `P${pos}`;
}
