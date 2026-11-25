import { describe, expect, it } from "vitest";
import { favouriteHitRate, finishLabel, formatPct, sortByWinProb } from "./lib";
import type { DriverPrediction } from "./types";

const d = (driver: string, win_prob: number, grid: number): DriverPrediction => ({
  driver, team: "T", grid, win_prob, podium_prob: 0, mean_finish: 0, dnf_prob: 0, actual_finish: null,
});

describe("formatPct", () => {
  it("formats normal values", () => expect(formatPct(0.4567)).toBe("45.7%"));
  it("does not show tiny chances as zero", () => expect(formatPct(0.0004)).toBe("<0.1%"));
  it("handles bad input", () => expect(formatPct(NaN)).toBe("–"));
});

describe("sortByWinProb", () => {
  it("sorts by probability, then grid", () => {
    const out = sortByWinProb([d("B", 0.2, 3), d("A", 0.5, 2), d("C", 0.2, 1)]);
    expect(out.map((x) => x.driver)).toEqual(["A", "C", "B"]);
  });
  it("does not mutate its input", () => {
    const input = [d("B", 0.1, 2), d("A", 0.9, 1)];
    sortByWinProb(input);
    expect(input[0].driver).toBe("B");
  });
});

describe("favouriteHitRate", () => {
  it("counts races where the favourite won", () => {
    const r = (w: string, f: string) => ({ race_id: "x", season: 2024, round: 1, name: "n", winner: w, predicted_favourite: f });
    expect(favouriteHitRate([r("A", "A"), r("A", "B")])).toBe(0.5);
    expect(favouriteHitRate([])).toBe(0);
  });
});

it("labels retirements", () => {
  expect(finishLabel(null)).toBe("DNF");
  expect(finishLabel(3)).toBe("P3");
});
