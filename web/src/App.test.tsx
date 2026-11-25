import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import App from "./App";

const summary = { test_seasons: "2024-2025", test_races: 1,
  model: { top1: 0.5, log_loss: 1.3, brier: 0.6 }, baseline_pole_wins: { top1: 0.5, log_loss: 4, brier: 0.9 },
  baseline_grid_prior: { top1: 0.5, log_loss: 1.3, brier: 0.6 } };
const races = [{ race_id: "2024-01", season: 2024, round: 1, name: "Bahrain", winner: "VER", predicted_favourite: "VER" }];
const detail = { ...races[0], drivers: [
  { driver: "VER", team: "Redline", grid: 1, win_prob: 0.6, podium_prob: 0.9, mean_finish: 1.5, dnf_prob: 0.05, actual_finish: 1 },
  { driver: "HAM", team: "Silverstar", grid: 2, win_prob: 0.3, podium_prob: 0.7, mean_finish: 2.5, dnf_prob: 0.05, actual_finish: null }] };

afterEach(() => vi.restoreAllMocks());

it("loads the summary and the latest race", async () => {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (url) => {
    const body = String(url).endsWith("/summary") ? summary : String(url).endsWith("/races") ? races : detail;
    return new Response(JSON.stringify(body), { status: 200 });
  });
  render(<App />);
  await waitFor(() => expect(screen.getByText("Redline")).toBeTruthy());
  expect(screen.getByText("60.0%")).toBeTruthy();
  expect(screen.getByText("DNF")).toBeTruthy();
});

it("shows an error when the API is down", async () => {
  vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("", { status: 503, statusText: "Service Unavailable" }));
  render(<App />);
  await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("503"));
});
