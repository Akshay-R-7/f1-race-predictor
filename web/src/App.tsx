import { useEffect, useState } from "react";
import { api } from "./api";
import { favouriteHitRate, finishLabel, formatPct, sortByWinProb } from "./lib";
import type { RaceDetail, RaceListItem, Scores, Summary } from "./types";

function ScoreRow({ label, s, best }: { label: string; s: Scores; best?: boolean }) {
  return (
    <tr className={best ? "best" : undefined}>
      <td>{label}</td>
      <td>{formatPct(s.top1)}</td>
      <td>{s.log_loss.toFixed(3)}</td>
      <td>{s.brier.toFixed(3)}</td>
    </tr>
  );
}

export default function App() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [races, setRaces] = useState<RaceListItem[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [race, setRace] = useState<RaceDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.summary(), api.races()])
      .then(([s, r]) => {
        setSummary(s);
        setRaces(r);
        if (r.length) setSelected(r[r.length - 1].race_id);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!selected) return;
    api.race(selected).then(setRace).catch((e: Error) => setError(e.message));
  }, [selected]);

  if (error) return <main><h1>F1 Race Predictor</h1><p role="alert">Could not load data: {error}</p></main>;
  if (!summary) return <main><h1>F1 Race Predictor</h1><p>Loading…</p></main>;

  return (
    <main>
      <h1>F1 Race Predictor</h1>
      <p className="muted">
        Win probabilities on held-out seasons {summary.test_seasons} ({summary.test_races} races).
        Favourite won {formatPct(favouriteHitRate(races))} of races.
      </p>

      <section>
        <h2>How good is it?</h2>
        <table className="scores">
          <thead><tr><th>Method</th><th>Favourite wins</th><th>Log loss ↓</th><th>Brier ↓</th></tr></thead>
          <tbody>
            <ScoreRow label="Final forecast" s={summary.model} best />
            {summary.simulator_only && <ScoreRow label="Simulator only" s={summary.simulator_only} />}
            <ScoreRow label="Baseline: grid position" s={summary.baseline_grid_prior} />
            <ScoreRow label="Baseline: pole always wins" s={summary.baseline_pole_wins} />
          </tbody>
        </table>
      </section>

      <section>
        <label>
          Race{" "}
          <select value={selected} onChange={(e) => setSelected(e.target.value)} aria-label="race">
            {races.map((r) => (
              <option key={r.race_id} value={r.race_id}>{r.season} R{r.round} – {r.name}</option>
            ))}
          </select>
        </label>
        {race && (
          <table className="drivers">
            <thead>
              <tr><th>Driver</th><th>Team</th><th>Grid</th><th>Win chance</th><th>Podium (sim)</th><th>Result</th></tr>
            </thead>
            <tbody>
              {sortByWinProb(race.drivers).map((d) => (
                <tr key={d.driver} className={d.driver === race.winner ? "winner" : undefined}>
                  <td>{d.driver}</td>
                  <td>{d.team}</td>
                  <td>{d.grid}</td>
                  <td>
                    <div className="bar"><span style={{ width: `${Math.min(100, d.win_prob * 100)}%` }} /></div>
                    {formatPct(d.win_prob)}
                  </td>
                  <td>{formatPct(d.podium_prob)}</td>
                  <td>{finishLabel(d.actual_finish)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </main>
  );
}
