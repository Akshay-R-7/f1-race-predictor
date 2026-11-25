import type { RaceDetail, RaceListItem, Summary } from "./types";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: ${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

export const api = {
  summary: () => get<Summary>("/api/summary"),
  races: () => get<RaceListItem[]>("/api/races"),
  race: (id: string) => get<RaceDetail>(`/api/races/${encodeURIComponent(id)}`),
};
