import { useState } from "react";
import type { ViewOverlayResponse } from "../types/api";

export interface SimulationState {
  overlay: ViewOverlayResponse | null;
  coverageScore: number | null;
  loading: boolean;
  error: string | null;
}

export function useSimulation(): SimulationState {
  const [overlay] = useState<ViewOverlayResponse | null>(null);
  const [coverageScore] = useState<number | null>(null);
  const [loading] = useState(false);
  const [error] = useState<string | null>(null);

  return { overlay, coverageScore, loading, error };
}
