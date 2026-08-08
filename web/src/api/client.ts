import type {
  ImportResponse,
  OptimizationRequest,
  OptimizationResponse,
  SimulateRequest,
  SimulateResponse,
  ViewOverlayResponse,
} from "../types/api";

const API_BASE = "/v1";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!response.ok) {
    throw new Error(`API error ${response.status}: ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string; version: string }>("/health"),

  simulateContact: (body: SimulateRequest) =>
    request<SimulateResponse>("/simulate/contact", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  getViewOverlay: (modelId: string, contactResultId: string) =>
    request<ViewOverlayResponse>(
      `/visualization/models/${modelId}/overlay?contact_result_id=${contactResultId}`,
    ),

  startOptimization: (body: OptimizationRequest) =>
    request<OptimizationResponse>("/optimization/runs", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  importSolidWorks: (body: { stl_path?: string; step_path?: string; sldprt_path?: string }) =>
    request<ImportResponse>("/import/solidworks", {
      method: "POST",
      body: JSON.stringify(body),
    }),
};
