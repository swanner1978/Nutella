/** Domain types mirroring API contracts. */

export interface SimulateRequest {
  model_id: string;
  jar_id?: string;
  simulation_profile?: string;
}

export interface SimulateResponse {
  coverage_score: number;
  contact_result_id?: string | null;
  feasible: boolean;
}

export interface ViewOverlayResponse {
  model_id: string;
  profile_svg: string;
  top_svg: string;
  /** Read-only copy of ComputeEngine coverage_score — never computed from pixels. */
  coverage_score_display: number;
}

export interface OptimizationRequest {
  jar_id?: string;
  design_space_id?: string;
  optimization_profile?: string;
}

export interface OptimizationResponse {
  run_id: string;
  status: string;
}

export interface ImportResponse {
  model_id: string;
  views_id?: string | null;
  source_hash: string;
}

export type ContactZone = "touched" | "untouched" | "scraper" | "jar";

export interface ViewLayer {
  id: string;
  zone: ContactZone;
  svgFragment: string;
}
