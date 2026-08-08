import { CoverageScoreBar } from "./components/CoverageScoreBar";
import { SideView } from "./components/SideView";
import { TopView } from "./components/TopView";
import { ContactLegend } from "./components/ContactLegend";
import { OptimizationPanel } from "./components/OptimizationPanel";
import { useSimulation } from "./hooks/useSimulation";

export default function App() {
  const { overlay, coverageScore, loading, error } = useSimulation();

  return (
    <div style={{ padding: "1rem", backgroundColor: "#000" }}>
      <header style={{ marginBottom: "1rem" }}>
        <h1 style={{ margin: 0, color: "#fff" }}>Nutella Scraper</h1>
        <CoverageScoreBar score={coverageScore} loading={loading} />
      </header>

      {error && <p style={{ color: "#f66" }}>{error}</p>}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
        <SideView svgContent={overlay?.profile_svg ?? null} />
        <TopView svgContent={overlay?.top_svg ?? null} />
      </div>

      <ContactLegend />
      <OptimizationPanel />
    </div>
  );
}
