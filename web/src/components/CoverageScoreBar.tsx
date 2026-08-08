interface CoverageScoreBarProps {
  score: number | null;
  loading?: boolean;
}

export function CoverageScoreBar({ score, loading }: CoverageScoreBarProps) {
  const display =
    loading ? "Calcul en cours..." : score !== null ? `${(score * 100).toFixed(1)} %` : "—";

  return (
    <div
      style={{
        marginTop: "0.5rem",
        padding: "0.5rem 1rem",
        backgroundColor: "#111",
        borderRadius: "4px",
        display: "inline-block",
      }}
    >
      <span style={{ color: "#888" }}>Couverture : </span>
      <strong style={{ color: "#0cf" }}>{display}</strong>
    </div>
  );
}
