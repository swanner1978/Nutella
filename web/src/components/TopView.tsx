interface TopViewProps {
  svgContent: string | null;
}

export function TopView({ svgContent }: TopViewProps) {
  return (
    <section
      aria-label="Vue dessus"
      style={{
        backgroundColor: "#000",
        border: "1px solid #333",
        borderRadius: "4px",
        minHeight: "400px",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      {svgContent ? (
        <div dangerouslySetInnerHTML={{ __html: svgContent }} />
      ) : (
        <span style={{ color: "#666" }}>
          Vue de dessus (Top View) — plan XY, selon Z — en attente de simulation
        </span>
      )}
    </section>
  );
}
