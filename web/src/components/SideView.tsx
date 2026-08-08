interface SideViewProps {
  svgContent: string | null;
}

export function SideView({ svgContent }: SideViewProps) {
  return (
    <section
      aria-label="Vue de côté"
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
          Vue de côté (Side View) — plan XZ, selon Y — en attente de simulation
        </span>
      )}
    </section>
  );
}
