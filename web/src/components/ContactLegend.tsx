export function ContactLegend() {
  const items = [
    { color: "#0f8", label: "Zone touchée" },
    { color: "#f80", label: "Zone non touchée" },
    { color: "#888", label: "Racloir" },
    { color: "#fff", label: "Contour pot" },
  ];

  return (
    <div style={{ marginTop: "1rem", display: "flex", gap: "1.5rem", flexWrap: "wrap" }}>
      {items.map((item) => (
        <span key={item.label} style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <span
            style={{
              width: "12px",
              height: "12px",
              backgroundColor: item.color,
              borderRadius: "2px",
            }}
          />
          <span style={{ color: "#aaa", fontSize: "0.875rem" }}>{item.label}</span>
        </span>
      ))}
    </div>
  );
}
