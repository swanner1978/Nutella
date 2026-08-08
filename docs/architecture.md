# Architecture

Voir le plan complet dans `.cursor/plans/` et [principles.md](principles.md).

## Couches

1. **Presentation** — CLI, API FastAPI, UI React
2. **Application** — Orchestrator, SimulationService, Container (DI)
3. **Engines** — Compute, Visualization, Optimization
4. **CAD Import** — SolidWorks pipeline
5. **Domain** — Modeles immuables, protocoles
6. **IO** — Config, persistence, export

## Flux principal

```
SLDPRT → cad_import → CanonicalModel3D → ComputeEngine → ContactResult
                                              ↓
                              VisualizationEngine (projection)
                                              ↓
                                           Web UI
```

## Contrats inter-modules

Documentes dans `domain/protocols/` et `domain/models/`.
