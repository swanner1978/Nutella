# DECISIONS.md — Journal des decisions architecturales

## ADR-001 — Source de verite 3D (CanonicalModel3D)

**Date** : 2026-08-05  
**Statut** : Accepte

Tous les calculs geometriques utilisent exclusivement STEP/STL normalises en `CanonicalModel3D`. Les fichiers SLDPRT passent par le pipeline d'import avant tout calcul.

## ADR-002 — Separation Visualisation / Calcul

**Date** : 2026-08-05  
**Statut** : Accepte

Les vues de côté (XZ, selon Y) et de dessus (XY, selon Z) sont `@visualization_only`. Interdit d'utiliser des projections 2D pour contact, couverture ou optimisation. Enforced par import-linter.

## ADR-003 — Trois moteurs independants

**Date** : 2026-08-05  
**Statut** : Accepte

- `ComputeEngine` — calcul 3D
- `VisualizationEngine` — rendu 2D read-only
- `OptimizationEngine` — appelle ComputeEngine via `IEvaluator`, jamais VisualizationEngine

## ADR-004 — Package cad_import vs import

**Date** : 2026-08-05  
**Statut** : Accepte

Le module d'import CAD est nomme `cad_import` car `import` est un mot-cle Python.

## ADR-005 — Stack hybride Python + React

**Date** : 2026-08-05  
**Statut** : Accepte

- Noyau Python 3.11+ (Typer, FastAPI)
- UI React/Vite/TypeScript
- Communication via REST + WebSocket (simulation temps reel)

## ADR-006 — Fabrication FDM

**Date** : 2026-08-05  
**Statut** : Accepte

Contraintes FDM (epaisseur min, surplombs, clearance) dans `FDMPrintabilityChecker`, config `configs/default.yaml`.

## ADR-007 — SolidWorks export strategy

**Date** : 2026-08-05  
**Statut** : Accepte

- Mode auto : COM API Windows (`pywin32`)
- Mode fallback : STEP/STL pre-exportes manuellement

## ADR-008 — Persistence SQLite

**Date** : 2026-08-05  
**Statut** : Accepte

Runs d'optimisation et resultats de contact en SQLite. View caches separes (visualisation only).
