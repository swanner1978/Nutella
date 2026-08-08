# AGENTS.md — Guide pour agents et contributeurs

## Contexte projet

Application hybride (CLI + API + UI) pour optimiser la geometrie d'un racloir Nutella imprimable en FDM.

## Regles absolues

1. **Calcul 3D uniquement** — `CanonicalModel3D` (STEP/STL) est la seule source pour contact, couverture, optimisation.
2. **Vues 2D de reference = CAD B-Rep** — `CadReferenceGeometry` (STEP/OpenCascade) alimente profil, dessus, contour interieur et enveloppe interne. Jamais Trimesh/STL/mesh pour ces vues.
3. **Mesh = calcul numerique** — la tessellation (`CanonicalModel3D.mesh`) sert contact, collision, distance — pas les vues 2D de reference.
4. **Vues 2D cache = visualisation** — `ViewProjectionCache` ne doit jamais alimenter ComputeEngine ou OptimizationEngine.
5. **Trois moteurs decouples** — pas d'import croise entre `engines/visualization` et `engines/optimization`.
6. **Pas de raccourci** — le score de couverture affiche provient de `ContactResult.coverage_score`, jamais de pixels.

## Modules cles

| Module | Chemin | Role |
|--------|--------|------|
| CanonicalModel3D | `domain/models/canonical.py` | Modele 3D normalise (mesh de calcul) |
| CadReferenceGeometry | `domain/models/cad_reference_geometry.py` | Geometrie CAD B-Rep de reference (vues 2D) |
| CAD Import | `cad_import/` | SLDPRT → STEP/STL → CanonicalModel3D + CadReferenceGeometry |
| DesignSpace | `domain/models/design_space.py` | Espace paramétrique |
| ContactSimulationEngine | `engines/compute/contact_simulator.py` | Simulation contact 3D |
| ObjectiveFunctions | `engines/compute/objective_functions.py` | Objectifs multi-criteres |
| ComputeEngine | `engines/compute/engine.py` | Facade calcul |
| VisualizationEngine | `engines/visualization/engine.py` | Vues côté/dessus (Side/Top) |
| OptimizationEngine | `engines/optimization/engine.py` | Optimisation |
| Persistence | `io/persistence/` | SQLite + view cache |
| Configuration | `io/config_loader.py`, `configs/` | YAML/JSON |

## Ou implementer quoi (prochaines phases)

- **Phase 1** : `GeometryNormalizer`, `ContactSimulationEngine`, `CoverageScorer`
- **Phase 2** : `ViewProjectionGenerator`, `ContactResultProjector`, UI temps reel
- **Phase 3** : `OptimizerRunner`, jobs async API
- **Phase 4** : export 3MF, tests golden

## Conventions

- Protocoles dans `domain/protocols/`
- DI via `application/container.py`
- Tests miroir de la structure `src/`
- `NotImplementedError` pour stubs en attente d'implementation

## Verification avant PR

```bash
pytest
lint-imports
ruff check src tests
cd web && npm run lint
```
