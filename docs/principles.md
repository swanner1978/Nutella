# Principes fondamentaux de l'architecture

Ces regles sont **non negociables** pour toute implementation future.

## P1 — Source de verite unique : le modele 3D canonique

- Le modele 3D issu de SolidWorks (STEP/STL) est la **seule source** pour tous les calculs geometriques.
- Toute geometrie importee est normalisee en `CanonicalModel3D`.
- Aucun calcul ne lit directement un fichier SLDPRT brut.

## P2 — Separation stricte Visualisation ≠ Calcul

- Les vues de côté (XZ, selon Y) et de dessus (XY, selon Z) sont `@visualization_only`.
- **Interdit** : utiliser une vue 2D comme entree du `ContactSimulator` ou de l'optimiseur.
- **Interdit** : deriver un score de couverture a partir de pixels.
- Le score affiche provient toujours du `ComputeEngine`.

## P3 — Trois moteurs independants

| Moteur | Peut lire | Ne peut pas lire |
|--------|-----------|------------------|
| ComputeEngine | CanonicalModel3D, JarCanonicalModel | ViewProjectionCache |
| VisualizationEngine | CanonicalModel3D, ContactResult | — ne recalcule jamais les metriques |
| OptimizationEngine | ComputeEngine via IEvaluator | VisualizationEngine, vues 2D |

## P4 — Flux unidirectionnel

```
SolidWorks → ImportPipeline → CanonicalModel3D → ComputeEngine → ContactResult
                                                                    ↓
                                              VisualizationEngine ← (read-only)
OptimizationEngine → ComputeEngine (jamais VisualizationEngine)
```

## P5 — Contrats types et tracabilite

- Chaque artefact porte un champ `provenance`.
- `ViewProjectionCache` et `CanonicalModel3D` sont dans des modules distincts.
- Tests import-linter verifient l'isolation des moteurs.

## P6 — Reproductibilite

- Chaque run logue : hash source, version maillage, parametres, seed.
- Les vues 2D ne sont jamais persistees comme reference de calcul.

## P7 — Pot Nutella en 3D

- `JarCanonicalModel` est un solide 3D pour la simulation de contact.
- Le profil JSON meridien sert a generer le 3D, pas a simuler en 2D.
