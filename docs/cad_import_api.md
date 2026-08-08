# CAD Import Pipeline — API publique

Module : `nutella_scraper.cad_import`

## Responsabilité

Importer des fichiers **STEP** (et STL secondaire), produire un `CanonicalModel3D` (source de vérité pour tous les calculs) et générer des vues 2D **exclusivement visuelles**.

## Séparation calcul / visualisation

```
STEP ──► GeometryNormalizer ──► CanonicalModel3D ──► ModelStore
                                      │
                                      └──► ViewProjectionGenerator ──► ViewProjectionCache
                                                    (@visualization_only)
```

Les vues de côté (XZ, vue selon Y) et de dessus (XY, vue selon Z) **ne doivent jamais** alimenter les moteurs de simulation ou d'optimisation.

## Installation

```bash
pip install "nutella-scraper[cad_import]"
```

Dépendances : `trimesh`, `cascadio` (lecture STEP).

## Usage

```python
from pathlib import Path
from nutella_scraper.cad_import import (
    GeometryNormalizer,
    ImportPipeline,
    ModelStore,
    ViewCacheStore,
    ViewProjectionGenerator,
)

pipeline = ImportPipeline(
    normalizer=GeometryNormalizer(),
    model_store=ModelStore(Path("data/models")),
    view_generator=ViewProjectionGenerator(),
    view_cache_store=ViewCacheStore(Path("data/views")),
)

result = pipeline.import_step(Path("racloir.step"))
print(result.canonical.geometry.volume_mm3)
print(result.views_id)  # visualisation uniquement
```

## Classes publiques

### `ImportPipeline`

| Méthode | Entrée | Sortie | Description |
|---------|--------|--------|-------------|
| `import_step(path, generate_views=True)` | `.step`/`.stp` | `ImportResult` | Chemin principal d'import |
| `import_stl(path, generate_views=True)` | `.stl` | `ImportResult` | Format secondaire |
| `import_step_stl(step, stl)` | STEP ou STL | `ImportResult` | Fallback manuel |
| `import_sldprt(path)` | `.sldprt` | `ImportResult` | Non implémenté (lève `NotImplementedError`) |

### `GeometryNormalizer`

| Méthode | Sortie |
|---------|--------|
| `normalize_from_step(step_path)` | `CanonicalModel3D` |
| `normalize_from_stl(stl_path)` | `CanonicalModel3D` |
| `normalize(ExportPaths)` | `CanonicalModel3D` (prefère STEP) |

### `GeometryValidator`

Valide le maillage avant construction du modèle canonique. Lève `InvalidGeometryError` avec la liste des violations.

### `ViewProjectionGenerator`

| Méthode | Sortie | Usage |
|---------|--------|-------|
| `generate(CanonicalModel3D)` | `ViewProjectionCache` | Visualisation UI uniquement |

### `ModelStore` / `ViewCacheStore`

Persistance fichier :
- Modèles : `{base}/{model_id}/canonical.stl` + `metadata.json`
- Vues : `{base}/{views_id}/side.svg` (alias `profile.svg`) + `top.svg` + `manifest.json`

## Modèles de données

### `CanonicalModel3D` (calcul)

- `mesh` : maillage 3D portable
- `geometry` : `GeometricMetadata` (bbox, dimensions, centre, axes principaux, volume)
- `provenance` : `"canonical_3d"`

### `GeometricMetadata`

| Champ | Description |
|-------|-------------|
| `bounding_box` | Boîte englobante axis-aligned (mm) |
| `dimensions_mm` | `(dx, dy, dz)` |
| `center_mm` | Centroïde |
| `principal_axes` | 3 axes orthonormés (PCA) |
| `volume_mm3` | Volume si maillage watertight, sinon `None` |
| `is_watertight` | État du maillage |
| `vertex_count`, `face_count` | Compteurs |

### `ViewProjectionCache` (visualisation)

- `provenance` : `"visualization_projection"`
- `profile_view` : vue de côté (Side View), projection plan **XZ**, vue selon **Y**
- `top_view` : vue de dessus (Top View), projection plan **XY**, vue selon **Z**

Conventions CAD (orthographiques) :

| Vue | Plan | Axe de vue | Axes affichés |
|-----|------|------------|---------------|
| Side View / vue de côté | XZ | Y | X × Z |
| Top View / vue de dessus | XY | Z | X × Y |

## Exceptions

| Exception | Cas |
|-----------|-----|
| `CadImportError` | Base |
| `UnsupportedFormatError` | Extension non supportée |
| `StepReadError` | Échec lecture STEP |
| `InvalidGeometryError` | Maillage invalide (violations explicites) |

## Interface `IMeshLoader`

Contrat pour remplacer le backend de chargement (`TrimeshLoader` par défaut) sans modifier le pipeline.
