# Import SolidWorks / STEP

Documentation complète de l'API : [cad_import_api.md](cad_import_api.md).

## Pipeline (implémenté)

1. `TrimeshLoader` — lecture STEP/STL
2. `GeometryValidator` — validation du maillage
3. `GeometryNormalizer` — construction `CanonicalModel3D` + `GeometricMetadata`
4. `ModelStore` — persistance
5. `ViewProjectionGenerator` — vues côté/dessus (Side/Top, `@visualization_only`)
6. `ImportPipeline` — orchestration

## Import STEP (chemin principal)

```bash
pip install "nutella-scraper[cad_import]"
```

```python
from pathlib import Path
from nutella_scraper.cad_import import ImportPipeline, GeometryNormalizer, ModelStore

pipeline = ImportPipeline(
    normalizer=GeometryNormalizer(),
    model_store=ModelStore(Path("data/models")),
)
result = pipeline.import_step(Path("racloir.step"))
```

## SLDPRT

Support prévu ultérieurement via `SolidWorksExporter` (COM Windows). Actuellement : `NotImplementedError`.
