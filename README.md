# Nutella Scraper

Optimisation automatique de la geometrie d'un racloir FDM pour pot Nutella.

## Architecture

Trois moteurs independants :

- **ComputeEngine** — simulation de contact 3D, metriques (source : `CanonicalModel3D`)
- **VisualizationEngine** — vues côté/dessus (Side/Top), overlays (`@visualization_only`)
- **OptimizationEngine** — exploration paramétrique via ComputeEngine uniquement

Voir [docs/principles.md](docs/principles.md) pour les regles de conception.

## Structure

```
src/nutella_scraper/
├── domain/           # Modeles et protocoles
├── cad_import/       # Pipeline SolidWorks → CanonicalModel3D
├── engines/
│   ├── compute/      # ContactSimulationEngine, ObjectiveFunctions
│   ├── visualization/
│   └── optimization/
├── application/      # Orchestration, DI
├── io/               # Config, persistence, exporters
├── cli/
└── api/
web/                  # UI React (profil + dessus, fond noir)
```

## Installation

```bash
# Python
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e ".[dev]"

# Web UI
cd web && npm install
```

## Usage

```bash
# Version
nutella-scraper version

# API
nutella-scraper serve-api

# Web UI (autre terminal)
cd web && npm run dev
```

## Verification (squelette)

```bash
# Syntaxe Python (sans dependances)
python scripts/verify_syntax.py

# Imports complets (apres pip install -e ".[dev]")
python scripts/verify_imports.py

# Tests
pytest

# Regles d'architecture
lint-imports

# Web
cd web && npm run lint && npm run build
```

## Statut

Squelette architectural — logique metier non implementee. Voir les `NotImplementedError` dans le code.

| Composant | Module | Statut |
|-----------|--------|--------|
| CanonicalModel3D | `domain/models/canonical.py` | Contrats definis |
| CAD Import Pipeline | `cad_import/` | Stubs |
| DesignSpace | `domain/models/design_space.py` | Contrats definis |
| ContactSimulationEngine | `engines/compute/contact_simulator.py` | Stub |
| ObjectiveFunctions | `engines/compute/objective_functions.py` | Stub |
| OptimizationEngine | `engines/optimization/` | Stubs |
| VisualizationEngine | `engines/visualization/` | Stubs |
| Web UI | `web/` | Squelette React |
| Persistence | `io/persistence/` | Stubs |
| Configuration | `io/config_loader.py`, `configs/` | Fonctionnel (chargement YAML/JSON) |
