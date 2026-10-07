# BIOMASS MAAP Hackathon 2026

Shared workspace for the **ESA Biomass MAAP Hackathon** (12–16 October 2026, ESOC, Darmstadt) - project templates, notebooks and resources.

## About

| | |
|---|---|
| **Event** | ESA Biomass MAAP Hackathon 2026 |
| **Dates** | 12–16 October 2026 (5 days, in-person) |
| **Location** | ESOC (European Space Operations Centre), Darmstadt, Germany |
| **Format** | Hands-on coding · English · Python |
| **Goal** | Evolve the BioPAL open-source processor suite ([BPS](https://github.com/BioPAL/BPS)) and the MAAP platform through collaborative coding |
| **Git support** | ACRI-ST - Git workflows & Q&A, on-site all week |

This repository is the shared home for all hackathon project groups. Add a folder, bring your code, and don't hesitate to ask for help with Git - that's what we're here for.

## Repository structure

```
biomass-hackathon-2026/
├── README.md
├── resources/            ← shared code, data-access helpers, course material links
└── topics/
    ├── 01-polsar-polinsar-analytics/
    ├── 02-rfi-removal/
    ├── 03-qgis-plugin/
    ├── 04-3d-forest-structure/
    ├── 05-biomass-gedi-intercomparison/
    ├── 06-biomass-validation/
    ├── 07-biomass-geology/
    ├── 08-biomass-cryosphere/
    ├── 09-biomass-ocean/
    ├── 10-biomass-ionosphere/
    ├── 11-biomass-biodiversity/
    └── 12-forest-height-tomosar/
```

Each project group works in its own folder under `topics/`. If yours doesn't exist yet, create it — see **How to contribute** below.

## Hackathon topics

| ID | Topic | Folder | Contact(s) |
|----|-------|--------|------------|
| 1 | PolSAR and PolInSAR Analytics | `topics/01-polsar-polinsar-analytics/` | Armando |
| 2 | RFI removal | `topics/02-rfi-removal/` | Francesco |
| 3 | QGIS plugin | `topics/03-qgis-plugin/` | Christiano |
| 4 | 3D forest structure visualisation | `topics/04-3d-forest-structure/` | Francesco (PolInSAR course code) |
| 5 | BIOMASS and GEDI intercomparison | `topics/05-biomass-gedi-intercomparison/` | — |
| 6 | BIOMASS validation | `topics/06-biomass-validation/` | Klaus (protocol) |
| 7 | BIOMASS for Geology | `topics/07-biomass-geology/` | Armando, Francesco |
| 8 | BIOMASS for Cryosphere | `topics/08-biomass-cryosphere/` | Armando, Francesco |
| 11 | BIOMASS for biodiversity | `topics/11-biomass-biodiversity/` | — |
| 12 | Forest Height from TomoSAR | `topics/12-forest-height-tomosar/` | — |

Full task descriptions and data sources for each topic: see the hackathon brief shared before the event.

## How to contribute

**1. Clone the repository**

```bash
git clone https://github.com/BioPAL/biomass-hackathon-2026.git
cd biomass-hackathon-2026
```

**2. Create your topic folder** (skip if it already exists)

```bash
mkdir -p topics/04-3d-forest-structure
```

**3. Work on a branch** — keeps `main` clean and avoids conflicts between groups

```bash
git checkout -b 04-3d-forest-structure/initial-notebook
```

**4. Commit often, with clear messages**

```bash
git add topics/04-3d-forest-structure/
git commit -m "Add first tomographic processing notebook for Lopé ALS data"
```

**5. Push your branch and open a Pull Request**

```bash
git push -u origin 04-3d-forest-structure/initial-notebook
```

Then open a Pull Request on GitHub targeting `main`. 

### Ground rules

- **No large data files** in the repo (point clouds, imagery, …) - link to the MAAP dataset or add a small download script instead.
- **No credentials or API keys**, ever.
- One folder per topic group; put shared/reusable code in `resources/` if several groups need it.
- Stuck on Git? ACRI-ST is on-site all week for exactly this — just ask.

## Data & resources

- [MAAP platform](https://maap-project.org/) - BIOMASS data, Brazil SFB ALS acquisitions, compute environment
- [PolSARpro](https://polsarpro.readthedocs.io/) — PolSAR/PolInSAR processing
- GEDI data — via MAAP or [gediDB](https://gedidb.readthedocs.io/)

## License

Code in this repository is released under the Apache License 2.0, matching [BioPAL/BPS](https://github.com/BioPAL/BPS).
