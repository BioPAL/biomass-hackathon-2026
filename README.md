# BIOMASS MAAP Hackathon 2026

Shared workspace for the **ESA Biomass MAAP Hackathon** (12 to 16 October 2026, ESOC, Darmstadt): project folders, notebooks and resources.

## About

|                 |                                                                                                                                         |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **Event**       | ESA Biomass MAAP Hackathon 2026                                                                                                         |
| **Dates**       | 12 to 16 October 2026 (5 days, in person)                                                                                               |
| **Location**    | ESOC (European Space Operations Centre), Darmstadt, Germany                                                                             |
| **Format**      | Hands on coding, English, Python                                                                                                        |
| **Goal**        | Evolve the BioPAL open source processor suite ([BPS](https://github.com/BioPAL/BPS)) and the MAAP platform through collaborative coding |
| **Git support** | ACRI-ST, Git workflows and Q&A, on site all week                                                                                        |

This repository is the shared home for all hackathon project groups. Add a folder, bring your code, and do not hesitate to ask for help with Git, that is what we are here for.

## Repository structure

```
biomass-hackathon-2026/
├── README.md
├── CONTRIBUTING.md
├── resources/                 shared code reusable across topics
│   ├── load/                  read products → arrays / xarray
│   ├── process/               arrays in → arrays out
│   ├── evaluate/              metrics, validation, stats
│   ├── plot/                  figures, maps
│   └── tests/                 pytest for shared code
└── topics/
    ├── 01-polsar-polinsar-analytics/
    ├── 02-rfi-removal/
    ├── 03-qgis-plugin/
    ├── 04-3d-forest-structure/
    ├── 05-biomass-gedi-intercomparison/
    ├── 06-biomass-validation/
    ├── 07-biomass-geology/
    ├── 08-biomass-cryosphere/
    ├── 11-biomass-biodiversity/
    └── 12-forest-height-tomosar/
```

Each project group works mainly in its own folder under `topics/`. Put **shared** helpers in `resources/` (see [resources/README.md](resources/README.md)). Every topic folder already has a README with the challenge title, goal, roadmap (foundation and deep dive), and data sources from the hackathon brief to get you started.

## Hackathon topics

| ID  | Topic                             | Folder                                    | Contact                          |
| --- | --------------------------------- | ----------------------------------------- | -------------------------------- |
| 1   | PolSAR and PolInSAR Analytics     | `topics/01-polsar-polinsar-analytics/`    | Topic referent                   |
| 2   | RFI removal                       | `topics/02-rfi-removal/`                  | Topic referent                   |
| 3   | QGIS plugin                       | `topics/03-qgis-plugin/`                  | Topic referent                   |
| 4   | 3D forest structure visualisation | `topics/04-3d-forest-structure/`          | Topic referent                   |
| 5   | BIOMASS and GEDI intercomparison  | `topics/05-biomass-gedi-intercomparison/` | not assigned                     |
| 6   | BIOMASS validation                | `topics/06-biomass-validation/`           | Topic referent                   |
| 7   | BIOMASS for Geology               | `topics/07-biomass-geology/`              | Topic referent                   |
| 8   | BIOMASS for Cryosphere            | `topics/08-biomass-cryosphere/`           | Topic referent                   |
| 11  | BIOMASS for biodiversity          | `topics/11-biomass-biodiversity/`         | not assigned                     |
| 12  | Forest Height from TomoSAR        | `topics/12-forest-height-tomosar/`        | not assigned                     |

## Contributing

Work from a **fork** (you do not push directly to `main`):

1. Fork [BioPAL/biomass-hackathon-2026](https://github.com/BioPAL/biomass-hackathon-2026) on GitHub.
2. Clone **your fork**, create a branch, commit your work.
3. Push to your fork and open a pull request targeting `BioPAL/biomass-hackathon-2026` branch `main`.

- Topic-specific code and notebooks → `topics/<num>-<name>/`
- Code useful to several teams → `resources/` (load / process / evaluate / plot)

Details, naming conventions, and ground rules: [CONTRIBUTING.md](CONTRIBUTING.md).

## Data and resources

- [MAAP platform](https://maap-project.org/), BIOMASS data, Brazil SFB ALS acquisitions, compute environment
- [PolSARpro](https://polsarpro.readthedocs.io/), PolSAR and PolInSAR processing
- GEDI data, via MAAP or [gediDB](https://gedidb.readthedocs.io/)
- Lope TLS and ALS datasets, ask the topic referent
- PolInSAR course example code (tomographic processor), ask the topic referent
- BIOMASS validation protocol, ask the topic referent

## License

Code in this repository is released under the Apache License 2.0, matching [BioPAL/BPS](https://github.com/BioPAL/BPS).
