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
├── resources/                 shared code reusable across groups
│   ├── load/                  read products → arrays / xarray
│   ├── process/               arrays in → arrays out
│   ├── evaluate/              metrics, validation, stats
│   ├── plot/                  figures, maps
│   └── tests/                 pytest for shared code
├── topics/                    one folder per group (same name as the GitHub label)
│   ├── 02-RFI/
│   ├── 03-QGIS/
│   ├── 04-3D-TOMO-CANOPY-VISUALIZER/
│   ├── 05_BIOMASS_AND_GEDI/
│   ├── 06A-CARBON/
│   ├── 06B-CARBON/
│   ├── 07-GEOLOGY/
│   ├── 08-ICE-QUEEN/
│   ├── 09-BIODIVERSITY/
│   ├── 10-TOMO-SAR-FOREST-HEIGHT/
│   ├── 11-A-SOIL-MOISTURE/
│   └── 11-B-FLOODED-FOREST/
└── non-chosen-topics/         briefs not running as a group this week
```

Each project group works in `topics/<GROUP-LABEL>/` — the folder name matches the GitHub group label. Put **shared** helpers in `resources/` (see [resources/README.md](resources/README.md)).

## Hackathon groups

| Group label | Topic | Folder |
| --- | --- | --- |
| `02-RFI` | RFI removal | `topics/02-RFI/` |
| `03-QGIS` | QGIS plugin | `topics/03-QGIS/` |
| `04-3D-TOMO-CANOPY-VISUALIZER` | 3D tomographic canopy visualiser | `topics/04-3D-TOMO-CANOPY-VISUALIZER/` |
| `05_BIOMASS_AND_GEDI` | BIOMASS and GEDI intercomparison | `topics/05_BIOMASS_AND_GEDI/` |
| `06A-CARBON` | Carbon map validation (table A) | `topics/06A-CARBON/` |
| `06B-CARBON` | Carbon map validation (table B) | `topics/06B-CARBON/` |
| `07-GEOLOGY` | BIOMASS for Geology | `topics/07-GEOLOGY/` |
| `08-ICE-QUEEN` | BIOMASS for Cryosphere | `topics/08-ICE-QUEEN/` |
| `09-BIODIVERSITY` | BIOMASS for biodiversity | `topics/09-BIODIVERSITY/` |
| `10-TOMO-SAR-FOREST-HEIGHT` | Forest height from TomoSAR | `topics/10-TOMO-SAR-FOREST-HEIGHT/` |
| `11-A-SOIL-MOISTURE` | Soil moisture | `topics/11-A-SOIL-MOISTURE/` |
| `11-B-FLOODED-FOREST` | Flooded forest | `topics/11-B-FLOODED-FOREST/` |

## Git from MAAP (use a Personal Access Token)

On **MAAP** (and often on ESA public Wi‑Fi):

- **SSH is blocked** (`git@github.com:...` will fail)
- You usually **cannot install** or use `gh auth login`
- Use **HTTPS + a GitHub Personal Access Token (PAT)** — that is the supported way to `git clone` / `git pull` / `git push`

### 1. Create a classic PAT

1. Open [github.com/settings/tokens](https://github.com/settings/tokens) (while logged into GitHub).
2. **Generate new token** → **Generate new token (classic)**.
3. Note: e.g. `maap-hackathon`.
4. Expiration: pick a short period (e.g. 7 or 30 days).
5. Scopes: tick **`repo`** (full control of private repositories — needed for private forks; harmless for this public hackathon).
6. **Generate token**, then **copy it immediately** (you will not see it again).

Fine-grained tokens also work: grant your fork **Contents: Read and write** (and **Metadata: Read**).

### 2. Clone / pull / push with the token

Always use an **HTTPS** URL (never `git@github.com:`):

```bash
git clone https://github.com/YOUR-USERNAME/biomass-hackathon-2026.git
cd biomass-hackathon-2026
```

When Git asks for credentials:

| Prompt | Enter |
| --- | --- |
| **Username** | your GitHub username |
| **Password** | the **PAT** (not your GitHub account password) |

Example push after you commit:

```bash
git push -u origin YOUR-BRANCH
# Username: YOUR-USERNAME
# Password: ghp_xxxxxxxx        ← paste the token
```

If Git never asks and fails with `Authentication failed` / `Permission denied`, clear old cached passwords, then retry (or embed the token only for this session — do not commit it):

```bash
# macOS / Linux credential helper reset examples — ask on-site support if unsure
git credential reject <<EOF
protocol=https
host=github.com
EOF
```

**Do not** paste the token into the README, notebooks, or chat logs. Revoke it at [github.com/settings/tokens](https://github.com/settings/tokens) if it leaks.

More Git workflow detail: [CONTRIBUTING.md](CONTRIBUTING.md).

## Contributing

Work from a **fork** (one fork per person; you do not push directly to `main`):

1. Create a **Task** issue and pick your **Group** (label is applied automatically).
2. Create a **PAT** (section above), fork [BioPAL/biomass-hackathon-2026](https://github.com/BioPAL/biomass-hackathon-2026), clone **your** fork with **HTTPS**, create a branch.
3. When you start the work, open a **draft** pull request early with `Closes #<issue-number>` in the description.
4. When ready for a maintainer, mark the PR **Ready for review**. A maintainer merges (do not merge yourself).

- Group-specific code and notebooks → `topics/<GROUP-LABEL>/`
- Code useful to several teams → `resources/` (load / process / evaluate / plot)

Details, naming conventions, and ground rules: [CONTRIBUTING.md](CONTRIBUTING.md).

## Project board (Kanban)

Public board: [Hackaton 2026 Board](https://github.com/orgs/BioPAL/projects/6)

The board shows **Task issues**, not pull requests. Example: [PR #93](https://github.com/BioPAL/biomass-hackathon-2026/pull/93) moves **issue #68** on the board (via `Closes #68` in the PR description) — you will not see card “#93”.

You do **not** need to drag cards — Status updates from GitHub events:

| Event | Status |
| --- | --- |
| Task issue opened | **Todo** (or Backlog) |
| **Draft** PR opened/updated with `Closes #N` | **Doing** (or Started) |
| PR marked **Ready for review** (or opened non-draft) | **In review** |
| PR merged / issue closed | **Done** |

Put `Closes #<issue-number>` in the PR description **when you open it** (or edit the description later — that also re-syncs). Open the draft PR as soon as you start so the board shows **Doing**.

## Data and resources

- [MAAP platform](https://maap-project.org/), BIOMASS data, Brazil SFB ALS acquisitions, compute environment
- [PolSARpro](https://polsarpro.readthedocs.io/), PolSAR and PolInSAR processing
- GEDI data, via MAAP or [gediDB](https://gedidb.readthedocs.io/)
- Lope TLS and ALS datasets, ask the topic referent
- PolInSAR course example code (tomographic processor), ask the topic referent
- BIOMASS validation protocol, ask the topic referent

## License

Code in this repository is released under the Apache License 2.0, matching [BioPAL/BPS](https://github.com/BioPAL/BPS).
