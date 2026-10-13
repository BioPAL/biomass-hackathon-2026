# Contributing

Thanks for contributing to the ESA Biomass MAAP Hackathon shared repository. This guide is for hackathon participants, 12 to 16 October at ESOC.

## Hackathon vs BioPAL/BPS

This is a **hackathon** repo, not the main BPS production codebase. This week:

- No signed commits required
- No mandatory CI gate on pull requests
- A maintainer reviews and merges; do not merge your own PR

[BioPAL/BPS](https://github.com/BioPAL/BPS) has stricter rules (approval gates, CODEOWNERS, and related checks). None of that applies here unless you are contributing to BPS itself.

## How a contribution flows

You do not have write access to this repository, so you will work from your own fork.

1. Create (or pick) a **Task** issue on GitHub and choose your **Group** (the group label is applied automatically).
2. Fork the repository to your own GitHub account (**one fork per person**).
3. Clone **your fork**, create a branch, work in your group folder (or in `resources/` for shared code).
4. Commit, push to your fork.
5. Open a **draft** pull request early: **base** = `BioPAL/biomass-hackathon-2026` / `main`, **head** = your fork + your branch. Put `Closes #<issue-number>` in the PR description (moves the board card to **Doing**).
6. When ready, mark the PR **Ready for review** (moves the card to **In review**). A maintainer merges.

`main` is protected: every change goes through a pull request. Do not merge your own pull request.

Board Status is automated (Todo → Doing → In review → Done). See [README.md](README.md#project-board-kanban).

## Getting set up with Git

**1. Fork the repository**

Open [github.com/BioPAL/biomass-hackathon-2026](https://github.com/BioPAL/biomass-hackathon-2026) and click "Fork" in the top right. This creates a copy under your own GitHub account.

**2. Clone your fork**

```bash
git clone https://github.com/YOUR-USERNAME/biomass-hackathon-2026.git
cd biomass-hackathon-2026
```

Replace `YOUR-USERNAME` with your GitHub username.

**3. Add the original repository as a remote**

This lets you pull in changes made by others later.

```bash
git remote add upstream https://github.com/BioPAL/biomass-hackathon-2026.git
```

**4. Create a branch**

Prefer `<GROUP-LABEL>/<short-description>` (same string as your group folder / GitHub label):

```bash
git checkout -b 04-3D-TOMO-CANOPY-VISUALIZER/short-description
```

For shared-code work, `resources/<short-description>` is fine.

**5. Work in your group folder**

Folder names under `topics/` match the GitHub group labels (see the table in [README.md](README.md)). Example:

```bash
cd topics/04-3D-TOMO-CANOPY-VISUALIZER
```

**6. Commit and push to your fork**

`origin` points to your fork, not to `BioPAL/biomass-hackathon-2026`, so this push goes to your own copy.

```bash
git add topics/04-3D-TOMO-CANOPY-VISUALIZER/
git commit -m "Add first tomographic processing notebook"
git push -u origin 04-3D-TOMO-CANOPY-VISUALIZER/short-description
```

**7. Open a draft pull request (early)**

Go to your fork on github.com. GitHub shows a banner suggesting a pull request for your new branch; click it. Check that:

- Base repository: `BioPAL/biomass-hackathon-2026`
- Base branch: `main`
- Head: your fork and branch
- Description includes `Closes #<issue-number>` (one PR = one issue)
- Create as **Draft** while you are still coding (board → **Doing**)

When the work is ready, click **Ready for review**. A maintainer will review and merge it.

**Keeping your fork up to date** (optional, useful if the hackathon runs for several days)

```bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

## Where to put code: `topics/` vs `resources/`

| Put it in… | When… |
| --- | --- |
| `topics/<GROUP-LABEL>/` | Notebooks, scripts, and helpers for **one** group only (folder = GitHub label) |
| `resources/load/` | Shared readers: products → arrays / xarray (no plotting) |
| `resources/process/` | Shared transforms: arrays in → arrays out (no plotting) |
| `resources/evaluate/` | Shared metrics, validation, stats |
| `resources/plot/` | Shared figures and maps |
| `resources/tests/` | `pytest` for shared helpers |

**Promote to `resources/`** when a second team needs the same helper, or when the function is clearly generic (e.g. a common MAAP load path). Prototype in your group folder first; then open an explicit PR that adds the cleaned version under the right `resources/` layer. See [resources/README.md](resources/README.md).

Edit mostly your own group folder. Touch `resources/` only when the PR is intentionally about shared code.

## Light coding conventions

- Prefer English for docs and comments aimed at other teams.
- License-compatible dependencies only (Apache 2.0–compatible), matching this repo and BioPAL/BPS.
- On **shared** code in `resources/`: NumPy-style docstrings and type hints on exported function signatures.
- If you introduce a linter config (e.g. Ruff), keep it lightweight and document it in the PR; there is no project-wide Ruff config yet.
- No large data files in Git: link to MAAP or ship a small download script.
- No credentials, API keys, or absolute machine paths (e.g. `/Users/...`) in committed code.

## Issues or Discussions

| Where | What for |
| --- | --- |
| Issues (Task forms) | Work items: **Task · my group** (code in your `topics/<GROUP-LABEL>/` folder) or **Task · resources** (shared helpers) |
| Discussions | Questions, ideas, show and tell, Git help that is not a concrete task |

Pick your **Group** in the Task form; the colored group label is applied automatically. Keep each Task small (S / M / L, ideally about 2 hours, at most 3).

## Ground rules

- No large data files in the repo. Link to the MAAP dataset, or add a small download script instead.
- No credentials or API keys, ever.
- Keep changes inside your own group folder unless the PR is explicitly for `resources/`.
- Contributions should remain Apache 2.0 compatible.

## Code of conduct

Be respectful, welcome newcomers, keep feedback constructive, respect different viewpoints and experience levels. This repo follows the same principles as the main BioPAL Code of Conduct. Contact biopal@esa.int if something needs to be reported.

## Getting help

We are on site all week for Git and technical questions. Ask in person first. For anything not urgent, use Discussions.
