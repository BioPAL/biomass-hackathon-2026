# Contributing

Thanks for contributing to the ESA Biomass MAAP Hackathon shared repository. This guide is for hackathon participants, 12 to 16 October at ESOC.

This is a hackathon repo, not the main BPS production codebase, so the process here is much lighter than BioPAL/BPS's own contributing guide. If you are also working on BPS itself, that repo has a more formal review process with an approval gate and CODEOWNERS routing. None of that applies here.

## How a contribution flows

You do not have write access to this repository, so you will work from your own fork.

1. Fork the repository to your own GitHub account.
2. Clone your fork, create a branch, find or create your topic folder under `topics/`.
3. Commit, push to your fork.
4. Open a pull request from your fork to `main` on `BioPAL/biomass-hackathon-2026`.

`main` is protected: every change goes through a pull request, and a maintainer reviews and merges it. Do not merge your own pull request.

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

```bash
git checkout -b 04-3d-forest-structure/short-description
```

**5. Create your topic folder if it does not exist yet**

```bash
mkdir -p topics/04-3d-forest-structure
```

**6. Commit and push to your fork**

`origin` points to your fork, not to `BioPAL/biomass-hackathon-2026`, so this push goes to your own copy.

```bash
git add topics/04-3d-forest-structure/
git commit -m "Add first tomographic processing notebook"
git push -u origin 04-3d-forest-structure/short-description
```

**7. Open a pull request**

Go to your fork on github.com. GitHub shows a banner suggesting a pull request for your new branch, click it. Check that the base repository is `BioPAL/biomass-hackathon-2026` and the base branch is `main`, then create the pull request. A maintainer will review and merge it.

**Keeping your fork up to date** (optional, useful if the hackathon runs for several days)

```bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

## Issues or Discussions

| Where | What for |
|---|---|
| Issues | Something specific and actionable: help wanted, a data access request, a bug in shared resources |
| Discussions | Open questions, ideas, show and tell, anything not tied to one task |

Issue templates are set up for this repo: Help wanted, Data access request, and Bug report.

## Ground rules

- No large data files in the repo. Link to the MAAP dataset, or add a small download script instead.
- No credentials or API keys, ever.
- Keep changes inside your own topic folder. Put anything shared by several groups in `resources/`.

## Code of conduct

Be respectful, welcome newcomers, keep feedback constructive, respect different viewpoints and experience levels. This repo follows the same principles as the main BioPAL Code of Conduct. Contact biopal@esa.int if something needs to be reported.

## Getting help

We are on site all week for Git and technical questions. Ask in person, or open a Help wanted issue. For anything not urgent, use Discussions.
