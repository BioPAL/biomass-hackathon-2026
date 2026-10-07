# Contributing

Thanks for contributing to the ESA Biomass MAAP Hackathon shared repository.

This is a hackathon repo, not the main BPS production codebase, so the process here is much lighter than BioPAL/BPS's own contributing guide. If you are also working on BPS itself, that repo has a more formal review process with an approval gate and CODEOWNERS routing. None of that applies here.

## Before the hackathon, fill in your topic folder

Every `topics/` folder already has a README with the goal, data sources and open actions, taken from the hackathon brief. If you are listed as a contact for a topic, please check your folder and fill in anything marked "still to source", "not assigned" or "TBD" before 12 October. You do not need Git for this:

1. Open your folder on github.com, for example `github.com/BioPAL/biomass-hackathon-2026/tree/main/topics/04-3d-forest-structure`
2. Click `README.md`, then the pencil icon to edit
3. Fill in or correct the content
4. Scroll down, choose "Create a new branch and start a pull request", then click "Propose changes"
5. On the pull request page that opens, click "Merge pull request"

No local setup needed. If step 5 is greyed out, or you are not sure about anything, ping Yoann and he will merge it for you.

## How a contribution flows, during the hackathon

1. Find or create your topic folder under `topics/`.
2. Work on a branch, commit, push.
3. Open a pull request targeting `main`.

No approval gate, no mandatory reviewers. You can merge your own pull request as long as it only touches your topic folder. The one rule that is enforced is that `main` is protected, so everything has to go through a pull request.

## Getting set up with Git

Clone the repo and create a branch:

```bash
git clone https://github.com/BioPAL/biomass-hackathon-2026.git
cd biomass-hackathon-2026
git checkout -b 04-3d-forest-structure/short-description
```

Create your topic folder if it does not exist yet:

```bash
mkdir -p topics/04-3d-forest-structure
```

Commit and push when ready:

```bash
git add topics/04-3d-forest-structure/
git commit -m "Add first tomographic processing notebook"
git push -u origin 04-3d-forest-structure/short-description
```

Then open a pull request on GitHub.

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

ACRI-ST is on site all week for Git and technical questions. Ask in person, or open a Help wanted issue. For anything not urgent, use Discussions.
