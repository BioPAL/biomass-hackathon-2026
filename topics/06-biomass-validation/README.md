# BIOMASS validation

**Challenge #6, The Truth-Engine: global carbon map cross-validation dashboard**

## Goal

Validate BIOMASS outputs against in-situ measurements, global canopy models, and historical baselines. Write a Jupyter notebook to access BIOMASS and Geo-Trees forest height and AGBD data, derive statistics following the official BIOMASS validation protocol, and write a script to automatically generate a report or ppt with the validation results.

## Roadmap

- **Foundation**: a notebook that maps BIOMASS data to Geo-Trees ground-truth sites and derives the validation statistics from the BIOMASS validation protocol.
- **Deep dive**: a script that auto-compiles the validation results into slide decks and documents.

## Extra challenges

1. Parallel ingestion streams for HydroGNSS AGBD, ESA CCI Biomass maps (from MAAP), Meta Forest Canopy data (AWS Data for Good), and global canopy height layers, for a cross-comparison dashboard. Add a validation of these products against Geo-Trees, with a cross comparison to BIOMASS.
2. Cross comparison of different BIOMASS product versions: forest height (FH) from the official ESA archive versus FH from the DLR Gabon map.

## Data sources

- BIOMASS core mission products
- Geo-Trees network ground-truth records, site still to be chosen
- Open global datasets (see extra challenges)

## Actions

- Validation protocol, from the topic referent.
- Geo-Trees data for at least one site, still to source.

## Status

Not started.

## Contact

Topic referent, feel free to ask on site.
