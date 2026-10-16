# Soil moisture (group 11-A-SOIL-MOISTURE)

1st ESA BIOMASS MAAP Hackathon, ESOC Darmstadt, 12-16 October 2026.

## Question
Under which conditions do BIOMASS P-band (435 MHz) L1C stacks carry soil-moisture information, and with which observable
(backscatter, polarimetric decomposition, InSAR coherence, closure phase)?

## Contents
| Folder | Content |
|---|---|
| `sm_pipeline/` | Main result. Processes any BIOMASS L1C STA stack at the ISMN stations inside its footprint or at a chosen point, with a dashboard of all stacks, per-stack reports and figures. Start with `sm_pipeline/README.md`. |
| `legacy/` | Exploratory notebooks and plots of the first hackathon days (station selection, L1C point extraction, ancillary data, Yamaguchi decomposition, acquisition plan, ISMN conversion). Kept for reference, not maintained. |

## Data
- BIOMASS L1C STA products from the ESA MAAP catalogue; the SAR part needs a personal MAAP token.
- ISMN in-situ soil moisture is not included: the ISMN terms do not allow onward distribution of the original data.
  How to use your own ISMN download: `sm_pipeline/README.md`, section *ISMN station data*.
- ERA5-Land, ERA5 and ECMWF IFS via Open-Meteo; Sentinel-2 L2A via Earth Search; SoilGrids 2.0; ESA WorldCover; Copernicus DEM.

## Team
Ochse (HeinerOch), Pfeiffer, Karamvasis (kleok), Borlaf-Mena (borlafgis)
