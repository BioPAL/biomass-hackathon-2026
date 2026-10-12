# QGIS plugin

**Challenge #3, The Biomass Gateway: production-ready QGIS plugin**

## Goal

Implement an open-source QGIS plugin with a reader and visualisation tools for BIOMASS data. The plugin should correctly expose the BIOMASS geolocation information and GCPs to QGIS, so products can be used with standard QGIS functions such as on-the-fly reprojection, clipping, and mosaicking.

## Roadmap

- **Foundation**: a robust directory reader that parses BIOMASS product folder structures and loads the COG layers on the fly.
- **Deep dive**: expose the internal BIOMASS geolocation grids and Ground Control Points (GCPs) to QGIS, and ensure dynamic reprojection, precision clipping, and multi-track mosaicking work.

## Data sources and tech stack

- BIOMASS Cloud-Optimized GeoTIFFs (COGs)
- GDAL (direct functions)
- QGIS plugin development guide: https://pyqgis.com/qgis-plugin-development/

## Status

Not started.

## Contact

Topic referent, feel free to ask on site.
