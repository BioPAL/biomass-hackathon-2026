# How to read the outputs of one run (team 11a soil-moisture pipeline)

One run = one BIOMASS L1C stack (7 acquisitions, 3 days apart) at one or more points (a station or a chosen location).
All time series, scatter plots and coherence figures show the co-polar channels HH and VV only; `window_images.png` shows all four
polarisations, and the tables (`metrics.csv`, `observables_long.csv`) keep all four.
Folder: `~/data/runs/<stack>/` with one subfolder per point. Colours: HH blue, VV vermillion, HV orange, VH purple (Okabe-Ito
palette, distinguishable with red-green colour-vision deficiency); station depths from shallow to deep in distinct colours and
markers (black triangle, orange square, sky-blue diamond, purple triangle down, green circle, brown ...), solid <= 0.4 m, dashed deeper.

## 0. The questions each run answers
1. Does P-band backscatter change when the soil moisture changes? (sigma0 panels, `sigma0_vs_reference.png`)
2. Does the interferometric phase carry a moisture signal? (coherence panels, `coherence_pairs.png`, `closure_check.png`)
3. Which part of the signal is soil, which is vegetation? (Freeman-Durden panel)
4. Under what conditions? (reference tier, slope, land cover, rain, temperature, snow)

## 1. `biomass_vs_reference.png` (main figure, at the top of the report; up to 6 panels, one time axis)
x axis = 5 days before the first to 2 days after the last acquisition; dotted vertical lines = the 7 BIOMASS acquisitions.
- **in-situ soil moisture [m3/m3]** (only if the point has a station): hourly values of every station depth (ISMN, flag G only;
  colours and markers per depth as above). Markers = the values at the overpass hours, i.e. what BIOMASS is compared with.
- **ERA5-Land 0-7 cm [m3/m3]**: hourly model soil moisture (~9 km grid; ERA5 or IFS if ERA5-Land is missing), markers at the
  overpasses. Without a station this is the only soil-moisture panel: then the point is a consistency check, not a test site.
  Compare the model's TIMING with the station, not its level (different soil).
- **sigma0 [dB], window mean**, HH (blue) and VV (vermillion): absolute values after the LUT calibration and noise subtraction.
  Typical P-band grassland: HH/VV -6 to -12 dB (cross-pol HV/VH -15 to -20 dB, in `window_images.png`). Soil wetting should raise HH/VV; expected P-band sensitivity ~ 0.1-0.3 dB per 0.01 m3/m3.
- **|coherence| 3-day pairs**, HH and VV: coherence of each pair of consecutive acquisitions, drawn as a step between
  its two dates; the y axis is scaled to the data (not 0-1), so compare the steps with each other. A low step = the surface changed
  between these dates (rain, wind on the canopy, moisture). Coherence to the primary is not plotted (1 at the primary by definition); the full matrix is in `coherence_pairs.png`.
- **rain [mm/day]**: daily sums from the station gauge if the station has one, else from the model (ERA5-Land has no hourly rain
  for 2026, then ERA5 or IFS).
- **temperature [deg C], hourly**: model 2 m air temperature (solid), soil temperature 0-7 cm (dashed), station air temperature
  (grey); snow depth as grey area on the right axis when > 0.5 cm. The axis is scaled to the data; the 0 deg C line appears only
  when the data cross it. Below 0 deg C or with snow the radar sees ice, not water.
The figure title names only stack and point. In the report, a small table above each main figure gives the site facts: elevation,
incidence (LUT / local), LUT and DEM slopes, land cover, soil 0-5 and 5-15 cm (SoilGrids), window size, valid share and the
processing decisions (flattening, denoising; see section 4).
`reference_sources.png` (all model sources per point) is shown in the report only for runs without SAR output (point outside the
swath or SAR skipped). Sentinel-1 is no longer used (dropped 15 Oct 2026: too hard to interpret quickly).

## 1b. `s2_ndvi_ndmi.png` (vegetation context from Sentinel-2)
Sentinel-2 L2A over the same square as the BIOMASS window, from 30 days before the first to 30 days after the last acquisition
(method of the team notebook 'Biomass L1C Station Ancillary'): NDVI = (B08 - B04) / (B08 + B04), 10 m; NDMI = (B8A - B11) /
(B8A + B11), 20 m (Gao 1996, doi:10.1016/S0034-4257(96)00067-3), per pixel and then averaged over the clear pixels of the box
(scene classification 4-7). Filled = clear scenes (>= 80 % of the box clear), error bar = std over the clear pixels; hollow = partly
cloudy. Fixed y ranges (NDVI 0-1, NDMI -1-1) as in the notebook. A change in NDVI or NDMI during the 18 days means the vegetation
changed (growth, drying, grazing, harvest) and can change sigma0 without any soil-moisture change; a flat NDVI makes a soil
origin of a sigma0 change more plausible. Values at the overpasses (nearest clear scene within 15 days and the interpolation
between clear scenes) are in `backscatter_raw.csv` (`s2_*`) and in the site table of the report.

## 2. `sigma0_vs_reference.png` (the sensitivity test)
Scatter: x = reference soil moisture (the two shallowest station depths and ERA5-Land 0-7 cm), y = sigma0 [dB] (absolute), one point per acquisition and polarisation,
with a least-squares line. Read: slope (dB per m3/m3), tightness (r). Interpretation rules:
- A slope is only meaningful if the x range is wide (>= 0.05 m3/m3). With 7 points along a monotonic dry-down, any variable that also
  decreases (vegetation water, phenology) gives a high r. Therefore `metrics.csv` also reports r_diff = correlation of day-to-day
  CHANGES: if r is high but r_diff is low, the agreement is a shared trend, not a response.
- Literature P-band/L-band values for bare/grass soil: 15-30 dB per m3/m3 (HH/VV). Values far above are a sign of a tiny x range.
- Compare depths: the depth with the highest r_diff is the depth the radar "sees" (P-band penetrates tens of cm in dry soil).

## 3. `coherence_pairs.png` and `closure_check.png` (the InSAR test)
- **coherence_pairs**: every pair of acquisitions (21). Top left: |coherence| vs days apart, HH and VV, lines = median per
  baseline (y axes of the scatter panels scaled to the data); the decay is temporal decorrelation. Top right: the full coherence matrices HH and VV (date x date) - a row that is
  darker than its neighbours marks a date with a surface change (rain, wind, moisture). Bottom: one panel per polarisation,
  |coherence| vs the moisture change |dSM| between the two dates, points coloured by the temporal baseline. If moisture matters,
  pairs with large |dSM| sit lower than pairs with the same colour (same time separation) but small |dSM|. The annotation is the
  regression |coh| = a + b_dt*dt + b_dsm*|dSM| (also in `metrics.csv`, "|coh| ... controlled for time"); b_dsm < 0 with a partial
  correlation r(|dSM|) clearly below 0 = moisture decorrelates. Caveat: in a monotonic dry-down dt and |dSM| are nearly
  collinear and the two effects cannot be separated (Maydantal); a rain event in the middle of the stack separates them (Hawaii).
- **closure_check**: the closure phase of a triplet (i, j, k) is arg(g_ij g_jk g_ki). Pure geometry/propagation gives 0; a change of
  the soil dielectric (moisture) between the dates gives a non-zero value (De Zan et al. 2014). x = closure phase predicted by the
  De Zan soil-volume model from the STATION moisture sequence (without a station: ERA5-Land, else ERA5 or IFS; Mironov
  dielectric model), y = observed. Points along the 1:1 line
  with r > 0.5: the phase carries the moisture signal with the modelled convention; along the anti-diagonal (r < 0): same signal,
  opposite sign convention (fix: conjugate); a cloud around 0 with |obs| < 2 deg: no detectable moisture phase (dry, little change,
  or decorrelated). Typical expected values for a 0.1 m3/m3 change: 5-15 deg.
- **sar_observables**: what is NOT in the main figure, on the same time axis as the main figure. Top: the closure phase of
  consecutive triplets (i-1, i, i+1), drawn like the 3-day coherence pairs of the main figure: a bar over the three dates of the
  triplet and a dot at the middle date i; a value far from 0 marks a moisture change around date i. Below: the PyPolSARpro
  partitions and H / A / alpha (next bullet). Freeman-Durden assigns all cross-polar power to volume, so on rough or tilted
  grassland it overestimates the volume share (SilverSword, 82 % grassland: volume about 48 %); read the change over the dates,
  not the level. The own Freeman-Durden was removed on 15 Oct 2026 (it duplicated the PyPolSARpro one).
- **PyPolSARpro partitions** (only if PyPolSARpro is installed): one stacked panel each for Freeman-Durden, Yamaguchi-4 with
  orientation compensation (y4r; surface, double-bounce, volume, helix) and van Zyl 1992 (surface blue, double-bounce vermillion,
  volume green, helix purple), and in the last panel entropy H (0 = one scattering mechanism, 1 = random), anisotropy A and mean
  alpha (< 40 deg surface, ~45 dipole/volume, > 50 double bounce). Without PyPolSARpro only an own H/alpha is shown. The
  absolute split depends strongly on the scattering model (SilverSword: surface 37 to 65 %); a soil-moisture effect would show as a common CHANGE over the dates (more surface power after rain), not as a level.
  `metrics.csv` has the share-vs-soil-moisture rows for every decomposition (`pp_<method> <component> share [%]`).
- **site table** above each main figure in the report: soil (clay, sand, silt at 0-5 and 5-15 cm, organic carbon and bulk
  density at 0-5 cm) from SoilGrids 2.0 (250 m, modelled), land cover, slopes, geometry and processing. Clay and sand are also the
  texture of the dielectric model unless --clay/--sand are set.
- **window_images**: multilooked sigma0 in display cells of about 50 m (`--image-look-m`; Hawaii: 8 x 1 pixels = 53 m azimuth x
  34 m slant range; the analysis itself uses 100 m looks) per polarisation and date on ONE colour scale, so
  that dates and polarisations are comparable; axes in metres (vertical = azimuth, horizontal = slant range), + = point; the last
  column is the temporal standard deviation per cell (dB): cells that change are the candidates for the moisture signal, a single
  bright cell that changes is probably a man-made target. With 8 pixels per display cell the speckle is larger than in the
  analysis looks (std of 10 log10 of the intensity for independent looks: 1.6 dB for 8 looks, 0.65 dB for the 45 pixels of a 100 m
  look in the Hawaii stack), so where the coherence between dates is low, part of the cell-to-cell pattern of the std map is speckle.
- **window_geometry**: LUT height, incidence angle, azimuth and range terrain slope and the beta0 -> sigma0 factor inside the window.
  Use it to judge whether the window is on one slope or mixes slopes (a slope mix changes the sigma0 mean without any moisture).
- **quicklooks_pauli**: left the full scene of the primary with the analysis window (dotted) and the point (small dot); right every
  acquisition zoomed to the window half width + 15 % (+-345 m for the 600 m window), window dotted, point as a small dot. The
  title gives the quicklook pixel size: the BPS quicklooks are strongly reduced, so the zoom cannot show finer detail. Slant-range geometry (vertical = azimuth, range pixels
  are ~6 times longer than azimuth pixels). Changes in the Pauli colours between dates = large-scale changes (rain, wind on the
  canopy, flooding); a steady image with a changing sigma0 mean is the clean case.

## 4. The processing decisions printed in the log (and in the figure title)
- `LUT geolocation: row, col (.. m from the node)`: the point found via the LUT lat/lon grid. Must be < 100 m. "GCP fit differs by
  N cols" = how wrong the simple georeferencing would have been (~1 km in mountains).
- `height, incidence, terrain slope az/rg, sigmaNought`: from the LUT at the point. sigmaNought should equal sin(incidence - range
  slope) within a few percent; printed side by side as a check.
- `flattening: applied (|coh| a -> b)` or `not applied`: the LUT phase screen (topographic phase) is applied only if it raises the
  coherence. "Not applied" means the product phases were already flattened by the processor.
- `denoising: applied; noise-to-signal per pol [dB]`: how far the signal is above the noise floor. HV/VH at -5 dB or worse means the
  cross-pol change is barely measurable; "NOT applied: noise >= signal" flags a LUT problem. The full NSR table (acquisition x
  polarisation) is in `sar_info.json`. Finding of 14 Oct: at Maydantal and Baytik the secondary LUTs carry noise maps around
  -5 dB (beta0) while the primary has -14 dB and the measured HV power is -13 dB; the secondaries' maps cannot be noise power in
  the product's units. Until clarified, denoising is skipped for those stacks and the sigma0 values contain the noise floor
  (co-pol at -3 dB beta0 is 10 dB above it, so the co-pol series are hardly affected; cross-pol changes are not interpretable).
- `flattening screen ... spread inside the window N rad per secondary`: the DEM + flat-earth phase from the LUT varies by tens of
  radians over a 600 m window; a spread of several hundred radians indicates steep terrain, where residual DEM errors decorrelate the
  100 m cells (Maydantal: coherence gain by flattening only 0.45 -> 0.46; Hawaii: 0.5 -> 0.75).
- `cross-check at the station node`: BPS's own coherence/interferogram to the primary (LUT) vs ours. They should agree in magnitude
  (ours is a window mean, theirs one node); the interferogram phase should be near 0 after flattening.
- `SAR skipped -> ...`: the point is outside the valid image (swath edge, nodata) or the LUT does not cover it. Not an error of the
  method; choose the neighbouring sub-swath stack.

## 4b. `sar_info.json` (one per point) and `all_tiles_summary.csv` (one row per stack x point, from run_batch.py)
`sar_info.json` holds the window facts: height, incidence, LUT slopes, valid pixels, flattening and denoising decisions, LUT vs
pipeline coherence and the 3-day pair coherences. `all_tiles_summary.csv` collects the key numbers of every computed stack x point
(reference used and its range, dsigma0 HH/VV, slopes, r, r_diff with p-value, mean 3-day coherence, closure-check r,
slope, land cover and soil as information, skip reason). With 7 acquisitions r_diff rests on 6 differences:
|r_diff| >= 0.81 is needed for p < 0.05 (with 6 acquisitions: 0.88); the report colours r_diff by its p-value. The dashboard reads this table for the tiles where BIOMASS was computed (point colours on the map: `has_ismn` and the skip
reason).

## 5. `metrics.csv` columns
station, reference (which SM series), observable (sigma0 HH ..., |coh| HH ..., pp_<method> <component> share ..., closure phase ..., closure-inversion dSM),
metric (what `value` is), value, ci_low/ci_high (95 % CI), r (Pearson), rho (Spearman), r_diff (correlation of first differences),
n, ref_range (SM range of the reference over the 7 dates), obs_range (dB range of the observable), note (e.g. "slope not quantifiable").
Rules of thumb: trust slopes only with ref_range >= 0.05 and r_diff > 0.5; closure phase row: r > 0.5 and sign agreement > 0.7 = the
phase carries the moisture signal; "closure-inversion dSM" rows are experimental (relative moisture from phases alone) and reported
for completeness, not as a result.

## 6. `backscatter_raw.csv`, `observables_long.csv`, `closure_phases.csv`, `slc_window.npz`
Raw numbers for your own analysis: per date and polarisation beta0, sigma0, coherence to the primary and to the previous acquisition (`coh_pair_with_previous_*`), Sentinel-2 NDVI / NDMI at the overpasses (`s2_*`), and every reference series
(`backscatter_raw.csv`); the shared long-format table for all points (`observables_long.csv`); every triplet with observed and
modelled closure phase (`closure_phases.csv`); the complex window stack after flattening for the InSAR/PolSAR colleagues
(`slc_window.npz`: slc[t, pol, rows, cols], names, datetimes, window, pixel spacing).
