# How to read the outputs of one run (team 11a soil-moisture pipeline)

One run = one BIOMASS L1C stack (7 acquisitions, 3 days apart) at one or more points (a station or a chosen location).
Folder: `~/data/runs/<stack>/` with one subfolder per point. Colours: HH blue, VV vermillion, HV orange, VH purple (Okabe-Ito
palette, distinguishable with red-green colour-vision deficiency); station depths from dark purple (shallow) to yellow (deep).

## 0. The questions each run answers
1. Does P-band backscatter change when the soil moisture changes? (sigma0 panels, `sigma0_vs_reference.png`)
2. Does the interferometric phase carry a moisture signal? (coherence panels, `coherence_pairs.png`, `closure_check.png`)
3. Which part of the signal is soil, which is vegetation? (Freeman-Durden panel)
4. Under what conditions? (reference tier, slope, land cover, rain, temperature, snow)

## 1. `reference_sources.png` (no token; "what happened on the ground")
Four panels per point, x axis = the 7 acquisition dates (dotted vertical lines).
- **in-situ soil moisture**: station sensors, one line per depth (triangles). Depths <= 0.4 m coloured, deeper grey dashed.
  Units m3/m3 (volume water per volume soil). Expect 0.05 (dry sand) to 0.45 (saturated loam). Only flag-G (good) values.
  Absent -> "no in-situ station data": this point is NOT a test site, only a consistency check.
- **model soil moisture**: ERA5-Land (red), ERA5 (orange), ECMWF IFS (purple); solid = 0-7 cm layer, dotted = 7-28 cm.
  These are 9-31 km model grid cells, not measurements. Compare their TIMING with the station, not their level: at Maydantal
  the station is 0.1-0.15 and ERA5-Land 0.2-0.4 because the model soil is a different soil.
- **rain [mm]**: bar at acquisition i = rain summed between acquisition i-1 and i ("rain since previous acquisition").
  Source in brackets: station gauge if the station has one, else ECMWF IFS (ERA5-Land hourly rain is missing for 2026).
  A bar of 20 mm before an acquisition should show up as a soil-moisture jump at that date; a bar at the first date is
  the rain in the 3 days before the stack began.
- **temperature [deg C]**: 2 m air temperature (solid), dew point (dotted), soil temperature 0-7 cm (dashed) from the models, station
  air temperature black triangles; snow depth as grey bars on the right axis when > 0.5 cm. Below 0 deg C or with snow the
  radar sees ice, not water: those dates must be excluded (ISMN flags them D01/D02/D04).

## 2. `biomass_vs_reference.png` (main figure, 7 panels, same x axis)
- **backscatter [dB], window mean**: circles = beta0 straight from the image (raw check), squares = sigma0 (beta0 x LUT factor,
  noise subtracted). Both per polarisation: HH blue, HV green, VH olive, VV red. Typical P-band grassland: HH/VV -6 to -12 dB,
  HV/VH -15 to -20 dB. HV and VH should be nearly equal (reciprocity); if not, something is wrong with the data.
  The raw and the sigma0 curve differ by a constant (the LUT factor, -2 to -4 dB); their SHAPE must be identical.
- **sigma0 anomaly [dB]**: sigma0 minus its 7-date mean, per polarisation. This is the signal we test: soil drying should give a
  decreasing curve, rain a step up. 1 dB = 26 % change in power. Expected moisture sensitivity at P-band ~ 0.1-0.3 dB per
  0.01 m3/m3 in HH/VV, less in HV. Vegetation growth/senescence and wind also move these curves, slowly (weeks) or randomly.
- **|coherence| to primary**: 1 at the primary (reference acquisition of the stack, marked * in quicklooks), decreasing with
  temporal distance. Values > 0.7 = stable surface; < 0.4 = decorrelated (vegetation movement, moisture change, rain).
  A V-shape centred on the primary is normal (temporal decorrelation). Asymmetry can indicate a rain event.
- **in-situ / model soil moisture / rain / temperature**: as in section 1, repeated here so BIOMASS and reference share the x axis.
Title suffix: "[flattening applied/not applied, denoising applied/not]" documents the processing decisions (see 5).

## 3. `sigma0_vs_reference.png` (the sensitivity test)
Scatter: x = reference soil moisture (station depths, ERA5-Land, IFS), y = sigma0 anomaly, one point per acquisition and polarisation,
with a least-squares line. Read: slope (dB per m3/m3), tightness (r). Interpretation rules:
- A slope is only meaningful if the x range is wide (>= 0.05 m3/m3). With 7 points along a monotonic dry-down, any variable that also
  decreases (vegetation water, phenology) gives a high r. Therefore `metrics.csv` also reports r_diff = correlation of day-to-day
  CHANGES: if r is high but r_diff is low, the agreement is a shared trend, not a response.
- Literature P-band/L-band values for bare/grass soil: 15-30 dB per m3/m3 (HH/VV). Values far above are a sign of a tiny x range.
- Compare depths: the depth with the highest r_diff is the depth the radar "sees" (P-band penetrates tens of cm in dry soil).

## 4. `coherence_pairs.png` and `closure_check.png` (the InSAR test)
- **coherence_pairs**: every pair of acquisitions (21). Top left: |coherence| vs days apart, all polarisations, lines = median per
  baseline; the decay is temporal decorrelation. Top right: the full coherence matrices HH and VV (date x date) - a row that is
  darker than its neighbours marks a date with a surface change (rain, wind, moisture). Bottom: one panel per polarisation,
  |coherence| vs the moisture change |dSM| between the two dates, points coloured by the temporal baseline. If moisture matters,
  pairs with large |dSM| sit lower than pairs with the same colour (same time separation) but small |dSM|. The annotation is the
  regression |coh| = a + b_dt*dt + b_dsm*|dSM| (also in `metrics.csv`, "|coh| ... controlled for time"); b_dsm < 0 with a partial
  correlation r(|dSM|) clearly below 0 = moisture decorrelates. Caveat: in a monotonic dry-down dt and |dSM| are nearly
  collinear and the two effects cannot be separated (Maydantal); a rain event in the middle of the stack separates them (Hawaii).
- **closure_check**: the closure phase of a triplet (i, j, k) is arg(g_ij g_jk g_ki). Pure geometry/propagation gives 0; a change of
  the soil dielectric (moisture) between the dates gives a non-zero value (De Zan et al. 2014). x = closure phase predicted by the
  De Zan soil-volume model from the STATION moisture sequence (Mironov dielectric model), y = observed. Points along the 1:1 line
  with r > 0.5: the phase carries the moisture signal with the modelled convention; along the anti-diagonal (r < 0): same signal,
  opposite sign convention (fix: conjugate); a cloud around 0 with |obs| < 2 deg: no detectable moisture phase (dry, little change,
  or decorrelated). Typical expected values for a 0.1 m3/m3 change: 5-15 deg.
- **sar_observables**: the same quantities as time series, plus the BPS LUT coherence at the node (grey crosses; the processor's own
  estimate, larger estimation window, so it is usually higher than our 100 m window value - the shape should agree), the
  Freeman-Durden decomposition as SHARES of the total power (surface blue, double-bounce vermillion, volume green; powers are
  forced non-negative, van Zyl et al. 2011), and the polarimetric entropy H (0 = one scattering mechanism, 1 = random) with the
  mean alpha angle (< 40 deg surface, ~45 dipole/volume, > 50 double bounce). For grassland the surface share should dominate and
  follow the moisture; a rising volume share means vegetation. The closure-phase panel shows consecutive triplets (i-1, i, i+1):
  a value far from 0 marks a moisture change around date i.
- **window_images**: multilooked sigma0 (same looks as the analysis, 100 m cells) per polarisation and date on ONE colour scale, so
  that dates and polarisations are comparable; axes in metres (vertical = azimuth, horizontal = slant range), + = point; the last
  column is the temporal standard deviation per cell (dB): cells that change are the candidates for the moisture signal, a single
  bright cell that changes is probably a man-made target.
- **window_geometry**: LUT height, incidence angle, azimuth and range terrain slope and the beta0 -> sigma0 factor inside the window.
  Use it to judge whether the window is on one slope or mixes slopes (a slope mix changes the sigma0 mean without any moisture).
- **quicklooks_pauli**: left the full scene of the primary with the window marked; right a +-6 km zoom of every acquisition with the
  600 m analysis window (orange), 500 m (white) and 1 km (yellow) rings. Slant-range geometry (vertical = azimuth, range pixels
  are ~6 times longer than azimuth pixels). Changes in the Pauli colours between dates = large-scale changes (rain, wind on the
  canopy, flooding); a steady image with a changing sigma0 mean is the clean case.

## 5. The processing decisions printed in the log (and in the figure title)
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
  radians over 600 m; a spread of several hundred radians indicates steep terrain, where residual DEM errors decorrelate the
  100 m cells (Maydantal: coherence gain by flattening only 0.45 -> 0.46; Hawaii: 0.5 -> 0.75).
- `cross-check at the station node`: BPS's own coherence/interferogram to the primary (LUT) vs ours. They should agree in magnitude
  (ours is a window mean, theirs one node); the interferogram phase should be near 0 after flattening.
- `SAR skipped -> ...`: the point is outside the valid image (swath edge, nodata) or the LUT does not cover it. Not an error of the
  method; choose the neighbouring sub-swath stack.

## 5a. `candidate_windows.png` / `.csv` (only with --auto-points or --candidates, or step 1b in the notebook)
Left: dominant WorldCover class per 600 m block of the footprint (official legend colours), suitable blocks hatched, chosen windows as
numbered stars, stations as triangles. Right: mean slope per block. The CSV has every block with lat, lon, all class percentages,
slope_mean_deg, slope_gt5_pct, suitable (True/False) and rank. Read it as: how much of the tile is usable for a soil signal at all
(Hawaii F138: 3 % of the blocks; the rest is lava, forest, ocean), and where the chosen windows sit relative to the stations.

## 5b. `sar_info.json` (one per point) and `all_tiles_summary.csv` (one row per stack x point, from run_batch.py)
Window location (image row/col, LUT offset, valid %), height, incidence, LUT slopes, local incidence, flattening and denoising
decisions, NSR table, acquisitions, BPS LUT coherence vs pipeline coherence to the primary, DEM slope, land cover, tier.
`all_tiles_summary.csv` adds the key numbers: reference used, its range, dsigma0 HH/VV over the stack, slopes, r, r_diff,
mean coherence, closure-check r and the skip reason, so that stacks and stations can be compared in one table.

## 6. `metrics.csv` columns
station, reference (which SM series), observable (sigma0 HH ..., |coh| HH ..., FD surface ..., closure phase ..., closure-inversion dSM),
metric (what `value` is), value, ci_low/ci_high (95 % CI), r (Pearson), rho (Spearman), r_diff (correlation of first differences),
n, ref_range (SM range of the reference over the 7 dates), obs_range (dB range of the observable), note (e.g. "slope not quantifiable").
Rules of thumb: trust slopes only with ref_range >= 0.05 and r_diff > 0.5; closure phase row: r > 0.5 and sign agreement > 0.7 = the
phase carries the moisture signal; "closure-inversion dSM" rows are experimental (relative moisture from phases alone) and reported
for completeness, not as a result.

## 7. `backscatter_raw.csv`, `observables_long.csv`, `closure_phases.csv`, `slc_window.npz`
Raw numbers for your own analysis: per date and polarisation beta0, sigma0, coherence to primary, and every reference series
(`backscatter_raw.csv`); the shared long-format table for all points (`observables_long.csv`); every triplet with observed and
modelled closure phase (`closure_phases.csv`); the complex window stack after flattening for the InSAR/PolSAR colleagues
(`slc_window.npz`: slc[t, pol, rows, cols], names, datetimes, window, pixel spacing).

## 8. First real-data readings (runs of 14 Oct 2026, 7 stacks, 18 stack x point rows; see all_tiles_summary.csv)
- Dry-down, no event (Maydantal S2_T009_F118, 25 Jul-12 Aug 2026, IFS rain 0.4-4.6 mm between acquisitions): station 10 cm 0.090 ->
  0.068 m3/m3 (range 0.022), sigma0 HH -5.8 -> -7.1 dB, VV -6.4 -> -7.7 dB (range 1.3 dB), r = 0.94/0.99 but r_diff 0.39/0.77 and
  the reference range is below 0.05 -> the slope (~50 dB per m3/m3 nominally) is not quantifiable; the drying is also seen by ERA5
  (r 0.99) but not by IFS (r 0.1). Coherence 0.35-0.6 to the primary, falling linearly with the baseline (0.6 at 3 d, 0.23 at 18 d);
  dt and |dSM| collinear. Closure phases +-5 deg, De Zan model predicts < 2 deg for this moisture change -> not testable here.
  Freeman-Durden: surface 30-50 %, volume ~35 %, H ~0.65, alpha ~30 deg (grass + 30 % trees, steep terrain). Denoising skipped (NSR).
- Second track over the same station (Maydantal S2_T001_F038, ascending, 28 Jul-15 Aug): sigma0 range only 0.7 (HH) / 0.3 dB (VV),
  r_diff 0.46 -> the signal of the first track is not reproduced in the second geometry; the station lies 1.1 km from the window centre
  in a valley (DEM slope 12 deg, window height range 130 m) -> geometry/terrain dominates at this site.
- Rain events (Hawaii, S2/S3_T018_F137/F138, 15 May-23 Jun 2026, station rain 10-25 mm between acquisitions): of the six SCAN
  stations, the three with a 5 cm range >= 0.08 m3/m3 show r_diff 0.78-0.99 in HH (Kukuihaele 0.08 / 0.78, Kainaliu 0.09 / 0.92,
  SilverSword 0.10 / 0.99; sigma0 HH ranges 0.6-1.35 dB; SilverSword: grassland 83 %, slope 15 deg, coherence 0.6-0.9, flattening
  gain 0.49 -> 0.73, denoising applied with HH NSR -11 dB). PuuwaiRanch (range 0.03) gives 0.63 and 0.87 in two stacks, WaimeaPlain
  (range 0.06, flat grassland) -0.31 - the counter-example to be explained (window on irrigated/mown pasture? check the quicklooks),
  ManaHouse (range 0.01) is not assessable. Tile centre without station (IFS range 0.14): HH range 1.3 dB, r_diff 0.65. ManaHouse
  and Kukuihaele fall outside the valid swath in S3 F138/F137 -> use the S2 frames for them.
- Snow/frost (Baytik S1_T001_F038, 24 Nov-9 Dec 2025, station 10 cm range 0.076): sigma0 range 1.2-1.3 dB, r_diff 0.30, coherence
  0.40, denoising skipped -> freezing, not liquid water; keep as negative control.
- Interior point without station (S1_T009_F118, 4-22 Jul 2026, IFS range 0.24): sigma0 range 5 dB (!) with r_diff 0.15 and coherence
  0.31 -> a large change that the model does not explain (irrigation? harvest? check the quicklooks before interpreting).
