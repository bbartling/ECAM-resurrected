# Meter concatenation, weather joins, and quality controls

## Timestamp and energy contract

The school CSVs use `Date,kW` at 15-minute intervals. The first reading at 00:15
and the sequence through midnight support an **interval-end assumption**; verify
this with the meter-export specification before a formal M&V study. Treat kW as
interval-average power and multiply by 0.25 hours. If the export uses instantaneous
power readings, that integration assumption needs review.

The meter location implies `America/Chicago`. Naive timestamps default to strict
DST rejection. `dst='infer'` is an explicit choice for chronological exports with
both repeated fall hours present. Localization occurs within each source file
before sorting. Spring-forward nonexistent labels are rejected, not shifted.
UTC duplicates and overlapping exports fail; no arbitrary keep-first deduplication
is allowed. Source filename and original CSV row number remain on each interval.

Hourly aggregation conserves observed energy and uses measured exposure. Missing
whole hours have null energy, duration 0, coverage 0. Partial hours retain their
actual energy/exposure/coverage. No meter gaps are interpolated or filled with
zero. `model_records` rejects these records by default; `missing='drop'` is an
explicit, audited modeling policy, not proof that incomplete coverage is acceptable
for a project's reporting-period savings calculation.

## Weather contract

Geocoding uses exact place/country/region matching and fails on ambiguity. The
school example selects Lake Geneva, Wisconsin, United States, coordinates
42.59168, -88.43343. The school address can be substituted with exact coordinates.
The archive API uses ERA5 throughout the historical interval, Celsius, km/h,
millimeters, GMT/UTC, and UNIX epoch timestamps. Requests cover full UTC dates,
then are trimmed to the needed interval. Returned weather grid coordinates can
differ from requested site coordinates; both are retained in metadata.

Dry-bulb temperature is instantaneous at the labeled hour. The join assigns the
UTC floor-hour of each meter interval start. There is no implied quarter-hour
weather interpolation. Precipitation is a preceding-hour sum; solar radiation
is a preceding-hour mean. Their repeated values in an interval-level joined table
are reference values, not interval totals. For an hourly radiation/precipitation
model, review whether the preceding-hour convention requires an explicit shift.

The join is validated many-to-one and preserves meter rows/index. It rejects
duplicate weather hours, timestamp ambiguity, column collisions and invalid
numeric values. Missing dry bulb fails by default; `missing='flag'` retains it
with a quality flag. Unselected reference nulls remain visible. The selected
regression column must be finite or explicitly excluded with a recorded audit.
No weather imputation is performed.

The client has timeout/bounded retry handling and a parameter-keyed disk cache.
Unit tests mock HTTP responses and do not depend on network availability. A live
integration run is separately recorded in the school dataset's audit report.

## Attribution and release boundary

Weather attribution: Open-Meteo and Copernicus ERA5. Follow the
[data license and API-use terms](https://open-meteo.com/en/terms) and the
[historical API documentation](https://open-meteo.com/en/docs/historical-weather-api).
Free API service eligibility and data redistribution licensing are separate
considerations; check the current terms for commercial deployment.

The synthetic practice generator ships with the package and needs no API.
Real meter exports, processed school data, caches, and site-specific reports
must not accidentally enter a release archive. Any real dataset distributed in
the repository needs an explicit permission statement and weather attribution.
Archive the original exports before removing them from the active input directory.
