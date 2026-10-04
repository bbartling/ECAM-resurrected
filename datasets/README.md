# School hourly meter and weather practice dataset

This directory contains a processed hourly school electricity record joined to
historical gridded weather, plus `school_weather_audit.json`, which records
processing provenance and source-file SHA-256 checksums. The dataset is
redistributed with the user's authorization dated October 4, 2026. Original
meter exports and the quarter-hour interval-level joined file are not included.

## Contents and preparation

`school_hourly_weather.csv` has 35,029 hourly rows from June 19, 2013 through
June 17, 2017. It was aggregated from 139,825 quarter-hour records in four
meter exports. The meter values are treated as interval-average kW, labeled at
the interval end, and converted using 15-minute duration to kWh. Naive meter
timestamps were localized to `America/Chicago`; the repeated fall DST hours
were inferred from chronological order. These are documented processing
assumptions that should be checked against the meter-export specification before
any formal study.

The observed hourly energy total is 2,753,981.6 kWh. There are 72 missing whole
hours and one partial final hour with 0.25 hours of coverage and 8.7 kWh. Missing
hours remain visible as missing energy and zero duration; the partial hour
retains its measured energy and coverage. No weather values are missing in the
processed table. The audit retains the four original source basenames and their
SHA-256 checksums; it does not contain machine-specific source paths.

Weather was obtained from the [Open-Meteo Historical Weather
API](https://open-meteo.com/en/docs/historical-weather-api), using the ERA5
model. Open-Meteo and Copernicus ERA5 weather data are attributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The requested site
coordinates and returned ERA5 grid coordinates are recorded in the audit.
Gridded reanalysis is not a measurement from a weather station at the school.
The user's redistribution authorization covers the processed school data in
this repository; it does not assign a third-party license to the underlying
meter records.

Temperature is the instantaneous hourly value. Precipitation is the preceding
hour's sum in millimeters; shortwave radiation is the preceding hour's mean in
W/m². Values are joined by the UTC floor-hour of each meter interval start. Do
not sum the repeated hourly weather values as if they were quarter-hour
interval totals. See `school_weather_audit.json` for detailed counts, units,
weather fields, and join checks.

## Reproducible exploratory fit

The CSV is already prepared for `ecam-mv`; its timestamp offsets preserve the
local daylight-saving transitions. For example:

```bash
mkdir -p work
ecam-mv fit datasets/school_hourly_weather.csv --prepared --exclude-incomplete \
  --model 3pC --group-by occupied --out work/school_baseline.json
```

`--exclude-incomplete` explicitly removes 73 hourly rows (the 72 missing hours
and one partial hour), leaving 34,956 model rows. The exploratory three-parameter
cooling model grouped by `occupied` has a CVRMSE of 43.95%, above the ASHRAE
Guideline 14 hourly calibration screen of 30%. It therefore fails that screen.
The example's default `occupied` schedule labels Monday through Friday from
08:00 to 18:00 as occupied; this illustrative schedule has not been verified
against the school's actual operating hours. This historical whole-period fit
is an example for practicing preparation and diagnostics, not a qualified
baseline and not evidence of energy savings. A real savings study also needs a
defined intervention, reporting period, and M&V plan.

The `--prepared` option requires the adjacent
`school_hourly_weather.csv.metadata.json` sidecar to be kept alongside the CSV.
Keep both files together when moving or copying the prepared dataset.

For synthetic examples that run without network access, use the package's
`practice_dataset()` generator or `examples/data/practice_hourly.csv`; those
records are generated practice data and are not school data.
