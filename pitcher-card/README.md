# xRuns Pitcher Card Producer

This folder contains a standalone Baseball Savant to PNG workflow for xRuns-style
single-game pitcher cards.

## 1. Build an MLB Pitch Baseline

Download a Baseball Savant Statcast Search CSV for the MLB date range you want to
grade against, then run:

```r
source("pitcher-card/build_pitch_baseline.R")
build_pitch_baseline(
  input_csvs = c("/path/to/statcast_search_2026.csv"),
  output_csv = "pitcher-card/mlb_pitch_baseline_2026.csv"
)
```

For the current local sample, `pitcher-card/pitch_data.csv` has already been
converted into `pitcher-card/mlb_pitch_baseline_2026.csv`.

The baseline is grouped by pitch type, pitch name, and pitcher handedness. It is
used for movement, command, whiff, CSW, and xwOBA grade comparisons.

## 2. Make a Card

### Upload App

Run the local app:

```r
shiny::runApp("pitcher-card")
```

Then upload a Baseball Savant game CSV, choose `Square` or `Landscape`, preview
the card, and download the PNG.

### Script

```r
source("pitcher-card/make_pitcher_card.R")
make_pitcher_card(
  game_csv = "/path/to/savant_game.csv",
  baseline_csv = "pitcher-card/mlb_pitch_baseline_2026.csv",
  output_png = "pitcher-card/pitcher_card.png"
)
```

The default output is a `2200x2200` square PNG. A landscape version is still
available:

```r
make_pitcher_card(
  game_csv = "/path/to/savant_game.csv",
  baseline_csv = "pitcher-card/mlb_pitch_baseline_2026.csv",
  output_png = "pitcher-card/pitcher_card_landscape.png",
  layout = "landscape"
)
```

The card attempts to fetch the pitcher headshot from MLB's public image endpoint
and the team logo from ESPN's public logo CDN; if either asset is unavailable,
it renders a text fallback instead.

## Notes

- `Run Value Added` is positive for pitcher value above the xRuns pitcher baseline.
- `xStuff+` blends movement/velocity shape and whiff rate versus league pitch-type baselines.
- `xPitching+` blends run-value prevention proxy and CSW% versus league pitch-type baselines.
- Both plus metrics are scaled to `100 = MLB average` and `10 = one standard deviation`.
- The game-level xERA input is estimated from the historical xRuns pitcher pool
  because pitch-level game CSVs do not include xERA.
- If a PA-ending pitch has no `estimated_woba_using_speedangle`, the card falls
  back to `woba_value`.
- A one-game CSV can be used as a smoke-test baseline, but real grades need a
  league-level Baseball Savant export.
