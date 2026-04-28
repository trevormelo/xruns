# MLB "KenPom-style" Dashboard — Multi-year

A Shiny app that ranks MLB teams by expected runs/game over an average opponent
in a neutral 9-inning game, and ranks individual players by predicted run value
derived from their expected stats.

## Quick start

```r
setwd("path/to/MLB Kenpom")
shiny::runApp("app_v2.R")
```

Packages required:

```r
install.packages(c("shiny","bslib","DT","dplyr","tidyr","readr","tibble",
                   "ggplot2","plotly","scales","purrr","mlbplotR"))
```

---

## File Naming Conventions

All file names follow a short, consistent pattern. Old names are still recognised
as a fallback so existing data does not need to be renamed.

### Prior / completed seasons — player-level files (`YYYY Data/`)

| File | What it contains |
|---|---|
| `bat_YYYY.csv` | Expected stats — batters |
| `pit_YYYY.csv` | Expected stats — pitchers |
| `rv_bat_YYYY.csv` | Run values — batters |
| `rv_pit_YYYY.csv` | Run values — pitchers |
| `br_YYYY.csv` | Baserunning run values |
| `fld_YYYY.csv` | Fielding run values |
| `std_YYYY.csv` | League standings |

### Current season — daily team snapshots (`YYYY Data/Snapshots/`)

| File | What it contains |
|---|---|
| `bat_t_YYYY-MM-DD.csv` | Team batting expected stats |
| `pit_t_YYYY-MM-DD.csv` | Team pitching expected stats |
| `br_t_YYYY-MM-DD.csv` | Team baserunning run values |
| `fld_t_YYYY-MM-DD.csv` | Team fielding run values |

> **Date format:** always `YYYY-MM-DD` (ISO 8601). This ensures files sort
> chronologically by filename, which the app relies on.

---

## Daily Upload Workflow

Each day you download four team-level CSV exports from Baseball Savant, rename
them to the short format above (substituting today's date), and drop them into:

```
2026 Data/Snapshots/
```

**That's it.** No other steps. The app picks up new files automatically on the
next launch (or app reload).

Example — after downloading on April 29, 2026:

```
2026 Data/Snapshots/
  bat_t_2026-04-29.csv
  pit_t_2026-04-29.csv
  br_t_2026-04-29.csv
  fld_t_2026-04-29.csv
```

If you miss a day, the app falls back gracefully: when you request "Last 7 Days"
but the oldest available snapshot is only 3 days back, it computes the delta
from whatever the earliest on-file snapshot is and shows a note saying so.

---

## Folder Layout

```
MLB Kenpom/
├── app_v2.R
├── README.md
│
├── 2022 Data/
│   ├── bat_2022.csv           (or old name: expected_stats_batter_2022.csv)
│   ├── pit_2022.csv
│   ├── rv_bat_2022.csv
│   ├── rv_pit_2022.csv
│   ├── br_2022.csv
│   ├── fld_2022.csv
│   └── std_2022.csv
│
├── 2023 Data/ ... (same pattern, _2023 suffix)
├── 2024 Data/ ... (same pattern, _2024 suffix)
│
├── 2025 Data/
│   ├── bat_2025.csv
│   ├── pit_2025.csv
│   ├── rv_bat_2025.csv
│   ├── rv_pit_2025.csv
│   ├── br_2025.csv
│   ├── fld_2025.csv
│   └── std_2025.csv
│
├── 2026 Data/
│   ├── Snapshots/                ← daily exports go here
│   │   ├── bat_t_2026-04-27.csv
│   │   ├── pit_t_2026-04-27.csv
│   │   ├── br_t_2026-04-27.csv
│   │   ├── fld_t_2026-04-27.csv
│   │   ├── bat_t_2026-04-28.csv
│   │   ├── ...
│   │
│   └── (optional player files for player leaderboard)
│       ├── bat_2026.csv
│       ├── pit_2026.csv
│       ├── rv_bat_2026.csv
│       └── rv_pit_2026.csv
│
└── 2026 Player Data/ ... (alternative location for player files)
```

---

## Time Period Filter

When viewing the current season (any year with a `Snapshots/` folder), a
**Period** selector appears in the Team Rankings tab:

| Option | What it shows |
|---|---|
| **Season** | Ratings based on the most recent snapshot (full season to date) |
| **30 Days** | Delta between the latest snapshot and the one from ≥ 30 days ago |
| **7 Days** | Delta between the latest snapshot and the one from ≥ 7 days ago |
| **Yesterday** | Delta between the latest and previous day's snapshots |

**How the delta math works:** since each snapshot is cumulative season-to-date,
the app subtracts an older snapshot from the newest to isolate a time window.
For rate stats (est_wOBA, BIP rate) it back-calculates the weighted totals
(`rate × PA`), subtracts them, then re-derives the rate from the PA delta. Run
values (baserunning, fielding) are already in totals so they subtract directly.

---

## How the Ratings Work

**One pooled model, many seasons.** A weighted linear regression is fit over
every player-season from every completed year. PA is the regression weight, so
high-sample players drive the fit.

- Batters:  `runs/PA ~ est_wOBA + BIP_rate`
- Pitchers: `runs/PA ~ est_wOBA + xERA`

Predictions are centered by that season's cohort mean so `0 = league-average`.
Multiplying by 38 PA/game converts the per-PA rate into a per-game rating.

- **Offensive Rating** = expected runs scored above average per game (hitting + baserunning).
- **Defensive Rating** = expected runs prevented above average per game (pitching + fielding).
- **Overall** = Off + Def = expected margin vs. a league-average team.

The model is trained purely on player-level expected stats. Team W-L and team
run totals are **never** used as training data. The Methodology tab validates
the ratings against actual run differential.

---

## Tuning Knobs

Edit the constants at the top of `app_v2.R`:

| Constant | Default | Description |
|---|---|---|
| `PA_PER_GAME` | 38 | Scales per-PA ratings into runs/game |
| `SUPPORTED_YEARS` | 2022:2026 | Years the app tries to load |
| `RECENCY_DECAY` | 0.5 | Per-year weight decay for the pooled model fit |
| `DISP_SIZE` | 8 | Negative Binomial dispersion for the matchup simulator |
