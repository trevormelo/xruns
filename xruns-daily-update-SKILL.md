---
name: xruns-daily-update
description: Performs the daily data refresh for Trevor's xruns MLB Shiny app. Downloads 12 CSV files from Baseball Savant leaderboards, places them in the correct subfolders of the xruns repository at /Users/trevormelo/Desktop/xruns, and updates the standings cache. Use this skill whenever Trevor asks to "update xruns data", "run the daily xruns update", "refresh the MLB data", "do today's data pull", "please do the update", or any similar phrasing referring to refreshing his Baseball Savant CSV data. This skill is designed to run end-to-end without prompting Trevor for input mid-flow.
---

# xruns Daily Data Update

## Purpose

This skill automates the daily refresh of Baseball Savant data for the **xruns** R Shiny app. It downloads 12 CSV files from public Baseball Savant leaderboards using `curl` (appending `&csv=true` to each URL), places them in the correct folders, and updates the standings cache using a Python script that hits the MLB Stats API directly.

Run unattended — no questions during the run. Only surface a problem if something genuinely blocks the workflow (e.g. a download returns 0 bytes or an HTTP error, a folder is missing, etc.).

## Workflow

1. Get today's date via `date +%Y-%m-%d` in bash.
2. Download all 12 CSV files using `curl` (see table below). Do all downloads in a single bash call for speed.
3. Update the standings cache using the Python script below.
4. Report a short summary.

**Do not attempt git operations.** Trevor handles commits himself.

## Paths

- **Repo root (bash):** `/sessions/hopeful-blissful-tesla/mnt/xruns`
- **Snapshots folder:** `/sessions/hopeful-blissful-tesla/mnt/xruns/2026 Data/Snapshots`
- **Player data folder:** `/sessions/hopeful-blissful-tesla/mnt/xruns/2026 Player Data`

Folder names contain spaces — always quote paths in shell commands.

## The 12 data sources

`{DATE}` = today's date in `YYYY-MM-DD` format.

Append `&csv=true` to URLs that already contain `?`, or `?csv=true` if they do not.

### Snapshots (4 files) → `2026 Data/Snapshots`

These use the full date in the filename and accumulate as historical daily snapshots. Never overwrite a prior day's file.

| Target filename | URL |
|-----------------|-----|
| `bat_t_{DATE}.csv` | `https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=batter-team&year=2026&position=&team=&filterType=bip&min=q` |
| `pit_t_{DATE}.csv` | `https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=pitcher-team&year=2026&position=&team=&filterType=bip&min=q` |
| `fld_t_{DATE}.csv` | `https://baseballsavant.mlb.com/leaderboard/fielding-run-value?gameType=Regular&seasonStart=2026&seasonEnd=2026&type=fielding-team&position=0&minInnings=q&minResults=1` |
| `br_t_{DATE}.csv` | `https://baseballsavant.mlb.com/leaderboard/baserunning-run-value?game_type=Regular&season_start=2026&season_end=2026&sortColumn=runner_runs_tot&sortDirection=desc&split=no&n=q&team=&type=Batting+Team&with_team_only=1` |

### Player data (8 files) → `2026 Player Data`

These use only the year in the filename and overwrite the existing file on each run (always-current cumulative leaderboards).

| Target filename | URL |
|-----------------|-----|
| `expected_stats_batter_2026.csv` | `https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=batter&year=2026&position=&team=&filterType=pa&min=25` |
| `expected_stats_pitcher_2026.csv` | `https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=pitcher&year=2026&position=&team=&filterType=pa&min=25` |
| `run_value_batter_2026.csv` | `https://baseballsavant.mlb.com/leaderboard/swing-take?year=2026&team=&leverage=Leveraged&group=Batter&type=All&sub_type=null&min=q` |
| `run_value_pitcher_2026.csv` | `https://baseballsavant.mlb.com/leaderboard/swing-take?year=2026&team=&leverage=Leveraged&group=Pitcher&type=All&sub_type=null&min=q` |
| `fielding_2026.csv` | `https://baseballsavant.mlb.com/leaderboard/fielding-run-value?gameType=Regular&seasonStart=2026&seasonEnd=2026&type=fielder&position=0&minInnings=q&minResults=1` |
| `running_2026.csv` | `https://baseballsavant.mlb.com/leaderboard/baserunning-run-value?game_type=Regular&season_start=2026&season_end=2026&sortColumn=runner_runs_tot&sortDirection=desc&split=no&n=0&team=&type=Batting&with_team_only=1` |
| `batting_stats_2026.csv` | `https://baseballsavant.mlb.com/leaderboard/custom?year=2026&type=batter&filter=&min=25&selections=k_percent%2Cbb_percent%2Con_base_percent%2Cbabip%2Cr_total_caught_stealing%2Cr_total_stolen_base%2Cxslg%2Cxwobacon%2Csquared_up_swing%2Cideal_angle_rate%2Cavg_hyper_speed%2Coz_swing_percent%2Cwhiff_percent%2Csprint_speed&chart=false&x=k_percent&y=k_percent&r=no&chartType=beeswarm&sort=whiff_percent&sortDir=asc` |
| `pitching_stats_2026.csv` | `https://baseballsavant.mlb.com/leaderboard/custom?year=2026&type=pitcher&filter=&min=5&selections=k_percent%2Cbb_percent%2Cxwobacon%2Cbarrel_batted_rate%2Cavg_hyper_speed%2Coz_swing_percent%2Cedge_percent%2Cwhiff_percent%2Cf_strike_percent%2Cfastball_avg_speed&chart=false&x=k_percent&y=k_percent&r=no&chartType=beeswarm&sort=1&sortDir=asc` |

## Step 1: Download all 12 files

Run all downloads in a single bash call:

```bash
DATE=$(date +%Y-%m-%d)
SNAP="/sessions/hopeful-blissful-tesla/mnt/xruns/2026 Data/Snapshots"
PLAYER="/sessions/hopeful-blissful-tesla/mnt/xruns/2026 Player Data"

echo "=== Snapshots ==="
curl -sL "https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=batter-team&year=2026&position=&team=&filterType=bip&min=q&csv=true" \
  -o "$SNAP/bat_t_${DATE}.csv" -w "bat_t: HTTP %{http_code}, %{size_download} bytes\n"
curl -sL "https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=pitcher-team&year=2026&position=&team=&filterType=bip&min=q&csv=true" \
  -o "$SNAP/pit_t_${DATE}.csv" -w "pit_t: HTTP %{http_code}, %{size_download} bytes\n"
curl -sL "https://baseballsavant.mlb.com/leaderboard/fielding-run-value?gameType=Regular&seasonStart=2026&seasonEnd=2026&type=fielding-team&position=0&minInnings=q&minResults=1&csv=true" \
  -o "$SNAP/fld_t_${DATE}.csv" -w "fld_t: HTTP %{http_code}, %{size_download} bytes\n"
curl -sL "https://baseballsavant.mlb.com/leaderboard/baserunning-run-value?game_type=Regular&season_start=2026&season_end=2026&sortColumn=runner_runs_tot&sortDirection=desc&split=no&n=q&team=&type=Batting+Team&with_team_only=1&csv=true" \
  -o "$SNAP/br_t_${DATE}.csv" -w "br_t: HTTP %{http_code}, %{size_download} bytes\n"

echo ""
echo "=== Player Data ==="
curl -sL "https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=batter&year=2026&position=&team=&filterType=pa&min=25&csv=true" \
  -o "$PLAYER/expected_stats_batter_2026.csv" -w "expected_stats_batter: HTTP %{http_code}, %{size_download} bytes\n"
curl -sL "https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=pitcher&year=2026&position=&team=&filterType=pa&min=25&csv=true" \
  -o "$PLAYER/expected_stats_pitcher_2026.csv" -w "expected_stats_pitcher: HTTP %{http_code}, %{size_download} bytes\n"
curl -sL "https://baseballsavant.mlb.com/leaderboard/swing-take?year=2026&team=&leverage=Leveraged&group=Batter&type=All&sub_type=null&min=q&csv=true" \
  -o "$PLAYER/run_value_batter_2026.csv" -w "run_value_batter: HTTP %{http_code}, %{size_download} bytes\n"
curl -sL "https://baseballsavant.mlb.com/leaderboard/swing-take?year=2026&team=&leverage=Leveraged&group=Pitcher&type=All&sub_type=null&min=q&csv=true" \
  -o "$PLAYER/run_value_pitcher_2026.csv" -w "run_value_pitcher: HTTP %{http_code}, %{size_download} bytes\n"
curl -sL "https://baseballsavant.mlb.com/leaderboard/fielding-run-value?gameType=Regular&seasonStart=2026&seasonEnd=2026&type=fielder&position=0&minInnings=q&minResults=1&csv=true" \
  -o "$PLAYER/fielding_2026.csv" -w "fielding: HTTP %{http_code}, %{size_download} bytes\n"
curl -sL "https://baseballsavant.mlb.com/leaderboard/baserunning-run-value?game_type=Regular&season_start=2026&season_end=2026&sortColumn=runner_runs_tot&sortDirection=desc&split=no&n=0&team=&type=Batting&with_team_only=1&csv=true" \
  -o "$PLAYER/running_2026.csv" -w "running: HTTP %{http_code}, %{size_download} bytes\n"
curl -sL "https://baseballsavant.mlb.com/leaderboard/custom?year=2026&type=batter&filter=&min=25&selections=k_percent%2Cbb_percent%2Con_base_percent%2Cbabip%2Cr_total_caught_stealing%2Cr_total_stolen_base%2Cxslg%2Cxwobacon%2Csquared_up_swing%2Cideal_angle_rate%2Cavg_hyper_speed%2Coz_swing_percent%2Cwhiff_percent%2Csprint_speed&chart=false&x=k_percent&y=k_percent&r=no&chartType=beeswarm&sort=whiff_percent&sortDir=asc&csv=true" \
  -o "$PLAYER/batting_stats_2026.csv" -w "batting_stats: HTTP %{http_code}, %{size_download} bytes\n"
curl -sL "https://baseballsavant.mlb.com/leaderboard/custom?year=2026&type=pitcher&filter=&min=5&selections=k_percent%2Cbb_percent%2Cxwobacon%2Cbarrel_batted_rate%2Cavg_hyper_speed%2Coz_swing_percent%2Cedge_percent%2Cwhiff_percent%2Cf_strike_percent%2Cfastball_avg_speed&chart=false&x=k_percent&y=k_percent&r=no&chartType=beeswarm&sort=1&sortDir=asc&csv=true" \
  -o "$PLAYER/pitching_stats_2026.csv" -w "pitching_stats: HTTP %{http_code}, %{size_download} bytes\n"
```

Check each response for HTTP 200 and a non-zero size. If any file comes back as 0 bytes or non-200, retry once before marking it failed.

## Step 2: Update standings cache

The standings cache lives at `2026 Data/standings_daily_2026.csv`. Run this via `python3 << 'EOF' ... EOF` in bash:

```python
import urllib.request, json, csv, os, re

season = 2026
repo_root = "/sessions/hopeful-blissful-tesla/mnt/xruns"
snapshot_dir = f"{repo_root}/2026 Data/Snapshots"
cache_path = f"{repo_root}/2026 Data/standings_daily_2026.csv"

team_lookup = {
    108: ("LAA","Los Angeles Angels"), 109: ("ARI","Arizona Diamondbacks"),
    110: ("BAL","Baltimore Orioles"), 111: ("BOS","Boston Red Sox"),
    112: ("CHC","Chicago Cubs"), 113: ("CIN","Cincinnati Reds"),
    114: ("CLE","Cleveland Guardians"), 115: ("COL","Colorado Rockies"),
    116: ("DET","Detroit Tigers"), 117: ("HOU","Houston Astros"),
    118: ("KC","Kansas City Royals"), 119: ("LAD","Los Angeles Dodgers"),
    120: ("WSH","Washington Nationals"), 121: ("NYM","New York Mets"),
    133: ("OAK","Athletics"), 134: ("PIT","Pittsburgh Pirates"),
    135: ("SD","San Diego Padres"), 136: ("SEA","Seattle Mariners"),
    137: ("SF","San Francisco Giants"), 138: ("STL","St. Louis Cardinals"),
    139: ("TB","Tampa Bay Rays"), 140: ("TEX","Texas Rangers"),
    141: ("TOR","Toronto Blue Jays"), 142: ("MIN","Minnesota Twins"),
    143: ("PHI","Philadelphia Phillies"), 144: ("ATL","Atlanta Braves"),
    145: ("CWS","Chicago White Sox"), 146: ("MIA","Miami Marlins"),
    147: ("NYY","New York Yankees"), 158: ("MIL","Milwaukee Brewers"),
}

dates = sorted(set(
    re.sub(r'^bat_t_(.+)\.csv$', r'\1', f)
    for f in os.listdir(snapshot_dir)
    if re.match(r'^bat_t_\d{4}-\d{2}-\d{2}\.csv$', f)
))

existing_rows = []
existing_keys = set()
if os.path.exists(cache_path):
    with open(cache_path, newline='') as f:
        for row in csv.DictReader(f):
            existing_rows.append(row)
            existing_keys.add((row['snapshot_date'], str(row['team_id'])))

def fetch_standings(d):
    url = (f"https://statsapi.mlb.com/api/v1/standings?"
           f"leagueId=103,104&season={season}&standingsTypes=regularSeason&date={d}")
    with urllib.request.urlopen(url, timeout=15) as resp:
        payload = json.loads(resp.read())
    rows = []
    for div in payload.get('records', []):
        for tr in div.get('teamRecords', []):
            tid = int(tr['team']['id'])
            abbrev, name = team_lookup.get(tid, ("UNK","Unknown"))
            rows.append({'snapshot_date': d, 'season': season, 'team_id': tid,
                         'abbrev': abbrev, 'team_name': name,
                         'W': int(tr['wins']), 'L': int(tr['losses']),
                         'W-L%': float(tr['winningPercentage'])})
    return rows

new_rows = []
for d in dates:
    if all((d, str(tid)) in existing_keys for tid in team_lookup):
        continue
    fetched = fetch_standings(d)
    print(f"Fetched {d}: {len(fetched)} teams")
    new_rows.extend(fetched)

all_rows = existing_rows.copy()
idx = {(r['snapshot_date'], str(r['team_id'])): i for i, r in enumerate(all_rows)}
for r in new_rows:
    key = (r['snapshot_date'], str(r['team_id']))
    if key in idx:
        all_rows[idx[key]] = {k: str(v) for k, v in r.items()}
    else:
        all_rows.append({k: str(v) for k, v in r.items()})

all_rows.sort(key=lambda r: (r['snapshot_date'], int(r['team_id'])))

fieldnames = ['snapshot_date','season','team_id','abbrev','team_name','W','L','W-L%']
with open(cache_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(all_rows)

print(f"Wrote {len(all_rows)} total rows to standings_daily_2026.csv")
from collections import Counter
for d in sorted(Counter(r['snapshot_date'] for r in all_rows).keys())[-3:]:
    c = sum(1 for r in all_rows if r['snapshot_date'] == d)
    print(f"  {d}: {c} teams")
```

This is idempotent — running twice on the same day is safe. It only fetches standings for dates not already fully in the cache.

## Step 3: Summary report

```
xruns daily update — YYYY-MM-DD

Snapshots (2026 Data/Snapshots):
  ✓ bat_t_YYYY-MM-DD.csv
  ✓ pit_t_YYYY-MM-DD.csv
  ✓ fld_t_YYYY-MM-DD.csv
  ✓ br_t_YYYY-MM-DD.csv

Player data (2026 Player Data):
  ✓ expected_stats_batter_2026.csv
  ✓ expected_stats_pitcher_2026.csv
  ✓ run_value_batter_2026.csv
  ✓ run_value_pitcher_2026.csv
  ✓ fielding_2026.csv
  ✓ running_2026.csv
  ✓ batting_stats_2026.csv
  ✓ pitching_stats_2026.csv

Standings: ✓ 30 teams added for YYYY-MM-DD (NNN total rows)
```

Use ✗ instead of ✓ for any file that failed, with a one-line reason.

## Things to watch out for

- **Don't run git.** Trevor handles commits himself.
- **Don't overwrite prior snapshots.** Files 1–4 have the date in the name, so this is safe by design.
- **The Snapshots folder grows daily.** That's intentional — don't clean it up.
- **Folder names have spaces.** Always quote paths in shell commands.
- **Don't ask Trevor questions.** Make the most reasonable choice and note anything unusual in the summary.
- **Year in URLs.** The URLs hardcode `2026`. At the start of a new season Trevor will update these manually — trust them as written.
