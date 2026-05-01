# =============================================================================
# xRuns: an MLB Rating System — Multi-year (2022 - 2026)
# -----------------------------------------------------------------------------
# Ranks teams by Offensive, Defensive, and Overall ratings expressed in
# "expected runs per 9 innings vs. an average team."
# Ranks individual players by predicted run value derived from expected stats.
#
# DATA LAYOUT:
#   2022 Data/ expected_stats_batter_2022.csv, expected_stats_pitcher_2022.csv,
#              run_value_batter_2022.csv,      run_value_pitcher_2022.csv,
#              league_standings_2022.csv
#   2023 Data/ ...same pattern ending in _2023...
#   2024 Data/ ...same pattern...
#   2025 Data/ ...uses mlb_standings_2025.csv for the standings file...
#   2026 Data/ ...no standings file (in-season)...
#
# MODEL:
#   - Pooled weighted linear regression fit on ALL available player-seasons
#     from 2022-2026 (weights = run-value PA, so high-sample players drive fit).
#   - Batters:  runs/PA ~ est_woba + bip_rate
#   - Pitchers: runs/PA ~ est_woba + xera
#   - Model predictions are applied back to every player-season, then PA-weighted
#     aggregated to teams for each year. Overall team rating = expected runs
#     above an average MLB team in a neutral 9-inning game.
#
# IMPORTANT: The model is trained only on player-level expected stats -> actual
# run value. It is NOT trained against team W-L records. Team records are used
# solely as an out-of-model validation (Standings Check tab).
# =============================================================================

# ---- packages ----------------------------------------------------------------
library(shiny)
library(bslib)
library(DT)
library(dplyr)
library(tidyr)
library(readr)
library(tibble)
library(ggplot2)
library(plotly)
library(scales)
library(purrr)
library(mlbplotR)
library(jsonlite)

# ---- constants ---------------------------------------------------------------
PA_PER_GAME     <- 38       # team PAs per 9-inning game
OUTS_PER_GAME   <- 27       # defensive outs per game at one fielding position
FIELDER_OUTS_PER_GAME <- 9 * OUTS_PER_GAME  # 243 fielder-outs per team-game (9 positions)
SUPPORTED_YEARS <- 2022:2026

# ---- reliability scalars for raw Statcast run values -------------------------
# Hitting and pitching run values are produced via a regression model that
# naturally dampens outliers toward expected outcomes. Baserunning and fielding
# run values are raw Statcast totals with no equivalent shrinkage, which causes
# single-season extremes to be over-weighted relative to their true signal.
#
# To place all four components on the same epistemological footing, we apply
# reliability-based scalars derived from published sabermetric research:
#
#   FIELDING_RELIABILITY = 0.50
#     UZR year-over-year correlation is ~0.50 at the player level, making
#     fielding the noisiest WAR component. Source: Lichtman (2010),
#     "The FanGraphs UZR Primer" (https://blogs.fangraphs.com/the-fangraphs-uzr-primer/)
#     and "How Reliable is UZR?" via The Hardball Times
#     (https://tht.fangraphs.com/tht-live/how-reliable-is-uzr/).
#     FanGraphs explicitly recommends 3-year samples for stable fielding estimates.
#
#   BASERUNNING_RELIABILITY = 0.70
#     BsR/UBR require more than one year of data to become reliably predictive,
#     but speed and baserunning instincts are more persistent skills than
#     fielding range, warranting lighter dampening. Source: FanGraphs Sabermetrics
#     Library — BsR (https://library.fangraphs.com/offense/bsr/) and
#     UBR (https://library.fangraphs.com/offense/ubr/).
FIELDING_RELIABILITY    <- 0.50
BASERUNNING_RELIABILITY <- 0.70

# ---- TEAM_META ---------------------------------------------------------------
# MLB team_id -> abbreviation / full name mapping (MLB Stats API IDs).
# mlb_abbrev maps our abbreviations to mlbplotR's conventions (KCR, WSN, etc.)
TEAM_META <- tibble::tribble(
  ~team_id, ~abbrev, ~team_name,
  108, "LAA", "Los Angeles Angels",
  109, "ARI", "Arizona Diamondbacks",
  110, "BAL", "Baltimore Orioles",
  111, "BOS", "Boston Red Sox",
  112, "CHC", "Chicago Cubs",
  113, "CIN", "Cincinnati Reds",
  114, "CLE", "Cleveland Guardians",
  115, "COL", "Colorado Rockies",
  116, "DET", "Detroit Tigers",
  117, "HOU", "Houston Astros",
  118, "KC",  "Kansas City Royals",
  119, "LAD", "Los Angeles Dodgers",
  120, "WSH", "Washington Nationals",
  121, "NYM", "New York Mets",
  133, "OAK", "Athletics",
  134, "PIT", "Pittsburgh Pirates",
  135, "SD",  "San Diego Padres",
  136, "SEA", "Seattle Mariners",
  137, "SF",  "San Francisco Giants",
  138, "STL", "St. Louis Cardinals",
  139, "TB",  "Tampa Bay Rays",
  140, "TEX", "Texas Rangers",
  141, "TOR", "Toronto Blue Jays",
  142, "MIN", "Minnesota Twins",
  143, "PHI", "Philadelphia Phillies",
  144, "ATL", "Atlanta Braves",
  145, "CWS", "Chicago White Sox",
  146, "MIA", "Miami Marlins",
  147, "NYY", "New York Yankees",
  158, "MIL", "Milwaukee Brewers"
) %>%
  dplyr::mutate(mlb_abbrev = dplyr::case_when(
    abbrev == "KC"  ~ "KCR",
    abbrev == "WSH" ~ "WSN",
    abbrev == "SD"  ~ "SDP",
    abbrev == "SF"  ~ "SFG",
    abbrev == "TB"  ~ "TBR",
    TRUE ~ abbrev
  ))

# Hardcode ESPN CDN logo URLs for all 30 teams.
# Using our own abbrev -> ESPN slug mapping avoids any mlbplotR join mismatches
# (mlbplotR uses KCR/SDP/SFG/TBR/WSN while our abbrevs are KC/SD/SF/TB/WSH).
ESPN_SLUGS <- c(
  LAA = "laa", ARI = "ari", BAL = "bal", BOS = "bos", CHC = "chc",
  CIN = "cin", CLE = "cle", COL = "col", DET = "det", HOU = "hou",
  KC  = "kc",  LAD = "lad", WSH = "wsh", NYM = "nym", OAK = "oak",
  PIT = "pit", SD  = "sd",  SEA = "sea", SF  = "sf",  STL = "stl",
  TB  = "tb",  TEX = "tex", TOR = "tor", MIN = "min", PHI = "phi",
  ATL = "atl", CWS = "chw", MIA = "mia", NYY = "nyy", MIL = "mil"
)
TEAM_META$team_logo_espn <- paste0(
  "https://a.espncdn.com/i/teamlogos/mlb/500/",
  ESPN_SLUGS[TEAM_META$abbrev],
  ".png"
)

# Hardcoded primary colors for all 30 MLB teams (used as definitive source).
# These are the official/canonical primary colors for each franchise. mlbplotR
# is checked first and its value wins only when present and valid; these hex
# codes serve as the complete, reliable fallback so every team always has a color.
TEAM_COLOR_FALLBACKS <- c(
  LAA = "#BA0021", ARI = "#A71930", BAL = "#DF4601", BOS = "#BD3039",
  CHC = "#0E3386", CIN = "#C6011F", CLE = "#0C2340", COL = "#33006F",
  DET = "#0C2340", HOU = "#002D62", KC  = "#004687", LAD = "#005A9C",
  WSH = "#AB0003", NYM = "#002D72", OAK = "#003831", PIT = "#FDB827",
  SD  = "#2F241D", SEA = "#0C2C56", SF  = "#FD5A1E", STL = "#C41E3A",
  TB  = "#092C5C", TEX = "#003278", TOR = "#134A8E", MIN = "#002B5C",
  PHI = "#E81828", ATL = "#CE1141", CWS = "#27251F", MIA = "#00A3E0",
  NYY = "#0C2340", MIL = "#12284B"
)

# Try to pull team primary colors from mlbplotR; fall back to hardcoded hex if
# unavailable or if the team is missing from the mlbplotR data. mlbplotR may
# not include every franchise (e.g. relocated teams, expansion) or may use
# different column names across package versions — we handle all cases here.
mlb_colors <- tryCatch({
  teams_raw <- mlbplotR::load_mlb_teams()
  # mlbplotR column may be team_color or team_color1 depending on version
  color_col <- intersect(c("team_color", "team_color1", "primary_color"), names(teams_raw))[1]
  if (is.na(color_col)) stop("no color column found")
  # Use base R rename to avoid tidy-eval issues inside tryCatch
  out <- teams_raw[, c("team_abbr", color_col)]
  names(out)[2] <- "team_color"
  out
}, error = function(e) NULL)

if (!is.null(mlb_colors)) {
  TEAM_META <- TEAM_META %>%
    dplyr::left_join(mlb_colors, by = c("mlb_abbrev" = "team_abbr"))
} else {
  TEAM_META$team_color <- NA_character_
}

# Coalesce: mlbplotR value → hardcoded fallback → dark navy default
TEAM_META$team_color <- dplyr::coalesce(
  # Only use mlbplotR color if it looks like a valid hex code
  dplyr::if_else(
    grepl("^#[0-9A-Fa-f]{6}$", as.character(TEAM_META$team_color)),
    as.character(TEAM_META$team_color),
    NA_character_
  ),
  unname(TEAM_COLOR_FALLBACKS[TEAM_META$abbrev]),
  "#151922"
)

# Map of names that may appear in the standings CSVs to a canonical team_id.
STANDING_NAME_ALIASES <- c(
  "Oakland Athletics" = 133L,
  "Athletics"         = 133L
)

# Abbreviation aliases: some data sources use non-standard abbreviations.
# Map them to the canonical abbrev used in TEAM_META.
ABBREV_ALIASES <- c(
  "ATH" = "OAK",   # Athletics (post-Oakland)
  "AZ"  = "ARI"    # Arizona Diamondbacks alternate abbrev
)

# ---- GitHub raw URL base -----------------------------------------------------
# All CSV files are read directly from GitHub. The folder structure in the repo
# must mirror the original local layout (e.g. "2022 Data/", "2026 Data/Snapshots/", …).
GITHUB_RAW_BASE <- "https://raw.githubusercontent.com/trevormelo/xruns/main"

# GitHub API base for listing directory contents (used to enumerate snapshot files).
GITHUB_API_BASE <- "https://api.github.com/repos/trevormelo/xruns/contents"

# Helper: fetch a CSV from a GitHub raw URL; returns NULL silently on 404/error.
read_github_csv <- function(url) {
  prefix <- paste0(GITHUB_RAW_BASE, "/")
  if (startsWith(url, prefix)) {
    local_path <- utils::URLdecode(sub(prefix, "", url, fixed = TRUE))
    if (file.exists(local_path)) {
      return(tryCatch(
        readr::read_csv(local_path, show_col_types = FALSE),
        error   = function(e) NULL,
        warning = function(w) NULL
      ))
    }
  }
  tryCatch(
    readr::read_csv(url, show_col_types = FALSE),
    error   = function(e) NULL,
    warning = function(w) NULL
  )
}

# Helper: list files in a GitHub directory via the API; returns a character
# vector of filenames (not full URLs), or character(0) on failure.
# 404s (e.g. years without a Snapshots/ folder) are silently ignored.
github_list_files <- function(path_in_repo) {
  if (dir.exists(path_in_repo)) {
    return(basename(list.files(path_in_repo, all.files = FALSE, no.. = TRUE)))
  }
  api_url <- paste0(GITHUB_API_BASE, "/", utils::URLencode(path_in_repo, repeated = TRUE))
  response <- tryCatch(
    suppressWarnings(jsonlite::fromJSON(api_url)),
    error = function(e) NULL
  )
  if (is.null(response) || !is.data.frame(response) || !"name" %in% names(response))
    return(character(0))
  response$name
}

# Helper: build a GitHub raw URL for a file inside a repo subdirectory.
# folder_path should be the path relative to repo root (e.g. "2026 Data/Snapshots").
github_raw_url <- function(folder_path, filename) {
  paste0(GITHUB_RAW_BASE, "/",
         utils::URLencode(folder_path, repeated = TRUE), "/",
         utils::URLencode(filename,    repeated = TRUE))
}

# ---- helpers -----------------------------------------------------------------
safe_numeric <- function(x) suppressWarnings(as.numeric(gsub('"', "", as.character(x))))

# Null-coalescing operator (defined here in case rlang is not loaded)
`%||%` <- function(a, b) if (!is.null(a)) a else b

split_name <- function(x) {
  parts <- strsplit(as.character(x), ",\\s*")
  vapply(parts, function(p) {
    if (length(p) == 2) paste(p[2], p[1]) else as.character(x[1])
  }, character(1))
}

resolve_team_id <- function(name_vec) {
  out <- integer(length(name_vec))
  for (i in seq_along(name_vec)) {
    nm <- name_vec[i]
    if (is.na(nm) || nm == "") { out[i] <- NA_integer_; next }
    if (nm %in% names(STANDING_NAME_ALIASES)) {
      out[i] <- STANDING_NAME_ALIASES[[nm]]; next
    }
    idx <- match(nm, TEAM_META$team_name)
    out[i] <- if (is.na(idx)) NA_integer_ else TEAM_META$team_id[idx]
  }
  out
}

# ---- per-year loading --------------------------------------------------------
# ---- FILE NAMING CONVENTIONS (short names — new standard) -------------------
# Player files (prior completed seasons):
#   bat_YYYY.csv      expected stats batters   (was expected_stats_batter_YYYY.csv)
#   pit_YYYY.csv      expected stats pitchers  (was expected_stats_pitcher_YYYY.csv)
#   rv_bat_YYYY.csv   run value batters        (was run_value_batter_YYYY.csv)
#   rv_pit_YYYY.csv   run value pitchers       (was run_value_pitcher_YYYY.csv)
#   br_YYYY.csv       baserunning              (was running_YYYY.csv)
#   fld_YYYY.csv      fielding                 (was fielding_YYYY.csv)
#   std_YYYY.csv      standings                (was league_standings_YYYY.csv / mlb_standings_YYYY.csv)
#
# Team snapshot files (current season, YYYY Data/Snapshots/):
#   bat_t_YYYY-MM-DD.csv   team batting expected stats
#   pit_t_YYYY-MM-DD.csv   team pitching expected stats
#   br_t_YYYY-MM-DD.csv    team baserunning run values
#   fld_t_YYYY-MM-DD.csv   team fielding run values
#
# Old names are still recognised as a fallback — files are tried in order and the
# first successful read wins.
# -----------------------------------------------------------------------------

# Try to read a CSV from one of several candidate URLs. Returns the first
# successfully read data frame, or NULL if all fail.
try_read_csv_urls <- function(...) {
  for (url in list(...)) {
    df <- read_github_csv(url)
    if (!is.null(df)) return(df)
  }
  NULL
}

# Returns the folder path (relative to repo root) that contains player files for
# year y, or NULL if neither candidate folder has the required files.
# Because we can't call dir.exists() on URLs, we probe by attempting reads.
find_player_folder_url <- function(base, y) {
  candidates <- c(
    paste0(y, " Data"),
    paste0(y, " Player Data")
  )
  for (folder in candidates) {
    # Probe with new short name first
    eb_url <- github_raw_url(folder, sprintf("bat_%d.csv", y))
    eb     <- read_github_csv(eb_url)
    if (!is.null(eb)) return(list(folder = folder, short_names = TRUE))
    # Probe with old name
    eb_url <- github_raw_url(folder, sprintf("expected_stats_batter_%d.csv", y))
    eb     <- read_github_csv(eb_url)
    if (!is.null(eb)) return(list(folder = folder, short_names = FALSE))
  }
  NULL
}

load_year <- function(base, y) {
  info <- find_player_folder_url(base, y)
  if (is.null(info)) return(NULL)
  f          <- info$folder
  short_names <- info$short_names
  
  if (short_names) {
    eb <- read_github_csv(github_raw_url(f, sprintf("bat_%d.csv",    y)))
    ep <- read_github_csv(github_raw_url(f, sprintf("pit_%d.csv",    y)))
    rb <- read_github_csv(github_raw_url(f, sprintf("rv_bat_%d.csv", y)))
    rp <- read_github_csv(github_raw_url(f, sprintf("rv_pit_%d.csv", y)))
  } else {
    eb <- read_github_csv(github_raw_url(f, sprintf("expected_stats_batter_%d.csv",  y)))
    ep <- read_github_csv(github_raw_url(f, sprintf("expected_stats_pitcher_%d.csv", y)))
    rb <- read_github_csv(github_raw_url(f, sprintf("run_value_batter_%d.csv",       y)))
    rp <- read_github_csv(github_raw_url(f, sprintf("run_value_pitcher_%d.csv",      y)))
  }
  if (any(sapply(list(eb, ep, rb, rp), is.null))) return(NULL)
  
  # ---- Baserunning (may be missing for older years) ---------------------------
  run_df_raw <- try_read_csv_urls(
    github_raw_url(f, sprintf("br_%d.csv",      y)),
    github_raw_url(f, sprintf("running_%d.csv", y))
  )
  run_df <- if (!is.null(run_df_raw)) {
    rd <- run_df_raw %>%
      transmute(
        player_id     = as.integer(player_id),
        br_runs       = safe_numeric(runner_runs_tot),
        br_opps       = safe_numeric(N_runner_moved)
      ) %>%
      filter(!is.na(player_id)) %>%
      group_by(player_id) %>%
      summarise(
        br_runs = sum(br_runs, na.rm = TRUE),
        br_opps = sum(br_opps, na.rm = TRUE),
        .groups = "drop"
      )
    rd
  } else {
    tibble(player_id = integer(0), br_runs = numeric(0), br_opps = numeric(0))
  }
  
  # ---- Fielding (may be missing for older years) ------------------------------
  fld_df_raw <- try_read_csv_urls(
    github_raw_url(f, sprintf("fld_%d.csv",      y)),
    github_raw_url(f, sprintf("fielding_%d.csv", y))
  )
  fld_df <- if (!is.null(fld_df_raw)) {
    fd <- fld_df_raw %>%
      transmute(
        player_id  = as.integer(id),
        fld_runs   = safe_numeric(total_runs),
        fld_outs   = safe_numeric(outs_total)
      ) %>%
      filter(!is.na(player_id)) %>%
      group_by(player_id) %>%
      summarise(
        fld_runs = sum(fld_runs, na.rm = TRUE),
        fld_outs = sum(fld_outs, na.rm = TRUE),
        .groups = "drop"
      )
    fd
  } else {
    tibble(player_id = integer(0), fld_runs = numeric(0), fld_outs = numeric(0))
  }
  
  # Search for standings — try the player folder and the primary year folder.
  year_data_folder <- paste0(y, " Data")
  st <- NULL
  for (search_folder in unique(c(f, year_data_folder))) {
    for (stem in c("std", "league_standings", "mlb_standings", "standings")) {
      st <- read_github_csv(github_raw_url(search_folder, sprintf("%s_%d.csv", stem, y)))
      if (!is.null(st)) break
    }
    if (!is.null(st)) break
  }
  if (!is.null(st)) {
    st <- st %>%
      filter(!is.na(Rk), Tm != "Average") %>%
      mutate(
        team_id        = resolve_team_id(Tm),
        W              = safe_numeric(W),
        L              = safe_numeric(L),
        `W-L%`         = safe_numeric(`W-L%`),
        R              = safe_numeric(R),
        RA             = safe_numeric(RA),
        Rdiff          = safe_numeric(Rdiff),
        rdiff_per_game = R - RA,
        year           = y
      )
  }
  
  name_col <- "last_name, first_name"
  eb$player <- split_name(eb[[name_col]])
  ep$player <- split_name(ep[[name_col]])
  rb$player <- split_name(rb[[name_col]])
  rp$player <- split_name(rp[[name_col]])
  ep$era  <- safe_numeric(ep$era)
  ep$xera <- safe_numeric(ep$xera)
  
  eb$season_year <- y; ep$season_year <- y
  rb$season_year <- y; rp$season_year <- y
  
  batters <- eb %>%
    select(player, player_id, season_year,
           pa_exp = pa, bip, ba, est_ba, slg, est_slg, woba, est_woba) %>%
    inner_join(rb %>% select(player_id, team_id, pa_rv = pa, pitches,
                             runs_all, runs_heart, runs_shadow,
                             runs_chase, runs_waste),
               by = "player_id") %>%
    left_join(TEAM_META, by = "team_id") %>%
    left_join(run_df, by = "player_id") %>%
    mutate(pa          = pmax(pa_exp, pa_rv, na.rm = TRUE),
           runs_per_pa = runs_all / pmax(pa_rv, 1),
           bip_rate    = bip / pmax(pa_exp, 1),
           br_runs     = dplyr::coalesce(br_runs, 0),
           br_opps     = dplyr::coalesce(br_opps, 0),
           br_per_pa   = ifelse(pa_rv > 0, br_runs / pa_rv, 0))
  
  pitchers <- ep %>%
    select(player, player_id, season_year,
           pa_exp = pa, bip, ba, est_ba, slg, est_slg, woba, est_woba, era, xera) %>%
    inner_join(rp %>% select(player_id, team_id, pa_rv = pa, pitches,
                             runs_all, runs_heart, runs_shadow,
                             runs_chase, runs_waste),
               by = "player_id") %>%
    left_join(TEAM_META, by = "team_id") %>%
    mutate(pa          = pmax(pa_exp, pa_rv, na.rm = TRUE),
           runs_per_pa = runs_all / pmax(pa_rv, 1),
           bip_rate    = bip / pmax(pa_exp, 1))
  
  # Attach fielding to whichever role that player appears in.
  # Rule: if the player appears in BOTH batters and pitchers (two-way, e.g. Ohtani),
  # attribute their fielding totals to whichever role has more PA — so team-level
  # team fielding aggregates don't double-count.
  if (nrow(fld_df) > 0) {
    bat_ids <- batters$player_id
    pit_ids <- pitchers$player_id
    dual_ids <- intersect(bat_ids, pit_ids)
    
    bat_pa_lookup <- setNames(batters$pa_rv,  as.character(batters$player_id))
    pit_pa_lookup <- setNames(pitchers$pa_rv, as.character(pitchers$player_id))
    
    # For dual-role players decide the primary role once.
    primary_role_for_dual <- vapply(as.character(dual_ids), function(id) {
      b_pa <- suppressWarnings(as.numeric(bat_pa_lookup[[id]])); if (is.na(b_pa)) b_pa <- 0
      p_pa <- suppressWarnings(as.numeric(pit_pa_lookup[[id]])); if (is.na(p_pa)) p_pa <- 0
      if (b_pa >= p_pa) "batter" else "pitcher"
    }, character(1))
    
    fld_bat_ids <- fld_df$player_id[fld_df$player_id %in% bat_ids]
    fld_pit_ids <- fld_df$player_id[fld_df$player_id %in% pit_ids]
    
    # Remove from the non-primary role for duals.
    dual_as_pitcher <- dual_ids[primary_role_for_dual == "pitcher"]
    dual_as_batter  <- dual_ids[primary_role_for_dual == "batter"]
    
    bat_fld <- fld_df %>% filter(player_id %in% setdiff(fld_bat_ids, dual_as_pitcher))
    pit_fld <- fld_df %>% filter(player_id %in% setdiff(fld_pit_ids, dual_as_batter))
    
    batters  <- batters  %>% left_join(bat_fld, by = "player_id")
    pitchers <- pitchers %>% left_join(pit_fld, by = "player_id")
  } else {
    batters  <- batters  %>% mutate(fld_runs = NA_real_, fld_outs = NA_real_)
    pitchers <- pitchers %>% mutate(fld_runs = NA_real_, fld_outs = NA_real_)
  }
  
  batters <- batters %>%
    mutate(
      fld_runs    = dplyr::coalesce(fld_runs, NA_real_),
      fld_outs    = dplyr::coalesce(fld_outs, NA_real_),
      fld_per_out = ifelse(!is.na(fld_outs) & fld_outs > 0, fld_runs / fld_outs, NA_real_)
    )
  pitchers <- pitchers %>%
    mutate(
      fld_runs    = dplyr::coalesce(fld_runs, NA_real_),
      fld_outs    = dplyr::coalesce(fld_outs, NA_real_),
      fld_per_out = ifelse(!is.na(fld_outs) & fld_outs > 0, fld_runs / fld_outs, NA_real_)
    )
  
  list(year = y, batters = batters, pitchers = pitchers, standings = st,
       run_df = run_df, fld_df = fld_df)
}

# ---- helper: canonicalise abbreviations found in team-level files ------------
canonicalize_abbrev <- function(abbrev_vec) {
  dplyr::coalesce(ABBREV_ALIASES[abbrev_vec], abbrev_vec)
}

# =============================================================================
# SNAPSHOT INFRASTRUCTURE — in-season daily team data
# =============================================================================
# Each day you export four files into  YYYY Data/Snapshots/  named:
#   bat_t_YYYY-MM-DD.csv    team batting expected stats
#   pit_t_YYYY-MM-DD.csv    team pitching expected stats
#   br_t_YYYY-MM-DD.csv     team baserunning run values
#   fld_t_YYYY-MM-DD.csv    team fielding run values
#
# Every file is a cumulative season-to-date snapshot.  Subtracting two
# snapshots gives stats for any rolling time window (last 7 days, etc.).
# =============================================================================

# ---- Parse and tidy one set of four team files into a snapshot list --------
# bat_p / pit_p / br_p / fld_p are now GitHub raw URLs (not local paths).
load_one_snapshot <- function(bat_p, pit_p, br_p, fld_p, y, standings_folder = NULL) {
  tb   <- read_github_csv(bat_p); if (is.null(tb))   return(NULL)
  tp   <- read_github_csv(pit_p); if (is.null(tp))   return(NULL)
  tbr  <- read_github_csv(br_p);  if (is.null(tbr))  return(NULL)
  tfld <- read_github_csv(fld_p); if (is.null(tfld)) return(NULL)
  
  # Normalise team abbreviation column
  get_abbrev <- function(df) {
    col <- if ("team_id" %in% names(df)) "team_id"
    else if ("abbrev" %in% names(df)) "abbrev"
    else NA
    if (is.na(col)) return(rep(NA_character_, nrow(df)))
    canonicalize_abbrev(as.character(df[[col]]))
  }
  tb$abbrev <- get_abbrev(tb)
  tp$abbrev <- get_abbrev(tp)
  
  tbr$abbrev <- if ("player_id" %in% names(tbr)) {
    TEAM_META$abbrev[match(as.integer(tbr$player_id), TEAM_META$team_id)]
  } else if ("entity_name" %in% names(tbr)) {
    canonicalize_abbrev(as.character(tbr$entity_name))
  } else {
    rep(NA_character_, nrow(tbr))
  }
  
  if ("id" %in% names(tfld)) {
    tfld$abbrev <- TEAM_META$abbrev[match(as.integer(tfld$id), TEAM_META$team_id)]
  } else if ("name" %in% names(tfld)) {
    tfld$abbrev <- TEAM_META$abbrev[match(tfld$name, TEAM_META$team_name)]
  }
  
  team_batting <- tb %>%
    dplyr::transmute(
      abbrev,
      pa       = safe_numeric(pa),
      bip      = safe_numeric(bip),
      est_ba   = safe_numeric(est_ba),
      est_slg  = safe_numeric(est_slg),
      est_woba = safe_numeric(est_woba)
    ) %>%
    dplyr::filter(!is.na(abbrev), !is.na(pa)) %>%
    dplyr::mutate(bip_rate = bip / pmax(pa, 1))
  
  team_pitching <- tp %>%
    dplyr::transmute(
      abbrev,
      pa       = safe_numeric(pa),
      bip      = safe_numeric(bip),
      est_ba   = safe_numeric(est_ba),
      est_slg  = safe_numeric(est_slg),
      est_woba = safe_numeric(est_woba)
    ) %>%
    dplyr::filter(!is.na(abbrev), !is.na(pa)) %>%
    dplyr::mutate(bip_rate = bip / pmax(pa, 1))
  
  team_br <- tbr %>%
    dplyr::transmute(abbrev, br_runs = safe_numeric(runner_runs_tot)) %>%
    dplyr::filter(!is.na(abbrev))
  
  team_fld <- tfld %>%
    dplyr::transmute(
      abbrev,
      fld_runs  = safe_numeric(total_runs),
      fld_plays = safe_numeric(if ("tot_plays" %in% names(tfld)) tot_plays else NA)
    ) %>%
    dplyr::filter(!is.na(abbrev))
  
  # Optional standings lookup
  st <- NULL
  if (!is.null(standings_folder)) {
    for (stem in c("std", "league_standings", "mlb_standings", "standings")) {
      st_raw <- read_github_csv(
        github_raw_url(standings_folder, sprintf("%s_%d.csv", stem, y))
      )
      if (!is.null(st_raw)) {
        st <- st_raw %>%
          dplyr::filter(!is.na(Rk), Tm != "Average") %>%
          dplyr::mutate(
            team_id        = resolve_team_id(Tm),
            W              = safe_numeric(W),
            L              = safe_numeric(L),
            `W-L%`         = safe_numeric(`W-L%`),
            R              = safe_numeric(R),
            RA             = safe_numeric(RA),
            Rdiff          = safe_numeric(Rdiff),
            rdiff_per_game = R - RA,
            year           = y
          )
        break
      }
    }
  }
  
  list(
    year          = y,
    team_mode     = TRUE,
    team_batting  = team_batting,
    team_pitching = team_pitching,
    team_br       = team_br,
    team_fld      = team_fld,
    standings     = st
  )
}

# ---- Scan YYYY Data/Snapshots/ and load every valid dated set of files ------
# Returns a named list keyed by "YYYY-MM-DD", sorted oldest → newest.
# File listing is done via the GitHub API.
scan_and_load_snapshots <- function(base, y) {
  snap_folder <- paste0(y, " Data/Snapshots")
  
  all_names <- github_list_files(snap_folder)
  if (length(all_names) == 0) return(list())
  
  pattern   <- sprintf("^bat_t_%d-\\d{2}-\\d{2}\\.csv$", y)
  bat_files <- sort(grep(pattern, all_names, value = TRUE))
  if (length(bat_files) == 0) return(list())
  
  year_data_folder <- paste0(y, " Data")
  result <- list()
  for (bf in bat_files) {
    d_str <- sub(sprintf("^bat_t_(%d-\\d{2}-\\d{2})\\.csv$", y), "\\1", bf)
    d     <- tryCatch(as.Date(d_str), error = function(e) NA_Date_)
    if (is.na(d)) next
    
    bat_p <- github_raw_url(snap_folder, sprintf("bat_t_%s.csv",  d_str))
    pit_p <- github_raw_url(snap_folder, sprintf("pit_t_%s.csv",  d_str))
    br_p  <- github_raw_url(snap_folder, sprintf("br_t_%s.csv",   d_str))
    fld_p <- github_raw_url(snap_folder, sprintf("fld_t_%s.csv",  d_str))
    
    snap <- load_one_snapshot(bat_p, pit_p, br_p, fld_p, y,
                              standings_folder = year_data_folder)
    if (is.null(snap)) next
    snap$date   <- d
    result[[d_str]] <- snap
  }
  result  # already sorted because bat_files was sorted alphabetically (ISO dates sort correctly)
}

# ---- Backward-compat: load old fixed-named team files as a single snapshot --
# Wraps the old expected_stats_team_batting_YYYY.csv style into the snapshot list.
load_year_team_compat <- function(base, y) {
  f <- paste0(y, " Data")
  
  tb_p  <- github_raw_url(f, sprintf("expected_stats_team_batting_%d.csv",  y))
  tp_p  <- github_raw_url(f, sprintf("expected_stats_team_pitching_%d.csv", y))
  br_p  <- github_raw_url(f, sprintf("baserunning_run_value_team_%d.csv",   y))
  fld_p <- github_raw_url(f, sprintf("fielding-run-value_team_%d.csv",      y))
  
  snap <- load_one_snapshot(tb_p, tp_p, br_p, fld_p, y, standings_folder = f)
  if (is.null(snap)) return(list())
  
  # Fall back to Apr 1 of the season year (no file mod-time available via URL)
  d_str <- sprintf("%d-04-01", y)
  snap$date <- as.Date(d_str)
  stats::setNames(list(snap), d_str)
}

# ---- Delta computation: back-calculate rate stats to totals, subtract, re-rate
delta_rate_df <- function(new_df, old_df) {
  dplyr::inner_join(
    new_df %>% dplyr::select(abbrev, pa_n = pa, bip_n = bip, ewoba_n = est_woba),
    old_df %>% dplyr::select(abbrev, pa_o = pa, bip_o = bip, ewoba_o = est_woba),
    by = "abbrev"
  ) %>%
    dplyr::mutate(
      pa       = pa_n - pa_o,
      bip      = bip_n - bip_o,
      est_woba = dplyr::if_else(pa > 0,
                                (ewoba_n * pa_n - ewoba_o * pa_o) / pa,
                                ewoba_n),
      bip_rate = dplyr::if_else(pa > 0, bip / pa, bip_n / pmax(pa_n, 1))
    ) %>%
    dplyr::filter(pa > 0) %>%
    dplyr::select(abbrev, pa, bip, est_woba, bip_rate)
}

# ---- Pick the right snapshot window and return a year_data list -------------
# window_days: Inf = full season, 30 / 7 / 1 = rolling windows.
# Gracefully falls back to oldest available baseline if history is too short.
compute_window_data <- function(snapshots, window_days) {
  if (length(snapshots) == 0) return(NULL)
  
  d_strs   <- names(snapshots)           # sorted oldest → newest
  latest_d <- d_strs[length(d_strs)]
  latest   <- snapshots[[latest_d]]
  
  # Full-season or only one snapshot → return latest as-is
  if (is.infinite(window_days) || length(d_strs) == 1) {
    out <- latest
    out$window_label    <- format(as.Date(latest_d), "%b %d, %Y")
    out$window_fallback <- FALSE
    out$fallback_msg    <- NULL
    return(out)
  }
  
  latest_date  <- as.Date(latest_d)
  cutoff_date  <- latest_date - window_days
  older_d_strs <- d_strs[as.Date(d_strs) <= cutoff_date]
  fallback_msg <- NULL
  
  if (length(older_d_strs) == 0) {
    # Not enough history — use oldest available as baseline
    older_d      <- d_strs[1]
    fallback_msg <- sprintf(
      "No snapshot from %d+ days ago — showing delta since %s (earliest on file).",
      window_days, format(as.Date(older_d), "%b %d")
    )
  } else {
    older_d <- older_d_strs[length(older_d_strs)]   # newest snapshot before cutoff
  }
  
  older <- snapshots[[older_d]]
  
  bat_d <- delta_rate_df(latest$team_batting,  older$team_batting)
  pit_d <- delta_rate_df(latest$team_pitching, older$team_pitching)
  
  # ---- Baserunning & Fielding: anchor to full-season rate ----------------------
  # BR and fielding run values are discrete and update infrequently — many teams
  # will show zero delta in any short window even after playing games.  Using the
  # raw period delta as the numerator would make those teams look average
  # regardless of their true quality, and a handful of teams gaining/losing a run
  # value would distort the whole league average.
  #
  # Strategy: all teams use their full-season (cumulative) br_runs and fld_runs
  # from the latest snapshot as the numerator, with full-season PA / fld_plays as
  # the denominator.  This locks in each team's season-long rate.  Teams whose
  # run value actually changed during the window are already reflected in the
  # latest snapshot total, so they naturally shift relative to teams that didn't
  # change — without distorting the others.
  full_season_pa <- latest$team_batting %>% dplyr::select(abbrev, pa_full = pa)
  
  br_d <- latest$team_br %>%
    dplyr::left_join(full_season_pa, by = "abbrev") %>%
    dplyr::select(abbrev, br_runs, pa_full)
  
  fld_d <- latest$team_fld %>%
    dplyr::transmute(
      abbrev,
      fld_runs  = dplyr::coalesce(fld_runs, 0),
      fld_plays = dplyr::coalesce(fld_plays, 0)
    )
  
  list(
    year           = latest$year,
    team_mode      = TRUE,
    team_batting   = bat_d,
    team_pitching  = pit_d,
    team_br        = br_d,
    team_fld       = fld_d,
    standings      = NULL,
    date           = latest$date,
    window_label   = sprintf("%s – %s",
                             format(as.Date(older_d),  "%b %d"),
                             format(latest_date,        "%b %d, %Y")),
    window_fallback = !is.null(fallback_msg),
    fallback_msg   = fallback_msg
  )
}

# ---- modelling ---------------------------------------------------------------
BAT_FORMULA <- runs_per_pa ~ est_woba + bip_rate
PIT_FORMULA <- runs_per_pa ~ est_woba + xera

# Recency multiplier: each season decays by this factor per year away from the
# most recent season. 0.5 = each prior year gets half the weight of the next.
RECENCY_DECAY <- 0.5

# =============================================================================
# BAYESIAN RECENCY BLENDING — season-view only
# =============================================================================
# When showing the "Season" tab, team ratings are a weighted blend of the
# full cumulative season rating and a recent 14-day window rating:
#
#   blended = (1 - α) × full_season  +  α × recent_14d
#
# where α grows with the recent-window sample size:
#
#   α  =  RECENCY_BLEND_ALPHA_MAX  ×  (PA_recent / (PA_recent + RECENCY_BLEND_K))
#
# This means:
#   • Early season (small PA_recent) → α ≈ 0, almost pure full-season rating.
#   • As a team sustains hot/cold performance over weeks, α rises toward α_max.
#   • α_max = 0.40 caps recency influence.  Because it is modulated by the PA
#     ratio it only approaches 0.40 asymptotically; in realistic mid-season
#     scenarios (14-day window ~350–700 PAs) effective weight is ~15–25%.
#   • k = 600 PAs ≈ 16 team games.  Meaningful recency signal starts ~3 weeks in.
#
# The 30d / 7d / 1d window chips bypass this logic entirely and use raw deltas.
# =============================================================================
RECENCY_BLEND_ALPHA_MAX <- 0.40
RECENCY_BLEND_K         <- 600   # PAs for half-max alpha

# Negative Binomial dispersion parameter for the matchup simulator.
# Replaces the Poisson (which forces variance = mean) with a distribution that
# allows variance = mu + mu^2/DISP_SIZE, producing heavier tails consistent
# with real MLB run distributions. Calibrated to historical game logs; r ~ 8
# fits observed shutout/blowout frequencies well. As DISP_SIZE -> Inf the NB
# collapses back to Poisson.
DISP_SIZE <- 8

fit_pooled_models <- function(years_list) {
  bat_pool <- bind_rows(lapply(years_list, `[[`, "batters"))
  pit_pool <- bind_rows(lapply(years_list, `[[`, "pitchers"))
  bat_pool <- bat_pool %>% filter(
    is.finite(runs_per_pa), is.finite(est_woba), is.finite(bip_rate), pa_rv >= 1
  )
  pit_pool <- pit_pool %>% filter(
    is.finite(runs_per_pa), is.finite(est_woba), is.finite(xera), pa_rv >= 1
  )
  # Apply recency weighting: most-recent year gets full weight (multiplier = 1),
  # each prior year gets RECENCY_DECAY^(years_back) applied on top of PA weight.
  max_year_bat <- max(bat_pool$season_year)
  max_year_pit <- max(pit_pool$season_year)
  bat_pool <- bat_pool %>%
    mutate(fit_weight = pa_rv * RECENCY_DECAY^(max_year_bat - season_year))
  pit_pool <- pit_pool %>%
    mutate(fit_weight = pa_rv * RECENCY_DECAY^(max_year_pit - season_year))
  bat_model <- lm(BAT_FORMULA, data = bat_pool, weights = fit_weight)
  pit_model <- lm(PIT_FORMULA, data = pit_pool, weights = fit_weight)
  list(
    bat_model      = bat_model,
    pit_model      = pit_model,
    bat_pool_n     = nrow(bat_pool),
    pit_pool_n     = nrow(pit_pool),
    bat_pool_years = sort(unique(bat_pool$season_year)),
    pit_pool_years = sort(unique(pit_pool$season_year))
  )
}

enrich_year <- function(year_data, models) {
  batters  <- year_data$batters
  pitchers <- year_data$pitchers
  
  batters$pred_runs_per_pa  <- predict(models$bat_model, batters)
  pitchers$pred_runs_per_pa <- predict(models$pit_model, pitchers)
  
  bat_mean_pa <- weighted.mean(batters$pred_runs_per_pa,  batters$pa_rv,  na.rm = TRUE)
  pit_mean_pa <- weighted.mean(pitchers$pred_runs_per_pa, pitchers$pa_rv, na.rm = TRUE)
  
  batters$adj_pred_runs_per_pa  <- batters$pred_runs_per_pa  - bat_mean_pa
  pitchers$adj_pred_runs_per_pa <- pitchers$pred_runs_per_pa - pit_mean_pa
  
  batters$pred_runs_total       <- batters$pred_runs_per_pa  * batters$pa
  pitchers$pred_runs_total      <- pitchers$pred_runs_per_pa * pitchers$pa
  batters$adj_pred_runs_total   <- batters$adj_pred_runs_per_pa  * batters$pa
  pitchers$adj_pred_runs_total  <- pitchers$adj_pred_runs_per_pa * pitchers$pa
  
  list(year = year_data$year, batters = batters, pitchers = pitchers,
       standings = year_data$standings,
       bat_mean_pa = bat_mean_pa, pit_mean_pa = pit_mean_pa)
}

build_team_ratings <- function(enriched) {
  # ---- Offense = hitting + baserunning ---------------------------------------
  # Hitting per game: PA-weighted mean of per-PA rate × 38 (team PA per game).
  # Baserunning per game: (team BR total / team PA) × 38 — same PA scale, so BR
  # contributions are on the same "runs/game above average" axis as hitting.
  bat_team <- enriched$batters %>%
    filter(!is.na(team_id)) %>%
    group_by(team_id, abbrev, team_name) %>%
    summarise(
      n_batters    = n(),
      total_bat_pa = sum(pa_rv, na.rm = TRUE),
      off_hit_rate = weighted.mean(adj_pred_runs_per_pa, pa_rv, na.rm = TRUE),
      team_br_runs = sum(br_runs, na.rm = TRUE),
      off_br_rate  = ifelse(sum(pa_rv, na.rm = TRUE) > 0,
                            sum(br_runs, na.rm = TRUE) / sum(pa_rv, na.rm = TRUE),
                            0),
      .groups = "drop"
    ) %>%
    mutate(
      off_hitting = off_hit_rate * PA_PER_GAME,
      off_br      = off_br_rate  * PA_PER_GAME * BASERUNNING_RELIABILITY,
      off_rating  = off_hitting + off_br
    )
  
  # ---- Defense = pitching + fielding -----------------------------------------
  # Pitching per game: PA-weighted mean of pitcher per-PA rate × 38.
  # Fielding per game: team fielding total scaled by FIELDER_OUTS_PER_GAME (243)
  # over the sum of fielder-outs played — this is the correct per-team-game
  # conversion for fielding, which is denominated in defensive-inning outs.
  pit_team <- enriched$pitchers %>%
    filter(!is.na(team_id)) %>%
    group_by(team_id, abbrev, team_name) %>%
    summarise(
      n_pitchers   = n(),
      total_pit_pa = sum(pa_rv, na.rm = TRUE),
      def_pit_rate = weighted.mean(adj_pred_runs_per_pa, pa_rv, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(def_pitching = def_pit_rate * PA_PER_GAME)
  
  # Team fielding aggregates from BOTH batters and pitchers (each player contributes
  # once — dual-role players were assigned to exactly one role during load).
  bat_fld_team <- enriched$batters %>%
    filter(!is.na(team_id)) %>%
    group_by(team_id) %>%
    summarise(bat_fld_runs = sum(fld_runs, na.rm = TRUE),
              bat_fld_outs = sum(fld_outs, na.rm = TRUE),
              .groups = "drop")
  pit_fld_team <- enriched$pitchers %>%
    filter(!is.na(team_id)) %>%
    group_by(team_id) %>%
    summarise(pit_fld_runs = sum(fld_runs, na.rm = TRUE),
              pit_fld_outs = sum(fld_outs, na.rm = TRUE),
              .groups = "drop")
  
  fld_team <- dplyr::full_join(bat_fld_team, pit_fld_team, by = "team_id") %>%
    mutate(
      bat_fld_runs = dplyr::coalesce(bat_fld_runs, 0),
      bat_fld_outs = dplyr::coalesce(bat_fld_outs, 0),
      pit_fld_runs = dplyr::coalesce(pit_fld_runs, 0),
      pit_fld_outs = dplyr::coalesce(pit_fld_outs, 0),
      team_fld_runs = bat_fld_runs + pit_fld_runs,
      team_fld_outs = bat_fld_outs + pit_fld_outs,
      def_fld = ifelse(team_fld_outs > 0,
                       (team_fld_runs / team_fld_outs) * FIELDER_OUTS_PER_GAME,
                       0) * FIELDING_RELIABILITY
    ) %>%
    select(team_id, team_fld_runs, team_fld_outs, def_fld)
  
  teams <- TEAM_META %>%
    left_join(bat_team %>% select(team_id, n_batters, total_bat_pa,
                                  off_hitting, off_br, off_rating),
              by = "team_id") %>%
    left_join(pit_team %>% select(team_id, n_pitchers, total_pit_pa, def_pitching),
              by = "team_id") %>%
    left_join(fld_team, by = "team_id") %>%
    mutate(
      off_hitting  = ifelse(is.na(off_hitting),  0, off_hitting),
      off_br       = ifelse(is.na(off_br),       0, off_br),
      off_rating   = ifelse(is.na(off_rating),   0, off_rating),
      def_pitching = ifelse(is.na(def_pitching), 0, def_pitching),
      def_fld      = ifelse(is.na(def_fld),      0, def_fld),
      def_rating   = def_pitching + def_fld,
      overall      = off_rating + def_rating
    ) %>%
    arrange(desc(overall)) %>%
    mutate(rank = row_number()) %>%
    select(rank, team_id, abbrev, team_name, overall,
           off_rating, off_hitting, off_br,
           def_rating, def_pitching, def_fld,
           n_batters, n_pitchers, total_bat_pa, total_pit_pa,
           team_fld_runs, team_fld_outs,
           team_color, team_logo_espn)
  
  if (!is.null(enriched$standings)) {
    teams <- teams %>% left_join(
      enriched$standings %>% select(team_id, W, L, `W-L%`, R, RA, Rdiff, rdiff_per_game),
      by = "team_id"
    )
  }
  teams
}

# ---- build team ratings when the year uses team-level data (not player rows) -
# Applies the regression model coefficients directly to team-level expected stats,
# then layers on the actual team baserunning and fielding run values for that year.
build_team_ratings_team_mode <- function(year_data, models) {
  tb   <- year_data$team_batting
  tp   <- year_data$team_pitching
  tbr  <- year_data$team_br
  tfld <- year_data$team_fld
  
  # ---- Hitting: apply bat_model to team-level batting stats -------------------
  # We need est_woba and bip_rate (same predictors as player model)
  bat_needed <- c("est_woba", "bip_rate")
  tb_clean <- tb %>%
    dplyr::filter(dplyr::if_all(dplyr::all_of(bat_needed), is.finite))
  
  tb_clean$pred_runs_per_pa <- predict(models$bat_model, tb_clean)
  
  # Centre to league mean (PA-weighted)
  bat_mean <- stats::weighted.mean(tb_clean$pred_runs_per_pa, tb_clean$pa, na.rm = TRUE)
  tb_clean <- tb_clean %>%
    dplyr::mutate(
      adj_pred_runs_per_pa = pred_runs_per_pa - bat_mean,
      off_hitting          = adj_pred_runs_per_pa * PA_PER_GAME
    )
  
  # ---- Baserunning: actual team BR run values (already runs above avg) --------
  # Normalise to runs/PA then scale to per-game.
  # When tbr carries a pa_full column (set by compute_window_data for date-
  # filtered windows), we use that full-season PA as the denominator so that
  # the rate is computed over a meaningful sample size regardless of the window.
  # For full-season (non-windowed) data pa_full is absent and we fall back to
  # the snapshot PA as before.
  has_pa_full <- "pa_full" %in% names(tbr)
  br_aug <- tb_clean %>%
    dplyr::select(abbrev, pa) %>%
    dplyr::left_join(
      if (has_pa_full) tbr %>% dplyr::select(abbrev, br_runs, pa_full)
      else             tbr %>% dplyr::select(abbrev, br_runs),
      by = "abbrev"
    ) %>%
    dplyr::mutate(
      br_runs   = dplyr::coalesce(br_runs, 0),
      # Use full-season PA when available; fall back to window PA otherwise.
      pa_denom  = if (has_pa_full) dplyr::coalesce(pa_full, pa) else pa,
      off_br    = ifelse(pa_denom > 0, (br_runs / pa_denom) * PA_PER_GAME, 0) * BASERUNNING_RELIABILITY
    )
  
  bat_team <- tb_clean %>%
    dplyr::left_join(br_aug %>% dplyr::select(abbrev, br_runs, off_br), by = "abbrev") %>%
    dplyr::mutate(
      total_bat_pa = pa,
      off_rating   = off_hitting + off_br
    ) %>%
    dplyr::select(abbrev, total_bat_pa, off_hitting, off_br, off_rating, br_runs)
  
  # ---- Pitching: apply pit_model to team-level pitching stats ----------------
  # The pitcher model uses est_woba + xera.  Team pitching files don't have xERA,
  # so we substitute the bat_model's xwOBA-only prediction pathway:
  # We only have est_woba + bip_rate from the pitching file; derive a surrogate
  # xERA from the historical average xERA-to-xwOBA relationship in the model pool.
  # Simple approach: set xera = mean(xera) from training data (neutral) so the
  # prediction is driven entirely by xwOBA.  This is conservative and consistent.
  mean_xera <- mean(models$pit_model$model[["xera"]], na.rm = TRUE)
  
  pit_needed <- c("est_woba", "bip_rate")
  tp_clean <- tp %>%
    dplyr::filter(dplyr::if_all(dplyr::all_of(pit_needed), is.finite)) %>%
    dplyr::mutate(xera = mean_xera)  # fill with mean so model can predict
  
  tp_clean$pred_runs_per_pa <- predict(models$pit_model, tp_clean)
  
  pit_mean <- stats::weighted.mean(tp_clean$pred_runs_per_pa, tp_clean$pa, na.rm = TRUE)
  tp_clean <- tp_clean %>%
    dplyr::mutate(
      adj_pred_runs_per_pa = pred_runs_per_pa - pit_mean,
      def_pitching         = adj_pred_runs_per_pa * PA_PER_GAME
    )
  
  pit_team <- tp_clean %>%
    dplyr::mutate(total_pit_pa = pa) %>%
    dplyr::select(abbrev, total_pit_pa, def_pitching)
  
  # ---- Fielding: actual 2026 team fielding run values ------------------------
  # Normalise to runs per fielder-out × FIELDER_OUTS_PER_GAME (243).
  # If fld_plays (total defensive plays) is available use it; otherwise
  # scale by PA as a proxy (each PA ~ 1 fielder opportunity on average).
  fld_team <- tfld %>%
    dplyr::mutate(
      fld_runs  = dplyr::coalesce(fld_runs, 0),
      fld_plays = dplyr::coalesce(fld_plays, 0)
    )
  
  # Compute def_fld using fld_plays as the outs denominator if > 0,
  # otherwise fall back to the season-average FIELDER_OUTS_PER_GAME×162 total.
  SEASON_FIELDER_OUTS <- FIELDER_OUTS_PER_GAME * 162
  fld_team <- fld_team %>%
    dplyr::mutate(
      def_fld = dplyr::case_when(
        fld_plays > 0 ~ (fld_runs / fld_plays) * FIELDER_OUTS_PER_GAME,
        TRUE          ~ (fld_runs / SEASON_FIELDER_OUTS) * FIELDER_OUTS_PER_GAME
      ) * FIELDING_RELIABILITY
    ) %>%
    dplyr::select(abbrev, team_fld_runs = fld_runs, def_fld)
  
  # ---- Assemble final table (join on abbrev → team_id via TEAM_META) ----------
  teams <- TEAM_META %>%
    dplyr::left_join(bat_team, by = "abbrev") %>%
    dplyr::left_join(pit_team, by = "abbrev") %>%
    dplyr::left_join(fld_team, by = "abbrev") %>%
    dplyr::mutate(
      off_hitting  = ifelse(is.na(off_hitting),  0, off_hitting),
      off_br       = ifelse(is.na(off_br),       0, off_br),
      off_rating   = ifelse(is.na(off_rating),   0, off_rating),
      def_pitching = ifelse(is.na(def_pitching), 0, def_pitching),
      def_fld      = ifelse(is.na(def_fld),      0, def_fld),
      def_rating   = def_pitching + def_fld,
      overall      = off_rating + def_rating,
      # Placeholders to keep downstream code happy
      n_batters     = NA_integer_,
      n_pitchers    = NA_integer_,
      total_bat_pa  = dplyr::coalesce(total_bat_pa, 0),
      total_pit_pa  = dplyr::coalesce(total_pit_pa, 0),
      team_fld_runs = dplyr::coalesce(team_fld_runs, 0),
      team_fld_outs = NA_real_
    ) %>%
    dplyr::filter(total_bat_pa > 0 | total_pit_pa > 0) %>%
    dplyr::arrange(dplyr::desc(overall)) %>%
    dplyr::mutate(rank = dplyr::row_number()) %>%
    dplyr::select(rank, team_id, abbrev, team_name, overall,
                  off_rating, off_hitting, off_br,
                  def_rating, def_pitching, def_fld,
                  n_batters, n_pitchers, total_bat_pa, total_pit_pa,
                  team_fld_runs, team_fld_outs,
                  team_color, team_logo_espn)
  
  if (!is.null(year_data$standings)) {
    teams <- teams %>% dplyr::left_join(
      year_data$standings %>% dplyr::select(team_id, W, L, `W-L%`, R, RA, Rdiff, rdiff_per_game),
      by = "team_id"
    )
  }
  teams
}

# =============================================================================
# compute_blended_season_ratings()
# =============================================================================
# Produces a "Season" team ratings table that blends full-season performance
# with a 14-day rolling window using the Bayesian recency scheme above.
#
# Arguments:
#   snapshots  — named list of snapshots for one year (output of
#                scan_and_load_snapshots / load_year_team_compat).
#   models     — fitted bat/pit models from fit_pooled_models().
#   standings  — optional standings data frame to attach (can be NULL).
#
# Returns a team ratings table identical in structure to
# build_team_ratings_team_mode(), with blended off_rating / def_rating /
# overall columns and an extra `recency_alpha` column for diagnostics.
# =============================================================================
compute_blended_season_ratings <- function(snapshots, models, standings = NULL) {
  if (length(snapshots) == 0) return(NULL)
  
  # ---- Full-season ratings (latest snapshot, window = Inf) -------------------
  full_data <- compute_window_data(snapshots, Inf)
  if (is.null(full_data)) return(NULL)
  full_tbl <- tryCatch(
    build_team_ratings_team_mode(full_data, models),
    error = function(e) NULL
  )
  if (is.null(full_tbl)) return(NULL)
  
  # ---- 14-day window ratings -------------------------------------------------
  recent_data <- compute_window_data(snapshots, 14)
  if (is.null(recent_data)) {
    # Not enough history for a 14-day window — return pure season ratings.
    if (!is.null(standings)) {
      full_tbl <- full_tbl %>% dplyr::left_join(
        standings %>% dplyr::select(team_id, W, L, `W-L%`, R, RA, Rdiff, rdiff_per_game),
        by = "team_id"
      )
    }
    return(full_tbl %>% dplyr::mutate(recency_alpha = 0))
  }
  
  recent_tbl <- tryCatch(
    build_team_ratings_team_mode(recent_data, models),
    error = function(e) NULL
  )
  if (is.null(recent_tbl)) {
    if (!is.null(standings)) {
      full_tbl <- full_tbl %>% dplyr::left_join(
        standings %>% dplyr::select(team_id, W, L, `W-L%`, R, RA, Rdiff, rdiff_per_game),
        by = "team_id"
      )
    }
    return(full_tbl %>% dplyr::mutate(recency_alpha = 0))
  }
  
  # ---- Compute per-team alpha using recent-window PA as the sample signal ----
  # recent_data$team_batting has the PA for the 14-day delta period.
  recent_pa_lookup <- recent_data$team_batting %>%
    dplyr::select(abbrev, pa_recent = pa)
  
  # ---- Blend at the component level so each sub-rating is correct ------------
  blended <- full_tbl %>%
    dplyr::left_join(recent_pa_lookup, by = "abbrev") %>%
    dplyr::mutate(
      pa_recent     = dplyr::coalesce(pa_recent, 0),
      recency_alpha = RECENCY_BLEND_ALPHA_MAX *
        (pa_recent / (pa_recent + RECENCY_BLEND_K))
    )
  
  # Pull recent component values into blended for the join.
  recent_components <- recent_tbl %>%
    dplyr::select(abbrev,
                  r_off_hitting  = off_hitting,
                  r_off_br       = off_br,
                  r_off_rating   = off_rating,
                  r_def_pitching = def_pitching,
                  r_def_fld      = def_fld,
                  r_def_rating   = def_rating,
                  r_overall      = overall)
  
  blended <- blended %>%
    dplyr::left_join(recent_components, by = "abbrev") %>%
    dplyr::mutate(
      # For any team missing from the recent window, fall back to full-season.
      r_off_hitting  = dplyr::coalesce(r_off_hitting,  off_hitting),
      r_off_br       = dplyr::coalesce(r_off_br,       off_br),
      r_off_rating   = dplyr::coalesce(r_off_rating,   off_rating),
      r_def_pitching = dplyr::coalesce(r_def_pitching, def_pitching),
      r_def_fld      = dplyr::coalesce(r_def_fld,      def_fld),
      r_def_rating   = dplyr::coalesce(r_def_rating,   def_rating),
      r_overall      = dplyr::coalesce(r_overall,      overall),
      # Blend each component
      off_hitting  = (1 - recency_alpha) * off_hitting  + recency_alpha * r_off_hitting,
      off_br       = (1 - recency_alpha) * off_br       + recency_alpha * r_off_br,
      off_rating   = (1 - recency_alpha) * off_rating   + recency_alpha * r_off_rating,
      def_pitching = (1 - recency_alpha) * def_pitching + recency_alpha * r_def_pitching,
      def_fld      = (1 - recency_alpha) * def_fld      + recency_alpha * r_def_fld,
      def_rating   = (1 - recency_alpha) * def_rating   + recency_alpha * r_def_rating,
      overall      = (1 - recency_alpha) * overall      + recency_alpha * r_overall
    ) %>%
    dplyr::select(-dplyr::starts_with("r_"), -pa_recent) %>%
    dplyr::arrange(dplyr::desc(overall)) %>%
    dplyr::mutate(rank = dplyr::row_number())
  
  if (!is.null(standings)) {
    blended <- blended %>% dplyr::left_join(
      standings %>% dplyr::select(team_id, W, L, `W-L%`, R, RA, Rdiff, rdiff_per_game),
      by = "team_id"
    )
  }
  blended
}

build_player_view <- function(enriched, use_single_season = FALSE) {
  # One row per player-role. All metrics expressed in "runs above avg per game"
  # so columns are directly comparable and sum cleanly into Overall.
  #
  # use_single_season = TRUE: use the preserved single_season_* columns so the
  # player leaderboard reflects only that year's performance (not multi-year
  # weighted rates). Falls back gracefully if those columns don't exist.
  batters  <- enriched$batters
  pitchers <- enriched$pitchers
  
  if (use_single_season && "single_season_adj_pred" %in% names(batters)) {
    batters <- batters %>%
      dplyr::mutate(
        adj_pred_runs_per_pa = single_season_adj_pred,
        br_per_pa            = single_season_br_per_pa,
        fld_per_out          = single_season_fld_per_out
      )
  }
  if (use_single_season && "single_season_adj_pred" %in% names(pitchers)) {
    pitchers <- pitchers %>%
      dplyr::mutate(
        adj_pred_runs_per_pa = single_season_adj_pred,
        fld_per_out          = single_season_fld_per_out
      )
  }
  
  bat_rows <- batters %>%
    transmute(
      player_id,
      Player      = player,
      Team        = abbrev,
      Role        = "Hitter",
      PA          = pa,
      xwOBA       = round(est_woba, 3),
      Hitting     = adj_pred_runs_per_pa * PA_PER_GAME,
      Baserunning = br_per_pa * PA_PER_GAME * BASERUNNING_RELIABILITY,
      Pitching    = NA_real_,
      Fielding    = fld_per_out * OUTS_PER_GAME * FIELDING_RELIABILITY
    )

  pit_rows <- pitchers %>%
    transmute(
      player_id,
      Player      = player,
      Team        = abbrev,
      Role        = "Pitcher",
      PA          = pa,
      xwOBA       = round(est_woba, 3),
      Hitting     = NA_real_,
      Baserunning = NA_real_,
      Pitching    = adj_pred_runs_per_pa * PA_PER_GAME,
      Fielding    = fld_per_out * OUTS_PER_GAME * FIELDING_RELIABILITY
    )
  
  all_rows <- dplyr::bind_rows(bat_rows, pit_rows)

  # ---- Two-way player merging (e.g. Ohtani) ------------------------------------
  # Any player_id that appears in BOTH bat_rows and pit_rows gets a combined row
  # with Role = "Player" that carries all four stat columns populated. The original
  # single-role rows are kept so that single-role filtering still works correctly.
  dual_ids <- intersect(bat_rows$player_id, pit_rows$player_id)

  if (length(dual_ids) > 0) {
    dual_bat <- bat_rows %>% dplyr::filter(player_id %in% dual_ids)
    dual_pit <- pit_rows %>% dplyr::filter(player_id %in% dual_ids)

    combined <- dplyr::inner_join(
      dual_bat %>% dplyr::select(player_id, Player, Team,
                                  PA_hit = PA, xwOBA_hit = xwOBA,
                                  Hitting, Baserunning,
                                  Fielding_hit = Fielding),
      dual_pit %>% dplyr::select(player_id,
                                  PA_pit = PA, xwOBA_pit = xwOBA,
                                  Pitching,
                                  Fielding_pit = Fielding),
      by = "player_id"
    ) %>%
      dplyr::mutate(
        Role    = "Player",
        PA      = PA_hit + PA_pit,
        xwOBA   = round((xwOBA_hit * PA_hit + xwOBA_pit * PA_pit) /
                          pmax(PA_hit + PA_pit, 1), 3),
        # Sum fielding across both roles; keep NA only if both were NA
        Fielding = ifelse(
          is.na(Fielding_hit) & is.na(Fielding_pit),
          NA_real_,
          dplyr::coalesce(Fielding_hit, 0) + dplyr::coalesce(Fielding_pit, 0)
        )
      ) %>%
      dplyr::select(player_id, Player, Team, Role, PA, xwOBA,
                    Hitting, Baserunning, Pitching, Fielding)

    # Bind: single-role rows stay intact; combined row is appended
    all_rows <- dplyr::bind_rows(all_rows, combined)
  }

  all_rows %>%
    dplyr::mutate(
      Overall = rowSums(
        cbind(Hitting, Baserunning, Pitching, Fielding),
        na.rm = TRUE
      ),
      Hitting     = round(Hitting, 2),
      Baserunning = round(Baserunning, 2),
      Pitching    = round(Pitching, 2),
      Fielding    = round(Fielding, 2),
      Overall     = round(Overall, 2)
    ) %>%
    dplyr::arrange(dplyr::desc(Overall)) %>%
    dplyr::mutate(Rank = dplyr::row_number()) %>%
    dplyr::select(Rank, player_id, Player, Team, Role, PA, xwOBA,
                  Hitting, Baserunning, Pitching, Fielding, Overall)
}

# ---- load and build all years -----------------------------------------------
# Strategy:
#   • A year with team-level files in "YYYY Data/" uses team-mode for TEAM RATINGS
#     (build_team_ratings_team_mode) regardless of whether player files also exist.
#   • Player files (in "YYYY Data/" or "YYYY Player Data/") are loaded separately
#     when available and used for: (a) model training if the year is complete, and
#     (b) the Player Rankings tab for any year.
#   • The regression model is trained ONLY on prior/completed seasons whose year
#     folder does NOT contain team-level aggregate files (i.e. 2022–2025).

# ---- Load all daily snapshots for each year ---------------------------------
# Years with a Snapshots/ subfolder → list of dated entries (new workflow).
# Years with old-style fixed team files → single-entry compat list.
# Years with neither → empty list (player-mode only).
all_snapshots_by_year <- setNames(
  lapply(SUPPORTED_YEARS, function(y) {
    snaps <- scan_and_load_snapshots(GITHUB_RAW_BASE, y)
    if (length(snaps) == 0) snaps <- load_year_team_compat(GITHUB_RAW_BASE, y)
    snaps
  }),
  as.character(SUPPORTED_YEARS)
)
all_snapshots_by_year <- Filter(function(x) length(x) > 0, all_snapshots_by_year)

# team_raw_years = latest snapshot per year (used as the "Season" baseline
# throughout the pre-computation below — same interface as before).
team_raw_years <- lapply(names(all_snapshots_by_year), function(yc) {
  snaps   <- all_snapshots_by_year[[yc]]
  latest  <- names(snaps)[length(names(snaps))]
  snaps[[latest]]
})
names(team_raw_years) <- names(all_snapshots_by_year)

# Load player-level data for each year (returns NULL if player files absent).
player_raw_years <- lapply(SUPPORTED_YEARS, function(y) load_year(GITHUB_RAW_BASE, y))
names(player_raw_years) <- as.character(SUPPORTED_YEARS)
player_raw_years <- Filter(Negate(is.null), player_raw_years)

if (length(team_raw_years) == 0 && length(player_raw_years) == 0)
  stop("No year data found at '", GITHUB_RAW_BASE, "'.")

# Years available = union of years with any data.
AVAILABLE_YEARS <- sort(unique(as.integer(c(names(team_raw_years),
                                            names(player_raw_years)))))
DEFAULT_YEAR    <- max(AVAILABLE_YEARS)

# "Model training" years = player-mode years that do NOT have team-level files
# (i.e. completed prior seasons). These exclusively drive the regression fit.
training_yr_names <- setdiff(names(player_raw_years), names(team_raw_years))
training_years    <- player_raw_years[training_yr_names]

if (length(training_years) == 0)
  stop("No prior-season player data found — cannot train model.")

# Train model on completed prior seasons only.
models <- fit_pooled_models(training_years)

# Enrich ALL player-mode years (including current year if player files exist).
# This powers both the Player Rankings tab and — for prior years — team ratings.
player_yr_names <- names(player_raw_years)
enriched_years  <- lapply(player_raw_years, enrich_year, models = models)
names(enriched_years) <- player_yr_names

# Build team tables:
#   - Years WITH team-level files → use build_team_ratings_team_mode (team stats)
#   - Years WITHOUT team-level files → use build_team_ratings (player aggregation)
team_tbl_by_year     <- list()
players_view_by_year <- list()

for (yc in as.character(AVAILABLE_YEARS)) {
  if (yc %in% names(team_raw_years)) {
    # Team-mode: use team-level expected stats for ratings.
    team_tbl_by_year[[yc]] <- build_team_ratings_team_mode(team_raw_years[[yc]], models)
  } else if (yc %in% player_yr_names) {
    # Player-mode (prior completed seasons): aggregate from player rows.
    team_tbl_by_year[[yc]] <- build_team_ratings(enriched_years[[yc]])
  }
  # Player view: available for any year that has player files.
  if (yc %in% player_yr_names) {
    players_view_by_year[[yc]] <- build_player_view(enriched_years[[yc]])
  } else {
    players_view_by_year[[yc]] <- NULL
  }
}

# Re-order by year.
yr_order <- as.character(sort(as.integer(names(team_tbl_by_year))))
team_tbl_by_year     <- team_tbl_by_year[yr_order]
players_view_by_year <- players_view_by_year[yr_order]

standings_check <- list()
make_standings_row <- function(yc, tt) {
  if (!"rdiff_per_game" %in% names(tt)) return(NULL)
  if (all(is.na(tt$rdiff_per_game)))    return(NULL)
  tt %>%
    dplyr::filter(!is.na(rdiff_per_game)) %>%
    dplyr::transmute(
      Rank             = rank,
      Team             = abbrev,
      `Team Name`      = team_name,
      `Overall Rating` = round(overall, 2),
      `Actual RDiff/G` = round(rdiff_per_game, 2),
      `Residual`       = round(overall - rdiff_per_game, 2),
      `Actual W`       = W,
      `Actual L`       = L,
      `Actual R/G`     = R,
      `Actual RA/G`    = RA,
      team_id          = team_id,
      team_logo_espn   = team_logo_espn,
      overall          = overall,
      rdiff_per_game   = rdiff_per_game
    )
}
for (yc in names(team_tbl_by_year)) {
  row <- make_standings_row(yc, team_tbl_by_year[[yc]])
  if (!is.null(row)) standings_check[[yc]] <- row
}

# =============================================================================
# Multi-year recency-weighted player ratings — player-mode years only
# =============================================================================
# For the most recent player-mode year we replace each player's
# adj_pred_runs_per_pa with a recency-weighted average across all player seasons:
#   combined_weight = PA * year_rank   (oldest year → rank 1, current → rank N)
# Players with only one season of data keep their current-year value unchanged.
{
  # yr_rank_map covers only player-mode years (the ones with enriched_years data).
  yr_rank_map <- setNames(
    seq_along(sort(as.integer(player_yr_names))),
    player_yr_names
  )
  
  most_recent_player_yr <- as.character(max(as.integer(player_yr_names)))
  
  # ---- batters: hitting rate -------------------------------------------------
  all_bat_rows <- dplyr::bind_rows(lapply(player_yr_names, function(yc) {
    enriched_years[[yc]]$batters %>%
      dplyr::filter(is.finite(adj_pred_runs_per_pa), pa_rv >= 1) %>%
      dplyr::select(player_id, pa_rv, adj_pred_runs_per_pa) %>%
      dplyr::mutate(yr_rank = yr_rank_map[yc], comb_w = pa_rv * yr_rank)
  }))
  
  bat_weighted_ratings <- all_bat_rows %>%
    dplyr::group_by(player_id) %>%
    dplyr::summarise(
      weighted_rating = weighted.mean(adj_pred_runs_per_pa, comb_w, na.rm = TRUE),
      .groups = "drop"
    )
  
  # ---- batters: baserunning rate (runs per PA) -------------------------------
  all_br_rows <- dplyr::bind_rows(lapply(player_yr_names, function(yc) {
    enriched_years[[yc]]$batters %>%
      dplyr::filter(is.finite(br_per_pa), pa_rv >= 1) %>%
      dplyr::select(player_id, pa_rv, br_per_pa) %>%
      dplyr::mutate(yr_rank = yr_rank_map[yc], comb_w = pa_rv * yr_rank)
  }))
  
  br_weighted_ratings <- all_br_rows %>%
    dplyr::group_by(player_id) %>%
    dplyr::summarise(
      weighted_br_per_pa = weighted.mean(br_per_pa, comb_w, na.rm = TRUE),
      .groups = "drop"
    )
  
  # ---- pitchers: pitching rate -----------------------------------------------
  all_pit_rows <- dplyr::bind_rows(lapply(player_yr_names, function(yc) {
    enriched_years[[yc]]$pitchers %>%
      dplyr::filter(is.finite(adj_pred_runs_per_pa), pa_rv >= 1) %>%
      dplyr::select(player_id, pa_rv, adj_pred_runs_per_pa) %>%
      dplyr::mutate(yr_rank = yr_rank_map[yc], comb_w = pa_rv * yr_rank)
  }))
  
  pit_weighted_ratings <- all_pit_rows %>%
    dplyr::group_by(player_id) %>%
    dplyr::summarise(
      weighted_rating = weighted.mean(adj_pred_runs_per_pa, comb_w, na.rm = TRUE),
      .groups = "drop"
    )
  
  # ---- batters + pitchers: fielding rate (runs per defensive out) ------------
  all_fld_rows <- dplyr::bind_rows(lapply(player_yr_names, function(yc) {
    dplyr::bind_rows(
      enriched_years[[yc]]$batters  %>% dplyr::select(player_id, fld_outs, fld_per_out),
      enriched_years[[yc]]$pitchers %>% dplyr::select(player_id, fld_outs, fld_per_out)
    ) %>%
      dplyr::filter(is.finite(fld_per_out), fld_outs >= 1) %>%
      dplyr::mutate(yr_rank = yr_rank_map[yc], comb_w = fld_outs * yr_rank)
  }))
  
  fld_weighted_ratings <- all_fld_rows %>%
    dplyr::group_by(player_id) %>%
    dplyr::summarise(
      weighted_fld_per_out = weighted.mean(fld_per_out, comb_w, na.rm = TRUE),
      .groups = "drop"
    )
  
  # ---- patch most-recent player-mode year with multi-year weighted rates -----
  # Preserve single-season values; team ratings will use weighted, player tab
  # uses single-season values.
  enriched_years[[most_recent_player_yr]]$batters <-
    enriched_years[[most_recent_player_yr]]$batters %>%
    dplyr::mutate(
      single_season_adj_pred    = adj_pred_runs_per_pa,
      single_season_br_per_pa   = br_per_pa,
      single_season_fld_per_out = fld_per_out
    ) %>%
    dplyr::left_join(bat_weighted_ratings, by = "player_id") %>%
    dplyr::left_join(br_weighted_ratings,  by = "player_id") %>%
    dplyr::left_join(fld_weighted_ratings, by = "player_id") %>%
    dplyr::mutate(
      adj_pred_runs_per_pa = dplyr::coalesce(weighted_rating,      adj_pred_runs_per_pa),
      br_per_pa            = dplyr::coalesce(weighted_br_per_pa,   br_per_pa),
      fld_per_out          = dplyr::coalesce(weighted_fld_per_out, fld_per_out),
      br_runs              = br_per_pa   * pa_rv,
      fld_runs             = fld_per_out * fld_outs
    ) %>%
    dplyr::select(-weighted_rating, -weighted_br_per_pa, -weighted_fld_per_out)
  
  enriched_years[[most_recent_player_yr]]$pitchers <-
    enriched_years[[most_recent_player_yr]]$pitchers %>%
    dplyr::mutate(
      single_season_adj_pred    = adj_pred_runs_per_pa,
      single_season_fld_per_out = fld_per_out
    ) %>%
    dplyr::left_join(pit_weighted_ratings, by = "player_id") %>%
    dplyr::left_join(fld_weighted_ratings, by = "player_id") %>%
    dplyr::mutate(
      adj_pred_runs_per_pa = dplyr::coalesce(weighted_rating,      adj_pred_runs_per_pa),
      fld_per_out          = dplyr::coalesce(weighted_fld_per_out, fld_per_out),
      fld_runs             = fld_per_out * fld_outs
    ) %>%
    dplyr::select(-weighted_rating, -weighted_fld_per_out)
  
  # Rebuild tables for the most recent player-mode year.
  # Team ratings: only rebuild from player rows if there is NO team-level file for
  # this year (team-level files take precedence for team ratings).
  if (!most_recent_player_yr %in% names(team_raw_years)) {
    team_tbl_by_year[[most_recent_player_yr]] <-
      build_team_ratings(enriched_years[[most_recent_player_yr]])
    row <- make_standings_row(most_recent_player_yr,
                              team_tbl_by_year[[most_recent_player_yr]])
    if (!is.null(row)) standings_check[[most_recent_player_yr]] <- row
  }
  
  # Player view always uses single-season values from the enriched data.
  players_view_by_year[[most_recent_player_yr]] <- build_player_view(
    enriched_years[[most_recent_player_yr]],
    use_single_season = TRUE
  )
}

# =============================================================================
# Player Profile — pre-computation
# =============================================================================
# Build a long-form history table: one row per player-year, with their Overall
# rating for that year. Used for the year-over-year trend chart.
# Also pre-compute all_players_by_role for percentile calculations.
{
  pp_history <- dplyr::bind_rows(lapply(names(players_view_by_year), function(yc) {
    pv <- players_view_by_year[[yc]]
    if (is.null(pv)) return(NULL)
    pv %>%
      dplyr::select(player_id, Player, Team, Role, PA, Overall,
                    Hitting, Baserunning, Pitching, Fielding) %>%
      dplyr::mutate(year = as.integer(yc))
  }))

  # For percentile lookups: all hitters and pitchers pooled across the most
  # recent player-mode year only (current-season leaderboard context).
  pp_current_yr <- as.character(most_recent_player_yr)
  pp_current_pv <- players_view_by_year[[pp_current_yr]]

  if (!is.null(pp_current_pv)) {
    pp_hitter_overalls  <- pp_current_pv %>%
      dplyr::filter(Role %in% c("Hitter", "Player")) %>%
      dplyr::pull(Overall)
    pp_pitcher_overalls <- pp_current_pv %>%
      dplyr::filter(Role %in% c("Pitcher", "Player")) %>%
      dplyr::pull(Overall)
    pp_hitter_hitting   <- pp_current_pv %>%
      dplyr::filter(!is.na(Hitting)) %>% dplyr::pull(Hitting)
    pp_pitcher_pitching <- pp_current_pv %>%
      dplyr::filter(!is.na(Pitching)) %>% dplyr::pull(Pitching)
    pp_hitter_br        <- pp_current_pv %>%
      dplyr::filter(!is.na(Baserunning)) %>% dplyr::pull(Baserunning)
    pp_fielding_all     <- pp_current_pv %>%
      dplyr::filter(!is.na(Fielding)) %>% dplyr::pull(Fielding)
  } else {
    pp_hitter_overalls  <- numeric(0)
    pp_pitcher_overalls <- numeric(0)
    pp_hitter_hitting   <- numeric(0)
    pp_pitcher_pitching <- numeric(0)
    pp_hitter_br        <- numeric(0)
    pp_fielding_all     <- numeric(0)
  }

  # Helper: percentile rank of x within pool (0–100, higher = better)
  pct_rank <- function(x, pool) {
    if (is.na(x) || length(pool) == 0) return(NA_real_)
    round(mean(pool <= x) * 100)
  }
}

# =============================================================================
# Matchup Simulator — pre-computation
# =============================================================================
# Recency-weighted pitcher ratings from ALL player-mode years only.
# (Team-mode years have no individual pitcher data to draw from.)
{
  yr_rank_map_mp <- setNames(
    seq_along(sort(as.integer(player_yr_names))),
    player_yr_names
  )
  
  mp_all_pitchers <- dplyr::bind_rows(lapply(player_yr_names, function(yc) {
    enriched_years[[yc]]$pitchers %>%
      dplyr::filter(is.finite(adj_pred_runs_per_pa), pa_rv >= 1) %>%
      dplyr::select(player_id, player, abbrev, pa_rv, adj_pred_runs_per_pa) %>%
      dplyr::mutate(yr_rank = yr_rank_map_mp[yc],
                    comb_w  = pa_rv * yr_rank)
  }))
  
  mp_pitcher_ratings <- mp_all_pitchers %>%
    dplyr::group_by(player_id, player) %>%
    dplyr::summarise(
      weighted_rating = weighted.mean(adj_pred_runs_per_pa, comb_w, na.rm = TRUE),
      .groups = "drop"
    )
}

# For the matchup simulator pitcher dropdowns, use the most recent player-mode
# year's pitchers (which have multi-year weighted ratings).
mp_current_pitchers <- enriched_years[[most_recent_player_yr]]$pitchers %>%
  dplyr::select(player_id, player, team_id, abbrev, pa_rv, adj_pred_runs_per_pa) %>%
  dplyr::left_join(
    mp_pitcher_ratings %>% dplyr::select(player_id, weighted_rating),
    by = "player_id"
  ) %>%
  dplyr::mutate(
    final_rating = dplyr::coalesce(weighted_rating, adj_pred_runs_per_pa)
  ) %>%
  dplyr::arrange(abbrev, dplyr::desc(pa_rv))

# Ordered team choices for the matchup dropdowns (all 30 MLB teams).
mp_team_choices <- TEAM_META %>%
  dplyr::arrange(team_name) %>%
  { setNames(.$abbrev, .$team_name) }

# =============================================================================
# UI
# =============================================================================
year_choices <- setNames(
  as.character(AVAILABLE_YEARS),
  paste0(AVAILABLE_YEARS, "   ")
)

season_picker_inline <- function(id_suffix = "") {
  selectInput(
    paste0("season_year_", id_suffix),
    label    = NULL,
    choices  = year_choices,
    selected = as.character(DEFAULT_YEAR),
    width    = "100px"
  )
}

# Heading + role filter checkboxes + picker all on one line (Player tab)
heading_with_filter_picker <- function(id_suffix, heading_output_id) {
  tags$div(
    class = "xruns-heading-row",
    tags$div(
      class = "xruns-heading-text",
      textOutput(heading_output_id, inline = TRUE)
    ),
    tags$div(
      class = "xruns-filter-group",
      tags$span(class = "xruns-filter-label", "Show:"),
      checkboxGroupInput(
        "players_role_filter",
        label    = NULL,
        choices  = c("Hitters" = "Hitter", "Pitchers" = "Pitcher"),
        selected = c("Hitter", "Pitcher"),
        inline   = TRUE
      )
    ),
    tags$div(
      class = "xruns-season-chip",
      season_picker_inline(id_suffix)
    )
  )
}

# ---- theme -------------------------------------------------------------------
app_theme <- bs_theme(
  version      = 5,
  bootswatch   = "litera",
  primary      = "#17202a",
  secondary    = "#b2342a",
  "link-color" = "#9d2b1f",
  base_font    = font_google("Inter"),
  heading_font = font_google("Inter")
)

# ---- custom CSS --------------------------------------------------------------
custom_css <- HTML("
  /* ---- Base & typography ---- */
  body { background: #f8fafc; color: #1e293b; }

  /* ---- Player Profile — search bar ---- */
  .xruns-pp-search-wrap {
    padding: 18px 0 28px 0;
    /* Reserve enough space so the dropdown doesn't overlap the header below */
    min-height: 52px;
  }
  .xruns-pp-search-inner {
    position: relative;
    max-width: 460px;
    z-index: 100;
  }
  .xruns-pp-search-icon {
    position: absolute;
    left: 12px;
    top: 50%;
    transform: translateY(-50%);
    color: #94a3b8;
    font-size: 14px;
    pointer-events: none;
  }
  .xruns-pp-search-input {
    width: 100%;
    padding: 9px 12px 9px 36px;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    font-size: 14px;
    color: #1e293b;
    background: #ffffff;
    outline: none;
    box-sizing: border-box;
    transition: border-color 0.15s;
  }
  .xruns-pp-search-input:focus {
    border-color: #1a365d;
  }
  .xruns-pp-search-dropdown {
    position: absolute;
    top: calc(100% + 4px);
    left: 0;
    right: 0;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.10);
    z-index: 9999;
    max-height: 280px;
    overflow-y: auto;
  }
  .xruns-pp-search-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 14px;
    cursor: pointer;
    border-bottom: 1px solid #f8fafc;
    font-size: 13.5px;
    color: #1e293b;
  }
  .xruns-pp-search-item:last-child { border-bottom: none; }
  .xruns-pp-search-item:hover { background: #f1f5f9; }
  .xruns-pp-search-item img { height: 18px; flex-shrink: 0; }
  .xruns-pp-search-item-name { font-weight: 600; flex: 1; }
  .xruns-pp-search-item-meta { font-size: 11.5px; color: #94a3b8; }
  .xruns-pp-search-none {
    padding: 12px 14px;
    font-size: 13px;
    color: #94a3b8;
  }

  /* ---- Player Profile tab ---- */
  .xruns-pp-header {
    background: #1a365d;
    margin: -1rem -1rem 0 -1rem;
    padding: 22px 28px 24px;
    display: flex;
    align-items: center;
    gap: 20px;
  }
  /* Headshot container: clips img, shows initials if img fails */
  .xruns-pp-avatar-wrap {
    position: relative;
    width: 80px;
    height: 80px;
    border-radius: 50%;
    flex-shrink: 0;
    overflow: hidden;
    border: 2px solid rgba(255,255,255,0.25);
    background: #2d4a7a;
  }
  .xruns-pp-headshot {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 50%;
  }
  .xruns-pp-name {
    font-size: 1.45rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.02em;
    line-height: 1.2;
  }
  .xruns-pp-meta {
    font-size: 12.5px;
    color: rgba(255,255,255,0.65);
    margin-top: 5px;
    display: flex;
    align-items: center;
    gap: 7px;
  }
  .xruns-pp-meta img {
    height: 18px;
    vertical-align: middle;
  }
  .xruns-pp-meta-dot { opacity: 0.4; }
  .xruns-pp-pill {
    display: inline-block;
    background: rgba(255,255,255,0.13);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 11px;
    color: rgba(255,255,255,0.8);
    margin-top: 8px;
  }
  .xruns-pp-overall {
    margin-left: auto;
    text-align: right;
    flex-shrink: 0;
    align-self: flex-start;
    padding-top: 3px;
  }
  .xruns-pp-overall-val {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.03em;
  }
  .xruns-pp-overall-val.pos { color: #68d391; }
  .xruns-pp-overall-val.neg { color: #fc8181; }
  .xruns-pp-overall-val.neu { color: rgba(255,255,255,0.6); }
  .xruns-pp-overall-label {
    font-size: 10.5px;
    color: rgba(255,255,255,0.45);
    margin-top: 3px;
  }
  .xruns-pp-overall .xruns-share-btn {
    margin-top: 13px;
  }
  /* (back button removed — player profile is now a standalone page) */

  /* ---- Mobile / narrow-screen responsive overrides ---- */

  /* Player profile header: stack vertically below 600px */
  @media (max-width: 600px) {
    .xruns-pp-header {
      grid-template-columns: 72px minmax(0, 1fr);
      gap: 12px 14px;
      padding: 16px;
    }
    .xruns-pp-avatar-wrap {
      width: 72px;
      height: 72px;
    }
    .xruns-pp-overall {
      margin-left: 0;
      text-align: left;
      width: 100%;
      padding-top: 0;
    }
    .xruns-pp-name { font-size: 1.12rem; }
    .xruns-pp-overall-val { font-size: 1.5rem; }
    .xruns-pp-overall .xruns-share-btn {
      width: 100%;
      margin-top: 10px;
    }
    .xruns-pp-grid {
      grid-template-columns: 1fr;
    }
  }

  @media (max-width: 900px) {
    .xruns-pp-header {
      display: grid;
      grid-template-columns: 86px minmax(0, 1fr);
      align-items: start;
      gap: 14px 16px;
      padding: 18px;
      margin-left: 0;
      margin-right: 0;
    }
    .xruns-pp-avatar-wrap {
      width: 86px;
      height: 86px;
    }
    .xruns-pp-header > div:nth-child(2) {
      min-width: 0;
      grid-column: 2;
    }
    .xruns-pp-name {
      font-size: 1.25rem;
      overflow-wrap: anywhere;
    }
    .xruns-pp-meta {
      flex-wrap: wrap;
      line-height: 1.35;
    }
    .xruns-pp-overall {
      margin-left: 0;
      text-align: left;
      grid-column: 1 / -1;
      width: 100%;
      padding-top: 0;
    }
    .xruns-pp-overall-label {
      margin-top: 1px;
    }
    .xruns-pp-overall-val {
      font-size: 1.85rem;
    }
    .xruns-pp-overall .xruns-share-btn {
      width: 100%;
      margin-top: 10px;
    }
    .xruns-pp-grid {
      grid-template-columns: 1fr !important;
    }
  }

  /* Team breakdown header pills: wrap tightly on narrow screens */
  @media (max-width: 640px) {
    .xruns-page { padding-left: 12px !important; padding-right: 12px !important; }
    .xruns-pp-search-inner { max-width: 100%; }
  }

  /* DT responsive: style the child-row expand button and detail rows */
  table.dataTable td.dtr-control {
    padding-left: 10px !important;
  }
  table.dataTable td.dtr-control::before {
    background-color: transparent !important;
    border: 1.5px solid #94a3b8 !important;
    box-shadow: none !important;
    width: 14px !important;
    height: 14px !important;
    margin-top: -7px !important;
    border-radius: 3px !important;
    font-size: 14px !important;
    line-height: 13px !important;
    color: #94a3b8 !important;
    content: \"+\" !important;
  }
  table.dataTable tr.parent td.dtr-control::before {
    content: \"-\" !important;
    border-color: #1a365d !important;
    color: #1a365d !important;
  }
  table.dataTable tr.child td.dtr-details {
    font-size: 12.5px;
    padding: 6px 8px;
  }
  /* Tighten table padding on small screens */
  @media (max-width: 640px) {
    table.dataTable tbody td {
      padding-top: 8px !important;
      padding-bottom: 8px !important;
      font-size: 12.5px !important;
    }
    table.dataTable thead th { font-size: 0.62rem !important; }
    .tab-body { padding: 10px 8px 16px !important; }
    /* Hide team name column on mobile — logo is enough */
    th.team-name-col, td.team-name-col { display: none !important; }
  }
  /* Section grid */
  .xruns-pp-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-top: 14px;
  }
  .xruns-pp-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px 18px;
  }
  .xruns-pp-card-label {
    font-size: 10.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #94a3b8;
    margin-bottom: 12px;
  }
  /* Percentile bars */
  .xruns-pp-pct-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 5px 0;
    border-bottom: 1px solid #f1f5f9;
  }
  .xruns-pp-pct-row:last-child { border-bottom: none; }
  .xruns-pp-pct-label {
    font-size: 12px;
    color: #475569;
    width: 96px;
    flex-shrink: 0;
  }
  .xruns-pp-pct-track {
    flex: 1;
    height: 6px;
    background: #f1f5f9;
    border-radius: 3px;
    overflow: hidden;
  }
  .xruns-pp-pct-fill {
    height: 100%;
    border-radius: 3px;
    background: #047857;
  }
  .xruns-pp-pct-fill.neg-fill { background: #b91c1c; }
  .xruns-pp-pct-num {
    font-size: 11.5px;
    font-weight: 700;
    width: 34px;
    text-align: right;
    flex-shrink: 0;
    color: #047857;
  }
  .xruns-pp-pct-num.neg-pct { color: #b91c1c; }
  /* Composition — diverging bar rows */
  .xruns-pp-comp-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 0;
    border-bottom: 1px solid #f1f5f9;
    font-size: 12.5px;
  }
  .xruns-pp-comp-row:last-child { border-bottom: none; }
  .xruns-pp-comp-label {
    width: 90px;
    flex-shrink: 0;
    color: #64748b;
    font-size: 12px;
  }
  /* The two halves of the diverging track */
  .xruns-pp-comp-neg-half {
    flex: 1;
    display: flex;
    justify-content: flex-end;
  }
  .xruns-pp-comp-pos-half {
    flex: 1;
    display: flex;
    justify-content: flex-start;
  }
  .xruns-pp-comp-bar {
    height: 8px;
    border-radius: 2px;
    min-width: 2px;
  }
  .xruns-pp-comp-bar.pos { background: #047857; }
  .xruns-pp-comp-bar.neg { background: #b91c1c; }
  .xruns-pp-comp-val {
    width: 42px;
    flex-shrink: 0;
    text-align: right;
    font-size: 12px;
    font-weight: 600;
  }
  .xruns-pp-comp-val.pos { color: #047857; }
  .xruns-pp-comp-val.neg { color: #b91c1c; }
  .xruns-pp-comp-val.neu { color: #94a3b8; font-weight: 400; }
  /* Centre tick (zero line) */
  .xruns-pp-comp-zero {
    width: 1px;
    background: #cbd5e1;
    align-self: stretch;
    flex-shrink: 0;
  }
  .navbar {
    background: #ffffff !important;
    border-bottom: 1px solid #e2e8f0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    padding-top: 0.5rem; padding-bottom: 0.5rem;
  }
  .navbar-brand {
    font-weight: 700;
    font-size: 1.1rem;
    color: #1a365d !important;
    letter-spacing: -0.02em;
  }
  .navbar-brand .brand-sub {
    font-weight: 400;
    font-size: 0.8rem;
    color: #64748b;
    margin-left: 4px;
  }
  .nav-link {
    font-size: 0.88rem;
    font-weight: 500;
    color: #475569 !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
  }
  .nav-link.active, .nav-link:hover {
    color: #1a365d !important;
  }

  /* ---- Info stat pill (used inside collapsible banners) ---- */
  .info-stat {
    display: inline-block;
    background: #f1f5f9;
    border-radius: 4px;
    padding: 1px 7px;
    font-size: 12px;
    color: #1a365d;
    font-weight: 600;
    margin: 0 2px;
  }

  /* ---- Heading row: heading + optional filter group + season chip ---- */
  .xruns-heading-row {
    display: flex;
    align-items: center;
    padding: 12px 20px 6px 20px;
    gap: 12px;
    flex-wrap: wrap;
  }
  .xruns-heading-text {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1a365d;
    letter-spacing: -0.02em;
    flex: 1;
    min-width: 160px;
    white-space: nowrap;
  }

  /* ---- Time period chip toggle (Team Rankings tab) ---- */
  .xruns-period-group {
    display: flex;
    align-items: center;
    gap: 2px;
    flex-shrink: 0;
  }
  .xruns-period-chip {
    background: none;
    border: none;
    padding: 2px 6px;
    font-size: 11px;
    font-weight: 500;
    color: #94a3b8;
    cursor: pointer;
    border-radius: 4px;
    line-height: 1.4;
    transition: color 0.15s, background 0.15s;
    white-space: nowrap;
    letter-spacing: 0.01em;
  }
  .xruns-period-chip:hover {
    color: #475569;
    background: #f1f5f9;
  }
  .xruns-period-chip.active {
    color: #1a365d;
    font-weight: 700;
    background: #e8eef7;
  }
  .xruns-period-sep {
    color: #cbd5e1;
    font-size: 10px;
    line-height: 1;
    user-select: none;
  }
  .xruns-window-note {
    margin: 0 20px 6px 20px;
    padding: 6px 12px;
    background: #fffbeb;
    border-left: 3px solid #f59e0b;
    border-radius: 0 6px 6px 0;
    font-size: 12px;
    color: #92400e;
  }

  /* Inline role filter group (Player tab only) */
  .xruns-filter-group {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-shrink: 0;
  }
  .xruns-filter-label {
    font-size: 12px;
    color: #64748b;
    font-weight: 500;
    white-space: nowrap;
    line-height: 1;
    align-self: center;
  }
  .xruns-filter-group .form-check {
    margin-bottom: 0;
    margin-right: 4px;
    display: flex;
    align-items: center;
  }
  .xruns-filter-group .form-check-input { margin-top: 0; }
  .xruns-filter-group .form-check-label { font-size: 12.5px; margin-bottom: 0; line-height: 1; }

  /* Season chip — slim styled pill that sits inline with the heading */
  .xruns-season-chip {
    flex-shrink: 0;
  }
  .xruns-season-chip .form-group { margin-bottom: 0; }
  .xruns-season-chip .form-select,
  .xruns-season-chip .form-control {
    font-size: 10px;
    font-weight: 500;
    color: #64748b;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 2px 22px 2px 10px;
    height: auto;
    line-height: 1.4;
    white-space: nowrap;
    width: 100px;
    cursor: pointer;
  }
  .xruns-season-chip .form-select:focus,
  .xruns-season-chip .form-control:focus {
    border-color: #1a365d;
    box-shadow: 0 0 0 2px rgba(26,54,93,0.12);
    outline: none;
  }

  /* ---- Section headings (used for scatter sub-headings etc.) ---- */
  .section-heading {
    font-size: 1.2rem;
    font-weight: 700;
    color: #1a365d;
    padding: 20px 20px 4px 20px;
    letter-spacing: -0.02em;
  }
  .section-subheading {
    font-size: 0.8rem;
    color: #94a3b8;
    font-weight: 400;
    margin-left: 6px;
    letter-spacing: 0;
  }

  /* ---- Season picker row (kept for methodology tab) ---- */
  .xruns-season-row .form-group { margin-bottom: 0; }
  .xruns-season-row .form-select,
  .xruns-season-row .form-control { width: auto !important; font-size: 13px; }

  /* ---- KPI cards — compact strip ---- */
  .kpi-row {
    display: flex;
    gap: 8px;
    padding: 8px 20px 6px 20px;
    flex-wrap: nowrap;
    overflow-x: auto;
  }
  .kpi-card {
    flex: 1;
    min-width: 0;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 6px 10px;
    display: flex;
    align-items: center;
    gap: 8px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    white-space: nowrap;
  }
  .kpi-logo { width: 26px; height: 26px; object-fit: contain; flex-shrink: 0; }
  .kpi-logo-placeholder {
    width: 26px; height: 26px; display: flex; align-items: center;
    justify-content: center; font-weight: 700; font-size: 10px;
    background: #f1f5f9; border-radius: 4px; color: #1a365d; flex-shrink: 0;
  }
  .kpi-text { display: flex; flex-direction: column; gap: 0px; min-width: 0; }
  .kpi-label {
    font-size: 9px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
  }
  .kpi-team {
    font-size: 11px;
    font-weight: 700;
    color: #1e293b;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 120px;
  }
  .kpi-value {
    font-size: 10px;
    font-weight: 600;
    color: #047857;
  }
  .kpi-value.negative { color: #b91c1c; }

  /* (players-filter-row removed — filter is now inline in heading row) */

  /* ---- Tables ---- */
  table.dataTable { font-size: 13.5px; font-family: 'Inter', sans-serif; }
  table.dataTable thead th {
    font-size: 11.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #64748b;
    border-bottom: 2px solid #e2e8f0 !important;
    background: #f8fafc;
  }
  table.dataTable tbody tr { transition: background 0.1s; }
  table.dataTable tbody tr:hover { background: #f0f9ff !important; }
  .dataTables_wrapper { overflow-x: auto; }
  .dataTables_info, .dataTables_filter, .dataTables_length {
    font-size: 12px; color: #94a3b8;
  }

  /* ---- Content body padding ---- */
  .tab-body { padding: 0 20px 28px 20px; }

  /* ---- Methodology cards ---- */
  .card { border: 1px solid #e2e8f0; border-radius: 10px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
  .card-header {
    background: #f8fafc !important;
    font-size: 13px; font-weight: 600;
    color: #1a365d;
    border-bottom: 1px solid #e2e8f0;
    border-radius: 10px 10px 0 0 !important;
  }
  pre { font-size: 12px; background: #f8fafc; border: none; }

  /* ---- Standings warning ---- */
  .standings-warning {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-radius: 8px;
    padding: 14px 18px;
    color: #78350f;
    font-size: 13px;
  }

  /* ---- Matchup simulator selector cards ---- */
  .mp-selectors-row {
    display: flex;
    flex-direction: row;       /* side-by-side on desktop */
    gap: 10px;
    align-items: flex-start;
    padding: 0 20px 10px 20px;
  }
  .mp-selector-card {
    flex: 1;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    min-width: 0;
  }
  .mp-selector-card .form-select,
  .mp-selector-card .form-control {
    font-size: 13px;
    width: 100%;
    box-sizing: border-box;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .mp-card-inputs { width: 100%; }
  .mp-card-inputs .form-group { width: 100%; margin-bottom: 2px; }
  /* Selectize fixed width */
  .mp-card-inputs .selectize-control { width: 100% !important; }
  .mp-card-inputs .selectize-input {
    width: 100% !important;
    box-sizing: border-box;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  /* VS divider — vertical bar on desktop */
  .mp-vs-divider {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.0rem;
    font-weight: 700;
    color: #cbd5e0;
    flex-shrink: 0;
    padding: 40px 0 0 0;   /* push down to logo midpoint */
    min-width: 28px;
  }

  /* ---- Desktop (768px+) ---- */
  @media (min-width: 768px) {
    .kpi-row { flex-wrap: nowrap; overflow-x: visible; }
  }

  /* ---- Mobile (max 640px): stack matchup cards vertically ---- */
  @media (max-width: 640px) {
    .mp-selectors-row {
      flex-direction: column;   /* stack A, VS, B top-to-bottom */
      padding: 0 14px 8px 14px;
      gap: 0;
    }
    .mp-selector-card {
      flex-direction: row;       /* logo | selects side-by-side inside each card */
      align-items: flex-start;
      padding: 10px 12px;
      gap: 10px;
      width: 100%;
    }
    /* Logo stays left, inputs fill the rest */
    .mp-selector-card > .mp-team-logo-lg,
    .mp-selector-card > .mp-placeholder-lg {
      flex-shrink: 0;
      margin-top: 4px;
    }
    /* Wrap the sub-label + select pairs in a column div in the card */
    .mp-card-inputs {
      display: flex;
      flex-direction: column;
      gap: 4px;
      flex: 1;
      min-width: 0;
    }
    /* VS becomes a slim horizontal rule between cards */
    .mp-vs-divider {
      font-size: 0.8rem;
      padding: 6px 0;
      text-align: center;
      width: 100%;
      border-top: 1px solid #e2e8f0;
      border-bottom: 1px solid #e2e8f0;
      margin: 4px 0;
      color: #94a3b8;
      min-width: unset;
    }
  }

  /* ---- Small mobile (max 576px) ---- */
  @media (max-width: 576px) {
    .xruns-heading-row { padding: 10px 14px 4px 14px; gap: 8px; }
    .xruns-heading-text { font-size: 1.0rem; }
    .xruns-filter-group { gap: 4px; }
    .section-heading { padding: 14px 14px 4px 14px; font-size: 1.05rem; }
    /* Hide KPI cards on mobile */
    .kpi-row { display: none; }
    .tab-body { padding: 0 10px 20px 10px; }
    table.dataTable { font-size: 12px; }
  }

  /* ---- Data source footer ---- */
  .xruns-data-footer {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 18px 20px 24px 20px;
    margin-top: 12px;
    border-top: 1px solid #e2e8f0;
    color: #94a3b8;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.01em;
  }
  .xruns-data-footer a {
    color: #64748b;
    text-decoration: none;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 5px;
  }
  .xruns-data-footer a:hover { color: #1a365d; text-decoration: underline; }
  .xruns-data-footer img { height: 16px; width: 16px; vertical-align: middle; }
")

product_css <- HTML("
  :root {
    --xr-ink: #151922;
    --xr-muted: #667085;
    --xr-line: #d9dee8;
    --xr-paper: #ffffff;
    --xr-field: #f5f7fb;
    --xr-red: #b2342a;
    --xr-blue: #245c8f;
    --xr-green: #24745a;
    --xr-gold: #b4852d;
  }

  html, body {
    background:
      linear-gradient(180deg, #eef2f7 0, #f7f8fb 340px, #f7f8fb 100%);
    color: var(--xr-ink);
    letter-spacing: 0;
  }

  .navbar {
    background: rgba(21, 25, 34, 0.97) !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 12px 34px rgba(21, 25, 34, 0.16);
    min-height: 64px;
    padding: 0 18px !important;
    position: relative;
    z-index: 10020;
  }
  /* Stretch navbar content full width so tabs anchor to the right edge */
  .navbar > .container,
  .navbar > .container-fluid,
  .navbar > .container-sm,
  .navbar > .container-md,
  .navbar > .container-lg,
  .navbar > .container-xl {
    display: flex !important;
    align-items: center !important;
    min-height: 64px;
    width: 100% !important;
    max-width: 100% !important;
  }
  .navbar-brand {
    color: #ffffff !important;
    font-weight: 850;
    letter-spacing: 0;
    display: flex;
    align-items: baseline;
    gap: 10px;
    white-space: nowrap;
  }
  .navbar-brand .brand-sub {
    color: rgba(255, 255, 255, 0.62);
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    font-size: 0.68rem;
    margin-left: 0;
  }
  .navbar-toggler,
  .navbar-toggle {
    --bs-navbar-toggler-border-color: transparent;
    --bs-navbar-toggler-border-radius: 0;
    border: 0 !important;
    border-radius: 6px !important;
    box-shadow: none !important;
    background: transparent !important;
    outline: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    -webkit-appearance: none;
    appearance: none;
    display: none !important;
    position: relative;
    z-index: 10000;
    width: 52px;
    height: 52px;
    min-width: 52px;
    cursor: pointer !important;
    display: flex !important;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    transition: background 0.15s;
  }
  .navbar-toggler:hover,
  .navbar-toggle:hover {
    background: rgba(255,255,255,0.10) !important;
  }
  .navbar .navbar-toggler,
  .navbar .navbar-toggle,
  .navbar .navbar-toggler.collapsed,
  .navbar .navbar-toggle.collapsed,
  .navbar .navbar-toggler:not(.collapsed),
  .navbar .navbar-toggle:not(.collapsed),
  .navbar .navbar-toggler:hover,
  .navbar .navbar-toggle:hover,
  .navbar .navbar-toggler:focus,
  .navbar .navbar-toggle:focus,
  .navbar .navbar-toggler:active,
  .navbar .navbar-toggle:active,
  .navbar .navbar-toggler:focus-visible,
  .navbar .navbar-toggle:focus-visible {
    background: transparent !important;
    background-color: transparent !important;
    border: 0 none transparent !important;
    border-color: transparent !important;
    border-radius: 0 !important;
    border-style: none !important;
    border-width: 0 !important;
    box-shadow: none !important;
    outline: 0 !important;
  }
  @media (min-width: 992px) {
    .navbar-toggler,
    .navbar-toggle {
      display: none !important;
    }
    /* Make container-fluid a flex row: brand left, collapse right */
    .navbar .container-fluid {
      display: flex !important;
      flex-direction: row !important;
      align-items: center !important;
      flex-wrap: nowrap !important;
      width: 100% !important;
    }
    .navbar-header {
      flex: 0 0 auto;
    }
    .navbar-collapse {
      display: flex !important;
      visibility: visible !important;
      height: auto !important;
      overflow: visible !important;
      position: static !important;
      transform: none !important;
      top: auto !important;
      right: auto !important;
      flex: 1 1 auto !important;
      justify-content: flex-end !important;
      align-items: center !important;
    }
    .navbar-nav {
      float: none !important;
      display: flex !important;
      flex-direction: row !important;
      flex-wrap: nowrap !important;
      align-items: center !important;
      flex: 0 0 auto !important;
      gap: 2px;
      margin: 0 !important;
    }
    .nav-link {
      padding-left: 0.55rem !important;
      padding-right: 0.55rem !important;
      font-size: clamp(0.72rem, 1.45vw, 0.86rem);
    }
  }
  @media (max-width: 991px) {
    .navbar {
      overflow: visible !important;
    }
    .navbar > .container,
    .navbar > .container-fluid,
    .navbar > .container-sm,
    .navbar > .container-md,
    .navbar > .container-lg,
    .navbar > .container-xl {
      position: relative;
      overflow: visible !important;
    }
    .navbar .navbar-toggler,
    .navbar .navbar-toggle,
    .navbar-toggler,
    .navbar-toggle {
      display: flex !important;
      width: 52px !important;
      height: 52px !important;
      min-width: 52px !important;
      cursor: pointer !important;
    }
    .navbar-collapse {
      position: absolute !important;
      top: calc(100% + 10px) !important;
      left: 12px !important;
      right: 12px !important;
      width: auto !important;
      max-width: 680px;
      margin-left: auto !important;
      margin-right: auto !important;
      display: block !important;
      visibility: visible !important;
      height: auto !important;
      max-height: min(72vh, 440px);
      overflow-y: auto !important;
      overflow-x: hidden !important;
      padding: 10px !important;
      border: 1px solid rgba(255, 255, 255, 0.13);
      border-radius: 14px;
      background: rgba(21, 25, 34, 0.86) !important;
      -webkit-backdrop-filter: blur(18px) saturate(140%);
      backdrop-filter: blur(18px) saturate(140%);
      box-shadow: 0 24px 60px rgba(15, 23, 42, 0.34);
      opacity: 0;
      pointer-events: none;
      transform: translateY(-10px) scale(0.98) !important;
      transform-origin: top center;
      transition:
        opacity 180ms ease,
        transform 180ms ease;
      z-index: 10030;
    }
    .navbar-collapse.show,
    .navbar-collapse.collapsing {
      opacity: 1;
      pointer-events: auto;
      transform: translateY(0) scale(1) !important;
    }
    .navbar-nav {
      float: none !important;
      display: flex !important;
      flex-direction: column !important;
      gap: 4px !important;
      margin: 0 !important;
      width: 100%;
    }
    .navbar-nav .nav-link {
      display: block;
      width: 100%;
      margin: 0;
      padding: 12px 14px !important;
      border-radius: 10px;
      font-size: 1rem !important;
      line-height: 1.15;
      color: rgba(255, 255, 255, 0.72) !important;
      transition:
        background 140ms ease,
        color 140ms ease,
        transform 140ms ease;
    }
    .navbar-nav .nav-link:hover {
      background: rgba(255, 255, 255, 0.10);
      transform: translateX(2px);
    }
    .navbar-nav .nav-link.active {
      color: #ffffff !important;
      background: rgba(255, 255, 255, 0.15);
      box-shadow: inset 0 -2px 0 rgba(255, 255, 255, 0.72);
    }
    .navbar-brand .brand-sub {
      display: inline !important;
      font-size: 0.58rem;
      letter-spacing: 0.01em;
    }
    .navbar-brand {
      gap: 7px;
    }
  }
  @media (max-width: 520px) {
    .navbar {
      padding-left: 14px !important;
      padding-right: 14px !important;
    }
    .navbar-collapse {
      left: 2px !important;
      right: 2px !important;
      border-radius: 12px;
    }
    .navbar-nav .nav-link {
      padding: 11px 12px !important;
      font-size: 0.95rem !important;
    }
  }
  .navbar-toggler:hover,
  .navbar-toggle:hover,
  .navbar-toggler:active,
  .navbar-toggle:active {
    background: transparent !important;
    border: 0 !important;
  }
  .navbar-toggler:focus,
  .navbar-toggle:focus {
    box-shadow: none !important;
    outline: none;
  }
  .navbar-toggler-icon,
  .navbar-toggle .icon-bar {
    position: relative;
    width: 1.7rem;
    height: 1.2rem;
    opacity: 1;
    filter: none;
    background-image: none !important;
    display: none !important;
  }
  .navbar-toggler:before,
  .navbar-toggle:before {
    content: '';
    display: block;
    width: 26px;
    height: 2.5px;
    background: #ffffff;
    border-radius: 2px;
    box-shadow:
      0 8px 0 #ffffff,
      0 16px 0 #ffffff;
    pointer-events: none;
    margin-top: -8px;
  }
  .nav-link {
    color: rgba(255, 255, 255, 0.72) !important;
    border-radius: 6px;
    font-size: 0.86rem;
    font-weight: 720;
    letter-spacing: 0;
    margin: 0 2px;
  }
  .nav-link.active,
  .nav-link:hover {
    color: #ffffff !important;
    background: rgba(255, 255, 255, 0.10);
  }

  .xruns-page {
    width: min(1240px, calc(100vw - 32px));
    margin: 22px auto 40px;
  }
  .xruns-hero {
    display: grid;
    grid-template-columns: minmax(0, 1.35fr) minmax(260px, 0.65fr);
    gap: 12px;
    align-items: stretch;
    margin-bottom: 14px;
  }
  .xruns-hero-main,
  .xruns-hero-panel {
    background: var(--xr-paper);
    border: 1px solid rgba(21, 25, 34, 0.10);
    border-radius: 8px;
    box-shadow: 0 8px 24px rgba(21, 25, 34, 0.06);
  }
  .xruns-hero-main {
    position: relative;
    overflow: hidden;
    padding: 18px 22px 16px;
    border-top: 4px solid var(--xr-red);
  }
  .xruns-hero-main:after {
    content: '';
    position: absolute;
    inset: 0 0 auto auto;
    width: 42%;
    height: 100%;
    background:
      repeating-linear-gradient(115deg,
        rgba(36, 92, 143, 0.08) 0,
        rgba(36, 92, 143, 0.08) 1px,
        transparent 1px,
        transparent 14px);
    pointer-events: none;
  }
  .xruns-kicker {
    color: var(--xr-red);
    font-size: 0.72rem;
    font-weight: 850;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    margin-bottom: 6px;
  }
  .xruns-hero h1,
  .xruns-feature-head h1 {
    margin: 0;
    color: var(--xr-ink);
    font-size: clamp(2rem, 4vw, 4.2rem);
    line-height: 0.95;
    font-weight: 900;
    letter-spacing: 0;
    max-width: 800px;
  }
  .xruns-hero-copy,
  .xruns-feature-copy {
    position: relative;
    max-width: 760px;
    margin: 8px 0 0;
    color: #485264;
    font-size: 0.9rem;
    line-height: 1.6;
    z-index: 1;
  }
  .xruns-hero-panel {
    padding: 14px 18px;
    display: grid;
    align-content: center;
    gap: 8px;
    border-top: 4px solid var(--xr-blue);
  }
  .xruns-panel-label {
    color: var(--xr-muted);
    font-size: 0.72rem;
    font-weight: 850;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  .xruns-panel-number {
    color: var(--xr-ink);
    font-size: 2.55rem;
    font-weight: 900;
    line-height: 1;
  }
  .xruns-panel-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }
  .xruns-panel-stat {
    background: #f7f8fb;
    border: 1px solid var(--xr-line);
    border-radius: 8px;
    padding: 12px;
  }
  .xruns-panel-stat b {
    display: block;
    color: var(--xr-ink);
    font-size: 1.24rem;
    line-height: 1;
  }
  .xruns-panel-stat span {
    display: block;
    margin-top: 5px;
    color: var(--xr-muted);
    font-size: 0.74rem;
    font-weight: 750;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  .xruns-feature-head {
    background: var(--xr-paper);
    border: 1px solid rgba(21, 25, 34, 0.10);
    border-top: 5px solid var(--xr-red);
    border-radius: 8px;
    padding: 30px 34px;
    margin-bottom: 18px;
    box-shadow: 0 18px 45px rgba(21, 25, 34, 0.07);
  }
  .xruns-feature-head h1 {
    font-size: clamp(1.85rem, 3.2vw, 3.2rem);
  }

  .xruns-explainer-wrap,
  .xruns-control-wrap,
  .tab-body,
  .section-heading,
  .mp-selectors-row,
  .xruns-action-row,
  .xruns-chart-wrap,
  .xruns-method-wrap {
    background: transparent;
  }
  .xruns-explainer-wrap {
    margin: 0 0 12px 0 !important;
  }
  .xruns-explainer-wrap details,
  details.xruns-explainer {
    background: #ffffff !important;
    border: 1px solid rgba(21, 25, 34, 0.10) !important;
    border-left: 1px solid rgba(21, 25, 34, 0.10) !important;
    border-radius: 8px !important;
    box-shadow: 0 8px 22px rgba(21, 25, 34, 0.04) !important;
  }
  .xruns-explainer-wrap summary,
  details.xruns-explainer summary {
    color: #6b7280 !important;
    font-size: 0.83rem !important;
    font-weight: 720 !important;
    letter-spacing: 0;
    padding: 8px 12px !important;
  }
  .xruns-explainer-wrap summary i,
  details.xruns-explainer summary i {
    color: #7b8494;
    font-size: 0.78rem;
  }
  .xruns-explainer-body {
    padding: 0 12px 10px 12px;
    color: #6b7280;
    line-height: 1.55;
    font-size: 0.78rem;
  }
  .info-stat {
    background: #f2f4f8;
    color: #6b7280;
    border: 1px solid #e5e8ef;
  }

  .xruns-heading-row {
    background: #ffffff;
    border: 1px solid rgba(21, 25, 34, 0.10);
    border-radius: 8px 8px 0 0;
    padding: 16px 18px 12px 18px;
    margin-top: 12px;
    gap: 10px;
  }
  .xruns-heading-text {
    color: var(--xr-ink);
    font-size: 1.12rem;
    font-weight: 880;
    white-space: normal;
    line-height: 1.2;
    min-width: 240px;
  }
  .xruns-share-row {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    margin: 8px 0 10px 0;
  }
  .xruns-share-btn,
  .xruns-share-btn:visited {
    display: inline-flex !important;
    align-items: center;
    justify-content: center;
    gap: 7px;
    max-width: 100%;
    min-height: 38px;
    padding: 8px 13px !important;
    border-radius: 6px !important;
    background: #1a365d !important;
    border: 1px solid #1a365d !important;
    color: #ffffff !important;
    font-size: 0.78rem !important;
    font-weight: 820 !important;
    line-height: 1.2 !important;
    white-space: normal !important;
    text-decoration: none !important;
    box-shadow: 0 8px 18px rgba(26, 54, 93, 0.16);
  }
  .xruns-share-btn:hover,
  .xruns-share-btn:focus {
    background: #142a49 !important;
    border-color: #142a49 !important;
    color: #ffffff !important;
    text-decoration: none !important;
  }
  .xruns-share-btn-secondary,
  .xruns-share-btn-secondary:visited {
    background: #ffffff !important;
    border-color: #cfd6e2 !important;
    color: #1a365d !important;
    box-shadow: none;
  }
  .xruns-share-btn-secondary:hover,
  .xruns-share-btn-secondary:focus {
    background: #f2f4f8 !important;
    border-color: #98a2b3 !important;
    color: #1a365d !important;
  }
  .xruns-share-inline {
    margin-left: auto;
    flex-shrink: 0;
  }
  .xruns-share-inline .xruns-share-btn {
    box-shadow: none;
  }
  .xruns-pp-inline-share {
    margin-top: 10px;
  }
  .xruns-season-chip .form-select,
  .xruns-season-chip .form-control,
  .xruns-season-row .form-select,
  .xruns-season-row .form-control,
  .mp-selector-card .form-select,
  .mp-selector-card .form-control,
  .selectize-input {
    border: 1px solid #cfd6e2 !important;
    border-radius: 6px !important;
    background: #ffffff !important;
    color: var(--xr-ink) !important;
    font-weight: 720 !important;
    box-shadow: none !important;
  }
  .mp-card-inputs .selectize-control {
    margin-bottom: 0;
  }
  .mp-card-inputs .selectize-control.single {
    background: #ffffff;
    border: 1px solid #cfd6e2;
    border-radius: 6px;
    box-shadow: none;
    overflow: visible;
  }
  .mp-card-inputs .selectize-control.single:focus-within {
    border-color: #98a2b3;
    box-shadow: 0 0 0 3px rgba(21, 25, 34, 0.06);
  }
  .mp-card-inputs .selectize-input,
  .mp-card-inputs .selectize-input.focus,
  .mp-card-inputs .selectize-input.input-active {
    min-height: 45px;
    padding: 10px 42px 10px 12px !important;
    line-height: 1.35 !important;
    background: #ffffff !important;
    border: 0 !important;
    box-shadow: none !important;
    outline: none !important;
  }
  .mp-card-inputs .selectize-input:after {
    right: 14px !important;
  }
  .mp-card-inputs .selectize-input > input {
    box-shadow: none !important;
  }
  .mp-card-inputs .selectize-dropdown {
    border: 1px solid #cfd6e2 !important;
    border-radius: 6px !important;
    box-shadow: 0 12px 26px rgba(21, 25, 34, 0.10) !important;
  }
  .xruns-period-chip {
    border: 1px solid transparent;
    border-radius: 999px;
    padding: 5px 10px;
    color: #5b6678;
    font-size: 0.73rem;
    font-weight: 780;
  }
  .xruns-period-chip:hover {
    background: #f3f5f9;
    color: var(--xr-ink);
  }
  .xruns-period-chip.active {
    background: #151922;
    color: #ffffff;
    border-color: #151922;
  }
  .xruns-period-sep {
    display: none;
  }
  .xruns-window-note,
  .standings-warning {
    background: #fff8e8;
    border: 1px solid #efd392;
    border-left: 5px solid var(--xr-gold);
    border-radius: 8px;
    color: #76531b;
    margin: 0 0 14px 0;
  }

  .kpi-row {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 8px;
    padding: 0;
    margin: 8px 0 10px 0;
    overflow: visible;
  }
  .kpi-card {
    border: 1px solid rgba(21, 25, 34, 0.10);
    border-radius: 6px;
    padding: 8px 10px;
    min-height: 54px;
    box-shadow: 0 6px 14px rgba(21, 25, 34, 0.04);
    position: relative;
    gap: 7px;
  }
  .kpi-card:before {
    content: '';
    position: absolute;
    inset: 0 auto 0 0;
    width: 3px;
    background: var(--xr-ink);
    border-radius: 6px 0 0 6px;
  }
  .kpi-logo { width: 24px; height: 24px; }
  .kpi-logo-placeholder { width: 24px; height: 24px; background: #eef2f7; color: var(--xr-ink); font-size: 9px; }
  .kpi-label {
    color: var(--xr-muted);
    font-size: 0.58rem;
    font-weight: 850;
    letter-spacing: 0.07em;
    display: flex;
    align-items: center;
    gap: 4px;
  }
  .kpi-label i { color: var(--xr-ink); font-size: 0.66rem; }
  .kpi-team {
    color: var(--xr-ink);
    font-size: 0.74rem;
    max-width: 150px;
  }
  .kpi-value {
    color: var(--xr-green);
    font-size: 0.64rem;
    font-weight: 820;
  }

  .tab-body {
    background: #ffffff;
    border: 1px solid rgba(21, 25, 34, 0.10);
    border-top: 0;
    border-radius: 0 0 8px 8px;
    padding: 16px 18px 22px 18px;
    box-shadow: 0 14px 34px rgba(21, 25, 34, 0.06);
  }
  .section-heading {
    color: var(--xr-ink);
    font-size: 1.28rem;
    font-weight: 890;
    padding: 28px 4px 10px 4px;
  }
  .section-subheading {
    color: var(--xr-muted);
    font-weight: 650;
  }
  table.dataTable {
    border-collapse: separate !important;
    border-spacing: 0;
    font-size: 13px;
  }
  table.dataTable thead th {
    background: #f2f4f8 !important;
    color: #505b6f;
    border-bottom: 1px solid var(--xr-line) !important;
    font-size: 0.68rem;
    font-weight: 850;
    letter-spacing: 0.08em;
  }
  table.dataTable tbody td {
    border-bottom: 1px solid #eef1f5;
    padding-top: 10px !important;
    padding-bottom: 10px !important;
  }
  table.dataTable tbody tr:hover {
    background: #f6f8fb !important;
  }
  .dataTables_filter input,
  .dataTables_length select {
    border: 1px solid #cfd6e2;
    border-radius: 6px;
    padding: 6px 8px;
    background: #ffffff;
  }

  .card {
    border: 1px solid rgba(21, 25, 34, 0.10);
    border-radius: 8px;
    box-shadow: 0 14px 34px rgba(21, 25, 34, 0.06);
  }
  .card-header {
    background: #f2f4f8 !important;
    color: var(--xr-ink);
    border-bottom: 1px solid var(--xr-line);
    border-radius: 8px 8px 0 0 !important;
    font-size: 0.78rem;
    font-weight: 850;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  pre {
    color: #283243;
    background: #f7f8fb;
    border-radius: 6px;
    padding: 14px;
  }

  .mp-selectors-row {
    padding: 0;
    margin-bottom: 12px;
    gap: 14px;
  }
  .mp-selector-card {
    border: 1px solid rgba(21, 25, 34, 0.10);
    border-radius: 8px;
    padding: 18px;
    box-shadow: 0 14px 34px rgba(21, 25, 34, 0.06);
    background: #ffffff;
  }
  .mp-vs-divider {
    color: var(--xr-red);
    font-weight: 900;
    letter-spacing: 0.08em;
  }
  .mp-sub-label {
    color: var(--xr-muted);
    font-weight: 850;
    letter-spacing: 0.08em;
  }
  .mp-team-logo-lg,
  .mp-placeholder-lg { width: 72px; height: 72px; }
  .mp-exp-runs,
  .mp-sim-score {
    color: var(--xr-ink);
  }
  .mp-win-bar {
    height: 34px;
    border-radius: 6px;
    background: #eef2f7;
  }
  .btn-primary,
  .btn-outline-secondary {
    border-radius: 6px !important;
    font-weight: 850 !important;
    letter-spacing: 0.01em;
  }
  .btn-primary {
    background: var(--xr-ink) !important;
    border-color: var(--xr-ink) !important;
  }
  .btn-primary:hover {
    background: #2a303b !important;
    border-color: #2a303b !important;
  }

  .xruns-method-wrap {
    padding: 0 !important;
  }
  .xruns-data-footer {
    background: #151922;
    color: rgba(255, 255, 255, 0.58);
    border-top: 0;
    margin-top: 36px;
  }
  .xruns-data-footer a {
    color: #ffffff;
  }

  @media (max-width: 760px) {
    .xruns-hero { grid-template-columns: 1fr; }
    .kpi-row { display: none !important; }
  }
  @media (max-width: 640px) {
    .xruns-page { width: min(100vw - 20px, 1240px); margin-top: 14px; }
    .xruns-hero-main,
    .xruns-feature-head { padding: 24px 20px; }
    .xruns-hero h1,
    .xruns-feature-head h1 { font-size: 2.15rem; line-height: 1; }
    .xruns-panel-grid { grid-template-columns: 1fr; }
    .kpi-row { display: none !important; }
    .xruns-heading-row { align-items: flex-start; }
    .xruns-heading-text { min-width: 100%; }
    .xruns-share-inline {
      width: 100%;
      margin-left: 0 !important;
    }
    .xruns-share-inline .xruns-share-btn {
      width: 100%;
    }
    .xruns-action-row {
      padding-left: 0 !important;
      padding-right: 0 !important;
    }
    .xruns-action-row .btn,
    .xruns-action-row .xruns-share-btn {
      width: min(100%, 300px);
      min-width: 0 !important;
    }
    #download_team_card,
    #download_rankings_card,
    #download_matchup_card,
    #download_player_card {
      width: 100%;
    }
    .xruns-period-group { flex-wrap: wrap; }
    .tab-body { padding: 12px 10px 18px 10px; }
    .mp-selectors-row { padding: 0; gap: 8px; }
  }
  @media (max-width: 820px) {
    .mp-selectors-row {
      flex-direction: column !important;
      padding: 0 !important;
      gap: 10px !important;
    }
    .mp-selector-card {
      width: 100% !important;
      flex-direction: row !important;
      align-items: center !important;
      gap: 12px !important;
      padding: 12px !important;
    }
    .mp-selector-card > .mp-team-logo-lg,
    .mp-selector-card > .mp-placeholder-lg {
      width: 56px !important;
      height: 56px !important;
      flex: 0 0 56px !important;
      margin-top: 0 !important;
    }
    .mp-card-inputs {
      min-width: 0 !important;
      flex: 1 1 auto !important;
    }
    .mp-sub-label {
      font-size: 0.68rem !important;
      line-height: 1.1 !important;
      margin-bottom: 3px !important;
    }
    .mp-selector-card .form-select,
    .mp-selector-card .form-control,
    .mp-card-inputs .selectize-input,
    #mp_starter_a .selectize-input,
    #mp_starter_b .selectize-input {
      min-height: 38px !important;
      padding: 7px 34px 7px 10px !important;
      font-size: 0.92rem !important;
      line-height: 1.25 !important;
      white-space: nowrap !important;
      overflow: hidden !important;
      text-overflow: ellipsis !important;
    }
    .mp-vs-divider {
      width: 100% !important;
      min-width: 0 !important;
      padding: 6px 0 !important;
      margin: 0 !important;
      border-top: 1px solid #e2e8f0;
      border-bottom: 1px solid #e2e8f0;
    }
    .mp-results-wrap {
      padding-left: 0 !important;
      padding-right: 0 !important;
    }
    .mp-header-row {
      flex-direction: column !important;
      justify-content: center !important;
      gap: 16px !important;
      padding: 14px 12px !important;
      overflow: hidden !important;
    }
    .mp-team-block {
      width: 100% !important;
      min-width: 0 !important;
    }
    .mp-team-name-lg {
      max-width: 100% !important;
      font-size: 1rem !important;
      line-height: 1.2 !important;
      overflow-wrap: anywhere !important;
    }
    .mp-exp-runs {
      font-size: 2.4rem !important;
    }
    .mp-score-row {
      grid-template-columns: 1fr !important;
      gap: 6px !important;
      align-items: flex-start !important;
    }
    .mp-score-label {
      font-size: 0.92rem !important;
      white-space: normal !important;
    }
    .mp-win-seg {
      font-size: 0.78rem !important;
      padding: 0 4px !important;
      white-space: nowrap !important;
    }
    .tb-header-card {
      display: grid !important;
      grid-template-columns: 72px minmax(0, 1fr) !important;
      align-items: center !important;
      gap: 14px !important;
      padding: 18px !important;
    }
    .tb-header-logo img,
    .tb-header-logo > div {
      width: 64px !important;
      height: 64px !important;
    }
    .tb-header-title {
      min-width: 0 !important;
    }
    .tb-header-title > div:first-child {
      font-size: 1.35rem !important;
      overflow-wrap: anywhere !important;
    }
    .tb-rating-pills {
      grid-column: 1 / -1 !important;
      display: grid !important;
      grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
      gap: 8px !important;
      width: 100% !important;
    }
    .tb-rating-pill {
      min-width: 0 !important;
      padding: 10px 8px !important;
    }
    .tb-rating-pill .tb-pill-value {
      font-size: 1.18rem !important;
    }
  }
  @media (max-width: 520px) {
    .tb-rating-pills {
      grid-template-columns: 1fr !important;
    }
  }

  /* ============================================================== */
  /* ====  MOBILE OVERRIDES (anchored banner + polish)          ==== */
  /* ============================================================== */

  /* Kill any default top spacing that pushes the banner down.
     Posit Connect Cloud iframes the page; if the body has a top
     margin or the navbar isn't sticky, the dark header drifts away
     from the URL bar and looks 'cut short'. */
  html, body {
    margin: 0 !important;
    padding: 0 !important;
  }
  body {
    /* Allow safe-area on notched iPhones without leaving a gap above the navbar */
    padding-top: 0 !important;
    padding-left: env(safe-area-inset-left, 0) !important;
    padding-right: env(safe-area-inset-right, 0) !important;
  }

  /* The page wrapper that bslib injects around content */
  .bslib-page-fill,
  .bslib-page-navbar,
  .bslib-page,
  body > .container-fluid:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
  }

  /* Anchor the dark banner to the very top of the viewport.
     This is the core fix: on mobile the navbar now sticks to the
     top edge so it visually butts up against the browser URL bar
     instead of floating in a sea of light grey. */
  .navbar {
    position: sticky !important;
    top: 0 !important;
    margin-top: 0 !important;
    width: 100% !important;
    z-index: 10050 !important;
    /* Respect iPhone notch safe area */
    padding-top: max(env(safe-area-inset-top, 0), 0px) !important;
  }

  /* On mobile specifically, tighten the navbar and make the brand
     scale gracefully so 'xRuns AN MLB RATINGS SYSTEM' never wraps
     awkwardly or runs into the hamburger button. */
  @media (max-width: 768px) {
    .navbar {
      min-height: 56px !important;
      padding: 0 14px !important;
    }
    .navbar > .container,
    .navbar > .container-fluid {
      min-height: 56px !important;
      padding-left: 0 !important;
      padding-right: 0 !important;
    }
    .navbar-brand {
      font-size: 1.18rem !important;
      gap: 8px !important;
      max-width: calc(100vw - 80px) !important;
      overflow: hidden !important;
    }
    .navbar-brand .xruns-brand-full {
      display: inline-flex !important;
      align-items: baseline !important;
      gap: 8px !important;
      min-width: 0 !important;
    }
    .navbar-brand .brand-sub {
      font-size: 0.6rem !important;
      letter-spacing: 0.06em !important;
      opacity: 0.72;
      white-space: nowrap !important;
      overflow: hidden !important;
      text-overflow: ellipsis !important;
    }
    .navbar-toggler,
    .navbar-toggle {
      width: 44px !important;
      height: 44px !important;
      min-width: 44px !important;
    }
  }

  /* On very narrow viewports (≤ 380px, e.g. iPhone SE), shorten the
     subtitle so 'xRuns' never gets squeezed by the hamburger. */
  @media (max-width: 380px) {
    .navbar-brand .brand-sub {
      font-size: 0.55rem !important;
    }
  }

  /* Pull the page content closer to the navbar on mobile so the
     hero block reads as part of the banner — no awkward grey gap. */
  @media (max-width: 768px) {
    .xruns-page {
      width: 100% !important;
      max-width: 100% !important;
      margin: 0 auto !important;
      padding: 14px 14px 32px !important;
    }
    .xruns-hero {
      gap: 10px !important;
      margin-bottom: 12px !important;
    }
    .xruns-hero-main {
      padding: 18px 18px 16px !important;
    }
    .xruns-hero-panel {
      padding: 14px 16px !important;
    }
    .xruns-kicker {
      font-size: 0.7rem !important;
      letter-spacing: 0.1em !important;
    }
    .xruns-hero-copy {
      font-size: 0.92rem !important;
      line-height: 1.55 !important;
    }
    .xruns-panel-label {
      font-size: 0.7rem !important;
    }
  }

  /* Soften the body gradient on mobile: the original gradient peaks
     at ~340px down, which on a phone viewport (~700px tall in the
     iframe) creates a visible light band right under the navbar
     that makes the banner look detached. */
  @media (max-width: 768px) {
    html, body {
      background:
        linear-gradient(180deg, #f5f7fa 0, #f7f8fb 120px, #f7f8fb 100%) !important;
    }
  }

  /* Make the share card downloads / table containers respect the
     full mobile width so nothing overflows horizontally. */
  @media (max-width: 768px) {
    .container-fluid,
    .tab-content,
    .tab-pane {
      padding-left: 0 !important;
      padding-right: 0 !important;
      max-width: 100% !important;
    }
    /* Shiny's auto-generated wrapper around nav_panel content */
    .tab-content > .tab-pane > .container-fluid {
      padding-left: 0 !important;
      padding-right: 0 !important;
    }
  }

  /* Prevent any horizontal scroll bleed from wide tables/cards */
  @media (max-width: 768px) {
    body, html {
      overflow-x: hidden !important;
      max-width: 100vw !important;
    }
  }

  /* Subtle: when navbar is sticky and user scrolls, give it a
     slightly stronger shadow so it reads as a true app bar. */
  .navbar {
    transition: box-shadow 180ms ease;
  }
")

# ---- Share card helpers ------------------------------------------------------
xruns_safe_text <- function(x, fallback = "") {
  if (is.null(x) || length(x) == 0 || is.na(x[1])) return(fallback)
  as.character(x[1])
}

xruns_fmt_signed <- function(x, digits = 2) {
  if (is.null(x) || length(x) == 0 || is.na(x[1]) || !is.finite(x[1])) return("N/A")
  sprintf(paste0("%+.", digits, "f"), x[1])
}

xruns_slug <- function(x) {
  slug <- tolower(gsub("[^A-Za-z0-9]+", "-", xruns_safe_text(x, "xruns")))
  slug <- gsub("(^-+|-+$)", "", slug)
  if (nzchar(slug)) slug else "xruns"
}

xruns_rating_color <- function(x) {
  if (is.null(x) || length(x) == 0 || is.na(x[1]) || !is.finite(x[1])) return("#64748b")
  if (x[1] > 0.01) "#047857" else if (x[1] < -0.01) "#b91c1c" else "#64748b"
}

xruns_team_color <- function(row, fallback = "#1a365d") {
  col <- if ("team_color" %in% names(row)) xruns_safe_text(row$team_color, fallback = NA_character_) else NA_character_
  if ((is.na(col) || !grepl("^#[0-9A-Fa-f]{6}$", col)) && "abbrev" %in% names(row)) {
    idx <- match(row$abbrev[1], TEAM_META$abbrev)
    if (!is.na(idx) && "team_color" %in% names(TEAM_META)) {
      col <- xruns_safe_text(TEAM_META$team_color[idx], fallback = NA_character_)
    }
  }
  if (is.na(col) || !grepl("^#[0-9A-Fa-f]{6}$", col)) fallback else col
}

xruns_headshot_url <- function(player_id, width = 220, height = width) {
  sprintf(
    "https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_%d,h_%d,c_fill,g_face,q_auto:best/v1/people/%d/headshot/67/current",
    width,
    height,
    as.integer(player_id)
  )
}

xruns_current_base_url <- function(session) {
  protocol <- session$clientData$url_protocol %||% "https:"
  host     <- session$clientData$url_hostname %||% "xruns"
  port     <- session$clientData$url_port %||% ""
  path     <- session$clientData$url_pathname %||% "/"
  port_part <- if (nzchar(port) && !(port %in% c("80", "443"))) paste0(":", port) else ""
  paste0(protocol, "//", host, port_part, path)
}

xruns_route_hash <- function(path) {
  path <- xruns_safe_text(path, "team-rankings")
  path <- sub("^[/# !]+", "", path)
  paste0("#/", path)
}

xruns_parse_route_hash <- function(hash) {
  route <- xruns_safe_text(hash, "")
  if (!nzchar(route)) return(list(tab = ""))
  route <- utils::URLdecode(route)
  route <- sub("^#!?/?", "", route)
  route <- sub("^/+", "", route)
  route <- sub("\\?.*$", "", route)
  route <- gsub("/+", "/", route)
  route <- gsub("_", "-", route)
  route <- tolower(route)
  parts <- unlist(strsplit(route, "/", fixed = TRUE), use.names = FALSE)
  parts <- parts[nzchar(parts)]
  if (length(parts) == 0) return(list(tab = ""))

  first <- parts[1]
  if (first %in% c("team-rankings", "rankings")) {
    return(list(tab = "rankings"))
  }
  if (first %in% c("player-rankings", "players")) {
    return(list(tab = "players"))
  }
  if (first %in% c("methodology", "model")) {
    return(list(tab = "methodology"))
  }

  collapsed <- paste(parts, collapse = "-")
  if (first == "team-profile" || startsWith(collapsed, "team-profile-")) {
    abbrev <- if (length(parts) >= 2) parts[2] else sub("^team-profile-", "", collapsed)
    abbrev <- toupper(sub("-.*$", "", abbrev))
    return(list(tab = "team", abbrev = abbrev))
  }

  if (first == "player-profile" || startsWith(collapsed, "player-profile-")) {
    id_source <- if (length(parts) >= 2) paste(parts[-1], collapse = "-") else sub("^player-profile-", "", collapsed)
    nums <- regmatches(id_source, gregexpr("[0-9]+", id_source))[[1]]
    pid <- if (length(nums) > 0) suppressWarnings(as.integer(tail(nums, 1))) else NA_integer_
    return(list(tab = "player", id = pid))
  }

  if (first == "matchup-simulator" || startsWith(collapsed, "matchup-simulator-")) {
    team_detail <- if (length(parts) >= 2) parts[2] else sub("^matchup-simulator-", "", collapsed)
    starter_detail <- if (length(parts) >= 3) paste(parts[-c(1, 2)], collapse = "-") else ""
    teams <- regmatches(team_detail, gregexpr("\\b[A-Za-z]{2,3}\\b", team_detail))[[1]]
    teams <- toupper(teams[teams %in% tolower(TEAM_META$abbrev) | toupper(teams) %in% TEAM_META$abbrev])
    teams <- unique(teams)
    nums <- regmatches(starter_detail, gregexpr("[0-9]{4,}", starter_detail))[[1]]
    starter_a <- if (length(nums) >= 1) suppressWarnings(as.integer(nums[1])) else NA_integer_
    starter_b <- if (length(nums) >= 2) suppressWarnings(as.integer(nums[2])) else NA_integer_
    return(list(
      tab = "matchup",
      team_a = if (length(teams) >= 1) teams[1] else NA_character_,
      team_b = if (length(teams) >= 2) teams[2] else NA_character_,
      starter_a = starter_a,
      starter_b = starter_b
    ))
  }

  list(tab = "")
}

xruns_player_profile_route <- function(row, player_id = NULL) {
  pid <- player_id %||% if (!is.null(row) && "player_id" %in% names(row)) row$player_id[1] else NA_integer_
  name <- if (!is.null(row) && "Player" %in% names(row)) row$Player[1] else "player"
  if (is.na(pid)) {
    xruns_route_hash("player-profile")
  } else {
    xruns_route_hash(paste0("player-profile/", xruns_slug(name), "-", as.integer(pid)))
  }
}

xruns_team_profile_route <- function(abbrev) {
  abbrev <- tolower(xruns_safe_text(abbrev, "lad"))
  xruns_route_hash(paste0("team-profile/", xruns_slug(abbrev)))
}

xruns_pitcher_route_label <- function(player_id) {
  pid <- suppressWarnings(as.integer(player_id))
  if (is.na(pid)) return(NULL)
  row <- mp_current_pitchers %>% dplyr::filter(player_id == pid) %>% head(1)
  name <- if (nrow(row) > 0) row$player[1] else "starter"
  paste0(xruns_slug(name), "-", pid)
}

xruns_matchup_route <- function(team_a, team_b, starter_a = NULL, starter_b = NULL) {
  matchup <- paste0(
    "matchup-simulator/",
    xruns_slug(team_a), "-vs-", xruns_slug(team_b)
  )
  starter_bits <- Filter(Negate(is.null), list(
    xruns_pitcher_route_label(starter_a),
    xruns_pitcher_route_label(starter_b)
  ))
  if (length(starter_bits) > 0) {
    matchup <- paste0(matchup, "/", paste(starter_bits, collapse = "-vs-"))
  }
  xruns_route_hash(matchup)
}

xruns_display_base_url <- function(base_url, card_type = "rankings") {
  url <- xruns_safe_text(base_url, fallback = "xruns")
  if (grepl("(^|//)(localhost|127\\.0\\.0\\.1|0\\.0\\.0\\.0)(:|/|$)", url)) {
    return("Dashboard preview")
  }
  host <- sub("^https?://", "", url)
  host <- sub("/.*$", "", host)
  host <- sub("^www\\.", "", host)
  path <- switch(
    card_type,
    "matchup" = "matchup-simulator",
    "team" = "team-profile",
    "player" = "player-profile",
    "rankings" = "team-rankings",
    card_type
  )
  label <- paste0(host, "/", path)
  if (nchar(label) > 58) paste0(substr(label, 1, 55), "...") else label
}

xruns_share_card_css <- function() {
'
  * { box-sizing: border-box; }
  html, body {
    margin: 0;
    padding: 0;
    background: transparent;
    color: #1e293b;
    font-family: Inter, Arial, sans-serif;
  }
  body { padding: 28px; }
  .xruns-share-card {
    width: 1200px;
    height: 675px;
    position: relative;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    background: #ffffff;
    border: 1px solid #dbe3ef;
    border-radius: 18px;
    box-shadow: 0 24px 70px rgba(15, 23, 42, 0.16);
  }
  .xruns-share-band {
    background: #1a365d;
    color: #ffffff;
    padding: 30px 42px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 26px;
  }
  .xruns-share-brand {
    color: rgba(255, 255, 255, 0.72);
    font-size: 20px;
    font-weight: 800;
    letter-spacing: 0;
  }
  .xruns-share-kicker {
    color: rgba(255, 255, 255, 0.62);
    font-size: 16px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.13em;
    margin-top: 5px;
  }
  .xruns-share-title {
    color: #ffffff;
    font-size: 52px;
    font-weight: 900;
    line-height: 1.02;
    letter-spacing: 0;
    margin-top: 8px;
  }
  .xruns-share-meta {
    color: rgba(255, 255, 255, 0.74);
    font-size: 20px;
    font-weight: 650;
    margin-top: 10px;
    line-height: 1.22;
  }
  .xruns-share-logo {
    width: 118px;
    height: 118px;
    object-fit: contain;
    flex: 0 0 auto;
  }
  .xruns-share-headshot {
    width: 138px;
    height: 138px;
    border-radius: 50%;
    background-color: #e2e8f0;
    border: 5px solid rgba(255, 255, 255, 0.34);
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.18);
    overflow: hidden;
    flex: 0 0 auto;
  }
  .xruns-share-headshot img {
    width: 100%;
    height: 100%;
    display: block;
    object-fit: cover;
    object-position: center center;
  }
  .xruns-share-body {
    padding: 30px 42px 70px;
    flex: 1 1 auto;
    min-height: 0;
    display: grid;
    grid-template-columns: 0.92fr 1.08fr;
    gap: 34px;
  }
  .xruns-share-stat {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 24px 26px;
  }
  .xruns-share-label {
    color: #64748b;
    font-size: 16px;
    font-weight: 850;
    letter-spacing: 0.10em;
    text-transform: uppercase;
  }
  .xruns-share-big {
    font-size: 88px;
    line-height: 0.95;
    font-weight: 900;
    letter-spacing: 0;
    margin-top: 14px;
  }
  .xruns-share-sub {
    color: #64748b;
    font-size: 19px;
    font-weight: 650;
    line-height: 1.38;
    margin-top: 14px;
  }
  .xruns-share-pills {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
  }
  .xruns-share-pill {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 18px 20px;
  }
  .xruns-share-pill-value {
    font-size: 37px;
    font-weight: 900;
    line-height: 1;
    margin-top: 8px;
  }
  .xruns-rankings-share-card .xruns-share-body {
    grid-template-columns: 0.94fr 1.06fr;
    padding-bottom: 92px;
  }
  .xruns-rankings-share-card .xruns-share-pills {
    grid-template-rows: repeat(2, minmax(0, 1fr));
    gap: 16px;
  }
  .xruns-rankings-share-card .xruns-share-stat {
    padding-top: 22px;
    padding-bottom: 18px;
  }
  .xruns-rankings-leader {
    display: flex;
    flex-direction: column;
    justify-content: center;
    min-height: 0;
  }
  .xruns-rankings-leader .xruns-share-pill-value {
    font-size: 50px;
    margin-top: 10px;
  }
  .xruns-rankings-leader-rating {
    color: #047857;
    font-size: 34px;
    font-weight: 900;
    line-height: 1;
    margin-top: 16px;
  }
  .xruns-matchup-share-card .xruns-share-band {
    min-height: 195px;
    padding-top: 26px;
    padding-bottom: 26px;
  }
  .xruns-matchup-share-card .xruns-share-title {
    font-size: 48px;
  }
  .xruns-matchup-share-card .xruns-share-meta {
    font-size: 19px;
  }
  .xruns-matchup-share-card .xruns-share-body {
    grid-template-columns: 0.92fr 1.08fr;
    gap: 24px;
    padding: 26px 42px 66px;
  }
  .xruns-matchup-share-card .xruns-share-stat {
    padding: 20px 24px;
    min-height: 0;
    overflow: hidden;
  }
  .xruns-matchup-share-card .xruns-share-label {
    font-size: 15px;
  }
  .xruns-share-bars {
    display: grid;
    gap: 16px;
  }
  .xruns-player-component-bars {
    margin-top: 22px;
    gap: 20px;
  }
  .xruns-share-bar-row {
    display: grid;
    grid-template-columns: 150px minmax(0, 1fr) 82px;
    gap: 14px;
    align-items: center;
  }
  .xruns-share-bar-name {
    color: #334155;
    font-size: 19px;
    font-weight: 800;
  }
  .xruns-share-bar-track {
    position: relative;
    height: 18px;
    background: #e8edf4;
    border-radius: 999px;
    overflow: hidden;
  }
  .xruns-share-bar-fill {
    height: 100%;
    min-width: 3px;
    border-radius: 999px;
  }
  .xruns-share-bar-value {
    font-size: 20px;
    font-weight: 900;
    text-align: right;
  }
  .xruns-player-component-row {
    grid-template-columns: 150px minmax(0, 1fr) 86px;
  }
  .xruns-player-pct-track {
    overflow: visible;
  }
  .xruns-player-pct-badge {
    position: absolute;
    top: 50%;
    transform: translate(-50%, -50%);
    min-width: 48px;
    height: 34px;
    padding: 0 8px;
    border-radius: 999px;
    border: 4px solid #ffffff;
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 17px;
    font-weight: 900;
    line-height: 1;
    box-shadow: 0 2px 4px rgba(15, 23, 42, 0.08);
  }
  .xruns-player-component-value {
    font-size: 19px;
    font-weight: 900;
    text-align: right;
  }
  .xruns-rank-list {
    display: grid;
    gap: 8px;
  }
  .xruns-rank-row {
    display: grid;
    grid-template-columns: 48px 40px minmax(0, 1fr) 86px;
    gap: 12px;
    align-items: center;
    min-height: 52px;
    padding: 8px 12px;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    background: #f8fafc;
  }
  .xruns-rank-num {
    color: #64748b;
    font-size: 18px;
    font-weight: 900;
  }
  .xruns-rank-logo {
    width: 34px;
    height: 34px;
    object-fit: contain;
  }
  .xruns-rank-team {
    color: #1e293b;
    font-size: 18px;
    font-weight: 850;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .xruns-rank-value {
    font-size: 22px;
    font-weight: 900;
    text-align: right;
  }
  .xruns-matchup-teams {
    display: flex;
    align-items: center;
    gap: 20px;
    flex: 0 0 auto;
  }
  .xruns-matchup-team {
    width: 155px;
    text-align: center;
  }
  .xruns-matchup-logo {
    width: 70px;
    height: 70px;
    object-fit: contain;
    display: block;
    margin: 0 auto 6px;
  }
  .xruns-matchup-name {
    color: #ffffff;
    font-size: 16px;
    font-weight: 900;
    line-height: 1.12;
  }
  .xruns-matchup-vs {
    color: rgba(255, 255, 255, 0.55);
    font-size: 22px;
    font-weight: 900;
  }
  .xruns-matchup-score {
    color: #1a365d;
    font-size: 62px;
    font-weight: 900;
    line-height: 1;
    margin-top: 16px;
  }
  .xruns-matchup-expected {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: 14px;
  }
  .xruns-matchup-expected .xruns-share-pill {
    padding: 14px 16px;
  }
  .xruns-matchup-exp-value {
    color: #1a365d;
    font-size: 36px;
    font-weight: 900;
    line-height: 1;
    margin-top: 7px;
  }
  .xruns-matchup-score-list {
    display: grid;
    gap: 8px;
    margin-top: 18px;
  }
  .xruns-matchup-score-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 76px;
    align-items: center;
    gap: 12px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    min-height: 46px;
    padding: 9px 14px;
  }
  .xruns-matchup-score-main {
    color: #1e293b;
    font-size: 24px;
    font-weight: 900;
    line-height: 1;
  }
  .xruns-matchup-score-prob {
    color: #64748b;
    font-size: 16px;
    font-weight: 900;
    text-align: right;
  }
  .xruns-matchup-pitchers {
    display: grid;
    gap: 18px;
    margin-top: 22px;
  }
  .xruns-matchup-pitcher {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    min-height: 104px;
    padding: 15px 18px;
    display: grid;
    grid-template-columns: 64px minmax(0, 1fr);
    gap: 18px;
    align-items: center;
  }
  .xruns-matchup-pitcher-photo {
    width: 64px;
    height: 64px;
    border-radius: 50%;
    background: #e2e8f0;
    border: 3px solid #f8fafc;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.10);
  }
  .xruns-matchup-pitcher-photo img {
    width: 100%;
    height: 100%;
    display: block;
    object-fit: cover;
    object-position: center center;
  }
  .xruns-matchup-pitcher-copy {
    min-width: 0;
  }
  .xruns-matchup-pitcher-name {
    color: #1e293b;
    font-size: 20px;
    font-weight: 900;
    line-height: 1.1;
    margin-top: 5px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .xruns-matchup-pitcher .xruns-share-sub {
    font-size: 16px;
    margin-top: 8px;
  }
  .xruns-matchup-prob {
    margin-top: 14px;
  }
  .xruns-matchup-prob-track {
    display: flex;
    height: 38px;
    overflow: hidden;
    border-radius: 999px;
    background: #e2e8f0;
    margin-top: 10px;
  }
  .xruns-matchup-prob-fill {
    display: flex;
    align-items: center;
    justify-content: center;
    color: #ffffff;
    font-size: 16px;
    font-weight: 900;
    min-width: 0;
  }
  .xruns-share-watermark {
    position: absolute;
    left: 42px;
    right: 42px;
    bottom: 18px;
    display: flex;
    justify-content: space-between;
    gap: 20px;
    color: #94a3b8;
    font-size: 14px;
    font-weight: 750;
    align-items: center;
  }
  .xruns-share-watermark span:first-child {
    max-width: 56%;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .xruns-share-watermark span:last-child {
    max-width: 38%;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    text-align: right;
  }
  .xruns-share-source-pill {
    padding: 6px 10px;
    border-radius: 999px;
    background: #f1f5f9;
    color: #64748b;
  }
  .xruns-team-rank-split {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 18px;
  }
  .xruns-team-mini-rank {
    padding-top: 2px;
  }
  .xruns-team-mini-rank-label {
    color: #64748b;
    font-size: 13px;
    font-weight: 850;
    letter-spacing: 0.09em;
    text-transform: uppercase;
  }
  .xruns-team-mini-rank-value {
    color: #1e293b;
    font-size: 34px;
    font-weight: 900;
    line-height: 1.15;
    margin-top: 8px;
  }
  .xruns-team-share-card .xruns-share-body {
    grid-template-columns: 0.88fr 1.12fr;
    gap: 28px;
    padding-bottom: 88px;
  }
  .xruns-team-rank-card {
    display: flex;
    flex-direction: column;
    min-height: 0;
    padding: 24px 26px 20px;
    overflow: hidden;
  }
  .xruns-team-rank-top {
    display: grid;
    grid-template-columns: 1.25fr 0.82fr 0.82fr;
    gap: 22px;
    align-items: start;
  }
  .xruns-team-season-rank-value {
    color: #1a365d;
    font-size: 82px;
    font-weight: 900;
    line-height: 0.95;
    margin-top: 16px;
  }
  .xruns-team-left-percentiles {
    border-top: 1px solid #e2e8f0;
    margin-top: 16px;
    padding-top: 14px;
    flex: 1 1 auto;
    min-height: 0;
  }
  .xruns-team-left-percentiles .xruns-share-label {
    font-size: 13px;
  }
  .xruns-team-left-percentile-bars {
    display: grid;
    gap: 12px;
    margin-top: 14px;
  }
  .xruns-team-pct-row {
    display: grid;
    grid-template-columns: 94px minmax(0, 1fr) 64px;
    gap: 12px;
    align-items: center;
  }
  .xruns-team-pct-name {
    color: #334155;
    font-size: 16px;
    font-weight: 850;
  }
  .xruns-team-pct-track {
    position: relative;
    height: 18px;
    border-radius: 999px;
    background: #e8edf4;
  }
  .xruns-team-pct-fill {
    height: 100%;
    min-width: 3px;
    border-radius: 999px;
  }
  .xruns-team-pct-badge {
    position: absolute;
    top: 50%;
    transform: translate(-50%, -50%);
    min-width: 44px;
    height: 28px;
    padding: 0 8px;
    border-radius: 999px;
    border: 3px solid #ffffff;
    color: #ffffff;
    font-size: 16px;
    font-weight: 900;
    display: flex;
    align-items: center;
    justify-content: center;
    line-height: 1;
  }
  .xruns-team-pct-value {
    font-size: 17px;
    font-weight: 900;
    text-align: right;
  }
  .xruns-team-leader-stack {
    display: grid;
    grid-template-rows: repeat(2, minmax(0, 1fr));
    gap: 16px;
    min-height: 0;
  }
  .xruns-team-leader-card {
    display: grid;
    grid-template-columns: 90px minmax(0, 1fr) 142px;
    gap: 20px;
    align-items: center;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 22px 24px;
    min-height: 0;
    overflow: hidden;
  }
  .xruns-team-player-photo {
    width: 90px;
    height: 90px;
    border-radius: 50%;
    overflow: hidden;
    background: #e2e8f0;
    border: 4px solid #ffffff;
    box-shadow: 0 3px 10px rgba(15, 23, 42, 0.10);
  }
  .xruns-team-player-photo img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center center;
    display: block;
  }
  .xruns-team-player-name {
    color: #1e293b;
    font-size: 25px;
    font-weight: 900;
    line-height: 1.08;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .xruns-team-player-value {
    font-size: 16px;
    font-weight: 900;
  }
  .xruns-team-player-values {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
    margin-top: 14px;
  }
  .xruns-team-player-value-chip {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 9px 10px;
  }
  .xruns-team-player-value-label {
    color: #64748b;
    font-size: 11px;
    font-weight: 850;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .xruns-team-player-value-number {
    font-size: 17px;
    font-weight: 900;
    margin-top: 4px;
    line-height: 1;
  }
  .xruns-team-player-primary {
    text-align: right;
    color: #1e293b;
  }
  .xruns-team-player-primary-number {
    font-size: 46px;
    font-weight: 900;
    line-height: 0.95;
  }
  .xruns-team-player-primary-unit {
    color: #64748b;
    font-size: 12px;
    font-weight: 850;
    letter-spacing: 0.07em;
    line-height: 1.22;
    margin-top: 8px;
    text-transform: uppercase;
  }
'
}

xruns_ordinal <- function(x) {
  if (is.na(x)) return("N/A")
  suffix <- if (x %% 100 %in% 11:13) {
    "th"
  } else {
    switch(as.character(x %% 10), "1" = "st", "2" = "nd", "3" = "rd", "th")
  }
  paste0(x, suffix)
}

xruns_player_component_rows <- function(components) {
  rows <- Filter(function(x) !is.null(x$pct) && !is.na(x$pct), components)
  lapply(rows, function(x) {
    color <- if (x$pct < 40) "#b91c1c" else "#047857"
    value_color <- xruns_rating_color(x$value)
    tags$div(
      class = "xruns-share-bar-row xruns-player-component-row",
      tags$div(class = "xruns-share-bar-name", x$label),
      tags$div(
        class = "xruns-share-bar-track xruns-player-pct-track",
        tags$div(
          class = "xruns-share-bar-fill",
          style = sprintf("width: %d%%; background: %s;", x$pct, color)
        ),
        tags$div(
          class = "xruns-player-pct-badge",
          style = sprintf("left: %d%%; background: %s;", x$pct, color),
          xruns_ordinal(x$pct)
        )
      ),
      tags$div(
        class = "xruns-player-component-value",
        style = sprintf("color: %s;", value_color),
        xruns_fmt_signed(x$value)
      )
    )
  })
}

xruns_team_percentile_rows <- function(components) {
  rows <- Filter(function(x) !is.null(x$pct) && !is.na(x$pct), components)
  lapply(rows, function(x) {
    color <- if (x$pct < 40) "#b91c1c" else "#047857"
    tags$div(
      class = "xruns-team-pct-row",
      tags$div(class = "xruns-team-pct-name", x$label),
      tags$div(
        class = "xruns-team-pct-track",
        tags$div(
          class = "xruns-team-pct-fill",
          style = sprintf("width: %d%%; background: %s;", x$pct, color)
        ),
        tags$div(
          class = "xruns-team-pct-badge",
          style = sprintf("left: %d%%; background: %s;", x$pct, color),
          xruns_ordinal(x$pct)
        )
      ),
      tags$div(
        class = "xruns-team-pct-value",
        style = sprintf("color: %s;", xruns_rating_color(x$value)),
        xruns_fmt_signed(x$value)
      )
    )
  })
}

xruns_card_data_label <- function(season, data_label = NULL) {
  label <- xruns_safe_text(data_label, fallback = "")
  if (nzchar(label)) label else paste0(season, " season data")
}

xruns_card_meta <- function(description, season, data_label = NULL) {
  paste(description, paste0("Data: ", xruns_card_data_label(season, data_label)), sep = " | ")
}

xruns_watermark <- function(season, base_url, data_label = NULL, card_type = "rankings") {
  tags$div(
    class = "xruns-share-watermark",
    tags$span(class = "xruns-share-source-pill", paste0("xRuns · ", season, " MLB ratings")),
    tags$span(xruns_display_base_url(base_url, card_type))
  )
}

xruns_player_share_card <- function(row, season, base_url, data_label = NULL) {
  role <- xruns_safe_text(row$Role)
  role_display <- switch(role,
    "Player" = "Two-Way Player",
    "Hitter" = "Hitter",
    "Pitcher" = "Pitcher",
    role
  )
  overall <- as.numeric(row$Overall[1])
  pct_value <- if (role %in% c("Hitter", "Player")) {
    pct_rank(overall, pp_hitter_overalls)
  } else {
    pct_rank(overall, pp_pitcher_overalls)
  }
  component_rows <- list(
    list(label = "Overall", value = overall, pct = pct_value),
    list(label = "Hitting", value = row$Hitting[1], pct = pct_rank(row$Hitting[1], pp_hitter_hitting)),
    list(label = "Baserunning", value = row$Baserunning[1], pct = pct_rank(row$Baserunning[1], pp_hitter_br)),
    list(label = "Pitching", value = row$Pitching[1], pct = pct_rank(row$Pitching[1], pp_pitcher_pitching)),
    list(label = "Fielding", value = row$Fielding[1], pct = pct_rank(row$Fielding[1], pp_fielding_all))
  )
  team_abbrev <- xruns_safe_text(row$Team)

  tags$div(
    class = "xruns-share-card xruns-player-share-card",
    tags$div(
      class = "xruns-share-band",
      tags$div(
        tags$div(class = "xruns-share-brand", "xRuns"),
        tags$div(class = "xruns-share-kicker", "Player Performance Card"),
        tags$div(class = "xruns-share-title", xruns_safe_text(row$Player)),
        tags$div(
          class = "xruns-share-meta",
          paste(paste(team_abbrev, role_display, sep = " | "), paste0("Data: ", xruns_card_data_label(season, data_label)), sep = " | ")
        )
      ),
      tags$div(
        class = "xruns-share-headshot",
        tags$img(src = xruns_headshot_url(row$player_id[1], width = 420), crossorigin = "anonymous", alt = xruns_safe_text(row$Player))
      )
    ),
    tags$div(
      class = "xruns-share-body",
      tags$div(
        class = "xruns-share-stat",
        tags$div(class = "xruns-share-label", "Overall Rating"),
        tags$div(
          class = "xruns-share-big",
          style = sprintf("color: %s;", xruns_rating_color(overall)),
          xruns_fmt_signed(overall)
        ),
        tags$div(class = "xruns-share-sub", "Average Run Value Added per 9 Innings")
      ),
      tags$div(
        class = "xruns-share-stat",
        tags$div(class = "xruns-share-label", "Component Percentiles"),
        tags$div(
          class = "xruns-share-bars xruns-player-component-bars",
          tagList(xruns_player_component_rows(component_rows))
        )
      )
    ),
    xruns_watermark(season, base_url, data_label, card_type = "player")
  )
}

xruns_team_share_card <- function(row, ranks, season, base_url, data_label = NULL,
                                  team_pool = NULL, recent_ranks = list(),
                                  featured_players = list()) {
  logo_url <- xruns_safe_text(row$team_logo_espn, fallback = "")
  team_pool <- team_pool %||% row

  mini_rank <- function(label, value) {
    tags$div(
      class = "xruns-team-mini-rank",
      tags$div(class = "xruns-team-mini-rank-label", label),
      tags$div(class = "xruns-team-mini-rank-value", if (is.na(value)) "N/A" else paste0("#", value))
    )
  }

  team_components <- list(
    list(label = "Offense", value = row$off_rating[1], pct = pct_rank(row$off_rating[1], team_pool$off_rating)),
    list(label = "Pitching", value = row$def_pitching[1], pct = pct_rank(row$def_pitching[1], team_pool$def_pitching)),
    list(label = "Defense", value = row$def_rating[1], pct = pct_rank(row$def_rating[1], team_pool$def_rating))
  )

  player_mini <- function(title, player_row, primary_col, primary_label, stat_specs = list()) {
    if (is.null(player_row) || nrow(player_row) == 0) {
      return(tags$div(
        class = "xruns-team-leader-card",
        tags$div(class = "xruns-team-player-photo"),
        tags$div(
          tags$div(class = "xruns-share-label", title),
          tags$div(class = "xruns-team-player-name", "Unavailable")
        )
      ))
    }
    primary_value <- player_row[[primary_col]][1]
    stat_chip <- function(label, value) {
      tags$div(
        class = "xruns-team-player-value-chip",
        tags$div(class = "xruns-team-player-value-label", label),
        tags$div(
          class = "xruns-team-player-value-number",
          style = sprintf("color: %s;", xruns_rating_color(value)),
          xruns_fmt_signed(value)
        )
      )
    }
    stat_block <- if (length(stat_specs) > 0) {
      tags$div(
        class = "xruns-team-player-values",
        tagList(lapply(stat_specs, function(x) stat_chip(x[[1]], player_row[[x[[2]]]][1])))
      )
    } else {
      NULL
    }
    tags$div(
      class = "xruns-team-leader-card",
      tags$div(
        class = "xruns-team-player-photo",
        tags$img(
          src = xruns_headshot_url(player_row$player_id[1], width = 220),
          crossorigin = "anonymous",
          alt = xruns_safe_text(player_row$Player)
        )
      ),
      tags$div(
        tags$div(class = "xruns-share-label", title),
        tags$div(class = "xruns-team-player-name", xruns_safe_text(player_row$Player)),
        stat_block
      ),
      tags$div(
        class = "xruns-team-player-primary",
        tags$div(
          class = "xruns-team-player-primary-number",
          style = sprintf("color: %s;", xruns_rating_color(primary_value)),
          xruns_fmt_signed(primary_value)
        ),
        tags$div(class = "xruns-team-player-primary-unit", primary_label)
      )
    )
  }

  tags$div(
    class = "xruns-share-card xruns-team-share-card",
    tags$div(
      class = "xruns-share-band",
      tags$div(
        tags$div(class = "xruns-share-brand", "xRuns"),
        tags$div(class = "xruns-share-kicker", "Team Performance Card"),
        tags$div(class = "xruns-share-title", xruns_safe_text(row$team_name)),
        tags$div(
          class = "xruns-share-meta",
          xruns_card_meta(paste0("Overall rank #", ranks$overall, " of 30"), season, data_label)
        )
      ),
      if (nzchar(logo_url)) {
        tags$img(class = "xruns-share-logo", src = logo_url, crossorigin = "anonymous", alt = xruns_safe_text(row$abbrev))
      }
    ),
    tags$div(
      class = "xruns-share-body",
      tags$div(
        class = "xruns-share-stat xruns-team-rank-card",
        tags$div(
          class = "xruns-team-rank-top",
          tags$div(
            tags$div(class = "xruns-share-label", "Season Rank"),
            tags$div(class = "xruns-team-season-rank-value", paste0("#", ranks$overall))
          ),
          mini_rank("Last 30d", recent_ranks$rank_30d %||% NA_integer_),
          mini_rank("Last 7d", recent_ranks$rank_7d %||% NA_integer_)
        ),
        tags$div(
          class = "xruns-team-left-percentiles",
          tags$div(class = "xruns-share-label", "Team Percentiles"),
          tags$div(
            class = "xruns-share-bars xruns-player-component-bars xruns-team-left-percentile-bars",
            tagList(xruns_team_percentile_rows(team_components))
          )
        )
      ),
      tags$div(
        class = "xruns-team-leader-stack",
        player_mini(
          "Best Hitter",
          featured_players$hitter,
          "Overall",
          "run value / 9 innings",
          list(c("Hit", "Hitting"), c("Field", "Fielding"), c("Run", "Baserunning"))
        ),
        player_mini(
          "Best Pitcher",
          featured_players$pitcher,
          "Pitching",
          "run value / 9 innings"
        )
      )
    ),
    xruns_watermark(season, base_url, data_label, card_type = "team")
  )
}

xruns_rankings_share_card <- function(tt, season, base_url, data_label = NULL) {
  top <- tt %>%
    dplyr::arrange(dplyr::desc(overall)) %>%
    dplyr::slice_head(n = 5)
  rank_rows <- lapply(seq_len(nrow(top)), function(i) {
    row <- top[i, ]
    logo_url <- xruns_safe_text(row$team_logo_espn, fallback = "")
    meta_name <- TEAM_META$team_name[match(row$abbrev[1], TEAM_META$abbrev)]
    team_name <- if (length(meta_name) == 1 && !is.na(meta_name)) meta_name else xruns_safe_text(row$team_name)
    tags$div(
      class = "xruns-rank-row",
      tags$div(class = "xruns-rank-num", paste0("#", i)),
      if (nzchar(logo_url)) {
        tags$img(class = "xruns-rank-logo", src = logo_url, crossorigin = "anonymous", alt = xruns_safe_text(row$abbrev))
      } else {
        tags$div(class = "xruns-rank-logo")
      },
      tags$div(class = "xruns-rank-team", team_name),
      tags$div(
        class = "xruns-rank-value",
        style = sprintf("color: %s;", xruns_rating_color(row$overall[1])),
        xruns_fmt_signed(row$overall[1])
      )
    )
  })
  leader <- function(label, metric) {
    row <- tt[which.max(tt[[metric]]), ]
    tags$div(
      class = "xruns-share-pill xruns-rankings-leader",
      tags$div(class = "xruns-share-label", label),
      tags$div(class = "xruns-share-pill-value", style = "color: #1a365d;", xruns_safe_text(row$abbrev)),
      tags$div(class = "xruns-rankings-leader-rating", xruns_fmt_signed(row[[metric]][1]))
    )
  }

  tags$div(
    class = "xruns-share-card xruns-rankings-share-card",
    tags$div(
      class = "xruns-share-band",
      tags$div(
        tags$div(class = "xruns-share-brand", "xRuns"),
        tags$div(class = "xruns-share-kicker", "MLB Performance Rankings"),
        tags$div(class = "xruns-share-title", paste0(season, " Team Rankings")),
        tags$div(
          class = "xruns-share-meta",
          xruns_card_meta("Runs per game above a league-average team", season, data_label)
        )
      )
    ),
    tags$div(
      class = "xruns-share-body",
      tags$div(
        class = "xruns-share-stat",
        tags$div(class = "xruns-share-label", "Top Overall"),
        tags$div(class = "xruns-rank-list", tagList(rank_rows))
      ),
      tags$div(
        class = "xruns-share-pills",
        leader("Best Overall", "overall"),
        leader("Best Offense", "off_rating"),
        leader("Best Pitching", "def_pitching"),
        leader("Best Fielding", "def_fld")
      )
    ),
    xruns_watermark(season, base_url, data_label, card_type = "rankings")
  )
}

xruns_matchup_share_card <- function(res, season, base_url, data_label = NULL) {
  ta <- res$ta
  tb <- res$tb
  col_a <- xruns_team_color(ta)
  col_b <- xruns_team_color(tb, "#334155")
  logo_a <- xruns_safe_text(ta$team_logo_espn, fallback = "")
  logo_b <- xruns_safe_text(tb$team_logo_espn, fallback = "")
  pct_a <- round(res$p_a_wins * 100, 1)
  pct_b <- round(100 - pct_a, 1)

  team_block <- function(row, logo_url) {
    tags$div(
      class = "xruns-matchup-team",
      if (nzchar(logo_url)) {
        tags$img(class = "xruns-matchup-logo", src = logo_url, crossorigin = "anonymous", alt = xruns_safe_text(row$abbrev))
      },
      tags$div(class = "xruns-matchup-name", xruns_safe_text(row$team_name))
    )
  }

  expected_pill <- function(abbrev, value, color) {
    tags$div(
      class = "xruns-share-pill",
      tags$div(class = "xruns-share-label", paste(abbrev, "Expected Runs")),
      tags$div(class = "xruns-matchup-exp-value", style = sprintf("color: %s;", color), sprintf("%.2f", value))
    )
  }

  score_rows <- lapply(seq_len(min(3, nrow(res$top_scores))), function(i) {
    score <- res$top_scores[i, ]
    tags$div(
      class = "xruns-matchup-score-row",
      tags$div(
        class = "xruns-matchup-score-main",
        sprintf(
          "%s %d - %d %s",
          xruns_safe_text(ta$abbrev),
          score$runs_a,
          score$runs_b,
          xruns_safe_text(tb$abbrev)
        )
      ),
      tags$div(class = "xruns-matchup-score-prob", sprintf("%.1f%%", score$prob * 100))
    )
  })

  pitcher_row <- function(abbrev, pitcher, color) {
    tags$div(
      class = "xruns-matchup-pitcher",
      style = sprintf("border-left: 5px solid %s;", color),
      tags$div(
        class = "xruns-matchup-pitcher-photo",
        tags$img(
          src = xruns_headshot_url(pitcher$player_id[1], width = 220),
          crossorigin = "anonymous",
          alt = xruns_safe_text(pitcher$player)
        )
      ),
      tags$div(
        class = "xruns-matchup-pitcher-copy",
        tags$div(class = "xruns-share-label", paste(abbrev, "Starter")),
        tags$div(class = "xruns-matchup-pitcher-name", xruns_safe_text(pitcher$player)),
        tags$div(class = "xruns-share-sub", sprintf("%+.2f pitching runs/game", pitcher$final_rating[1] * PA_PER_GAME))
      )
    )
  }

  tags$div(
    class = "xruns-share-card xruns-matchup-share-card",
    tags$div(
      class = "xruns-share-band",
      tags$div(
        tags$div(class = "xruns-share-brand", "xRuns"),
        tags$div(class = "xruns-share-kicker", "Matchup Simulator"),
        tags$div(class = "xruns-share-title", paste0(xruns_safe_text(ta$abbrev), " vs ", xruns_safe_text(tb$abbrev))),
        tags$div(
          class = "xruns-share-meta",
          xruns_card_meta(paste0(season, " projected matchup"), season, data_label)
        )
      ),
      tags$div(
        class = "xruns-matchup-teams",
        team_block(ta, logo_a),
        tags$div(class = "xruns-matchup-vs", "VS"),
        team_block(tb, logo_b)
      )
    ),
    tags$div(
      class = "xruns-share-body",
      tags$div(
        class = "xruns-share-stat",
        tags$div(class = "xruns-share-label", "Expected Runs"),
        tags$div(
          class = "xruns-matchup-expected",
          expected_pill(xruns_safe_text(ta$abbrev), res$exp_a, col_a),
          expected_pill(xruns_safe_text(tb$abbrev), res$exp_b, col_b)
        ),
        tags$div(
          class = "xruns-matchup-score-list",
          tags$div(class = "xruns-share-label", "Most Likely Final Scores"),
          tagList(score_rows)
        )
      ),
      tags$div(
        class = "xruns-share-stat",
        tags$div(class = "xruns-share-label", "Win Probability"),
        tags$div(
          class = "xruns-matchup-prob",
          tags$div(
            class = "xruns-matchup-prob-track",
            tags$div(
              class = "xruns-matchup-prob-fill",
              style = sprintf("width: %.1f%%; background: %s;", pct_a, col_a),
              paste0(xruns_safe_text(ta$abbrev), " ", pct_a, "%")
            ),
            tags$div(
              class = "xruns-matchup-prob-fill",
              style = sprintf("width: %.1f%%; background: %s;", pct_b, col_b),
              paste0(xruns_safe_text(tb$abbrev), " ", pct_b, "%")
            )
          )
        ),
        tags$div(
          class = "xruns-matchup-pitchers",
          pitcher_row(xruns_safe_text(ta$abbrev), res$sa, col_a),
          pitcher_row(xruns_safe_text(tb$abbrev), res$sb, col_b)
        )
      )
    ),
    xruns_watermark(season, base_url, data_label, card_type = "matchup")
  )
}

xruns_apply_rounded_png_mask <- function(file, radius_px = NULL) {
  if (!requireNamespace("magick", quietly = TRUE)) {
    return(invisible(file))
  }

  img <- magick::image_read(file)
  info <- magick::image_info(img)[1, ]
  width <- info$width
  height <- info$height
  radius_px <- radius_px %||% (min(width / 1200, height / 675) * 18)
  radius_ratio <- radius_px / min(width, height)

  mask <- magick::image_blank(width, height, color = "transparent")
  mask <- magick::image_draw(mask)
  grid::grid.roundrect(
    x = grid::unit(0.5, "npc"),
    y = grid::unit(0.5, "npc"),
    width = grid::unit(1, "npc"),
    height = grid::unit(1, "npc"),
    r = grid::unit(radius_ratio, "snpc"),
    gp = grid::gpar(fill = "white", col = NA)
  )
  grDevices::dev.off()

  img <- magick::image_composite(img, mask, operator = "CopyOpacity")
  magick::image_write(img, path = file, format = "png")
  invisible(file)
}

# ---- UI ----------------------------------------------------------------------
ui <- page_navbar(
  id    = "main_nav",
  title = tags$a(
    href    = "javascript:void(0)",
    onclick = "(function(){
      var links = document.querySelectorAll('.navbar-nav .nav-link');
      for (var i = 0; i < links.length; i++) {
        if (links[i].textContent.trim().indexOf('Team Rankings') >= 0) {
          links[i].click(); break;
        }
      }
    })()",
    style   = "text-decoration:none; color:inherit; cursor:pointer;",
    tags$span(class = "xruns-brand-full",
              "xRuns",
              tags$span(class = "brand-sub", "an MLB Ratings System")
    )
  ),
  window_title = "xRuns: an MLB Ratings System",
  theme        = app_theme,
  fillable     = FALSE,
  navbar_options = navbar_options(collapsible = TRUE),
  
  footer = tags$div(
    class = "xruns-data-footer",
    "Data sourced from",
    tags$a(
      href   = "https://baseballsavant.mlb.com/",
      target = "_blank",
      rel    = "noopener noreferrer",
      tags$img(
        src   = "https://baseballsavant.mlb.com/favicon.ico",
        alt   = "Baseball Savant",
        style = "height:16px; width:16px; vertical-align:middle; margin-right:4px;"
      ),
      "Baseball Savant"
    )
  ),
  
  header = tagList(
    tags$head(
      tags$meta(name = "viewport",
                content = "width=device-width, initial-scale=1, shrink-to-fit=no"),
      # Font Awesome for tab icons
      tags$link(
        rel  = "stylesheet",
        href = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css"
      ),
      tags$link(
        rel = "stylesheet",
        href = "https://fonts.googleapis.com/css2?family=Inter:wght@500;650;750;800;850;900&display=swap"
      ),
      tags$script(src = "https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js"),
      tags$style(custom_css),
      tags$style(product_css),
      tags$style(HTML(xruns_share_card_css())),
      tags$style(HTML("
        #xruns-share-stage {
          position: fixed;
          left: -10000px;
          top: 0;
          width: 1300px;
          height: 760px;
          padding: 28px;
          background: transparent;
          pointer-events: none;
          z-index: -1;
        }
      ")),
      # JS: handle year-change reset of time period chips back to Season
      tags$script(HTML("
        (function() {
          function xrunsPublicBase() {
            var loc = window.location;
            var host = loc.hostname || '';
            var path = loc.pathname || '/';
            var isLocal = /^(localhost|127\\.0\\.0\\.1|0\\.0\\.0\\.0)$/.test(host);
            var isConnectCloud = /connect\\.posit\\.cloud$/i.test(host);

            if (isConnectCloud) return loc.origin + '/';
            if (!isLocal && /^\\/_w_[A-Za-z0-9]+/.test(path)) return loc.origin + '/';
            return loc.origin + path;
          }

          function xrunsNotifyRoute() {
            if (!window.Shiny) return;
            if (!window.location.hash) return;
            Shiny.setInputValue('xruns_route_hash', window.location.hash || '', {priority: 'event'});
          }

          Shiny.addCustomMessageHandler('xrunsRoute', function(msg) {
            var route = msg && msg.route ? String(msg.route) : '#/team-rankings';
            if (route.charAt(0) !== '#') route = '#/' + route.replace(/^\\/+/, '');
            var target = xrunsPublicBase().replace(/[#?].*$/, '') + route;
            if (window.location.href !== target) {
              window.history.replaceState(null, document.title, target);
            }
          });

          document.addEventListener('shiny:connected', xrunsNotifyRoute);
          window.addEventListener('hashchange', xrunsNotifyRoute);
          window.addEventListener('popstate', xrunsNotifyRoute);
        })();

        function xrunsDownloadClientShareCard(cardId) {
          var node = document.getElementById(cardId);
          if (!node) return;
          var card = node.querySelector('.xruns-share-card');
          if (!card || !window.html2canvas) return;
          var filename = node.getAttribute('data-filename') ||
            (node.firstElementChild && node.firstElementChild.getAttribute('data-filename')) ||
            'xruns-share-card.png';
          html2canvas(card, {
            backgroundColor: null,
            scale: 2,
            useCORS: true,
            allowTaint: false,
            logging: false,
            width: 1200,
            height: 675,
            windowWidth: 1300,
            windowHeight: 760
          }).then(function(canvas) {
            canvas.toBlob(function(blob) {
              if (!blob) return;
              var url = URL.createObjectURL(blob);
              var a = document.createElement('a');
              a.href = url;
              a.download = filename;
              document.body.appendChild(a);
              a.click();
              a.remove();
              setTimeout(function() { URL.revokeObjectURL(url); }, 1000);
            }, 'image/png');
          }).catch(function(err) {
            console.error('Unable to create share PNG', err);
            window.alert('The share image could not be created in this browser. Please refresh and try again.');
          });
        }
        Shiny.addCustomMessageHandler('resetTimePeriod', function(msg) {
          Shiny.setInputValue('time_period', 'season', {priority: 'event'});
          document.querySelectorAll('.xruns-period-chip').forEach(function(el) {
            el.classList.remove('active');
            if (el.textContent.trim() === 'Season') el.classList.add('active');
          });
        });
        // Auto-collapse navbar dropdown when a nav link is clicked on mobile
        document.addEventListener('click', function(e) {
          var link = e.target.closest('.navbar-nav .nav-link');
          if (link) {
            var collapse = document.querySelector('.navbar-collapse');
            if (collapse && collapse.classList.contains('show')) {
              collapse.classList.remove('show');
            }
          }
        });
        document.addEventListener('DOMContentLoaded', function() {
          var style = document.createElement('style');
          style.textContent = [
            '.navbar .navbar-toggler,.navbar .navbar-toggle{display:none!important;background:transparent!important;background-color:transparent!important;border:0!important;border-radius:0!important;box-shadow:none!important;outline:0!important;}',
            '@media (max-width:991px){.navbar .navbar-toggler,.navbar .navbar-toggle{display:flex!important;width:52px!important;height:52px!important;min-width:52px!important;cursor:pointer!important;}}',
            '@media (max-width:991px){.navbar{overflow:visible!important;z-index:10020!important;}.navbar>.container,.navbar>.container-fluid{position:relative;overflow:visible!important;}.navbar-collapse{position:absolute!important;top:calc(100% + 10px)!important;left:12px!important;right:12px!important;width:auto!important;max-width:680px;margin-left:auto!important;margin-right:auto!important;display:block!important;visibility:visible!important;height:auto!important;max-height:min(72vh,440px);overflow-y:auto!important;overflow-x:hidden!important;padding:10px!important;z-index:10030!important;border:1px solid rgba(255,255,255,0.13);border-radius:14px;background:rgba(21,25,34,0.86)!important;-webkit-backdrop-filter:blur(18px) saturate(140%);backdrop-filter:blur(18px) saturate(140%);box-shadow:0 24px 60px rgba(15,23,42,0.34);opacity:0!important;pointer-events:none!important;transform:translateY(-10px) scale(.98)!important;transform-origin:top center;transition:opacity 180ms ease,transform 180ms ease;}}',
            '@media (max-width:991px){.navbar-collapse.show,.navbar-collapse.collapsing{opacity:1!important;pointer-events:auto!important;transform:translateY(0) scale(1)!important;}}',
            '@media (max-width:991px){.navbar-nav{float:none!important;display:flex!important;flex-direction:column!important;gap:4px!important;margin:0!important;width:100%;}.navbar-nav .nav-link{display:block;width:100%;margin:0;padding:12px 14px!important;border-radius:10px;font-size:1rem!important;line-height:1.15;color:rgba(255,255,255,.72)!important;transition:background 140ms ease,color 140ms ease,transform 140ms ease;}.navbar-nav .nav-link:hover{background:rgba(255,255,255,.10)!important;transform:translateX(2px);}.navbar-nav .nav-link.active{color:#fff!important;background:rgba(255,255,255,.15)!important;box-shadow:inset 0 -2px 0 rgba(255,255,255,.72);}}',
            '@media (max-width:520px){.navbar-collapse{left:2px!important;right:2px!important;border-radius:12px;}.navbar-nav .nav-link{padding:11px 12px!important;font-size:.95rem!important;}}',
            '.navbar .navbar-toggler-icon,.navbar .navbar-toggle .icon-bar{display:none!important;}',
            '.navbar .navbar-toggler:before,.navbar .navbar-toggle:before{content:\"\";display:block;width:27px;height:2px;background:#fff;border-radius:2px;box-shadow:0 8px 0 #fff,0 16px 0 #fff;}'
          ].join('\\n');
          document.head.appendChild(style);
        });
      "))
    ),
    tags$div(
      id = "xruns-share-stage",
      uiOutput("client_rankings_share_card"),
      uiOutput("client_team_share_card"),
      uiOutput("client_player_share_card"),
      uiOutput("client_matchup_share_card")
    )
  ),
  
  # ---- Tab: Team Rankings ----
  nav_panel(
    title = "Team Rankings",
    tags$main(
      class = "xruns-page",

      # ---- Hero / landing block ----
      tags$div(
        class = "xruns-hero",
        tags$div(
          class = "xruns-hero-main",
          tags$div(class = "xruns-kicker", "MLB Performance Ratings"),
          tags$p(
            class = "xruns-hero-copy",
            "xRuns rates every MLB team and player using expected performance stats — ",
            "what they should have scored, not just what they did. ",
            "No luck. No noise. Just signal."
          )
        ),
        tags$div(
          class = "xruns-hero-panel",
          tags$div(class = "xruns-panel-label", "How ratings work"),
          tags$p(
            style = "font-size:12.5px; line-height:1.6; color:#485264; margin:0;",
            tags$b("Overall"), " = runs/game above a league-average team. ",
            tags$b("Positive is good"), "; 0 = exactly average. ",
            tags$b("Offense"), " = hitting + baserunning. ",
            tags$b("Defense"), " = pitching + fielding."
          ),
          tags$p(
            style = "font-size:12px; line-height:1.55; color:#64748b; margin:4px 0 0;",
            "Built from Statcast expected stats (xwOBA, xERA) — ",
            "a hot streak or lucky week won't inflate a rating."
          )
        )
      ),
      
      # Heading row with time-period filter (only shown for years with snapshot history)
      tags$div(
        class = "xruns-heading-row",
        tags$div(class = "xruns-heading-text",
                 textOutput("team_heading", inline = TRUE)
        ),
        uiOutput("time_period_ui"),
        tags$div(class = "xruns-season-chip", season_picker_inline("team")),
        tags$div(
          class = "xruns-share-inline",
          actionButton(
            "download_rankings_card_client",
            tagList(tags$i(class = "fa-solid fa-image"), "Share the Rankings"),
            class = "xruns-share-btn",
            onclick = "xrunsDownloadClientShareCard('client_rankings_share_card')"
          )
        )
      ),
      
      # Fallback / window note (shown when time period filter uses graceful fallback)
      uiOutput("window_note_ui"),
      
      uiOutput("kpi_row"),
      
      tags$div(class = "tab-body", DTOutput("team_table", height = "auto")),
      
      tags$div(class = "section-heading",
               "Offense vs. Defense",
               tags$span(class = "section-subheading", "runs/game above average")
      ),
      tags$div(class = "tab-body", plotlyOutput("team_scatter", height = "620px"))
    )
  ),
  
  # ---- Tab: Matchup Simulator ----
  nav_panel(
    title = "Matchup Simulator",
    
    tags$head(tags$style(HTML("
      .mp-vs { font-size:1.5rem; font-weight:700; color:#cbd5e0;
               display:flex; align-items:center; justify-content:center; height:100%; }
      .mp-team-logo-lg { width:60px; height:60px; object-fit:contain; }
      .mp-team-logo-sm { width:32px; height:32px; object-fit:contain; flex-shrink:0; }
      .mp-placeholder-lg { width:60px; height:60px; background:#f1f5f9; border-radius:8px;
               display:flex; align-items:center; justify-content:center;
               font-weight:700; color:#1a365d; font-size:13px; }
      .mp-placeholder-sm { width:32px; height:32px; background:#f1f5f9; border-radius:6px;
               display:flex; align-items:center; justify-content:center;
               font-weight:700; color:#1a365d; font-size:10px; flex-shrink:0; }
      .mp-header-row { display:flex; align-items:center; justify-content:space-around;
               padding:18px 24px 14px; }
      .mp-team-block { display:flex; flex-direction:column; align-items:center;
               gap:6px; flex:1; }
      .mp-team-name-lg { font-size:1.05rem; font-weight:700; color:#1e293b;
               text-align:center; }
      .mp-sub-label { font-size:11px; font-weight:600; text-transform:uppercase;
               letter-spacing:.05em; color:#94a3b8; margin-bottom:2px; }
      .mp-exp-runs { font-size:2.2rem; font-weight:800; color:#1a365d; line-height:1; }
      .mp-win-bar { height:30px; border-radius:15px; overflow:hidden; display:flex; }
      .mp-win-seg { display:flex; align-items:center; justify-content:center;
               font-size:12px; font-weight:600; color:#fff; transition:width .4s; }
      .mp-score-row { display:flex; align-items:center; justify-content:space-between;
               padding:9px 14px; border-radius:8px; margin-bottom:6px;
               background:#f8fafc; border:1px solid #e2e8f0; }
      .mp-score-label { font-size:1rem; font-weight:700; color:#1e293b; }
      .mp-score-pct   { font-size:13px; font-weight:700; color:#047857; }
      .mp-pitcher-chip { display:flex; align-items:center; gap:10px;
               background:#f8fafc; border:1px solid #e2e8f0;
               border-radius:8px; padding:10px 14px; }
      .mp-chip-text-main { font-weight:700; font-size:13px; color:#1e293b; }
      .mp-chip-text-sub  { font-size:11.5px; color:#64748b; margin-top:1px; }
      /* Pitcher selectize — clear selection feedback */
      #mp_starter_a .selectize-input,
      #mp_starter_b .selectize-input {
        font-size: 13px;
        font-weight: 720;
        color: #151922;
        background: #ffffff;
        border: 0;
        border-radius: 6px;
        padding: 10px 42px 10px 12px;
        min-height: 45px;
        box-shadow: none;
        outline: none;
        cursor: pointer;
      }
      #mp_starter_a .selectize-input.focus,
      #mp_starter_b .selectize-input.focus,
      #mp_starter_a .selectize-input.input-active,
      #mp_starter_b .selectize-input.input-active {
        box-shadow: none;
      }
      #mp_starter_a .selectize-input:hover,
      #mp_starter_b .selectize-input:hover {
        background: #ffffff;
      }
      #mp_starter_a .selectize-dropdown,
      #mp_starter_b .selectize-dropdown {
        font-size: 13px;
        border: 1px solid #cfd6e2;
        border-radius: 6px;
        box-shadow: 0 12px 26px rgba(21,25,34,0.10);
      }
      #mp_starter_a .selectize-dropdown-content .option,
      #mp_starter_b .selectize-dropdown-content .option {
        padding: 8px 12px;
        border-bottom: 1px solid #f1f5f9;
      }
      #mp_starter_a .selectize-dropdown-content .option.active,
      #mp_starter_b .selectize-dropdown-content .option.active {
        background: #f2f4f8;
        color: #151922;
        font-weight: 700;
      }
      /* Simulation card */
      .mp-sim-score-block { display:flex; align-items:center; justify-content:center;
               gap:22px; padding:18px 0 10px; }
      .mp-sim-team  { display:flex; flex-direction:column; align-items:center; gap:8px; }
      .mp-sim-score { font-size:3.4rem; font-weight:900; color:#1a365d; line-height:1; }
      .mp-sim-dash  { font-size:2.8rem; font-weight:200; color:#cbd5e0; }
      .mp-sim-footer { display:flex; align-items:center; justify-content:center;
               gap:10px; margin-top:8px; flex-wrap:wrap; }
      .mp-sim-note  { font-size:11px; color:#94a3b8; text-align:center;
               margin-top:6px; }
    "))),
    
    tags$main(
      class = "xruns-page",
      # ---- Selector cards — flex row, collapses cleanly on mobile ----
      tags$div(
        class = "mp-selectors-row",
        # Team A card
        tags$div(
          class = "mp-selector-card",
          uiOutput("mp_logo_a"),
          tags$div(
            class = "mp-card-inputs",
            tags$div(class = "mp-sub-label", style = "margin-top:2px;", "Team"),
            selectInput("mp_team_a", NULL,
                        choices  = mp_team_choices,
                        selected = mp_team_choices[[1]],
                        width    = "100%"),
            tags$div(class = "mp-sub-label", "Starting Pitcher"),
            selectizeInput("mp_starter_a", NULL, choices = NULL, width = "100%",
                           options = list(placeholder = "Select pitcher...",
                                          dropdownParent = "body"))
          )
        ),
        tags$div(class = "mp-vs-divider", "VS"),
        # Team B card
        tags$div(
          class = "mp-selector-card",
          uiOutput("mp_logo_b"),
          tags$div(
            class = "mp-card-inputs",
            tags$div(class = "mp-sub-label", style = "margin-top:2px;", "Team"),
            selectInput("mp_team_b", NULL,
                        choices  = mp_team_choices,
                        selected = mp_team_choices[[min(2L, length(mp_team_choices))]],
                        width    = "100%"),
            tags$div(class = "mp-sub-label", "Starting Pitcher"),
            selectizeInput("mp_starter_b", NULL, choices = NULL, width = "100%",
                           options = list(placeholder = "Select pitcher...",
                                          dropdownParent = "body"))
          )
        )
      ),
      
      tags$div(
        class = "xruns-action-row",
        style = "padding:6px 0 18px 0; display:flex; justify-content:center; align-items:center; gap:10px; flex-wrap:wrap;",
        actionButton("mp_predict", "Generate Matchup",
                     class = "btn btn-primary",
                     style = "padding:9px 30px; font-size:13px; min-width:190px;"),
        uiOutput("mp_share_button")
      ),
      
      # ---- Results (rendered server-side) ----
      uiOutput("mp_results"),
      
      # ---- Simulation result card (re-rolls independently) ----
      uiOutput("mp_sim_card"),
      
      tags$div(
        class = "tab-body",
        style = "border-top:1px solid rgba(21,25,34,0.10); border-radius:8px;",
        plotlyOutput("mp_heatmap", height = "500px")
      )
    )
  ),
  
  # ---- Tab: Team Profile ----
  nav_panel(
    title = "Team Profile",
    tags$main(
      class = "xruns-page",

      # Controls row: team selector + year picker
      tags$div(
        style = "display:flex; align-items:center; gap:16px; flex-wrap:wrap; margin-bottom:24px;",
        tags$div(
          style = "flex:1; min-width:220px; max-width:320px;",
          selectInput(
            "tb_team",
            label   = NULL,
            choices = setNames(TEAM_META$abbrev, TEAM_META$team_name),
            selected = "LAD"
          )
        ),
        tags$div(season_picker_inline("tb")),
        tags$div(
          class = "xruns-share-inline",
          style = "margin-left:auto;",
          actionButton(
            "download_team_card_client",
            tagList(tags$i(class = "fa-solid fa-image"), "Share Team Profile"),
            class = "xruns-share-btn",
            onclick = "xrunsDownloadClientShareCard('client_team_share_card')"
          )
        )
      ),

      # Section 1 — Team Header
      uiOutput("tb_header"),

      # Section 2 — Rating Decomposition Bar Chart
      tags$div(
        class = "section-heading",
        "Rating Breakdown",
        tags$span(class = "section-subheading", "runs/game above average by component")
      ),
      tags$div(class = "tab-body", plotlyOutput("tb_bar", height = "280px")),

      # Section 3 — Roster Breakdown
      tags$div(
        class = "section-heading",
        "Roster Breakdown",
        tags$span(class = "section-subheading", "individual player ratings for this team")
      ),
      tags$div(class = "tab-body", DTOutput("tb_players_table")),

      # Section 4 — League Context
      tags$div(
        class = "section-heading",
        "League Context",
        tags$span(class = "section-subheading", "how this team compares across all 30")
      ),
      tags$div(class = "tab-body", plotlyOutput("tb_league_context", height = "260px"))
    )
  ),

  # ---- Tab: Player Profile ----
  nav_panel(
    title = "Player Profile",
    value = "Player Profile",
    tags$main(
      class = "xruns-page",
      style = "padding: 0 20px 32px;",

      # ---- Search bar row ----
      tags$div(
        class = "xruns-pp-search-wrap",
        tags$div(
          class = "xruns-pp-search-inner",
          tags$i(class = "fa-solid fa-magnifying-glass xruns-pp-search-icon"),
          tags$input(
            id          = "pp_player_search",
            type        = "text",
            class       = "xruns-pp-search-input",
            placeholder = "Search for a player…",
            autocomplete = "off"
          ),
          # Dropdown results (populated server-side via uiOutput)
          uiOutput("pp_search_results")
        )
      ),

      # Header card + player content (hidden until a player is selected)
      uiOutput("pp_header"),

      # Two-column grid: percentiles LEFT, composition RIGHT
      tags$div(
        class = "xruns-pp-grid",
        uiOutput("pp_percentiles"),
        uiOutput("pp_composition")
      ),

      # Year-over-year trend (full width)
      tags$div(
        style = "margin-top: 14px;",
        uiOutput("pp_trend_wrap")
      )
    )
  ),

  # ---- Tab: Player Rankings ----
  nav_panel(
    title = "Player Rankings",
    tags$main(
      class = "xruns-page",
      tags$div(
        class = "xruns-explainer-wrap",
        # Always-visible plain-English summary
        tags$p(
          style = "margin: 0 0 8px; font-size:13.5px; color:#485264; line-height:1.55;",
          tags$b("+1.0"), " = one extra run per 9 innings above a league-average player. ",
          "Positive is good. Blanks (—) mean that stat doesn't apply to that player's role."
        ),
        tags$details(
          class = "xruns-explainer",
          tags$summary(
            tags$i(class = "fa-solid fa-circle-info me-2"),
            "Technical details"
          ),
          tags$div(
            class = "xruns-explainer-body",
            "Each column measures how many extra runs per 9 innings that player contributes ",
            "compared to a league-average player at that skill. For example, a ",
            tags$b("Hitting"), " rating of +0.50 means that player generates 0.50 more runs ",
            "per 9 innings than average through their hitting alone. A value of +1.0 is elite; ",
            "most qualified players fall between -1.0 and +1.0. ",
            tags$b("Overall"), " = Hitting + Baserunning + Pitching + Fielding, the total ",
            "runs per 9 innings that player adds above an average player across all facets of the game. ",
            "Dashes (—) appear where a stat is not applicable — hitters have no Pitching rating; ",
            "pitchers have no Hitting or Baserunning rating."
          )
        )
      ),
      heading_with_filter_picker("players", "players_heading"),
      tags$div(class = "tab-body", DTOutput("players_table"))
    )
  ),

  # ---- Tab: Methodology ----
  nav_panel(
    title = "Methodology",
    tags$main(
      class = "xruns-page",
      tags$section(
        class = "xruns-feature-head",
        tags$div(class = "xruns-kicker", "Model notes and validation"),
        tags$h1("The method behind the board."),
        tags$p(
          class = "xruns-feature-copy",
          "The ratings are trained from player-level expected statistics and then checked against actual team run differential. This page keeps the assumptions, formulas, and validation results close to the product instead of hiding them in a footnote."
        )
      ),
      tags$div(class = "xruns-method-wrap",
               layout_columns(
                 col_widths = c(6, 6),
                 card(
                   card_header("Batter model — runs/PA ~ xwOBA + BIP rate"),
                   verbatimTextOutput("bat_model_summary")
                 ),
                 card(
                   card_header("Pitcher model — runs/PA ~ xwOBA + xERA"),
                   verbatimTextOutput("pit_model_summary")
                 )
               ),
               card(
                 card_header("Approach"),
                 tags$div(style = "padding: 10px 16px; line-height:1.65; font-size:13.5px;",
                          tags$p("For each player we fit a weighted linear regression that maps ",
                                 tags$b("expected stats"), " (xwOBA and balls-in-play rate for batters; ",
                                 "xwOBA + xERA for pitchers) to their actual ", tags$b("run value per PA"),
                                 " from Statcast run-value tables. PA is the regression weight."),
                          tags$p("The model is trained on ", tags$b("all player-seasons from completed years (2022–2025) pooled"),
                                 ", not one year at a time. This gives larger, more stable coefficients ",
                                 "and allows any single season to serve as a partial out-of-sample check."),
                          tags$p("Predictions are centered by the ", tags$em("cohort"), " (year) mean, so ",
                                 "0 = league-average player ", tags$em("for that season"),
                                 ". Multiplying by 38 PA/game converts per-PA to per-game. ",
                                 "For completed seasons, teams aggregate their rostered players with a PA-weighted mean."),
                          tags$p(tags$b("Current-season ratings (", DEFAULT_YEAR, ")"),
                                 " use team-level expected stats (xwOBA, BIP rate) fed through the",
                                 " same regression coefficients trained on prior years. This captures the team's",
                                 " production as a whole without relying on individual player run values."),
                          tags$p(tags$b("Baserunning and fielding"),
                                 " are layered on top of the regression as independent",
                                 " Statcast run-value components (runner_runs and fielding total_runs,",
                                 " both already in \u2018runs above average\u2019 units). For the current season,",
                                 " actual year-to-date team run values are used directly.",
                                 " Offense = hitting + baserunning;",
                                 " Defense = pitching + fielding. Team baserunning is normalized to runs per",
                                 " team-PA \u00d7 38 PA/game; team fielding is normalized to runs per defensive",
                                 " fielder-out \u00d7 243 fielder-outs/game."),
                          tags$p(tags$b("Reliability dampening:"),
                                 " hitting and pitching run values emerge from the regression model, which",
                                 " naturally shrinks single-season outliers toward expected outcomes.",
                                 " Raw Statcast baserunning and fielding totals have no equivalent shrinkage,",
                                 " so extreme values are over-weighted relative to their true signal.",
                                 " To place all four components on the same epistemological footing,",
                                 " each raw component is multiplied by a reliability scalar derived from",
                                 " published sabermetric research on year-over-year repeatability:"),
                          tags$ul(
                            style = "margin: 4px 0 10px 18px; font-size:13.5px; line-height:1.7;",
                            tags$li(
                              tags$b("Fielding \u00d7 0.50 \u2014 "),
                              "UZR year-over-year correlation is approximately 0.50 at the player level,",
                              " making fielding the noisiest WAR component by a wide margin. FanGraphs",
                              " explicitly recommends three-year samples before treating fielding values as",
                              " stable. Sources: Lichtman (2010), ",
                              tags$a(href = "https://blogs.fangraphs.com/the-fangraphs-uzr-primer/",
                                     target = "_blank", "\u201cThe FanGraphs UZR Primer\u201d"),
                              "; ",
                              tags$a(href = "https://tht.fangraphs.com/tht-live/how-reliable-is-uzr/",
                                     target = "_blank", "\u201cHow Reliable is UZR?\u201d"),
                              " via The Hardball Times; ",
                              tags$a(href = "https://library.fangraphs.com/defense/uzr/",
                                     target = "_blank", "FanGraphs UZR library page"),
                              "."
                            ),
                            tags$li(
                              tags$b("Baserunning \u00d7 0.70 \u2014 "),
                              "BsR and UBR require more than one year of data to become reliably",
                              " predictive, but speed and baserunning instincts are more persistent",
                              " skills than fielding range, warranting lighter dampening. Sources: ",
                              tags$a(href = "https://library.fangraphs.com/offense/bsr/",
                                     target = "_blank", "FanGraphs BsR library page"),
                              "; ",
                              tags$a(href = "https://library.fangraphs.com/offense/ubr/",
                                     target = "_blank", "FanGraphs UBR library page"),
                              "."
                            )
                          ),
                          tags$p(tags$b("Important:"), " the model is NOT trained on W-L records or team run totals. ",
                                 "The accuracy check below validates it against actual run differential per game.")
                 )
               ),
               card(
                 card_header("Model fit vs. actual run differential — all historical seasons"),
                 verbatimTextOutput("model_fit_summary")
               ),
               
               # ---- Standings accuracy check (moved from its own tab) ------------------
               tags$hr(style = "margin: 24px 0; border-color: #e2e8f0;"),
               tags$div(
                 style = "font-size:1.1rem; font-weight:700; color:#1a365d; margin-bottom:4px;
                 letter-spacing:-0.02em;",
                 "Year-to-Year Model Accuracy vs. Actual Standings"
               ),
               tags$div(
                 style = "display:flex; align-items:center; gap:10px; flex-wrap:wrap;
                 padding: 4px 0 14px 0;",
                 tags$span(style = "font-size:13px; color:#64748b; font-weight:500;", "Season:"),
                 season_picker_inline("sc")
               ),
               tags$div(
                 style = "color:#475569; font-size:13.5px; line-height:1.65; margin-bottom:12px;",
                 tags$p(
                   "The model is fit purely on player expected stats. This section checks how ",
                   "closely the team-level overall rating tracks actual season run ",
                   "differential and wins. Residual = Overall Rating − Actual RDiff/G — ",
                   "a positive residual means the model was bullish vs. the team's real runs."
                 ),
                 tags$p(tags$b("If the current season has no final standings yet, this section will say so."))
               ),
               uiOutput("standings_section"),
               tags$div(
                 class = "tab-body",
                 style = "border-top:1px solid rgba(21,25,34,0.10); border-radius:8px; margin-top:16px;",
                 plotlyOutput("standings_scatter", height = "580px")
               )
      )
    )
  )
)

# =============================================================================
# Server
# =============================================================================
server <- function(input, output, session) {
  # ---- Season picker sync ----
  picker_ids     <- c("season_year_team", "season_year_players",
                      "season_year_sc", "season_year_tb")
  current_year_rv <- reactiveVal(as.character(DEFAULT_YEAR))
  
  for (pid in picker_ids) local({
    this_id <- pid
    observeEvent(input[[this_id]], {
      new_val <- input[[this_id]]
      if (is.null(new_val) || new_val == current_year_rv()) return()
      current_year_rv(new_val)
      for (other in picker_ids)
        if (other != this_id)
          updateSelectInput(session, other, selected = new_val)
      # Reset time period to Season whenever the year changes
      session$sendCustomMessage("resetTimePeriod", list())
    }, ignoreInit = TRUE)
  })
  
  current_year         <- reactive({ current_year_rv() })
  current_players_view <- reactive(players_view_by_year[[current_year()]])

  current_share_data_label <- reactive({
    yc <- current_year()
    tp <- input$time_period %||% "season"
    snapshots <- all_snapshots_by_year[[yc]]

    if (is.null(snapshots) || length(snapshots) == 0) {
      return(paste0(yc, " season data"))
    }

    if (identical(tp, "season")) {
      latest_d <- names(snapshots)[length(snapshots)]
      return(paste0(yc, " season data through ", format(as.Date(latest_d), "%b %d, %Y")))
    }

    window_days <- switch(tp, "30d" = 30, "7d" = 7, "1d" = 1, Inf)
    wd <- compute_window_data(snapshots, window_days)
    if (is.null(wd)) {
      return(paste0(yc, " season data"))
    }

    period <- switch(tp, "30d" = "Last 30 days", "7d" = "Last 7 days", "1d" = "Last day", "Selected period")
    paste0(period, " (", wd$window_label, ")")
  })

  # ---- Selected player ID (reactive value, persists across renders) ----
  selected_player_id <- reactiveVal(NULL)

  # ---- Shareable URL state --------------------------------------------------
  route_state_ready <- reactiveVal(FALSE)
  pending_starter_a <- reactiveVal(NULL)
  pending_starter_b <- reactiveVal(NULL)
  pending_matchup_generate <- reactiveVal(FALSE)
  mp_generation_token <- reactiveVal(0L)

  restore_route_state <- function(hash = "", qs = list()) {
    route <- xruns_parse_route_hash(hash)
    tab <- xruns_safe_text(route$tab, "")

    if (!nzchar(tab)) {
      tab <- tolower(xruns_safe_text(qs$tab, "rankings"))
      if (identical(tab, "player")) {
        route$id <- suppressWarnings(as.integer(xruns_safe_text(qs$id, "")))
      } else if (identical(tab, "team")) {
        route$abbrev <- toupper(xruns_safe_text(qs$abbrev, ""))
      } else if (identical(tab, "matchup")) {
        route$team_a <- toupper(xruns_safe_text(qs$team_a, qs$a %||% ""))
        route$team_b <- toupper(xruns_safe_text(qs$team_b, qs$b %||% ""))
        route$starter_a <- suppressWarnings(as.integer(xruns_safe_text(qs$starter_a, qs$sa %||% "")))
        route$starter_b <- suppressWarnings(as.integer(xruns_safe_text(qs$starter_b, qs$sb %||% "")))
      }
    }

    if (identical(tab, "player")) {
      pid <- suppressWarnings(as.integer(route$id %||% NA_integer_))
      if (!is.na(pid)) selected_player_id(pid)
      nav_select("main_nav", "Player Profile")
    } else if (identical(tab, "team")) {
      abbrev <- toupper(xruns_safe_text(route$abbrev, ""))
      if (abbrev %in% TEAM_META$abbrev) {
        updateSelectInput(session, "tb_team", selected = abbrev)
      }
      nav_select("main_nav", "Team Profile")
    } else if (identical(tab, "matchup")) {
      team_a <- toupper(xruns_safe_text(route$team_a, ""))
      team_b <- toupper(xruns_safe_text(route$team_b, ""))
      if (team_a %in% TEAM_META$abbrev) updateSelectInput(session, "mp_team_a", selected = team_a)
      if (team_b %in% TEAM_META$abbrev) updateSelectInput(session, "mp_team_b", selected = team_b)
      starter_a <- suppressWarnings(as.integer(route$starter_a %||% NA_integer_))
      starter_b <- suppressWarnings(as.integer(route$starter_b %||% NA_integer_))
      pending_starter_a(if (!is.na(starter_a)) starter_a else NULL)
      pending_starter_b(if (!is.na(starter_b)) starter_b else NULL)
      pending_matchup_generate(TRUE)
      nav_select("main_nav", "Matchup Simulator")
    } else if (identical(tab, "players")) {
      nav_select("main_nav", "Player Rankings")
    } else if (identical(tab, "methodology")) {
      nav_select("main_nav", "Methodology")
    } else {
      nav_select("main_nav", "Team Rankings")
    }
  }

  observe({
    qs <- getQueryString(session)
    hash <- getUrlHash(session)
    isolate({
      if (route_state_ready()) return()
      restore_route_state(hash, qs)
      session$onFlushed(function() {
        route_state_ready(TRUE)
      }, once = TRUE)
    })
  })

  observe({
    req(route_state_ready())
    req(input$main_nav)
    active_tab <- input$main_nav %||% "Team Rankings"
    route <- switch(active_tab,
      "Player Profile" = {
        pid <- selected_player_id()
        row <- tryCatch(current_player_row(), error = function(e) NULL)
        xruns_player_profile_route(row, pid)
      },
      "Team Profile" = {
        abbrev <- input$tb_team %||% "LAD"
        xruns_team_profile_route(abbrev)
      },
      "Matchup Simulator" = {
        xruns_matchup_route(
          input$mp_team_a %||% mp_team_choices[[1]],
          input$mp_team_b %||% mp_team_choices[[min(2L, length(mp_team_choices))]],
          input$mp_starter_a,
          input$mp_starter_b
        )
      },
      "Player Rankings" = xruns_route_hash("player-rankings"),
      "Methodology" = xruns_route_hash("methodology"),
      "Team Rankings" = xruns_route_hash("team-rankings"),
      xruns_route_hash("team-rankings")
    )
    session$sendCustomMessage("xrunsRoute", list(route = route))
  })

  observeEvent(input$xruns_route_hash, {
    if (!route_state_ready()) return()
    restore_route_state(input$xruns_route_hash, list())
  }, ignoreInit = TRUE)

  # ---- Team table row click → navigate to Team Profile ----
  observeEvent(input$team_table_row_clicked, {
    abbrev_clicked <- input$team_table_row_clicked
    if (is.null(abbrev_clicked) || abbrev_clicked == "") return()
    updateSelectInput(session, "tb_team", selected = abbrev_clicked)
    nav_select("main_nav", "Team Profile")
  })

  # ---- Player table row click → navigate to Player Profile tab ----
  observeEvent(input$player_table_row_clicked, {
    pid <- as.integer(input$player_table_row_clicked)
    if (is.null(pid) || is.na(pid)) return()
    selected_player_id(pid)
    nav_select("main_nav", "Player Profile")
  })

  # ---- Player search: only include players with a profile in the selected year ----
  profile_search_players <- reactive({
    pv <- current_players_view()
    if (is.null(pv)) return(NULL)

    dual_player_ids <- pv$player_id[pv$Role == "Player"]

    pv %>%
      dplyr::filter(
        Role == "Player" | (Role %in% c("Hitter", "Pitcher") &
                              !(player_id %in% dual_player_ids))
      ) %>%
      dplyr::arrange(Player) %>%
      dplyr::select(player_id, Player, Team, Role) %>%
      dplyr::distinct(player_id, .keep_all = TRUE) %>%
      dplyr::left_join(
        TEAM_META %>% dplyr::select(abbrev, team_logo_espn),
        by = c("Team" = "abbrev")
      )
  })

  # ---- Search dropdown results ----
  output$pp_search_results <- renderUI({
    query <- input$pp_player_search
    if (is.null(query) || nchar(trimws(query)) < 2) return(NULL)

    query_clean <- trimws(query)
    search_players <- profile_search_players()
    if (is.null(search_players) || nrow(search_players) == 0) return(NULL)

    matches <- search_players %>%
      dplyr::filter(grepl(query_clean, Player, ignore.case = TRUE)) %>%
      dplyr::arrange(Player) %>%
      dplyr::slice_head(n = 10)

    if (nrow(matches) == 0) {
      return(tags$div(
        class = "xruns-pp-search-dropdown",
        tags$div(class = "xruns-pp-search-none", "No Qualified Players Found")
      ))
    }

    items <- lapply(seq_len(nrow(matches)), function(i) {
      m        <- matches[i, ]
      logo_url <- if (!is.na(m$team_logo_espn)) m$team_logo_espn else NULL
      tags$div(
        class   = "xruns-pp-search-item",
        onclick = sprintf(
          "Shiny.setInputValue('pp_search_select', %d, {priority:'event'}); document.getElementById('pp_player_search').value = '%s'; this.closest('.xruns-pp-search-dropdown').style.display='none';",
          m$player_id, gsub("'", "\\\\'", m$Player)
        ),
        if (!is.null(logo_url))
          tags$img(src = logo_url, alt = m$Team),
        tags$span(class = "xruns-pp-search-item-name", m$Player),
        tags$span(class = "xruns-pp-search-item-meta",
                  paste0(m$Team, " · ", m$Role))
      )
    })

    tags$div(class = "xruns-pp-search-dropdown", tagList(items))
  })

  # ---- Search item selected ----
  observeEvent(input$pp_search_select, {
    pid <- as.integer(input$pp_search_select)
    if (is.null(pid) || is.na(pid)) return()
    selected_player_id(pid)
  })

  # ---- Reactive: current player data row ----
  current_player_row <- reactive({
    pid <- selected_player_id()
    if (is.null(pid)) return(NULL)
    pv  <- current_players_view()
    if (is.null(pv)) return(NULL)
    # Prefer the merged "Player" row for two-way players
    pv %>%
      dplyr::filter(player_id == pid) %>%
      dplyr::arrange(dplyr::case_when(Role == "Player" ~ 0L, TRUE ~ 1L)) %>%
      dplyr::slice(1)
  })

  output$client_player_share_card <- renderUI({
    row <- current_player_row()
    if (is.null(row) || nrow(row) == 0) return(NULL)
    name <- xruns_slug(row$Player)
    tags$div(
      `data-filename` = paste0("xruns-player-", name, "-", current_year(), ".png"),
      xruns_player_share_card(
        row = row,
        season = current_year(),
        base_url = xruns_current_base_url(session),
        data_label = paste0(current_year(), " season data")
      )
    )
  })

  # ---- Helper: signed value string ----
  signed_str <- function(v, digits = 2) {
    fmt <- paste0("%+.", digits, "f")
    sprintf(fmt, v)
  }

  # ---- pp_header ----
  output$pp_header <- renderUI({
    row <- current_player_row()
    if (is.null(row) || nrow(row) == 0) {
      return(tags$div(
        style = "padding: 48px 0 16px; color: #94a3b8; font-size: 14px; text-align: center;",
        tags$div(style = "font-size: 2rem; margin-bottom: 10px; opacity: 0.3;", "⚾"),
        tags$div("Search for a player above, or click any name in Player Rankings.")
      ))
    }

    pid         <- row$player_id
    player_name <- row$Player
    team_abbrev <- row$Team
    role        <- row$Role
    pa          <- row$PA
    overall     <- row$Overall
    xwoba       <- row$xwOBA

    role_display <- switch(role,
      "Player"  = "Two-Way Player",
      "Hitter"  = "Hitter",
      "Pitcher" = "Pitcher",
      role
    )

    overall_cls  <- if (overall > 0.01) "pos" else if (overall < -0.01) "neg" else "neu"
    overall_sign <- signed_str(overall)

    # Overall percentile pill text (role-appropriate)
    pct_text <- if (role %in% c("Hitter", "Player") && length(pp_hitter_overalls) > 0) {
      h_pct <- pct_rank(overall, pp_hitter_overalls)
      if (!is.na(h_pct)) sprintf("%dth percentile among hitters", h_pct) else NULL
    } else if (role == "Pitcher" && length(pp_pitcher_overalls) > 0) {
      p_pct <- pct_rank(overall, pp_pitcher_overalls)
      if (!is.na(p_pct)) sprintf("%dth percentile among pitchers", p_pct) else NULL
    } else NULL

    # Team logo
    team_logo_url <- TEAM_META %>%
      dplyr::filter(abbrev == team_abbrev) %>%
      dplyr::pull(team_logo_espn)
    team_logo_url <- if (length(team_logo_url) == 1 && !is.na(team_logo_url))
      team_logo_url else NULL

    # Headshot: MLB Cloudinary CDN. The d_people:generic fallback param means
    # the CDN itself serves a silhouette if the player has no photo — no JS needed.
    headshot_url <- sprintf(
      "https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_180,q_auto:best/v1/people/%d/headshot/67/current",
      pid
    )

    tags$div(
      class = "xruns-pp-header",
      # Avatar: Cloudinary handles missing photos server-side with its fallback param
      tags$div(
        class = "xruns-pp-avatar-wrap",
        tags$img(
          src   = headshot_url,
          class = "xruns-pp-headshot",
          alt   = player_name
        )
      ),
      # Name / meta
      tags$div(
        tags$div(class = "xruns-pp-name", player_name),
        tags$div(
          class = "xruns-pp-meta",
          if (!is.null(team_logo_url))
            tags$img(src = team_logo_url, alt = team_abbrev),
          tags$span(team_abbrev),
          tags$span(class = "xruns-pp-meta-dot", "·"),
          tags$span(role_display),
          tags$span(class = "xruns-pp-meta-dot", "·"),
          tags$span(pa, " PA"),
          if (!is.na(xwoba))
            tagList(
              tags$span(class = "xruns-pp-meta-dot", "·"),
              tags$span(sprintf("%.3f xwOBA", xwoba))
            )
        ),
        if (!is.null(pct_text))
          tags$div(tags$span(class = "xruns-pp-pill", pct_text))
      ),
      # Overall rating
      tags$div(
        class = "xruns-pp-overall",
        tags$div(class = paste("xruns-pp-overall-val", overall_cls), overall_sign),
        tags$div(class = "xruns-pp-overall-label", "Overall - runs/9 vs avg"),
        actionButton(
          "download_player_card_client",
          tagList(tags$i(class = "fa-solid fa-image"), "Share Player Card"),
          class = "xruns-share-btn xruns-share-btn-secondary",
          onclick = "xrunsDownloadClientShareCard('client_player_share_card')"
        )
      )
    )
  })

  # ---- pp_percentiles (LEFT card) ----
  output$pp_percentiles <- renderUI({
    row <- current_player_row()
    if (is.null(row) || nrow(row) == 0) return(NULL)

    role        <- row$Role
    overall     <- row$Overall
    hitting     <- row$Hitting
    baserunning <- row$Baserunning
    pitching    <- row$Pitching
    fielding    <- row$Fielding

    is_hitter  <- role %in% c("Hitter", "Player")
    is_pitcher <- role %in% c("Pitcher", "Player")

    make_pct_row <- function(label, val, pool) {
      p <- pct_rank(val, pool)
      if (is.na(p)) return(NULL)
      is_neg  <- p < 40
      fill_pct <- p
      tags$div(
        class = "xruns-pp-pct-row",
        tags$span(class = "xruns-pp-pct-label", label),
        tags$div(
          class = "xruns-pp-pct-track",
          tags$div(
            class = if (is_neg) "xruns-pp-pct-fill neg-fill" else "xruns-pp-pct-fill",
            style = sprintf("width: %d%%;", fill_pct)
          )
        ),
        tags$span(
          class = if (is_neg) "xruns-pp-pct-num neg-pct" else "xruns-pp-pct-num",
          sprintf("%dth", p)
        )
      )
    }

    rows <- list()

    if (is_hitter && is_pitcher) {
      # Two-way: show hitter pool for hitting-side, pitcher pool for pitching-side
      rows[["h_overall"]] <- make_pct_row("Overall (hit)", overall, pp_hitter_overalls)
      rows[["p_overall"]] <- make_pct_row("Overall (pit)", overall, pp_pitcher_overalls)
      rows[["hitting"]]   <- make_pct_row("Hitting", hitting, pp_hitter_hitting)
      rows[["pitching"]]  <- make_pct_row("Pitching", pitching, pp_pitcher_pitching)
      rows[["br"]]        <- make_pct_row("Baserunning", baserunning, pp_hitter_br)
      rows[["fielding"]]  <- make_pct_row("Fielding", fielding, pp_fielding_all)
    } else if (is_hitter) {
      rows[["overall"]]   <- make_pct_row("Overall", overall, pp_hitter_overalls)
      rows[["hitting"]]   <- make_pct_row("Hitting", hitting, pp_hitter_hitting)
      rows[["br"]]        <- make_pct_row("Baserunning", baserunning, pp_hitter_br)
      rows[["fielding"]]  <- make_pct_row("Fielding", fielding, pp_fielding_all)
    } else {
      rows[["overall"]]   <- make_pct_row("Overall", overall, pp_pitcher_overalls)
      rows[["pitching"]]  <- make_pct_row("Pitching", pitching, pp_pitcher_pitching)
      rows[["fielding"]]  <- make_pct_row("Fielding", fielding, pp_fielding_all)
    }

    pool_note <- if (is_hitter && is_pitcher) {
      "Percentiles vs. all hitters / pitchers in current season."
    } else if (is_hitter) {
      "Percentiles vs. all hitters in current season."
    } else {
      "Percentiles vs. all pitchers in current season."
    }

    tags$div(
      class = "xruns-pp-card",
      tags$div(class = "xruns-pp-card-label", "League percentiles"),
      tagList(Filter(Negate(is.null), rows)),
      tags$div(
        style = "font-size:10.5px; color:#94a3b8; margin-top:8px;",
        pool_note
      )
    )
  })

  # ---- pp_composition (RIGHT card) — diverging bars ----
  output$pp_composition <- renderUI({
    row <- current_player_row()
    if (is.null(row) || nrow(row) == 0) return(NULL)

    components <- list(
      list(label = "Hitting",     val = row$Hitting),
      list(label = "Baserunning", val = row$Baserunning),
      list(label = "Pitching",    val = row$Pitching),
      list(label = "Fielding",    val = row$Fielding)
    )
    components <- Filter(function(x) !is.na(x$val), components)
    if (length(components) == 0) return(NULL)

    # Scale bars relative to the largest absolute value among components
    max_abs <- max(sapply(components, function(x) abs(x$val)), na.rm = TRUE)
    if (max_abs == 0) max_abs <- 1

    comp_rows <- lapply(components, function(x) {
      v      <- x$val
      pct    <- min(abs(v) / max_abs * 100, 100)
      is_pos <- v > 0.01
      is_neg <- v < -0.01
      val_cls_str <- if (is_pos) "pos" else if (is_neg) "neg" else "neu"
      sign_s <- signed_str(v)

      tags$div(
        class = "xruns-pp-comp-row",
        # Label
        tags$span(class = "xruns-pp-comp-label", x$label),
        # Negative half (bar grows rightward from centre for negative values)
        tags$div(
          class = "xruns-pp-comp-neg-half",
          if (is_neg)
            tags$div(class = "xruns-pp-comp-bar neg",
                     style = sprintf("width:%.1f%%;", pct))
          else
            tags$div(style = "width:0;")
        ),
        # Centre zero tick
        tags$div(class = "xruns-pp-comp-zero"),
        # Positive half
        tags$div(
          class = "xruns-pp-comp-pos-half",
          if (is_pos)
            tags$div(class = "xruns-pp-comp-bar pos",
                     style = sprintf("width:%.1f%%;", pct))
          else
            tags$div(style = "width:0;")
        ),
        # Value label
        tags$span(class = paste("xruns-pp-comp-val", val_cls_str), sign_s)
      )
    })

    tags$div(
      class = "xruns-pp-card",
      tags$div(class = "xruns-pp-card-label", "Component breakdown"),
      tagList(comp_rows),
      tags$div(
        style = "font-size:10.5px; color:#94a3b8; margin-top:8px;",
        "Bars show each component's contribution relative to the largest. Runs above avg per 9 inn."
      )
    )
  })

  # ---- pp_trend_wrap (full-width trend chart) ----
  output$pp_trend_wrap <- renderUI({
    row <- current_player_row()
    if (is.null(row) || nrow(row) == 0) return(NULL)
    tags$div(
      class = "xruns-pp-card",
      tags$div(class = "xruns-pp-card-label", "Year-over-year overall rating"),
      plotlyOutput("pp_trend_chart", height = "180px")
    )
  })

  output$pp_trend_chart <- renderPlotly({
    row <- current_player_row()
    if (is.null(row) || nrow(row) == 0) return(NULL)
    pid <- row$player_id

    hist <- pp_history %>%
      dplyr::filter(player_id == pid) %>%
      # For two-way players prefer the "Player" row; else take whichever role exists
      dplyr::arrange(year, dplyr::case_when(Role == "Player" ~ 0L, TRUE ~ 1L)) %>%
      dplyr::distinct(year, .keep_all = TRUE) %>%
      dplyr::arrange(year)

    if (nrow(hist) == 0) return(NULL)

    hover_txt <- paste0(
      "<b>", hist$year, "</b><br>",
      "Overall: ", sprintf("%+.2f", hist$Overall), " runs/9<br>",
      hist$Team, " · ", hist$PA, " PA"
    )

    # Zero reference line — draw first so it sits behind the data line
    x_pad <- if (nrow(hist) > 1) 0.3 else 0.5

    plot_ly() %>%
      # Dotted zero line
      add_segments(
        x = min(hist$year) - x_pad, xend = max(hist$year) + x_pad,
        y = 0, yend = 0,
        line = list(color = "#cbd5e1", width = 1, dash = "dot"),
        showlegend = FALSE, hoverinfo = "none"
      ) %>%
      # Data line — clean navy, no markers
      add_trace(
        x         = hist$year,
        y         = hist$Overall,
        type      = "scatter",
        mode      = "lines",
        line      = list(color = "#1a365d", width = 2.5),
        text      = hover_txt,
        hoverinfo = "text"
      ) %>%
      layout(
        xaxis = list(
          title      = "",
          tickformat = "d",
          tickvals   = hist$year,
          zeroline   = FALSE,
          showgrid   = FALSE,
          color      = "#94a3b8"
        ),
        yaxis = list(
          title    = "Runs / 9 inn vs avg",
          zeroline = FALSE,
          showgrid = TRUE,
          gridcolor = "#f1f5f9",
          color    = "#94a3b8",
          tickfont = list(size = 11)
        ),
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        margin        = list(t = 8, r = 16, b = 32, l = 52),
        font          = list(family = "Inter", size = 11, color = "#94a3b8"),
        showlegend    = FALSE,
        hoverlabel    = list(bgcolor = "#1a365d", font = list(color = "#fff", size = 12))
      ) %>%
      config(displayModeBar = FALSE)
  })

  # (pp_underlying removed — replaced by cleaner component breakdown card)
  
  # ---- Time period chip toggle (only for years with snapshot history) --------
  # Rendered as plain text chips — quiet by default, minimal visual footprint.
  # Clicking a chip calls Shiny.setInputValue so input$time_period updates
  # exactly like a radioButtons input would, with no extra packages needed.
  output$time_period_ui <- renderUI({
    yc        <- current_year()
    has_multi <- yc %in% names(all_snapshots_by_year) &&
      length(all_snapshots_by_year[[yc]]) > 1
    if (!has_multi) return(NULL)
    
    sel <- isolate(input$time_period) %||% "season"
    
    make_chip <- function(value, label) {
      is_active <- identical(sel, value)
      tags$button(
        class   = paste("xruns-period-chip", if (is_active) "active" else ""),
        onclick = sprintf(
          "Shiny.setInputValue('time_period', '%s', {priority: 'event'});
           document.querySelectorAll('.xruns-period-chip').forEach(function(el){el.classList.remove('active');});
           this.classList.add('active');",
          value
        ),
        label
      )
    }
    
    tags$div(
      class = "xruns-period-group",
      make_chip("season", "Season"),
      tags$span(class = "xruns-period-sep", "·"),
      make_chip("30d", "30d"),
      tags$span(class = "xruns-period-sep", "·"),
      make_chip("7d", "7d"),
      tags$span(class = "xruns-period-sep", "·"),
      make_chip("1d", "Yesterday")
    )
  })
  
  output$window_note_ui <- renderUI({
    yc <- current_year()
    tp <- input$time_period %||% "season"
    if (tp == "season") return(NULL)
    if (!(yc %in% names(all_snapshots_by_year))) return(NULL)
    
    window_days <- switch(tp, "30d" = 30, "7d" = 7, "1d" = 1, Inf)
    wd <- compute_window_data(all_snapshots_by_year[[yc]], window_days)
    if (is.null(wd) || !isTRUE(wd$window_fallback)) return(NULL)
    
    tags$div(class = "xruns-window-note",
             tags$i(class = "fa-solid fa-circle-info me-1"),
             wd$fallback_msg
    )
  })
  
  # ---- Team table helper — supports season and rolling windows ---------------
  team_tbl_for_period <- function(tp = "season") {
    yc <- current_year()
    
    has_snaps <- yc %in% names(all_snapshots_by_year) &&
      length(all_snapshots_by_year[[yc]]) > 1
    
    # ---- "Season" view: Bayesian-blended full-season + 14-day window ----------
    # When snapshots are available for the selected year, blend the full-season
    # rating with a 14-day rolling window using the recency scheme defined by
    # RECENCY_BLEND_ALPHA_MAX and RECENCY_BLEND_K.  Falls back to the
    # pre-computed pure-season table if blending fails or no snapshots exist.
    if (tp == "season") {
      if (!has_snaps) return(team_tbl_by_year[[yc]])
      standings_df <- if (!is.null(team_raw_years[[yc]]$standings))
        team_raw_years[[yc]]$standings
      else NULL
      blended <- tryCatch(
        compute_blended_season_ratings(all_snapshots_by_year[[yc]], models, standings_df),
        error = function(e) NULL
      )
      return(blended %||% team_tbl_by_year[[yc]])
    }
    
    # ---- Rolling window views (30d / 7d / 1d): raw delta, no blending --------
    window_days <- switch(tp, "30d" = 30, "7d" = 7, "1d" = 1, Inf)
    year_data   <- compute_window_data(all_snapshots_by_year[[yc]], window_days)
    if (is.null(year_data)) return(team_tbl_by_year[[yc]])
    
    tryCatch(
      build_team_ratings_team_mode(year_data, models),
      error = function(e) team_tbl_by_year[[yc]]
    )
  }

  # ---- Team table reactive — responds to year AND time period ----------------
  current_team_tbl <- reactive({
    team_tbl_for_period(input$time_period %||% "season")
  })

  output$client_rankings_share_card <- renderUI({
    tt <- current_team_tbl()
    if (is.null(tt) || nrow(tt) == 0) return(NULL)
    tags$div(
      `data-filename` = paste0("xruns-rankings-", current_year(), ".png"),
      xruns_rankings_share_card(
        tt = tt,
        season = current_year(),
        base_url = xruns_current_base_url(session),
        data_label = current_share_data_label()
      )
    )
  })
  
  # ---- Headings ----
  output$team_heading    <- renderText(sprintf("%s Team Performance Ratings", current_year()))
  output$players_heading <- renderText(sprintf("%s Player Leaderboard", current_year()))
  
  # ---- KPI cards ----
  output$kpi_row <- renderUI({
    tt <- current_team_tbl()
    
    make_card <- function(label, row, value_fmt, css_class = "positive") {
      logo_tag <- if (!is.na(row$team_logo_espn)) {
        tags$img(src = row$team_logo_espn, class = "kpi-logo",
                 alt = row$abbrev)
      } else {
        tags$div(class = "kpi-logo-placeholder", row$abbrev)
      }
      tags$div(class = "kpi-card",
               logo_tag,
               tags$div(class = "kpi-text",
                        tags$div(class = "kpi-label", label),
                        tags$div(class = "kpi-team",  row$team_name),
                        tags$div(class = paste("kpi-value", css_class), value_fmt)
               )
      )
    }
    
    top_overall  <- tt[which.max(tt$overall),      ]
    top_off      <- tt[which.max(tt$off_rating),   ]
    top_pit      <- tt[which.max(tt$def_pitching), ]
    top_fld      <- tt[which.max(tt$def_fld),      ]
    
    tags$div(class = "kpi-row",
             make_card(tagList(tags$i(class = "fa-solid fa-ranking-star"), "Best Overall"), top_overall,
                       sprintf("%+.2f runs/game", top_overall$overall)),
             make_card(tagList(tags$i(class = "fa-solid fa-baseball-bat-ball"), "Best Offense"), top_off,
                       sprintf("%+.2f runs/game", top_off$off_rating)),
             make_card(tagList(tags$i(class = "fa-solid fa-baseball"), "Best Pitching"), top_pit,
                       sprintf("%+.2f runs/game", top_pit$def_pitching)),
             make_card(tagList(tags$i(class = "fa-solid fa-shield-halved"), "Best Fielding"), top_fld,
                       sprintf("%+.2f runs/game", top_fld$def_fld))
    )
  })
  
  # ---- Signed render JS for DT ----
  signed_render <- JS(
    "function(data, type, row) {",
    "  if (type === 'display' || type === 'filter') {",
    "    if (data === null || data === undefined || data === '' || isNaN(data)) return '—';",
    "    var n = Number(data);",
    "    return (n > 0 ? '+' : '') + n.toFixed(2);",
    "  }",
    "  return data;",
    "}"
  )
  
  # ---- Team Rankings table ----
  output$team_table <- renderDT({
    tt <- current_team_tbl()
    
    # Pre-compute ranks for each sub-category (descending = higher is better)
    off_rank <- rank(-tt$off_rating,   ties.method = "min")
    pit_rank <- rank(-tt$def_pitching, ties.method = "min")
    fld_rank <- rank(-tt$def_fld,      ties.method = "min")
    
    # Helper: build an HTML cell with signed value + grey rank badge
    make_ranked_cell <- function(value, r) {
      val_str <- if (!is.finite(value)) "" else {
        sprintf("%s%.2f", if (value > 0) "+" else "", value)
      }
      color <- if (!is.finite(value)) "#64748b" else if (value > 0.01) "#047857" else if (value < -0.01) "#b91c1c" else "#64748b"
      sprintf(
        '<span style="font-weight:700; color:%s;">%s</span> <span style="font-size:10px; color:#94a3b8; font-weight:500;">(%d)</span>',
        color, val_str, r
      )
    }
    
    dt <- tt %>%
      mutate(
        ` `          = ifelse(
          !is.na(team_logo_espn),
          paste0('<img src="', team_logo_espn,
                 '" height="26" title="', team_name,
                 '" style="vertical-align:middle; margin-right:2px;">'),
          paste0('<span style="font-weight:700;">', abbrev, '</span>')
        ),
        Offense      = mapply(make_ranked_cell, off_rating,   off_rank),
        Pitching     = mapply(make_ranked_cell, def_pitching, pit_rank),
        Fielding     = mapply(make_ranked_cell, def_fld,      fld_rank),
        # Hidden numeric columns used for sorting (invisible in table)
        off_sort     = round(off_rating,   4),
        pit_sort     = round(def_pitching, 4),
        fld_sort     = round(def_fld,      4)
      ) %>%
      transmute(
        Rank     = rank,
        ` `,
        Team     = team_name,            # col 2 — hidden on mobile via CSS
        Overall  = round(overall, 4),   # col 3 — always visible
        Offense,
        Pitching,
        Fielding,
        off_sort,
        pit_sort,
        fld_sort,
        # Hidden: abbrev for row-click navigation (col index 10)
        abbrev_  = abbrev
      )

    # Column indices (0-indexed for DT columnDefs):
    #   0 Rank  1 Logo  2 Team  3 Overall  4 Offense  5 Pitching  6 Fielding
    #   7 off_sort (hidden)  8 pit_sort (hidden)  9 fld_sort (hidden)
    #   10 abbrev_ (hidden)
    # Team col (2) is always in the DOM; CSS class "team-name-col" hides it on mobile.
    centered_cols  <- c(0, 1, 3, 4, 5, 6)

    row_click_js <- JS(
      "function(row, data, index) {",
      "  $(row).css('cursor', 'pointer');",
      "  $(row).on('click', function(e) {",
      "    Shiny.setInputValue('team_table_row_clicked', data[10], {priority: 'event'});",
      "  });",
      "}"
    )

    datatable(
      dt,
      rownames    = FALSE,
      escape      = FALSE,
      class       = "compact stripe hover",
      options     = list(
        pageLength = 30,
        dom        = "t",
        scrollX    = TRUE,
        # Default sort: Overall descending (column index 3)
        order      = list(list(3, "desc")),
        autoWidth  = FALSE,
        rowCallback = row_click_js,
        columnDefs = list(
          list(targets = 3,             render = signed_render),
          list(className = "dt-center", targets = centered_cols),
          list(orderable = FALSE,       targets = 1),
          list(width = "44px",          targets = 1),
          # Team column: assign class so CSS can hide it on mobile
          list(targets = 2, className = "team-name-col"),
          # Run value columns: clicking sorts descending first (higher = better)
          list(targets = c(3, 4, 5, 6), orderSequence = list("desc", "asc")),
          # Rank column: clicking sorts ascending first (lower number = better)
          list(targets = 0, orderSequence = list("asc", "desc")),
          # Point HTML display columns at their hidden numeric sort columns
          list(targets = 4, orderData = 7),
          list(targets = 5, orderData = 8),
          list(targets = 6, orderData = 9),
          # Hide the sort helper and abbrev columns
          list(targets = c(7, 8, 9, 10), visible = FALSE)
        )
      )
    ) %>%
      formatStyle(columns = seq_len(ncol(dt)),
                  fontSize = "13.5px", lineHeight = "1.4") %>%
      formatStyle("Team", fontWeight = "bold") %>%
      formatStyle("Overall",
                  color      = styleInterval(c(-0.01, 0.01),
                                             c("#b91c1c", "#64748b", "#047857")),
                  fontWeight = "bold")
  })
  
  # ---- Offense vs Pitching scatter (plotly with team logos) ----
  output$team_scatter <- renderPlotly({
    tt <- current_team_tbl()
    
    x_vals <- tt$off_rating
    y_vals <- tt$def_rating
    
    x_pad <- diff(range(x_vals)) * 0.18
    y_pad <- diff(range(y_vals)) * 0.18
    x_rng <- c(min(x_vals) - x_pad, max(x_vals) + x_pad)
    y_rng <- c(min(y_vals) - y_pad, max(y_vals) + y_pad)
    
    img_w <- diff(x_rng) * 0.075
    img_h <- diff(y_rng) * 0.075
    
    hover_txt <- paste0(
      "<b>", tt$team_name, "</b>",
      "<br>Overall  ", sprintf("%+.2f", tt$overall),
      "<br>Off  ", sprintf("%+.2f", tt$off_rating),
      "  (Hit ", sprintf("%+.2f", tt$off_hitting),
      " · BR ", sprintf("%+.2f", tt$off_br), ")",
      "<br>Def  ", sprintf("%+.2f", tt$def_rating),
      "  (Pit ", sprintf("%+.2f", tt$def_pitching),
      " · Fld ", sprintf("%+.2f", tt$def_fld), ")"
    )

    has_logos <- !all(is.na(tt$team_logo_espn))

    p <- plot_ly() %>%
      # Quadrant reference lines
      add_segments(x = 0, xend = 0, y = y_rng[1], yend = y_rng[2],
                   line = list(color = "#cbd5e0", width = 1, dash = "dot"),
                   showlegend = FALSE, hoverinfo = "none") %>%
      add_segments(x = x_rng[1], xend = x_rng[2], y = 0, yend = 0,
                   line = list(color = "#cbd5e0", width = 1, dash = "dot"),
                   showlegend = FALSE, hoverinfo = "none") %>%
      # Invisible markers for hover tooltips — sized to cover the logo footprint
      add_trace(
        x           = x_vals,
        y           = y_vals,
        type        = "scatter",
        mode        = "markers",
        marker      = list(size = 48, color = "rgba(0,0,0,0)"),
        text        = hover_txt,
        hoverinfo   = "text",
        hoverlabel  = list(
          bgcolor     = "#1e293b",
          bordercolor = "#334155",
          font        = list(color = "#f8fafc", family = "Inter", size = 13)
        ),
        showlegend  = FALSE
      )
    
    if (has_logos) {
      # Logo images at each data point
      logo_images <- lapply(seq_len(nrow(tt)), function(i) {
        list(
          source    = tt$team_logo_espn[i],
          x         = tt$off_rating[i],
          y         = tt$def_rating[i],
          xref      = "x",
          yref      = "y",
          sizex     = img_w,
          sizey     = img_h,
          xanchor   = "center",
          yanchor   = "middle",
          layer     = "above"
        )
      })
    } else {
      # Fallback: colored abbreviation labels
      logo_images <- list()
      p <- p %>%
        add_trace(
          x          = x_vals,
          y          = y_vals,
          type       = "scatter",
          mode       = "text",
          text       = tt$abbrev,
          textfont   = list(size = 11, color = "#1a365d", family = "Inter"),
          showlegend = FALSE,
          hoverinfo  = "none"
        )
    }
    
    # Quadrant label positions: near the corners of the plot
    quad_lbl_x_pos <- x_rng[2] - diff(x_rng) * 0.03
    quad_lbl_x_neg <- x_rng[1] + diff(x_rng) * 0.03
    quad_lbl_y_pos <- y_rng[2] - diff(y_rng) * 0.04
    quad_lbl_y_neg <- y_rng[1] + diff(y_rng) * 0.04
    
    quadrant_annotations <- list(
      list(x = quad_lbl_x_pos, y = quad_lbl_y_pos, xref = "x", yref = "y",
           text = "Good Offense\nGood Defense", showarrow = FALSE,
           font = list(size = 10.5, color = "#047857", family = "Inter"),
           align = "right", xanchor = "right", yanchor = "top"),
      list(x = quad_lbl_x_neg, y = quad_lbl_y_pos, xref = "x", yref = "y",
           text = "Bad Offense\nGood Defense", showarrow = FALSE,
           font = list(size = 10.5, color = "#64748b", family = "Inter"),
           align = "left", xanchor = "left", yanchor = "top"),
      list(x = quad_lbl_x_pos, y = quad_lbl_y_neg, xref = "x", yref = "y",
           text = "Good Offense\nBad Defense", showarrow = FALSE,
           font = list(size = 10.5, color = "#64748b", family = "Inter"),
           align = "right", xanchor = "right", yanchor = "bottom"),
      list(x = quad_lbl_x_neg, y = quad_lbl_y_neg, xref = "x", yref = "y",
           text = "Bad Offense\nBad Defense", showarrow = FALSE,
           font = list(size = 10.5, color = "#b91c1c", family = "Inter"),
           align = "left", xanchor = "left", yanchor = "bottom")
    )
    
    p %>% layout(
      images      = logo_images,
      annotations = quadrant_annotations,
      xaxis     = list(title      = list(text = "Offensive Rating — hitting + baserunning (runs/game)",
                                         font = list(size = 12.5, color = "#475569")),
                       range      = x_rng,
                       zeroline   = FALSE,
                       gridcolor  = "#f1f5f9",
                       tickfont   = list(size = 11, color = "#94a3b8")),
      yaxis     = list(title      = list(text = "Defensive Rating — pitching + fielding (runs/game)",
                                         font = list(size = 12.5, color = "#475569")),
                       range      = y_rng,
                       zeroline   = FALSE,
                       gridcolor  = "#f1f5f9",
                       tickfont   = list(size = 11, color = "#94a3b8")),
      paper_bgcolor = "#ffffff",
      plot_bgcolor  = "#fafafa",
      margin        = list(t = 20, r = 20, b = 50, l = 60),
      font          = list(family = "Inter")
    ) %>%
      config(displayModeBar = FALSE)
  })
  
  # ---- Player table (unified batters + pitchers) ----
  output$players_table <- renderDT({
    pv <- current_players_view()
    
    # Team-mode years have no individual player data yet.
    if (is.null(pv)) {
      yr <- current_year()
      return(datatable(
        data.frame(
          Message = sprintf(
            "Individual player data for %s is not yet available. Place player-level files in '%s Player Data/' to enable this tab.",
            yr, yr
          )
        ),
        rownames = FALSE,
        options  = list(dom = "t", pageLength = 1)
      ))
    }
    
    roles_selected <- input$players_role_filter
    if (is.null(roles_selected) || length(roles_selected) == 0) {
      roles_selected <- c("Hitter", "Pitcher")
    }

    both_selected <- all(c("Hitter", "Pitcher") %in% roles_selected)

    # player_ids that have a merged "Player" row (two-way players like Ohtani)
    dual_player_ids <- pv$player_id[pv$Role == "Player"]

    pv <- pv %>%
      dplyr::filter(
        if (both_selected) {
          # Show merged "Player" row for duals; suppress their individual rows
          Role == "Player" | (Role %in% c("Hitter", "Pitcher") &
                                !(player_id %in% dual_player_ids))
        } else {
          # Single role: only that role's rows; never show merged "Player" rows
          Role %in% roles_selected & Role != "Player"
        }
      ) %>%
      dplyr::arrange(dplyr::desc(Overall)) %>%
      dplyr::mutate(Rank = dplyr::row_number()) %>%
      dplyr::left_join(
        TEAM_META %>% dplyr::select(abbrev, team_logo_espn),
        by = c("Team" = "abbrev")
      ) %>%
      dplyr::mutate(
        Logo = ifelse(
          !is.na(team_logo_espn),
          paste0('<img src="', team_logo_espn,
                 '" height="22" style="vertical-align:middle;">'),
          ""
        ),
        # Alias player_id as player_id_ so DT doesn't drop it as a dup
        player_id_ = player_id
      ) %>%
      dplyr::select(Rank, Logo, Player, Overall, Team, Role, PA,
                    Hitting, Baserunning, Pitching, Fielding,
                    player_id_)  # hidden — used for row-click

    # Column indices (0-indexed): Rank=0, Logo=1, Player=2, Overall=3, Team=4,
    #   Role=5, PA=6, Hitting=7, Baserunning=8, Pitching=9, Fielding=10
    #   player_id_=11 (hidden)
    signed_cols   <- c(3, 7, 8, 9, 10)
    centered_cols <- c(0, 3, 4, 5, 6, 7, 8, 9, 10)

    player_row_click_js <- JS(
      "function(row, data, index) {",
      "  $(row).css('cursor', 'pointer');",
      "  $(row).on('click', function(e) {",
      "    if ($(e.target).closest('td.dtr-control').length > 0) return;",
      "    Shiny.setInputValue('player_table_row_clicked', data[11], {priority: 'event'});",
      "  });",
      "}"
    )

    datatable(
      pv,
      rownames   = FALSE,
      escape     = FALSE,
      class      = "compact stripe hover",
      options    = list(
        pageLength  = 25,
        # Default sort: Overall descending (column index 3)
        order       = list(list(3, "desc")),
        dom         = "frtip",
        scrollX     = TRUE,
        searchDelay = 100,
        rowCallback = player_row_click_js,
        columnDefs  = list(
          list(orderable = FALSE, targets = 1),
          list(width = "36px",    targets = 1),
          list(className = "dt-center", targets = centered_cols),
          list(targets = signed_cols, render = signed_render),
          # Run value columns sort descending first (higher = better)
          list(targets = c(3, 7, 8, 9, 10), orderSequence = list("desc", "asc")),
          # Rank sorts ascending first (lower = better)
          list(targets = 0, orderSequence = list("asc", "desc")),
          # Show dash for N/A cells
          list(targets = c(7, 8, 9, 10), defaultContent = "—"),
          # Hide the player_id column
          list(targets = 11, visible = FALSE)
        )
      )
    ) %>%
      formatStyle(columns = seq_len(ncol(pv)), fontSize = "13.5px") %>%
      formatStyle("Overall",
                  color      = styleInterval(c(-0.01, 0.01),
                                             c("#b91c1c", "#64748b", "#047857")),
                  fontWeight = "bold") %>%
      formatStyle(c("Hitting", "Baserunning", "Pitching", "Fielding"),
                  color = styleInterval(c(-0.01, 0.01),
                                        c("#b91c1c", "#64748b", "#047857")))
  }, server = FALSE)

  # ---- Standings Check ----
  standings_for_year <- reactive({ standings_check[[current_year()]] })
  
  output$standings_section <- renderUI({
    sc <- standings_for_year()
    if (is.null(sc) || nrow(sc) == 0) {
      return(tags$div(
        class = "standings-warning",
        sprintf("No final standings data for %s yet — nothing to validate against. Check back when the season wraps up.",
                current_year())
      ))
    }
    tagList(
      verbatimTextOutput("standings_corrs"),
      DTOutput("standings_table")
    )
  })
  
  output$standings_corrs <- renderPrint({
    sc <- standings_for_year()
    if (is.null(sc) || nrow(sc) == 0) return(invisible(NULL))
    pearson  <- cor(sc$overall, sc$rdiff_per_game)
    spearman <- cor(sc$overall, sc$rdiff_per_game, method = "spearman")
    rmse     <- sqrt(mean((sc$overall - sc$rdiff_per_game)^2))
    cat(sprintf(
      "Rating vs. actual run differential/game — Pearson r = %.3f | Spearman rho = %.3f | RMSE = %.3f runs/game (%d teams)\n",
      pearson, spearman, rmse, nrow(sc)
    ))
  })
  
  output$standings_table <- renderDT({
    sc <- standings_for_year()
    if (is.null(sc) || nrow(sc) == 0) return(NULL)
    dt <- sc %>% select(-team_id, -team_logo_espn, -overall, -rdiff_per_game)
    datatable(
      dt,
      rownames = FALSE,
      class    = "compact stripe hover",
      options  = list(pageLength = 30, dom = "t",
                      order = list(list(0, "asc")), scrollX = TRUE)
    ) %>%
      formatStyle(columns = seq_len(ncol(dt)), fontSize = "13.5px") %>%
      formatStyle("Overall Rating",
                  color      = styleInterval(c(-0.01, 0.01),
                                             c("#b91c1c", "#64748b", "#047857")),
                  fontWeight = "bold") %>%
      formatStyle("Actual RDiff/G",
                  color = styleInterval(c(-0.01, 0.01),
                                        c("#b91c1c", "#64748b", "#047857")))
  })
  
  output$standings_scatter <- renderPlotly({
    sc <- standings_for_year()
    
    if (is.null(sc) || nrow(sc) == 0) {
      return(
        plot_ly() %>%
          layout(
            xaxis = list(visible = FALSE), yaxis = list(visible = FALSE),
            annotations = list(list(
              x = 0.5, y = 0.5, xref = "paper", yref = "paper",
              text = sprintf("No final standings for %s yet.", current_year()),
              showarrow = FALSE,
              font = list(size = 14, color = "#94a3b8")
            )),
            paper_bgcolor = "#ffffff", plot_bgcolor = "#fafafa"
          )
      )
    }
    
    # Join logos back in
    sc <- sc %>%
      left_join(TEAM_META %>% select(abbrev, team_logo_espn),
                by = c("Team" = "abbrev"), suffix = c("", ".meta")) %>%
      mutate(team_logo_espn = dplyr::coalesce(team_logo_espn, team_logo_espn.meta))
    
    rng_all <- range(c(sc$overall, sc$rdiff_per_game))
    pad     <- diff(rng_all) * 0.12
    rng     <- c(rng_all[1] - pad, rng_all[2] + pad)
    img_sz  <- diff(rng) * 0.065
    
    hover_txt <- paste0(
      "<b>", sc$Team, " — ", sc$`Team Name`, "</b>",
      "<br>Model Rating  ", sprintf("%+.2f", sc$overall),
      "<br>Actual RDiff/G  ", sprintf("%+.2f", sc$rdiff_per_game),
      "<br>Record  ", sc$`Actual W`, "–", sc$`Actual L`
    )

    has_logos <- !all(is.na(sc$team_logo_espn))

    p <- plot_ly() %>%
      add_segments(x = rng[1], xend = rng[2], y = rng[1], yend = rng[2],
                   line = list(color = "#cbd5e0", width = 1, dash = "dot"),
                   showlegend = FALSE, hoverinfo = "none") %>%
      add_segments(x = 0, xend = 0, y = rng[1], yend = rng[2],
                   line = list(color = "#e2e8f0", width = 1),
                   showlegend = FALSE, hoverinfo = "none") %>%
      add_segments(x = rng[1], xend = rng[2], y = 0, yend = 0,
                   line = list(color = "#e2e8f0", width = 1),
                   showlegend = FALSE, hoverinfo = "none") %>%
      add_trace(
        x          = sc$overall,
        y          = sc$rdiff_per_game,
        type       = "scatter",
        mode       = "markers",
        marker     = list(size = 48, color = "rgba(0,0,0,0)"),
        text       = hover_txt,
        hoverinfo  = "text",
        hoverlabel = list(
          bgcolor     = "#1e293b",
          bordercolor = "#334155",
          font        = list(color = "#f8fafc", family = "Inter", size = 13)
        ),
        showlegend = FALSE
      )

    if (has_logos) {
      logo_images <- lapply(seq_len(nrow(sc)), function(i) {
        list(
          source  = sc$team_logo_espn[i],
          x       = sc$overall[i],
          y       = sc$rdiff_per_game[i],
          xref    = "x", yref = "y",
          sizex   = img_sz, sizey = img_sz,
          xanchor = "center", yanchor = "middle",
          layer   = "above"
        )
      })
    } else {
      logo_images <- list()
      p <- p %>% add_trace(
        x = sc$overall, y = sc$rdiff_per_game,
        type = "scatter", mode = "text",
        text = sc$Team,
        textfont = list(size = 11, color = "#1a365d"),
        showlegend = FALSE, hoverinfo = "none"
      )
    }
    
    # Annotations: Q2 (upper-left) = Overperforming, Q4 (lower-right) = Underperforming
    # Q2: high actual run diff, low model rating → team beat expectations
    # Q4: low actual run diff, high model rating → team fell short of expectations
    sc_annotations <- list(
      list(
        x = rng[1] + diff(rng) * 0.05,
        y = rng[2] - diff(rng) * 0.05,
        xref = "x", yref = "y",
        text = "↑ Overperforming Model",
        showarrow = FALSE,
        font = list(size = 11, color = "#047857", family = "Inter"),
        align = "left", xanchor = "left", yanchor = "top"
      ),
      list(
        x = rng[2] - diff(rng) * 0.05,
        y = rng[1] + diff(rng) * 0.05,
        xref = "x", yref = "y",
        text = "↓ Underperforming Model",
        showarrow = FALSE,
        font = list(size = 11, color = "#b91c1c", family = "Inter"),
        align = "right", xanchor = "right", yanchor = "bottom"
      )
    )
    
    p %>% layout(
      images      = logo_images,
      annotations = sc_annotations,
      xaxis     = list(title     = list(text = "Model Overall Rating (runs/game)",
                                        font = list(size = 12.5, color = "#475569")),
                       range     = rng, zeroline  = FALSE,
                       gridcolor = "#f1f5f9",
                       tickfont  = list(size = 11, color = "#94a3b8")),
      yaxis     = list(title     = list(text = "Actual Run Differential per Game",
                                        font = list(size = 12.5, color = "#475569")),
                       range     = rng, zeroline  = FALSE,
                       gridcolor = "#f1f5f9",
                       tickfont  = list(size = 11, color = "#94a3b8")),
      paper_bgcolor = "#ffffff",
      plot_bgcolor  = "#fafafa",
      margin        = list(t = 20, r = 20, b = 50, l = 60),
      font          = list(family = "Inter")
    ) %>%
      config(displayModeBar = FALSE)
  })
  
  # ---- Methodology ----
  output$bat_model_summary <- renderPrint({
    s <- summary(models$bat_model)
    cat("Weighted OLS (pooled ", paste(range(models$bat_pool_years), collapse = "–"),
        "): runs/PA ~ est_woba + bip_rate\n", sep = "")
    cat("Observations:", models$bat_pool_n, "\n")
    cat("R-squared:", round(s$r.squared, 4),
        " | Adj R²:", round(s$adj.r.squared, 4), "\n\n")
    print(s$coefficients)
  })
  
  output$pit_model_summary <- renderPrint({
    s <- summary(models$pit_model)
    cat("Weighted OLS (pooled ", paste(range(models$pit_pool_years), collapse = "–"),
        "): runs/PA ~ est_woba + xera\n", sep = "")
    cat("Observations:", models$pit_pool_n, "\n")
    cat("R-squared:", round(s$r.squared, 4),
        " | Adj R²:", round(s$adj.r.squared, 4), "\n\n")
    print(s$coefficients)
  })
  
  output$model_fit_summary <- renderPrint({
    rows <- list()
    for (yc in names(standings_check)) {
      sc <- standings_check[[yc]]
      rows[[yc]] <- data.frame(
        Season       = yc,
        Teams        = nrow(sc),
        Pearson_r    = round(cor(sc$overall, sc$rdiff_per_game), 3),
        Spearman_rho = round(cor(sc$overall, sc$rdiff_per_game, method = "spearman"), 3),
        RMSE         = round(sqrt(mean((sc$overall - sc$rdiff_per_game)^2)), 3)
      )
    }
    if (length(rows) == 0) {
      cat("No seasons with final standings available to validate against yet.\n")
      return(invisible())
    }
    df     <- do.call(rbind, rows)
    pooled <- do.call(rbind, standings_check)
    cat("Per-season fit vs. actual run differential per game:\n\n")
    print(df, row.names = FALSE)
    cat(sprintf(
      "\nPooled across all %d team-seasons: Pearson r = %.3f | Spearman rho = %.3f | RMSE = %.3f runs/game.\n",
      nrow(pooled),
      cor(pooled$overall, pooled$rdiff_per_game),
      cor(pooled$overall, pooled$rdiff_per_game, method = "spearman"),
      sqrt(mean((pooled$overall - pooled$rdiff_per_game)^2))
    ))
    cat("NOTE: the model is trained on player-level runs/PA only. Team records were never used as training data.\n")
  })
  
  # ============================================================================
  # Matchup Simulator — server
  # ============================================================================
  
  # Small helper: return logo img tag (lg or sm size) or a text placeholder.
  mp_logo_tag <- function(abbrev, size = c("lg", "sm")) {
    size <- match.arg(size)
    meta <- TEAM_META[TEAM_META$abbrev == abbrev, ]
    if (nrow(meta) == 0) return(tags$span(abbrev))
    if (is.na(meta$team_logo_espn[1])) {
      cls <- if (size == "lg") "mp-placeholder-lg" else "mp-placeholder-sm"
      return(tags$div(class = cls, abbrev))
    }
    cls <- if (size == "lg") "mp-team-logo-lg" else "mp-team-logo-sm"
    tags$img(src = meta$team_logo_espn[1], class = cls, alt = abbrev)
  }
  
  # Logo renderers for the selector cards (update reactively with team choice).
  output$mp_logo_a <- renderUI({ mp_logo_tag(req(input$mp_team_a), "lg") })
  output$mp_logo_b <- renderUI({ mp_logo_tag(req(input$mp_team_b), "lg") })
  
  # Populate pitcher dropdowns whenever team selection changes.
  observe({
    req(input$mp_team_a)
    ps <- mp_current_pitchers %>%
      dplyr::filter(abbrev == input$mp_team_a) %>%
      dplyr::arrange(dplyr::desc(pa_rv))
    ch <- setNames(as.character(ps$player_id), ps$player)
    pending <- pending_starter_a()
    selected <- if (!is.null(pending) && as.character(pending) %in% ch) {
      pending_starter_a(NULL)
      as.character(pending)
    } else if (length(ch) > 0) {
      if (!is.null(pending)) pending_starter_a(NULL)
      ch[[1]]
    } else {
      if (!is.null(pending)) pending_starter_a(NULL)
      character(0)
    }
    updateSelectizeInput(session, "mp_starter_a", choices = ch,
                         selected = selected)
  })
  
  observe({
    req(input$mp_team_b)
    ps <- mp_current_pitchers %>%
      dplyr::filter(abbrev == input$mp_team_b) %>%
      dplyr::arrange(dplyr::desc(pa_rv))
    ch <- setNames(as.character(ps$player_id), ps$player)
    pending <- pending_starter_b()
    selected <- if (!is.null(pending) && as.character(pending) %in% ch) {
      pending_starter_b(NULL)
      as.character(pending)
    } else if (length(ch) > 0) {
      if (!is.null(pending)) pending_starter_b(NULL)
      ch[[1]]
    } else {
      if (!is.null(pending)) pending_starter_b(NULL)
      character(0)
    }
    updateSelectizeInput(session, "mp_starter_b", choices = ch,
                         selected = selected)
  })

  observeEvent(input$mp_predict, {
    mp_generation_token(mp_generation_token() + 1L)
  }, ignoreInit = TRUE)

  observe({
    req(pending_matchup_generate())
    req(input$mp_team_a, input$mp_team_b, input$mp_starter_a, input$mp_starter_b)
    req(is.null(pending_starter_a()), is.null(pending_starter_b()))
    pending_matchup_generate(FALSE)
    mp_generation_token(mp_generation_token() + 1L)
  })
  
  # Core prediction (fires when button is clicked).
  mp_result <- eventReactive(mp_generation_token(), {
    req(input$mp_team_a, input$mp_team_b)
    
    tt <- current_team_tbl()
    ta <- tt %>% dplyr::filter(abbrev == input$mp_team_a)
    tb <- tt %>% dplyr::filter(abbrev == input$mp_team_b)
    
    pitcher_pool_a <- mp_current_pitchers %>%
      dplyr::filter(abbrev == input$mp_team_a) %>%
      dplyr::arrange(dplyr::desc(pa_rv))
    pitcher_pool_b <- mp_current_pitchers %>%
      dplyr::filter(abbrev == input$mp_team_b) %>%
      dplyr::arrange(dplyr::desc(pa_rv))
    
    starter_a_id <- suppressWarnings(as.integer(input$mp_starter_a))
    starter_b_id <- suppressWarnings(as.integer(input$mp_starter_b))
    if (is.na(starter_a_id) && nrow(pitcher_pool_a) > 0) starter_a_id <- pitcher_pool_a$player_id[1]
    if (is.na(starter_b_id) && nrow(pitcher_pool_b) > 0) starter_b_id <- pitcher_pool_b$player_id[1]
    
    sa <- pitcher_pool_a %>% dplyr::filter(player_id == starter_a_id)
    sb <- pitcher_pool_b %>% dplyr::filter(player_id == starter_b_id)
    
    if (nrow(ta) == 0 || nrow(tb) == 0 ||
        nrow(sa) == 0 || nrow(sb) == 0) return(NULL)
    
    LEAGUE_AVG <- 4.5   # typical MLB runs per team per 9-inning game
    SP_INN     <- 5
    TOT_INN    <- 9
    
    # Starter rating in runs/game units (positive = better, saves more runs).
    sa_rpg <- sa$final_rating[1] * PA_PER_GAME
    sb_rpg <- sb$final_rating[1] * PA_PER_GAME
    
    # Back-calculate bullpen rating from team PITCHING rate (fielding excluded, since
    # fielding value is shared by all pitchers and is added to the composite defense
    # separately below).
    calc_bp_rpg <- function(team_row, sp_pa, sp_rpg) {
      tot_pa <- team_row$total_pit_pa
      if (is.na(tot_pa) || tot_pa <= sp_pa || sp_pa <= 0)
        return(team_row$def_pitching)
      team_rate <- team_row$def_pitching / PA_PER_GAME
      sp_rate   <- sp_rpg / PA_PER_GAME
      bp_pa     <- tot_pa - sp_pa
      bp_rate   <- (team_rate * tot_pa - sp_pa * sp_rate) / bp_pa
      bp_rate * PA_PER_GAME
    }
    
    sa_bp_rpg <- calc_bp_rpg(ta, sa$pa_rv[1], sa_rpg)
    sb_bp_rpg <- calc_bp_rpg(tb, sb$pa_rv[1], sb_rpg)
    
    # Composite pitching contribution for this matchup (5 SP inn + 4 BP inn),
    # PLUS the opposing team's season fielding contribution (constant).
    tb_pitch_vs_a <- sb_rpg * (SP_INN / TOT_INN) + sb_bp_rpg * ((TOT_INN - SP_INN) / TOT_INN)
    ta_pitch_vs_b <- sa_rpg * (SP_INN / TOT_INN) + sa_bp_rpg * ((TOT_INN - SP_INN) / TOT_INN)
    
    tb_def_vs_a <- tb_pitch_vs_a + tb$def_fld[1]
    ta_def_vs_b <- ta_pitch_vs_b + ta$def_fld[1]
    
    # Expected runs: league_avg + own offense (hitting + BR) - opponent defense (pit + fld).
    exp_a <- max(0.5, min(20, LEAGUE_AVG + ta$off_rating[1] - tb_def_vs_a))
    exp_b <- max(0.5, min(20, LEAGUE_AVG + tb$off_rating[1] - ta_def_vs_b))
    
    # Score probability matrix via Negative Binomial approximation.
    # NB has variance = mu + mu^2/DISP_SIZE, capturing the over-dispersion
    # (heavier tails) observed in real MLB run distributions vs. Poisson.
    runs_seq  <- 0:13
    prob_a    <- dnbinom(runs_seq, mu = exp_a, size = DISP_SIZE)
    prob_b    <- dnbinom(runs_seq, mu = exp_b, size = DISP_SIZE)
    score_mat <- outer(prob_a, prob_b)
    # score_mat[i, j] = P(Team A scores runs_seq[i], Team B scores runs_seq[j])
    
    cmp_a_wins  <- outer(runs_seq, runs_seq, ">")
    cmp_b_wins  <- outer(runs_seq, runs_seq, "<")
    p_a_wins_9  <- sum(score_mat[cmp_a_wins])
    p_b_wins_9  <- sum(score_mat[cmp_b_wins])
    p_tied_9    <- sum(diag(score_mat))
    
    # Baseball has no ties. Model extra innings by distributing the tied-after-9
    # probability proportionally to each team's expected runs — the higher-scoring
    # team has a slight edge in extras (same logic a Pythagorean formula uses).
    extras_a <- exp_a / (exp_a + exp_b)
    extras_b <- 1 - extras_a
    p_a_wins <- p_a_wins_9 + p_tied_9 * extras_a
    p_b_wins <- p_b_wins_9 + p_tied_9 * extras_b
    p_tie    <- 0  # no ties in baseball
    
    top_scores <- expand.grid(runs_a = runs_seq, runs_b = runs_seq) %>%
      dplyr::mutate(prob = dnbinom(runs_a, mu = exp_a, size = DISP_SIZE) * dnbinom(runs_b, mu = exp_b, size = DISP_SIZE)) %>%
      dplyr::filter(runs_a != runs_b) %>%   # ties are impossible in baseball
      dplyr::arrange(dplyr::desc(prob)) %>%
      head(5)
    
    list(
      ta = ta, tb = tb, sa = sa, sb = sb,
      exp_a = exp_a, exp_b = exp_b,
      p_a_wins = p_a_wins, p_b_wins = p_b_wins, p_tie = p_tie,
      score_mat = score_mat, runs_seq = runs_seq,
      top_scores = top_scores,
      sa_rpg = sa_rpg, sb_rpg = sb_rpg,
      sa_bp_rpg = sa_bp_rpg, sb_bp_rpg = sb_bp_rpg,
      tb_pitch_vs_a = tb_pitch_vs_a, ta_pitch_vs_b = ta_pitch_vs_b,
      tb_def_vs_a   = tb_def_vs_a,   ta_def_vs_b   = ta_def_vs_b
    )
  }, ignoreNULL = TRUE)

  output$mp_share_button <- renderUI({
    res <- mp_result()
    if (is.null(res)) return(NULL)
    actionButton(
      "download_matchup_card_client",
      tagList(tags$i(class = "fa-solid fa-image"), "Share Matchup"),
      class = "xruns-share-btn xruns-share-btn-secondary",
      onclick = "xrunsDownloadClientShareCard('client_matchup_share_card')"
    )
  })

  output$client_matchup_share_card <- renderUI({
    res <- mp_result()
    if (is.null(res)) return(NULL)
    tags$div(
      `data-filename` = paste0(
        "xruns-matchup-",
        xruns_slug(res$ta$abbrev),
        "-vs-",
        xruns_slug(res$tb$abbrev),
        "-",
        current_year(),
        ".png"
      ),
      xruns_matchup_share_card(
        res = res,
        season = current_year(),
        base_url = xruns_current_base_url(session),
        data_label = current_share_data_label()
      )
    )
  })
  
  # Renders the full results panel (header, expected runs, win bar, pitchers, scores).
  mp_team_color <- function(row, fallback = "#151922") {
    col <- if ("team_color" %in% names(row)) suppressWarnings(as.character(row$team_color[1])) else NA_character_
    if ((is.na(col) || !grepl("^#[0-9A-Fa-f]{6}$", col)) && "abbrev" %in% names(row)) {
      idx <- match(row$abbrev[1], TEAM_META$abbrev)
      if (!is.na(idx) && "team_color" %in% names(TEAM_META)) {
        col <- suppressWarnings(as.character(TEAM_META$team_color[idx]))
      }
    }
    if (is.na(col) || !grepl("^#[0-9A-Fa-f]{6}$", col)) fallback else col
  }
  
  contrast_text <- function(hex) {
    hex <- gsub("#", "", hex)
    rgb <- grDevices::col2rgb(paste0("#", hex)) / 255
    rgb <- ifelse(rgb <= 0.03928, rgb / 12.92, ((rgb + 0.055) / 1.055) ^ 2.4)
    lum <- 0.2126 * rgb[1, ] + 0.7152 * rgb[2, ] + 0.0722 * rgb[3, ]
    ifelse(lum > 0.48, "#151922", "#ffffff")
  }
  
  output$mp_results <- renderUI({
    res <- mp_result()
    if (is.null(res)) return(NULL)
    
    ta <- res$ta; tb <- res$tb
    col_a <- mp_team_color(ta)
    col_b <- mp_team_color(tb, "#334155")
    txt_a <- contrast_text(col_a)
    txt_b <- contrast_text(col_b)
    pct_a <- round(res$p_a_wins * 100, 1)
    pct_b <- 100 - pct_a   # derived so the two always sum exactly to 100
    
    # Most likely scores rows.
    score_rows <- lapply(seq_len(nrow(res$top_scores)), function(i) {
      r      <- res$top_scores[i, ]
      winner <- if (r$runs_a > r$runs_b) ta$abbrev else tb$abbrev
      badge_col <- if (r$runs_a > r$runs_b) col_a else col_b
      badge_txt <- if (r$runs_a > r$runs_b) txt_a else txt_b
      tags$div(class = "mp-score-row",
               tags$span(class = "mp-score-label",
                         sprintf("%s  %d – %d  %s", ta$abbrev, r$runs_a, r$runs_b, tb$abbrev)),
               tags$div(style = "display:flex; align-items:center; gap:8px;",
                        tags$span(style = sprintf("font-size:11px; font-weight:700; color:%s;
                                     background:%s; border-radius:4px; padding:2px 7px;",
                                                  badge_txt, badge_col), winner),
                        tags$span(class = "mp-score-pct", sprintf("%.1f%%", r$prob * 100))
               )
      )
    })
    
    tags$div(
      class = "mp-results-wrap",
      style = "padding:0 20px 8px 20px;",
      
      # Header: logos + team names + season ratings.
      card(
        style = "margin-bottom:14px; border-top:4px solid #151922;",
        card_body(
          tags$div(class = "mp-header-row",
                   tags$div(class = "mp-team-block",
                            mp_logo_tag(ta$abbrev, "lg"),
                            tags$div(class = "mp-team-name-lg", style = sprintf("color:%s;", col_a), ta$team_name),
                            tags$div(class = "mp-sub-label",
                                     sprintf("Off %+.2f  |  Def %+.2f", ta$off_rating, ta$def_rating))
                   ),
                   tags$div(style = "font-size:1.6rem; font-weight:700; color:#cbd5e0;
                              padding:0 20px;", "VS"),
                   tags$div(class = "mp-team-block",
                            mp_logo_tag(tb$abbrev, "lg"),
                            tags$div(class = "mp-team-name-lg", style = sprintf("color:%s;", col_b), tb$team_name),
                            tags$div(class = "mp-sub-label",
                                     sprintf("Off %+.2f  |  Def %+.2f", tb$off_rating, tb$def_rating))
                   )
          )
        )
      ),
      
      # Expected runs side by side.
      layout_columns(
        col_widths = c(6, 6),
        card(
          style = sprintf("text-align:center; margin-bottom:14px; border-top:4px solid %s;", col_a),
          card_header(paste(ta$abbrev, "Expected Runs")),
          card_body(
            tags$div(class = "mp-exp-runs", style = sprintf("color:%s;", col_a), sprintf("%.2f", res$exp_a)),
            tags$div(style = "margin-top:10px; font-size:12px; color:#64748b; line-height:1.8;",
                     tags$div(sprintf("Offense (hit + BR): %+.2f rpg", ta$off_rating)),
                     tags$div(sprintf("Opp. defense (pit + fld): %+.2f rpg", -res$tb_def_vs_a))
            )
          )
        ),
        card(
          style = sprintf("text-align:center; margin-bottom:14px; border-top:4px solid %s;", col_b),
          card_header(paste(tb$abbrev, "Expected Runs")),
          card_body(
            tags$div(class = "mp-exp-runs", style = sprintf("color:%s;", col_b), sprintf("%.2f", res$exp_b)),
            tags$div(style = "margin-top:10px; font-size:12px; color:#64748b; line-height:1.8;",
                     tags$div(sprintf("Offense (hit + BR): %+.2f rpg", tb$off_rating)),
                     tags$div(sprintf("Opp. defense (pit + fld): %+.2f rpg", -res$ta_def_vs_b))
            )
          )
        )
      ),
      
      # Win probability bar.
      card(
        style = "margin-bottom:14px;",
        card_header("Win Probability"),
        card_body(
          tags$div(class = "mp-win-bar", style = "margin-bottom:8px;",
                   tags$div(class = "mp-win-seg",
                            style = sprintf("width:%.1f%%; background:%s; color:%s;", pct_a, col_a, txt_a),
                            if (pct_a >= 9) paste0(ta$abbrev, "  ", pct_a, "%") else ""),
                   tags$div(class = "mp-win-seg",
                            style = sprintf("width:%.1f%%; background:%s; color:%s;", pct_b, col_b, txt_b),
                            if (pct_b >= 9) paste0(pct_b, "%  ", tb$abbrev) else "")
          ),
          tags$div(style = "font-size:12px; color:#94a3b8; text-align:center;",
                   sprintf("%s: %.1f%%   |   %s: %.1f%%", ta$abbrev, pct_a, tb$abbrev, pct_b))
        )
      ),
      
      # Starting pitcher cards.
      card(
        style = "margin-bottom:14px;",
        card_header("Starting Pitchers"),
        card_body(
          layout_columns(
            col_widths = c(6, 6),
            tags$div(
              tags$div(class = "mp-sub-label", style = "margin-bottom:6px;",
                       ta$abbrev),
              tags$div(class = "mp-pitcher-chip", style = sprintf("border-left:4px solid %s;", col_a),
                       mp_logo_tag(ta$abbrev, "sm"),
                       tags$div(
                         tags$div(class = "mp-chip-text-main", res$sa$player[1]),
                         tags$div(class = "mp-chip-text-sub",
                                  sprintf("Pitching run value: %+.2f rpg  ·  %d PA",
                                          res$sa_rpg, res$sa$pa_rv[1]))
                       )
              )
            ),
            tags$div(
              tags$div(class = "mp-sub-label", style = "margin-bottom:6px;",
                       tb$abbrev),
              tags$div(class = "mp-pitcher-chip", style = sprintf("border-left:4px solid %s;", col_b),
                       mp_logo_tag(tb$abbrev, "sm"),
                       tags$div(
                         tags$div(class = "mp-chip-text-main", res$sb$player[1]),
                         tags$div(class = "mp-chip-text-sub",
                                  sprintf("Pitching run value: %+.2f rpg  ·  %d PA",
                                          res$sb_rpg, res$sb$pa_rv[1]))
                       )
              )
            )
          )
        )
      ),
      
      # Most likely final scores.
      card(
        card_header("Most Likely Final Scores"),
        card_body(
          tags$div(do.call(tagList, score_rows))
        )
      )
    )
  })
  
  # Score probability heatmap.
  output$mp_heatmap <- renderPlotly({
    res <- mp_result()
    if (is.null(res)) {
      return(
        plot_ly() %>%
          layout(
            xaxis = list(visible = FALSE), yaxis = list(visible = FALSE),
            annotations = list(list(
              x = 0.5, y = 0.5, xref = "paper", yref = "paper",
              text = "Select two teams and click Generate Matchup.",
              showarrow = FALSE,
              font = list(size = 14, color = "#94a3b8", family = "Inter")
            )),
            paper_bgcolor = "#ffffff", plot_bgcolor = "#fafafa"
          ) %>%
          config(displayModeBar = FALSE)
      )
    }
    
    ta <- res$ta; tb <- res$tb
    runs_seq  <- res$runs_seq
    score_mat <- res$score_mat
    
    # Zero out the diagonal — tied final scores are impossible in baseball;
    # that probability has already been redistributed to extra-innings outcomes.
    score_mat_plot <- score_mat
    diag(score_mat_plot) <- 0
    
    plot_ly(
      x = runs_seq,
      y = runs_seq,
      z = round(score_mat_plot * 100, 3),
      type      = "heatmap",
      colorscale = list(
        list(0,   "#f0f9ff"),
        list(0.25, "#bae6fd"),
        list(0.6,  "#0284c7"),
        list(1,    "#0c4a6e")
      ),
      hovertemplate = paste0(
        "<b>", ta$abbrev, " %{y} \u2013 %{x} ", tb$abbrev, "</b>",
        "<br>Probability: %{z:.3f}%<extra></extra>"
      ),
      showscale = TRUE,
      colorbar  = list(
        title     = list(text = "Prob (%)", font = list(size = 12, family = "Inter")),
        tickfont  = list(size = 11),
        thickness = 14
      )
    ) %>%
      layout(
        title  = list(
          text     = "Score Probability Grid",
          font     = list(size = 14, color = "#1a365d", family = "Inter"),
          x = 0.02, xanchor = "left"
        ),
        xaxis  = list(
          title    = list(text = paste(tb$abbrev, "Runs"),
                          font = list(size = 12.5, color = "#475569")),
          tickfont = list(size = 11, color = "#94a3b8"),
          gridcolor = "#f1f5f9"
        ),
        yaxis  = list(
          title    = list(text = paste(ta$abbrev, "Runs"),
                          font = list(size = 12.5, color = "#475569")),
          tickfont = list(size = 11, color = "#94a3b8"),
          gridcolor = "#f1f5f9"
        ),
        paper_bgcolor = "#ffffff",
        plot_bgcolor  = "#fafafa",
        font   = list(family = "Inter"),
        margin = list(t = 40, r = 20, b = 60, l = 60)
      ) %>%
      config(displayModeBar = FALSE)
  })
  
  # ============================================================================
  # Matchup Simulator — "Your Simulation" random draw
  # ============================================================================
  
  # Increments when the Re-roll button is clicked; mp_sim also re-runs when
  # mp_result() itself changes (new matchup), so no need to reset manually.
  sim_trigger <- reactiveVal(0L)
  observeEvent(input$mp_reroll, sim_trigger(sim_trigger() + 1L), ignoreInit = TRUE)
  
  mp_sim <- reactive({
    res <- mp_result()   # re-run when matchup changes
    sim_trigger()        # also re-run on re-roll
    if (is.null(res)) return(NULL)
    
    # Add log-normal noise to the expected run totals (22% std-dev on log scale).
    # Mean correction -σ²/2 keeps E[exp(noise)] = 1 so the draw is unbiased.
    noise_scale <- 0.22
    lambda_a <- res$exp_a * exp(rnorm(1, -noise_scale^2 / 2, noise_scale))
    lambda_b <- res$exp_b * exp(rnorm(1, -noise_scale^2 / 2, noise_scale))
    lambda_a <- max(lambda_a, 0.1)
    lambda_b <- max(lambda_b, 0.1)
    
    sim_a <- as.integer(rnbinom(1, mu = lambda_a, size = DISP_SIZE))
    sim_b <- as.integer(rnbinom(1, mu = lambda_b, size = DISP_SIZE))
    
    # No ties in baseball — resolve via proportional extra-innings coin flip.
    if (sim_a == sim_b) {
      if (runif(1) < res$exp_a / (res$exp_a + res$exp_b))
        sim_a <- sim_a + 1L
      else
        sim_b <- sim_b + 1L
    }
    
    # Look up base probability from the unperturbed score matrix.
    runs_seq <- res$runs_seq
    max_idx  <- max(runs_seq)
    sim_prob <- if (sim_a <= max_idx && sim_b <= max_idx)
      res$score_mat[sim_a + 1L, sim_b + 1L] * 100
    else
      NA_real_
    
    list(sim_a = sim_a, sim_b = sim_b, sim_prob = sim_prob)
  })
  
  output$mp_sim_card <- renderUI({
    res <- mp_result()
    if (is.null(res)) return(NULL)
    sim <- mp_sim()
    if (is.null(sim)) return(NULL)
    
    ta <- res$ta; tb <- res$tb
    col_a <- mp_team_color(ta)
    col_b <- mp_team_color(tb, "#334155")
    txt_a <- contrast_text(col_a)
    txt_b <- contrast_text(col_b)
    winner_name <- if (sim$sim_a > sim$sim_b) ta$team_name else tb$team_name
    badge_col   <- if (sim$sim_a > sim$sim_b) col_a else col_b
    badge_txt   <- if (sim$sim_a > sim$sim_b) txt_a else txt_b
    prob_txt    <- if (!is.na(sim$sim_prob))
      sprintf("Base probability: %.2f%%", sim$sim_prob)
    else
      "Low-probability outcome"
    
    tags$div(
      style = "padding:0 20px 14px 20px;",
      card(
        card_header(
          tags$div(
            style = "display:flex; align-items:center; justify-content:space-between;",
            tags$span(
              tags$i(class = "fa-solid fa-dice me-2"),
              tags$strong("Your Simulation")
            ),
            actionButton(
              "mp_reroll",
              label = tagList(tags$i(class = "fa-solid fa-rotate-right me-1"), "Re-roll"),
              class = "btn btn-outline-secondary btn-sm",
              style = "font-size:12px; padding:3px 12px; border-radius:6px;"
            )
          )
        ),
        card_body(
          tags$div(class = "mp-sim-score-block",
                   tags$div(class = "mp-sim-team",
                            mp_logo_tag(ta$abbrev, "lg"),
                            tags$div(class = "mp-sim-score", style = sprintf("color:%s;", col_a), sim$sim_a)
                   ),
                   tags$div(class = "mp-sim-dash", "\u2014"),
                   tags$div(class = "mp-sim-team",
                            mp_logo_tag(tb$abbrev, "lg"),
                            tags$div(class = "mp-sim-score", style = sprintf("color:%s;", col_b), sim$sim_b)
                   )
          ),
          tags$div(
            style = "display:flex; align-items:center; justify-content:center;
                     gap:10px; margin-top:6px; flex-wrap:wrap;",
            tags$span(
              style = sprintf(
                "font-size:12px; font-weight:700; color:%s; background:%s;
                 border-radius:4px; padding:3px 12px;",
                badge_txt, badge_col
              ),
              paste(winner_name, "Win")
            ),
            tags$span(class = "mp-sim-prob", style = "font-size:12px; color:#64748b;",
                      prob_txt)
          ),
          tags$div(class = "mp-sim-note",
                   "Sampled from the score probability model with added game-to-game variance. Re-roll for a new draw."
          )
        )
      )
    )
  })
  

  # ==========================================================================
  # Team Profile tab
  # ==========================================================================

  # Reactive: one-row slice of the current team table for the selected team
  selected_team_row <- reactive({
    tt  <- current_team_tbl()
    sel <- input$tb_team %||% "LAD"
    row <- tt %>% dplyr::filter(abbrev == sel)
    if (nrow(row) == 0) return(NULL)
    row[1, ]
  })

  output$client_team_share_card <- renderUI({
    sel <- input$tb_team %||% "LAD"
    tt <- team_tbl_for_period("season")
    row <- tt %>% dplyr::filter(abbrev == sel)
    if (is.null(row) || nrow(row) == 0) return(NULL)
    row <- row[1, ]
    rank_for <- function(tbl, metric) {
      if (is.null(tbl) || nrow(tbl) == 0 || !metric %in% names(tbl)) return(NA_integer_)
      rank_val <- which(dplyr::arrange(tbl, dplyr::desc(.data[[metric]]))$abbrev == row$abbrev[1])
      if (length(rank_val) == 0) NA_integer_ else rank_val
    }
    ranks <- list(
      overall = rank_for(tt, "overall"),
      offense = rank_for(tt, "off_rating"),
      pitching = rank_for(tt, "def_pitching"),
      fielding = rank_for(tt, "def_fld")
    )
    if (is.na(ranks$overall)) ranks$overall <- row$rank[1]
    recent_ranks <- list(
      rank_30d = rank_for(team_tbl_for_period("30d"), "overall"),
      rank_7d = rank_for(team_tbl_for_period("7d"), "overall")
    )
    pv <- current_players_view()
    team_players <- if (!is.null(pv)) pv %>% dplyr::filter(Team == row$abbrev[1]) else NULL
    if (!is.null(team_players) && nrow(team_players) > 0) {
      dual_player_ids <- team_players$player_id[team_players$Role == "Player"]
      team_players <- team_players %>%
        dplyr::filter(Role == "Player" | (Role %in% c("Hitter", "Pitcher") & !(player_id %in% dual_player_ids)))
      best_hitter <- team_players %>%
        dplyr::filter(!is.na(Hitting)) %>%
        dplyr::arrange(dplyr::desc(Hitting)) %>%
        dplyr::slice_head(n = 1)
      best_pitcher <- team_players %>%
        dplyr::filter(!is.na(Pitching)) %>%
        dplyr::arrange(dplyr::desc(Pitching)) %>%
        dplyr::slice_head(n = 1)
    } else {
      best_hitter <- NULL
      best_pitcher <- NULL
    }
    tags$div(
      `data-filename` = paste0("xruns-team-", xruns_slug(row$abbrev), "-", current_year(), ".png"),
      xruns_team_share_card(
        row = row,
        ranks = ranks,
        season = current_year(),
        base_url = xruns_current_base_url(session),
        data_label = current_share_data_label(),
        team_pool = tt,
        recent_ranks = recent_ranks,
        featured_players = list(hitter = best_hitter, pitcher = best_pitcher)
      )
    )
  })

  # Reactive: player view filtered to the selected team
  selected_team_players <- reactive({
    pv  <- current_players_view()
    sel <- input$tb_team %||% "LAD"
    if (is.null(pv)) return(NULL)
    pv %>% dplyr::filter(Team == sel)
  })

  # ---- Section 1: Team Header ----
  output$tb_header <- renderUI({
    row <- selected_team_row()
    tt  <- current_team_tbl()
    if (is.null(row)) return(NULL)

    rank_overall <- which(dplyr::arrange(tt, dplyr::desc(overall))$abbrev  == row$abbrev[1])
    rank_off     <- which(dplyr::arrange(tt, dplyr::desc(off_rating))$abbrev == row$abbrev[1])
    rank_def     <- which(dplyr::arrange(tt, dplyr::desc(def_rating))$abbrev == row$abbrev[1])
    n            <- nrow(tt)

    fmt_rank <- function(r, n) sprintf("#%d of %d", r, n)
    fmt_val  <- function(v) sprintf("%+.2f", v)

    pill <- function(label, val, rank, color) {
      tags$div(
        class = "tb-rating-pill",
        style = paste0(
          "display:flex; flex-direction:column; align-items:center; ",
          "background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; ",
          "padding:12px 20px; min-width:110px;"
        ),
        tags$span(style = "font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:.06em; color:#94a3b8;", label),
        tags$span(class = "tb-pill-value", style = paste0("font-size:1.55rem; font-weight:800; color:", color, "; line-height:1.15; margin-top:2px;"), fmt_val(val)),
        tags$span(style = "font-size:11px; color:#94a3b8; margin-top:2px;", fmt_rank(rank, n))
      )
    }

    val_color <- function(v) if (v >= 0) "#047857" else "#b91c1c"

    logo_tag <- if (!is.na(row$team_logo_espn))
      tags$div(class = "tb-header-logo",
               tags$img(src = row$team_logo_espn, style = "width:72px; height:72px; object-fit:contain;"))
    else
      tags$div(class = "tb-header-logo",
               tags$div(style = "width:72px; height:72px; background:#f1f5f9; border-radius:10px;"))

    tags$div(
      class = "tb-header-card",
      style = paste0(
        "display:flex; align-items:center; gap:24px; flex-wrap:wrap; ",
        "background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; ",
        "padding:20px 28px; margin-bottom:24px;"
      ),
      logo_tag,
      tags$div(
        class = "tb-header-title",
        style = "flex:1;",
        tags$div(style = "font-size:1.4rem; font-weight:800; color:#1e293b; line-height:1.2;", row$team_name),
        tags$div(style = "font-size:12px; color:#94a3b8; margin-top:3px;", current_year())
      ),
      tags$div(
        class = "tb-rating-pills",
        style = "display:flex; gap:12px; flex-wrap:wrap;",
        pill("Overall",  row$overall,     rank_overall, val_color(row$overall)),
        pill("Offense",  row$off_rating,  rank_off,     val_color(row$off_rating)),
        pill("Defense",  row$def_rating,  rank_def,     val_color(row$def_rating))
      )
    )
  })

  # ---- Section 2: Rating Decomposition Bar Chart ----
  output$tb_bar <- renderPlotly({
    row <- selected_team_row()
    if (is.null(row)) return(plot_ly() %>% layout(paper_bgcolor = "#ffffff", plot_bgcolor = "#fafafa"))

    # Fixed display order: Hitting, Baserunning, Pitching, Fielding
    components  <- c("Hitting", "Baserunning", "Pitching", "Fielding")
    values      <- c(row$off_hitting, row$off_br, row$def_pitching, row$def_fld)
    colors      <- ifelse(values >= 0, "#047857", "#b91c1c")
    label_txt   <- sprintf("%+.2f", values)

    hover_txt <- paste0(
      "<b>", components, "</b><br>",
      sprintf("%+.3f", values), " runs/game"
    )

    plot_ly(
      x            = components,
      y            = values,
      type         = "bar",
      marker       = list(color = colors, line = list(color = "#ffffff", width = 1)),
      text         = label_txt,
      textposition = "outside",
      cliponaxis   = FALSE,
      textfont     = list(size = 13, color = "#475569", family = "Inter"),
      hoverinfo    = "none",
      showlegend   = FALSE
    ) %>%
      layout(
        # Force category order so bars always appear L→R as specified
        xaxis = list(
          title         = NULL,
          categoryorder = "array",
          categoryarray = components,
          tickfont      = list(size = 13, color = "#475569", family = "Inter"),
          gridcolor     = "#f1f5f9"
        ),
        yaxis = list(
          title    = list(text = "runs/game above avg", font = list(size = 12, color = "#94a3b8")),
          tickfont = list(size = 11, color = "#94a3b8"),
          zeroline = FALSE, gridcolor = "#f1f5f9",
          # Pad based on the full data spread so small values still get headroom
          range = list(
            min(values) - max(diff(range(values)) * 0.22, 0.18),
            max(values) + max(diff(range(values)) * 0.22, 0.18)
          )
        ),
        shapes = list(list(
          type = "line", x0 = -0.5, x1 = 3.5, y0 = 0, y1 = 0,
          line = list(color = "#cbd5e0", width = 1, dash = "dot")
        )),
        paper_bgcolor = "#ffffff",
        plot_bgcolor  = "#fafafa",
        margin = list(t = 30, r = 20, b = 40, l = 60),
        font   = list(family = "Inter")
      ) %>%
      config(displayModeBar = FALSE)
  })

  # ---- Section 3: League Context — dot/lollipop chart ----
  # One chart with three traces (Overall, Offense, Defense). Each team is a
  # row on the y-axis, sorted by Overall. The selected team row is highlighted.
  output$tb_league_context <- renderPlotly({
    tt  <- current_team_tbl()
    sel <- input$tb_team %||% "LAD"

    is_sel    <- tt$abbrev == sel
    sel_color <- tt$team_color[is_sel]
    if (length(sel_color) == 0 || is.na(sel_color)) sel_color <- "#1e293b"

    muted_colors <- tt$team_color

    # Hover text: team name + the relevant rating
    hover_for <- function(metric_label, vals) {
      paste0(
        "<b>", tt$team_name, "</b><br>",
        metric_label, ":  ", sprintf("%+.2f", vals), " runs/game"
      )
    }

    # Y-axis categories (fixed order, top-to-bottom)
    cats <- c("Overall", "Offense", "Defense")

    # x values per category
    x_overall <- tt$overall
    x_offense <- tt$off_rating
    x_defense <- tt$def_rating

    # Build one trace per team so each dot can have its own color + opacity.
    # This is necessary because Plotly marker color arrays don't support
    # per-point opacity; we need separate traces for the selected team.
    #
    # Strategy: one trace for all non-selected teams (vectorised, fast),
    # one trace for the selected team on top.
    not_sel <- !is_sel

    # Flatten all three rows into long-format vectors for the non-selected trace
    x_bg <- c(x_overall[not_sel], x_offense[not_sel], x_defense[not_sel])
    y_bg <- c(rep("Overall", sum(not_sel)),
               rep("Offense", sum(not_sel)),
               rep("Defense", sum(not_sel)))
    col_bg    <- rep(muted_colors[not_sel], 3)
    hover_bg  <- c(
      hover_for("Overall", x_overall)[not_sel],
      hover_for("Offense", x_offense)[not_sel],
      hover_for("Defense", x_defense)[not_sel]
    )

    # Rows in display order (top → bottom): Overall, Offense, Defense
    row_labels <- c("Overall", "Offense", "Defense")

    p <- plot_ly() %>%
      # All non-selected teams — single vectorised trace with per-point colors
      add_trace(
        x          = x_bg,
        y          = factor(y_bg, levels = row_labels),
        type       = "scatter",
        mode       = "markers",
        marker     = list(
          size    = 9,
          color   = col_bg,
          opacity = 0.30,
          line    = list(width = 0)
        ),
        text      = hover_bg,
        hoverinfo = "text",
        hoverlabel = list(bgcolor = "#1e293b", bordercolor = "#334155",
                          font = list(color = "#f8fafc", family = "Inter", size = 13)),
        showlegend = FALSE,
        name       = "Other teams"
      ) %>%
      # Selected team — three points (one per row), full color, larger, no label
      add_trace(
        x    = c(x_overall[is_sel], x_offense[is_sel], x_defense[is_sel]),
        y    = factor(c("Overall", "Offense", "Defense"), levels = row_labels),
        type = "scatter",
        mode = "markers",
        marker = list(
          size  = 20,
          color = sel_color,
          line  = list(color = "#ffffff", width = 2.5)
        ),
        hovertext = c(
          hover_for("Overall", x_overall)[is_sel],
          hover_for("Offense", x_offense)[is_sel],
          hover_for("Defense", x_defense)[is_sel]
        ),
        hoverinfo  = "text",
        hoverlabel = list(bgcolor = "#1e293b", bordercolor = "#334155",
                          font = list(color = "#f8fafc", family = "Inter", size = 13)),
        showlegend = FALSE,
        name       = sel
      ) %>%
      layout(
        xaxis = list(
          title    = list(text = "runs/game above avg",
                          font = list(size = 12, color = "#94a3b8")),
          zeroline = FALSE, gridcolor = "#f1f5f9",
          tickfont = list(size = 11, color = "#94a3b8")
        ),
        yaxis = list(
          title    = NULL,
          type     = "category",
          categoryorder = "array",
          categoryarray = rev(row_labels),  # Overall on top
          tickfont = list(size = 13, color = "#1e293b", family = "Inter"),
          gridcolor = "#e2e8f0"
        ),
        # Zero reference line via shapes (avoids corrupting y-axis type)
        shapes = list(list(
          type = "line",
          x0 = 0, x1 = 0,
          y0 = -0.5, y1 = 2.5,
          yref = "y",
          line = list(color = "#cbd5e0", width = 1, dash = "dot")
        )),
        paper_bgcolor = "#ffffff",
        plot_bgcolor  = "#fafafa",
        margin = list(t = 20, r = 50, b = 50, l = 80),
        font   = list(family = "Inter")
      ) %>%
      config(displayModeBar = FALSE)

    p
  })

  # ---- Section 4: Roster Breakdown Table ----
  output$tb_players_table <- renderDT({
    pv  <- selected_team_players()
    yr  <- current_year()
    sel <- input$tb_team %||% "LAD"

    if (is.null(pv) || nrow(pv) == 0) {
      msg <- if (is.null(current_players_view()))
        sprintf("Individual player data for %s is not yet available.", yr)
      else
        sprintf("No player data found for %s in %s.", sel, yr)
      return(datatable(
        data.frame(Message = msg),
        rownames = FALSE,
        options  = list(dom = "t", pageLength = 1)
      ))
    }

    dual_player_ids <- pv$player_id[pv$Role == "Player"]

    pv_display <- pv %>%
      dplyr::filter(
        Role == "Player" | (Role %in% c("Hitter", "Pitcher") &
                              !(player_id %in% dual_player_ids))
      ) %>%
      dplyr::arrange(dplyr::desc(Overall)) %>%
      dplyr::mutate(Rank = dplyr::row_number()) %>%
      dplyr::select(Rank, Player, Overall, Role,
                    Hitting, Baserunning, Pitching, Fielding,
                    player_id)

    # Keep raw numeric copies for sorting BEFORE converting to HTML strings.
    # NA → -Inf so dashes sort below all real values.
    pv_display <- pv_display %>%
      dplyr::mutate(
        ovr_sort = dplyr::coalesce(Overall,     -Inf),
        hit_sort = dplyr::coalesce(Hitting,     -Inf),
        br_sort  = dplyr::coalesce(Baserunning, -Inf),
        pit_sort = dplyr::coalesce(Pitching,    -Inf),
        fld_sort = dplyr::coalesce(Fielding,    -Inf),
        player_id_ = player_id
      )

    fmt_signed <- function(x) {
      ifelse(is.na(x), "—",
             ifelse(x >= 0,
                    sprintf('<span style="color:#047857">%+.2f</span>', x),
                    sprintf('<span style="color:#b91c1c">%+.2f</span>', x)))
    }

    pv_display <- pv_display %>%
      dplyr::mutate(
        Overall     = fmt_signed(Overall),
        Hitting     = fmt_signed(Hitting),
        Baserunning = fmt_signed(Baserunning),
        Pitching    = fmt_signed(Pitching),
        Fielding    = fmt_signed(Fielding)
      ) %>%
      dplyr::select(-player_id)

    # Column indices (0-indexed):
    #  0 Rank | 1 Player | 2 Overall | 3 Role |
    #  4 Hitting | 5 Baserunning | 6 Pitching | 7 Fielding
    #  8 ovr_sort | 9 hit_sort | 10 br_sort | 11 pit_sort | 12 fld_sort
    #  13 player_id_ (hidden, row-click navigation)
    centered_cols <- c(0, 2, 3, 4, 5, 6, 7)
    run_val_cols  <- c(2, 4, 5, 6, 7)

    roster_row_click_js <- JS(
      "function(row, data, index) {",
      "  $(row).css('cursor', 'pointer');",
      "  $(row).on('click', function(e) {",
      "    Shiny.setInputValue('player_table_row_clicked', data[13], {priority: 'event'});",
      "  });",
      "}"
    )

    datatable(
      pv_display,
      rownames   = FALSE,
      escape     = FALSE,
      class      = "compact stripe hover",
      options    = list(
        dom        = "tp",
        pageLength = 25,
        ordering   = TRUE,
        scrollX    = TRUE,
        # Default: sort by Overall (col 2, via hidden sort col 8) descending
        order      = list(list(8, "desc")),
        rowCallback = roster_row_click_js,
        columnDefs = list(
          list(className      = "dt-center", targets = centered_cols),
          # Run value cols: first click sorts descending
          list(orderSequence = list("desc", "asc"), targets = run_val_cols),
          # Rank col: first click sorts ascending
          list(orderSequence = list("asc", "desc"),  targets = 0),
          # Point each HTML column to its hidden numeric sort twin
          list(orderData = 8,  targets = 2),
          list(orderData = 9,  targets = 4),
          list(orderData = 10, targets = 5),
          list(orderData = 11, targets = 6),
          list(orderData = 12, targets = 7),
          # Hide the sort helper columns
          list(visible = FALSE, targets = c(8, 9, 10, 11, 12, 13))
        )
      ),
      caption = tags$caption(
        style = "color:#94a3b8; font-size:11.5px; text-align:left; padding:6px 0;",
        "Only qualified players are shown (minimum PA / BIP thresholds apply)."
      )
    ) %>%
      formatStyle(columns = seq_len(ncol(pv_display)),
                  fontSize = "13.5px", lineHeight = "1.4") %>%
      formatStyle("Rank",   fontWeight = "bold",  color = "#94a3b8") %>%
      formatStyle("Player", fontWeight = "600",   color = "#1e293b") %>%
      formatStyle("Role",   color = "#64748b",    fontStyle = "italic")
  })

}

shinyApp(ui, server)
