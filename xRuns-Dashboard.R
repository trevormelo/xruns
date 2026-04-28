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

# Try to pull team primary colors from mlbplotR; fall back to navy if unavailable.
mlb_colors <- tryCatch({
  mlbplotR::load_mlb_teams() |>
    dplyr::select(team_abbr, team_color)
}, error = function(e) NULL)

if (!is.null(mlb_colors)) {
  TEAM_META <- TEAM_META %>%
    dplyr::left_join(mlb_colors, by = c("mlb_abbrev" = "team_abbr"))
} else {
  TEAM_META$team_color <- "#1a365d"
}

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
      fld_runs    = dplyr::coalesce(fld_runs, 0),
      fld_outs    = dplyr::coalesce(fld_outs, 0),
      fld_per_out = ifelse(fld_outs > 0, fld_runs / fld_outs, 0)
    )
  pitchers <- pitchers %>%
    mutate(
      fld_runs    = dplyr::coalesce(fld_runs, 0),
      fld_outs    = dplyr::coalesce(fld_outs, 0),
      fld_per_out = ifelse(fld_outs > 0, fld_runs / fld_outs, 0)
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
      off_br      = off_br_rate  * PA_PER_GAME,
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
                       0)
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
      off_br    = ifelse(pa_denom > 0, (br_runs / pa_denom) * PA_PER_GAME, 0)
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
      )
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
      Baserunning = br_per_pa * PA_PER_GAME,
      Pitching    = NA_real_,
      Fielding    = fld_per_out * OUTS_PER_GAME
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
      Fielding    = fld_per_out * OUTS_PER_GAME
    )

  dplyr::bind_rows(bat_rows, pit_rows) %>%
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

# Compact heading+picker row: heading left, picker chip right
heading_with_picker <- function(id_suffix, heading_output_id) {
  tags$div(
    class = "xruns-heading-row",
    tags$div(
      class = "xruns-heading-text",
      textOutput(heading_output_id, inline = TRUE)
    ),
    tags$div(
      class = "xruns-season-chip",
      season_picker_inline(id_suffix)
    )
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

season_selector_row <- function(id_suffix, label_text) {
  tags$div(
    class = "xruns-season-row",
    style = "display:flex; align-items:center; gap:10px; flex-wrap:wrap;
             padding: 4px 20px 14px 20px;",
    tags$span(style = "font-size:13px; color:#64748b; font-weight:500;", label_text),
    season_picker_inline(id_suffix),
    tags$span(style = "font-size:12px; color:#94a3b8;", "switch to view a different season")
  )
}

# ---- theme -------------------------------------------------------------------
app_theme <- bs_theme(
  version      = 5,
  bootswatch   = "litera",
  primary      = "#1a365d",
  "link-color" = "#2b6cb0",
  base_font    = font_google("Inter"),
  heading_font = font_google("Inter")
)

# ---- custom CSS --------------------------------------------------------------
custom_css <- HTML("
  /* ---- Base & typography ---- */
  body { background: #f8fafc; color: #1e293b; }

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

  /* ---- Desktop (≥768px) ---- */
  @media (min-width: 768px) {
    .kpi-row { flex-wrap: nowrap; overflow-x: visible; }
  }

  /* ---- Mobile (≤640px): stack matchup cards vertically ---- */
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

  /* ---- Small mobile (≤576px) ---- */
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
    tags$span(class = "d-none d-md-inline",
      "xRuns",
      tags$span(class = "brand-sub", "an MLB Rating System")
    ),
    tags$span(class = "d-inline d-md-none", "xRuns")
  ),
  window_title = "xRuns: an MLB Rating System",
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
      tags$style(custom_css),
      # JS: handle year-change reset of time period chips back to Season
      tags$script(HTML("
        Shiny.addCustomMessageHandler('resetTimePeriod', function(msg) {
          Shiny.setInputValue('time_period', 'season', {priority: 'event'});
          document.querySelectorAll('.xruns-period-chip').forEach(function(el) {
            el.classList.remove('active');
            if (el.textContent.trim() === 'Season') el.classList.add('active');
          });
        });
      "))
    )
  ),

  # ---- Tab: Team Rankings ----
  nav_panel(
    title = tagList(tags$i(class = "fa-solid fa-ranking-star me-1"), "Team Rankings"),
    # Info banner — collapsible, only on this tab
    tags$div(
      style = "margin: 10px 20px 0 20px;",
      tags$details(
        style = "background:#ffffff; border-left:4px solid #1a365d; border-radius:0 8px 8px 0;
                 box-shadow:0 1px 3px rgba(0,0,0,0.05); font-size:13px; color:#334155; line-height:1.65;",
        tags$summary(
          style = "cursor:pointer; padding:10px 16px; font-weight:600; font-size:13px;
                   color:#1a365d; list-style:none; display:flex; align-items:center; gap:6px;
                   user-select:none;",
          tags$span(style="font-size:11px; color:#94a3b8;", "▼"),
          "How to read the ratings"
        ),
        tags$div(
          style = "padding:4px 16px 12px 16px;",
          tags$b("Overall"), " = expected run margin vs. an average MLB team in a 9-inning game. ",
          tags$b("Offense"), " = hitting + baserunning runs scored above average per game. ",
          tags$b("Defense"), " = pitching + fielding runs prevented above average per game.",
          tags$br(),
          tags$span(style = "color:#64748b; font-size:12px;",
            "Regression model trained on ",
            tags$span(class = "info-stat", textOutput("pool_bat_n", inline = TRUE)),
            " batter-seasons and ",
            tags$span(class = "info-stat", textOutput("pool_pit_n", inline = TRUE)),
            " pitcher-seasons (2022–", most_recent_player_yr, "). ",
            "Current season (", DEFAULT_YEAR, ") ratings use team-level expected stats applied through that model; ",
            "baserunning and fielding are from current-year Statcast run-value data. ",
            "Ratings are centered at 0 = league average each year."
          )
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
      tags$div(class = "xruns-season-chip", season_picker_inline("team"))
    ),

    # Fallback / window note (shown when time period filter uses graceful fallback)
    uiOutput("window_note_ui"),

    # KPI cards (hidden on mobile via CSS)
    uiOutput("kpi_row"),

    # Table
    tags$div(class = "tab-body", DTOutput("team_table", height = "auto")),

    # Scatter
    tags$div(class = "section-heading",
      "Offense vs. Defense",
      tags$span(class = "section-subheading", "runs/game above average")
    ),
    tags$div(class = "tab-body", plotlyOutput("team_scatter", height = "620px"))
  ),

  # ---- Tab: Matchup Simulator ----
  nav_panel(
    title = tagList(tags$i(class = "fa-solid fa-bolt me-1"), "Matchup Simulator"),

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
        font-weight: 600;
        color: #1a365d;
        background: #f0f7ff;
        border: 2px solid #3b82f6;
        border-radius: 6px;
        padding: 7px 10px;
        cursor: pointer;
      }
      #mp_starter_a .selectize-input:hover,
      #mp_starter_b .selectize-input:hover {
        border-color: #1d4ed8;
        background: #dbeafe;
      }
      #mp_starter_a .selectize-dropdown,
      #mp_starter_b .selectize-dropdown {
        font-size: 13px;
        border: 1px solid #3b82f6;
        border-radius: 6px;
        box-shadow: 0 4px 12px rgba(59,130,246,0.15);
      }
      #mp_starter_a .selectize-dropdown-content .option,
      #mp_starter_b .selectize-dropdown-content .option {
        padding: 8px 12px;
        border-bottom: 1px solid #f1f5f9;
      }
      #mp_starter_a .selectize-dropdown-content .option.active,
      #mp_starter_b .selectize-dropdown-content .option.active {
        background: #dbeafe;
        color: #1a365d;
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

    tags$div(class = "section-heading", "Matchup Simulator"),
    tags$div(
      style = "padding:2px 20px 10px 20px; color:#64748b; font-size:13px; line-height:1.5;",
      "Select two teams and their starting pitchers to project a matchup result."
    ),

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
                         options = list(placeholder = "Select pitcher…",
                                        dropdownParent = "body"))
        )
      ),
      # VS divider
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
                         options = list(placeholder = "Select pitcher…",
                                        dropdownParent = "body"))
        )
      )
    ),

    tags$div(
      style = "padding:4px 20px 16px 20px;",
      actionButton("mp_predict", "Generate Matchup",
                   class = "btn btn-primary",
                   style = "font-weight:600; padding:8px 28px; font-size:13px;
                            border-radius:8px;")
    ),

    # ---- Results (rendered server-side) ----
    uiOutput("mp_results"),

    # ---- Simulation result card (re-rolls independently) ----
    uiOutput("mp_sim_card"),

    tags$div(
      style = "padding:0 20px 28px 20px;",
      plotlyOutput("mp_heatmap", height = "500px")
    )
  ),

  # ---- Tab: Player Rankings ----
  nav_panel(
    title = tagList(tags$i(class = "fa-solid fa-baseball-bat-ball me-1"), "Player Rankings"),
    # Player ratings collapsible banner — mirrors the team ratings banner style
    tags$div(
      style = "margin: 10px 20px 0 20px;",
      tags$details(
        style = "background:#ffffff; border-left:4px solid #1a365d; border-radius:0 8px 8px 0;
                 box-shadow:0 1px 3px rgba(0,0,0,0.05); font-size:13px; color:#334155; line-height:1.65;",
        tags$summary(
          style = "cursor:pointer; padding:10px 16px; font-weight:600; font-size:13px;
                   color:#1a365d; list-style:none; display:flex; align-items:center; gap:6px;
                   user-select:none;",
          tags$span(style="font-size:11px; color:#94a3b8;", "▼"),
          "How to read the player ratings"
        ),
        tags$div(
          style = "padding:4px 16px 12px 16px;",
          "Each column measures how many extra runs per 9 innings that player contributes ",
          "compared to a league-average player at that skill. For example, a ",
          tags$b("Hitting"), " rating of +0.50 means that player generates 0.50 more runs ",
          "per 9 innings than average through their hitting alone. A value of +1.0 is elite; ",
          "most qualified players fall between −1.0 and +1.0. ",
          tags$b("Overall"), " = Hitting + Baserunning + Pitching + Fielding — the total ",
          "runs per 9 innings that player adds above an average player across all facets of the game."
        )
      )
    ),
    heading_with_filter_picker("players", "players_heading"),
    tags$div(class = "tab-body", DTOutput("players_table"))
  ),

  # ---- Tab: Methodology ----
  nav_panel(
    title = tagList(tags$i(class = "fa-solid fa-flask me-1"), "Methodology"),
    tags$div(style = "padding: 20px;",
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
                 " both already in 'runs above average' units). For the current season,",
                 " actual year-to-date team run values are used directly.",
                 " Offense = hitting + baserunning;",
                 " Defense = pitching + fielding. Team baserunning is normalized to runs per",
                 " team-PA \u00d7 38 PA/game; team fielding is normalized to runs per defensive",
                 " fielder-out \u00d7 243 fielder-outs/game."),
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
      tags$div(style = "margin-top:16px;", plotlyOutput("standings_scatter", height = "580px"))
    )
  )
)

# =============================================================================
# Server
# =============================================================================
server <- function(input, output, session) {

  # Pool stats for the info banner
  output$pool_bat_n <- renderText(format(models$bat_pool_n, big.mark = ","))
  output$pool_pit_n <- renderText(format(models$pit_pool_n, big.mark = ","))

  # ---- Season picker sync ----
  picker_ids     <- c("season_year_team", "season_year_players",
                      "season_year_sc")
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

  # ---- Team table reactive — responds to year AND time period ----------------
  current_team_tbl <- reactive({
    yc <- current_year()
    tp <- input$time_period %||% "season"

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
  })

  # ---- Headings ----
  output$team_heading    <- renderText(sprintf("%s Team Efficiency Ratings", current_year()))
  output$players_heading <- renderText(sprintf("%s Player Leaderboard", current_year()))
  output$sc_heading      <- renderText(sprintf("%s — Model vs. Actual Standings", current_year()))

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
      make_card("\U0001F3C6 Best Overall",  top_overall,
                sprintf("%+.2f runs/game", top_overall$overall)),
      make_card("\U26BE Best Offense",      top_off,
                sprintf("%+.2f runs/game", top_off$off_rating)),
      make_card("\U26BE Best Pitching",     top_pit,
                sprintf("%+.2f runs/game", top_pit$def_pitching)),
      make_card("\U0001F6E1 Best Fielding", top_fld,
                sprintf("%+.2f runs/game", top_fld$def_fld))
    )
  })

  # ---- Signed render JS for DT ----
  signed_render <- JS(
    "function(data, type, row) {",
    "  if (type === 'display' || type === 'filter') {",
    "    if (data === null || data === undefined || data === '' || isNaN(data)) return '';",
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
    n_teams <- nrow(tt)
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
                 '" height="26" style="vertical-align:middle; margin-right:2px;">'),
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
        Team     = team_name,
        Overall  = round(overall, 4),
        Offense,
        Pitching,
        Fielding,
        off_sort,
        pit_sort,
        fld_sort
      )

    # Column indices (0-indexed for DT columnDefs):
    #   0 Rank  1 Logo  2 Team  3 Overall  4 Offense  5 Pitching  6 Fielding
    #   7 off_sort (hidden)  8 pit_sort (hidden)  9 fld_sort (hidden)
    centered_cols  <- c(0, 3, 4, 5, 6)

    datatable(
      dt,
      rownames = FALSE,
      escape   = FALSE,
      class    = "compact stripe hover",
      options  = list(
        pageLength = 30,
        dom        = "t",
        # Default sort: Overall descending (column index 3)
        order      = list(list(3, "desc")),
        autoWidth  = FALSE,
        scrollX    = TRUE,
        columnDefs = list(
          list(targets = 3,             render = signed_render),
          list(className = "dt-center", targets = centered_cols),
          list(orderable = FALSE,       targets = 1),
          list(width = "44px",          targets = 1),
          # Run value columns: clicking sorts descending first (higher = better)
          list(targets = c(3, 4, 5, 6), orderSequence = list("desc", "asc")),
          # Rank column: clicking sorts ascending first (lower number = better)
          list(targets = 0, orderSequence = list("asc", "desc")),
          # Point HTML display columns at their hidden numeric sort columns
          list(targets = 4, orderData = 7),
          list(targets = 5, orderData = 8),
          list(targets = 6, orderData = 9),
          # Hide the numeric sort helper columns
          list(targets = c(7, 8, 9), visible = FALSE)
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
      "<br>Overall: ", sprintf("%+.2f", tt$overall),
      "<br>Offense: ", sprintf("%+.2f", tt$off_rating),
      "  (Hit ",       sprintf("%+.2f", tt$off_hitting),
      " + BR ",        sprintf("%+.2f", tt$off_br), ")",
      "<br>Defense: ", sprintf("%+.2f", tt$def_rating),
      "  (Pit ",       sprintf("%+.2f", tt$def_pitching),
      " + Fld ",       sprintf("%+.2f", tt$def_fld), ")"
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
      # Invisible markers for hover tooltips
      add_trace(
        x           = x_vals,
        y           = y_vals,
        type        = "scatter",
        mode        = "markers",
        marker      = list(size = 32, color = "rgba(0,0,0,0)"),
        text        = hover_txt,
        hoverinfo   = "text",
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

    pv <- pv %>%
      dplyr::filter(Role %in% roles_selected) %>%
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
        )
      ) %>%
      dplyr::select(Rank, Logo, Player, Team, Role, PA,
                    Hitting, Baserunning, Pitching, Fielding, Overall)

    # Column indices (0-indexed): Rank=0, Logo=1, Player=2, Team=3, Role=4, PA=5,
    #   Hitting=6, Baserunning=7, Pitching=8, Fielding=9, Overall=10
    signed_cols   <- c(6, 7, 8, 9, 10)
    centered_cols <- c(0, 3, 4, 5, 6, 7, 8, 9, 10)

    datatable(
      pv,
      rownames = FALSE,
      escape   = FALSE,
      class    = "compact stripe hover",
      options  = list(
        pageLength  = 25,
        # Default sort: Overall descending (column index 10)
        order       = list(list(10, "desc")),
        scrollX     = TRUE,
        dom         = "frtip",
        columnDefs  = list(
          list(orderable = FALSE, targets = 1),
          list(width = "36px",    targets = 1),
          list(className = "dt-center", targets = centered_cols),
          list(targets = signed_cols, render = signed_render),
          # Run value columns sort descending first (higher = better)
          list(targets = c(6, 7, 8, 9, 10), orderSequence = list("desc", "asc")),
          # Rank sorts ascending first (lower = better)
          list(targets = 0, orderSequence = list("asc", "desc"))
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
  })

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
      "<br>Model Rating: ", sprintf("%+.2f", sc$overall),
      "<br>Actual RDiff/G: ", sprintf("%+.2f", sc$rdiff_per_game),
      "<br>Record: ", sc$`Actual W`, "–", sc$`Actual L`
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
        marker     = list(size = 32, color = "rgba(0,0,0,0)"),
        text       = hover_txt,
        hoverinfo  = "text",
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
    updateSelectInput(session, "mp_starter_a", choices = ch,
                      selected = if (length(ch) > 0) ch[[1]] else NULL)
  })

  observe({
    req(input$mp_team_b)
    ps <- mp_current_pitchers %>%
      dplyr::filter(abbrev == input$mp_team_b) %>%
      dplyr::arrange(dplyr::desc(pa_rv))
    ch <- setNames(as.character(ps$player_id), ps$player)
    updateSelectInput(session, "mp_starter_b", choices = ch,
                      selected = if (length(ch) > 0) ch[[1]] else NULL)
  })

  # Core prediction (fires when button is clicked).
  mp_result <- eventReactive(input$mp_predict, {
    req(input$mp_team_a, input$mp_team_b,
        input$mp_starter_a, input$mp_starter_b)

    tt <- current_team_tbl()
    ta <- tt %>% dplyr::filter(abbrev == input$mp_team_a)
    tb <- tt %>% dplyr::filter(abbrev == input$mp_team_b)

    sa <- mp_current_pitchers %>%
      dplyr::filter(player_id == suppressWarnings(as.integer(input$mp_starter_a)),
                    abbrev    == input$mp_team_a)
    sb <- mp_current_pitchers %>%
      dplyr::filter(player_id == suppressWarnings(as.integer(input$mp_starter_b)),
                    abbrev    == input$mp_team_b)

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

  # Renders the full results panel (header, expected runs, win bar, pitchers, scores).
  output$mp_results <- renderUI({
    res <- mp_result()
    if (is.null(res)) return(NULL)

    ta <- res$ta; tb <- res$tb
    pct_a <- round(res$p_a_wins * 100, 1)
    pct_b <- 100 - pct_a   # derived so the two always sum exactly to 100
    pct_t <- round(res$p_tie    * 100, 1)

    # Most likely scores rows.
    score_rows <- lapply(seq_len(nrow(res$top_scores)), function(i) {
      r      <- res$top_scores[i, ]
      winner <- if (r$runs_a > r$runs_b) ta$abbrev else tb$abbrev
      badge_col <- if (r$runs_a > r$runs_b) "#1a365d" else "#b91c1c"
      tags$div(class = "mp-score-row",
        tags$span(class = "mp-score-label",
          sprintf("%s  %d – %d  %s", ta$abbrev, r$runs_a, r$runs_b, tb$abbrev)),
        tags$div(style = "display:flex; align-items:center; gap:8px;",
          tags$span(style = sprintf("font-size:11px; font-weight:700; color:#fff;
                                     background:%s; border-radius:4px; padding:2px 7px;",
                                    badge_col), winner),
          tags$span(class = "mp-score-pct", sprintf("%.1f%%", r$prob * 100))
        )
      )
    })

    tags$div(
      style = "padding:0 20px 8px 20px;",

      # Header: logos + team names + season ratings.
      card(
        style = "margin-bottom:14px;",
        card_body(
          tags$div(class = "mp-header-row",
            tags$div(class = "mp-team-block",
              mp_logo_tag(ta$abbrev, "lg"),
              tags$div(class = "mp-team-name-lg", ta$team_name),
              tags$div(class = "mp-sub-label",
                sprintf("Off %+.2f  |  Def %+.2f", ta$off_rating, ta$def_rating))
            ),
            tags$div(style = "font-size:1.6rem; font-weight:700; color:#cbd5e0;
                              padding:0 20px;", "VS"),
            tags$div(class = "mp-team-block",
              mp_logo_tag(tb$abbrev, "lg"),
              tags$div(class = "mp-team-name-lg", tb$team_name),
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
          style = "text-align:center; margin-bottom:14px;",
          card_header(paste(ta$abbrev, "Expected Runs")),
          card_body(
            tags$div(class = "mp-exp-runs", sprintf("%.2f", res$exp_a)),
            tags$div(style = "margin-top:10px; font-size:12px; color:#64748b; line-height:1.8;",
              tags$div(sprintf("Offense (hit + BR): %+.2f rpg", ta$off_rating)),
              tags$div(sprintf("Opp. defense (pit + fld): %+.2f rpg", -res$tb_def_vs_a))
            )
          )
        ),
        card(
          style = "text-align:center; margin-bottom:14px;",
          card_header(paste(tb$abbrev, "Expected Runs")),
          card_body(
            tags$div(class = "mp-exp-runs", sprintf("%.2f", res$exp_b)),
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
              style = sprintf("width:%.1f%%; background:#1a365d;", pct_a),
              if (pct_a >= 9) paste0(ta$abbrev, "  ", pct_a, "%") else ""),
            tags$div(class = "mp-win-seg",
              style = sprintf("width:%.1f%%; background:#b91c1c;", pct_b),
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
              tags$div(class = "mp-pitcher-chip",
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
              tags$div(class = "mp-pitcher-chip",
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
    winner_name <- if (sim$sim_a > sim$sim_b) ta$team_name else tb$team_name
    badge_col   <- if (sim$sim_a > sim$sim_b) "#1a365d" else "#b91c1c"
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
              tags$div(class = "mp-sim-score", sim$sim_a)
            ),
            tags$div(class = "mp-sim-dash", "\u2014"),
            tags$div(class = "mp-sim-team",
              mp_logo_tag(tb$abbrev, "lg"),
              tags$div(class = "mp-sim-score", sim$sim_b)
            )
          ),
          tags$div(
            style = "display:flex; align-items:center; justify-content:center;
                     gap:10px; margin-top:6px; flex-wrap:wrap;",
            tags$span(
              style = sprintf(
                "font-size:12px; font-weight:700; color:#fff; background:%s;
                 border-radius:4px; padding:3px 12px;",
                badge_col
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

}

shinyApp(ui, server)
