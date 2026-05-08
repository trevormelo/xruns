suppressPackageStartupMessages({
  library(dplyr)
  library(jsonlite)
  library(purrr)
  library(readr)
  library(tibble)
})

`%||%` <- function(a, b) if (!is.null(a)) a else b

season <- 2026L
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- if (length(script_arg) > 0) sub("^--file=", "", script_arg[1]) else "."
repo_root <- normalizePath(file.path(dirname(script_path), ".."), mustWork = FALSE)
if (!dir.exists(file.path(repo_root, "2026 Data", "Snapshots"))) {
  repo_root <- normalizePath(getwd(), mustWork = TRUE)
}

snapshot_dir <- file.path(repo_root, sprintf("%d Data", season), "Snapshots")
cache_path <- file.path(repo_root, sprintf("%d Data", season), sprintf("standings_daily_%d.csv", season))

team_lookup <- tibble::tribble(
  ~team_id, ~abbrev, ~team_name,
  108L, "LAA", "Los Angeles Angels",
  109L, "ARI", "Arizona Diamondbacks",
  110L, "BAL", "Baltimore Orioles",
  111L, "BOS", "Boston Red Sox",
  112L, "CHC", "Chicago Cubs",
  113L, "CIN", "Cincinnati Reds",
  114L, "CLE", "Cleveland Guardians",
  115L, "COL", "Colorado Rockies",
  116L, "DET", "Detroit Tigers",
  117L, "HOU", "Houston Astros",
  118L, "KC",  "Kansas City Royals",
  119L, "LAD", "Los Angeles Dodgers",
  120L, "WSH", "Washington Nationals",
  121L, "NYM", "New York Mets",
  133L, "OAK", "Athletics",
  134L, "PIT", "Pittsburgh Pirates",
  135L, "SD",  "San Diego Padres",
  136L, "SEA", "Seattle Mariners",
  137L, "SF",  "San Francisco Giants",
  138L, "STL", "St. Louis Cardinals",
  139L, "TB",  "Tampa Bay Rays",
  140L, "TEX", "Texas Rangers",
  141L, "TOR", "Toronto Blue Jays",
  142L, "MIN", "Minnesota Twins",
  143L, "PHI", "Philadelphia Phillies",
  144L, "ATL", "Atlanta Braves",
  145L, "CWS", "Chicago White Sox",
  146L, "MIA", "Miami Marlins",
  147L, "NYY", "New York Yankees",
  158L, "MIL", "Milwaukee Brewers"
)

snapshot_dates <- list.files(
  snapshot_dir,
  pattern = sprintf("^bat_t_%d-[0-9]{2}-[0-9]{2}\\.csv$", season),
  full.names = FALSE
) %>%
  sub("^bat_t_", "", .) %>%
  sub("\\.csv$", "", .) %>%
  as.Date() %>%
  sort()

if (length(snapshot_dates) == 0) {
  stop("No snapshot dates found in ", snapshot_dir)
}

fetch_standings_for_date <- function(snapshot_date) {
  url <- paste0(
    "https://statsapi.mlb.com/api/v1/standings?",
    "leagueId=103,104",
    "&season=", season,
    "&standingsTypes=regularSeason",
    "&date=", format(snapshot_date, "%Y-%m-%d")
  )
  
  payload <- jsonlite::fromJSON(url, flatten = TRUE)
  if (is.null(payload$records) || nrow(payload$records) == 0) {
    return(tibble())
  }
  
  team_records <- purrr::map_dfr(payload$records$teamRecords, tibble::as_tibble)
  if (nrow(team_records) == 0) return(tibble())
  
  team_records %>%
    transmute(
      snapshot_date = as.Date(snapshot_date),
      season = season,
      team_id = as.integer(.data[["team.id"]]),
      W = as.numeric(.data[["wins"]]),
      L = as.numeric(.data[["losses"]]),
      `W-L%` = as.numeric(.data[["winningPercentage"]])
    ) %>%
    left_join(team_lookup, by = "team_id") %>%
    select(snapshot_date, season, team_id, abbrev, team_name, W, L, `W-L%`) %>%
    arrange(team_id)
}

existing <- if (file.exists(cache_path)) {
  readr::read_csv(cache_path, show_col_types = FALSE) %>%
    mutate(snapshot_date = as.Date(snapshot_date))
} else {
  tibble()
}

fresh <- purrr::map_dfr(snapshot_dates, fetch_standings_for_date)

if (nrow(fresh) == 0) {
  stop("No standings rows were fetched.")
}

out <- bind_rows(existing, fresh) %>%
  mutate(
    snapshot_date = as.Date(snapshot_date),
    season = as.integer(season),
    team_id = as.integer(team_id),
    W = as.numeric(W),
    L = as.numeric(L),
    `W-L%` = as.numeric(`W-L%`)
  ) %>%
  arrange(snapshot_date, team_id) %>%
  distinct(snapshot_date, team_id, .keep_all = TRUE)

readr::write_csv(out, cache_path)

counts <- out %>%
  count(snapshot_date, name = "teams") %>%
  arrange(snapshot_date)

print(counts, n = nrow(counts))
message("Wrote ", nrow(out), " rows to ", cache_path)
