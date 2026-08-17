# Build MLB pitch-type baselines from one or more Baseball Savant pitch-level CSVs.
#
# Usage:
#   source("pitcher-card/build_pitch_baseline.R")
#   build_pitch_baseline(
#     input_csvs = c("/path/to/statcast_search_2026.csv"),
#     output_csv = "pitcher-card/mlb_pitch_baseline_2026.csv"
#   )

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

pc_safe_num <- function(x) suppressWarnings(as.numeric(gsub('"', "", as.character(x))))

pc_first_present <- function(df, cols) {
  hit <- cols[cols %in% names(df)]
  if (length(hit) == 0) return(rep(NA_real_, nrow(df)))
  df[[hit[1]]]
}

pc_add_pitch_features <- function(df) {
  desc <- as.character(df$description %||% NA_character_)
  px <- pc_safe_num(pc_first_present(df, c("plate_x")))
  pz <- pc_safe_num(pc_first_present(df, c("plate_z")))
  bot <- pc_safe_num(pc_first_present(df, c("sz_bot")))
  top <- pc_safe_num(pc_first_present(df, c("sz_top")))
  in_zone <- !is.na(px) & !is.na(pz) & !is.na(bot) & !is.na(top) &
    px >= -0.83 & px <= 0.83 & pz >= bot & pz <= top
  swing <- desc %in% c("swinging_strike", "swinging_strike_blocked", "foul",
                      "foul_tip", "foul_bunt", "missed_bunt", "hit_into_play")
  whiff <- desc %in% c("swinging_strike", "swinging_strike_blocked",
                      "missed_bunt")
  called <- desc == "called_strike"
  pa_end <- !is.na(df$events) & as.character(df$events) != ""
  xwoba <- pc_safe_num(pc_first_present(df, c("estimated_woba_using_speedangle")))
  woba <- pc_safe_num(pc_first_present(df, c("woba_value")))
  pa_xwoba <- dplyr::coalesce(xwoba, woba)

  df %>%
    mutate(
      velo = pc_safe_num(pc_first_present(., c("release_speed"))),
      ivb = pc_safe_num(pc_first_present(., c("pfx_z"))) * 12,
      hb = pc_safe_num(pc_first_present(., c("pfx_x"))) * 12,
      spin = pc_safe_num(pc_first_present(., c("release_spin_rate"))),
      ext = pc_safe_num(pc_first_present(., c("release_extension"))),
      zone_flag = in_zone,
      chase_flag = swing & !in_zone,
      out_zone_flag = !in_zone,
      swing_flag = swing,
      whiff_flag = whiff,
      csw_flag = called | whiff,
      pa_end_flag = pa_end,
      pa_xwoba = pa_xwoba
    )
}

`%||%` <- function(x, y) if (is.null(x)) y else x

pc_mean <- function(x) if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
pc_sd <- function(x) {
  out <- sd(x, na.rm = TRUE)
  if (is.na(out) || out == 0) NA_real_ else out
}
pc_rate <- function(num, den) ifelse(den > 0, num / den, NA_real_)

build_pitch_baseline <- function(input_csvs, output_csv = "pitcher-card/mlb_pitch_baseline_2026.csv") {
  if (length(input_csvs) == 0) stop("Provide at least one Baseball Savant pitch-level CSV.")
  missing <- input_csvs[!file.exists(input_csvs)]
  if (length(missing) > 0) stop("These input CSVs do not exist: ", paste(missing, collapse = ", "))

  pitches <- bind_rows(lapply(input_csvs, readr::read_csv, show_col_types = FALSE)) %>%
    pc_add_pitch_features() %>%
    filter(!is.na(pitch_type), !is.na(pitch_name), !is.na(p_throws))

  baseline <- pitches %>%
    group_by(pitch_type, pitch_name, p_throws) %>%
    summarise(
      baseline_pitches = n(),
      baseline_pa = sum(pa_end_flag, na.rm = TRUE),
      velo_mean = pc_mean(velo), velo_sd = pc_sd(velo),
      ivb_mean = pc_mean(ivb), ivb_sd = pc_sd(ivb),
      hb_mean = pc_mean(hb), hb_sd = pc_sd(hb),
      spin_mean = pc_mean(spin), spin_sd = pc_sd(spin),
      ext_mean = pc_mean(ext), ext_sd = pc_sd(ext),
      zone_mean = mean(zone_flag, na.rm = TRUE), zone_sd = pc_sd(as.numeric(zone_flag)),
      chase_mean = pc_rate(sum(chase_flag, na.rm = TRUE), sum(out_zone_flag, na.rm = TRUE)),
      chase_sd = pc_sd(as.numeric(chase_flag[out_zone_flag])),
      whiff_mean = pc_rate(sum(whiff_flag, na.rm = TRUE), sum(swing_flag, na.rm = TRUE)),
      whiff_sd = pc_sd(as.numeric(whiff_flag[swing_flag])),
      csw_mean = mean(csw_flag, na.rm = TRUE), csw_sd = pc_sd(as.numeric(csw_flag)),
      xwoba_mean = pc_mean(pa_xwoba[pa_end_flag]), xwoba_sd = pc_sd(pa_xwoba[pa_end_flag]),
      .groups = "drop"
    ) %>%
    arrange(pitch_type, p_throws)

  dir.create(dirname(output_csv), showWarnings = FALSE, recursive = TRUE)
  readr::write_csv(baseline, output_csv)
  invisible(baseline)
}
