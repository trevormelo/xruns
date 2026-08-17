# xRuns pitcher card producer.
#
# Usage:
#   source("pitcher-card/make_pitcher_card.R")
#   make_pitcher_card(
#     game_csv = "/Users/trevormelo/Downloads/savant_data (1).csv",
#     baseline_csv = "pitcher-card/mlb_pitch_baseline_2026.csv",
#     output_png = "pitcher-card/payton_tolle_2026-04-23.png"
#   )

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(grid)
  library(readr)
  library(ragg)
  library(scales)
  library(stringr)
  library(tidyr)
})

PC_PA_PER_GAME <- 38
PC_RECENCY_DECAY <- 0.5

pc_pitch_colors <- c(
  FF = "#ff4f73", SI = "#f0a33a", FC = "#b95aa5", SL = "#48c47a",
  ST = "#ee8ed9", CU = "#5167f5", KC = "#7863d8", CH = "#4fd1b2",
  FS = "#2fb7c9", SV = "#9b6cff", KN = "#c7c7c7"
)
pc_pitch_color <- function(pitch_type, fallback = "#d1d5db") {
  col <- unname(pc_pitch_colors[as.character(pitch_type)])
  ifelse(is.na(col), fallback, col)
}

pc_espn_slugs <- c(
  LAA = "laa", ARI = "ari", BAL = "bal", BOS = "bos", CHC = "chc",
  CIN = "cin", CLE = "cle", COL = "col", DET = "det", HOU = "hou",
  KC  = "kc",  LAD = "lad", WSH = "wsh", NYM = "nym", OAK = "oak",
  PIT = "pit", SD  = "sd",  SEA = "sea", SF  = "sf",  STL = "stl",
  TB  = "tb",  TEX = "tex", TOR = "tor", MIN = "min", PHI = "phi",
  ATL = "atl", CWS = "chw", MIA = "mia", NYY = "nyy", MIL = "mil"
)

`%||%` <- function(x, y) if (is.null(x)) y else x
pc_safe_num <- function(x) suppressWarnings(as.numeric(gsub('"', "", as.character(x))))
pc_mean <- function(x) if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
pc_mean0 <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) 0 else mean(x)
}
pc_wmean <- function(x, w) {
  ok <- is.finite(x) & is.finite(w) & w > 0
  if (!any(ok)) NA_real_ else weighted.mean(x[ok], w[ok])
}
pc_scaled_mean_z <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) return(0)
  pc_clamp(mean(x) * sqrt(length(x)), -4, 4)
}
pc_rate <- function(num, den) ifelse(den > 0, num / den, NA_real_)
pc_clamp <- function(x, lo, hi) pmax(lo, pmin(hi, x))

pc_require_cols <- function(df, cols, label) {
  missing <- setdiff(cols, names(df))
  if (length(missing) > 0) {
    stop(label, " is missing required columns: ", paste(missing, collapse = ", "), call. = FALSE)
  }
}

pc_first_present <- function(df, cols) {
  hit <- cols[cols %in% names(df)]
  if (length(hit) == 0) return(rep(NA_real_, nrow(df)))
  df[[hit[1]]]
}

pc_clean_player_name <- function(x) {
  x <- as.character(x[1])
  if (is.na(x) || !nzchar(x)) return("Unknown Pitcher")
  if (grepl(",", x)) {
    parts <- trimws(strsplit(x, ",", fixed = TRUE)[[1]])
    if (length(parts) >= 2) return(paste(parts[2], parts[1]))
  }
  x
}

pc_add_pitch_features <- function(df) {
  desc <- as.character(df$description %||% NA_character_)
  events <- as.character(df$events %||% NA_character_)
  px <- pc_safe_num(pc_first_present(df, c("plate_x")))
  pz <- pc_safe_num(pc_first_present(df, c("plate_z")))
  bot <- pc_safe_num(pc_first_present(df, c("sz_bot")))
  top <- pc_safe_num(pc_first_present(df, c("sz_top")))
  in_zone <- !is.na(px) & !is.na(pz) & !is.na(bot) & !is.na(top) &
    px >= -0.83 & px <= 0.83 & pz >= bot & pz <= top
  swing <- desc %in% c("swinging_strike", "swinging_strike_blocked", "foul",
                      "foul_tip", "foul_bunt", "missed_bunt", "hit_into_play")
  whiff <- desc %in% c("swinging_strike", "swinging_strike_blocked", "missed_bunt")
  called <- desc == "called_strike"
  pa_end <- !is.na(events) & events != ""
  xwoba <- pc_safe_num(pc_first_present(df, c("estimated_woba_using_speedangle")))
  woba <- pc_safe_num(pc_first_present(df, c("woba_value")))

  df %>%
    mutate(
      velo = pc_safe_num(pc_first_present(., c("release_speed"))),
      ivb = pc_safe_num(pc_first_present(., c("pfx_z"))) * 12,
      hb = pc_safe_num(pc_first_present(., c("pfx_x"))) * 12,
      spin = pc_safe_num(pc_first_present(., c("release_spin_rate"))),
      ext = pc_safe_num(pc_first_present(., c("release_extension"))),
      plate_x_num = px,
      plate_z_num = pz,
      zone_flag = in_zone,
      chase_flag = swing & !in_zone,
      out_zone_flag = !in_zone,
      swing_flag = swing,
      whiff_flag = whiff,
      csw_flag = called | whiff,
      strike_flag = as.character(type) %in% c("S", "X"),
      pa_end_flag = pa_end,
      pa_xwoba = dplyr::coalesce(xwoba, woba),
      pitcher_re = pc_safe_num(pc_first_present(., c("delta_pitcher_run_exp")))
    )
}

pc_fit_xruns_model <- function(data_dir = ".") {
  exp_files <- Sys.glob(file.path(data_dir, "* Player Data", "expected_stats_pitcher_*.csv"))
  rv_files <- Sys.glob(file.path(data_dir, "* Player Data", "run_value_pitcher_*.csv"))
  if (length(exp_files) == 0 || length(rv_files) == 0) {
    stop("Could not find xRuns pitcher expected-stat and run-value files under ", normalizePath(data_dir), call. = FALSE)
  }

  read_year <- function(exp_file) {
    year <- as.integer(stringr::str_match(basename(exp_file), "(\\d{4})")[, 2])
    rv_file <- rv_files[grepl(as.character(year), basename(rv_files))]
    if (length(rv_file) == 0) return(NULL)
    ep <- readr::read_csv(exp_file, show_col_types = FALSE)
    rp <- readr::read_csv(rv_file[1], show_col_types = FALSE)
    ep %>%
      transmute(
        player_id,
        season_year = year,
        pa_exp = pc_safe_num(pa),
        est_woba = pc_safe_num(est_woba),
        xera = pc_safe_num(xera)
      ) %>%
      inner_join(
        rp %>%
          transmute(player_id, pa_rv = pc_safe_num(pa), runs_all = pc_safe_num(runs_all)),
        by = "player_id"
      ) %>%
      mutate(
        pa = pmax(pa_exp, pa_rv, na.rm = TRUE),
        runs_per_pa = runs_all / pmax(pa_rv, 1)
      )
  }

  pool <- bind_rows(lapply(exp_files, read_year)) %>%
    filter(is.finite(runs_per_pa), is.finite(est_woba), is.finite(xera), pa_rv >= 1)
  if (nrow(pool) < 10) stop("Not enough pitcher rows to fit the xRuns model.", call. = FALSE)

  max_year <- max(pool$season_year, na.rm = TRUE)
  pool <- pool %>%
    mutate(fit_weight = pa_rv * PC_RECENCY_DECAY^(max_year - season_year))
  pit_model <- lm(runs_per_pa ~ est_woba + xera, data = pool, weights = fit_weight)
  xera_model <- lm(xera ~ est_woba, data = pool, weights = pmax(pa_exp, 1))
  pool$pred_runs_per_pa <- predict(pit_model, pool)

  list(
    pit_model = pit_model,
    xera_model = xera_model,
    pit_mean_pa = weighted.mean(pool$pred_runs_per_pa, pool$pa_rv, na.rm = TRUE),
    pool = pool
  )
}

pc_estimate_xruns_saved <- function(est_woba, pa, model_bundle) {
  if (!is.finite(est_woba) || !is.finite(pa) || pa <= 0) {
    return(list(xera = NA_real_, pred_rpa = NA_real_, saved = NA_real_))
  }
  xera <- as.numeric(predict(model_bundle$xera_model, newdata = data.frame(est_woba = est_woba)))
  xera <- pc_clamp(xera, 0, 15)
  pred_rpa <- as.numeric(predict(model_bundle$pit_model, newdata = data.frame(est_woba = est_woba, xera = xera)))
  saved <- (pred_rpa - model_bundle$pit_mean_pa) * pa
  list(xera = xera, pred_rpa = pred_rpa, saved = saved)
}

pc_outs_from_events <- function(events) {
  e <- as.character(events)
  dplyr::case_when(
    e %in% c("strikeout", "field_out", "force_out", "sac_fly", "sac_bunt",
             "fielders_choice_out", "pickoff_1b", "pickoff_2b", "pickoff_3b") ~ 1,
    e %in% c("double_play", "grounded_into_double_play", "strikeout_double_play",
             "sac_fly_double_play", "fielders_choice") ~ 2,
    e %in% c("triple_play") ~ 3,
    TRUE ~ 0
  )
}

pc_ip_label <- function(outs) paste0(outs %/% 3, ".", outs %% 3)

pc_grade_letter <- function(score) {
  dplyr::case_when(
    !is.finite(score) ~ "NA",
    score >= 70 ~ "A+",
    score >= 65 ~ "A",
    score >= 60 ~ "A-",
    score >= 55 ~ "B+",
    score >= 50 ~ "B",
    score >= 45 ~ "C+",
    score >= 40 ~ "C",
    score >= 35 ~ "D",
    TRUE ~ "F"
  )
}

pc_z <- function(value, mean, sd, invert = FALSE, abs_dev = FALSE) {
  if (!is.finite(value) || !is.finite(mean) || !is.finite(sd) || sd <= 0) return(NA_real_)
  z <- (value - mean) / sd
  if (abs_dev) z <- abs(z)
  if (invert) z <- -z
  pc_clamp(z, -3, 3)
}

pc_make_pitch_summary <- function(pitches, baseline, model_bundle) {
  total_pitches <- nrow(pitches)
  total_pas <- sum(pitches$pa_end_flag, na.rm = TRUE)

  raw <- pitches %>%
    group_by(pitch_type, pitch_name) %>%
    summarise(
      p_throws = dplyr::first(na.omit(p_throws)),
      count = n(),
      pitch_pct = count / total_pitches,
      pa = sum(pa_end_flag, na.rm = TRUE),
      velo = pc_mean(velo),
      ivb = pc_mean(ivb),
      hb = pc_mean(hb),
      spin = pc_mean(spin),
      ext = pc_mean(ext),
      zone = mean(zone_flag, na.rm = TRUE),
      chase = pc_rate(sum(chase_flag, na.rm = TRUE), sum(out_zone_flag, na.rm = TRUE)),
      whiff = pc_rate(sum(whiff_flag, na.rm = TRUE), sum(swing_flag, na.rm = TRUE)),
      csw = mean(csw_flag, na.rm = TRUE),
      xwoba = pc_mean(pa_xwoba[pa_end_flag]),
      re = sum(pitcher_re, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    left_join(baseline, by = c("pitch_type", "pitch_name", "p_throws"))

  values <- lapply(seq_len(nrow(raw)), function(i) {
    pc_estimate_xruns_saved(raw$xwoba[i], raw$pa[i], model_bundle)
  })
  raw$xera <- vapply(values, `[[`, numeric(1), "xera")
  raw$xruns_saved <- vapply(values, `[[`, numeric(1), "saved")

  raw %>%
    rowwise() %>%
    mutate(
      z_results = pc_z(xwoba, xwoba_mean, xwoba_sd, invert = TRUE),
      z_whiff = pc_z(whiff, whiff_mean, whiff_sd),
      z_csw = pc_z(csw, csw_mean, csw_sd),
      z_chase = pc_z(chase, chase_mean, chase_sd),
      z_zone = pc_z(zone, zone_mean, zone_sd),
      z_velo = pc_z(velo, velo_mean, velo_sd),
      z_ivb = pc_z(ivb, ivb_mean, ivb_sd, abs_dev = TRUE),
      z_hb = pc_z(hb, hb_mean, hb_sd, abs_dev = TRUE),
      z_ext = pc_z(ext, ext_mean, ext_sd),
      baseline_found = is.finite(xwoba_mean) | is.finite(whiff_mean) | is.finite(csw_mean),
      shape_component = pc_mean0(c(z_velo, z_ivb, z_hb, z_ext)),
      command_component = pc_mean0(c(z_zone, z_chase)),
      result_component = pc_mean0(c(z_results, z_whiff, z_csw)),
      grade_z = pc_mean0(c(result_component, result_component, command_component, shape_component)),
      grade_score = pc_clamp(round(50 + 10 * grade_z), 20, 80),
      grade = pc_grade_letter(grade_score),
      stuff_z = pc_scaled_mean_z(c(z_velo, z_ivb, z_hb, z_ext, z_whiff)),
      pitching_z = pc_scaled_mean_z(c(z_results, z_csw)),
      xruns_stuff_plus = round(pc_clamp(100 + 10 * stuff_z, 50, 150)),
      xruns_pitching_plus = round(pc_clamp(100 + 10 * pitching_z, 50, 150)),
      confidence = case_when(
        !baseline_found ~ "No BL",
        count >= 15 & pa >= 3 ~ "High",
        count >= 8 | pa >= 2 ~ "Med",
        TRUE ~ "Low"
      )
    ) %>%
    ungroup() %>%
    arrange(desc(count)) %>%
    mutate(total_pa = total_pas)
}

pc_card_theme <- function(base_size = 10) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.background = element_rect(fill = "#111827", color = NA),
      panel.background = element_rect(fill = "#17233a", color = NA),
      panel.grid = element_line(color = "#38506f", linewidth = 0.25),
      axis.text = element_text(color = "#c9d4e5"),
      axis.title = element_text(color = "#e7edf7", face = "bold"),
      plot.title = element_text(color = "#ffffff", face = "bold", hjust = 0.5),
      plot.subtitle = element_text(color = "#9fb4d1", hjust = 0.5),
      legend.position = "none"
    )
}

pc_location_plot <- function(pitches,
                             title = "Pitch Locations",
                             subtitle = NULL,
                             base_size = 9,
                             point_size = 2.2) {
  ggplot(pitches, aes(plate_x_num, plate_z_num, color = pitch_type)) +
    annotate("rect", xmin = -0.83, xmax = 0.83, ymin = 1.5, ymax = 3.5,
             fill = NA, color = "#dbe7f8", linewidth = 0.6) +
    geom_point(size = point_size, alpha = 0.88) +
    scale_color_manual(values = pc_pitch_colors, na.value = "#d1d5db") +
    coord_cartesian(xlim = c(-2.2, 2.2), ylim = c(0, 5), expand = FALSE) +
    labs(title = title, subtitle = subtitle, x = NULL, y = NULL) +
    pc_card_theme(base_size) +
    theme(
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      panel.grid = element_blank()
    )
}

pc_movement_plot <- function(pitches, base_size = 9, point_size = 2.2) {
  throws <- as.character(pitches$p_throws[which(!is.na(pitches$p_throws))[1]])
  arm_x <- ifelse(throws == "L", -13, 13)
  glove_x <- -arm_x
  pitches <- pitches %>% mutate(mound_hb = -hb)
  ellipse_data <- pitches %>%
    group_by(pitch_type) %>%
    filter(n() >= 4, sd(mound_hb, na.rm = TRUE) > 0, sd(ivb, na.rm = TRUE) > 0) %>%
    ungroup()
  ggplot(pitches, aes(mound_hb, ivb, color = pitch_type)) +
    geom_hline(yintercept = 0, color = "#8aa0bd", linewidth = 0.35) +
    geom_vline(xintercept = 0, color = "#8aa0bd", linewidth = 0.35) +
    stat_ellipse(data = ellipse_data, aes(fill = pitch_type), geom = "polygon", alpha = 0.09,
                 color = NA, show.legend = FALSE, type = "norm") +
    geom_point(size = point_size, alpha = 0.46) +
    annotate("text", x = arm_x, y = 21.5, label = "Arm", color = "#aebed5",
             fontface = "bold", size = base_size / 5.4) +
    annotate("text", x = glove_x, y = 21.5, label = "Glove", color = "#aebed5",
             fontface = "bold", size = base_size / 5.4) +
    scale_color_manual(values = pc_pitch_colors, na.value = "#d1d5db") +
    scale_fill_manual(values = pc_pitch_colors, na.value = "#d1d5db") +
    coord_cartesian(xlim = c(-25, 25), ylim = c(-25, 25), expand = FALSE) +
    labs(title = "Movement", x = NULL, y = NULL) +
    pc_card_theme(base_size) +
    theme(
      axis.text = element_text(color = "#9fb4d1", size = base_size * 0.72),
      panel.grid = element_line(color = "#314761", linewidth = 0.22)
    )
}

pc_short_pitch_name <- function(x) {
  x <- as.character(x)
  x <- stringr::str_replace(x, "4-Seam Fastball", "4-Seam")
  x <- stringr::str_replace(x, "Split-Finger", "Splitter")
  x <- stringr::str_replace(x, "Knuckle Curve", "Knuckle-Curve")
  x
}

pc_usage_grob <- function(pitches, summary_df) {
  usage <- pitches %>%
    count(stand, pitch_type, pitch_name) %>%
    group_by(stand) %>%
    mutate(pct = n / sum(n)) %>%
    ungroup()
  rows <- summary_df %>%
    arrange(desc(count)) %>%
    transmute(pitch_type, pitch_name, overall = pitch_pct) %>%
    head(7)
  side_counts <- pitches %>% count(stand)
  l_total <- side_counts$n[side_counts$stand == "L"]
  r_total <- side_counts$n[side_counts$stand == "R"]
  l_total <- ifelse(length(l_total) == 0 || is.na(l_total), 0, l_total)
  r_total <- ifelse(length(r_total) == 0 || is.na(r_total), 0, r_total)
  max_pct <- max(usage$pct, 0.5, na.rm = TRUE)

  grid.grabExpr({
    grid.text("Usage", x = 0.5, y = 0.96, gp = gpar(col = "#ffffff", fontsize = 15, fontface = "bold"))
    grid.text(sprintf("vs LHH (%d)", l_total), x = 0.28, y = 0.86, gp = gpar(col = "#9fb4d1", fontsize = 8.5, fontface = "bold"))
    grid.text(sprintf("vs RHH (%d)", r_total), x = 0.72, y = 0.86, gp = gpar(col = "#9fb4d1", fontsize = 8.5, fontface = "bold"))
    grid.lines(x = c(0.5, 0.5), y = c(0.14, 0.82), gp = gpar(col = "#47627f", lwd = 1))
    row_y <- seq(0.74, 0.2, length.out = max(1, nrow(rows)))
    for (i in seq_len(nrow(rows))) {
      pt <- rows$pitch_type[i]
      col <- pc_pitch_color(pt)
      lhs <- usage$pct[usage$stand == "L" & usage$pitch_type == pt]
      rhs <- usage$pct[usage$stand == "R" & usage$pitch_type == pt]
      lhs <- ifelse(length(lhs) == 0 || is.na(lhs), 0, lhs)
      rhs <- ifelse(length(rhs) == 0 || is.na(rhs), 0, rhs)
      y <- row_y[i]
      lw <- 0.23 * lhs / max_pct
      rw <- 0.23 * rhs / max_pct
      if (lw > 0) {
        grid.roundrect(x = 0.43 - lw / 2, y = y, width = lw, height = 0.07,
                       r = unit(0.012, "npc"), gp = gpar(fill = col, col = NA, alpha = 0.95))
      }
      if (rw > 0) {
        grid.roundrect(x = 0.57 + rw / 2, y = y, width = rw, height = 0.07,
                       r = unit(0.012, "npc"), gp = gpar(fill = col, col = NA, alpha = 0.95))
      }
      grid.text(scales::percent(lhs, accuracy = 1), x = 0.19, y = y,
                just = "right", gp = gpar(col = col, fontsize = 8.2, fontface = "bold"))
      grid.text(scales::percent(rhs, accuracy = 1), x = 0.81, y = y,
                just = "left", gp = gpar(col = col, fontsize = 8.2, fontface = "bold"))
      grid.points(x = 0.5, y = y, pch = 21, size = unit(4.2, "mm"),
                  gp = gpar(fill = col, col = "#0f172a", lwd = 0.7))
    }
  })
}

pc_metric_fill <- function(z) {
  if (!is.finite(z)) return("#243754")
  z <- pc_clamp(z, -2, 2)
  ramp <- grDevices::colorRamp(c("#3f7ee8", "#243754", "#d75b66"))
  rgb <- ramp((z + 2) / 4)
  grDevices::rgb(rgb[1], rgb[2], rgb[3], maxColorValue = 255)
}

pc_table_grob <- function(summary_df,
                          header_size = 7.6,
                          text_size = 7.4,
                          header_h = 0.18,
                          overall_rv_added = NULL,
                          overall_xwoba = NULL) {
  max_runs <- max(abs(summary_df$xruns_saved), na.rm = TRUE)
  if (!is.finite(max_runs) || max_runs <= 0) max_runs <- 1
  overall_stuff_z <- pc_scaled_mean_z(c(
    pc_wmean(summary_df$z_velo, summary_df$count),
    pc_wmean(summary_df$z_ivb, summary_df$count),
    pc_wmean(summary_df$z_hb, summary_df$count),
    pc_wmean(summary_df$z_ext, summary_df$count),
    pc_wmean(summary_df$z_whiff, summary_df$count)
  ))
  overall_pitching_z <- pc_scaled_mean_z(c(
    pc_wmean(summary_df$z_results, pmax(summary_df$pa, 1)),
    pc_wmean(summary_df$z_csw, summary_df$count)
  ))
  overall_row <- tibble::tibble(
    pitch_type = "ALL",
    pitch_name = "All",
    count = sum(summary_df$count, na.rm = TRUE),
    pitch_pct = 1,
    pa = sum(summary_df$pa, na.rm = TRUE),
    velo = pc_wmean(summary_df$velo, summary_df$count),
    ivb = pc_wmean(summary_df$ivb, summary_df$count),
    hb = pc_wmean(summary_df$hb, summary_df$count),
    zone = pc_wmean(summary_df$zone, summary_df$count),
    chase = pc_wmean(summary_df$chase, summary_df$count),
    whiff = pc_wmean(summary_df$whiff, summary_df$count),
    csw = pc_wmean(summary_df$csw, summary_df$count),
    xwoba = if (!is.null(overall_xwoba) && is.finite(overall_xwoba)) overall_xwoba else pc_wmean(summary_df$xwoba, pmax(summary_df$pa, 1)),
    xruns_saved = if (!is.null(overall_rv_added) && is.finite(overall_rv_added)) overall_rv_added else sum(summary_df$xruns_saved, na.rm = TRUE),
    xruns_stuff_plus = round(pc_clamp(100 + 10 * overall_stuff_z, 50, 150)),
    xruns_pitching_plus = round(pc_clamp(100 + 10 * overall_pitching_z, 50, 150)),
    z_chase = pc_wmean(summary_df$z_chase, summary_df$count),
    z_whiff = pc_wmean(summary_df$z_whiff, summary_df$count),
    z_csw = pc_wmean(summary_df$z_csw, summary_df$count),
    z_results = pc_wmean(summary_df$z_results, pmax(summary_df$pa, 1)),
    stuff_z = overall_stuff_z,
    pitching_z = overall_pitching_z
  )
  display_df <- bind_rows(summary_df, overall_row)
  tbl <- display_df %>%
    transmute(
      Pitch = ifelse(pitch_type == "ALL", "All", paste(pitch_type, pitch_name)),
      `#` = count,
      `%` = scales::percent(pitch_pct, accuracy = 1),
      Velo = sprintf("%.1f", velo),
      IVB = sprintf("%+.1f", ivb),
      HB = sprintf("%+.1f", hb),
      `Zone%` = scales::percent(zone, accuracy = 1),
      `Chase%` = ifelse(is.na(chase), "--", scales::percent(chase, accuracy = 1)),
      `Whiff%` = ifelse(is.na(whiff), "--", scales::percent(whiff, accuracy = 1)),
      CSW = ifelse(is.na(csw), "--", scales::percent(csw, accuracy = 1)),
      xwOBA = ifelse(is.na(xwoba), "--", sprintf("%.3f", xwoba)),
      `RV Added` = ifelse(is.na(xruns_saved), "--", sprintf("%+.2f", xruns_saved)),
      `xStuff+` = xruns_stuff_plus,
      `xPitching+` = xruns_pitching_plus,
      pitch_type = pitch_type,
      z_chase = z_chase,
      z_whiff = z_whiff,
      z_csw = z_csw,
      z_results = z_results,
      z_runs = xruns_saved / max_runs,
      z_stuff = stuff_z,
      z_pitching = pitching_z
    )
  cols <- c("Pitch", "#", "%", "Velo", "IVB", "HB", "Zone%", "Chase%", "Whiff%", "CSW", "xwOBA", "RV Added", "xStuff+", "xPitching+")
  col_labels <- cols
  col_labels[col_labels == "xStuff+"] <- "xRuns\nStuff+"
  col_labels[col_labels == "xPitching+"] <- "xRuns\nPitching+"
  widths <- c(0.20, 0.045, 0.05, 0.06, 0.06, 0.06, 0.065, 0.075, 0.075, 0.065, 0.07, 0.08, 0.075, 0.085)
  widths <- widths / sum(widths)
  x_left <- c(0, cumsum(widths)[-length(widths)])
  n <- nrow(tbl)
  grid.grabExpr({
    grid.rect(gp = gpar(fill = "#101827", col = NA))
    row_h <- (1 - header_h) / max(n, 1)
    for (j in seq_along(cols)) {
      grid.rect(x = x_left[j] + widths[j] / 2, y = 1 - header_h / 2,
                width = widths[j], height = header_h,
                gp = gpar(fill = "#1f6feb", col = "#365b91", lwd = 0.6))
      grid.text(col_labels[j], x = x_left[j] + widths[j] / 2, y = 1 - header_h / 2,
                gp = gpar(col = "#ffffff", fontsize = header_size, fontface = "bold"))
    }
    for (i in seq_len(n)) {
      y <- 1 - header_h - (i - 0.5) * row_h
      base_fill <- ifelse(i %% 2 == 1, "#17233a", "#142036")
      for (j in seq_along(cols)) {
        col_name <- cols[j]
        fill <- base_fill
        if (col_name == "Chase%") fill <- pc_metric_fill(tbl$z_chase[i])
        if (col_name == "Whiff%") fill <- pc_metric_fill(tbl$z_whiff[i])
        if (col_name == "xwOBA") fill <- pc_metric_fill(tbl$z_results[i])
        if (col_name == "CSW") fill <- pc_metric_fill(tbl$z_csw[i])
        if (col_name == "RV Added") fill <- pc_metric_fill(tbl$z_runs[i] * 2)
        if (col_name == "xStuff+") fill <- pc_metric_fill(tbl$z_stuff[i])
        if (col_name == "xPitching+") fill <- pc_metric_fill(tbl$z_pitching[i])
        grid.rect(x = x_left[j] + widths[j] / 2, y = y,
                  width = widths[j], height = row_h,
                  gp = gpar(fill = fill, col = "#29405f", lwd = 0.55))
        txt <- as.character(tbl[[col_name]][i])
        txt_col <- if (col_name == "Pitch") pc_pitch_color(tbl$pitch_type[i], "#ffffff") else "#edf6ff"
        font <- if (col_name %in% c("Pitch", "RV Added", "xStuff+", "xPitching+")) "bold" else "plain"
        grid.text(txt, x = x_left[j] + widths[j] / 2, y = y,
                  gp = gpar(col = txt_col, fontsize = text_size, fontface = font))
      }
    }
  })
}

pc_draw_panel <- function(grob, x, y, width, height, radius_col = "#21b6d7") {
  pushViewport(viewport(x = x, y = y, width = width, height = height, just = c("left", "bottom")))
  grid.roundrect(gp = gpar(fill = "#17233a", col = radius_col, lwd = 1.4), r = unit(0.018, "npc"))
  pushViewport(viewport(x = 0.5, y = 0.5, width = 0.965, height = 0.93))
  grid.draw(grob)
  popViewport(2)
}

pc_draw_plot_area <- function(grob, x, y, width, height) {
  pushViewport(viewport(x = x, y = y, width = width, height = height, just = c("left", "bottom")))
  grid.roundrect(gp = gpar(fill = "#17233a", col = NA), r = unit(0.012, "npc"))
  pushViewport(viewport(x = 0.5, y = 0.5, width = 0.97, height = 0.94))
  grid.draw(grob)
  popViewport(2)
}

pc_draw_text <- function(label, x, y, size, col = "#ffffff", fontface = "plain", just = "left") {
  grid.text(label, x = x, y = y, just = just, gp = gpar(col = col, fontsize = size, fontface = fontface))
}

pc_headshot_url <- function(player_id, width = 240, height = width) {
  sprintf(
    "https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_%d,h_%d,c_fill,g_face,q_auto:best/v1/people/%d/headshot/67/current",
    width, height, as.integer(player_id)
  )
}

pc_team_logo_url <- function(abbrev) {
  slug <- pc_espn_slugs[[as.character(abbrev)]]
  if (is.null(slug) || is.na(slug)) return(NA_character_)
  paste0("https://a.espncdn.com/i/teamlogos/mlb/500/", slug, ".png")
}

pc_read_asset <- function(url, timeout = 8) {
  if (is.na(url) || !nzchar(url) || !requireNamespace("magick", quietly = TRUE)) return(NULL)
  old_timeout <- getOption("timeout")
  options(timeout = timeout)
  on.exit(options(timeout = old_timeout), add = TRUE)
  tmp <- tempfile(fileext = ".png")
  tryCatch({
    if (requireNamespace("curl", quietly = TRUE)) {
      handle <- curl::new_handle(timeout = timeout, connecttimeout = timeout)
      curl::curl_download(url, tmp, quiet = TRUE, handle = handle)
    } else {
      utils::download.file(url, tmp, quiet = TRUE, mode = "wb")
    }
    if (!file.exists(tmp) || file.info(tmp)$size <= 0) return(NULL)
    img <- magick::image_read(tmp)
    as.raster(img)
  }, error = function(e) NULL)
}

pc_draw_asset <- function(url, x, y, width, height, fallback, timeout = 8,
                          fill = "#203354", border = "#21b6d7") {
  grid.roundrect(x = x, y = y, width = width, height = height,
                 gp = gpar(fill = fill, col = border, lwd = 0.8),
                 r = unit(0.014, "npc"))
  asset <- pc_read_asset(url, timeout = timeout)
  if (is.null(asset)) {
    grid.text(fallback, x = x, y = y, gp = gpar(col = "#c2d2e8", fontsize = 9, fontface = "bold"))
  } else {
    grid.raster(asset, x = x, y = y, width = width * 0.9, height = height * 0.9, interpolate = TRUE)
  }
}

pc_draw_pitch_legend <- function(summary_df, y = 0.796, abbr_only = FALSE) {
  rows <- summary_df %>%
    arrange(desc(count)) %>%
    mutate(label = paste0(pitch_type, " ", pc_short_pitch_name(pitch_name))) %>%
    head(7)
  if (nrow(rows) == 0) return(invisible(NULL))
  grid.roundrect(x = 0.5, y = y, width = 0.62, height = 0.035,
                 gp = gpar(fill = "#142036", col = "#294b6f", lwd = 0.8),
                 r = unit(0.012, "npc"))
  xs <- seq(0.24, 0.76, length.out = nrow(rows))
  for (i in seq_len(nrow(rows))) {
    col <- pc_pitch_color(rows$pitch_type[i])
    if (abbr_only) {
      grid.roundrect(x = xs[i], y = y, width = 0.043, height = 0.023,
                     r = unit(0.006, "npc"), gp = gpar(fill = col, col = NA))
      grid.text(rows$pitch_type[i], x = xs[i], y = y,
                gp = gpar(col = "#ffffff", fontsize = 7.2, fontface = "bold"))
    } else {
      grid.points(x = xs[i] - 0.028, y = y, pch = 21, size = unit(4.2, "mm"),
                  gp = gpar(fill = col, col = "#0f172a", lwd = 0.6))
      grid.text(rows$label[i], x = xs[i] - 0.018, y = y, just = "left",
                gp = gpar(col = "#eaf3ff", fontsize = 7.5, fontface = "bold"))
    }
  }
}

pc_draw_stat_table <- function(stats, x = 0.5, y = 0.74, width = 0.90, height = 0.055,
                               label_size = 9, value_size = 13) {
  n <- length(stats)
  cell_w <- width / n
  left <- x - width / 2
  for (i in seq_along(stats)) {
    cx <- left + (i - 0.5) * cell_w
    grid.roundrect(x = cx, y = y, width = cell_w * 0.985, height = height,
                   r = unit(0.006, "npc"),
                   gp = gpar(fill = "#111827", col = "#21b6d7", lwd = 0.65))
    grid.rect(x = cx, y = y + height * 0.25, width = cell_w * 0.985, height = height * 0.5,
              gp = gpar(fill = "#1f6feb", col = NA))
    grid.text(names(stats)[i], x = cx, y = y + height * 0.25,
              gp = gpar(col = "#ffffff", fontsize = label_size, fontface = "bold"))
    grid.text(stats[[i]], x = cx, y = y - height * 0.25,
              gp = gpar(col = "#eaf3ff", fontsize = value_size, fontface = "bold"))
  }
}

pc_render_square_card <- function(output_png, width, height, game, pitch_summary, table_grob,
                                  pitcher, pitcher_id, p_throws, team, opp, game_date,
                                  game_value, stat_values, game_xwoba) {
  loc_grob <- ggplotGrob(pc_location_plot(
    game,
    title = "Locations",
    subtitle = sprintf("%d pitches", nrow(game)),
    base_size = 11,
    point_size = 3.0
  ))
  mov_grob <- ggplotGrob(pc_movement_plot(game, base_size = 9.3, point_size = 2.7))
  use_grob <- pc_usage_grob(game, pitch_summary)

  ragg::agg_png(output_png, width = width, height = height, units = "px", res = 220, background = "#111827")
  grid.newpage()
  grid.rect(gp = gpar(fill = "#111827", col = NA))

  grid.roundrect(x = 0.5, y = 0.92, width = 0.95, height = 0.135,
                 gp = gpar(fill = "#1b2e4d", col = "#21b6d7", lwd = 1.2),
                 r = unit(0.016, "npc"))
  pc_draw_asset(pc_headshot_url(pitcher_id, 320), 0.10, 0.92, 0.105, 0.105,
                fallback = "P", fill = "#10213a")
  pc_draw_asset(pc_team_logo_url(team), 0.90, 0.92, 0.09, 0.09,
                fallback = team, fill = "#1b2e4d", border = "#2b7fb8")
  pc_draw_text("xRuns", 0.17, 0.958, 14, "#35d0ff", "bold", "left")
  pc_draw_text(pitcher, 0.17, 0.928, 25, "#ffffff", "bold", "left")
  pc_draw_text(sprintf("%sHP | %s vs %s | %s", p_throws, team, opp, game_date),
               0.17, 0.899, 10.5, "#c2d2e8", "bold", "left")
  pc_draw_text(sprintf("%+.2f", game_value$saved), 0.755, 0.932, 27,
               ifelse(is.na(game_value$saved) || game_value$saved < 0, "#ff728a", "#66e39c"),
               "bold", "center")
  pc_draw_text("Run Value Added", 0.755, 0.900, 9.5, "#c2d2e8", "bold", "center")

  pc_draw_stat_table(stat_values, x = 0.5, y = 0.775, width = 0.90, height = 0.058,
                     label_size = 8.5, value_size = 13.5)

  pc_draw_plot_area(loc_grob, 0.05, 0.415, 0.28, 0.28)
  pc_draw_plot_area(mov_grob, 0.365, 0.415, 0.27, 0.28)
  pc_draw_plot_area(use_grob, 0.67, 0.415, 0.28, 0.28)

  pc_draw_pitch_legend(pitch_summary, y = 0.39, abbr_only = TRUE)

  grid.roundrect(x = 0.5, y = 0.205, width = 0.95, height = 0.31,
                 gp = gpar(fill = "#17233a", col = "#21b6d7", lwd = 1.3),
                 r = unit(0.014, "npc"))
  pc_draw_text("Pitch Type Metrics", 0.5, 0.345, 16, "#e8f2ff", "bold", "center")
  pushViewport(viewport(x = 0.5, y = 0.205, width = 0.91, height = 0.235))
  grid.draw(table_grob)
  popViewport()
  pc_draw_text(sprintf("%.3f game xwOBA | RV Added = Run Value Added | xStuff+ and xPitching+ are scaled 100 avg / 10 SD.", game_xwoba),
               0.5, 0.055, 7.5, "#8fa3c0", "plain", "center")
  grid.rect(x = 0, y = 1, width = 0.012, height = 0.012, just = c("left", "top"),
            gp = gpar(fill = "#111827", col = NA))
  dev.off()
}

make_pitcher_card <- function(game_csv,
                              baseline_csv,
                              output_png,
                              data_dir = ".",
                              layout = c("square", "landscape"),
                              width = NULL,
                              height = NULL) {
  layout <- match.arg(layout)
  if (is.null(width)) width <- if (layout == "square") 2200 else 1600
  if (is.null(height)) height <- if (layout == "square") 2200 else 900
  if (!file.exists(game_csv)) stop("Game CSV does not exist: ", game_csv, call. = FALSE)
  if (!file.exists(baseline_csv)) {
    stop(
      "Baseline CSV does not exist: ", baseline_csv, "\n",
      "Create one from a local Baseball Savant league pitch-level export with:\n",
      "  source('pitcher-card/build_pitch_baseline.R')\n",
      "  build_pitch_baseline('/path/to/statcast_search.csv', '", baseline_csv, "')",
      call. = FALSE
    )
  }

  game <- readr::read_csv(game_csv, show_col_types = FALSE)
  pc_require_cols(game, c("pitch_type", "pitch_name", "game_date", "player_name", "events",
                          "description", "type", "stand", "p_throws", "home_team", "away_team",
                          "plate_x", "plate_z", "pfx_x", "pfx_z"), "Game CSV")
  game <- game %>% pc_add_pitch_features() %>% filter(!is.na(pitch_type))
  if (nrow(game) == 0) stop("No usable pitch rows found in game CSV.", call. = FALSE)

  baseline <- readr::read_csv(baseline_csv, show_col_types = FALSE)
  pc_require_cols(baseline, c("pitch_type", "pitch_name", "p_throws", "xwoba_mean", "xwoba_sd",
                              "whiff_mean", "whiff_sd", "csw_mean", "csw_sd"), "Baseline CSV")

  model_bundle <- pc_fit_xruns_model(data_dir)
  pitch_summary <- pc_make_pitch_summary(game, baseline, model_bundle)

  total_pa <- sum(game$pa_end_flag, na.rm = TRUE)
  game_xwoba <- pc_mean(game$pa_xwoba[game$pa_end_flag])
  game_value <- pc_estimate_xruns_saved(game_xwoba, total_pa, model_bundle)
  outs <- sum(pc_outs_from_events(game$events), na.rm = TRUE)
  events <- as.character(game$events)
  hits <- sum(events %in% c("single", "double", "triple", "home_run"), na.rm = TRUE)
  walks <- sum(events %in% c("walk", "intent_walk"), na.rm = TRUE)
  strikeouts <- sum(events == "strikeout", na.rm = TRUE)
  homers <- sum(events == "home_run", na.rm = TRUE)
  whiffs <- sum(game$whiff_flag, na.rm = TRUE)
  csw <- mean(game$csw_flag, na.rm = TRUE)
  strike_pct <- mean(game$strike_flag, na.rm = TRUE)
  stat_values <- c(
    IP = pc_ip_label(outs),
    PA = as.character(total_pa),
    H = as.character(hits),
    BB = as.character(walks),
    K = as.character(strikeouts),
    HR = as.character(homers),
    `Strike%` = scales::percent(strike_pct, accuracy = 1),
    Whiffs = as.character(whiffs),
    CSW = scales::percent(csw, accuracy = 1)
  )

  pitcher <- pc_clean_player_name(game$player_name)
  game_date <- as.character(game$game_date[which(!is.na(game$game_date))[1]])
  p_throws <- as.character(game$p_throws[which(!is.na(game$p_throws))[1]])
  home <- as.character(game$home_team[1])
  away <- as.character(game$away_team[1])
  fielder_team <- ifelse(game$inning_topbot %||% "" == "Top", home, away)
  team <- names(sort(table(fielder_team), decreasing = TRUE))[1] %||% ""
  opp <- ifelse(team == home, away, home)
  pitcher_id <- suppressWarnings(as.integer(game$pitcher[which(!is.na(game$pitcher))[1]]))

  dir.create(dirname(output_png), showWarnings = FALSE, recursive = TRUE)

  if (layout == "square") {
    table_grob <- pc_table_grob(
      pitch_summary,
      header_size = 9.4,
      text_size = 9.1,
      header_h = 0.18,
      overall_rv_added = game_value$saved,
      overall_xwoba = game_xwoba
    )
    pc_render_square_card(
      output_png = output_png,
      width = width,
      height = height,
      game = game,
      pitch_summary = pitch_summary,
      table_grob = table_grob,
      pitcher = pitcher,
      pitcher_id = pitcher_id,
      p_throws = p_throws,
      team = team,
      opp = opp,
      game_date = game_date,
      game_value = game_value,
      stat_values = stat_values,
      game_xwoba = game_xwoba
    )
    return(invisible(list(
      output_png = output_png,
      pitcher = pitcher,
      game_date = game_date,
      pitch_summary = pitch_summary,
      game_xwoba = game_xwoba,
      xruns_saved = game_value$saved
    )))
  }

  loc_grob <- ggplotGrob(pc_location_plot(game))
  mov_grob <- ggplotGrob(pc_movement_plot(game))
  use_grob <- pc_usage_grob(game, pitch_summary)
  table_grob <- pc_table_grob(pitch_summary, overall_rv_added = game_value$saved, overall_xwoba = game_xwoba)

  ragg::agg_png(output_png, width = width, height = height, units = "px", res = 160, background = "#111827")
  grid.newpage()
  grid.rect(gp = gpar(fill = "#111827", col = NA))
  grid.roundrect(x = 0.5, y = 0.93, width = 0.97, height = 0.125,
                 gp = gpar(fill = "#1b2e4d", col = "#2b7fb8", lwd = 1.2),
                 r = unit(0.018, "npc"))
  pc_draw_asset(pc_headshot_url(pitcher_id), 0.07, 0.93, 0.07, 0.10,
                fallback = "P", fill = "#10213a")
  pc_draw_text("xRuns", 0.122, 0.968, 16, "#35d0ff", "bold")
  pc_draw_text(pitcher, 0.122, 0.928, 23, "#ffffff", "bold")
  pc_draw_text(sprintf("%sHP | %s vs %s | %s", p_throws, team, opp, game_date),
               0.122, 0.895, 10.5, "#c2d2e8", "bold")
  pc_draw_text("PITCHER PERFORMANCE CARD", 0.895, 0.962, 12, "#c2d2e8", "bold", "right")
  pc_draw_text(sprintf("%+.2f", game_value$saved), 0.78, 0.922, 28,
               ifelse(is.na(game_value$saved) || game_value$saved < 0, "#ff728a", "#66e39c"),
               "bold", "right")
  pc_draw_text("Run Value Added", 0.79, 0.922, 10, "#c2d2e8", "bold", "left")
  pc_draw_asset(pc_team_logo_url(team), 0.94, 0.925, 0.065, 0.085,
                fallback = team, fill = "#1b2e4d", border = "#2b7fb8")

  stat_line <- sprintf(
    "%s IP   %d PA   %d H   %d BB   %d K   %d HR   %s Strike   %d Whiffs   %s CSW   %.3f xwOBA",
    pc_ip_label(outs), total_pa, hits, walks, strikeouts, homers,
    scales::percent(strike_pct, accuracy = 1), whiffs, scales::percent(csw, accuracy = 1),
    game_xwoba
  )
  grid.roundrect(x = 0.5, y = 0.855, width = 0.97, height = 0.055,
                 gp = gpar(fill = "#0f172a", col = "#21b6d7", lwd = 1.1),
                 r = unit(0.015, "npc"))
  pc_draw_text(stat_line, 0.5, 0.855, 14, "#eef6ff", "bold", "center")
  pc_draw_pitch_legend(pitch_summary, y = 0.802)

  pc_draw_panel(loc_grob, 0.025, 0.40, 0.30, 0.365)
  pc_draw_panel(mov_grob, 0.35, 0.40, 0.30, 0.365)
  pc_draw_panel(use_grob, 0.675, 0.40, 0.30, 0.365)

  grid.roundrect(x = 0.5, y = 0.19, width = 0.95, height = 0.34,
                 gp = gpar(fill = "#17233a", col = "#21b6d7", lwd = 1.3),
                 r = unit(0.014, "npc"))
  pc_draw_text("Pitch Type Metrics", 0.5, 0.345, 15, "#e8f2ff", "bold", "center")
  pushViewport(viewport(x = 0.5, y = 0.188, width = 0.91, height = 0.245))
  grid.draw(table_grob)
  popViewport()

  pc_draw_text("Run Value Added uses the xRuns pitcher model adapted from game xwOBA; plus metrics compare this outing to the supplied MLB pitch baseline.",
               0.5, 0.042, 7.6, "#8fa3c0", "plain", "center")
  grid.rect(x = 0, y = 1, width = 0.012, height = 0.012, just = c("left", "top"),
            gp = gpar(fill = "#111827", col = NA))
  dev.off()

  invisible(list(
    output_png = output_png,
    pitcher = pitcher,
    game_date = game_date,
    pitch_summary = pitch_summary,
    game_xwoba = game_xwoba,
    xruns_saved = game_value$saved
  ))
}
