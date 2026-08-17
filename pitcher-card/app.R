library(shiny)
library(bslib)

app_dir <- normalizePath(getwd(), mustWork = TRUE)
source(file.path(app_dir, "make_pitcher_card.R"), local = TRUE)

default_baseline <- file.path(app_dir, "mlb_pitch_baseline_2026.csv")
default_pitch_data <- file.path(app_dir, "pitch_data.csv")
export_dir <- file.path(app_dir, "exports")
export_prefix <- "xruns-card-exports"

dir.create(export_dir, showWarnings = FALSE, recursive = TRUE)
if (!export_prefix %in% names(resourcePaths())) {
  addResourcePath(export_prefix, export_dir)
}

ensure_default_baseline <- function() {
  if (file.exists(default_baseline)) return(default_baseline)
  if (!file.exists(default_pitch_data)) return(NA_character_)
  source(file.path(app_dir, "build_pitch_baseline.R"), local = TRUE)
  build_pitch_baseline(default_pitch_data, default_baseline)
  default_baseline
}

ui <- page_fillable(
  theme = bs_theme(
    version = 5,
    bg = "#111827",
    fg = "#eaf3ff",
    primary = "#35d0ff",
    base_font = font_google("Inter")
  ),
  tags$style(HTML("
    .xruns-app-shell { max-width: 1180px; margin: 0 auto; width: 100%; }
    .xruns-title { color:#35d0ff; font-weight:800; letter-spacing:.02em; }
    .xruns-card {
      background:#17233a; border:1px solid #21b6d7; border-radius:14px;
      padding:18px; box-shadow:0 18px 45px rgba(0,0,0,.22);
    }
    .form-label, .control-label { color:#c2d2e8; font-weight:700; }
    .help-block { color:#8fa3c0; }
    .btn-primary { color:#07111f; font-weight:800; }
    .xruns-download {
      display:inline-block; margin-top:10px; text-decoration:none;
    }
    .xruns-download-disabled {
      display:inline-block; margin-top:10px; opacity:.45; pointer-events:none;
    }
    .xruns-preview img {
      max-width:100%; height:auto; border-radius:10px; border:1px solid #294b6f;
      background:#111827;
    }
    .shiny-notification { background:#17233a; color:#eaf3ff; border-color:#21b6d7; }
  ")),
  div(
    class = "xruns-app-shell",
    layout_columns(
      col_widths = c(4, 8),
      div(
        class = "xruns-card",
        h2(class = "xruns-title", "xRuns Pitcher Card Maker"),
        p("Upload a Baseball Savant pitch-level game CSV and export a shareable PNG."),
        fileInput(
          "game_csv",
          "Game CSV",
          accept = c(".csv", "text/csv")
        ),
        fileInput(
          "baseline_csv",
          "Optional baseline CSV",
          accept = c(".csv", "text/csv"),
          placeholder = "Uses mlb_pitch_baseline_2026.csv by default"
        ),
        radioButtons(
          "layout",
          "Card Layout",
          choices = c("Square" = "square", "Landscape" = "landscape"),
          selected = "square",
          inline = TRUE
        ),
        actionButton("build", "Create Card", class = "btn-primary"),
        uiOutput("download_link"),
        hr(),
        p(class = "help-block",
          "Default baseline: pitcher-card/mlb_pitch_baseline_2026.csv. ",
          "If it is missing, the app will try to build it from pitcher-card/pitch_data.csv."
        )
      ),
      div(
        class = "xruns-card xruns-preview",
        h3("Preview"),
        uiOutput("status_text"),
        imageOutput("preview", height = "auto")
      )
    )
  )
)

server <- function(input, output, session) {
  current_card <- reactiveVal(NULL)
  current_name <- reactiveVal("xruns_pitcher_card.png")
  current_href <- reactiveVal(NULL)
  current_status <- reactiveVal("Upload a game CSV, then create a card.")

  observeEvent(input$build, {
    req(input$game_csv)
    baseline <- if (!is.null(input$baseline_csv)) {
      input$baseline_csv$datapath
    } else {
      ensure_default_baseline()
    }

    if (is.na(baseline) || !file.exists(baseline)) {
      showNotification("No baseline found. Upload a baseline CSV or add pitch_data.csv to pitcher-card/.", type = "error")
      return()
    }

    ext <- if (identical(input$layout, "landscape")) "_landscape.png" else "_square.png"
    out <- tempfile(pattern = "xruns_pitcher_card_", fileext = ext)

    withProgress(message = "Creating pitcher card...", value = 0.2, {
      result <- tryCatch(
        {
          make_pitcher_card(
            game_csv = input$game_csv$datapath,
            baseline_csv = baseline,
            output_png = out,
            layout = input$layout,
            data_dir = dirname(app_dir)
          )
        },
        error = function(e) e
      )
      incProgress(0.8)

      if (inherits(result, "error")) {
        showNotification(conditionMessage(result), type = "error", duration = 10)
        current_status("Card creation failed. Check that the upload is a Baseball Savant pitch-level CSV.")
        return()
      }

      safe_player <- gsub("[^A-Za-z0-9]+", "_", tolower(result$pitcher))
      safe_date <- gsub("[^0-9-]+", "", result$game_date)
      file_name <- paste0("xruns_", safe_player, "_", safe_date, "_", input$layout, ".png")
      export_path <- file.path(export_dir, file_name)
      file.copy(out, export_path, overwrite = TRUE)
      current_name(file_name)
      current_card(export_path)
      current_href(paste0(export_prefix, "/", file_name))
      current_status(sprintf(
        "%s card created for %s: %+0.2f Run Value Added.",
        tools::toTitleCase(input$layout),
        result$pitcher,
        result$xruns_saved
      ))
    })
  })

  output$status_text <- renderUI({
    p(class = "help-block", current_status())
  })

  output$download_link <- renderUI({
    href <- current_href()
    if (is.null(href)) {
      tags$a(
        class = "btn btn-primary xruns-download-disabled",
        href = "#",
        "Download PNG"
      )
    } else {
      tags$a(
        class = "btn btn-primary xruns-download",
        href = href,
        download = current_name(),
        target = "_blank",
        "Download PNG"
      )
    }
  })

  output$preview <- renderImage({
    path <- current_card()
    if (is.null(path) || !file.exists(path)) return(NULL)
    list(src = path, contentType = "image/png", alt = "xRuns pitcher card")
  }, deleteFile = FALSE)

}

shinyApp(ui, server)
