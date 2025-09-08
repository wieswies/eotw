Sys.setenv(GITHUB_PAT = "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")
library(paperboy)
library(httr)
library(dplyr)
library(progress)

base_path <- "/Users/wiesruyters/Library/CloudStorage/OneDrive-SharedLibraries-WageningenUniversity&Research/Business - Documents/Wies/WhD/Repositories for publication/embedding_on_the_wall__NL-UK/b__data-collection-with-web-scraping/"

# Input paths
uk_urls <- read.csv(file.path(base_path, "datasets/news/UK input/uk_article_urls.csv"))
nl_urls <- read.csv(file.path(base_path, "datasets/news/NL input/nl_article_urls.csv"))

# Output paths
out_dir <- file.path(base_path, "datasets/news")
dir.create(out_dir, showWarnings = FALSE)  # Ensure directory exists

uk_csv  <- file.path(out_dir, "uk_paperboy_results.csv")
nl_csv  <- file.path(out_dir, "nl_paperboy_results.csv")
uk_rds  <- file.path(out_dir, "uk_paperboy_results.rds")
nl_rds  <- file.path(out_dir, "nl_paperboy_results.rds")

flatten_df <- function(df) {
  for (col in names(df)) {
    if (is.list(df[[col]])) {
      df[[col]] <- sapply(df[[col]], function(x) {
        if (is.null(x)) return("")
        paste(unlist(x), collapse = "; ")
      })
    }
  }
  df
}

# Enhanced CSV writing function
write_results_csv <- function(df, path) {
  flattened <- flatten_df(df)
  
  # Convert all columns to UTF-8
  flattened[] <- lapply(flattened, function(x) {
    if (is.character(x)) enc2utf8(x) else x
  })
  
  # Write with proper encoding and quotes where needed
  write.csv(flattened, 
            file = path,
            row.names = FALSE,
            fileEncoding = "UTF-8",
            quote = TRUE,
            na = "")
}

# Modified scraper function with better CSV handling
scrape_urls <- function(urls, csv_path, rds_path, failed_path) {
  # Initialize empty files if they don't exist
  if (!file.exists(csv_path)) {
    write.csv(data.frame(), csv_path, row.names = FALSE)
  }
  if (!file.exists(failed_path)) {
    writeLines(character(), failed_path)
  }
  
  # Resume: load previous results if available
  results_list <- if (file.exists(rds_path)) readRDS(rds_path) else list()
  processed <- names(results_list)
  
  # Progress bar
  n <- nrow(urls)
  pb <- progress_bar$new(
    format = "Scraping [:bar] :current/:total (:percent) ETA: :eta",
    total = nrow(urls), clear = FALSE, width = 80
  )
  for (i in seq_len(n)) {
    url <- urls[i, "url"]
    
    # Skip if already processed
    if (url %in% processed) {
        next
      }
      
      tryCatch({
        # Scrape (suppress console spam)
        article <- suppressMessages(pb_deliver(url))
        
        # Check
        if (is.data.frame(article) && nrow(article) > 0) {
          # Save in results list
          results_list[[url]] <- article
          
          # 1) Write complete CSV each time (overwrite)
          combined_df <- bind_rows(results_list)
          write_results_csv(combined_df, csv_path)
          
          # 2) Save full RDS every 25 articles for better recovery
          if (i %% 25 == 0) saveRDS(results_list, rds_path)
          
        } else {
          stop("Empty response")
        }
        
      }, error = function(e) {
        cat("Failed:", url, "(", e$message, ")\n")
        write(url, file = failed_path, append = TRUE)
      })
      
    pb$tick()
    Sys.sleep(runif(1, 0, 2))  # More conservative delay
  }
  
  saveRDS(results_list, rds_path)  # final save
  cat("\nScraping finished! Results saved to:\n",
      "- CSV:", csv_path, "\n",
      "- RDS:", rds_path, "\n",
      "- Failed URLs:", failed_path, "\n")
}

# Scrape UK
scrape_urls(
  uk_urls,
  csv_path    = uk_csv,
  rds_path    = uk_rds,
  failed_path = file.path(out_dir, "uk_paperboy_failed_responses.csv")
)
