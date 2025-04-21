rm(list = ls())

# Load required packages
pacman::p_load(
  ggplot2, glmnet, reshape2, haven, irr, future.apply, ICC, broom, stringr, knitr,
  kableExtra, officer, flextable, cowplot, EconGeo, modi, eurostat, sf, raster,
  DataVisualizations, plotly, xlsx, plm, fixest, countrycode, ggpmisc, car, plyr,
  pals, htmlwidgets, forcats, ds4psy, rsample, dplyr, rpart, rpart.plot, ipred, 
  caret, BMS, Matrix, BMA, data.table, modelsummary, sandwich, boot, sp, rgdal, 
  treemap, RColorBrewer, parallel, foreach, doParallel, randomForest, nnet, xgboost
)


set.seed(8765)

# Custom function
source("./misc/R_functions/vlookup.R")

# PARAMETERS
lag_parameter <- 150
normalization_option <- "log"
k <- 10
HPI_incl <- "Y"
rescale_regions <- "Y"
shapley <- "Y"
standarderrors <- "Y"
n_bootstraps <- 500
bootstrap_CI <- c(0.05, 0.95)
n_draws <- 3

version <- paste0(Sys.Date(), "_COMPLETE")

# Create output directories
output_dirs <- c(
  "figures", "figures_SI", "figures_SI/ECI", "figures_SI/LASSO_results", 
  "figures_SI/SVD", "figures_SI/RF_results",  # <--- ADD THIS LINE
  "maps", "maps/famous_individuals", "maps/Maddison", 
  "maps/Maddison+ML", "maps/Maddison+ML/countries", "maps/Maddison+ML/regions"
)

base_dir <- paste0("./genfiles_", version)
if (dir.exists(base_dir)) unlink(base_dir, recursive = TRUE)
dir.create(base_dir)
lapply(file.path(base_dir, output_dirs), dir.create)

# Load data
data <- readRDS("./misc/data_inputToML_v1.rds")

labeled_data <- subset(data, !is.na(GDPpc))
unlabeled_data <- subset(data, is.na(GDPpc))

# Compute model performance
message("🔍 Running model performance...")
source("./scripts/07_model_performance.R")

# Run final model
message("⚙️ Running final model and predictions...")
final_model_success <- tryCatch({
  source("./scripts/08_finalmodel.R")
  TRUE
}, error = function(e) {
  warning("❌ Error in final model: ", conditionMessage(e))
  FALSE
})

# Show structure of data to make sure finalmodel didn’t wipe it
message("🧾 Data structure after final model:")
print(str(data))  # helpful for debugging

# Generate maps regardless of model outcome
message("🗺️ Generating maps...")
tryCatch({
  source("./scripts/06_generate_maps.R")
  message("✅ Maps generated successfully.")
}, error = function(e) {
  message("❌ Error in generating maps: ", conditionMessage(e))
  print(e)
})

# Generate descriptives
message("📊 Generating descriptives...")
tryCatch({
  source("./scripts/09_descriptives.R")
  message("✅ Descriptives generated successfully.")
}, error = function(e) {
  warning("❌ Error in descriptives: ", conditionMessage(e))
})
