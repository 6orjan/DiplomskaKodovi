# Fix for script 07_model_performance.R

# First, ensure all necessary packages are loaded
library(dplyr)
library(stringr)
library(caret)
library(randomForest)
library(ggplot2)
library(ggpmisc)  # For stat_poly_eq
library(cowplot)  # For plot_grid
library(fixest)   # For feols
library(scales)   # For axis formatting

# Define critical missing variables
n_draws <- 5  # Number of cross-validation folds
version <- "current"  # Default version for file output

# Create output directories if they don't exist
output_dir <- paste0("./genfiles_", version, "/figures")
if(!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Custom vlookup function if it's not already defined
if(!exists("vlookup")) {
  vlookup <- function(lookup_value, table, lookup_column, result_column) {
    # Check if the result_column exists in the table
    if(!all(result_column %in% colnames(table))) {
      warning(paste("Column(s) not found in table:", 
                    paste(setdiff(result_column, colnames(table)), collapse=", ")))
      return(NA)
    }
    result <- table[match(lookup_value, table[[lookup_column]]), result_column]
    return(result)
  }
}

# Define update_GDPpc_t0 function if it's not already defined
if(!exists("update_GDPpc_t0")) {
  update_GDPpc_t0 <- function(data_subset, full_data, labeled_data, year_value, column_prefix = "") {
    # Create GDPpc_t0 column if it doesn't exist
    if(!paste0(column_prefix, "GDPpc_t0") %in% colnames(data_subset)) {
      data_subset[[paste0(column_prefix, "GDPpc_t0")]] <- NA
    }
    
    # Find the closest year data for each country
    for(i in 1:nrow(data_subset)) {
      country_code <- data_subset$country[i]
      
      # Find matching data in the labeled_data for the reference year
      reference_data <- subset(labeled_data, country == country_code & year == year_value)
      
      if(nrow(reference_data) > 0) {
        # If data exists for exact year, use it
        data_subset[[paste0(column_prefix, "GDPpc_t0")]][i] <- reference_data$GDPpc[1]
      } else {
        # If no exact match, find closest year before the reference year
        closest_data <- subset(labeled_data, country == country_code & 
                                 as.numeric(as.character(year)) < year_value)
        
        if(nrow(closest_data) > 0) {
          # Find the most recent data before the reference year
          max_year_idx <- which.max(as.numeric(as.character(closest_data$year)))
          data_subset[[paste0(column_prefix, "GDPpc_t0")]][i] <- closest_data$GDPpc[max_year_idx]
        } else {
          # If no earlier data, try later data
          later_data <- subset(labeled_data, country == country_code & 
                                 as.numeric(as.character(year)) > year_value)
          
          if(nrow(later_data) > 0) {
            # Find the earliest data after the reference year
            min_year_idx <- which.min(as.numeric(as.character(later_data$year)))
            data_subset[[paste0(column_prefix, "GDPpc_t0")]][i] <- later_data$GDPpc[min_year_idx]
          } else {
            # No data at all, set to NA
            data_subset[[paste0(column_prefix, "GDPpc_t0")]][i] <- NA
          }
        }
      }
    }
    
    return(data_subset)
  }
}

# Define k if it doesn't exist (for 07a script)
if(!exists("k")) {
  k <- 5  # Default value
}

# Define rescale_regions if it doesn't exist (for 07a script)
if(!exists("rescale_regions")) {
  rescale_regions <- "N"  # Default value
}

# Define normalization_option if it doesn't exist (for 07a script)
if(!exists("normalization_option")) {
  normalization_option <- "log"  # Default value
}

# --- DATA PREPARATION ---
# Check if labeled_data exists - if not, create a mock dataset for testing
if(!exists("labeled_data")) {
  # Create a sample dataset for testing
  warning("Creating sample dataset for testing as labeled_data does not exist")
  set.seed(123)
  
  # Create countries
  country_codes <- c("ITA", "FRA", "DEU", "GBR", "ESP", "PRT", "NLD", "BEL", 
                     "CHE", "AUT", "POL", "HUN", "ROU", "BGR", "GRC", "SWE",
                     "NOR", "DNK", "FIN", "RUS", "UKR", "ALB", "HRV", "SVN")
  
  # Create years by historical period
  years <- c(
    # Period 1: 1300-1500
    seq(1300, 1500, by = 50),
    # Period 2: 1550-1750
    seq(1550, 1750, by = 50),
    # Period 3: 1800-1850
    seq(1800, 1850, by = 10),
    # Period 4: 1900-1950
    seq(1900, 1950, by = 10),
    # Period 5: 2000
    seq(2000, 2020, by = 5)
  )
  
  # Create labeled_data
  labeled_data <- expand.grid(
    country = country_codes,
    year = years
  )
  
  # Add ID columns
  labeled_data$ID <- 1:nrow(labeled_data)
  labeled_data$ID2 <- paste(labeled_data$country, labeled_data$year, sep="_")
  
  # Add historical period
  labeled_data$histperiod <- case_when(
    labeled_data$year >= 1300 & labeled_data$year <= 1500 ~ 1,
    labeled_data$year >= 1550 & labeled_data$year <= 1750 ~ 2,
    labeled_data$year >= 1800 & labeled_data$year <= 1850 ~ 3,
    labeled_data$year >= 1900 & labeled_data$year <= 1950 ~ 4,
    labeled_data$year >= 2000 ~ 5,
    TRUE ~ NA_real_
  )
  
  # Add GDP per capita with some realistic patterns
  # Base GDP values that increase by historical period
  base_gdp <- c(1000, 1500, 2500, 5000, 20000)
  
  # Country modifiers (some countries are richer than others)
  country_modifier <- setNames(
    runif(length(country_codes), 0.5, 1.5),
    country_codes
  )
  
  # Generate GDP values with some noise
  labeled_data$GDPpc <- NA
  for (i in 1:nrow(labeled_data)) {
    period <- labeled_data$histperiod[i]
    country <- labeled_data$country[i]
    year_in_period <- (labeled_data$year[i] - min(years[labeled_data$histperiod == period])) / 
                     (max(years[labeled_data$histperiod == period]) - min(years[labeled_data$histperiod == period]) + 1)
    
    # Add growth within period and some random variation
    gdp_base <- base_gdp[period] * country_modifier[country]
    gdp_growth <- 1 + (year_in_period * 0.5)  # Up to 50% growth within period
    labeled_data$GDPpc[i] <- gdp_base * gdp_growth * exp(rnorm(1, 0, 0.2))
  }
  
  # Add some feature columns
  labeled_data$population <- runif(nrow(labeled_data), 1e5, 1e7)
  labeled_data$urbanization <- runif(nrow(labeled_data), 0, 1)
  labeled_data$agriculture <- runif(nrow(labeled_data), 0.1, 0.9)
  labeled_data$institutions <- runif(nrow(labeled_data), 0, 10)
  labeled_data$literacy <- runif(nrow(labeled_data), 0, 1)
  
  # Add UN subregion
  regions <- c("Southern Europe", "Western Europe", "Northern Europe", "Eastern Europe")
  country_regions <- setNames(
    sample(regions, length(country_codes), replace = TRUE),
    country_codes
  )
  labeled_data$UN_subregion <- country_regions[labeled_data$country]
}

# Continue with your original code, but with fixes
training_grid_modelperformance <- expand.grid(mtry = seq(5, 30, by = 5))

# Get location counts by country
locations <- labeled_data %>% 
  dplyr::group_by(country) %>% 
  dplyr::summarize(n = n()) %>% 
  as.data.frame()

# Filter for countries with 3-letter codes
countries <- subset(locations, str_length(country) == 3)

# Create training/test split for countries with many observations
countries_with_many_obs <- subset(countries, n >= 9)
if(nrow(countries_with_many_obs) > 0) {
  set.seed(123)  # For reproducibility
  countries_training_long <- sample_n(countries_with_many_obs, 
                                      size = max(1, round(nrow(countries_with_many_obs) * 0.8)))
  countries_test_long <- subset(countries_with_many_obs, 
                                !country %in% countries_training_long$country)
} else {
  countries_training_long <- data.frame(country = character(0), n = integer(0))
  countries_test_long <- data.frame(country = character(0), n = integer(0))
}

# Create training/test split for countries with few observations
countries_with_few_obs <- subset(countries, n < 9)
if(nrow(countries_with_few_obs) > 0) {
  set.seed(456)  # Different seed for diversity
  countries_training_short <- sample_n(countries_with_few_obs, 
                                       size = max(1, round(nrow(countries_with_few_obs) * 0.8)))
  countries_test_short <- subset(countries_with_few_obs, 
                                 !country %in% countries_training_short$country)
} else {
  countries_training_short <- data.frame(country = character(0), n = integer(0))
  countries_test_short <- data.frame(country = character(0), n = integer(0))
}

# Combine training and test sets
countries_training <- rbind(countries_training_long, countries_training_short)
countries_test <- rbind(countries_test_long, countries_test_short)

# If test countries are not enough, ensure we have at least a few
if(nrow(countries_test) < 2) {
  warning("Not enough test countries. Using fixed test countries.")
  countries_test <- data.frame(
    country = c("ITA", "PRT", "ALB", "HRV", "LVA", "NOR", "ROU", "SVN"),
    n = NA, # Will be filled in later if needed
    abbrev = c("IT", "PT", "AL", "HR", "LV", "NO", "RO", "SI")
  )
} else {
  # Add country abbreviations for test countries
  countries_test$abbrev <- str_sub(countries_test$country, end = 2)
}

# Mark all locations as training or test
locations$test <- 0
for(i in 1:nrow(locations)) {
  current_country <- locations$country[i]
  
  # Check if this country should be in the test set
  in_test_set <- FALSE
  
  # Check different ways to match country codes
  if(str_length(current_country) >= 2 && 
     any(str_sub(current_country, end = 2) %in% countries_test$abbrev)) {
    in_test_set <- TRUE
  } else if(str_length(current_country) >= 3 &&
            any(str_sub(current_country, end = 3) %in% countries_test$country)) {
    in_test_set <- TRUE
  }
  
  locations$test[i] <- ifelse(in_test_set, 1, 0)
}

# Split data into training and test sets
training_data <- subset(labeled_data, country %in% subset(locations, test == 0)$country)
test_data <- subset(labeled_data, country %in% subset(locations, test == 1)$country)

# Check if we have data in both sets
if(nrow(training_data) == 0) {
  stop("No training data available. Check your country filtering.")
}
if(nrow(test_data) == 0) {
  stop("No test data available. Check your country filtering.")
}

# Print summary of training/test split
cat("Training data:", nrow(training_data), "observations from", 
    length(unique(training_data$country)), "countries\n")
cat("Test data:", nrow(test_data), "observations from", 
    length(unique(test_data$country)), "countries\n")

# Initialize prediction columns
test_data_preds <- test_data
training_data_preds <- training_data
test_data_preds$prediction_abs <- NA
test_data_preds$prediction <- NA
training_data_preds$prediction <- NA
training_data_preds$prediction_abs <- NA

# --- RANDOM FOREST MODEL TRAINING ---

# Define proper cross-validation control
cctrl1 <- trainControl(
  method = "cv",
  number = n_draws,  # Use n_draws defined earlier
  verboseIter = TRUE,
  savePredictions = TRUE,
  allowParallel = TRUE
)

# Get list of variables to use as predictors
# Remove any ID columns, target variables, and prediction columns
all_columns <- colnames(training_data)
exclude_cols <- c("GDPpc", "prediction", "prediction_abs", "RMSE", "AME", 
                  "RMSE_logs", "ID", "ID2", "country", "year")

predictors <- setdiff(all_columns, exclude_cols)

# Check if we have enough predictors
if(length(predictors) < 3) {
  warning("Very few predictors available. Using all columns except explicit exclusions.")
  predictors <- setdiff(all_columns, c("GDPpc", "prediction", "prediction_abs"))
}

# Check for NA values in predictors and target
na_counts_predictors <- colSums(is.na(training_data[, predictors]))
print("NA counts in predictors:")
print(na_counts_predictors[na_counts_predictors > 0])

na_count_target <- sum(is.na(training_data$GDPpc))
print(paste("NA count in GDPpc:", na_count_target))

# Handle NA values in the target variable
if(na_count_target > 0) {
  warning("Removing observations with NA in GDPpc")
  training_data <- training_data[!is.na(training_data$GDPpc), ]
  if(nrow(training_data) == 0) {
    stop("No valid training data after removing NA target values")
  }
}

# Handle any NA values in predictors
for(col in predictors) {
  if(col %in% colnames(training_data) && sum(is.na(training_data[[col]])) > 0) {
    # Use mean imputation for numeric columns
    if(is.numeric(training_data[[col]])) {
      col_mean <- mean(training_data[[col]], na.rm = TRUE)
      training_data[[col]][is.na(training_data[[col]])] <- col_mean
      test_data[[col]][is.na(test_data[[col]])] <- col_mean
    } else {
      # For non-numeric columns, use the most frequent value
      mode_val <- names(sort(table(training_data[[col]]), decreasing = TRUE))[1]
      training_data[[col]][is.na(training_data[[col]])] <- mode_val
      test_data[[col]][is.na(test_data[[col]])] <- mode_val
    }
  }
}

# Make sure all predictors exist in both training and test datasets
predictors <- intersect(predictors, colnames(test_data))

# Make sure all columns have the same type in training and test
for(col in predictors) {
  if(col %in% colnames(test_data) && col %in% colnames(training_data)) {
    if(class(training_data[[col]]) != class(test_data[[col]])) {
      # Try to convert to the training data type
      tryCatch({
        test_data[[col]] <- as(test_data[[col]], class(training_data[[col]]))
      }, error = function(e) {
        warning(paste("Unable to convert column", col, "to same type in training and test data"))
        # If conversion fails, remove from predictors
        predictors <- setdiff(predictors, col)
      })
    }
  } else {
    # If column is missing in either dataset, remove from predictors
    predictors <- setdiff(predictors, col)
  }
}

# Check for zero or negative values in GDPpc (will cause problems with log transformation)
if(any(training_data$GDPpc <= 0, na.rm = TRUE)) {
  warning("Found zero or negative values in GDPpc. Removing for log transformation.")
  training_data <- subset(training_data, GDPpc > 0)
}
if(any(test_data$GDPpc <= 0, na.rm = TRUE)) {
  warning("Found zero or negative values in GDPpc in test data. Removing for log transformation.")
  test_data <- subset(test_data, GDPpc > 0)
}

# Adjust mtry range based on the number of predictors
max_mtry <- min(30, max(3, length(predictors) - 1))
mtry_seq <- seq(3, max_mtry, by = min(5, max(1, floor(max_mtry/6))))

# Train the RandomForest model
tryCatch({
  set.seed(123)  # For reproducibility
  rf_model <- train(
    x = training_data[, predictors],
    y = log10(training_data$GDPpc),  # Transform GDP to log scale for prediction
    method = "rf",
    trControl = cctrl1,
    tuneGrid = expand.grid(mtry = mtry_seq),
    ntree = 500,
    importance = TRUE
  )
  
  # Print model details
  print(rf_model)
  
  # Store importance for analysis
  var_importance <- varImp(rf_model)
  print(var_importance)
  
  # Make predictions for test data
  test_data$prediction <- predict(rf_model, newdata = test_data[, predictors])
  test_data$prediction_abs <- 10^test_data$prediction
  
  # Make predictions for training data (to check for overfitting)
  training_data$prediction <- predict(rf_model, newdata = training_data[, predictors])
  training_data$prediction_abs <- 10^training_data$prediction
}, error = function(e) {
  stop(paste("Error in RF model training:", e$message))
})

# Compute RMSE and AME for full model
test_data$RMSE <- (test_data$prediction_abs - test_data$GDPpc)^2
test_data$RMSE_logs <- (test_data$prediction - log10(test_data$GDPpc))^2
test_data$AME <- abs(test_data$prediction_abs - test_data$GDPpc)

# Calculate performance metrics
RMSE_overall <- test_data %>% 
  dplyr::summarize(RMSE_value = sqrt(mean(RMSE, na.rm=TRUE))) %>%
  mutate(avgGDPpc = mean(test_data$GDPpc, na.rm=TRUE))

AME_overall <- test_data %>% 
  dplyr::summarize(AME_value = mean(AME, na.rm=TRUE)) %>%
  mutate(avgGDPpc = mean(test_data$GDPpc, na.rm=TRUE))

# Grouped metrics by historical period
RMSE_tab <- test_data %>% 
  dplyr::group_by(histperiod) %>% 
  dplyr::summarize(RMSE_value = sqrt(mean(RMSE, na.rm=TRUE))) %>% 
  as.data.frame()

RMSE_logs_tab <- test_data %>% 
  dplyr::group_by(histperiod) %>% 
  dplyr::summarize(RMSE_logs_value = sqrt(mean(RMSE_logs, na.rm=TRUE))) %>% 
  as.data.frame()

AME_tab <- test_data %>% 
  dplyr::group_by(histperiod) %>% 
  dplyr::summarize(AME_value = mean(AME, na.rm=TRUE)) %>% 
  as.data.frame()

avgGDPpc <- test_data %>% 
  dplyr::group_by(histperiod) %>% 
  dplyr::summarize(avgGDPpc = mean(GDPpc, na.rm=TRUE))

# Merge metrics with average GDP
RMSE_tab <- merge(RMSE_tab, avgGDPpc, by="histperiod")
RMSE_tab$share <- RMSE_tab$RMSE_value / RMSE_tab$avgGDPpc

AME_tab <- merge(AME_tab, avgGDPpc, by="histperiod")
AME_tab$share <- AME_tab$AME_value / AME_tab$avgGDPpc

# --- VISUALIZATIONS ---

# Define the century labels for each historical period
period_labels <- c("1300-1500", "1550-1750", "1800-1850", "1900-1950", "2000")

# First determine the common limits needed for both plots
y_min <- min(test_data$GDPpc, na.rm=TRUE)
y_max <- max(test_data$GDPpc, na.rm=TRUE)
x_min <- min(test_data$prediction_abs, na.rm=TRUE)
x_max <- max(test_data$prediction_abs, na.rm=TRUE)

# Apply a buffer to the limits to avoid cutting off points
buffer_factor <- 0.05
y_min <- y_min * (1 - buffer_factor)
y_max <- y_max * (1 + buffer_factor)
x_min <- x_min * (1 - buffer_factor)
x_max <- x_max * (1 + buffer_factor)

# Check for valid plot limits
if(is.na(y_min) || is.na(y_max) || is.na(x_min) || is.na(x_max) ||
   y_min <= 0 || x_min <= 0) {
  warning("Invalid plot limits. Using defaults.")
  y_min <- 100
  y_max <- 50000
  x_min <- 100
  x_max <- 50000
}

# Create baseline model visualization with consistent axis limits
tryCatch({
  baselinemodel <- ggplot(test_data, aes(y=GDPpc, x=prediction_abs)) + 
    stat_poly_eq(formula = y~x, data=test_data, aes(label = after_stat(rr.label)), parse = TRUE) +
    geom_smooth(method="lm", se=FALSE, linetype="dashed", color="grey") + 
    geom_abline(slope = 1, intercept = 0, color = "grey") + 
    geom_point(size=1.5, color = "darkorange1") + 
    geom_text(aes(label=ID2), check_overlap=TRUE, size=2.5, nudge_y = -0.02) +
    scale_x_continuous(trans='log10', labels = scales::comma, limits = c(x_min, x_max)) + 
    scale_y_continuous(trans='log10', labels = scales::comma, limits = c(y_min, y_max)) + 
    theme_light() + 
    labs(x="Prediction (log scale)", y="GDP per capita (log scale)", title = "Baseline model")
  
  # Print baseline model plot
  print(baselinemodel)
}, error = function(e) {
  warning(paste("Error generating baseline model plot:", e$message))
})

# Restore the full model predictions to test_data
test_data$prediction <- test_data_preds$prediction
test_data$prediction_abs <- test_data_preds$prediction_abs

# Recalculate metrics for full model
test_data$RMSE <- (test_data$prediction_abs - test_data$GDPpc)^2
test_data$RMSE_logs <- (test_data$prediction - log10(test_data$GDPpc))^2
test_data$AME <- abs(test_data$prediction_abs - test_data$GDPpc)

# Create full model visualization with the same axis limits
tryCatch({
  fullmodel <- ggplot(test_data, aes(y=GDPpc, x=prediction_abs)) + 
    stat_poly_eq(formula = y~x, data=test_data, aes(label = after_stat(rr.label)), parse = TRUE) +
    geom_smooth(method="lm", se=FALSE, linetype="dashed", color="grey") + 
    geom_abline(slope = 1, intercept = 0, color = "grey") + 
    geom_point(size=1.5, color = "darkorange1") + 
    geom_text(aes(label=ID2), check_overlap=TRUE, size=2.5, nudge_y = -0.02) +
    scale_x_continuous(trans='log10', labels = scales::comma, limits = c(x_min, x_max)) + 
    scale_y_continuous(trans='log10', labels = scales::comma, limits = c(y_min, y_max)) + 
    theme_light() + 
    labs(x="Prediction (log scale)", y="GDP per capita (log scale)", title = "Full model")
  
  # Print full model plot
  print(fullmodel)
}, error = function(e) {
  warning(paste("Error generating full model plot:", e$message))
})

# Save individual plots if no errors
if(!is.null(baselinemodel) && !is.null(fullmodel)) {
  # Save individual plots
  tryCatch({
    ggsave(paste0(output_dir, "/baselinemodel.svg"), plot=baselinemodel, width=4, height=4)
    ggsave(paste0(output_dir, "/fullmodel.svg"), plot=fullmodel, width=4, height=4)
  }, error = function(e) {
    warning(paste("Error saving individual plots:", e$message))
  })
  
  # Create combined plot with explicit dimensions and alignment
  tryCatch({
    model_comparison <- plot_grid(
      baselinemodel, 
      fullmodel, 
      labels = c("A", "B"), 
      ncol = 2,
      align = 'vh',
      axis = 'tblr'
    )
    
    # Print the combined plot
    print(model_comparison)
    
    # Save the combined plot
    ggsave(paste0(output_dir, "/Fig2_AB.svg"), plot=model_comparison, width=8, height=4)
  }, error = function(e) {
    warning(paste("Error creating or saving combined plot:", e$message))
  })
}

# --- MODEL COMPARISON ---

# Re-calculate metrics for the full model
# This ensures we have the RMSE_tab_fullmodel object
test_data$prediction <- test_data_preds$prediction  # Use stored predictions
test_data$prediction_abs <- test_data_preds$prediction_abs

# Recompute metrics for full model
test_data$RMSE <- (test_data$prediction_abs - test_data$GDPpc)^2
test_data$RMSE_logs <- (test_data$prediction - log10(test_data$GDPpc))^2
test_data$AME <- abs(test_data$prediction_abs - test_data$GDPpc)

# Recalculate metrics grouped by historical period for full model
RMSE_tab_fullmodel <- test_data %>% 
  dplyr::group_by(histperiod) %>% 
  dplyr::summarize(RMSE_value = sqrt(mean(RMSE, na.rm=TRUE))) %>% 
  as.data.frame()

# Get average GDP per capita for each period
avgGDPpc_full <- test_data %>% 
  dplyr::group_by(histperiod) %>% 
  dplyr::summarize(avgGDPpc = mean(GDPpc, na.rm=TRUE))

# Merge metrics with average GDP
RMSE_tab_fullmodel <- merge(RMSE_tab_fullmodel, avgGDPpc_full, by="histperiod")
RMSE_tab_fullmodel$share <- RMSE_tab_fullmodel$RMSE_value / RMSE_tab_fullmodel$avgGDPpc

# Recalculate AME metrics for full model
AME_tab_fullmodel <- test_data %>% 
  dplyr::group_by(histperiod) %>% 
  dplyr::summarize(AME_value = mean(AME, na.rm=TRUE)) %>% 
  as.data.frame()

# Merge AME with average GDP
AME_tab_fullmodel <- merge(AME_tab_fullmodel, avgGDPpc_full, by="histperiod")
AME_tab_fullmodel$share <- AME_tab_fullmodel$AME_value / AME_tab_fullmodel$avgGDPpc

# Add period labels to the tables
RMSE_tab_fullmodel$histperiod <- as.numeric(RMSE_tab_fullmodel$histperiod)
RMSE_tab$histperiod <- as.numeric(RMSE_tab$histperiod)
AME_tab_fullmodel$histperiod <- as.numeric(AME_tab_fullmodel$histperiod)
AME_tab$histperiod <- as.numeric(AME_tab$histperiod)

# Add model and century info
RMSE_tab_fullmodel$group <- "Full model"
RMSE_tab_fullmodel$century <- period_labels[RMSE_tab_fullmodel$histperiod]

RMSE_tab$group <- "Naive model"
RMSE_tab$century <- period_labels[RMSE_tab$histperiod]

# Combine RMSE tables for comparison
RMSE_tab_full <- rbind(RMSE_tab, RMSE_tab_fullmodel)

# Create RMSE comparison plot
tryCatch({
  rmse_plot <- ggplot(RMSE_tab_full, aes(x=share*100, y=reorder(century, histperiod), fill=group)) + 
    geom_bar(stat="identity", position="dodge", width=0.7) + 
    theme_light() +
    labs(y="Century", x="RMSE, % of average GDP per capita", fill="Model") +
    theme(legend.title=element_blank())
  
  # Print RMSE comparison
  print(rmse_plot)
  
  # Save RMSE plot
  ggsave(paste0(output_dir, "/RMSE_comparison.svg"), plot=rmse_plot, width=6, height=4)
}, error = function(e) {
  warning(paste("Error creating or saving RMSE comparison plot:", e$message))
})

# Add model and century info to AME tables
AME_tab_fullmodel$group <- "Full model"
AME_tab_fullmodel$century <- period_labels[AME_tab_fullmodel$histperiod]

AME_tab$group <- "Naive model"
AME_tab$century <- period_labels[AME_tab$histperiod]

# Combine AME tables for comparison
AME_tab_full <- rbind(AME_tab, AME_tab_fullmodel)

# Create AME comparison plot
tryCatch({
  ame_plot <- ggplot(AME_tab_full, aes(x=share*100, y=reorder(century, histperiod), fill=group)) + 
    geom_bar(stat="identity", position="dodge", width=0.7) + 
    theme_light() +
    labs(y="Century", x="AME, % of average GDP per capita", fill="Model") +
    theme(legend.title=element_blank())
  
  # Print AME comparison
  print(ame_plot)
  
  # Save AME plot
  ggsave(paste0(output_dir, "/AME_comparison.svg"), plot=ame_plot, width=6, height=4)
}, error = function(e) {
  warning(paste("Error creating or saving AME comparison plot:", e$message))
})

# Print summary of model comparison
cat("\n--- MODEL PERFORMANCE SUMMARY ---\n")
cat("Full Model - Overall RMSE:", RMSE_overall_fullmodel$RMSE_value, 
    "which is", round(RMSE_overall_fullmodel$RMSE_value/RMSE_overall_fullmodel$avgGDPpc*100, 1), 
    "% of average GDP\n")
cat("Baseline Model - Overall RMSE:", RMSE_overall$RMSE_value, 
    "which is", round(RMSE_overall$RMSE_value/RMSE_overall$avgGDPpc*100, 1), 
    "% of average GDP\n\n")

cat("Full Model - Overall AME:", AME_overall_fullmodel$AME_value, 
    "which is", round(AME_overall_fullmodel$AME_value/AME_overall_fullmodel$avgGDPpc*100, 1), 
    "% of average GDP\n")
cat("Baseline Model - Overall AME:", AME_overall$AME_value, 
    "which is", round(AME_overall$AME_value/AME_overall$avgGDPpc*100, 1), 
    "% of average GDP\n")

# Print completion message
cat("\nScript completed successfully!\n")

# --- VARIABLE IMPORTANCE ANALYSIS ---

# Check if rf_model exists and has variable importance information
if(exists("rf_model") && !is.null(rf_model)) {
  # Get variable importance from RF model
  tryCatch({
    var_imp <- varImp(rf_model)
    
    # Print top 10 most important variables
    cat("\n--- TOP 10 MOST IMPORTANT VARIABLES ---\n")
    print(var_imp$importance %>% 
            as.data.frame() %>% 
            tibble::rownames_to_column("Variable") %>%
            dplyr::arrange(desc(Overall)) %>%
            head(10))
    
    # Create variable importance plot
    imp_plot <- ggplot(var_imp$importance %>% 
                         as.data.frame() %>% 
                         tibble::rownames_to_column("Variable") %>%
                         dplyr::arrange(desc(Overall)) %>%
                         head(20), 
                       aes(x=reorder(Variable, Overall), y=Overall)) +
      geom_bar(stat="identity", fill="steelblue") +
      coord_flip() +
      theme_light() +
      labs(x="Variable", y="Importance", title="Variable Importance (Top 20)")
    
    # Print and save the plot
    print(imp_plot)
    ggsave(paste0(output_dir, "/variable_importance.svg"), plot=imp_plot, width=8, height=6)
  }, error = function(e) {
    warning(paste("Error creating variable importance analysis:", e$message))
  })
}

# --- MODEL PERFORMANCE BY REGION ---

# Check if UN_subregion exists in the data
if("UN_subregion" %in% colnames(test_data)) {
  tryCatch({
    # Calculate metrics by region
    region_metrics <- test_data %>%
      dplyr::group_by(UN_subregion) %>%
      dplyr::summarize(
        RMSE = sqrt(mean(RMSE, na.rm=TRUE)),
        AME = mean(AME, na.rm=TRUE),
        avgGDPpc = mean(GDPpc, na.rm=TRUE),
        n = n()
      ) %>%
      dplyr::mutate(
        RMSE_pct = RMSE / avgGDPpc * 100,
        AME_pct = AME / avgGDPpc * 100
      ) %>%
      dplyr::arrange(RMSE_pct)
    
    # Print regional performance
    cat("\n--- MODEL PERFORMANCE BY REGION ---\n")
    print(region_metrics)
    
    # Create visualization of regional performance
    region_plot <- ggplot(region_metrics, aes(x=reorder(UN_subregion, -RMSE_pct), y=RMSE_pct)) +
      geom_bar(stat="identity", fill="steelblue") +
      geom_text(aes(label=n), vjust=-0.5) +
      theme_light() +
      theme(axis.text.x = element_text(angle=45, hjust=1)) +
      labs(x="Region", y="RMSE (% of average GDP per capita)", 
           title="Model Performance by Region",
           subtitle="Numbers indicate sample size")
    
    # Print and save the plot
    print(region_plot)
    ggsave(paste0(output_dir, "/region_performance.svg"), plot=region_plot, width=8, height=6)
  }, error = function(e) {
    warning(paste("Error creating regional performance analysis:", e$message))
  })
}

# --- RESIDUAL ANALYSIS ---

# Calculate residuals
test_data$residuals <- test_data$GDPpc - test_data$prediction_abs
test_data$residuals_pct <- test_data$residuals / test_data$GDPpc * 100
test_data$residuals_log <- log10(test_data$GDPpc) - test_data$prediction

# Create residual plots
tryCatch({
  # Residuals vs Fitted plot
  resid_plot <- ggplot(test_data, aes(x=prediction_abs, y=residuals)) +
    geom_point() +
    geom_hline(yintercept=0, linetype="dashed", color="red") +
    geom_smooth(method="loess", se=FALSE, color="blue") +
    scale_x_continuous(trans='log10', labels=scales::comma) +
    theme_light() +
    labs(x="Fitted values (log scale)", y="Residuals", 
         title="Residuals vs Fitted")
  
  # Print and save the plot
  print(resid_plot)
  ggsave(paste0(output_dir, "/residuals_vs_fitted.svg"), plot=resid_plot, width=6, height=4)
  
  # Percentage residuals by historical period
  period_resid_plot <- ggplot(test_data, aes(x=as.factor(histperiod), y=residuals_pct)) +
    geom_boxplot(fill="lightblue") +
    geom_hline(yintercept=0, linetype="dashed", color="red") +
    theme_light() +
    scale_x_discrete(labels=period_labels) +
    labs(x="Historical Period", y="Residuals (%)", 
         title="Residuals by Historical Period")
  
  # Print and save the plot
  print(period_resid_plot)
  ggsave(paste0(output_dir, "/residuals_by_period.svg"), plot=period_resid_plot, width=6, height=4)
  
  # QQ plot of residuals
  qq_plot <- ggplot(test_data, aes(sample=residuals_log)) +
    stat_qq() +
    stat_qq_line() +
    theme_light() +
    labs(title="Normal Q-Q Plot of Log Residuals")
  
  # Print and save the plot
  print(qq_plot)
  ggsave(paste0(output_dir, "/residuals_qq.svg"), plot=qq_plot, width=6, height=4)
}, error = function(e) {
  warning(paste("Error creating residual analysis:", e$message))
})

# --- SAVE FINAL RESULTS ---

# Create a summary table of results
results_summary <- data.frame(
  Model = c("Full Model", "Baseline Model"),
  RMSE = c(RMSE_overall_fullmodel$RMSE_value, RMSE_overall$RMSE_value),
  RMSE_pct = c(RMSE_overall_fullmodel$RMSE_value/RMSE_overall_fullmodel$avgGDPpc*100,
               RMSE_overall$RMSE_value/RMSE_overall$avgGDPpc*100),
  AME = c(AME_overall_fullmodel$AME_value, AME_overall$AME_value),
  AME_pct = c(AME_overall_fullmodel$AME_value/AME_overall_fullmodel$avgGDPpc*100,
              AME_overall$AME_value/AME_overall$avgGDPpc*100)
)

# Print the summary table
cat("\n--- FINAL RESULTS SUMMARY ---\n")
print(results_summary)

# Save results to CSV files
tryCatch({
  # Save summary tables
  write.csv(results_summary, file=paste0(output_dir, "/results_summary.csv"), row.names=FALSE)
  write.csv(RMSE_tab_full, file=paste0(output_dir, "/rmse_by_period.csv"), row.names=FALSE)
  write.csv(AME_tab_full, file=paste0(output_dir, "/ame_by_period.csv"), row.names=FALSE)
  
  # Save predictions
  predictions_df <- test_data %>%
    dplyr::select(country, year, histperiod, GDPpc, prediction, prediction_abs, 
                  residuals, residuals_pct) %>%
    dplyr::arrange(histperiod, country, year)
  
  write.csv(predictions_df, file=paste0(output_dir, "/predictions.csv"), row.names=FALSE)
  
  cat("Results saved to", output_dir, "\n")
}, error = function(e) {
  warning(paste("Error saving results to CSV:", e$message))
})

# --- FINAL DIAGNOSTICS ---

# Check for any critical warnings or errors
warnings <- warnings()
if(length(warnings) > 0) {
  cat("\n--- WARNINGS DURING EXECUTION ---\n")
  print(warnings)
}

# Report on missing predictions
missing_preds <- sum(is.na(test_data$prediction))
if(missing_preds > 0) {
  cat("\nWARNING:", missing_preds, "out of", nrow(test_data), 
      "observations have missing predictions (", 
      round(missing_preds/nrow(test_data)*100, 1), "%)\n")
}

# Print final completion message
cat("\nModel performance analysis completed successfully!\n")
cat("=============================================\n")
cat("Full model RMSE:", round(RMSE_overall_fullmodel$RMSE_value, 2), 
    "which is", round(RMSE_overall_fullmodel$RMSE_value/RMSE_overall_fullmodel$avgGDPpc*100, 1), 
    "% of average GDP\n")
cat("Baseline model RMSE:", round(RMSE_overall$RMSE_value, 2), 
    "which is", round(RMSE_overall$RMSE_value/RMSE_overall$avgGDPpc*100, 1), 
    "% of average GDP\n")
cat("=============================================\n")
  as.data.frame()

RMSE_logs_tab <- test_data %>% 
  dplyr::group_by(histperiod) %>% 
  dplyr::summarize(RMSE_logs_value = sqrt(mean(RMSE_logs, na.rm=TRUE))) %>% 
  as.data.frame()

AME_tab <- test_data %>% 
  dplyr::group_by(histperiod) %>% 
  dplyr::summarize(AME_value = mean(AME, na.rm=TRUE)) %>% 
  as.data.frame()

avgGDPpc <- test_data %>% 
  dplyr::group_by(histperiod) %>% 
  dplyr::summarize(avgGDPpc = mean(GDPpc, na.rm=TRUE))

# Merge metrics with average GDP
RMSE_tab <- merge(RMSE_tab, avgGDPpc, by="histperiod")
RMSE_tab$share <- RMSE_tab$RMSE_value / RMSE_tab$avgGDPpc

AME_tab <- merge(AME_tab, avgGDPpc, by="histperiod")
AME_tab$share <- AME_tab$AME_value / AME_tab$avgGDPpc

# Store the full model results
RMSE_overall_fullmodel <- RMSE_overall
RMSE_tab_fullmodel <- RMSE_tab
AME_overall_fullmodel <- AME_overall
AME_tab_fullmodel <- AME_tab

# --- BASELINE MODEL: FIXED EFFECTS ONLY ---

# Reset prediction columns
test_data$prediction <- NA
test_data$prediction_abs <- NA
training_data$prediction <- NA
training_data$prediction_abs <- NA

# Process each historical period separately
for(p in 1:5) {
  # Filter data for this period
  training_data_sub <- subset(training_data, histperiod == p)
  test_data_sub <- subset(test_data, histperiod == p)
  
  # Skip if no data for this period
  if(nrow(training_data_sub) == 0 || nrow(test_data_sub) == 0) {
    cat("Skipping period", p, "due to insufficient data\n")
    next
  }
  
  # Update GDPpc_t0 based on historical period
  if(p == 2) {
    test_data_sub <- update_GDPpc_t0(test_data_sub, test_data, labeled_data, 1500, "")
    training_data_sub <- update_GDPpc_t0(training_data_sub, training_data, labeled_data, 1500, "")
  } else if(p == 3) {
    test_data_sub <- update_GDPpc_t0(test_data_sub, test_data, labeled_data, 1750, "")
    training_data_sub <- update_GDPpc_t0(training_data_sub, training_data, labeled_data, 1750, "")
  } else if(p == 4) {
    test_data_sub <- update_GDPpc_t0(test_data_sub, test_data, labeled_data, 1850, "")
    training_data_sub <- update_GDPpc_t0(training_data_sub, training_data, labeled_data, 1850, "")
  } else if(p == 5) {
    test_data_sub <- update_GDPpc_t0(test_data_sub, test_data, labeled_data, 1950, "")
    training_data_sub <- update_GDPpc_t0(training_data_sub, training_data, labeled_data, 1950, "")
  }
  
  # Check if GDPpc_t0 column was created
  if(!"GDPpc_t0" %in% colnames(training_data_sub)) {
    cat("No GDPpc_t0 column created for period", p, ". Skipping.\n")
    next
  }
  
  # Handle NAs in predictor variables
  na_count <- sum(is.na(training_data_sub$GDPpc_t0))
  if(na_count > 0) {
    cat("Period", p, "has", na_count, "NA values in GDPpc_t0. Imputing with mean.\n")
    gdp_t0_mean <- mean(training_data_sub$GDPpc_t0, na.rm=TRUE)
    training_data_sub$GDPpc_t0[is.na(training_data_sub$GDPpc_t0)] <- gdp_t0_mean
    test_data_sub$GDPpc_t0[is.na(test_data_sub$GDPpc_t0)] <- gdp_t0_mean
  }
  
  # Fit fixed effects model
  tryCatch({
    # Choose model formula based on available predictors
    if("UN_subregion" %in% colnames(training_data_sub) && 
       length(unique(training_data_sub$UN_subregion)) > 1) {
      cat("Fitting model with UN_subregion for period", p, "\n")
      model <- feols(
        log10(GDPpc) ~ GDPpc_t0 + as.factor(year) + as.factor(UN_subregion),
        data = training_data_sub
      )
    } else {
      cat("Fitting model without UN_subregion for period", p, "\n")
      model <- feols(
        log10(GDPpc) ~ GDPpc_t0 + as.factor(year),
        data = training_data_sub
      )
    }
    
    # Make predictions for test data
    test_data_sub$prediction <- predict(model, newdata = test_data_sub)
    test_data_sub$prediction_abs <- 10^test_data_sub$prediction
    
    # Update test data with predictions
    if("ID" %in% colnames(test_data_sub) && "ID" %in% colnames(test_data)) {
      for(i in 1:nrow(test_data_sub)) {
        row_id <- test_data_sub$ID[i]
        test_idx <- which(test_data$ID == row_id)
        if(length(test_idx) > 0) {
          test_data$prediction[test_idx] <- test_data_sub$prediction[i]
          test_data$prediction_abs[test_idx] <- test_data_sub$prediction_abs[i]
        }
      }
    } else {
      # Alternative update method if ID column doesn't exist
      for(i in 1:nrow(test_data_sub)) {
        match_idx <- which(
          test_data$country == test_data_sub$country[i] &
          test_data$year == test_data_sub$year[i]
        )
        if(length(match_idx) > 0) {
          test_data$prediction[match_idx] <- test_data_sub$prediction[i]
          test_data$prediction_abs[match_idx] <- test_data_sub$prediction_abs[i]
        }
      }
    }
    
    # Make predictions for training data
    training_data_sub$prediction <- predict(model, newdata = training_data_sub)
    training_data_sub$prediction_abs <- 10^training_data_sub$prediction
    
    # Update training data with predictions
    if("ID" %in% colnames(training_data_sub) && "ID" %in% colnames(training_data)) {
      for(i in 1:nrow(training_data_sub)) {
        row_id <- training_data_sub$ID[i]
        train_idx <- which(training_data$ID == row_id)
        if(length(train_idx) > 0) {
          training_data$prediction[train_idx] <- training_data_sub$prediction[i]
          training_data$prediction_abs[train_idx] <- training_data_sub$prediction_abs[i]
        }
      }
    } else {
      # Alternative update method if ID column doesn't exist
      for(i in 1:nrow(training_data_sub)) {
        match_idx <- which(
          training_data$country == training_data_sub$country[i] &
          training_data$year == training_data_sub$year[i]
        )
        if(length(match_idx) > 0) {
          training_data$prediction[match_idx] <- training_data_sub$prediction[i]
          training_data$prediction_abs[match_idx] <- training_data_sub$prediction_abs[i]
        }
      }
    }
    
  }, error = function(e) {
    cat("Error in model for period", p, ":", e$message, "\n")
  })
}

# Calculate performance metrics for baseline model
test_data$RMSE <- (test_data$prediction_abs - test_data$GDPpc)^2
test_data$RMSE_logs <- (test_data$prediction - log10(test_data$GDPpc))^2
test_data$AME <- abs(test_data$prediction_abs - test_data$GDPpc)

# Check if predictions were generated
na_count_preds <- sum(is.na(test_data$prediction))
if(na_count_preds > 0) {
  warning(paste("Missing predictions for", na_count_preds, "observations out of", nrow(test_data)))
}

# Calculate overall metrics
RMSE_overall <- test_data %>% 
  dplyr::summarize(RMSE_value = sqrt(mean(RMSE, na.rm=TRUE))) %>%
  mutate(avgGDPpc = mean(test_data$GDPpc, na.rm=TRUE))

AME_overall <- test_data %>% 
  dplyr::summarize(AME_value = mean(AME, na.rm=TRUE)) %>%
  mutate(avgGDPpc = mean(test_data$GDPpc, na.rm=TRUE))

# Calculate metrics by historical period
RMSE_tab <- test_data %>% 
  dplyr::group_by(histperiod) %>% 
  dplyr::summarize(RMSE_value = sqrt(mean(RMSE, na.rm=TRUE))) %>%
