### Fixed 07a_RF_for_modelperformance.R ###

# Load necessary packages
library(randomForest)
library(ranger)
library(caret)
library(dplyr)
library(stringr)

# Initialize prediction variables
test_data_preds <- test_data
test_data_preds$prediction_abs <- NA
test_data_preds$prediction <- NA

training_data_preds <- training_data
training_data_preds$prediction <- NA
training_data_preds$prediction_abs <- NA

# Function to update GDPpc_t0 - Unified implementation combining both approaches
update_GDPpc_t0 <- function(data_subset, full_data, labeled_data, year_value, column_prefix = "") {
  # Create GDPpc_t0 column if it doesn't exist
  if(!paste0(column_prefix, "GDPpc_t0") %in% colnames(data_subset)) {
    data_subset[[paste0(column_prefix, "GDPpc_t0")]] <- NA
  }
  
  for(m in 1:nrow(data_subset)) {
    # Try the direct ID2 lookup first
    tempGDP <- NA
    tryCatch({
      id2_key <- paste(data_subset$country[m], year_value, sep="_")
      if("ID2" %in% colnames(full_data)) {
        matched_rows <- full_data$ID2 == id2_key
        if(any(matched_rows) && "prediction" %in% colnames(full_data)) {
          tempGDP <- full_data$prediction[matched_rows][1]
        }
      }
    }, error = function(e) {
      warning(paste("Error looking up ID2:", id2_key, "Error:", e$message))
    })
    
    # Try alternate lookup methods if first one failed
    if(is.na(tempGDP) && "country_0" %in% colnames(data_subset)) {
      # Try using country_0 with labeled_data for GDPpc
      tryCatch({
        id2_alt_key <- paste(data_subset$country_0[m], year_value, sep="_")
        matched_rows <- labeled_data$ID2 == id2_alt_key
        if(any(matched_rows) && "GDPpc" %in% colnames(labeled_data)) {
          gdp_value <- labeled_data$GDPpc[matched_rows][1]
          if(is.numeric(gdp_value) && gdp_value > 0) {
            tempGDP <- log10(gdp_value)
          }
        }
      }, error = function(e) {
        # Silent error handling
      })
      
      # Try in full_data if still NA
      if(is.na(tempGDP)) {
        tryCatch({
          id2_alt_key <- paste(data_subset$country_0[m], year_value, sep="_")
          matched_rows <- full_data$ID2 == id2_alt_key
          if(any(matched_rows) && "prediction" %in% colnames(full_data)) {
            tempGDP <- full_data$prediction[matched_rows][1]
          }
        }, error = function(e) {
          # Silent error handling
        })
      }
    }
    
    # Update GDPpc_t0 if we found a value
    if(!is.na(tempGDP)) {
      # Check if we should override with the new value
      override <- TRUE
      if("ID2" %in% colnames(labeled_data) && "GDPpc" %in% colnames(labeled_data)) {
        id2_check <- paste(data_subset$country[m], year_value, sep="_")
        matched_rows <- labeled_data$ID2 == id2_check
        if(any(matched_rows)) {
          existing_gdp <- labeled_data$GDPpc[matched_rows][1]
          override <- is.na(existing_gdp)
        }
      }
      
      if(override) {
        data_subset[[paste0(column_prefix, "GDPpc_t0")]][m] <- tempGDP
      }
    }
  }
  
  return(data_subset)
}

# Define cross-validation control
n_draws <- ifelse(exists("n_draws"), n_draws, 5)  # Use n_draws if defined, else default to 5
k <- min(5, ifelse(exists("training_data"), nrow(training_data), 5))  # Use k if defined

cctrl1 <- trainControl(
  method = "cv",
  number = n_draws,
  verboseIter = TRUE,
  savePredictions = TRUE,
  allowParallel = TRUE
)

# Define consistent training grid
training_grid <- expand.grid(
  mtry = seq(5, 30, by = 5)  # Match the grid from the main script
)

# When using ranger method, add the additional parameters
ranger_grid <- expand.grid(
  mtry = seq(5, 30, by = 5),
  splitrule = "variance",
  min.node.size = c(5)  # Default for regression
)

# Process each historical period
for(selperiod in 1:5) {
  startcolumn <- 14  # Starting column for predictors
  
  # Get data for the current period
  training_data_sub <- subset(training_data, histperiod == selperiod)
  test_data_sub <- subset(test_data, histperiod == selperiod)
  
  # Update GDPpc_t0 with historical data
  if(selperiod == 2) {
    test_data_sub <- update_GDPpc_t0(test_data_sub, test_data_preds, labeled_data, 1500)
    training_data_sub <- update_GDPpc_t0(training_data_sub, training_data_preds, labeled_data, 1500)
  } else if(selperiod == 3) {
    test_data_sub <- update_GDPpc_t0(test_data_sub, test_data_preds, labeled_data, 1750)
    training_data_sub <- update_GDPpc_t0(training_data_sub, training_data_preds, labeled_data, 1750)
  } else if(selperiod == 4) {
    test_data_sub <- update_GDPpc_t0(test_data_sub, test_data_preds, labeled_data, 1850)
    training_data_sub <- update_GDPpc_t0(training_data_sub, training_data_preds, labeled_data, 1850)
  } else if(selperiod == 5) {
    test_data_sub <- update_GDPpc_t0(test_data_sub, test_data_preds, labeled_data, 1950)
    training_data_sub <- update_GDPpc_t0(training_data_sub, training_data_preds, labeled_data, 1950)
  }
  
  # Skip processing if insufficient data
  if(nrow(training_data_sub) < 5) {
    cat("Skipping period", selperiod, "due to insufficient data\n")
    next
  }
  
  # Prepare data - handle missing values
  # Get predictor columns (from startcolumn to end)
  predictors <- colnames(training_data_sub)[startcolumn:ncol(training_data_sub)]
  
  # Handle any NA values
  for(col in predictors) {
    if(sum(is.na(training_data_sub[[col]])) > 0) {
      # Use mean imputation for numeric columns
      if(is.numeric(training_data_sub[[col]])) {
        col_mean <- mean(training_data_sub[[col]], na.rm = TRUE)
        training_data_sub[[col]][is.na(training_data_sub[[col]])] <- col_mean
        test_data_sub[[col]][is.na(test_data_sub[[col]])] <- col_mean
      } else {
        # For non-numeric columns, use the most frequent value
        mode_val <- names(sort(table(training_data_sub[[col]]), decreasing = TRUE))[1]
        training_data_sub[[col]][is.na(training_data_sub[[col]])] <- mode_val
        test_data_sub[[col]][is.na(test_data_sub[[col]])] <- mode_val
      }
    }
  }
  
  # Ensure GDPpc is safe for log transform
  training_data_sub$GDPpc_safe <- pmax(training_data_sub$GDPpc, 1e-10)
  test_data_sub$GDPpc_safe <- pmax(test_data_sub$GDPpc, 1e-10)
  
  # Train Random Forest model using ranger for better performance
  set.seed(123)  # For reproducibility
  
  # Check if we have enough predictors for requested mtry
  max_mtry <- min(length(predictors), max(ranger_grid$mtry))
  ranger_grid_adj <- ranger_grid
  ranger_grid_adj$mtry <- ranger_grid$mtry[ranger_grid$mtry <= max_mtry]
  
  if(length(ranger_grid_adj$mtry) == 0) {
    # If all mtry values too large, use half the predictors
    ranger_grid_adj$mtry <- floor(length(predictors)/2)
  }
  
  rf_model <- caret::train(
    x = training_data_sub[, predictors],
    y = log10(training_data_sub$GDPpc_safe),
    method = "ranger",
    trControl = cctrl1,
    tuneGrid = ranger_grid_adj,
    importance = 'impurity',
    num.trees = 500
  )
  
  # Make predictions for test data
  test_data_sub$prediction <- predict(rf_model, newdata = test_data_sub[, predictors])
  test_data_sub$prediction_abs <- 10^test_data_sub$prediction
  
  # Make predictions for training data
  training_data_sub$prediction <- predict(rf_model, newdata = training_data_sub[, predictors])
  training_data_sub$prediction_abs <- 10^training_data_sub$prediction
  
  # Update main datasets with prediction results
  # For test data
  if("ID" %in% colnames(test_data_sub) && "ID" %in% colnames(test_data_preds)) {
    for(i in 1:nrow(test_data_sub)) {
      row_id <- test_data_sub$ID[i]
      test_idx <- which(test_data_preds$ID == row_id)
      if(length(test_idx) > 0) {
        test_data_preds$prediction[test_idx] <- test_data_sub$prediction[i]
        test_data_preds$prediction_abs[test_idx] <- test_data_sub$prediction_abs[i]
      }
    }
  }
  
  # For training data
  if("ID" %in% colnames(training_data_sub) && "ID" %in% colnames(training_data_preds)) {
    for(i in 1:nrow(training_data_sub)) {
      row_id <- training_data_sub$ID[i]
      train_idx <- which(training_data_preds$ID == row_id)
      if(length(train_idx) > 0) {
        training_data_preds$prediction[train_idx] <- training_data_sub$prediction[i]
        training_data_preds$prediction_abs[train_idx] <- training_data_sub$prediction_abs[i]
      }
    }
  }
}

# Update test_data with predictions
test_data <- test_data_preds
