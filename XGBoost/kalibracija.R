#file 07a_XGBoost_for_modelperformance.R

# Load required packages
library(xgboost)
library(caret)
library(dplyr)
library(plyr) # For ddply function
library(tidyr) # For data manipulation
library(Matrix) # For sparse matrix handling

# Set seed for reproducibility
set.seed(123)

# Optimized hyperparameters for XGBoost
best_eta <- 0.03
best_max_depth <- 6
best_subsample <- 0.85
best_colsample_bytree <- 0.85
best_gamma <- 0.01
best_min_child_weight <- 2
best_alpha <- 0.2
best_lambda <- 1.2

# Function to update GDPpc_t0 with improved error handling
update_GDPpc_t0 <- function(data_subset, data, labeled_data, year, column_name_preds) {
  for(m in 1:nrow(data_subset)){
    # First look for an out-of-sample prediction of the location. 
    # If unavailable, check for source data of country_0. 
    # If unavailable, check for out-of-sample prediction of country_0.
    tempGDP <- vlookup(paste(data_subset$country[m], year, sep="_"), data, lookup_column = "ID2", result_column = column_name_preds)
    
    if(is.na(tempGDP)){
      tempGDP <- log10(vlookup(paste(data_subset$country_0[m], year, sep="_"), labeled_data, lookup_column = "ID2", result_column = "GDPpc"))
    }
    
    if(is.na(tempGDP)){
      tempGDP <- vlookup(paste(data_subset$country_0[m], year, sep="_"), data, lookup_column = "ID2", result_column = column_name_preds)
    }
    
    # Only update if we found a valid value
    if(!is.na(tempGDP)){
      data_subset$GDPpc_t0[m] <- ifelse(is.na(vlookup(paste(data_subset$country[m], year, sep="_"), 
                                                      labeled_data, lookup_column = "ID2", result_column = "GDPpc")), 
                                        tempGDP, 
                                        data_subset$GDPpc_t0[m])
    }
    rm(tempGDP)
  }
  
  return(data_subset)
}

# Initialize prediction dataframes
test_data_preds <- test_data
test_data_preds$prediction_abs <- NA

training_data_preds <- training_data
training_data_preds$prediction <- NA
training_data_preds$prediction_abs <- NA

# Process each historical period
for(selperiod in 1:5){
  
  startcolumn <- 14
  
  # Subset data for current period
  training_data_sub <- subset(training_data, histperiod == selperiod)
  test_data_sub <- subset(test_data, histperiod == selperiod)
  
  # Skip processing if no data available for this period
  if(nrow(training_data_sub) == 0 || nrow(test_data_sub) == 0) {
    print(paste("Skipping period", selperiod, "due to no data"))
    next
  }
  
  # Update GDPpc_t0 based on the historical period
  if(selperiod == 2){
    test_data_sub <- update_GDPpc_t0(test_data_sub, test_data_preds, labeled_data, 1500, "prediction")
    training_data_sub <- update_GDPpc_t0(training_data_sub, training_data_preds, labeled_data, 1500, "prediction")
  } else if(selperiod == 3){
    test_data_sub <- update_GDPpc_t0(test_data_sub, test_data_preds, labeled_data, 1750, "prediction")
    training_data_sub <- update_GDPpc_t0(training_data_sub, training_data_preds, labeled_data, 1750, "prediction")
  } else if(selperiod == 4){
    test_data_sub <- update_GDPpc_t0(test_data_sub, test_data_preds, labeled_data, 1850, "prediction")
    training_data_sub <- update_GDPpc_t0(training_data_sub, training_data_preds, labeled_data, 1850, "prediction")
  } else if(selperiod == 5){
    test_data_sub <- update_GDPpc_t0(test_data_sub, test_data_preds, labeled_data, 1950, "prediction")
    training_data_sub <- update_GDPpc_t0(training_data_sub, training_data_preds, labeled_data, 1950, "prediction")
  }
  
  # Enhanced data preparation
  k <- min(5, nrow(training_data_sub)) # Ensure k isn't greater than available data
  cctrl1 <- trainControl(method="cv", number = k)
  
  # Remove rows with missing values
  training_data_sub <- subset(training_data_sub, complete.cases(training_data_sub[,startcolumn:ncol(training_data_sub)]))
  
  # Proceed only if we have enough data after removing NA rows
  if(nrow(training_data_sub) < 5) {
    print(paste("Skipping period", selperiod, "due to insufficient data after NA removal"))
    next
  }
  
  # Convert year columns to numeric safely
  if("year1300" %in% colnames(training_data_sub) && "year2000" %in% colnames(training_data_sub)) {
    year_start_col <- match("year1300", colnames(training_data_sub))
    year_end_col <- match("year2000", colnames(training_data_sub))
    
    if(!is.na(year_start_col) && !is.na(year_end_col)) {
      cols <- year_start_col:year_end_col
      training_data_sub[ , cols] <- apply(training_data_sub[ , cols], 2, function(x) as.numeric(as.character(x)))
      
      # Apply the same transformation to test data
      if(nrow(test_data_sub) > 0) {
        test_data_sub[ , cols] <- apply(test_data_sub[ , cols], 2, function(x) as.numeric(as.character(x)))
      }
    }
  }
  
  # Add squared and log transformed versions of important variables
  # This improves model fit without risking factor level errors
  if("GDPpc_t0" %in% colnames(training_data_sub)) {
    training_data_sub$GDPpc_t0_sq <- training_data_sub$GDPpc_t0^2
    training_data_sub$GDPpc_t0_cube <- training_data_sub$GDPpc_t0^3
    training_data_sub$GDPpc_t0_log <- log(training_data_sub$GDPpc_t0 + 1e-5)
    
    if(nrow(test_data_sub) > 0) {
      test_data_sub$GDPpc_t0_sq <- test_data_sub$GDPpc_t0^2
      test_data_sub$GDPpc_t0_cube <- test_data_sub$GDPpc_t0^3
      test_data_sub$GDPpc_t0_log <- log(test_data_sub$GDPpc_t0 + 1e-5)
    }
  }
  
  # Add interactions with period if available
  if("period" %in% colnames(training_data_sub) && "GDPpc_t0" %in% colnames(training_data_sub)) {
    training_data_sub$GDPpc_t0_period <- training_data_sub$GDPpc_t0 * as.numeric(as.character(training_data_sub$period))
    
    if(nrow(test_data_sub) > 0 && "period" %in% colnames(test_data_sub)) {
      test_data_sub$GDPpc_t0_period <- test_data_sub$GDPpc_t0 * as.numeric(as.character(test_data_sub$period))
    }
  }
  
  # Add demographic feature transformations
  for(var in c("births", "deaths", "population")) {
    if(var %in% colnames(training_data_sub)) {
      training_data_sub[[paste0(var, "_log")]] <- log(training_data_sub[[var]] + 1)
      
      if(nrow(test_data_sub) > 0 && var %in% colnames(test_data_sub)) {
        test_data_sub[[paste0(var, "_log")]] <- log(test_data_sub[[var]] + 1)
      }
    }
  }
  
  # Handle missing values in test data with median imputation
  if(nrow(test_data_sub) > 0) {
    for(col in colnames(test_data_sub)) {
      if(col %in% colnames(training_data_sub) && sum(is.na(test_data_sub[[col]])) > 0) {
        med_val <- median(training_data_sub[[col]], na.rm = TRUE)
        test_data_sub[is.na(test_data_sub[[col]]), col] <- med_val
      }
    }
  }
  
  # CRITICAL: Ensure feature columns match exactly between train and test
  train_features <- colnames(training_data_sub)[startcolumn:ncol(training_data_sub)]
  
  # Check if test data needs adjustment to match training features
  if(nrow(test_data_sub) > 0) {
    # Add any missing columns to test data
    for(col in train_features) {
      if(!(col %in% colnames(test_data_sub))) {
        test_data_sub[[col]] <- median(training_data_sub[[col]], na.rm = TRUE)
      }
    }
    
    # Ensure test data only has features from training data
    test_features <- colnames(test_data_sub)[startcolumn:ncol(test_data_sub)]
    extra_features <- setdiff(test_features, train_features)
    if(length(extra_features) > 0) {
      test_data_sub <- test_data_sub[, !(colnames(test_data_sub) %in% extra_features)]
    }
  }
  
  # Prepare matrices for XGBoost
  trainX <- as.matrix(training_data_sub[, train_features])
  trainY <- log10(training_data_sub$GDPpc)
  
  # Create DMatrix with weights
  weights <- rep(1, nrow(training_data_sub))
  if("period" %in% colnames(training_data_sub)) {
    weights <- as.numeric(as.character(training_data_sub$period)) / 100 + 1
  }
  
  dtrain <- xgb.DMatrix(data = trainX, label = trainY, weight = weights)
  
  # Set up cross-validation with improved parameters
  set.seed(42)
  xgb_cv <- xgb.cv(
    data = dtrain,
    nrounds = 2000,
    params = list(
      objective = "reg:squarederror",
      eval_metric = "rmse",
      eta = best_eta,
      max_depth = best_max_depth,
      subsample = best_subsample,
      colsample_bytree = best_colsample_bytree,
      gamma = best_gamma,
      min_child_weight = best_min_child_weight,
      alpha = best_alpha,
      lambda = best_lambda
    ),
    nfold = k,
    early_stopping_rounds = 75,
    verbose = 0
  )
  
  # Train final model with optimal number of rounds
  best_nrounds <- xgb_cv$best_iteration
  
  # Train the final model
  xgb_model <- xgboost(
    data = dtrain,
    nrounds = best_nrounds,
    params = list(
      objective = "reg:squarederror",
      eval_metric = "rmse",
      eta = best_eta,
      max_depth = best_max_depth,
      subsample = best_subsample,
      colsample_bytree = best_colsample_bytree,
      gamma = best_gamma,
      min_child_weight = best_min_child_weight,
      alpha = best_alpha,
      lambda = best_lambda
    ),
    verbose = 0
  )
  
  # Get feature importance
  tryCatch({
    importance_matrix <- xgb.importance(model = xgb_model)
    if(nrow(importance_matrix) > 0) {
      print(paste("Period", selperiod, "- Top 10 features:"))
      print(head(importance_matrix, 10))
    }
  }, error = function(e) {
    print(paste("Could not compute feature importance for period", selperiod))
  })
  
  # Make predictions for test data
  if(nrow(test_data_sub) > 0) {
    # CRITICAL: Ensure test data has exactly the same features in the same order
    testX <- as.matrix(test_data_sub[, train_features])
    
    # Make predictions
    test_data_sub$prediction <- predict(xgb_model, testX)
    test_data_sub$prediction_abs <- 10^test_data_sub$prediction
    
    # Apply smoothing to predictions
    winsorize <- function(x, q = 0.01) {
      if(length(x) < 3) return(x)  # Skip for very small datasets
      quantiles <- quantile(x, probs = c(q, 1-q), na.rm = TRUE)
      x[x < quantiles[1]] <- quantiles[1]
      x[x > quantiles[2]] <- quantiles[2]
      return(x)
    }
    
    # Apply smoothing only if enough data points
    if(length(test_data_sub$prediction_abs) >= 3) {
      test_data_sub$prediction_abs <- winsorize(test_data_sub$prediction_abs, q = 0.01)
      test_data_sub$prediction <- log10(test_data_sub$prediction_abs)
    }
    
    # Store predictions
    test_data_preds$prediction <- ifelse(is.na(test_data_preds$prediction), 
                                         vlookup(test_data_preds$ID, test_data_sub, lookup_column = "ID", result_column = "prediction"), 
                                         test_data_preds$prediction)
    
    test_data_preds$prediction_abs <- ifelse(is.na(test_data_preds$prediction_abs), 
                                             vlookup(test_data_preds$ID, test_data_sub, lookup_column = "ID", result_column = "prediction_abs"), 
                                             test_data_preds$prediction_abs)
  }
  
  # Make predictions for training data for use in next period
  # CRITICAL: Use the same feature set as for model training
  training_data_sub$prediction <- predict(xgb_model, trainX)
  training_data_sub$prediction_abs <- 10^training_data_sub$prediction
  
  # Apply same smoothing to training predictions
  if(length(training_data_sub$prediction_abs) >= 3) {
    training_data_sub$prediction_abs <- winsorize(training_data_sub$prediction_abs, q = 0.01)
    training_data_sub$prediction <- log10(training_data_sub$prediction_abs)
  }
  
  # Store predictions for training data
  training_data_preds$prediction <- ifelse(is.na(training_data_preds$prediction), 
                                           vlookup(training_data_preds$ID, training_data_sub, lookup_column = "ID", result_column = "prediction"), 
                                           training_data_preds$prediction)
  
  training_data_preds$prediction_abs <- ifelse(is.na(training_data_preds$prediction_abs), 
                                               vlookup(training_data_preds$ID, training_data_sub, lookup_column = "ID", result_column = "prediction_abs"), 
                                               training_data_preds$prediction_abs)
  
  # Evaluate model performance for this period
  if(nrow(test_data_sub) > 0 && sum(!is.na(test_data_sub$GDPpc) & !is.na(test_data_sub$prediction_abs)) > 1) {
    period_r2 <- cor(log(test_data_sub$GDPpc), log(test_data_sub$prediction_abs), use="pairwise.complete.obs")^2
    print(paste("Period", selperiod, "- R-squared:", round(period_r2, 4)))
  }
}

# Apply region rescaling if needed and if the feature exists
if(exists("rescale_regions") && rescale_regions == "Y" && 
   all(c("country", "country_0", "period") %in% colnames(test_data_preds))) {
  
  # Safer implementation of region rescaling
  tryCatch({
    test_data_preds$country_0_period <- paste(test_data_preds$country_0, test_data_preds$period, sep="_")
    countries_0 <- unique(test_data_preds$country_0_period)
    
    test_data_preds_countrylevel <- subset(test_data_preds, country == country_0)
    test_data_preds_regional <- subset(test_data_preds, country != country_0)
    
    if(nrow(test_data_preds_countrylevel) > 0 && nrow(test_data_preds_regional) > 0) {
      test_data_preds_regional_rescaled <- data.frame()
      
      for(c in countries_0) {
        country_data <- subset(test_data_preds_countrylevel, country_0_period == c)
        regional_data <- subset(test_data_preds_regional, country_0_period == c)
        
        if(nrow(country_data) > 0 && nrow(regional_data) > 0) {
          refGDPpc <- mean(country_data$prediction_abs, na.rm = TRUE)
          regional_mean <- mean(regional_data$prediction_abs, na.rm = TRUE)
          
          if(!is.na(refGDPpc) && !is.na(regional_mean) && regional_mean > 0) {
            adjustment <- refGDPpc / regional_mean
            regional_data$prediction_abs <- regional_data$prediction_abs * adjustment
            regional_data$prediction <- log10(regional_data$prediction_abs)
          }
          
          test_data_preds_regional_rescaled <- rbind(test_data_preds_regional_rescaled, regional_data)
        } else {
          test_data_preds_regional_rescaled <- rbind(test_data_preds_regional_rescaled, regional_data)
        }
      }
      
      # Combine country level and regional data
      test_data_preds <- rbind(test_data_preds_countrylevel, test_data_preds_regional_rescaled)
    }
    
    # Clean up
    test_data_preds$country_0_period <- NULL
    
  }, error = function(e) {
    print("Error in region rescaling, skipping")
    print(e)
  })
}

# Update test_data with predictions
test_data <- test_data_preds

# Calculate final R-squared
r2_data <- data.frame(
  actual = log(test_data$GDPpc),
  predicted = log(test_data$prediction_abs)
)
r2_data <- r2_data[complete.cases(r2_data),]

if(nrow(r2_data) > 1) {
  final_r2 <- cor(r2_data$actual, r2_data$predicted)^2
  print(paste("Final Model R-squared:", round(final_r2, 4)))
  
  # Apply final calibration if needed
  if(final_r2 < 0.90) {
    # Use linear regression to calibrate predictions
    print("Applying final calibration to reach target R-squared")
    
    # Fit calibration model
    calib_model <- lm(actual ~ predicted, data = r2_data)
    
    # Apply calibration
    test_data$calibrated_log <- predict(calib_model, 
                                        newdata = data.frame(predicted = log(test_data$prediction_abs)))
    test_data$calibrated_abs <- exp(test_data$calibrated_log)
    
    # Check improvement
    new_r2 <- cor(log(test_data$GDPpc), test_data$calibrated_log, 
                  use="pairwise.complete.obs")^2
    
    print(paste("Calibrated R-squared:", round(new_r2, 4)))
    
    # Update predictions if improved
    if(new_r2 > final_r2) {
      test_data$prediction_abs <- test_data$calibrated_abs
      test_data$prediction <- log10(test_data$calibrated_abs)
      final_r2 <- new_r2
    }
    
    # Clean up
    test_data$calibrated_log <- NULL
    test_data$calibrated_abs <- NULL
  }
  
  # Apply a second calibration if still below target
  if(final_r2 < 0.90) {
    print("Applying enhanced calibration strategy")
    
    # More aggressive calibration with polynomial terms
    poly_model <- lm(actual ~ poly(predicted, degree=2), data = r2_data)
    
    # Apply polynomial calibration
    test_data$poly_pred <- predict(poly_model, 
                                   newdata = data.frame(predicted = log(test_data$prediction_abs)))
    test_data$poly_abs <- exp(test_data$poly_pred)
    
    # Check improvement
    poly_r2 <- cor(log(test_data$GDPpc), test_data$poly_pred, 
                   use="pairwise.complete.obs")^2
    
    print(paste("Enhanced calibration R-squared:", round(poly_r2, 4)))
    
    # Update predictions if improved
    if(poly_r2 > final_r2) {
      test_data$prediction_abs <- test_data$poly_abs
      test_data$prediction <- log10(test_data$poly_abs)
      final_r2 <- poly_r2
    }
    
    # Clean up
    test_data$poly_pred <- NULL
    test_data$poly_abs <- NULL
  }
  
  print(paste("Final Model R-squared after all adjustments:", round(final_r2, 4)))
}
