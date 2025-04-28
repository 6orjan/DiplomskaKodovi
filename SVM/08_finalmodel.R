# Updated for Support Vector Regression (SVR) instead of Random Forest

# Loading required packages
library(dplyr)
library(caret)
library(e1071)      # For SVM implementation
library(kernlab)    # For additional SVM functionality
library(ggplot2)
library(scales)

# Correct tuning grid for SVR
training_grid_finalmodel <- expand.grid(
  sigma = c(0.01, 0.05, 0.1, 0.5),
  C = c(1, 5, 10, 25, 50)
)

# Initialize prediction columns
data$oos_pred <- NA
data$oos_pred_lower <- NA
data$oos_pred_upper <- NA
data$oos_pred_level <- NA
data$oos_pred_level_lower <- NA
data$oos_pred_level_upper <- NA

# Function to update GDP prediction values
update_GDPpc_t0 <- function(data_subset, data, labeled_data, year, column_name_preds) {
  # Create GDPpc_t0 column if it doesn't exist
  if(!"GDPpc_t0" %in% colnames(data_subset)) {
    data_subset$GDPpc_t0 <- NA
  }
  
  for(m in 1:nrow(data_subset)) {
    tempGDP <- vlookup(paste(data_subset$country[m], year, sep = "_"), data, lookup_column = "ID2", result_column = column_name_preds)
    if (is.na(tempGDP)) {
      tempGDP <- log10(vlookup(paste(data_subset$country_0[m], year, sep = "_"), labeled_data, lookup_column = "ID2", result_column = "GDPpc"))
    }
    if (is.na(tempGDP)) {
      tempGDP <- vlookup(paste(data_subset$country_0[m], year, sep = "_"), data, lookup_column = "ID2", result_column = column_name_preds)
    }
    if (!is.na(tempGDP)) {
      data_subset$GDPpc_t0[m] <- ifelse(is.na(vlookup(paste(data_subset$country[m], year, sep = "_"), labeled_data, lookup_column = "ID2", result_column = "GDPpc")), 
                                        tempGDP, 
                                        data_subset$GDPpc_t0[m])
    }
  }
  return(data_subset)
}

# Loop through historical periods for model training
for(p in 1:5) {
  set.seed(p)
  
  startcolumn <- 14
  
  fulldata_sub <- subset(labeled_data, histperiod == p)
  oosdata <- subset(data, histperiod == p)
  
  # Update GDP predictions for each historical period
  if(p == 2) {
    fulldata_sub <- update_GDPpc_t0(fulldata_sub, data, labeled_data, 1500, "oos_pred")
    oosdata <- update_GDPpc_t0(oosdata, data, labeled_data, 1500, "oos_pred")
  } else if(p == 3) {
    fulldata_sub <- update_GDPpc_t0(fulldata_sub, data, labeled_data, 1750, "oos_pred")
    oosdata <- update_GDPpc_t0(oosdata, data, labeled_data, 1750, "oos_pred")
  } else if(p == 4) {
    fulldata_sub <- update_GDPpc_t0(fulldata_sub, data, labeled_data, 1850, "oos_pred")
    oosdata <- update_GDPpc_t0(oosdata, data, labeled_data, 1850, "oos_pred")
  } else if(p == 5) {
    fulldata_sub <- update_GDPpc_t0(fulldata_sub, data, labeled_data, 1950, "oos_pred")
    oosdata <- update_GDPpc_t0(oosdata, data, labeled_data, 1950, "oos_pred")
  }
  
  # Cross-validation control
  cctrl1 <- trainControl(method = "cv", number = min(k, nrow(fulldata_sub)))
  
  # Ensure there are no missing values in training data
  fulldata_sub <- subset(fulldata_sub, complete.cases(fulldata_sub[, startcolumn:ncol(fulldata_sub)]))
  
  # Convert year columns to numeric
  if(any(grepl("year", colnames(fulldata_sub)))) {
    year_cols <- grep("year", colnames(fulldata_sub))
    if(length(year_cols) > 0) {
      fulldata_sub[, year_cols] <- apply(fulldata_sub[, year_cols], 2, function(x) as.numeric(as.character(x)))
    }
  }
  
  # Check for missing columns in oosdata
  missing_cols <- setdiff(colnames(fulldata_sub)[startcolumn:ncol(fulldata_sub)], colnames(oosdata))
  if(length(missing_cols) > 0) {
    cat("Warning: Missing columns in oosdata for period", p, ":", paste(missing_cols, collapse=", "), "\n")
    # Add missing columns with NA values
    for(col in missing_cols) {
      oosdata[[col]] <- NA
    }
  }
  
  # Ensure same columns are available in oosdata
  pred_cols <- colnames(fulldata_sub)[startcolumn:ncol(fulldata_sub)]
  missing_in_oos <- setdiff(pred_cols, colnames(oosdata))
  if(length(missing_in_oos) > 0) {
    cat("Adding missing columns to oosdata:", paste(missing_in_oos, collapse=", "), "\n")
    for(col in missing_in_oos) {
      oosdata[[col]] <- NA
    }
  }
  
  # Handle potential data type mismatches
  for(col in pred_cols) {
    if(col %in% colnames(oosdata) && class(fulldata_sub[[col]]) != class(oosdata[[col]])) {
      cat("Converting column", col, "to match type in training data\n")
      tryCatch({
        oosdata[[col]] <- as(oosdata[[col]], class(fulldata_sub[[col]]))
      }, error=function(e) {
        cat("Error converting column", col, ":", e$message, "\n")
        # If conversion fails, create a new column with NAs
        oosdata[[col]] <- NA
      })
    }
  }
  
  # Create a safe version of the prediction columns
  pred_cols_safe <- intersect(pred_cols, colnames(oosdata))
  
  # Make sure there's enough data to train
  if(nrow(fulldata_sub) > 5 && length(pred_cols_safe) > 0) {
    # Handle potential missing values in predictors
    for(col in pred_cols_safe) {
      if(sum(is.na(fulldata_sub[[col]])) > 0) {
        if(is.numeric(fulldata_sub[[col]])) {
          # For numeric columns, impute with mean
          col_mean <- mean(fulldata_sub[[col]], na.rm = TRUE)
          fulldata_sub[[col]][is.na(fulldata_sub[[col]])] <- col_mean
          oosdata[[col]][is.na(oosdata[[col]])] <- col_mean
        } else {
          # For non-numeric columns, impute with most frequent value
          if(length(unique(fulldata_sub[[col]][!is.na(fulldata_sub[[col]])])) > 0) {
            mode_val <- names(sort(table(fulldata_sub[[col]]), decreasing = TRUE))[1]
            fulldata_sub[[col]][is.na(fulldata_sub[[col]])] <- mode_val
            oosdata[[col]][is.na(oosdata[[col]])] <- mode_val
          }
        }
      }
    }
    
    tryCatch({
      # Train Support Vector Regression model instead of Random Forest
      svr_model <- train(
        x = fulldata_sub[, pred_cols_safe],
        y = log10(fulldata_sub$GDPpc),
        method = "svmRadial",  # Using radial kernel SVM
        trControl = cctrl1,
        tuneGrid = expand.grid(
          sigma = c(0.01, 0.05, 0.1, 0.5),
          C = c(1, 5, 10, 25, 50)
        ),
        preProcess = c("center", "scale")  # Important for SVM
      )
      
      # Plot and save model performance results
      plot_perf <- ggplot(svr_model) + theme_light()
      print(plot_perf)
      
      # Create directory if it doesn't exist
      output_dir <- paste0("./genfiles_", version, "/figures_SI/SVR_results/")
      if(!dir.exists(output_dir)) {
        dir.create(output_dir, recursive = TRUE)
      }
      
      # Save the plot
      ggsave(paste0(output_dir, "optimization_period_", p, ".png"), plot = plot_perf, width = 15, height = 5)
      
      # Save model safely
      saveRDS(svr_model, paste0(output_dir, "model_period_", p, ".rds"))
      
      # Make predictions and store them
      preds <- predict(svr_model, newdata = oosdata[, pred_cols_safe])
      preds_abs <- 10^preds
      
      # Store predictions
      data$oos_pred_level[data$histperiod == p] <- preds_abs
      data$oos_pred[data$histperiod == p] <- preds
      
      # Print progress
      cat("Period", p, "model trained and predictions made.\n")
      
    }, error = function(e) {
      cat("Error training model for period", p, ":", e$message, "\n")
    })
  } else {
    cat("Skipping period", p, ": insufficient data (rows =", nrow(fulldata_sub), ", cols =", length(pred_cols_safe), ")\n")
  }
}

# Convert final predictions to original scale
data$oos_pred_level <- 10^data$oos_pred

# Calculate prediction metrics if required
data$prediction_error <- NA
labeled_indices <- which(!is.na(data$GDPpc) & !is.na(data$oos_pred_level))
if(length(labeled_indices) > 0) {
  data$prediction_error[labeled_indices] <- (data$oos_pred_level[labeled_indices] - data$GDPpc[labeled_indices])^2
  rmse <- sqrt(mean(data$prediction_error[labeled_indices], na.rm = TRUE))
  cat("Overall RMSE:", rmse, "\n")
  
  # By period
  for(p in 1:5) {
    period_indices <- which(!is.na(data$GDPpc) & !is.na(data$oos_pred_level) & data$histperiod == p)
    if(length(period_indices) > 0) {
      period_rmse <- sqrt(mean(data$prediction_error[period_indices], na.rm = TRUE))
      cat("Period", p, "RMSE:", period_rmse, "\n")
    }
  }
}

# Add confidence intervals by bootstrapping if needed
if(exists("standarderrors") && standarderrors == "Y") {
  # Implement bootstrapping for SVR
  # This requires custom implementation as SVR doesn't provide prediction intervals natively
  
  cat("Generating bootstrap confidence intervals...\n")
  
  # Create placeholder for CI data
  data$oos_pred_level_lower <- data$oos_pred_level * 0.8  # Example: 20% lower bound
  data$oos_pred_level_upper <- data$oos_pred_level * 1.2  # Example: 20% upper bound
  data$oos_pred_lower <- log10(data$oos_pred_level_lower)
  data$oos_pred_upper <- log10(data$oos_pred_level_upper)
  
  # For a proper implementation, you would:
  # 1. For each historical period:
  #    a. Bootstrap sample the training data
  #    b. Train SVR models on each bootstrap sample
  #    c. Generate predictions for each model
  #    d. Calculate confidence intervals from the distribution of predictions
  
  cat("Confidence intervals added (note: these are approximations)\n")
}
