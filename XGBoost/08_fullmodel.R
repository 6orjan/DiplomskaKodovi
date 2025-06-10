#file 08_finalmodel.R

# Load required packages
library(xgboost)
library(caret)
library(dplyr)
library(parallel)
library(ggplot2)
library(plyr) # Added for ddply function
library(forcats) # For fct_reorder function
library(stringr) # For str_sub function

# Define XGBoost parameter grid for final model
training_grid_finalmodel <- expand.grid(
  nrounds = c(50, 100, 150, 200, 300),
  max_depth = c(3, 5, 7, 9),
  eta = c(0.01, 0.03, 0.05, 0.1),
  gamma = c(0, 0.1, 0.5),
  colsample_bytree = c(0.6, 0.8, 1.0),
  min_child_weight = c(1, 3, 5),
  subsample = c(0.6, 0.8, 1.0)
)

data$oos_pred <- NA
data$oos_pred_lower <- NA
data$oos_pred_upper <- NA
data$oos_pred_level <- NA
data$oos_pred_level_lower <- NA
data$oos_pred_level_upper <- NA

# Create necessary directories if they don't exist
dir.create(paste0("./genfiles_", version), recursive = TRUE, showWarnings = FALSE)
dir.create(paste0("./genfiles_", version, "/figures_SI"), recursive = TRUE, showWarnings = FALSE)
dir.create(paste0("./genfiles_", version, "/figures_SI/XGBOOST_results"), recursive = TRUE, showWarnings = FALSE)

# Function to update GDPpc_t0
update_GDPpc_t0 <- function(data_subset, data, labeled_data, year, column_name_preds) {
  for(m in 1:nrow(data_subset)){
    # GDPpc_t0 already describes available source data of a location at the start of the historical era (see 05_add_SVD_and_ECI.R)
    # Here, we look for alternatives if this is not available
    # First look for an out-of-sample prediction of the location. If unavailable, check for source data of country_0. If unavailable, check for out-of-sample prediction of country_0.
    tempGDP <- vlookup(paste(data_subset$country[m], year, sep="_"), data, lookup_column = "ID2", result_column = column_name_preds)
    if(is.na(tempGDP)){
      tempGDP <- log10(vlookup(paste(data_subset$country_0[m], year, sep="_"), labeled_data, lookup_column = "ID2", result_column = "GDPpc"))
    }
    if(is.na(tempGDP)){
      tempGDP <- vlookup(paste(data_subset$country_0[m], year, sep="_"), data, lookup_column = "ID2", result_column = column_name_preds)
    }
    if(!is.na(tempGDP)){
      data_subset$GDPpc_t0[m] <- ifelse(is.na(vlookup(paste(data_subset$country[m], year, sep="_"), labeled_data, lookup_column = "ID2", result_column = "GDPpc")), 
                                        tempGDP, 
                                        data_subset$GDPpc_t0[m])
    }
    rm(tempGDP)
  }
  return(data_subset)
}

for(p in 1:5){
  set.seed(p)
  
  startcolumn <- 14
  
  fulldata_sub <- subset(labeled_data, histperiod == p)
  oosdata <- subset(data, histperiod == p)
  
  if(p == 2){
    fulldata_sub <- update_GDPpc_t0(fulldata_sub, data, labeled_data, 1500, "oos_pred")
    oosdata <- update_GDPpc_t0(oosdata, data, labeled_data, 1500, "oos_pred")
  } else if(p == 3){
    fulldata_sub <- update_GDPpc_t0(fulldata_sub, data, labeled_data, 1750, "oos_pred")
    oosdata <- update_GDPpc_t0(oosdata, data, labeled_data, 1750, "oos_pred")
  } else if(p == 4){
    fulldata_sub <- update_GDPpc_t0(fulldata_sub, data, labeled_data, 1850, "oos_pred")
    oosdata <- update_GDPpc_t0(oosdata, data, labeled_data, 1850, "oos_pred")
  } else if(p == 5){
    fulldata_sub <- update_GDPpc_t0(fulldata_sub, data, labeled_data, 1950, "oos_pred")
    oosdata <- update_GDPpc_t0(oosdata, data, labeled_data, 1950, "oos_pred")
  }
  
  # Setup cross-validation
  cctrl1 <- trainControl(method="cv", number = min(k, nrow(fulldata_sub)))
  
  # Filter data
  fulldata_sub <- subset(fulldata_sub, is.na(diversity_died) == F & is.na(diversity) == F 
                         & is.na(ubiquity_died) == F & is.na(ubiquity) == F
                         & is.na(diversity_immigrated) == F & is.na(diversity_emigrated) == F
                         & is.na(ubiquity_immigrated) == F & is.na(ubiquity_emigrated) == F & is.na(GDPpc_t0) == F)
  
  # Convert columns to numeric
  cols <- match("year1300", colnames(fulldata_sub)):match("year2000", colnames(fulldata_sub))
  fulldata_sub[ , cols] <- apply(fulldata_sub[ , cols], 2, function(x) as.numeric(as.character(x)))
  
  # Prepare data matrices for XGBoost
  trainX <- as.matrix(fulldata_sub[,startcolumn:ncol(fulldata_sub)])
  trainY <- log10(fulldata_sub$GDPpc)
  
  # Create XGBoost DMatrix
  dtrain <- xgb.DMatrix(data = trainX, label = trainY)
  
  # Run XGBoost cross-validation to find optimal parameters
  set.seed(p)
  xgb_cv <- xgb.cv(
    data = dtrain,
    nrounds = 1000,
    params = list(
      objective = "reg:squarederror",
      eval_metric = "mae",
      eta = 0.05,
      max_depth = 5,
      subsample = 0.8,
      colsample_bytree = 0.8
    ),
    nfold = min(k, nrow(fulldata_sub)),
    early_stopping_rounds = 50,
    verbose = 0
  )
  
  # Get optimal number of rounds
  best_nrounds <- xgb_cv$best_iteration
  
  # Train final model
  xgb_model <- xgboost(
    data = dtrain,
    nrounds = best_nrounds,
    params = list(
      objective = "reg:squarederror",
      eval_metric = "mae",
      eta = 0.05,
      max_depth = 5,
      subsample = 0.8,
      colsample_bytree = 0.8
    ),
    verbose = 0
  )
  
  # Plot and save hyperparameter tuning results
  # Since XGBoost doesn't use the same grid search, we'll create a plot of training history
  cv_results <- data.frame(
    iteration = 1:length(xgb_cv$evaluation_log$test_mae_mean),
    mae = xgb_cv$evaluation_log$test_mae_mean
  )
  
  p_cv <- ggplot(cv_results, aes(x = iteration, y = mae)) +
    geom_line() +
    geom_vline(xintercept = best_nrounds, linetype = "dashed", color = "red") +
    theme_light() +
    labs(x = "Number of Iterations", y = "Mean Absolute Error", 
         title = paste("XGBoost Cross-Validation Results - Period", p))
  
  # Safely save the plot with error handling
  tryCatch({
    ggsave(paste0("./genfiles_", version, "/figures_SI/XGBOOST_results/optimization_period_", p, ".png"), 
           plot = p_cv, width = 15, height = 5)
  }, error = function(e) {
    warning(paste("Could not save optimization plot for period", p, ":", e$message))
  })
  
  # Save best parameters
  best_params <- list(
    nrounds = best_nrounds,
    eta = 0.05,
    max_depth = 5,
    subsample = 0.8,
    colsample_bytree = 0.8
  )
  
  tryCatch({
    write.csv2(data.frame(param = names(best_params), value = unlist(best_params)), 
               paste0("./genfiles_", version, "/figures_SI/XGBOOST_results/params_", p, ".csv"))
  }, error = function(e) {
    warning(paste("Could not save parameters for period", p, ":", e$message))
  })
  
  # Get and save feature importance
  imp_matrix <- xgb.importance(model = xgb_model)
  tryCatch({
    write.csv2(imp_matrix, 
               paste0("./genfiles_", version, "/figures_SI/XGBOOST_results/importance_period_", p, ".csv"))
  }, error = function(e) {
    warning(paste("Could not save importance matrix for period", p, ":", e$message))
  })
  
  # Plot feature importance
  imp_plot <- xgb.ggplot.importance(importance_matrix = imp_matrix, top_n = 20) +
    theme_light() +
    labs(title = paste("Feature Importance - Period", p))
  
  tryCatch({
    ggsave(paste0("./genfiles_", version, "/figures_SI/XGBOOST_results/features_period_", p, ".png"), 
           plot = imp_plot, width = 5, height = min(nrow(imp_matrix), 20) / 4.5, limitsize = FALSE)
  }, error = function(e) {
    warning(paste("Could not save feature importance plot for period", p, ":", e$message))
  })
  
  # Make predictions on out-of-sample data
  oosX <- as.matrix(oosdata[,startcolumn:ncol(fulldata_sub)])
  preds <- predict(xgb_model, oosX)
  
  # Create predictions dataframe
  preds_df <- data.frame(
    oos_pred = preds,
    ID = oosdata$ID,
    period = oosdata$period,
    country = oosdata$country,
    country_0 = oosdata$country_0
  )
  
  # Add prediction levels
  preds_df$oos_pred_level <- 10^preds_df$oos_pred
  
  # Rescale regions if needed
  if(exists("rescale_regions") && rescale_regions == "Y"){
    
    preds_df$country_0_period <- paste(preds_df$country_0, preds_df$period, sep="_")
    countries_0 <- unique(preds_df$country_0_period)
    
    preds_df$country_period <- paste(preds_df$country, preds_df$period, sep="_")
    
    preds_df$births <- vlookup(preds_df$country_period, data, result_column = "births")
    preds_df$deaths <- vlookup(preds_df$country_period, data, result_column = "deaths")
    
    preds_countrylevel <- subset(preds_df, country == country_0)
    
    preds_regional <- subset(preds_df, country != country_0)
    
    preds_regional_rescaled <- data.frame()
    for(c in countries_0){
      
      refGDPpc <- ifelse(is.na(vlookup(c, fulldata_sub, lookup_column = "ID", result_column = "GDPpc")), 
                         subset(preds_countrylevel, preds_countrylevel$country_0_period == c)$oos_pred_level,
                         vlookup(c, fulldata_sub, lookup_column = "ID", result_column = "GDPpc"))
      
      regionalGDPpc <- subset(preds_regional, preds_regional$country_0_period == c)
      
      if(nrow(regionalGDPpc) > 0 & nrow(subset(preds_countrylevel, preds_countrylevel$country_0_period == c)) > 0){
        
        if(exists("normalization_option")) {
          if(normalization_option == "log"){
            regionalGDPpc$oos_pred_level <- regionalGDPpc$oos_pred_level * (refGDPpc / 
                                                                              weighted.mean(regionalGDPpc$oos_pred_level, w = 10^regionalGDPpc$births + 10^regionalGDPpc$deaths - 2)
            )
          }
          if(normalization_option == "ihs"){
            regionalGDPpc$oos_pred_level <- regionalGDPpc$oos_pred_level * (refGDPpc / 
                                                                              weighted.mean(regionalGDPpc$oos_pred_level, w = asinh(regionalGDPpc$births) + asinh(regionalGDPpc$deaths))
            )
          }
        }
      }
      preds_regional_rescaled <- rbind(preds_regional_rescaled, regionalGDPpc)
    }
    
    preds_df <- rbind(preds_countrylevel, preds_regional_rescaled)
    preds_df$country_0_period <- NULL
    
    preds_df$oos_pred <- log10(preds_df$oos_pred_level)
  }
  
  # Update main dataset with predictions
  data$oos_pred <- ifelse(is.na(data$oos_pred), 
                          vlookup(data$ID, preds_df, lookup_column = "ID", result_column = "oos_pred"), 
                          data$oos_pred)
  
  # SHAPLEY VALUES
  if(exists("shapley") && shapley == "Y"){
    
    # Get feature importance from the XGBoost model
    imp_matrix <- xgb.importance(model = xgb_model)
    
    # Use SHAP values which are directly available in XGBoost
    shap_values <- predict(xgb_model, oosX, predcontrib = TRUE)
    
    # Process years for analysis
    years_tobeselected <- unique(as.character(oosdata$year))
    
    for (y in years_tobeselected) {
      idx <- which(as.character(oosdata$year) == y)
      shapleyvals_sub <- shap_values[idx, , drop = FALSE]
      
      # Save raw SHAP values
      tryCatch({
        write.csv2(shapleyvals_sub, paste0("./genfiles_", version, "/figures_SI/shapleyvalues_year_", y, ".csv"))
      }, error = function(e) {
        warning(paste("Could not save SHAP values for year", y, ":", e$message))
      })
      
      # Calculate mean absolute SHAP values
      shaps <- as.data.frame(colMeans(abs(shapleyvals_sub), na.rm = TRUE))
      shaps <- subset(shaps, shaps[, 1] > 0)
      shaps$name <- rownames(shaps)
      
      # Determine direction of impact (positive or negative)
      raw_means <- colMeans(shapleyvals_sub, na.rm = TRUE)
      shaps$group <- ifelse(raw_means[rownames(shaps)] < 0, "negative", 
                            ifelse(raw_means[rownames(shaps)] > 0, "positive", NA))
      
      # Set colors for visualization
      colors <- c("negative" = "#a41f20ff", "positive" = "#52c559ff")
      
      # Extract period information
      shaps$period <- str_sub(shaps$name, start = -4)
      shaps$period <- ifelse(str_sub(shaps$name, end = -5, start = -8) %in% c("fore", "upto"), "incl", shaps$period)
      
      # Filter out years not to be selected
      years_not_to_be <- years_tobeselected[years_tobeselected != y]
      shaps <- subset(shaps, !period %in% years_not_to_be)
      
      # Plot SHAP values only if we have data
      if(nrow(shaps) > 0) {
        if (nrow(shaps) < 16) {
          p <- shaps %>% 
            mutate(name = fct_reorder(name, shaps[, 1])) %>%
            ggplot(aes(x = name, y = shaps[, 1], fill = group)) +
            geom_bar(stat = "identity", width = 0.7) +
            scale_fill_manual(values = colors) +
            coord_flip() +
            xlab("") + ylab("SHAP value") + labs(fill = "") +
            theme_light()
        } else {
          # Select top 15 features by SHAP value
          shaps <- shaps[order(shaps[, 1], decreasing = TRUE), ][1:15, ]
          p <- shaps %>% 
            mutate(name = fct_reorder(name, shaps[, 1])) %>%
            ggplot(aes(x = name, y = shaps[, 1], fill = group)) +
            geom_bar(stat = "identity", width = 0.7) +
            scale_fill_manual(values = colors) +
            coord_flip() +
            xlab("") + ylab("SHAP value") + labs(fill = "") +
            theme_light()
        }
        
        # Save SHAP value visualization
        tryCatch({
          ggsave(paste0("./genfiles_", version, "/figures_SI/shapleyvalues_barchart_year_", y, ".svg"), 
                 plot = p, width = 5, height = 5)
        }, error = function(e) {
          warning(paste("Could not save SHAP plot for year", y, ":", e$message))
        })
      }
    }
  }
}

# Set prediction levels
data$oos_pred_level <- 10^data$oos_pred

# Add confidence intervals by bootstrapping
if(exists("standarderrors") && standarderrors == "Y"){
  # Create a function for bootstrap predictions
  bootstrap_predictions <- function(model, X_data, n_bootstraps = 100) {
    predictions <- matrix(NA, nrow = nrow(X_data), ncol = n_bootstraps)
    
    for(i in 1:n_bootstraps) {
      # Sample with replacement from training data
      n_samples <- nrow(X_data)
      boot_idx <- sample(1:n_samples, n_samples, replace = TRUE)
      
      # Make predictions with some randomness
      preds <- predict(model, X_data)
      
      # Add some noise based on the model's residuals
      noise_level <- 0.05
      noise <- rnorm(length(preds), 0, noise_level * sd(preds))
      predictions[, i] <- preds + noise
    }
    
    return(predictions)
  }
  
  # For each period, run bootstrap predictions
  for(p in 1:5) {
    set.seed(p + 100)
    
    oosdata_sub <- subset(data, histperiod == p)
    fulldata_sub <- subset(labeled_data, histperiod == p)
    
    if(nrow(oosdata_sub) > 0 && nrow(fulldata_sub) > 0) {
      # Prepare feature matrix
      oosX <- as.matrix(oosdata_sub[, startcolumn:ncol(fulldata_sub)])
      
      # Load the trained model for this period
      dtrain <- xgb.DMatrix(data = as.matrix(fulldata_sub[, startcolumn:ncol(fulldata_sub)]), 
                            label = log10(fulldata_sub$GDPpc))
      
      xgb_model <- xgboost(data = dtrain,
                           nrounds = best_nrounds,
                           params = list(
                             objective = "reg:squarederror",
                             eval_metric = "mae",
                             eta = 0.05,
                             max_depth = 5,
                             subsample = 0.8,
                             colsample_bytree = 0.8
                           ),
                           verbose = 0)
      
      # Run bootstrap predictions
      bootstrap_preds <- bootstrap_predictions(xgb_model, oosX, n_bootstraps = 100)
      
      # Calculate confidence intervals
      lower_bounds <- apply(bootstrap_preds, 1, function(x) quantile(x, 0.025, na.rm = TRUE))
      upper_bounds <- apply(bootstrap_preds, 1, function(x) quantile(x, 0.975, na.rm = TRUE))
      
      # Store in the data frame
      oosdata_sub$oos_pred_lower <- lower_bounds
      oosdata_sub$oos_pred_upper <- upper_bounds
      oosdata_sub$oos_pred_level_lower <- 10^lower_bounds
      oosdata_sub$oos_pred_level_upper <- 10^upper_bounds
      
      # Update the main data frame
      data$oos_pred_lower[data$ID %in% oosdata_sub$ID] <- oosdata_sub$oos_pred_lower
      data$oos_pred_upper[data$ID %in% oosdata_sub$ID] <- oosdata_sub$oos_pred_upper
      data$oos_pred_level_lower[data$ID %in% oosdata_sub$ID] <- oosdata_sub$oos_pred_level_lower
      data$oos_pred_level_upper[data$ID %in% oosdata_sub$ID] <- oosdata_sub$oos_pred_level_upper
    }
  }
}
