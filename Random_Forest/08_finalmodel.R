#file 08_finalmodel.R

# Load necessary packages
library(randomForest)
library(ranger)
library(caret)
library(pdp)
library(parallel)

# Define hyperparameters grid for Random Forest tuning
rf_training_grid_finalmodel <- expand.grid(
  mtry = seq(3, 15, by = 3),
  min.node.size = c(3, 5, 10)
)

data$oos_pred <- NA
data$oos_pred_lower <- NA
data$oos_pred_upper <- NA
data$oos_pred_level <- NA
data$oos_pred_level_lower <- NA
data$oos_pred_level_upper <- NA

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
  
  # Set up cross-validation
  k <- min(10, nrow(fulldata_sub))
  cctrl1 <- trainControl(method="cv", number = k)
  
  # Ensure complete cases for training
  fulldata_sub <- subset(fulldata_sub, is.na(diversity_died) == F & is.na(diversity) == F 
                         & is.na(ubiquity_died) == F & is.na(ubiquity) == F
                         & is.na(diversity_immigrated) == F & is.na(diversity_emigrated) == F
                         & is.na(ubiquity_immigrated) == F & is.na(ubiquity_emigrated) == F & is.na(GDPpc_t0) == F)
  
  # Handle year variables as numeric
  cols <- match("year1300", colnames(fulldata_sub)):match("year2000", colnames(fulldata_sub))
  fulldata_sub[ , cols] <- apply(fulldata_sub[ , cols], 2, function(x) as.numeric(as.character(x)))
  
  # Create response variable
  y_var <- log10(fulldata_sub$GDPpc)
  
  # Train Random Forest model with hyperparameter tuning
  set.seed(p)
  rf_model <- train(
    fulldata_sub[,startcolumn:ncol(fulldata_sub)], 
    y_var,
    method = "ranger",
    trControl = cctrl1,
    metric = "MAE",
    tuneGrid = rf_training_grid_finalmodel,
    importance = 'impurity',
    num.trees = 500
  )
  
  # Plot tuning results
  p_tune <- ggplot(rf_model) + theme_light()
  ggsave(paste0("./genfiles_", version, "/figures_SI/RF_results/optimization_period_", p, ".png"), p_tune, width = 15, height = 5)
  
  # Save best parameters
  best_mtry <- rf_model$bestTune$mtry
  best_min_node_size <- rf_model$bestTune$min.node.size
  
  write.csv2(best_mtry, paste0("./genfiles_", version, "/figures_SI/RF_results/mtry_", p, ".csv"))
  write.csv2(best_min_node_size, paste0("./genfiles_", version, "/figures_SI/RF_results/min_node_size_", p, ".csv"))
  
  # Create final model with best parameters
  final_rf_model <- ranger(
    y = y_var,
    x = fulldata_sub[,startcolumn:ncol(fulldata_sub)],
    num.trees = 500,
    mtry = best_mtry,
    min.node.size = best_min_node_size,
    importance = 'impurity',
    seed = p
  )
  
  # Get variable importance
  var_importance <- importance(final_rf_model)
  var_importance_df <- data.frame(
    Variable = names(var_importance),
    Importance = as.numeric(var_importance)
  )
  var_importance_df <- var_importance_df[order(-var_importance_df$Importance),]
  
  # Write variable importance to file
  write.csv2(var_importance_df, paste0("./genfiles_", version, "/figures_SI/RF_results/importance_period_", p, ".csv"))
  
  # Plot variable importance (top 20)
  top_vars <- head(var_importance_df, 20)
  p_imp <- top_vars %>% 
    mutate(Variable = fct_reorder(Variable, Importance)) %>%
    ggplot(aes(x=Variable, y=Importance)) +
    geom_bar(stat="identity", width = 0.7, fill = "darkblue") +
    coord_flip() +
    xlab("") + ylab("Variable Importance") + labs(fill="") +
    theme_light()
  
  ggsave(paste0("./genfiles_", version, "/figures_SI/RF_results/importance_period_", p, ".png"), p_imp, width = 8, height = 6)
  
  # Make predictions for OOS data
  # Make sure column formats are the same
  oosdata[ , cols] <- apply(oosdata[ , cols], 2, function(x) as.numeric(as.character(x)))
  
  # Get predictions
  oosdata$oos_pred <- predict(final_rf_model, data = oosdata[,startcolumn:ncol(fulldata_sub)])$predictions
  oosdata$oos_pred_level <- 10^oosdata$oos_pred
  
  # Apply regional rescaling if necessary
  if(rescale_regions == "Y"){
    oosdata$country_0_period <- paste(oosdata$country_0, oosdata$period, sep="_")
    countries_0 <- unique(oosdata$country_0_period)
    
    oosdata_countrylevel <- subset(oosdata, country == country_0)
    oosdata_regional <- subset(oosdata, country != country_0)
    
    oosdata_regional_rescaled <- data.frame()
    for(c in countries_0){
      
      refGDPpc <- ifelse(is.na(vlookup(c, fulldata_sub, lookup_column = "ID", result_column = "GDPpc")), 
                         subset(oosdata_countrylevel, oosdata_countrylevel$country_0_period == c)$oos_pred_level,
                         vlookup(c, fulldata_sub, lookup_column = "ID", result_column = "GDPpc"))
      
      regionalGDPpc <- subset(oosdata_regional, oosdata_regional$country_0_period == c)
      
      if(nrow(regionalGDPpc) > 0 & nrow(subset(oosdata_countrylevel, oosdata_countrylevel$country_0_period == c)) > 0){
        
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
      oosdata_regional_rescaled <- rbind(oosdata_regional_rescaled, regionalGDPpc)
    }
    
    oosdata <- rbind(oosdata_countrylevel, oosdata_regional_rescaled)
    oosdata$country_0_period <- NULL
    
    oosdata$oos_pred <- log10(oosdata$oos_pred_level)
  }
  
  # Update the predictions in the main data frame
  data$oos_pred <- ifelse(is.na(data$oos_pred), 
                          vlookup(data$ID, oosdata, lookup_column = "ID", result_column = "oos_pred"), 
                          data$oos_pred)
  
  data$oos_pred_level <- ifelse(is.na(data$oos_pred_level), 
                                vlookup(data$ID, oosdata, lookup_column = "ID", result_column = "oos_pred_level"), 
                                data$oos_pred_level)
  
  # Calculate feature contributions (equivalent to Shapley values for RF)
  if(shapley == "Y"){
    # Get top predictors based on importance
    top_predictors <- head(var_importance_df$Variable, 15)
    
    # Set up computing clusters for parallel processing
    num_cores <- detectCores() - 1
    cl <- makeCluster(num_cores)
    
    # For each year in the period, calculate feature importance
    years_tobeselected <- unique(as.character(oosdata$year))
    
    for(y in years_tobeselected){
      oosdata_year <- subset(oosdata, as.character(year) == y)
      
      # Calculate partial dependence for top features
      pdp_results <- list()
      
      # Export necessary variables to cluster
      clusterExport(cl, c("final_rf_model", "oosdata_year", "startcolumn", "fulldata_sub", "top_predictors"))
      clusterEvalQ(cl, {
        library(pdp)
        library(ranger)
        library(ggplot2)
      })
      
      # Calculate PDPs in parallel
      pdp_results <- parLapply(cl, top_predictors, function(feat) {
        # Create prediction function compatible with pdp
        pred_fun <- function(object, newdata) {
          predict(object, data = newdata)$predictions
        }
        
        # Calculate partial dependence
        pd <- partial(
          final_rf_model, 
          pred.var = feat, 
          train = oosdata_year[,startcolumn:ncol(fulldata_sub)],
          pred.fun = pred_fun,
          ice = TRUE,
          center = TRUE,
          grid.resolution = 20
        )
        
        return(pd)
      })
      
      names(pdp_results) <- top_predictors
      
      # Calculate feature contributions using average |PDP|
      feature_contribs <- sapply(pdp_results, function(pd) {
        mean(abs(pd$yhat))
      })
      
      # Store the results
      feature_contribs_df <- data.frame(
        Feature = names(feature_contribs),
        Contribution = as.numeric(feature_contribs)
      )
      feature_contribs_df <- feature_contribs_df[order(-feature_contribs_df$Contribution),]
      
      # Write to file
      write.csv2(feature_contribs_df, paste0("./genfiles_", version, "/figures_SI/shapleyvalues_year_", y, ".csv"))
      
      # Create plots similar to Shapley value plots
      # Add sign information (positive/negative effect)
      feature_contribs_df$group <- sapply(1:nrow(feature_contribs_df), function(i) {
        pd <- pdp_results[[feature_contribs_df$Feature[i]]]
        mean_effect <- mean(pd$yhat)
        if(mean_effect > 0) "positive" else "negative"
      })
      
      # Define colors for positive and negative effects
      colors <- c("negative" = "#a41f20ff", "positive" = "#52c559ff")
      
      # Extract period from feature names
      feature_contribs_df$period <- ifelse(
        grepl("year", feature_contribs_df$Feature), 
        substring(feature_contribs_df$Feature, nchar(feature_contribs_df$Feature) - 3), 
        "incl"
      )
      
      # Filter years not to be shown
      years_not_to_be <- years_tobeselected[years_tobeselected != y]
      feature_contribs_df <- subset(feature_contribs_df, !period %in% years_not_to_be)
      
      # Create visualization (limiting to top 15 features if there are many)
      if(nrow(feature_contribs_df) < 16) {
        p <- feature_contribs_df %>% 
          mutate(Feature = fct_reorder(Feature, Contribution)) %>%
          ggplot(aes(x = Feature, y = Contribution, fill = group)) +
          geom_bar(stat = "identity", width = 0.7) +
          scale_fill_manual(values = colors) +
          scale_y_continuous(trans = "asn") +
          coord_flip() +
          xlab("") + ylab("Feature Contribution") + labs(fill = "") +
          theme_light()
      } else {
        feature_contribs_df <- feature_contribs_df[order(feature_contribs_df$Contribution, decreasing = TRUE), ][1:15, ]
        p <- feature_contribs_df %>% 
          mutate(Feature = fct_reorder(Feature, Contribution)) %>%
          ggplot(aes(x = Feature, y = Contribution, fill = group)) +
          geom_bar(stat = "identity", width = 0.7) +
          scale_fill_manual(values = colors) +
          scale_y_continuous(trans = "asn") +
          coord_flip() +
          xlab("") + ylab("Feature Contribution") + labs(fill = "") +
          theme_light()
      }
      
      # Save the plot
      ggsave(paste0("./genfiles_", version, "/figures_SI/shapleyvalues_barchart_year_", y, ".svg"), plot = p, width = 5, height = 5)
    }
    
    # Stop the parallel cluster
    stopCluster(cl)
  }
}

# Convert predictions to level form
data$oos_pred_level <- 10^data$oos_pred

# Add confidence intervals by bootstrapping if required
if(standarderrors == "Y"){
  # Create a function to generate predictions from bootstrapped samples
  generate_bootstrap_predictions <- function(period, n_bootstrap = 100){
    set.seed(period * 100)
    
    fulldata_sub <- subset(labeled_data, histperiod == period)
    oosdata <- subset(data, histperiod == period)
    
    # Ensure complete cases for training
    fulldata_sub <- subset(fulldata_sub, is.na(diversity_died) == F & is.na(diversity) == F 
                           & is.na(ubiquity_died) == F & is.na(ubiquity) == F
                           & is.na(diversity_immigrated) == F & is.na(diversity_emigrated) == F
                           & is.na(ubiquity_immigrated) == F & is.na(ubiquity_emigrated) == F & is.na(GDPpc_t0) == F)
    
    # Handle year variables as numeric
    cols <- match("year1300", colnames(fulldata_sub)):match("year2000", colnames(fulldata_sub))
    fulldata_sub[ , cols] <- apply(fulldata_sub[ , cols], 2, function(x) as.numeric(as.character(x)))
    
    startcolumn <- 14
    
    # Set best parameters (these would come from previous tuning)
    best_mtry <- as.numeric(read.csv2(paste0("./genfiles_", version, "/figures_SI/RF_results/mtry_", period, ".csv")))
    best_min_node_size <- as.numeric(read.csv2(paste0("./genfiles_", version, "/figures_SI/RF_results/min_node_size_", period, ".csv")))
    
    # Prepare OOS data
    oosdata[ , cols] <- apply(oosdata[ , cols], 2, function(x) as.numeric(as.character(x)))
    
    # Store bootstrap predictions
    all_predictions <- matrix(NA, nrow = nrow(oosdata), ncol = n_bootstrap)
    
    for(b in 1:n_bootstrap){
      # Create bootstrap sample
      boot_idx <- sample(1:nrow(fulldata_sub), nrow(fulldata_sub), replace = TRUE)
      boot_data <- fulldata_sub[boot_idx, ]
      
      # Train on bootstrap sample
      boot_model <- ranger(
        y = log10(boot_data$GDPpc),
        x = boot_data[, startcolumn:ncol(fulldata_sub)],
        num.trees = 500,
        mtry = best_mtry,
        min.node.size = best_min_node_size,
        importance = 'none',
        seed = period * 1000 + b
      )
      
      # Predict on OOS data
      all_predictions[, b] <- predict(boot_model, data = oosdata[, startcolumn:ncol(fulldata_sub)])$predictions
    }
    
    # Calculate confidence intervals
    lower_quantile <- 0.025
    upper_quantile <- 0.975
    
    oosdata$oos_pred_lower <- apply(all_predictions, 1, function(x) quantile(x, lower_quantile, na.rm = TRUE))
    oosdata$oos_pred_upper <- apply(all_predictions, 1, function(x) quantile(x, upper_quantile, na.rm = TRUE))
    
    # Convert to level
    oosdata$oos_pred_level_lower <- 10^oosdata$oos_pred_lower
    oosdata$oos_pred_level_upper <- 10^oosdata$oos_pred_upper
    
    return(oosdata[, c("ID", "oos_pred_lower", "oos_pred_upper", "oos_pred_level_lower", "oos_pred_level_upper")])
  }
  
  # Run bootstrap for each period
  bootstrap_results <- list()
  for(p in 1:5){
    bootstrap_results[[p]] <- generate_bootstrap_predictions(p)
  }
  
  # Combine results
  all_bootstrap_results <- do.call(rbind, bootstrap_results)
  
  # Update data with confidence intervals
  data$oos_pred_lower <- vlookup(data$ID, all_bootstrap_results, lookup_column = "ID", result_column = "oos_pred_lower")
  data$oos_pred_upper <- vlookup(data$ID, all_bootstrap_results, lookup_column = "ID", result_column = "oos_pred_upper")
  data$oos_pred_level_lower <- vlookup(data$ID, all_bootstrap_results, lookup_column = "ID", result_column = "oos_pred_level_lower")
  data$oos_pred_level_upper <- vlookup(data$ID, all_bootstrap_results, lookup_column = "ID", result_column = "oos_pred_level_upper")
}
