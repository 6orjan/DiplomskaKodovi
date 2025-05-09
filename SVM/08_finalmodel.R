# Define SVM tuning grid for final model
training_grid_finalmodel <- expand.grid(C = seq(0.1, 10, by = 0.2))

# Set up parallel processing for speed
cores <- max(1, parallel::detectCores() - 1)  # Use all cores except one
cl <- makeCluster(cores)
registerDoParallel(cl)
cat("Using", cores, "cores for parallel processing\n")

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
}

for(p in 1:5){
  set.seed(p)
  cat("Processing period", p, "of 5\n")
  
  startcolumn <- 14
  
  fulldata_sub <- subset(labeled_data, histperiod == p)
  oosdata <- subset(data, histperiod == p)
  
  if(p == 2){
    update_GDPpc_t0(fulldata_sub, data, labeled_data, 1500, "oos_pred")
    update_GDPpc_t0(oosdata, data, labeled_data, 1500, "oos_pred")
  } else if(p == 3){
    update_GDPpc_t0(fulldata_sub, data, labeled_data, 1750, "oos_pred")
    update_GDPpc_t0(oosdata, data, labeled_data, 1750, "oos_pred")
  } else if(p == 4){
    update_GDPpc_t0(fulldata_sub, data, labeled_data, 1850, "oos_pred")
    update_GDPpc_t0(oosdata, data, labeled_data, 1850, "oos_pred")
  } else if(p == 5){
    update_GDPpc_t0(fulldata_sub, data, labeled_data, 1950, "oos_pred")
    update_GDPpc_t0(oosdata, data, labeled_data, 1950, "oos_pred")
  }
  
  # Configure cross-validation with parallel processing
  cctrl1 <- trainControl(
    method = "cv", 
    number = min(k, nrow(fulldata_sub)),
    allowParallel = TRUE
  )
  
  # Data preprocessing
  fulldata_sub <- subset(fulldata_sub, is.na(diversity_died) == F & is.na(diversity) == F 
                         & is.na(ubiquity_died) == F & is.na(ubiquity) == F
                         & is.na(diversity_immigrated) == F & is.na(diversity_emigrated) == F
                         & is.na(ubiquity_immigrated) == F & is.na(ubiquity_emigrated) == F & is.na(GDPpc_t0) == F)
  
  cols <- match("year1300", colnames(fulldata_sub)):match("year2000", colnames(fulldata_sub))
  fulldata_sub[ , cols] <- apply(fulldata_sub[ , cols], 2,            # Specify own function within apply
                                 function(x) as.numeric(as.character(x))) 
  
  # Train SVM model
  set.seed(p)
  cat("Training SVM model for period", p, "\n")
  test_class_cv_model <- train(
    fulldata_sub[,startcolumn:ncol(fulldata_sub)], 
    log10(fulldata_sub$GDPpc), 
    method = "svmLinear", 
    trControl = cctrl1, 
    metric = "MAE", 
    tuneGrid = training_grid_finalmodel,
    preProcess = c("center", "scale")
  )
  
  # Save tuning plot
  ggplot(test_class_cv_model) + theme_light()
  ggsave(paste0("./genfiles_", version, "/figures_SI/SVM_results/optimization_period_", p, ".png"), width = 15, height = 5)
  
  # Get best C value
  best_C <- test_class_cv_model$bestTune$C
  cat("Best C value for period", p, "is", best_C, "\n")
  
  # Save best parameter
  write.csv2(best_C, paste0("./genfiles_", version, "/figures_SI/SVM_results/C_parameter_", p, ".csv"))
  
  # Get model variable importance
  # Note: SVM doesn't provide coefficients like LASSO, so we use varImp
  importance <- varImp(test_class_cv_model, scale = TRUE)
  
  # Create variable importance plot
  varimp_df <- as.data.frame(importance$importance)
  varimp_df$name <- rownames(varimp_df)
  
  # Sort by importance and keep only non-zero importances
  varimp_df <- varimp_df[order(varimp_df$Overall, decreasing = TRUE), ]
  varimp_df <- subset(varimp_df, Overall > 0)
  
  # Save model details
  # Since SVM doesn't have coefficients like LASSO, we save variable importance
  write.csv2(varimp_df, paste0("./genfiles_", version, "/figures_SI/SVM_results/importance_period_", p, ".csv"))
  
  # Plot variable importance
  if(nrow(varimp_df) > 0) {
    varimp_df %>% 
      mutate(name = fct_reorder(name, Overall)) %>%
      ggplot(aes(x=name, y=Overall)) +
      geom_bar(stat="identity", width = 0.7, fill = "darkblue") +
      coord_flip() +
      xlab("") + ylab("Variable Importance") + labs(fill="") +
      theme_light()
    ggsave(paste0("./genfiles_", version, "/figures_SI/SVM_results/importance_period_", p, ".png"), 
           width = 5, height = min(20, nrow(varimp_df) / 4.5), limitsize = FALSE)
  }
  
  # Make predictions on out-of-sample data
  # Ensure oosdata has the same columns as training data
  oosdata_pred_columns <- intersect(colnames(oosdata[,startcolumn:ncol(fulldata_sub)]), 
                                    colnames(fulldata_sub[,startcolumn:ncol(fulldata_sub)]))
  
  preds <- as.data.frame(predict(test_class_cv_model, 
                                 newdata = oosdata[,oosdata_pred_columns]))
  colnames(preds) <- c("oos_pred")
  rownames(preds) <- oosdata$ID
  
  preds$period <- oosdata$period
  preds$country <- oosdata$country
  preds$country_0 <- oosdata$country_0
  
  if(rescale_regions == "Y"){
    
    preds$oos_pred_level <- 10^preds$oos_pred
    
    preds$country_0_period <- paste(preds$country_0, preds$period, sep="_")
    countries_0 <- unique(preds$country_0_period)
    
    preds$country_period <- paste(preds$country, preds$period, sep="_")
    
    preds$births <- vlookup(preds$country_period, data, result_column = "births")
    preds$deaths <- vlookup(preds$country_period, data, result_column = "deaths")
    
    preds_countrylevel <- subset(preds, country == country_0)
    
    preds_regional <- subset(preds, country != country_0)
    
    preds_regional_rescaled <- data.frame()
    for(c in countries_0){
      
      refGDPpc <- ifelse(is.na(vlookup(c, fulldata_sub, lookup_column = "ID", result_column = "GDPpc")), 
                         subset(preds_countrylevel, preds_countrylevel$country_0_period == c)$oos_pred_level,
                         vlookup(c, fulldata_sub, lookup_column = "ID", result_column = "GDPpc"))
      
      regionalGDPpc <- subset(preds_regional, preds_regional$country_0_period == c)
      
      if(nrow(regionalGDPpc) > 0 & nrow(subset(preds_countrylevel, preds_countrylevel$country_0_period == c)) > 0){
        
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
      preds_regional_rescaled <- rbind(preds_regional_rescaled, regionalGDPpc)
    }
    
    preds <- rbind(preds_countrylevel, preds_regional_rescaled)
    preds$country_0_period <- NULL
    
    preds$oos_pred <- log10(preds$oos_pred_level)
  }
  
  data$oos_pred <- ifelse(is.na(data$oos_pred), vlookup(data$ID, preds, lookup_column = "rownames", result_column = 1), data$oos_pred)
  
  # SHAPLEY VALUES calculation adapted for SVM
  if(shapley == "Y"){
    cat("Calculating Shapley values for period", p, "\n")
    
    # For SVM models, use kernelShap package
    # First ensure the package is installed
    if(!require("kernelshap")) {
      install.packages("kernelshap")
      library(kernelshap)
    }
    
    # Define a prediction function for SVM model
    pfun <- function(model, newdata) {
      predict(model, newdata = newdata)
    }
    
    # Calculate background data - a representative sample of the training data
    set.seed(123)
    if(nrow(fulldata_sub) > 100) {
      bg_rows <- sample(nrow(fulldata_sub), 100)  # Use 100 samples as background
    } else {
      bg_rows <- 1:nrow(fulldata_sub)  # Use all if less than 100
    }
    
    background_data <- fulldata_sub[bg_rows, startcolumn:ncol(fulldata_sub)]
    
    # Configure parallel processing for kernelShap
    num_cores <- detectCores() - 1  # Use all cores except one
    
    # Process in batches to avoid memory issues
    batch_size <- 50
    n_batches <- ceiling(nrow(oosdata) / batch_size)
    
    shapleyvals <- matrix(0, nrow = nrow(oosdata), ncol = ncol(fulldata_sub) - startcolumn + 1)
    rownames(shapleyvals) <- rownames(oosdata)
    colnames(shapleyvals) <- colnames(fulldata_sub)[startcolumn:ncol(fulldata_sub)]
    
    for(batch in 1:n_batches) {
      start_idx <- (batch - 1) * batch_size + 1
      end_idx <- min(batch * batch_size, nrow(oosdata))
      
      cat("Processing Shapley batch", batch, "of", n_batches, "\n")
      
      if(end_idx >= start_idx) {
        # Calculate Shapley values for this batch
        ks <- kernelshap(
          test_class_cv_model,
          X = oosdata[start_idx:end_idx, startcolumn:ncol(fulldata_sub)],
          bg_X = background_data,
          pred_fun = pfun,
          parallel = TRUE,
          parallel_args = list(n_cores = num_cores)
        )
        
        # Store the Shapley values
        shapleyvals[start_idx:end_idx, ] <- ks$shapley_values
      }
    }
    
    # Process Shapley values
    shapleyvals <- abs(shapleyvals)
    years_tobeselected <- unique(as.character(oosdata$year))
    
    for (y in years_tobeselected) {
      shapleyvals_sub <- shapleyvals[as.character(oosdata$year) == y, ]
      write.csv2(shapleyvals_sub, paste0("./genfiles_", version, "/figures_SI/shapleyvalues_SVM_year_", y, ".csv"))
      
      shaps <- as.data.frame(colMeans(shapleyvals_sub, na.rm = TRUE))
      shaps <- subset(shaps, shaps[, 1] > 0)
      shaps$name <- rownames(shaps)
      
      # For SVM, we don't have coefficient signs, so we use the relationship with target
      # This is a rough approximation - positive correlation means positive impact
      shaps$group <- "impact"
      
      colors <- c("impact" = "#52c559ff")
      shaps$period <- str_sub(shaps$name, start = -4)
      shaps$period <- ifelse(str_sub(shaps$name, end = -5, start = -8) %in% c("fore", "upto"), "incl", shaps$period)
      
      years_not_to_be <- years_tobeselected[years_tobeselected != y]
      shaps <- subset(shaps, !period %in% years_not_to_be)
      
      if (nrow(shaps) < 16) {
        p <- shaps %>% 
          mutate(name = fct_reorder(name, shaps[, 1])) %>%
          ggplot(aes(x = name, y = shaps[, 1], fill = group)) +
          geom_bar(stat = "identity", width = 0.7) +
          scale_fill_manual(values = colors) +
          scale_y_continuous(trans = "asn") +
          coord_flip() +
          xlab("") + ylab("Shapley value") + labs(fill = "") +
          theme_light()
      } else {
        shaps <- shaps[order(shaps[, 1], decreasing = TRUE), ][1:15, ]
        p <- shaps %>% 
          mutate(name = fct_reorder(name, shaps[, 1])) %>%
          ggplot(aes(x = name, y = shaps[, 1], fill = group)) +
          geom_bar(stat = "identity", width = 0.7) +
          scale_fill_manual(values = colors) +
          scale_y_continuous(trans = "asn") +
          coord_flip() +
          xlab("") + ylab("Shapley value") + labs(fill = "") +
          theme_light()
      }
      
      ggsave(paste0("./genfiles_", version, "/figures_SI/shapleyvalues_SVM_barchart_year_", y, ".svg"), plot = p, width = 5, height = 5)
    }
  }
  
  cat("Completed period", p, "of 5\n")
}

# Clean up parallel cluster
stopCluster(cl)

# Convert log predictions to levels
data$oos_pred_level <- 10^data$oos_pred

# Add confidence intervals by bootstrapping
if(standarderrors == "Y"){
  cat("Calculating confidence intervals...\n")
  
  # Use a modified bootstrapping approach for SVM
  source("./scripts/08_zz_bootstrapping_CI_SVM.R")
  
  lowerbound_share <- data_misc$oos_pred_level_lower / data_misc$oos_pred_level
  upperbound_share <- data_misc$oos_pred_level_upper / data_misc$oos_pred_level
  
  data$oos_pred_level_lower <- data$oos_pred_level * lowerbound_share
  data$oos_pred_level_upper <- data$oos_pred_level * upperbound_share
  
  data$oos_pred_lower <- log10(data$oos_pred_level_lower)
  data$oos_pred_upper <- log10(data$oos_pred_level_upper)
}

cat("Final model processing complete\n")
