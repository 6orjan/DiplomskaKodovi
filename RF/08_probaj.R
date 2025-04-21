# Updated for Random Forest (RF) instead of LASSO

# Correct tuning grid for Random Forest
training_grid_finalmodel <- expand.grid(mtry = seq(5, 30, by = 5))

# Initialize prediction columns
data$oos_pred <- NA
data$oos_pred_lower <- NA
data$oos_pred_upper <- NA
data$oos_pred_level <- NA
data$oos_pred_level_lower <- NA
data$oos_pred_level_upper <- NA

# Function to update GDP prediction values
update_GDPpc_t0 <- function(data_subset, data, labeled_data, year, column_name_preds) {
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
}

# Loop through historical periods for model training
for(p in 1:5) {
  set.seed(p)
  
  startcolumn <- 14
  
  fulldata_sub <- subset(labeled_data, histperiod == p)
  oosdata <- subset(data, histperiod == p)
  
  # Update GDP predictions for each historical period
  if(p == 2) {
    update_GDPpc_t0(fulldata_sub, data, labeled_data, 1500, "oos_pred")
    update_GDPpc_t0(oosdata, data, labeled_data, 1500, "oos_pred")
  } else if(p == 3) {
    update_GDPpc_t0(fulldata_sub, data, labeled_data, 1750, "oos_pred")
    update_GDPpc_t0(oosdata, data, labeled_data, 1750, "oos_pred")
  } else if(p == 4) {
    update_GDPpc_t0(fulldata_sub, data, labeled_data, 1850, "oos_pred")
    update_GDPpc_t0(oosdata, data, labeled_data, 1850, "oos_pred")
  } else if(p == 5) {
    update_GDPpc_t0(fulldata_sub, data, labeled_data, 1950, "oos_pred")
    update_GDPpc_t0(oosdata, data, labeled_data, 1950, "oos_pred")
  }
  
  # Cross-validation control
  cctrl1 <- trainControl(method = "cv", number = min(k, nrow(fulldata_sub)))
  
  # Ensure there are no missing values in training data
  fulldata_sub <- subset(fulldata_sub, complete.cases(fulldata_sub[, startcolumn:ncol(fulldata_sub)]))
  
  # Convert year columns to numeric
  cols <- match("year1300", colnames(fulldata_sub)):match("year2000", colnames(fulldata_sub))
  fulldata_sub[, cols] <- apply(fulldata_sub[, cols], 2, function(x) as.numeric(as.character(x)))
  
  # Train Random Forest model
  rf_model <- train(
    x = fulldata_sub[, startcolumn:ncol(fulldata_sub)],
    y = log10(fulldata_sub$GDPpc),
    method = "rf",
    trControl = cctrl1,
    tuneGrid = expand.grid(mtry = seq(3, 10, by = 1)),  # Explore mtry values in a smaller range
    ntree = 500,  # Simplify the model by reducing the number of trees
    importance = TRUE
  )
  
  # Plot and save model performance results
  ggplot(rf_model) + theme_light()
  ggsave(paste0("./genfiles_", version, "/figures_SI/RF_results/optimization_period_", p, ".png"), width = 15, height = 5)
  
  # Save model and predictions
  output_dir <- paste0("./genfiles_", version, "/figures_SI/RF_results/")
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  # Save model safely
  saveRDS(rf_model, paste0(output_dir, "model_period_", p, ".rds"))
  
  # Make predictions and store them
  preds <- predict(rf_model, newdata = oosdata[, startcolumn:ncol(fulldata_sub)])
  preds <- 10^preds
  
  data$oos_pred_level[data$histperiod == p] <- preds
  data$oos_pred[data$histperiod == p] <- log10(preds)
}

# Convert final predictions to original scale
data$oos_pred_level <- 10^data$oos_pred

# Add confidence intervals by bootstrapping if needed
if(standarderrors == "Y") {
  source("./scripts/08_zz_bootstrapping_CI.R")
  lowerbound_share <- data_misc$oos_pred_level_lower / data_misc$oos_pred_level
  upperbound_share <- data_misc$oos_pred_level_upper / data_misc$oos_pred_level
  
  data$oos_pred_level_lower <- data$oos_pred_level * lowerbound_share
  data$oos_pred_level_upper <- data$oos_pred_level * upperbound_share
  data$oos_pred_lower <- log10(data$oos_pred_level_lower)
  data$oos_pred_upper <- log10(data$oos_pred_level_upper)
}

# Create a figure for the best model performance
# Filter out rows where we have both predictions and actual values
model_eval_data <- subset(data, !is.na(oos_pred_level) & !is.na(GDPpc))

# Create a plot of predicted vs actual values
best_model_plot <- ggplot(model_eval_data, aes(x = oos_pred_level, y = GDPpc)) +
  geom_point(color = "darkorange1", size = 1.5) +
  geom_abline(slope = 1, intercept = 0, color = "grey") +
  geom_smooth(method = "lm", se = FALSE, linetype = "dashed", color = "grey") +
  stat_poly_eq(formula = y ~ x, aes(label = paste(after_stat(rr.label), sep = "~~~")), parse = TRUE) +
  scale_x_continuous(trans = 'log10', labels = scales::comma) +
  scale_y_continuous(trans = 'log10', labels = scales::comma) +
  labs(x = "Predicted GDP per capita (log scale)", 
       y = "Actual GDP per capita (log scale)",
       title = "Overall Random Forest Model Performance") +
  theme_light()

# Print the plot to view it
print(best_model_plot)

# Save the best model performance plot
ggsave(paste0("./genfiles_", version, "/figures_SI/RF_results/best_model_performance.png"), 
       plot = best_model_plot, width = 8, height = 6)

# Optionally, create period-specific performance plots
if(exists("by_period") && by_period == TRUE) {
  # Create a list to store each period's plot
  period_plots <- list()
  
  # Create separate plots for each historical period
  for(p in 1:5) {
    period_data <- subset(model_eval_data, histperiod == p)
    
    if(nrow(period_data) > 5) {  # Only create plot if we have enough data points
      period_plot <- ggplot(period_data, aes(x = oos_pred_level, y = GDPpc)) +
        geom_point(color = "darkorange1", size = 1.5) +
        geom_abline(slope = 1, intercept = 0, color = "grey") +
        geom_smooth(method = "lm", se = FALSE, linetype = "dashed", color = "grey") +
        stat_poly_eq(formula = y ~ x, aes(label = paste(after_stat(rr.label), sep = "~~~")), parse = TRUE) +
        scale_x_continuous(trans = 'log10', labels = scales::comma) +
        scale_y_continuous(trans = 'log10', labels = scales::comma) +
        labs(x = "Predicted GDP per capita (log scale)", 
             y = "Actual GDP per capita (log scale)",
             title = paste("Period", p, "Model Performance")) +
        theme_light()
      
      period_plots[[p]] <- period_plot
      
      # Save individual period plots
      ggsave(paste0("./genfiles_", version, "/figures_SI/RF_results/period_", p, "_performance.png"), 
             plot = period_plot, width = 6, height = 6)
    }
  }
  
  # Create a combined plot of all periods if cowplot is available
  if(requireNamespace("cowplot", quietly = TRUE) && length(period_plots) > 1) {
    # Filter out NULL elements
    period_plots <- period_plots[!sapply(period_plots, is.null)]
    
    # Calculate grid dimensions
    n_plots <- length(period_plots)
    n_cols <- min(2, n_plots)
    n_rows <- ceiling(n_plots / n_cols)
    
    # Create the combined plot
    combined_plot <- cowplot::plot_grid(plotlist = period_plots, ncol = n_cols)
    
    # Save the combined plot
    ggsave(paste0("./genfiles_", version, "/figures_SI/RF_results/all_periods_performance.png"), 
           plot = combined_plot, width = 12, height = 8)
  }
}
