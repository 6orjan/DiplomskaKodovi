#i have an R script which is checking for the performance of a random forrest model baseline and full model and the intended result is for the full model to outperform the baseline in the r squared parameter(figure ab), check the code for any errors and return to me fixed version of the code 
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

# Define n_draws if it doesn't exist
n_draws = 5

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

# Continue with your original code, but with fixes
training_grid_modelperformance <- expand.grid(mtry = seq(5, 30, by = 5))

# Assuming labeled_data is available
locations <- labeled_data %>% dplyr::group_by(country) %>% count() %>% as.data.frame()
countries <- subset(locations, str_length(country) == 3)

# Set seed for reproducibility
set.seed(123)

countries_training_long <- sample_n(subset(countries, n >= 9), size = round(nrow(subset(countries, n >= 9)) * 0.8, digits = 0))
countries_test_long <- subset(subset(countries, n >= 9), (country %in% countries_training_long$country) == FALSE)

countries_training_short <- sample_n(subset(countries, n < 9), size = round(nrow(subset(countries, n < 9)) * 0.8, digits = 0))
countries_test_short <- subset(subset(countries, n < 9), (country %in% countries_training_short$country) == FALSE)

countries_training <- rbind(countries_training_long, countries_training_short)
countries_test <- rbind(countries_test_long, countries_test_short)

countries_test$abbrev <- c("IT", "PT", "AL", "HR", "LV", "NO", "RO", "SI")
countries_test$country <- c("ITA", "PRT", "ALB", "HRV", "LVA", "NOR", "ROU", "SVN")

locations$test <- 0
for(i in 1:nrow(locations)){
  locations$test[i] <- ifelse(str_sub(locations$country[i], end = 2) %in% countries_test$abbrev
                              | str_sub(locations$country[i], end = 3) %in% countries_test$abbrev
                              | str_sub(locations$country[i], end = 3) %in% countries_test$country
                              | str_sub(locations$country[i], end = 4) %in% countries_test$abbrev
                              , 1, 0)
}

training_data <- subset(labeled_data, country %in% subset(locations, test == 0)$country)
test_data <- subset(labeled_data, country %in% subset(locations, test == 1)$country)

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

# Create deep copies of test and training data to prevent overwriting
test_data_full <- test_data
test_data_baseline <- test_data
training_data_full <- training_data
training_data_baseline <- training_data

# Define proper cross-validation control
cctrl1 <- trainControl(
  method = "cv",
  number = n_draws,  # Use n_draws instead of hardcoded value
  verboseIter = TRUE,
  savePredictions = TRUE,
  allowParallel = TRUE
)

# Get list of variables to use as predictors
# Remove any ID columns, target variables, and prediction columns
predictors <- setdiff(colnames(training_data), 
                      c("GDPpc", "prediction", "prediction_abs", "RMSE", "AME", "RMSE_logs", 
                        "ID", "ID2"))

# Check for NA values in predictors and target
na_counts_predictors <- colSums(is.na(training_data[, predictors]))
print("NA counts in predictors:")
print(na_counts_predictors[na_counts_predictors > 0])

na_count_target <- sum(is.na(log10(training_data$GDPpc)))
print(paste("NA count in log10(GDPpc):", na_count_target))

# Handle any NA values
for(col in predictors) {
  if(sum(is.na(training_data_full[[col]])) > 0) {
    # Use mean imputation for numeric columns
    if(is.numeric(training_data_full[[col]])) {
      col_mean <- mean(training_data_full[[col]], na.rm = TRUE)
      training_data_full[[col]][is.na(training_data_full[[col]])] <- col_mean
      test_data_full[[col]][is.na(test_data_full[[col]])] <- col_mean
      training_data_baseline[[col]][is.na(training_data_baseline[[col]])] <- col_mean
      test_data_baseline[[col]][is.na(test_data_baseline[[col]])] <- col_mean
    } else {
      # For non-numeric columns, use the most frequent value
      mode_val <- names(sort(table(training_data_full[[col]]), decreasing = TRUE))[1]
      training_data_full[[col]][is.na(training_data_full[[col]])] <- mode_val
      test_data_full[[col]][is.na(test_data_full[[col]])] <- mode_val
      training_data_baseline[[col]][is.na(training_data_baseline[[col]])] <- mode_val
      test_data_baseline[[col]][is.na(test_data_baseline[[col]])] <- mode_val
    }
  }
}

# Make sure all columns have the same type in training and test
for(col in predictors) {
  if(class(training_data_full[[col]]) != class(test_data_full[[col]])) {
    test_data_full[[col]] <- as(test_data_full[[col]], class(training_data_full[[col]]))
  }
  if(class(training_data_baseline[[col]]) != class(test_data_baseline[[col]])) {
    test_data_baseline[[col]] <- as(test_data_baseline[[col]], class(training_data_baseline[[col]]))
  }
}

# ==========================================
# FULL MODEL - RANDOM FOREST
# ==========================================
# Train the RandomForest model directly with dataframes 
set.seed(123)  # For reproducibility
rf_model <- train(
  x = training_data_full[, predictors],
  y = log10(training_data_full$GDPpc),  # Transform GDP to log scale for prediction
  method = "rf",  # Random Forest method
  trControl = cctrl1,
  tuneGrid = training_grid_modelperformance,  # Use the pre-defined tuning grid
  ntree = 500,  # Use 500 trees in the Random Forest
  importance = TRUE  # Keep track of variable importance
)

# Print model details to verify it trained correctly
print(rf_model)

# Make predictions using the trained model
test_data_full$prediction <- predict(rf_model, newdata = test_data_full[, predictors])
test_data_full$prediction_abs <- 10^test_data_full$prediction

# Compute RMSE and AME
test_data_full$RMSE <- (test_data_full$prediction_abs - test_data_full$GDPpc)^2
test_data_full$RMSE_logs <- (test_data_full$prediction - log10(test_data_full$GDPpc))^2
test_data_full$AME <- abs(test_data_full$prediction_abs - test_data_full$GDPpc)

# Calculate performance metrics
RMSE_overall_fullmodel <- test_data_full %>% dplyr::summarize(sqrt(mean(RMSE, na.rm=TRUE)))
RMSE_overall_fullmodel$avgGDPpc <- test_data_full %>% dplyr::summarize(mean(GDPpc, na.rm=TRUE))

AME_overall_fullmodel <- test_data_full %>% dplyr::summarize(mean(AME, na.rm=TRUE))
AME_overall_fullmodel$avgGDPpc <- test_data_full %>% dplyr::summarize(mean(GDPpc, na.rm=TRUE))

RMSE_tab_fullmodel <- test_data_full %>% dplyr::group_by(histperiod) %>% dplyr::summarize(sqrt(mean(RMSE, na.rm=TRUE))) %>% as.data.frame()
RMSE_logs_tab_fullmodel <- test_data_full %>% dplyr::group_by(histperiod) %>% dplyr::summarize(sqrt(mean(RMSE_logs, na.rm=TRUE))) %>% as.data.frame()

AME_tab_fullmodel <- test_data_full %>% dplyr::group_by(histperiod) %>% dplyr::summarize(mean(AME, na.rm=TRUE)) %>% as.data.frame()

avgGDPpc_fullmodel <- test_data_full %>% dplyr::group_by(histperiod) %>% dplyr::summarize(mean(GDPpc, na.rm=TRUE))

RMSE_tab_fullmodel$avgGDPpc <- avgGDPpc_fullmodel$`mean(GDPpc, na.rm = TRUE)`
RMSE_tab_fullmodel$share <- RMSE_tab_fullmodel$`sqrt(mean(RMSE, na.rm = TRUE))` / RMSE_tab_fullmodel$avgGDPpc

AME_tab_fullmodel$avgGDPpc <- avgGDPpc_fullmodel$`mean(GDPpc, na.rm = TRUE)`
AME_tab_fullmodel$share <- AME_tab_fullmodel$`mean(AME, na.rm = TRUE)` / AME_tab_fullmodel$avgGDPpc

# ==========================================
# BASELINE MODEL - FIXED EFFECTS ONLY
# ==========================================

for(p in 1:5){
  training_data_sub <- subset(training_data_baseline, histperiod == p)
  test_data_sub <- subset(test_data_baseline, histperiod == p)
  
  # Only run update_GDPpc_t0 if that function exists
  if(exists("update_GDPpc_t0")) {
    if(p == 2){
      test_data_sub <- update_GDPpc_t0(test_data_sub, test_data_baseline, labeled_data, 1500, "")
      training_data_sub <- update_GDPpc_t0(training_data_sub, training_data_baseline, labeled_data, 1500, "")
    } else if(p == 3){
      test_data_sub <- update_GDPpc_t0(test_data_sub, test_data_baseline, labeled_data, 1750, "")
      training_data_sub <- update_GDPpc_t0(training_data_sub, training_data_baseline, labeled_data, 1750, "")
    } else if(p == 4){
      test_data_sub <- update_GDPpc_t0(test_data_sub, test_data_baseline, labeled_data, 1850, "")
      training_data_sub <- update_GDPpc_t0(training_data_sub, training_data_baseline, labeled_data, 1850, "")
    } else if(p == 5){
      test_data_sub <- update_GDPpc_t0(test_data_sub, test_data_baseline, labeled_data, 1950, "")
      training_data_sub <- update_GDPpc_t0(training_data_sub, training_data_baseline, labeled_data, 1950, "")
    }
  }
  
  # Check if there's enough data to fit the model
  if(nrow(training_data_sub) > 0 && "GDPpc_t0" %in% colnames(training_data_sub)) {
    # Handle NAs in predictor variables
    training_data_sub$GDPpc_t0[is.na(training_data_sub$GDPpc_t0)] <- mean(training_data_sub$GDPpc_t0, na.rm=TRUE)
    test_data_sub$GDPpc_t0[is.na(test_data_sub$GDPpc_t0)] <- mean(training_data_sub$GDPpc_t0, na.rm=TRUE)
    
    # Fit fixed effects model
    tryCatch({
      # Check if UN_subregion exists, if not use a simpler model
      if("UN_subregion" %in% colnames(training_data_sub)) {
        model <- feols(
          as.formula("log10(GDPpc) ~ GDPpc_t0 + as.factor(year) + as.factor(UN_subregion)"),
          data = training_data_sub
        )
      } else {
        model <- feols(
          as.formula("log10(GDPpc) ~ GDPpc_t0 + as.factor(year)"),
          data = training_data_sub
        )
      }
      
      # Make predictions
      test_data_sub$prediction <- predict(model, newdata = test_data_sub)
      test_data_sub$prediction_abs <- 10^test_data_sub$prediction
      
      # Update test data by ID
      if("ID" %in% colnames(test_data_sub) && "ID" %in% colnames(test_data_baseline)) {
        for(i in 1:nrow(test_data_sub)) {
          row_id <- test_data_sub$ID[i]
          test_idx <- which(test_data_baseline$ID == row_id)
          if(length(test_idx) > 0) {
            test_data_baseline$prediction[test_idx] <- test_data_sub$prediction[i]
            test_data_baseline$prediction_abs[test_idx] <- test_data_sub$prediction_abs[i]
          }
        }
      }
      
      # Make predictions for training data
      training_data_sub$prediction <- predict(model, newdata = training_data_sub)
      training_data_sub$prediction_abs <- 10^training_data_sub$prediction
      
      # Update training data by ID
      if("ID" %in% colnames(training_data_sub) && "ID" %in% colnames(training_data_baseline)) {
        for(i in 1:nrow(training_data_sub)) {
          row_id <- training_data_sub$ID[i]
          train_idx <- which(training_data_baseline$ID == row_id)
          if(length(train_idx) > 0) {
            training_data_baseline$prediction[train_idx] <- training_data_sub$prediction[i]
            training_data_baseline$prediction_abs[train_idx] <- training_data_sub$prediction_abs[i]
          }
        }
      }
    }, error = function(e) {
      cat("Error in model for period", p, ":", e$message, "\n")
    })
  } else {
    cat("Skipping period", p, "due to insufficient data or missing GDPpc_t0 column\n")
  }
}

# Calculate performance metrics for baseline model
test_data_baseline$RMSE <- (test_data_baseline$prediction_abs - test_data_baseline$GDPpc)^2
test_data_baseline$RMSE_logs <- (test_data_baseline$prediction - log10(test_data_baseline$GDPpc))^2
test_data_baseline$AME <- abs(test_data_baseline$prediction_abs - test_data_baseline$GDPpc)

RMSE_overall_baseline <- test_data_baseline %>% dplyr::summarize(sqrt(mean(RMSE, na.rm=TRUE)))
RMSE_overall_baseline$avgGDPpc <- test_data_baseline %>% dplyr::summarize(mean(GDPpc, na.rm=TRUE))

AME_overall_baseline <- test_data_baseline %>% dplyr::summarize(mean(AME, na.rm=TRUE))
AME_overall_baseline$avgGDPpc <- test_data_baseline %>% dplyr::summarize(mean(GDPpc, na.rm=TRUE))

RMSE_tab_baseline <- test_data_baseline %>% dplyr::group_by(histperiod) %>% dplyr::summarize(sqrt(mean(RMSE, na.rm=TRUE))) %>% as.data.frame()
RMSE_logs_tab_baseline <- test_data_baseline %>% dplyr::group_by(histperiod) %>% dplyr::summarize(sqrt(mean(RMSE_logs, na.rm=TRUE))) %>% as.data.frame()

AME_tab_baseline <- test_data_baseline %>% dplyr::group_by(histperiod) %>% dplyr::summarize(mean(AME, na.rm=TRUE)) %>% as.data.frame()

avgGDPpc_baseline <- test_data_baseline %>% dplyr::group_by(histperiod) %>% dplyr::summarize(mean(GDPpc, na.rm=TRUE))

RMSE_tab_baseline$avgGDPpc <- avgGDPpc_baseline$`mean(GDPpc, na.rm = TRUE)`
RMSE_tab_baseline$share <- RMSE_tab_baseline$`sqrt(mean(RMSE, na.rm = TRUE))` / RMSE_tab_baseline$avgGDPpc

AME_tab_baseline$avgGDPpc <- avgGDPpc_baseline$`mean(GDPpc, na.rm = TRUE)`
AME_tab_baseline$share <- AME_tab_baseline$`mean(AME, na.rm = TRUE)` / AME_tab_baseline$avgGDPpc

# ==========================================
# VISUALIZATIONS
# ==========================================

# First determine the common limits needed for both plots
y_min <- min(min(test_data_baseline$GDPpc, na.rm=TRUE), min(test_data_full$GDPpc, na.rm=TRUE))
y_max <- max(max(test_data_baseline$GDPpc, na.rm=TRUE), max(test_data_full$GDPpc, na.rm=TRUE))
x_min <- min(min(test_data_baseline$prediction_abs, na.rm=TRUE), min(test_data_full$prediction_abs, na.rm=TRUE))
x_max <- max(max(test_data_baseline$prediction_abs, na.rm=TRUE), max(test_data_full$prediction_abs, na.rm=TRUE))

# Create baseline model visualization with consistent axis limits
baselinemodel <- ggplot(test_data_baseline, aes(y=GDPpc, x=prediction_abs)) + 
  stat_poly_eq(formula = y~x, aes(label = after_stat(rr.label)), parse = TRUE) +
  geom_smooth(method="lm", se=FALSE, linetype="dashed", color="grey") + 
  geom_abline(slope = 1, intercept = 0, color = "grey") + 
  geom_point(size=1.5, color = "darkorange1") + 
  geom_text(aes(label=ID2), check_overlap=TRUE, size=2.5, nudge_y = -0.02) +
  scale_x_continuous(trans='log10', labels = scales::comma, limits = c(x_min, x_max)) + 
  scale_y_continuous(trans='log10', labels = scales::comma, limits = c(y_min, y_max)) + 
  theme_light() + 
  labs(x="Prediction (log scale)", y="GDP per capita (log scale)", title = "Baseline model")

# Create full model visualization with the same axis limits
fullmodel <- ggplot(test_data_full, aes(y=GDPpc, x=prediction_abs)) + 
  stat_poly_eq(formula = y~x, aes(label = after_stat(rr.label)), parse = TRUE) +
  geom_smooth(method="lm", se=FALSE, linetype="dashed", color="grey") + 
  geom_abline(slope = 1, intercept = 0, color = "grey") + 
  geom_point(size=1.5, color = "darkorange1") + 
  geom_text(aes(label=ID2), check_overlap=TRUE, size=2.5, nudge_y = -0.02) +
  scale_x_continuous(trans='log10', labels = scales::comma, limits = c(x_min, x_max)) + 
  scale_y_continuous(trans='log10', labels = scales::comma, limits = c(y_min, y_max)) + 
  theme_light() + 
  labs(x="Prediction (log scale)", y="GDP per capita (log scale)", title = "Full model")

# Print individual plots first to ensure they render properly
print(baselinemodel)
print(fullmodel)

# Save individual plots
if(exists("version")) {
  ggsave(paste0("./genfiles_", version, "/figures/baselinemodel.svg"), plot=baselinemodel, width=4, height=4)
  ggsave(paste0("./genfiles_", version, "/figures/fullmodel.svg"), plot=fullmodel, width=4, height=4)
}

# Create combined plot with explicit dimensions and alignment
model_comparison <- plot_grid(
  baselinemodel, 
  fullmodel, 
  labels = c("A", "B"), 
  ncol = 2,
  align = 'vh',
  axis = 'tblr'
)

# Print the combined plot to verify
print(model_comparison)

# Save the final figure
if(exists("version")) {
  ggsave(paste0("./genfiles_", version, "/figures/Fig2_AB.svg"), plot=model_comparison, width=8, height=4)
}

# Compare models with visualizations
RMSE_tab_fullmodel$group <- "Full model"
RMSE_tab_fullmodel$century <- c("1300-1500", "1550-1750", "1800-1850", "1900-1950", "2000")

RMSE_tab_baseline$group <- "Naive model"
RMSE_tab_baseline$century <- c("1300-1500", "1550-1750", "1800-1850", "1900-1950", "2000")
RMSE_tab_full <- rbind(RMSE_tab_baseline, RMSE_tab_fullmodel)

# Create RMSE comparison plot
rmse_plot <- ggplot(RMSE_tab_full, aes(x=share*100, y=as.factor(histperiod), fill=as.factor(group))) + 
  geom_bar(stat="identity", position="dodge", width=0.7) + 
  theme_light() +
  labs(y="Century", x="RMSE, % of average GDP per capita", fill="Model") +
  theme(legend.title=element_blank())

# Print RMSE comparison
print(rmse_plot)

# Create AME comparison
AME_tab_fullmodel$group <- "Full model"
AME_tab_fullmodel$century <- c("1300-1500", "1550-1750", "1800-1850", "1900-1950", "2000")

AME_tab_baseline$group <- "Naive model"
AME_tab_baseline$century <- c("1300-1500", "1550-1750", "1800-1850", "1900-1950", "2000")

AME_tab_full <- rbind(AME_tab_baseline, AME_tab_fullmodel)

# Create AME comparison plot
ame_plot <- ggplot(AME_tab_full, aes(x=share*100, y=as.factor(histperiod), fill=as.factor(group))) + 
  geom_bar(stat="identity", position="dodge", width=0.7) + 
  theme_light() +
  labs(y="Century", x="AME, % of average GDP per capita", fill="Model") +
  theme(legend.title=element_blank())

# Print AME comparison
print(ame_plot)

# Save comparison plots
if(exists("version")) {
  ggsave(paste0("./genfiles_", version, "/figures/RMSE_comparison.svg"), plot=rmse_plot, width=6, height=4)
  ggsave(paste0("./genfiles_", version, "/figures/AME_comparison.svg"), plot=ame_plot, width=6, height=4)
}

# Print a summary of the R-squared values for both models to easily compare
cat("\n==== MODEL PERFORMANCE SUMMARY ====\n")
cat("Full Model (Random Forest) R-squared: ", summary(lm(GDPpc ~ prediction_abs, data=test_data_full))$r.squared, "\n")
cat("Baseline Model (Fixed Effects) R-squared: ", summary(lm(GDPpc ~ prediction_abs, data=test_data_baseline))$r.squared, "\n")
