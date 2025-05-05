# Fix for script 07_model_performance.R

# First, ensure all necessary packages are loaded
library(dplyr)
library(stringr)
library(caret)
library(randomForest)
library(ranger)  # Add ranger package for better Random Forest implementation
library(ggplot2)
library(ggpmisc)  # For stat_poly_eq
library(cowplot)  # For plot_grid
library(fixest)   # For feols
library(scales)   # For axis formatting

# Define n_draws if it doesn't exist (this was missing in your code)
if(!exists("n_draws")) {
  n_draws <- 5  # Default value
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

# Modified training grid parameters as specified
ranger_grid <- expand.grid(
  mtry = seq(2, 40, by = 2),          # Wider range with finer steps
  splitrule = c("variance", "extratrees", "maxstat"),  # Try all available split rules for regression
  min.node.size = c(1, 3, 5, 10)      # Test smaller and larger minimum node sizes
)

# Note: num.trees is not a tuning parameter in caret's ranger implementation
# It should be passed directly to the train function

# Assuming labeled_data is available
locations <- labeled_data %>% dplyr::group_by(country) %>% count() %>% as.data.frame()
countries <- subset(locations, str_length(country) == 3)

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

# BEGIN MODIFIED 07a_Rf_for_modelperformance.R CONTENT
test_data_preds <- test_data
training_data_preds <- training_data
test_data_preds$prediction_abs <- NA
test_data_preds$prediction <- NA
training_data_preds$prediction <- NA
training_data_preds$prediction_abs <- NA

# Updated update_GDPpc_t0 function for 07a
update_GDPpc_t0_07a <- function(data_subset, data, labeled_data, year, column_name_preds) {
  # Create GDPpc_t0 column if it doesn't exist
  if(!"GDPpc_t0" %in% colnames(data_subset)) {
    data_subset$GDPpc_t0 <- NA
  }
  
  for(m in 1:nrow(data_subset)){
    # Check if country and ID2 columns exist
    if(!"country" %in% colnames(data_subset) || !"ID2" %in% colnames(data)) {
      next  # Skip if required columns don't exist
    }
    
    # Create ID2-like identifier
    id2_key <- paste(data_subset$country[m], year, sep="_")
    
    # Check if column_name_preds exists in data
    if(!column_name_preds %in% colnames(data)) {
      data_subset$GDPpc_t0[m] <- NA
      next  # Skip this iteration
    }
    
    # Try to get tempGDP, with error handling
    tempGDP <- NA
    tryCatch({
      matched_rows <- data$ID2 == id2_key
      if(any(matched_rows)) {
        tempGDP <- data[[column_name_preds]][matched_rows][1]
      }
    }, error = function(e) {
      warning(paste("Error looking up ID2:", id2_key, "Error:", e$message))
    })
    
    # Try alternate lookup methods if first one failed
    if(is.na(tempGDP) && "country_0" %in% colnames(data_subset)) {
      id2_alt_key <- paste(data_subset$country_0[m], year, sep="_")
      
      # Try in labeled_data
      tryCatch({
        matched_rows <- labeled_data$ID2 == id2_alt_key
        if(any(matched_rows) && "GDPpc" %in% colnames(labeled_data)) {
          # Ensure GDPpc is positive before log transform
          gdp_value <- labeled_data$GDPpc[matched_rows][1]
          if(is.numeric(gdp_value) && gdp_value > 0) {
            tempGDP <- log10(gdp_value)
          }
        }
      }, error = function(e) {
        warning(paste("Error looking up ID2 in labeled_data:", id2_alt_key, "Error:", e$message))
      })
      
      # Try in data if still NA
      if(is.na(tempGDP)) {
        tryCatch({
          matched_rows <- data$ID2 == id2_alt_key
          if(any(matched_rows)) {
            tempGDP <- data[[column_name_preds]][matched_rows][1]
          }
        }, error = function(e) {
          warning(paste("Error looking up alt ID2 in data:", id2_alt_key, "Error:", e$message))
        })
      }
    }
    
    # Update GDPpc_t0 if we found a value
    if(!is.na(tempGDP)) {
      # Check if labeled_data has the required columns
      if(all(c("ID2", "GDPpc") %in% colnames(labeled_data))) {
        id2_check <- paste(data_subset$country[m], year, sep="_")
        matched_rows <- labeled_data$ID2 == id2_check
        existing_gdp <- NA
        if(any(matched_rows)) {
          existing_gdp <- labeled_data$GDPpc[matched_rows][1]
        }
        
        data_subset$GDPpc_t0[m] <- ifelse(is.na(existing_gdp), tempGDP, data_subset$GDPpc_t0[m])
      } else {
        data_subset$GDPpc_t0[m] <- tempGDP
      }
    }
  }
  return(data_subset)
}

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

# Ensure GDPpc is positive before log transform
training_data$GDPpc_safe <- pmax(training_data$GDPpc, 1e-10)
test_data$GDPpc_safe <- pmax(test_data$GDPpc, 1e-10)

na_count_target <- sum(is.na(log10(training_data$GDPpc_safe)))
print(paste("NA count in log10(GDPpc):", na_count_target))

# Handle any NA values
for(col in predictors) {
  if(sum(is.na(training_data[[col]])) > 0) {
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

# Make sure all columns have the same type in training and test
for(col in predictors) {
  if(class(training_data[[col]]) != class(test_data[[col]])) {
    test_data[[col]] <- as(test_data[[col]], class(training_data[[col]]))
  }
}

# Train the model using ranger method instead of rf to accommodate our expanded parameter grid
set.seed(123)  # For reproducibility
rf_model <- caret::train(
  x = training_data[, predictors],
  y = log10(training_data$GDPpc_safe),  # Transform GDP to log scale for prediction
  method = "ranger",  # Use ranger instead of rf for better performance with our expanded grid
  trControl = cctrl1,
  tuneGrid = ranger_grid,  # Use our expanded parameter grid
  num.trees = 1000,        # Set number of trees directly
  importance = 'impurity',  # Track variable importance
  respect.unordered.factors = TRUE,  # Properly handle categorical variables
  verbose = TRUE  # Provide more detailed output during training
)

# Print model details to verify it trained correctly
print(rf_model)

# Make predictions using the trained model
test_data$prediction <- predict(rf_model, newdata = test_data[, predictors])
test_data$prediction_abs <- 10^test_data$prediction

# Compute RMSE and AME
test_data$RMSE <- (test_data$prediction_abs - test_data$GDPpc)^2
test_data$RMSE_logs <- (test_data$prediction - log10(test_data$GDPpc_safe))^2
test_data$AME <- abs(test_data$prediction_abs - test_data$GDPpc)

# Calculate performance metrics
RMSE_overall <- test_data %>% dplyr::summarize(RMSE_val = sqrt(mean(RMSE, na.rm=TRUE)))
RMSE_overall$avgGDPpc <- test_data %>% dplyr::summarize(avg_GDP = mean(GDPpc, na.rm=TRUE)) %>% pull(avg_GDP)

AME_overall <- test_data %>% dplyr::summarize(AME_val = mean(AME, na.rm=TRUE))
AME_overall$avgGDPpc <- test_data %>% dplyr::summarize(avg_GDP = mean(GDPpc, na.rm=TRUE)) %>% pull(avg_GDP)

RMSE_tab <- test_data %>% 
  dplyr::group_by(histperiod) %>% 
  dplyr::summarize(RMSE_val = sqrt(mean(RMSE, na.rm=TRUE))) %>% 
  as.data.frame()

RMSE_logs_tab <- test_data %>% 
  dplyr::group_by(histperiod) %>% 
  dplyr::summarize(RMSE_logs_val = sqrt(mean(RMSE_logs, na.rm=TRUE))) %>% 
  as.data.frame()

AME_tab <- test_data %>% 
  dplyr::group_by(histperiod) %>% 
  dplyr::summarize(AME_val = mean(AME, na.rm=TRUE)) %>% 
  as.data.frame()

avgGDPpc <- test_data %>% 
  dplyr::group_by(histperiod) %>% 
  dplyr::summarize(avg_GDP = mean(GDPpc, na.rm=TRUE))

RMSE_tab$avgGDPpc <- avgGDPpc$avg_GDP
RMSE_tab$share <- RMSE_tab$RMSE_val / RMSE_tab$avgGDPpc

AME_tab$avgGDPpc <- avgGDPpc$avg_GDP
AME_tab$share <- AME_tab$AME_val / AME_tab$avgGDPpc

# Store the full model results - SAVE COPY OF TEST DATA WITH FULL MODEL PREDICTIONS
test_data_full <- test_data  # Create a copy with full model predictions
RMSE_overall_fullmodel <- RMSE_overall
RMSE_tab_fullmodel <- RMSE_tab
AME_overall_fullmodel <- AME_overall
AME_tab_fullmodel <- AME_tab

# BASELINE TAKING ONLY FIXED EFFECTS
# Reset prediction columns
test_data$prediction <- NA
test_data$prediction_abs <- NA
training_data$prediction <- NA
training_data$prediction_abs <- NA

for(p in 1:5){
  training_data_sub <- subset(training_data, histperiod == p)
  test_data_sub <- subset(test_data, histperiod == p)
  
  # Only run update_GDPpc_t0 if that function exists and there's data
  if(exists("update_GDPpc_t0") && nrow(training_data_sub) > 0 && nrow(test_data_sub) > 0) {
    if(p == 2){
      test_data_sub <- update_GDPpc_t0(test_data_sub, test_data, labeled_data, 1500, "")
      training_data_sub <- update_GDPpc_t0(training_data_sub, training_data, labeled_data, 1500, "")
    } else if(p == 3){
      test_data_sub <- update_GDPpc_t0(test_data_sub, test_data, labeled_data, 1750, "")
      training_data_sub <- update_GDPpc_t0(training_data_sub, training_data, labeled_data, 1750, "")
    } else if(p == 4){
      test_data_sub <- update_GDPpc_t0(test_data_sub, test_data, labeled_data, 1850, "")
      training_data_sub <- update_GDPpc_t0(training_data_sub, training_data, labeled_data, 1850, "")
    } else if(p == 5){
      test_data_sub <- update_GDPpc_t0(test_data_sub, test_data, labeled_data, 1950, "")
      training_data_sub <- update_GDPpc_t0(training_data_sub, training_data, labeled_data, 1950, "")
    }
  }
  
  # Check if there's enough data to fit the model
  if(nrow(training_data_sub) > 0 && "GDPpc_t0" %in% colnames(training_data_sub)) {
    # Handle NAs in predictor variables
    training_data_sub$GDPpc_t0[is.na(training_data_sub$GDPpc_t0)] <- mean(training_data_sub$GDPpc_t0, na.rm=TRUE)
    
    # Ensure test data also has GDPpc_t0 and handle NAs
    if(!"GDPpc_t0" %in% colnames(test_data_sub)) {
      test_data_sub$GDPpc_t0 <- NA
    }
    test_data_sub$GDPpc_t0[is.na(test_data_sub$GDPpc_t0)] <- mean(training_data_sub$GDPpc_t0, na.rm=TRUE)
    
    # Fit fixed effects model
    tryCatch({
      # Check if UN_subregion exists, if not use a simpler model
      if("UN_subregion" %in% colnames(training_data_sub)) {
        model <- feols(
          as.formula("log10(GDPpc_safe) ~ GDPpc_t0 + as.factor(year) + as.factor(UN_subregion)"),
          data = training_data_sub
        )
      } else {
        model <- feols(
          as.formula("log10(GDPpc_safe) ~ GDPpc_t0 + as.factor(year)"),
          data = training_data_sub
        )
      }
      
      # Make predictions
      test_data_sub$prediction <- predict(model, newdata = test_data_sub)
      test_data_sub$prediction_abs <- 10^test_data_sub$prediction
      
      # Update test data by ID
      if("ID" %in% colnames(test_data_sub) && "ID" %in% colnames(test_data)) {
        for(i in 1:nrow(test_data_sub)) {
          row_id <- test_data_sub$ID[i]
          test_idx <- which(test_data$ID == row_id)
          if(length(test_idx) > 0) {
            test_data$prediction[test_idx] <- test_data_sub$prediction[i]
            test_data$prediction_abs[test_idx] <- test_data_sub$prediction_abs[i]
          }
        }
      }
      
      # Make predictions for training data
      training_data_sub$prediction <- predict(model, newdata = training_data_sub)
      training_data_sub$prediction_abs <- 10^training_data_sub$prediction
      
      # Update training data by ID
      if("ID" %in% colnames(training_data_sub) && "ID" %in% colnames(training_data)) {
        for(i in 1:nrow(training_data_sub)) {
          row_id <- training_data_sub$ID[i]
          train_idx <- which(training_data$ID == row_id)
          if(length(train_idx) > 0) {
            training_data$prediction[train_idx] <- training_data_sub$prediction[i]
            training_data$prediction_abs[train_idx] <- training_data_sub$prediction_abs[i]
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
test_data$RMSE <- (test_data$prediction_abs - test_data$GDPpc)^2
test_data$RMSE_logs <- (test_data$prediction - log10(test_data$GDPpc_safe))^2
test_data$AME <- abs(test_data$prediction_abs - test_data$GDPpc)

RMSE_overall <- test_data %>% dplyr::summarize(RMSE_val = sqrt(mean(RMSE, na.rm=TRUE)))
RMSE_overall$avgGDPpc <- test_data %>% dplyr::summarize(avg_GDP = mean(GDPpc, na.rm=TRUE)) %>% pull(avg_GDP)

AME_overall <- test_data %>% dplyr::summarize(AME_val = mean(AME, na.rm=TRUE))
AME_overall$avgGDPpc <- test_data %>% dplyr::summarize(avg_GDP = mean(GDPpc, na.rm=TRUE)) %>% pull(avg_GDP)

RMSE_tab <- test_data %>% 
  dplyr::group_by(histperiod) %>% 
  dplyr::summarize(RMSE_val = sqrt(mean(RMSE, na.rm=TRUE))) %>% 
  as.data.frame()

RMSE_logs_tab <- test_data %>% 
  dplyr::group_by(histperiod) %>% 
  dplyr::summarize(RMSE_logs_val = sqrt(mean(RMSE_logs, na.rm=TRUE))) %>% 
  as.data.frame()

AME_tab <- test_data %>% 
  dplyr::group_by(histperiod) %>% 
  dplyr::summarize(AME_val = mean(AME, na.rm=TRUE)) %>% 
  as.data.frame()

avgGDPpc <- test_data %>% 
  dplyr::group_by(histperiod) %>% 
  dplyr::summarize(avg_GDP = mean(GDPpc, na.rm=TRUE))

RMSE_tab$avgGDPpc <- avgGDPpc$avg_GDP
RMSE_tab$share <- RMSE_tab$RMSE_val / RMSE_tab$avgGDPpc

AME_tab$avgGDPpc <- avgGDPpc$avg_GDP
AME_tab$share <- AME_tab$AME_val / AME_tab$avgGDPpc

# First determine the common limits needed for both plots
y_min <- min(min(test_data$GDPpc, na.rm=TRUE), min(test_data_full$GDPpc, na.rm=TRUE))
y_max <- max(max(test_data$GDPpc, na.rm=TRUE), max(test_data_full$GDPpc, na.rm=TRUE))
x_min <- min(min(test_data$prediction_abs, na.rm=TRUE), min(test_data_full$prediction_abs, na.rm=TRUE))
x_max <- max(max(test_data$prediction_abs, na.rm=TRUE), max(test_data_full$prediction_abs, na.rm=TRUE))

# Create baseline model visualization with consistent axis limits
baselinemodel <- ggplot(test_data, aes(y=GDPpc, x=prediction_abs)) + 
  stat_poly_eq(formula = y~x, data=test_data, aes(label = after_stat(rr.label)), parse = TRUE) +
  geom_smooth(method="lm", se=FALSE, linetype="dashed", color="grey") + 
  geom_abline(slope = 1, intercept = 0, color = "grey") + 
  geom_point(size=1.5, color = "darkorange1") + 
  geom_text(label=test_data$ID2, check_overlap=TRUE, size=2.5, nudge_y = -0.02) +
  scale_x_continuous(trans='log10', labels = scales::comma, limits = c(x_min, x_max)) + 
  scale_y_continuous(trans='log10', labels = scales::comma, limits = c(y_min, y_max)) + 
  theme_light() + 
  labs(x="Prediction (log scale)", y="GDP per capita (log scale)", title = "Baseline model")

# Update full model visualization with the same axis limits - USE test_data_full
fullmodel <- ggplot(test_data_full, aes(y=GDPpc, x=prediction_abs)) + 
  stat_poly_eq(formula = y~x, data=test_data_full, aes(label = after_stat(rr.label)), parse = TRUE) +
  geom_smooth(method="lm", se=FALSE, linetype="dashed", color="grey") + 
  geom_abline(slope = 1, intercept = 0, color = "grey") + 
  geom_point(size=1.5, color = "darkorange1") + 
  geom_text(label=test_data_full$ID2, check_overlap=TRUE, size=2.5, nudge_y = -0.02) +
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

# Compare models with visualizations - Use original column names for RMSE_tab_fullmodel
RMSE_tab_fullmodel$group <- "Full model"
RMSE_tab_fullmodel$century <- c("1300-1500", "1550-1750", "1800-1850", "1900-1950", "2000")

RMSE_tab$group <- "Naive model"
RMSE_tab$century <- c("1300-1500", "1550-1750", "1800-1850", "1900-1950", "2000")

# Ensure column names match for rbind
names(RMSE_tab)[names(RMSE_tab) == "RMSE_val"] <- names(RMSE_tab_fullmodel)[which(grepl("RMSE", names(RMSE_tab_fullmodel)))[1]]
RMSE_tab_full <- rbind(RMSE_tab, RMSE_tab_fullmodel)

# Create RMSE comparison plot
rmse_plot <- ggplot(RMSE_tab_full, aes(x=share*100, y=as.factor(histperiod), fill=as.factor(group))) + 
  geom_bar(stat="identity", position="dodge", width=0.7) + 
  theme_light() +
  labs(y="Century", x="RMSE, % of average GDP per capita", fill="Model") +
  theme(legend.title=element_blank())

# Print RMSE comparison
print(rmse_plot)

# Save RMSE comparison plot
if(exists("version")) {
  ggsave(paste0("./genfiles_", version, "/figures/RMSE_comparison.svg"), plot=rmse_plot, width=8, height=4)
}

# Optional: Extract and print variable importance from the model
if(!is.null(rf_model$finalModel) && "ranger" %in% class(rf_model$finalModel)) {
  # Extract variable importance if available
  var_imp <- ranger::importance(rf_model$finalModel)
  
  # Sort and format for plotting
  var_imp_df <- data.frame(
    Variable = names(var_imp),
    Importance = as.numeric(var_imp)
  ) %>%
    arrange(desc(Importance)) %>%
    head(20)  # Top 20 most important variables
  
  # Create variable importance plot
  var_imp_plot <- ggplot(var_imp_df, aes(x=reorder(Variable, Importance), y=Importance)) +
    geom_col(fill="steelblue") +
    coord_flip() +
    theme_light() +
    labs(x="Variable", y="Importance", title="Top Variable Importance in Random Forest Model")
  
  # Print variable importance plot
  print(var_imp_plot)
  
  # Save variable importance plot
  if(exists("version")) {
    ggsave(paste0("./genfiles_", version, "/figures/variable_importance.svg"), plot=var_imp_plot, width=8, height=6)
  }
}

# Print summary of the best tuning parameters
cat("\nBest tuning parameters:\n")
print(rf_model$bestTune)

# Print performance summary
cat("\nModel performance metrics:\n")
cat("Full model RMSE (overall):", RMSE_overall_fullmodel$RMSE_val, "\n")
cat("Naive model RMSE (overall):", RMSE_overall$RMSE_val, "\n")
cat("Improvement (%):", round((1 - RMSE_overall_fullmodel$RMSE_val/RMSE_overall$RMSE_val) * 100, 2), "%\n")

# Save performance results to CSV
if(exists("version")) {
  performance_summary <- data.frame(
    Model = c("Full model", "Naive model"),
    RMSE_overall = c(RMSE_overall_fullmodel$RMSE_val, RMSE_overall$RMSE_val),
    AME_overall = c(AME_overall_fullmodel$AME_val, AME_overall$AME_val)
  )
  
  write.csv(performance_summary, 
            paste0("./genfiles_", version, "/tables/model_performance_summary.csv"),
            row.names = FALSE)
  
  # Also save detailed by-period metrics
  RMSE_tab_full$century <- factor(RMSE_tab_full$century, 
                                  levels = c("1300-1500", "1550-1750", "1800-1850", "1900-1950", "2000"))
  
  write.csv(RMSE_tab_full,
            paste0("./genfiles_", version, "/tables/RMSE_by_period.csv"),
            row.names = FALSE)
}
