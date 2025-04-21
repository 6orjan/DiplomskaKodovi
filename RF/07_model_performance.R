# First, ensure all necessary packages are loaded
library(dplyr)
library(stringr)
library(caret)
library(randomForest)
library(ggplot2)
library(ggpmisc)  # For stat_poly_eq
library(cowplot)  # For plot_grid
library(fixest)   # For feols

# Custom vlookup function if it's not already defined
if(!exists("vlookup")) {
  vlookup <- function(lookup_value, table, lookup_column, result_column) {
    result <- table[match(lookup_value, table[[lookup_column]]), result_column]
    return(result)
  }
}

# Define update_GDPpc_t0 function if it's not already defined
if(!exists("update_GDPpc_t0")) {
  update_GDPpc_t0 <- function(data_subset, full_data, labeled_data, year_value, column_prefix = "") {
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

# Define proper cross-validation control
cctrl1 <- trainControl(
  method = "cv",
  number = 5, #how many times the model will run
  verboseIter = TRUE,  # Set to TRUE to see progress
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

# Train the RandomForest model directly with dataframes 
# (avoid matrix conversion which can cause issues)
set.seed(123)  # For reproducibility
rf_model <- train(
  x = training_data[, predictors],
  y = log10(training_data$GDPpc),  # Transform GDP to log scale for prediction
  method = "rf",  # Random Forest method
  trControl = cctrl1,
  tuneGrid = expand.grid(mtry = seq(3, 10, by = 1)),  # Adjust mtry range for Random Forest
  ntree = 500,  # Use 500 trees in the Random Forest
  importance = TRUE  # Keep track of variable importance
)

# Print model details to verify it trained correctly
print(rf_model)

# Make predictions using the trained model
test_data$prediction <- predict(rf_model, newdata = test_data[, predictors])
test_data$prediction_abs <- 10^test_data$prediction

# Compute RMSE and AME
test_data$RMSE <- (test_data$prediction_abs - test_data$GDPpc)^2
test_data$RMSE_logs <- (test_data$prediction - log10(test_data$GDPpc))^2
test_data$AME <- abs(test_data$prediction_abs - test_data$GDPpc)

# Calculate performance metrics
RMSE_overall <- test_data %>% dplyr::summarize(sqrt(mean(RMSE, na.rm=TRUE)))
RMSE_overall$avgGDPpc <- test_data %>% dplyr::summarize(mean(GDPpc, na.rm=TRUE))

AME_overall <- test_data %>% dplyr::summarize(mean(AME, na.rm=TRUE))
AME_overall$avgGDPpc <- test_data %>% dplyr::summarize(mean(GDPpc, na.rm=TRUE))

RMSE_tab <- test_data %>% dplyr::group_by(histperiod) %>% dplyr::summarize(sqrt(mean(RMSE, na.rm=TRUE))) %>% as.data.frame()
RMSE_logs_tab <- test_data %>% dplyr::group_by(histperiod) %>% dplyr::summarize(sqrt(mean(RMSE_logs, na.rm=TRUE))) %>% as.data.frame()

AME_tab <- test_data %>% dplyr::group_by(histperiod) %>% dplyr::summarize(mean(AME, na.rm=TRUE)) %>% as.data.frame()

avgGDPpc <- test_data %>% dplyr::group_by(histperiod) %>% dplyr::summarize(mean(GDPpc, na.rm=TRUE))

RMSE_tab$avgGDPpc <- avgGDPpc$`mean(GDPpc, na.rm = TRUE)`
RMSE_tab$share <- RMSE_tab$`sqrt(mean(RMSE, na.rm = TRUE))` / RMSE_tab$avgGDPpc

AME_tab$avgGDPpc <- avgGDPpc$`mean(GDPpc, na.rm = TRUE)`
AME_tab$share <- AME_tab$`mean(AME, na.rm = TRUE)` / AME_tab$avgGDPpc

# Visualization
fullmodel <- ggplot(test_data, aes(y=GDPpc, x=prediction_abs)) + stat_poly_eq(formula = y~x, data=test_data, aes(label = after_stat(rr.label)), parse = TRUE) +
  geom_smooth(method="lm", se=FALSE, linetype="dashed", color="grey") + 
  geom_abline(slope = 1, intercept = 0, color = "grey") + 
  geom_point(size=1.5, color = "darkorange1") + 
  geom_text(label=test_data$ID2, check_overlap=TRUE, size=2.5, nudge_y = -0.02) +
  scale_x_continuous(trans='log10') + 
  scale_y_continuous(trans='log10') + 
  theme_light() + 
  labs(x="prediction", y="log(GDP per capita)", title = "Full model")

# Save the first visualization
if(exists("version")) {
  ggsave(paste0("./genfiles_", version, "/figures/Fig2_AB.svg"), width = 8, height = 4)
}

# Store the full model results
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
  
  # Only run update_GDPpc_t0 if that function exists
  if(exists("update_GDPpc_t0")) {
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
    
    # Fit fixed effects model
    tryCatch({
      model <- feols(as.formula("log10(GDPpc) ~ GDPpc_t0 + as.factor(year) + as.factor(UN_subregion)"),
                     data = training_data_sub)
      
      # Make predictions
      test_data_sub$prediction <- predict(model, newdata = test_data_sub)
      test_data_sub$prediction_abs <- 10^test_data_sub$prediction
      
      # Update test data
      for(i in 1:nrow(test_data_sub)) {
        row_id <- test_data_sub$ID[i]
        test_idx <- which(test_data$ID == row_id)
        if(length(test_idx) > 0) {
          test_data$prediction[test_idx] <- test_data_sub$prediction[i]
          test_data$prediction_abs[test_idx] <- test_data_sub$prediction_abs[i]
        }
      }
      
      # Make predictions for training data
      training_data_sub$prediction <- predict(model, newdata = training_data_sub)
      training_data_sub$prediction_abs <- 10^training_data_sub$prediction
      
      # Update training data
      for(i in 1:nrow(training_data_sub)) {
        row_id <- training_data_sub$ID[i]
        train_idx <- which(training_data$ID == row_id)
        if(length(train_idx) > 0) {
          training_data$prediction[train_idx] <- training_data_sub$prediction[i]
          training_data$prediction_abs[train_idx] <- training_data_sub$prediction_abs[i]
        }
      }
    }, error = function(e) {
      cat("Error in model for period", p, ":", e$message, "\n")
    })
  } else {
    cat("Skipping period", p, "due to insufficient data\n")
  }
}

# Calculate performance metrics for baseline model
test_data$RMSE <- (test_data$prediction_abs - test_data$GDPpc)^2
test_data$RMSE_logs <- (test_data$prediction - log10(test_data$GDPpc))^2
test_data$AME <- abs(test_data$prediction_abs - test_data$GDPpc)

RMSE_overall <- test_data %>% dplyr::summarize(sqrt(mean(RMSE, na.rm=TRUE)))
RMSE_overall$avgGDPpc <- test_data %>% dplyr::summarize(mean(GDPpc, na.rm=TRUE))

AME_overall <- test_data %>% dplyr::summarize(mean(AME, na.rm=TRUE))
AME_overall$avgGDPpc <- test_data %>% dplyr::summarize(mean(GDPpc, na.rm=TRUE))

RMSE_tab <- test_data %>% dplyr::group_by(histperiod) %>% dplyr::summarize(sqrt(mean(RMSE, na.rm=TRUE))) %>% as.data.frame()
RMSE_logs_tab <- test_data %>% dplyr::group_by(histperiod) %>% dplyr::summarize(sqrt(mean(RMSE_logs, na.rm=TRUE))) %>% as.data.frame()

AME_tab <- test_data %>% dplyr::group_by(histperiod) %>% dplyr::summarize(mean(AME, na.rm=TRUE)) %>% as.data.frame()

avgGDPpc <- test_data %>% dplyr::group_by(histperiod) %>% dplyr::summarize(mean(GDPpc, na.rm=TRUE))

RMSE_tab$avgGDPpc <- avgGDPpc$`mean(GDPpc, na.rm = TRUE)`
RMSE_tab$share <- RMSE_tab$`sqrt(mean(RMSE, na.rm = TRUE))` / RMSE_tab$avgGDPpc

AME_tab$avgGDPpc <- avgGDPpc$`mean(GDPpc, na.rm = TRUE)`
AME_tab$share <- AME_tab$`mean(AME, na.rm = TRUE)` / AME_tab$avgGDPpc

# Create baseline model visualization
baselinemodel <- ggplot(test_data, aes(y=GDPpc, x=prediction_abs)) + 
  stat_poly_eq(formula = y~x, data=test_data, aes(label = after_stat(rr.label)), parse = TRUE) +
  geom_smooth(method="lm", se=FALSE, linetype="dashed", color="grey") + 
  geom_abline(slope = 1, intercept = 0, color = "grey") + 
  geom_point(size=1.5, color = "darkorange1") + 
  geom_text(label=test_data$ID2, check_overlap=TRUE, size=2.5, nudge_y = -0.02) +
  scale_x_continuous(trans='log10') + 
  scale_y_continuous(trans='log10') + 
  theme_light() + 
  labs(x="prediction", y="log(GDP per capita)", title = "Baseline model")

# Compare models with visualizations
RMSE_tab_fullmodel$group <- "Full model"
RMSE_tab_fullmodel$century <- c("1300-1500", "1550-1750", "1800-1850", "1900-1950", "2000")

RMSE_tab$group <- "Naive model"
RMSE_tab$century <- c("1300-1500", "1550-1750", "1800-1850", "1900-1950", "2000")
RMSE_tab_full <- rbind(RMSE_tab, RMSE_tab_fullmodel)

# Create RMSE comparison plot
rmse_plot <- ggplot(RMSE_tab_full, aes(x=share*100, y=as.factor(histperiod), fill=as.factor(group))) + 
  geom_bar(stat="identity", position="dodge", width=0.7) + 
  theme_light() +
  labs(y="Century", x="RMSE, % of average GDP per capita", fill="Model") +
  theme(legend.title=element_blank())

# Create AME comparison
AME_tab_fullmodel$group <- "Full model"
AME_tab_fullmodel$century <- c("1300-1500", "1550-1750", "1800-1850", "1900-1950", "2000")

AME_tab$group <- "Naive model"
AME_tab$century <- c("1300-1500", "1550-1750", "1800-1850", "1900-1950", "2000")

AME_tab_full <- rbind(AME_tab, AME_tab_fullmodel)

# Create AME comparison plot
ame_plot <- ggplot(AME_tab_full, aes(x=share*100, y=as.factor(histperiod), fill=as.factor(group))) + 
  geom_bar(stat="identity", position="dodge", width=0.7) + 
  theme_light() +
  labs(y="Century", x="AME, % of average GDP per capita", fill="Model") +
  theme(legend.title=element_blank())

# Compare baseline and full models
model_comparison <- plot_grid(baselinemodel, fullmodel, labels=c("A", "B"), ncol=2)

# Save the final figure
if(exists("version")) {
  ggsave(paste0("./genfiles_", version, "/figures/Fig2_AB.svg"), plot=model_comparison, width=8, height=4)
}
