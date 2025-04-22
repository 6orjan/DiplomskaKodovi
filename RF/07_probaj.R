# Store full model predictions in dedicated columns
test_data$prediction_full <- predict(rf_model, newdata = test_data[, predictors])
test_data$prediction_abs_full <- 10^test_data$prediction_full

# Later store baseline model predictions in different columns
test_data$prediction_base <- test_data$prediction
test_data$prediction_abs_base <- test_data$prediction_abs

# Create plots using the specific columns
fullmodel <- ggplot(test_data, aes(y=GDPpc, x=prediction_abs_full)) +
  # rest of plotting code

baselinemodel <- ggplot(test_data, aes(y=GDPpc, x=prediction_abs_base)) +
  # rest of plotting code
