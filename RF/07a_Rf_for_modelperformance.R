test_data_preds <- test_data
training_data_preds <- training_data
test_data_preds$prediction_abs <- NA
test_data_preds$prediction <- NA
training_data_preds$prediction <- NA
training_data_preds$prediction_abs <- NA

update_GDPpc_t0 <- function(data_subset, data, labeled_data, year, column_name_preds) {
  for(m in 1:nrow(data_subset)){
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
  }
  return(data_subset)
}

for(selperiod in 1:5){
  startcolumn <- 14
  training_data_sub <- subset(training_data, histperiod == selperiod)
  test_data_sub <- subset(test_data, histperiod == selperiod)
  
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
  
  cctrl1 <- trainControl(method = "cv", number = min(k, nrow(training_data_sub)))
  training_data_sub <- subset(training_data_sub, complete.cases(training_data_sub[, startcolumn:ncol(training_data_sub)]))
  
  cols <- match("year1300", colnames(training_data_sub)):match("year2000", colnames(training_data_sub))
  training_data_sub[ , cols] <- apply(training_data_sub[ , cols], 2, as.numeric)
  
  rf_model <- train(
    x = training_data_sub[, startcolumn:ncol(training_data_sub)],
    y = log10(training_data_sub$GDPpc),
    method = "rf",
    trControl = cctrl1,
    tuneGrid = training_grid_modelperformance,
    metric = "MAE"
  )
  
  test_data_sub[ , cols] <- apply(test_data_sub[ , cols], 2, as.numeric)
  test_data_sub$prediction <- predict(rf_model, newdata = test_data_sub[, startcolumn:ncol(test_data_sub)])
  test_data_sub$prediction_abs <- 10^test_data_sub$prediction
  
  test_data_preds$prediction <- ifelse(is.na(test_data_preds$prediction), vlookup(test_data_preds$ID, test_data_sub, lookup_column = "ID", result_column = "prediction"), test_data_preds$prediction)
  test_data_preds$prediction_abs <- ifelse(is.na(test_data_preds$prediction_abs), vlookup(test_data_preds$ID, test_data_sub, lookup_column = "ID", result_column = "prediction_abs"), test_data_preds$prediction_abs)
  
  if(rescale_regions == "Y"){
    test_data_preds$country_0_period <- paste(test_data_preds$country_0, test_data_preds$period, sep="_")
    countries_0 <- unique(test_data_preds$country_0_period)
    test_data_preds_countrylevel <- subset(test_data_preds, country == country_0)
    test_data_preds_regional <- subset(test_data_preds, country != country_0)
    
    test_data_preds_regional_rescaled <- do.call(rbind, lapply(countries_0, function(c){
      refGDPpc <- ifelse(is.na(vlookup(c, test_data_sub, lookup_column = "ID", result_column = "GDPpc")), 
                         subset(test_data_preds_countrylevel, country_0_period == c)$prediction_abs,
                         vlookup(c, test_data_sub, lookup_column = "ID", result_column = "GDPpc"))
      
      regionalGDPpc <- subset(test_data_preds_regional, country_0_period == c)
      if(nrow(regionalGDPpc) > 0 & nrow(subset(test_data_preds_countrylevel, country_0_period == c)) > 0){
        if(normalization_option == "log"){
          regionalGDPpc$prediction_abs <- regionalGDPpc$prediction_abs * (refGDPpc / weighted.mean(regionalGDPpc$prediction_abs, w = 10^regionalGDPpc$births + 10^regionalGDPpc$deaths))
        }
        if(normalization_option == "ihs"){
          regionalGDPpc$prediction_abs <- regionalGDPpc$prediction_abs * (refGDPpc / weighted.mean(regionalGDPpc$prediction_abs, w = asinh(regionalGDPpc$births) + asinh(regionalGDPpc$deaths)))
        }
      }
      return(regionalGDPpc)
    }))
    
    test_data_preds <- rbind(test_data_preds_countrylevel, test_data_preds_regional_rescaled)
    test_data_preds$country_0_period <- NULL
  }
  
  training_data_sub$prediction <- predict(rf_model, newdata = training_data_sub[, startcolumn:ncol(training_data_sub)])
  training_data_sub$prediction_abs <- 10^training_data_sub$prediction
  training_data_preds$prediction <- ifelse(is.na(training_data_preds$prediction), vlookup(training_data_preds$ID, training_data_sub, lookup_column = "ID", result_column = "prediction"), training_data_preds$prediction)
  training_data_preds$prediction_abs <- ifelse(is.na(training_data_preds$prediction_abs), vlookup(training_data_preds$ID, training_data_sub, lookup_column = "ID", result_column = "prediction_abs"), training_data_preds$prediction_abs)
  
  if(rescale_regions == "Y"){
    training_data_preds$country_0_period <- paste(training_data_preds$country_0, training_data_preds$period, sep="_")
    countries_0 <- unique(training_data_preds$country_0_period)
    training_data_preds_countrylevel <- subset(training_data_preds, country == country_0)
    training_data_preds_regional <- subset(training_data_preds, country != country_0)
    
    training_data_preds_regional_rescaled <- do.call(rbind, lapply(countries_0, function(c){
      refGDPpc <- ifelse(is.na(vlookup(c, training_data_sub, lookup_column = "ID", result_column = "GDPpc")), 
                         subset(training_data_preds_countrylevel, country_0_period == c)$prediction_abs,
                         vlookup(c, training_data_sub, lookup_column = "ID", result_column = "GDPpc"))
      
      regionalGDPpc <- subset(training_data_preds_regional, country_0_period == c)
      if(nrow(regionalGDPpc) > 0 & nrow(subset(training_data_preds_countrylevel, country_0_period == c)) > 0){
        if(normalization_option == "log"){
          regionalGDPpc$prediction_abs <- regionalGDPpc$prediction_abs * (refGDPpc / weighted.mean(regionalGDPpc$prediction_abs, w = 10^regionalGDPpc$births + 10^regionalGDPpc$deaths))
        }
        if(normalization_option == "ihs"){
          regionalGDPpc$prediction_abs <- regionalGDPpc$prediction_abs * (refGDPpc / weighted.mean(regionalGDPpc$prediction_abs, w = asinh(regionalGDPpc$births) + asinh(regionalGDPpc$deaths)))
        }
      }
      return(regionalGDPpc)
    }))
    
    training_data_preds <- rbind(training_data_preds_countrylevel, training_data_preds_regional_rescaled)
    training_data_preds$country_0_period <- NULL
  }
}

test_data <- test_data_preds
