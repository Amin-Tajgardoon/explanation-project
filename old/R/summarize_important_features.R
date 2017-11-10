

###### RF_TP FEATURES #####################

### create dataframe with columns: feature, supporting_counts, opposing_counts, weight

rf_tp_w = read.csv(
  "output/explain_feature_weights/feature_weights/rf_tp_feature_weights.csv",
  check.names = FALSE
)

supportive_counts = apply(rf_tp_w[, 2:ncol(rf_tp_w)], 2, function(x)
  sum(x > 0, na.rm = TRUE))

opposing_counts = apply(rf_tp_w[, 2:ncol(rf_tp_w)], 2, function(x)
  sum(x < 0, na.rm = TRUE))

all_counts = data.frame(cbind(supportive_counts, opposing_counts),
                        row.names = names(supportive_counts))
avrg_weight = apply(rf_tp_w[, 2:ncol(rf_tp_w)], 2, function(x)
  mean(x, na.rm = TRUE))

rf_tp_features = merge(all_counts,
                       as.data.frame(avrg_weight),
                       by = "row.names",
                       all = TRUE)

write.csv(x = rf_tp_features[order(rf_tp_features$avrg_weight, decreasing = TRUE), ], 
          file = "output/explain_feature_weights/rf_tp_features.csv", row.names = FALSE)


### Extract top 10 features based on absolute weights

rf_tpw.o = rf_tp_features[order(abs(rf_tp_features$avrg_weight), decreasing = TRUE),]

write.csv(x = rf_tpw.o[1:10,][order(rf_tpw.o$avrg_weight[1:10], decreasing = T),],
          file = "output/explain_feature_weights/rf_tp_top_10_absolute.csv",
          row.names = F)




###### RF_TN FEATURES #####################


### create dataframe with columns: feature, supporting_counts, opposing_counts, weight


rf_tn_w = read.csv(
  "output/explain_feature_weights/feature_weights/rf_tn_feature_weights.csv",
  check.names = FALSE
)

supportive_counts = apply(rf_tn_w[, 2:ncol(rf_tn_w)], 2, function(x)
  sum(-x < 0, na.rm = TRUE))

opposing_counts = apply(rf_tn_w[, 2:ncol(rf_tn_w)], 2, function(x)
  sum(-x > 0, na.rm = TRUE))

all_counts = data.frame(cbind(supportive_counts, opposing_counts),
                        row.names = names(supportive_counts))
avrg_weight = apply(rf_tn_w[, 2:ncol(rf_tn_w)], 2, function(x)
  mean(-x, na.rm = TRUE))

rf_tn_features = merge(all_counts,
                       as.data.frame(avrg_weight),
                       by = "row.names",
                       all = TRUE)

write.csv(x = rf_tn_features[order(rf_tn_features$avrg_weight, decreasing = TRUE), ], 
          file = "output/explain_feature_weights/rf_tn_features.csv", row.names = FALSE)

### Extract top 10 features based on absolute weights

rf_tnw.o = rf_tn_features[order(abs(rf_tn_features$avrg_weight), decreasing = TRUE),]

write.csv(x = rf_tnw.o[1:10,][order(rf_tnw.o$avrg_weight[1:10], decreasing = F),],
          file = "output/explain_feature_weights/rf_tn_top_10_absolute.csv",
          row.names = F)


