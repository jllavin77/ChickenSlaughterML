# =====================================================================
# SCRIPT 2: MULTI-MODEL TRAINING, LEAKAGE CONTROL
# & Extended Metrics
# =====================================================================

library(lattice)
library(munsell)
library(ggplot2)
library(caret)
library(dplyr)
library(magrittr)

setwd("Path_to_WorkDir")

# Load data (mandatory renaming variables 'farm_id' and category as 'CATEG_LUMAGORRI' in the input table)
dataset <- read.delim(file.choose(), sep = "\t", header = TRUE, row.names = 1)
head(dataset)

# get factors
dataset$CATEG_LUMAGORRI <- as.factor(dataset$CATEG_LUMAGORRI)
dataset$farm_id <- as.factor(dataset$farm_id) # mandatory for group control

PA2 <- dataset

# =====================================================================
# Training Config (DATA LEAKAGE CONTROL)
# =====================================================================
num_granjas <- length(unique(PA2$farm_id))
folds_agrupados <- groupKFold(PA2$farm_id, k = min(10, num_granjas))

# Control unificado con validación cruzada por granjas
control_agrupado <- trainControl(
  method = "cv", 
  index = folds_agrupados, 
  savePredictions = "final",
  classProbs = TRUE,
  summaryFunction = multiClassSummary 
)

metric <- "Mean_Balanced_Accuracy"   

# =====================================================================
# MODEL TRAINING (RF, CART, KNN, SVM)
# =====================================================================

# 1. Random Forest
set.seed(7)
fit.rf <- train(CATEG_LUMAGORRI ~ ., data = PA2, method = "rf", metric = metric, trControl = control_agrupado, na.action = na.pass)
saveRDS(fit.rf, file = "model_VFDB_rf_grouped.rda")

# 2. CART
set.seed(7)
fit.cart <- train(CATEG_LUMAGORRI ~ ., data = PA2, method = "rpart", metric = metric, trControl = control_agrupado, na.action = na.pass)
saveRDS(fit.cart, file = "model_VFDB_cart_grouped.rda")

# 3. kNN
set.seed(7)
fit.knn <- train(CATEG_LUMAGORRI ~ ., data = PA2, method = "knn", metric = metric, trControl = control_agrupado, na.action = na.pass)
saveRDS(fit.knn, file = "model_VFDB_knn_grouped.rda")

# 4. SVM Radial
set.seed(7)
fit.svm <- train(CATEG_LUMAGORRI ~ ., data = PA2, method = "svmRadial", metric = metric, trControl = control_agrupado, na.action = na.pass)
saveRDS(fit.svm, file = "model_VFDB_svm_grouped.rda")

# =====================================================================
# SuMMARY and MODEL COMPARISON
# =====================================================================
sink("Models_Accuracy_Grouped.txt")
results <- resamples(list(cart = fit.cart, knn = fit.knn, svm = fit.svm, rf = fit.rf))
summary(results)
print("_______________________________________")
print(fit.cart)
print("_______________________________________")
print(fit.knn)
print("_______________________________________")
print(fit.svm)
print("_______________________________________")
print(fit.rf)
sink()

pdf('Models_Accuracy_Comparison.pdf')
dotplot(results)
dotplot(results, metric = "Mean_Balanced_Accuracy")
dotplot(results, metric = "Accuracy")
dotplot(results, metric = "Kappa")
dev.off()

# Per individual model fold:
fit.rf$resample    # data.frame con una fila por fold y todas las columnas de métricas

#  Métrics per class 
cm_rf <- confusionMatrix(fit.rf$pred$pred, fit.rf$pred$obs, mode = "everything")  # sensibilidad/especificidad por categoría
sink("Metricas_CV_RandomForest.txt")
print(cm_rf)
sink()

cm_cart<- confusionMatrix(fit.cart$pred$pred, fit.cart$pred$obs, mode = "everything")
sink("Metricas_CV_CART.txt")
print(cm_cart)
sink()

cm_knn<- confusionMatrix(fit.knn$pred$pred, fit.knn$pred$obs, mode = "everything")
sink("Metricas_CV_KNN.txt")
print(cm_knn)
sink()

cm_svm <- confusionMatrix(fit.svm$pred$pred, fit.svm$pred$obs, mode = "everything")
sink("Metricas_CV_SVM.txt")
print(cm_svm)
sink()

# =====================================================================
# PREDICTIONS & EXTENDED EVALUATION WITH NEWLY COLLECTED DATA
# =====================================================================
# Upload dataset for external validation
dataTest_complete <- read.delim(file.choose(), sep = "\t", header = TRUE, row.names = 1)
dataTest_complete$CATEG_LUMAGORRI <- as.factor(dataTest_complete$CATEG_LUMAGORRI)

# --- 1. Random Forest ---
predt_rf <- predict(fit.rf, dataTest_complete)
cmt_rf <- confusionMatrix(predt_rf, dataTest_complete$CATEG_LUMAGORRI, mode = "everything")
sink("Metricas_Detalladas_RF_Test.txt")
print(cmt_rf)
sink()
resultadoTest <- cbind(dataTest_complete, predictionTest = predt_rf)
write.table(resultadoTest, file = "RF_predictions_Test.tsv", sep = "\t", row.names = FALSE)

# --- 2. SVM ---
predt_svm <- predict(fit.svm, dataTest_complete)
cmt_svm <- confusionMatrix(predt_svm, dataTest_complete$CATEG_LUMAGORRI, mode = "everything")
sink("Metricas_Detalladas_SVM_Test.txt")
print(cmt_svm)
sink()
resultadoTest2 <- cbind(dataTest_complete, predictionTest2 = predt_svm)
write.table(resultadoTest2, file = "SVM_predictions_Test.tsv", sep = "\t", row.names = FALSE)

# --- 3. kNN ---
predt_knn <- predict(fit.knn, dataTest_complete)
cmt_knn <- confusionMatrix(predt_knn, dataTest_complete$CATEG_LUMAGORRI, mode = "everything")
sink("Metricas_Detalladas_KNN_Test.txt")
print(cmt_knn)
sink()
resultadoTest3 <- cbind(dataTest_complete, predictionTest3 = predt_knn)
write.table(resultadoTest3, file = "KNN_predictions_Test.tsv", sep = "\t", row.names = FALSE)

# --- 4. CART ---
predt_cart <- predict(fit.cart, dataTest_complete)
cmt_cart <- confusionMatrix(predt_cart, dataTest_complete$CATEG_LUMAGORRI, mode = "everything")
sink("Metricas_Detalladas_CART_Test.txt")
print(cmt_cart)
sink()
resultadoTest4 <- cbind(dataTest_complete, predictionTest4 = predt_cart)
write.table(resultadoTest4, file = "CART_predictions_Test.tsv", sep = "\t", row.names = FALSE)

# =====================================================================
# MULTI_PREDICTION TABLE SUMMARY
# =====================================================================
resultadoTest_fix  <- resultadoTest[c("CATEG_LUMAGORRI", "predictionTest")] %>% rename(RFpred = predictionTest)
resultadoTest_fix2 <- resultadoTest2[c("CATEG_LUMAGORRI", "predictionTest2")] %>% rename(SVMpred = predictionTest2)
resultadoTest_fix3 <- resultadoTest3[c("CATEG_LUMAGORRI", "predictionTest3")] %>% rename(KNNpred = predictionTest3)
resultadoTest_fix4 <- resultadoTest4[c("CATEG_LUMAGORRI", "predictionTest4")] %>% rename(CARTpred = predictionTest4)

de  <- merge(resultadoTest_fix, resultadoTest_fix2, by = "row.names")
de2 <- merge(de, resultadoTest_fix3, by.x = "Row.names", by = "row.names")
de3 <- merge(de2, resultadoTest_fix4, by.x = "Row.names", by = "row.names")

de3f <- de3[c("Row.names", "CATEG_LUMAGORRI.x", "CARTpred", "RFpred", "KNNpred", "SVMpred")]
write.table(de3f, file = "MULTI_Predictions_with_names.tab", sep = "\t", row.names = FALSE)