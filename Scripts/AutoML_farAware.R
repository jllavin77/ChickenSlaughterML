# =====================================================================
# OPTIMIZED SCRIPT: AutoML with Blind Validation 
# =====================================================================

library(ggplot2)
library(caret)
library(dplyr)
library(magrittr)
library(automl)

setwd("C:/R_data/Pavos_AWARE/Transformadas/AML_0")

# 1. Load and initial data cleaning
PA_table <- read.delim(file.choose(), sep = "\t", header = TRUE, row.names = 1)
PA_table$X <- NULL
colnames(PA_table) <- gsub("X.", "", colnames(PA_table))

# Ensure target variable format
PA_table$categoria <- as.factor(PA_table$categoria)

# 2. Real and robust split (90% Train / 10% Blind Test)
# Note: If you have 'farm_id', a group-based split would be ideal here 
# to ensure that test farms are entirely unseen.
set.seed(123)
train_indices <- sample(1:nrow(PA_table), 0.9 * nrow(PA_table))
train_data <- PA_table[train_indices, ]
test_data  <- PA_table[-train_indices, ]

# 3. Exclusive preparation of matrices using the training set
xmat_train <- train_data %>% select(-categoria)
ymat_train <- as.numeric(train_data$categoria)

# 4. Optimized training (using definitive tuning)
set.seed(123)
amlmodel <- automl_train_manual(
  Xref = xmat_train,
  Yref = ymat_train,
  hpar = list(
    learningrate = 0.01,
    minibatchsize = 4,      # 2^2
    numiterations = 60
  )
)

# 5. Save the trained model securely
saveRDS(amlmodel, file = "amlmodel_optimized.rda")

# 6. REAL EVALUATION ON THE BLIND TEST SET (Reserved 10%)
xmat_test <- test_data %>% select(-categoria)
ymat_test <- as.numeric(test_data$categoria)

# Prediction on entirely new data
raw_preds <- automl_predict(model = amlmodel, X = xmat_test)

# Transform prediction into 3-class factors (according to your logic)
pred_classes <- ifelse(raw_preds > 2.5, 3, ifelse(raw_preds > 1.5, 2, 1)) %>% as.factor()
true_classes <- as.factor(ymat_test)

# 7. Generate Real Confusion Matrix (External)
cm_external <- confusionMatrix(pred_classes, true_classes, mode = "everything")
print(cm_external)

# Export results from the blind evaluation
res_external <- data.frame(Actual = true_classes, Predicted = pred_classes)
write.table(res_external, file = "External_Test_Predictions.tsv", sep = "\t", row.names = FALSE)
