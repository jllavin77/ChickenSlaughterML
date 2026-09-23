# =====================================================================
# FROM CORRELATION TO RFE
# =====================================================================

setwd("Path_to_WorkDir")
 
library("cluster")
library("factoextra")
library("magrittr")
library("caret")
library("RColorBrewer")
library("gplots")
library("Hmisc")
library("randomForest")

# 1. Load Input Data
my_df <- read.delim(file.choose())
head(my_df)

# =====================================================================
# DATA PREPROCESSING
# =====================================================================

metadata <- my_df[, c("sampleID", "farm_id", "CATEG_LUMAGORRI")] 
predictors_raw <- my_df[, !(names(my_df) %in% c("sampleID", "farm_id", "CATEG_LUMAGORRI"))]

# =====================================================================
# Zero Variance filtering
# =====================================================================
nsv <- nearZeroVar(predictors_raw, saveMetrics = TRUE)
print("Resumen de varianza:")
print(nsv)

if (any(nsv$nzv)) {
  predictors_filtered <- predictors_raw[, !nsv$nzv]
  print("Variables eliminadas por varianza cero:")
  print(rownames(nsv)[nsv$nzv])
} else {
  predictors_filtered <- predictors_raw
  print("No se encontraron variables de varianza cero.")
}

# =====================================================================
# Re-join clean dataset
# =====================================================================

dataset <- cbind(metadata, predictors_filtered)

rownames(dataset) <- dataset$sampleID
dataset$sampleID <- NULL 

 #=====================================================================
# CORRELATIONS
# =====================================================================
res2 <- cor(dataset[3:ncol(dataset)], use = "pairwise.complete.obs")
round(res2, 2)

res3 <- rcorr(as.matrix(dataset[3:ncol(dataset)]), type = c("pearson", "spearman"))

flattenCorrMatrix <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor = (cormat)[ut],
    p = pmat[ut]
  )
}

tabprob <- flattenCorrMatrix(res3$r, res3$P)
write.table(tabprob, file = "Tabla_correlaciones.tab", sep = "\t", eol = "\n", row.names = FALSE)

selres2 <- res2[1:ncol(res2), 1:ncol(res2)]
pres3 <- res3$P[1:ncol(res2), 1:ncol(res2)]

# Correlation graphics
pdf("correlaciones_05_full.pdf")
corrplot(selres2, type = "upper", order = "hclust", mar = c(2, 0, 1, 0), tl.cex = 0.9,
         col = brewer.pal(n = 8, name = "RdYlBu"))
corrplot(selres2, method = "circle", mar = c(2, 0, 1, 0), tl.col = "black", tl.cex = 0.9)

library("PerformanceAnalytics")
chart.Correlation(selres2, histogram = TRUE, pch = 19)
dev.off()

# Heatmap
pdf("Heatmap_Correlacion_Variables.pdf", width = 10, height = 10)
heatmap.2(selres2,
          main = "Correlation",
          notecol = "black",
          trace = "none",
          margins = c(12, 9),
          col = colorRampPalette(c("blue", "white", "yellow"))(75),
          dendrogram = "column")
dev.off()

# =====================================================================
# Dataset preprocessing for RFE
# =====================================================================
set.seed(7)

dataset <- dataset[complete.cases(dataset), ]
dataset$CATEG_LUMAGORRI <- as.factor(dataset$CATEG_LUMAGORRI)

# IMPORTANT: make sure your farm variable column name is "farm_id" 
#and your categories are uion a column named  "CATEG_LUMAGORRI"

dataset$farm_id <- as.factor(dataset$farm_id)

# PreProcessing and scaling 
features_to_scale <- dataset[, !(names(dataset) %in% c("CATEG_LUMAGORRI", "farm_id"))]
process <- preProcess(features_to_scale, method = c("range"))
norm_scale <- predict(process, as.data.frame(features_to_scale))

norm_scale$CATEG_LUMAGORRI <- dataset$CATEG_LUMAGORRI
norm_scale$farm_id <- dataset$farm_id

# =====================================================================
# RFE with cross validation per farm
# =====================================================================
set.seed(7)

# Folds defined per farm to avoid data leakage
num_granjas <- length(unique(norm_scale$farm_id))
folds_agrupados <- groupKFold(norm_scale$farm_id, k = min(10, num_granjas))

# RFE Control adapted to group indexing
control_rfe <- rfeControl(
  functions = rfFuncs, 
  method = "cv", 
  index = folds_agrupados # <--- AQUÍ SE CORRIGE EL DATA LEAKAGE EN EL RFE
)

subsets <- c(1:5, 10, 15, 19)

# Run RFE (excluding Category & farm_id variables)
x_vars <- norm_scale[, !(names(norm_scale) %in% c("CATEG_LUMAGORRI", "farm_id"))]
y_target <- norm_scale$CATEG_LUMAGORRI

results <- rfe(x_vars, y_target, sizes = subsets, rfeControl = control_rfe)

sink("Top_Variables.txt")
print(results)
print("############# TOP VARIABLES #############")
print(predictors(results))
sink()

# RFE graphics
pdf("Variables_Accuracy.pdf", width = 10, height = 10)
plot(results, type = c("g", "o"))
print(ggplot(data = results, metric = "Accuracy") + theme_bw())
print(ggplot(data = results, metric = "Kappa") + theme_bw())
dev.off()

# Variable importance barplot
varimp_data <- data.frame(
  feature = row.names(varImp(results))[1:min(16, nrow(varImp(results)))],
  importance = varImp(results)[1:min(16, nrow(varImp(results))), 1]
)

tiff("Variables_Importance_barplot.tiff", units = "in", width = 9, height = 6, res = 500)
ggplot(data = varimp_data, aes(x = reorder(feature, -importance), y = importance, fill = feature)) +
  geom_bar(stat = "identity") + 
  labs(x = "Features", y = "Variable Importance") + 
  geom_text(aes(label = round(importance, 2)), vjust = 1.6, color = "white", size = 4) + 
  theme_bw() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))
dev.off()

######################Get optimal variables from RFE

variables_rfe <- predictors(results)
cols_necesarias <- c("Category", "farm_id", variables_rfe)
dataset_para_modelos <- dataset[, colnames(dataset) %in% cols_necesarias]

# RECOVED SAMPLEID
dataset_para_modelos$sampleID <- rownames(dataset_para_modelos)
dataset_para_modelos <- dataset_para_modelos[, c("sampleID", setdiff(names(dataset_para_modelos), "sampleID"))]

head(dataset_para_modelos)

# Export table to TSV 
write.table(dataset_para_modelos, file = "Dataset_Variables_Seleccionadas.tsv", sep = "\t", eol = "\n", row.names = FALSE)

