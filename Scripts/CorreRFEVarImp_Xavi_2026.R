# =====================================================================
# SCRIPT DEPURADO Y CORREGIDO CONTRA DATA LEAKAGE (GRUPO-GRANJA)
# =====================================================================

setwd("C:/Users/jllavin/Desktop/Paper_Xavi_ML_SciRep_2026")
# Cargar paquetes esenciales
library("cluster")
library("factoextra")
library("magrittr")
library("caret")
library(RColorBrewer)
library(gplots)
library("Hmisc")
library("randomForest")

# 1. Cargar datos
my_df <- read.delim(file.choose())
head(my_df)

# =====================================================================
# 1. SEPARAR METADATOS Y PREDICTORES DESDE EL INICIO
# =====================================================================
# Asegúrate de que estos son los nombres exactos de tus columnas de identificación/objetivo
metadata <- my_df[, c("sampleID", "farm_id", "CATEG_LUMAGORRI")] 
predictors_raw <- my_df[, !(names(my_df) %in% c("sampleID", "farm_id", "CATEG_LUMAGORRI"))]

# =====================================================================
# 2. APLICAR nearZeroVar ÚNICAMENTE A LOS PREDICTORES
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
# 3. RECONSTRUIR EL DATASET LIMPIO
# =====================================================================
# Volvemos a unir los metadatos limpios con las variables predictoras filtradas
dataset <- cbind(metadata, predictors_filtered)

# Asignar rownames y preparar para los siguientes pasos
rownames(dataset) <- dataset$sampleID
dataset$sampleID <- NULL # Ya cumplió su función como nombre de fila

# A partir de aquí, 'dataset' ya tiene las ZV eliminadas y la estructura intacta

# =====================================================================
# MATRIZ DE CORRELACIÓN Y ESTADÍSTICOS
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

# Gráficos de correlación
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
# PREPARACIÓN DEL DATASET PARA MODELIZACIÓN
# =====================================================================
set.seed(7)
# dataset <- my_df
dataset <- dataset[complete.cases(dataset), ]
dataset$CATEG_LUMAGORRI <- as.factor(dataset$CATEG_LUMAGORRI)

# IMPORTANTE: Asegúrate de tener una columna que identifique la granja (ej. farm_id)
# Si se llama de otra forma en tu tabla, cambia "farm_id" por su nombre exacto:
dataset$farm_id <- as.factor(dataset$farm_id)

# rownames(dataset) <- dataset$sampleID
# dataset$sampleID <- NULL

# PreProcesamiento y escalado (excluyendo la columna de granja para que no se escale)
features_to_scale <- dataset[, !(names(dataset) %in% c("CATEG_LUMAGORRI", "farm_id"))]
process <- preProcess(features_to_scale, method = c("range"))
norm_scale <- predict(process, as.data.frame(features_to_scale))

# Volvemos a integrar Category y farm_id en el dataset normalizado
norm_scale$CATEG_LUMAGORRI <- dataset$CATEG_LUMAGORRI
norm_scale$farm_id <- dataset$farm_id

# =====================================================================
# SELECCIÓN DE CARACTERÍSTICAS (RFE) CON VALIDACIÓN CRUZADA POR GRANJA
# =====================================================================
set.seed(7)

# Definir los folds agrupados por granja para evitar Data Leakage
# (Garantiza que las muestras de una misma granja no se dividen entre train y validación)
num_granjas <- length(unique(norm_scale$farm_id))
folds_agrupados <- groupKFold(norm_scale$farm_id, k = min(10, num_granjas))

# Control de RFE adaptado con los índices de grupo
control_rfe <- rfeControl(
  functions = rfFuncs, 
  method = "cv", 
  index = folds_agrupados # <--- AQUÍ SE CORRIGE EL DATA LEAKAGE EN EL RFE
)

subsets <- c(1:5, 10, 15, 19)

# Ejecutar RFE (excluyendo Category y farm_id de las variables predictoras x)
x_vars <- norm_scale[, !(names(norm_scale) %in% c("CATEG_LUMAGORRI", "farm_id"))]
y_target <- norm_scale$CATEG_LUMAGORRI

results <- rfe(x_vars, y_target, sizes = subsets, rfeControl = control_rfe)

sink("Top_Variables.txt")
print(results)
print("############# TOP VARIABLES #############")
print(predictors(results))
sink()

# Gráficos de resultados del RFE
pdf("Variables_Accuracy.pdf", width = 10, height = 10)
plot(results, type = c("g", "o"))
print(ggplot(data = results, metric = "Accuracy") + theme_bw())
print(ggplot(data = results, metric = "Kappa") + theme_bw())
dev.off()

# Gráfico de barras de importancia de variables del RFE
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

######################1. Extraer los nombres de las variables óptimas del RFE
# 1. Seleccionar tus variables (del RFE o elegidas)
variables_rfe <- predictors(results) # o tus_variables_elegidas
cols_necesarias <- c("Category", "farm_id", variables_rfe)
dataset_para_modelos <- dataset[, colnames(dataset) %in% cols_necesarias]

# 2. RECUPERAR EL SAMPLEID: Convertir los nombres de fila actuales en una columna real
dataset_para_modelos$sampleID <- rownames(dataset_para_modelos)

# 3. Reordenar para que 'sampleID' sea la primera columna (opcional, pero queda muy ordenado)
dataset_para_modelos <- dataset_para_modelos[, c("sampleID", setdiff(names(dataset_para_modelos), "sampleID"))]

# Comprobación en consola (verás tus IDs reales en texto)
head(dataset_para_modelos)

# 4. Exportar a TSV (ahora sí, con row.names = FALSE porque sampleID ya es una columna de verdad)
write.table(dataset_para_modelos, file = "Dataset_Variables_Seleccionadas.tsv", sep = "\t", eol = "\n", row.names = FALSE)

# ##############SELECTED VARIABLES####################################################
# # 1. Escribe tú mismo los nombres exactos de las variables que quieres conservar
# mis_variables_elegidas <- c("Variable_1", "Variable_2", "Variable_3", "Variable_4", "Variable_5", "Variable_6")
# 
# # 2. Construir el data.frame final con Category, farm_id y tus variables elegidas
# cols_necesarias <- c("Category", "farm_id", mis_variables_elegidas)
# dataset_para_modelos <- dataset[, colnames(dataset) %in% cols_necesarias]
# 
# # Comprobación rápida
# head(dataset_para_modelos)
# 
# # Exportar el dataframe resultante a un archivo TSV
# write.table(dataset_para_modelos, file = "Dataset_Variables_SeleccionadasManualmente.tsv", sep = "\t", eol = "\n", row.names = FALSE)