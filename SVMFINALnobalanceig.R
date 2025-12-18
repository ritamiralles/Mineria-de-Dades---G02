# ===============================
# SVM – sense balanceig + KPIs
# ===============================

library(dplyr)
library(caret)
library(e1071)
library(pROC)

set.seed(2025)

# 1) Split train / test (dividim el dataset incial segons train i test)
df <- dades_imputades
df_tr <- df %>% filter(Group == "train")
df_te <- df %>% filter(Group == "test")

# 2) Agafem unicament les Variables relevants 
#i la varaiable resposta: extited 
vars_x <- c("Geography", "Balance", "Age",
            "IsActiveMember", "NumOfProducts",
            "NetPromoterScore", "Gender")
target <- "Exited"


# 3) Exited estava posat com: 0-1, ho transformem en format factor Yes / No
df_tr[[target]] <- factor(
  ifelse(as.character(df_tr[[target]]) %in% c("1"), "Yes", "No"),
  levels = c("Yes","No")
)


# 4) Converteix variables categòriques en números (dummies), ja que SVM nomes enten numeros
#Això es fa només amb el train, i després s’aplica igual al test(acaben tenint exctament les mateixes columes)
dmy <- dummyVars(~ ., data = df_tr[, vars_x, drop = FALSE], fullRank = TRUE)
X_tr <- as.data.frame(predict(dmy, df_tr[, vars_x, drop = FALSE]))
X_te <- as.data.frame(predict(dmy, df_te[, vars_x, drop = FALSE]))

# 5) Escalat 
#Posem totes les variables a la mateixa escala, calculant la mitjana  i deviacio tipica
#i aixi evitem que una variable “domini” el model
num_cols <- sapply(X_tr, is.numeric)
X_tr_num <- X_tr[, num_cols, drop = FALSE]
X_te_num <- X_te[, colnames(X_tr_num), drop = FALSE]

pp <- preProcess(X_tr_num, method = c("center","scale"))
X_tr_sc <- predict(pp, X_tr_num)
X_te_sc <- predict(pp, X_te_num)

# 6) Ara ja dividim el dataset final en train i test 
#perque amb els canvis fets ja podem fer SVM
df_tr_svm <- cbind(Exited = df_tr[[target]], X_tr_sc)
df_te_svm <- X_te_sc

# 7) ) Validació interna 70/30 (per avaluar)
#Del train:fem 70% per entrenar i 30% per validar (val_in)
#Això és per veure si el model funciona abans d’enviar a Kaggle
set.seed(123)
idx <- createDataPartition(df_tr_svm$Exited, p = 0.7, list = FALSE)
train_in <- df_tr_svm[idx, ]
val_in   <- df_tr_svm[-idx, ]

# 8) Entrenament SVM sense balanceig
svm_in <- svm(
  Exited ~ .,      #varaible a predir(resta com predictors)
  data   = train_in,  #utilitzem nomes 70% train 
  kernel = "radial",    #kernel radial es el que funciona millor per relacions no lineals, mes complexes
  cost   = 1,    #perque no sobreajusti ni sigui masa felxible
  gamma  = 0.1       #valor intermig
)

# 9) ara el model prediu el 30% que no ha vist i ho comparem amb la realitat
pred_train <- predict(svm_in, train_in)
cm_train   <- confusionMatrix(pred_train, train_in$Exited, positive = "Yes")

prob_train <- attr(
  predict(svm_in, train_in, decision.values = TRUE),
  "decision.values"
)

roc_train <- roc(train_in$Exited, as.numeric(prob_train))
auc_train <- as.numeric(auc(roc_train))

pred_val <- predict(svm_in, val_in)
cm_val   <- confusionMatrix(pred_val, val_in$Exited, positive = "Yes")

prob_val <- attr(
  predict(svm_in, val_in, decision.values = TRUE),
  "decision.values"
)

roc_val <- roc(val_in$Exited, as.numeric(prob_val))
auc_val <- as.numeric(auc(roc_val))

# ===============================
# TAULA FINAL TRAIN vs TEST
# ===============================
results_no_bal <- data.frame(
  Model     = "SVM sense balanceig",
  Dataset   = c("Train","Test"),
  Accuracy  = round(c(cm_train$overall["Accuracy"],
                      cm_val$overall["Accuracy"]), 4),
  Precision = round(c(cm_train$byClass["Precision"],
                      cm_val$byClass["Precision"]), 4),
  Recall    = round(c(cm_train$byClass["Recall"],
                      cm_val$byClass["Recall"]), 4),
  F1        = round(c(cm_train$byClass["F1"],
                      cm_val$byClass["F1"]), 4),
  Kappa     = round(c(cm_train$overall["Kappa"],
                      cm_val$overall["Kappa"]), 4),
  AUC       = round(c(auc_train, auc_val), 4)
)

print(results_no_bal)

# ===============================
# SUBMISSIÓ KAGGLE (sense balanceig)
# ===============================
svm_final <- svm(
  Exited ~ .,
  data   = df_tr_svm,
  kernel = "radial",
  cost   = 1,
  gamma  = 0.1
)

pred_test <- predict(svm_final, df_te_svm)

svm_submit <- data.frame(
  ID = df_te$ID,
  Exited = pred_test
)

write.csv(svm_submit,
          "svm_reduides_submission.csv",
          row.names = FALSE)


# ==============================
# SVM AMB BALANCEIG + KPI
# ==============================
library(dplyr)
library(caret)
library(e1071)
library(pROC)

set.seed(2025)

# 1) Split train / test
df <- dades_imputades
df_tr <- df %>% filter(Group == "train")
df_te <- df %>% filter(Group == "test")

# 2) Variables
vars_x <- c("Geography", "Balance", "Age",
            "IsActiveMember", "NumOfProducts",
            "NetPromoterScore", "Gender")
target <- "Exited"

# 3) Target Yes / No
df_tr[[target]] <- factor(
  ifelse(as.character(df_tr[[target]]) %in% c("1"), "Yes", "No"),
  levels = c("Yes","No")
)

# 4) Dummies
dmy <- dummyVars(~ ., data = df_tr[, vars_x, drop = FALSE], fullRank = TRUE)
X_tr <- as.data.frame(predict(dmy, df_tr[, vars_x, drop = FALSE]))
X_te <- as.data.frame(predict(dmy, df_te[, vars_x, drop = FALSE]))

# 5) Escalat
num_cols <- sapply(X_tr, is.numeric)
X_tr_num <- X_tr[, num_cols, drop = FALSE]
X_te_num <- X_te[, colnames(X_tr_num), drop = FALSE]

pp <- preProcess(X_tr_num, method = c("center","scale"))
X_tr_sc <- predict(pp, X_tr_num)
X_te_sc <- predict(pp, X_te_num)

# 6) Dataset final per SVM
df_tr_svm <- cbind(Exited = df_tr[[target]], X_tr_sc)
df_te_svm <- X_te_sc

# =============================
# BALANCEIG AMB OVERSAMPLING
# =============================
set.seed(2025)
train_bal <- upSample(
  x     = df_tr_svm[, colnames(df_tr_svm) != "Exited", drop = FALSE],
  y     = df_tr_svm$Exited,
  yname = "Exited"
)

# 7) Validació interna 70/30
set.seed(123)
idx_bal      <- createDataPartition(train_bal$Exited, p = 0.7, list = FALSE)
train_in_bal <- train_bal[idx_bal, ]
val_in_bal   <- train_bal[-idx_bal, ]

# 8) Entrenament SVM balancejat
svm_in_bal <- svm(
  Exited ~ .,
  data   = train_in_bal,
  kernel = "radial",
  cost   = 1,
  gamma  = 0.1,
  decision.values = TRUE
)

# =============================
# KPIs TRAIN
# =============================
pred_train <- predict(svm_in_bal, train_in_bal)
cm_train   <- confusionMatrix(pred_train, train_in_bal$Exited, positive = "Yes")

prob_train <- attr(
  predict(svm_in_bal, train_in_bal, decision.values = TRUE),
  "decision.values"
)

roc_train <- roc(train_in_bal$Exited, as.numeric(prob_train))
auc_train <- as.numeric(auc(roc_train))

# =============================
# KPIs TEST (validació)
# =============================
pred_val <- predict(svm_in_bal, val_in_bal)
cm_val   <- confusionMatrix(pred_val, val_in_bal$Exited, positive = "Yes")

prob_val <- attr(
  predict(svm_in_bal, val_in_bal, decision.values = TRUE),
  "decision.values"
)

roc_val <- roc(val_in_bal$Exited, as.numeric(prob_val))
auc_val <- as.numeric(auc(roc_val))

# =============================
# TAULA FINAL TRAIN vs TEST
# =============================
results_bal <- data.frame(
  Model     = "SVM amb oversampling",
  Dataset   = c("Train", "Test"),
  Accuracy  = round(c(cm_train$overall["Accuracy"],
                      cm_val$overall["Accuracy"]), 4),
  Precision = round(c(cm_train$byClass["Precision"],
                      cm_val$byClass["Precision"]), 4),
  Recall    = round(c(cm_train$byClass["Recall"],
                      cm_val$byClass["Recall"]), 4),
  F1        = round(c(cm_train$byClass["F1"],
                      cm_val$byClass["F1"]), 4),
  Kappa     = round(c(cm_train$overall["Kappa"],
                      cm_val$overall["Kappa"]), 4),
  AUC       = round(c(auc_train, auc_val), 4)
)

print(results_bal)

# =============================
# SUBMISSIÓ KAGGLE
# =============================
svm_final_bal <- svm(
  Exited ~ .,
  data   = train_bal,
  kernel = "radial",
  cost   = 1,
  gamma  = 0.1
)

pred_test_bal <- predict(svm_final_bal, newdata = df_te_svm)

svm_submit_bal <- data.frame(
  ID     = df_te$ID,
  Exited = pred_test_bal
)

write.csv(
  svm_submit_bal,
  "svm_reduides_bal_submission.csv",
  row.names = FALSE
)

