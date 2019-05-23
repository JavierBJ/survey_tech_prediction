# Technology survey prediction
# Javier Beltrán

library(foreign)
library(mlr)
library(caret)

# 0. Load dataset and clean it
#-----------------------------

# Read dataset and columns of interest
original_dataset <- read.dta("tc_data.dta")
cols <- c('sindicato','industria','paro_self_1','paro_self_2',
         'habilidades_1','habilidades_2','habilidades_3','habilidades_4','habilidades_5','habilidades_6',
         'habilidades_7','habilidades_8','habilidades_9','habilidades_10',
         'female','age','educacion', 'income_2', 'empleo','ocupacion','routine','paro_razon_all','riesgo',
         'ideol','beneficios_1','beneficios_2','beneficios_3','beneficios_4', 'exp1_tech_cons')
dataset <- original_dataset[cols]

# Impute missing values (NA) in all columns, using the mean in numeric data and the mode in categorical data
imp <- impute(dataset, target="exp1_tech_cons", classes=list(numeric=imputeMean(), factor=imputeMode()))
dataset <- imp$data

# Variable to predict must be an (ordered) factor, instead of numeric, or cannot run classification algorithm
dataset$exp1_tech_cons <- factor(dataset$exp1_tech_cons, ordered=TRUE)

# Specify which factor variables are ordinal
dataset$age <- factor(dataset$age, ordered=TRUE)
dataset$educacion <- factor(dataset$educacion, ordered=TRUE)
dataset$income_2 <- factor(dataset$income_2, ordered=TRUE)
dataset$routine <- factor(dataset$routine, ordered=TRUE)
dataset$riesgo <- factor(dataset$riesgo, ordered=TRUE)
dataset$ideol <- factor(dataset$ideol, ordered=TRUE)
dataset$beneficios_1 <- factor(dataset$beneficios_1, ordered=TRUE)
dataset$beneficios_2 <- factor(dataset$beneficios_2, ordered=TRUE)
dataset$beneficios_3 <- factor(dataset$beneficios_3, ordered=TRUE)
dataset$beneficios_4 <- factor(dataset$beneficios_4, ordered=TRUE)

# 1. Fix Income variable by imputing a value to people who prefered not to answer
#--------------------------------------------------------------------------------

income_dataset <- dataset[,!(names(dataset) %in% c("exp1_tech_cons"))]
income_labeled <- income_dataset[income_dataset$income_2!="Prefers not to respond",]

# Remove "Prefers not to respond" level from dataset "income_labeled", as no rows contain this value
income_labeled$income_2 <- factor(income_labeled$income_2, levels=levels(income_labeled$income_2), exclude="Prefers not to respond")

# Evaluation settings
control <- trainControl(method="cv", number=10, search="random") # 10-fold cross-validation using random grid search for speed
metric <- "Accuracy"
income_grid <- expand.grid(k = c(1,3,5,7)) # These are the k values tried for knn imputation

# The model varies a lot between executions. Run it several times and keep the best model
best_score <- 0
for (i in 1:30) {
  # Train K-nn to predict unlabeled incomes
  fit_income <- train(income_2~., income_labeled, method='knn', metric=metric, trControl=control, tuneGrid=income_grid)
  predictions_income <- predict(fit_income, dataset)
  
  # Only apply predicted incomes to rows where its original value was "Prefers not to respond"
  imputeknn <- function(a, b) {
    if (a == "Prefers not to respond") {
      b
    } else {
      a
    }
  }
  new_income <- mapply(imputeknn, dataset$income_2, predictions_income) # Apply imputeknn element by element
  
  # Restore new income_2 factors, as now there are 0 values with level "Prefers not to respond"
  dataset$income_2 <- factor(new_income, levels=levels(dataset$income_2), exclude="Prefers not to respond")
  
  # 2. Predict "exp1_tech_cons" with supervised learning
  # ----------------------------------------------------
  

  # Train-validation split of labeled data
  labeled = dataset[!is.na(dataset$exp1_tech_cons),]
  validation_index <- createDataPartition(labeled$exp1_tech_cons, p=0.80, list=FALSE)
  validation <- labeled[-validation_index,]
  training <- labeled[validation_index,]
  
  # Subsample training corpus to have proportional classes
  # Function renames class variable to "Class", we rename it to our previous "exp1_tech_cons"
  training <- downSample(training[, -ncol(training)], training$exp1_tech_cons)
  names(training)[names(training)=="Class"] <- "exp1_tech_cons"
  
  # Fit model on training, predict on validation. Model used is Random Forest
  fit <- train(exp1_tech_cons~., data=training, method="rf", metric=metric, trControl=control)
  predictions <- predict(fit, validation)
  
  # Accuracy
  conf <- confusionMatrix(validation$exp1_tech_cons, predictions)

  # Measure how good this run was, using the mean of balanced accuracies for each class
  score <- mean(conf$byClass[,"Balanced Accuracy"])
  
  # Predict unlabeled rows
  results <- predict(fit, dataset)
  
  # If these are the best results obtained, save them
  if (score > best_score) {
    best_score = score
    best_results <- results
    best_conf <- conf
  }
}

# Only apply predicted results to rows that were NA originally
imputetech <- function(a, b) {
  if (is.na(a)) {
    b
  } else {
    a
  }
}
new_tech <- mapply(imputeknn, dataset$income_2, predictions_income) # Apply imputeknn element by element

# Create new dataset with imputed columns
out_dataset <- data.frame("ResponseId"=original_dataset$ResponseId, "income"=new_income, "exp1_tech_cons"=new_tech)
write.csv(out_dataset, "tc_data_imputed.csv")