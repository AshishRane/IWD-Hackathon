# All State Insurance Claims Severity Prediction
# Date: 27 Oct 2016

# Load Required packages

library(xgboost)
library(randomForest)
library(data.table)
library(Matrix)
library(FeatureHashing)
library(dplyr)
library(readr)
library(Metrics)
library(caret)
library(ggplot2)
library(data.table)

#Set the current working directory
setwd("/Users/ashish/Documents/Kaggle/AllStateInsuranceClaimSeverity")

#Load the datasets
train_orig <- fread('train-2.csv')
test_orig <- fread('test-2.csv')

#Feature engineering
#New feature - 'losscat' feature has been added to categorize the losses based on their values

#Exploratory Data Analysis

#Check the structure of the datasets

str(train_orig)
str(test_orig)

#Check corelation between continuous variables

co_var <- paste0("cont", seq(1, 14))
summary(train_orig[,co_var]) #No missing values
train_orig_df <- as.data.frame(train_orig)
corelation <- cor(train_orig_df[,co_var])
head(round(corelation, 2))                  
                   
#Plot corelation graph

# correlations plot between predictors
corrplot(corelation, type="upper", order="hclust",col=brewer.pal(n=8, name="PuOr"),addCoef.col = "grey",diag=FALSE)
                   
#Load the IDs from the test dataset to a variable for later use
testID <- test_orig$id

#Setting levels
cvar <- names(train_orig)[sapply(train_orig, is.character)]
cat(cvar)
for(var in cvar) {
  foo.levels <- unique(c(train_orig[[var]], test_orig[[var]]))
  set(train_orig, j = var, value = factor(train_orig[[var]], levels = foo.levels))
  set(test_orig, j = var, value = factor(test_orig[[var]], levels = foo.levels))
}

#Dependent variable from train dataset is saved to a new variable for later use
response <- train_orig$loss

#Set Id & loss variables from train dataset to NULL 
train_orig[, id := NULL]
train_orig[, loss := NULL]

#Set Id variable from test dataset to NULL
test_orig[, id := NULL]

#Merge the train and test datasets
merge <- rbind(train_orig, test_orig)

merge$i <- 1:dim(merge)[1]

#Check for factor variables
fac_var <- names(train_orig)[sapply(train_orig, is.factor)]

merge[, (fac_var) := lapply(.SD, as.numeric), .SDcols = fac_var]

#Creating a sparse matrix 
merge.sparse <- sparseMatrix(merge$i, merge[,cvar[1], with = FALSE][[1]])

for(var in cvar[-1]){
  merge.sparse <- cbind(merge.sparse, sparseMatrix(merge$i, merge[,var, with = FALSE][[1]])) 
  cat('Combining: ', var, '\n')
}

#Separating the test and train dataset
merge.sparse <- cbind(merge.sparse, as.matrix(merge[,-c(char.var, 'i'), with = FALSE]))
dim(merge.sparse)
train <- merge.sparse[1:(dim(train_orig)[1]),]
test <- merge.sparse[(dim(train_orig)[1] + 1):nrow(merge),]

# Function & attributes from
# https://www.kaggle.com/nigelcarpenter/allstate-claims-severity/farons-xgb-starter-ported-to-r/code

xg_eval_mae <- function (yhat, dtrain) {
  y = getinfo(dtrain, "label")
  err= mae(exp(y),exp(yhat) )
  return (list(metric = "error", value = err))
}

xgb_params = list(
  seed = 0,
  colsample_bytree = 0.7,
  subsample = 0.7,
  eta = 0.075,
  objective = 'reg:linear',
  max_depth = 6,
  num_parallel_tree = 1,
  min_child_weight = 1,
  base_score = 7
)

#Get a random sample of data
sample.index <- sample(1:nrow(train_orig), nrow(train_orig) * 0.9)


validation <- xgb.DMatrix(train[-sample.index,], label = log(response[-sample.index]))
train <- xgb.DMatrix(train[sample.index,], label = log(response[sample.index]))
test <- xgb.DMatrix(test)


xgboost_model <- xgb.train(xgb_params,
                      train,
                      nrounds=500,
                      print_every_n = 5,
                      verbose= 1,
                      watchlist = list(valid_score = validation),
                      feval=xg_eval_mae,
                      early_stop_rounds = 20,
                      maximize=FALSE)

xgboost_prediction <- predict(xgboost_model, test)

submission <- data.frame(id = testID, loss = exp(xgboost_prediction))

write.csv(submission, 'xgboost_AR4.csv', row.names = FALSE)
