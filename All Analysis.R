## Program     : Coursera Data Science - Practical Machine Learning
## Written By  : Gabriel Mohanna
## Date Created: July 26, 2014
##
## Narrative   : 
##
## Background  : 
##
## TBD         : 
##
## \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Code is Poetry >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
##
## **********************************************************************************************************************
## Steps
## -----
## (0) Define Paths, Input Files & Load Libraries
## (1) Read and Preprocess the Data
## (2) Build Models with & without PCA
## (3) Predict Against Validation
##
## **********************************************************************************************************************
## Notes
## ---------------------
##
## **********************************************************************************************************************

# ***********************************************************************************************************************
# (0) Define Paths, Input Files & Load Libraries
# ***********************************************************************************************************************
# Clear Workspace
rm(list=ls())

# Define Data Path
root      <- "/Users/gabrielm/OneDrive/Documents/HW/Coursera/Data Science Specialization/"
data_path <- "8 - Machine Learning/Project/Data/"
out_path  <- "8 - Machine Learning/Project/Output/"

# Define Input Files
data       <- "pml-training.csv"
data.valid <- "pml-testing.csv"

# Load Libraries
library(caret)
library(randomForest)
library(corrplot)

# Define Paths, Input Files & Load Libraries


# ***********************************************************************************************************************
# (1) Read and Preprocess the Data
# ***********************************************************************************************************************
# Set Working Directory
setwd(paste0(root, data_path))

# Read Data
data       <- read.csv(data      , na.string=c("NA", ""))
data.valid <- read.csv(data.valid, na.string=c("NA", ""))

# Determine Data's NA Columns
table(round(apply(data, 2, function(x) sum(is.na(x)))/nrow(data), 2))
columns.naCount <- apply(data, 2, function(x) sum(is.na(x)))

# Remove Data's NA Columns
columns.NA    <- names(columns.naCount)[columns.naCount> 0]
columns.notNA <- names(columns.naCount)[columns.naCount==0]
data <- data[, columns.notNA]
nearZeroVar(data, saveMetrics = T)
data <- data[, -c(1:7)]

# Set Seet
set.seed(3599)

# Partition Data
inTrain <- createDataPartition(y=data$classe, p=.75, list=F)
data.train <- data[ inTrain, ]
data.test  <- data[-inTrain, ]

# Clean Up
rm(inTrain, data, columns.naCount, columns.NA, columns.notNA)

# End Read and Preprocess the Data


# ***********************************************************************************************************************
# (2) Build Random Forest Models with & without PCA
# ***********************************************************************************************************************
# Explore Data
cor.data.train <- cor(data.train[, -53])
par(mar = c(0, 0, 0, 0))
corrplot(cor.data.train, method="color")

# Train Models without PCA
# Random Forest
modFit.RF <- randomForest(classe~., data=data.train)

# Neural Network
modFit.NN <- nnet(classe~., data=data.train, size=1, decay=5e-4, maxit=200)

# Random Forest in Caret Package
modFit.Caret.RF <- train(classe ~ ., method="rf", data=data.train, 
                         trControl=trainControl(method="repeatedcv", 
                                                number=2, repeats=2))

# Regression Trees in Caret Package
modFit.Caret.RT <- train(classe ~ ., method="rpart", data=data.train, 
                         trControl=trainControl(method="repeatedcv", 
                                                number=2, repeats=2))

# PCA
preProc <- preProcess(data.train[, -53], method="pca")
trainPC <- predict(preProc, data.train[, -53])
testPC  <- predict(preProc, data.test [, -53])

# Train Models with PCA
# Random Forest
modFit.RF.PCA <- randomForest(classe~., data=trainPC)

# Neural Network
modFit.NN.PCA <- nnet(classe~., data=trainPC, size=1, decay=5e-4, maxit=200)

# Random Forest in Caret Package
modFit.Caret.RF.PCA <- train(classe ~ ., method="rf", data=trainPC, 
                             trControl=trainControl(method="repeatedcv", 
                                                    number=2, repeats=2))

# Regression Trees in Caret Package
modFit.Caret.Rt.PCA <- train(classe ~ ., method="rpart", data=trainPC, 
                             trControl=trainControl(method="repeatedcv", 
                                                    number=2, repeats=2))

# Compare Out-of-Sample Errors
percent <- function(x, digits = 2, format = "f", ...) {
    paste0(formatC(100 * x, format = format, digits = digits, ...), "%")
}

percent(1-confusionMatrix(data.test$classe, predict(modFit.RF      , data.test))$overall[[1]])
percent(1-confusionMatrix(data.test$classe, predict(modFit.NN      , data.test))$overall[[1]])
percent(1-confusionMatrix(data.test$classe, predict(modFit.Caret.RF, data.test))$overall[[1]])
percent(1-confusionMatrix(data.test$classe, predict(modFit.Caret.RT, data.test))$overall[[1]])

percent(1-confusionMatrix(data.test$classe, predict(modFit.RF.PCA      , data.test))$overall[[1]])
percent(1-confusionMatrix(data.test$classe, predict(modFit.NN.PCA      , data.test))$overall[[1]])
percent(1-confusionMatrix(data.test$classe, predict(modFit.Caret.RF.PCA, data.test))$overall[[1]])
percent(1-confusionMatrix(data.test$classe, predict(modFit.Caret.RT.PCA, data.test))$overall[[1]])

# End Build Models with & without PCA

# ***********************************************************************************************************************
# (3) Predict Against Validation
# ***********************************************************************************************************************
answers = predict(modFit.RF, data.valid)

temp==answers
setwd(paste0(root, out_path))
pml_write_files <- function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i], file=filename, quote=FALSE, row.names=FALSE, col.names=FALSE)
    }
}

pml_write_files(answers)

# End Predict Against Validation
