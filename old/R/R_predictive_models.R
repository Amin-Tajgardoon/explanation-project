library(MLmetrics)
source("get_measures.R")
source("resample.R")

train = read.csv("data/port_train.csv")
train$X217.DIREOUT[train$X217.DIREOUT == "1"] = "0"
train$X217.DIREOUT[train$X217.DIREOUT == "2"] = "1"
train[] = lapply(train, factor)

test = read.csv("data/port_test.csv")
test$X217.DIREOUT[test$X217.DIREOUT == "1"] = "0"
test$X217.DIREOUT[test$X217.DIREOUT == "2"] = "1"
test[] = lapply(test, factor)

for(n in names(test)) {
  levels(test[[n]]) = levels(train[[n]]) 
}

yt  <-  as.numeric(test$X217.DIREOUT)
yt[yt == 1]  <-  0
yt[yt == 2]  <-  1

############# RANDOM FOREST ###################

set.seed(1)

library(randomForest)
rf = randomForest(formula = X217.DIREOUT ~ ., data = train, 
                  ntree = 500,  importance=FALSE, proximity=FALSE)

rf.pred = predict(rf, test, type="prob",
        norm.votes=TRUE, predict.all=FALSE, proximity=FALSE, nodes=FALSE)

rf.probs = rf.pred[,2]

getMeasures(yt, rf.probs)

getConfMat(yt, rf.probs)

############# Naive Bayes ###################

set.seed(1)

library(e1071)

nb <- naiveBayes(formula = X217.DIREOUT ~ ., laplace = 10, data = train)
#summary(nb)
#print(nb)

nb.pred = predict(nb, test, type = "raw")
nb.probs = nb.pred[,2]

getMeasures(yt, nb.probs)
#Precision(y_true = yt, y_pred = as.numeric(nb.probs >= .5), positive = 1)
#Recall(y_true = yt, y_pred = as.numeric(nb.probs >= .5), positive = 1)
#F1_Score(y_true = yt, y_pred = as.numeric(nb.probs >= .5), positive = 1) 
#getConfMat(yt, nb.probs)

############ Resampling - Naive Bayes #####################
set.seed(1)
train.oversample= data.resample(X = train[,-ncol(train)],
                                Y = train$X217.DIREOUT, positive = "1", type = "ubSMOTE", percOver = 500)
colnames(train.oversample)[ncol(train.oversample)] = 'X217.DIREOUT'
table(train.oversample$X217.DIREOUT)/nrow(train.oversample)
nb <- naiveBayes(formula = X217.DIREOUT ~ ., laplace = 10, data = train.oversample)
nb.pred = predict(nb, test, type = "raw")
nb.probs = nb.pred[,2]
getMeasures(yt, nb.probs)
getConfMat(yt, nb.probs)

############ Resampling - RANDOM FOREST #####################
set.seed(1)
train.oversample= data.resample(X = train[,-ncol(train)],
                                Y = train$X217.DIREOUT, positive = "1", type = "ubSMOTE", percOver = 300)
colnames(train.oversample)[ncol(train.oversample)] = 'X217.DIREOUT'
table(train.oversample$X217.DIREOUT)/nrow(train.oversample)

rf = randomForest(formula = X217.DIREOUT ~ ., data = train.oversample, 
                  ntree = 500,  importance=FALSE, proximity=FALSE)

rf.pred = predict(rf, test, type="prob",
                  norm.votes=TRUE, predict.all=FALSE, proximity=FALSE, nodes=FALSE)

rf.probs = rf.pred[,2]

getMeasures(yt, rf.probs)
getConfMat(yt, rf.probs)
