install.packages("pROC")
library(pROC)
setwd("C:/Users/mot16/Box Sync/Projects/Port-project/")
train = read.csv("data/port_train_new_subset.csv")
dim(train)
lr.prob = read.csv("output/probs_LR_L1_1.0_SMOTE.csv", header = FALSE)
dim(lr.prob)
rf.prob = read.csv("output/probs_RF_500_SMOTE.csv", header = FALSE)
dim(rf.prob)
nb.prob = read.csv("output/probs_NB_10_SMOTE.csv", header = FALSE)
dim(nb.prob)
svm.prob = read.csv("output/probs_SVM_0.1_SMOTE.csv", header = FALSE)
dim(svm.prob)

resp = train$X217.DIREOUT
length(resp)

rf.roc = roc(response = resp, predictor = rf.prob$V1)
lr.roc = roc(response = resp, predictor = lr.prob$V1)
nb.roc = roc(response = resp, predictor = nb.prob$V1)
svm.roc = roc(response = resp, predictor = svm.prob$V1)

ci.auc(rf.roc)
ci.auc(lr.roc)
ci.auc(nb.roc)
ci.auc(svm.roc)

#testobj <- roc.test(rf.roc, lr.roc, method = "delong", alternative = "two.sided", paired = TRUE)

