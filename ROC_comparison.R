install.packages("pROC")
library(pROC)
#setwd("C:/Users/mot16/Box Sync/My CoSBBI/Data")
test = read.csv("data/port_test.csv")
dim(test)
rf.prob = read.csv("output/rf_probs.csv", header = FALSE)
dim(rf.prob)
lr.prob = read.csv("output/nb_probs.csv", header = FALSE)
dim(lr.prob)
resp = test$X217.DIREOUT
resp[resp == 1] = 0
resp[resp == 2] = 1


rf.roc = roc(response = resp, predictor = rf.prob$V2)
lr.roc = roc(response = resp, predictor = lr.prob$V2)
auc(rf.roc)
auc(lr.roc)

par(mar = c(4, 4, 4, 4)+.1)

plot.roc(rf.roc, col = "green", legacy.axes = TRUE, asp = NA, print.auc = FALSE
         , main = "ROC curves. Random Forest VS. Logistic Regression\n DeLong's test"
         ,cex.main = .9)

plot.roc(lr.roc, add=TRUE, col="blue", legacy.axes = TRUE, print.auc = FALSE)

testobj <- roc.test(rf.roc, lr.roc, method = "delong", alternative = "two.sided", paired = TRUE)

#testobj2 <- roc.test(rf.roc, lr.roc, method = "delong", alternative = "g", paired = TRUE)

text(.5, .5, labels=paste("RF.AUC =", round(auc(rf.roc), 3)), adj=c(0, .5))
text(.5, .4, labels=paste("LR.AUC =", round(auc(lr.roc), 3)), adj=c(0, .5))
text(.5, .3, labels=paste("p-value =", format.pval(testobj$p.value)), adj=c(0, .5))


legend("bottomright", legend=c("RF", "LR"),
       col=c("green", "blue"), lwd=2)
