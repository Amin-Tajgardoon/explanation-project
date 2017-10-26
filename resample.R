library(unbalanced)

data.resample <- function (X, Y, positive , type="ubSMOTE", percOver=300, percUnder=150, verbose=TRUE,
                           k=5, perc=50, method="percPos", w=NULL){
  
  start.time <- Sys.time()
  
  data <- ubBalance(X, Y, type=type, positive = positive, percOver=percOver, percUnder=percUnder, verbose=verbose,
                    k=k, perc=perc, method=method, w=w)
  
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  print("data balancing time: ")
  print(time.taken)
  
  balancedData <- cbind(data$X,data$Y)
  return (balancedData)
  
}