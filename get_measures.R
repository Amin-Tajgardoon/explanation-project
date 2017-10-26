
getMeasures = function(YTE, PTE ){
  #   GETMEASURES Summary of this function goes here
  if (sum(!is.finite(PTE))>0){
    print('there are some nan value in predictions')
  }
  
  if (sum(!is.finite(YTE))>0){
    print('there are some nan value in predictions')
  }
  
  
  idx = is.finite(YTE)&is.finite(PTE);
  YTE = YTE[idx]; PTE = PTE[idx];
  
  res = list();
  res$RMSE = getRMSE(YTE,PTE);
  res$AUC = getAUC(YTE,PTE);
  res$ACC = sum(YTE==(PTE>=0.5))/length(YTE);    
  res$MCE = getMCE(YTE,PTE); #Computing the Max Calibratio Error among all bins 
  res$ECE = getECE(YTE,PTE); #Computing Average Calinration Error on different binns
  res$R2 = getR2(YTE, PTE)
  PRF = getPRF(YTE, PTE)
  res$PREC = PRF$prec
  res$RECALL = PRF$rec
  res$F1 = PRF$f1
  return(res)
}

getAUC = function(Actual,Predicted){
  stopifnot( (length(unique(Actual))==2)&(max(unique(Actual))==1))
  nTarget     = sum(Actual == 1);
  nBackground = sum(Actual != 1);
  # Rank data
  R = rank(Predicted, ties.method = "average");  # % 'tiedrank' from Statistics Toolbox  
  #   Calculate AUC
  AUC = (sum(R[Actual == 1]) - (nTarget^2 + nTarget)/2) / (nTarget * nBackground);
  AUC = max(AUC,1-AUC);
  return(AUC)
}

getMCE = function ( Y, P, nBin = 10 ){
  predictions = P;
  labels = Y;
  sortObj = sort.int(predictions, index.return=TRUE)
  predictions = sortObj$x
  labels = labels[sortObj$ix]
  ordered = cbind(predictions,labels)
  N = length(predictions);
  rest = N%%nBin;
  S=rep(0,nBin)
  for (i in 1:nBin){
    if (i <= rest){
      startIdx = as.integer((i-1) * ceiling(N / nBin) + 1)
      endIdx = as.integer(i * ceiling(N / nBin))
    }else{
      startIdx = as.integer(rest + (i-1)*floor(N/nBin)+1)
      endIdx = as.integer(rest + i*floor(N/nBin))    
    }
    group = ordered[startIdx:endIdx,];
    
    n = dim(group)[1];
    observed = mean(group[,2]);
    expected = mean(group[,1]);
    S[i] = abs(expected-observed);
  }
  res = max(S);
  return(res)
}

getECE = function ( Y, P, nBin = 10 ){
  predictions = P;
  labels = Y;
  sortObj = sort.int(predictions, index.return=TRUE)
  predictions = sortObj$x
  labels = labels[sortObj$ix]
  ordered = cbind(predictions,labels)
  N = length(predictions);
  rest = N%%nBin;
  S=rep(0,nBin)
  W=rep(0,nBin)
  for (i in 1:nBin){
    if (i <= rest){
      startIdx = as.integer((i-1) * ceiling(N / nBin) + 1)
      endIdx = as.integer(i * ceiling(N / nBin))
    }else{
      startIdx = as.integer(rest + (i-1)*floor(N/nBin)+1)
      endIdx = as.integer(rest + i*floor(N/nBin))    
    }
    group = ordered[startIdx:endIdx,];
    
    n = dim(group)[1];
    observed = mean(group[,2]);
    expected = mean(group[,1]);
    S[i] = abs(expected-observed);
    W[i] = n/N;
  }
  res = sum(S*W);
  return(res)
}

getRMSE = function( Y, P ){
  res = sqrt(sum((Y-P)*(Y-P))/length(Y));
}

getR2 <- function(Y, P){
  y_bar = mean(Y)
  ss_tot = sum( (Y - y_bar)**2 )
  ss_res = sum( (P - Y)**2 )
  r2 = 1 - (ss_res/ss_tot)
  return(r2)
}

getPRF <- function(Y, P){
  confmat = getConfMat(Y,P)
  prec = confmat$tp / (confmat$tp + confmat$fp)
  rec = confmat$tp / (confmat$tp + confmat$fn)
  f1 = (2*prec*rec)/(prec + rec)
  prf = list()
  prf$prec = prec
  prf$rec = rec
  prf$f1 = f1
  return(prf)
}

getConfMat <- function(Y, P){
  conf.mat = list();
  conf.mat$tp = sum(Y[which(P >= .5)] == 1)
  conf.mat$fp = sum(Y[which(P >= .5)] == 0)
  conf.mat$fn = sum(Y[which(P < .5)] == 1)
  conf.mat$tn = sum(Y[which(P < .5)] == 0)
  return(conf.mat)
}

calib.xy = function ( Y, P, nBin = 10 ){
  predictions = P;
  labels = Y;
  sortObj = sort.int(predictions, index.return=TRUE)
  predictions = sortObj$x
  labels = labels[sortObj$ix]
  ordered = cbind(predictions,labels)
  N = length(predictions);
  rest = N%%nBin;
  observed=rep(0,nBin)
  expected=rep(0,nBin)
  for (i in 1:nBin){
    if (i <= rest){
      startIdx = as.integer((i-1) * ceiling(N / nBin) + 1)
      endIdx = as.integer(i * ceiling(N / nBin))
    }else{
      startIdx = as.integer(rest + (i-1)*floor(N/nBin)+1)
      endIdx = as.integer(rest + i*floor(N/nBin))    
    }
    group = ordered[startIdx:endIdx,];
    
    # n = dim(group)[1];
    observed[i] = mean(group[,2]);
    expected[i] = mean(group[,1]);
    # S[i] = abs(expected-observed);
  }

  res = data.frame(observed, expected)
  res.sorted = res[ order(res$observed,decreasing = FALSE), ]
  return(res.sorted)
}


getCE <- function(df, pos_class, row.index, P){
  
  df = df[row.index, ]
  y_frac = getActualFrac(df,pos_class)
  CE = mean(abs(p-y_frac))
  return(CE)
}

getActualFrac = function(df, pos_class){
  
  y_frac = apply(df,1, function(x){
            sum(apply(df, 1, function(z) all(z == c(x[1:ncol(df)-1],pos_class))))/
              sum(apply(df, 1, function(z) all(z[-ncol(df)] == x[-ncol(df)])))
  })
  
  return(y_frac)
}

getZscore = function(PTE, YTE){
  
  PTE[PTE == 0] = 1e-4
  PTE[PTE == 1] = 1 - 1e-4
  
  E = sum(expectation(PTE))
  
  V = sum(variance(PTE))
  
  S = sum(- (log(PTE)*YTE + log(1-PTE)*(1-YTE)))
  
  Z = (S-E)/sqrt(V)
  pvalue = pnorm(abs(Z), lower.tail = FALSE)
  
  return (list(z = Z, pvalue = pvalue))
}


expectation = function(p){
  return (- (p*log(p) + (1-p)*log(1-p)))
}

variance = function(p){
  v= (p*log(p)^2 + (1-p)*log(1-p)^2) - expectation(p)^2
  return (v)
}

getLocalScore = function(fitted, target, DTE){
  t = fitted[[target]]
  parents = t$parents
  responseName = "prob"
  probs = as.data.frame(x = t$prob, responseName = responseName)
  if(ncol(probs) == 2)
    colnames(probs) <- c(target, responseName)
  p = localProbs(DTE, probs,vars = c(target, parents))
  
  S = - sum(log(p))
  E = sum(expectation(p))
  V = sum(variance(p))
  Z = (S-E)/sqrt(V)
  pvalue = pnorm(abs(Z), lower.tail = FALSE)
  
  return (list(z = Z, pvalue = pvalue))
  
}

buildCondition = function(vars, d){
  cond = ""
  for(v in vars){
    c = paste0(v, "==", "\"", d[,v], "\"")
    if(cond == "")
      cond = c
    else
      cond = paste(cond,c,sep = " & ")
  }
  
  return(parse(text = cond))
}

localProbs = function(DTE, probs, vars){
  probs = sapply(1:nrow(DTE), function(i){
    c = buildCondition(vars, DTE[i,])
    with(probs, probs[eval(c), ncol(probs)])
  })
  
  return(probs)
}