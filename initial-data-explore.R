data = read.csv("data/base3_imputed.txt", sep = "")

port.train = read.csv("data/port_train.csv")
port.all = read.csv("data/port_all.csv")
dim(port.all)

diff = colnames(data)[!(colnames(data) %in% colnames(port.train))]

grep("WEIGTH", colnames(port.train), value = TRUE)

diff.names = sub(".*\\.", "", diff)
diff.names[61] = "P.riskscip."

info = read.csv("data/PORT_var_info_all.csv")
dim(info)
colnames(info)
varnames = lapply(info["VAR.NAME"], as.character)
diff.info = info[unlist(varnames) %in% diff.names,]
dim(diff.info)
write.csv(diff.info, file = "data/diff-info.csv")

cooper.varnames = read.csv("data/Cooper_varnames_all.csv", stringsAsFactors = FALSE)

cp.vars = sapply(cooper.varnames$varname, tolower)
p.vars = sapply(sub(".*\\.", "", colnames(port.train)), tolower)

cooper.varnames$varname[!(cp.vars %in% p.vars)]

svc.prob = read.csv("output/svc_probs.csv", header = FALSE)

source("get_measures.R")
getMeasures(resp, rf.prob$V2)
getMeasures(resp, lr.prob$V2)
getMeasures(resp, svc.prob$V2)


