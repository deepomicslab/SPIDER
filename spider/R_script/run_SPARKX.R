library('SPARK')
args = commandArgs()
set.seed(20230617)

if (length(args)==0) {
  stop("not enough input", call.=FALSE)
}

count_f <- args[4]
meta_f <- args[5]
out_f <- args[6]

counts <- t(read.csv(count_f, row.names=1, check.names=F, stringsAsFactors=FALSE))
info <- read.csv(meta_f, stringsAsFactors=FALSE, row.names=1, check.names=F)
sparkx <- sparkx(as.matrix(counts),as.matrix(info[,1:2]),numCores=strtoi(args[7]),option="mixture")
write.csv(sparkx$res_mtest, paste0(out_f,"SPARKX.csv"), row.names = TRUE)