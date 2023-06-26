library(nnSVG)
library(scran)
library(SpatialExperiment)

args = commandArgs()

if (length(args)==0) {
  stop("not enough input", call.=FALSE)
}

count_f <- args[4]
meta_f <- args[5]
out_f <- args[6]

counts <- read.csv(count_f, row.names=1, check.names=F, stringsAsFactors=FALSE)
colData <- read.csv(meta_f, stringsAsFactors=FALSE, row.names=1, check.names=F)
rowData <- data.frame(gene_name=colnames(counts))
head(rowData)
spe <-  SpatialExperiment(
    assay = list(counts = t(counts)), 
    colData = colData, 
    rowData = rowData,
    spatialCoordsNames = c("row", "col"))
spe <- filter_genes(spe)
spe <- computeLibraryFactors(spe)
spe <- logNormCounts(spe)

set.seed(20230617)
spe <- nnSVG(spe, n_threads = 1)

write.csv(rowData(spe), paste0(out_f,"nnSVG.csv"), row.names = TRUE)