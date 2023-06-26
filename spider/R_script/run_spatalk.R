library(SpaTalk)
# running SpaTalk 
# Shao, X., et al. Knowledge-graph-based cell-cell communication inference for spatially resolved transcriptomic data with SpaTalk. Nat Commun 13, 4429 (2022). https://doi.org/10.1038/s41467-022-32111-8
# Configurations used on SPIDER are described in Supplementary Methods > Screening ligand-receptor pairs based on the activation of downstream markers
set.seed(20230617)

args = commandArgs()

if (length(args)==0) {
  stop("not enough input", call.=FALSE)
}

count_f <- args[4]
meta_f <- args[5]
species <- args[6]
out_f <- args[7]

colData <- read.csv(meta_f, stringsAsFactors=FALSE, row.names=1, check.names=F)
counts <- read.csv(count_f, row.names=1, check.names=F, stringsAsFactors=FALSE)

obj <- createSpaTalk(st_data = t(as.matrix(counts)),
                     st_meta = colData[-4],
                     species = species,
                     if_st_is_sc = T,
                     spot_max_cell = 1,
                     celltype = colData$celltype)
obj <- find_lr_path(object = obj , lrpairs = lrpairs, pathways = pathways, if_doParallel = F, use_n_cores=1)

cellname <- unique(colData$celltype)

for (i in 1:length(cellname)) {
    try(obj <- dec_cci(object = obj, 
               celltype_sender = cellname[i],
               celltype_receiver =  cellname[i], 
               pvalue=0.1, n_neighbor = 20,
               if_doParallel = T,  use_n_cores=10))
}


try(obj <- dec_cci_all(object = obj, if_doParallel = T, pvalue=0.1, use_n_cores=10, n_neighbor = 20))


write.csv(obj@lrpair, paste0(out_f,"_lrpair.csv"), row.names = TRUE)
save(obj, file = paste0(out_f,"_spatalk.RData"))
