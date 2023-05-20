library(SpaTalk)


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
               pvalue=0.1, 
               if_doParallel = T,  use_n_cores=1))
}

# n_neis <- c(5, 10, 15, 20, 25, 30)
# pvals <- c(0.01, 0.05, 0.1, 0.15, 0.2)
# min_p <- c(2, 5, 10)
# co_exp <- c(0.05, 0.1, 0.2, 0.5)
''
# for (i in 1:length(n_neis)) {
#   for (j in 1:length(pvals)) {
#     for (k in 1:length(min_p)) {
#       for (l in 1:length(co_exp)) {
#         print(paste(pvals[j],n_neis[i], min_p[k], co_exp[l]))
#         try(obj <- dec_cci(object = obj, 
#                     celltype_sender = "Stroma",
#                     celltype_receiver =   "Stroma", 
#                     pvalue=pvals[j], n_neighbor = n_neis[i],
#                     min_pairs = min_p[k], co_exp_ratio = co_exp[l],
#                     if_doParallel = F))
#       }
#     }
#   }
# }

try(obj <- dec_cci_all(object = obj, if_doParallel = T,  use_n_cores=1))
# obj <- dec_cci_all(object = obj, pvalue=0.1, if_doParallel = F,  use_n_cores=10)

# celltype_pair <- NULL
# for (i in 1:length(cellname)) {
#     d1 <- data.frame(celltype_sender = rep(cellname[i], length(cellname)), celltype_receiver = cellname,
#         stringsAsFactors = F)
#     celltype_pair <- rbind(celltype_pair, d1)
# }

# tf_path <- NULL
# path_pvalue <- NULL

# for (i in 1:nrow(celltype_pair)) {
#     celltype_sender <- celltype_pair$celltype_sender[i]
#     celltype_receiver <- celltype_pair$celltype_receiver[i]
#     try({obj_lr_path <- get_lr_path(object = obj, celltype_sender = celltype_sender, celltype_receiver = celltype_receiver)
#     tf_path <- rbind(tf_path, obj_lr_path$tf_path)
#     path_pvalue <- rbind(path_pvalue, obj_lr_path$path_pvalue)})
# }

write.csv(obj@lrpair, paste0(out_f,"_lrpair.csv"), row.names = TRUE)
save(obj, file = paste0(out_f,"_spatalk.RData"))
# write.csv(tf_path, paste0(out_f,"_tfpath.csv"), row.names = TRUE)
# write.csv(path_pvalue, paste0(out_f,"_pathpvalue.csv"), row.names = TRUE)
