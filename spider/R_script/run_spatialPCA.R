library(SpatialPCA)
set.seed(20230617)
args = commandArgs()

if (length(args)==0) {
  stop("not enough input", call.=FALSE)
}

colData <- read.csv(args[4], stringsAsFactors=FALSE, row.names=1, check.names=F)
xy_coords = as.matrix(colData)
counts <- t(read.csv(args[5], row.names=1, check.names=F, stringsAsFactors=FALSE))
rownames(xy_coords) = colnames(counts)
rownames(counts) = gsub("_", "-", rownames(counts))
customGenelist = rownames(counts)

LIBD = CreateSpatialPCAObject(counts=counts, location=xy_coords, project = "SpatialPCA",gene.type="custom",
                              customGenelist=customGenelist,min.loctions = 20, min.features=20)

LIBD = SpatialPCA_buildKernel(LIBD, kerneltype="gaussian", bandwidthtype="SJ",bandwidth.set.by.user=NULL)
LIBD = SpatialPCA_EstimateLoading(LIBD,fast=FALSE,SpatialPCnum=strtoi(args[10])) 
LIBD = SpatialPCA_SpatialPCs(LIBD, fast=FALSE)

clusterlabel= walktrap_clustering(clusternum=strtoi(args[6]),latent_dat=LIBD@SpatialPCs,knearest=strtoi(args[7]))
clusterlabel_refine = refine_cluster_10x(clusterlabels=clusterlabel,location=LIBD@location,shape="square")
fname=args[8]
write.csv(cbind(LIBD@location, clusterlabel), paste(fname, "interface_label.csv", sep=""))
write.csv(cbind(LIBD@location, clusterlabel_refine), paste(fname, "refined_interface_label.csv", sep=""))
write.csv(LIBD@SpatialPCs, paste(fname, "interface_SpatialPCs.csv", sep=""), row.names=TRUE)
library(slingshot)
sim<- SingleCellExperiment(assays = counts)
reducedDims(sim) <- SimpleList(DRM = t(LIBD@SpatialPCs))
colData(sim)$clusterlabel <- factor(clusterlabel)    
sim <-slingshot(sim, clusterLabels = 'clusterlabel', reducedDim = 'DRM',start.clus=clusterlabel[strtoi(args[9])]) 
pseudotime_traj1 = slingAvgPseudotime(sim)
gridnum = 10
cbp = c( "#FD7446" ,"#709AE1", "#31A354","#9EDAE5",
        "#DE9ED6" ,"#BCBD22", "#CE6DBD" ,"#DADAEB" ,
        "yellow", "#FF9896","#91D1C2", "#C7E9C0" ,
        "#6B6ECF", "#7B4173" )
p_traj1 = plot_trajectory(pseudotime_traj1, LIBD@location,clusterlabel,gridnum,cbp,pointsize=5 ,arrowlength=0.2,arrowsize=1,textsize=15 )
write.csv(cbind(LIBD@location, pseudotime_traj1), paste(fname, "interface_pseudotime.csv", sep=""))
write.csv(p_traj1$Arrowplot1$layers[[1]]$data, paste(fname, "interface_arrow.csv", sep=""))
colData(sim)$clusterlabel_refine <- factor(clusterlabel_refine)    
sim <-slingshot(sim, clusterLabels = 'clusterlabel_refine', reducedDim = 'DRM',start.clus=clusterlabel_refine[strtoi(args[9])]) 
pseudotime_traj1 = slingAvgPseudotime(sim)
p_traj1 = plot_trajectory(pseudotime_traj1, LIBD@location,clusterlabel_refine,gridnum,cbp,pointsize=5 ,arrowlength=0.2,arrowsize=1,textsize=15 )
write.csv(cbind(LIBD@location, pseudotime_traj1), paste(fname, "refined_interface_pseudotime.csv", sep=""))
write.csv(p_traj1$Arrowplot1$layers[[1]]$data, paste(fname, "refined_interface_arrow.csv", sep=""))