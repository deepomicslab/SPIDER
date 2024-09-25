library(nnSVG)
library(scran)
library(SpatialExperiment)
library(BRISC)

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

tryCatch({
  spe <-  SpatialExperiment(
    assay = list(counts = t(counts)), 
    colData = colData, 
    rowData = rowData,
    spatialCoordsNames = c("row", "col"))
  ix_zero_genes <- rowSums(counts(spe)) == 0
  if (sum(ix_zero_genes) > 0) {
    spe <- spe[!ix_zero_genes, ]
  }

  spe <- computeLibraryFactors(spe)
  size_factors <- sizeFactors(spe)
  zero_sf_cells <- names(size_factors[size_factors == 0])
  spe <- spe[, !colnames(spe) %in% zero_sf_cells]
  spe <- logNormCounts(spe)
  # replace nan with 0
  counts(spe)[is.na(counts(spe))] <- 0

  # remove genes with zero expression
  ix_zero_genes <- rowSums(counts(spe)) == 0
  if (sum(ix_zero_genes) > 0) {
    spe <- spe[!ix_zero_genes, ]
  }
  # remove spots with zero expression
  ix_zero_spots <- colSums(counts(spe)) == 0
  if (sum(ix_zero_spots) > 0) {
    spe <- spe[, !ix_zero_spots]
  }
  set.seed(20230617)
  spe <- nnSVG(spe, n_threads = strtoi(args[7]))
  write.csv(rowData(spe), paste0(out_f,"nnSVG.csv"), row.names = TRUE)

}, error = function(e) {
  print(e)
  print('retrying with pseudo count')
  tryCatch({
    spe <-  SpatialExperiment(
      assay = list(counts = t(counts)+1), 
      colData = colData, 
      rowData = rowData,
      spatialCoordsNames = c("row", "col"))
    ix_zero_genes <- rowSums(counts(spe)) == 0
    if (sum(ix_zero_genes) > 0) {
      spe <- spe[!ix_zero_genes, ]
    }
    spe <- computeLibraryFactors(spe)
    size_factors <- sizeFactors(spe)
    zero_sf_cells <- names(size_factors[size_factors == 0])
    spe <- spe[, !colnames(spe) %in% zero_sf_cells]
    spe <- logNormCounts(spe)
    # replace nan with 0
    counts(spe)[is.na(counts(spe))] <- 0

    # remove genes with zero expression
    ix_zero_genes <- rowSums(counts(spe)) == 0
    if (sum(ix_zero_genes) > 0) {
      spe <- spe[!ix_zero_genes, ]
    }
    # remove spots with zero expression
    ix_zero_spots <- colSums(counts(spe)) == 0
    if (sum(ix_zero_spots) > 0) {
      spe <- spe[, !ix_zero_spots]
    }

    set.seed(20230617)
    spe <- nnSVG(spe, n_threads = strtoi(args[7]))
    write.csv(rowData(spe), paste0(out_f,"nnSVG.csv"), row.names = TRUE)
  }, error = function(e) {
    print(e)
    print('retrying with single run')
    spe <-  SpatialExperiment(
      assay = list(counts = t(counts)+1), 
      colData = colData, 
      rowData = rowData,
      spatialCoordsNames = c("row", "col"))
    ix_zero_genes <- rowSums(counts(spe)) == 0
    if (sum(ix_zero_genes) > 0) {
      spe <- spe[!ix_zero_genes, ]
    }
    spe <- computeLibraryFactors(spe)
    size_factors <- sizeFactors(spe)
    zero_sf_cells <- names(size_factors[size_factors == 0])
    spe <- spe[, !colnames(spe) %in% zero_sf_cells]
    spe <- logNormCounts(spe)
    # replace nan with 0
    counts(spe)[is.na(counts(spe))] <- 0

    # remove genes with zero expression
    ix_zero_genes <- rowSums(counts(spe)) == 0
    if (sum(ix_zero_genes) > 0) {
      spe <- spe[!ix_zero_genes, ]
    }
    # remove spots with zero expression
    ix_zero_spots <- colSums(counts(spe)) == 0
    if (sum(ix_zero_spots) > 0) {
      spe <- spe[, !ix_zero_spots]
    }

    spatial_coords = NULL
    X = NULL
    assay_name = "logcounts"
    n_neighbors = 10
    order = "AMMD"
    n_threads = 1
    BPPARAM = NULL
    verbose = FALSE
    y <- assays(spe)[[assay_name]]
    coords <- spatialCoords(spe)
    range_all <- max(apply(coords, 2, function(col) diff(range(col))))
    coords <- apply(coords, 2, function(col) (col - min(col)) / range_all)
    order_brisc <- BRISC_order(coords, order = order, verbose = verbose)
    nn_brisc <- BRISC_neighbor(coords, n.neighbors = n_neighbors, n_omp = 1, 
                              search.type = "tree", ordering = order_brisc, 
                              verbose = verbose)
    out_brisc <- list()
    genes <- list()
    for(i in seq_len(nrow(y))) {
      y_i <- y[i, ]
      suppressWarnings({
        runtime <- system.time({
          tryCatch({
            out_i <- BRISC_estimation(coords = coords, y = y_i, x = X, 
                                      cov.model = "exponential",
                                      ordering = order_brisc, neighbor = nn_brisc,
                                      verbose = verbose)
            out_i["sigma.sq"] <-out_i$Theta["sigma.sq"]
            out_i["tau.sq"] <- out_i$Theta["tau.sq"]
            out_i["loglik"] <- out_i$log_likelihood
            out_brisc[[i]] <- out_i
            genes[i] <- i
          }, error = function(e) {
            print(e)
        })
      })
      })}
    gene_idx <- unlist(genes)
    out_brisc_sub <- out_brisc[gene_idx]
    mat_brisc <- do.call("rbind", out_brisc_sub)
    genes <- rownames(spe)[gene_idx]
    spe <- spe[rownames(spe) %in% genes, ]
    lc <- logcounts(spe)
    mat_brisc <- cbind(
      mat_brisc, 
      mean = rowMeans(lc), useNames = TRUE
    )


    sigmasq <- unlist(mat_brisc[, "sigma.sq"])
    tausq <- unlist(mat_brisc[, "tau.sq"])
    meanarr <- unlist(mat_brisc[, "mean"])
    mat_brisc <- cbind(
      mat_brisc, 
      spcov = sqrt(sigmasq) / meanarr, useNames = TRUE
    )
    mat_brisc <- cbind(
      mat_brisc, 
      prop_sv = sigmasq / (sigmasq + tausq), useNames = TRUE
    )
    nrows <- nrow(spe)
    ncols <- ncol(spe)
    loglik_lm <- vapply(seq_len(nrows), function(i) {
      y_i <- y[i, ]
      if (is.null(X)) {
        X <- rep(1, ncols)
      }
      as.numeric(logLik(lm(y_i ~ X - 1)))
    }, numeric(1))
    mat_brisc <- cbind(
      mat_brisc, 
      loglik_lm = loglik_lm, useNames = TRUE
    )
    loglik_lmarr = unlist(mat_brisc[, "loglik_lm"])
    loglik_arr = unlist(mat_brisc[, "loglik"])
    LR_stat <- -2 * (loglik_lmarr - loglik_arr)
    pval <- 1 - pchisq(LR_stat, df = 2)
    padj <- p.adjust(pval, method = "BH")
    LR_rank <- rank(-1 * LR_stat)
    mat_brisc <- cbind(
      mat_brisc, 
      LR_stat = LR_stat, 
      rank = LR_rank, 
      pval = pval, 
      padj = padj, useNames = TRUE
    )
    stopifnot(nrow(spe) == nrow(mat_brisc))
    rowData(spe) <- cbind(rowData(spe), mat_brisc)
    cols <- c("gene_name","sigma.sq","tau.sq","loglik","mean","spcov","prop_sv","loglik_lm","LR_stat","rank","pval","padj")
    df <- as.data.frame(rowData(spe))[cols]
    library(data.table)
    fwrite(df, paste0(out_f,"nnSVG.csv"))
  })
})


