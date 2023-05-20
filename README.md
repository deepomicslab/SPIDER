# SPIDER: SPtial Interaction-encoDed intErface decipheR

SPIDER constructs cell-cell interaction interfaces with minimized Dirichlet energy, models interface profiles with knowledge-graph-informed interaction signals, and identifies spatially variable signals with multiple probabilistic models.

## Install
For python version of SPIDER, simply call:
```shell
pip install spider-st
```

To also use the R packages in SPIDER, you need to first install:
```shell
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("nnSVG")

install.packages('devtools')
devtools::install_github('xzhoulab/SPARK')
devtools::install_github('linxihui/NNLM')
devtools::install_github('ZJUFanLab/SpaTalk')
```

## Quick Start

### Run `spider.SPIDER`
This example shows the usage of `spider.SPIDER`.

```python
import anndata 
import spider
op=spider.SPIDER()

# interface consutrction
idata = op.prep(adata, out_f, cluster_key='region', is_human=True, coord_type='grid')

# running SVI tests
idata, abstract_idata = op.find_svi(idata, out_f)

# visualize SVI
op.vis.pattern_LRI(idata,show_SVI=10)
```

Outputs:

![Metrics](https://github.com/deepomicslab/SPIDER/raw/main/demo/human_pdac_st_patterns.png)

```python
# combine SVI with p-value threshold
svi_df, svi_df_strict = op.svi.combine_SVI(idata,threshold=0.01)

# plot evaluation metrics
op.svi.eva_SVI(idata, svi_df_strict)
```

Outputs:

![Metrics](https://github.com/deepomicslab/SPIDER/raw/main/demo/PDAC_metric.png)

```python
# transform SVI pattern from interfaces to spots
op.svi.idata_pattern_to_spot(idata)

# SVI-based spot clustering
op.cl.unsupervised_spot_clust(idata, adata, n_cluster=7)
```

Outputs:

![Metrics](https://github.com/deepomicslab/SPIDER/raw/main/demo/PDAC_boundary.png)



