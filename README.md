# SPIDER: SPtial Interaction-encoDed intErface decipheR

SPIDER constructs cell-cell interaction interfaces with minimized Dirichlet energy, models interface profiles with knowledge-graph-informed interaction signals, and identifies spatially variable signals with multiple probabilistic models.

## Install
```shell
pip install spider-st
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


