# SPIDER: SPtial Interaction-encoDed intErface decipheR

Spatial transcriptomics has emerged as a groundbreaking tool to study ligand-receptor interactions between cells, and these interactions exhibit spatial variability. To identify spatially variable ligand-receptor interactions with activation evidence, we present SPIDER, which constructs cell-cell interaction interfaces constrained by cellular interaction capacity, and profiles and identifies spatially variable interaction signals with support from downstream transcript factors and multiple probabilistic models. SPIDER exhibited consistent performance over seven datasets from four platforms of various tissues. 

## 
> [!IMPORTANT]  
> Try SPIDER on CodeOcean at **[Capsule 3209985](https://codeocean.com/capsule/3209985/tree)** now - no installation required!

## Install
First, creast a conda enviroment with python=3.8
```shell
conda create -n spider python=3.8
conda activate spider
conda install -c conda-forge somoclu fa2
```
Make sure you have scgco installed with 
```shell
pip install Cython
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=TRUE \ 
    pip install sklearn
pip install scgco
```

If you have problem installing the pygco package required by scgco, try clone the pygco repo, in which you need to change the name in the setup function in setup.py from gco-wrapper to pygco, and install pygco with setup.py
```shell
git clone https://github.com/Borda/pyGCO.git
cd pyGCO
<!-- change name in setup.py from gco-wrapper to pygco -->
pip install -r requirements.txt
python setup.py install
```

Then use pip to install spider-st:
```shell
pip install spider-st
```

To also use the R packages in SPIDER, you need to first install:
```shell
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("SpatialExperiment")
BiocManager::install("scran")
BiocManager::install("nnSVG")

install.packages('devtools')
devtools::install_github('xzhoulab/SPARK')
```

You also need to provide the executive R path in your enviroment, you can normally obtain this by checking the output of
```shell
which R
```

## Quick Start

### Run `spider.SPIDER`
This example shows the usage of `spider.SPIDER`.

```python
import anndata 
import spider
op=spider.SPIDER()

# interface consutrction
idata = op.prep(adata, cluster_key=adata.uns['cluster_key'], is_human=adata.uns['is_human'], is_sc=adata.uns['is_sc'], itermax=1000, imputation=True, normalize_total=True)

# TF scoring
op.svi.tf_corr(idata, adata, adata.uns['is_human'], out_f, threshold=0.3)

# running SVI tests
idata, meta_idata = op.find_svi(idata, out_f, R_path, alpha=0.3, overwrite=True, n_jobs=1, svi_number=0)

# visualize SVI
op.vis.pattern_LRI(idata,show_SVI=10, spot_size=10)
```

Outputs:

![Metrics](https://github.com/deepomicslab/SPIDER/raw/main/demo/demo_pattern.png)

```python
# combine SVI with p-value threshold
svi_df, svi_df_strict = op.svi.combine_SVI(idata,threshold=0.01)

# plot evaluation metrics
op.svi.eva_SVI(idata, svi_df_strict)
```

Outputs:

![Metrics](https://github.com/deepomicslab/SPIDER/raw/main/demo/demo_eva_svi.png)

```python
# plot correlations of patterns and member SVIs
op.svi.eva_pattern(idata)
```

Outputs:

![Metrics](https://github.com/deepomicslab/SPIDER/raw/main/demo/demo_pattern_eva.png)


Check the correlations between SVIs and deconvoluted celltypes:

![Metrics](https://github.com/deepomicslab/SPIDER/raw/main/demo/demo_celltypes.png)


