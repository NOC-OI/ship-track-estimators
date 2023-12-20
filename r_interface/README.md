# Usage instructions

## 1. NOC servers

If running these scripts on NOC servers, ensure to load the necessary modules first:

```bash
module load R
module load gcc/11.2.0
```

## 2. Reticulate Installation

The only package required to execute the `.R` scripts in this directory is `reticulate`. Install `reticulate` from CRAN using either of the following methods:

### Command Line

```bash
Rscript -e "install.packages('https://cran.r-project.org/src/contrib/Archive/reticulate/reticulate_1.4.tar.gz', repos=NULL, type='source')"
```

### R Studio

```R
install.packages("reticulate")
```

Once installed, you can use the `track_estimators` package from within R.

## 3. Run the scripts

Before executing the scripts, ensure that your Conda or pyenv environment is activated:

```bash
conda activate shiptrack-estimators
# pyenv activate shiptrack-estimators
```

Run the scripts in this directory simply by using, for example:

```bash
Rscript ./example_ukf.r
```

# References

1. [Reticulate documentation](https://rstudio.github.io/reticulate/)
