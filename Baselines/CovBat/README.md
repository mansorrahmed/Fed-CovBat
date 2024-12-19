# CovBat_Harmonization
### Correcting Covariance Batch Effects (CovBat): Harmonization of mean and covariance for multi-site data

--------
**Maintainer**: Andrew Chen, andrewac@pennmedicine.upenn.edu

**License**: Artistic License 2.0

**References**: If you are using CovBat, please cite the ComBat papers:

|       | Citation     | Paper Link
| -------------  | -------------  | -------------  |
| ComBat for multi-site DTI data    | Jean-Philippe Fortin, Drew Parker, Birkan Tunc, Takanori Watanabe, Mark A Elliott, Kosha Ruparel, David R Roalf, Theodore D Satterthwaite, Ruben C Gur, Raquel E Gur, Robert T Schultz, Ragini Verma, Russell T Shinohara. **Harmonization Of Multi-Site Diffusion Tensor Imaging Data**. NeuroImage, 161, 149-170, 2017  |[Link](https://www.sciencedirect.com/science/article/pii/S1053811917306948?via%3Dihub#!)| 
| ComBat for multi-site cortical thickness measurements    | Jean-Philippe Fortin, Nicholas Cullen, Yvette I. Sheline, Warren D. Taylor, Irem Aselcioglu, Philip A. Cook, Phil Adams, Crystal Cooper, Maurizio Fava, Patrick J. McGrath, Melvin McInnis, Mary L. Phillips, Madhukar H. Trivedi, Myrna M. Weissman, Russell T. Shinohara. **Harmonization of cortical thickness measurements across scanners and sites**. NeuroImage, 167, 104-120, 2018  |[Link](https://www.sciencedirect.com/science/article/pii/S105381191730931X)| 
| Original ComBat paper for gene expression array    |  W. Evan Johnson and Cheng Li, **Adjusting batch effects in microarray expression data using empirical Bayes methods**. Biostatistics, 8(1):118-127, 2007.      | [Link](https://academic.oup.com/biostatistics/article/8/1/118/252073/Adjusting-batch-effects-in-microarray-expression) |

If you are using CovBat for harmonization of mean and covariance, please cite the following preprint:

|       | Citation     | Paper Link
| -------------  | -------------  | -------------  |
| Original CovBat paper  | Andrew A. Chen, Joanne C. Beer, Nicholas J. Tustison, Philip A. Cook, Russell T. Shinohara, Haochang Shou, for the Alzheimer’s Disease Neuroimaging Initiative **Removal of Scanner Effects in Covariance Improves Multivariate Pattern Analysis in Neuroimaging Data**. BioRxiv 858415 [Preprint], December 2, 2019. Available from: https://doi.org/10.1101/858415. |[Link](https://www.biorxiv.org/content/10.1101/858415v1)| 

## Table of content
- [1. Installation](#id-section1)
- [2. Background](#id-section2)
- [3. Software](#id-section3)

<div id='id-section1'/>

## 1. Installation
This Python script can be loaded by moving `covbat.py` to your working directory and running

```
import covbat
```

<div id='id-section2'/>

## 2. Testing
From the original ComBat Python implementation ([https://github.com/brentp/combat.py](https://github.com/brentp/combat.py)), we include a script `test.py` to compare the R and Python implementations of both ComBat and CovBat. This script finds the maximum difference between the two implementations in the `bladderbatch` dataset and confirms that it is less than 10<sup>-4</sup>. You first need to run the R script `R-test.R`, which generates the input data from the `bladderbatch` package and harmonized outputs for the R implementations `r-combat.txt` and `r-covbat.txt`.

<div id='id-section3'/>

## 3. Software
The reference implementation (Standard Version) of ComBat, developed for gene expression analyses, is written in R and is part of the `sva` package available through the Bioconductor project [here](https://bioconductor.org/packages/release/bioc/html/sva.html). This package is an extension of the original ComBat method for harmonization of covariance for multivariate data. We use the same open-source license as the `sva` package, that is the Artistic License 2.0.

This Python script is adapted from a port of ComBat done by *brentp* on GitHub ([https://github.com/brentp/combat.py](https://github.com/brentp/combat.py))