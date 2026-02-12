![](figures/Carbide.png)

![repo version](https://img.shields.io/badge/Version-v.%201.0.1-green)
![python version](https://img.shields.io/badge/python-v.3.11-blue)
![license](https://img.shields.io/badge/license-CC_BY_4.0-orange)

Repository accompanying paper "Beyond Molecular Structure: Investigating Demographic Factors in Drug-Induced Cardiotoxicity Prediction Models". https://doi.org/10.26434/chemrxiv.10001782/v1

## Prerequisites
To download and run the code make sure you have installed the following:
* miniforge https://github.com/conda-forge/miniforge
* git https://github.com/git-guides/install-git

## Cloning the repository
Enter command line and run: <div>
`` git clone https://github.com/M-Iwan/CARBIDE ``

## Installing virtual environments
The base environment can be installed using: <div>
`` conda env create -f environment.yml ``
To install an environment with all required dependencies: <div>
`` conda env create -f environment-full.yml ``

And activated using: <div>
`` conda activate carbide ``

## Optional dependencies
The following require additional packages / setup:
* Using the class for processing drug descriptions requires additional libraries: pubchempy, Levenshtein, rapidfuzz
* Calculating ChemBERTA embeddings requires additional libraries: transformers, torch
* Calculating CDDD embeddings requires creating a separate environment. The setup files are provided in src/cddd.

## Associated files
The prepared CARBIDE dataset is available in the repository under data/carbide in form of: <div>
* .joblib files: compatibile with the rest of the code / notebooks
* .tsv files: for non-Python users / easier access

Additionally, the dataset, raw data, aggregated raw results, and other associated files are available at Zenodo for LTS: <div>
https://zenodo.org/records/18605832

## Running the code
We provide notebooks that allow to recreate the results of this study:
* 1_Mapping.ipynb - Preparation of mapping from raw drug descriptions to active ingredients
* 2_Datasets.ipynb - Preparation of CARBIDE variants: processing of FAERS data and Disproportionality Analysis
* 3_Calculations.ipynb - Model training and prediction differences
* 4_Analysis.ipynb - Aggregation and analysis of obtained data
* 5_Figures.ipynb - Plotting of figures

## Funding
This study was funded by the Horizon Europe funding programme, under the Marie Skłodowska-Curie Actions Doctoral Networks grant agreement “Explainable AI for Molecules - AiChemist” no. 101120466. This work used the Dutch national e-infrastructure with the support of the SURF Cooperative using grant no. EINF-14313 to MI. 

## How to cite
If you use this code or parts thereof, please cite the following paper: <div>
Iwan M., Roncaglioni A., and Grisoni F. "Beyond Molecular Structure: Investigating Demographic Factors in Drug-Induced Cardiotoxicity Prediction Models", https://doi.org/10.26434/chemrxiv.10001782/v1
