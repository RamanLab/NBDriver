# NBDriver
NBDriver(**N**EIGH**B**ORHOOD **Driver**) is a tool used to differentiate between driver and passenger mutations using features derived from the neighborhood sequences of somatic mutations.

## Table of Contents

- [Description](#description)
- [Overall Workflow of NBDriver](#overall-workflow-of-NBDriver)
- [Data](#data)
- [Dependencies](#dependencies)
- [Preprint Link](#links)

## Description
Using missense mutation data from experimental assays, we build a binary classifier by extracting features from the neighborhood sequences of driver and passenger mutations. Our key results are three-fold. First, we use generative models to derive the distances between the underlying probability estimates of the neighborhood sequences for the two classes of mutations. Then, we build robust classification models using repeated cross-validation experiments to derive the median values of the metrics designed to estimate the classification performances. Finally, we demonstrate our models’ ability to predict unseen coding mutations from independent test datasets derived from large mutational databases. 

## Overall Workflow of NBDriver
Raw nucleotide sequences were derived from the reference genome GRCH37. Then, seven feature representations, namely, TFIDF Vectorizer (*k*-mer sizes 2,3 and 4), Count Vectorizer (*k*-mer sizes 2,3 and 4) and One-hot encoding were used to convert the string-based features to numerical formats. This was followed by estimating the underlying probabilit distributions using kernel density estimation and repeated cross-validation experiments using Random Forests, KDE classifer and Extra Trees classifier. The final model (NBDriver) was obtained using a training set derived after removing all overalapping mutations with an independent test set published by Martelotto et al. Subsequent validation with four separate independent validation sets containing pathogenic data from landamrk studies was also performed to judge the ability of NBDriver in predicting unseen test instances. The overall workflow is summarized below.  
![workflow](https://user-images.githubusercontent.com/7888886/108251643-a94a4680-717d-11eb-9d8c-9cd92217a620.png)

## Data
Training data was derived from a study by Brown et al., where they published mutation data from experimental assays labelled as drivers/passengers.
> -- <cite>Brown AL, Li M, Goncearenco A, Panchenko AR (2019) Finding driver mutations in cancer: Elucidating the role of background mutational processes. PLOS Computational Biology 15(4): e1006981. https://doi.org/10.1371/journal.pcbi.1006981</cite>

## Dependencies
scikit-learn - 0.22.1
pandas - 0.25.3
numpy - 1.18.5
imblearn - 0.5.0
ggplot2 - 3.3.2

## Preprint Link
[Link](https://www.biorxiv.org/content/10.1101/2021.02.09.430460v1)
