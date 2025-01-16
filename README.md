# yeastlifespan

The yeast genome has around 6400 genes. After several decades of research, it is still not fully known which genes compose the elusive gene regulatory network governing yeast lifespan (either at the single-cell level or at the population level). However, as for many things in genetics/biology, lifespan regulation should occur through concerted efforts of a gene set. As a side note: the genes of this gene set, whatever they are, are almost certainly part of other known pathways (associated with other phenotypes) in yeast, because many genes are usually shared among many pathways (and that’s OK). I use the terms pathway and network interchangibly in this document.

If some genes come together and compose the network regulating/governing single-cell lifepan, then there are two implicit expectations here: the specific expression levels of those genes are important for lifespan regulation (i.e. lifespan regulation is dependent on the specific expression levels of those genes) and the structure/connectivity or wiring diagram among those genes is important (i.e. which gene is up/down-regulated by which other gene(s) in the network governing single-cell lifespan).

Therefore what we need to know to solve this puzzle (i.e. predict the single-cell lifespan based on the expression levels of the genes governing lifespan, instead of performing the tedious lifespan-measurement experiment under the microscope over multiple days):

1. The names of the genes composing the network governing lifespan (which ones, out of the 6400 total genes?)  
2. The wiring diagram (network structure) of the network governing lifespan  
3. The expression levels of the genes composing the network governing lifespan

Now, while there are some available datasets as explained below, there is no data set or knowledge about:

1. how gene expression level of a gene changes during the course of aging of a single cell. On the other hand, what we have is: gene expression level of a gene when the cell is young. This may not be a huge problem because we could build the model by taking young-cells gene expresssion levels as input. Also there is the concept of gene expression memory across aging, so young gene expresssion levels are still valuable. The final say will be said be the model based on lifespan-predictive capability, anyways.  
2. No one has the information about how the network wiring structure (of the network governing lifespan) changes during the aging of a cell. On the other hand, we have this information on young cells (the third dataset mentioned below). I think the assumption of no-change-in-wiring-diagram during aging is a good one, because the opposite would be too extreme (i.e. protein structure, protein-protein interactions and protein-DNA interactions changing during aging); in other words, we assume that gene expression changes are the main drivers of aging and the resulting lifespan. The fact that old cells still divide and show gene expression signatures not deviating in extreme manners from young cells support the assumption that there is no-change-in-wiring-diagram during aging.   
   

 After this introduction, here is further information on the three key datasets (the fourth dataset is redundant):

**processed\_lifespan\_data:** This file is a processed/simplified version of the McCormick dataset (therefore the mentioned redundancy). In the unprocessed dataset is named lifespan\_mccormick\_data. 

Here we see average lifespan values from each of the 4633 strains when one of the 4633 genes is deleted from the yeast genome. When no gene is deleted, the strain is called wild-type; “strain” is just a nomenclature indicating the resulting cell type when a genetic modification is made on the wild-type (WT). In the processed data file, you can see the deleted gene’s name on each row under the column name “set\_genotype”.

The CV values are not very trustworthy in this dataset because the means are calculated for each strain from the experimental lifespan analysis of 20-40 single cells (low numberr of cells). We will need to do something to deal with this situation (for example, by synthetic data generation based on the emirical mean values for each strain; maybe by generating a continuous gaussian based on the mean and SD of the single-cell lifespan data coming from 20-40 cells and then randomly sampling thousands of semi-synthetic single-cell lifespan values by making sure to exclude the outlier cell lifespan of the 20-40 single cells)


**expression\_log\_fold\_changes:** In this file, the rows under the “DELETED\_GENE” column represent the deleted gene from the yeast genome (one gene is deleted at a time). We have around 1500 gene deletions in this manner. The columns, on the other hand, cover the genes of the entire yeast genome (6000+). The entries of this matrix: for a given entry (i, j), we have the expression log-fold change (LFC) that gene j encounters when gene i is deleted (the “change” is given relative to the wild-type). Put more simply, for a given row (meaning a single gene deletion), we have the expression change information for the rest of the genes in the yeast genome. For a deletion gene i, LFC is measured by: 

LFC = log(expression of gene j under deletion i / wildtype expression of gene j)

Note that this matrix is not square (1500 x 6400\) because the experimenters collcted data from only 1500 yeast strains when 1500 genes are deleted, one gene at a time. Overall, this dataset reveals expression dependencies between genes (in young cells as mentioned above).

Note that this is not a single-cell level dataset, but it’s population level meaning many cells were pooled and broken to measure their average expression levels for all genes when a gene is deleted from the yeast genome. 

**interaction\_strengths\_of\_yeastgenes:** This file covers the whole genome, and carries a square matrix (6400 x 6400). Both row and columns show gene names. Its entries show experimentally obtained interaction strengths between yeast genes. This dataset could be used to construct the wiring diagram (network structure) of the network governing yeast lifespan. Yet another independent dataset also exist, showing which transcription factor (TF) protein activates/represses which gene.

These two datasets do not have to be at the single-cell level because these interaction strangths are meant to be at the molecular level (protein-protein or protein-DNA).

**Final note:** In addition to these datasets, we also have single-cell gene expression data obtained from thousands of individual wild-type cells. The only caveat is that we don’t have this single-cell resolution data obtained from gene-deleted strains. Otherwise, we know the gene expression levels of \*all\* yeast genes in the wild-type genetic background, obtained at single-cell resolution. Yiğit is currently in the process of generating/analyzing this dataset. Deep learning based modeling approaches usually work better with single-cell resolution datasets, instead of bulk/average value representing each gene’s expression in the yeast genome.       

**Addendum:** 300+ yeast straininden toplanmış single cell lifespan data, aşağıdaki makalemizin Supplementary File S2 adlı online dosyasında mevcut. Makale de open access.

https://www.nature.com/articles/s41467-023-43233-y
