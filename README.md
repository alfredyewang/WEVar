## WEVar: A novel statistical learning framework for predicting non-coding genetic variants

We propose WEVar, a new weighted ensemble learning framework, to predict and prioritize functional relevant non-coding variations. The WEVar framework integrates nine algorithms on functional annotation of non-coding variants by training a constrained logistic regression with the transformed score.
<center>

<div align=center><img width="1200" height="380" src="https://raw.githubusercontent.com/alfredyewang/WEVar/master/doc/WEVAR.jpg"/></div>
</center>  



## Requirements and Installation

WEVar is implemented by Python3.

- Python 3.6
- pyranges == 0.0.79
- numpy >= 1.15.4
- scipy >= 1.2.1
- scikit-learn >= 0.20.3
- seaborn >=0.9.0
- matplotlib >=3.1.0
- cvxpy >= 1.0.23
- cloudpickle >=1.1.1

Download [Base Score](https://drive.google.com/file/d/1Jwwuo01ZL1MlJHd1VVR0PaJdyI9MLgaz/view?usp=sharing)

```
unzip Base.zip
```


Download WEVar:
```
git clone https://github.com/alfredyewang/WEVar
```
Install requirements
```
pip3 install -r requirements
```
## Usage
You can see the input arguments for WEVar by help option:

```
usage: WEVar.py [-h] [--prediction] [--data_dir <data_directory>]
                [--test_file TEST_FILE] [--res_dir <data_directory>]
                [--Context CONTEXT] [--train] [--train_file TRAIN_FILE]
                [--num_fold NUM_FOLD] [--kde_kde_bandwidth KDE_KDE_BANDWIDTH]

WEVar: A novel statistical learning framework for predicting non-coding
genetic variants

optional arguments:
  -h, --help            show this help message and exit
  --prediction          Use this option for predict WEVar score
  --data_dir <data_directory>
                        The data directory for training and prediction
  --test_file TEST_FILE
                        The test SNPs file
  --res_dir <data_directory>
                        The data directory for save res
  --Context CONTEXT     Use this option for select Context. (HGMD, eQTL, GWAS,
                        Allele_imbanlace)
  --train               Use this option for train WEVar model
  --train_file TRAIN_FILE
                        The test SNPs file
  --num_fold NUM_FOLD   k-fold cross validation
  --kde_kde_bandwidth KDE_KDE_BANDWIDTH
                        the bandwidth of kde estimation

```

### Input File Format
WEVar takes UCSC Genome Browser BED file. The BED fields are:

- Chromosome: The name of the chromosome (e.g. chr3, chrY).
- Start:  The starting position of the feature in the chromosome.
- End:  The ending position of the feature in the chromosome.
- Labels: Functional or Benign

The first three are required for predicting WEVar score, and all are necessary for training WEVar model. Please see data/test_SNPs file as reference

### Predict WEVar Score

```
python3 src/WEVar.py --prediction --Context HGMD --data_dir data --test_file test_snps --res_dir res

```
The WEVar score will be saved res_dir folder. You may select different context (eg: HGMD, eQTL, GWAS, Allele_imbanlace)

### Train WEVar model

```
python3 src/WEVar.py --train --data_dir data --train_file CAGI --num_fold 10 --kde_kde_bandwidth 0.1

```
The program will train WEVar model based on k-fold cross validation, save all files to pdf directory, and draw plots.

<center>
<div align=center><img width="450" height="300" src="https://raw.githubusercontent.com/alfredyewang/WEVar/master/doc/CAGI.png"/></div>
</center>  
