## WEVar: A novel statistical learning framework for predicting non-coding genetic variants

We propose WEVar, a new weighted ensemble learning framework, to predict and prioritize functional relevant non-coding variations. The WEVar framework integrates nine algorithms on functional annotation of non-coding variants by training a constrained logistic regression with the transformed score.
<center>

<div align=center><img width="1200" height="400" src="https://raw.githubusercontent.com/alfredyewang/WEVar/master/doc/WEVAR.jpg"/></div>
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

Download [Base Score](https://indiana-my.sharepoint.com/personal/yw146_iu_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyw146%5Fiu%5Fedu%2FDocuments%2FBase%2Ezip&parent=%2Fpersonal%2Fyw146%5Fiu%5Fedu%2FDocuments&originalPath=aHR0cHM6Ly9pbmRpYW5hLW15LnNoYXJlcG9pbnQuY29tLzp1Oi9nL3BlcnNvbmFsL3l3MTQ2X2l1X2VkdS9FVlZ3N3FFckNXSkZsMVhHTkVTSnN4Z0JtU0dqV0JWN1ZESXV1czZsNWZJdmZRP3J0aW1lPWgzVHg5VFFTMkVn)

```
unzip Base.zip Base
```


Download WEVar:
```
git clone https://github.com/alfredyewang/WEVar
```
Install requirements
```
pip3 install -r requirements.txt
```
## Usage
You can see the input arguments for WEVar by help option:

```
usage: WEVar.py [-h] [--prediction] [--data_dir <data_directory>]
                [--test_file TEST_FILE] [--res_dir <data_directory>]
                [--Context CONTEXT] [--train] [--train_file TRAIN_FILE]

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


```

## Input File Format
WEVar takes UCSC Genome Browser BED file. The BED fields are:

- Chromosome  The name of the chromosome (e.g. chr3, chrY, chr2_random) or scaffold (e.g. scaffold10671).
- Start The starting position of the feature in the chromosome or scaffold. The first base in a chromosome is numbered 0.
- End The ending position of the feature in the chromosome or scaffold.
- Labels Functional or Benign

The first three are required for predicting WEVar score, and all are necessary for training WEVar model. Please see data/test_SNPs file as reference

### Predict WEVar Score

```
python3 src/WEVar.py --prediction

```
The WEVar score will be saved res_dir folder
#### Evaluate the well-trained model

```

```
The program will evaluate the well-trained model, draw a R-squared figure, and save it to result directory.

<center>
<div align=center><img width="400" height="300" src="https://github.com/alfredyewang/MDeep/blob/master/result/USA/result.jpg"/></div>
</center>  


#### Test the model with unlabelled data

```
python3 src/MDeep.py --test --test_file data/USA/X_test.npy  --correlation_file data/USA/c.npy --result_dir result/USA --model_dir model --outcome_type continous --batch_size 16 --max_epoch 2000 --learning_rate 5e-3 --dropout_rate 0.5 --window_size 8 8 8 --kernel_size 64 64 32 --strides 4 4 4
```
The program will take the unlabelled test file and save the prediction result to result directory.


### Malawian Twin pairs Human Gut Microbiome data (Binary-Outcome)
#### Train the model
The USA Human Gut Microbiome data contains 995 samples with 2291 OTUs.
```
python3 src/MDeep.py --train --data_dir data/Malawiantwin_pairs --model_dir model --outcome_type binary --batch_size 32 --max_epoch 500 --learning_rate 1e-4 --dropout_rate 0.5 --window_size 128 4 --kernel_size 32 32 --strides 64 2
```
#### Evaluate the well-trained model

```
python3 src/MDeep.py --evaluation --data_dir data/Malawiantwin_pairs --result_dir result/Malawiantwin_pairs --model_dir model --outcome_type binary --batch_size 32 --max_epoch 500 --learning_rate 1e-4 --dropout_rate 0.5 --window_size 128 4 --kernel_size 32 32 --strides 64 2
```
The program will draw a ROC figure and save it to result directory.

<center>
<div align=center><img width="400" height="300" src="https://github.com/alfredyewang/MDeep/blob/master/result/Malawiantwin_pairs/result.jpg"/></div>
</center>  

#### Test the model with unlabelled data
```
python3 src/MDeep.py --test --test_file data/Malawiantwin_pairs/X_test.npy --correlation_file data/Malawiantwin_pairs/c.npy --result_dir result/Malawiantwin_pairs --model_dir model --outcome_type binary --batch_size 32 --max_epoch 500 --learning_rate 1e-4 --dropout_rate 0.5 --window_size 128 4 --kernel_size 32 32 --strides 64 2
```
