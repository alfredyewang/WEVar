

"""Example of generating correlated normally distributed random samples."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
np.random.seed(123)
from scipy.stats import gaussian_kde
from decimal import Decimal
from os import listdir
from os.path import isfile, join
path= 'data/'
files = [f for f in listdir(path) if isfile(join(path, f))]




for k, file in enumerate(files):
    file = str(file)
    print(path + file)
    df = pd.read_csv(path + file, sep='\t')
    # print(df.shape)
    a = df.shape[0]

    if file in ['test1','test2','train']:
        continue

    df['Eigen'] = df['Eigen'].replace('.', np.nan, regex = False)
    df['CADD'] = df['CADD'].replace('.', np.nan, regex = False)
    df['DANN'] = df['DANN'].replace('.', np.nan, regex = False)
    df['FATHMM-MKL'] = df['FATHMM-MKL'].replace('.', np.nan, regex = False)
    df['FunSeq2'] = df['FunSeq2'].replace('.', np.nan, regex = False)
    df['Gwava_Region'] = df['Gwava_Region'].replace('.', np.nan, regex = False)
    df['Gwava_TSS'] = df['Gwava_TSS'].replace('.', np.nan, regex = False)
    df['LINSIGHT'] = df['LINSIGHT'].replace('.', np.nan, regex = False)
    df['Gwava_unmatched'] = df['Gwava_unmatched'].replace('.', np.nan, regex = False)
    # df = df[["Eigen-raw", "CADD", "DANN", "Fathmm-mkl",
                # "FIRE",
                # "Funseq2",
                # "GenoCanyon",
                # "Gwava_Region", "Gwava_TSS",
                # "Gwava_unmatched",
                # "Fitcon",
                # "LinSight",
                # "SuRFR",
             # 'labels'
             #    ]]
    # df = df.fillna('.')
    df =df[['Eigen','CADD', 'DANN', 'FATHMM-MKL', 'FunSeq2', 'Gwava_Region','Gwava_TSS','Gwava_unmatched','LINSIGHT','label']]
    # df['Eigen'] =df['Eigen'].str.replace('.','')
    # print(df.iloc[11050])
    df = df.dropna()
    # print(df.iloc[11050])
    df['Eigen'] = df['Eigen'].astype(np.float64)
    df["CADD"] = df["CADD"].astype(np.float64)
    df["DANN"] = df["DANN"].astype(np.float64)
    df['FATHMM-MKL'] = df['FATHMM-MKL'].astype(np.float64)
    df["FunSeq2"] = df["FunSeq2"].astype(np.float64)
    df["Gwava_Region"] = df["Gwava_Region"].astype(np.float64)
    df['Gwava_TSS'] = df['Gwava_TSS'].astype(np.float64)
    df["Gwava_unmatched"] = df["Gwava_unmatched"].astype(np.float64)
    df["LINSIGHT"] = df["LINSIGHT"].astype(np.float64)

    score = df[['Eigen','CADD', 'DANN', 'FATHMM-MKL', 'FunSeq2', 'Gwava_Region','Gwava_TSS','Gwava_unmatched','LINSIGHT'
                ]].to_numpy()
    # score['Eigen'] = score['Eigen'].
    # label = df[["labels"]].to_numpy()
    # print(score.shape)
    label = df['label']
    # print(score)
    # score = score.dropna()
    # print(float(score.shape[0])/a)
    # exit(22)
    # label = df[["labels"]].to_numpy()
    # print(score.shape)
    # print(label.shape)
    # exit(1)
    # print(score)
    # np.save('data_npy/'+'{}'.format(file)+'_score.npy',score)
    # np.save('data_npy/'+'{}'.format(file)+'_label.npy',label)
    # print(score.shape)
    print(label.sum())
    print(score.shape[0]-label.sum())