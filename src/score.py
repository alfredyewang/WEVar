
import numpy as np
import pandas as pd
import pyranges as pr
path= './'
df = pd.read_csv(path+'Base', sep='\t')
df['#Chrom'] = 'chr' + df['#Chrom'].astype(str)
df['End'] = df['Pos_end']
df = df[['#Chrom', 'Pos_end','End','Eigen', 'CADD', 'DANN', 'FATHMM-MKL', 'FunSeq2', 'LINSIGHT',"Gwava_Region",'Gwava_TSS',"Gwava_unmatched"]]
df = df.rename(columns={'#Chrom':'Chromosome', 'Pos_end':'Start'})
pr2 = pr.PyRanges(df)

def get_score(test):
    test1 = pr.PyRanges(test)
    res = test1.k_nearest(pr2, k=1,ties="first")

    data = res.as_df()[['Eigen', 'CADD', 'DANN', 'FATHMM-MKL', 'FunSeq2', 'LINSIGHT',"Gwava_Region",'Gwava_TSS',"Gwava_unmatched"]].to_numpy()
    return data

