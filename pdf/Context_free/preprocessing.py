import pandas as pd

import numpy as np
from os import listdir
from os.path import isfile, join

path = 'bed_res'

files =['gwas'
,'allele_imbanlance_0.1_FDR.noNA.score.noNA.score'
,'dsQTL_deltaSVM.noNA.score'
,'eQTL_tss_unique.noNA.score'
,'validated_regulatory_SNP.noNA.score',
'cosmic',
        'clinvar']

for file in files:


    df1 = pd.read_csv('./cepip/{}_reg.txt'.format(file), sep='\t')
    # print(df.shape)
    # print(df.iloc[0])
    res =df1[['#Chrom','Pos_end','Eigen','CADD', 'DANN', 'FATHMM-MKL', 'FunSeq2','LINSIGHT']]
    res = res.drop_duplicates(subset=['#Chrom','Pos_end'],keep='first')
    # res.to_csv('./2', sep='\t', header=True, index=False)
    # bed=df1[['#Chrom','Pos_start', 'Pos_end','Ref','Alts']]
    # bed['#Chrom'] = bed['#Chrom'].astype(str)
    # bed['#Chrom'] = 'chr' + bed['#Chrom'].astype(str)
    # bed.to_csv('./2_bed', sep='\t', header=True, index=False)
    # print(bed.iloc[0])
    # exit(2)
    res['#Chrom'] = res['#Chrom'].astype(str)
    df = pd.read_csv('./cepip/{}_gwava.txt'.format(file), sep='\t', header=None)
    df.columns=["#Chrom","Pos_start",'Pos_end','a','label',"Gwava_Region",'Gwava_TSS',"Gwava_unmatched"]
    df=df[["#Chrom","Pos_end","label","Gwava_Region",'Gwava_TSS',"Gwava_unmatched"]]
    df['label'] = df['label'].replace('+', 1, regex=False)
    df['label'] = df['label'].replace('-', 0, regex=False)
    df['#Chrom'] = df['#Chrom'].astype(str)
        # df['chr'] = df['chr'].map(lambda x: x.rstrip('chr'))
    df['#Chrom'] = df['#Chrom'].map(lambda x: str(x)[3:])
    df['Pos_end'] = df['Pos_end'].astype(int)
        # print(df.shape)
    print(df.iloc[0])
    res = pd.merge(res,df, on=["#Chrom", "Pos_end"], how="left")
    del df
    print('add Gwava')
    res = res.drop_duplicates(keep='first')
    print(res.shape)
    res=res[['#Chrom','Pos_end','Eigen','CADD', 'DANN', 'FATHMM-MKL', 'FunSeq2', 'Gwava_Region','Gwava_TSS','Gwava_unmatched','LINSIGHT', 'label']]
    res.to_csv('./data/{}'.format(file), sep='\t', header=True, index=False)
    print(res.iloc[1])
