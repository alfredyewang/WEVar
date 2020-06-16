import get_score
import transformed_score
import model
import pandas as pd


test_file ='test_snps'
test_SNPs = pd.read_csv(test_file, sep='\t')

data =get_score.get_score(test_SNPs)


'''
kde get transformed score
'''
methods ='GWAS'

trans_score = transformed_score.transform(data, methods)

'''
load WEVar model, and predict WEVar score
'''

res = model.WEVar(trans_score, methods)

test_SNPs['WEVar_{}'.format(methods)]= res
test_SNPs.to_csv('res/{}'.format(test_file), index=False, sep='\t')

