import pandas as pd
import score
import model
import transform
import train_model
import argparse



def main(args):


    prediction = args.prediction
    train = args.train
    if prediction:
        test_path = args.data_dir
        test_file = args.test_file
        context =  args.Context
        res_dir = args.res_dir
        test_SNPs = pd.read_csv(test_path + '/' + test_file, sep='\t')


        test_SNPs = test_SNPs[['Chromosome','Start','End']]
        data = score.get_score(test_SNPs)

        '''
        kde get transformed score
        '''

        trans_score = transform.transform(data, context)

        '''
        load WEVar model, and predict WEVar score
        '''
        res = model.WEVar(trans_score, context)

        test_SNPs['WEVar_{}'.format(context)] = res
        test_SNPs.to_csv('{}/{}'.format(res_dir,test_file), index=False, sep='\t')

    elif train:

        data_path = args.data_dir
        train_file = args.train_file
        train_SNPs = pd.read_csv(data_path + '/'  + train_file, sep='\t')
        X = train_SNPs[['Chromosome', 'Start', 'End']]
        X = score.get_score(X)
        Y = train_SNPs['Labels'].to_numpy()
        train_model.train(X,Y, train_file)
        train_model.test(X,Y, train_file)



def parse_arguments(parser):

    parser.add_argument('--prediction', dest='prediction', action='store_true', help='Use this option for predict WEVar score')
    parser.set_defaults(prediction=False)
    parser.add_argument('--data_dir', type=str, default='data/', metavar='<data_directory>',
                        help='The data directory for training and prediction')
    parser.add_argument('--test_file', type=str, default='test_snps', help='The test SNPs file')
    parser.add_argument('--res_dir', type=str, default='res/', metavar='<data_directory>',
                        help='The data directory for save res')
    parser.add_argument('--Context', type=str, default='HGMD', help='Use this option for select Context. (HGMD, eQTL, GWAS, Allele_imbanlace)')

    parser.add_argument('--train', dest='train', action='store_true', help='Use this option for train WEVar model')
    parser.set_defaults(train=True)
    parser.add_argument('--train_file', type=str, default='CAGI', help='The test SNPs file')
    parser.add_argument('--num_fold', type=int, default=10, help='k-fold cross validation')
    parser.add_argument('--kde_kde_bandwidth', type=float, default=0.1, help='the bandwidth of kde estimation')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='WEVar: A novel statistical learning framework for predicting non-coding genetic variants')
    args = parse_arguments(parser)
    main(args)
