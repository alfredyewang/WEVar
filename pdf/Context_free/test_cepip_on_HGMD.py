import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import auc
from sklearn.metrics import r2_score
from sklearn import metrics
import seaborn as sns
np.random.seed(123)
sns.set_style("ticks")
sns.despine()

# sns.set()
cur = [(0.2980392156862745, 0.4470588235294118, 0.6901960784313725), (0.8666666666666667, 0.5176470588235295, 0.3215686274509804), (0.3333333333333333, 0.6588235294117647, 0.40784313725490196), (0.7686274509803922, 0.3058823529411765, 0.3215686274509804), (0.5058823529411764, 0.4470588235294118, 0.7019607843137254), (0.5764705882352941, 0.47058823529411764, 0.3764705882352941), (0.8549019607843137, 0.5450980392156862, 0.7647058823529411), (0.5490196078431373, 0.5490196078431373, 0.5490196078431373), (0.8, 0.7254901960784313, 0.4549019607843137) , (1.0, 0.8509803921568627, 0.1843137254901961), (0.39215686274509803, 0.7098039215686275, 0.803921568627451)]

order = [1,
         0,
         2,3,
         4,7,
         6,5,8,9,10]
cur = [cur[i] for i in order]

import matplotlib
plt.rcParams["font.family"] = "Times New Roman"
del matplotlib.font_manager.weight_dict['roman']
matplotlib.font_manager._rebuild()


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm



files = [
    'allele_imbanlance_0.1_FDR.noNA.score',
    # 'clinvar',
    # 'cosmic',
    # 'dsQTL_deltaSVM.noNA.score',
    'eQTL_tss_unique.noNA.score',
    'gwas',
        # 'HGMD',
    'validated_regulatory_SNP.noNA.score',
    'TestData_MPRA_E116.txt',
    # 'TestData_MPRA_E116_unbalanced.txt',
    # 'TestData_MPRA_E118.txt',
    'TestData_MPRA_E123.txt'

]

names = [
    'Allele imbanlance',
    # 'clinvar',
    # 'COSMIC',
    # 'dsQTL',
    'Fine mapping eQTLs',
    'GWAS',
        # 'HGMD',
    'Experimentally validated SNPs',
    'GM12878 lymphoblastoid',
    # 'TestData_MPRA_E116_unbalanced.txt',
    # 'MPRA_E118.txt',
    'K562 leukemia'
]


b= 0.1
model = '_1'
fig, axs = plt.subplots(2, 3,  figsize=(22, 14))

for j, file in enumerate(files):
    print(file)
    beta = np.load('./HGMD_pdf/{}/model/model{}.npy'.format(b,model))
    # print(beta)
    AUC = []
    AUC_PR = []
    r2 = []
    X1= np.load('./HGMD_pdf/{}/{}_1.npy'.format(b,file))
    X2= np.load('./HGMD_pdf/{}/{}_2.npy'.format(b,file))

    x = X2 / X1

    x = np.ma.log(x)
    x = x.filled(0)
    label = np.load('./data_npy/{}_label.npy'.format(file))

    x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)

    res = sigmoid(np.dot(x, beta))
    labels = label.reshape(label.shape[0], 1)


    score0 = np.load('./data_npy/{}_score.npy'.format(file))
    score1 = np.load('./data_npy/{}_score.npy'.format(file))
    for i in range(score1.shape[1]):
        score1[:, i] = (score1[:, i] - score1[:, i].min()) / (score1[:, i].max() - score1[:, i].min())

    baseline = score1.mean(axis=1).reshape(score1.shape[0],1)
    # print(baseline.shape)
    # print(res.shape)

    score = np.concatenate((score0, baseline), axis=1)

    score = np.concatenate((score,res),axis=1)
    #
    # score_new = np.concatenate((score0,res),axis=1)
    #
    # # exit(2)
    # # print(score_new.shape)
    # # print(labels.shape)
    # # print(sum(labels))
    # # exit(2)
    # zero_idx = np.where(labels == 0.0)[0]
    # # print(zero_idx)
    # one_idx = np.where(labels == 1)[0]
    # zero_idx_sample = np.random.choice(zero_idx, len(one_idx))
    #
    # score = np.concatenate((score_new[one_idx,:],score_new[zero_idx_sample,:]),axis=0)
    # # print(score.shape)
    # label = np.concatenate((np.ones(len(one_idx),),np.zeros(len(one_idx),)),axis=0)
    # # print(label.shape)

    method = [
        "Eigen",
        "CADD", "DANN", "FATHMM_MKL", "FunSeq2"
        , "GWAVA_Region", "GWAVA_TSS", "GWAVA_Unmatched",'LINSIGHT',
        'Unweighted',
    ]

    for k, m in enumerate(method):
        fpr, tpr, threshold = metrics.roc_curve(label, score[:, k])
        auc = metrics.roc_auc_score(label, score[:, k])
        AUC.append(auc)
            # plt.plot(fpr, tpr, label='{} ({:.02f})'.format(m, auc))
            # print(auc, end='\t')
        precision, recall, thresholds = precision_recall_curve(label, score[:, k])
            # print(precision.shape)
            # print(recall.shape)
        auc_pr = metrics.auc(recall, precision)
        # auc_pr = average_precision_score(label, score[:, k])
        AUC_PR.append(auc_pr)
            # print(label.shape)
            # print(score[:,k].shape)
        # print(label.reshape(label.shape[0],).shape)
        r = np.corrcoef(label.reshape(label.shape[0],),score[:,k])
            # print(r)
            # r1= r2_score(label,score[:,k])

        r2.append(r[0,1])

    x = j // 3
    y = j % 3
    # print(int(x), int(y))
    # continue
    # fig = plt.figure(figsize=(8, 6))
    df_testing = pd.DataFrame({'Methods': method, 'AUROC': AUC, 'AUPR': AUC_PR, 'R2': r2})
    print(df_testing)
    df_testing.to_csv('../draw_figure5/res/{}_context_free'.format(file),index=False, sep='\t')
            # plot training
    sns.scatterplot(x="AUPR", y="AUROC", hue="Methods", size="R2", palette=cur,
                    hue_order=["CADD", "Eigen", 'LINSIGHT', "DANN", "GWAVA_Unmatched", "GWAVA_TSS", "GWAVA_Region",
                               "FunSeq2", "FATHMM_MKL", 'Unweighted', 'WEVar'],
                    sizes= (100,500),data=df_testing, ax=axs[int(x)][int(y)])

    print(df_testing)

    # if x == 1 and y == 2:
    handles, labels = axs[int(x)][int(y)].get_legend_handles_labels()
    # print(handles)
    # print(labels)
        # axs[int(x)][int(y)].legend(handles[1:12], labels[1:12], loc='center left', bbox_to_anchor=(-1.5, -0.3), ncol=4,
        #                             fontsize=16)
    axs[int(x)][int(y)].get_legend().remove()
    # axs[int(x), int(y)].legend().set_visible(False)




    # fig.legend().set_visible(True)
    axs[int(x)][int(y)].set_title('{}'.format(names[j]), fontsize=16)
    axs[int(x)][int(y)].set_title('{}'.format(names[j]), fontsize=24)
    plt.setp(axs[int(x)][int(y)].get_xticklabels(), fontsize=16)
    plt.setp(axs[int(x)][int(y)].get_yticklabels(), fontsize=16)
    plt.setp(axs[int(x)][int(y)].set_xlabel('AUPR'), fontsize=18)
    plt.setp(axs[int(x)][int(y)].set_ylabel('AUROC'), fontsize=18)
    if x==1 and y ==2:
        fig.legend(handles[1:12], labels[1:12], loc='center left', bbox_to_anchor=(0.23, 0.03), ncol=6,
                   fontsize=16, frameon=False)
fig.tight_layout()
fig.subplots_adjust(bottom=0.12)
# plt.subplots_adjust(hspace = 0.5)
    # fig.savefig('./TIVAN.png')
    # fig.savefig("./Figure/AD_on_HGMD/{}.pdf".format(file), bbox_inches='tight')

sns.despine()
fig.savefig("./Figure33.pdf")

plt.show()