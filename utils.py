import os
import logging
import random
import numpy as np
import torch
import torch_geometric as pyg
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
from scipy import sparse
from itertools import product, permutations
from sklearn.metrics import precision_recall_curve, roc_curve, auc

def change_into_current_py_path():
    current_file_path = __file__
    current_dir = os.path.dirname(current_file_path)
    os.chdir(current_dir)

def set_logger():
    logger = logging.getLogger(__name__)        
    logger.setLevel(logging.DEBUG)              
    console_handler = logging.StreamHandler()   
    logger.addHandler(console_handler)  
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - \33[32m%(message)s\033[0m')     
    console_handler.setFormatter(formatter) 
    logger.propagate = False
    return logger

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pyg.seed_everything(seed)

def visual(true, preds, gene_name, name):
    """
    visualize the prediction and ground truth of gene expression
    """
    plt.figure()
    plt.title(gene_name)
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def prepare_data_for_R(adata: sc.AnnData,
                       save_path: str,
                       reducedDim = None):
    """
    Process the AnnData object and save the necessary data to files.
    These data files are prepared for running the `slingshot_MAST_script.R` or `MAST_script.R` scripts.
    """
    if 'log_transformed' not in adata.layers:
        raise ValueError(
            f'Did not find `log_transformed` in adata.layers.'
        )

    if isinstance(adata.layers['log_transformed'], sparse.csr_matrix):
        exp_normalized = adata.layers['log_transformed'].A
    else:
        exp_normalized = adata.layers['log_transformed']

    # The normalized and log transformed data is used for MAST
    normalized_counts = pd.DataFrame(exp_normalized,
                                     index=adata.obs_names,
                                     columns=adata.var_names)
    normalized_counts.to_csv(save_path + '/exp_normalized.csv', sep=',')

    # The reduced dimension data is used for Slingshot
    if reducedDim is not None:
        reducedDim_data = pd.DataFrame(adata.obsm[reducedDim], dtype='float32', index=None)
        reducedDim_data.to_csv(save_path + '/data_reducedDim.csv', index=None)
    else:
        if 'lineages' not in adata.uns:
            raise ValueError(
                f'Did not find `lineages` in adata.uns.'
            )
        else:
            pseudotime_all = pd.DataFrame(index=adata.obs_names)
            for li in adata.uns['lineages']:
                pseudotime_all[li] = adata.obs[li + '_pseudo_time']
            pseudotime_all.to_csv(save_path + '/pseudotime_lineages.csv', index=True)

def EarlyPrec(trueEdgesDF: pd.DataFrame, predEdgesDF: pd.DataFrame,
              weight_key: str = 'weights_combined', TFEdges: bool = True):
    """
        Computes early precision for a given set of predictions in the form of a DataFrame.
        The early precision is defined as the fraction of true positives in the top-k edges,
        where k is the number of edges in the ground truth network (excluding self loops).

    :param trueEdgesDF:   A pandas dataframe containing the true edges.
    :type trueEdgesDF: DataFrame

    :param predEdgesDF:   A pandas dataframe containing the edges and their weights from
        the predicted network. Use param `weight_key` to assign the column name of edge weights.
        Higher the weight, higher the edge confidence.
    :type predEdgesDF: DataFrame

    :param weight_key:   A str represents the column name containing weights in predEdgeDF.
    :type weight_key: str

    :param TFEdges:   A flag to indicate whether to consider only edges going out of TFs (TFEdges = True)
        or not (TFEdges = False) from evaluation.
    :type TFEdges: bool

    :returns:
        - Eprec: Early precision value
        - Erec: Early recall value
        - EPR: Early precision ratio
    """

    print("Calculating the EPR(early prediction rate)...")

    # Remove self-loops
    trueEdgesDF = trueEdgesDF.loc[(trueEdgesDF['Gene1'] != trueEdgesDF['Gene2'])]
    if 'Score' in trueEdgesDF.columns:
        trueEdgesDF = trueEdgesDF.sort_values('Score', ascending=False)
    trueEdgesDF = trueEdgesDF.drop_duplicates(keep='first', inplace=False).copy()
    trueEdgesDF.reset_index(drop=True, inplace=True)

    predEdgesDF = predEdgesDF.loc[(predEdgesDF['Gene1'] != predEdgesDF['Gene2'])]
    if weight_key in predEdgesDF.columns:
        predEdgesDF = predEdgesDF.sort_values(weight_key, ascending=False)
    predEdgesDF = predEdgesDF.drop_duplicates(keep='first', inplace=False).copy()
    predEdgesDF.reset_index(drop=True, inplace=True)

    uniqueNodes = np.unique(trueEdgesDF.loc[:, ['Gene1', 'Gene2']])
    if TFEdges:
        # Consider only edges going out of source genes
        print("  Consider only edges going out of source genes")

        # Get a list of all possible TF to gene interactions
        possibleEdges_TF = set(product(set(trueEdgesDF.Gene1), set(uniqueNodes)))   #笛卡尔积

        # Get a list of all possible interactions 
        possibleEdges_noSelf = set(permutations(uniqueNodes, r=2))  #两两边的排列组合

        # Find intersection of above lists to ignore self edges
        # TODO: is there a better way of doing this?
        possibleEdges = possibleEdges_TF.intersection(possibleEdges_noSelf)
        possibleEdges = pd.DataFrame(possibleEdges, columns=['Gene1', 'Gene2'], dtype=str)

        # possibleEdgesDict = {'|'.join(p): 0 for p in possibleEdges}
        possibleEdgesDict = possibleEdges['Gene1'] + "|" + possibleEdges['Gene2']

        trueEdges = trueEdgesDF['Gene1'].astype(str) + "|" + trueEdgesDF['Gene2'].astype(str)
        trueEdges = trueEdges[trueEdges.isin(possibleEdgesDict)]
        print("  {} TF Edges in ground-truth".format(len(trueEdges)))
        numEdges = len(trueEdges)

        predEdgesDF['Edges'] = predEdgesDF['Gene1'].astype(str) + "|" + predEdgesDF['Gene2'].astype(str)
        # limit the predicted edges to the genes that are in the ground truth
        predEdgesDF = predEdgesDF[predEdgesDF['Edges'].isin(possibleEdgesDict)]
        print("  {} Predicted TF edges are considered".format(len(predEdgesDF)))

        M = len(set(trueEdgesDF.Gene1)) * (len(uniqueNodes) - 1)        #笛卡尔积可能的边的数量，减去1是因为不考虑self_loop

    else:
        trueEdges = trueEdgesDF['Gene1'].astype(str) + "|" + trueEdgesDF['Gene2'].astype(str)
        trueEdges = set(trueEdges.values)
        numEdges = len(trueEdges)
        print("  {} edges in ground-truth".format(len(trueEdges)))

        M = len(uniqueNodes) * (len(uniqueNodes) - 1)

    if not predEdgesDF.shape[0] == 0:
        # Use num True edges or the number of
        # edges in the dataframe, which ever is lower
        maxk = min(predEdgesDF.shape[0], numEdges)
        edgeWeightTopk = predEdgesDF.iloc[maxk - 1][weight_key]

        nonZeroMin = np.nanmin(predEdgesDF[weight_key].values)
        bestVal = max(nonZeroMin, edgeWeightTopk)

        newDF = predEdgesDF.loc[(predEdgesDF[weight_key] >= bestVal)]
        predEdges = set(newDF['Gene1'].astype(str) + "|" + newDF['Gene2'].astype(str))
        print("  {} Top-k edges selected".format(len(predEdges)))
    else:
        predEdges = set([])

    if len(predEdges) != 0:
        intersectionSet = predEdges.intersection(trueEdges)
        print("  {} true-positive edges".format(len(intersectionSet)))
        Eprec = len(intersectionSet) / len(predEdges)
        Erec = len(intersectionSet) / len(trueEdges)
    else:
        Eprec = 0
        Erec = 0

    random_EP = len(trueEdges) / M
    EPR = Erec / random_EP
    return Eprec, Erec, EPR

def computeScores(trueEdgesDF: pd.DataFrame, predEdgesDF: pd.DataFrame,
                  weight_key: str = 'weights_combined', selfEdges: bool = False):
    """
        Computes precision-recall and ROC curves using scikit-learn
        for a given set of predictions in the form of a DataFrame.

    :param trueEdgesDF:   A pandas dataframe containing the true edges.
    :type trueEdgesDF: DataFrame

    :param predEdgesDF:   A pandas dataframe containing the edges and their weights from
        the predicted network. Use param `weight_key` to assign the column name of edge weights.
        Higher the weight, higher the edge confidence.
    :type predEdgesDF: DataFrame

    :param weight_key:   A str represents the column name containing weights in predEdgeDF.
    :type weight_key: str

    :param selfEdges:   A flag to indicate whether to include self-edges (selfEdges = True)
        or exclude self-edges (selfEdges = False) from evaluation.
    :type selfEdges: bool

    :returns:
        - prec: A list of precision values (for PR plot)
        - recall: A list of precision values (for PR plot)
        - fpr: A list of false positive rates (for ROC plot)
        - tpr: A list of true positive rates (for ROC plot)
        - AUPRC: Area under the precision-recall curve
        - AUROC: Area under the ROC curve
    """
    print("Calculating the AUPRC and AUROC...")

    trueEdgesDF = trueEdgesDF.drop_duplicates(keep='first', inplace=False).copy()
    trueEdgesDF.reset_index(drop=True, inplace=True)

    if weight_key in predEdgesDF.columns:
        predEdgesDF = predEdgesDF.sort_values(weight_key, ascending=False)
    predEdgesDF = predEdgesDF.drop_duplicates(keep='first', inplace=False).copy()
    predEdgesDF.reset_index(drop=True, inplace=True)

    # Initialize dictionaries with all
    # possible edges
    if selfEdges:
        possibleEdges = list(product(np.unique(trueEdgesDF.loc[:, ['Gene1', 'Gene2']]),
                                     repeat=2))
    else:
        possibleEdges = list(permutations(np.unique(trueEdgesDF.loc[:, ['Gene1', 'Gene2']]),
                                          r=2))
        trueEdgesDF = trueEdgesDF.loc[(trueEdgesDF['Gene1'] != trueEdgesDF['Gene2'])]
        predEdgesDF = predEdgesDF.loc[(predEdgesDF['Gene1'] != predEdgesDF['Gene2'])]

    TrueEdgeDict = pd.DataFrame({'|'.join(p): 0 for p in possibleEdges}, index=['label']).T
    PredEdgeDict = pd.DataFrame({'|'.join(p): 0 for p in possibleEdges}, index=['label']).T

    # Compute TrueEdgeDict Dictionary
    # 1 if edge is present in the ground-truth
    # 0 if edge is not present in the ground-truth
    TrueEdgeDict.loc[np.array(trueEdgesDF['Gene1'] + "|" + trueEdgesDF['Gene2']), 'label'] = 1
    PredEdgeDict.loc[np.array(predEdgesDF['Gene1'] + "|" + predEdgesDF['Gene2']), 'label'] = np.abs(
        predEdgesDF[weight_key].values)

    # Combine into one dataframe
    # to pass it to sklearn
    outDF = pd.DataFrame([TrueEdgeDict['label'].values, PredEdgeDict['label'].values]).T
    outDF.columns = ['TrueEdges', 'PredEdges']
    
    fpr, tpr, thresholds = roc_curve(y_true=outDF['TrueEdges'],
                                     y_score=outDF['PredEdges'], pos_label=1)

    prec, recall, thresholds = precision_recall_curve(y_true=outDF['TrueEdges'],
                                                      probas_pred=outDF['PredEdges'], pos_label=1)
    AUPRC = auc(recall, prec)
    AUROC = auc(fpr, tpr)
    
    # folder_path = './results/'
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)
    
    # ## Make PR curves
    # plt.plot(recall,prec)
    
    # plt.xlim(0,1)    
    # plt.ylim(0,1)
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.legend('(AUPRC = ' + str("%.2f" % (AUPRC))+')') 
    # plt.savefig(os.path.join(folder_path,'AUPRC.jpg'))
    # plt.clf()

    # ## Make ROC curves
    # plt.plot(fpr,tpr)
    # plt.plot([0, 1], [0, 1], linewidth = 1.5, color = 'k', linestyle = '--')

    # plt.xlim(0,1)    
    # plt.ylim(0,1)
    # plt.xlabel('FPR')
    # plt.ylabel('TPR')
    # plt.legend('(AUROC = ' + str("%.2f" % (AUROC))+')') 
    # plt.savefig(os.path.join(folder_path,'AUROC.jpg'))
    # plt.clf()
    return AUPRC, AUROC

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss

def MAE(pred, true):
    return np.abs(pred - true).mean(axis=1)

def MSE(pred, true):
    return ((pred - true) ** 2).mean(axis=1)

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)

    return mae, mse