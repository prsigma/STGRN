import argparse
from argparse import Namespace
import multiprocessing
import pandas as pd
import scanpy as sc
import networkx as nx
import copy
import celloracle as co
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import pytorch_warmup as warmup
from utils import *
from stgrn_data import STGRN_data
from model import STGRN
from Baseline_models import *

change_into_current_py_path()       
logger = set_logger()     
tf_dict = {
        "hESC": "human",
        "hHep": "human",
        "mDC": "mouse",
        "mESC": "mouse",
        "mHSC-E": "mouse",
        "mHSC-GM": "mouse",
        "mHSC-L": "mouse",
    }       

def parse_args():
    parser = argparse.ArgumentParser(description='expriments for GRN')

    data_group = parser.add_argument_group('data')
    data_group.add_argument('--dataset_name',type=str,required=True, help='name of dataset')
    data_group.add_argument('--data_path', type=str, required=True, help='expression data file') 
    data_group.add_argument('--gt_path',type=str,required=True, help='ground truth file') 
    data_group.add_argument('--prior_network',type=str,required=True, help='prior network for GCN')
    data_group.add_argument('--time_info',required=True, type=str, help='PseudoTime file')
    data_group.add_argument('--output_dir', type=str, required=True, help='location to store results')
    
    stgrn_group = parser.add_argument_group('STGRN')
    stgrn_group.add_argument('--stgrn_seg_len', type=int, default=8, help='segment length (L_seg)')
    stgrn_group.add_argument('--stgrn_input_len',type=int, default=64, help='input length')
    stgrn_group.add_argument('--stgrn_output_len',type=int,default=64,help='output length')
    stgrn_group.add_argument('--stgrn_d_model', type=int, default=128, help='dimension of hidden states of time layer transformer')
    stgrn_group.add_argument('--stgrn_d_ff', type=int, default=512, help='dimension of MLP in transformer')
    stgrn_group.add_argument('--stgrn_n_heads', type=int, default=4, help='num of heads')
    stgrn_group.add_argument('--stgrn_dropout', type=float, default=0.2, help='dropout')
    stgrn_group.add_argument('--stgrn_batch_size',type=int,default=4,help='batch size')
    stgrn_group.add_argument('--stgrn_layers',default=4,type=int,help='layers of transformer')
    stgrn_group.add_argument('--stgrn_train_epochs', type=int, default=50, help='train epochs')
    stgrn_group.add_argument('--stgrn_warm_up_steps',type=int,default=5,help='warm up steps')
    stgrn_group.add_argument('--stgrn_learning_rate', type=float, default=1e-3, help='optimizer initial learning rate')
    stgrn_group.add_argument('--stgrn_num_workers',type=int,default=8)
    stgrn_group.add_argument('--stgrn_patience',type=int,default=10)

    cefcon_group = parser.add_argument_group('CEFCON')
    cefcon_group.add_argument("--cefcon_hidden_dim", type=int, default=128,help="hidden dimension of the GNN encoder")
    cefcon_group.add_argument("--cefcon_output_dim", type=int, default=64,help="output dimension of the GNN encoder")
    cefcon_group.add_argument("--cefcon_heads", type=int, default=4,help="number of heads of multi-head attention. Default is 4")
    cefcon_group.add_argument("--cefcon_attention", type=str, default='COS', choices=['COS', 'AD', 'SD'],help="type of attention scoring function (\'COS\', \'AD\', \'SD\'). Default is \'COS\'")
    cefcon_group.add_argument('--cefcon_miu', type=float, default=0.5,help='parameter for considering the importance of attention coefficients of the first GNN layer')
    cefcon_group.add_argument('--cefcon_epochs', type=int, default=350,help='number of epochs for one run')
    cefcon_group.add_argument("--cefcon_edge_threshold_param", type=int, default=8,help="threshold for selecting top-weighted edges (larger values means more edges)")
    cefcon_group.add_argument("--cefcon_remove_self_loops", action="store_true",help="remove self loops")
    cefcon_group.add_argument('--cefcon_additional_edges_pct', type=float, default=0.01)

    args = parser.parse_args()

    return args

class Entry_STGRN(nn.Module):
    def __init__(self, data_args,stgrn_args,adata, ground_truth, together_genes_list,save_path_for_stgrn):
        super(Entry_STGRN, self).__init__()
        self.data_args = data_args
        self.stgrn_args = stgrn_args
        self.adata = adata
        self.ground_truth = ground_truth
        self.together_genes_list = together_genes_list
        self.save_path = save_path_for_stgrn

    def _build_model(self,gene_num):
        model = STGRN(gene_num,
                      self.stgrn_args.input_len,
                      self.stgrn_args.output_len,
                      self.stgrn_args.seg_len,
                      self.stgrn_args.d_model,
                      self.stgrn_args.d_ff,
                      self.stgrn_args.heads,
                      self.stgrn_args.dropout,
                      self.stgrn_args.layers,
                      self.stgrn_args.device).to(self.stgrn_args.device)
        return model

    def _get_data(self, flag):
        data_loader,prior_mask,idx_GeneName_map, ground_truth,gene_num = prepare_data_for_stgrn(self.data_args,self.stgrn_args, self.adata, self.ground_truth, self.together_genes_list, flag)
        return data_loader,prior_mask,idx_GeneName_map, ground_truth,gene_num

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.stgrn_args.learning_rate)
        return model_optim

    def _select_cosine_scheduler(self,num_steps,optimizer):
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
        return lr_scheduler
    
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_loader, criterion, prior_mask):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for _, (batch_x, batch_y, batch_y_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.stgrn_args.device)
                batch_y = batch_y.float()
                batch_y_mask = batch_y_mask.to(self.stgrn_args.device)

                outputs,_ = self.model(batch_x, prior_mask)
                batch_y = batch_y.to(self.stgrn_args.device)

                loss = criterion(outputs, batch_y)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self):
        train_loader, prior_mask, _ , _, gene_num= self._get_data(flag='train')
        vali_loader, _ , _ , _, _= self._get_data(flag='val')

        self.model = self._build_model(gene_num)

        early_stopping = EarlyStopping(patience=self.stgrn_args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        num_steps = self.stgrn_args.train_epochs
        lr_scheduler = self._select_cosine_scheduler(num_steps,model_optim)
        warmup_scheduler = warmup.LinearWarmup(model_optim, self.stgrn_args.warm_up_steps)

        for _ in range(self.stgrn_args.train_epochs):
            train_loss = []

            self.model.train()
            for (batch_x, batch_y, batch_y_mask) in tqdm(train_loader):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.stgrn_args.device)
                batch_y = batch_y.float().to(self.stgrn_args.device)
                batch_y_mask = batch_y_mask.to(self.stgrn_args.device)

                outputs,_ = self.model(batch_x,prior_mask)
                batch_y = batch_y.to(self.stgrn_args.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

            vali_loss = self.vali(vali_loader, criterion, prior_mask)
            with warmup_scheduler.dampening():
                    if warmup_scheduler.last_step + 1 > self.stgrn_args.warm_up_steps:
                        lr_scheduler.step()

            early_stopping(vali_loss, self.model, self.save_path)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break
        return

    def test(self):
        test_loader, prior_mask, idx_GeneName_map, ground_truth,_ = self._get_data(flag='test')
        
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'checkpoint.pth')))

        whole_attention = None
        preds = []
        trues = []
        inputs = []

        self.model.eval()
        with torch.no_grad():
            count = 0
            for (batch_x, batch_y, batch_y_mask) in tqdm(test_loader):
                count += 1
                batch_x = batch_x.float().to(self.stgrn_args.device)
                batch_y = batch_y.float().to(self.stgrn_args.device)
                batch_y_mask = batch_y_mask.to(self.stgrn_args.device)

                outputs, attention = self.model(batch_x, prior_mask)

                attention = torch.cat(attention).mean(0).detach().cpu().numpy()
                if whole_attention is None:
                    whole_attention = attention
                else:
                    whole_attention += attention

                # outputs = outputs * batch_y_mask
                # batch_y = batch_y * batch_y_mask

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                inputs.append(batch_x.detach().cpu().numpy())
        
        preds = np.array(preds)
        trues = np.array(trues)
        inputs = np.array(inputs)
        
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])

        mae, mse = metric(preds, trues)
        
        min_mae = mae.min()
        row_index_list,col_index_list = np.where(mae == min_mae)
        gene_name_list = idx_GeneName_map.iloc[col_index_list]['geneName'].values

        row_index = row_index_list[0]
        col_index = col_index_list[0]
        gene_name = gene_name_list[0]

        gt = np.concatenate((inputs[row_index, :, col_index], trues[row_index, :, col_index]), axis=0)
        pr = np.concatenate((inputs[row_index, :, col_index], preds[row_index, :, col_index]), axis=0)
        visual(gt, pr, gene_name, os.path.join(self.save_path, str(count) + '.pdf'))
        
        whole_attention /= len(test_loader)
        whole_attention *= (1- prior_mask)

        num_nodes = whole_attention.shape[0]
        mat_indicator_all = np.zeros([num_nodes, num_nodes])
        
        mat_indicator_all[abs(whole_attention) > 0] = 1
        idx_rec, idx_send = np.where(mat_indicator_all)
        edges_df = pd.DataFrame(
            {'Gene1': idx_GeneName_map.iloc[idx_send]['geneName'].tolist(), 'Gene2': idx_GeneName_map.iloc[idx_rec]['geneName'].tolist(), 'weights_combined': (whole_attention[idx_rec,idx_send])})
        edges_df = edges_df.sort_values('weights_combined', ascending=False)
        predicted_grn = edges_df.iloc[:len(ground_truth),:]
        predicted_grn.to_csv(os.path.join(self.save_path,'grn.csv'))

        Eprec, Erec, EPR = EarlyPrec(ground_truth,predicted_grn)
        AUPRC, AUROC = computeScores(ground_truth,predicted_grn)

        logger.info(f'****************STGRN********************** \n Eprec:{Eprec} \t Erec:{Erec} \t EPR:{EPR} \t AUPRC:{AUPRC} \t AUROC:{AUROC}')

        return Eprec, Erec, EPR, AUPRC, AUROC

def preprocess_data(data_args):
    logger.info('preprocessing data...')
    if not os.path.exists(os.path.join(data_args.output_dir,'gene_expression.h5ad')):
        # Gene Expression information
        adata = sc.read_csv(data_args.data_path, first_column_names=True)
        adata = adata.T     # cell * gene
        adata.var_names = adata.var_names.str.upper()

        # Ground Truth information
        ground_truth = pd.read_csv(data_args.gt_path)
        ground_truth = ground_truth[['Gene1','Gene2']]
        ground_truth = ground_truth.applymap(lambda x: x.upper())

        # Filter the genes that exist in both single cell data and the ground truth network
        ground_truth_genes = set(ground_truth['Gene1'].tolist()) | set(ground_truth['Gene2'].tolist())
        expression_genes = set(adata.var_names.values)
        together_genes_set = ground_truth_genes & expression_genes
        together_genes_list = list(together_genes_set)

        # Filter the gene expression
        adata = adata[:,together_genes_list]

        # Filter the ground truth
        ground_truth = ground_truth.loc[ground_truth['Gene1'].isin(together_genes_list) & ground_truth['Gene2'].isin(together_genes_list)]
        ground_truth = ground_truth.drop_duplicates(subset=['Gene1', 'Gene2'], keep='first', inplace=False)

        # Save the processed data
        adata.write(os.path.join(data_args.output_dir,'gene_expression.h5ad'))  #cell * gene
        ground_truth.to_csv(os.path.join(data_args.output_dir,'ground_truth.csv'),index=False)
    
    else:
        adata = sc.read(os.path.join(data_args.output_dir,'gene_expression.h5ad'))
        ground_truth = pd.read_csv(os.path.join(data_args.output_dir,'ground_truth.csv'))
        together_genes_list = adata.var_names.tolist()

    return adata, ground_truth, together_genes_list

def prepare_data_for_stgrn(data_args, stgrn_args, adata, ground_truth, together_genes_list, flag):
    # TF information
    if stgrn_args.species == 'human':
        TFs_df = pd.read_csv('./TF_data/human-tfs.csv')
    else:
        TFs_df = pd.read_csv('./TF_data/mouse-tfs.csv')
    TFs_df = TFs_df.applymap(lambda x: x.upper())
    TF_set = set(TFs_df['TF'].tolist())
    together_TF = TF_set & set(together_genes_list)

    # for stgrn, we restrict the gene1 is the TF
    ground_truth = ground_truth.loc[ground_truth['Gene1'].isin(together_TF) & ground_truth['Gene2'].isin(together_genes_list)] 
    ground_truth = ground_truth.drop_duplicates(subset=['Gene1', 'Gene2'], keep='first', inplace=False)

    # Prior network information
    if stgrn_args.species == 'mouse':
        base_GRN = co.data.load_mouse_scATAC_atlas_base_GRN()
        base_GRN = base_GRN.drop(columns='peak_id')
        base_GRN = base_GRN.set_index('gene_short_name')
        base_GRN.index = base_GRN.index.map(lambda x:x.upper())
        base_GRN.columns = base_GRN.columns.map(lambda x:x.upper())
    else:       #human
        base_GRN = co.data.load_human_promoter_base_GRN()
        base_GRN = base_GRN.drop(columns='peak_id')
        base_GRN = base_GRN.set_index('gene_short_name')
        base_GRN.index = base_GRN.index.map(lambda x:x.upper())
        base_GRN.columns = base_GRN.columns.map(lambda x:x.upper())
    
    location = np.where(base_GRN.values == 1)
    netData = pd.DataFrame(np.array(location).T,columns=['from','to'])
    index_map = {x:y for x,y in enumerate(base_GRN.index)}
    columns_map = {x:y for x,y in enumerate(base_GRN.columns)}
    netData['from'] = netData['from'].map(index_map)
    netData['to'] = netData['to'].map(columns_map)
    netData['from'] = netData['from'].str.upper()
    netData['to'] = netData['to'].str.upper()
    netData = netData.loc[netData['from'].isin(together_TF)
                          & netData['to'].isin(together_genes_list), :]
    netData = netData.drop_duplicates(subset=['from', 'to'], keep='first', inplace=False)
    netData = netData.iloc[:,:2]
    netData.columns = ['Gene1','Gene2']

    # Refine the prior network
    prior_network = pd.concat([netData,ground_truth],axis=0,ignore_index=True)
    prior_network = prior_network.drop_duplicates(subset=['Gene1', 'Gene2'], keep='first', inplace=False)

    # Get gene id map
    gene_id_map = {x:y for y,x in enumerate(together_genes_list)}
    idx_GeneName_map = pd.DataFrame({'idx': range(len(together_genes_list)),
                                     'geneName': together_genes_list},index=range(len(together_genes_list)))
    
    prior_network['Gene1'] = prior_network['Gene1'].map(gene_id_map)
    prior_network['Gene2'] = prior_network['Gene2'].map(gene_id_map)

    # get prior mask
    prior_mask = np.ones((len(together_genes_list),len(together_genes_list)))   
    prior_mask[prior_network['Gene2'],prior_network['Gene1']] = 0

    # Time infomation
    time_info = pd.read_csv(data_args.time_info)
    time_info = time_info.rename(columns={'Unnamed: 0':'cell'})
    time_info = time_info.sort_values(by='PseudoTime',ascending=True)
    cell_index_sorted_by_time = time_info['cell']
    adata = adata[cell_index_sorted_by_time]        #sort the cells
    
    dropout_mask = (adata.X != 0).astype(int)
    expression_data = adata.X

    data_set = STGRN_data(expression_data,dropout_mask,size=[stgrn_args.input_len,stgrn_args.output_len],flag=flag)

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = 1  # bsz=1 for evaluation
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = stgrn_args.batch_size  # bsz for train and valid
        
    logger.info(f'{flag}, {len(data_set)}')
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=stgrn_args.num_workers,
        drop_last=drop_last)
    
    return data_loader, prior_mask, idx_GeneName_map, ground_truth, len(idx_GeneName_map)

def prepare_data_for_cefcon(data_args, adata, ground_truth, together_genes_list, species, save_path, additional_edges_pct=0.01):
    # Prior network data
    netData = pd.read_csv(data_args.prior_network, index_col=None, header=0)
    
    # make sure the genes of prior network are in the input scRNA-seq data
    netData['from'] = netData['from'].str.upper()
    netData['to'] = netData['to'].str.upper()
    netData = netData.loc[netData['from'].isin(together_genes_list)
                          & netData['to'].isin(together_genes_list), :]
    netData = netData.drop_duplicates(subset=['from', 'to'], keep='first', inplace=False)

    # Transfer into networkx object
    priori_network = nx.from_pandas_edgelist(netData, source='from', target='to', create_using=nx.DiGraph)
    priori_network_nodes = np.array(priori_network.nodes())

    # in_degree, out_degree (centrality)
    in_degree = pd.DataFrame.from_dict(nx.in_degree_centrality(priori_network),
                                       orient='index', columns=['in_degree'])
    out_degree = pd.DataFrame.from_dict(nx.out_degree_centrality(priori_network),
                                        orient='index', columns=['out_degree'])
    centrality = pd.concat([in_degree, out_degree], axis=1)
    centrality = centrality.loc[priori_network_nodes, :]

    # A mapper for node index and gene name
    idx_GeneName_map = pd.DataFrame({'idx': range(len(priori_network_nodes)),
                                     'geneName': priori_network_nodes},
                                    index=priori_network_nodes)

    edgelist = pd.DataFrame({'from': idx_GeneName_map.loc[netData['from'].tolist(), 'idx'].tolist(),
                             'to': idx_GeneName_map.loc[netData['to'].tolist(), 'idx'].tolist()})

    # TF information
    is_TF = np.ones(len(priori_network_nodes), dtype=int)
    if species == 'human':
        TFs_df = pd.read_csv('./TF_data/human-tfs.csv')
    else:
        TFs_df = pd.read_csv('./TF_data/mouse-tfs.csv')
    TF_list = TFs_df.iloc[:, 0].str.upper()
    is_TF[~np.isin(priori_network_nodes, TF_list)] = 0

    # Only keep the genes that exist in both single cell data and the prior gene interaction network
    adata = adata[:, priori_network_nodes]
    adata.var['is_TF'] = is_TF
    adata.varm['centrality_prior_net'] = centrality
    adata.varm['idx_GeneName_map'] = idx_GeneName_map
    adata.uns['lineages'] = ['all']

    # Additional edges with high spearman correlation
    if isinstance(adata.X, sparse.csr_matrix):
        gene_exp = pd.DataFrame(adata.X.A.T, index=priori_network_nodes)
    else:
        gene_exp = pd.DataFrame(adata.X.T, index=priori_network_nodes)

    ori_edgeNum = len(edgelist)
    edges_corr = np.absolute(np.array(gene_exp.T.corr('spearman')))
    np.fill_diagonal(edges_corr, 0.0)
    x, y = np.where(edges_corr > 0.6)
    addi_top_edges = pd.DataFrame({'from': x, 'to': y, 'weight': edges_corr[x, y]})
    addi_top_k = int(gene_exp.shape[0] * (gene_exp.shape[0] - 1) * additional_edges_pct)
    if len(addi_top_edges) > addi_top_k:
        addi_top_edges = addi_top_edges.sort_values(by=['weight'], ascending=False)
        addi_top_edges = addi_top_edges.iloc[0:addi_top_k, 0:2]
    edgelist = pd.concat([edgelist, addi_top_edges.iloc[:, 0:2]], ignore_index=True)
    edgelist = edgelist.drop_duplicates(subset=['from', 'to'], keep='first', inplace=False)
    logger.info('{} extra edges (Spearman correlation > 0.6) are added into the prior gene interaction network.\n'
            '    Total number of edges: {}.'.format((len(edgelist) - ori_edgeNum), len(edgelist)))

    adata.uns['edgelist'] = edgelist

    # Differential expression scores
    adata.layers['log_transformed'] = adata.X.copy()
    adata_temp = copy.deepcopy(adata)

    # Time infomation
    time_info = pd.read_csv(data_args.time_info)
    time_info = time_info.rename(columns={'Unnamed: 0':'cell'})
    time_info = time_info.sort_values(by='PseudoTime',ascending=True)
    cell_index_sorted_by_time = time_info['cell']
    adata_temp = adata_temp[cell_index_sorted_by_time]    
    adata_temp.obs['all_pseudo_time'] = time_info['PseudoTime'].values

    genes_DE_path = os.path.join(save_path,f'DEgenes_MAST_sp4_all.csv')
    if os.path.exists(genes_DE_path):
        logger.info('loading DE info...')
        genes_DE = pd.read_csv(genes_DE_path, index_col=0, header=0)
    else:
        prepare_data_for_R(adata_temp, save_path)
        R_script_path = './Benchmark/CEFCON/MAST_script.R'
        logger.info("prepare run MAST")
        command = f'Rscript {R_script_path} {save_path}'
        logger.info('Running MAST using: \'{}\'\n'.format(command))
        logger.info('It will take a few minutes ...')
        os.system(command)
        logger.info(f'Done. The results are saved in \'{save_path}\'.')
        genes_DE = pd.read_csv(genes_DE_path, index_col=0, header=0)

    genes_DE = pd.DataFrame(genes_DE).iloc[:, 0]
    genes_DE.index = genes_DE.index.str.upper()
    genes_DE = genes_DE[genes_DE.index.isin(priori_network_nodes)].abs().dropna()
    node_score_auxiliary = pd.Series(np.zeros(len(priori_network_nodes)), index=priori_network_nodes)
    node_score_auxiliary[genes_DE.index] = genes_DE.values
    node_score_auxiliary = np.array(node_score_auxiliary)
    adata.var['node_score_auxiliary'] = node_score_auxiliary

    logger.info(f"n_genes × n_cells = {adata.n_vars} × {adata.n_obs}")

    return adata, ground_truth

def prepare_data_for_deepsem(data_args, adata, ground_truth):
    if 'specific' in data_args.gt_path and 'non_specifice' not in data_args.gt_path:
        task = 'celltype_GRN'
    else:
        task = 'non_celltype_GRN'
    adata = adata.T
    return adata, ground_truth, task

def prepare_data_for_grnboost2(adata,ground_truth,species):
    if species == 'human':
        TFs_df = pd.read_csv('./TF_data/human-tfs.csv')
    else:
        TFs_df = pd.read_csv('./TF_data/mouse-tfs.csv')
    
    TFs_df = TFs_df.applymap(lambda x: x.upper())
    tf_names = TFs_df['TF'].tolist()

    gene_expression = adata.to_df()
    return gene_expression, tf_names, ground_truth

def prepare_data_for_genie3(adata,ground_truth,species):
    if species == 'human':
        TFs_df = pd.read_csv('./TF_data/human-tfs.csv')
    else:
        TFs_df = pd.read_csv('./TF_data/mouse-tfs.csv')
    
    TFs_df = TFs_df.applymap(lambda x: x.upper())
    tf_names = TFs_df['TF'].tolist()

    gene_expression = adata.to_df()
    return gene_expression, tf_names, ground_truth

def run_CEFCON(data_args,cefcon_args,adata, ground_truth, together_genes_list,species,seed,device):
    logger.info('running CEFCON...')
    save_path_for_cefcon = os.path.join(data_args.output_dir,'cefcon')
    os.makedirs(save_path_for_cefcon, exist_ok=True)
    data_for_cefcon, ground_truth_for_cefcon = prepare_data_for_cefcon(data_args, adata, ground_truth, together_genes_list,species, save_path_for_cefcon, additional_edges_pct=cefcon_args.additional_edges_pct)

    cefcon_GRN_model = NetModel(hidden_dim=cefcon_args.hidden_dim,
                                output_dim=cefcon_args.output_dim,
                                heads=cefcon_args.heads,
                                attention_type=cefcon_args.attention,
                                miu=cefcon_args.miu,
                                epochs=cefcon_args.epochs,
                                repeats=cefcon_args.repeats,
                                seed=seed,
                                cuda=device,
                                )
    
    cefcon_GRN_model.run(data_for_cefcon, device, showProgressBar=True)
    _ = cefcon_GRN_model.get_network(keep_self_loops=False,
                                               edge_threshold_avgDegree=cefcon_args.edge_threshold_param,
                                               edge_threshold_zscore=None,
                                               output_file=os.path.join(save_path_for_cefcon,'grn.csv'))
    grn_cefcon = pd.read_csv(os.path.join(save_path_for_cefcon, 'grn.csv'),header=None)
    grn_cefcon.columns=['Gene1','Gene2','weights_combined']
    Eprec, Erec, EPR = EarlyPrec(ground_truth_for_cefcon,grn_cefcon)
    AUPRC, AUROC = computeScores(ground_truth_for_cefcon,grn_cefcon)

    logger.info(f'*******************CEFCON******************* \n Eprec:{Eprec} \t Erec:{Erec} \t EPR:{EPR} \t AUPRC:{AUPRC} \t AUROC:{AUROC}')
    return Eprec,Erec,EPR,AUPRC,AUROC

def run_STGRN(data_args,stgrn_args,adata, ground_truth, together_genes_list):
    logger.info('running STGRN...')
    save_path_for_stgrn = os.path.join(data_args.output_dir,'stgrn')
    os.makedirs(save_path_for_stgrn, exist_ok=True)

    Eprec_list = list()
    Erec_list = list()
    EPR_list = list()
    AUPRC_list = list()
    AUROC_list = list()

    for _ in range(stgrn_args.repeats):
        stgrn = Entry_STGRN(data_args,stgrn_args,adata, ground_truth, together_genes_list,save_path_for_stgrn)  # set experiments
        logger.info('>>>>>>>start training :>>>>>>>>>>>>>>>>>>>>>>>>>>')
        stgrn.train()

        logger.info('>>>>>>>testing :<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        Eprec, Erec, EPR, AUPRC, AUROC = stgrn.test()

        Eprec_list.append(Eprec)
        Erec_list.append(Erec)
        EPR_list.append(EPR)
        AUPRC_list.append(AUPRC)
        AUROC_list.append(AUROC)
        torch.cuda.empty_cache()
    
    avg_Eprec = np.array(Eprec_list).mean()
    avg_Erec = np.array(Erec_list).mean()
    avg_EPR = np.array(EPR_list).mean()
    avg_AUPRC = np.array(AUPRC_list).mean()
    avg_AUROC = np.array(AUROC_list).mean()

    return avg_Eprec,avg_Erec,avg_EPR,avg_AUPRC,avg_AUROC

def run_DeepSEM(data_args,adata, ground_truth):
    logger.info('running DeepSEM...')
    data_for_deepsem, ground_truth_for_deepsem, task_for_deepsem = prepare_data_for_deepsem(data_args,adata, ground_truth)
    save_path_for_deepsem = os.path.join(data_args.output_dir,'deepsem')
    os.makedirs(save_path_for_deepsem, exist_ok=True)
    save_expression_path = os.path.join(save_path_for_deepsem,'expression_data.h5ad')
    save_gt_path = os.path.join(save_path_for_deepsem,'ground_truth.csv')
    data_for_deepsem.write(save_expression_path)
    ground_truth_for_deepsem.to_csv(save_gt_path,index=False)

    os.system(f'python ./Baseline_models/DeepSEM/main.py --task {task_for_deepsem} --data_file {save_expression_path} --net_file {save_gt_path} --setting new --alpha 0.1 --beta 0.01 --n_epochs 150  --save_name {save_path_for_deepsem}')
    grn_deepsem = pd.read_csv(os.path.join(save_path_for_deepsem,'grn.tsv'),sep='\t')
    grn_deepsem.columns = ['Gene1','Gene2','weights_combined']

    Eprec, Erec, EPR = EarlyPrec(ground_truth_for_deepsem,grn_deepsem)
    AUPRC, AUROC = computeScores(ground_truth_for_deepsem,grn_deepsem)

    logger.info(f'****************DeepSEM********************** \n Eprec:{Eprec} \t Erec:{Erec} \t EPR:{EPR} \t AUPRC:{AUPRC} \t AUROC:{AUROC}')

    return Eprec, Erec, EPR, AUPRC, AUROC

def run_GRNBoost2(data_args,species,seed,adata,ground_truth):
    logger.info('running GRNBoost2...')
    data_for_grnboost2, tf_names ,ground_truth_for_grnboost2 = prepare_data_for_grnboost2(adata,ground_truth,species)
    save_path_for_grnboost2 = os.path.join(data_args.output_dir,'grnboost2')
    os.makedirs(save_path_for_grnboost2, exist_ok=True)

    gene_expression_path = os.path.join(save_path_for_grnboost2,'gene_expression.csv')
    data_for_grnboost2.to_csv(gene_expression_path,index=False)

    tf_name_path = os.path.join(save_path_for_grnboost2,'tf_names.txt')
    with open(tf_name_path,'w') as f:
        for item in tf_names:
            f.write("%s\n" % item) 
    
    save_grn_path = os.path.join(save_path_for_grnboost2,'grn.csv')
    os.system(f'python ./Baseline_models/GRNBoost2/arboreto_with_multiprocessing.py \
                {gene_expression_path} \
                {tf_name_path} \
                --method grnboost2 \
                --output {save_grn_path} \
                --num_workers 20 \
                --seed {seed}')
    
    grn_boost2 = pd.read_csv(os.path.join(save_path_for_grnboost2,'grn.csv'))
    grn_boost2.columns = ['Gene1','Gene2','weights_combined']

    Eprec, Erec, EPR = EarlyPrec(ground_truth_for_grnboost2,grn_boost2)
    AUPRC, AUROC = computeScores(ground_truth_for_grnboost2,grn_boost2)

    logger.info(f'*******************GRNBoost2******************* \n Eprec:{Eprec} \t Erec:{Erec} \t EPR:{EPR} \t AUPRC:{AUPRC} \t AUROC:{AUROC}')
    return Eprec, Erec, EPR, AUPRC, AUROC

def run_GENIE3(data_args,species,seed,adata,ground_truth):
    logger.info('running GENIE3...')
    data_for_genie3, tf_names ,ground_truth_for_genie3 = prepare_data_for_genie3(adata,ground_truth,species)
    save_path_for_genie3 = os.path.join(data_args.output_dir,'genie3')
    os.makedirs(save_path_for_genie3, exist_ok=True)

    gene_expression_path = os.path.join(save_path_for_genie3,'gene_expression.csv')
    data_for_genie3.to_csv(gene_expression_path,index=False)

    tf_name_path = os.path.join(save_path_for_genie3,'tf_names.txt')
    with open(tf_name_path,'w') as f:
        for item in tf_names:
            f.write("%s\n" % item) 
    
    save_grn_path = os.path.join(save_path_for_genie3,'grn.csv')
    os.system(f'python ./Baseline_models/GRNBoost2/arboreto_with_multiprocessing.py \
                {gene_expression_path} \
                {tf_name_path} \
                --method genie3 \
                --output {save_grn_path} \
                --num_workers {multiprocessing.cpu_count()} \
                --seed {seed}')
    
    grn_boost2 = pd.read_csv(os.path.join(save_path_for_genie3,'grn.csv'))
    grn_boost2.columns = ['Gene1','Gene2','weights_combined']

    Eprec, Erec, EPR = EarlyPrec(ground_truth_for_genie3,grn_boost2)
    AUPRC, AUROC = computeScores(ground_truth_for_genie3,grn_boost2)

    logger.info(f'*******************GENIE3******************* \n Eprec:{Eprec} \t Erec:{Erec} \t EPR:{EPR} \t AUPRC:{AUPRC} \t AUROC:{AUROC}')
    return Eprec, Erec, EPR, AUPRC, AUROC

def main():
    algorithm_name = list()
    prec_list = list()
    Erec_list = list()
    EPR_list = list()
    AUPRC_list = list()
    AUROC_list = list()

    args = parse_args()
    
    data_args = Namespace()
    stgrn_args = Namespace()
    cefcon_args = Namespace()

    data_args.dataset_name = args.dataset_name
    data_args.data_path = args.data_path
    data_args.gt_path = args.gt_path
    data_args.prior_network = args.prior_network
    data_args.time_info = args.time_info
    data_args.output_dir = args.output_dir

    stgrn_args.seg_len = args.stgrn_seg_len
    stgrn_args.input_len = args.stgrn_input_len
    stgrn_args.output_len = args.stgrn_output_len
    stgrn_args.d_model = args.stgrn_d_model
    stgrn_args.d_ff = args.stgrn_d_ff
    stgrn_args.heads = args.stgrn_n_heads
    stgrn_args.dropout = args.stgrn_dropout
    stgrn_args.layers = args.stgrn_layers
    stgrn_args.train_epochs = args.stgrn_train_epochs
    stgrn_args.warm_up_steps = args.stgrn_warm_up_steps
    stgrn_args.learning_rate = args.stgrn_learning_rate
    stgrn_args.batch_size = args.stgrn_batch_size
    stgrn_args.num_workers = args.stgrn_num_workers
    stgrn_args.patience = args.stgrn_patience

    cefcon_args.hidden_dim = args.cefcon_hidden_dim
    cefcon_args.output_dim = args.cefcon_output_dim
    cefcon_args.heads = args.cefcon_heads
    cefcon_args.attention = args.cefcon_attention
    cefcon_args.miu = args.cefcon_miu
    cefcon_args.epochs = args.cefcon_epochs
    cefcon_args.edge_threshold_param = args.cefcon_edge_threshold_param
    cefcon_args.remove_self_loops = args.cefcon_remove_self_loops
    cefcon_args.additional_edges_pct = args.cefcon_additional_edges_pct

    #add new args
    species = tf_dict[data_args.dataset_name]
    seed = 2024
    seed_everything(seed)
    device = torch.device('cuda:7')
    repeats = 1 
    
    cefcon_args.repeats = repeats

    stgrn_args.repeats = repeats
    stgrn_args.device = device
    stgrn_args.species = species

    # preprocess data
    adata, ground_truth, together_genes_list = preprocess_data(data_args)

    # #running STGRN
    # STGRN_Eprec, STGRN_Erec, STGRN_EPR, STGRN_AUPRC, STGRN_AUROC = run_STGRN(data_args,stgrn_args, adata, ground_truth, together_genes_list)
    # algorithm_name.append('STGRN')
    # prec_list.append(STGRN_Eprec)
    # Erec_list.append(STGRN_Erec)
    # EPR_list.append(STGRN_EPR)
    # AUPRC_list.append(STGRN_AUPRC)
    # AUROC_list.append(STGRN_AUROC)

    # #running CEFCON
    # CEFCON_Eprec,CEFCON_Erec,CEFCON_EPR,CEFCON_AUPRC,CEFCON_AUROC = run_CEFCON(data_args,cefcon_args,adata, ground_truth, together_genes_list,species,seed,device)
    # algorithm_name.append('CEFCON')
    # prec_list.append(CEFCON_Eprec)
    # Erec_list.append(CEFCON_Erec)
    # EPR_list.append(CEFCON_EPR)
    # AUPRC_list.append(CEFCON_AUPRC)
    # AUROC_list.append(CEFCON_AUROC)

    # #running DeepSEM
    # deepsem_Eprec, deepsem_Erec, deepsem_EPR, deepsem_AUPRC, deepsem_AUROC = run_DeepSEM(data_args,adata, ground_truth)
    # algorithm_name.append('DeepSEM')
    # prec_list.append(deepsem_Eprec)
    # Erec_list.append(deepsem_Erec)
    # EPR_list.append(deepsem_EPR)
    # AUPRC_list.append(deepsem_AUPRC)
    # AUROC_list.append(deepsem_AUROC)

    # running GRNboost2
    grnboost2_Eprec, grnboost2_Erec, grnboost2_EPR, grnboost2_AUPRC, grnboost2_AUROC = run_GRNBoost2(data_args,species,seed,adata, ground_truth)
    algorithm_name.append('GRNBoost2')
    prec_list.append(grnboost2_Eprec)
    Erec_list.append(grnboost2_Erec)
    EPR_list.append(grnboost2_EPR)
    AUPRC_list.append(grnboost2_AUPRC)
    AUROC_list.append(grnboost2_AUROC)

    # running GENIE3
    genie3_Eprec, genie3_Erec, genie3_EPR, genie3_AUPRC, genie3_AUROC = run_GENIE3(data_args,species,seed,adata, ground_truth)
    algorithm_name.append('GENIE3')
    prec_list.append(genie3_Eprec)
    Erec_list.append(genie3_Erec)
    EPR_list.append(genie3_EPR)
    AUPRC_list.append(genie3_AUPRC)
    AUROC_list.append(genie3_AUROC)

    pd.DataFrame({'algorithm':algorithm_name, 'Eprec':prec_list, 'Erec':Erec_list, 'EPR':EPR_list, 'AUPRC':AUPRC_list, 'AUROC':AUROC_list}).to_csv(os.path.join(data_args.output_dir,'metric.csv'),index=False)

if __name__ == "__main__":
    main()    