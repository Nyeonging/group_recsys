"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world_group
import torch
import time
#from dataloader import BasicDataset
from dataloader_group import BasicDatasetGroup
from torch import nn
import scipy.sparse as sp
import numpy as np
from sparsesvd import sparsesvd
from tqdm import tqdm


class BasicModel(nn.Module):  #이 모델이 가져야할 최소한의 기능들
    def __init__(self): 
        super(BasicModel, self).__init__() 

    def getUsersRating(self, users): #사용자의 평가를 얻는 함수 #getUsersRating, users(userID)
        raise NotImplementedError #뒤에 하위클래스에서 이 메서드를 구현
    
    def getGroupsRating(self, groups):  #group의 평가를 얻는 함수 #getGroupsRating, groups(groupID)
        raise NotImplementedError  
    

class PairWiseModel(BasicModel):  #basicmodel 상속 
    def __init__(self): 
        super(PairWiseModel, self).__init__() 

    def user_bpr_loss(self, users, pos, neg):   #loss function, groups 
        """
        Parameters: 
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return: 
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    def group_bpr_loss(self, groups, pos, neg):   #loss function, groups
        """
        Parameters:
            groups: groups list 
            pos: positive items for corresponding groups
            neg: negative items for corresponding groups
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError

class PureMF(BasicModel):   #user 와 item 간의 interactio 을 기반으로 추천하는 모델(순수행렬분해모델)
    def __init__(self,  
                 config:dict, 
                 dataset:BasicDatasetGroup): 
        super(PureMF, self).__init__() 
        self.num_users  = dataset.n_users 
        self.num_groups = dataset.n_groups 
        self.num_items  = dataset.m_items 
        self.num_group_items = dataset.m_group_items 
        self.latent_dim = config['latent_dim_rec'] 
        self.f = nn.Sigmoid() 
        self.__init_weight() 
        self.latent_dim = config['latent_dim_rec'] 
        self.f = nn.Sigmoid() 
        self.__init_weight() 

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(  
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_group = torch.nn.Embedding( #embedding_group 
            num_embeddings=self.num_groups, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_group_item = torch.nn.Embedding( #embedding_group_items 
            num_embeddings=self.num_group_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users): #user-item 평가 점수 계산 
        users = users.long() 
        users_emb = self.embedding_user(users) #groups_emb 
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    

    def getGroupsRating(self, groups): #group-item 평가 점수 계산
        groups = groups.long() 
        groups_emb = self.embedding_group(groups) #groups_emb
        items_emb = self.embedding_group_item.weight
        scores = torch.matmul(groups_emb, items_emb.t())
        return self.f(scores)
    
    def user_bpr_loss(self, users, pos, neg): #BPR loss 계산 
        users_emb = self.embedding_user(users.long()) 
        pos_emb   = self.embedding_item(pos.long()) 
        neg_emb   = self.embedding_item(neg.long()) 
        pos_scores= torch.sum(users_emb*pos_emb, dim=1) 
        neg_scores= torch.sum(users_emb*neg_emb, dim=1) 
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores)) #pos 와 neg 간의 점수 차이 계산
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
    
    #group BPR loss 계산 
    def group_bpr_loss(self, groups, pos, neg): 
        groups_emb = self.embedding_group(groups.long()) 
        pos_emb   = self.embedding_group_item(pos.long()) 
        neg_emb   = self.embedding_group_item(neg.long()) 
        pos_scores= torch.sum(groups_emb*pos_emb, dim=1) 
        neg_scores= torch.sum(groups_emb*neg_emb, dim=1) 
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores)) #pos 와 neg 간의 점수 차이 계산
        reg_loss = (1/2)*(groups_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(groups))
        return loss, reg_loss

    def forward(self, users, items, group, group_items): 
        # train group
        if (group is not None) and (users is None):
            out = self.group_forward(group, group_items)
        # train user
        else:
            out = self.user_forward(users, items)
        return out
    
    def user_forward(self, users, items): 
        users = users.long() 
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)
    
    #groups
    def group_forward(self, groups, items): 
        groups = groups.long() 
        items = items.long()
        group_emb = self.embedding_group(groups) #groups_emb
        items_emb = self.embedding_group_item(items)
        scores = torch.sum(group_emb*items_emb, dim=1)
        return self.f(scores)
    
class LightGCN(BasicModel): 
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDatasetGroup):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader_group.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world_group.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
    
class LGCN_IDE(object): 
    def __init__(self, adj_mat):
        self.adj_mat = adj_mat
        
    def train(self):
        adj_mat = self.adj_mat
        start = time.time()
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        d_mat_i = d_mat
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        d_mat_u = d_mat
        d_mat_u_inv = sp.diags(1/d_inv)
        norm_adj = norm_adj.dot(d_mat)
        self.norm_adj = norm_adj.tocsr()
        end = time.time()
        print('training time for LGCN-IDE', end-start)
        
    def getUsersRating(self, batch_users, ds_name):
        norm_adj = self.norm_adj
        batch_test = np.array(norm_adj[batch_users,:].todense())
        U_1 = batch_test @ norm_adj.T @ norm_adj
        if(ds_name == 'gowalla'):
            U_2 = U_1 @ norm_adj.T @ norm_adj
            return U_2
        else:
            return U_1

class GF_CF(object):
    def __init__(self, user_adj_mat, group_adj_mat): 
        self.user_adj_mat = user_adj_mat   
        self.group_adj_mat = group_adj_mat 

#     def train(self):
#         user_adj_mat = self.user_adj_mat
#         group_adj_mat = self.group_adj_mat
#         start = time.time()
#         user_rowsum = np.array(user_adj_mat.sum(axis=1))
#         group_rowsum = np.array(group_adj_mat.sum(axis=1))

#         user_d_inv = np.power(user_rowsum, -0.5).flatten()
#         group_d_inv = np.power(group_rowsum, -0.5).flatten()
    
#         user_d_inv[np.isinf(user_d_inv)] = 0.
#         group_d_inv[np.isinf(group_d_inv)] = 0.
#         user_d_mat = sp.diags(user_d_inv)
#         group_d_mat = sp.diags(group_d_inv)
#         user_norm_adj = user_d_mat.dot(user_adj_mat)
#         group_norm_adj = group_d_mat.dot(group_adj_mat)
#         user_colsum = np.array(user_adj_mat.sum(axis=0))
#         group_colsum = np.array(group_adj_mat.sum(axis=0))
#         user_d_inv = np.power(user_colsum, -0.5).flatten()
#         group_d_inv = np.power(group_colsum, -0.5).flatten()
#         user_d_inv[np.isinf(user_d_inv)] = 0.
#         group_d_inv[np.isinf(group_d_inv)] = 0.
#         user_d_mat = sp.diags(user_d_inv)
#         group_d_mat = sp.diags(group_d_inv)
#         self.user_d_mat_i = user_d_mat
#         self.group_d_mat_i = group_d_mat
#         self.user_d_mat_i_inv = sp.diags(1/user_d_inv)
#         self.group_d_mat_i_inv = sp.diags(1/group_d_inv)
#         self.user_d_mat_i = user_d_mat
#         self.group_d_mat_i = group_d_mat
#         user_norm_adj = user_norm_adj.dot(user_d_mat)
#         group_norm_adj = group_norm_adj.dot(group_d_mat)
#         self.user_norm_adj = user_norm_adj.tocsc()
#         self.group_norm_adj = group_norm_adj.tocsc()
#         user_ut, user_s, self.user_vt = sparsesvd(self.user_norm_adj, 256) #svd 계산도 데이터셋에 맞게 다시
#         group_ut, group_s, self.group_vt = sparsesvd(self.group_norm_adj, 256) #svd 계산도 데이터셋에 맞게 다시
#         end = time.time()
#         print('training time for GF-CF', end-start)

    # def train(self):
    #     user_adj_mat = self.user_adj_mat
    #     group_adj_mat = self.group_adj_mat

    #     start = time.time()
    #     user_rowsum = np.array(user_adj_mat.sum(axis=1))
    #     group_rowsum = np.array(group_adj_mat.sum(axis=1))

    #     # Prevent divide by zero error
    #     user_d_inv = np.where(user_rowsum != 0, np.power(user_rowsum, -0.5), 0)
    #     group_d_inv = np.where(group_rowsum != 0, np.power(group_rowsum, -0.5), 0)

    #     user_d_mat = sp.diags(user_d_inv)
    #     group_d_mat = sp.diags(group_d_inv)

    #     # # 사용자 대각 행렬 생성
    #     # user_d_mat = sp.diags(user_d_inv, offsets=0)
        
    #     # # 그룹 대각 행렬 생성
    #     # group_d_mat = sp.diags(group_d_inv, offsets=0)
    

    #     user_norm_adj = user_d_mat.dot(user_adj_mat)
    #     group_norm_adj = group_d_mat.dot(group_adj_mat)

    #     user_colsum = np.array(user_adj_mat.sum(axis=0))
    #     group_colsum = np.array(group_adj_mat.sum(axis=0))

    #     EPSILON = 1e-8  # 아주 작은 값
    #     user_d_inv = np.where(user_colsum != 0, np.power(user_colsum + EPSILON, -0.5), 0).flatten()
    #     group_d_inv = np.where(group_colsum != 0, np.power(group_colsum + EPSILON, -0.5), 0).flatten()

    #     user_d_mat = sp.diags(user_d_inv)
    #     group_d_mat = sp.diags(group_d_inv)

    #     self.user_d_mat_i = user_d_mat
    #     self.group_d_mat_i = group_d_mat

    #     self.user_d_mat_i_inv = sp.diags(1/user_d_inv)
    #     self.group_d_mat_i_inv = sp.diags(1/group_d_inv)

    #     self.user_d_mat_i = user_d_mat
    #     self.group_d_mat_i = group_d_mat

    #     user_norm_adj = user_norm_adj.dot(user_d_mat)
    #     group_norm_adj = group_norm_adj.dot(group_d_mat)

    #     self.user_norm_adj = user_norm_adj.tocsc()
    #     self.group_norm_adj = group_norm_adj.tocsc()

    #     user_ut, user_s, self.user_vt = sparsesvd(self.user_norm_adj, 256) #svd 계산도 데이터셋에 맞게 다시
    #     group_ut, group_s, self.group_vt = sparsesvd(self.group_norm_adj, 256) #svd 계산도 데이터셋에 맞게 다시

    #     end = time.time()
    #     print('training time for GF-CF', end-start)

    def train(self):
        user_adj_mat = self.user_adj_mat
        group_adj_mat = self.group_adj_mat
        start = time.time()
        user_rowsum = np.array(user_adj_mat.sum(axis=1))
        group_rowsum = np.array(group_adj_mat.sum(axis=1))
        user_d_inv = np.where(user_rowsum != 0, np.power(user_rowsum, -0.5), 0).flatten()
        group_d_inv = np.where(group_rowsum != 0, np.power(group_rowsum, -0.5), 0).flatten()
        # user_d_inv = np.power(user_rowsum, -0.5).flatten()
        # group_d_inv = np.power(group_rowsum, -0.5).flatten()
        user_d_inv[np.isinf(user_d_inv)] = 0.
        group_d_inv[np.isinf(group_d_inv)] = 0.
        user_d_mat = sp.diags(user_d_inv)
        group_d_mat = sp.diags(group_d_inv)
        user_norm_adj = user_d_mat.dot(user_adj_mat)
        group_norm_adj = group_d_mat.dot(group_adj_mat)
        user_colsum = np.array(user_adj_mat.sum(axis=0))
        group_colsum = np.array(group_adj_mat.sum(axis=0))
        EPSILON = 1e-8  # 아주 작은 값
        user_d_inv = np.where(user_colsum != 0, np.power(user_colsum + EPSILON, -0.5), 0).flatten()
        group_d_inv = np.where(group_colsum != 0, np.power(group_colsum + EPSILON, -0.5), 0).flatten()
        user_d_inv[np.isinf(user_d_inv)] = 0.
        group_d_inv[np.isinf(group_d_inv)] = 0.
        user_d_mat = sp.diags(user_d_inv)
        group_d_mat = sp.diags(group_d_inv)
        self.user_d_mat_i = user_d_mat
        self.group_d_mat_i = group_d_mat
        # self.user_d_mat_i_inv = sp.diags(1/user_d_inv)
        # self.group_d_mat_i_inv = sp.diags(1/group_d_inv)
        user_nonzero_indices = user_d_inv != 0
        group_nonzero_indices = group_d_inv != 0
        # 0이 아닌 경우에만 역수를 취한 후, 대각 행렬을 생성합니다.
        self.user_d_mat_i_inv = sp.diags(1/user_d_inv[user_nonzero_indices], offsets=0, shape=(np.sum(user_nonzero_indices), np.sum(user_nonzero_indices)))
        self.group_d_mat_i_inv = sp.diags(1/group_d_inv[group_nonzero_indices], offsets=0, shape=(np.sum(group_nonzero_indices), np.sum(group_nonzero_indices)))
        user_norm_adj = user_norm_adj.dot(user_d_mat)
        group_norm_adj = group_norm_adj.dot(group_d_mat)
        self.user_norm_adj = user_norm_adj.tocsc()
        self.group_norm_adj = group_norm_adj.tocsc()
        user_ut, user_s, self.user_vt = sparsesvd(self.user_norm_adj, 256) #svd 계산도 데이터셋에 맞게 다시
        group_ut, group_s, self.group_vt = sparsesvd(self.group_norm_adj, 256) #svd 계산도 데이터셋에 맞게 다시
        end = time.time()
        print('training time for GF-CF', end-start)


    def getUsersRating(self, batch_users, ds_name): #ds_name 바꿔주어, 평가 점수 계산 방법 맞게 
        norm_adj = self.norm_adj
        adj_mat = self.adj_mat
        batch_test = np.array(adj_mat[batch_users,:].todense())
        U_2 = batch_test @ norm_adj.T @ norm_adj
        if(ds_name == 'amazon-book'):   
            ret = U_2 
        else: 
            U_1 = batch_test @  self.d_mat_i @ self.vt.T @ self.vt @ self.d_mat_i_inv 
            ret = U_2 + 0.3 * U_1 
        return ret
    
    def getGroupsRating(self, batch_users, ds_name): #ds_name 바꿔주어, 평가 점수 계산 방법 맞게 
        
        user_norm_adj = self.user_norm_adj
        group_norm_adj = self.group_norm_adj

        user_adj_mat = self.user_adj_mat
        group_adj_mat = self.group_adj_mat

        user_batch_test = np.array(user_adj_mat[batch_users,:].todense())
        group_batch_test = np.array(group_adj_mat[batch_users,:].todense()) #batch 수는 user 와 같게 설정한다는 가정 하에

        U_2 = user_batch_test @ user_norm_adj.T @ user_norm_adj
        if(ds_name == 'agree-data'):
            U_1 = group_batch_test @ group_norm_adj.T @ group_norm_adj
            ret = U_2 + 0.3 * U_1
        else:  
            ret = U_2  
        return ret
