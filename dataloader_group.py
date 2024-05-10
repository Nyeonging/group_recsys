import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world_group
from world_group import cprint
from time import time


class BasicDatasetGroup(Dataset): # abstract class
    def __init__(self):
        print("init dataset")
    # user
    @property
    def n_users(self): # number of users
        raise NotImplementedError
    @property
    def m_items(self): # number of items
        raise NotImplementedError
    @property
    def trainDataSize(self):
        raise NotImplementedError
    @property
    def testDict(self): # test data
        raise NotImplementedError
    @property
    def allPos(self): # positive items
        raise NotImplementedError
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    def getUserPosItems(self, users): # positive items of users
        raise NotImplementedError
    def getUserNegItems(self, users): # negative items of users
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    def getSparseGraph(self): # get the graph
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A =
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError
    # group
    @property
    def n_groups(self):
        raise NotImplementedError
    @property
    def m_group_items(self):
        raise NotImplementedError
    @property
    def trainDataSizeGroup(self):
        return NotImplementedError
    @property
    def testDict_group(self):
        return NotImplementedError
    @property
    def allPos_group(self): # positive items
        raise NotImplementedError
    def getGroupItemFeedback(self, groups, items):
        raise NotImplementedError
    def getGroupPosItems(self, groups): # positive items of groups
        raise NotImplementedError
    def getGroupNegItems(self, groups): # negative items of groups
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    def getSparseGraphGroup(self): # get the graph
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A =
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

    
class GroupLoader(BasicDatasetGroup):
    """
    Dataset type for pytorch \n
    Incldue graph information
    CAMRa2011 dataset
    """
    def __init__(self, config = world_group.config,path="../data/CAMRa2011"):
        # train or test
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train'] #0
        #user
        self.n_user = 0
        self.m_item = 0

        #group data 불러오기 
        group_train_file = path + '/groupRatingTrain.txt'
        group_test_file = path + '/groupRatingTest.txt'
        train_file = path + '/userRatingTrain.txt'
        test_file = path + '/userRatingTest.txt'
        self.path = path

        #user data
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        group_train_data, group_test_data = {}, {}
        self.traindataSize = 0
        self.testDataSize = 0

        # Loading userRatingTrain.txt
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
        self.Graph = None

        # Loading userRatingTest.txt
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0]) #user id
                    testUniqueUsers.append(uid) 
                    testUser.extend([uid] * len(items)) 
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        self.Graph_group = None
        self.Graph = None

        # Loading groupRatingTrain.txt
        with open(group_train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    group_id = int(l[0])
                    items = [int(i) for i in l[1:]]
                    group_train_data[group_id] = items

        # Loading groupRatingTest.txt
        with open(group_test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    group_id = int(l[0])
                    items = [int(i) for i in l[1:]]
                    group_test_data[group_id] = items

        self.group_train_data = group_train_data # 딕셔너리 형태
        self.group_test_data = group_test_data

        # Calculating sparsity
        sparsity = (self.trainDataSize + self.testDataSize) / self.n_users / self.m_items
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"Sparsity: {sparsity}")
        self.UserItemNet, self.groupItemNet = self.constructUserItemNet(trainUser, trainItem, group_train_data)

        # Calculate users_D and items_D using UserItemNet
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()

        # Calculate groups_D and groups_D using groupItemNet
        group_users_D = np.array(self.groupItemNet.sum(axis=1)).squeeze()
        group_items_D = np.array(self.groupItemNet.sum(axis=0)).squeeze()

        # Handle cases where there are no interactions
        self.users_D[self.users_D == 0.] = 1
        self.items_D[self.items_D == 0.] = 1.
        group_users_D[group_users_D == 0.] = 1
        group_items_D[group_items_D == 0.] = 1.

        # Pre-calculate for UserItemNet
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()

        # Pre-calculate for groupItemNet
        self._allPos_group = self.getGroupPosItems(list(range(self.n_groups)))
        self.__testDict_group = self.__build_test_group()
        print("Dataset is ready to go")

    def constructUserItemNet(self, trainUser, trainItem, group_train_data):
        """
        Construct User-Item Network including group data.
        Args:
            trainUser: array of user indices
            trainItem: array of item indices
            group_train_data: dictionary containing group ratings
        Returns:
            tuple: (User-Item Network, Group-Item Network)
        """
        UserItemNet = csr_matrix((np.ones(len(trainUser)), (trainUser, trainItem)),
                                      shape=(self.n_user, self.m_item))
        group_indices_map = {}
        current_index = 0
        for group_id, items in group_train_data.items():
            group_indices_map[group_id] = current_index
            current_index += len(items)
        row_indices = []
        col_indices = []
        for group_id, items in group_train_data.items():
            for item_id in items:
                row_indices.append(group_id)
                col_indices.append(item_id)
        GroupItemNet = csr_matrix((np.ones(len(row_indices)), (row_indices, col_indices)),
                                shape=(self.n_groups, self.m_item))
        return (UserItemNet, GroupItemNet)


    @property
    def n_users(self):
        return self.n_user
    @property
    def m_items(self):
        return self.m_item
    @property
    def trainDataSize(self):
        return self.traindataSize
    @property
    def testDict(self):
        return self.__testDict
    @property
    def allPos(self):
        return self._allPos


    #group
    @property
    def n_groups(self):
        return len(self.group_train_data)
    @property
    def n_group_items(self):
        return self.groupItemNet.shape[1]
    @property
    def TrainDataSizeGroup(self):
        return sum(len(items) for items in self.group_train_data.values())
    @property
    def testDict_group(self):
        return self.__testDict_group
    @property
    def allPos_group(self):
        return self._allPos_group
    def _split_A_hat(self,A, is_group=False): # split the graph
        A_fold = []
        if not is_group:  # If splitting User-Item Network
            fold_len = (self.n_users + self.m_items) // self.folds
            for i_fold in range(self.folds):
                start = i_fold * fold_len
                if i_fold == self.folds - 1:
                    end = self.n_users + self.m_items
                else:
                    end = (i_fold + 1) * fold_len
                A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world_group.device))
        else:  # If splitting Group-Item Network
            fold_len = (self.n_groups + self.m_group_items) // self.folds
            for i_fold in range(self.folds):
                start = i_fold * fold_len
                if i_fold == self.folds - 1:
                    end = self.n_groups + self.m_group_items
                else:
                    end = (i_fold + 1) * fold_len
                A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world_group.device))
        return A_fold
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
    
    def getSparseGraph(self):
        print("loading user-item adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)
            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj, is_group=False)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world_group.device)
                print("don't split the matrix")
        return self.Graph
    
    def getSparseGraphGroup(self):
        print("loading group-item adjacency matrix")
        if self.Graph_group is None:
            print("generating adjacency matrix")
            s = time()
            adj_mat = sp.dok_matrix((self.n_groups + self.m_group_items, self.n_groups + self.m_group_items), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = self.groupItemNet.tolil()
            adj_mat[:self.n_groups, self.n_groups:] = R
            adj_mat[self.n_groups:, :self.n_groups] = R.T
            adj_mat = adj_mat.todok()
            # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            end = time()
            print(f"costing {end-s}s, saved norm_mat...")
            sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)
        if self.split == True:
            self.Graph_group = self._split_A_hat(norm_adj, is_group=True)
            print("done split matrix")
        else:
            self.Graph_group = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph_group = self.Graph_group.coalesce().to(world_group.device)
            print("don't split the matrix")
        return self.Graph_group
    
    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
    def __build_test_group(self):
        """
        return:
            dict: {group: [items]}
        """
        test_data = {}
        for group, item in self.group_test_data.items(): #self.group_testdata:dictionary
            if test_data.get(group):
                test_data[group].append(item)
            else:
                test_data[group] = [item]
        return test_data
    
    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))
    
    def getGroupItemFeedback(self, groups, group_items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        return np.array(self.groupItemNet[groups, group_items]).astype('uint8').reshape((-1,))
    
    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    def getGroupPosItems(self, groups):
        posItems = []
        for group in groups:
            posItems.append(self.groupItemNet[group].nonzero()[1])
        return posItems
    
