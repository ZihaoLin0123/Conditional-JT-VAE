import os, random
import pickle as pickle
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from JTVAE.fast_jtnn.mol_tree import MolTree
from JTVAE.fast_jtnn.jtnn_enc import JTNNEncoder
from JTVAE.fast_jtnn.mpn import MPN
from JTVAE.fast_jtnn.jtmpn import JTMPN

class PairTreeFolder(object):

    def __init__(self, data_folder, vocab, batch_size, num_workers=4, shuffle=True, y_assm=True, replicate=None):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.y_assm = y_assm
        self.shuffle = shuffle

        if replicate is not None: 
            self.data_files = self.data_files * replicate

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, 'rb') as f:
                data = pickle.load(f)

            if self.shuffle: 
                random.shuffle(data) 

            batches = [data[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)]
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            dataset = PairTreeDataset(batches, self.vocab, self.y_assm)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x:x[0], num_workers=self.num_workers)

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader

class MolTreeFolder(object):

    def __init__(self, data_folder, vocab, batch_size, num_workers=8, shuffle=True, assm=True, replicate=None,
                 cond_lnKD_path=None, cond_SelectPtoM_path=None, cond_lnKD_size=0, cond_SelectPtoM_size=0):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm
        self.cond_lnKD_size = cond_lnKD_size
        self.cond_SelectPtoM_size = cond_SelectPtoM_size
        self.cond_lnKD_path = cond_lnKD_path
        self.cond_SelectPtoM_path = cond_SelectPtoM_path
        self.cond_lnKD, self.cond_SelectPtoM = None, None
        self.set_cond()

        if replicate is not None: 
            self.data_files = self.data_files * replicate

    def set_cond(self):

        if self.cond_lnKD_path is not None:
            self.cond_lnKD = np.loadtxt(self.cond_lnKD_path).tolist()

        if self.cond_SelectPtoM_path is not None:
            self.cond_SelectPtoM = np.loadtxt(self.cond_SelectPtoM_path).tolist()
        
        # print("len(self.cond_lnKD): ", len(self.cond_lnKD))
        # print("len(self.cond_SelectPtoM): ", len(self.cond_SelectPtoM))

    def __iter__(self):

        fn_list = []
        for fn in self.data_files:
            if fn[:7] in 'tensors':
                fn_list.append(fn)
                print("we have file ", fn)
        # print("self.data_files: ", self.data_files)
        # print("fn_list: ", fn_list)
        # reorder the file path list
        # print(good)
        fn_list_new = []
        path_head = fn_list[0][:-5]
        for i in range(len(fn_list)):
            fn_list_new.append(path_head + str(i) + '.pkl')
        
        print("total file names: ", fn_list_new)

        for fn in fn_list_new:
            fn = os.path.join(self.data_folder, fn)
            print("current file name is ", fn)
            with open(fn, 'rb') as f:
                data = pickle.load(f)

            data_list = []
            for d in data:
                if self.cond_lnKD is not None:
                    d_lnKD = self.cond_lnKD[0]
                    self.cond_lnKD.pop(0)
                else:
                    d_lnKD = None

                if self.cond_SelectPtoM is not None:
                    d_SelectPtoM = self.cond_SelectPtoM[0]
                    self.cond_SelectPtoM.pop(0)
                else:
                    d_SelectPtoM = None

                data_list.append((d, d_lnKD, d_SelectPtoM))

            del data
            data = data_list
            # each item in data list is a tuple with 3 items: (data, lnKD, SelectPtoM), the last two could be None

            if self.shuffle: 
                random.shuffle(data) 

            batches = [data[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)]
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            dataset = MolTreeDataset(batches, self.vocab, self.cond_lnKD_size, self.cond_SelectPtoM_size, self.assm)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x:x[0], num_workers=self.num_workers, pin_memory=True)

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader, data_list

class PairTreeDataset(Dataset):

    def __init__(self, data, vocab, cond_lnKD_size, cond_SelectPtoM_size, y_assm):
        self.data = data
        self.vocab = vocab
        self.y_assm = y_assm
        self.cond_lnKD_size = cond_lnKD_size
        self.cond_SelectPtoM_size = cond_SelectPtoM_size

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        batch0, batch1 = list(zip(*self.data[idx]))
        return tensorize(batch0, self.vocab, self.cond_lnKD_size, self.cond_SelectPtoM_size, assm=False),\
               tensorize(batch1, self.vocab, self.cond_lnKD_size, self.cond_SelectPtoM_size, assm=self.y_assm)

class MolTreeDataset(Dataset):

    def __init__(self, data, vocab, cond_lnKD_size, cond_SelectPtoM_size, assm=True):
        self.data = data
        self.vocab = vocab
        self.assm = assm
        self.cond_lnKD_size = cond_lnKD_size
        self.cond_SelectPtoM_size = cond_SelectPtoM_size

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return tensorize(self.data[idx], self.vocab, self.cond_lnKD_size, self.cond_SelectPtoM_size, assm=self.assm, type="train")

def tensorize(batch, vocab, cond_lnKD_size, cond_SelectPtoM_size, assm=True, type="train"):  # type could be train or pretrain
    
    tree_batch = []
    lnKD_batch = []
    SelectPtoM_batch = []
   
    # print("JTVAE/fast_molvae/jtnnvae_train.py, line 177, batch: ", batch) 
    if type == "train":
        for item in batch:
            tree_batch.append(item[0])
            lnKD_batch.append(item[1])
            SelectPtoM_batch.append(item[2])
    else:
        tree_batch = list(batch[0])
        lnKD_batch = list(batch[1])
        SelectPtoM_batch = list(batch[2])
   
    if lnKD_batch[0] is not None:
        # print("jtvae/fast_jtnn/datautils.py, line 183, lnKD_batch: ", lnKD_batch)
        lnKD_batch = set_condition(lnKD_batch, -13, 0)
    if SelectPtoM_batch[0] is not None:
        SelectPtoM_batch = set_condition(SelectPtoM_batch, -9, 8)

    set_batch_nodeID(tree_batch, vocab) 
    smiles_batch = [tree.smiles for tree in tree_batch]
    
    jtenc_holder,mess_dict = JTNNEncoder.tensorize(tree_batch)
    mpn_holder = MPN.tensorize(smiles_batch)
    _, _, _, _, jtenc_scope = jtenc_holder
    _, _, _, _, mpn_scope, mpn_scope_bonds = mpn_holder
    jtenc_condition_holder = condition_tensorize(jtenc_scope, None, lnKD_batch, cond_lnKD_size, SelectPtoM_batch, cond_SelectPtoM_size)
    mpn_condition_holder = condition_tensorize(mpn_scope, mpn_scope_bonds, lnKD_batch, cond_lnKD_size, SelectPtoM_batch, cond_SelectPtoM_size)
    cond_lnKD_batch = batch_condition_tensorize(lnKD_batch, cond_lnKD_size) if lnKD_batch is not None else None
    cond_SelectPtoM_batch = batch_condition_tensorize(SelectPtoM_batch, cond_SelectPtoM_size) if SelectPtoM_batch is not None else None
    batch_condition_holder = cond_lnKD_batch, cond_SelectPtoM_batch

    if assm is False:
        return tree_batch, jtenc_holder, mpn_holder, jtenc_condition_holder, mpn_condition_holder, batch_condition_holder

    cands = []
    batch_idx = []
    for i,mol_tree in enumerate(tree_batch):
        for node in mol_tree.nodes:
            if node.is_leaf or len(node.cands) == 1: continue
            cands.extend( [(cand, mol_tree.nodes, node) for cand in node.cands] )
            batch_idx.extend([i] * len(node.cands))
            
    jtmpn_holder = JTMPN.tensorize(cands, mess_dict)
    batch_idx = torch.LongTensor(batch_idx)

    return tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder,batch_idx), jtenc_condition_holder, mpn_condition_holder, batch_condition_holder

def batch_condition_tensorize(cond, cond_size):
    cond_batch = None
    for i in range(len(cond)):
        cond_cls = cond[i]
        cond_vec = torch.zeros((1, cond_size))
        cond_vec[:,cond_cls] = 1
        if i == 0:
            cond_batch = cond_vec
        else:
            cond_batch = torch.cat((cond_batch, cond_vec), dim=0)

    return cond_batch


def create_cond_vec(cond, cond_size, scope):
    # print("len(cond): ", len(cond))
    # print("cond_size: ", cond_size)
    # print(scope)
    cond_final = None
    for i in range(len(cond)):
        num_node = scope[i][1]
        cond_cls = cond[i]
        cond_vec = torch.zeros((num_node, cond_size))
        cond_vec[:, cond_cls] = 1
        if i == 0:
            cond_final = cond_vec
        else:
            cond_final = torch.cat((cond_final, cond_vec), dim=0)
    return cond_final


def condition_tensorize(scope, scope_bonds, lnKD_batch, cond_lnKD_size, SelectPtoM_batch, cond_SelectPtoM_size):

    # cat fnode with condition
    if lnKD_batch[0] is not None:
        cond_lnKD = create_cond_vec(lnKD_batch, cond_lnKD_size, scope)
        cond_lnKD_bonds = None if scope_bonds is None else create_cond_vec(lnKD_batch, cond_lnKD_size, scope_bonds)
    else:
        cond_lnKD, cond_lnKD_bonds = None, None

    if SelectPtoM_batch[0] is not None:
        cond_SelectPtoM = create_cond_vec(SelectPtoM_batch, cond_SelectPtoM_size, scope)
        cond_SelectPtoM_bonds = None if scope_bonds is None else create_cond_vec(SelectPtoM_batch, cond_SelectPtoM_size, scope_bonds)
    else:
        cond_SelectPtoM, cond_SelectPtoM_bonds = None, None

    if scope_bonds is None:
        return cond_lnKD, cond_SelectPtoM
    else:
        return cond_lnKD, cond_SelectPtoM, cond_lnKD_bonds, cond_SelectPtoM_bonds


def set_condition(source, min_, max_):
    source = torch.tensor(source)  # change list to tensor
    s = torch.clamp(source, max=max_, min=min_)  # 将值限制在一定范围内，lnKD: [-13, 0], SelectPtoM: [-9, 8]
    s = torch.round(s)  # 四舍五入取整
    s = s - min_  # 将取值限制到从0 开始
    s = s.int().tolist()  # 转为int，转为list
    return s

def set_batch_nodeID(mol_batch, vocab): 
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1

class PropDataset(Dataset):

    def __init__(self, data_file, prop_file, cond_lnKD_path, cond_SelectPtoM_path):
        self.prop_data = np.loadtxt(prop_file)
        with open(data_file) as f:
            self.data = [line.strip("\r\n ").split()[0] for line in f]

        if cond_lnKD_path is not None:
            self.cond_lnKD = np.loadtxt(cond_lnKD_path).tolist()
        else:
            self.cond_lnKD = None

        if cond_SelectPtoM_path is not None:
            self.cond_SelectPtoM = np.loadtxt(cond_SelectPtoM_path).tolist()
        else:
            self.cond_SelectPtoM = None

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        smiles = self.data[idx]
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        mol_tree.assemble()

        if self.cond_lnKD is not None:
            cond_lnKD = self.cond_lnKD[idx]
        else:
            cond_lnKD = None

        if self.cond_SelectPtoM is not None:
            cond_SelectPtoM = self.cond_SelectPtoM[idx]
        else:
            cond_SelectPtoM = None

        return mol_tree, self.prop_data[idx], cond_lnKD, cond_SelectPtoM

def smiles_to_moltree(smiles, assm=True):
    mol_tree = MolTree(smiles)
    mol_tree.recover()
    if assm:
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)
    return mol_tree
