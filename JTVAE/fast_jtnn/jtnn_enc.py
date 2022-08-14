from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from JTVAE.fast_jtnn.mol_tree import Vocab, MolTree
from JTVAE.fast_jtnn.nnutils import create_var, index_select_ND

class JTNNEncoder(nn.Module):

    def __init__(self, hidden_size,cond_lnKD_size, cond_SelectPtoM_size, depth, embedding):
        super(JTNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.embedding = embedding  # [vocab_size, hidden_size]
        self.outputNN = nn.Sequential(
            nn.Linear(2 * hidden_size + cond_lnKD_size + cond_SelectPtoM_size, hidden_size),
            nn.ReLU()
        )
        self.GRU = GraphGRU(hidden_size, hidden_size, cond_lnKD_size, cond_SelectPtoM_size, depth=depth)

    def forward(self, fnode, fmess, node_graph, mess_graph, scope, cond_lnKD, cond_SelectPtoM):
        
        """
        @param fnode: 一维，长度为batch中node个数，存储的是node在vocabulary中的index
        @param fmess: 一维，长度为 node对 的个数，存储的是该 node对 中的左侧node的index
        @param node_graph: 二维，第一个维度是node的个数，每一个位置存储的是该node作为右侧(j)node时所有的node对在messages的indexes
        @param mess_graph: 二维，第一个维度是messages的长度，每一个位置存储的是当前edge信息的所有源头（上一级edge）
        @param scope: list of tuple，每一个tuple中有两个元素，左侧元素表示当前数据前所有的tree的node数目之和，右侧元素表示当前tree的node数目
        @param cond_lnKD: 二维，第一个维度是node个数，第二个维度是cond_lnKD_size
        @param cond_SelectPtoM: 二维，第一个维度是node个数，第二个为都市cond_SelectPtoM_size
        """

        fnode = create_var(fnode)
        fmess = create_var(fmess)
        node_graph = create_var(node_graph)
        mess_graph = create_var(mess_graph)
        messages = create_var(torch.zeros(mess_graph.size(0), self.hidden_size))  # 维度：[len(messages), hidden_size], 初始化为0

        fnode = self.embedding(fnode)  # 先对node进行embedding  [number of nodes, hidden_size]
        if cond_lnKD is not None:
            cond_lnKD = create_var(cond_lnKD)
            fnode = torch.cat((fnode, cond_lnKD), dim=1)
        if cond_SelectPtoM is not None:
            cond_SelectPtoM = create_var(cond_SelectPtoM)
            fnode = torch.cat((fnode, cond_SelectPtoM), dim=1)                
        fmess = index_select_ND(fnode, 0, fmess)  # 应该需要在这个后面加上对应的condition, [number of nodes pairs, hidden_size]
        messages = self.GRU(messages, fmess, mess_graph)  # [len(messages), hidden_size]]

        mess_nei = index_select_ND(messages, 0, node_graph)
        node_vecs = torch.cat([fnode, mess_nei.sum(dim=1)], dim=-1)
        node_vecs = self.outputNN(node_vecs)

        max_len = max([x for _,x in scope])
        batch_vecs = []
        for st,le in scope:
            cur_vecs = node_vecs[st]
            batch_vecs.append( cur_vecs )

        tree_vecs = torch.stack(batch_vecs, dim=0) 
        return tree_vecs, messages  
        
    @staticmethod
    def tensorize(tree_batch):
        # scope是一个list of tuple of two items
        # scope中，每一个tuple中有两个元素，左侧元素表示当前数据前所有的tree的node数目之和，右侧元素表示当前tree的node数目
        # 该方法中没有用到scope
        # node_batch是一个node实例列表

        node_batch = [] 
        scope = []
        for tree in tree_batch:
            scope.append((len(node_batch), len(tree.nodes)))
            node_batch.extend(tree.nodes)
        
        return JTNNEncoder.tensorize_nodes(node_batch, scope)
    
    @staticmethod
    def tensorize_nodes(node_batch, scope):
        # scope是一个list of tuple of two items
        # scope中，每一个tuple中有两个元素，左侧元素表示当前node的index，右侧元素表示该元素的neighbor个数 ？
        # 该方法中没有用到scope
        # node_batch是一个node实例列表

        messages,mess_dict = [None],{}
        fnode = []
        for x in node_batch:
            fnode.append(x.wid)  # x.wid 表示当前node在vocabulary中的idx (以下以)
            for y in x.neighbors:
                mess_dict[(x.idx,y.idx)] = len(messages)  # x.idx 表示当前node在当前batch下的index（以下表示为"index"）
                messages.append((x,y))  # # messages存储的是(x,y)实例对，其长度为所有实例对的总和，其中包括了正序&反序
                # 即 假设有node A 和 node B，那么就会存储一个(A, B)，也会存储一个(B, A)
        
        node_graph = [[] for i in range(len(node_batch))]
        mess_graph = [[] for i in range(len(messages))]
        fmess = [0] * len(messages)

        for x,y in messages[1:]:
            mid1 = mess_dict[(x.idx,y.idx)]  # mid1表示当前(x,y)node对在messages中的次序, m_ij
            fmess[mid1] = x.idx              # fmess 存储了当前次序下的 node对 中左侧node的index
            node_graph[y.idx].append(mid1)   # node_graph 在右侧node的index位置上添加了当前node对在messages中的次序
            for z in y.neighbors:
                if z.idx == x.idx: continue      # 如果y的neighbor就是x，那么跳过该步骤
                mid2 = mess_dict[(y.idx,z.idx)]  # mid2 表示当前(y,x) node对 在messages中的次序，m_ji
                mess_graph[mid2].append(mid1)    # mess_graph 在mid2的位置上 添加了 (x,y)node对在messages中的次序
        # 下面要统一维度
        max_len = max([len(t) for t in node_graph] + [1])
        for t in node_graph:
            pad_len = max_len - len(t)
            t.extend([0] * pad_len)

        max_len = max([len(t) for t in mess_graph] + [1])
        for t in mess_graph:
            pad_len = max_len - len(t)
            t.extend([0] * pad_len)

        mess_graph = torch.LongTensor(mess_graph)  # 二维的，第一个维度是messages的长度
        node_graph = torch.LongTensor(node_graph)  # 二维的，第一个维度是node的个数
        fmess = torch.LongTensor(fmess)   # 一维的，长度是messages的长度
        fnode = torch.LongTensor(fnode)   # 一维的，长度是node的个数
        
        return (fnode, fmess, node_graph, mess_graph, scope), mess_dict

class GraphGRU(nn.Module):

    def __init__(self, input_size, hidden_size, cond_lnKD_size, cond_SelectPtoM_size, depth):
        super(GraphGRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size + cond_lnKD_size + cond_SelectPtoM_size
        self.depth = depth

        self.W_z = nn.Linear(self.input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(self.input_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(self.input_size + hidden_size, hidden_size)

    def forward(self, h, x, mess_graph):
        mask = torch.ones(h.size(0), 1)
        mask[0] = 0
        mask = create_var(mask)
        for it in range(self.depth):
            h_nei = index_select_ND(h, 0, mess_graph)
            sum_h = h_nei.sum(dim=1)
            z_input = torch.cat([x, sum_h], dim=1)
            z = torch.sigmoid(self.W_z(z_input))

            r_1 = self.W_r(x).view(-1, 1, self.hidden_size)
            r_2 = self.U_r(h_nei)
            r = torch.sigmoid(r_1 + r_2)
            
            gated_h = r * h_nei
            sum_gated_h = gated_h.sum(dim=1)
            h_input = torch.cat([x, sum_gated_h], dim=1)
            pre_h = torch.tanh(self.W_h(h_input))
            h = (1.0 - z) * sum_h + z * pre_h
            h = h * mask

        return h
