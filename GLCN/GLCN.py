import numpy as np
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from torch import Tensor
from collections import OrderedDict
import sys
sys.path.append("..")  # 상위 폴더를 import할 수 있도록 경로 추가
from cfg import get_cfg
from GTN.inits import glorot
cfg = get_cfg()
print(torch.cuda.device_count())
device =torch.device(cfg.cuda if torch.cuda.is_available() else "cpu")
print(device)


def sample_adjacency_matrix(weight_matrix):
    # weight_matrix는 n x n 텐서이며, 각 원소는 연결 확률을 나타냅니다.
    # 0과 1 사이의 uniform random matrix를 생성합니다.
    random_matrix = torch.rand(weight_matrix.size()).to(device)

    # weight_matrix의 확률 값을 사용하여 0 또는 1을 샘플링합니다.
    # weight_matrix의 각 원소가 해당 위치에서의 연결 확률을 나타내므로,
    # random_matrix가 그 확률 이하인 경우에는 연결(1)로, 그렇지 않으면 비연결(0)으로 판단합니다.
    adjacency_matrix = (random_matrix < weight_matrix).int()
    #print(adjacency_matrix)
    return adjacency_matrix


def gumbel_sigmoid(logits: Tensor, tau: float = 1, hard: bool = False, threshold: float = 0.5) -> Tensor:
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
    y_soft = gumbels.sigmoid()


    if hard:
        # Straight through.

        indices = (y_soft > threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        ret = y_hard - y_soft.detach() + y_soft

    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def weight_init_xavier_uniform(submodule):
    if isinstance(submodule, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(submodule.weight)
        submodule.bias.data.fill_(0.01)
    if isinstance(submodule, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(submodule.weight)
    elif isinstance(submodule, torch.nn.BatchNorm2d):
        submodule.weight.data.fill_(1.0)
        submodule.bias.data.zero_()


class GLCN(nn.Module):
    def __init__(self, feature_size, graph_embedding_size, link_prediction = True, feature_obs_size = None, skip_connection = False):
        super(GLCN, self).__init__()
        self.graph_embedding_size = graph_embedding_size
        self.link_prediction = link_prediction
        if self.link_prediction == True:
            self.feature_obs_size = feature_obs_size
            self.a_link = nn.Parameter(torch.empty(size=(self.feature_obs_size, 1)))
            nn.init.xavier_uniform_(self.a_link.data, gain=1.414)
            self.k_hop = int(os.environ.get("k_hop",2))
            self.sampling = bool(os.environ.get("sampling", True))
            self.skip_connection = skip_connection

            if self.skip_connection == True:
                graph_embedding_size = feature_size
                self.graph_embedding_size = feature_size

            if self.sampling == True:
                self.Ws = [nn.Parameter(torch.Tensor(feature_size, graph_embedding_size)) if k == 0 else nn.Parameter(torch.Tensor(size=(graph_embedding_size, graph_embedding_size))) for k in range(self.k_hop)]
                [glorot(W) for W in self.Ws]

                self.a = [nn.Parameter(torch.empty(size=(2 * graph_embedding_size, 1))) if k == 0 else nn.Parameter(torch.empty(size=(2 * graph_embedding_size, 1))) for k in range(self.k_hop)]
                [nn.init.xavier_uniform_(self.a[k].data, gain=1.414) for k in range(self.k_hop)]

                self.Ws = nn.ParameterList(self.Ws)
                self.a = nn.ParameterList(self.a)
            else:
                self.W = [nn.Parameter(torch.Tensor(size=(feature_size, graph_embedding_size))) if k == 0 else nn.Parameter(torch.Tensor(size=(graph_embedding_size, graph_embedding_size))) for k in range(self.k_hop)]
                [glorot(W) for W in self.W]
                self.W = nn.ParameterList(self.W)
        else:
            self.Ws = nn.Parameter(torch.Tensor(feature_size, graph_embedding_size))
            glorot(self.Ws)
            self.a = nn.Parameter(torch.empty(size=(2 * graph_embedding_size, 1)))
            nn.init.xavier_uniform_(self.a.data, gain=1.414)





    def _link_prediction(self, h, dead_masking, mini_batch = False):

        h = h.detach()
        h = h[:, :self.feature_obs_size]
        h = torch.einsum("ijk,kl->ijl", torch.abs(h.unsqueeze(1) - h.unsqueeze(0)), self.a_link)
        h = h.squeeze(2)
        A = gumbel_sigmoid(h, tau = float(os.environ.get("gumbel_tau",1.4)), hard = True, threshold = 0.5)
        D = torch.diag(torch.diag(A))
        A = A-D
        I = torch.eye(A.size(0)).to(device)
        A = A+I

        return A







    def _prepare_attentional_mechanism_input(self, Wq, Wv, k = None):
        if k == None:

            # N = Wq.size(0)
            #
            # # Prepare repeat and transpose tensors for broadcasting
            # Wh_repeated_in_chunks = Wq.repeat_interleave(N, dim=0)
            # Wh_repeated_alternating = Wq.repeat(N, 1)
            # all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating],dim=1)  # (N*N, 2*out_features)
            # e = torch.matmul(all_combinations_matrix, self.a).squeeze(1)
            # e = e.view(N, N)
            # print("전",e.shape)
            Wh1 = Wq
            Wh1 = torch.matmul(Wh1, self.a[:self.graph_embedding_size, : ])
            Wh2 = Wv
            Wh2 = torch.matmul(Wh2, self.a[self.graph_embedding_size:, :])
            e = Wh1 + Wh2.T
            # print("후", e.shape)
        else:
            Wh1 = Wq
            Wh1 = torch.matmul(Wh1, self.a[k][:self.graph_embedding_size, : ])
            Wh2 = Wv
            Wh2 = torch.matmul(Wh2, self.a[k][self.graph_embedding_size:, :])
            e = Wh1 + Wh2.T
            # N = Wq.size(0)
            #
            # # Prepare repeat and transpose tensors for broadcasting
            # Wh_repeated_in_chunks = Wq.repeat_interleave(N, dim=0)
            # Wh_repeated_alternating = Wq.repeat(N, 1)
            # all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating],dim=1)  # (N*N, 2*out_features)
            # e = torch.matmul(all_combinations_matrix, self.a[k]).squeeze(1)
            # e = e.view(N, N)
        return F.leaky_relu(e, negative_slope=cfg.negativeslope)



    def forward(self, A, X, dead_masking = False, mini_batch = False):
        if self.link_prediction == False:
            if mini_batch == False:
                E = A.to(device)
                num_nodes = X.shape[0]
                E = torch.sparse_coo_tensor(E.clone().detach(), torch.ones(torch.tensor(E.clone().detach()).shape[1]).to(device), (num_nodes, num_nodes)).long().to(device).to_dense()
                Wh = X @ self.Ws
                a = self._prepare_attentional_mechanism_input(Wh, Wh)
                zero_vec = -9e15 * torch.ones_like(E)
                a = torch.where(E > 0, a, zero_vec)
                a = F.softmax(a, dim = 1)
                H = torch.matmul(a, Wh)
            else:
                batch_size = X.shape[0]
                num_nodes = X.shape[1]
                Hs = torch.zeros([batch_size, num_nodes, self.graph_embedding_size]).to(device)

                for b in range(batch_size):
                    X_t = X[b,:,:]
                    E = torch.tensor(A[b]).long().to(device)
                    E = torch.sparse_coo_tensor(E.clone().detach(), torch.ones(torch.tensor(E.clone().detach()).shape[1]).to(device), (num_nodes, num_nodes)).long().to(device).to_dense()
                    Wh = X_t @ self.Ws
                    a = self._prepare_attentional_mechanism_input(Wh, Wh)
                    zero_vec = -9e15 * torch.ones_like(E)
                    a = torch.where(E > 0, a, zero_vec)
                    a = F.softmax(a, dim = 1)
                    H = torch.matmul(a, Wh)
                    Hs[b, :, :] = H
                H = Hs

            return H
        else:
            if mini_batch == False:
                A = self._link_prediction(X, dead_masking, mini_batch = mini_batch)
                H = X
                for k in range(self.k_hop):
                    Wh = H @ self.Ws[k]
                    a = self._prepare_attentional_mechanism_input(Wh, Wh, k=k)
                    zero_vec = -9e15 * torch.ones_like(A)
                    a = torch.where(A > 0, A * a, zero_vec)
                    a = F.softmax(a, dim=1)
                    H = torch.matmul(a, Wh)
                return H, A, X
            else:
                num_nodes = X.shape[1]
                batch_size = X.shape[0]
                Hs = torch.zeros([batch_size, num_nodes, self.graph_embedding_size]).to(device)
                As = torch.zeros([batch_size, num_nodes, num_nodes]).to(device)
                for b in range(batch_size):
                    A = self._link_prediction(X[b], dead_masking[b], mini_batch = mini_batch)
                    As[b, :, :] = A
                    H = X[b, :, :]
                    for k in range(self.k_hop):
                        # if k != 0:
                        #     A = A.detach()
                        Wh = H @ self.Ws[k]
                        a = self._prepare_attentional_mechanism_input(Wh, Wh, k = k)
                        zero_vec = -9e15 * torch.ones_like(A)
                        a = torch.where(A > 0, A*a, zero_vec)
                        a = F.softmax(a, dim=1)
                        H = torch.matmul(a, Wh)
                        if k+1 == self.k_hop:
                            Hs[b,:, :] = H

                return Hs, As, X, 1