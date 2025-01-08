import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch
import torch.optim as optim
import os
from torch.optim.lr_scheduler import StepLR
from collections import OrderedDict
from GDN import NodeEmbedding
from utils import *
from GAT.model import GAT
from GAT.layers import device

import numpy as np
from GLCN.GLCN import GLCN
from cfg import get_cfg

cfg = get_cfg()
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPONetwork(nn.Module):
    def __init__(self, state_size, state_action_size, layers=[8, 12]):
        super(PPONetwork, self).__init__()
        self.state_size = state_size
        self.NN_sequential = OrderedDict()
        layers = eval(layers)
        self.fc_pi = nn.Linear(state_action_size, layers[0])
        self.fc_v = nn.Linear(state_size, layers[0])
        self.fcn = OrderedDict()
        last_layer = layers[0]
        for i in range(1, len(layers)):
            layer = layers[i]
            if i <= len(layers)-2:
                self.fcn['linear{}'.format(i)] = nn.Linear(last_layer, layer)
                self.fcn['activation{}'.format(i)] = nn.ELU()
                last_layer = layer
            #else:
        self.forward_cal = nn.Sequential(self.fcn)
        self.output_pi = nn.Linear(last_layer, 1)
        self.output_v = nn.Linear(last_layer, 1)


    def forward(self, x, visualize = False):
        if visualize == False:
            x = self.fc_pi(x)
            x = F.elu(x)
            x = self.forward_cal(x)
            pi = self.output_pi(x)
            return pi
        else:
            x = self.fc_pi(x)
            x = F.elu(x)
            x = self.forward_cal(x)
            pi = self.output_pi(x)
            return x



class ValueNetwork(nn.Module):
    def __init__(self, state_size, state_action_size, layers=[8, 12]):
        super(ValueNetwork, self).__init__()
        self.state_size = state_size
        self.NN_sequential = OrderedDict()
        layers = eval(layers)
        self.fc_pi = nn.Linear(state_action_size, layers[0])
        self.fc_v = nn.Linear(state_size, layers[0])
        self.fcn = OrderedDict()
        last_layer = layers[0]
        for i in range(1, len(layers)):
            layer = layers[i]
            if i <= len(layers)-2:
                self.fcn['linear{}'.format(i)] = nn.Linear(last_layer, layer)
                self.fcn['activation{}'.format(i)] = nn.ELU()
                last_layer = layer
            #else:
        self.forward_cal = nn.Sequential(self.fcn)
        self.output_pi = nn.Linear(last_layer, 1)
        self.output_v = nn.Linear(last_layer, 1)

    def forward(self, x):
        x = self.fc_v(x)
        x = F.elu(x)
        x = self.forward_cal(x)
        v = self.output_v(x)
        return v


class Agent:
    def __init__(self,
                 params
                 ):

        # action_size,
        # feature_size,
        # hidden_size_obs,
        # hidden_size_action,
        # n_representation_obs,
        # n_representation_action,
        # graph_embedding,
        # learning_rate = cfg.lr,
        # gamma = cfg.gamma,
        # lmbda = cfg.lmbda,
        # eps_clip = cfg.eps_clip,
        # K_epoch = cfg.K_epoch,
        # layers = list(eval(cfg.ppo_layers))

        self.action_size = params["action_size"]
        self.feature_size = params["feature_size"]
        self.hidden_size_obs = params["hidden_size_obs"]
        self.hidden_size_comm = params["hidden_size_comm"]
        self.hidden_size_action = params["hidden_size_action"]
        self.n_representation_obs = params["n_representation_obs"]
        self.n_representation_comm = params["n_representation_comm"]
        self.n_representation_action = params["n_representation_action"]
        self.graph_embedding = params["graph_embedding"]
        self.graph_embedding_comm = params["graph_embedding_comm"]
        self.learning_rate = params["learning_rate"]
        self.gamma = params["gamma"]
        self.gamma1 = params["gamma1"]
        self.gamma2 = params["gamma2"]
        self.learning_rate_graph = params["learning_rate_graph"]
        self.n_data_parallelism = params["n_data_parallelism"]



        self.lmbda = params["lmbda"]
        self.eps_clip = params["eps_clip"]
        self.K_epoch = params["K_epoch"]
        self.layers = params["ppo_layers"]
        self.data = []





        """
        
        NodeEmbedding 수정해야 함
        
        """
        self.node_representation =   NodeEmbedding(feature_size=self.feature_size,
                                                   hidden_size=self.hidden_size_obs,
                                                   n_representation_obs=self.n_representation_obs).to(device)  # 수정사항

        self.node_representation_comm = NodeEmbedding(feature_size =self.feature_size-1,
                                                      hidden_size  =self.hidden_size_comm,
                                                      n_representation_obs=self.n_representation_comm).to(device)  # 수정사항

        self.action_representation = NodeEmbedding(feature_size=self.feature_size + 5,
                                                   hidden_size=self.hidden_size_action,
                                                   n_representation_obs=self.n_representation_action).to(device)  # 수정사항


        self.func_obs  = GLCN(feature_size=self.n_representation_obs,
                              graph_embedding_size=self.graph_embedding, link_prediction = False).to(device)
        if cfg.given_edge == True:
            self.func_glcn = GLCN(feature_size=self.graph_embedding,
                                graph_embedding_size=self.graph_embedding_comm, link_prediction = False).to(device)
            self.func_glcn2 = GLCN(feature_size=self.graph_embedding_comm,graph_embedding_size=self.graph_embedding_comm, link_prediction=False).to(device)
        else:
            self.func_glcn = GLCN(feature_size=self.graph_embedding,
                                  feature_obs_size=self.graph_embedding,
                                graph_embedding_size=self.graph_embedding_comm, link_prediction = True).to(device)

        self.network = PPONetwork(state_size=self.graph_embedding_comm,
                                  state_action_size=self.graph_embedding_comm + self.n_representation_action,
                                  layers=self.layers).to(device)
        self.valuenetwork = ValueNetwork(state_size=self.graph_embedding_comm,
                                  state_action_size=self.graph_embedding_comm + self.n_representation_action,
                                  layers=self.layers).to(device)

        if cfg.given_edge == True:
            self.eval_params = list(self.network.parameters()) + \
                               list(self.valuenetwork.parameters()) + \
                               list(self.node_representation.parameters()) + \
                               list(self.node_representation_comm.parameters()) + \
                               list(self.action_representation.parameters()) + \
                               list(self.func_obs.parameters()) + \
                               list(self.func_glcn.parameters()) + \
                               list(self.func_glcn2.parameters())
        else:
            self.eval_params = list(self.network.parameters()) + \
                               list(self.valuenetwork.parameters()) + \
                               list(self.node_representation.parameters()) + \
                               list(self.node_representation_comm.parameters()) + \
                               list(self.action_representation.parameters()) + \
                               list(self.func_obs.parameters()) + \
                               list(self.func_glcn.parameters())

        if cfg.optimizer == 'ADAM':
            self.optimizer1 = optim.Adam(self.eval_params, lr=self.learning_rate)  #
        if cfg.optimizer == 'ADAMW':
            self.optimizer = optim.AdamW(self.eval_params, lr=self.learning_rate)  #
        self.scheduler = StepLR(optimizer=self.optimizer1, step_size=cfg.scheduler_step, gamma=cfg.scheduler_ratio)

        self.node_features_list = list()
        self.edge_index_enemy_list = list()
        self.avail_action_list = list()
        self.action_list = list()
        self.prob_list = list()
        self.action_feature_list = list()
        self.reward_list = list()
        self.done_list = list()
        self.edge_index_comm_list = list()
        self.factorize_pi_list = list()
        self.dead_masking = list()
        self.state_list = list()
        self.batch_store = []


    def batch_reset(self):
        self.batch_store = []

    @torch.no_grad()
    def get_td_target(self, ship_features, node_features_missile, heterogenous_edges, possible_actions, action_feature, reward, done):
        obs_next, act_graph = self.get_node_representation(ship_features,node_features_missile, heterogenous_edges,mini_batch=False)
        td_target = reward + self.gamma * self.network.v(obs_next) * (1 - done)
        return td_target.tolist()[0][0]

    # num_agent = env1.get_env_info()["n_agents"],
    # num_enemy = env1.get_env_info()["n_enemies"],
    # feature_size = env1.get_env_info()["node_features"],
    # action_size = env1.get_env_info()["n_actions"],

    @torch.no_grad()
    def sample_action(self, node_representation, action_feature, avail_action, num_agent):
        """
        node_representation 차원 : n_agents X n_representation_comm
        action_feature 차원      : action_size X n_action_feature
        avail_action 차원        : n_agents X action_size
        """
        mask = torch.tensor(avail_action, device=device).bool()
        action_feature = torch.tensor(action_feature, device=device, dtype = torch.float64).float()
        action_size = action_feature.shape[0]
        action = []
        probs = []
        action_embedding = self.action_representation(action_feature)
        for n in range(num_agent):
            obs = node_representation[n].expand(action_size, node_representation[n].shape[0])
            obs_cat_action = torch.concat([obs, action_embedding], dim = 1)                           # shape :
            logit = self.network(obs_cat_action).squeeze(1)
            logit = logit.masked_fill(mask[n, :]==0, -1e8)
            prob = torch.softmax(logit, dim=-1)             # 에이전트 별 확률
            m = Categorical(prob)
            u = m.sample().item()
            action.append(u)
            probs.append(prob[u].item())
        factorized_probs = probs[:]
        probs = torch.exp(torch.sum(torch.log(torch.tensor(probs))))
        return action, probs, factorized_probs


    def get_node_representation_gpo(self, node_feature, edge_index_obs,edge_index_comm, n_agent, dead_masking, mini_batch = False):
        if mini_batch == False:
            with torch.no_grad():
                node_feature = torch.tensor(node_feature, dtype=torch.float,device=device)
                node_embedding_obs = self.node_representation(node_feature)
                #node_embedding_comm = self.node_representation_comm(node_feature[:,:-1])
                edge_index_obs = torch.tensor(edge_index_obs, dtype=torch.long, device=device)
                edge_index_comm = torch.tensor(edge_index_comm, dtype=torch.long, device=device)
                node_embedding_obs = self.func_obs(X = node_embedding_obs, A = edge_index_obs)
                #cat_embedding = torch.cat([node_embedding_obs, node_embedding_comm], dim = 1)
                if cfg.given_edge == True:
                    node_embedding = self.func_glcn(X=node_embedding_obs[:n_agent, :], A=edge_index_comm)
                    node_embedding = self.func_glcn2(X=node_embedding, A=edge_index_comm)
                    return node_embedding
                else:
                    node_embedding, A, X = self.func_glcn(dead_masking= dead_masking, X = node_embedding_obs[:n_agent, :], A = None)
                    return node_embedding, A, X
        else:
            node_feature = torch.tensor(node_feature, dtype=torch.float, device=device)
            node_embedding_obs = self.node_representation(node_feature)
            # node_embedding_comm = self.node_representation_comm(node_feature[:, :, :-1])
            node_embedding_obs = self.func_obs(X = node_embedding_obs, A = edge_index_obs, mini_batch = mini_batch)
            # cat_embedding = torch.cat([node_embedding_obs, node_embedding_comm], dim=2)
            if cfg.given_edge == True:
                node_embedding = self.func_glcn(X=node_embedding_obs[:, :n_agent, :], A=edge_index_comm, mini_batch=mini_batch)
                node_embedding = self.func_glcn2(X=node_embedding , A=edge_index_comm, mini_batch=mini_batch)
                return node_embedding
            else:
                node_embedding, A, X, D = self.func_glcn(dead_masking= dead_masking, X = node_embedding_obs[:, :n_agent, :], A = None, mini_batch = mini_batch)
                return node_embedding, A, X, D


    def put_data(self, transition):
        self.node_features_list.append(transition[0])
        self.edge_index_enemy_list.append(transition[1])
        self.avail_action_list.append(transition[2])
        self.action_list.append(transition[3])
        self.prob_list.append(transition[4])
        self.action_feature_list.append(transition[5])
        self.reward_list.append(transition[6])
        self.done_list.append(transition[7])
        self.edge_index_comm_list.append(transition[8])
        self.factorize_pi_list.append(transition[9])
        self.dead_masking.append(transition[10])
        self.state_list.append(transition[11])

        if transition[7] == True:
            batch_data = (
                self.node_features_list,
                self.edge_index_enemy_list,
                self.avail_action_list,
                self.action_list,
                self.prob_list,
                self.action_feature_list,
                self.reward_list,
                self.done_list,
                self.edge_index_comm_list,
                self.factorize_pi_list,
                self.dead_masking,
                self.state_list
                )

            self.batch_store.append(batch_data) # batch_store에 저장함
            self.node_features_list = list()
            self.edge_index_enemy_list = list()
            self.avail_action_list = list()
            self.action_list = list()
            self.prob_list = list()
            self.action_feature_list = list()
            self.reward_list = list()
            self.done_list = list()
            self.edge_index_comm_list = list()
            self.factorize_pi_list = list()
            self.dead_masking = list()
            self.state_list = list()


    def make_batch(self, batch_data):
        node_features_list = batch_data[0]
        edge_index_enemy_list = batch_data[1]
        avail_action_list = batch_data[2]
        action_list = batch_data[3]
        prob_list = batch_data[4]
        action_feature_list = batch_data[5]
        reward_list = batch_data[6]
        done_list = batch_data[7]
        edge_index_comm_list = batch_data[8]
        factorize_pi_list = batch_data[9]
        dead_masking = batch_data[10]
        state_list = batch_data[11]
        factorize_pi_list = torch.tensor(factorize_pi_list, dtype = torch.float).to(device)
        dead_masking = torch.tensor(dead_masking, dtype=torch.float).to(device)

        node_features_list = torch.tensor(node_features_list, dtype = torch.float).to(device)

        edge_index_enemy_list = edge_index_enemy_list
        avail_action_list = torch.tensor(avail_action_list, dtype=torch.float).to(device)
        action_list = torch.tensor(action_list, dtype=torch.float).to(device)
        state_list = torch.tensor(state_list, dtype=torch.float).to(device)


        return node_features_list, edge_index_enemy_list, avail_action_list,action_list,prob_list,action_feature_list,reward_list,done_list, edge_index_comm_list, factorize_pi_list,dead_masking,state_list





    def learn(self, cum_loss = 0):

        cum_surr = 0
        cum_value_loss = 0
        cum_lap_quad = 0
        cum_sec_eig_upperbound = 0


        for i in range(self.K_epoch):
            if i == 0:
                v_s_old_list = list()
                v_s_next_old_list = list()
            for l in range(len(self.batch_store)):
                batch_data = self.batch_store[l]
                node_features_list, \
                edge_index_enemy_list, \
                avail_action_list,\
                action_list,\
                prob_list,\
                action_feature_list,\
                reward_list,\
                done_list, \
                edge_index_comm_list,  factorize_pi_list, dead_masking, state_list = self.make_batch(batch_data)
                self.eval_check(eval=False)
                action_feature = torch.tensor(action_feature_list, dtype= torch.float).to(device)
                action_list = torch.tensor(action_list, dtype = torch.long).to(device)
                mask = torch.tensor(avail_action_list, dtype= torch.float).to(device)
                done = torch.tensor(done_list, dtype = torch.float).to(device)
                reward = torch.tensor(reward_list, dtype= torch.float).to(device)
                pi_old = torch.tensor(prob_list, dtype= torch.float).to(device)
                factorize_pi_old =torch.tensor(factorize_pi_list, dtype= torch.float).to(device)

                num_nodes = node_features_list.shape[1]
                num_agent = mask.shape[1]
                num_action = action_feature.shape[1]
                time_step = node_features_list.shape[0]
                if cfg.given_edge == True:
                    node_embedding = self.get_node_representation_gpo(node_features_list,
                                                                      edge_index_enemy_list,
                                                                      edge_index_comm_list,
                                                                      mini_batch=True,
                                                                      dead_masking=dead_masking,
                                                                      n_agent = num_agent
                                                                      )

                else:
                    node_embedding, A, X, _ = self.get_node_representation_gpo(
                                                                            node_features_list,
                                                                            edge_index_enemy_list,
                                                                            edge_index_comm_list,
                                                                            dead_masking = dead_masking,
                                                                            mini_batch=True,
                                                                            n_agent=num_agent
                                                                            )



                action_feature = action_feature.reshape(time_step*num_action, -1).to(device)
                action_embedding = self.action_representation(action_feature)
                action_embedding = action_embedding.reshape(time_step, num_action, -1)




                node_embedding = node_embedding[:, :num_agent, :]
                empty = torch.zeros(1, num_agent, node_embedding.shape[2]).to(device)
                node_embedding_next = torch.cat((node_embedding, empty), dim = 0)[:-1, :, :]





                v_s = self.valuenetwork(node_embedding.reshape(num_agent*time_step,-1))
                v_s = v_s.reshape(time_step, num_agent)
                v_next = self.valuenetwork(node_embedding_next.reshape(num_agent*time_step,-1))
                v_next = v_next.reshape(time_step, num_agent)
                if i == 0:e)
                advantage_lst.reverse()
                    v_s_old_list.append(v_s)
                    v_s_next_old_list.append(v_next)

                done =  done.unsqueeze(1).repeat(1, num_agent)

                reward =  reward.unsqueeze(1).repeat(1, num_agent)
                td_target = reward + self.gamma * v_next * (1-done)
                delta = td_target - v_s
                delta = delta.cpu().detach().numpy()
                advantage_lst = []
                advantage = torch.zeros(num_agent)
                for delta_t in delta[:, :]:
                    advantage = self.gamma * self.lmbda * advantage + delta_t
                    advantage_lst.append(advantag
                advantage = torch.stack(advantage_lst).to(device)

                for n in range(num_agent):
                    obs = node_embedding[:, n, :].unsqueeze(1).expand(time_step,  num_action, node_embedding.shape[2])
                    obs_cat_action = torch.concat([obs, action_embedding], dim=2)
                    obs_cat_action = obs_cat_action.reshape(time_step*num_action, -1)
                    logit = self.network(obs_cat_action).squeeze(1)
                    logit = logit.reshape(time_step, num_action, -1)
                    logit = logit.squeeze(-1).masked_fill(mask[:, n, :] == 0, -1e8)
                    prob = torch.softmax(logit, dim=-1)
                    actions = action_list[:, n].unsqueeze(1)
                    pi = prob.gather(1, actions)
                    pi_old = factorize_pi_old[:,n].unsqueeze(1)
                    advantage_i = advantage[:, n].unsqueeze(1)
                    ratio = torch.exp(torch.log(pi) - torch.log(pi_old).detach())  # a/b == exp(log(a)-log(b))
                    surr1 = ratio * (advantage_i.detach().squeeze())
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * (advantage_i.detach().squeeze())
                    entropy = -(pi * torch.log(pi)).sum(1).mean()



                    if n == 0:
                        surr = torch.min(surr1, surr2).mean()/num_agent#+0.01*entropy.mean()/num_agent

                    else:
                        surr+= torch.min(surr1, surr2).mean()/num_agent#+0.01*entropy.mean()/num_agent

                val_surr1 = v_s
                val_surr2 = torch.clamp(v_s, v_s_old_list[l].detach() - self.eps_clip,v_s_old_list[l].detach() + self.eps_clip)
                value_loss = torch.max(F.smooth_l1_loss(val_surr1, td_target.detach()), F.smooth_l1_loss(val_surr2, td_target.detach())).mean()

                if cfg.given_edge == True:
                    loss = -surr + 0.5 * value_loss
                else:
                    gamma1 = self.gamma1
                    gamma2 = self.gamma2
                    lap_quad, sec_eig_upperbound, L = get_graph_loss(X, A, num_nodes)
                    if cfg.softmax == True:
                        loss = -surr + 0.5 * value_loss + gamma1 * lap_quad + gamma2 * gamma1 * frobenius_norm.mean()
                    else:
                        loss = -surr + 0.5 * value_loss+  gamma1* lap_quad - gamma2 * gamma1 * sec_eig_upperbound


            #print(np.array([np.linalg.eigh(L[t, :, :].cpu().detach().numpy())[0][1] for t in range(time_step)]))
                if l == 0:
                    cum_loss = loss / self.n_data_parallelism
                else:
                    cum_loss = cum_loss + loss / self.n_data_parallelism

                if l == 0:
                    cum_surr               += surr.tolist() / (self.n_data_parallelism * self.K_epoch)
                    cum_value_loss         += value_loss.tolist() / (self.n_data_parallelism * self.K_epoch)
                    if cfg.given_edge == True:
                        cum_lap_quad =0
                        cum_sec_eig_upperbound =0
                    else:
                        cum_lap_quad           += lap_quad.tolist() / (self.n_data_parallelism * self.K_epoch)
                        cum_sec_eig_upperbound += sec_eig_upperbound.tolist()  / (self.n_data_parallelism * self.K_epoch)
                else:
                    cum_surr       += surr.tolist() / (self.n_data_parallelism * self.K_epoch)
                    cum_value_loss += value_loss.tolist() / (self.n_data_parallelism * self.K_epoch)
                    if cfg.given_edge == True:pass
                    else:
                        cum_lap_quad += lap_quad.tolist() / (self.n_data_parallelism * self.K_epoch)
                        cum_sec_eig_upperbound += sec_eig_upperbound.tolist() / (self.n_data_parallelism * self.K_epoch)

            grad_clip = float(os.environ.get("grad_clip", 10))
            cum_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.eval_params, grad_clip)
            self.optimizer1.step()
            self.optimizer1.zero_grad()
            # if l == 0:
            #     print(second_eigenvalue)


        self.batch_store = list()
        if cfg.given_edge== True:
            return cum_surr, cum_value_loss, 0, 0, 0  # second_eigenvalue
        else:
            return cum_surr, cum_value_loss, cum_lap_quad, cum_sec_eig_upperbound, _  # second_eigenvalue

    # def load_network(self, file_dir):
    #     print(file_dir)
    #     checkpoint = torch.load(file_dir)
    #     self.network.load_state_dict(checkpoint["network"])
    #     self.node_representation_ship_feature.load_state_dict(checkpoint["node_representation_ship_feature"])
    #     self.func_meta_path.load_state_dict(checkpoint["func_meta_path"])
    #     self.func_meta_path2.load_state_dict(checkpoint["func_meta_path2"])
    #     self.func_meta_path3.load_state_dict(checkpoint["func_meta_path3"])
    #     self.func_meta_path4.load_state_dict(checkpoint["func_meta_path4"])
    #     self.func_meta_path5.load_state_dict(checkpoint["func_meta_path5"])
    #     self.func_meta_path6.load_state_dict(checkpoint["func_meta_path6"])
    #     try:
    #         self.node_representation_wo_graph.load_state_dict(checkpoint["node_representation_wo_graph"])
    #     except KeyError:pass

    def eval_check(self, eval):
        if eval == True:
            self.network.eval()
            self.node_representation.eval()
            self.node_representation_comm.eval()
            self.action_representation.eval()
            self.func_obs.eval()
            self.func_glcn.eval()

        else:
            self.network.train()
            self.node_representation.train()
            self.node_representation_comm.train()
            self.action_representation.train()
            self.func_obs.train()
            self.func_glcn.train()
