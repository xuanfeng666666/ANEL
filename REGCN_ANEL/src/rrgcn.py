import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
import dgl.nn as dglnn

# from rgcn.layers import RGCNBlockLayer as RGCNLayer
from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer, AttentionRGCNLayer, StructuralAttentionLayer
from src.model import BaseRGCN
from src.decoder import ConvTransE, ConvTransR
import dgl.function as fn




class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            h = g.ndata['h']
            h = self.linear(h)
            return h



class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        elif self.encoder_name == "attentiongcn":
            return AttentionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn" or self.encoder_name == "attentiongcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')


class RecurrentRGCN(nn.Module):
    def __init__(self,basic_model_weight, k, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, h_dim, opn,
                 sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1, discount=0, angle=0, use_static=False,
                 entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu=0, analysis=False):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.gpu = gpu

        self.lin_residual = nn.Linear(h_dim, h_dim)
        ##################
        self.w0 = torch.nn.Parameter(torch.Tensor(self.h_dim*2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w0)

        self.w1 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w1)

        self.w2 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w2)

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)

        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels * 2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False,
                                                    skip_connect=False)
            self.static_loss = torch.nn.MSELoss()

        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()

        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)

        self.AttentionLayer = StructuralAttentionLayer(h_dim,
                                                       h_dim,
                                                       4,
                                                       0.2,
                                                       0.2,
                                                       True)
        self.rgcnAttention = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             "attentiongcn",
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)

        
        self.gcn = GCNLayer(in_feats=self.h_dim, out_feats=self.h_dim)

        # GRU cell for relation evolving
        self.relation_cell_1 = nn.GRUCell(self.h_dim * 2, self.h_dim)
        self.entity_cell_1 = nn.GRUCell(self.h_dim, self.h_dim)
        self.entity_cell_2 = nn.GRUCell(self.h_dim, self.h_dim)
        # decoder
        if decoder_name == "convtranse":
            self.decoder_ob = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError
        ##############
        self.k = k
        self.theta_d = nn.parameter.Parameter(torch.randn(num_ents,1))
        self.weight_t1 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t1 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.weight_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.para = nn.parameter.Parameter(torch.randn(1,))
        self.lenda = basic_model_weight

    def forward(self, fore_neighbor_list, t_idx, global_rep, neighbor_list, g_list, static_graph, use_cuda):
        gate_list = []
        degree_list = []

       
        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        else:
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None

        history_embs = []

        for i, g in enumerate(g_list):
            g = g.to(self.gpu)
            g.r_to_e = g.r_to_e.long() 
            temp_e = self.h[g.r_to_e]
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(
                self.num_rels * 2, self.h_dim).float()

            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1], :]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean

            if i == 0:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.emb_rel)  
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            else:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.h_0)  
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0

            current_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0])
            current_h = F.normalize(current_h) if self.layer_norm else current_h

           
            if self.k == 0:
                
                num_nodes = g.number_of_nodes()
                init_features = torch.randn(num_nodes, self.h_dim, device=self.gpu)  
                temp_g = g  
                current_node_list = list(range(num_nodes))
                new_current_h = init_features  
            else:
                
                if i == 0:
                    temp_g, current_node_list = self.densify_graph(g, neighbor_list[i], fore_neighbor_list, current_h)
                else:
                    temp_g, current_node_list = self.densify_graph(g, neighbor_list[i], [neighbor_list[i - 1]], current_h)
                out, unique_targets = self.AttentionLayer(temp_g, current_h, self.h_0, current_node_list)
                new_current_h = self.attention_layer(current_h, out, unique_targets)

            num_neighbors = neighbor_list[i].unsqueeze(1).float().to(self.gpu)
            h1 = self.entity_cell_1(current_h, self.h)
            h1 = F.normalize(h1) if self.layer_norm else h1
            h2 = self.entity_cell_2(new_current_h, self.h)
            h2 = F.normalize(h2) if self.layer_norm else h2
            # self.h = self.lenda * h1 + (1 - self.lenda) / (1 + torch.exp(num_neighbors)) * h2
            beta = 1 / (1 + torch.exp(num_neighbors))
            self.h = h1 + beta * h2
            #self.h = h1 + h2

            self.h = F.normalize(self.h) if self.layer_norm else self.h
            history_embs.append(self.h)

        return history_embs, static_emb, self.h_0, gate_list, degree_list, F.normalize(global_rep, p=2, dim=0).clone().detach()



    def predict(self,fore_neighbor_list, global_rep,t_idx, neighbor_list, test_graph, num_rels, static_graph, test_triplets, use_cuda):
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels  
            all_triples = torch.cat((test_triplets, inverse_test_triplets))

            evolve_embs, _, r_emb, _, _ ,global_rep= self.forward(fore_neighbor_list, t_idx, global_rep,neighbor_list, test_graph, static_graph,
                                                       use_cuda)
            embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

            score = self.decoder_ob.forward(embedding, r_emb, all_triples, mode="test")
            score_rel = self.rdecoder.forward(embedding, r_emb, all_triples, mode="test")
            return all_triples, score, score_rel, global_rep

    def get_loss(self, fore_neighbor_list, t_idx, global_rep, neighbor_list, glist, triples, static_graph, use_cuda):
        """
        :param glist:
        :param triplets:
        :param static_graph:
        :param use_cuda:
        :return:
        """
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_static = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        inverse_triples = triples[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.gpu)
        evolve_embs, static_emb, r_emb, _, _,global_rep = self.forward(fore_neighbor_list, t_idx, global_rep, neighbor_list, glist, static_graph,
                                                            use_cuda)

        pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]


        if self.entity_prediction:
            scores_ob = self.decoder_ob.forward(pre_emb, r_emb, all_triples).view(-1, self.num_ents)
            loss_ent += self.loss_e(scores_ob, all_triples[:, 2])

        if self.relation_prediction:
            score_rel = self.rdecoder.forward(pre_emb, r_emb, all_triples, mode="train").view(-1, 2 * self.num_rels)
            loss_rel += self.loss_r(score_rel, all_triples[:, 1])

        if self.use_static:
            if self.discount == 1:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
            elif self.discount == 0:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))

        return loss_ent, loss_rel, loss_static, global_rep


    # def cal_similarity_graph(self, node_embeddings):
    #     similarity_graph = torch.mm(node_embeddings[:, :int(self.h_dim/2)], node_embeddings[:, :int(self.h_dim/2)].t())
    #     similarity_graph += torch.mm(node_embeddings[:, int(self.h_dim/2):], node_embeddings[:, int(self.h_dim/2):].t())
    #     return similarity_graph

    def cal_similarity_graph(self, node_embeddings, mask=None):
        similarity_graph = torch.mm(node_embeddings[:, :int(self.h_dim/2)], node_embeddings[:, :int(self.h_dim/2)].t())
        similarity_graph += torch.mm(node_embeddings[:, int(self.h_dim/2):], node_embeddings[:, int(self.h_dim/2):].t())
        
        if mask is not None:
            similarity_graph = similarity_graph * mask.float()
        
        return similarity_graph


    def _sparse_graph(self, new_graph, K):
        sparse_graph = torch.zeros_like(new_graph)  
        node_compensate_list = []
        for i in range(new_graph.size(0)):
            row = new_graph[i].clone() 
            row[i] = -1  
            node_count = int(torch.sum(row > 0))

            if(node_count >= K):

                top_k_indices = row.topk(K, dim=-1).indices

                sparse_graph[i, top_k_indices] = new_graph[i, top_k_indices]

                node_compensate = 0
                node_compensate_list.append(node_compensate)

            else:
                sparse_graph[i, :] = new_graph[i, :]
                node_compensate = K - node_count
                node_compensate_list.append(node_compensate)

        return sparse_graph, node_compensate_list

    def sparse_graph_from_pre(self, similarity_adj, node_compensate_num):
   
        sparse_graph = torch.zeros_like(similarity_adj)
        for i in range(similarity_adj.size(0)):
            row = similarity_adj[i].clone()  

            top_k_indices = row.topk(node_compensate_num[i], dim=-1).indices 
         
            sparse_graph[i, top_k_indices] = similarity_adj[i, top_k_indices]

        return sparse_graph

    def densify_graph(self, g, neighbor_list, pre_neighbor_list, current_h):
        if len(pre_neighbor_list) != 0:
            not_first = True
        else:
            not_first = False

        current_node_t = torch.nonzero(neighbor_list.float() > 0.5).squeeze()
        current_node_list = current_node_t.tolist()
        current_node_embed = current_h[current_node_list, :]

        adj_matrix = g.adjacency_matrix().to_dense().to(self.gpu)
        adj_matrix = adj_matrix[current_node_list, :][:, current_node_list]

        mask = adj_matrix == 0
        # print(mask)

        Adj_new = self.cal_similarity_graph(current_node_embed, mask=mask)

        Adj_new, node_compensate_list = self._sparse_graph(Adj_new, self.k)

        src_new = []
        dst_new = []
        if sum(node_compensate_list) != 0 and not_first:
            node_compensate_index = [i for i,count in enumerate(node_compensate_list) if count > 0]
            node_compensate_num = [count for i,count in enumerate(node_compensate_list) if count > 0]
            node_compensate_embed = current_node_embed[node_compensate_index, :]

            pre_node_t = torch.nonzero(pre_neighbor_list[0].float() > 0.5).squeeze()
            pre_node_list = pre_node_t.tolist()

            pre_node_list_new = [idx for idx in pre_node_list if idx not in current_node_list]

            pre_node_embed = current_h[pre_node_list_new, :]

            combined_embed = torch.cat((node_compensate_embed, pre_node_embed), dim=0)

            compensate_node_len = node_compensate_embed.size(0)

         
            similarity_adj = self.cal_similarity_graph(combined_embed)

            similarity_adj = similarity_adj[:compensate_node_len,compensate_node_len:]

           
            similarity_adj = self.sparse_graph_from_pre(similarity_adj, node_compensate_num)

            
            edge_indices_new = torch.nonzero(similarity_adj).t()
            src_new, dst_new = edge_indices_new[0].tolist(), edge_indices_new[1].tolist()
            src_new = [current_node_list[node_compensate_index[src_new[i]]] for i in range(len(src_new))]
            dst_new = [pre_node_list_new[dst_new[i]] for i in range(len(dst_new))]

       
        edge_indices = torch.nonzero(Adj_new).t()
        src, dst = edge_indices[0].tolist(), edge_indices[1].tolist()
        src = [current_node_list[src[i]] for i in range(len(src))]
        dst = [current_node_list[dst[i]] for i in range(len(dst))]

        
        src.extend(src_new)
        dst.extend(dst_new)

        
        num_nodes = max(max(src), max(dst)) + 1
        
      
        temp_g = dgl.graph((src, dst), num_nodes=num_nodes)
        temp_g = temp_g.to(self.gpu)

       
        edge_types = torch.zeros(len(src), dtype=torch.int64, device=self.gpu)
        temp_g.edata['type'] = edge_types

       
        all_node_list = sorted(list(set(src + dst)))
        all_node_embed = current_h[all_node_list]

        
        temp_g.ndata['h'] = torch.zeros(num_nodes, current_node_embed.size(1), device=self.gpu)
        temp_g.ndata['h'][all_node_list] = all_node_embed

        return temp_g, current_node_list




    def attention_layer(self, current_h, out, unique_targets):
        new_current_h = torch.zeros_like(current_h)
        mask = torch.ones_like(current_h)
        mask[unique_targets] = 0
        new_current_h += mask * current_h
        new_current_h[unique_targets] += out
        new_current_h = F.normalize(new_current_h) if self.layer_norm else new_current_h
        return new_current_h
