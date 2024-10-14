import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import softmax
from torch_scatter import scatter
class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=None, activation=None,
                 self_loop=False, skip_connect=False, dropout=0.0, layer_norm=False):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.skip_connect = skip_connect
        self.layer_norm = layer_norm

        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            # self.loop_weight = nn.Parameter(torch.eye(out_feat), requires_grad=False)

        if self.skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))   # 和self-loop不一样，是跨层的计算
            nn.init.xavier_uniform_(self.skip_connect_weight,
                                    gain=nn.init.calculate_gain('relu'))

            self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.skip_connect_bias)  # 初始化设置为0

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if self.layer_norm:
            self.normalization_layer = nn.LayerNorm(out_feat, elementwise_affine=False)

    # define how propagation is done in subclass
    def propagate(self, g):
        raise NotImplementedError

    def forward(self, g, prev_h=[]):
        if self.self_loop:
            #print(self.loop_weight)
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)
        # self.skip_connect_weight.register_hook(lambda g: print("grad of skip connect weight: {}".format(g)))
        if len(prev_h) != 0 and self.skip_connect:
            skip_weight = F.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)     # 使用sigmoid，让值在0~1
            # print("skip_ weight")
            # print(skip_weight)
            # print("skip connect weight")
            # print(self.skip_connect_weight)
            # print(torch.mm(prev_h, self.skip_connect_weight))

        self.propagate(g)  # 这里是在计算从周围节点传来的信息

        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias
        # print(len(prev_h))
        if len(prev_h) != 0 and self.skip_connect:   # 两次计算loop_message的方式不一样，前者激活后再加权
            previous_node_repr = (1 - skip_weight) * prev_h
            if self.activation:
                node_repr = self.activation(node_repr)
            if self.self_loop:
                if self.activation:
                    loop_message = skip_weight * self.activation(loop_message)
                else:
                    loop_message = skip_weight * loop_message
                node_repr = node_repr + loop_message
            node_repr = node_repr + previous_node_repr
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message
            if self.layer_norm:
                node_repr = self.normalization_layer(node_repr)
            if self.activation:
                node_repr = self.activation(node_repr)
            # print("node_repr")
            # print(node_repr)
        g.ndata['h'] = node_repr
        return node_repr


class RGCNBasisLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(RGCNBasisLayer, self).__init__(in_feat, out_feat, bias, activation)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.is_input_layer = is_input_layer
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # add basis weights
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                self.out_feat))
        if self.num_bases < self.num_rels:
            # linear combination coefficients
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels,
                                                    self.num_bases))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))

    def propagate(self, g):
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.view(self.num_bases,
                                      self.in_feat * self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(
                self.num_rels, self.in_feat, self.out_feat)
        else:
            weight = self.weight

        if self.is_input_layer:
            def msg_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                embed = weight.view(-1, self.out_feat)
                index = edges.data['type'] * self.in_feat + edges.src['id']
                return {'msg': embed.index_select(0, index)}
        else:
            def msg_func(edges):
                w = weight.index_select(0, edges.data['type'])
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                return {'msg': msg}

        def apply_func(nodes):
            return {'h': nodes.data['h'] * nodes.data['norm']}

        g.update_all(msg_func, fn.sum(msg='msg', out='h'), apply_func)


class RGCNBlockLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=None,
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False, layer_norm=False):
        super(RGCNBlockLayer, self).__init__(in_feat, out_feat, bias,
                                             activation, self_loop=self_loop, skip_connect=skip_connect,
                                             dropout=dropout)
        self.num_rels = num_rels
        self.num_bases = num_bases

        assert self.num_bases > 0

        self.out_feat = out_feat
        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases

        # assuming in_feat and out_feat are both divisible by num_bases
        self.weight = nn.Parameter(torch.Tensor(
            self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def msg_func(self, edges):
        weight = self.weight.index_select(0, edges.data['type']).view(
                    -1, self.submat_in, self.submat_out)    # [edge_num, submat_in, submat_out]
        node = edges.src['h'].view(-1, 1, self.submat_in)   # [edge_num * num_bases, 1, submat_in]->
        msg = torch.bmm(node, weight).view(-1, self.out_feat)   # [edge_num, out_feat]
        return {'msg': msg}

    def propagate(self, g):
        g.update_all(self.msg_func, fn.sum(msg='msg', out='h'), self.apply_func)
        # g.updata_all ({'msg': msg} , fn.sum(msg='msg', out='h'), {'h': nodes.data['h'] * nodes.data[''norm]})

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}


class UnionRGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1,  bias=None,
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False, rel_emb=None):
        super(UnionRGCNLayer, self).__init__()

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.num_rels = num_rels
        self.rel_emb = None
        self.skip_connect = skip_connect
        self.ob = None
        self.sub = None

        # WL
        self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

        if self.skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))   # 和self-loop不一样，是跨层的计算
            nn.init.xavier_uniform_(self.skip_connect_weight,gain=nn.init.calculate_gain('relu'))
            self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.skip_connect_bias)  # 初始化设置为0

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def propagate(self, g):
        g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h'), self.apply_func)

    def forward(self, g, prev_h, emb_rel):
        self.rel_emb = emb_rel
        # self.sub = sub
        # self.ob = ob
        if self.self_loop:
            #loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            # masked_index = torch.masked_select(torch.arange(0, g.number_of_nodes(), dtype=torch.long), (g.in_degrees(range(g.number_of_nodes())) > 0))
            masked_index = torch.masked_select(
                torch.arange(0, g.number_of_nodes(), dtype=torch.long).cuda(),
                (g.in_degrees(range(g.number_of_nodes())) > 0))
            loop_message = torch.mm(g.ndata['h'], self.evolve_loop_weight)
            loop_message[masked_index, :] = torch.mm(g.ndata['h'], self.loop_weight)[masked_index, :]
        if len(prev_h) != 0 and self.skip_connect:
            skip_weight = F.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)     # 使用sigmoid，让值在0~1

        # calculate the neighbor message with weight_neighbor
        self.propagate(g)
        node_repr = g.ndata['h']

        # print(len(prev_h))
        if len(prev_h) != 0 and self.skip_connect:  # 两次计算loop_message的方式不一样，前者激活后再加权
            if self.self_loop:
                node_repr = node_repr + loop_message
            node_repr = skip_weight * node_repr + (1 - skip_weight) * prev_h
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message

        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout is not None:
            node_repr = self.dropout(node_repr)
        g.ndata['h'] = node_repr
        return node_repr

    def msg_func(self, edges):
        # if reverse:
        #     relation = self.rel_emb.index_select(0, edges.data['type_o']).view(-1, self.out_feat)
        # else:
        #     relation = self.rel_emb.index_select(0, edges.data['type_s']).view(-1, self.out_feat)
        relation = self.rel_emb.index_select(0, edges.data['type']).view(-1, self.out_feat)
        edge_type = edges.data['type']
        edge_num = edge_type.shape[0]
        node = edges.src['h'].view(-1, self.out_feat)
        # node = torch.cat([torch.matmul(node[:edge_num // 2, :], self.sub),
        #                  torch.matmul(node[edge_num // 2:, :], self.ob)])
        # node = torch.matmul(node, self.sub)

        # after add inverse edges, we only use message pass when h as tail entity
        # 这里计算的是每个节点发出的消息，节点发出消息时其作为头实体
        # msg = torch.cat((node, relation), dim=1)
        msg = node + relation
        # calculate the neighbor message with weight_neighbor
        msg = torch.mm(msg, self.weight_neighbor)
        return {'msg': msg}

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}
class AttentionRGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1,  bias=None,
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False, rel_emb=None):
        super(AttentionRGCNLayer, self).__init__()

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.num_rels = num_rels
        self.rel_emb = None
        self.skip_connect = skip_connect
        self.ob = None
        self.sub = None

        # WL
        self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))

        # Attention parameters
        self.a = nn.Parameter(torch.Tensor(3*in_feat,1))
        nn.init.xavier_uniform_(self.a, gain=nn.init.calculate_gain('relu'))
        self.W3 = nn.Parameter(torch.Tensor(3*in_feat,3*in_feat))
        nn.init.xavier_uniform_(self.W3, gain=nn.init.calculate_gain('relu'))

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

        if self.skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))   # 和self-loop不一样，是跨层的计算
            nn.init.xavier_uniform_(self.skip_connect_weight,gain=nn.init.calculate_gain('relu'))
            self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.skip_connect_bias)  # 初始化设置为0

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def propagate(self, g):
        g.apply_edges(self.compute_alpha)
        g.update_all(lambda x: self.msg_func(x), fn.sum('msg', 'h'))
        g.apply_nodes(self.apply_func)

    def forward(self, g, prev_h, emb_rel):
        self.rel_emb = emb_rel
        # self.sub = sub
        # self.ob = ob
        if self.self_loop:
            #loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            # masked_index = torch.masked_select(torch.arange(0, g.number_of_nodes(), dtype=torch.long), (g.in_degrees(range(g.number_of_nodes())) > 0))
            masked_index = torch.masked_select(
                torch.arange(0, g.number_of_nodes(), dtype=torch.long).cuda(),
                (g.in_degrees(range(g.number_of_nodes())) > 0))
            loop_message = torch.mm(g.ndata['h'], self.evolve_loop_weight)
            loop_message[masked_index, :] = torch.mm(g.ndata['h'], self.loop_weight)[masked_index, :]
        if len(prev_h) != 0 and self.skip_connect:
            skip_weight = F.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)     # 使用sigmoid，让值在0~1

        # calculate the neighbor message with weight_neighbor
        self.propagate(g)
        node_repr = g.ndata['h']

        # print(len(prev_h))
        if len(prev_h) != 0 and self.skip_connect:  # 两次计算loop_message的方式不一样，前者激活后再加权
            if self.self_loop:
                node_repr = node_repr + loop_message
            node_repr = skip_weight * node_repr + (1 - skip_weight) * prev_h
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message

        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout is not None:
            node_repr = self.dropout(node_repr)
        g.ndata['h'] = node_repr
        return node_repr

    def compute_alpha(self, edges):
        zi = edges.src['h']
        zj = edges.dst['h']
        zr = self.rel_emb.index_select(0, edges.data['type']).view(-1, self.out_feat)
        emb_cat = torch.cat([zi, zj, zr], dim=-1)
        W = self.W3 @ self.a
        alpha_ij = torch.exp(F.leaky_relu(torch.matmul(emb_cat,W)))
        alpha_ij = alpha_ij / torch.sum(alpha_ij, dim=1, keepdim=True)
        return {'alpha': alpha_ij}

    def msg_func(self, edges):
        relation = self.rel_emb.index_select(0, edges.data['type']).view(-1, self.out_feat)
        node = edges.src['h'].view(-1, self.out_feat)
        alpha = edges.data['alpha']
        msg = node + relation
        msg = torch.mm(msg, self.weight_neighbor)
        msg = msg * alpha
        return {'msg': msg}

    def apply_func(self, nodes):
        return {'h': F.relu(nodes.data['h'])}


from torch_geometric.utils import softmax

class StructuralAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads, attn_drop, ffd_drop, residual):
        super(StructuralAttentionLayer, self).__init__()
        self.out_dim = output_dim // n_heads  # 每个头的输出维度
        self.n_heads = n_heads
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.relu = nn.ReLU()

        # 定义线性变换层
        self.lin_s = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)  # 源节点
        self.lin_r = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)  # 关系类型
        self.lin_o = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)  # 目标节点
        self.lin_w = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)  # 残差连接

        # 注意力参数 a^l
        self.w_1 = nn.Linear(2 * self.out_dim + self.out_dim, self.out_dim, bias=True)  # 对应KBGAT_conv的注意力机制
        self.w_2 = nn.Linear(self.out_dim, 1, bias=False)  # 计算注意力权重

        # Dropout 层
        self.attn_drop = nn.Dropout(attn_drop)
        self.ffd_drop = nn.Dropout(ffd_drop)

        self.residual = residual

        # 初始化参数
        self.xavier_init()

    def forward(self, graph, h, h_0, current_node_list):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 确保设备是 GPU

        # 获取源节点和目标节点索引，并迁移到 GPU
        edge_index = graph.edges()
        edge_index = (edge_index[0].to(device), edge_index[1].to(device))  # 分别迁移 source 和 target 到 GPU

        # 迁移 r_type、h 和 h_0 到 GPU
        r_type = graph.edata['type'].to(device)
        
        random_vector_h = torch.randn_like(h)
        random_vector_h_0 = torch.randn_like(h_0)
        h = random_vector_h.to(device)
        h_0 = random_vector_h_0.to(device)

        H, C = self.n_heads, self.out_dim

        # 获取节点和关系的特征
        z_s = h[edge_index[0]]  # 源节点特征
        z_o = h[edge_index[1]]  # 目标节点特征

        # 处理关系类型，确保索引不越界
        max_rel_type = h_0.size(0)
        r_type = torch.clamp(r_type, max=max_rel_type - 1)
        z_r = h_0[r_type]  # 从关系嵌入中选择的关系特征

        # 线性变换，映射到多头空间
        Wz_s = self.lin_s(z_s).view(-1, H, C)
        Wz_o = self.lin_o(z_o).view(-1, H, C)
        Wz_r = self.lin_r(z_r).view(-1, H, C)

        # **注意力机制 (KBGAT_conv)**
        # 拼接源节点、目标节点和关系的特征
        triple_emb = torch.cat([Wz_s, Wz_o, Wz_r], dim=-1)
        c = self.w_1(triple_emb)  # 线性变换得到注意力输入
        b = self.w_2(self.leaky_relu(c))  # 计算注意力权重

        # softmax归一化得到alpha
        # alpha = F.softmax(b, dim=0)
        alpha = softmax(b, edge_index[0])


        # 应用 Dropout
        alpha = self.attn_drop(alpha)
        Wz_s = self.ffd_drop(Wz_s)
        Wz_r = self.ffd_drop(Wz_r)

        alpha = alpha.view(-1, H, 1)

        # 消息传递：加权相乘并聚合消息
        msg = (Wz_s + Wz_r) * alpha

        # 获取唯一的目标节点及其索引映射
        unique_targets, inverse_indices = torch.unique(edge_index[1], return_inverse=True)
        indices = inverse_indices.to(device)

        # 消息聚合
        out = scatter(msg, indices, dim=0, dim_size=unique_targets.size(0), reduce="sum")

        # 计算残差项
        z_o_prev = h[unique_targets]  # 残差连接的源节点特征
        res = self.lin_w(z_o_prev).view(-1, H, C)

        # 残差相加并激活
        out = self.relu(out + res)

        # 调整输出形状
        out = out.view(-1, H * C)

        # 返回 `out` 和 `unique_targets`
        return out, unique_targets



    def xavier_init(self):
        # Xavier 初始化
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)
        nn.init.xavier_uniform_(self.lin_s.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.lin_o.weight)
        nn.init.xavier_uniform_(self.lin_w.weight)

