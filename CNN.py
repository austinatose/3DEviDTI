import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from compound_encode import AtomEmbedding, BondEmbedding, BondFloatRBF, BondAngleFloatRBF, DistanceFloatRBF
from torch_geometric.nn import TransformerConv, global_max_pool as gmp,global_mean_pool, GATConv
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class LightAttention(nn.Module):
    def __init__(self, cfg, output_dim=2, kernel_size=9, n_layers=6):

        super(LightAttention, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.atom_names = ["atomic_num", "chiral_tag", "degree",
                       "explicit_valence", "formal_charge", "hybridization",
                       "implicit_valence"]
        self.bond_names = ["bond_dir", "bond_type", "is_in_ring"]
        self.bond_float_names = ['bond_length']
        self.bond_angle_float_names = ['bond_angle']
        self.distance = ['distance']
        self.n_layers = n_layers
        self.p_input_dim = cfg.PROTEIN_LA['INPUT_DIM']
        self.p_kernel_size = cfg.PROTEIN_LA['KERNEL_SIZE']
        self.stride = 1

        self.d_kernel_size = cfg.DRUG['KERNEL_SIZE']
        self.d_2d_dnn_input_dim = cfg.DRUG['D2_DNN_INPUT_DIM']
        self.d_2d_dnn_hidden_dim = cfg.DRUG['D2_DNN_HIDDEN_DIM']
        self.d_2d_dnn_out_dim = cfg.DRUG['D2_DNN_OUT_DIM']
        self.d_3d_input_dim = cfg.DRUG['D3_INPUT_DIM']
        self.d_3d_dnn_input_dim = cfg.DRUG['D3_DNN_INPUT_DIM']
        self.d_3d_dnn_hidden_dim = cfg.DRUG['D3_DNN_HIDDEN_DIM']
        self.d_3d_dnn_out_dim = cfg.DRUG['D3_DNN_OUT_DIM']
        self.d_dropout = cfg.DRUG['DROPOUT']
        self.d_dnn_input_dim = cfg.DRUG['D_DNN_INPUT_DIM']
        self.d_dnn_out_dim = cfg.DRUG['D_DNN_OUT_DIM']
        self.final_in_dim = cfg.DECODER['IN_DIM']
        self.final_hidden_dim = cfg.DECODER['HIDDEN_DIM']
        self.final_out_dim = cfg.DECODER['OUT_DIM']

        self.embed_dim = 64
        self.output_dim = 128
        self.dropout_rate = 0.2
        # self.layer_num = 8
        self.readout = "mean"
        self.norm = nn.LayerNorm(self.embed_dim)
        self.feature_convolution = nn.Conv1d(self.p_input_dim, self.p_input_dim,
                                             self.p_kernel_size, stride=self.stride,
                                             padding=self.p_kernel_size // 2)
        self.attention_convolution = nn.Conv1d(self.p_input_dim, self.p_input_dim,
                                               self.p_kernel_size, stride=self.stride,
                                               padding=self.p_kernel_size // 2)
        self.d_conv1 = nn.Sequential(nn.Conv1d(in_channels=1,out_channels=8,
                                               kernel_size=self.d_kernel_size,
                                               stride=self.stride,padding=1),
                                     nn.ReLU(),
                                     nn.MaxPool1d(kernel_size=2))
        self.d_conv2 = nn.Sequential(nn.Conv1d(8,16,self.d_kernel_size,1,1),
                                     nn.ReLU(),
                                     nn.MaxPool1d(2))
        self.d_fc1 = nn.Sequential(nn.Linear(self.d_2d_dnn_input_dim, self.d_2d_dnn_hidden_dim),
                                   nn.Dropout(self.dropout_rate),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(self.d_2d_dnn_hidden_dim))
        self.d_fc2 = nn.Sequential(nn.Linear(self.d_2d_dnn_hidden_dim, self.d_2d_dnn_out_dim),
                                   nn.Dropout(self.d_dropout),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(self.d_2d_dnn_out_dim))
        self.init_atom_embedding = AtomEmbedding(self.atom_names, self.embed_dim)
        self.init_bond_embedding = BondEmbedding(self.bond_names, self.embed_dim)
        self.init_bond_float_rbf = BondFloatRBF(self.bond_float_names, self.embed_dim)
        self.bond_angle_float_rbf =  BondAngleFloatRBF(self.bond_angle_float_names, self.embed_dim)
        self.distance_rbf = DistanceFloatRBF(self.distance,self.embed_dim)
        # self.graph_pool = gmp()
        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(self.dropout_rate)

        self.linear = nn.Sequential(
            nn.Linear(self.d_dnn_input_dim, self.d_dnn_out_dim),
            # nn.Linear(embeddings_dim, 64),
            nn.Dropout(self.d_dropout),
            nn.ReLU(),
            nn.BatchNorm1d(self.d_dnn_out_dim)
        )

        self.embed_abg_node = torch.nn.Linear(self.d_3d_input_dim * 7, self.d_3d_input_dim)
        self.embed_abg_edge = torch.nn.Linear(self.d_3d_input_dim * 4, self.d_3d_input_dim)
        self.embed_bag_node = torch.nn.Linear(self.d_3d_input_dim * 4, self.d_3d_input_dim)

        self.abg_layers = nn.ModuleList()
        for i in range(n_layers):
            self.abg_layers.append(
                TransformerConv(in_channels=self.d_3d_input_dim, out_channels=self.d_3d_input_dim, heads=4,
                                concat=False, edge_dim=self.d_3d_input_dim)
            )
        self.bag_layers = nn.ModuleList()
        for i in range(n_layers):
            self.bag_layers.append(
                TransformerConv(in_channels=self.d_3d_input_dim, out_channels=self.d_3d_input_dim, heads=4,
                                concat=False, edge_dim=self.d_3d_input_dim)
            )

        self.abg_fc1 = torch.nn.Linear(self.d_3d_dnn_input_dim, self.d_3d_dnn_hidden_dim)
        self.abg_bn1 = nn.BatchNorm1d(self.d_3d_dnn_hidden_dim)
        self.abg_fc2 = torch.nn.Linear(self.d_3d_dnn_hidden_dim, self.d_3d_dnn_out_dim)
        self.abg_bn2 = nn.BatchNorm1d(self.d_3d_dnn_out_dim)

        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.final_in_dim, self.final_hidden_dim[0])
        self.fc2 = nn.Linear(self.final_hidden_dim[0], self.final_hidden_dim[1])
        self.fc3 = nn.Linear(self.final_hidden_dim[1], self.final_hidden_dim[2])
        self.out = nn.Linear(self.final_hidden_dim[2], self.final_out_dim)
        # self.output = nn.Linear(128*4, output_dim)
        self.final_activation = nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, batch, mask, **kwargs):
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        data = batch
        x, edge_index, batch, edge_attr_list = data.x, data.edge_index,data.batch, data.edge_attr
        batch = batch.to(device)
        t_1D_embedding = data.t_1D_feature
        d_2D_embedding = data.d_2D_feature
        bag = data.bag
        # t_graph = data.t_3D_graph
        bag_x = torch.Tensor([])
        bag_edge_index = torch.Tensor([])
        bag_edge_attr = torch.Tensor([])
        bag_batch = torch.Tensor([])
        j = 0
        for g in bag:
            g_x = g.x
            bag_x = torch.cat((bag_x,g_x))
            temp = np.full(len(g_x), j)
            bag_batch = torch.cat((bag_batch, torch.tensor(temp)))
            g_edge_index = g.edge_index
            bag_edge_index = torch.cat((bag_edge_index,g_edge_index),dim=1)
            g_edge_attr = torch.tensor(g.edge_attr)
            bag_edge_attr = torch.cat((bag_edge_attr,g_edge_attr))
            j += 1
        # bag_x = bag_x.to(device).to(torch.float32)
        bag_edge_index = bag_edge_index.to(device).to(torch.int64)
        bag_edge_attr = bag_edge_attr.to('cpu').to(torch.float32)
        bag_edge_attr = torch.unsqueeze(bag_edge_attr, dim=1)

        x = x.to('cpu').to(torch.int64)
        bag_x = bag_x.to('cpu').to(torch.int64)
        edge_index = edge_index.to(device)

        # d_node_hidden = self.init_atom_embedding(d_3D_embedding)
        t_1D = [torch.tensor(item) for item in t_1D_embedding]
        t_1D= pad_sequence(t_1D, batch_first=True)
        t_1D = t_1D.permute(0, 2, 1)
        t_1D = t_1D.to(device).to(torch.float32)
        d_2D= torch.tensor(np.array([item for item in d_2D_embedding]))
        d_2D = d_2D.to(device).to(torch.float32)
        d_2D = d_2D.reshape([d_2D.shape[0], 1, d_2D.shape[1]])
        a = []
        a = torch.Tensor(a)
        for item in edge_attr_list:
            temp = torch.tensor(item)
            a = torch.cat((a, temp))

        edge_attr_list = a

        # target lightattention
        t_o = self.feature_convolution(t_1D)  # [batch_size, embeddings_dim, sequence_length]
        t_o = self.dropout(t_o)  # [batch_gsize, embeddings_dim, sequence_length]
        attention = self.attention_convolution(t_1D)  # [batch_size, embeddings_dim, sequence_length]

        # mask out the padding to which we do not want to pay any attention (we have the padding because the sequences have different lenghts).
        # This padding is added by the dataloader when using the padded_permuted_collate function in utils/general.py
        attention = attention.masked_fill(mask[:, None, :] == False, -1e9)

        # code used for extracting embeddings for UMAP visualizations
        # extraction =  torch.sum(x * self.softmax(attention), dim=-1)
        # extraction = self.id0(extraction)
        att_AA = torch.mean(attention,dim = 1)

        t_o1 = torch.sum(t_o * self.softmax(attention), dim=-1)  # [batchsize, embeddings_dim]
        t_o2, _ = torch.max(t_o, dim=-1)  # [batchsize, embeddings_dim]
        t_o = torch.cat([t_o1, t_o2], dim=-1)  # [batchsize, 2*embeddings_dim]
        t_o = self.linear(t_o)  # [batchsize, 64]

        # # drug_2d
        # d_o = self.d_fc(d_2D)
        d_o = self.d_conv1(d_2D)
        d_o = self.d_conv2(d_o)
        d_o = d_o.view(d_o.size(0), -1)
        d_o = self.d_fc1(d_o)
        d_o = self.d_fc2(d_o)

        # # drug_3D
        #初始化
        node_hidden = self.init_atom_embedding(x)
        node_hidden = node_hidden.to(device)
        edge_int_feat = edge_attr_list[:, 0:3].to('cpu').to(torch.int64)
        edge_float_feat = edge_attr_list[:, 3].to('cpu').to(torch.float32)
        bond_embed = self.init_bond_embedding(edge_int_feat)
        # edge_hidden = bond_embed + self.init_bond_float_rbf(edge_float_feat)
        edge_hidden = torch.cat((bond_embed, self.init_bond_float_rbf(edge_float_feat)), dim=1)
        edge_hidden = edge_hidden.to(device)
        node_hidden = self.embed_abg_node(node_hidden)
        edge_hidden = self.embed_abg_edge(edge_hidden)

        # 1
        for layer_id in range(self.n_layers):
            if layer_id != self.n_layers:
                atom_h = self.abg_layers[layer_id](node_hidden, edge_index, edge_hidden)
                atom_h = self.norm(atom_h)
                atom_h = self.dropout(atom_h)
                # 节点信息 = 本层信息+上层信息
                atom_h = atom_h + node_hidden
                node_hidden = atom_h
            else:
                atom_h = self.abg_layers[layer_id](node_hidden, edge_index, edge_hidden)
                atom_h = self.norm(atom_h)
                atom_h = self.relu(atom_h)
                atom_h = self.dropout(atom_h)
                atom_h = atom_h + node_hidden
                node_hidden = atom_h

            edge_int_feat = edge_attr_list[:, 0:3].to('cpu').to(torch.int64)
            edge_float_feat = edge_attr_list[:, 3].to('cpu').to(torch.float32)
            cur_edge_hidden = self.init_bond_embedding(edge_int_feat)
            # cur_edge_hidden = cur_edge_hidden + self.init_bond_float_rbf(edge_float_feat)
            cur_edge_hidden = torch.cat((cur_edge_hidden, self.init_bond_float_rbf(edge_float_feat)), dim=1)
            cur_edge_hidden = cur_edge_hidden.to(device)
            cur_angle_hidden = self.bond_angle_float_rbf(bag_edge_attr)
            cur_angle_hidden = cur_angle_hidden.to(device)
            cur_edge_hidden = self.embed_bag_node(cur_edge_hidden)
            edge_h = self.bag_layers[layer_id](cur_edge_hidden, bag_edge_index, cur_angle_hidden)
            # edge_h = self.bag_conv1(cur_edge_hidden,bag_edge_index,cur_angle_hidden)
            edge_h = self.norm(edge_h)
            edge_h = self.dropout(edge_h)
            edge_hidden = edge_h + cur_edge_hidden
            # edge_h = self.relu(edge_h)

        atom_h = gmp(atom_h,batch)
        atom_h = self.abg_fc1(atom_h)
        atom_h = self.abg_bn1(atom_h)
        atom_h = self.abg_fc2(atom_h)
        atom_h = self.abg_bn2(atom_h)

        cat_v = torch.cat((t_o, d_o, atom_h), 1)
        # cat_v = torch.cat((t_o,d_o), 1)
        # cat_v = self.output(cat_v) # [batchsize, output_dim]
        fully1 = self.leaky_relu(self.fc1(cat_v))
        fully1 = self.dropout(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
        output = nn.functional.softplus(predict) + 1

        # return output, att_AA
        return output