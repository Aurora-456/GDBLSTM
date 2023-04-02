import numpy as np
import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.utils import to_dense_adj
import torch_geometric.utils as u
from scipy import sparse
import torch_sparse
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp, InnerProductDecoder
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class ResidualBlock(torch.nn.Module):
  def __init__(self, outfeature):
    super(ResidualBlock, self).__init__()
    self.outfeature = outfeature
    self.gcn = GCNConv(outfeature, outfeature)
    self.ln = torch.nn.Linear(outfeature, outfeature, bias=False)
    self.relu = nn.ReLU()


  def init_weights(self):
    nn.init.xavier_uniform_(self.gcn.weight)

  def forward(self, x, edge_index):
    identity = x
    out = self.gcn(x, edge_index)
    out = self.relu(out)
    out = self.ln(out)
    out += identity
    out = self.relu(out)
    return out

class GATGCNNet(torch.nn.Module):
  def __init__(self,embed_dim, num_layer, device, num_feature_xd=78,n_output=1, num_feature_xt=25,
              output_dim=128, dropout=0.2):
    super(GATGCNNet,self).__init__()

    self.device = device
    self.num_layer = num_layer
    self.embed_dim = embed_dim
    self.num_rblock = 4

    # GAT+GCN
    self.n_output = n_output


    self.conv1 = GATConv(num_feature_xd, num_feature_xd, heads=10)
    self.conv2 = GCNConv(num_feature_xd * 10, num_feature_xd * 10 * 2 )
    self.conv3 = GCNConv(num_feature_xd*2*10, num_feature_xd*2*2*10)
    self.rblock_xd = ResidualBlock(num_feature_xd * 2*2*10)
    self.fc_g1 = torch.nn.Linear(num_feature_xd * 10 *2*2, 1500)
    self.fc_g2 = torch.nn.Linear(1500, output_dim)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(dropout)

    #protien sequence branch (LSTM)
    self.embedding_xt = nn.Embedding(num_feature_xt+1,embed_dim)

    self.LSTM_xt_1 = nn.LSTM(self.embed_dim,self.embed_dim,self.num_layer,batch_first = True,bidirectional=True)
    self.fc_xt1 = nn.Linear(1000*256,1500)
    # 多加一层全连接层
    self.fc_xt2 = nn.Linear(1500, output_dim)

    #combined layers
    self.fc1 = nn.Linear(2*output_dim,1024)
    self.fc2 = nn.Linear(1024,512)
    self.out = nn.Linear(512,n_output)

  def forward(self,data,hidden,cell):
    x , edge_index, batch = data.x,data.edge_index,data.batch
    target = data.target

    # print('x shape = ', x.shape)
    x = self.conv1(x, edge_index)
    x = self.relu(x)
    x = self.conv2(x, edge_index)
    x = self.relu(x)
    x = self.conv3(x, edge_index)
    x = self.relu(x)
    for i in range(self.num_rblock):
      x = self.rblock_xd(x, edge_index)
    x = gmp(x, batch)
    x = self.relu(self.fc_g1(x))
    x = self.dropout(x)
    x = self.fc_g2(x)

    # LSTM layer
    embedded_xt = self.embedding_xt(target)
    LSTM_xt,(hidden,cell) = self.LSTM_xt_1(embedded_xt,(hidden,cell))
    # conv_xt = self.conv_xt_1(embedded_xt)
    xt = LSTM_xt.contiguous().view(-1,1000*256)
    xt = self.fc_xt1(xt)
    xt = self.fc_xt2(xt)

    #concat
    xc = torch.cat((x,xt),1)
    # add some dense layers
    xc = self.fc1(xc)
    xc = self.relu(xc)
    xc = self.dropout(xc)
    xc = self.fc2(xc)
    xc = self.relu(xc)
    xc = self.dropout(xc)
    out = self.out(xc)
    return out


  def init_hidden(self, batch_size):

    hidden = torch.zeros(4,batch_size,self.embed_dim).to(self.device)
    cell = torch.zeros(4,batch_size,self.embed_dim).to(self.device)
    return hidden,cell



