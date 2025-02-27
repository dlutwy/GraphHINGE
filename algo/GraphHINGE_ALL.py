import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv
import os
import sys
import gc
sys.path.append("..")
from utils import data_loader,utils
import argparse
import random
import pickle as pkl
import tqdm

class Interaction(nn.Module):
    def __init__(self, s_padding, t_padding):
        super(Interaction, self).__init__()
        self.ms = torch.nn.ZeroPad2d((0,0,0,s_padding))
        self.mt = torch.nn.ZeroPad2d((0,0,0,t_padding))
       
    def forward(self, s, t):
        #s,t: B*L*E*N
        s_imaginary=s*0
        t_imaginary=t*0
        fs=torch.stack((s,s_imaginary),4)
        ft=torch.stack((t,t_imaginary),4)
        ft=self.mt(ft)
        fs=self.ms(fs)
        fs=torch.Tensor.fft(fs,1)
        ft=torch.Tensor.fft(ft,1)
        H=[]
        fft=ft
        for i in range(0,s.shape[1]):
            ffs=torch.roll(fs, i, 1)
            rr=torch.Tensor.mul(ffs[:,:,:,:,0],fft[:,:,:,:,0])
            ii=torch.Tensor.mul(ffs[:,:,:,:,1],fft[:,:,:,:,1])
            ri=torch.Tensor.mul(ffs[:,:,:,:,0],fft[:,:,:,:,1])
            ir=torch.Tensor.mul(ffs[:,:,:,:,1],fft[:,:,:,:,0])
            h=torch.stack((rr-ii,ri+ir),axis=4) #B*L*E*N*2
            H.append(h)
        H = torch.cat(H,1)
        H=torch.Tensor.ifft(H,1)
        H=H[:,:,:,:,0]
        H=H.permute(0,1,3,2) #B*L*(Is+It-1)*E
        return H

class NodeAttention(nn.Module):
    def __init__(self, in_size, out_size=128, atn_heads = 3, temp = 0.2):
        super(NodeAttention, self).__init__()
        '''
        h: B*L*N*E
        in_size = out_size = E
        '''
        self.in_size = in_size
        self.out_size = out_size
        self.atn_heads = atn_heads
        self.softmax = torch.nn.Softmax(dim=1)
        self.temp = temp
        
        self.Wt = nn.Parameter(utils.glorot((atn_heads, in_size, out_size)))
        self.Ws = nn.Parameter(utils.glorot((atn_heads, in_size, out_size)))
        self.Wc = nn.Parameter(utils.glorot((atn_heads, in_size, out_size)))

        self.path_att = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.Tanh(),
            nn.Linear(out_size, 1, bias=False)
        )
        
    
    def forward(self, h):

        H=torch.reshape(h,(h.shape[0]*h.shape[1],h.shape[2],h.shape[3])) #(B*L)*N*E
        for i in range(self.atn_heads):
            hi = torch.Tensor.matmul(H[:,0,:].unsqueeze(1), self.Wt[i,:,:]) #(B*L)*1*E_
            hj = torch.Tensor.matmul(H, self.Ws[i,:,:]) #(B*L)*N*E_
            hij = (torch.Tensor.bmm(hj, hi.permute(0, 2, 1))/self.temp) #(B*L)*N*1
            alpha = self.softmax(hij).expand(h.shape[0]*h.shape[1],h.shape[2],h.shape[3]) #(B*L)*N*1 -> (B*L)*N*E
            Hc = torch.Tensor.matmul(H, self.Wc[i,:,:]) #(B*L)*N*E -> (B*L)*N*E
            if i==0:
                Z = (alpha * Hc).sum(1) #(B*L)*N*E ->(B*L)*E_
            else:
                Z += (alpha * Hc).sum(1) #(B*L)*E_
                
        Z = torch.reshape(Z,(h.shape[0],h.shape[1],self.out_size)) #B*L*E
        Z = Z/self.atn_heads
        return Z

class PathAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128, temp=0.2):
        super(PathAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.temp=temp

    def forward(self, z):
        '''
        z: B*M*L*E (batch_size,num_metas,num_paths,hidden)
        '''
        z=z.reshape(z.shape[0],z.shape[1]*z.shape[2],z.shape[3]) #B*(M*L)*E
        w = self.project(z)/self.temp #B*(M*L)*1            
        beta = F.softmax(w,1) #B*(M*L)*1
        beta = beta.expand(z.shape) 

        return (beta * z).sum(1)          #B*E                

 
class GraphHINGE(nn.Module):
    
    def __init__(self, user_num, item_num, attr1_num, attr2_num, attr3_num, in_size, hidden_size, out_size, num_heads, temp1=0.2, temp2=0.2):
        super(GraphHINGE, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.num_heads = num_heads
        self.user_emb = nn.Embedding(user_num+1, in_size, padding_idx=0)
        self.item_emb = nn.Embedding(item_num+1, in_size, padding_idx=0)
        self.attr1_emb = nn.Embedding(attr1_num+1, in_size, padding_idx=0)
        self.attr2_emb = nn.Embedding(attr2_num+1, in_size, padding_idx=0)
        self.attr3_emb = nn.Embedding(attr3_num+1, in_size, padding_idx=0)
        #self.Interaction0 = Interaction(1,1)
        #self.Interaction = Interaction(3,3)
        self.NodeAttention = nn.ModuleList()
        for i in range(0,5):
            self.NodeAttention.append(NodeAttention(in_size, hidden_size, num_heads, temp1))
        self.PathAttention = PathAttention(hidden_size, hidden_size, temp2)
        self.final_linear = nn.Sequential(
            nn.Linear(3*hidden_size, out_size),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(out_size, 1)
        )
        
    def interaction(self, s, t):
        #s,t: B*L*E*N
        length = s.shape[-1] + t.shape[-1]
        fs = torch.fft.fft(s, n=length)
        ft = torch.fft.fft(t, n=length)
        H = []
        for i in range(0, s.shape[1]):
            rfs = torch.roll(fs, i, 1)
            h = rfs*ft
            h = torch.fft.ifft(h)
            H.append(h.permute(0,1,3,2).float()) #B*L*(Is+It-1)*E
        h = torch.cat(H, 1)
        return h
        
    def forward(self, UI, IU, UIUI, IUIU, UIAI1, IAIU1, UIAI2, IAIU2, UIAI3, IAIU3):
        user_idx = UI[:,0,0] #B
        item_idx = IU[:,0,0] #B
        source_feature = self.user_emb(user_idx) #B*E
        target_feature = self.item_emb(item_idx) #B*E

        #UI & IU B*L*N (batch_size, num_paths, num_nodes)
        ui = torch.stack((self.user_emb(UI[:,:,0]), self.item_emb(UI[:,:,1])),3) #B*L*E*N
        iu = torch.stack((self.item_emb(IU[:,:,0]), self.user_emb(IU[:,:,1])),3) #B*L*E*N
        uiui = torch.stack((self.user_emb(UIUI[:,:,0]), self.item_emb(UIUI[:,:,1]), self.user_emb(UIUI[:,:,2]), self.item_emb(UIUI[:,:,3])),3) #B*L*E*N
        iuiu = torch.stack((self.item_emb(IUIU[:,:,0]), self.user_emb(IUIU[:,:,1]), self.item_emb(IUIU[:,:,2]), self.user_emb(IUIU[:,:,3])),3) #B*L*E*N
        uiai1 = torch.stack((self.user_emb(UIAI1[:,:,0]), self.item_emb(UIAI1[:,:,1]), self.attr1_emb(UIAI1[:,:,2]), self.item_emb(UIAI1[:,:,3])),3) #B*L*E*N
        iaiu1 = torch.stack((self.item_emb(IAIU1[:,:,0]), self.attr1_emb(IAIU1[:,:,1]), self.item_emb(IAIU1[:,:,2]), self.user_emb(IAIU1[:,:,3])),3) #B*L*E*N
        uiai2 = torch.stack((self.user_emb(UIAI2[:,:,0]), self.item_emb(UIAI2[:,:,1]), self.attr2_emb(UIAI2[:,:,2]), self.item_emb(UIAI2[:,:,3])),3) #B*L*E*N
        iaiu2 = torch.stack((self.item_emb(IAIU2[:,:,0]), self.attr2_emb(IAIU2[:,:,1]), self.item_emb(IAIU2[:,:,2]), self.user_emb(IAIU2[:,:,3])),3) #B*L*E*N
        uiai3 = torch.stack((self.user_emb(UIAI3[:,:,0]), self.item_emb(UIAI3[:,:,1]), self.attr3_emb(UIAI3[:,:,2]), self.item_emb(UIAI3[:,:,3])),3) #B*L*E*N
        iaiu3 = torch.stack((self.item_emb(IAIU3[:,:,0]), self.attr3_emb(IAIU3[:,:,1]), self.item_emb(IAIU3[:,:,2]), self.user_emb(IAIU3[:,:,3])),3) #B*L*E*N
        
        user_features = [uiui,uiai1,uiai2,uiai3]
        item_features = [iuiu,iaiu1,iaiu2,iaiu3]
        H=[]

        #h=self.Interaction0(ui,iu)
        h=self.interaction(ui, iu)
        h=self.NodeAttention[0](h)
        H.append(h)
        for i in range(0,len(user_features)):
            #h = self.Interaction(user_features[i], item_features[i])
            h = self.interaction(user_features[i], item_features[i])
            h = self.NodeAttention[i+1](h)
            H.append(h)
        
        Z = torch.stack(H,1)
        pred = self.PathAttention(Z)
        pred = torch.cat((source_feature,target_feature,pred),1) #B*3E 
        pred=self.final_linear(pred)          
        pred=torch.sigmoid(pred)
        return pred