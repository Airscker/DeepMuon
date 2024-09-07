'''
Author: airscker
Date: 2024-07-14 12:45:30
LastEditors: airscker
LastEditTime: 2024-07-14 13:13:00
Description: NULL

Copyright (C) 2024 by Airscker(Yufeng), All Rights Reserved. 
'''
import math
import torch
from torch import nn
from torch.nn import functional as F

class AtomAttentionEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,):
        pass

class TriangularUpdate(nn.Module):
    def __init__(self, num_channels=4, hidden_channels=32, incoming=True) -> None:
        super().__init__()
        self.incoming = incoming
        self.num_channels = num_channels
        self.hidden_channels = hidden_channels
        self.channel_linear_a = nn.Linear(num_channels, hidden_channels)
        self.channel_linear_b = nn.Linear(num_channels, hidden_channels)
        self.input_gate_a = nn.Sequential(
            nn.Linear(num_channels, hidden_channels),
            nn.Sigmoid(),
        )
        self.input_gate_b = nn.Sequential(
            nn.Linear(num_channels, hidden_channels),
            nn.Sigmoid(),
        )
        self.output_gate = nn.Sequential(
            nn.Linear(num_channels, num_channels),
            nn.Sigmoid(),
        )
        self.output_linear = nn.Linear(hidden_channels, num_channels)
    def masked_gate_linear(self, x, mask):
        if mask is not None:
            return x * mask.unsqueeze(-1)
        else:
            return x
    def forward(self, biased_molecular_matrix: torch.Tensor, mask: torch.Tensor=None):
        """
        ### Args:
            - biased_molecular_matrix: torch.Tensor, shape=(B, n_atoms, n_atoms, num_channels)
            - mask: torch.Tensor, shape=(B, n_atoms, n_atoms)
        ### Returns:
            - torch.Tensor, shape=(B, n_atoms, n_atoms, num_channels)
        """
        _, n_atoms, _, _ = biased_molecular_matrix.shape
        biased_molecular_matrix = F.layer_norm(biased_molecular_matrix, normalized_shape=(n_atoms, n_atoms, self.num_channels))
        _a = self.masked_gate_linear(self.input_gate_a(biased_molecular_matrix) * self.channel_linear_a(biased_molecular_matrix),mask) # (B, n_atoms, n_atoms, hidden_channels)
        _b = self.masked_gate_linear(self.input_gate_b(biased_molecular_matrix) * self.channel_linear_b(biased_molecular_matrix),mask)  # (B, n_atoms, n_atoms, hidden_channels)
        _g = self.output_gate(biased_molecular_matrix)
        if self.incoming:
            _z = torch.einsum("bikd,bjkd->bijd", _a, _b)
        else:
            _z = torch.einsum("bkid,bkjd->bijd", _a, _b)
        # plt.figure(figsize=(10,8))
        # plt.subplot(131)
        # plt.imshow(np.abs(_z.detach().numpy()[0,:,:,0]))
        # plt.subplot(132)
        # plt.imshow(np.abs(biased_molecular_matrix.detach().numpy()[0,:,:,0]))
        # plt.subplot(133)
        # plt.imshow(np.abs(mask.detach().numpy()[0,:,:]))
        # plt.show()
        _z = F.layer_norm(_z, normalized_shape=(n_atoms, n_atoms, self.hidden_channels))
        return self.output_linear(_z) * _g
class AtomAttention(nn.Module):
    def __init__(
        self,
        atom_embedding_dim=256,
        num_channels=4,
        num_heads=4,
    ):
        super().__init__()
        self.atom_embedding_dim = atom_embedding_dim
        self.num_channels = num_channels
        self.num_heads = num_heads
        # self.atom_embedding = nn.Embedding(118, atom_embedding_dim)
        self.bias_att_heads = nn.Sequential(
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, num_heads, bias=False)
        )
        self.q_linear = nn.Linear(atom_embedding_dim, atom_embedding_dim * num_heads, bias=False)
        self.k_linear = nn.Linear(atom_embedding_dim, atom_embedding_dim * num_heads, bias=False)
        self.v_linear = nn.Linear(atom_embedding_dim, atom_embedding_dim * num_heads, bias=False)
        self.gate_linear = nn.Sequential(
            nn.Linear(atom_embedding_dim, atom_embedding_dim * num_heads),
            nn.Sigmoid(),
        )
        self.output_linear = nn.Linear(atom_embedding_dim * num_heads, atom_embedding_dim)
    def attention_score(self, embedding: torch.Tensor):
        batch_size, num_seq, _ = embedding.shape
        q = self.q_linear(embedding).reshape(batch_size, num_seq, self.atom_embedding_dim, self.num_heads)
        k = self.k_linear(embedding).reshape(batch_size, num_seq, self.atom_embedding_dim, self.num_heads)
        att = torch.einsum("bqdh,bkdh->bqkh", q, k)
        return att
    def forward(self, molecular_matrix: torch.Tensor, atom_embed: torch.Tensor, embedding_mask: torch.Tensor=None):
        """
        ### Args:
            - molecular_matrix: torch.Tensor, shape=(B, n_atoms, n_atoms, num_channels)
            - atom_embed: torch.Tensor, shape=(B, n_atoms, atom_embedding_dim)
            - embedding_mask: torch.Tensor, shape=(B, n_atoms)
        ### Returns:
            - Updated Multi-Atom Embeddings, torch.Tensor, shape=(B, n_atoms, atom_embedding_dim)
            - Biased Inter-Molecular Attention, torch.Tensor, shape=(B, n_atoms, n_atoms, num_channels)
        """
        batch_size, n_atoms, _, _ = molecular_matrix.shape
        # atom_embed = self.atom_embedding(atomic_num)  # (B, n_atoms, atom_embedding_dim)
        attention_bias = self.bias_att_heads(molecular_matrix)  # (B, n_atoms, n_atoms, num_heads)
        attention_score = self.attention_score(atom_embed)
        attention_score = attention_score + attention_bias
        if embedding_mask is not None:
            attention_score = attention_score.masked_fill(~embedding_mask.unsqueeze(1).unsqueeze(3), -torch.finfo(attention_score.dtype).max)
        attention_score = F.softmax(
            attention_score/math.sqrt(self.atom_embedding_dim),
            dim=2
        ) # (B, n_atoms, n_atoms, num_heads), make sure the key_dim is normalized so that the value_dim can be weighted using seq_dim
        atom_embed_v = self.v_linear(atom_embed).reshape(
            batch_size, n_atoms, self.atom_embedding_dim, self.num_heads
        ) # (B, n_atoms, atom_embedding_dim, num_heads)
        weighted_output = torch.einsum("bqkh,bvdh->bqdh", attention_score, atom_embed_v) # (B, n_atoms, atom_embedding_dim, num_heads)
        gate_weight = self.gate_linear(atom_embed).reshape(batch_size, n_atoms, self.atom_embedding_dim, self.num_heads)
        weighted_output = weighted_output * gate_weight
        weighted_output = weighted_output.reshape(batch_size, n_atoms, self.atom_embedding_dim * self.num_heads)
        output = self.output_linear(weighted_output) # (B, n_atoms, atom_embedding_dim)
        return output
class TriangularAttention(nn.Module):
    def __init__(self,num_channels=4,num_heads=4,hidden_dim=32,starting_node=True) -> None:
        super().__init__()
        self.num_channels=num_channels
        self.hidden_dim=hidden_dim
        self.num_heads=num_heads
        self.starting_node=starting_node
        self.q_linear=nn.Linear(num_channels,hidden_dim*num_heads,bias=False)
        self.k_linear=nn.Linear(num_channels,hidden_dim*num_heads,bias=False)
        self.v_linear=nn.Linear(num_channels,hidden_dim*num_heads,bias=False)
        self.b_linear=nn.Linear(num_channels,num_heads,bias=False)
        self.gate_linear=nn.Sequential(
            nn.Linear(num_channels,hidden_dim*num_heads),
            nn.Sigmoid()
        )
        self.output_linear=nn.Linear(hidden_dim*num_heads,num_channels)
    def forward(self,biased_molecular_matrix: torch.Tensor,mask: torch.Tensor=None):
        """
        ### Args:
            - biased_molecular_matrix: torch.Tensor, shape=(B, n_atoms, n_atoms, num_channels)
            - mask: torch.Tensor, shape=(B, n_atoms, n_atoms)
        ### Returns:
            - torch.Tensor, shape=(B, n_atoms, n_atoms, num_channels)
        """
        batch_size, n_atoms, _, _ = biased_molecular_matrix.shape
        biased_molecular_matrix=F.layer_norm(biased_molecular_matrix,normalized_shape=(n_atoms,n_atoms,self.num_channels))
        _q=self.q_linear(biased_molecular_matrix).reshape(batch_size,n_atoms,n_atoms,self.hidden_dim,self.num_heads)
        _k=self.k_linear(biased_molecular_matrix).reshape(batch_size,n_atoms,n_atoms,self.hidden_dim,self.num_heads)
        _v=self.v_linear(biased_molecular_matrix).reshape(batch_size,n_atoms,n_atoms,self.hidden_dim,self.num_heads)
        _b=self.b_linear(biased_molecular_matrix) # (B, n_atoms, n_atoms, num_heads)
        if self.starting_node:
            attention_score=torch.einsum("bijdh,bikdh->bijkh",_q,_k)+_b.unsqueeze(1)
            if mask is not None:
                attention_score=attention_score.masked_fill(~mask.unsqueeze(-1).unsqueeze(-1),-torch.finfo(attention_score.dtype).max)
            attention_score=F.softmax(attention_score/math.sqrt(self.hidden_dim),dim=3)
            weighted_output=torch.einsum("bijkh,bikdh->bijdh",attention_score,_v) # (B, n_atoms, n_atoms, hidden_dim, num_heads)
        else:
            attention_score=torch.einsum("bijdh,bkjdh->bijkh",_q,_k)+_b.permute(0,2,1,3).unsqueeze(1)
            if mask is not None:
                attention_score=attention_score.masked_fill(~mask.unsqueeze(-1).unsqueeze(-1),-torch.finfo(attention_score.dtype).max)
            attention_score=F.softmax(attention_score/math.sqrt(self.hidden_dim),dim=3)
            weighted_output=torch.einsum("bijkh,bkjdh->bijdh",attention_score,_v)
        gate_weight=self.gate_linear(biased_molecular_matrix).reshape(batch_size,n_atoms,n_atoms,self.hidden_dim,self.num_heads)
        weighted_output=weighted_output*gate_weight
        output=self.output_linear(weighted_output.reshape(batch_size,n_atoms,n_atoms,self.hidden_dim*self.num_heads))
        return output
class TransitionLayer(nn.Module):
    def __init__(self,in_dim,expand_factor:int=4) -> None:
        super().__init__()
        assert isinstance(expand_factor,int) and expand_factor>=1
        self.mlp=nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim,in_dim*expand_factor),
            nn.ReLU(),
            nn.Linear(in_dim*expand_factor,in_dim)
        )
    def forward(self,embedding: torch.Tensor):
        '''
        ### Args:
            - embedding: torch.Tensor, shape=(B, _, embedding_dim)
        ### Returns:
            - torch.Tensor, shape=(B, _, embedding_dim)
        '''
        return self.mlp(embedding)

class InteractionAttention(nn.Module):
    def __init__(self,energy_dim=100, hidden_dim=256, num_channels=4, num_heads=4, expand_factor=4, column_wise_att=False) -> None:
        super().__init__()
        self.energy_dim=energy_dim
        self.hidden_dim=hidden_dim
        self.num_channels=num_channels
        self.num_heads=num_heads
        self.column_wise_att=column_wise_att
        self.matrix_linear=nn.Linear(num_channels,num_heads,bias=False)
        self.rq_linear=nn.Linear(hidden_dim,hidden_dim*num_heads,bias=False)
        self.rk_linear=nn.Linear(hidden_dim,hidden_dim*num_heads,bias=False)
        self.rv_linear=nn.Linear(hidden_dim,hidden_dim*num_heads,bias=False)
        self.rgate_linear=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim*num_heads,bias=False),
            nn.Sigmoid()
        )
        self.routput_linear=nn.Linear(hidden_dim*num_heads,hidden_dim)
        if self.column_wise_att:
            self.cq_linear=nn.Linear(hidden_dim,hidden_dim*num_heads,bias=False)
            self.ck_linear=nn.Linear(hidden_dim,hidden_dim*num_heads,bias=False)
            self.cv_linear=nn.Linear(hidden_dim,hidden_dim*num_heads,bias=False)
            self.cgate_linear=nn.Sequential(
                nn.Linear(hidden_dim,hidden_dim*num_heads,bias=False),
                nn.Sigmoid()
            )
            self.coutput_linear=nn.Linear(hidden_dim*num_heads,hidden_dim)
        self.transition=TransitionLayer(hidden_dim,expand_factor)
    def row_wise_attention(self,atom_abs_att:torch.Tensor,att_bias: torch.Tensor,batch_size:int,n_atoms:int,mask:torch.Tensor=None):
        '''
        ### Args:
            - atom_abs_att: torch.Tensor, shape=(B, energy_dim, n_atoms, hidden_dim)
            - att_bias: torch.Tensor, shape=(B, n_atoms, n_atoms, num_channels)
            - batch_size: int
            - n_atoms: int
            - mask: torch.Tensor, shape=(B, n_atoms, n_atoms)
        ### Returns:
            - torch.Tensor, shape=(B, energy_dim, n_atoms, hidden_dim)
        '''
        q=self.rq_linear(atom_abs_att).reshape(batch_size,self.energy_dim,n_atoms,self.hidden_dim,self.num_heads)
        k=self.rk_linear(atom_abs_att).reshape(batch_size,self.energy_dim,n_atoms,self.hidden_dim,self.num_heads)
        v=self.rv_linear(atom_abs_att).reshape(batch_size,self.energy_dim,n_atoms,self.hidden_dim,self.num_heads)
        att_bias=self.matrix_linear(att_bias).reshape(batch_size,n_atoms,n_atoms,self.num_heads)
        # att=torch.einsum("blidh,bljdh->blijdh",q,k)+att_bias.unsqueeze(1).unsqueeze(4)
        att=torch.einsum("blidh,bljdh->blijh",q,k)+att_bias.unsqueeze(1)

        if mask is not None:
            att=att.masked_fill(~mask.unsqueeze(1).unsqueeze(-1),-torch.finfo(att.dtype).max)

        att=F.softmax(att/math.sqrt(self.hidden_dim),dim=3)
        # output=torch.einsum("blijdh,bljdh->blidh",att,v)
        output=torch.einsum("blijh,bljdh->blidh",att,v)
        gate_weights=self.rgate_linear(atom_abs_att).reshape(batch_size,self.energy_dim,n_atoms,self.hidden_dim,self.num_heads) # (B, energy_dim, n_atoms, hidden_dim, num_heads)
        output=output*gate_weights
        output=self.routput_linear(output.reshape(batch_size,self.energy_dim,n_atoms,self.hidden_dim*self.num_heads))
        return output
    def column_wise_attention(self,atom_abs_att:torch.Tensor,batch_size:int,n_atoms:int,mask:torch.Tensor=None):
        '''
        ### Args:
            - atom_abs_att: torch.Tensor, shape=(B, energy_dim, n_atoms, hidden_dim)
            - batch_size: int
            - n_atoms: int
            - mask: torch.Tensor, shape=(B, n_atoms)
        ### Returns:
            - torch.Tensor, shape=(B, energy_dim, n_atoms, hidden_dim)
        '''
        q=self.cq_linear(atom_abs_att).reshape(batch_size,self.energy_dim,n_atoms,self.hidden_dim,self.num_heads)
        k=self.ck_linear(atom_abs_att).reshape(batch_size,self.energy_dim,n_atoms,self.hidden_dim,self.num_heads)
        v=self.cv_linear(atom_abs_att).reshape(batch_size,self.energy_dim,n_atoms,self.hidden_dim,self.num_heads)
        # att=torch.einsum("blndh,bkndh->blkndh",q,k)
        att=torch.einsum("blndh,bkndh->blknh",q,k)

        if mask is not None:
            att=att.masked_fill(~mask.unsqueeze(1).unsqueeze(1).unsqueeze(-1),-torch.finfo(att.dtype).max)

        att=F.softmax(att/math.sqrt(self.hidden_dim),dim=2)
        # output=torch.einsum("blkndh,bkndh->blndh",att,v).reshape(batch_size,self.energy_dim,n_atoms,self.hidden_dim*self.num_heads)
        output=torch.einsum("blknh,bkndh->blndh",att,v).reshape(batch_size,self.energy_dim,n_atoms,self.hidden_dim*self.num_heads)
        gate_weights=self.cgate_linear(atom_abs_att)
        output=output*gate_weights
        output=self.coutput_linear(output)
        return output

    def forward(self,
                energy_range: torch.Tensor,
                embedding: torch.Tensor,
                molecular_matrix: torch.Tensor,
                embedding_mask: torch.Tensor=None,
                matrix_mask: torch.Tensor=None):
        '''
        ### Args:
            - energy_range: torch.Tensor, shape=(B, energy_dim, hidden_dim)
            - embedding: torch.Tensor, shape=(B, n_atoms, hidden_dim)
            - molecular_matrix: torch.Tensor, shape=(B, n_atoms, n_atoms, num_channels)
            - embedding_mask: torch.Tensor, shape=(B, n_atoms)
            - matrix_mask: torch.Tensor, shape=(B, n_atoms, n_atoms)
        ### Returns:
            - torch.Tensor, shape=(B, energy_dim, hidden_dim)
        '''
        batch_size,n_atoms,_=embedding.shape
        energy_range=F.layer_norm(energy_range,normalized_shape=(self.energy_dim,self.hidden_dim))
        embedding=F.layer_norm(embedding,normalized_shape=(n_atoms,self.hidden_dim))
        # print(energy_range.shape,embedding.shape,molecular_matrix.shape)
        atom_abs=torch.einsum('bid,bjd->bijd',energy_range,embedding) # (B, energy_dim, n_atoms, hidden_dim)
        if embedding_mask is not None:
            atom_abs=atom_abs*embedding_mask.unsqueeze(1).unsqueeze(3)
        # plt.figure(figsize=(10,8))
        # plt.subplot(181)
        # plt.imshow(np.abs(atom_abs.detach().numpy()[0,:,:,0]))
        # plt.subplot(182)
        # plt.imshow(np.abs(atom_abs.detach().numpy()[1,:,:,0]))
        # plt.subplot(183)
        # plt.imshow(np.abs(atom_abs.detach().numpy()[2,:,:,0]))
        # plt.subplot(184)
        # plt.imshow(np.abs(embedding_mask.detach().numpy()))
        # # plt.subplot(185)
        # # plt.imshow(np.abs(_atom_abs.detach().numpy()[0,:,:,0]))
        # # plt.subplot(186)
        # # plt.imshow(np.abs(_atom_abs.detach().numpy()[1,:,:,0]))
        # # plt.subplot(187)
        # # plt.imshow(np.abs(_atom_abs.detach().numpy()[2,:,:,0]))
        # plt.show()
        
        atom_abs=self.row_wise_attention(atom_abs,molecular_matrix,batch_size,n_atoms,matrix_mask)
        if self.column_wise_att:
            atom_abs=self.column_wise_attention(atom_abs,batch_size,n_atoms,embedding_mask)
        atom_abs=self.transition(atom_abs).mean(dim=2) # (B, energy_dim, hidden_dim)
        return atom_abs

class FourierEmbedding(nn.Module):
    def __init__(self,energy_dim=100,embedding_dim=256) -> None:
        super().__init__()
        self.linear=nn.Linear(energy_dim,energy_dim)
        self.embedding=nn.Linear(1,embedding_dim)
    def forward(self,energy_range: torch.Tensor):
        '''
        ### Args:
            - energy_range: torch.Tensor, shape=(B, energy_dim)
        ### Returns:
            - torch.Tensor, shape=(B, energy_dim, embedding_dim)
        '''
        embedding=torch.cos(2*torch.pi*self.linear(energy_range))
        return self.embedding(embedding.unsqueeze(-1))

class AtomFormerBlock(nn.Module):
    def __init__(self,num_channels=4,energy_dim=100,hidden_dim=256,num_heads=4,expand_factor=4,hidden_channels=32) -> None:
        super().__init__()
        self.gated_attention=AtomAttention(hidden_dim,num_channels,num_heads)
        self.triangular_update_outgoing=TriangularUpdate(num_channels,hidden_channels,incoming=False)
        self.triangular_update_incoming=TriangularUpdate(num_channels,hidden_channels,incoming=True)
        self.triangular_attention_start=TriangularAttention(num_channels,num_heads,hidden_dim,starting_node=True)
        self.triangular_attention_end=TriangularAttention(num_channels,num_heads,hidden_dim,starting_node=False)
        self.matrix_transition=TransitionLayer(num_channels,expand_factor)
        self.interaction=InteractionAttention(energy_dim,hidden_dim,num_channels,num_heads,expand_factor,column_wise_att=False)
        self.embedding_transition=TransitionLayer(hidden_dim,expand_factor)
    def forward(self,energy_range: torch.Tensor,
                embedding: torch.Tensor,
                molecular_matrix: torch.Tensor,
                embedding_mask: torch.Tensor=None,
                matrix_mask: torch.Tensor=None):
        '''
        ### Args:
            - energy_range: torch.Tensor, shape=(B, energy_dim, hidden_dim)
            - embedding: torch.Tensor, shape=(B, n_atoms, hidden_dim)
            - molecular_matrix: torch.Tensor, shape=(B, n_atoms, n_atoms, num_channels)
            - embedding_mask: torch.Tensor, shape=(B, n_atoms)
            - matrix_mask: torch.Tensor, shape=(B, n_atoms, n_atoms)
        ### Returns:
            - torch.Tensor, shape=(B, n_atoms, hidden_dim)
        '''
        triupdate_outgoing=self.triangular_update_outgoing(molecular_matrix,matrix_mask)+molecular_matrix
        triupdate_incoming=self.triangular_update_incoming(molecular_matrix,matrix_mask)+triupdate_outgoing
        triatt_start=self.triangular_attention_start(triupdate_incoming,matrix_mask)+triupdate_incoming
        triatt_end=self.triangular_attention_end(triatt_start,matrix_mask)+triatt_start
        molecular_matrix_out=self.matrix_transition(triatt_end)+triatt_end

        updated_embedding=self.gated_attention(molecular_matrix_out,embedding,embedding_mask)+embedding
        updated_embedding=self.embedding_transition(updated_embedding)+updated_embedding

        energy_spec=self.interaction(energy_range,updated_embedding,molecular_matrix_out,embedding_mask,matrix_mask)
        return molecular_matrix_out,updated_embedding,energy_spec
class AlphaXAS(nn.Module):
    def __init__(self,
                 energy_dim=100,
                 hidden_dim=256,
                 num_heads=4,
                 num_channels=4,
                 hidden_channels=32,
                 expand_factor=4,
                 atom_former_blocks=8) -> None:
        super().__init__()
        self.fourier_embedding=FourierEmbedding(energy_dim=energy_dim,embedding_dim=hidden_dim)
        self.atom_formers=nn.ModuleList([
            AtomFormerBlock(num_channels=num_channels,
                            energy_dim=energy_dim,
                            hidden_dim=hidden_dim,
                            num_heads=num_heads,
                            expand_factor=expand_factor,
                            hidden_channels=hidden_channels) for _ in range(atom_former_blocks)
        ])
        self.atom_embedding=nn.Embedding(119,hidden_dim)
        self.xas_module=InteractionAttention(energy_dim=energy_dim,
                                             hidden_dim=hidden_dim,
                                             num_channels=num_channels,
                                             num_heads=num_heads,
                                             expand_factor=expand_factor,
                                             column_wise_att=False)
        self.output_linear=nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim,hidden_dim*expand_factor),
            nn.ReLU(),
            nn.Linear(hidden_dim*expand_factor,1)
        )
    def forward(self,input:list,device:torch.device):
        '''
        ### Args:
            - energy_range: torch.Tensor, shape=(B, energy_dim)
            - embedding: torch.Tensor, shape=(B, n_atoms)
            - molecular_matrix: torch.Tensor, shape=(B, n_atoms, n_atoms, num_channels)
            - embedding_mask: torch.Tensor, shape=(B, n_atoms)
            - matrix_mask: torch.Tensor, shape=(B, n_atoms, n_atoms)
        '''
        energy_range,embedding,molecular_matrix,embedding_mask,matrix_mask=input
        energy_range=energy_range.to(device)
        embedding=embedding.to(device)
        molecular_matrix=molecular_matrix.to(device)
        if embedding_mask is not None:
            embedding_mask=embedding_mask.to(device)
        else:
            embedding_mask=None
        if matrix_mask is not None:
            matrix_mask=matrix_mask.to(device)
        else:
            matrix_mask=None
        energy_range=self.fourier_embedding(energy_range) # (B, energy_dim, hidden_dim)
        embedding=self.atom_embedding(embedding) # (B, n_atoms, hidden_dim)
        if embedding_mask is not None:
            embedding=embedding*embedding_mask.unsqueeze(-1)
        for atom_former in self.atom_formers:
            molecular_matrix,embedding,energy_range=atom_former(energy_range,embedding,molecular_matrix,embedding_mask,matrix_mask)
        energy_spec=self.xas_module(energy_range,embedding,molecular_matrix,embedding_mask,matrix_mask)
        energy_spec=self.output_linear(energy_spec).squeeze(-1)
        return energy_spec