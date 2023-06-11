# coding: utf-8
"""
Attention modules
"""

from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
import torch.distributions as dist

from entmax import Sparsemax, Entmax15, EntmaxBisect
from .basis_functions import (PowerBasisFunctions,
                                     SineBasisFunctions,
                                     CosineBasisFunctions,
                                     GaussianBasisFunctions)
from .continuous_sparsemax import ContinuousSparsemax
from .continuous_softmax import ContinuousSoftmax

import math

import numpy as np

import pickle

import matplotlib.pyplot as plt



class LongTermAttention(nn.Module):
    def __init__(
        self,
        head_size:int ,
        length: int,
        target_len:int,
        attn_func: str,
        attn_num_basis: int,
        continuous: bool,
        attn_drop: float,
        infinite_memory: bool,
        n_layers: int,
        n_heads: int,
        affines: bool,
        mask: bool,
        mask_type: str,
        mask_dropout: float,
        kl_regularizer: bool,
        sigma_0,
        mu_0,
        sticky_memories,
        **kwargs
    ):
        super(LongTermAttention, self).__init__()

        self.device = 'cuda'
        self.length = length #memory length
        self.target_len = target_len #target length / transformer length
        self.head_size = head_size
        self.attn_num_basis = attn_num_basis
        self.continuous = continuous # whether attention over memory vectors is continuous
        self.attn_func = attn_func # normalizing function
        self.n_head = n_heads
        
        self.kl_regularizer = kl_regularizer
        self.sigma_0 = sigma_0
        self.mu_0 = mu_0

        self.proj_query = nn.Linear(n_heads * head_size, n_heads * head_size, bias = False)
        self.proj_key = nn.Linear(n_heads * head_size, n_heads * head_size, bias = False)
        self.proj_value = nn.Linear(n_heads * head_size, n_heads * head_size, bias = False)

        self.attn_out = nn.Linear(n_heads * head_size, n_heads * head_size, bias = False)
        self.attn_dropout = nn.Dropout(attn_drop)
        
        self.affines = affines # whether mu, sigma should be computed using affine transformations

        self.mask = mask
        self.mask_type = mask_type
        self.mask_dropout_value = mask_dropout

        if self.mask:
            if self.mask_type == 'affine':
                self.mask_net = nn.Linear(length,length)
            elif self.mask_type == 'cnn':
                self.mask_net = torch.nn.Conv1d(n_heads*head_size, n_heads*head_size, 3, padding = 1)
        
        if self.mask_dropout_value>0:
            self.mask_dropout = nn.Dropout(self.mask_dropout_value)

        self.sticky_memories = sticky_memories
        if self.sticky_memories:
            self.attn_past = None

        # TODO - remove this in favor of self.reset_inf()
        # self.mem_threshold = 2048
        self.infinite_memory = infinite_memory # whether the memory is infinite

        self.nb_samples = 512 # number of samples used for update
        self.tau = 0.5 #compressing factor
        self.count = 0

        # TODO - determine if x_part is ever needed
        # self.x_past = None # previous memory vectors
        self.B_past = None # previous coefficient matrix

        self.ridge_penalty=0.5 # ridge penalty
        padding = True

        self.spacing = 'linear'

        def compute_G(l, psi, positions, padding = True):
            F = torch.zeros(self.attn_num_basis, positions.size(0))

            basis_functions = psi
            F[:, :] = basis_functions.evaluate(positions.unsqueeze(1)).t()

            I = torch.eye(self.attn_num_basis)
            G = F.t().matmul((F.matmul(F.t()) + self.ridge_penalty * I).inverse())

            if padding:
                if l % 2:
                    G = G[((l-1)//2):(-(l-1)//2), :]
                else:
                    G = G[(l//2):-(l//2), :]

            return G.to(self.device)

        if self.continuous:
            if self.affines:
                self.mu = nn.Linear(attn_num_basis, 1, bias = False)
                self.sigma = nn.Linear(attn_num_basis, 1, bias = False)
                self.softplus = torch.nn.Softplus()

            # normalizing function
            if attn_func == 'softmax':
                self.transform = ContinuousSoftmax(psi=None)
            elif attn_func == 'sparsemax':
                self.transform = ContinuousSparsemax(psi=None)
            else:
                assert False

            # get basis functions psi
            sigmas = [.005,.01] # basis function sigmas
            if attn_num_basis % len(sigmas):
                attn_num_basis += (len(sigmas) - attn_num_basis % len(sigmas))

            self.psi = [ None ]
            self.Gs = [ None for _ in range(length+1) ]
            self.psi = [ None ]

            lengths = []
            for i in range(length):
                self.psi.append([])
                if (i + 1) % target_len == 0:
                    lengths.append(i + 1)

            if length not in lengths:
                lengths.append(length)

            for l in lengths:
                # get positions for memory vectors
                self.add_gaussian_basis_functions(self.psi[l], attn_num_basis, sigmas, device = self.device)

                if self.spacing == 'linear':
                    if padding:
                        if l % 2:
                            shift = 1 / float(l)
                            positions = torch.linspace(-.5+shift, 1.5-shift, 2*l-1).to(self.device)
                        else:
                            shift = 1 / float(2*l)
                            positions = torch.linspace(-.5+shift, 1.5-shift, 2*l).to(self.device)
                    else:
                        shift = 1 / float(2*l)
                        positions = torch.linspace(shift, 1-shift, l).to(self.device)
                elif self.spacing == 'log':
                    if padding:
                        if l % 2:
                            shift = 1 / float(l)
                            positions = torch.linspace(-.5+shift, 1.5-shift, 2*l-1).to(self.device)
                        else:
                            shift = 1 / float(2*l)
                            positions = torch.linspace(-.5+shift, 1.5-shift, 2*l).to(self.device)

                        pos = np.e**(np.log(1+1)*torch.arange(1,length+1)/length)-1
                        positions = torch.cat([positions[:int(l/2)],pos.to(self.device),positions[-int(l/2):]])

                    else:
                        positions = np.e**(np.log(1+1)*torch.arange(1,length+1)/length)-1

                # compute basis functions
                self.Gs[l] = compute_G(l, self.psi[l][0], positions, padding = padding) # [L,N]
                self.positions = positions[int(l/2):-int(l/2)]

            # compute samples for memory update
            if self.infinite_memory:
                tm_tau = torch.arange(1, self.nb_samples + 1).float()
                tm_l = torch.arange(self.nb_samples + 1, length+self.nb_samples+1).float()
                tm_tau = tm_tau * self.tau/self.nb_samples # positions of old vectors
                tm_l = self.tau + (1 - self.tau) * (tm_l - self.nb_samples) / length # positions of new vectors
                positions_inf = torch.cat([tm_tau, tm_l], 0).to(self.device) # positions

                if padding:
                    if l % 2:
                        shift = 1 / float(length + self.nb_samples)
                        positions_pad_ = torch.linspace(-.5 + shift, 0, 2 * (length + self.nb_samples) - 1).to(self.device)
                    else:
                        shift = 1 / float(2 * length + self.nb_samples)
                        positions_pad = torch.linspace(-.5 + shift, 1.5 - shift, 2 * (length + self.nb_samples)).to(self.device)
                    positions_pad_ = torch.FloatTensor([i for i in positions_pad if i < 0]).to(self.device)
                    positions_pad__ = torch.FloatTensor([i for i in positions_pad if i > 1]).to(self.device)
                    positions_inf = torch.cat([positions_pad_, positions_inf, positions_pad__], dim = 0)

                self.samples = None
                for t in tm_tau:
                    if self.samples is None:
                        self.samples = self.psi[l][0].evaluate(t / self.tau)
                    else:
                        self.samples = torch.cat([self.samples,self.psi[l][0].evaluate(t / self.tau)], dim = 0)

                # compute G for the infinite case
                self.G_inf = compute_G(self.nb_samples + length, self.psi[l][0], positions_inf, padding = padding) #[L+nb_samples,N]

                if self.sticky_memories:
                    self.bins = torch.linspace(0, 1, 129).to(device = self.device) #self.positions
                    self.nb_bins_cat = 1
                    self.bins_cat = dist.Categorical(torch.ones(self.nb_bins_cat))

        elif self.attn_func == 'sparsemax':
            self.sparsemax = Sparsemax(dim = -1)

    def add_gaussian_basis_functions(self, psi, nb_basis, sigmas, device):
        mu, sigma = torch.meshgrid(torch.linspace(0, 1, nb_basis // len(sigmas)), torch.Tensor(sigmas))
        mu = mu.flatten().to(device)
        sigma = sigma.flatten().to(device)
        self.basis_mu = mu
        self.basis_sigma = sigma
        assert mu.size(0) == nb_basis
        psi.append(GaussianBasisFunctions(mu = mu, sigma = sigma))

    def score(self, query, keys):
        query = query / (self.d_head ** 0.5) # divide by sqrt(d_head) [B,h,q,d]
        keys = keys.transpose(-1, -2) #[B,h,d,N]
        scores = torch.matmul(query, keys) #[B,h,q,N] 
        return scores

    def value_function(self, x, inf = False):
        if inf:
            G = self.G_inf # [nb_sample+L,N]
        else:
            G = self.Gs[x.size(-1)] # [L,N]

        B = torch.matmul(x, G) # [B,e,N]
        B = B.permute(0,2,1) # [B,N,e]

        return B

    def reset_inf(self):
        self.B_past = None
        # TODO - determine if x_part is ever needed
        # self.x_past = None
        self.count = 0

    def update_inf(self, x):
        if self.B_past is not None:
            if self.sticky_memories:
                n = dist.Normal(self.attn_past[0],self.attn_past[1])

                bins = self.bins.clone()
                bins[0] = -.000001
                bins[-1] = 1.000001

                p = (n.cdf(bins[1:].repeat(self.attn_past[0].size(1), x.size(0), 1).permute(2, 1, 0))
                    -n.cdf(bins[:-1].repeat(self.attn_past[0].size(1), x.size(0), 1).permute(2, 1, 0))).sum(-1).transpose(1, 0)

                p = (p / p.sum(-1).repeat(p.size(-1), 1).transpose(1, 0))
                p = dist.Categorical(p)

                b = p.sample((self.nb_samples,))

                t = self.bins_cat.sample((self.nb_samples, self.attn_past[0].size(0))).to(device = self.device)

                ts = (t * (self.bins[b + 1] - self.bins[b]) / self.nb_bins_cat + self.bins[b]).transpose(1, 0)

                ts = torch.sort(ts, -1)[0]

                samples = torch.zeros(x.size(0), self.nb_samples, self.attn_num_basis).to(device = self.device)
                for i in range(len(ts)):
                    samples[i] = self.psi[self.length][0].batch_evaluate(ts[i])

                xm_tau = self.B_past.transpose(-1, -2).matmul(samples.transpose(-1, -2)) # [B,e,nb_samples]

            else:
                xm_tau = self.B_past.transpose(-1, -2).matmul(self.samples.transpose(-1, -2)) # [B,e,nb_samples]

            x = torch.cat([xm_tau,x], dim = 2) # [B,e,nb_samples+L]
            B = self.value_function(x, inf = True) # [B,N,e]
        else:
            B = self.value_function(x)

        self.B_past = B.detach()
        # TODO - determine if x_part is ever needed
        # self.x_past = x

        return B

    def forward(self, k, q):
        if self.continuous:
            batch_size = k.size(0) #batch size
            qlen = q.size(2) #query length
            klen = k.size(1) #key length
            self.d_head = self.head_size #head size

            # TODO - remove this in favor of self.reset_inf()
            # clean memory if going through different document
            # if self.count >= self.mem_threshold:
            #     self.B_past = None
            #     self.x_past = None
            #     self.count = 0

            k = k.permute(0, 2, 1) # [B,e,L]
            if self.mask_dropout_value > 0:
                k = self.mask_dropout(k)

            if self.mask:
                reg_mask=torch.sigmoid(self.mask_net(k))
                k = k * reg_mask
            elif self.mask:
                k = k * reg_mask

            # perform memory update
            if self.infinite_memory:
                B = self.update_inf(k)
                self.count += klen
            else: # compute input continuous approximation
                B = self.value_function(k) # [B,N,e]

            keys = self.proj_key(B)
            values = self.proj_value(B)

            query = q
            keys = keys.view(batch_size, self.attn_num_basis, self.n_head, self.d_head).transpose(1, 2) # [B,h,N,d]
            values = values.view(batch_size, self.attn_num_basis, self.n_head, self.d_head).transpose(1, 2) # [B,h,N,d]

            #compute scores
            scores = self.score(query, keys) #[B,h,q,N] 

            #compute mu and sigma
            if self.affines:
                mu = torch.sigmoid(self.mu(scores)) #[B,h,q] 
                sigma_sq = self.softplus(self.sigma(scores))#[B,h,q] 
                mu = mu.view(-1)
                sigma_sq = torch.clamp(sigma_sq, min = 1e-4).view(-1)

                if self.sticky_memories:
                    self.attn_past = [mu.view(batch_size, -1), sigma_sq.view(batch_size, -1) ** (1/2)]
            else:
                scores = torch.softmax(scores,dim = -1)
                mu = torch.matmul(scores, self.basis_mu)
                sigma_sq = torch.matmul(scores, self.basis_mu ** 2 + self.basis_sigma ** 2) - mu ** 2
                mu = mu.view(-1)
                sigma_sq = sigma_sq.view(-1)

            if self.kl_regularizer:
                sigma_0_sq = self.sigma_0 ** 2
                if self.mu_0 > 0:
                    kl_reg = 1/2 * (
                        sigma_sq.view(batch_size, -1) / sigma_0_sq -
                        torch.log(sigma_sq.view(batch_size, -1) / sigma_0_sq) - 1
                    )
                else:
                    kl_reg = 1/2 * (
                        sigma_sq.view(batch_size, -1) / sigma_0_sq -
                        torch.log(sigma_sq.view(batch_size, -1) / sigma_0_sq) - 1 +
                        (mu.view(batch_size, -1) - self.mu_0)**2 / sigma_0_sq
                    )

            # pass parameters to theta
            theta = torch.zeros(batch_size * self.n_head * qlen, 2, device = self.device)  # [B*h*q, 2]
            theta[:, 0] = mu / sigma_sq
            theta[:, 1] = -1. / (2. * sigma_sq)

            # get basis functions
            self.transform.psi = self.psi[klen]

            #compute basis functions expectation
            r = self.transform(theta) # [B*h*q,N] 

            r = r.view(batch_size, self.n_head, qlen, self.attn_num_basis).permute(0, 1, 3, 2) # [B,h,N,q]

            values = values.transpose(-1, -2) # [B,h,d,N]
            
            context = torch.matmul(values, r) # [B,h,d,q]

            context = context.permute(0, 3, 1, 2) # [q,B,h,d]
            context = context.contiguous().view(batch_size, qlen, self.n_head * self.d_head) # [q,B,e]

            context = self.attn_out(context)

            output_density = False
            if output_density:
                positions = torch.linspace(0, 1, klen,device=query.device)
                if type(self.transform) == ContinuousSoftmax:
                    try:
                        if self.alphas_save:
                            aux = True
                    except:
                        self.alphas_save = []
                    import math
                    sgm = torch.sqrt(sigma_sq).unsqueeze(1)
                    density = (
                        (1. / torch.sqrt(2 * math.pi * sgm ** 2)) * 
                        torch.exp(-((positions.unsqueeze(0) - mu.unsqueeze(1)) ** 2 /(2 * sgm ** 2)))
                    ).unsqueeze(1)

                    #density = torch.clamp(density, min=1e-6)
                    alphas = density / torch.sum(density, dim = 2).unsqueeze(-1)
                    alphas = alphas.view(qlen, batch_size, self.n_head, klen).cpu().data
                    self.alphas_save.append(alphas)

                    with open('./alphas_layer_' + str(layer_n),'wb') as f:
                        pickle.dump(self.alphas_save,f)
                else:
                    sgm = torch.sqrt(sigma_sq).unsqueeze(1)
                    lbd = -.5 * (3 / (2 * sgm)) ** (2./3)
                    density = (-lbd - ((positions.unsqueeze(0) - mu.unsqueeze(1)) ** 2 / (2 * sgm ** 2))).unsqueeze(1)
                    density = torch.clamp(density, min=0.)
                    alphas = density / torch.sum(density, dim=2).unsqueeze(-1)
                    alphas = alphas.view(batch_size, qlen, n_head, klen)
        else:
            d_head = k.size(-1)
            n_head = k.size(-2)

            AC = torch.einsum('ibnd,jbnd->ijbn', (q, k))             # qlen x klen x bsz x n_head

            BD = torch.einsum('ibnd,jnd->ijbn', (q, r_k))              # qlen x klen x bsz x n_head

            attn_score = AC + BD
            attn_score.mul_(1 / (d_head ** 0.5))
            attn_prob = torch.softmax(attn_score, dim=1)

            #### compute attention vector
            attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, v))

            # [qlen x bsz x n_head x d_head]
            context = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), n_head * d_head)

            analyse = False
            if analyse:
                attn_prob = attn_prob.permute(0,2,3,1)
                alphas = attn_prob
                alphas_sorted = torch.sort(alphas, dim=-1, descending=True)[0]
                alphas_sorted = torch.cumsum(alphas_sorted, dim=-1)

                n = (alphas_sorted[:,:,:]<=0.9).sum(-1)

                with open('n_ltm_discrete_layer_'+str(layer_n),'ab') as f:
                    pickle.dump(n.cpu().data,f)

        if self.kl_regularizer:
            return context, kl_reg
        else:
            return context
        
    @property
    def _query_dim(self):
        return self.query_layer.in_features

    def __repr__(self):
        return "ContinuousAttention"

