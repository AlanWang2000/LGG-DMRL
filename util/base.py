from turtle import forward
from torch import layer_norm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal
from torch.nn.modules.container import ModuleList
import math


class DNN(nn.Module):
    def __init__(self, dims, activation='sigmoid', last=True, softmax=False):
        super(DNN, self).__init__()
        # net
        self.net = ModuleList()
        layers = len(dims) - 1
        for i in range(layers):
            self.net.append(nn.Linear(dims[i], dims[i + 1]))
        # activation function
        self.act = None
        if activation is None:
            pass
        elif activation == 'relu':
            self.act = nn.ReLU(True)
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'softplus':
            self.act = nn.Softplus()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'prelu':
            self.act = nn.PReLU()
        elif activation.startswith('leakyrelu'):
            self.act = nn.LeakyReLU(float(activation.split(":")[1]))
        # activation or not after the last layer
        self.last = last
        # softmax or not after the last layer
        self.softmax = softmax

    def forward(self, x):
        res = x
        layer = len(self.net)
        for i in range(layer - 1):
            res = self.net[i](res)
            res = self.act(res) if self.act is not None else res
        res = self.net[-1](res)
        if self.last and self.act is not None:
            res = self.act(res)
        if self.softmax:
            res = F.softmax(res, dim=1)
        return res


class VAE_DNN(nn.Module):
    def __init__(self, dims, activation='sigmoid'):
        super(VAE_DNN, self).__init__()
        self.dim = dims[-1]
        # net
        self.net = ModuleList()
        layers = len(dims) - 1
        for i in range(layers - 1):
            self.net.append(nn.Linear(dims[i], dims[i + 1]))
        # activation function
        self.act = None
        if activation is None:
            pass
        elif activation == 'relu':
            self.act = nn.ReLU(True)
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'softplus':
            self.act = nn.Softplus()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'prelu':
            self.act = nn.PReLU()
        elif activation.startswith('leakyrelu'):
            self.act = nn.LeakyReLU(float(activation.split(":")[1]))
        self.net1 = nn.Linear(dims[-2], dims[-1])
        self.net2 = nn.Linear(dims[-2], dims[-1])

    def forward(self, x):
        res = x
        layer = len(self.net)
        for i in range(layer):
            res = self.net[i](res)
            res = self.act(res) if self.act is not None else res
        # mu, sigma = self.net1(res), self.net2(res)
        # sigma = F.softplus(sigma) + 1e-7  # make sigma always positive
        # return Independent(Normal(loc=mu, scale=sigma), 1), mu, sigma
        mu, logvar = self.net1(res), torch.sigmoid(self.net2(res))
        return Independent(Normal(loc=mu, scale=logvar.exp()), 1), mu, logvar


class Attention(nn.Module):
    def __init__(self, dims, tau=10.0):
        super(Attention, self).__init__()
        self.tau = tau
        self.output_layer = DNN(dims, activation=None, last=False)
        self.weights = None

    def forward(self, xs=[], calculate=True):
        x = torch.cat(xs, dim=1)
        act = self.output_layer(x)
        e = F.softmax(torch.sigmoid(act) / self.tau, dim=1)
        self.weights = torch.mean(e, dim=0)
        if calculate:
            output = torch.sum(self.weights[None, None, :] * torch.stack(xs, dim=-1), dim=-1)
            return self.weights, output
        else:
            return self.weights


class MI(nn.Module):
    def __init__(self, dims, activation='relu'):
        super(MI, self).__init__()
        net = []
        layers = len(dims) - 1
        net.append(nn.Linear(dims[0] * 2, dims[1]))
        for i in range(1, layers):
            if activation is None:
                pass
            elif activation == 'relu':
                self.act = nn.ReLU(True)
            elif activation == 'sigmoid':
                net.append(nn.Sigmoid())
            elif activation == 'softplus':
                net.append(nn.Softplus())
            elif activation == 'tanh':
                net.append(nn.Tanh())
            elif activation == 'prelu':
                self.act = nn.PReLU()
            elif activation.startswith('leakyrelu'):
                self.act = nn.LeakyReLU(float(activation.split(":")[1]))
            net.append(nn.Linear(dims[i], dims[i + 1]))
        self.net = nn.Sequential(*net)

    def forward(self, x1, x2):
        pos = self.net(torch.cat([x1, x2], 1))  # Positive Samples
        # neg = self.net(torch.cat([torch.roll(x1, 1, 0), x2], 1))  # Negative Samples
        neg = self.net(torch.cat([x1, x2[torch.randperm(x2.shape[0])]], 1))  # Negative Samples
        return -F.softplus(-pos).mean() - F.softplus(neg).mean(), pos.mean() - neg.exp().mean() + 1


class MIB_decoder(nn.Module):
    def __init__(self, dims, activation='relu'):
        super(MIB_decoder, self).__init__()
        self.mi = MI(dims, activation=activation)

    def forward(self, p1, p2, beta=1):
        # Read the two views v1 and v2
        # Sample from the posteriors with reparametrization
        z1 = p1.rsample()
        z2 = p2.rsample()
        # Mutual information estimation
        mi_gradient, mi_estimation = self.mi(z1, z2)
        mi_gradient = mi_gradient.mean()
        mi_estimation = mi_estimation.mean()
        # Symmetrized Kullback-Leibler divergence
        kl_1_2 = p1.log_prob(z1) - p2.log_prob(z1)
        kl_2_1 = p2.log_prob(z2) - p1.log_prob(z2)
        skl = (kl_1_2 + kl_2_1).mean() / 2.
        # Computing the loss function
        loss = - mi_gradient + beta * skl
        return loss


class SelfRepresentation(nn.Module):
    def __init__(self, N):
        super(SelfRepresentation, self).__init__()
        self.C = nn.Parameter(1.0e-4 * torch.ones(N, N))

    def forward(self, x):
        C = (self.C + self.C.T) / 2
        coef = C - torch.diag(torch.diag(C))
        output = torch.matmul(coef, x)

        coef_12 = torch.norm(coef, p=1, dim=0)
        loss_12norm = torch.norm(coef_12, p=2)
        loss_c = (torch.cdist(x, x) * coef).mean()
        loss_re = F.mse_loss(x, output)
        return output, 0.0001 * loss_12norm + loss_c + loss_re

    def get_C(self):
        C = (self.C + self.C.T) / 2
        coef = C - torch.diag(torch.diag(C))
        return coef

    def get_L(self, K, C=None, knn=None, set_one=True, d=6, alpha=8):
        if C is None:
            C = self.C
        C = C.detach().cpu().numpy()
        import numpy as np
        from sklearn.preprocessing import normalize
        from scipy.sparse.linalg import svds
        # C: coefficient matrix, K: number of clusters, knn: k-nearest neighbor, d: dimension of each subspace
        C = 0.5*(C + C.T)
        r = d*K + 1
        U, S, _ = svds(C,r,v0 = np.ones(C.shape[0]))
        U = U[:,::-1]
        S = np.sqrt(S[::-1])
        S = np.diag(S)
        U = U.dot(S)
        U = normalize(U, norm='l2', axis = 1)
        Z = U.dot(U.T)
        Z = Z * (Z>0)
        L = np.abs(Z ** alpha)
        L = L/L.max()
        L = 0.5 * (L + L.T)
        # L = L - np.diag(np.diag(L))
        if knn is not None:
            L = k_neighbor(L, knn, set_one)
        return L


class SelfWeight(nn.Module):
    def __init__(self, N):
        super(SelfWeight, self).__init__()
        self.W = nn.Parameter(torch.ones(N) / N)

    def forward(self, x):
        W = F.softmax(self.W, dim=0)
        res = None
        for i in range(W.shape[0]):
            res = x[i] * W[i] if res is None else res + x[i] * W[i]
        return self.W, res


# Graph Conv Network
class GCN(nn.Module):
    def __init__(self, dims, activation='relu', last=True):
        super(GCN, self).__init__()
        # net
        self.net = ModuleList()
        layers = len(dims) - 1
        for i in range(layers):
            self.net.append(nn.Linear(dims[i], dims[i + 1]))
        # activation function
        self.act = None
        if activation == 'relu':
            self.act = nn.ReLU(True)
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'softplus':
            self.act = nn.Softplus()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'leakyrelu':
            self.act = nn.LeakyReLU()
        elif activation == 'prelu':
            self.act = nn.PReLU()
        # activation or not after the last layer
        self.last = last

    def forward(self, x, W):
        res = x
        D_ = (1 / torch.sqrt(W.sum(dim=0))).diag()
        DWD = D_.matmul(W).matmul(D_)
        layer = len(self.net)
        for i in range(layer - 1):
            res = torch.matmul(DWD, self.net[i](res))
            res = self.act(res) if self.act is not None else res
        res = torch.matmul(DWD, self.net[-1](res))
        if self.act is not None and self.last:
            res = self.act(res)
        return res


# Graph Conv Variational Network
class GCVN(nn.Module):
    def __init__(self, dims, activation='relu'):
        super(GCVN, self).__init__()
        # net
        self.net = ModuleList()
        layers = len(dims) - 1
        for i in range(layers - 1):
            self.net.append(nn.Linear(dims[i], dims[i + 1]))
        # activation function
        self.act = None
        if activation == 'relu':
            self.act = nn.ReLU(True)
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'softplus':
            self.act = nn.Softplus()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'leakyrelu':
            self.act = nn.LeakyReLU()
        elif activation == 'prelu':
            self.act = nn.PReLU()
        self.net1 = nn.Linear(dims[-2], dims[-1])
        self.net2 = nn.Linear(dims[-2], dims[-1])

    def forward(self, x, W):
        res = x
        D_ = 1 / torch.sqrt(W.sum(dim=0))
        DWD = D_.matmul(W).matmul(D_)
        layer = len(self.net)
        for i in range(layer):
            res = torch.matmul(DWD, self.net[i](res))
            res = self.act(res) if self.act is not None else res
        mu, logvar = self.net1(torch.matmul(DWD, res)), torch.sigmoid(self.net2(torch.matmul(DWD, res)))
        return Independent(Normal(loc=mu, scale=logvar.exp()), 1), mu, logvar


# Graph Attention Encoder
class GAE(nn.Module):
    def __init__(self, dims, activation='sigmoid'):
        super(GAE, self).__init__()
        self.attention1 = nn.ParameterList()
        self.attention2 = nn.ParameterList()
        # encoder
        self.en = ModuleList()
        self.de = ModuleList()
        layers = len(dims) - 1
        for i in range(layers):
            self.attention1.append(nn.Parameter(torch.ones(dims[i + 1], 1) * 1e-4))
            self.attention2.append(nn.Parameter(torch.ones(dims[i + 1], 1) * 1e-4))
            self.en.append(nn.Linear(dims[i], dims[i + 1]))
            self.de.append(nn.Linear(dims[layers - i], dims[layers - i - 1]))
        # activation function
        self.act = None
        if activation == 'relu':
            self.act = nn.ReLU(True)
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'softplus':
            self.act = nn.Softplus()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'leakyrelu':
            self.act = nn.LeakyReLU()
        elif activation == 'prelu':
            self.act = nn.PReLU()

    def forward(self, x, A):
        res = x
        attentions = []
        layer = len(self.en)
        for i in range(layer):
            Wh = self.en[i](res)
            Wh1 = torch.matmul(Wh, self.attention1[i])
            Wh2 = torch.matmul(Wh, self.attention2[i])
            e = Wh1 + Wh2.T
            attention = A * e
            attention = self.act(attention)
            attention = F.softmax(attention, dim=1)
            attentions.append(attention)
            res = torch.matmul(attention, Wh)
        hidden = res
        for i in range(layer):
            Wh = self.de[i](res)
            res = torch.matmul(attentions[layer-i-1], Wh)
        G = torch.matmul(hidden, hidden.T)
        return hidden, res, G

    def encode(self, x, A):
        res = x
        attentions = []
        layer = len(self.en)
        for i in range(layer):
            Wh = self.en[i](res)
            Wh1 = torch.matmul(Wh, self.attention1[i])
            Wh2 = torch.matmul(Wh, self.attention2[i])
            e = Wh1 + Wh2.T
            attention = A * e
            attention = self.act(attention)
            attention = F.softmax(attention, dim=1)
            attentions.append(attention)
            res = torch.matmul(attention, Wh)
        G = torch.matmul(res, res.T)
        return res, G, attentions

    def decode(self, hidden, attentions):
        res = hidden
        layer = len(self.en)
        for i in range(layer):
            Wh = self.de[i](res)
            res = torch.matmul(attentions[layer-i-1], Wh)
        return res


def kl_Loss(mu, logvar):
    """KL Divergence.
    @param mu: torch.Tensor
               mean of latent distribution
    @param logvar: torch.Tensor
                   log-variance of latent distribution
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def orthogonal(x):
    xx = torch.mm(x.t(), x)
    xx = xx + torch.eye(x.shape[1]).to(x.device)
    L = torch.cholesky(xx)
    orth = torch.mm(x, torch.pinverse(L).t() * math.sqrt(x.shape[0]))
    return orth


def knn(X, k=15, sigma=None):
    n = X.shape[0]
    D = torch.cdist(X, X)
    Ds, ind = torch.sort(D, dim=0, descending=False)
    indx = ind[0:k, :].T.reshape(-1)
    dist = Ds[0:k, :].T.reshape(-1)
    if sigma is None: sigma = torch.mean(dist) ** 2
    w = torch.exp(- dist ** 2 / sigma)
    W = torch.zeros((n, n)).to(w.device)
    for i in range(n):
        e = indx[i*k: k+i*k]
        W[e, i] = torch.sqrt(torch.sum(w[e]))
    W = (W + W.T) / 2
    return W


def laplacian(W):
    D = W.sum(dim=1)
    D = torch.maximum(D, torch.zeros_like(D) + 1e-6)
    D_ = (1 / (D.sqrt())).diag()
    L = D.diag() - W
    Ln = torch.matmul(torch.matmul(D_, L), D_)
    return Ln


def hyperLaplacian(W, k=15):
    De = (W > 0).sum(dim=0)
    Dv = (W * W).sum(dim=1)
    Dv_ = (1 / torch.sqrt(Dv)).diag()
    De_ = (1 / De).diag()
    L = torch.eye(W.shape[0]).to(W.device) - Dv_.matmul(W).matmul(De_).matmul(W.T).matmul(Dv_)
    return L


def k_neighbor(W, k=10, set_one=True):
    import numpy as np
    ind = np.argsort(W)
    N = W.shape[0]
    ind0 = ind[:, :N-k]
    ind1 = ind[:, N-k:]
    for i in range(N):
        W[i, ind0[i]] = 0.
        if set_one:
            W[i, ind1[i]] = 1.
    W = (W + W.T) / 2
    return W