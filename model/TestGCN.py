import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.modules.container import ModuleList
from torch.autograd import Variable
from itertools import chain
import timeit

from util.dataset import MultiDataset
from util.base import DNN, VAE_DNN, GCN, MI, kl_Loss


class TestGCN:
    def __init__(self, X, W, gt, dims, activations, device='cuda:0'):
        self.device = device
        self.V = len(X)
        self.N = gt.shape[0]
        self.gt = torch.from_numpy(gt).view(self.N)
        self.clusters = len(torch.unique(self.gt))
        self.X = {}
        for i in range(self.V):
            self.X[i] = torch.from_numpy(X[i] * 1.0).type(torch.FloatTensor)
        self.W = torch.from_numpy(W).type(torch.FloatTensor).to(device)
        self.W = self.W + self.W.matmul(self.W)
        self.dataset = MultiDataset(self.X, self.gt)
        self.H = torch.normal(mean=torch.zeros([self.N, dims['h']]), std=0.01).to(device)
        self.en = ModuleList()
        self.gcn = ModuleList()
        self.de = ModuleList()
        self.dg = ModuleList()
        self.mi = MI(dims['mi']).to(device)
        for i in range(self.V):
            self.en.append(DNN([X[i].shape[1]]+dims['en'][i], activations['en'], last=False).to(device))
            self.gcn.append(GCN([X[i].shape[1]]+dims['gcn'][i], activations['gcn'], last=False).to(device))
            self.de.append(DNN(dims['de'][i]+[X[i].shape[1]], activations['de'], last=False).to(device))
            self.dg.append(VAE_DNN([dims['h']] + dims['dg'], activations['dg']).to(device))

    def train(self, lr, epochs, params, batch_size, log_show=True, log_epoch=1, log_batch=10):
        start = timeit.default_timer()
        dataloader = DataLoader(dataset=self.dataset, batch_size=batch_size, shuffle=False)

        op_pre = torch.optim.Adam(chain(self.en.parameters(), self.gcn.parameters(), self.de.parameters()), lr=lr['pre'])
        op_s = torch.optim.Adam(chain(self.en.parameters(), self.gcn.parameters(), self.de.parameters()), lr=lr['ae'])
        op_dg = torch.optim.Adam(chain(self.dg.parameters(), self.mi.parameters()), lr=lr['dg'])

        # pre-train
        for epoch in range(epochs['pre']):
            for batch_idx, (x, y, idx) in enumerate(dataloader):
                for i in range(self.V):
                    x[i] = x[i].to(self.device)
                op_pre.zero_grad()
                loss_ae = 0
                for i in range(self.V):
                    s1 = self.gcn[i](x[i], self.W)
                    s2 = self.en[i](x[i])
                    s = (s1 + s2) / 2
                    # s = s1
                    # s = s2
                    x_re = self.de[i](s)
                    loss_ae += F.mse_loss(x_re, x[i])
                total_loss = loss_ae
                total_loss.backward()
                op_pre.step()

                # log
                if log_show and  epoch % log_epoch == 0 and batch_idx % log_batch == 0:
                    output = 'Pre train Epoch : {:2.0f} -- Batch : {:2.0f}'.format(epoch, batch_idx)
                    output += '  ===> Total training loss = {:.4f} : loss_ae = {:.4f}'.format(total_loss, loss_ae)
                    print(output)

        # train
        for epoch in range(epochs['total']):
            for batch_idx, (x, y, idx) in enumerate(dataloader):
                for i in range(self.V):
                    x[i] = x[i].to(self.device)

                # step1: S
                h_batch = self.H[idx]
                op_s.zero_grad()
                loss_ae = 0
                loss_dg = 0
                loss_kl = 0
                for i in range(self.V):
                    s1 = self.gcn[i](x[i], self.W)
                    s2 = self.en[i](x[i])
                    p_h_dg, h_dg, logvar = self.dg[i](h_batch)
                    a1, a2 = torch.square(s1 - h_dg).sum(dim=1, keepdim=True).detach(), torch.square(s2 - h_dg).sum(dim=1, keepdim=True).detach()
                    # a1, a2 = F.mse_loss(s1, h_dg).detach(), F.mse_loss(s2, h_dg).detach()
                    s = (a2 * s1 + a1 * s2) / (a1 + a2)
                    # s = s1
                    # s = s2
                    x_re = self.de[i](s)
                    loss_ae += F.mse_loss(x_re, x[i])
                    loss_dg += F.mse_loss(s, h_dg.detach())
                    loss_kl += kl_Loss(h_dg, logvar)
                loss_kl = loss_kl.mean()
                total_loss = loss_ae + params['alpha'] * loss_dg + params['beta'] * loss_kl
                total_loss.backward()
                op_s.step()

                # step2: degradation net H
                h_batch = self.H[idx]
                op_dg.zero_grad()
                loss_dg = 0
                loss_kl = 0
                loss_mi = 0
                for i in range(self.V):
                    s1 = self.gcn[i](x[i], self.W)
                    s2 = self.en[i](x[i])
                    p_h_dg, h_dg, logvar = self.dg[i](h_batch)
                    a1, a2 = torch.square(s1 - h_dg).sum(dim=1, keepdim=True).detach(), torch.square(s2 - h_dg).sum(dim=1, keepdim=True).detach()
                    # a1, a2 = F.mse_loss(s1, h_dg).detach(), F.mse_loss(s2, h_dg).detach()
                    s = (a2 * s1 + a1 * s2) / (a1 + a2)
                    # s = s1
                    # s = s2
                    loss_dg += F.mse_loss(h_dg, s.detach())
                    loss_kl += kl_Loss(h_dg, logvar).mean()
                    for j in range(i+1, self.V):
                        p_h_dg_, h_dg_, logvar_ = self.dg[j](h_batch)
                        mi_gradient, _ = self.mi(h_dg, h_dg_)
                        loss_mi += - mi_gradient.mean()
                loss_kl = loss_kl.mean()
                total_loss = params['alpha'] * loss_dg + params['beta'] * loss_kl + params['gamma'] * loss_mi
                total_loss.backward()
                op_dg.step()

                # step3: H
                h_batch = self.H[idx]
                h = Variable(h_batch, requires_grad=True)
                op_h = torch.optim.Adam([h], lr=lr['h'])
                op_h.zero_grad()
                loss_dg = 0
                loss_kl = 0
                loss_mi = 0
                for i in range(self.V):
                    s1 = self.gcn[i](x[i], self.W)
                    s2 = self.en[i](x[i])
                    p_h_dg, h_dg, logvar = self.dg[i](h)
                    a1, a2 = torch.square(s1 - h_dg).sum(dim=1, keepdim=True).detach(), torch.square(s2 - h_dg).sum(dim=1, keepdim=True).detach()
                    # a1, a2 = F.mse_loss(s1, h_dg).detach(), F.mse_loss(s2, h_dg).detach()
                    s = (a2 * s1 + a1 * s2) / (a1 + a2)
                    # s = s1
                    # s = s2
                    loss_dg += F.mse_loss(h_dg, s.detach())
                    loss_kl += kl_Loss(h_dg, logvar)
                    for j in range(i+1, self.V):
                        p_h_dg_, h_dg_, logvar_ = self.dg[j](h_batch)
                        mi_gradient, _ = self.mi(h_dg, h_dg_)
                        loss_mi += - mi_gradient.mean()
                loss_kl = loss_kl.mean()
                total_loss = params['alpha'] * loss_dg + params['beta'] * loss_kl + params['gamma'] * loss_mi
                total_loss.backward()
                op_h.step()
                self.H[idx] = h.data

                # log
                if log_show and epoch % log_epoch == 0 and batch_idx % log_batch == 0:
                    loss_ae = 0
                    loss_dg = 0
                    loss_kl = 0
                    loss_mi = 0
                    h_batch = self.H[idx]
                    for i in range(self.V):
                        s1 = self.gcn[i](x[i], self.W)
                        s2 = self.en[i](x[i])
                        p_h_dg, h_dg, logvar = self.dg[i](h_batch)
                        a1, a2 = torch.square(s1 - h_dg).sum(dim=1, keepdim=True).detach(), torch.square(s2 - h_dg).sum(dim=1, keepdim=True).detach()
                        # a1, a2 = F.mse_loss(s1, h_dg).detach(), F.mse_loss(s2, h_dg).detach()
                        s = (a2 * s1 + a1 * s2) / (a1 + a2)
                        # s = s1
                        # s = s2
                        x_re = self.de[i](s)
                        loss_ae += F.mse_loss(x_re, x[i]).detach().cpu()
                        loss_dg += F.mse_loss(h_dg, s).detach().cpu()
                        loss_kl = kl_Loss(h_dg, logvar)
                        for j in range(i+1, self.V):
                            p_h_dg_, h_dg_, logvar_ = self.dg[j](h_batch)
                            mi_gradient, _ = self.mi(h_dg, h_dg_)
                            loss_mi += - mi_gradient.mean()
                    loss_kl = loss_kl.mean().detach().cpu()
                    total_loss = loss_ae + params['alpha'] * loss_dg + params['beta'] * loss_kl + params['gamma'] * loss_mi
                    output = 'Epoch : {:2.0f} -- Batch : {:2.0f}'.format(epoch, batch_idx)
                    output += '  ===> Total training loss = {:.4f} : loss_ae = {:.4f}, loss_dg = {:.4f}, loss_kl = {:.4f}, loss_mi = {:.4f}'\
                            .format(total_loss, loss_ae, loss_dg, loss_kl, loss_mi)
                    print(output)
            
        S = {}
        for batch_idx, (x, y, idx) in enumerate(dataloader):
            h_batch = self.H[idx]
            for i in range(self.V):
                x[i] = x[i].to(self.device)
                s1 = self.gcn[i](x[i], self.W)
                s2 = self.en[i](x[i])
                p_h_dg, h_dg, logvar = self.dg[i](h_batch)
                a1, a2 = torch.square(s1 - h_dg).sum(dim=1, keepdim=True).detach(), torch.square(s2 - h_dg).sum(dim=1, keepdim=True).detach()
                S[i] = ((a2 * s1 + a1 * s2) / (a1 + a2)).detach().cpu().numpy()


        self.H = self.H.cpu().numpy()
        self.gt = self.gt.cpu().numpy()
        elapsed = (timeit.default_timer() - start)
        print("Time used: {}, model ran on: {}".format(elapsed, self.device))
        return S, self.H, self.gt
