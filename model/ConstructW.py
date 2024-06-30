import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.modules.container import ModuleList
from itertools import chain
import timeit

from util.dataset import MultiDataset
from util.base import DNN, SelfRepresentation


class ConstructW:
    def __init__(self, X, gt, dims, activations, device='cuda:0'):
        self.device = device
        self.V = len(X)
        self.N = gt.shape[0]
        self.gt = torch.from_numpy(gt).view(self.N)
        self.clusters = len(torch.unique(self.gt))
        self.X = {}
        for i in range(self.V):
            self.X[i] = torch.from_numpy(X[i] * 1.0).type(torch.FloatTensor)
        self.dataset = MultiDataset(self.X, self.gt)
        self.en = ModuleList()
        self.de = ModuleList()
        self.sr = SelfRepresentation(self.N).to(device)
        for i in range(self.V):
            dim_en = [X[i].shape[1]] + dims['ae'][i]
            dim_de = dim_en.copy()
            dim_de.reverse()
            self.en.append(DNN(dim_en, activations['ae'], last=False).to(device))
            self.de.append(DNN(dim_de, activations['ae'], last=False).to(device))

    def train(self, lr, epochs, params, batch_size, log_show=True, log_epoch=1, log_batch=10):
        start = timeit.default_timer()
        dataloader = DataLoader(dataset=self.dataset, batch_size=batch_size, shuffle=False)
        op = torch.optim.Adam(chain(self.en.parameters(), self.de.parameters(), self.sr.parameters()), lr=lr['ae'])

        for epoch in range(epochs['total']):
            for batch_idx, (x, y, idx) in enumerate(dataloader):
                for i in range(self.V):
                    x[i] = x[i].to(self.device)

                op.zero_grad()
                loss_ae = 0
                loss_sr = 0
                s = {}
                s_ = {}
                for i in range(self.V):
                    s_[i] = self.en[i](x[i])
                    s[i], loss_sr_ = self.sr(s_[i])
                    x_re = self.de[i](s[i])
                    loss_sr += loss_sr_
                    loss_ae += F.mse_loss(x_re, x[i])
                total_loss = loss_ae + params['alpha'] * loss_sr
                total_loss.backward()
                op.step()

                # log
                if log_show and epoch % log_epoch == 0 and batch_idx % log_batch == 0:
                    output = 'ConstructW train Epoch : {:2.0f} -- Batch : {:2.0f}'.format(epoch, batch_idx)
                    output += '  ===> Total training loss = {:.4f} : loss_ae = {:.4f}, loss_sr = {:.4f}'.format(total_loss, loss_ae, loss_sr)
                    print(output)
        

        L = self.sr.get_L(self.clusters, knn=params['knn'], set_one=params['set_one'])
        elapsed = (timeit.default_timer() - start)
        print("Time used: {}, model ran on: {}".format(elapsed, self.device))
        return L