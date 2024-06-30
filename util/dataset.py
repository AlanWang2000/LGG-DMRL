from torch.utils.data import Dataset


class MultiDataset(Dataset):
    def __init__(self, x, gt=None):
        self.x = x
        self.gt = gt
        self.v = len(x)

    def __getitem__(self, item):
        xs = {}
        for i in range(self.v):
            xs[i] = self.x[i][item]
        res = [xs]
        if self.gt is not None:
            res.append(self.gt[item])
        res.append(item)
        return tuple(res)

    def __len__(self):
        return self.x[0].shape[0]
