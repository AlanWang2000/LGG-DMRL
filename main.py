import yaml

from model.ConstructW import ConstructW
from model.MvRL import MvRL
from util.readfile import readfile
from util.metric import cluster, classify



dataset = 'handwritten'
X, Y, V, N, clusters = readfile(dataset)

args = yaml.load(open('./configs/ConstructW.yml','r'), Loader=yaml.FullLoader)[dataset]
model = ConstructW(X, Y, args['dims'], args['act'])
W = model.train(args['lr'], args['epochs'], args['params'], args['batch_size'], log_epoch=10, log_show=True)
args = yaml.load(open('./configs/MvRL.yml','r'), Loader=yaml.FullLoader)[dataset]
model = MvRL(X, W, Y, args['dims'], args['act'])
S, H, Y = model.train(args['lr'], args['epochs'], args['params'], args['batch_size'], log_epoch=10, log_show=True)
cluster(clusters, H, Y, method='KMeans')
classify(H, Y)
