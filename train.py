import argparse
from sklearn import metrics
from Model import Network
from utils_func import load_data, generateAdj, CLDataGenerater, Cell_level_Loss, Cluster_level_Loss
import torch
from torch_sparse import SparseTensor
from torch import optim
from sklearn.metrics import adjusted_rand_score, silhouette_score
import pickle as pkl
from torch.utils.data import DataLoader

def setup_args():
    args = argparse.ArgumentParser()

    args.add_argument("-dataset_name", "--dataset_name",
                      type=str, default='Romanov')
    args.add_argument("-n_clusters", "--n_clusters", type=int, default=7)

    args.add_argument("-pca_dim", "--pca_dim", type=int, default=512)
    args.add_argument("-hidden", "--hidden", type=int, default=128)
    args.add_argument("-output", "--output", type=int, default=64)
    args.add_argument("-head", "--head", type=int, default=4)
    args.add_argument("-k", "--k", type=int, default=20)
    args.add_argument("-epoch", "--epoch", type=int, default=100)
    args.add_argument("-dropout", "--dropout", type=float, default=0.4)
    args.add_argument("-cellT", "--cellT", type=float, default=0.5)
    args.add_argument("-clusterT", "--clusterT", type=float, default=1.5)
    args.add_argument("-lr", "--lr", type=float, default=3e-4)

    args.add_argument("-device", "--device", type=str, default='cpu')
    args.add_argument("-is_val", "--is_val", type=bool, default=True)

    return args



def train(network, train_loader, cell_labels, loss_device, args, A):

    network = network.to(loss_device)

    optimizer = optim.Adam(network.parameters(), lr=args.lr)
    criterion_cell = Cell_level_Loss(args.cellT).to(loss_device)
    criterion_cluster = Cluster_level_Loss(n_clusters,
                                           args.clusterT).to(loss_device)

    for i in range(args.epoch):
        loss_epoch = 0
        network.train()
        for step, (x_i, x_j) in enumerate(train_loader):
            optimizer.zero_grad()

            x_i = x_i.to(loss_device)
            x_j = x_j.to(loss_device)
            z_i, z_j, c_i, c_j = network(x_i, x_j)
            loss_cell = criterion_cell(z_i, x_i, x_j, A)
            loss_cluster = criterion_cluster(c_i, c_j)
            loss = loss_cell + loss_cluster
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
        # if i % 50 == 0:
        if True:
            network.eval()
            with torch.no_grad():
                predict, hidden = network.forward_cluster()
                predict = predict.detach().cpu().numpy()

            ari = adjusted_rand_score(cell_labels, predict)
            NMI = metrics.normalized_mutual_info_score(cell_labels, predict)
            print(f"Iter: {i}, labels: [{len(set(predict))}/{args.n_clusters}], loss_cluster: {loss_cluster.item()}, ARI: {ari}, NMI: {NMI}")

def val(network, train_loader, cell_labels, loss_device, args):
    # network = network.to(loss_device)

    network.eval()
    with torch.no_grad():
        predict, hidden = network.forward_cluster()
        predict = predict.detach().cpu().numpy()

    ari = adjusted_rand_score(cell_labels, predict)
    NMI = metrics.normalized_mutual_info_score(cell_labels, predict)
    print('ARI: {:.8f}, NMI: {:.8f}'.format(ari, NMI))

if __name__ == '__main__':

    args = setup_args().parse_args()
    n_clusters = args.n_clusters


    print('dataset: ' + args.dataset_name)

    data_path = 'data/'+ args.dataset_name + '/data.tsv'
    data_pca = 'data/'+ args.dataset_name + '/data_pca'

    X = load_data(data_path=data_path, data_pca=data_pca, PCA_dim=args.pca_dim)
    X = torch.FloatTensor(X)
    adj = generateAdj(X, k=args.k)
    A = adj.to_dense().float()

    data_load = CLDataGenerater(A)
    train_loader = DataLoader(data_load, batch_size=data_load.__len__(), shuffle=False)

    with open('data/'+ args.dataset_name +'/label', 'rb') as label:
        cell_labels = pkl.load(label)

    loss_device = torch.device(args.device)
    X = X.to(loss_device)
    A = A.fill_diagonal_(1.0)
    adj = SparseTensor.from_dense(A).to(loss_device)

    network = Network(pca_dim=args.pca_dim,
                      hidden_dim=args.hidden,
                      feature_dim=args.output,
                      head=args.head,
                      dropout_rate=args.dropout,
                      class_num=args.n_clusters,
                      X=X, adj=adj)

    network.load_state_dict(torch.load('data/' + args.dataset_name + '/Romanov_model.pkl'))

    if args.is_val:
        val(network, train_loader, cell_labels, loss_device, args)
    else:
        train(network, train_loader, cell_labels, loss_device, args, A)




