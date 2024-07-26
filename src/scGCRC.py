import torch
import scGCRC_Utils as utils
from scGCRC_Model import CLNetwork

if __name__ == '__main__':

    device = torch.device('cpu')
    dataset_name = 'Klein'

    # Step1: reading dataset
    cell_matrix, cells, genes = utils.load_data('dataset/' + dataset_name + '/' + dataset_name + '_top2000.csv')
    cell_labels, n_cluster = utils.load_label('dataset/' + dataset_name + '/label')

    # Step2: get graph neibours
    pca_path = 'dataset/' + dataset_name + '/pca_data'
    adjs4GAT_mask, adjs4CL = utils.generateNeibour(cell_matrix, pca_path=pca_path)

    # Step3: training model
    X = torch.FloatTensor(utils.normalization(cell_matrix)).to(device)
    model = CLNetwork(X.shape[-1], 256, 32, n_cluster, adjs4GAT_mask, 4)
    ## 3.1 loading trained model
    model.load_state_dict(torch.load('dataset/' + dataset_name + '/' + dataset_name + '.pkl'))
    model = model.to(device)
    ## 3.2 or training
    # model = utils.training(X, model, adjs4CL, device, cell_labels)

    # Step4: predict
    model.eval()
    with torch.no_grad():
        _, hidden = model.forward_cluster(X)
        hidden = hidden.detach().cpu().numpy()

    predict = utils.LeidenClustering(hidden, resolution=0.01)


    # Step5: evaluation
    print('nclusters:{}, predict clusters:{}'.format(n_cluster, len(set(predict))))
    print('ARI: {:.3f}, NMI: {:.3f}'.format(utils.getARI(cell_labels, predict), utils.getNMI(cell_labels, predict)))

