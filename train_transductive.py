import logging
from absl import app
from absl import flags

from torch.nn.functional import cosine_similarity
from torch.optim import AdamW
from tqdm import tqdm

from GCH.data import *
from GCH.GCH import *
from GCH.linear_eval_ppi import *
from GCH.logistic_regression_eval import *
from GCH.models import *
from GCH.predictors import *
from GCH.scheduler import *
from GCH.transforms import *
from GCH.utils import *
from torch_geometric.utils import sort_edge_index, degree, add_remaining_self_loops, remove_self_loops, get_laplacian, to_undirected, to_dense_adj, to_networkx  # 导入torch_geometric库中的一些实用工具
import copy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from torch_geometric.transforms import GDC
import numpy

import scipy.sparse as sp
import torch
import os
import torch.nn
from torch_geometric.transforms import GDC  # 导入torch_geometric库中的GDC图变换

'''形参
python3 train_transductive.py
--flagfile=config/amazon-photos.cfg\
--logdir=./runs/amazon-photos-256\
--predictor_hidden_size=256
python3 linear_eval_transductive.py --flagfile=config-eval/amazon-photos.cfg
'''
log = logging.getLogger(__name__)
FLAGS = flags.FLAGS
flags.DEFINE_integer('model_seed', None, 'Random seed used for model initialization and training.')
flags.DEFINE_integer('data_seed', 1, 'Random seed used to generate train/val/test split.')
flags.DEFINE_integer('num_eval_splits', 3, 'Number of different train/test splits the model will be evaluated over.')

# Dataset：定义了数据集相关的命令行参数
flags.DEFINE_enum('dataset', 'cora',
                  ['amazon-computers', 'amazon-photos', 'coauthor-cs', 'coauthor-physics', 'wiki-cs', 'cora', 'citeseer', 'pubmed'],
                  'Which graph dataset to use.')
flags.DEFINE_string('dataset_dir', './data', 'Where the dataset resides.')

# Architecture：定义了模型架构相关的命令行参数
flags.DEFINE_multi_integer('graph_encoder_layer', None, 'Conv layer sizes.')
flags.DEFINE_integer('predictor_hidden_size', 512, 'Hidden size of projector.')

# Training hyperparameters：定义了训练超参数相关的命令行参数
flags.DEFINE_integer('epochs', 10000, 'The number of training epochs.')
flags.DEFINE_float('lr', 1e-5, 'The learning rate for model training.')
flags.DEFINE_float('weight_decay', 1e-5, 'The value of the weight decay for training.')
flags.DEFINE_float('mm', 0.99, 'The momentum for moving average.')
flags.DEFINE_integer('lr_warmup_epochs', 1000, 'Warmup period for learning rate.')

# Logging and checkpoint：定义了日志和检查点相关的命令行参数
flags.DEFINE_string('logdir', None, 'Where the checkpoint and logs are stored.')

flags.DEFINE_integer('log_steps', 10, 'Log information at every log_steps.')

# Evaluation
flags.DEFINE_integer('eval_epochs', 5, 'Evaluate every eval_epochs.')
flags.DEFINE_string('drop_scheme', 'degree', 'method')
flags.DEFINE_float('drop_edge_rate_1', 0.2, 'sbl')
flags.DEFINE_float('drop_edge_rate_2', 0.4, 'sbl')
flags.DEFINE_float('drop_f_rate_1', 0.1, 'sbl')
flags.DEFINE_float('drop_f_rate_2', 0.1, 'sbl')
flags.DEFINE_float('lambd', 1e-3, 'lamd1')
flags.DEFINE_float('theta', 0.4, 'sbl')
flags.DEFINE_float('tau', 0.7, 'sbl')
flags.DEFINE_integer('k', 4, 'nei')
flags.DEFINE_integer('augk', 4, 'nei')
flags.DEFINE_float('alpha', 0.05, 'nei')
flags.DEFINE_integer('avg_degree', 4, 'nei')
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # 设置可见的CUDA设备
flags.DEFINE_float('gamma', 0.005, 'gamma')
flags.DEFINE_float('ratio', 0.5, 'ratio')
flags.DEFINE_float('threshold', 0.001, 'threshold')


def main(argv):
    # use CUDA_VISIBLE_DEVICES to select gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log.info('Using {} for training.'.format(device))


    # set random seed 设置随机种子，以确保实验的可重复性。
    if FLAGS.model_seed is not None:
        log.info('Random seed set to {}.'.format(FLAGS.model_seed))
        set_random_seeds(random_seed=FLAGS.model_seed)

    os.makedirs(FLAGS.logdir, exist_ok=True)
    with open(os.path.join(FLAGS.logdir, 'config.cfg'), "w") as file:
        file.write(FLAGS.flags_into_string())

# load data
    if FLAGS.dataset != 'wiki-cs':
        dataset = get_dataset(FLAGS.dataset_dir, FLAGS.dataset)
        num_eval_splits = FLAGS.num_eval_splits
    else:
        dataset, train_masks, val_masks, test_masks = get_wiki_cs(FLAGS.dataset_dir)
        num_eval_splits = train_masks.shape[1]

    data = dataset[0]
    zm = dataset[0]
    log.info('Dataset {}, {}.'.format(dataset.__class__.__name__, data))
    data = data.to(device)
    adj = data.edge_index


    # 创建SVD图
    knn_graph = data.edge_index.to('cpu').numpy()
    data.edge_index = data.edge_index.to('cpu').numpy()
    # 2E->NN
    NN = sp.coo_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index[0, :], data.edge_index[1, :])),
                       shape=(data.x.shape[0], data.x.shape[0]),
                       dtype=np.float32)
    svd = scipy_sparse_mat_to_torch_sparse_tensor(NN).coalesce().cuda(torch.device(device))
    svd_q = 5
    svd_u, s, svd_v = torch.svd_lowrank(svd, q=svd_q)
    s = torch.diag(s)
    SVD1 = torch.mm(svd_u, s)
    SVD = torch.mm(SVD1, svd_v.T)

    print('threshold',FLAGS.threshold)
    # 获取非零元素的坐标
    nonzero_coords = torch.nonzero(torch.abs(SVD) > FLAGS.threshold)
    # 构造COO稀疏矩阵
    sparse_matrix = sp.coo_matrix((SVD[nonzero_coords[:, 0], nonzero_coords[:, 1]].cpu().numpy(),
                                   (nonzero_coords[:, 0].cpu().numpy(), nonzero_coords[:, 1].cpu().numpy())),
                                  shape=SVD.shape)
    # 转换为PyTorch稀疏张量
    sparse_tensor = torch.sparse.FloatTensor(torch.LongTensor([sparse_matrix.row, sparse_matrix.col]),
                                             torch.FloatTensor(sparse_matrix.data))

    sparse_matrix = sparse_tensor.coalesce()._indices()

    data.edge_index = sparse_matrix
    data = data.to(device)

    log.info('Dataset {},{}.'.format('SVD                         ', data.edge_index.size(1)))

    knn_graph = knngraph(data,FLAGS.augk)
    log.info('Dataset {}, {}.'.format('KNN                           ', knn_graph.size(1)))

    data.edge_index = globlegraph(data.x, data.edge_index, adj, knn_graph)
    log.info('Dataset {}, {}.'.format('add                          ',
                                      data.edge_index.size(1) - zm.edge_index.size(1)))


    input_size, representation_size = data.x.size(1), FLAGS.graph_encoder_layer[-1]
    encoder = GCN([input_size] + FLAGS.graph_encoder_layer, batchnorm=True)
    predictor = MLP_Predictor(representation_size, representation_size,
                              hidden_size=FLAGS.predictor_hidden_size)

    model = BGRL(encoder, predictor, 0.7).to(device)
    model = model.to(device)
    # 下面就可以正常使用了

    # optimizer
    optimizer = AdamW(model.trainable_parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    # scheduler
    lr_scheduler = CosineDecayScheduler(FLAGS.lr, FLAGS.lr_warmup_epochs, FLAGS.epochs)
    mm_scheduler = CosineDecayScheduler(1 - FLAGS.mm, 0, FLAGS.epochs)


    layout = {'accuracy': {'accuracy/test': ['Multiline', [f'accuracy/test_{i}' for i in range(num_eval_splits)]]}}

    log.info('Dataset {}, {}.'.format(dataset.__class__.__name__, data))
    global drop_weights # FLAGS.drop_scheme='degree'
    if FLAGS.drop_scheme == 'degree':
        drop_weights = degree_drop_weights(data.edge_index).to(device)
    elif FLAGS.drop_scheme == 'pr':
        drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200).to(device)
    elif FLAGS.drop_scheme == 'evc':
        drop_weights = evc_drop_weights(data).to(device)
    else:
        drop_weights = None

    # FLAGS.drop_scheme='degree'
    if FLAGS.drop_scheme == 'degree':
        edge_index_ = to_undirected(data.edge_index)
        node_deg = degree(edge_index_[1])
        if FLAGS.dataset == 'wiki-cs':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_deg).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
    elif FLAGS.drop_scheme == 'pr':
        node_pr = compute_pr(data.edge_index)
        if FLAGS.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_pr).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_pr).to(device)
    elif FLAGS.drop_scheme == 'evc':
        node_evc = eigenvector_centrality(data)
        if FLAGS.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_evc).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_evc).to(device)
    else:
        feature_weights = torch.ones((data.x.size(1),)).to(device)


# 定义训练和评估函数
    def train(step):
        model.train()
        # update learning rate
        lr = lr_scheduler.get(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # update momentum
        mm = 1 - mm_scheduler.get(step)

        # forward
        optimizer.zero_grad()


        def drop_edge1():
            # global drop_weights

            if FLAGS.drop_scheme == 'uniform':
                return dropout_adj(data.edge_index, p=FLAGS.drop_edge_rate_1)[0]
            elif FLAGS.drop_scheme in ['degree', 'evc', 'pr']:
                return drop_edge_weighted(data.edge_index, drop_weights,  p=FLAGS.drop_edge_rate_1,
                                          threshold=0.7)
            else:
                raise Exception(f'undefined drop scheme: {FLAGS.drop_scheme}')

        def drop_edge2():
            # global drop_weights

            if FLAGS.drop_scheme == 'uniform':# FLAGS.drop_scheme=degree
                return dropout_adj(data.edge_index, p=FLAGS.drop_edge_rate_2)[0]
            elif FLAGS.drop_scheme in ['degree', 'evc', 'pr']:
                return drop_edge_weighted(data.edge_index, drop_weights,  p=FLAGS.drop_edge_rate_2,
                                          threshold=0.7)
            else:
                raise Exception(f'undefined drop scheme: {FLAGS.drop_scheme}')

        edge_index_1 = dropout_adj(data.edge_index, p=FLAGS.drop_edge_rate_1)[0]
        edge_index_2 = drop_edge2()

        x_2 = drop_feature(data.x, FLAGS.drop_f_rate_2)
        if FLAGS.drop_scheme in ['pr', 'degree', 'evc']:
            x_1 = drop_feature_weighted_2(data.x, feature_weights, FLAGS.drop_f_rate_1)

        data1 = copy.deepcopy(data)
        data2 = copy.deepcopy(data)

        data1.x = x_1
        data2.x = x_2
        #adj
        data1.edge_index = edge_index_1
        data2.edge_index = edge_index_2
        q1, y2, ind1= model.forward(data1, data2, FLAGS.k)
        q2, y1, ind2= model.forward(data2, data1, FLAGS.k)

        loss2 = model.loss(q1, y2.detach(),) + model.loss(q2, y1.detach())
        loss3 = model.loss_fn(q1[ind1[0]], y2[ind1[1]].detach()).mean()+model.loss_fn(q2[ind2[0]], y1[ind2[1]].detach()).mean()
        loss = FLAGS.theta * loss2 + (1 - FLAGS.theta) * loss3

        loss.backward()
        # update online network
        optimizer.step()
        # update target network
        model.update_target_network(mm)

        def eval(epoch):
        # make temporary copy of encoder
        tmp_encoder = copy.deepcopy(model.online_encoder).eval()
        representations, labels = compute_representations(tmp_encoder, dataset, device)
        z = representations
        @torch.no_grad()
        def plot_points(colors, z):
            model.eval()
            z = z
            z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
            y = data.y.cpu().numpy()

            plt.figure(figsize=(50, 50), dpi=1000)
            for i in range(dataset.num_classes):
                plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
            plt.axis('off')
            plt.show()

        colors = ['#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535', '#ffd700', '#e60000']
        plot_points(colors, z)

        if FLAGS.dataset != 'wiki-cs':
            scores = fit_logistic_regression(representations.cpu().numpy(), labels.cpu().numpy(),FLAGS.dataset,
                                             data_random_seed=FLAGS.data_seed, repeat=FLAGS.num_eval_splits)# num_eval_splits=3=repeat
        else:
            scores = fit_logistic_regression_preset_splits(representations.cpu().numpy(), labels.cpu().numpy(),
                                                           train_masks, val_masks, test_masks,FLAGS.dataset)


    f = open("result_" + FLAGS.dataset + ".txt", "a")
    f.write(str("-----------------------------------------") + "\n")

    for epoch in tqdm(range(1, FLAGS.epochs + 1)):
        train(epoch-1)
        if epoch % FLAGS.eval_epochs == 0:# eval_epochs=1000
            eval(epoch)

    torch.save({'model': model.online_encoder.state_dict()}, os.path.join(FLAGS.logdir, 'online.pt'))


if __name__ == "__main__":
    log.info('PyTorch version: %s' % torch.__version__)
    app.run(main)
