import pandas as pd

from sklearn.cluster import SpectralClustering, KMeans
import numpy as np
import os
import pickle

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


# def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
#     data = {}
#     for category in ['train', 'val', 'test']:
#         cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
#         data['x_' + category] = cat_data['x']
#         data['y_' + category] = cat_data['y']
#     scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
#     # Data format
#     for category in ['train', 'val', 'test']:
#         data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
#
#     data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
#     data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size)
#     data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size)
#     data['scaler'] = scaler
#     return data
#
#     if args.w_init == 'group':
#         city_loc = np.load(os.path.join(path, 'loc_filled.npy'), allow_pickle=True)
#         kmeans = KMeans(n_clusters=args.group_num, random_state=0).fit(city_loc)
#         group_list = kmeans.labels_.tolist()
#         w = np.random.randn(args.city_num, args.group_num)
#         w = w * 0.1
#         for i in range(len(group_list)):
#             w[i, group_list[i]] = 1.0
#         w = torch.FloatTensor(w).to(device, non_blocking=True)



# loc = pd.read_csv('./data/sensor_graph/graph_sensor_locations.csv')
# loc_np = loc.to_numpy()
# loc_np = loc_np[:, -2:]
# kmeans = SpectralClustering(n_clusters=10, random_state=0).fit(loc_np)
# group_list = kmeans.labels_.tolist()
#
# w = np.random.randn(207, 10)
# w = w * 0.1
# for i in range(len(group_list)):
#     w[i, group_list[i]] = 1.0

# np.save('./data/sensor_graph/assignment_matrix', w)
w = np.load('./data/sensor_graph/assignment_matrix.npy')
w_t = w.transpose()
_, _, adj = load_pickle('./data/sensor_graph/adj_mx.pkl')
adj_cluster = np.einsum('mn, nl->ml ', w_t, adj)
adj_cluster = np.einsum('mn, nk->mk ', adj_cluster, w)
# w = torch.FloatTensor(w).to(device, non_blocking=True)

np.save('/home/hjl/deep_learning_workspace/data_metrla/1212/adj_cluster', adj_cluster)
np.save('/home/hjl/deep_learning_workspace/data_metrla/1212/transmit', w)

dataset_dir = '/home/hjl/deep_learning_workspace/data_metrla/1212'
data = {}
train_cluster = {}
val_cluster = {}
test_cluster = {}


for category in ['train', 'val', 'test']:
    cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
    data['x_' + category] = cat_data['x']
    data['y_' + category] = cat_data['y']

    data['x_' + category] = np.einsum('btnc, nm ->btmc', data['x_' + category], w)
    data['y_' + category] = np.einsum('btnc, nm ->btmc', data['y_' + category], w)

    if category == 'train':
        train_cluster['x'] = data['x_' + category]
        train_cluster['y'] = data['y_' + category]
    elif category == 'val':
        val_cluster['x'] = data['x_' + category]
        val_cluster['y'] = data['y_' + category]
    else:
        test_cluster['x'] = data['x_' + category]
        test_cluster['y'] = data['y_' + category]

np.savez('/home/hjl/deep_learning_workspace/data_metrla/1212/train_cluster_x', train_cluster['x'])
np.savez('/home/hjl/deep_learning_workspace/data_metrla/1212/train_cluster_y', train_cluster['y'])
np.savez('/home/hjl/deep_learning_workspace/data_metrla/1212/val_cluster_x', val_cluster['x'])
np.savez('/home/hjl/deep_learning_workspace/data_metrla/1212/val_cluster_y', val_cluster['y'])
np.savez('/home/hjl/deep_learning_workspace/data_metrla/1212/test_cluster_x', test_cluster['x'])
np.savez('/home/hjl/deep_learning_workspace/data_metrla/1212/test_cluster_y', test_cluster['y'])
# train_c = np.load('/home/hjl/deep_learning_workspace/data_metrla/1212/train_cluster.npz', allow_pickle=True)
# val_c = np.load('/home/hjl/deep_learning_workspace/data_metrla/1212/val_cluster.npz')
# test_c = np.load('/home/hjl/deep_learning_workspace/data_metrla/1212/test_cluster.npz')
#
# train_cluster_c = train_c['arr_0']

data_c = {}
# for category in ['train', 'val', 'test']:
    # data_c['x_c_' + category] =
    # data_c['y_c_' + category] =