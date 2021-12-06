import logging
import numpy as np
import os
import pickle
import scipy.sparse as sp
import sys
import tensorflow as tf

from scipy.sparse import linalg


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False,
                 irregularity=None):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
        self.xs = xs
        self.ys = ys

        # TODO(piyush) remove
        self.keep = None
        if irregularity is not None:
            print(f"USING IRREGULAR PROB {irregularity['prob']}")

            if irregularity["keep_path"] is not None:
                print(f"LOADING MASK FROM PATH {irregularity['keep_path']}")
                import pickle
                with open(irregularity["keep_path"], "rb") as f:
                    self.keep = pickle.load(f)
            else:
                import torch
                self.keep = torch.rand(len(self.xs), 12) > irregularity["prob"]
                self.keep[:, 0] = True  # Never discard the first time step
                print(f"GENERATED MASK: {self.keep.shape}, {self.keep.float().mean()}% true")

            if irregularity["mode"] == "MOSTRECENT":
                print("USING MOSTRECENT IRREGULARITY")
                self.irreg_func = most_recent_irreg_func
            elif irregularity["mode"] == "ZERO":
                print("USING ZERO IRREGULARITY")
                self.irreg_func = zero_irreg_func
            elif irregularity["mode"] == "LINEAR":
                print("USING LINEAR IRREGULARITY")
                self.irreg_func = linear_irreg_func
            else:
                raise ValueError(f"Invalid irregularity mode: {irregularity['mode']}")

            self.labelmask = irregularity["labelmask"]
            self.scaler = irregularity["scaler"]
            if self.labelmask:
                print("USING LABEL MASKING")

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]

                # TODO(piyush) remove
                if self.keep is not None:
                    keep = self.keep[start_ind : end_ind, ...]
                    x_i = self.irreg_func(x_i, keep)

                    if self.labelmask:
                        # Make a copy to avoid making permanent changes to the data loader.
                        masked_y = np.empty_like(y_i)
                        masked_y[:] = y_i

                        masked_y[~keep] = self.scaler.transform(0)
                        y_i = masked_y

                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def add_simple_summary(writer, names, values, global_step):
    """
    Writes summary for a list of scalars.
    :param writer:
    :param names:
    :param values:
    :param global_step:
    :return:
    """
    for name, value in zip(names, values):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        writer.add_summary(summary, global_step)


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)


def config_logging(log_dir, log_filename='info.log', level=logging.INFO):
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Create the log directory if necessary.
    try:
        os.makedirs(log_dir)
    except OSError:
        pass
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level=level)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level=level)
    logging.basicConfig(handlers=[file_handler, console_handler], level=level)


def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger


def get_total_trainable_parameter_size():
    """
    Calculates the total number of trainable parameters in the current graph.
    :return:
    """
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        total_parameters += np.product([x.value for x in variable.get_shape()])
    return total_parameters


def load_dataset(dataset_dir, batch_size, test_batch_size=None, **kwargs):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])
    # TODO(piyush) remove
    irregularity, val_irregularity = None, None
    if "IRREGULARITY" in os.environ:
        irregularity = {
            "mode": os.environ.get("IRREGULARITY", None),
            "prob": float(os.environ.get("PROB", 0.0)),
            "keep_path": os.environ.get("KEEP_PATH", None),
            "labelmask": "LABELMASK" in os.environ,
            "scaler": scaler,
        }
        print("USING IRREGULARITY:")
        print(irregularity)
    if "VAL_IRREGULARITY" in os.environ:
        val_irregularity = {
            "mode": os.environ.get("IRREGULARITY", None),
            "prob": float(os.environ.get("PROB", 0.0)),
            "keep_path": None,
            "labelmask": False,
            "scaler": None,
        }
    data['train_loader'] = DataLoader(
        data['x_train'], data['y_train'], batch_size, shuffle=True, irregularity=irregularity)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size, shuffle=False,
                                    irregularity=val_irregularity)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size, shuffle=False,
                                     irregularity=val_irregularity)
    data['scaler'] = scaler

    return data


def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


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


def most_recent_irreg_func(x, keep):
    # Make a copy to avoid making permanent changes to the data loader.
    irreg_x = np.empty_like(x)
    irreg_x[:] = x
    for i in range(x.shape[0]):
        for t in range(1, x.shape[1]):
            if not keep[i, t]:
                irreg_x[i, t, ...] = irreg_x[i, t - 1, ...]
    return irreg_x

def zero_irreg_func(x, keep, zero_val=0):
    # Make a copy to avoid making permanent changes to the data loader.
    irreg_x = np.empty_like(x)
    irreg_x[:] = x
    for i in range(x.shape[0]):
        for t in range(1, x.shape[1]):
            if not keep[i, t]:
                irreg_x[i, t, ...] = zero_val
    return irreg_x

def linear_irreg_func(x, keep):
    # Make a copy to avoid making permanent changes to the data loader.
    irreg_x = np.empty_like(x)
    irreg_x[:] = x
    for i in range(x.shape[0]):
        t = 1
        while t < x.shape[1]:
            if not keep[i, t]:
                start = t
                while t < x.shape[1] and not keep[i, t]:
                    t += 1
                end = t

                irreg_x[i, start : end, ...] = np.array([
                    [
                        np.interp(
                            x=range(start, end), xp=[start - 1, end],
                            fp=[irreg_x[i, start - 1, j1, j2],
                                irreg_x[i, end, j1, j2]])
                        for j2 in range(x.shape[3])
                    ]
                    for j1 in range(x.shape[2])
                ])
            t += 1
    return irreg_x
