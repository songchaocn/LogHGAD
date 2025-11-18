from CONSTANTS import *


class Batch(object):
    def __init__(self, x, hyper_edge_index, edge_attr, y):
        self.x = x
        self.hyper_edge_index = hyper_edge_index
        self.edge_attr = edge_attr
        self.y = y


def batch_preprocess(batch):
    x = [item.x for item in batch]
    hyper_edge_index = torch.stack([item.hyper_edge_index for item in batch])
    edge_attr = [item.edge_attr for item in batch]
    y = [item.y for item in batch]
    return Batch(x, hyper_edge_index, edge_attr, y)


def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        datas = [data[i * batch_size + b] for b in range(cur_batch_size)]
        yield datas


def data_loader(data, batch_size, shuffle=True):
    batched_data = []
    if shuffle: np.random.shuffle(data)
    batched_data.extend(list(batch_slice(data, batch_size)))
    if shuffle: np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch
