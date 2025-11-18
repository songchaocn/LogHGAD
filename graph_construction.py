import os.path
from CONSTANTS import *
import torch
from torch_geometric.data import Data
from template_embedding import TemplateEmbedding
from util import logger

GraphConstructionLogger = logger('graph_construction')


def parameters_set(df):
    param_set = set()
    for param_list in df['Parameters']:
        for param in param_list:
            if not param.isdigit():
                param_set.add(param)
    return param_set


def BGL_hyper_edge_extract(log_dataframe):
    hyper_edges = [[i, i + 1] for i in range(len(log_dataframe) - 1)]

    type_groups = log_dataframe.groupby('Type')
    hyper_edges.extend([group.index.tolist() for _, group in type_groups if len(group) > 1])
    component_groups = log_dataframe.groupby('Component')
    hyper_edges.extend([group.index.tolist() for _, group in component_groups if len(group) > 1])

    param_set = parameters_set(log_dataframe)
    param_groups = {}
    for param in param_set:
        param_groups[param] = log_dataframe[log_dataframe['Parameters'].apply(lambda x: param in x)]
    for param, group in param_groups.items():
        if len(group) > 1:
            hyper_edges.append(group.index.tolist())

    return hyper_edges


def Hadoop_hyper_edge_extract(log_dataframe):
    hyper_edges = [[i, i + 1] for i in range(len(log_dataframe) - 1)]

    process_groups = log_dataframe.groupby('Process')
    hyper_edges.extend([group.index.tolist() for _, group in process_groups if len(group) > 1])
    process_groups = log_dataframe.groupby('Component')
    hyper_edges.extend([group.index.tolist() for _, group in process_groups if len(group) > 1])

    param_set = parameters_set(log_dataframe)
    param_groups = {}
    for param in param_set:
        param_groups[param] = log_dataframe[log_dataframe['Parameters'].apply(lambda x: param in x)]
    for param, group in param_groups.items():
        if len(group) > 1:
            hyper_edges.append(group.index.tolist())

    return hyper_edges


def HDFS_hyper_edge_extract(log_dataframe):
    hyper_edges = [[i, i + 1] for i in range(len(log_dataframe) - 1)]

    pid_groups = log_dataframe.groupby('PID')
    hyper_edges.extend([group.index.tolist() for _, group in pid_groups if len(group) > 1])
    component_groups = log_dataframe.groupby('Component')
    hyper_edges.extend([group.index.tolist() for _, group in component_groups if len(group) > 1])

    param_set = parameters_set(log_dataframe)
    param_groups = {}
    for param in param_set:
        param_groups[param] = log_dataframe[log_dataframe['Parameters'].apply(lambda x: param in x)]
    for param, group in param_groups.items():
        if len(group) > 1:
            hyper_edges.append(group.index.tolist())

    return hyper_edges


def Spirit_hyper_edge_extract(log_dataframe):
    hyper_edges = [[i, i + 1] for i in range(len(log_dataframe) - 1)]

    pid_groups = log_dataframe.groupby('PID')
    hyper_edges.extend([group.index.tolist() for _, group in pid_groups if len(group) > 1])
    component_groups = log_dataframe.groupby('Component')
    hyper_edges.extend([group.index.tolist() for _, group in component_groups if len(group) > 1])
    location_groups = log_dataframe.groupby('Location')
    hyper_edges.extend([group.index.tolist() for _, group in location_groups if len(group) > 1])

    param_set = parameters_set(log_dataframe)
    param_groups = {}
    for param in param_set:
        param_groups[param] = log_dataframe[log_dataframe['Parameters'].apply(lambda x: param in x)]
    for param, group in param_groups.items():
        if len(group) > 1:
            hyper_edges.append(group.index.tolist())

    return hyper_edges


def Thunderbird_hyper_edge_extract(log_dataframe):
    hyper_edges = [[i, i + 1] for i in range(len(log_dataframe) - 1)]

    pid_groups = log_dataframe.groupby('PID')
    hyper_edges.extend([group.index.tolist() for _, group in pid_groups if len(group) > 1])
    component_groups = log_dataframe.groupby('Component')
    hyper_edges.extend([group.index.tolist() for _, group in component_groups if len(group) > 1])
    location_groups = log_dataframe.groupby('Location')
    hyper_edges.extend([group.index.tolist() for _, group in location_groups if len(group) > 1])

    param_set = parameters_set(log_dataframe)
    param_groups = {}
    for param in param_set:
        param_groups[param] = log_dataframe[log_dataframe['Parameters'].apply(lambda x: param in x)]
    for param, group in param_groups.items():
        if len(group) > 1:
            hyper_edges.append(group.index.tolist())

    return hyper_edges


hyper_edge_extract_funcs = {
    'BGL': BGL_hyper_edge_extract,
    'Hadoop': Hadoop_hyper_edge_extract,
    'HDFS': HDFS_hyper_edge_extract,
    'Spirit': Spirit_hyper_edge_extract,
    'Thunderbird': Thunderbird_hyper_edge_extract
}

normal_label_dict = {
    'BGL': {'INFO', 'WARNING'},
    'Hadoop': {'INFO', 'WARN'},
    'HDFS': {'error', 'exception'},
    'Spirit': {'-'},
    'Thunderbird': {'-'}
}

grouping_strategy = {
    'BGL': 'Node',
    'Hadoop': 'Container',
    'HDFS': 'BlkID',
    'Spirit': 'User',
    'Thunderbird': 'User'
}


class HypergraphBuilder:
    def __init__(self, node_event_list, hyper_edges, label, hyper_edge_features=None):
        self.node_event_list = node_event_list
        self.hyper_edges = hyper_edges
        self.label = label
        self.hyper_edge_features = hyper_edge_features
        if hyper_edge_features is not None:
            self.hyper_edge_features = torch.tensor(hyper_edge_features, dtype=torch.float)
        self.num_nodes = len(node_event_list)
        self.num_hyper_edges = len(self.hyper_edges)

    def build_hypergraph(self):
        node_indices = []
        hyper_edge_indices = []
        for hyper_edge_idx, hyper_edge in enumerate(self.hyper_edges):
            for node_idx in hyper_edge:
                node_indices.append(node_idx)
                hyper_edge_indices.append(hyper_edge_idx)
        hyper_edge_index = torch.tensor([node_indices, hyper_edge_indices], dtype=torch.long)

        x = torch.tensor(self.node_event_list, dtype=torch.long)
        x = x - 1

        hypergraph = Data(x=x, hyper_edge_index=hyper_edge_index, y=self.label,
                          edge_attr=self.hyper_edge_features)
        return hypergraph


class Log2Hypergraph:
    def __init__(self, dataset, window_size=50, step_size=50):
        self.dataset = dataset
        self.template_embedding = TemplateEmbedding(self.dataset)
        self.window_size = window_size
        self.step_size = step_size
        self.hypergraphs = None
        os.makedirs(os.path.join(PROJECT_ROOT, f'datasets/{self.dataset}/inputs/'), exist_ok=True)
        self.hypergraphs_file = os.path.join(PROJECT_ROOT, f'datasets/{self.dataset}/inputs/',
                                             f'window_size-{window_size}_step_size-{step_size}_hypergraphs.pkl')
        if os.path.exists(self.hypergraphs_file):
            GraphConstructionLogger.info(f"Loading {self.dataset} hypergraphs from {self.hypergraphs_file}")
            self.hypergraphs = load_hypergraphs(self.hypergraphs_file)
        else:
            GraphConstructionLogger.info(f"hypergraphs_file {self.hypergraphs_file} does not exist, start to generate.")
            GraphConstructionLogger.info(f"Starting to load {self.dataset} log dataframe")
            self.log_dataframe = self._load_log_dataframe()
            self.events = self.log_dataframe['EventID'].unique()
            GraphConstructionLogger.info(f"Load {self.dataset} log dataframe successfully")
            self.hypergraphs = self._convert_to_hypergraph()

    def get_hypergraphs(self):
        return self.hypergraphs

    def get_template_embedding(self):
        return self.template_embedding

    def _get_hypergraph_label(self, window_seq):
        normal_label_set = normal_label_dict[self.dataset]
        if self.dataset == 'BGL' or self.dataset == 'Hadoop':
            log_dataframe_label_set = set(window_seq['Level'])
        elif self.dataset == 'HDFS':
            log_messages = window_seq['Content']
            for normal_label in normal_label_dict[self.dataset]:
                if any(normal_label in message.split() for message in log_messages):
                    return 1
            return 0
        else:
            log_dataframe_label_set = set(window_seq['Label'])
        return 1 if len(log_dataframe_label_set - normal_label_set) > 0 else 0

    def _get_hypergraph_features(self, window_seq):
        node_event_list = window_seq['EventID'].tolist()
        hyper_edges = hyper_edge_extract_funcs[self.dataset](window_seq)
        label = self._get_hypergraph_label(window_seq)
        return node_event_list, hyper_edges, label

    def _convert_to_hypergraph(self):
        GraphConstructionLogger.info(f"Starting to convert {self.dataset} log to hypergraph")

        hypergraphs = []

        groups = self.log_dataframe.groupby(grouping_strategy[self.dataset])

        abnormal_count = 0
        for _, group_dataframe in tqdm(groups):
            if len(group_dataframe) == 1:
                continue
            group_dataframe = group_dataframe.reset_index(drop=True)
            end = len(group_dataframe)
            for start in range(0, end, self.step_size):
                if start + self.window_size > end:
                    window_seq = group_dataframe.iloc[start:].reset_index(drop=True)
                else:
                    window_seq = group_dataframe.iloc[start:start + self.window_size].reset_index(drop=True)
                if len(window_seq) < 2:
                    continue
                node_event_list, hyper_edges, label = self._get_hypergraph_features(window_seq)
                hb = HypergraphBuilder(node_event_list, hyper_edges, label)
                hypergraphs.append(hb.build_hypergraph())
                if label == 1:
                    abnormal_count += 1

        GraphConstructionLogger.info(
            f"Convert {self.dataset} log to hypergraph successfully, total hypergraphs: {len(hypergraphs)}, anomalies: {abnormal_count}")
        save_hypergraphs(hypergraphs, self.hypergraphs_file)
        del self.log_dataframe
        gc.collect()
        return hypergraphs

    def _load_log_dataframe(self):
        return pd.read_csv(os.path.join(PROJECT_ROOT, f'datasets/{self.dataset}/{self.dataset}.log_structured.csv'))


def save_hypergraphs(hypergraphs, file_path):
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(hypergraphs, file)
        print(f"Hypergraphs saved to {file_path}")
    except Exception as e:
        print(f"Error saving hypergraphs to {file_path}: {e}")


def load_hypergraphs(file_path):
    try:
        with open(file_path, 'rb') as file:
            loaded_list = pickle.load(file)
        print(f"Hypergraphs loaded from {file_path}")
        return loaded_list
    except Exception as e:
        print(f"Error loading hypergraphs from {file_path}: {e}")
        return None


if __name__ == "__main__":
    Log2Hypergraph("HDFS", window_size=50, step_size=50)
