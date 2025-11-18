from CONSTANTS import *


def logger(name) -> Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

    file_handler = logging.FileHandler(os.path.join(LOG_ROOT, f'{name}.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.info(
        f'Construct {name}Logger success, current working directory: {os.getcwd()}, logs will be written in {LOG_ROOT}')

    return logger


def hyperedge_to_incidence_matrix(hyperedge_index, num_nodes):
    hyperedge_index = torch.unique(hyperedge_index, dim=1)

    nodes = hyperedge_index[0]
    edges = hyperedge_index[1]  

    num_hyperedges = edges.max().item() + 1  

    indices = torch.stack([edges, nodes], dim=0)  
    values = torch.ones(indices.size(1), dtype=torch.float)  

    sparse_incidence = torch.sparse_coo_tensor(
        indices,
        values,
        size=(num_hyperedges, num_nodes)
    )

    incidence_matrix = sparse_incidence.to_dense()

    return incidence_matrix


def cut_by_82_with_shuffle(graphs):
    np.random.seed(seed)
    np.random.shuffle(graphs)
    train_split = math.ceil(0.8 * len(graphs))
    train = graphs[:train_split]
    test = graphs[train_split:]
    return train, None, test


def cut_by_613(graphs):
    random.seed(seed)
    random.shuffle(graphs)
    train_split = int(0.7 * len(graphs))
    train, test = graphs[:train_split], graphs[train_split:]
    random.shuffle(train)
    train_split = int(0.6 * len(graphs))
    dev = train[train_split:]
    train = train[:train_split]
    random.shuffle(train)
    
    return train, dev, test
