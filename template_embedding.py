from typing import Optional, List, Union
from drain3.template_miner_config import TemplateMinerConfig
from CONSTANTS import *
from util import logger

TemplateEmbeddingLogger = logger('template_embedding')


def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


def split(sentence):
    regex = re.compile('[^a-zA-Z]')
    tokens = regex.sub(' ', sentence).split()
    splitted_tokens = []
    for token in tokens:
        splitted_tokens.extend(camel_case_split(token))

    return splitted_tokens


def load_vectors():
    fin = io.open(os.path.join(PROJECT_ROOT, 'datasets/wiki-news-300d-1M.vec'), 'r', encoding='utf-8', newline='\n',
                  errors='ignore')
    n, d = map(int, fin.readline().split())
    word2vector = {}
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        word2vector[tokens[0]] = list(map(float, tokens[1:]))
    return n, d, word2vector


class TemplateEmbedding:
    def __init__(self, dataset):
        self.dim = None
        self.dataset = dataset
        self.embedding_dict = {}
        self.embedding = None
        self.load()

    def load(self):
        embedding_file = os.path.join(PROJECT_ROOT, f'datasets/{self.dataset}/{self.dataset}_templates.vec')
        if not os.path.exists(embedding_file):
            n, d, word2vector = load_vectors()
            self.dim = d

            TemplateEmbeddingLogger.info(f'{self.dataset} template embedding file does not exist, start to generate.')
            config = TemplateMinerConfig()
            config.load(os.path.join(PROJECT_ROOT, f'conf/{self.dataset}.ini'))
            template_file = os.path.join(PROJECT_ROOT, f'datasets/{self.dataset}/persistences',
                                         'drain_depth-' + str(config.drain_depth) \
                                         + '_st-' + str(config.drain_sim_th), 'templates.txt')
            with open(template_file, 'r') as file:
                templates = file.readlines()
                oov_list = set()
                for index, template in tqdm(enumerate(templates)):
                    template = template.split(':')[2]

                    template_tokens = split(template)
                    tokens_embedding_list = []
                    for token in template_tokens:
                        if token in word2vector:
                            tokens_embedding_list.append(torch.tensor(word2vector[token], dtype=torch.float))
                        else:
                            oov_list.add(token)
                            word2vector[token] = torch.rand(d).tolist()
                            tokens_embedding_list.append(torch.tensor(word2vector[token], dtype=torch.float))
                    if len(tokens_embedding_list) == 0:
                        tokens_embedding_list.append(torch.rand(d))
                    tokens_embedding = torch.vstack(tokens_embedding_list)
                    self.embedding_dict[index] = tokens_embedding.mean(dim=0)
                TemplateEmbeddingLogger.info(f'OOV count: {len(oov_list)},OOV : {oov_list}')

            with open(embedding_file, 'w') as f:
                for event_id, embedding in self.embedding_dict.items():
                    embedding_values = embedding.tolist()
                    line = f"{event_id} {' '.join(map(str, embedding_values))}\n"
                    f.write(line)
                TemplateEmbeddingLogger.info('Template embedding file saved to %s successfully' % embedding_file)
        else:
            with open(embedding_file, 'r') as file:
                for line in file.readlines():
                    tokens = line.strip().split(' ')
                    self.dim = len(tokens) - 1
                    self.embedding_dict[int(tokens[0])] = torch.tensor(list(map(float, tokens[1:])), dtype=torch.float)
                TemplateEmbeddingLogger.info('Template embedding file loaded from %s successfully' % embedding_file)
        self._embedding()

    def _embedding(self):
        TemplateEmbeddingLogger.info("Start to create embedding")
        num_embeddings = max(self.embedding_dict.keys()) + 1
        embedding_dim = self.dim
        embedding_matrix = torch.zeros(num_embeddings, embedding_dim)
        for key, tensor in self.embedding_dict.items():
            embedding_matrix[key] = tensor

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        TemplateEmbeddingLogger.info("Embedding created successfully")

    def get_embeddings(self, input_indices: Union[torch.Tensor, List]) -> torch.Tensor:
        if isinstance(input_indices, list):
            return torch.stack([self.embedding(indices) for indices in input_indices])

        return self.embedding(input_indices)

    def get_embedding_by_id(self, event_id: int) -> torch.Tensor:
        return self.embedding_dict[event_id]


if __name__ == '__main__':
    embedding = TemplateEmbedding(dataset='BGL')
