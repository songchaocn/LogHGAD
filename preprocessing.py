import re

from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig

from CONSTANTS import *
from util import logger

PreprocessingLogger = logger('preprocessing')


def BGL_preprocessing():
    structured_log_file = os.path.join(PROJECT_ROOT, 'datasets/BGL/BGL.log_structured.csv')
    if os.path.exists(structured_log_file):
        return

    PreprocessingLogger.info('BGL template csv does not exist.')

    config = TemplateMinerConfig()
    config.load(os.path.join(PROJECT_ROOT, f'conf/BGL.ini'))
    persistence_folder = os.path.join(PROJECT_ROOT, f'datasets/BGL/persistences',
                                      'drain_depth-' + str(config.drain_depth) \
                                      + '_st-' + str(config.drain_sim_th))
    persistence_file = os.path.join(persistence_folder, 'persistence')

    template_miner = TemplateMiner(config=config, persistence_handler=FilePersistence(persistence_file))
    template_miner.load_state()
    PreprocessingLogger.info('BGL template miner Loaded.')

    label_list, timestamp_list, date_list, node_list, datetime_list = [], [], [], [], []
    node_repeat_list, type_list, component_list, level_list, content_list = [], [], [], [], []
    event_id_list, event_list, parameters_list = [], [], []

    PreprocessingLogger.info('Starting to extract BGL template csv...')
    with open(os.path.join(PROJECT_ROOT, 'datasets/BGL/BGL.log'), 'r', encoding='utf-8') as reader:
        for line in tqdm(reader.readlines()):
            tokens = line.split()
            content = ' '.join(tokens[9:])
            label_list.append(tokens[0])
            timestamp_list.append(tokens[1])
            date_list.append(tokens[2])
            node_list.append(tokens[3])
            datetime_list.append(tokens[4])
            node_repeat_list.append(tokens[5])
            type_list.append(tokens[6])
            component_list.append(tokens[7])
            level_list.append(tokens[8])
            content_list.append(content)
            matched_cluster = template_miner.match(content)
            event_id_list.append(matched_cluster.cluster_id)
            event_list.append(matched_cluster.get_template())
            parameters_list.append(template_miner.get_parameter_list(matched_cluster.get_template(), content))

    parsed_log_df = pd.DataFrame({
        'Label': label_list,
        'Timestamp': timestamp_list,
        'Date': date_list,
        'Node': node_list,
        'DateTime': datetime_list,
        'NodeRepeat': node_repeat_list,
        'Type': type_list,
        'Component': component_list,
        'Level': level_list,
        'Content': content_list,
        'EventID': event_id_list,
        'Event': event_list,
        'Parameters': parameters_list
    })
    PreprocessingLogger.info('BGL template csv extracted.')

    parsed_log_df.to_csv(structured_log_file, index=False)
    PreprocessingLogger.info('BGL template csv saved.')


def Hadoop_preprocessing():
    structured_log_file = os.path.join(PROJECT_ROOT, 'datasets/Hadoop/Hadoop.log_structured.csv')
    if os.path.exists(structured_log_file):
        return

    PreprocessingLogger.info('Hadoop template csv does not exist.')

    config = TemplateMinerConfig()
    config.load(os.path.join(PROJECT_ROOT, f'conf/Hadoop.ini'))
    persistence_folder = os.path.join(PROJECT_ROOT, f'datasets/Hadoop/persistences',
                                      'drain_depth-' + str(config.drain_depth) \
                                      + '_st-' + str(config.drain_sim_th))
    persistence_file = os.path.join(persistence_folder, 'persistence')

    template_miner = TemplateMiner(config=config, persistence_handler=FilePersistence(persistence_file))
    template_miner.load_state()
    PreprocessingLogger.info('Hadoop template miner Loaded.')

    container_list, date_list, time_list, process_list = [], [], [], []
    component_list, level_list, content_list, event_id_list, event_list, parameters_list = [], [], [], [], [], []

    PreprocessingLogger.info('Starting to extract Hadoop template csv...')
    with open(os.path.join(PROJECT_ROOT, 'datasets/Hadoop/Hadoop.log'), 'r', encoding='utf-8') as reader:
        for line in tqdm(reader.readlines()):
            tokens = line.split()
            container_list.append(tokens[0])
            date_list.append(tokens[1])
            time_list.append(tokens[2])
            level_list.append(tokens[3])
            line = ' '.join(tokens[4:])
            process_list.append(re.search(r'\[(.*?)\]', line).group(1))
            tokens = re.sub(r'\[.*?\]', '', line, count=1).strip().split()
            component_list.append(tokens[0][:-1])
            content = ' '.join(tokens[1:])
            content_list.append(content)
            matched_cluster = template_miner.match(content)
            event_id_list.append(matched_cluster.cluster_id)
            event_list.append(matched_cluster.get_template())
            parameters_list.append(template_miner.get_parameter_list(matched_cluster.get_template(), content))

    parsed_log_df = pd.DataFrame({
        'Container': container_list,
        'Date': date_list,
        'Time': time_list,
        'Level': level_list,
        'Process': process_list,
        'Content': content_list,
        'Component': component_list,
        'EventID': event_id_list,
        'Event': event_list,
        'Parameters': parameters_list
    })
    PreprocessingLogger.info('Hadoop template csv extracted.')

    parsed_log_df.to_csv(structured_log_file, index=False)
    PreprocessingLogger.info('Hadoop template csv saved.')


def HDFS_preprocessing():
    structured_log_file = os.path.join(PROJECT_ROOT, 'datasets/HDFS/HDFS.log_structured.csv')
    if os.path.exists(structured_log_file):
        return

    PreprocessingLogger.info('HDFS template csv does not exist.')

    config = TemplateMinerConfig()
    config.load(os.path.join(PROJECT_ROOT, f'conf/HDFS.ini'))
    persistence_folder = os.path.join(PROJECT_ROOT, f'datasets/HDFS/persistences',
                                      'drain_depth-' + str(config.drain_depth) \
                                      + '_st-' + str(config.drain_sim_th))
    persistence_file = os.path.join(persistence_folder, 'persistence')

    template_miner = TemplateMiner(config=config, persistence_handler=FilePersistence(persistence_file))
    template_miner.load_state()
    PreprocessingLogger.info('HDFS template miner Loaded.')

    date_list, time_list, pid_list, level_list = [], [], [], []
    component_list, content_list, event_id_list, event_list, parameters_list = [], [], [], [], []
    blkid_list = []

    PreprocessingLogger.info('Starting to extract HDFS template csv...')
    with open(os.path.join(PROJECT_ROOT, 'datasets/HDFS/HDFS.log'), 'r', encoding='utf-8') as reader:
        blk_rex = re.compile(r'blk_[-]{0,1}[0-9]+')
        for line in tqdm(reader.readlines()):
            tokens = line.split()
            content = ' '.join(tokens[5:])
            date_list.append(tokens[0])
            time_list.append(tokens[1])
            pid_list.append(tokens[2])
            level_list.append(tokens[3])
            component_list.append(tokens[4][:-1])
            blkid_list.append(re.findall(blk_rex, content)[0].strip())
            content_list.append(content)
            matched_cluster = template_miner.match(content)
            event_id_list.append(matched_cluster.cluster_id)
            event_list.append(matched_cluster.get_template())
            parameters_list.append(template_miner.get_parameter_list(matched_cluster.get_template(), content))

    parsed_log_df = pd.DataFrame({
        'Date': date_list,
        'Time': time_list,
        'PID': pid_list,
        'Level': level_list,
        'Component': component_list,
        'BlkID': blkid_list,
        'Content': content_list,
        'EventID': event_id_list,
        'Event': event_list,
        'Parameters': parameters_list
    })
    PreprocessingLogger.info('HDFS template csv extracted.')

    parsed_log_df.to_csv(structured_log_file, index=False)
    PreprocessingLogger.info('HDFS template csv saved.')


def Spirit_preprocessing():
    structured_log_file = os.path.join(PROJECT_ROOT, 'datasets/Spirit/Spirit.log_structured.csv')
    if os.path.exists(structured_log_file):
        return

    PreprocessingLogger.info('Spirit template csv does not exist.')

    config = TemplateMinerConfig()
    config.load(os.path.join(PROJECT_ROOT, f'conf/Spirit.ini'))
    persistence_folder = os.path.join(PROJECT_ROOT, f'datasets/Spirit/persistences',
                                      'drain_depth-' + str(config.drain_depth) \
                                      + '_st-' + str(config.drain_sim_th))
    persistence_file = os.path.join(persistence_folder, 'persistence')

    template_miner = TemplateMiner(config=config, persistence_handler=FilePersistence(persistence_file))
    template_miner.load_state()
    PreprocessingLogger.info('Spirit template miner Loaded.')

    label_list, timestamp_list, date_list, user_list, month_list = [], [], [], [], []
    day_list, time_list, location_list, component_list, pid_list, content_list = [], [], [], [], [], []
    event_id_list, event_list, parameters_list = [], [], []

    PreprocessingLogger.info('Starting to extract Spirit template csv...')
    with open(os.path.join(PROJECT_ROOT, 'datasets/Spirit/Spirit.log'), 'r', encoding='utf-8') as reader:
        for line in tqdm(reader.readlines()):
            tokens = line.split()
            content = ' '.join(tokens[9:])
            label_list.append(tokens[0])
            timestamp_list.append(tokens[1])
            date_list.append(tokens[2])
            user_list.append(tokens[3])
            month_list.append(tokens[4])
            day_list.append(tokens[5])
            time_list.append(tokens[6])
            location_list.append(tokens[7])
            component_list.append(tokens[8].partition('[')[0])
            pid_list.append(tokens[8].partition('[')[2][:-2] if len(tokens[8].partition('[')) > 1 else '')
            content_list.append(content)
            matched_cluster = template_miner.match(content)
            event_id_list.append(matched_cluster.cluster_id)
            event_list.append(matched_cluster.get_template())
            parameters_list.append(template_miner.get_parameter_list(matched_cluster.get_template(), content))

    parsed_log_df = pd.DataFrame({
        'Label': label_list,
        'Timestamp': timestamp_list,
        'Date': date_list,
        'User': user_list,
        'Month': month_list,
        'Day': date_list,
        'Time': time_list,
        'Location': location_list,
        'Component': component_list,
        'PID': pid_list,
        'Content': content_list,
        'EventID': event_id_list,
        'Event': event_list,
        'Parameters': parameters_list
    })
    PreprocessingLogger.info('Spirit template csv extracted.')

    parsed_log_df.to_csv(structured_log_file, index=False)
    PreprocessingLogger.info('Spirit template csv saved.')


def Thunderbird_preprocessing():
    structured_log_file = os.path.join(PROJECT_ROOT, 'datasets/Thunderbird/Thunderbird.log_structured.csv')
    if os.path.exists(structured_log_file):
        return

    PreprocessingLogger.info('Thunderbird template csv does not exist.')
    config = TemplateMinerConfig()
    config.load(os.path.join(PROJECT_ROOT, f'conf/Thunderbird.ini'))
    persistence_folder = os.path.join(PROJECT_ROOT, f'datasets/Thunderbird/persistences',
                                      'drain_depth-' + str(config.drain_depth) \
                                      + '_st-' + str(config.drain_sim_th))
    persistence_file = os.path.join(persistence_folder, 'persistence')

    template_miner = TemplateMiner(config=config, persistence_handler=FilePersistence(persistence_file))
    template_miner.load_state()
    PreprocessingLogger.info('Thunderbird template miner Loaded.')

    label_list, timestamp_list, date_list, user_list, month_list = [], [], [], [], []
    day_list, time_list, location_list, component_list, pid_list, content_list = [], [], [], [], [], []
    event_id_list, event_list, parameters_list = [], [], []

    PreprocessingLogger.info('Starting to extract Thunderbird template csv...')
    with open(os.path.join(PROJECT_ROOT, 'datasets/Thunderbird/Thunderbird.log'), 'r', encoding='utf-8') as reader:
        for line in tqdm(reader.readlines()):
            tokens = line.split()
            content = ' '.join(tokens[9:])
            label_list.append(tokens[0])
            timestamp_list.append(tokens[1])
            date_list.append(tokens[2])
            user_list.append(tokens[3])
            month_list.append(tokens[4])
            day_list.append(tokens[5])
            time_list.append(tokens[6])
            location_list.append(tokens[7])
            component_list.append(tokens[8].partition('[')[0])
            pid_list.append(tokens[8].partition('[')[2][:-2] if len(tokens[8].partition('[')) > 1 else '')
            content_list.append(content)
            matched_cluster = template_miner.match(content)
            event_id_list.append(matched_cluster.cluster_id)
            event_list.append(matched_cluster.get_template())
            parameters_list.append(template_miner.get_parameter_list(matched_cluster.get_template(), content))
    parsed_log_df = pd.DataFrame({
        'Label': label_list,
        'Timestamp': timestamp_list,
        'Date': date_list,
        'User': user_list,
        'Month': month_list,
        'Day': date_list,
        'Time': time_list,
        'Location': location_list,
        'Component': component_list,
        'PID': pid_list,
        'Content': content_list,
        'EventID': event_id_list,
        'Event': event_list,
        'Parameters': parameters_list
    })
    PreprocessingLogger.info('Thunderbird template csv extracted.')
    parsed_log_df.to_csv(structured_log_file, index=False)
    PreprocessingLogger.info('Thunderbird template csv saved.')


dataset_preprocess = {
    'BGL': BGL_preprocessing,
    'Hadoop': Hadoop_preprocessing,
    'HDFS': HDFS_preprocessing,
    'Spirit': Spirit_preprocessing,
    'Thunderbird': Thunderbird_preprocessing
}

if __name__ == '__main__':
    HDFS_preprocessing()


