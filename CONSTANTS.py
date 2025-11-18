import sys
from logging import Logger

sys.path.extend([".", ".."])
import os, gc, math, abc, pickle, argparse
import random
import time
import hashlib
import numpy as np
import pandas as pd
import torch
import datetime
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
from tqdm import *
import regex as re
import logging
import io
from collections import Counter
from torch.nn.parameter import Parameter
from multiprocessing import Manager, Pool
from copy import deepcopy
import matplotlib.pyplot as plt
print('Seeding everything...')
seed = 6
random.seed(seed)  
np.random.seed(seed)  
torch.manual_seed(seed)  
torch.cuda.manual_seed(seed)  
torch.cuda.manual_seed_all(seed)  
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print('Seeding Finished')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SESSION = hashlib.md5(
    time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time() + 8 * 60 * 60)).encode('utf-8')).hexdigest()
SESSION = 'SESSION_' + SESSION


def get_project_root():
    current_abspath = os.path.abspath(__file__)
    while True:
        current_dir = os.path.split(current_abspath)[1]
        if current_dir == 'LogHGAD':
            project_root = current_abspath
            break
        else:
            current_abspath = os.path.dirname(current_abspath)
    return project_root


def get_logs_root():
    project_root = get_project_root()
    log_file_root = os.path.join(project_root, 'logs')
    if not os.path.exists(log_file_root):
        os.makedirs(log_file_root)
    return log_file_root


LOG_ROOT = get_logs_root()
PROJECT_ROOT = get_project_root()

