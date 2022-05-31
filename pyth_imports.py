from sentence_transformers import SentenceTransformer
import os
import csv
import time

import pandas as pd
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.activation import Sigmoid
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from transformers import BertTokenizer, TFBertModel, BertModel
# import random


if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

results_loc = "RESULTS/results.txt"
EMBEDDING_SIZE = 1