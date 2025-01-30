from utils import fine_tunning
from model import pre_trained_model
from dataset import proc_fablesDataset

fine_tunning(pre_trained_model, proc_fablesDataset)

