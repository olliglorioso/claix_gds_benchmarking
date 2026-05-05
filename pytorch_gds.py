import torch 
from torch.utils.serialization import config as serialization_config

serialization_config.save.storage_alignment = 4096