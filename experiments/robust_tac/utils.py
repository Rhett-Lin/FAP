import csv
import os
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def count_parameters(*modules):
    trainable = 0
    total = 0
    for module in modules:
        for param in module.parameters():
            n = param.numel()
            total += n
            if param.requires_grad:
                trainable += n
    return trainable, total


def count_trainable_parameters(module):
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def count_parameters_from_iterable(parameters):
    return sum(param.numel() for param in parameters)


class AverageMeter:
    def __init__(self):
        self.values = []

    def update(self, value):
        self.values.append(float(value))

    @property
    def avg(self):
        if not self.values:
            return 0.0
        return float(sum(self.values) / len(self.values))


def write_summary_csv(path, row):
    ensure_dir(os.path.dirname(path))
    fieldnames = list(row.keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
