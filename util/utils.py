import csv
import datetime
import json
import math
import random
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch


class SmoothedValue(object):
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = '{median:.4f} ({global_avg:.4f})'
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self):
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self):
        return self.total / max(self.count, 1)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(median=self.median, avg=self.avg, global_avg=self.global_avg, value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter='  '):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for name, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.meters[name].update(float(value))

    def __str__(self):
        return self.delimiter.join(f'{name}: {meter}' for name, meter in self.meters.items())

    def log_every(self, iterable, print_freq, header=''):
        start = end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        total = len(iterable)
        width = len(str(total))
        for i, obj in enumerate(iterable):
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == total - 1:
                eta = str(datetime.timedelta(seconds=int(iter_time.global_avg * (total - i - 1))))
                print(self.delimiter.join([header, f'[{i:{width}d}/{total}]', f'eta: {eta}', str(self),
                                           f'time: {iter_time}']), flush=True)
            end = time.time()
        total_time = str(datetime.timedelta(seconds=int(time.time() - start)))
        print(f'{header} Total time: {total_time}', flush=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def autocast(device, enabled=True):
    device = torch.device(device)
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=enabled and device.type == 'cuda')


def cosine_with_warmup(optimizer, warmup_steps, total_steps):
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(obj, torch.Tensor):
        return to_jsonable(obj.detach().cpu().numpy())
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), indent=2) + '\n')


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text())


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: to_jsonable(v) for k, v in row.items()})


def read_csv(path):
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))
