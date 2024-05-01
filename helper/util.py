from __future__ import print_function

import json

import numpy as np
import torch
import torch.distributed as dist


def adjust_learning_rate_new(epoch, optimizer, LUT):
    """
    new learning rate schedule according to RotNet
    """
    lr = next((lr for (max_epoch, lr) in LUT if max_epoch > epoch), LUT[-1][1])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class EarlyStopping:
    def __init__(self, patience=1, min_delta=0.0, cumulative_delta=True, min_mode=False):
        """
        EarlyStopping Function class initializer

        @param patience: Number of calls to wait if no improvement and then indicate stopping condition.
        @param min_delta: A minimum change in score to potentially qualify as an improvement
        @param cumulative_delta: If True, min_delta defines an increase since the last patience reset, otherwise,
                                it defines an increase after the last call.
        @param min_mode: If True, the lower the score the better, otherwise the higher the score the better.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.cumulative_delta = cumulative_delta
        self.min_mode = min_mode
        self.score = -np.inf

        self.counter = 0

    def __call__(self, score):
        current_score = -score if self.min_mode else score
        if (current_score + self.min_delta) > self.score:   # Improvement detected -> Patience reset
            self.counter = 0
            self.score = current_score
            return False
        # No improvement
        self.counter += 1
        if not self.cumulative_delta:
            self.score = current_score
        if self.counter >= self.patience:
            return True
        return False


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'a') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)

def load_json_to_dict(json_path):
    """Loads json file to dict 

    Args:
        json_path: (string) path to json file
    """
    with open(json_path, 'r') as f:
        params = json.load(f)
    return params


def parser_config_save(args,PATH):
    import json
    with open(PATH+'/'+'config.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def reduce_tensor(tensor, world_size = 1, op='avg'):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if world_size > 1:
        rt = torch.true_divide(rt, world_size)
    return rt


if __name__ == '__main__':

    pass
