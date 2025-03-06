import os
import sys
import math
import pprint

import torch
import torch_geometric.data

from torchvision import datasets
from torchdrug import core, models
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import util
from s3f import dataset, model, task, gvp


def train_and_validate(cfg, solver):
    if cfg.train.num_epoch == 0:
        return

    step = math.ceil(cfg.train.num_epoch / 50)
    best_result = float("-inf")
    best_epoch = -1

    for i in range(0, cfg.train.num_epoch, step):
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i)
        solver.train(**kwargs)
        solver.save("model_epoch_%d.pth" % solver.epoch)
        metric = solver.evaluate("valid")
        result = metric[cfg.metric]
        if result > best_result:
            best_result = result
            best_epoch = solver.epoch

    solver.load("model_epoch_%d.pth" % best_epoch)
    return solver


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + comm.get_rank())

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver = util.build_solver(cfg, dataset)

    train_and_validate(cfg, solver)