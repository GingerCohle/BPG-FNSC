#!/usr/bin/env python
import argparse
import csv
import logging
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from configs import Config
from eval import evaluate
from model import get_model
from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AutoSSVH checkpoint mAP metrics.")
    parser.add_argument("--config", required=True, help="Config file path.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pth path.")
    parser.add_argument("--gpu", default="0", help="GPU id.")
    parser.add_argument("--output", default=None, help="Optional CSV output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    device = "cuda:" + args.gpu

    logger = logging.getLogger("eval_checkpoint_maps")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.handlers = [handler]

    set_seed(cfg)
    torch.cuda.set_device(int(args.gpu))

    logger.info("config: %s", args.config)
    logger.info("checkpoint: %s", args.checkpoint)
    logger.info("gpu: %s", args.gpu)

    model = get_model(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    maps = evaluate(cfg, model, cfg.test_num_sample, logger, device, return_all=True)
    topks = [5, 20, 40, 60, 80, 100]

    print("dataset,bit,checkpoint," + ",".join("mAP@{}".format(k) for k in topks))
    print("{},{},{},".format(cfg.dataset, cfg.nbits, args.checkpoint) + ",".join("{:.8f}".format(v) for v in maps))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_header = not output.exists()
        with output.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["dataset", "bit", "config", "checkpoint"] + ["mAP@{}".format(k) for k in topks])
            writer.writerow([cfg.dataset, cfg.nbits, args.config, args.checkpoint] + maps)
        logger.info("wrote: %s", output)


if __name__ == "__main__":
    main()
