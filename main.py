import argparse
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import train
from src.utils import get_data


def build_parser():
    parser = argparse.ArgumentParser(description="MulT sentiment analysis")
    parser.add_argument("-f", default="", type=str)
    parser.add_argument(
        "--preset",
        type=str,
        default="",
        choices=["", "mosi_paper"],
        help="named experiment preset",
    )

    parser.add_argument("--model", type=str, default="MulT", help="name of the model to use")
    parser.add_argument("--vonly", action="store_true", help="use the crossmodal fusion into v")
    parser.add_argument("--aonly", action="store_true", help="use the crossmodal fusion into a")
    parser.add_argument("--lonly", action="store_true", help="use the crossmodal fusion into l")
    parser.add_argument("--aligned", action="store_true", help="consider aligned experiment or not")
    parser.add_argument("--dataset", type=str, default="mosei_senti", help="dataset to use")
    parser.add_argument("--data_path", type=str, default="data", help="dataset directory")

    parser.add_argument("--attn_dropout", type=float, default=0.1, help="attention dropout")
    parser.add_argument("--attn_dropout_a", type=float, default=0.0, help="audio attention dropout")
    parser.add_argument("--attn_dropout_v", type=float, default=0.0, help="visual attention dropout")
    parser.add_argument("--relu_dropout", type=float, default=0.1, help="relu dropout")
    parser.add_argument("--embed_dropout", type=float, default=0.25, help="embedding dropout")
    parser.add_argument("--res_dropout", type=float, default=0.1, help="residual block dropout")
    parser.add_argument("--out_dropout", type=float, default=0.0, help="output layer dropout")

    parser.add_argument("--nlevels", type=int, default=5, help="number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=5, help="number of attention heads")
    parser.add_argument("--proj_dim", type=int, default=30, help="shared projection dimension")
    parser.add_argument("--kernel_l", type=int, default=1, help="temporal conv kernel for text")
    parser.add_argument("--kernel_a", type=int, default=1, help="temporal conv kernel for audio")
    parser.add_argument("--kernel_v", type=int, default=1, help="temporal conv kernel for vision")
    parser.add_argument("--attn_mask", action="store_false", help="use attention mask for Transformer")

    parser.add_argument("--batch_size", type=int, default=24, metavar="N", help="batch size")
    parser.add_argument("--clip", type=float, default=0.8, help="gradient clip value")
    parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
    parser.add_argument("--optim", type=str, default="Adam", help="optimizer to use")
    parser.add_argument("--num_epochs", type=int, default=40, help="number of epochs")
    parser.add_argument("--when", type=int, default=20, help="when to decay learning rate")
    parser.add_argument("--batch_chunk", type=int, default=1, help="number of chunks per batch")
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="optimizer step interval for gradient accumulation",
    )

    parser.add_argument("--log_interval", type=int, default=30, help="frequency of result logging")
    parser.add_argument("--seed", type=int, default=1111, help="random seed")
    parser.add_argument("--no_cuda", action="store_true", help="do not use cuda")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="device selection override",
    )
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader worker count")
    parser.add_argument("--no_amp", action="store_true", help="disable automatic mixed precision")
    parser.add_argument("--no_prompt", action="store_true", help="disable end-of-run prompt")
    parser.add_argument("--name", type=str, default="mult", help="name of the trial")
    return parser


def _cli_flags():
    flags = set()
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            flags.add(arg.split("=", 1)[0])
    return flags


def _kernel_is_valid(kernel_size):
    return kernel_size > 0 and kernel_size % 2 == 1


def apply_preset(args, passed_flags):
    if args.preset != "mosi_paper":
        return args

    preset_values = {
        "dataset": "mosi",
        "batch_size": 128,
        "lr": 1e-3,
        "num_epochs": 100,
        "nlevels": 4,
        "num_heads": 10,
        "proj_dim": 40,
        "kernel_l": 1,
        "kernel_a": 3,
        "kernel_v": 3,
        "embed_dropout": 0.2,
        "attn_dropout": 0.2,
        "attn_dropout_a": 0.2,
        "attn_dropout_v": 0.2,
        "relu_dropout": 0.2,
        "res_dropout": 0.2,
        "out_dropout": 0.1,
        "clip": 0.8,
        "seed": 1111,
        "name": "mosi_paper",
    }

    for key, value in preset_values.items():
        if f"--{key}" not in passed_flags:
            setattr(args, key, value)

    if "--aligned" not in passed_flags:
        args.aligned = False
    return args


def resolve_device(args):
    if args.device == "cpu" or args.no_cuda:
        return torch.device("cpu")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")


def seed_everything(seed, use_cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_run_config(args, dataset, device):
    print("=" * 50)
    print("Preset:", args.preset or "custom")
    print("Dataset:", dataset)
    print("Aligned:", args.aligned)
    print("Device:", device)
    print("AMP:", args.use_amp)
    print("Batch size:", args.batch_size)
    print("Grad accumulation:", args.grad_accum_steps)
    print("Effective batch size:", args.batch_size * args.grad_accum_steps)
    print("Proj dim:", args.proj_dim)
    print("Kernels (l/a/v):", (args.kernel_l, args.kernel_a, args.kernel_v))
    print("Heads / layers:", (args.num_heads, args.nlevels))
    print("Epochs / lr:", (args.num_epochs, args.lr))
    print("Run name:", args.name)
    print("=" * 50)


parser = build_parser()
args = parser.parse_args()
args = apply_preset(args, _cli_flags())

if args.grad_accum_steps < 1:
    raise ValueError("--grad_accum_steps must be >= 1")
for kernel_name in ("kernel_l", "kernel_a", "kernel_v"):
    if not _kernel_is_valid(getattr(args, kernel_name)):
        raise ValueError(f"--{kernel_name} must be a positive odd integer")

device = resolve_device(args)
use_cuda = device.type == "cuda"
dataset = args.dataset.strip().lower()
valid_partial_mode = args.lonly + args.vonly + args.aonly

if valid_partial_mode == 0:
    args.lonly = args.vonly = args.aonly = True
elif valid_partial_mode != 1:
    raise ValueError("You can only choose one of {l/v/a}only.")

seed_everything(args.seed, use_cuda)

output_dim_dict = {"mosi": 1, "mosei_senti": 1, "iemocap": 8}
criterion_dict = {"iemocap": "CrossEntropyLoss"}

os.makedirs(args.data_path, exist_ok=True)
os.makedirs("pre_trained_models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

print("Start loading the data....")
train_data = get_data(args, dataset, "train")
valid_data = get_data(args, dataset, "valid")
test_data = get_data(args, dataset, "test")

loader_kwargs = {"num_workers": args.num_workers, "pin_memory": use_cuda}
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

print("Finish loading the data....")
if not args.aligned:
    print("### Note: You are running in unaligned mode.")

hyp_params = args
hyp_params.device = device
hyp_params.use_cuda = use_cuda
hyp_params.use_amp = use_cuda and not args.no_amp
hyp_params.dataset = dataset
hyp_params.layers = args.nlevels
hyp_params.when = args.when
hyp_params.batch_chunk = args.batch_chunk
hyp_params.n_train = len(train_data)
hyp_params.n_valid = len(valid_data)
hyp_params.n_test = len(test_data)
hyp_params.model = args.model.strip().upper()
hyp_params.output_dim = output_dim_dict.get(dataset, 1)
hyp_params.criterion = criterion_dict.get(dataset, "L1Loss")
hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = train_data.get_dim()
hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = train_data.get_seq_len()
hyp_params.effective_batch_size = args.batch_size * args.grad_accum_steps

print_run_config(hyp_params, dataset, device)

if __name__ == "__main__":
    train.initiate(hyp_params, train_loader, valid_loader, test_loader)
