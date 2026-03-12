import os

import torch

from src.dataset import Multimodal_Datasets


def torch_load_compat(path, map_location=None):
    kwargs = {}
    if map_location is not None:
        kwargs["map_location"] = map_location
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def get_data(args, dataset, split="train"):
    alignment = "a" if args.aligned else "na"
    data_path = os.path.join(args.data_path, dataset) + f"_{split}_{alignment}.dt"
    if not os.path.exists(data_path):
        print(f"  - Creating new {split} data")
        data = Multimodal_Datasets(args.data_path, dataset, split, args.aligned)
        torch.save(data, data_path)
    else:
        print(f"  - Found cached {split} data")
        data = torch_load_compat(data_path, map_location="cpu")
    return data


def save_load_name(args, name=""):
    if args.aligned:
        name = name if len(name) > 0 else "aligned_model"
    else:
        name = name if len(name) > 0 else "nonaligned_model"
    return name + "_" + args.model


def checkpoint_path(args, name=""):
    os.makedirs("pre_trained_models", exist_ok=True)
    return os.path.join("pre_trained_models", f"{save_load_name(args, name)}.pt")


def save_model(args, model, name=""):
    model_to_save = model.module if hasattr(model, "module") else model
    payload = {"model_state_dict": model_to_save.state_dict()}
    torch.save(payload, checkpoint_path(args, name))


def load_model(args, model_cls, name=""):
    payload = torch_load_compat(checkpoint_path(args, name), map_location=args.device)
    model = model_cls(args)
    model.load_state_dict(payload["model_state_dict"])
    return model.to(args.device)
