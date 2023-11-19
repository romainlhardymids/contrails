import argparse
import data
import gc
import numpy as np
import torch
import yaml

from scripts.classification.train import ClassificationModule
from data.dataset import ContrailsDataset
from torchmetrics import AveragePrecision, Dice, FBetaScore
from torchmetrics.functional import dice
from torch.utils.data import DataLoader
from train import SegmentationModule2d


def load_model(module, config, checkpoint_path):
    """Loads a 2D segmentation model checkpoint."""
    model = module(config)
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    return model


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/config.yaml", required=True)
    args = parser.parse_args()
    return args


def flip_batch(x, i):
    """Flips a batch of images along a specific axis."""
    if i == 0:
        return x
    elif i == 1:
        return x.flip(3)
    elif i == 2:
        return x.flip(4)
    else:
        return x.flip(3).flip(4)


def evaluate(config):
    """Evaluates a set of models on held-out data."""
    df = data.utils.data_split("../data/data_split.csv")
    df = df[df.split == "validation"]
    print(f"Images: {df.shape[0]}")

    clf_models = config["clf_models"]
    seg_models = config["seg_models"]
    device = config["device"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    bins = config["bins"]
    thresholds = [i / bins for i in range(1, bins)]
    total_clf_weight = sum([clf_models[k]["weight"] * len(clf_models[k]["checkpoint_paths"]) for k in clf_models])
    total_seg_weight = sum([seg_models[k]["weight"] * len(seg_models[k]["checkpoint_paths"]) for k in seg_models])

    # Classification filtering
    predictions = {}
    labels = {}
    for name, model in clf_models.items():
        weight = model["weight"]
        config_path = model["config_path"]
        checkpoint_paths = model["checkpoint_paths"]

        print(f"Loading classification model `{name}` with configuration: {config_path}")
        with open(config_path, "rb") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            
        for checkpoint_path in checkpoint_paths:
            print(f"Checkpoint: {checkpoint_path}")
            model = load_model(ClassificationModule, config["model"], checkpoint_path)
            model.to(device)
            model.eval()
            
            dataset = ContrailsDataset(
                df, 
                timesteps=[i for i in range(8)], 
                image_size=config["model"]["data"]["image_size"],
                cutmix_prob=0.,
                split="validation"
            )
            dataloader = DataLoader(
                dataset, 
                shuffle=False, 
                batch_size=batch_size // 8,
                num_workers=num_workers
            )

            fbeta = [FBetaScore(task="binary", beta=1., threshold=t).to(device) for t in thresholds]

            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    x = batch["frames"].to(device)
                    y = batch["label"].to(device)
                    n, c, t, h, w = x.shape
                    x = torch.permute(x, (0, 2, 1, 3, 4)).contiguous()
                    x = x.view(-1, c, h, w)
                    logits, _ = model.model(x)
                    probs = torch.sigmoid(logits).view(n, t)
                    for j, (pp, yy) in enumerate(zip(probs, y)):
                        record_id = df.iloc[i * batch_size // 8 + j]["record_id"]
                        predictions.setdefault(record_id, torch.zeros(pp.shape, dtype=torch.float32).to(device))
                        predictions[record_id] += pp * weight / total_clf_weight
                        labels[record_id] = yy
                    for fb in fbeta:
                        fb.update(probs[:, 4], y)

            scores = [fb.compute().cpu().numpy() for fb in fbeta]
            idx = np.argmax(scores)
            print(f"F1 coefficient (t = {thresholds[idx]}): {scores[idx]:.04f}")

            del model, dataset, dataloader, fbeta
            torch.cuda.empty_cache()
            gc.collect()

    ensemble_fbeta = [FBetaScore(task="binary", beta=1., threshold=t).to(device) for t in thresholds]
    for record_id in predictions:
        probs, y = predictions[record_id], labels[record_id]
        for ef in ensemble_fbeta:
            ef.update(probs[None, 4], y[None])    
            
    scores = [ef.compute().cpu().numpy() for ef in ensemble_fbeta]
    idx = np.argmax(scores)
    threshold, score = thresholds[idx], scores[idx]
    threshold *= 0.5
    negative_records = set([record_id for record_id in predictions if (predictions[record_id] > threshold).sum() < 2])
    print(f"Ensemble F1 coefficient (t = {threshold}): {score:.04f}")
    print(f"Negative records: {len(negative_records)}")

    # Segmentation predictions
    predictions = {}
    labels = {}
    for name, model in seg_models.items():
        weight = model["weight"]
        use_tta = model["use_tta"]
        config_path = model["config_path"]
        checkpoint_paths = model["checkpoint_paths"]

        print(f"Loading segmentation model `{name}` with configuration: {config_path}")
        with open(config_path, "rb") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            
        for checkpoint_path in checkpoint_paths:
            print(f"Checkpoint: {checkpoint_path}")
            model = load_model(SegmentationModule2d, config["model"], checkpoint_path)
            model.to(device)
            model.eval()
            
            image_size = config["model"]["data"]["image_size"]
            timesteps = config["model"]["data"]["timesteps"]
            
            dataset = ContrailsDataset(
                df, 
                timesteps=timesteps,
                image_size=image_size,
                cutmix_prob=0.,
                split="validation"
            )
            dataloader = DataLoader(
                dataset, 
                shuffle=False, 
                batch_size=batch_size,
                num_workers=num_workers
            )

            global_dice = [Dice(threshold=t).to(device) for t in thresholds]
            auprc = AveragePrecision(task="binary").to(device)

            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    x = batch["frames"].to(device)
                    y = batch["mask"].to(device)
                    
                    # Full resolution prediction
                    logits = model.model(x)
                    if logits.shape[-1] != 256:
                        logits = torch.nn.functional.interpolate(logits, size=256, mode="bilinear")
                    probs = torch.sigmoid(logits)
                    probs = probs.view(probs.size(0), 1, 256, 256)

                    # TTA
                    if use_tta:
                        probs *= 3.
                        for j in range(3):
                            x_ = flip_batch(x, j + 1)
                            logits_ = model.model(x_)
                            if logits_.shape[-1] != 256:
                                logits_ = torch.nn.functional.interpolate(logits_, size=256, mode="bilinear")
                            probs_ = torch.sigmoid(logits_)
                            probs_ = probs_.view(probs_.size(0), 1, 1, 256, 256)
                            probs_ = flip_batch(probs_, j + 1)[:, 0]
                            probs += probs_
                        probs /= 6.
                        
                    for j, (pp, yy) in enumerate(zip(probs, y)):
                        record_id = df.iloc[i * batch_size + j]["record_id"]
                        predictions.setdefault(record_id, torch.zeros(pp.shape, dtype=torch.float32).to(device))
                        if record_id in negative_records:
                            pp = torch.zeros(pp.shape, dtype=torch.float32).to(device)
                        predictions[record_id] += pp * weight / total_seg_weight
                        labels[record_id] = yy
                        for gd in global_dice:
                            gd.update(pp, yy)
                        auprc.update(pp, yy)

            scores = [gd.compute().cpu().numpy() for gd in global_dice]
            idx = np.argmax(scores)
            print(f"DICE coefficient (t = {thresholds[idx]}): {scores[idx]:.04f}")
            print(f"AUPRC: {auprc.compute():.04f}")

            del model, dataset, dataloader, global_dice, auprc
            torch.cuda.empty_cache()
            gc.collect()
    
    ensemble_dice = [Dice(threshold=t).to(device) for t in thresholds]
    for record_id in predictions:
        probs, y = predictions[record_id], labels[record_id]
        for ed in ensemble_dice:
            ed.update(probs, y)    
            
    scores = [ed.compute().cpu().numpy() for ed in ensemble_dice]
    idx = np.argmax(scores)
    threshold, score = thresholds[idx], scores[idx]
    print(f"Ensemble DICE coefficient (t = {threshold}): {score:.04f}")


def main():
    args = parse_args()
    with open(args.config, "rb") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    evaluate(config)
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()