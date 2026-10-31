"""
Evaluation pipeline for ADNI MRI classification.

Supports:
    - Baseline CNN and ConvNeXt-Tiny
    - Patient-independent validation/test evaluation
    - Accuracy, precision, recall, F1, AUROC
    - Confusion matrix and ROC curve
    - Calibration analysis
    - Misclassification examples
    - Inference latency

Classes:
    NC = 0
    AD = 1
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    brier_score_loss,
)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from modules import create_model, count_parameters
from dataset import get_transforms

CLASS_NAMES = ["NC", "AD"]


class EvaluationDataset(Dataset):
    """Load ADNI MRI slices without random augmentation."""

    def __init__(self, records, image_size=224):
        self.records = records

        self.transform = get_transforms(
            image_size=image_size,
            training=False,
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        # Adjust this key if your dataset.py uses a different name.
        image_path = record["path"]

        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("L"))

        return (
            tensor,
            int(record["label"]),
            str(image_path),
            str(record["patient_id"]),
        )


def load_evaluation_data(split_path, split, image_size, batch_size, workers):
    with open(split_path, "r") as f:
        splits = json.load(f)

    if split not in ("val", "test"):
        raise ValueError("Evaluation split must be 'val' or 'test'.")

    dataset = EvaluationDataset(splits[split], image_size=image_size)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    return loader


def load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model_name = checkpoint["model_name"]

    model = create_model(
        model_name=model_name,
        pretrained=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


@torch.inference_mode()
def collect_predictions(model, loader, device):
    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_paths = []
    all_patient_ids = []

    elapsed_seconds = 0.0
    total_images = 0

    for images, labels, paths, patient_ids in loader:
        images = images.to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)[:, 1]

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_seconds += time.perf_counter() - start
        total_images += len(labels)

        predictions = (probabilities >= 0.5).long()

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_probabilities.extend(probabilities.cpu().tolist())
        all_paths.extend(paths)
        all_patient_ids.extend(patient_ids)

    return {
        "labels": np.asarray(all_labels),
        "predictions": np.asarray(all_predictions),
        "probabilities": np.asarray(all_probabilities),
        "paths": all_paths,
        "patient_ids": all_patient_ids,
        "latency_ms_per_image": elapsed_seconds / total_images * 1000,
    }


def calculate_metrics(results):
    y_true = results["labels"]
    y_pred = results["predictions"]
    y_prob = results["probabilities"]

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "brier_score": float(
            brier_score_loss(y_true, y_prob)
        ),
    }

    if len(np.unique(y_true)) == 2:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["auroc"] = None

    return metrics


def aggregate_by_patient(results):
    """
    Average slice probabilities per patient.

    This is a simple research aggregation strategy,
    not a clinically validated diagnostic method.
    """

    patients = {}

    for label, probability, patient_id in zip(
        results["labels"],
        results["probabilities"],
        results["patient_ids"],
    ):
        if patient_id not in patients:
            patients[patient_id] = {
                "labels": [],
                "probabilities": [],
            }

        patients[patient_id]["labels"].append(int(label))
        patients[patient_id]["probabilities"].append(float(probability))

    labels = []
    probabilities = []

    for patient_id, values in patients.items():
        unique_labels = set(values["labels"])

        if len(unique_labels) != 1:
            raise ValueError(
                f"Patient {patient_id} has inconsistent labels."
            )

        labels.append(values["labels"][0])
        probabilities.append(np.mean(values["probabilities"]))

    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)

    return {
        "labels": labels,
        "predictions": (probabilities >= 0.5).astype(int),
        "probabilities": probabilities,
    }


def save_confusion_matrix(results, output_dir, name):
    matrix = confusion_matrix(
        results["labels"],
        results["predictions"],
        labels=[0, 1],
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_confusion_matrix.png", dpi=150)
    plt.close()


def save_roc_curve(results, output_dir, name):
    if len(np.unique(results["labels"])) < 2:
        return

    fpr, tpr, _ = roc_curve(
        results["labels"],
        results["probabilities"],
    )

    auc = roc_auc_score(
        results["labels"],
        results["probabilities"],
    )

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUROC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_roc_curve.png", dpi=150)
    plt.close()


def save_calibration_plot(results, output_dir):
    fraction_positive, mean_predicted = calibration_curve(
        results["labels"],
        results["probabilities"],
        n_bins=10,
        strategy="uniform",
    )

    plt.figure(figsize=(6, 5))
    plt.plot(
        mean_predicted,
        fraction_positive,
        marker="o",
        label="Model",
    )
    plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed AD Frequency")
    plt.title("Probability Calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "calibration_curve.png", dpi=150)
    plt.close()


def save_misclassified_examples(results, output_dir, max_examples=5):
    mistakes = np.flatnonzero(
        results["labels"] != results["predictions"]
    )

    if len(mistakes) == 0:
        print("No misclassified images found.")
        return

    # Prioritise high-confidence mistakes.
    confidence = np.maximum(
        results["probabilities"],
        1 - results["probabilities"],
    )

    mistakes = sorted(
        mistakes,
        key=lambda index: confidence[index],
        reverse=True,
    )[:max_examples]

    fig, axes = plt.subplots(
        1,
        len(mistakes),
        figsize=(4 * len(mistakes), 4),
    )

    axes = np.atleast_1d(axes)

    for ax, index in zip(axes, mistakes):
        with Image.open(results["paths"][index]) as image:
            ax.imshow(image.convert("L"), cmap="gray")

        true_label = CLASS_NAMES[results["labels"][index]]
        predicted_label = CLASS_NAMES[results["predictions"][index]]
        probability = results["probabilities"][index]

        ax.set_title(
            f"True: {true_label}\n"
            f"Pred: {predicted_label}\n"
            f"P(AD): {probability:.3f}"
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "misclassified_examples.png", dpi=150)
    plt.close()


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, checkpoint = load_model(args.checkpoint, device)

    image_size = checkpoint.get("image_size", 224)

    loader = load_evaluation_data(
        args.splits,
        args.split,
        image_size,
        args.batch_size,
        args.workers,
    )

    output_dir = Path(args.output_dir) / checkpoint["model_name"] / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ADNI MRI EVALUATION")
    print("=" * 60)
    print("Model:", checkpoint["model_name"])
    print("Split:", args.split)
    print("Device:", device)
    print("Images:", len(loader.dataset))

    results = collect_predictions(model, loader, device)

    slice_metrics = calculate_metrics(results)

    patient_results = aggregate_by_patient(results)
    patient_metrics = calculate_metrics(patient_results)

    slice_metrics["latency_ms_per_image"] = (
        results["latency_ms_per_image"]
    )

    slice_metrics["total_parameters"] = count_parameters(model)

    print("\nSLICE-LEVEL METRICS")
    print(json.dumps(slice_metrics, indent=2))

    print("\nPATIENT-LEVEL METRICS")
    print(json.dumps(patient_metrics, indent=2))

    save_confusion_matrix(results, output_dir, "slice")
    save_confusion_matrix(patient_results, output_dir, "patient")

    save_roc_curve(results, output_dir, "slice")
    save_roc_curve(patient_results, output_dir, "patient")

    save_calibration_plot(results, output_dir)
    save_misclassified_examples(results, output_dir)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "slice_level": slice_metrics,
                "patient_level": patient_metrics,
            },
            f,
            indent=2,
        )

    print("\nEvaluation complete.")
    print("Results saved to:", output_dir)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
    )
    parser.add_argument("--splits", default="splits.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
