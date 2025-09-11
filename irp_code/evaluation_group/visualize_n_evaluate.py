"""
visualize_n_evaluate.py
This script is designed to work with a trained anomaly detection
model for strawberry images.
It includes functions to visualize the reconstruction of images,
infer anomaly scores, and evaluate the model's
performance using various metrics such as MSE, MAE, and SSIM.
It also provides a comprehensive evaluation summary and plots
for better understanding of the model's performance.
It is assumed that the model is an instance of `AnomalyNet` or similar, which
is a PyTorch neural network model designed for anomaly detection in images.
strawberry images.

NB: The AnomalyEvaluator class was created with the help of GitHub Copilot.

Requirements:
    - Python 3.x
    - `torch` for model handling: `pip install torch`
    - `torchvision` for image transformations: `pip install torchvision`
    - `matplotlib` for visualization: `pip install matplotlib`
    - `numpy` for numerical operations: `pip install numpy`
    - `PIL` for image handling: `pip install pillow`
    - `torchmetrics` for SSIM metric: `pip install torchmetrics`
    - `pytorch-ignite` for SSIM metric: `pip install pytorch-ignite`
    - `scikit-learn` for evaluation metrics: `pip install scikit-learn`
"""

import os
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIMMetric
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
import csv
from torchvision import transforms as T


PATCH_SIZE = 128
EPOCHS = 50
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
anetgan_model = "placeholder"
compute_reconstruction_error = "placeholder"
evaluate_dataset = "placeholder"

transform_patch = transforms.Compose(
    [transforms.Resize((PATCH_SIZE, PATCH_SIZE)), transforms.ToTensor()]
)


class PatchDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        for subdir in os.listdir(root_dir):
            # Check if it's a directory
            if not os.path.isdir(os.path.join(root_dir, subdir)):
                continue
            subdir_path = os.path.join(self.root_dir, subdir)
            for f in os.listdir(subdir_path):
                if f.endswith(".jpg"):
                    self.image_files.append(os.path.join(subdir, f))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


# Inference function to visualize and categorize strawberry anomalies
def infer_and_plot(image_path, model):
    """
    Function to infer and plot anomalies in strawberry images
    based on a trained unsupervised anomaly detection model,
    and using SSIM for anomaly scoring.
    """
    model.eval()
    image = Image.open(image_path).convert("RGB")
    tf = transform_patch
    x = tf(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
    anomaly = torch.abs(out - x).squeeze(0).mean(dim=0).cpu().numpy()
    # Use SSIMMetric for thresholding
    ssim = SSIMMetric().to(DEVICE)
    ssim_score = 1 - ssim(out.to(DEVICE), x.to(DEVICE))

    if ssim_score >= 0.24:
        status = "Most likely DISEASED, 93% detection confidence"
    elif ssim_score >= 0.1595 and ssim_score < 0.24:
        status = "POTENTIALLY DISEASED, 85% detection confidence"
    elif ssim_score > 0.13 and ssim_score < 0.1595:
        status = "POTENTIALLY DISEASED, 79% detection confidence"
    elif ssim_score > 0.0383 and ssim_score <= 0.1:
        status = "POTENTIALLY HEALTHY, 80% detection confidence"
    elif ssim_score <= 0.0383:
        status = "HEALTHY, 99.96% detection confidence"
    else:
        status = "POTENTIALLY DISEASED, further assessment required"
    print(f"Anomaly Score: {ssim_score:.4f} - Status: {status}")
    # Convert output tensor to numpy for visualization
    output_img = out.squeeze(0).cpu().permute(1, 2, 0).numpy()
    output_img = np.clip(output_img, 0, 1)
    # Plot the original image and the anomaly map
    image_np = np.array(image.resize(anomaly.shape[::-1]))
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title("Original")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.title("Reconstructed")
    plt.imshow(output_img)
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(anomaly, cmap="hot")
    plt.title("Anomaly Map")
    plt.colorbar()
    plt.axis("off")
    plt.tight_layout()
    plt.show()


class AnomalyEvaluator:
    """
    A class of functions for evaluating anomaly detection models.
    """

    def __init__(
        self, model, device: str = "cuda" if
        torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

        # Initialize SSIM metric based on available library
        self.ssim_metric = SSIMMetric(data_range=1.0,
                                      kernel_size=11, sigma=1.5).to(
            device
        )

    def compute_reconstruction_error(
        self, images: torch.Tensor, error_type: str = "mse"
    ) -> torch.Tensor:
        """
        Compute the reconstruction error between original and
        reconstructed images based on the specified error type.
        Error types can be one of "mse", "mae", "ssim".
        """
        with torch.no_grad():
            reconstructions = self.model(images)

            if error_type == "mse":
                errors = torch.mean((images - reconstructions)
                                    ** 2, dim=(1, 2, 3))
            elif error_type == "mae":
                errors = torch.mean(torch.abs(images -
                                              reconstructions), dim=(1, 2, 3))
            elif error_type == "ssim":
                # Compute SSIM for each image in the batch
                ssim_values = []
                for i in range(images.size(0)):
                    img_orig = images[i:i + 1]  # Take a single image batch
                    img_recon = reconstructions[i:i + 1]
                    # Ensure images are in the correct format for SSIM
                    ssim_val = self.ssim_metric(img_recon, img_orig)
                    ssim_values.append(ssim_val)

                ssim_tensor = torch.stack(ssim_values)
                errors = 1 - ssim_tensor
            else:
                raise ValueError(f"Unsupported error type: {error_type}")

        return errors

    def evaluate_dataset(
        self, dataloader: DataLoader, error_types: List[str] =
        ["mse", "mae", "ssim"]
    ) -> Dict:
        """
        Evaluate the model on a dataset with multiple error metrics

        Args:
            dataloader: DataLoader for the test dataset
            labels: List of labels (0 for healthy, 1 for diseased)
            error_types: List of error types to compute

        Returns:
            Dictionary containing evaluation metrics
        """
        all_errors = {error_type: [] for error_type in error_types}
        all_images = []
        all_reconstructions = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if isinstance(batch, tuple):
                    images, batch_labels = batch
                else:
                    images = batch

                images = images.to(self.device)
                reconstructions = self.model(images)

                # Compute different types of reconstruction errors
                for error_type in error_types:
                    errors = self.compute_reconstruction_error(images,
                                                               error_type)
                    all_errors[error_type].extend(errors.cpu().numpy())

                all_images.extend(images.cpu().numpy())
                all_reconstructions.extend(reconstructions.cpu().numpy())

        # Convert to numpy arrays
        for error_type in error_types:
            all_errors[error_type] = np.array(all_errors[error_type])

        results = {
            "reconstruction_errors": all_errors,
            "images": np.array(all_images),
            "reconstructions": np.array(all_reconstructions),
            "error_types": error_types,
        }

        return results

    def print_evaluation_summary(self, results: Dict):
        """Print a comprehensive evaluation summary for all error types"""
        print("ANOMALY DETECTION EVALUATION SUMMARY")
        print("=" * 60)

        error_types = results.get("error_types", ["mse"])

        print(
            f"Total samples evaluated: \
            {len(results['reconstruction_errors'][error_types[0]])}"
        )

        # Print metrics for each error type
        for error_type in error_types:
            errors = results["reconstruction_errors"][error_type]
            print(f"\n{error_type.upper()} Error Statistics:")
            print(f"  Mean: {np.mean(errors):.4f}")
            print(f"  Std: {np.std(errors):.4f}")
            print(f"  Min: {np.min(errors):.4f}")
            print(f"  Max: {np.max(errors):.4f}")

    def plot_evaluation_results(self, results: Dict, save_path: str = None):
        error_types = results.get("error_types", ["mse"])

        # Create subplots for just the reconstruction images
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

        fig.suptitle("Anomaly Detection Evaluation Results", fontsize=16)
        # Sample reconstructions
        if "images" in results and "reconstructions" in results:
            n_samples = min(6, len(results["images"]))

            for i in range(min(3, n_samples)):
                # Original image
                img = results["images"][i].transpose(1, 2, 0)
                recon = results["reconstructions"][i].transpose(1, 2, 0)

                # Create side-by-side comparison
                comparison = np.hstack([img, recon])
                axes[i].imshow(comparison)

                # Add error information
                error_info = []
                for error_type in error_types:
                    error_val = results["reconstruction_errors"][error_type][i]
                    error_info.append(f"{error_type.upper()}: {error_val:.3f}")

                axes[i].set_title(f'Original | Reconstruction\n\
                                  {", ".join(error_info)}')
                axes[i].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


def catch_anomaly(dataloader, model, threshold=0.1):
    """
    Evaluate anomaly detection on a dataset

    Args:
        dataloader: DataLoader with images
        model: Model that reconstructs images
        threshold: Threshold for anomaly detection (default 0.1)

    Returns:
        Dictionary with scores, statuses, and statistics
    """
    model.eval()
    model.to(DEVICE)
    ssim_metric = SSIMMetric().to(DEVICE)

    all_scores = []
    all_status = []
    total_samples = 0
    anomaly_count = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch.to(DEVICE)
            outputs = model(images)

            # Compute SSIM for each image pair in the batch
            batch_scores = []
            for i in range(images.size(0)):
                img_orig = images[i:i + 1]  # Take a single image batch
                img_recon = outputs[i:i + 1]
                ssim_val = ssim_metric(img_recon, img_orig)
                anomaly_score = 1 - ssim_val
                batch_scores.append(anomaly_score.item())

            # Process scores
            for score in batch_scores:
                status = "POTENTIALLY DISEASED" if \
                    score > threshold else "HEALTHY"
                all_scores.append(score)
                all_status.append(status)

                if score > threshold:
                    anomaly_count += 1
                total_samples += 1

    # Calculate statistics
    scores_array = np.array(all_scores)
    stats = {
        "mean": np.mean(scores_array),
        "std": np.std(scores_array),
        "min": np.min(scores_array),
        "max": np.max(scores_array),
        "anomaly_rate":
        anomaly_count / total_samples if total_samples > 0 else 0,
        "total_samples": total_samples,
        "anomaly_count": anomaly_count,
    }

    # Print summary
    print("\nEvaluation Summary:")
    print(f"Total samples: {total_samples}")
    print(f"Detected anomalies: {anomaly_count}(\
          {stats['anomaly_rate']*100:.2f}%)")
    print(f"Score statistics: Mean={stats['mean']:.4f},\
           Std={stats['std']:.4f}, Min={stats['min']:.4f},\
            Max={stats['max']:.4f}")

    return {"scores": all_scores, "statuses": all_status, "stats": stats}


def find_optimal_threshold(
    good_loader, bad_loader, model, threshold_range=None, step=0.005
):
    """
    Find optimal threshold for anomaly detection by evaluating
    metrics across a range of thresholds

    Args:
        good_loader: DataLoader with healthy samples
        bad_loader: DataLoader with anomalous samples
        model: The anomaly detection model
        threshold_range: Tuple of (min_threshold, max_threshold)
        or None to use default
        step: Step size for threshold values

    Returns:
        Dictionary with optimal thresholds and performance metrics
    """
    if threshold_range is None:
        threshold_range = (0.05, 0.2)  # Default range based on observed data

    # Prepare model and evaluation metric
    model.eval()
    ssim_metric = SSIMMetric().to(DEVICE)

    # Get anomaly scores for good and bad samples
    good_scores = []
    bad_scores = []

    # Process good samples
    with torch.no_grad():
        for batch in good_loader:
            images = batch.to(DEVICE)
            outputs = model(images)

            # Compute SSIM for each image pair in the batch
            for i in range(images.size(0)):
                img_orig = images[i:i + 1]
                img_recon = outputs[i:i + 1]
                ssim_val = ssim_metric(img_recon, img_orig)
                anomaly_score = 1 - ssim_val
                good_scores.append(anomaly_score.item())

    # Process bad samples
    with torch.no_grad():
        for batch in bad_loader:
            images = batch.to(DEVICE)
            outputs = model(images)

            # Compute SSIM for each image pair in the batch
            for i in range(images.size(0)):
                img_orig = images[i:i + 1]
                img_recon = outputs[i:i + 1]
                ssim_val = ssim_metric(img_recon, img_orig)
                anomaly_score = 1 - ssim_val
                bad_scores.append(anomaly_score.item())

    # Calculate metrics for different thresholds
    thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
    results = {
        "thresholds": thresholds,
        "precision": [],
        "recall": [],
        "f1_score": [],
        "accuracy": [],
        "specificity": [],
    }

    for threshold in thresholds:
        # Apply threshold
        good_preds = [score > threshold for score in good_scores]
        bad_preds = [score > threshold for score in bad_scores]

        # True Negatives: good samples correctly identified as good
        tn = sum(not pred for pred in good_preds)
        # False Positives: good samples incorrectly identified as bad
        fp = sum(good_preds)
        # True Positives: bad samples correctly identified as bad
        tp = sum(bad_preds)
        # False Negatives: bad samples incorrectly identified as good
        fn = sum(not pred for pred in bad_preds)

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Store results
        results["precision"].append(precision)
        results["recall"].append(recall)
        results["f1_score"].append(f1)
        results["accuracy"].append(accuracy)
        results["specificity"].append(specificity)

    # Find optimal thresholds
    best_f1_idx = np.argmax(results["f1_score"])
    best_accuracy_idx = np.argmax(results["accuracy"])
    best_balanced_idx = np.argmax(
        np.array(results["recall"]) + np.array(results["specificity"])
    )

    # Create results dictionary
    optimal_results = {
        "best_f1": {
            "threshold": thresholds[best_f1_idx],
            "f1_score": results["f1_score"][best_f1_idx],
            "precision": results["precision"][best_f1_idx],
            "recall": results["recall"][best_f1_idx],
            "accuracy": results["accuracy"][best_f1_idx],
            "specificity": results["specificity"][best_f1_idx],
        },
        "best_accuracy": {
            "threshold": thresholds[best_accuracy_idx],
            "f1_score": results["f1_score"][best_accuracy_idx],
            "precision": results["precision"][best_accuracy_idx],
            "recall": results["recall"][best_accuracy_idx],
            "accuracy": results["accuracy"][best_accuracy_idx],
            "specificity": results["specificity"][best_accuracy_idx],
        },
        "best_balanced": {
            "threshold": thresholds[best_balanced_idx],
            "f1_score": results["f1_score"][best_balanced_idx],
            "precision": results["precision"][best_balanced_idx],
            "recall": results["recall"][best_balanced_idx],
            "accuracy": results["accuracy"][best_balanced_idx],
            "specificity": results["specificity"][best_balanced_idx],
        },
        "all_results": results,
    }

    # Print summary
    print("\nOptimal Threshold Results:")
    print(f"Best F1 Score: {optimal_results['best_f1']['f1_score']:.4f}\
           at threshold {optimal_results['best_f1']['threshold']:.4f}")
    print(f"  Precision: {optimal_results['best_f1']['precision']:.4f}")
    print(f"  Recall: {optimal_results['best_f1']['recall']:.4f}")
    print(f"  Accuracy: {optimal_results['best_f1']['accuracy']:.4f}")
    print(f"  Specificity: {optimal_results['best_f1']['specificity']:.4f}")

    print(
        f"\nBest Balanced Accuracy (Recall + Specificity): \
           {optimal_results['best_accuracy']['accuracy']:.4f} \
              at threshold {optimal_results['best_accuracy']['threshold']:.4f}"
                  )
    print(f"  F1 Score: {optimal_results['best_accuracy']['f1_score']:.4f}")
    print(f"  Precision: {optimal_results['best_accuracy']['precision']:.4f}")
    print(f"  Recall: {optimal_results['best_accuracy']['recall']:.4f}")
    print(
        f"  Specificity: {optimal_results['best_accuracy']['specificity']:.4f}"
        )

    # Plot the results
    plt.figure(figsize=(15, 10))

    # Plot precision-recall curve
    plt.subplot(2, 2, 1)
    plt.plot(
        results["recall"], results["precision"],
        "b-", label="Precision-Recall curve"
    )
    plt.scatter(
        optimal_results["best_f1"]["recall"],
        optimal_results["best_f1"]["precision"],
        color="red",
        label=f'Best F1: {optimal_results["best_f1"]["threshold"]:.4f}',
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)

    # Plot metrics vs threshold
    plt.subplot(2, 2, 2)
    plt.plot(thresholds, results["precision"], "g-", label="Precision")
    plt.plot(thresholds, results["recall"], "r-", label="Recall")
    plt.plot(thresholds, results["f1_score"], "b-", label="F1 Score")
    plt.plot(thresholds, results["accuracy"], "y-", label="Accuracy")
    plt.plot(thresholds, results["specificity"], "m-", label="Specificity")
    plt.axvline(
        x=optimal_results["best_f1"]["threshold"],
        color="k",
        linestyle="--",
        label="Best F1 Threshold",
    )
    plt.axvline(
        x=optimal_results["best_accuracy"]["threshold"],
        color="c",
        linestyle="--",
        label="Best Accuracy Threshold",
    )
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Metrics vs Threshold")
    plt.legend()
    plt.grid(True)

    # Histogram of scores
    plt.subplot(2, 2, 3)
    plt.hist(good_scores, alpha=0.5, bins=30, label="Healthy", color="green")
    plt.hist(bad_scores, alpha=0.5, bins=30, label="Diseased", color="red")
    plt.axvline(
        x=optimal_results["best_f1"]["threshold"],
        color="k",
        linestyle="--",
        label="Best F1 Threshold",
    )
    plt.axvline(
        x=optimal_results["best_accuracy"]["threshold"],
        color="c",
        linestyle="--",
        label="Best Accuracy Threshold",
    )
    plt.xlabel("Anomaly Score")
    plt.ylabel("Count")
    plt.title("Distribution of Anomaly Scores")
    plt.legend()
    plt.grid(True)

    # ROC curve
    plt.subplot(2, 2, 4)
    fpr = [1 - spec for spec in results["specificity"]]
    plt.plot(fpr, results["recall"], "b-", label="ROC curve")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return optimal_results

# CLASSIFIER UTILITY FUNCTIONS


def _norm(s: str) -> str:
    return (s or "").strip().lower()


class PatchLoaderCL(Dataset):
    def __init__(
        self,
        root_dir,
        csv_file="classes.csv",
        transform=None,
        label_columns=None,
        strict_one_hot=True,
        skip_labels=(
            "Unlabeled", "Leaf Spot", 'Healthy-Leaf -Strawberry', 'Mulch'
            ),
        drop_labels=True,  # when True, remove skipped labels (reindex)
        valid_exts=(".jpg", ".jpeg", ".png"),
        verbose=True,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.targets = []

        # Original label columns (order defines class indices)
        self.label_columns = label_columns or [
            "Anthracnose Fruit Rot",
            "Gray Mold",
            "Healthy-Leaf -Strawberry",
            "Leaf Spot",
            "Mulch",
            "Powdery Mildew Fruit",
            "Powdery Mildew Leaf",
            "Unlabeled",
            "Withered fruit",
            "Healthy-Strawberry",
        ]

        # Normalize for robust comparisons
        self._label_columns_norm = [_norm(x) for x in self.label_columns]
        skip_norm = {_norm(x) for x in (skip_labels or [])}

        # Build mapping old_idx -> new_idx excluding skipped labels
        keep_old_indices = [i for i, name in enumerate(
            self._label_columns_norm) if name not in skip_norm]
        if drop_labels:
            self.old_to_new = {old: new for new, old in enumerate(
                keep_old_indices)}
            self.class_names = [
                self.label_columns[i] for i in keep_old_indices
                ]
        else:
            # keep label space unchanged
            self.old_to_new = {i: i for i in range(len(self.label_columns))}
            self.class_names = list(self.label_columns)

        if verbose:
            dropped = [self.label_columns[i] for i in range(
                len(self.label_columns)) if i not in keep_old_indices]
            print(f"Skipping labels: {sorted(skip_norm)}")
            if drop_labels:
                print(f"Dropping from class space and remapping.\
                       Kept classes ({len(self.class_names)}):\
                          {self.class_names}")
                if dropped:
                    print(f"Dropped classes: {dropped}")
            else:
                print("Not dropping from class space (indices unchanged).")

        csv_path = os.path.join(root_dir, csv_file)
        with open(csv_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fn = (row.get("filename") or "").strip()
                if not fn:
                    continue

                # Read row label values in the declared order
                y = [float(row.get(col, 0)) for col in self.label_columns]

                if strict_one_hot:
                    s = sum(int(v > 0.5) for v in y)
                    if s != 1:
                        # skip non one-hot rows to avoid ambiguity
                        continue

                # Pick the active class (argmax as a fallback)
                label_idx_old = int(max(range(len(y)), key=lambda i: y[i]))
                label_name_norm = self._label_columns_norm[label_idx_old]

                # Skip if it's in the skip set
                if label_name_norm in skip_norm:
                    continue

                # Remap to new contiguous class index
                if label_idx_old not in self.old_to_new:
                    # This can only happen if drop_labels=
                    # True and label is skipped
                    continue
                label_idx_new = self.old_to_new[label_idx_old]

                full_path = os.path.join(root_dir, fn)
                if os.path.exists(
                        full_path
                        ) and full_path.lower().endswith(tuple(valid_exts)):
                    self.image_files.append(full_path)
                    self.targets.append(label_idx_new)

        if verbose:
            print(f"Loaded {len(self.image_files)}\
                   images after filtering. Num classes:\
                   {len(self.class_names)}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        return img, target


def _is_normalize(obj) -> bool:
    # Works for torchvision.transforms. Normalize and compatible wrappers
    if T is not None and isinstance(obj, T.Normalize):
        return True
    # Fallback: duck-typing by attribute presence
    return hasattr(obj, "mean") and hasattr(
        obj, "std") and hasattr(obj, "__call__"
                                )


def remove_normalize_from_transform(transform):
    """
    Return a version of `transform` with any Normalize steps removed.
    If it's not a torchvision Compose, returns it unchanged.
    """
    if T is None or not isinstance(transform, T.Compose):
        return transform
    kept = [t for t in transform.transforms if not _is_normalize(t)]
    return T.Compose(kept)


def add_or_replace_normalize(transform, mean: List[float], std: List[float]):
    """
    Append Normalize(mean, std) at the end of the transform,
    or replace an existing Normalize.
    Does not add or change any other transform steps.
    """
    if T is None:
        raise RuntimeError(
            "torchvision is required to modify transforms for normalization."
            )
    norm = T.Normalize(mean=mean, std=std)

    if isinstance(transform, T.Compose):
        new_list = []
        replaced = False
        for t in transform.transforms:
            if _is_normalize(t) and not replaced:
                new_list.append(norm)
                replaced = True
            else:
                new_list.append(t)
        if not replaced:
            new_list.append(norm)  # add at the end
        return T.Compose(new_list)

    # If it's a single transform (not Compose),
    # wrap into a Compose and add Normalize
    return T.Compose([transform, norm])


@contextmanager
def temporary_transform(dataset, temp_transform):
    """
    Temporarily set dataset.transform to `temp_transform` during the context.
    Always restores the original afterward.
    """
    old = getattr(dataset, "transform", None)
    dataset.transform = temp_transform
    try:
        yield
    finally:
        dataset.transform = old


@torch.no_grad()
def compute_mean_std_from_dataset(
    dataset,
    batch_size: int = 64,
    num_workers: int = 4,
    max_batches: Optional[int] = None,
) -> Tuple[List[float], List[float]]:
    """
    Compute per-channel mean and std from a dataset,
    using its existing transforms
    EXCEPT normalization (which is temporarily removed).
    No other transforms are added or changed.

    Returns (mean, std) lists suitable for torchvision.transforms.Normalize.

    Note:
    - Assumes the dataset's transform outputs tensors
    in [0, 1] range (e.g., after ToTensor()).
    - If your current transform lacks ToTensor,
    add it in your original script before using this.
    """
    # Use the same transforms as your dataset, but without Normalize
    base_transform = getattr(dataset, "transform", None)
    transform_no_norm = remove_normalize_from_transform(base_transform)

    channel_sum = None
    channel_sq_sum = None
    pixel_count = 0

    dl = DataLoader(dataset, batch_size=batch_size,
                    shuffle=False, num_workers=num_workers, pin_memory=True)
    batches_seen = 0

    with temporary_transform(dataset, transform_no_norm):
        for images, _ in dl:
            # images: [B, C, H, W]
            if images.ndim != 4:
                raise ValueError(
                    "Expected images as tensors with shape [B, C, H, W]. \
                        Check your dataset transform."
                        )

            b, c, h, w = images.shape
            imgs = images.float()  # ensure float
            sum_ = imgs.sum(dim=(0, 2, 3))                 # [C]
            sq_sum = (imgs * imgs).sum(dim=(0, 2, 3))      # [C]
            n_pixels = b * h * w

            if channel_sum is None:
                channel_sum = sum_
                channel_sq_sum = sq_sum
            else:
                channel_sum += sum_
                channel_sq_sum += sq_sum

            pixel_count += n_pixels
            batches_seen += 1
            if max_batches is not None and batches_seen >= max_batches:
                break

    if pixel_count == 0 or channel_sum is None:
        raise RuntimeError(
            "Failed to compute mean/std; dataset may"
            " be empty or transform is incompatible."
        )

    mean = channel_sum / pixel_count
    var = (channel_sq_sum / pixel_count) - mean.pow(2)
    std = torch.sqrt(var.clamp_min(1e-12))

    return mean.tolist(), std.tolist()
