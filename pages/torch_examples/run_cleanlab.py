import cleanlab 
from skorch import NeuralNetClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from cleanlab.filter import find_label_issues
import numpy as np
from data_generation import NUM_SAMPLES
from train_utils import create_loaders
import torch
from torch.utils.data import TensorDataset
import numpy as np
import torch.nn.functional as F
import pandas as pd
from sklearn.datasets import make_circles
from csv_utils import format_csv
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Tuple
from train_utils import train_model
from loguru import logger
import argparse

BATCH_SIZE = 32

def sigmoid(x):
    return 1/(1 + np.exp(-x))


def create_data(output_path: Path, label_mixup: bool = True, ratio: float = 0.8) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    np.random.seed(96578)
    data_circles, label_circles = make_circles(n_samples=NUM_SAMPLES, factor=0.5, noise=0.05)

    if label_mixup:
        logger.info('Corrupting labels...')
        corrupted_indices = np.random.choice(range(len(label_circles)), size = 20, replace = False)
        df = pd.DataFrame({'x1': data_circles[:,0], 'x2': data_circles[:,1], 'labels': label_circles})
        df.to_csv(output_path/'circles_uncorrupted.csv', index = False)
        
        label_circles[corrupted_indices] = abs(label_circles[corrupted_indices] - 1)
        df = pd.DataFrame({'x1': data_circles[:,0], 'x2': data_circles[:,1], 'labels': label_circles})
        df.to_csv(output_path/'circles_corrupted_test.csv', index = False)

    # create dataloaders
    data_tensor_circles = TensorDataset(
                torch.Tensor(data_circles), F.one_hot(torch.Tensor(label_circles).to(torch.int64), num_classes=2).float() 
            )
    train_loader_circles, val_loader_circles = create_loaders(data = data_tensor_circles, ratio = ratio, batch_size = BATCH_SIZE, num_workers = 0,
                                                        shuffle_train = True, shuffle_val = False)

    logger.info(
        f"Created dataloaders with {ratio*100:.0f}/{(1 - ratio)*100:.0f}"
        "train/test split \U0001F52A"
    )

    return  train_loader_circles, val_loader_circles, data_circles, label_circles

def run_cleanlab(simple_dnn, data_circles: np.ndarray, label_circles: np.ndarray, output_path: Path):
    model_skorch = NeuralNetClassifier(simple_dnn.encoder)
    #cl = cleanlab.classification.CleanLearning(model_skorch)

    pred_probs_logits = cross_val_predict(
        model_skorch,
        np.float32(data_circles),
        label_circles,
        cv=3,
        method="predict_proba",
        )

    pred_probs = np.apply_along_axis(sigmoid, 0, pred_probs_logits)

    predicted_labels = pred_probs.argmax(axis=1)
    acc = accuracy_score(label_circles, predicted_labels)
    logger.info(f"Cross-validated estimate of accuracy on held-out data: {acc}")

    ranked_label_issues = find_label_issues(
    label_circles,
    pred_probs,
    return_indices_ranked_by="self_confidence",
    )

    # save for plotting
    df = pd.DataFrame({'x1': data_circles[ranked_label_issues[:20],0], 
                    'x2': data_circles[ranked_label_issues[:20],1],
                    'labels': label_circles[ranked_label_issues[:20]]*0 + 2})

    df.to_csv(output_path/'circles_corrupted_cleanlab_pred.csv', index = False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir_logger', 
                        type = str, 
                        required = True)
    parser.add_argument('--name_logger', 
                        type = str, 
                        required = True)    
    parser.add_argument('--output_file_name', 
                        type = str, 
                        required = True) 
    args = parser.parse_args()  

    train_loader, val_loader, data_circles, label_circles = create_data(label_mixup = True, output_path = Path(args.save_dir_logger))

    simple_dnn = train_model(train_loader = train_loader,
                val_loader = val_loader,
                save_dir_logger = args.save_dir_logger,
                name_logger = args.name_logger,
                break_activations = False,
                output_dim = 2,
                freeze_weights= False,
                freeze_bias = False,
                weight_watcher = False,
                return_model = True,
                )

    input_path = Path(args.save_dir_logger) / args.name_logger / 'version_0' / 'metrics.csv'
    output_path = Path('reformatted_metrics') / (args.output_file_name + '.csv')
    format_csv(input_path, ['train_loss', 'val_loss'], output_path)

    logger.info('Running cleanlab...')
    run_cleanlab(simple_dnn, data_circles, label_circles, output_path = Path(args.save_dir_logger))


if __name__ == '__main__':
    main()