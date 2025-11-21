import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from pathlib import Path
import yaml
import argparse
from tqdm import tqdm
import json
from typing import Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingManager:

    def __init__(self, config_path: str, model_type: str = "gnn"):

        self.config_path = config_path
        self.model_type = model_type
        self.config = self._load_config()
        self.device = torch.device(self.config['training']['device'])

        logger.info(f"Device: {self.device}")
        logger.info(f"Model type: {model_type}")

    def _load_config(self) -> dict:

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def build_model(self) -> nn.Module:

        if self.model_type == "gnn":
            from src.models.gnn_models import SpatioTemporalGCN
            model = SpatioTemporalGCN(
                input_dim=self.config['model']['gnn']['input_dim'],
                hidden_dim=self.config['model']['gnn']['hidden_dim'],
                num_layers=self.config['model']['gnn']['num_layers'],
                dropout=self.config['model']['gnn']['dropout'],
                output_dim=self.config['model']['gnn']['output_dim']
            )

        elif self.model_type == "transformer":
            from src.models.transformer_models import TransformerBehaviorPredictor
            model = TransformerBehaviorPredictor(
                input_dim=self.config['model']['transformer']['input_dim'],
                d_model=self.config['model']['transformer']['d_model'],
                num_heads=self.config['model']['transformer']['num_heads'],
                num_layers=self.config['model']['transformer']['num_layers'],
                dropout=self.config['model']['transformer']['dropout'],
                output_dim=self.config['model']['transformer']['output_dim']
            )

        elif self.model_type == "convlstm":
            from src.models.transformer_models import ConvLSTMBehaviorDetector
            model = ConvLSTMBehaviorDetector(
                input_channels=self.config['model']['convlstm']['input_channels'],
                hidden_channels=self.config['model']['convlstm']['hidden_channels'],
                num_layers=self.config['model']['convlstm']['num_layers'],
                dropout=self.config['model']['convlstm']['dropout'],
                output_dim=self.config['model']['convlstm']['output_dim']
            )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        model = model.to(self.device)
        logger.info(f"Model created: {self.model_type}")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        return model

    def create_dummy_dataset(self, num_samples: int = 100, seq_len: int = 30):

        X = torch.randn(num_samples, seq_len, 6)

        y = torch.randint(0, 2, (num_samples, 1)).float()

        dataset = TensorDataset(X, y)
        return dataset

    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                   optimizer: optim.Optimizer, criterion: nn.Module) -> float:

        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc="Training")
        for X, y in pbar:
            X = X.to(self.device)
            y = y.to(self.device)

            outputs, _ = model(X)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': total_loss / num_batches})

        return total_loss / num_batches

    def validate(self, model: nn.Module, val_loader: DataLoader,
                criterion: nn.Module) -> Tuple[float, dict]:

        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(self.device)
                y = y.to(self.device)

                outputs, _ = model(X)
                loss = criterion(outputs, y)

                total_loss += loss.item()
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        from sklearn.metrics import f1_score, roc_auc_score

        preds_binary = (all_preds > 0.5).astype(int)
        f1 = f1_score(all_labels, preds_binary, zero_division=0)

        try:
            auc = roc_auc_score(all_labels, all_preds)
        except:
            auc = 0.0

        metrics = {
            'loss': total_loss / len(val_loader),
            'f1': f1,
            'auc': auc
        }

        return metrics['loss'], metrics

    def train(self, model: nn.Module, train_dataset, val_dataset = None,
             num_epochs: int = None, batch_size: int = None):

        num_epochs = num_epochs or self.config['training']['num_epochs']
        batch_size = batch_size or self.config['training']['batch_size']

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                num_workers=0
            )
        else:
            val_loader = None

        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

        criterion = nn.BCELoss()

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            logger.info(f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.4f}")

            if val_loader:
                val_loss, metrics = self.validate(model, val_loader, criterion)
                logger.info(f"  val_loss={val_loss:.4f}, f1={metrics['f1']:.4f}, auc={metrics['auc']:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_checkpoint(model, epoch, train_loss, metrics)
                else:
                    patience_counter += 1
                    if patience_counter >= self.config['training']['early_stopping']['patience']:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break

        logger.info("Training complete")
        return model

def main():

    parser = argparse.ArgumentParser(description="Train crowd behavior prediction model")
    parser.add_argument("--config", default="configs/model_config.yaml",
                       help="Config file path")
    parser.add_argument("--model_type", default="gnn",
                       choices=["gnn", "transformer", "convlstm"],
                       help="Model type")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--dataset", help="Dataset path")

    args = parser.parse_args()

    manager = TrainingManager(args.config, args.model_type)

    model = manager.build_model()

    dataset = manager.create_dummy_dataset(num_samples=1000)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    manager.train(
        model,
        train_dataset,
        val_dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
