"""
Graph Convolutional Network (GCN) for coordinated account detection.

Implements GCN using PyTorch Geometric for node classification
(IO vs. Control account detection).
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class GCN(torch.nn.Module):
    """Graph Convolutional Network architecture."""

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # Input layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        # Output layer
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final layer (no activation)
        x = self.convs[-1](x, edge_index)
        return x


class GCNModel:
    """GCN training and evaluation wrapper."""

    def __init__(self, config: dict):
        self.config = config
        self.gcn_config = config.get("models", {}).get("gcn", {})
        self.hidden_channels = self.gcn_config.get("hidden_channels", 64)
        self.num_layers = self.gcn_config.get("num_layers", 2)
        self.dropout = self.gcn_config.get("dropout", 0.5)
        self.learning_rate = self.gcn_config.get("learning_rate", 0.001)
        self.epochs = self.gcn_config.get("epochs", 100)
        self.patience = self.gcn_config.get("patience", 10)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._train_history = []

    def _detect_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def build_model(self, in_channels: int, out_channels: int = 2) -> GCN:
        """Build GCN model."""
        self.model = GCN(
            in_channels=in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=out_channels,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        logger.info(f"Built GCN: {in_channels} -> {self.hidden_channels} "
                   f"-> {out_channels}, {self.num_layers} layers")
        return self.model

    def train(self, data: Data, train_mask: torch.Tensor,
              val_mask: torch.Tensor) -> Dict:
        """
        Train the GCN model with early stopping.

        Args:
            data: PyG Data object
            train_mask: Boolean mask for training nodes
            val_mask: Boolean mask for validation nodes

        Returns:
            Training history dict
        """
        if self.model is None:
            self.build_model(data.x.shape[1], num_classes=data.y.unique().shape[0])

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=5e-4)

        # Handle class imbalance
        num_classes = data.y.unique().shape[0]
        class_counts = torch.bincount(data.y[train_mask], minlength=num_classes)
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum() * num_classes
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(self.device))

        data = data.to(self.device)

        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "val_f1": []}

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)
            loss = criterion(out[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                out = self.model(data.x, data.edge_index)
                val_loss = criterion(out[val_mask], data.y[val_mask])

                val_preds = out[val_mask].argmax(dim=1)
                val_true = data.y[val_mask]

                from sklearn.metrics import f1_score
                val_f1 = f1_score(
                    val_true.cpu(), val_preds.cpu(),
                    average="weighted", zero_division=0
                )

            history["train_loss"].append(loss.item())
            history["val_loss"].append(val_loss.item())
            history["val_f1"].append(val_f1)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                # Save best model state
                self._best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch} "
                          f"(best epoch: {best_epoch})")
                break

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs}: "
                          f"train_loss={loss.item():.4f}, "
                          f"val_loss={val_loss.item():.4f}, "
                          f"val_f1={val_f1:.4f}")

        # Restore best model
        if hasattr(self, "_best_state"):
            self.model.load_state_dict(self._best_state)

        self._train_history = history
        logger.info(f"Training complete. Best epoch: {best_epoch}, "
                   f"Best val_loss: {best_val_loss:.4f}")
        return history

    def predict(self, data: Data) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on the full graph.

        Returns:
            predictions: (num_nodes,) array of class predictions
            probabilities: (num_nodes, num_classes) array of probabilities
        """
        self.model.eval()
        data = data.to(self.device)

        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            probs = F.softmax(out, dim=1)
            predictions = out.argmax(dim=1)

        return predictions.cpu().numpy(), probs.cpu().numpy()

    def get_node_embeddings(self, data: Data) -> np.ndarray:
        """Get intermediate node embeddings from the GCN."""
        self.model.eval()
        data = data.to(self.device)

        embeddings = []
        with torch.no_grad():
            x = data.x
            for i in range(self.model.num_layers - 1):
                x = self.model.convs[i](x, data.edge_index)
                x = self.model.bns[i](x)
                x = F.relu(x)
            embeddings = x.cpu().numpy()

        return embeddings

    def evaluate(self, data: Data, mask: torch.Tensor) -> Dict[str, float]:
        """Evaluate the model on a data split."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, average_precision_score
        )

        predictions, probabilities = self.predict(data)
        y_true = data.y[mask].cpu().numpy()
        y_pred = predictions[mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask]
        y_scores = probabilities[mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask]

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        if len(np.unique(y_true)) > 1:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_scores[:, 1])
                metrics["pr_auc"] = average_precision_score(y_true, y_scores[:, 1])
            except (ValueError, IndexError):
                metrics["roc_auc"] = 0.0
                metrics["pr_auc"] = 0.0

        return metrics
