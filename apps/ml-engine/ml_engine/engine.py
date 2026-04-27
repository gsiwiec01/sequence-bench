from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ml_engine.model_registry import BaseRNNModel
from ml_engine.utils import get_gpu_memory_mb

Hidden = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


class TBPTTEngine:
    def __init__(
        self,
        model: BaseRNNModel,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        k1: int,
        k2: int,
        device: torch.device,
        max_grad_norm: float = 1.0,
        gradient_monitor: Any = None,
        gradient_log_interval: int = 10,
        checkpoint_interval: int = 10,
        checkpoint_dir: str = "./checkpoints",
        early_stopping_metric: str = "val_loss",
        early_stopping_mode: str = "min",
    ) -> None:
        if k1 < 1 or k2 < 1:
            raise ValueError("k1 and k2 must be >= 1")
        if k1 > k2:
            raise ValueError("k1 must be <= k2 (TBPTT convention)")
        if early_stopping_mode not in ("min", "max"):
            raise ValueError("early_stopping_mode must be 'min' or 'max'")

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.k1 = k1
        self.k2 = k2
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.gradient_monitor = gradient_monitor
        self.gradient_log_interval = gradient_log_interval
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_mode = early_stopping_mode

    def train(
        self,
        train_loader: Iterable[Any],
        val_loader: Iterable[Any],
        max_epochs: int,
        early_stopping_patience: int = 10,
        callback: Callable[[int, dict[str, float]], None] | None = None,
    ) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        best_value: float | None = None
        patience_counter = 0

        for epoch in range(max_epochs):
            t0 = time.perf_counter()

            monitor_active = (
                self.gradient_monitor is not None
                and epoch % self.gradient_log_interval == 0
            )
            if monitor_active:
                self.gradient_monitor.attach()

            train_metrics = self._train_epoch(
                train_loader,
                monitor=self.gradient_monitor if monitor_active else None,
            )

            if monitor_active:
                self.gradient_monitor.save(epoch)
                self.gradient_monitor.detach()

            val_metrics = self._eval_epoch(val_loader)

            epoch_metrics: dict[str, float] = {
                **train_metrics,
                **val_metrics,
                "epoch": float(epoch),
                "epoch_time": time.perf_counter() - t0,
                "gpu_memory_mb": get_gpu_memory_mb(),
            }
            history.append(epoch_metrics)

            if callback is not None:
                callback(epoch, epoch_metrics)

            if (epoch + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(epoch)

            current = epoch_metrics.get(self.early_stopping_metric)
            if current is not None:
                improved = (
                    best_value is None
                    or (self.early_stopping_mode == "min" and current < best_value)
                    or (self.early_stopping_mode == "max" and current > best_value)
                )
                if improved:
                    best_value = current
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        break

        return history

    def _train_epoch(self, dataloader: Iterable[Any], monitor: Any = None) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_steps = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            if isinstance(batch_y, torch.Tensor):
                batch_y = batch_y.to(self.device)

            seq_len = batch_x.size(1)
            batch_size = batch_x.size(0)

            if seq_len < self.k2:
                h = self.model.init_hidden(batch_size, self.device)
                chunk_y = self._slice_target(batch_y, 0, seq_len, seq_len)
                output, _ = self.model(batch_x, h)
                loss = self._compute_loss(output, chunk_y)
                self._backward_step(loss)
                if monitor is not None:
                    monitor.step()
                total_loss += loss.item()
                total_steps += 1
                continue

            h: Hidden = self.model.init_hidden(batch_size, self.device)
            t = 0
            while t + self.k2 <= seq_len:
                chunk_x = batch_x[:, t : t + self.k2]
                chunk_y = self._slice_target(batch_y, t, t + self.k2, seq_len)

                output, h_new = self.model(chunk_x, h)
                loss = self._compute_loss(output, chunk_y)
                self._backward_step(loss)
                if monitor is not None:
                    monitor.step()

                total_loss += loss.item()
                total_steps += 1

                next_t = t + self.k1
                if next_t + self.k2 > seq_len:
                    break

                if self.k1 == self.k2:
                    h = self._detach_hidden(h_new)
                else:
                    h_detached = self._detach_hidden(h)
                    with torch.no_grad():
                        _, h = self.model(batch_x[:, t:next_t], h_detached)

                t = next_t

        return {"train_loss": total_loss / max(total_steps, 1)}

    def _eval_epoch(self, dataloader: Iterable[Any]) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_steps = 0

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                if isinstance(batch_y, torch.Tensor):
                    batch_y = batch_y.to(self.device)

                seq_len = batch_x.size(1)
                h = self.model.init_hidden(batch_x.size(0), self.device)
                output, _ = self.model(batch_x, h)
                chunk_y = self._slice_target(batch_y, 0, seq_len, seq_len)
                loss = self._compute_loss(output, chunk_y)
                total_loss += loss.item()
                total_steps += 1

        return {"val_loss": total_loss / max(total_steps, 1)}

    def _backward_step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def _compute_loss(self, output: torch.Tensor, target: Any) -> torch.Tensor:
        task_type = getattr(self.model, "task_type", "classification")
        if task_type in ("seq2seq", "language_model") and output.dim() == 3:
            output_flat = output.reshape(-1, output.size(-1))

            if isinstance(target, torch.Tensor) and target.dim() >= 2:
                target_flat: Any = (
                    target.reshape(-1)
                    if target.dim() == 2
                    else target.reshape(-1, target.size(-1))
                )
            else:
                target_flat = target

            return self.criterion(output_flat, target_flat)

        return self.criterion(output, target)

    @staticmethod
    def _slice_target(target: Any, start: int, end: int, seq_len: int) -> Any:
        if not isinstance(target, torch.Tensor):
            return target

        if target.dim() >= 2 and target.size(1) == seq_len:
            return target[:, start:end]

        return target

    @staticmethod
    def _detach_hidden(h: Hidden) -> Hidden:
        if isinstance(h, tuple):
            return tuple(t.detach() for t in h)  # type: ignore[return-value]

        return h.detach()

    def _save_checkpoint(self, epoch: int) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(self.model.state_dict(), path)
