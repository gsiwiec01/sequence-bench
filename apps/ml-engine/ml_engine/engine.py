from __future__ import annotations
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
from ml_engine.model_registry import BaseRNNModel
from ml_engine.utils import get_gpu_memory_mb

Hidden = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


@dataclass
class MonitorConfig:
    gradient_monitor: Any = None
    gradient_log_interval: int = 1
    gradient_experiment_id: str | None = None
    gradient_save_callback: Callable[[int, Path], None] | None = None
    weight_tracker: Any = None
    weight_log_interval: int = 5
    weight_save_callback: Callable[[int, Path], None] | None = None


class TBPTTEngine:
    def __init__(
            self,
            model: BaseRNNModel,
            optimizer: torch.optim.Optimizer,
            criterion: nn.Module,
            k1: int,
            k2: int,
            device: torch.device,
            task_type: str | None = None,
            max_grad_norm: float = 1.0,
            grad_clip_mode: str = "norm",
            max_grad_value: float | None = None,
            checkpoint_interval: int = 10,
            checkpoint_dir: str = "./checkpoints",
            early_stopping_metric: str = "val_loss",
            early_stopping_mode: str = "min",
            compute_eval_extra: Callable[..., dict[str, float]] | None = None,
            collect_logits: bool = False,
            monitor: MonitorConfig | None = None,
    ) -> None:
        if k1 < 1 or k2 < 1:
            raise ValueError("k1 and k2 must be >= 1")

        if k1 > k2:
            raise ValueError("k1 must be <= k2 (TBPTT convention)")

        if early_stopping_mode not in ("min", "max"):
            raise ValueError("early_stopping_mode must be 'min' or 'max'")

        if grad_clip_mode not in ("norm", "value", "none"):
            raise ValueError("grad_clip_mode must be 'norm', 'value', or 'none'")

        if grad_clip_mode == "value" and max_grad_value is None:
            raise ValueError("max_grad_value required when grad_clip_mode='value'")

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.k1 = k1
        self.k2 = k2
        self.device = device
        self.task_type = task_type or getattr(model, "task_type", "classification")
        self.max_grad_norm = max_grad_norm
        self.grad_clip_mode = grad_clip_mode
        self.max_grad_value = max_grad_value
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_mode = early_stopping_mode
        self.compute_eval_extra = compute_eval_extra
        self.collect_logits = collect_logits

        mc = monitor or MonitorConfig()
        self.gradient_monitor = mc.gradient_monitor
        self.gradient_log_interval = mc.gradient_log_interval
        self.gradient_experiment_id = mc.gradient_experiment_id
        self.gradient_save_callback = mc.gradient_save_callback
        self.weight_tracker = mc.weight_tracker
        self.weight_log_interval = mc.weight_log_interval
        self.weight_save_callback = mc.weight_save_callback

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
        last_weight_save: int = -1
        final_epoch: int = -1

        for epoch in range(max_epochs):
            final_epoch = epoch
            t0 = time.perf_counter()

            monitor_active = (self.gradient_monitor is not None and epoch % self.gradient_log_interval == 0)
            if monitor_active:
                self.gradient_monitor.attach()

            train_metrics = self._train_epoch(
                train_loader,
                monitor=self.gradient_monitor if monitor_active else None,
                _debug_nan=getattr(self, "_debug_nan", False),
            )

            if monitor_active:
                saved_path = self.gradient_monitor.save(epoch, self.gradient_experiment_id)
                self.gradient_monitor.detach()
                if self.gradient_save_callback is not None:
                    self.gradient_save_callback(epoch, saved_path)

            val_metrics = self._eval_epoch(val_loader)

            epoch_metrics: dict[str, float] = {
                **train_metrics,
                **val_metrics,
                "epoch": float(epoch),
                "epoch_time_s": time.perf_counter() - t0,
                "gpu_memory_mb": get_gpu_memory_mb(),
            }
            history.append(epoch_metrics)

            if callback is not None:
                callback(epoch, epoch_metrics)

            if (epoch + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(epoch)

            if self.weight_tracker is not None:
                self.weight_tracker.step()
                if (epoch + 1) % self.weight_log_interval == 0:
                    saved_wt = self.weight_tracker.save(epoch)
                    last_weight_save = epoch
                    if self.weight_save_callback is not None:
                        self.weight_save_callback(epoch, saved_wt)

            current = epoch_metrics.get(self.early_stopping_metric)
            if current is not None:
                if (
                        best_value is None
                        or (self.early_stopping_mode == "min" and current < best_value)
                        or (self.early_stopping_mode == "max" and current > best_value)
                ):
                    best_value = current
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        break

        if self.weight_tracker is not None and final_epoch >= 0 and last_weight_save < final_epoch:
            saved_wt = self.weight_tracker.save(final_epoch)
            if self.weight_save_callback is not None:
                self.weight_save_callback(final_epoch, saved_wt)

        return history

    def _train_epoch(self, dataloader: Iterable[Any], monitor: Any = None, _debug_nan: bool = False) -> dict[
        str, float]:
        self.model.train()
        total_loss = 0.0
        total_steps = 0
        total_correct = 0
        total_count = 0
        grad_norms: list[float] = []
        _nan_windows = 0
        _total_windows = 0

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
                _total_windows += 1

                if not loss.isnan():
                    norm = self._backward_step(loss)
                    grad_norms.append(norm)
                    if monitor is not None:
                        monitor.step()
                    total_loss += loss.item()
                    total_steps += 1
                    c, n = self._compute_correct_total(output.detach(), chunk_y)
                    total_correct += c
                    total_count += n
                else:
                    _nan_windows += 1

                continue

            if self.task_type in ("classification", "regression"):
                h = self.model.init_hidden(batch_size, self.device)
                tail_start = seq_len - self.k2
                if tail_start > 0:
                    with torch.no_grad():
                        _, h = self.model(batch_x[:, :tail_start], h)

                chunk_x = batch_x[:, tail_start:]
                chunk_y = self._slice_target(batch_y, tail_start, seq_len, seq_len)
                output, _ = self.model(chunk_x, h)
                loss = self._compute_loss(output, chunk_y)
                _total_windows += 1

                if not loss.isnan():
                    norm = self._backward_step(loss)
                    grad_norms.append(norm)
                    if monitor is not None:
                        monitor.step()
                    total_loss += loss.item()
                    total_steps += 1
                    c, n = self._compute_correct_total(output.detach(), chunk_y)
                    total_correct += c
                    total_count += n
                else:
                    _nan_windows += 1
                continue

            h: Hidden = self.model.init_hidden(batch_size, self.device)
            h_new: Hidden = h
            t = 0
            while t + self.k2 <= seq_len:
                chunk_x = batch_x[:, t: t + self.k2]
                chunk_y = self._slice_target(batch_y, t, t + self.k2, seq_len)

                output, h_new = self.model(chunk_x, h)
                loss = self._compute_loss(output, chunk_y)
                _total_windows += 1

                if not loss.isnan():
                    norm = self._backward_step(loss)
                    grad_norms.append(norm)
                    if monitor is not None:
                        monitor.step()
                    total_loss += loss.item()
                    total_steps += 1
                    c, n = self._compute_correct_total(output.detach(), chunk_y)
                    total_correct += c
                    total_count += n
                else:
                    _nan_windows += 1

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

            last_covered = t + self.k2
            if last_covered < seq_len:
                tail_start = seq_len - self.k2
                h_tail = self._detach_hidden(h_new)
                if tail_start > last_covered:
                    with torch.no_grad():
                        _, h_tail = self.model(batch_x[:, last_covered:tail_start], h_tail)

                chunk_x = batch_x[:, tail_start:]
                chunk_y = self._slice_target(batch_y, tail_start, seq_len, seq_len)
                output, _ = self.model(chunk_x, h_tail)
                loss = self._compute_loss(output, chunk_y)
                _total_windows += 1

                if not loss.isnan():
                    norm = self._backward_step(loss)
                    grad_norms.append(norm)
                    if monitor is not None:
                        monitor.step()
                    total_loss += loss.item()
                    total_steps += 1
                    c, n = self._compute_correct_total(output.detach(), chunk_y)
                    total_correct += c
                    total_count += n
                else:
                    _nan_windows += 1

        if _debug_nan:
            rate = _nan_windows / max(_total_windows, 1)

        metrics: dict[str, float] = {"train_loss": total_loss / max(total_steps, 1)}
        if total_count > 0:
            metrics["train_accuracy"] = total_correct / total_count

        if grad_norms:
            metrics["grad_norm_mean"] = float(np.mean(grad_norms))
            metrics["grad_norm_max"] = float(np.max(grad_norms))

        metrics["learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
        return metrics

    def _eval_epoch(self, dataloader: Iterable[Any]) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_steps = 0
        total_correct = 0
        total_count = 0

        collect = self.compute_eval_extra is not None
        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        all_logits: list[np.ndarray] | None = [] if (collect and self.collect_logits) else None

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
                c, n = self._compute_correct_total(output, chunk_y)
                total_correct += c
                total_count += n

                if collect:
                    self._collect_preds(output, chunk_y, all_preds, all_targets, all_logits)

        metrics: dict[str, float] = {"val_loss": total_loss / max(total_steps, 1)}
        if total_count > 0:
            metrics["val_accuracy"] = total_correct / total_count

        if collect and all_targets:
            preds_np = np.concatenate(all_preds) if all_preds else np.array([])
            targets_np = np.concatenate(all_targets) if all_targets else np.array([])
            logits_np = np.concatenate(all_logits) if all_logits else None
            extra = self.compute_eval_extra(preds_np, targets_np, logits_np)  # type: ignore[misc]
            metrics.update(extra)

        return metrics

    def _collect_preds(
            self,
            output: torch.Tensor,
            target: Any,
            all_preds: list[np.ndarray],
            all_targets: list[np.ndarray],
            all_logits: list[np.ndarray] | None,
    ) -> None:
        task_type = self.task_type
        ignore_index = getattr(self.criterion, "ignore_index", -100)

        if task_type in ("regression", "forecasting"):
            all_preds.append(output.reshape(-1).cpu().numpy())
            if isinstance(target, torch.Tensor):
                all_targets.append(target.reshape(-1).cpu().numpy())
            return

        if output.dim() == 3:
            out_flat = output.reshape(-1, output.size(-1))
            tgt_flat = target.reshape(-1) if (isinstance(target, torch.Tensor) and target.dim() == 2) else target
        else:
            out_flat = output
            tgt_flat = target

        if not isinstance(tgt_flat, torch.Tensor):
            return

        mask = tgt_flat != ignore_index
        pred_flat = out_flat.argmax(dim=-1)
        all_preds.append(pred_flat[mask].cpu().numpy())
        all_targets.append(tgt_flat[mask].cpu().numpy())
        if all_logits is not None:
            all_logits.append(out_flat[mask].cpu().numpy())

    def _backward_step(self, loss: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        loss.backward()

        if self.grad_clip_mode == "norm":
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        elif self.grad_clip_mode == "value":
            total_norm = self._compute_grad_norm()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.max_grad_value)
        else:  # "none"
            total_norm = self._compute_grad_norm()

        self.optimizer.step()

        return float(total_norm)

    def _compute_grad_norm(self) -> float:
        total = sum(
            p.grad.detach().norm(2).item() ** 2
            for p in self.model.parameters()
            if p.grad is not None
        )
        return total ** 0.5

    def _compute_correct_total(self, output: torch.Tensor, target: Any) -> tuple[int, int]:
        task_type = self.task_type
        if task_type == "regression" or not isinstance(target, torch.Tensor):
            return 0, 0

        pred = output.argmax(dim=-1)
        ignore_index = getattr(self.criterion, "ignore_index", -100)
        mask = target != ignore_index
        total = int(mask.sum().item())
        if total == 0:
            return 0, 0

        correct = int(((pred == target) & mask).sum().item())
        return correct, total

    def _compute_loss(self, output: torch.Tensor, target: Any) -> torch.Tensor:
        task_type = self.task_type
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
