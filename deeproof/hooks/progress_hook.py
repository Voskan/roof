import time
import threading

from mmengine.hooks import Hook
from mmengine.registry import HOOKS
try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@HOOKS.register_module()
class DeepRoofProgressHook(Hook):
    """Explicit training progress heartbeat for notebook/interactive runs."""

    priority = 'BELOW_NORMAL'

    def __init__(
        self,
        interval: int = 10,
        flush: bool = True,
        heartbeat_sec: int = 30,
        dataloader_warn_sec: int = 90,
    ):
        self.interval = max(int(interval), 1)
        self.flush = bool(flush)
        self.heartbeat_sec = max(int(heartbeat_sec), 5)
        self.dataloader_warn_sec = max(int(dataloader_warn_sec), 30)
        self._start_time = None
        self._stop_event = threading.Event()
        self._heartbeat_thread = None
        self._runner = None
        self._last_iter_seen = -1
        self._dataloader_warned = False
        self._ever_seen_before_train_iter = False
        self._in_train_iter = False
        self._active_iter = -1
        self._iter_start_time = None
        self._iter_warned = False

    @staticmethod
    def _max_iters(runner) -> int:
        train_loop = getattr(runner, 'train_loop', None)
        if train_loop is not None and hasattr(train_loop, 'max_iters'):
            return int(train_loop.max_iters)
        return int(getattr(runner, 'max_iters', 0))

    def _emit(self, msg: str):
        print(msg, flush=self.flush)

    def _gpu_mem_str(self) -> str:
        if torch is None or (not torch.cuda.is_available()):
            return ''
        try:
            dev = torch.cuda.current_device()
            total = float(torch.cuda.get_device_properties(dev).total_memory)
            alloc = float(torch.cuda.memory_allocated(dev))
            reserved = float(torch.cuda.memory_reserved(dev))
            util_alloc = 100.0 * (alloc / max(total, 1.0))
            util_reserved = 100.0 * (reserved / max(total, 1.0))
            return (
                f' | gpu_alloc={alloc / (1024 ** 3):.1f}/{total / (1024 ** 3):.1f}GB ({util_alloc:.0f}%)'
                f' | gpu_reserved={reserved / (1024 ** 3):.1f}/{total / (1024 ** 3):.1f}GB ({util_reserved:.0f}%)'
            )
        except Exception:
            return ''

    def _format_lr(self, runner) -> str:
        try:
            lr_obj = runner.optim_wrapper.get_lr()
        except Exception:
            return 'N/A'
        if isinstance(lr_obj, dict):
            if not lr_obj:
                return 'N/A'
            first_key = next(iter(lr_obj))
            vals = lr_obj[first_key]
            if isinstance(vals, (list, tuple)) and vals:
                return f'{float(vals[0]):.3e}'
            return f'{float(vals):.3e}'
        if isinstance(lr_obj, (list, tuple)) and lr_obj:
            return f'{float(lr_obj[0]):.3e}'
        try:
            return f'{float(lr_obj):.3e}'
        except Exception:
            return 'N/A'

    def _heartbeat_loop(self):
        # Time-based heartbeat so user sees liveness even when first iter is slow.
        while not self._stop_event.wait(self.heartbeat_sec):
            runner = self._runner
            if runner is None:
                continue
            train_loop = getattr(runner, 'train_loop', None)
            cur = int(getattr(train_loop, '_iter', 0)) if train_loop is not None else 0
            max_iters = self._max_iters(runner)
            elapsed = 0.0 if self._start_time is None else (time.time() - self._start_time)
            in_iter = self._in_train_iter and self._active_iter == cur
            if in_iter:
                iter_elapsed = 0.0 if self._iter_start_time is None else (time.time() - self._iter_start_time)
                self._emit(
                    f'[DeepRoofProgress] alive | iter={cur}/{max_iters} | '
                    f'elapsed={elapsed:.1f}s | running_train_step=1 | step_elapsed={iter_elapsed:.1f}s'
                    f'{self._gpu_mem_str()}'
                )
                if (not self._iter_warned) and cur == 0 and iter_elapsed >= float(self.dataloader_warn_sec):
                    self._emit(
                        '[DeepRoofProgress] WARNING: first train step is taking too long. '
                        'Data is already loaded; bottleneck is likely model compute/memory. '
                        'Try lower batch size / image size, and check GPU memory.'
                    )
                    self._iter_warned = True
            elif cur <= self._last_iter_seen:
                self._emit(
                    f'[DeepRoofProgress] alive | iter={cur}/{max_iters} | '
                    f'elapsed={elapsed:.1f}s | waiting_next_batch=1'
                    f'{self._gpu_mem_str()}'
                )
                if ((not self._dataloader_warned)
                        and (not self._ever_seen_before_train_iter)
                        and cur == 0
                        and elapsed >= float(self.dataloader_warn_sec)):
                    self._emit(
                        '[DeepRoofProgress] WARNING: first batch is still not ready. '
                        'Likely dataloader stall (workers/path/augmentation). '
                        'Try num_workers=0 and persistent_workers=False for notebook runs.'
                    )
                    self._dataloader_warned = True
            else:
                self._emit(
                    f'[DeepRoofProgress] alive | iter={cur}/{max_iters} | '
                    f'elapsed={elapsed:.1f}s | lr={self._format_lr(runner)}'
                    f'{self._gpu_mem_str()}'
                )
                self._last_iter_seen = cur

    def before_train(self, runner):
        self._start_time = time.time()
        self._runner = runner
        self._stop_event.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name='deeproof-progress-heartbeat')
        self._heartbeat_thread.start()
        self._emit(
            f'[DeepRoofProgress] TRAIN START | max_iters={self._max_iters(runner)} | work_dir={runner.work_dir}'
        )
        self._emit(
            f'[DeepRoofProgress] logger_interval={self.interval} | heartbeat_sec={self.heartbeat_sec} | '
            f'lr={self._format_lr(runner)}'
        )
        self._dataloader_warned = False
        self._ever_seen_before_train_iter = False
        self._in_train_iter = False
        self._active_iter = -1
        self._iter_start_time = None
        self._iter_warned = False

    def before_train_iter(self, runner, batch_idx: int, data_batch=None):
        self._ever_seen_before_train_iter = True
        self._in_train_iter = True
        self._active_iter = int(batch_idx)
        self._iter_start_time = time.time()

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None):
        self._in_train_iter = False
        self._active_iter = -1
        self._iter_start_time = None
        cur_iter = int(batch_idx) + 1
        if cur_iter != 1 and (cur_iter % self.interval) != 0:
            return
        elapsed = 0.0 if self._start_time is None else (time.time() - self._start_time)
        msg = f'[DeepRoofProgress] iter={cur_iter}/{self._max_iters(runner)} | elapsed={elapsed:.1f}s'
        msg += f' | lr={self._format_lr(runner)}'
        msg += self._gpu_mem_str()
        if isinstance(outputs, dict) and 'loss' in outputs:
            try:
                loss_val = float(outputs['loss'])
                msg += f' | loss={loss_val:.4f}'
            except Exception:
                pass
        self._emit(msg)
        self._last_iter_seen = cur_iter

    def before_val(self, runner):
        self._emit('[DeepRoofProgress] VALIDATION START')

    def after_val(self, runner):
        self._emit('[DeepRoofProgress] VALIDATION END')

    def after_train(self, runner):
        elapsed = 0.0 if self._start_time is None else (time.time() - self._start_time)
        self._emit(f'[DeepRoofProgress] TRAIN END | total_elapsed={elapsed:.1f}s')
        self._stop_event.set()

    def after_run(self, runner):
        self._stop_event.set()
