import time
import threading

from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class DeepRoofProgressHook(Hook):
    """Explicit training progress heartbeat for notebook/interactive runs."""

    priority = 'BELOW_NORMAL'

    def __init__(
        self,
        interval: int = 10,
        flush: bool = True,
        heartbeat_sec: int = 30,
    ):
        self.interval = max(int(interval), 1)
        self.flush = bool(flush)
        self.heartbeat_sec = max(int(heartbeat_sec), 5)
        self._start_time = None
        self._stop_event = threading.Event()
        self._heartbeat_thread = None
        self._runner = None
        self._last_iter_seen = -1

    @staticmethod
    def _max_iters(runner) -> int:
        train_loop = getattr(runner, 'train_loop', None)
        if train_loop is not None and hasattr(train_loop, 'max_iters'):
            return int(train_loop.max_iters)
        return int(getattr(runner, 'max_iters', 0))

    def _emit(self, msg: str):
        print(msg, flush=self.flush)

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
            if cur <= self._last_iter_seen:
                self._emit(
                    f'[DeepRoofProgress] alive | iter={cur}/{max_iters} | '
                    f'elapsed={elapsed:.1f}s | waiting_next_batch=1'
                )
            else:
                self._emit(
                    f'[DeepRoofProgress] alive | iter={cur}/{max_iters} | '
                    f'elapsed={elapsed:.1f}s | lr={self._format_lr(runner)}'
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

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None):
        cur_iter = int(batch_idx) + 1
        if cur_iter != 1 and (cur_iter % self.interval) != 0:
            return
        elapsed = 0.0 if self._start_time is None else (time.time() - self._start_time)
        msg = f'[DeepRoofProgress] iter={cur_iter}/{self._max_iters(runner)} | elapsed={elapsed:.1f}s'
        msg += f' | lr={self._format_lr(runner)}'
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
