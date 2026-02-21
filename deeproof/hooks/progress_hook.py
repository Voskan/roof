import time

from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class DeepRoofProgressHook(Hook):
    """Explicit training progress heartbeat for notebook/interactive runs."""

    priority = 'BELOW_NORMAL'

    def __init__(self, interval: int = 10, flush: bool = True):
        self.interval = max(int(interval), 1)
        self.flush = bool(flush)
        self._start_time = None

    @staticmethod
    def _max_iters(runner) -> int:
        train_loop = getattr(runner, 'train_loop', None)
        if train_loop is not None and hasattr(train_loop, 'max_iters'):
            return int(train_loop.max_iters)
        return int(getattr(runner, 'max_iters', 0))

    def _emit(self, msg: str):
        print(msg, flush=self.flush)

    def before_train(self, runner):
        self._start_time = time.time()
        self._emit(
            f'[DeepRoofProgress] TRAIN START | max_iters={self._max_iters(runner)} | work_dir={runner.work_dir}'
        )

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None):
        cur_iter = int(batch_idx) + 1
        if cur_iter != 1 and (cur_iter % self.interval) != 0:
            return
        elapsed = 0.0 if self._start_time is None else (time.time() - self._start_time)
        msg = f'[DeepRoofProgress] iter={cur_iter}/{self._max_iters(runner)} | elapsed={elapsed:.1f}s'
        if isinstance(outputs, dict) and 'loss' in outputs:
            try:
                loss_val = float(outputs['loss'])
                msg += f' | loss={loss_val:.4f}'
            except Exception:
                pass
        self._emit(msg)

    def before_val(self, runner):
        self._emit('[DeepRoofProgress] VALIDATION START')

    def after_val(self, runner):
        self._emit('[DeepRoofProgress] VALIDATION END')

    def after_train(self, runner):
        elapsed = 0.0 if self._start_time is None else (time.time() - self._start_time)
        self._emit(f'[DeepRoofProgress] TRAIN END | total_elapsed={elapsed:.1f}s')
