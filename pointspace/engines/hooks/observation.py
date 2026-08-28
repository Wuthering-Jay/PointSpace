"""OC-HPSD 单次训练 curriculum hook。"""

from .builder import HOOKS
from .default import HookBase


@HOOKS.register_module()
class ObservationCurriculumHook(HookBase):
    """将可由 epoch/iteration 重建的归一化进度传给模型。"""

    def __init__(self, strict=True):
        self.strict = bool(strict)
        self._model = None

    def before_train(self):
        model = self.trainer.model
        self._model = model.module if hasattr(model, "module") else model
        if not hasattr(self._model, "set_train_progress"):
            if self.strict:
                raise AttributeError(
                    "ObservationCurriculumHook requires model.set_train_progress()"
                )
            self._model = None
            return
        self._model.set_train_progress(0.0)

    def before_step(self):
        if self._model is None:
            return
        steps_per_epoch = len(self.trainer.train_loader)
        total_steps = self.trainer.max_epoch * steps_per_epoch
        current_step = self.trainer.epoch * steps_per_epoch + int(
            self.trainer.comm_info["iter"]
        )
        progress = current_step / max(total_steps - 1, 1)
        self._model.set_train_progress(progress)
