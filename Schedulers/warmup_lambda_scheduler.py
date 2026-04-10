from functools import partial
from torch.optim.lr_scheduler import LRScheduler


def _warmup_lr_lambda(step, warmup_steps):
    return min(step / warmup_steps, 1.0)


class WarmupLambdaLR(LRScheduler):
    """Linear warmup scheduler.

    lr increases linearly from 0 to learning_rate over warmup_steps,
    then stays fixed at learning_rate.

        lr_t = base_lr * min(t / warmup_steps, 1.0)

    Args:
        optimizer:    wrapped optimizer
        warmup_steps: number of steps to warm up over
    """

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.lr_lambda = partial(
            _warmup_lr_lambda,
            warmup_steps=warmup_steps,
        )
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch
        factor = self.lr_lambda(t)
        return [base_lr * factor for base_lr in self.base_lrs]