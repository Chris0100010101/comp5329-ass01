from Schedulers.cosine_scheduler import CosineAnnealingLR
from Schedulers.lambda_scheduler import LambdaLR
from Schedulers.step_scheduler import StepLR
from Schedulers.warmup_lambda_scheduler import WarmupLambdaLR
from functools import partial

def _constant_lr(step, learning_rate):
    return learning_rate

def cosine_scheduler(optimizer, args):
    return CosineAnnealingLR(
        optimizer,
        T_max=args.num_steps,
    )

def step_scheduler(optimizer, args):
    return StepLR(
        optimizer,
        step_size=getattr(args, "lr_step_size", 10000),
        gamma=getattr(args, "lr_gamma", 0.5),
    )

def lambda_scheduler(optimizer, args):
    fn = partial(_constant_lr, learning_rate=args.learning_rate)
    return LambdaLR(optimizer, lr_lambda=fn)

def warmup_lambda_scheduler(optimizer, args):
    return WarmupLambdaLR(
        optimizer,
        warmup_steps=getattr(args, "warmup_steps", 1000),
    )

schedulers = {
    "cosine":        cosine_scheduler,
    "step":          step_scheduler,
    "lambda":        lambda_scheduler,
    "warmup_lambda": warmup_lambda_scheduler,
}