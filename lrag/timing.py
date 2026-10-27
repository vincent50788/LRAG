"""Prints the time that each step takes. Use it to see where the time goes.

The timer (T0) starts when Python first imports this module.
rag.py imports this module first. Thus the timer also measures the heavy imports after it, such as torch.
"""
import time

T0 = _last = time.perf_counter()


def tick(step):
    """Print the time of this step and the total time since the start."""
    global _last
    now = time.perf_counter()
    print(f"Elapsed time: +{now - _last:5.2f}s | {now - T0:6.2f}s  {step}", flush=True)
    _last = now
