"""Process memory monitoring helpers used to skip memory-intensive models."""

import logging
from typing import Optional, Tuple

from .config import PipelineConfig


def monitor_memory(step_name: str, log_warning: bool = False) -> Tuple[Optional[float], Optional[float]]:
    try:
        import psutil
        info = psutil.Process().memory_info()
        rss_mb = info.rss / 1024 ** 2
        vms_mb = info.vms / 1024 ** 2
        fn = logging.warning if log_warning else logging.info
        fn(f"[Memory{'Warning' if log_warning else ''}] {step_name}: RSS={rss_mb:.1f}MB, VMS={vms_mb:.1f}MB")
        return rss_mb, vms_mb
    except ImportError:
        msg = f"[Memory{'Warning' if log_warning else ''}] {step_name}: psutil not available"
        (logging.warning if log_warning else logging.info)(msg)
        return None, None


def _check_memory_skip(config: PipelineConfig) -> bool:
    """Return True if memory-intensive models should be skipped this fold."""
    if not config._raw_get('memory_safety.monitor_memory', True):
        return False
    rss_mb, _ = monitor_memory("fold memory check")
    if rss_mb is None:  # psutil unavailable — safe default
        return False
    max_mb = config._raw_get('memory_safety.max_memory_mb', 8000)
    if rss_mb > max_mb and config._raw_get('memory_safety.skip_memory_intensive_models', True):
        logging.warning(f"High memory ({rss_mb:.0f}MB > {max_mb}MB): skipping BGM/RF this fold")
        return True
    return False


# ============================================================================
# VCF PARSING HELPERS
# ============================================================================


