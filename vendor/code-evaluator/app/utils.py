import os
import signal
from multiprocessing import process

from loguru import logger


def kill_proc(p: process.BaseProcess):
    if not p:
        return
    if p.is_alive():
        p.terminate()
        p.join(0.1)
    if p.is_alive():
        try:
            if p.pid is not None:
                os.kill(p.pid, signal.SIGKILL)
        except Exception:
            logger.debug(f"failed to kill subprocess: {p.pid}")
        p.join(0.1)
    try:
        p.close()
    except Exception:
        logger.debug(f"failed to close subprocess: {p.pid}")
