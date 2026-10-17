import logging

def get_pylogger(name: str = __name__) -> logging.Logger:
    """HW4 inference uses one process and does not require Lightning logging."""
    return logging.getLogger(name)
