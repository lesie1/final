"""Typed worker failures; filesystem paths and arbitrary log text are not evidence."""
import errno
import re

RESOURCE_EXIT_CODE = 75


class ResourceLimitError(RuntimeError):
    """An explicit controller check found insufficient runtime resources."""


class WorkerFailure(ValueError):
    def __init__(self, message, category='training', exception_type=None, returncode=None):
        super().__init__(message)
        self.category=category
        self.exception_type=exception_type
        self.returncode=returncode


def classify(exc):
    if isinstance(exc,(ResourceLimitError,MemoryError,TimeoutError)):
        return 'resource'
    if isinstance(exc,OSError) and exc.errno in (errno.ENOMEM,errno.ENOSPC,errno.EMFILE,errno.ENFILE):
        return 'resource'
    # XGBoost reports CUDA allocation failures as XGBoostError, not MemoryError.
    # Match specific error signatures only, never the words "memory" or "cuda".
    if type(exc).__name__=='XGBoostError' and type(exc).__module__.startswith('xgboost'):
        if re.search(r'cudaErrorMemoryAllocation|CUDA_ERROR_OUT_OF_MEMORY|out of memory|std::bad_alloc|'
                     r'cudaErrorInsufficientDriver|CUDA driver version is insufficient|'
                     r'no CUDA-capable device|invalid device ordinal',str(exc),re.IGNORECASE):
            return 'resource'
    return 'training'
