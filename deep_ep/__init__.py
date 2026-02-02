import os
import torch

from .utils import EventOverlap


def _detect_backend():
    backend_env = os.environ.get('DEEP_EP_BACKEND', '').lower()
    if backend_env in ('cuda', 'rocm'):
        return backend_env

    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        return 'rocm'

    if os.environ.get('ROCM_PATH') or os.environ.get('HIP_PATH'):
        return 'rocm'

    return 'cuda'


_backend = _detect_backend()

if _backend == 'rocm':
    from .buffer_rocm import BufferROCm as Buffer, Config
    from mori.cpp import topk_idx_t
else:
    from .buffer import Buffer
    # noinspection PyUnresolvedReferences
    from deep_ep_cpp import Config, topk_idx_t
