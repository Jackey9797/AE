from pathlib import Path, PurePath
from typing import Callable, Generic, Optional, TypeVar, Union

import torch

_T = TypeVar('_T')


class TensorStorage(Generic[_T]):
    def __init__(self,
                 path: Union[str, PurePath],
                 tensor_getter: Callable[[], _T],) -> None:
        self.path = Path(path)
        self.get_tensor = tensor_getter

    def _save(self, tensor: _T):
        torch.save(tensor, self.path)

    def _load(self, device: Optional[str] = None) -> Optional[_T]:
        if self.path.exists():
            tensor: _T = torch.load(str(self.path), map_location=device)
            return tensor
        return None

    def get(self, device: Optional[str] = None):
        tensor = self._load(device)
        if tensor is None:
            tensor = self.get_tensor()
            self._save(tensor)
        return tensor
