from __future__ import annotations
import math
from typing import Callable, Any
from tinygrad import Tensor
from tinygrad.nn import Linear
from tinygrad.nn.state import get_state_dict

class LoRALinear:
  """
  Linear layer with LoRA.

  - Paper: https://arxiv.org/abs/2106.09685

  ```python exec="true" source="above" session="tensor" result="python"
  from tinygrad import nn
  layer = nn.Linear(64, 32)
  lora_layer = LoRALinear.from_linear(layer, r=8, alpha=16)
  ```
  Covert an existing model to LoRA
  ```python exec="true" source="above" session="tensor" result="python"
  from extra.lora import apply_lora
  apply_lora(model, r=16, alpha=32)
  ```
  """
  def __init__(self, in_features:int, out_features:int, r:int=8, alpha:int=16, bias:bool=True):
    self.in_features, self.out_features = in_features, out_features
    self.r, self.alpha, self.scale = r, alpha, alpha / r if r > 0 else 0.0
    bound = 1 / math.sqrt(in_features)
    self.weight = Tensor.uniform(out_features, in_features, low=-bound, high=bound).is_param_(False)
    self.bias = (Tensor.uniform(out_features, low=-bound, high=bound).is_param_(False) if bias else None)
    if r > 0:
      self.lora_A:Tensor|None = Tensor.kaiming_uniform(r, in_features, a=(1/math.sqrt(r)))
      self.lora_B:Tensor|None = Tensor.zeros(out_features, r)
    else: self.lora_A, self.lora_B = None, None

  @classmethod
  def from_linear(cls, linear:Linear, r:int=8, alpha:int=16):
    out_features, in_features = int(linear.weight.shape[0]), int(linear.weight.shape[1])
    lora = cls(in_features, out_features, r, alpha, bias=linear.bias is not None)
    lora.weight = linear.weight.is_param_(False)
    if linear.bias is not None: lora.bias = linear.bias.is_param_(False)
    return lora

  def __call__(self, x:Tensor) -> Tensor:
    base = x.linear(self.weight.T, self.bias)
    if isinstance(self.lora_A, Tensor) and isinstance(self.lora_B, Tensor):
      base += (x @ self.lora_A.T @ self.lora_B.T) * self.scale
    return base

  def merge(self):
    assert self.lora_A is not None and self.lora_B is not None
    self.weight = (self.weight + (self.lora_B @ self.lora_A) * self.scale).is_param_(False)
    self.lora_A, self.lora_B, self.r = None, None, 0

def _swap_linear_with_lora(model, r:int, alpha:int, target_cls:type[Linear],
                           module_filter_fn:Callable[[Any,str],bool]|None, fqn:str,
                           parent:Any|None, attr_name:str, visited:set):
  # TODO: Could this function be combined with extra/fp8/fp8_linear.py::_swap_linear_with_fp8?
  if id(model) in visited: return
  visited.add(id(model))
  if isinstance(model, (str, int, float, bool, type(None), Tensor)): return
  elif isinstance(model, target_cls):
    if module_filter_fn is not None and not module_filter_fn(model, fqn): return
    lora = LoRALinear.from_linear(model, r, alpha)
    if parent is not None and attr_name: setattr(parent, attr_name, lora)
  elif isinstance(model, list):
    for i, item in enumerate(model):
      child_fqn = f"{fqn}.{i}" if fqn else str(i)
      if isinstance(item, target_cls) and (module_filter_fn is None or module_filter_fn(item, child_fqn)):
        model[i] = LoRALinear.from_linear(item, r, alpha)
      else: _swap_linear_with_lora(item, r, alpha, target_cls, module_filter_fn, child_fqn, None, "", visited)
  elif isinstance(model, dict):
    for key, item in list(model.items()):
      child_fqn = f"{fqn}.{key}" if fqn else str(key)
      if isinstance(item, target_cls) and (module_filter_fn is None or module_filter_fn(item, child_fqn)):
        model[key] = LoRALinear.from_linear(item, r, alpha)
      else: _swap_linear_with_lora(item, r, alpha, target_cls, module_filter_fn, child_fqn, None, "", visited)
  elif hasattr(model, "__dict__"):
    for attr_key in list(vars(model).keys()):
      try: attr = getattr(model, attr_key)
      except Exception: continue
      child_fqn = f"{fqn}.{attr_key}" if fqn else attr_key
      _swap_linear_with_lora(attr, r, alpha, target_cls, module_filter_fn, child_fqn, model, attr_key, visited)

def apply_lora(model, r:int=8, alpha:int=16, target_cls:type[Linear]=Linear,
               module_filter_fn:Callable[[Any,str],bool]|None=None):
  _swap_linear_with_lora(model, r, alpha, target_cls, module_filter_fn, "", None, "", set())
  return model

def get_lora_params(model) -> list[Tensor]:
  return [v for k, v in get_state_dict(model).items() if "lora_" in k]
