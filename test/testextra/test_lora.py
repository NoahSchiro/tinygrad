#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad import Tensor, Context
from tinygrad.nn import Linear, optim
from tinygrad.nn.state import get_state_dict, get_parameters
from extra.lora import LoRALinear, apply_lora, get_lora_params

BS, T, IN_DIM, OUT_DIM, R = 4, 2, 32, 16, 8
np.random.seed(1337)
Tensor.manual_seed(1337)

def _make_pair(in_dim=IN_DIM, out_dim=OUT_DIM, r=R, alpha=16, bias=True):
  linear = Linear(in_dim, out_dim, bias=bias)
  lora = LoRALinear.from_linear(linear, r=r, alpha=alpha)
  return linear, lora

class TestLoRALinear(unittest.TestCase):
  def test_forward_parity_initial(self):
    li, lo = _make_pair()
    li_no_bias, lo_no_bias = _make_pair(bias=False)
    x2d, x3d = Tensor.randn(BS, IN_DIM), Tensor.randn(BS, T, IN_DIM)
    np.testing.assert_allclose(li(x2d).numpy(), lo(x2d).numpy(), atol=1e-6)
    np.testing.assert_allclose(li(x3d).numpy(), lo(x3d).numpy(), atol=1e-6)
    np.testing.assert_allclose(li_no_bias(x2d).numpy(), lo_no_bias(x2d).numpy(), atol=1e-6)
    np.testing.assert_allclose(li_no_bias(x3d).numpy(), lo_no_bias(x3d).numpy(), atol=1e-6)

  def test_forward_with_lora(self):
    linear, lora = _make_pair()
    lora.lora_A = Tensor.randn(R, IN_DIM) * 0.1
    lora.lora_B = Tensor.randn(OUT_DIM, R) * 0.1
    x = Tensor.randn(BS, IN_DIM)
    delta = (x @ lora.lora_A.T @ lora.lora_B.T) * lora.scale
    np.testing.assert_allclose(lora(x).numpy(), (linear(x)+delta).numpy(), atol=1e-6)

  def test_r_zero(self):
    lora = LoRALinear(IN_DIM, OUT_DIM, r=0)
    self.assertIsNone(lora.lora_A)
    self.assertIsNone(lora.lora_B)
    self.assertEqual(lora.scale, 0.0)

  def test_backward_only_lora_has_grad(self):
    _, lora = _make_pair()
    lora(Tensor.randn(BS, IN_DIM)).sum().backward()
    self.assertIsNotNone(lora.lora_A.grad)
    self.assertIsNotNone(lora.lora_B.grad)

  def test_merge(self):
    _, lora = _make_pair()
    lora.lora_A = Tensor.randn(R, IN_DIM) * 0.1
    lora.lora_B = Tensor.randn(OUT_DIM, R) * 0.1
    x = Tensor.randn(BS, IN_DIM)
    y_before = lora(x)
    lora.merge()
    self.assertIsNone(lora.lora_A)
    self.assertIsNone(lora.lora_B)
    self.assertEqual(lora.r, 0)
    y_after = lora(x)
    np.testing.assert_allclose(y_after.numpy(), y_before.numpy(), atol=1e-6)

  def test_from_linear_preserves_weight(self):
    linear = Linear(IN_DIM, OUT_DIM)
    w_orig, b_orig = linear.weight.numpy(), linear.bias.numpy()
    lora = LoRALinear.from_linear(linear, r=R)
    np.testing.assert_allclose(lora.weight.numpy(), w_orig, atol=0)
    np.testing.assert_allclose(lora.bias.numpy(), b_orig, atol=0)
  def test_base_weight_is_frozen(self):
    _, lora = _make_pair()
    self.assertFalse(lora.weight.is_param)
    self.assertFalse(lora.bias.is_param)
    self.assertTrue(lora.lora_A.is_param)
    self.assertTrue(lora.lora_B.is_param)

class TestApplyLoRA(unittest.TestCase):
  def test_apply_to_simple_model(self):
    class Model:
      def __init__(self):
        self.fc1 = Linear(IN_DIM, OUT_DIM)
        self.fc2 = Linear(OUT_DIM, IN_DIM)
      def __call__(self, x):
        return self.fc2(self.fc1(x).relu())
    model = Model()
    apply_lora(model, r=R, alpha=16)
    self.assertIsInstance(model.fc1, LoRALinear)
    self.assertIsInstance(model.fc2, LoRALinear)
  def test_apply_with_filter(self):
    class Model:
      def __init__(self):
        self.qkv = Linear(IN_DIM, OUT_DIM)
        self.o = Linear(OUT_DIM, IN_DIM)
        self.ffn = Linear(IN_DIM, OUT_DIM)
      def __call__(self, x):
        return self.o(self.qkv(x)) + self.ffn(x)
    model = Model()
    apply_lora(model, r=R, alpha=16, module_filter_fn=lambda _, fqn: "qkv" in fqn or fqn.endswith(".o") or fqn == "o")
    self.assertIsInstance(model.qkv, LoRALinear)
    self.assertIsInstance(model.o, LoRALinear)
    self.assertIsInstance(model.ffn, Linear)
    self.assertNotIsInstance(model.ffn, LoRALinear)

  def test_apply_with_nested_model(self):
    class Block:
      def __init__(self):
        self.proj = Linear(IN_DIM, OUT_DIM)
    class Model:
      def __init__(self):
        self.blocks = [Block() for _ in range(3)]
        self.head = Linear(OUT_DIM, IN_DIM)
    model = Model()
    apply_lora(model, r=R, alpha=16)
    self.assertIsInstance(model.blocks[0].proj, LoRALinear)
    self.assertIsInstance(model.blocks[2].proj, LoRALinear)
    self.assertIsInstance(model.head, LoRALinear)

  def test_apply_preserves_forward(self):
    class Model:
      def __init__(self):
        self.fc = Linear(IN_DIM, OUT_DIM)
      def __call__(self, x):
        return self.fc(x)
    model = Model()
    x = Tensor.randn(BS, IN_DIM)
    y_before = model(x).numpy()
    apply_lora(model, r=R, alpha=16)
    y_after = model(x).numpy()
    np.testing.assert_allclose(y_after, y_before, atol=1e-6)

class TestLoRAOptimizer(unittest.TestCase):
  def test_get_lora_params(self):
    class Model:
      def __init__(self):
        self.fc1 = Linear(IN_DIM, OUT_DIM)
        self.fc2 = Linear(OUT_DIM, IN_DIM)
    model = Model()
    apply_lora(model, r=R, alpha=16)
    params = get_lora_params(model)
    self.assertEqual(len(params), 4)  # lora_A, lora_B for each layer

  def test_get_lora_params_are_trainable(self):
    class Model:
      def __init__(self):
        self.fc = Linear(IN_DIM, OUT_DIM)
    model = Model()
    apply_lora(model, r=R, alpha=16)
    for p in get_lora_params(model):
      self.assertTrue(p.is_param)

  def test_optimizer_only_updates_lora(self):
    class Model:
      def __init__(self):
        self.fc1 = Linear(IN_DIM, OUT_DIM)
        self.fc2 = Linear(OUT_DIM, IN_DIM)
      def __call__(self, x):
        return self.fc2(self.fc1(x).relu())
    model = Model()
    apply_lora(model, r=R, alpha=16)
    lora_params = get_lora_params(model)
    optimizer = optim.AdamW(lora_params, lr=1e-3)
    x = Tensor.randn(BS, IN_DIM)
    y = Tensor.randn(BS, IN_DIM)
    w_before = model.fc1.weight.numpy()
    b_before = model.fc1.lora_B.numpy()
    with Context(TRAINING=1):
      loss = (model(x) - y).abs().mean()
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
    np.testing.assert_allclose(model.fc1.weight.numpy(), w_before, atol=0, err_msg="base weight should not change")
    self.assertFalse(np.allclose(model.fc1.lora_B.numpy(), b_before, atol=1e-6), "lora_B should change")

  def test_optimizer_with_all_params_skips_frozen(self):
    # passing all params (including frozen) to optimizer should still only update trainable
    class Model:
      def __init__(self):
        self.fc = Linear(IN_DIM, OUT_DIM)
      def __call__(self, x):
        return self.fc(x)
    model = Model()
    apply_lora(model, r=R, alpha=16)
    all_params = get_parameters(model)
    optimizer = optim.SGD(all_params, lr=1e-2)
    self.assertEqual(len(optimizer.params), 2)  # only lora_A, lora_B
    self.assertEqual(len(optimizer.buffers), 2)  # weight, bias

  def test_lora_state_dict(self):
    class Model:
      def __init__(self):
        self.fc = Linear(IN_DIM, OUT_DIM)
    model = Model()
    apply_lora(model, r=R, alpha=16)
    sd = get_state_dict(model)
    self.assertIn("fc.weight", sd)
    self.assertIn("fc.bias", sd)
    self.assertIn("fc.lora_A", sd)
    self.assertIn("fc.lora_B", sd)

if __name__ == "__main__":
  unittest.main()
