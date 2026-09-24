from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import max_abs_error
from .pretrained_shadow import PretrainedHeadGeometry, reconstruct_head


def _finite_array(value, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def slice_gpt2_head(
    c_attn_weight,
    c_attn_bias,
    c_proj_weight,
    *,
    n_head: int,
    head_index: int,
    scale_attn_weights: bool = True,
    scale_attn_by_inverse_layer_idx: bool = False,
    reorder_and_upcast_attn: bool = False,
    layer_index: int = 0,
) -> PretrainedHeadGeometry:
    packed_w = _finite_array(c_attn_weight, name="c_attn_weight", ndim=2)
    packed_b = _finite_array(c_attn_bias, name="c_attn_bias", ndim=1)
    proj_w = _finite_array(c_proj_weight, name="c_proj_weight", ndim=2)
    if not isinstance(n_head, (int, np.integer)) or int(n_head) <= 0:
        raise ValueError("n_head must be a positive integer")
    n_head = int(n_head)
    if not isinstance(head_index, (int, np.integer)):
        raise ValueError("head_index must be an integer")
    head_index = int(head_index)
    if not isinstance(layer_index, (int, np.integer)) or int(layer_index) < 0:
        raise ValueError("layer_index must be a non-negative integer")
    if scale_attn_by_inverse_layer_idx:
        raise ValueError("inverse-layer attention scaling is outside this gate")
    if reorder_and_upcast_attn:
        raise ValueError("reordered/upcast GPT-2 attention is outside this gate")

    d_model = packed_w.shape[0]
    if packed_w.shape != (d_model, 3 * d_model):
        raise ValueError("c_attn_weight must have shape (d_model, 3*d_model)")
    if packed_b.shape != (3 * d_model,):
        raise ValueError("c_attn_bias must have shape (3*d_model,)")
    if proj_w.shape != (d_model, d_model):
        raise ValueError("c_proj_weight must have shape (d_model, d_model)")
    if d_model % n_head != 0:
        raise ValueError("d_model must be divisible by n_head")
    if not 0 <= head_index < n_head:
        raise ValueError("head_index out of range")

    d_head = d_model // n_head
    start = head_index * d_head
    stop = start + d_head
    q_slice = slice(start, stop)
    k_slice = slice(d_model + start, d_model + stop)
    v_slice = slice(2 * d_model + start, 2 * d_model + stop)
    score_scale = 1.0 / np.sqrt(float(d_head)) if bool(scale_attn_weights) else 1.0
    return PretrainedHeadGeometry(
        d_model=d_model,
        d_head=d_head,
        w_q=packed_w[:, q_slice],
        w_k=packed_w[:, k_slice],
        w_v=packed_w[:, v_slice],
        b_q=packed_b[q_slice],
        b_k=packed_b[k_slice],
        b_v=packed_b[v_slice],
        w_o=proj_w[start:stop, :],
        score_scale=score_scale,
    )


@dataclass(frozen=True)
class GPT2Capture:
    x: np.ndarray
    attention: np.ndarray
    pre_cproj: np.ndarray
    cproj_output: np.ndarray

    def __post_init__(self) -> None:
        x = _finite_array(self.x, name="x", ndim=2)
        attention = _finite_array(self.attention, name="attention", ndim=2)
        pre = _finite_array(self.pre_cproj, name="pre_cproj", ndim=2)
        out = _finite_array(self.cproj_output, name="cproj_output", ndim=2)
        sequence = x.shape[0]
        if sequence <= 0:
            raise ValueError("capture sequence must be non-empty")
        if attention.shape != (sequence, sequence):
            raise ValueError("attention must have shape (sequence, sequence)")
        if pre.shape[0] != sequence or out.shape[0] != sequence:
            raise ValueError("captured tensors must share the sequence dimension")
        object.__setattr__(self, "x", x.copy())
        object.__setattr__(self, "attention", attention.copy())
        object.__setattr__(self, "pre_cproj", pre.copy())
        object.__setattr__(self, "cproj_output", out.copy())


@dataclass(frozen=True)
class GPT2Parity:
    max_attention_abs_error: float
    max_mixture_abs_error: float
    max_full_module_abs_error: float
    passed: bool


def verify_gpt2_capture(
    geometry: PretrainedHeadGeometry,
    capture: GPT2Capture,
    *,
    c_proj_weight,
    c_proj_bias,
    head_index: int,
    tolerance: float = 1e-5,
) -> GPT2Parity:
    proj_w = _finite_array(c_proj_weight, name="c_proj_weight", ndim=2)
    proj_b = _finite_array(c_proj_bias, name="c_proj_bias", ndim=1)
    if capture.x.shape[1] != geometry.d_model:
        raise ValueError("capture x width must equal geometry d_model")
    if capture.pre_cproj.shape != (capture.x.shape[0], geometry.d_model):
        raise ValueError("pre_cproj must have shape (sequence, d_model)")
    if capture.cproj_output.shape != (capture.x.shape[0], geometry.d_model):
        raise ValueError("cproj_output must have shape (sequence, d_model)")
    if proj_w.shape != (geometry.d_model, geometry.d_model):
        raise ValueError("c_proj_weight must have shape (d_model, d_model)")
    if proj_b.shape != (geometry.d_model,):
        raise ValueError("c_proj_bias must have shape (d_model,)")
    if not isinstance(head_index, (int, np.integer)) or int(head_index) < 0:
        raise ValueError("head_index must be a non-negative integer")
    head_index = int(head_index)
    if geometry.d_model % geometry.d_head != 0:
        raise ValueError("geometry d_model must be divisible by d_head")
    n_head = geometry.d_model // geometry.d_head
    if head_index >= n_head:
        raise ValueError("head_index out of range")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")

    reference = reconstruct_head(geometry, capture.x)
    head_start = head_index * geometry.d_head
    head_stop = head_start + geometry.d_head
    attention_error = max_abs_error(capture.attention, reference.attention)
    mixture_error = max_abs_error(
        capture.pre_cproj[:, head_start:head_stop],
        reference.mixture,
    )
    full_reconstructed = capture.pre_cproj @ proj_w + proj_b
    full_error = max_abs_error(capture.cproj_output, full_reconstructed)
    return GPT2Parity(
        max_attention_abs_error=attention_error,
        max_mixture_abs_error=mixture_error,
        max_full_module_abs_error=full_error,
        passed=max(attention_error, mixture_error, full_error) <= tolerance,
    )


def capture_gpt2_context(
    model,
    input_ids,
    *,
    layer_index: int,
    head_index: int,
    tolerance: float = 1e-5,
):
    """Capture and verify one GPT-2 self-attention head from a batch-size-1 run."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for GPT-2 capture") from exc

    if not isinstance(layer_index, (int, np.integer)) or int(layer_index) < 0:
        raise ValueError("layer_index must be a non-negative integer")
    layer_index = int(layer_index)
    if layer_index >= len(model.transformer.h):
        raise ValueError("layer_index out of range")
    block = model.transformer.h[layer_index]
    attn_module = block.attn
    n_head = int(attn_module.num_heads)
    if not 0 <= int(head_index) < n_head:
        raise ValueError("head_index out of range")
    if bool(getattr(attn_module, "scale_attn_by_inverse_layer_idx", False)):
        raise ValueError("inverse-layer attention scaling is outside this gate")
    if bool(getattr(attn_module, "reorder_and_upcast_attn", False)):
        raise ValueError("reordered/upcast GPT-2 attention is outside this gate")

    captured: dict[str, object] = {}

    def attn_pre_hook(_module, args):
        captured["x"] = args[0].detach().cpu()

    def proj_pre_hook(_module, args):
        captured["pre_cproj"] = args[0].detach().cpu()

    def proj_hook(_module, _args, output):
        captured["cproj_output"] = output.detach().cpu()

    handles = [
        attn_module.register_forward_pre_hook(attn_pre_hook),
        attn_module.c_proj.register_forward_pre_hook(proj_pre_hook),
        attn_module.c_proj.register_forward_hook(proj_hook),
    ]
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    if getattr(outputs, "attentions", None) is None:
        raise ValueError("model did not return attentions")
    if len(captured) != 3:
        raise ValueError("GPT-2 hooks did not capture all required tensors")
    x_tensor = captured["x"]
    pre_tensor = captured["pre_cproj"]
    out_tensor = captured["cproj_output"]
    if x_tensor.ndim != 3 or x_tensor.shape[0] != 1:
        raise ValueError("canonical capture requires batch size 1")
    if pre_tensor.ndim != 3 or out_tensor.ndim != 3 or pre_tensor.shape[0] != 1 or out_tensor.shape[0] != 1:
        raise ValueError("captured projection tensors must have batch size 1")

    c_attn_weight = attn_module.c_attn.weight.detach().cpu().numpy()
    c_attn_bias = attn_module.c_attn.bias.detach().cpu().numpy()
    c_proj_weight = attn_module.c_proj.weight.detach().cpu().numpy()
    c_proj_bias = attn_module.c_proj.bias.detach().cpu().numpy()
    geometry = slice_gpt2_head(
        c_attn_weight,
        c_attn_bias,
        c_proj_weight,
        n_head=n_head,
        head_index=int(head_index),
        scale_attn_weights=bool(getattr(attn_module, "scale_attn_weights", True)),
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        layer_index=layer_index,
    )
    attention_tensor = outputs.attentions[layer_index]
    if attention_tensor.ndim != 4 or attention_tensor.shape[0] != 1:
        raise ValueError("returned attention tensor must have shape (1, heads, sequence, sequence)")
    capture = GPT2Capture(
        x=x_tensor[0].numpy(),
        attention=attention_tensor[0, int(head_index)].detach().cpu().numpy(),
        pre_cproj=pre_tensor[0].numpy(),
        cproj_output=out_tensor[0].numpy(),
    )
    parity = verify_gpt2_capture(
        geometry,
        capture,
        c_proj_weight=c_proj_weight,
        c_proj_bias=c_proj_bias,
        head_index=int(head_index),
        tolerance=tolerance,
    )
    return geometry, capture, parity
