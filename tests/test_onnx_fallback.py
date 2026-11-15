"""The ONNX session falls back to safer settings and finally to the CPU instead of crashing (DirectML failures)."""
import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper

from colorstream.config import config_from_dict
from colorstream.models import onnx_backend as ob


def identity_model(path, size=64):
    node = helper.make_node("Identity", ["x"], ["y"])
    g = helper.make_graph([node], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, size, size])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, size, size])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, str(path))


class FailingGpu:
    """Wraps InferenceSession: any session asked to use a GPU provider fails, like DirectML did."""

    def __init__(self, real, fail_levels):
        self.real, self.fail_levels, self.calls = real, fail_levels, []

    def __call__(self, path, sess_options=None, providers=None):
        first = providers[0] if isinstance(providers[0], str) else providers[0][0]
        self.calls.append((first, sess_options.graph_optimization_level))
        if first == "DmlExecutionProvider" and sess_options.graph_optimization_level in self.fail_levels:
            raise RuntimeError("Exception during initialization: 80070057 The parameter is incorrect.")
        providers = [p for p in providers if (p if isinstance(p, str) else p[0]) == "CPUExecutionProvider"]
        return self.real(path, sess_options=sess_options, providers=providers)


ALL_LEVELS = {ort.GraphOptimizationLevel.ORT_ENABLE_ALL, ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
              ort.GraphOptimizationLevel.ORT_DISABLE_ALL}


def cfg_for(tmp_path, **onnx_cfg):
    return config_from_dict({"paths": {"models_dir": str(tmp_path)}, "onnx": onnx_cfg})


def test_safer_optimization_level_is_tried_before_giving_up_on_the_gpu(tmp_path, monkeypatch):
    p = tmp_path / "m.onnx"
    identity_model(p)
    monkeypatch.setattr(ob, "select_providers", lambda cfg: ["DmlExecutionProvider", "CPUExecutionProvider"])
    fake = FailingGpu(ort.InferenceSession, {ort.GraphOptimizationLevel.ORT_ENABLE_ALL})
    monkeypatch.setattr(ort, "InferenceSession", fake)
    sess, notes = ob.create_session(str(p), cfg_for(tmp_path))
    assert notes and "optimization full" in notes[0]
    assert len(fake.calls) == 2 and fake.calls[1][1] == ort.GraphOptimizationLevel.ORT_ENABLE_BASIC


def test_falls_back_to_cpu_when_the_gpu_never_works(tmp_path, monkeypatch, caplog):
    p = tmp_path / "m.onnx"
    identity_model(p)
    monkeypatch.setattr(ob, "select_providers", lambda cfg: ["DmlExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(ort, "InferenceSession", FailingGpu(ort.InferenceSession, ALL_LEVELS))
    with caplog.at_level("WARNING"):
        sess, notes = ob.create_session(str(p), cfg_for(tmp_path))
    assert sess.get_providers()[0] == "CPUExecutionProvider"
    assert any("using the CPU instead" in r.message for r in caplog.records)
    out = sess.run(None, {"x": np.ones((1, 3, 64, 64), np.float32)})[0]
    assert out.shape == (1, 3, 64, 64)


def test_cpu_fallback_can_be_switched_off(tmp_path, monkeypatch):
    p = tmp_path / "m.onnx"
    identity_model(p)
    monkeypatch.setattr(ob, "select_providers", lambda cfg: ["DmlExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(ort, "InferenceSession", FailingGpu(ort.InferenceSession, ALL_LEVELS))
    with pytest.raises(RuntimeError, match="could not be loaded"):
        ob.create_session(str(p), cfg_for(tmp_path, allow_cpu_fallback=False))


def test_plain_cpu_setup_just_works(tmp_path):
    p = tmp_path / "m.onnx"
    identity_model(p)
    sess, notes = ob.create_session(str(p), cfg_for(tmp_path))
    assert not notes and sess.get_providers()[0] == "CPUExecutionProvider"
