"""Catch the Hopper backward incompatibility before a full model load."""

from types import SimpleNamespace

import pytest

from scripts import check_env


@pytest.mark.parametrize(
    "affected,available,enabled,disabled,import_error,want",
    [
        (False, False, False, False, False, True),  # fixed Triton / other GPU
        (True, False, True, False, False, False),  # missing TileLang or nvcc
        (True, True, False, False, False, False),  # FLA_TILELANG=0
        (True, True, True, True, False, False),  # dispatch disabled
        (True, True, True, False, True, False),  # broken TileLang installation
        (True, True, True, False, False, True),
    ],
)
def test_hopper_backward_prerequisites(
    monkeypatch, capsys, affected, available, enabled, disabled, import_error, want,
):
    monkeypatch.setenv("FLA_DISABLE_BACKEND_DISPATCH", "1" if disabled else "0")
    backend = SimpleNamespace(is_available=lambda: available, is_enabled=lambda: enabled)
    modules = {
        "fla.utils": SimpleNamespace(
            IS_NVIDIA_HOPPER=affected, TRITON_ABOVE_3_4_0=True, TRITON_ABOVE_3_7_1=False,
        ),
        "fla.ops.common.backends.tilelang": SimpleNamespace(TileLangBackend=backend),
        "tilelang": SimpleNamespace(),
    }

    def load(name):
        if name == "tilelang" and import_error:
            raise ImportError("missing shared library")
        return modules[name]

    monkeypatch.setattr(check_env.importlib, "import_module", load)
    assert check_env._check_gdn_backward() is want
    assert ("[FAIL]" in capsys.readouterr().out) is not want


def test_main_fails_when_backward_prerequisites_fail(monkeypatch, capsys):
    import torch

    monkeypatch.setattr(check_env, "_version", lambda name: "installed")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(check_env, "_check_gdn_backward", lambda: False)

    def optimized_dispatch():
        implementation = lambda: None
        is_new_implementation = True

        def wrapped():
            return implementation() if is_new_implementation else None

        return wrapped

    # These wrappers have the same dispatch closure as the HF decorators.
    mq = SimpleNamespace(**{
        name: optimized_dispatch() for name in (
            "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
            "causal_conv1d_fn", "causal_conv1d_update",
        )
    })
    import sys

    monkeypatch.setitem(sys.modules, "transformers.models.qwen3_5.modeling_qwen3_5", mq)
    monkeypatch.setattr("transformers.models.qwen3_5.modeling_qwen3_5", mq)
    assert check_env.main() == 1
    assert "[OK]" not in capsys.readouterr().out
