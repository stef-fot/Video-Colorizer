"""Convert PyTorch colorization weights to ONNX files that ColorStream can load.

DDColor (needs a clone of https://github.com/piddnad/DDColor):
    git clone https://github.com/piddnad/DDColor external/DDColor
    python scripts/download_models.py ddcolor-tiny
    python scripts/export_onnx.py ddcolor-tiny --repo external/DDColor

ECCV16 / SIGGRAPH17 (needs a clone of https://github.com/richzhang/colorization):
    git clone https://github.com/richzhang/colorization external/colorization
    python scripts/export_onnx.py eccv16 --repo external/colorization

Add --fp16 to also write a half precision copy (do NOT use it on GTX 16xx cards).

NOTE: these exporters follow the upstream inference code of each project. Upstream
APIs change, so if an import fails, compare with the project's own inference script.
After exporting, run 'python -m colorstream benchmark' and check one real clip by eye.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from colorstream.config import load_config  # noqa: E402

DDCOLOR_ENCODERS = {"ddcolor-tiny": "convnext-t", "ddcolor-large": "convnext-l", "ddcolor-artistic": "convnext-l"}
DDCOLOR_FILES = {"ddcolor-tiny": "ddcolor_paper_tiny.pth", "ddcolor-large": "ddcolor_paper.pth",
                 "ddcolor-artistic": "ddcolor_artistic.pth"}


def export_ddcolor(name, repo, cfg, size, opset):
    import torch

    sys.path.insert(0, str(Path(repo).resolve()))
    from basicsr.archs.ddcolor_arch import DDColor  # from the DDColor repository

    pth = Path(cfg.paths.models_dir) / DDCOLOR_FILES[name]
    if not pth.exists():
        raise SystemExit(f"{pth} not found. Run: python scripts/download_models.py {name}")
    model = DDColor(encoder_name=DDCOLOR_ENCODERS[name], decoder_name="MultiScaleColorDecoder",
                    input_size=[size, size], num_output_channels=2, last_norm="Spectral",
                    do_normalize=False, num_queries=100, num_scales=3, dec_layers=9)
    state = torch.load(pth, map_location="cpu")
    model.load_state_dict(state["params"] if "params" in state else state, strict=False)
    return model.eval(), torch.zeros(1, 3, size, size)


def export_zhang(name, repo, cfg, size, opset):
    import torch

    sys.path.insert(0, str(Path(repo).resolve()))
    import colorizers  # from the richzhang/colorization repository

    net = (colorizers.eccv16 if name == "eccv16" else colorizers.siggraph17)(pretrained=True).eval()

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, L):  # L: (N,1,H,W), lightness 0..100 -> ab (N,2,H,W)
            return self.m(L)

    return Wrap(net).eval(), torch.zeros(1, 1, size, size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["ddcolor-tiny", "ddcolor-large", "ddcolor-artistic", "eccv16", "siggraph17"])
    ap.add_argument("--repo", required=True, help="path to the upstream repository clone")
    ap.add_argument("--size", type=int, default=0, help="fixed input side (default 512 for DDColor, 256 for Zhang)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("-c", "--config")
    a = ap.parse_args()

    import torch

    cfg = load_config(a.config)
    spec = cfg.model_spec(a.model)
    size = a.size or (512 if a.model.startswith("ddcolor") else 256)
    fn = export_ddcolor if a.model.startswith("ddcolor") else export_zhang
    model, dummy = fn(a.model, a.repo, cfg, size, a.opset)
    out = Path(cfg.paths.models_dir) / spec.file
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(model, dummy, str(out), input_names=["input"], output_names=["ab"], opset_version=a.opset,
                          dynamic_axes={"input": {0: "batch"}, "ab": {0: "batch"}})
    print(f"Wrote {out}")
    if a.fp16:
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16

        m16 = convert_float_to_float16(onnx.load(str(out)), keep_io_types=True)
        out16 = out.with_name(out.stem + "_fp16.onnx")
        onnx.save(m16, str(out16))
        print(f"Wrote {out16} (set a model entry with this file in config.yaml to use it)")
    print(f"Input size is fixed at {size}x{size}. Verify with: python -m colorstream benchmark")


if __name__ == "__main__":
    main()
