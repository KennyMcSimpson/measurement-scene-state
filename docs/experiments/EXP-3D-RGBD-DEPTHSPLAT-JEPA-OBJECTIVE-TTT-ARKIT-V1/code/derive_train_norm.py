"""Derives train_jepa_arkit_norm.py from EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1/code/train_jepa_arkit_overlap.py by
exact replacements (each must occur exactly once). The only change is the training objective: accuracy only. The
relative L1 of the alpha-normalized rendered depth and the L1 of the alpha-normalized rendered RGB, both on valid
measured target pixels that the state covers (alpha >= 0.5, the mask carries no gradient), so an uncovered pixel costs
nothing and coverage is never rewarded. Usage: python3 derive_train_norm.py SRC DST
"""
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
text = src.read_text()
REPLACEMENTS = [
    ('"""Overlap-aware reverse-JEPA training of DepthSplat on real sensor data (EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1):\n',
     '"""Accuracy-only reverse-JEPA training of DepthSplat on real sensor data (EXP-3D-RGBD-DEPTHSPLAT-JEPA-OBJECTIVE-TTT-ARKIT-V1, JN):\n'
     'the overlap-aware script of EXP-3D-RGBD-DEPTHSPLAT-JEPA-ARKIT-OVERLAP-V1 with one more change, the objective: the\n'
     'relative L1 of the alpha-normalized rendered depth and the L1 of the alpha-normalized rendered RGB on the covered\n'
     '(alpha >= 0.5, no gradient through the mask) valid target pixels; uncovered pixels cost nothing. The rest follows.\n'),
    ('def grounded_loss(depth, color, z, rgb):\n'
     '    """Relative L1 of raw rendered depth on valid measured pixels + RGB_WEIGHT x RGB L1."""\n'
     '    valid = z > 0\n'
     '    depth_term = ((depth - z).abs() / z.clamp_min(1e-6))[valid].mean()\n'
     '    return depth_term + RGB_WEIGHT * (color - rgb).abs().mean(), float(depth_term)\n',
     'def grounded_loss(depth, color, z, rgb, alpha):\n'
     '    """Accuracy only: relative L1 of the alpha-normalized rendered depth + RGB_WEIGHT x L1 of the alpha-normalized\n'
     '    rendered RGB, both on valid measured pixels the state covers (alpha >= ALPHA_MIN; the mask carries no gradient).\n'
     '    An uncovered pixel costs nothing, so coverage is never rewarded."""\n'
     '    covered = (z > 0) & (alpha.detach() >= ALPHA_MIN)\n'
     '    if not covered.any():\n'
     '        return (depth * 0).sum(), 0.0\n'
     '    a = alpha.clamp_min(1e-6)\n'
     '    depth_term = ((depth / a - z).abs() / z.clamp_min(1e-6))[covered].mean()\n'
     '    rgb_term = (color / a.unsqueeze(1) - rgb).abs().permute(0, 2, 3, 1)[covered].mean()\n'
     '    return depth_term + RGB_WEIGHT * rgb_term, float(depth_term)\n'),
    ('        depth, color = render(decoder, gaussians, target)\n'
     '        z = measured_z(record["frames"], target_ids).to(device)\n'
     '        loss, depth_term = grounded_loss(depth, color, z, target["image"][0])\n',
     '        depth, color = render(decoder, gaussians, target)\n'
     '        alpha = render_alpha(gaussians, target, RENDER)[0]\n'
     '        z = measured_z(record["frames"], target_ids).to(device)\n'
     '        loss, depth_term = grounded_loss(depth, color, z, target["image"][0], alpha)\n'),
    ('"source": "train_jepa_arkit_overlap.py"', '"source": "train_jepa_arkit_norm.py"'),
]
for old, new in REPLACEMENTS:
    if text.count(old) != 1:
        raise SystemExit(f"replacement anchor found {text.count(old)} times: {old[:60]!r}")
    text = text.replace(old, new)
dst.write_text(text)
print("DERIVED", dst)
