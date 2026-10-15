"""EXP-3D-CAJU3D-JEPAGATE-V1 descriptive diagnostic (after the DEV decision; arrived frames only, no ground truth, no
TTT): which measurements each gate drops. Recomputes the DEV episodes' scores and gates exactly as gate_ttt.py does
(same seeds, same generator order) and profiles the dropped pixels against all valid pixels:
  no cross-view hit   no point of the other six views lands on the pixel (DEPTH's z-buffer; DEPTH scores it 0)
  no JEPA cell hit    no source latent cell is warped onto the pixel's latent cell
  cosine, edge        incidence cosine and depth jump >= 0.1 (engine_core.incidence_weight)
  relative depth      depth / the view's median valid depth
  stream view         one of the 4 stream views (else one of the 3 context views)
  log P, log err      the JEPA's predicted log variance and log mean (z - z_hat)^2 of the pixel's cell
  depth / photometric disagreement | hit   DEPTH's score and the channel-mean |RGB - RGB of the nearest point of the
                      other views|, over the pixels with a cross-view hit
  dropped neighbour share   the share of a dropped pixel's valid 4-neighbours that are dropped too (spatial clumping;
                      a scattered 10 % drop gives ~0.10)
Run with the DepthSplat environment from the DepthSplat repository root (PYTHONPATH=.):
  python gate_dropped_profile.py ROOT OUT.json
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import gate_ttt as T  # noqa: E402  (path set-up and imports of the replay)

G, J, v1, engine_core, ds_predict = T.G, T.J, T.v1, T.engine_core, T.ds_predict
ARMS = ("JEPA", "DEPTH", "INC", "RANDOM")
COHORTS = ("DEV_SYSTEM", "DEV_SYSTEM2")


def cross_view_hit(planar, rgb, k, c2w, min_depth=0.05):
    """The z-buffer of gates.depth_disagreement: the hit mask [V, H, W] and the photometric disagreement [V, H, W]
    (channel-mean |RGB_v - RGB of the nearest point of the other six views|, 0 where no hit)."""
    v_n, h, w = planar.shape
    rows = torch.arange(h, dtype=torch.float32, device=planar.device).view(h, 1)
    cols = torch.arange(w, dtype=torch.float32, device=planar.device).view(1, w)
    pts, ok, col = [], [], []
    for s in range(v_n):
        d = planar[s]
        cam = torch.stack([(cols - k[s, 0, 2]) / k[s, 0, 0] * d, (rows - k[s, 1, 2]) / k[s, 1, 1] * d, d], dim=-1)
        pts.append(cam.reshape(-1, 3) @ c2w[s, :3, :3].T + c2w[s, :3, 3])
        ok.append((d > 0).reshape(-1))
        col.append(rgb[s].reshape(3, -1).T)
    hits, photo = [], []
    for v in range(v_n):
        others = [s for s in range(v_n) if s != v]
        p = torch.cat([pts[s] for s in others])
        valid = torch.cat([ok[s] for s in others])
        c = torch.cat([col[s] for s in others])
        cam = (p - c2w[v, :3, 3]) @ c2w[v, :3, :3]
        z = cam[:, 2]
        zs = z.clamp_min(1e-6)
        u = torch.round(k[v, 0, 0] * cam[:, 0] / zs + k[v, 0, 2]).long()
        r = torch.round(k[v, 1, 1] * cam[:, 1] / zs + k[v, 1, 2]).long()
        good = valid & (z > min_depth) & (u >= 0) & (u < w) & (r >= 0) & (r < h)
        flat = r.clamp(0, h - 1) * w + u.clamp(0, w - 1)
        key = torch.where(good, z, torch.full_like(z, float("inf")))
        nearest = torch.full((h * w,), float("inf"), device=planar.device).scatter_reduce(0, flat, key, reduce="amin")
        win = (good & (key <= nearest[flat] * (1 + 1e-6))).float()
        count = torch.zeros(h * w, device=planar.device).scatter_add(0, flat, win)
        total = torch.zeros(h * w, 3, device=planar.device).scatter_add(0, flat[:, None].expand(-1, 3), c * win[:, None])
        hit = (count > 0).view(h, w) & (planar[v] > 0)
        rep = (total / count.clamp_min(1.0)[:, None]).T.reshape(3, h, w)
        hits.append(hit)
        photo.append(torch.where(hit, (rgb[v] - rep).abs().mean(dim=0), torch.zeros_like(planar[v])))
    return torch.stack(hits), torch.stack(photo)


def neighbour_share(mask, valid):
    """Per pixel of `mask`: the share of its valid 4-neighbours that are also in `mask`."""
    m, ok = mask.float(), valid.float()
    pad = lambda x: F.pad(x[:, None], (1, 1, 1, 1))[:, 0]  # noqa: E731
    mp, op = pad(m), pad(ok)
    same = mp[:, :-2, 1:-1] + mp[:, 2:, 1:-1] + mp[:, 1:-1, :-2] + mp[:, 1:-1, 2:]
    total = op[:, :-2, 1:-1] + op[:, 2:, 1:-1] + op[:, 1:-1, :-2] + op[:, 1:-1, 2:]
    return same / total.clamp_min(1.0)


@torch.no_grad()
def jepa_terms(model, rgb, ray, k, c2w):
    """Per view: log P, log mean latent error and the cell hit of the leave-one-out prediction, at the pixels."""
    out = {"log_p": [], "log_err": [], "cell_hit": []}
    for v in range(rgb.shape[0]):
        others = [s for s in range(rgb.shape[0]) if s != v]
        res = model(rgb[others][None], ray[others][None], k[others][None], c2w[others][None],
                    rgb[v][None], ray[v][None], k[v][None], c2w[v][None])
        err = ((res["z"] - res["z_hat"]) ** 2).mean(dim=1, keepdim=True)
        for name, x in (("log_p", res["log_p"]), ("log_err", torch.log(err.clamp_min(1e-12))),
                        ("cell_hit", res["hit"])):
            out[name].append(F.interpolate(x, size=(J.H, J.W), mode="nearest")[0, 0])
    return {n: torch.stack(x) for n, x in out.items()}


def main():
    root, out_path = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    locks = json.loads((root / "frame_role_lock.json").read_text())["cohorts"]
    jepa = J.GeometricReverseJEPA()
    jepa.load_state_dict(torch.load(root / "model/hjepa_final.pth", map_location="cpu", weights_only=False)["state_dict"])
    jepa = jepa.to(device).eval()
    sums = {name: {} for name in ("ALL",) + ARMS}
    pooled = {"log_nu": [], "log_p": [], "log_err": []}
    agree = {arm: [] for arm in ARMS}
    episodes = 0
    for cohort in COHORTS:
        lock = locks[cohort]
        if T.sha(lock["manifest"]) != lock["manifest_sha256"]:
            raise PermissionError(f"{cohort}: manifest changed")
        records = {r["scene_id"]: r for r in json.loads(Path(lock["manifest"]).read_text())["scenes"]}
        sealed = {}
        for part in sorted((root / "gate/DEV").glob("gate_seal_*of*.json")):
            sealed.update(json.loads(part.read_text())["episodes_log"])
        for e in lock["episodes"]:
            frames = records[e["scene_id"]]["frames"]
            rows = {f["frame_id"]: f for f in frames}
            inputs = e["context"] + e["stream"]
            key = f"{e['scene_id']}/{e['role']}"
            generator = torch.Generator(device=device).manual_seed(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
            scores, valid = T.episode_scores(jepa, frames, inputs, device, generator)
            gates = {arm: G.matched_gate(scores[arm], valid, T.SHARE, generator) for arm in ARMS}
            dropped = {arm: valid & (g == 0) for arm, g in gates.items()}
            overlap = {f"{a}|{b}": float((dropped[a] & dropped[b]).sum() / max(int(dropped[a].sum()), 1))
                       for i, a in enumerate(ARMS) for b in ARMS[i + 1:]}
            for name, v in sealed[key]["arms"]["dropped_overlap"].items():
                agree[name.split("|")[0]].append(abs(v - overlap[name]))
            planar = T.measured_z(frames, inputs).to(device)
            k = torch.tensor(np.array([rows[i]["intrinsics"] for i in inputs], dtype=np.float32), device=device)
            c2w = torch.tensor(np.array([ds_predict.orthonormal(np.array(rows[i]["c2w"], dtype=np.float64))
                                         for i in inputs], dtype=np.float32), device=device)
            rgb = T.views(frames, inputs, None, v1.NEAR, v1.FAR, device)["image"][0]
            terms = jepa_terms(jepa, rgb, planar * J.ray_factor(k, device=device), k, c2w)
            _, cosine, jump = engine_core.incidence_weight(planar, k, 0.1, 0.4, 0.1)
            med = torch.stack([planar[v][planar[v] > 0].median() for v in range(planar.shape[0])]).view(-1, 1, 1)
            stream = torch.zeros_like(valid)
            stream[len(e["context"]):] = True
            hit, photo = cross_view_hit(planar, rgb, k, c2w)
            # name -> (per-pixel value, condition): conditional features average over the set's pixels with a hit
            feats = {"no_cross_view_hit": ((~hit).float(), None),
                     "no_jepa_cell_hit": ((terms["cell_hit"] == 0).float(), None), "cosine": (cosine, None),
                     "edge": ((jump >= 0.1).float(), None), "relative_depth": (planar / med, None),
                     "stream_view": (stream.float(), None), "log_p": (terms["log_p"], None),
                     "log_err": (terms["log_err"], None), "depth_disagreement|hit": (scores["DEPTH"], hit),
                     "photometric_disagreement|hit": (photo, hit)}
            for name, mask in (("ALL", valid),) + tuple(dropped.items()):
                s = sums[name]
                s["pixels"] = s.get("pixels", 0) + int(mask.sum())
                for f, (x, cond) in feats.items():
                    m = mask if cond is None else mask & cond
                    s[f] = s.get(f, 0.0) + float(x[m].double().sum())
                    if cond is not None:
                        s[f"#{f}"] = s.get(f"#{f}", 0) + int(m.sum())
                share = neighbour_share(mask, valid)
                s["dropped_neighbour_share"] = s.get("dropped_neighbour_share", 0.0) + float(share[mask].double().sum())
            pick = valid & (torch.rand(valid.shape, generator=generator, device=device) < 0.02)
            pooled["log_nu"].append(torch.log(scores["JEPA"][pick].clamp_min(1e-12)).cpu())
            pooled["log_p"].append(terms["log_p"][pick].cpu())
            pooled["log_err"].append(terms["log_err"][pick].cpu())
            episodes += 1
    profile = {}
    for name, s in sums.items():
        profile[name] = {"pixels": s["pixels"]}
        for f, v in s.items():
            if f == "pixels" or f.startswith("#"):
                continue
            profile[name][f] = v / (s[f"#{f}"] if f"#{f}" in s else s["pixels"])
    x = {n: torch.cat(v).double().numpy() for n, v in pooled.items()}
    corr = {f"log_nu~{n}": float(np.corrcoef(x["log_nu"], x[n])[0, 1]) for n in ("log_p", "log_err")}
    result = {"episodes": episodes, "profile": profile, "pooled_correlations": corr,
              "overlap_reproduction_max_abs_diff": {a: float(max(v)) for a, v in agree.items() if v},
              "ground_truth_read": False, "query_frames_read": False}
    out_path.write_text(json.dumps(result, indent=1))
    names = ("ALL",) + ARMS
    print("feature".ljust(30), "".join(n.rjust(10) for n in names))
    for f in ("pixels",) + tuple(feats) + ("dropped_neighbour_share",):
        print(f.ljust(30), "".join(f"{profile[n][f]:10.4f}" if f != "pixels" else f"{profile[n][f]:10d}" for n in names))
    print("pooled correlations", json.dumps({k: round(v, 3) for k, v in corr.items()}))
    print("overlap reproduction max |diff|", json.dumps(result["overlap_reproduction_max_abs_diff"]))


if __name__ == "__main__":
    main()
