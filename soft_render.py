"""
soft_render.py — numpy だけで動く簡易レンダラ(点スプラット + Zバッファ + 影)

用途:
  1. Genesis のヘッドレス描画(EGL等)が動かない環境でのフォールバック (--renderer soft)
  2. rain_fx の見た目調整・単体テスト
rr には rain_fx.Camera(numpy の幾何)を渡す。
出力は Genesis の cam.render と同じ約束: rgb(uint8, HxWx3) と depth(カメラ前方距離[m]、背景は 0)。
"""

import numpy as np



def _sample(V, F, fc, density, rng):
    tri = V[F]
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cr, axis=1)
    nrm = cr / (2 * area[:, None] + 1e-18)
    n = max(1, int(area.sum() * density))
    idx = rng.choice(len(F), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    f = u + v > 1
    u[f], v[f] = 1 - u[f], 1 - v[f]
    t = tri[idx]
    P = t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])
    C = np.full((n, 3), 150.0) if fc is None else np.asarray(fc, np.float32)[idx, :3]
    return P, nrm[idx], C


def soft_render(rr, meshes, light=(-0.35, -0.45, 0.82), seed=0):
    """meshes: [(V(N,3), F(M,3), face_colors(M,4) or None)] すべてワールド座標。"""
    H, W = rr.H, rr.W
    rng = np.random.default_rng(seed)
    dens = 2.5 * (rr.fpix / rr.zf) ** 2  # 1ピクセルの見かけ寸法の約0.6倍の点間隔
    parts = [_sample(V, F, fc, dens, rng) for V, F, fc in meshes]
    P = np.vstack([p[0] for p in parts])
    N = np.vstack([p[1] for p in parts])
    C = np.vstack([p[2] for p in parts])
    L = np.array(light, float)
    L /= np.linalg.norm(L)

    # --- シェーディング(曇天: 環境光が強く、影は弱い) ---
    view = rr.pos - P
    view /= np.linalg.norm(view, axis=1, keepdims=True)
    N = N * np.where((N * view).sum(1, keepdims=True) < 0, -1.0, 1.0)
    diff = np.clip(N @ L, 0, 1)
    hemi = 0.5 + 0.5 * N[:, 2]
    spec = np.clip((N * ((L + view) / np.linalg.norm(L + view, axis=1, keepdims=True))).sum(1), 0, 1) ** 30
    col = C * (0.30 + 0.48 * diff + 0.22 * hemi)[:, None] + spec[:, None] * 40.0

    # --- Zバッファ(2x2スプラットで穴を埋める) ---
    u, v, z = rr.proj(P)
    ok = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    u, v, z, col = u[ok].astype(int), v[ok].astype(int), z[ok], col[ok]
    idx = np.concatenate([(v + dv) * W + (u + du) for dv in (0, 1) for du in (0, 1)])
    zz = np.tile(z, 4)
    cc = np.tile(col, (4, 1))
    zbuf = np.full(H * W, np.inf)
    np.minimum.at(zbuf, idx, zz)
    win = zz <= zbuf[idx] + 1e-6
    cbuf = np.zeros((H * W, 3), np.float32)
    cbuf[idx[win]] = cc[win]

    # --- 地面(アスファルト) + 影(物体点を光の向きに地面へ投影した影マップ) ---
    ray_z = rr.ray[..., 2]
    below = ray_z < -1e-4
    t_g = np.where(below, -rr.pos[2] / np.where(below, ray_z, -1.0), np.inf)
    t_safe = np.where(below, t_g, 0.0)  # 地平線より上(t=inf)は計算しない
    gx = rr.pos[0] + t_safe * rr.ray[..., 0]
    gy = rr.pos[1] + t_safe * rr.ray[..., 1]
    cell, ext = 0.02, 4.0
    n_c = int(2 * ext / cell)
    sh = np.zeros((n_c, n_c), np.float32)
    keep = P[:, 2] > 0.05
    sp = P[keep] - L * (P[keep, 2:3] / L[2])
    ix = ((sp[:, 0] + ext) / cell).astype(int)
    iy = ((sp[:, 1] + ext) / cell).astype(int)
    m = (ix >= 0) & (ix < n_c) & (iy >= 0) & (iy < n_c)
    np.add.at(sh, (iy[m], ix[m]), 1.0)
    import cv2

    sh = cv2.GaussianBlur(sh, (0, 0), 5.0)
    sh = np.clip(sh / (0.25 * dens * cell * cell + 1e-9), 0, 1)  # 影の濃さ(密度で正規化)
    gi = ((gx + ext) / cell).astype(int).clip(0, n_c - 1)
    gj = ((gy + ext) / cell).astype(int).clip(0, n_c - 1)
    shadow = np.where(below, sh[gj, gi], 0.0)
    tex = cv2.GaussianBlur(rng.normal(size=(H, W)).astype(np.float32), (0, 0), 1.0) * 6
    ground = np.array([84, 85, 90], np.float32)[None, None, :] * (1 - 0.35 * shadow[..., None]) + tex[..., None]

    obj = zbuf.reshape(H, W)
    cbuf = cbuf.reshape(H, W, 3)
    use_obj = np.isfinite(obj) & (obj < t_g)
    sky = rr.sky()
    img = np.where(use_obj[..., None], cbuf, np.where(below[..., None], ground, sky))
    depth = np.where(use_obj, obj, np.where(below, t_g, 0.0)).astype(np.float32)  # 背景は0(Genesisと同じ想定)
    return np.clip(img, 0, 255).astype(np.uint8), depth
