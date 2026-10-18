"""
相合傘 最適角度探索 v2(立ち止まり)

v1 からの主な変更:
  1. どんなモデルでも動くよう、前処理を頑健化
     - 上方向・傘の軸を自動推定(手動上書き可)、スケール/原点を自動正規化
     - 非水密/法線反転/穴あき/縮退面/複数パーツ/地面や遠方のゴミに耐える(表面ボクセル殻のみ使用)
  2. 傘と人の干渉を解決: 傘を上へ持ち上げて(腕を伸ばして)クリアランスを確保。無理なら INFEASIBLE
  3. すり抜け無しを構造的に保証 + --verify で実測検証(表面を必ず横切る線分でミス率を数える)
  4. 実際の雨に近づけた雨粒
     - Marshall-Palmer 粒径分布、粒径依存の終端速度(Atlas式)、風、水量(体積)で濡れを評価
     - 傘に当たった水は縁から滴下(傘の傾き方向の低い縁ほど多く滴る)し、人に落ちれば濡れに加算
     - 衝突時の飛沫(見た目のみ)、人体に当たった位置の濡れ染み(蓄積表示)
  5. 傘が傾いた/複数パーツで構成されていても対応(任意方向の回転対称軸を探索、パーツ一覧・除外、確認図)
  6. OBJ/FBX 対応(自前ローダー)、単位(m/cm/mm)の自動推定、二人の自動検出(A/B分割・軸合わせ・距離調整)
  7. GPU 活用
     - 全角度(--env-batch 個ずつ)を1つのテンソルにまとめて同時にシミュレーション(傘の姿勢は (E,3,3) の行列)
     - シミュレーションのステップ内にホスト同期(.item()/.nonzero())が無い(再投入・滴下の発生も where/cumsum/searchsorted)
     - 映像合成(ぼかし・雨の筋・反射・波紋・ブルーム)も GPU。粒子・画像は最後までGPU上に置く
     - 既定でGPU優先(--device auto): デバイス・SSAA倍率・同時角度数・エンコーダ(NVENC)を自動選択
  8. 傘・人の位置関係を Genesis で描画(傘の姿勢のみ変更するので静止シーンを1回だけレンダリング)

実行例:
  uv run python umbrella_rain_v2.py --couple assets/couple.glb --umbrella assets/umbrella.glb --preview
  uv run python umbrella_rain_v2.py --couple assets/couple.glb --umbrella assets/umbrella.glb --verify
  uv run python umbrella_rain_v2.py --couple assets/couple.glb --umbrella assets/umbrella.glb --phi 0
  uv run python umbrella_rain_v2.py --couple assets/couple.glb --umbrella assets/umbrella.glb --phi 90 --no-video
"""

import warnings

warnings.filterwarnings("ignore", message=".*torch.jit.script.*")  # Genesis 内部の非推奨警告(動作に影響しない)

import argparse
import csv
import sys
import time
import types
from pathlib import Path
import math
import os

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from scipy.spatial.transform import Rotation as Rot

import genesis as gs
from rain_fx import Camera, GpuCamera, RainRenderer
from soft_render import soft_render

V_DRIP = math.pi / 6 * 4.0**3  # 滴下する水滴(直径4mm)の体積 [mm^3]
AXES = {"+x": (0, 1), "-x": (0, -1), "+y": (1, 1), "-y": (1, -1), "+z": (2, 1), "-z": (2, -1)}


# =============================================================================
# 1. モデルの頑健な前処理
# =============================================================================
def unit(ax, sign):
    v = np.zeros(3)
    v[ax] = sign
    return v


def auto_couple_axis(pts):
    lo, hi = np.percentile(pts, [1, 99], axis=0)
    ax = int(np.argmax(hi - lo))  # 立っている人は高さが最大寸法
    sign = 1
    if ax == 0:  # x が上は異例。重心の位置で推定
        sign = 1 if pts[:, 0].mean() > (lo[0] + hi[0]) / 2 else -1
    return ax, sign  # y/z は慣習どおり + を上と仮定(違えば --couple-axis で上書き)


def _perp_basis(a):
    h = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(a, h)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(a, e1)


def sym_score(pts, c, a, nr=16, nt=24):
    """軸 (c, a) まわりの回転対称性からのずれ(小さいほど対称)。
    (半径, 軸方向位置) のビンごとに、角度分布が一様かを調和成分1・2で測る。
    ノイズ由来の期待値(2/n)は差し引く。リブ本数や形状に依存しない。"""
    e1, e2 = _perp_basis(a)
    d = pts - c
    t = d @ a
    x, y = d @ e1, d @ e2
    r = np.hypot(x, y)
    ph = np.arctan2(y, x)
    rmax = np.percentile(r, 99.5) + 1e-9
    tl, th = np.percentile(t, [0.5, 99.5])
    ri = np.clip((r / rmax * nr).astype(int), 0, nr - 1)
    ti = np.clip(((t - tl) / (th - tl + 1e-9) * nt).astype(int), 0, nt - 1)
    b = ri * nt + ti
    nb = nr * nt
    cnt = np.bincount(b, minlength=nb).astype(float)
    R2 = 0.0
    for h in (1, 2):
        cs = np.bincount(b, weights=np.cos(h * ph), minlength=nb)
        sn = np.bincount(b, weights=np.sin(h * ph), minlength=nb)
        R2 = R2 + (cs**2 + sn**2) / np.maximum(cnt, 1) ** 2
    excess = np.maximum(R2 - 2.0 / np.maximum(cnt, 1), 0)
    w = cnt.copy()
    w[:nt] = 0  # 軸付近(最内周)は角度が不定なので除外
    w[cnt < 20] = 0
    ws = w.sum()
    return float((excess * w).sum() / ws) if ws > 0 else 1.0


def fibonacci_dirs(n):
    i = np.arange(n) + 0.5
    ph = np.arccos(1 - i / n)
    th = math.pi * (1 + 5**0.5) * i
    return np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], 1)


def fit_umbrella_axis(pts, part_axes, rng, n_dirs=400, verbose=True):
    """任意方向の回転対称軸を探す。
    1) 候補: 球面一様分布 + x/y/z + 各パーツの主成分軸(細長いパーツ=軸の手がかり)
    2) 粗い探索で上位を選び、軸の向きと中心位置を乱数で局所探索して精密化
    戻り値: (軸単位ベクトル(符号なし), 軸が通る点, 対称スコア, 全候補中央値に対する比)"""
    c0 = pts.mean(0)
    ext = float(np.ptp(np.percentile(pts, [1, 99], axis=0), axis=0).max())
    sub = pts[rng.choice(len(pts), min(20000, len(pts)), replace=False)]
    cands = [np.eye(3)[i] for i in range(3)] + list(fibonacci_dirs(n_dirs)) + [v for v in part_axes]
    scores = np.array([sym_score(sub, c0, a) for a in cands])
    fine = pts[rng.choice(len(pts), min(100000, len(pts)), replace=False)]
    best = (np.inf, None, None)
    for ti in np.argsort(scores)[:5]:
        a, c = np.array(cands[ti], float), c0.copy()
        s = sym_score(fine, c, a)
        for it in range(120):
            sg = 0.97**it
            a2 = a + rng.normal(size=3) * math.radians(4.0) * sg
            a2 /= np.linalg.norm(a2)
            c2 = c + rng.normal(size=3) * 0.03 * ext * sg
            s2 = sym_score(fine, c2, a2)
            if s2 < s:
                a, c, s = a2, c2, s2
        if s < best[0]:
            best = (s, a, c)
    s, a, c = best
    # 軸の通る位置を、縁(最大半径付近)のリングの中心に合わせて微補正(乱数探索の残差を除く)
    e1, e2 = _perp_basis(a)
    d = pts - c
    for e in (e1, e2):  # 垂直面での両端(縁)の中点 = 傘の中心。リング平均より安定
        x = d @ e
        c = c + e * float((np.percentile(x, 0.5) + np.percentile(x, 99.5)) / 2)
    return a, c, s, s / (np.median(scores) + 1e-12)


def umbrella_sign(pts, c, a):
    """軸の向きを決める。縁(最大半径の位置)より片側には布(ドーム)が広がり、
    反対側には柄だけがある。表面積が多い側を『上』とする。戻り値: (符号, 確信度0-1)"""
    d = pts - c
    t = d @ a
    r = np.linalg.norm(d - t[:, None] * a, axis=1)
    rim = t[r > 0.95 * np.percentile(r, 99.5)]
    t_rim = float(np.median(rim)) if len(rim) else float(np.median(t))
    frac = float((t > t_rim).mean())
    return (1 if frac >= 0.5 else -1), abs(frac - 0.5) * 2


def part_table(parts):
    """パーツ一覧を表示し、各パーツの主成分軸 [(固有ベクトル3x3, 細長度)] を返す。"""
    tot = sum(p.area for p in parts) + 1e-12
    out = []
    for i, p in enumerate(parts):
        s, _ = trimesh.sample.sample_surface(p, 3000)
        w, V = np.linalg.eigh(np.cov(s.T))
        w = np.maximum(w, 1e-18)
        elong = math.sqrt(w[2] / w[1])
        name = str(p.metadata.get("name", p.metadata.get("node", "?")))
        print(f"  part[{i}] {name[:24]:24s} 面積{100 * p.area / tot:5.1f}%  寸法={np.round(p.extents, 3)}  細長度={elong:.1f}")
        out.append((V, elong))
    return out


def save_inspect(path, pts, labels, c, a, title):
    """軸推定の確認図: 左=半径-軸方向の断面形(傘のシルエットになるはず) / 中=軸方向から見た図(円盤になるはず) / 右=横から"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e1, e2 = _perp_basis(a)
    d = pts - c
    t, x, y = d @ a, d @ e1, d @ e2
    sel = np.random.default_rng(0).choice(len(pts), min(15000, len(pts)), replace=False)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].scatter(np.hypot(x, y)[sel], t[sel], c=labels[sel], s=1, cmap="tab20")
    ax[0].set_title("profile: radius vs axis (should look like the umbrella outline)")
    ax[1].scatter(x[sel], y[sel], c=labels[sel], s=1, cmap="tab20")
    ax[1].set_title("view along axis (should be a symmetric disk)")
    ax[2].scatter(x[sel], t[sel], c=labels[sel], s=1, cmap="tab20")
    ax[2].set_title("side view (dome should be on top)")
    for q in ax:
        q.set_aspect("equal")
    fig.suptitle(title)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


PALETTE = [  # 人物A(x<0) / 人物B(x>0)
    dict(skin=(218, 166, 134), hair=(42, 31, 27), top=(163, 184, 212), pants=(54, 58, 70), shoes=(34, 31, 31)),
    dict(skin=(233, 186, 156), hair=(80, 52, 37), top=(190, 108, 118), pants=(46, 60, 90), shoes=(226, 226, 229)),
]
EYE_WHITE, EYE_DARK = (238, 238, 236), (32, 24, 22)


def _is_plain_grey(g):
    try:
        fc = np.asarray(g.visual.face_colors)
        return bool(np.all(fc.reshape(-1, fc.shape[-1])[:, :3] == 150))
    except Exception:
        return False


def humanize_couple(parts):
    """3ds Max のワイヤー表示色(役割と無関係な緑・紫・赤など)を、自然な人間の配色(肌・髪・服・靴・目)に置き換える。
    パーツごとに、人物ごとの身長 Hp に対する高さ・大きさ・位置から役割を判定する(座標は正規化後: z が高さ、x が二人を結ぶ軸)。
      目      : 小さく(面積<0.012m²)頭の高さ(>0.85Hp)にある。三角形が多いものは白目、他は黒目
      靴      : 上端が 0.12Hp 未満
      頭まわり: 下端が 0.5Hp より上で、重心が 0.8Hp より上 または 上端が 0.95Hp 以上。その中で最も上に届くものが髪、残りは肌(頭・首)
      全身の肌: 下端が地面付近で上端が 0.9Hp 以上 → 肌(服は別パーツ)
      服      : 残りは重心が 0.55Hp より上ならトップス、下ならズボン
      体と服が1枚のメッシュ(全身の肌が無く、服のパーツも無い人物の体): 面ごとに
              首・頭=肌、靴、手(腕を下ろした位置)=肌、胴と袖=トップス、脚=ズボン と塗り分ける
    配色は、「体と服が一体のメッシュ」を持つ人物にパレット0、そうでない人物にパレット1を使う(両方同じなら x<0 を0)。"""
    feats = []
    for g in parts:
        V, F = np.asarray(g.vertices), np.asarray(g.faces)
        t = V[F]
        feats.append(dict(
            area=float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum()),
            zmin=float(V[:, 2].min()), zmax=float(V[:, 2].max()), zc=float(V[:, 2].mean()), cx=float(V[:, 0].mean()), n=len(F)))
    who = [0 if f["cx"] < 0 else 1 for f in feats]
    roles, Hps = {0: {}, 1: {}}, {}
    for pid in (0, 1):
        idx = [i for i, w in enumerate(who) if w == pid]
        if not idx:
            continue
        Hp = max(feats[i]["zmax"] for i in idx)
        Hps[pid] = Hp
        role = roles[pid]
        for i in idx:
            f = feats[i]
            if f["area"] < 0.012 and f["zc"] > 0.85 * Hp:
                role[i] = "eye_white" if f["n"] >= 600 else "eye_dark"
            elif f["area"] < 0.012:
                role[i] = "eye_dark"
            elif f["zmax"] < 0.12 * Hp:
                role[i] = "shoes"
        rest = [i for i in idx if i not in role]
        head = [i for i in rest if feats[i]["zmin"] > 0.5 * Hp and (feats[i]["zc"] > 0.8 * Hp or feats[i]["zmax"] >= 0.95 * Hp)]
        if head:
            hair = max(head, key=lambda i: feats[i]["zmax"])
            for i in head:
                role[i] = "hair" if i == hair else "skin"
        rest = [i for i in idx if i not in role]
        has_skin_body = any(feats[i]["zmin"] < 0.05 * Hp and feats[i]["zmax"] >= 0.9 * Hp for i in rest)
        for i in rest:
            f = feats[i]
            if f["zmin"] < 0.05 * Hp and f["zmax"] >= 0.9 * Hp:
                role[i] = "skin"
            elif not has_skin_body and f["zmin"] < 0.05 * Hp and f["zmax"] > 0.6 * Hp:
                role[i] = "painted"
            else:
                role[i] = "top" if f["zc"] > 0.55 * Hp else "pants"
    painted = [pid for pid in (0, 1) if "painted" in roles[pid].values()]
    pal_of = {0: 0, 1: 1}
    if len(painted) == 1:
        pal_of = {painted[0]: 0, 1 - painted[0]: 1}
    summary = {0: {}, 1: {}}
    for pid in (0, 1):
        pal, Hp = PALETTE[pal_of[pid]], Hps.get(pid)
        for i, r in roles[pid].items():
            g = parts[i]
            n = len(g.faces)
            if r == "painted":
                V, F = np.asarray(g.vertices), np.asarray(g.faces)
                C = V[F].mean(1)
                z, dx = C[:, 2], np.abs(C[:, 0] - V[:, 0].mean())
                col = np.empty((n, 4), np.uint8)
                col[:] = (*pal["pants"], 255)
                col[z >= 0.53 * Hp] = (*pal["top"], 255)  # 胴と袖
                col[(z > 0.43 * Hp) & (z < 0.59 * Hp) & (dx > 0.115 * Hp)] = (*pal["skin"], 255)  # 下ろした手
                col[z < 0.05 * Hp] = (*pal["shoes"], 255)
                col[z > 0.835 * Hp] = (*pal["skin"], 255)  # 首・頭
                g.visual.face_colors = col
            else:
                c = {"skin": pal["skin"], "hair": pal["hair"], "top": pal["top"], "pants": pal["pants"], "shoes": pal["shoes"],
                     "eye_white": EYE_WHITE, "eye_dark": EYE_DARK}[r]
                g.visual.face_colors = [c[0], c[1], c[2], 255]
            summary[pid][r] = summary[pid].get(r, 0) + 1
    print("  人物の色: ワイヤー表示色を自然な配色(肌・髪・服・靴・目)に置き換えました "
          + " / ".join(f"{'AB'[p]}: " + ", ".join(f"{k}×{v}" for k, v in sorted(d.items())) for p, d in summary.items() if d))


def infer_scale(raw_size, lo, hi, fallback):
    """元データの単位を推定(m/cm/mm)。想定サイズ範囲[lo,hi]m に入る倍率を採用、無ければ既定サイズに正規化。"""
    for f, nm in ((1.0, "m"), (0.01, "cm"), (0.001, "mm")):
        if lo <= raw_size * f <= hi:
            return f, f"単位を{nm}と推定"
    return fallback / raw_size, f"想定サイズ範囲外のため {fallback}m に正規化"


def split_two(s, ref):
    """1次元座標 s(二人を結ぶ軸方向)を2クラスタに分け、二人の間の谷の位置を返す。
    戻り値: (谷の位置, 中心A, 中心B, 二人と判定できたか)。ref=身長(元の単位)"""
    a_, b_ = np.percentile(s, 10), np.percentile(s, 90)
    for _ in range(50):
        m = np.abs(s - a_) < np.abs(s - b_)
        if m.all() or (~m).all():
            break
        a2, b2 = s[m].mean(), s[~m].mean()
        done = abs(a2 - a_) + abs(b2 - b_) < 1e-9 * (np.ptp(s) + 1e-12)
        a_, b_ = a2, b2
        if done:
            break
    if abs(b_ - a_) < 1e-9:
        return float(np.median(s)), a_, b_, False
    lo_, hi_ = min(a_, b_), max(a_, b_)
    hist, edges = np.histogram(s, bins=120, range=(lo_, hi_))
    sm = np.convolve(hist, np.ones(5) / 5, mode="same")
    j = int(np.argmin(sm[5:-5])) + 5
    thr = float((edges[j] + edges[j + 1]) / 2)
    ok = (hi_ - lo_) > 0.25 * ref and sm[j] < 0.3 * np.median(sm[sm > 0])
    return thr, lo_, hi_, bool(ok)


def prep_model(path, kind, axis_arg, size, grip, cache, keep_all, a=None):
    if str(path).lower().endswith(".fbx"):  # trimesh は FBX を読めないため自前ローダー(階層変換込み)
        from fbx_loader import load_fbx_parts

        raw, _ = load_fbx_parts(path)
        dumped = []
        for q in raw:
            g = trimesh.Trimesh(vertices=q["V"], faces=q["F"], process=False, metadata={"name": q["name"]})
            g.visual.face_colors = [150, 150, 150, 255]  # FBXの材質は読まないので無地グレー
            dumped.append(g)
    elif str(path).lower().endswith(".obj"):  # 3ds Max 等の大きなOBJも速く読める自前ローダー(パーツ名・材質色を保持)
        from obj_loader import load_obj_parts, mtl_color

        dumped = []
        for q in load_obj_parts(path):
            g = trimesh.Trimesh(vertices=q["V"], faces=q["F"], process=False,
                                metadata={"name": q["name"], "wire": mtl_color(q["mtl"]) is not None})  # 3ds Max のワイヤー表示色か
            col = mtl_color(q["mtl"]) or (150, 150, 150)  # 'wire_RRRGGGBBB' 形式なら色として使う
            g.visual.face_colors = [col[0], col[1], col[2], 255]
            dumped.append(g)
    else:
        dumped = trimesh.load(path, force="scene").dump()
    parts = []
    for g in dumped:
        if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0:
            try:
                g.remove_infinite_values()
                g.update_faces(g.nondegenerate_faces())
                g.remove_unreferenced_vertices()
            except Exception:
                pass
            if len(g.faces) > 0 and np.isfinite(g.vertices).all():
                parts.append(g)
    if not parts:
        raise RuntimeError(f"{path}: 三角形メッシュが見つかりません(点群/線のみ?)")

    removed = 0
    if not keep_all and len(parts) > 1:
        ext_max = max(p.extents.max() for p in parts)
        if kind == "couple":  # 地面のような薄く巨大なパーツを除去
            kept = [p for p in parts if not (p.extents.min() < 0.01 * p.extents.max() and p.extents.max() >= 0.8 * ext_max)]
            parts = kept or parts
        main = max(parts, key=lambda p: p.area)
        mc, mext = main.bounds.mean(0), main.extents.max()
        kept = [p for p in parts if np.linalg.norm(p.bounds.mean(0) - mc) < 3 * mext]
        removed = len(dumped) - len(kept)
        parts = kept

    ucol = getattr(a, "umbrella_color", None)
    if kind == "umbrella" and not ucol:  # 指定が無く、材質の色情報も無い(全パーツが既定のグレー)なら紺色にする

        def _plain(g):
            try:
                fc = np.asarray(g.visual.face_colors)
                return bool(np.all(fc.reshape(-1, fc.shape[-1])[:, :3] == 150))
            except Exception:
                return False

        if all(_plain(g) for g in parts):
            ucol = (35, 60, 125)
            print("  傘の色: 材質の色情報が無いため既定の紺色を適用します(--umbrella-color R G B で変更)")
    if kind == "umbrella" and ucol:  # 面積の大きいパーツ(=布)だけ着色。骨・柄は金属グレーのまま
        tot_a = sum(g.area for g in parts) + 1e-12
        for g in parts:
            if g.area / tot_a > 0.10:
                g.visual.face_colors = [int(ucol[0]), int(ucol[1]), int(ucol[2]), 255]
    ids = list(range(len(parts)))
    part_info = []
    if kind == "umbrella":
        print(f"[umbrella] パーツ一覧 ({len(parts)}個)")
        part_info = part_table(parts)
        drop = {int(x) for x in str(getattr(a, "umbrella_drop_parts", "") or "").split(",") if x.strip()}
        if drop:
            keep = [(i, p) for i, p in zip(ids, parts) if i not in drop]
            ids, parts = [i for i, _ in keep], [p for _, p in keep]
            print(f"  パーツ {sorted(drop)} を除外して続行")
        part_info = [part_info[i] for i in ids]
    m = trimesh.util.concatenate(parts)
    tot_area = sum(p.area for p in parts) + 1e-12
    pl, ll = [], []
    for i, p in zip(ids, parts):  # 面積に比例してパーツごとにサンプリング(ラベル付き)
        n = max(300, int(200000 * p.area / tot_area))
        s_, _ = trimesh.sample.sample_surface(p, n)
        pl.append(s_)
        ll.append(np.full(n, i))
    pts, labels = np.vstack(pl), np.concatenate(ll)

    names = "xyz"
    pair = dict(ok=False, sep=0.0)
    if kind == "couple":
        if axis_arg == "auto":
            ax, sign = auto_couple_axis(pts)
            how = "自動"
        else:
            ax, sign = AXES[axis_arg]
            how = "指定"
        T1 = trimesh.geometry.align_vectors(unit(ax, sign), [0, 0, 1])
        p1 = pts @ T1[:3, :3].T
        lo, hi = np.percentile(p1, [0.1, 99.9], axis=0)
        raw_h = float(hi[2] - lo[2])
        if size:
            k, how_u = size / raw_h, f"身長{size}mに正規化"
        else:
            k, how_u = infer_scale(raw_h, 1.4, 2.3, 1.7)
        print(f"[couple] 上方向 = {'+' if sign > 0 else '-'}{names[ax]} ({how}) / スケール: {how_u} (x{k:.5f}) / 高さ={raw_h * k:.2f}m / 除去パーツ={removed} / 三角形数={len(m.faces)}")
        # 二人を検出 → 二人を結ぶ軸を x に合わせ、二人の間(谷)を原点にする(wetness の A/B 分割もここ)
        _, V = np.linalg.eigh(np.cov(p1[:, :2].T))
        u = V[:, 1]
        thr, ca, cb, ok = split_two(p1[:, :2] @ u, raw_h)
        Rz = np.eye(4)
        if ok:
            Rz[:3, :3] = Rot.from_euler("z", -math.atan2(u[1], u[0])).as_matrix()
            pair = dict(ok=True, sep=(cb - ca) * k)
            print(f"  二人を検出: 中心間距離={pair['sep']:.2f} m (二人の間を原点・結ぶ軸を x に合わせます。x<0 が人A、x>0 が人B)")
        else:
            print("  ⚠ 二人を分離して検出できませんでした(1人/密着/重なり)。bbox中心を原点にし、A/B は x=0 で分割します。")
        p2 = p1 @ Rz[:3, :3].T
        lo2, hi2 = np.percentile(p2, [0.1, 99.9], axis=0)
        xc = thr if ok else float((lo2[0] + hi2[0]) / 2)
        Tr = np.eye(4)
        Tr[:3, 3] = [-xc * k, -(lo2[1] + hi2[1]) / 2 * k, -lo2[2] * k]
        T = Tr @ np.diag([k, k, k, 1.0]) @ Rz @ T1
    else:
        rng = np.random.default_rng(0)
        c = pts.mean(0)
        vec_arg = getattr(a, "umbrella_axis_vec", None)
        warn = ""
        if vec_arg is not None:
            vec = np.array(vec_arg, float)
            vec /= np.linalg.norm(vec)
            how = "ベクトル指定"
            score = sym_score(pts[::5], c, vec)
        elif axis_arg != "auto":
            ax, sign = AXES[axis_arg]
            vec = unit(ax, sign)
            how = f"軸指定 {axis_arg}"
            score = sym_score(pts[::5], c, vec)
        else:
            cand = [V[:, j] for V, _ in part_info for j in range(3)]
            ax_u, c, score, ratio = fit_umbrella_axis(pts, cand, rng)
            sg, sconf = umbrella_sign(pts, c, ax_u)
            vec = ax_u * sg
            how = f"自動(任意方向) 確信度ratio={ratio:.2f}(低いほど良い) 上下の確信={sconf:.2f}"
            if ratio > 0.3:
                warn += "  ⚠ 回転対称性が弱い: 余計なパーツ(手/台/看板)が混ざっていないか確認し、--umbrella-drop-parts で除外してください\n"
            if sconf < 0.15:
                warn += "  ⚠ 上下の判定が曖昧: inspect_umbrella.png を見て、逆なら --umbrella-axis-vec に逆向きを指定してください\n"
            for V, el in part_info:  # 独立した手がかり: 細長いパーツ(柄)の向きと一致するか
                if el > 5:
                    ang = math.degrees(math.acos(min(1.0, abs(float(V[:, 2] @ vec)))))
                    if ang > 5:
                        warn += f"  ⚠ 細長いパーツ(柄?)の向きと推定軸が {ang:.1f}° ずれています\n"
        print(f"[umbrella] 軸(元データ座標) = {np.round(vec, 4)} / {how} / 対称スコア={score:.4f} / パーツ={len(parts)} / 除去={removed}")
        if warn:
            print(warn, end="")
        os.makedirs(cache, exist_ok=True)
        save_inspect(os.path.join(cache, "inspect_umbrella.png"), pts, labels, c, vec, f"umbrella axis={np.round(vec, 3)} score={score:.4f}")
        T1 = trimesh.geometry.align_vectors(vec, [0, 0, 1])
        Tc = np.eye(4)
        Tc[:3, 3] = -c
        p1 = (pts - c) @ T1[:3, :3].T
        zlo = np.percentile(p1[:, 2], 0.1)
        raw_d = float(2 * np.percentile(np.hypot(p1[:, 0], p1[:, 1]), 99.9))  # 直径 = 縁までの半径×2
        if size:
            k, how_u = size / raw_d, f"直径{size}mに正規化"
        else:
            k, how_u = infer_scale(raw_d, 0.8, 1.6, 1.1)
        print(f"  傘のスケール: {how_u} (x{k:.5f}) / 直径={raw_d * k:.2f}m")
        Tr = np.eye(4)
        Tr[:3, 3] = [0.0, 0.0, -zlo * k - grip]
        T = Tr @ np.diag([k, k, k, 1.0]) @ T1 @ Tc

    for g in parts:
        g.apply_transform(T)
    if kind == "couple" and getattr(a, "couple_spacing", None):
        if pair["ok"]:
            dx = (a.couple_spacing - pair["sep"]) / 2
            straddle = 0
            for g in parts:
                if g.bounds[0, 0] < -0.05 and g.bounds[1, 0] > 0.05:
                    straddle += 1  # 二人にまたがるパーツ(動かせない)
                g.apply_translation([-dx if g.vertices[:, 0].mean() < 0 else dx, 0.0, 0.0])
            print(f"  二人の中心間距離を {pair['sep']:.2f} → {a.couple_spacing:.2f} m に調整" + (f"(⚠ 二人にまたがるパーツ {straddle} 個は歪みます)" if straddle else ""))
        else:
            print("  ⚠ 二人を検出できていないため --couple-spacing は無視します")
    if kind == "couple":
        mode = getattr(a, "couple_colors", "auto")
        unnatural = any(g.metadata.get("wire") for g in parts) or all(_is_plain_grey(g) for g in parts)
        if mode == "natural" or (mode == "auto" and unnatural):
            if pair["ok"]:
                humanize_couple(parts)
            else:
                print("  ⚠ 二人を検出できていないため、人物の色の自動配色は行いません")
    sc = trimesh.Scene(parts)
    phys = trimesh.util.concatenate(parts)
    fcs = []  # 面ごとの色(ソフトレンダラ用)。パーツの並び順 = concatenate後の面の並び順
    for g in parts:
        fc = np.asarray(g.visual.face_colors, np.uint8)
        fcs.append(np.tile(fc[:4], (len(g.faces), 1)) if fc.ndim == 1 else fc)
    phys.metadata["face_colors"] = np.vstack(fcs)
    glb = os.path.join(cache, f"{kind}_prepared.glb")
    try:
        sc.export(glb)
    except Exception as e:  # 表示用データが壊れていても形状だけで書き出す
        print(f"  (警告) テクスチャ付き書き出し失敗 -> 形状のみ: {e}")
        trimesh.Trimesh(phys.vertices, phys.faces).export(glb)
    print(f"  正規化後サイズ(x,y,z)[m] = {np.round(phys.extents, 3)}")
    return phys, glb


# =============================================================================
# 2. ボクセル殻(すり抜け防止の要)と占有判定
# =============================================================================
def voxel_shell(mesh, pitch):
    """表面だけをボクセル化(穴・法線反転・非水密に依存しない)。"""
    vg = mesh.voxelized(pitch)
    if np.prod(vg.matrix.shape) > 4e8:
        raise MemoryError("ボクセルが巨大すぎます。--voxel を大きくしてください。")
    return np.asarray(vg.matrix).copy(), np.array(vg.transform[:3, 3])


def to_grid(raw, pitch, r, dev):
    """殻を r ボクセル太らせる(境界で欠けないよう r だけパディング)。
    r=1 → 厚み3ボクセル。サンプル間隔 <= pitch/2 なので横切る線分は必ず5点以上が殻内に入る。"""
    m = torch.from_numpy(raw[0]).to(dev)
    o = raw[1]
    if r > 0:
        m = F.pad(m.float(), (r,) * 6)
        m = F.max_pool3d(m[None, None], 2 * r + 1, 1, r)[0, 0] > 0
        o = o - r * pitch
    return m, torch.tensor(o, dtype=torch.float32, device=dev), pitch


def occ(grid, pts):
    m, o, p = grid
    idx = torch.round((pts - o) / p).long()
    sx, sy, sz = m.shape  # Python の整数で比較(毎回デバイスにテンソルを作らない)
    ix, iy, iz = idx[:, 0], idx[:, 1], idx[:, 2]
    inb = (ix >= 0) & (ix < sx) & (iy >= 0) & (iy < sy) & (iz >= 0) & (iz < sz)
    return m[ix.clamp(0, sx - 1), iy.clamp(0, sy - 1), iz.clamp(0, sz - 1)] & inb


def swept(grids, P0, disp, pitch):
    """各粒子の移動線分 P0 -> P0+disp を (pitch/2) 以下の間隔で走査し、最初に当たった物体を返す。
    grids: [(grid, R, H)] R=None ならワールド座標、そうでなければ (p-H)@R でローカルへ。
    戻り値: tag (-1=当たらず / 0..=grids の番号), pos(当たった位置)"""
    N = len(P0)
    tag = torch.full((N,), -1, dtype=torch.long, device=P0.device)
    pos = torch.zeros_like(P0)
    n_sub = max(1, math.ceil(float(disp.norm(dim=1).max()) / (0.5 * pitch)))
    for k in range(1, n_sub + 1):
        Pk = P0 + disp * (k / n_sub)
        free = tag < 0
        for gi, (g, R, H) in enumerate(grids):
            q = Pk if R is None else (Pk - H) @ R
            h = free & occ(g, q)
            tag[h] = gi
            pos[h] = Pk[h]
            free &= ~h
    return tag, pos


def swept_env(ugrid, pgrid, R, H, has_u, P0, disp, pitch, n_sub):
    """全環境(角度)をまとめた掃引衝突判定。P0, disp: (E,N,3) / R: (E,3,3) / H: (E,3) / has_u: (E,) bool。
    n_sub は呼び出し側で最大速度から決めた定数なので、ホストとの同期(.item())が無い。
    戻り値: tag (E,N) [-1=当たらず, 0=傘, 1=人], pos (E,N,3)"""
    E, N, _ = P0.shape
    tag = torch.full((E, N), -1, dtype=torch.long, device=P0.device)
    pos = torch.zeros_like(P0)
    Hb = H[:, None, :]
    for k in range(1, n_sub + 1):
        Pk = P0 + disp * (k / n_sub)
        free = tag < 0
        hu = free & has_u[:, None] & occ(ugrid, ((Pk - Hb) @ R).reshape(-1, 3)).reshape(E, N)  # 傘ローカル = R^T(p-H)
        tag = torch.where(hu, torch.zeros_like(tag), tag)
        pos = torch.where(hu[..., None], Pk, pos)
        free = free & ~hu
        hp = free & occ(pgrid, Pk.reshape(-1, 3)).reshape(E, N)
        tag = torch.where(hp, torch.ones_like(tag), tag)
        pos = torch.where(hp[..., None], Pk, pos)
    return tag, pos


# =============================================================================
# 3. アセット構築・配置(干渉解決)
# =============================================================================
class Assets:
    pass


def tilt(theta_deg, phi_deg):
    ph, th = math.radians(phi_deg), math.radians(theta_deg)
    axis = np.array([-math.sin(ph), math.cos(ph), 0.0])
    rot = Rot.from_rotvec(th * axis)
    q = rot.as_quat()
    return rot.as_matrix(), np.array([q[3], q[0], q[1], q[2]])  # wxyz


def build_assets(a, dev):
    cache = os.path.join(a.out, "_cache")
    os.makedirs(cache, exist_ok=True)
    A = Assets()
    A.cm, A.couple_glb = prep_model(a.couple, "couple", a.couple_axis, a.couple_height, 0.0, cache, a.keep_all, a)
    A.um, A.umb_glb = prep_model(a.umbrella, "umbrella", a.umbrella_axis, a.umbrella_diameter, a.grip_offset, cache, a.keep_all, a)
    pitch = a.voxel
    rc = math.ceil(a.clearance / pitch)
    A.p_raw, A.u_raw = voxel_shell(A.cm, pitch), voxel_shell(A.um, pitch)
    A.pgrid, A.ugrid = to_grid(A.p_raw, pitch, 1, dev), to_grid(A.u_raw, pitch, 1, dev)
    A.pgrid_c, A.ugrid_c = to_grid(A.p_raw, pitch, 1 + rc, dev), to_grid(A.u_raw, pitch, 1 + rc, dev)

    def smp(m, n):
        return torch.tensor(trimesh.sample.sample_surface(m, n)[0], dtype=torch.float32, device=dev)

    A.u_pts, A.p_pts = smp(A.um, 40000), smp(A.cm, 40000)
    A.couple_h = float(A.cm.bounds[1, 2])  # 正規化後の実際の高さ[m]
    A.hold = np.array(a.hold if a.hold else [0.0, 0.0, A.couple_h * a.hold_z_ratio], dtype=np.float32)
    A.dev = dev
    return A


def interferes(A, R, H):
    if occ(A.pgrid_c, A.u_pts @ R.T + H).any():
        return True
    return bool(occ(A.ugrid_c, (A.p_pts - H) @ R).any())


def place(A, a, theta):
    """傘を theta だけ傾け、人と干渉しない最小の持ち上げ量を探す。戻り値の H が None なら実現不可能。"""
    R_np, quat = tilt(theta, a.phi)
    R = torch.tensor(R_np, dtype=torch.float32, device=A.dev)
    H0 = torch.tensor(A.hold, device=A.dev)
    max_lift = max(0.0, a.max_hold_ratio * A.couple_h - float(A.hold[2]))
    for lift in np.arange(0.0, max_lift + 1e-9, 0.01):
        H = H0 + torch.tensor([0.0, 0.0, float(lift)], device=A.dev)
        if not interferes(A, R, H):
            return R, quat, H, float(lift)
    return R, quat, None, None


def verify(A, a, n=30000, quiet=False):
    """すり抜け検証: 必ず表面を横切る線分(表面点を通る)を大量に作り、ボクセル走査が見逃す数を数える。
    戻り値: (見逃し数, 検査した線分数)"""
    rng = np.random.default_rng(0)
    if not quiet:
        print(f"最大1ステップ移動量(想定) ≈ {(9.65 - 10.3 * math.exp(-0.6 * 5.5)) * a.dt * 100:.1f} cm / ボクセル {a.voxel * 100:.1f} cm")
    tot_miss = tot_n = 0
    for name, mesh, grid in [("couple", A.cm, A.pgrid), ("umbrella", A.um, A.ugrid)]:
        p, _ = trimesh.sample.sample_surface(mesh, n)
        d = rng.normal(size=(n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        L = rng.uniform(0.002, 0.045, n)
        u = rng.uniform(0, 1, n)
        p0 = p - d * (u * L)[:, None]
        t = lambda x: torch.tensor(x, dtype=torch.float32, device=A.dev)
        tag, _ = swept([(grid, None, None)], t(p0), t(d * L[:, None]), grid[2])
        miss = int((tag < 0).sum())
        tot_miss, tot_n = tot_miss + miss, tot_n + n
        if not quiet:
            print(f"[{name}] 表面を横切る {n} 線分のうち、見逃し = {miss} ({100 * miss / n:.4f}%)")
    return tot_miss, tot_n


# =============================================================================
# 4. 粒子雨(粒径分布・終端速度・風・滴下)
# =============================================================================
class FrameRec:
    """1フレーム分の衝突位置の記録(映像の濡れ・飛沫・波紋用)。すべてデバイス上のテンソル。"""

    def __init__(self):
        self.items = {"p": [], "u": [], "g": []}  # 人 / 傘 / 地面 (環境番号, 位置[, 粒径])

    def add(self, kind, env, pos):
        self.items[kind].append((env, pos))

    def get(self, e, kind, dev):
        out = [pos[env == e] for env, pos in self.items[kind]]
        return torch.cat(out) if out else torch.zeros((0, 3), device=dev)


class RainSim:
    """E 個の環境(傘の角度ごと)を1つのテンソルにまとめて同時に進める。
    ステップ内に .item()/.nonzero() などのホスト同期が無い(再投入・滴下の発生も where/cumsum/searchsorted で行う)。"""

    def __init__(self, a, A, E):
        self.a, self.A, self.dev, self.E = a, A, A.dev, E
        self.pitch = A.pgrid[2]
        self.L, self.Hb = a.box, a.box_height
        R = a.rain_mm_h
        self.lam = 4.1 * R**-0.21  # Marshall-Palmer の傾き [1/mm]
        # 粒径の離散分布(0.4〜5.5mm)。p=濃度分布(0.4mm未満は下限へ丸める) / q=流束重み p*vt(再投入用)。
        # 遅い小粒ほど箱に長く留まるため、再投入を p で引くと定常状態で小粒に偏り、水量が不足する。
        # 地面に達する(=再投入される)頻度は p*vt に比例するので、再投入は q から引く。
        Dg = np.linspace(0.4, 5.5, 600)
        mid = np.concatenate([[0.0], (Dg[:-1] + Dg[1:]) / 2, [np.inf]])
        mass = np.diff(1.0 - np.exp(-self.lam * mid))
        vt_g = np.clip(9.65 - 10.3 * np.exp(-0.6 * Dg), 0.8, None)
        vol_g = math.pi / 6 * Dg**3
        q = mass * vt_g
        t32 = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=self.dev)
        self.Dg, self.dD = t32(Dg), float(Dg[1] - Dg[0])
        self.cdf_p, self.cdf_q = t32(np.cumsum(mass) / mass.sum()), t32(np.cumsum(q) / q.sum())
        # 水量の流束 [mm^3/m^2/s] = nt * E[vol*vt] が降水強度 R[mm/h] (=R*277.8 mm^3/m^2/s) に一致するよう濃度を決める
        evt = float((mass * vol_g * vt_g).sum())
        nt = R * 277.8 / evt  # 粒子数濃度 [個/m^3]
        full = nt * self.L * self.L * self.Hb
        self.n_sd = int(min(a.max_drops, full))  # 箱の中に定常的に存在する粒子数(間引き後)
        self.weight = full / self.n_sd  # 粒子数を間引いた分は水量に上乗せして降水強度を保つ
        self.n = int(self.n_sd * 1.08) + 64  # プール容量(影の分だけ実際の粒子数は n_sd より少なくなる)
        # 上空からの注入レート [個/s] = 濃度 × 平均終端速度 × 面積。地面や障害物での消滅とは無関係に一定。
        self.rate = (nt / self.weight) * float((mass * vt_g).sum()) * self.L * self.L
        if self.weight > 1.01 and not getattr(A, "_info_printed", False):
            print(f"(情報) 雨粒 {int(full)} 個 → {self.n_sd} 個に間引き。1粒の水量を x{self.weight:.2f} で補正")
            A._info_printed = True
        self.wind = torch.tensor([a.wind_x, a.wind_y], dtype=torch.float32, device=self.dev)
        self.nd = 30000
        vmax = math.hypot(math.hypot(a.wind_x, a.wind_y), 9.4)  # 終端速度の上限+風 → 掃引の分割数を定数にする
        self.n_sub = max(1, math.ceil(vmax * a.dt / (0.5 * self.pitch)))
        u = A.u_pts
        r = torch.linalg.norm(u[:, :2], dim=1)
        ring = u[r > 0.93 * torch.quantile(r, 0.99)]
        self.ring = ring if len(ring) else u  # 傘の縁(傘ローカル座標)。滴下の発生源
        self.reset()

    def _draw(self, shape, flux):
        """粒径をテーブル引きでサンプル(同期なし)。flux=True なら流束重み(再投入用)、False なら濃度分布(初期配置用)。"""
        cdf = self.cdf_q if flux else self.cdf_p
        idx = torch.searchsorted(cdf, torch.rand(*shape, device=self.dev)).clamp(0, len(self.Dg) - 1)
        D = self.Dg[idx] + (torch.rand(*shape, device=self.dev) - 0.5) * self.dD * (idx > 0).float()  # ビン内を連続化
        D = D.clamp(0.4, 5.5)
        vt = (9.65 - 10.3 * torch.exp(-0.6 * D)).clamp(min=0.8)  # Atlas et al. 1973
        return vt, math.pi / 6 * D**3  # 終端速度[m/s], 体積[mm^3]

    def reset(self):
        torch.manual_seed(self.a.seed)
        E, N, dev = self.E, self.n, self.dev
        self.vt, self.vol = self._draw((E, N), flux=False)
        self.alive = torch.zeros(E, N, dtype=torch.bool, device=dev)
        self.alive[:, :self.n_sd] = True  # 初期は箱全体に濃度 c で一様に配置(残りのスロットは空き)
        self.in_carry = torch.zeros(E, device=dev)  # 注入数の端数
        sc = torch.tensor([self.L, self.L, self.Hb], dtype=torch.float32, device=dev)
        off = torch.tensor([self.L / 2, self.L / 2, 0.0], dtype=torch.float32, device=dev)
        self.P = torch.rand(E, N, 3, device=dev) * sc - off
        self.acc = torch.zeros(E, 5, device=dev)  # [傘が受けた量, 雨A, 雨B, 滴下A, 滴下B] mm^3
        self.set_poses([None] * E)

    def set_poses(self, poses):
        """poses: 環境ごとの (R(3,3), H(3,)) か None(傘なし)。"""
        E, dev, nd = self.E, self.dev, self.nd
        R = torch.eye(3, device=dev).unsqueeze(0).expand(E, 3, 3).clone()
        H = torch.zeros(E, 3, device=dev)
        has = torch.zeros(E, dtype=torch.bool, device=dev)
        for e, pz in enumerate(poses):
            if pz is not None:
                R[e], H[e], has[e] = pz[0], pz[1], True
        self.R, self.H, self.has_u = R, H, has
        self.d_P, self.d_v = torch.zeros(E, nd, 3, device=dev), torch.zeros(E, nd, 3, device=dev)
        self.d_act = torch.zeros(E, nd, dtype=torch.bool, device=dev)
        self.carry = torch.zeros(E, device=dev)
        w = torch.matmul(self.ring[None], R.permute(0, 2, 1)) + H[:, None, :]  # (E,M,3) 縁のワールド座標
        z = w[..., 2]
        # 縁の点は傘の表面そのもの(=殻の内側)なので、そのまま発生させると即「傘に当たった」と判定されて消える。
        # 外側へ1.5cm・下へ3cmずらして殻の外から落とす。
        out = w[..., :2] - H[:, None, :2]
        out = out / torch.linalg.norm(out, dim=-1, keepdim=True).clamp(min=1e-6)
        self.rim_pts = torch.cat([w[..., :2] + out * 0.015, w[..., 2:3] - 0.03], dim=-1)
        wgt = torch.softmax(-(z - torch.amin(z, dim=1, keepdim=True)) / 0.03, dim=1)  # 低い縁ほど滴る
        cdf = torch.cumsum(wgt, dim=1)
        self.rim_cdf = cdf / cdf[:, -1:]

    @torch.no_grad()
    def step(self, dt, count=True, rec=None):
        a, A, dev, E, N = self.a, self.A, self.dev, self.E, self.n
        v = torch.cat([self.wind.view(1, 1, 2).expand(E, N, 2), -self.vt[..., None]], dim=-1)
        disp = v * dt
        tag, pos = swept_env(A.ugrid, A.pgrid, self.R, self.H, self.has_u, self.P, disp, self.pitch, self.n_sub)
        alive = self.alive
        tag = torch.where(alive, tag, torch.full_like(tag, -1))  # 空きスロットは無視
        um, pe = tag == 0, tag == 1
        side = pos[..., a.split_axis] < 0
        vol_w = self.vol * self.weight
        vu = (vol_w * um.float()).sum(1)
        if count:
            self.acc[:, 0] += vu
            self.acc[:, 1] += (vol_w * (pe & side).float()).sum(1)
            self.acc[:, 2] += (vol_w * (pe & ~side).float()).sum(1)
        self.carry = self.carry + vu * a.drip_fraction / V_DRIP
        if rec is not None:
            ip = pe.nonzero()
            iu = um.nonzero()
            rec.add("p", ip[:, 0], pos[ip[:, 0], ip[:, 1]])
            rec.add("u", iu[:, 0], pos[iu[:, 0], iu[:, 1]])
        newP = self.P + disp
        on_ground = alive & (tag < 0) & (newP[..., 2] < 0)
        if rec is not None:  # 地面に着いた粒(波紋・飛沫の見た目用)。環境ごとに最大150個: [x, y, 粒径mm]
            keep = on_ground & (torch.cumsum(on_ground.long(), dim=1) <= 150)
            ig = keep.nonzero()
            Dm = (6.0 * self.vol[ig[:, 0], ig[:, 1]] / math.pi) ** (1.0 / 3.0)
            rec.add("g", ig[:, 0], torch.stack([newP[ig[:, 0], ig[:, 1], 0], newP[ig[:, 0], ig[:, 1], 1], Dm], 1))
        # 当たった粒・地面に着いた粒は「消える」だけ。使い回さない。
        alive = alive & ~((tag >= 0) | on_ground)
        # 上空(箱の上面)から一定レートで新しい粒を注入する。粒径は流束重み(速い粒ほど多く通過する)。
        # 注入数は下流で何が起きるかに依存しない。数は端数を繰り越して決める(同期なし)。
        self.in_carry = self.in_carry + self.rate * dt
        kin = torch.floor(self.in_carry)
        self.in_carry = self.in_carry - kin
        free = ~alive
        inject = free & ((torch.cumsum(free.long(), dim=1) - 1) < kin[:, None].long())
        nvt, nvol = self._draw((E, N), flux=True)
        xy = torch.rand(E, N, 2, device=dev) * self.L - self.L / 2
        z = self.Hb - torch.rand(E, N, device=dev) * dt * nvt  # 1ステップ分の厚みの層に一様に
        newP = torch.where(inject[..., None], torch.cat([xy, z[..., None]], -1), newP)
        self.vt, self.vol = torch.where(inject, nvt, self.vt), torch.where(inject, nvol, self.vol)
        self.alive = alive | inject
        self.P = torch.cat([torch.remainder(newP[..., :2] + self.L / 2, self.L) - self.L / 2, newP[..., 2:]], -1)

        # ---- 傘の縁からの滴下(スロット固定のプール。発生数は carry から cumsum で決める) ----
        k = torch.floor(self.carry)
        self.carry = self.carry - k
        free = ~self.d_act
        rank = torch.cumsum(free.long(), dim=1) - 1
        spawn = free & (rank < k[:, None].long()) & self.has_u[:, None]
        M = self.rim_pts.shape[1]
        sel = torch.searchsorted(self.rim_cdf, torch.rand(E, self.nd, device=dev)).clamp(0, M - 1)
        spos = self.rim_pts[torch.arange(E, device=dev)[:, None], sel]
        self.d_P = torch.where(spawn[..., None], spos, self.d_P)
        self.d_v = torch.where(spawn[..., None], torch.zeros_like(self.d_v), self.d_v)
        act = self.d_act | spawn
        vz = (self.d_v[..., 2] - 9.81 * dt).clamp(min=-7.0)
        vxy = self.d_v[..., :2] + (self.wind - self.d_v[..., :2]) * min(1.0, dt / 0.3)
        dv = torch.cat([vxy, vz[..., None]], -1)
        dd = dv * dt
        t2, p2 = swept_env(A.ugrid, A.pgrid, self.R, self.H, self.has_u, self.d_P, dd, self.pitch, self.n_sub)
        hpd = (t2 == 1) & act
        if count:
            sd = p2[..., a.split_axis] < 0
            self.acc[:, 3] += V_DRIP * (hpd & sd).float().sum(1)
            self.acc[:, 4] += V_DRIP * (hpd & ~sd).float().sum(1)
        nP = self.d_P + dd
        kill = act & ((t2 >= 0) | (nP[..., 2] < 0))
        self.d_P = torch.where(act[..., None], nP, self.d_P)
        self.d_v = torch.where(act[..., None], dv, self.d_v)
        self.d_act = act & ~kill

    def snapshot(self, e, draw_max):
        """環境 e の描画用粒子(位置・速度・粒径mm)。すべてデバイス上のまま返す。"""
        dev = self.dev
        sel = torch.arange(0, self.n, max(1, self.n // draw_max), device=dev)
        sel = sel[self.alive[e, sel]]  # 生きている粒だけ(空きスロットは描かない)
        P, vt = self.P[e, sel], self.vt[e, sel]
        V = torch.cat([self.wind.view(1, 2).expand(len(sel), 2), -vt[:, None]], dim=1)
        D = (6.0 * self.vol[e, sel] / math.pi) ** (1.0 / 3.0)
        act = self.d_act[e]
        k = int(act.sum().item())
        if k:
            P = torch.cat([P, self.d_P[e][act]])
            V = torch.cat([V, self.d_v[e][act]])
            D = torch.cat([D, torch.full((k,), 4.0, device=dev)])
        return P, V, D


# =============================================================================
# 5. 描画
# =============================================================================
def to_np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


# =============================================================================
OOM_ERRORS = tuple(e for e in (getattr(torch.cuda, "OutOfMemoryError", None),) if e) or (MemoryError,)
MODEL_EXTS = (".obj", ".fbx", ".glb", ".gltf", ".stl", ".ply", ".dae", ".off")
UMB_WORDS = ("umbrella", "parasol", "kasa", "傘")
COUPLE_WORDS = ("couple", "pair", "people", "person", "human", "woman", "カップル", "二人", "人")
FAST_PROFILE = dict(max_drops=60000, measure_s=2.0, warmup_s=1.0, step=15, res=[854, 480], draw_max=6000)


def is_explicit(parser, dest):
    """そのオプションがコマンドラインで明示指定されたか(明示指定された値は自動調整で上書きしない)。"""
    argv = sys.argv[1:]
    return any(t == o or t.startswith(o + "=") for ac in parser._actions if ac.dest == dest for o in ac.option_strings for t in argv)


def resolve_models(a):
    """--couple / --umbrella が無ければ、実行フォルダ・./assets・./models・このスクリプトのフォルダから探す。"""
    if a.couple and a.umbrella:
        return
    here = Path(__file__).resolve().parent
    dirs = []
    for base in (Path.cwd(), here):
        for sub in ("", "assets", "models", "model", "data"):
            d = base / sub if sub else base
            if d.is_dir() and d.resolve() not in [x.resolve() for x in dirs]:
                dirs.append(d)
    files = [f for d in dirs for f in sorted(d.iterdir()) if f.is_file() and f.suffix.lower() in MODEL_EXTS]
    has = lambda f, words: any(w in f.stem.lower() for w in words)
    umb = a.umbrella or next((str(f) for f in files if has(f, UMB_WORDS)), None)
    cpl = a.couple or next((str(f) for f in files if str(f) != umb and has(f, COUPLE_WORDS)), None)
    rest = [f for f in files if str(f) not in (umb, cpl)]
    if cpl is None and umb is not None and len(rest) == 1:
        cpl = str(rest[0])  # 傘以外のモデルがちょうど1つなら、それをカップルとみなす
    if umb is None and cpl is not None and len(rest) == 1:
        umb = str(rest[0])
    if not (cpl and umb):
        print("⚠ 3Dモデルを自動検出できませんでした。")
        print("  探した場所: " + ", ".join(str(d) for d in dirs))
        print("  見つかったモデル: " + (", ".join(f.name for f in files) if files else "(なし)"))
        print("  対処: assets/ フォルダに couple.obj と Umbrella.obj を置くか、--couple <パス> --umbrella <パス> を指定してください。")
        raise SystemExit(2)
    print(f"モデルを自動検出: couple = {cpl} / umbrella = {umb}")
    a.couple, a.umbrella = cpl, umb


def apply_profile(a, parser, use_cuda):
    """GPUが無い環境では、既定のままだと極端に遅いので軽い設定に自動で切り替える(明示指定した値は優先)。"""
    prof = a.quality if a.quality != "auto" else ("high" if use_cuda else "fast")
    if prof == "fast":
        for k, v in FAST_PROFILE.items():
            if not is_explicit(parser, k):
                setattr(a, k, list(v) if isinstance(v, list) else v)
    return prof


def build_parser():
    p = argparse.ArgumentParser(
        description="相合傘の最適な傾き角を探す。引数なしで実行すると、実行フォルダ・./assets・このスクリプトのフォルダから"
                    "couple / umbrella のモデルを自動検出し、φ=0°と90°の全角度を測定して out/rain/ に動画・CSV・summary.md を出力する。")
    p.add_argument("--couple", default=None, help="カップルの3Dモデルのパス。省略時は自動検出(名前に couple/pair/people 等)")
    p.add_argument("--umbrella", default=None, help="傘の3Dモデルのパス。省略時は自動検出(名前に umbrella/傘 等)")
    p.add_argument("--quality", default="auto", choices=["auto", "high", "fast"], help="auto=GPUがあればhigh、無ければfast(粒子数・測定時間・刻み・解像度を軽くする。個別に指定した値が優先)")
    p.add_argument("--dry-run", action="store_true", help="実行計画(検出したモデル・デバイス・設定)だけ表示して終了")
    p.add_argument("--couple-axis", default="auto", choices=["auto"] + list(AXES), help="モデルの上方向(元データ上の軸)")
    p.add_argument("--umbrella-axis", default="auto", choices=["auto"] + list(AXES), help="傘の軸(石突→持ち手の逆=上)")
    p.add_argument("--keep-all", action="store_true", help="ゴミ/地面パーツの自動除去を無効化")
    p.add_argument("--umbrella-axis-vec", type=float, nargs=3, default=None,
                   help="傘の軸を元データ座標のベクトルで指定(傘の頂点側へ向ける)。例: --umbrella-axis-vec 0.2 0.9 0.1")
    p.add_argument("--umbrella-drop-parts", default="", help="除外する傘パーツ番号(パーツ一覧の番号)。例: 3,7")
    p.add_argument("--inspect", action="store_true", help="モデルの前処理(軸推定)だけ行って確認図を出力して終了")
    p.add_argument("--couple-height", type=float, default=None, help="省略時は単位(m/cm/mm)を自動推定。指定すると最大寸法をこの身長[m]に正規化")
    p.add_argument("--umbrella-diameter", type=float, default=None, help="省略時は単位を自動推定。指定すると傘の直径[m]に正規化")
    p.add_argument("--couple-colors", default="auto", choices=["auto", "natural", "original"], help="人物の色。auto=3ds Maxのワイヤー表示色など実際の色でない場合に自然な配色(肌・髪・服・靴・目)にする / natural=常に自然な配色 / original=元の色のまま")
    p.add_argument("--umbrella-color", type=int, nargs=3, default=None, help="傘の布の色 R G B(0-255)。省略時は、モデルに色情報が無ければ紺(35 60 125)")
    p.add_argument("--couple-spacing", type=float, default=None, help="二人の中心間距離[m]に調整(省略=モデルのまま)")
    p.add_argument("--grip-offset", type=float, default=0.25)
    p.add_argument("--hold", type=float, nargs=3, default=None)
    p.add_argument("--hold-z-ratio", type=float, default=0.85, help="握り初期高さ=身長×この値")
    p.add_argument("--max-hold-ratio", type=float, default=1.25, help="握り高さの上限=身長×この値(腕の届く範囲)")
    p.add_argument("--clearance", type=float, default=0.03, help="傘と人の最小クリアランス[m]")
    p.add_argument("--split-axis", type=int, default=0)
    # 雨
    p.add_argument("--rain-mm-h", type=float, default=15.0, help="降水強度 [mm/h](水量の流束がこの値に一致するよう粒子数濃度を決める)")
    p.add_argument("--max-drops", type=int, default=300000)
    p.add_argument("--wind-x", type=float, default=0.0)
    p.add_argument("--wind-y", type=float, default=0.0)
    p.add_argument("--drip-fraction", type=float, default=0.6, help="傘に当たった水のうち縁から滴下する割合")
    p.add_argument("--box", type=float, default=4.0)
    p.add_argument("--box-height", type=float, default=3.5)
    p.add_argument("--voxel", type=float, default=0.01)
    p.add_argument("--dt", type=float, default=1 / 240)
    # スイープ
    p.add_argument("--phi", type=float, nargs="+", default=[0.0, 90.0], help="傾ける方位[度]。複数指定可(0=二人を結ぶ軸方向、90=それに直交)。既定は 0 と 90 の両方")
    p.add_argument("--theta-min", type=int, default=-45)
    p.add_argument("--theta-max", type=int, default=45)
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--warmup-s", type=float, default=1.5)
    p.add_argument("--measure-s", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    # 動画
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--crf", type=int, default=20, help="動画の画質(小さいほど高画質・大容量。18〜24が目安)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="auto=CUDAがあればGPU(既定)")
    p.add_argument("--encoder", default="auto", choices=["auto", "libx264", "nvenc"], help="auto=GPUならNVENC、失敗または非GPUならlibx264")
    p.add_argument("--env-batch", type=int, default=-1, help="同時に計算する角度数。-1=VRAMの空きから自動(既定) / 0=全角度 / 正の数=指定")
    p.add_argument("--res", type=int, nargs=2, default=[1280, 720])
    p.add_argument("--cam-pos", type=float, nargs=3, default=[2.0, 4.3, 1.5], help="人は+y向きに歩いている想定(正面斜め前)。向きが違えば変更")
    p.add_argument("--cam-lookat", type=float, nargs=3, default=[0.0, 0.0, 1.15])
    p.add_argument("--fov", type=float, default=40)
    p.add_argument("--draw-max", type=int, default=12000, help="領域内で描く雨粒の最大数(領域外の遠景は密度を揃えて自動追加)")
    p.add_argument("--streak-s", type=float, default=1 / 60, help="シャッター時間[s]。雨の筋の長さを決める")
    p.add_argument("--renderer", default="genesis", choices=["genesis", "soft"], help="genesis=Genesisで描画(失敗したら自動でsoftに切替) / soft=CPUのみの簡易描画")
    p.add_argument("--ssaa", type=int, default=0, help="スーパーサンプリング倍率。0=自動(GPUなら2、CPUなら1)")
    p.add_argument("--depth-mode", default="auto", choices=["auto", "zdepth", "distance"], help="Genesisのdepthの意味。auto=実際に1枚描いて自動判定")
    p.add_argument("--dof", type=float, default=0.05, help="被写界深度の強さ(大きいほど手前/奥の雨が大きくボケる)")
    p.add_argument("--fog", type=float, default=0.05, help="霧の濃さ")
    p.add_argument("--gloss", type=float, default=2.2, help="路面の濡れ(反射)の強さ")
    p.add_argument("--ripples-per-s", type=int, default=220)
    p.add_argument("--grain", type=float, default=2.5)
    p.add_argument("--bloom", type=float, default=0.30)
    p.add_argument("--vignette", type=float, default=0.30)
    p.add_argument("--no-depth", action="store_true")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--preview", action="store_true")
    p.add_argument("--verify", action="store_true", help="すり抜け検証だけ実行して終了")
    p.add_argument("--out", default="out/rain")
    return p


def theta_list(a):
    """測定する傾き量[度]。刻みの倍数だけを使うので、範囲に含まれていれば 0° は必ず入る。"""
    return sorted(t for t in range(a.theta_min, a.theta_max + 1) if t % a.step == 0)


def zup_kw(path):
    """glb を Z-up として読むための Genesis の Mesh キーワード。バージョンで名前が違う
    (旧 parse_glb_with_zup → 新 file_meshes_are_zup)ので、受け付けられる方を返す。"""
    for k in ("file_meshes_are_zup", "parse_glb_with_zup"):
        try:
            gs.morphs.Mesh(file=path, **{k: True})
            return {k: True}
        except Exception:
            continue
    return {}


def print_plan(a, use_cuda, prof, phis):
    n_theta = len(theta_list(a))
    print("=" * 64)
    print("実行計画")
    print(f"  モデル   : couple = {a.couple}")
    print(f"             umbrella = {a.umbrella}")
    print(f"  デバイス : {'GPU (CUDA)' if use_cuda else 'CPU'} / 品質プロファイル: {prof}")
    print(f"  測定     : φ = {[f'{p:.0f}' for p in phis]} × θ = {theta_list(a)[0]}〜{theta_list(a)[-1]}° を {a.step}° 刻み({n_theta} 角度)、"
          f"1角度あたり測定 {a.measure_s}s(助走 {a.warmup_s}s)")
    print(f"  雨       : {a.rain_mm_h} mm/h、粒子 最大 {a.max_drops}、風 ({a.wind_x}, {a.wind_y}) m/s、縁からの滴下の割合 {a.drip_fraction}")
    print(f"  映像     : {'なし' if a.no_video else f'{a.res[0]}x{a.res[1]} {a.fps}fps'} / 出力先: {a.out}")
    print("=" * 64)


def write_summary(a, rows, phis):
    """全方位・全角度を通した結論を summary.md に書き、結論部分をコンソールにも表示する。"""
    ok = [r for r in rows if r["feasible"]]
    L = ["# 相合傘 最適角度の結果", "",
         f"- 降水強度 {a.rain_mm_h} mm/h / 風 ({a.wind_x}, {a.wind_y}) m/s / 縁からの滴下の割合 {a.drip_fraction} / 1角度あたり測定 {a.measure_s} s",
         "- **濡れ率** = 傘なしを1とした水量比(小さいほど濡れない)。± は標準誤差(測定を5ブロックに分けて算出)",
         "- 傾き: 傘の頂点が方位 φ の向き(φ=0 は二人を結ぶ軸の +x 側=人B側、φ=90 は +y 側)へ θ 度倒れる(θ<0 は逆向き)", ""]
    concl = []
    if not ok:
        concl = ["実現可能な角度がありませんでした(すべて INFEASIBLE)。--hold-z-ratio / --max-hold-ratio / --couple-spacing を調整してください。"]
    else:
        bt = min(ok, key=lambda r: r["wet_total"])
        bd = min(ok, key=lambda r: r["wet_direct"])
        bm = min(ok, key=lambda r: r["wet_max"])
        se2 = lambda r, b, k: 2 * math.hypot(r[k], b[k])
        tie_t = [(int(r["phi"]), r["theta"]) for r in ok if r["wet_total"] <= bt["wet_total"] + se2(r, bt, "wet_total_se")]
        tie_d = [(int(r["phi"]), r["theta"]) for r in ok if r["wet_direct"] <= bd["wet_direct"] + se2(r, bd, "wet_direct_se")]
        concl = [
            f"- **二人の合計が最も濡れない**: φ={bt['phi']:.0f}°, θ={bt['theta']:+d}°"
            f"(合計 {bt['wet_total']:.3f}±{bt['wet_total_se']:.3f} / 人A {bt['wet_A']:.3f} / 人B {bt['wet_B']:.3f} / 傘の持ち上げ {bt['lift_m']:.2f} m)",
            f"- **濡れの多い方が最小(二人の公平さ)**: φ={bm['phi']:.0f}°, θ={bm['theta']:+d}°"
            f"(max {bm['wet_max']:.3f} / 人A {bm['wet_A']:.3f} / 人B {bm['wet_B']:.3f})",
            f"- **直接の雨だけで見た最小**(縁からの滴下の仮定を除く): φ={bd['phi']:.0f}°, θ={bd['theta']:+d}°(合計 {bd['wet_direct']:.3f}±{bd['wet_direct_se']:.3f})",
        ]
        if len(tie_t) > 1:
            concl.append(f"- 注意: 合計の濡れ率は {tie_t} が誤差(2σ)の範囲で区別できません。--measure-s を長く / --max-drops を増やすと絞れます。")
        if (bt["phi"], bt["theta"]) != (bd["phi"], bd["theta"]):
            concl.append("- 注意: 滴下を含めた場合と直接の雨だけの場合で最小の角度が異なります。滴下の仮定(--drip-fraction)に結論が左右されます。")
    L += ["## 結論", ""] + concl + [""]
    for phi in phis:
        L += [f"## φ={phi:.0f}° の全角度", "", "| θ | 持上げ[m] | 人A | 人B | 合計 | 直接の雨のみ | max(A,B) |", "|---|---|---|---|---|---|---|"]
        for r in [r for r in rows if r["phi"] == phi]:
            if not r["feasible"]:
                L.append(f"| {r['theta']:+d}° | - | INFEASIBLE | | | | |")
            else:
                L.append(f"| {r['theta']:+d}° | {r['lift_m']:.2f} | {r['wet_A']:.3f} | {r['wet_B']:.3f} | "
                         f"{r['wet_total']:.3f}±{r['wet_total_se']:.3f} | {r['wet_direct']:.3f} | {r['wet_max']:.3f} |")
        L.append("")
    path = os.path.join(a.out, "summary.md")
    open(path, "w", encoding="utf-8").write("\n".join(L))
    print("\n" + "=" * 64)
    print("結論(全方位を通した結果)")
    for c in concl:
        print(c)
    print(f"詳細: {path}")
    print("=" * 64)


def main():
    parser = build_parser()
    a = parser.parse_args()
    phis = list(a.phi)
    a.phi = phis[0]  # 以降の関数は φ を1つの数値として扱う(測定ループで都度設定する)
    resolve_models(a)
    os.makedirs(a.out, exist_ok=True)
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda が指定されましたが、CUDA が利用できません。")
    use_cuda = torch.cuda.is_available() if a.device == "auto" else a.device == "cuda"
    prof = apply_profile(a, parser, use_cuda)
    print_plan(a, use_cuda, prof, phis)
    if a.dry_run:
        return
    soft = a.renderer == "soft"
    dev = None
    if not soft:
        try:
            gs.init(backend=gs.gpu if use_cuda else gs.cpu, logging_level="error", seed=a.seed)  # 描画専用モデルの「非水密」警告を出さない
            dev = gs.device
        except Exception as e:
            print(f"⚠ Genesis を初期化できないため、CPUソフトレンダラに切り替えます({type(e).__name__}: {e})")
            soft = True
    if dev is None:
        dev = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        p_ = torch.cuda.get_device_properties(0)
        print(f"デバイス: GPU ({p_.name}, VRAM {p_.total_memory / 2**30:.1f} GB) — 物理・映像合成はGPU上で実行")
    else:
        print("⚠ GPU(CUDA)が見つかりません。CPUで実行します(軽い設定に自動調整済み。それでも時間がかかります)。")
    A = build_assets(a, dev)

    if a.inspect:
        print(f"確認図: {os.path.join(a.out, '_cache', 'inspect_umbrella.png')}")
        return

    if a.verify:
        verify(A, a)
        return

    # 自動のすり抜け検証(見逃しがあればボクセルを半分にして作り直す)
    for _ in range(3):
        miss, tot = verify(A, a, n=4000, quiet=True)
        if miss == 0:
            print(f"すり抜け検証: 表面を横切る {tot} 線分をすべて検出(見逃し 0) ✔")
            break
        if a.voxel <= 0.0025:
            print(f"⚠ すり抜け検証: 見逃し {miss}/{tot}。ボクセルは十分細かいので続行します(--verify で詳細を確認してください)")
            break
        a.voxel /= 2
        print(f"⚠ すり抜け検証: 見逃し {miss}/{tot} → ボクセルを {a.voxel * 100:.2f}cm に細かくして作り直します")
        A = build_assets(a, dev)

    # カメラ: 明示指定が無ければ、モデルの大きさに合わせて正面斜め前から(人は+y向きに歩いている想定)
    sc_ = A.couple_h / 1.8
    if not is_explicit(parser, "cam_pos"):
        a.cam_pos = [2.0 * sc_, 4.3 * sc_, 1.5 * sc_]
    if not is_explicit(parser, "cam_lookat"):
        a.cam_lookat = [0.0, 0.0, 1.15 * sc_]

    ss = a.ssaa if a.ssaa > 0 else (2 if use_cuda else 1)
    enc_pref = a.encoder if a.encoder != "auto" else ("nvenc" if use_cuda else "libx264")
    W0, H0 = a.res
    W, Hh = W0 * ss, H0 * ss
    state = dict(R=None, H=None)
    umb = cam = scene = None
    cam_geo = Camera(a.cam_pos, a.cam_lookat, a.fov, W, Hh)
    gcam = GpuCamera(cam_geo, dev)  # カメラ幾何はデバイスに1つだけ置き、全角度のレンダラで共有する
    depth_cfg = dict(mode="distance" if a.depth_mode == "distance" else "zdepth", use=not a.no_depth)

    def ground_frac(d):  # depth から再構成した点のうち、地面(z≈0)にある割合
        fin = np.isfinite(d) & (d > 0.05) & (d < 200)
        wz = cam_geo.pos[2] + np.where(fin, d, 0.0) * cam_geo.ray[..., 2]
        return float((fin & (np.abs(wz) < 0.03)).mean())

    if not soft:
        try:
            zup = zup_kw(os.path.abspath(A.couple_glb))
            vis = None
            for kw in (dict(background_color=(0.55, 0.60, 0.66), ambient_light=(0.55, 0.55, 0.60), shadow=True,
                             lights=[dict(type="directional", dir=(-0.35, -0.45, -0.82), color=(1.0, 1.0, 1.02), intensity=3.0)]),
                       dict(background_color=(0.55, 0.60, 0.66), ambient_light=(0.5, 0.5, 0.55)), dict()):
                try:  # Genesisのバージョンで受け付ける VisOptions が違うため、通る組合せまで落とす
                    vis = gs.options.VisOptions(**kw)
                    break
                except Exception:
                    continue
            scene = gs.Scene(sim_options=gs.options.SimOptions(gravity=(0, 0, 0)), vis_options=vis, show_viewer=False)
            try:
                scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.20, 0.20, 0.22))))
            except Exception:
                scene.add_entity(gs.morphs.Plane())
            scene.add_entity(gs.morphs.Mesh(file=os.path.abspath(A.couple_glb), fixed=True, collision=False, **zup))
            umb = scene.add_entity(gs.morphs.Mesh(file=os.path.abspath(A.umb_glb), pos=tuple(float(x) for x in A.hold),
                                                  fixed=False, collision=False, **zup))
            cam = scene.add_camera(res=(W, Hh), pos=tuple(a.cam_pos), lookat=tuple(a.cam_lookat), fov=a.fov, GUI=False)
            scene.build()
            # 描画の自己診断: 実際に1枚描き、depth の定義(前方距離か直線距離か)を自動判定する
            out = cam.render(rgb=True, depth=not a.no_depth)
            to_np(out[0])
            if not a.no_depth:
                d_ = np.squeeze(to_np(out[1])).astype(np.float32)
                g1, g2 = ground_frac(d_), ground_frac(d_ / cam_geo.raylen)
                if a.depth_mode == "auto":
                    depth_cfg["mode"] = "zdepth" if g1 >= g2 else "distance"
                if max(g1, g2) < 0.01:
                    depth_cfg["use"] = False
                    print("⚠ depth から地面を検出できないため、depth を使わずに描画します(空・路面・霧の処理を省略)。")
                else:
                    print(f"Genesis の描画を確認: depth の定義 = {depth_cfg['mode']}(地面検出 {100 * max(g1, g2):.1f}%)")
        except Exception as e:
            print(f"⚠ Genesis での描画に失敗したため、CPUソフトレンダラに切り替えます({type(e).__name__}: {e})")
            soft = True
            umb = cam = scene = None

    def make_renderer():
        rr = RainRenderer(gcam, a, ss=ss, seed=a.seed)
        rr.box_half = a.box / 2
        return rr

    def set_umbrella(R, H, quat):
        state["R"], state["H"] = R.cpu().numpy().astype(np.float64), H.cpu().numpy().astype(np.float64)
        if soft:
            return
        umb.set_pos(H.cpu().numpy().astype(np.float32), zero_velocity=True)
        umb.set_quat(quat.astype(np.float32), zero_velocity=True)
        scene.step()

    def render():
        if soft:
            um_w = A.um.vertices @ state["R"].T + state["H"]
            return soft_render(cam_geo, [(A.cm.vertices, A.cm.faces, A.cm.metadata.get("face_colors")),
                                         (um_w, A.um.faces, A.um.metadata.get("face_colors"))], seed=a.seed)
        out = cam.render(rgb=True, depth=depth_cfg["use"])
        rgb = np.ascontiguousarray(to_np(out[0])).astype(np.uint8)
        depth = None
        if depth_cfg["use"]:
            depth = np.squeeze(to_np(out[1])).astype(np.float32)
            if depth_cfg["mode"] == "distance":  # 視点からの直線距離 → 前方距離へ
                depth = depth / cam_geo.raylen
        return rgb, depth

    def put_text(img, lines):
        img = np.ascontiguousarray(img)  # OpenCV(5系)は非連続配列に描けない
        sc = img.shape[1] / 1280.0
        for i, t in enumerate(lines):
            cv2.putText(img, t, (int(20 * sc), int((40 + 34 * i) * sc)), cv2.FONT_HERSHEY_SIMPLEX, 0.85 * sc,
                        (255, 255, 255), max(1, int(2 * sc)), cv2.LINE_AA)
        return img

    class Writer:
        """動画書き出し。nvenc が使えなければ最初のフレームで自動的に libx264 へ切り替える。"""

        def __init__(self, path):
            self.path, self.first, self.w, self.enc = path, [], None, enc_pref
            self._open()

        def _open(self):
            if self.enc == "nvenc":
                self.w = imageio.get_writer(self.path, fps=a.fps, codec="h264_nvenc", macro_block_size=2, output_params=["-cq", str(a.crf), "-preset", "p5"])
            else:
                self.w = imageio.get_writer(self.path, fps=a.fps, codec="libx264", macro_block_size=2, output_params=["-crf", str(a.crf), "-preset", "medium"])

        def append(self, img):
            try:
                self.w.append_data(img)
                if len(self.first) < 2:
                    self.first.append(img)
            except Exception:
                if self.enc != "nvenc":
                    raise
                print("  (nvenc が使えないため libx264 に切り替えます)")
                self.enc = "libx264"
                self._open()
                for f in self.first + [img]:
                    self.w.append_data(f)

        def close(self):
            self.w.close()

    sub = max(1, round(1.0 / (a.fps * a.dt)))
    sync = (lambda: torch.cuda.synchronize()) if torch.cuda.is_available() else (lambda: None)

    def make_frame(sim_, rr, e, rec):
        P, V, D = sim_.snapshot(e, a.draw_max)
        g = lambda k: rec.get(e, k, dev)
        hp, hu_, hg = g("p"), g("u"), g("g")
        return rr.frame(P, V, D, hp, hu_, hg, torch.cat([hp, hu_]), 1.0 / a.fps)

    def simulate(poses, count_s, videos=None, headers=None):
        """poses: [(R,H,quat) or None] を E 個まとめて同時にシミュレーション。videos があれば映像も作る。
        戻り値: (水量 (E,5) mL/s [傘, 雨A, 雨B, 滴下A, 滴下B], ブロックごとの水量 (E,nb,5) mL/s)。
        測定時間を nb 個のブロックに分けて記録しておき、ブロック間のばらつきから標準誤差を出す。"""
        E = len(poses)
        sim = RainSim(a, A, E)
        sim.set_poses([None if pz is None else (pz[0], pz[1]) for pz in poses])
        t_sim = t_fx = t_enc = 0.0
        t0 = time.perf_counter()
        for _ in range(int(a.warmup_s / a.dt)):
            sim.step(a.dt, count=False)
        sync()
        t_sim += time.perf_counter() - t0
        sim.acc.zero_()
        rrs, writers = [], []
        if videos:
            for e in range(E):  # 角度ごとの静的な絵(傘の姿勢が違う)を作る
                set_umbrella(*poses[e][:3])  # (R, H, quat)
                rgb, depth = render()
                rr = make_renderer()
                rr.prepare(rgb, depth)
                rrs.append(rr)
                writers.append(Writer(videos[e]))
        n_frames = int(count_s * a.fps)
        nb = max(1, min(5, n_frames))
        edges = [round(n_frames * (i + 1) / nb) for i in range(nb)]
        marks = [sim.acc.clone()]
        for fi in range(n_frames):
            rec = FrameRec() if videos else None
            t0 = time.perf_counter()
            for _ in range(sub):
                sim.step(a.dt, rec=rec)
            sync()
            t_sim += time.perf_counter() - t0
            if videos:
                acc = (sim.acc / 1000.0).cpu().numpy()
                for e in range(E):
                    t0 = time.perf_counter()
                    img = make_frame(sim, rrs[e], e, rec)
                    t1 = time.perf_counter()
                    writers[e].append(put_text(img, headers[e] + [f"water on A:{acc[e, 1] + acc[e, 3]:.2f} mL  B:{acc[e, 2] + acc[e, 4]:.2f} mL"]))
                    t_fx += t1 - t0
                    t_enc += time.perf_counter() - t1
            if fi + 1 in edges:
                marks.append(sim.acc.clone())
        for w in writers:
            w.close()
        res = (sim.acc / (n_frames * sub * a.dt) / 1000.0).cpu().numpy()
        mk = torch.stack(marks).cpu().numpy()  # (nb+1, E, 5)
        t_blk = np.diff([0] + edges) * sub * a.dt  # 各ブロックの時間[s]
        blk = (np.diff(mk, axis=0) / t_blk[:, None, None] / 1000.0).transpose(1, 0, 2)  # (E, nb, 5) mL/s
        print(f"  [{E}角度同時] シミュレーション {t_sim:.1f}s / 映像合成 {t_fx:.1f}s / エンコード等 {t_enc:.1f}s")
        return res, blk

    if a.preview:
        R, quat, H, lift = place(A, a, 0)
        if H is None:
            print("θ=0 でも干渉が解消できません。--hold-z-ratio/--max-hold-ratio/--umbrella-diameter を調整してください。")
            return
        set_umbrella(R, H, quat)
        sim = RainSim(a, A, 1)
        sim.set_poses([(R, H)])
        for _ in range(int(1.0 / a.dt)):
            sim.step(a.dt, count=False)
        rgb, depth = render()
        rr = make_renderer()
        rr.prepare(rgb, depth)
        gr = float(rr.ground.float().mean())
        if depth is not None:  # 空の判定の診断: 地平線より上を向く画素の depth(空が遠い定数で返る環境の確認用)
            up_ = cam_geo.ray[..., 2] >= 0
            du = depth[up_]
            du = du[np.isfinite(du) & (du > 0)]
            if len(du):
                print(f"地平線より上の画素の depth: 最小 {du.min():.1f} / 中央値 {np.median(du):.1f} / 最大 {du.max():.1f} m(空は遠い定数になっているはず)")
        print(f"地面として検出されたピクセル: {100 * gr:.1f}%  背景(空)として扱ったピクセル: {100 * float((~torch.isfinite(rr.depth)).float().mean()):.1f}%")
        if not a.no_depth and gr < 0.01:
            print("⚠ 地面を検出できません。depth の定義が違う可能性があります → --depth-mode distance を試すか、--no-depth で切り分けてください。")
        for _ in range(12):  # 波紋・濡れ・飛沫が溜まるまで進める
            rec = FrameRec()
            for _ in range(sub):
                sim.step(a.dt, rec=rec)
            img = make_frame(sim, rr, 0, rec)
        path = os.path.join(a.out, "preview.png")
        imageio.imwrite(path, put_text(img, [f"PREVIEW theta=0 lift={lift:.2f}m"]))
        print(f"プレビュー保存: {path}")
        return

    # --- baseline(傘なし) ---
    res_b, _ = simulate([None], a.measure_s * 2)
    b = res_b[0]
    base = b[1] + b[2]
    base_each = [b[1], b[2]]
    print(f"baseline(傘なし) 水量 mL/s: A={b[1]:.3f} B={b[2]:.3f}")

    def sweep(phi):
        """方位 phi について全角度を測定し、動画・CSV・グラフを出力して行のリストを返す。"""
        a.phi = phi
        # --- 各角度の傘の姿勢を決める(干渉解決) → 実現可能なものを env-batch 個ずつまとめて同時に計算 ---
        thetas = theta_list(a)
        feas = {}
        rows = {}
        for theta in thetas:
            R, quat, H, lift = place(A, a, theta)
            if H is None:
                print(f"θ={theta:+3d}°  INFEASIBLE(傘が人に干渉。腕の届く上限でも解消不可)")
                rows[theta] = dict(theta=theta, phi=a.phi, lift_m=float("nan"), feasible=0, wet_A=float("nan"),
                                   wet_B=float("nan"), wet_total=float("nan"), wet_max=float("nan"), drip_share=float("nan"),
                                   wet_direct=float("nan"), wet_total_se=float("nan"), wet_A_se=float("nan"),
                                   wet_B_se=float("nan"), wet_direct_se=float("nan"))
            else:
                feas[theta] = (R, H, quat, lift)
        todo = list(feas)
        if a.env_batch == 0:
            bs = len(todo)
        elif a.env_batch > 0:
            bs = a.env_batch
        elif use_cuda and todo:  # 自動: VRAMの空きに収まる最大数(物理テンソル + 映像用の角度ごとの状態)
            probe = RainSim(a, A, 1)
            per_sim = probe.n * 320 + probe.nd * 80
            per_fx = 0 if a.no_video else W * Hh * 25  # 角度ごとに常駐する画像・濡れマップ
            reserve = 0 if a.no_video else W * Hh * 250  # 合成の一時バッファ(角度で使い回す)
            free = torch.cuda.mem_get_info()[0]
            bs = int(max(1, min(len(todo), (0.7 * free - reserve) // (per_sim + per_fx))))
            del probe
            print(f"同時に計算する角度数: {bs} (VRAM空き {free / 2**30:.1f} GB から自動決定。--env-batch で変更可)")
        else:
            bs = min(len(todo), 4) if todo else 1
        pos_i = 0
        while pos_i < len(todo):
            chunk = todo[pos_i:pos_i + max(1, bs)]
            vids = None if a.no_video else [os.path.join(a.out, f"phi{int(a.phi):03d}_theta{t:+03d}.mp4") for t in chunk]
            hdr = [[f"tilt theta={t:+d} deg (phi={a.phi:.0f})  lift={feas[t][3]:.2f} m"] for t in chunk]
            try:
                acc, blk = simulate([feas[t] for t in chunk], a.measure_s, vids, hdr)
            except OOM_ERRORS:  # GPUメモリ不足: 同時に計算する角度数を半分にして、同じ角度グループをやり直す
                if bs <= 1:
                    raise
                bs = max(1, len(chunk) // 2)
                print(f"⚠ GPUメモリ不足 → 同時に計算する角度数を {bs} に減らして再試行します(--ssaa 1 や --env-batch でも調整できます)")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            pos_i += len(chunk)
            for t, c, bk in zip(chunk, acc, blk):
                wA, wB = (c[1] + c[3]) / base_each[0], (c[2] + c[4]) / base_each[1]
                tot = (c[1] + c[2] + c[3] + c[4]) / base
                nbk = len(bk)
                se = lambda x: float(np.std(x, ddof=1) / math.sqrt(nbk)) if nbk > 1 else float("nan")  # ブロック間の標準誤差
                se_tot = se((bk[:, 1] + bk[:, 2] + bk[:, 3] + bk[:, 4]) / base)
                drip = (c[3] + c[4]) / max(1e-9, c[1] + c[2] + c[3] + c[4])
                direct = (c[1] + c[2]) / base  # 縁からの滴下を含めない(直接の雨だけ)。滴下の仮定に左右されない値
                rows[t] = dict(theta=t, phi=a.phi, lift_m=feas[t][3], feasible=1, wet_A=wA, wet_B=wB, wet_total=tot,
                               wet_max=max(wA, wB), drip_share=drip, wet_direct=direct, wet_total_se=se_tot,
                               wet_A_se=se((bk[:, 1] + bk[:, 3]) / base_each[0]), wet_B_se=se((bk[:, 2] + bk[:, 4]) / base_each[1]),
                               wet_direct_se=se((bk[:, 1] + bk[:, 2]) / base))
                print(f"θ={t:+3d}°  持上げ={feas[t][3]:.2f}m  濡れ率 A={wA:.3f} B={wB:.3f} 合計={tot:.3f}±{se_tot:.3f} (直接の雨のみ {direct:.3f} / 滴下の寄与 {100 * drip:.1f}%)")
        rows = [rows[t] for t in thetas]

        csv_path = os.path.join(a.out, f"result_phi{int(a.phi):03d}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        th = [r["theta"] for r in rows]
        plt.figure(figsize=(7, 4))
        for k, lab, ls in [("wet_A", "A", "-"), ("wet_B", "B", "-"), ("wet_total", "total (with rim drips)", "-"),
                           ("wet_direct", "total (direct rain only)", "--"), ("wet_max", "max(A,B)", "-")]:
            plt.plot(th, [r[k] for r in rows], marker="o", linestyle=ls, label=lab)
        plt.errorbar(th, [r["wet_total"] for r in rows], yerr=[r["wet_total_se"] for r in rows], fmt="none", ecolor="gray", capsize=3)
        plt.xlabel("tilt angle [deg]")
        plt.ylabel("wetness ratio (1 = no umbrella)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(a.out, f"curve_phi{int(a.phi):03d}.png"), dpi=150, bbox_inches="tight")

        ok = [r for r in rows if r["feasible"]]
        if ok:
            bt, bm = min(ok, key=lambda r: r["wet_total"]), min(ok, key=lambda r: r["wet_max"])
            print(f"合計最小: θ={bt['theta']:+d}° / 不公平さ最小(maxの最小): θ={bm['theta']:+d}°  (INFEASIBLEの角度は除外)")
        if ok:  # 最小値と誤差(2σ)の範囲で区別できない角度を示す(粒子数・測定時間が少ないと最適角度は決められない)
            tie = [r["theta"] for r in ok if r["wet_total"] <= bt["wet_total"] + 2 * math.hypot(r["wet_total_se"], bt["wet_total_se"])]
            if len(tie) > 1:
                print(f"注意: 合計の濡れ率は θ={tie} が誤差(2σ)の範囲で区別できません。--measure-s を長く、--max-drops を増やすと絞れます。")
        print(f"結果: {csv_path}")
        return rows

    all_rows = []
    try:
        for phi in phis:
            print(f"\n===== 傾ける方位 φ={phi:.0f}° の測定 =====")
            all_rows += sweep(phi)
    except KeyboardInterrupt:  # Ctrl+C: ここまでに完了した方位の結果で summary を作る
        print("\n中断されました。ここまでに完了した方位の結果で summary.md を作ります。")
    if all_rows:
        write_summary(a, all_rows, [p for p in phis if any(r["phi"] == p for r in all_rows)])
    else:
        print("完了した測定がありません。")

if __name__ == "__main__":
    main()