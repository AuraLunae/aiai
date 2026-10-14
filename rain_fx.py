"""
rain_fx.py — 雨映像のリアル合成(PyTorch。GPU があれば GPU、無ければ CPU で動く)

入力: ベース画像(rgb uint8) + depth(カメラ前方への距離[m]。背景は 0/inf/NaN/巨大値でよい)
処理(すべてテンソル演算。画像・粒子・波紋は最後までGPU上に置き、CPUへ戻すのは完成フレームだけ):
  静的(角度ごとに1回): 空(曇天) / 濡れた路面(フレネル反射+ゆらぎ) / 距離霧
  動的(毎フレーム):
    - 濡れ: 人に当たった位置に濡れ染み(暗化+鏡面ハイライト)、傘に水玉のきらめき
    - 地面: 着水の波紋と飛沫
    - 雨: 粒径別の輝度、シャッター時間による筋の長さ、被写界深度(錯乱円で層分けしてぼかす)
    - 仕上げ: ブルーム、フィルム調コントラスト、寒色、ビネット、粒状ノイズ、SSAA縮小
線の描画は「線分上に約1pxごとにサンプルを置いて、ビリニア加重で index_put_(accumulate)」する方式で、
cv2 を使わずに全粒子を1回のカーネルで描ける。
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

SIGMAS = [0.0, 0.8, 2.0, 4.5, 9.0]  # 錯乱円の大きさ別のぼかし量[px]
EDGES = [1.5, 4.0, 9.0, 18.0]  # 錯乱円径[px]の区切り


# =============================================================================
# カメラ幾何(numpy。ソフトレンダラとも共有)
# =============================================================================
class Camera:
    def __init__(self, pos, lookat, fov, W, H):
        self.pos = np.array(pos, float)
        f = np.array(lookat, float) - self.pos
        self.zf = float(np.linalg.norm(f))  # ピント距離
        self.f = f / self.zf
        r = np.cross(self.f, [0, 0, 1.0])
        self.r = r / np.linalg.norm(r)
        self.u = np.cross(self.r, self.f)
        self.fpix = (H / 2) / math.tan(math.radians(fov) / 2)
        self.W, self.H = W, H
        xs = ((np.arange(W) + 0.5) - W / 2) / self.fpix
        ys = -((np.arange(H) + 0.5) - H / 2) / self.fpix
        self.ray = (self.f[None, None, :] + xs[None, :, None] * self.r[None, None, :]
                    + ys[:, None, None] * self.u[None, None, :]).astype(np.float32)
        self.raylen = np.linalg.norm(self.ray, axis=2)

    def proj(self, P):
        d = P - self.pos
        z = d @ self.f
        z = np.where(z < 0.1, np.nan, z)
        return self.W / 2 + (d @ self.r) / z * self.fpix, self.H / 2 - (d @ self.u) / z * self.fpix, z

    def sky(self):
        elev = np.clip(self.ray[..., 2] / self.raylen, 0, 1)
        t = np.clip(elev * 2.2, 0, 1)[..., None]
        hor, zen = np.array([178, 188, 198], np.float32), np.array([96, 106, 120], np.float32)
        return hor * (1 - t) + zen * t


class GpuCamera:
    """カメラ幾何をデバイスへ置いたもの。全環境(角度)のレンダラで共有する。"""

    def __init__(self, cam: Camera, device):
        self.cam, self.device = cam, device
        self.W, self.H = cam.W, cam.H
        t = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=device)
        self.pos, self.f, self.r, self.u = t(cam.pos), t(cam.f), t(cam.r), t(cam.u)
        self.fpix, self.zf = float(cam.fpix), float(cam.zf)
        self.ray, self.raylen = t(cam.ray), t(cam.raylen)
        self.sky = t(cam.sky())
        yy, xx = torch.meshgrid(torch.arange(cam.H, device=device).float(), torch.arange(cam.W, device=device).float(), indexing="ij")
        self.rad2 = ((xx - cam.W / 2) / (cam.W / 2)) ** 2 + ((yy - cam.H / 2) / (cam.H / 2)) ** 2
        sp = (torch.rand(cam.H, cam.W, device=device) > 0.997).float()
        sp = blur2(sp, 0.9)
        self.sparkle = sp / sp.max().clamp(min=1e-6)

    def proj(self, P):
        d = P - self.pos
        z = d @ self.f
        z = torch.where(z < 0.1, torch.full_like(z, float("nan")), z)
        return self.W / 2 + (d @ self.r) / z * self.fpix, self.H / 2 - (d @ self.u) / z * self.fpix, z


# =============================================================================
# テンソル演算ユーティリティ
# =============================================================================
def gblur(x, sigma):
    """ガウスぼかし(分離畳み込み)。x: (B,1,H,W)"""
    if sigma <= 0:
        return x
    k = int(math.ceil(3 * sigma))
    xs = torch.arange(-k, k + 1, device=x.device, dtype=x.dtype)
    w = torch.exp(-0.5 * (xs / sigma) ** 2)
    w = w / w.sum()
    x = F.conv2d(F.pad(x, (k, k, 0, 0), mode="reflect"), w.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, k, k), mode="reflect"), w.view(1, 1, -1, 1))


def blur2(m, s):  # (H,W)
    return gblur(m[None, None], s)[0, 0]


def blur3(img, s):  # (H,W,3)
    return gblur(img.permute(2, 0, 1)[:, None], s)[:, 0].permute(1, 2, 0)


def splat_segments(canvas, H, W, u0, v0, u1, v1, inten, offset=None, nmax=400):
    """線分 (u0,v0)-(u1,v1) を約1pxごとのサンプルで canvas(平坦 H*W*層数) へビリニア加算する。"""
    n = (torch.ceil(torch.hypot(u1 - u0, v1 - v0)) + 1).clamp(1, nmax).long()
    tot = int(n.sum().item())
    if tot == 0:
        return
    seg = torch.repeat_interleave(torch.arange(len(n), device=n.device), n)
    start = torch.cumsum(n, 0) - n
    k = torch.arange(tot, device=n.device) - start[seg]
    t = k.float() / (n[seg] - 1).clamp(min=1).float()
    x = u0[seg] + (u1 - u0)[seg] * t
    y = v0[seg] + (v1 - v0)[seg] * t
    w = inten[seg]
    base = offset[seg] if offset is not None else 0
    x0, y0 = torch.floor(x), torch.floor(y)
    fx, fy = x - x0, y - y0
    for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = (x0 + dx).long(), (y0 + dy).long()
        ok = ((xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)).float()  # 範囲外は重み0(ブール添字を避けて同期を減らす)
        idx = yi.clamp(0, H - 1) * W + xi.clamp(0, W - 1) + base
        canvas.index_put_((idx,), (w * wt * ok).float(), accumulate=True)


def splat_points(canvas, H, W, x, y, w):
    x0, y0 = torch.floor(x), torch.floor(y)
    fx, fy = x - x0, y - y0
    for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = (x0 + dx).long(), (y0 + dy).long()
        ok = ((xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)).float()
        canvas.index_put_((yi.clamp(0, H - 1) * W + xi.clamp(0, W - 1),), (w * wt * ok).float(), accumulate=True)


# =============================================================================
class RainRenderer:
    """1つの環境(1つの傘角度)の映像を作る。幾何(GpuCamera)は共有、状態(濡れ・波紋・飛沫)は個別。"""

    def __init__(self, G: GpuCamera, p=None, ss=1, seed=0):
        g = lambda k, d: getattr(p, k, d) if p is not None else d
        self.G, self.ss, self.dev = G, ss, G.device
        self.dof = g("dof", 0.05)
        self.shutter = g("streak_s", 1 / 60)
        self.fog_k = g("fog", 0.05)
        self.gloss = g("gloss", 2.2)
        self.ripples_per_s = g("ripples_per_s", 220)
        self.box_half = g("box_half", 2.0)
        self.far_extent = g("far_extent", 9.0)
        self.far_decay = g("far_decay", 3.0)
        self.grain = g("grain", 2.5)
        self.bloom = g("bloom", 0.30)
        self.vig = g("vignette", 0.30)
        self.rain_gain = g("rain_gain", 0.85)
        d = self.dev
        self.rain_alpha = torch.tensor([0.72, 0.78, 0.88], device=d)
        self.edges = torch.tensor(EDGES, device=d)
        self.ang = torch.linspace(0, 2 * math.pi, 23, device=d)  # 22分割の閉曲線(始点=終点)
        self.depth = None

    # ------------------------------------------------------------------ 幾何
    def _visible(self, P):
        G = self.G
        u, v, z = G.proj(P)
        ok = torch.isfinite(u) & torch.isfinite(v) & (u >= 0) & (u < G.W - 1) & (v >= 0) & (v < G.H - 1)
        iu = torch.nan_to_num(u).long().clamp(0, G.W - 1)
        iv = torch.nan_to_num(v).long().clamp(0, G.H - 1)
        ok = ok & (z < self.depth[iv, iu] + 0.03)  # 背景(depth=inf)の前にある粒は描く
        return u, v, z, ok

    # ------------------------------------------------------------------ 静的パス
    def prepare(self, rgb, depth):
        """角度ごとに1回: 静的な見た目を作り、動的状態をリセットする。rgb: uint8 (H,W,3) / depth: (H,W) or None"""
        G, dev = self.G, self.dev
        H, W = G.H, G.W
        img = torch.as_tensor(np.ascontiguousarray(rgb), device=dev).float()
        if depth is None:  # depth が無い場合は空/地面/霧の処理を行わず、雨と濡れだけ重ねる
            self.depth = torch.full((H, W), float("inf"), device=dev)
            self.ground = torch.zeros((H, W), dtype=torch.bool, device=dev)
            self.base = img
            self._reset_dynamic()
            return
        d = torch.as_tensor(np.ascontiguousarray(depth), device=dev).float()
        fin = torch.isfinite(d) & (d > 0.05) & (d < 200.0)  # 背景は 0 や巨大値で返る環境もある
        d = torch.where(fin, d, torch.full_like(d, float("inf")))
        wz = G.pos[2] + torch.where(fin, d, torch.zeros_like(d)) * G.ray[..., 2]
        ground = fin & (torch.abs(wz) < 0.03)
        img = torch.where(fin[..., None], img, G.sky)
        if bool(ground.any()):
            lum = img.mean(dim=2)
            ref_l = torch.quantile(lum[ground], 0.5) + 1e-6
            shade = (blur2(lum, 20) / ref_l).clamp(0.5, 1.5)  # 影・照明の濃淡だけ残す
            asphalt = torch.tensor([52.0, 53.0, 58.0], device=dev) * shade[..., None]
            asphalt = asphalt + (blur2(torch.randn(H, W, device=dev), 1.2) * 90.0 * 0.15)[..., None]
            v0 = float(G.proj(torch.zeros(1, 3, device=dev))[1][0])  # 足元の画面行を鏡の軸にする
            ys = torch.round(2 * v0 - torch.arange(H, device=dev).float()).clamp(0, H - 1).long()
            src = img[ys]
            nx = blur2(torch.randn(H, W, device=dev), 6) * 90.0
            ny = blur2(torch.randn(H, W, device=dev), 6) * 40.0
            gx = (torch.arange(W, device=dev).float()[None, :] + nx + 0.5) / W * 2 - 1
            gy = (torch.arange(H, device=dev).float()[:, None] + ny + 0.5) / H * 2 - 1
            grid = torch.stack([gx, gy], dim=-1)[None]
            refl = F.grid_sample(src.permute(2, 0, 1)[None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0].permute(1, 2, 0)
            refl = blur3(refl, 2.5)
            cosi = torch.abs(G.ray[..., 2]) / G.raylen
            fres = 0.02 + 0.98 * (1 - cosi) ** 5
            k = (fres * self.gloss).clamp(0, 0.85)[..., None]
            img = torch.where(ground[..., None], asphalt * (1 - k) + refl * k, img)
        dist = torch.where(fin, d * G.raylen, torch.zeros_like(d))
        f = (1 - torch.exp(-self.fog_k * dist))[..., None]
        img = torch.where(fin[..., None], img * (1 - f) + torch.tensor([172.0, 182.0, 192.0], device=dev) * f, img)
        self.depth, self.ground, self.base = d, ground, img
        self._reset_dynamic()

    def _reset_dynamic(self):
        G, dev = self.G, self.dev
        self.wet_p = torch.zeros(G.H * G.W, device=dev)
        self.wet_u = torch.zeros(G.H * G.W, device=dev)
        self.rip = torch.zeros((0, 4), device=dev)  # x, y, age, strength
        self.sp_P, self.sp_v, self.sp_l = torch.zeros((0, 3), device=dev), torch.zeros((0, 3), device=dev), torch.zeros(0, device=dev)

    # ------------------------------------------------------------------ 動的パス
    def _accumulate(self, wet, hits):
        if hits is None or len(hits) == 0:
            return
        u, v, z, ok = self._visible(hits)
        idx = v[ok].long() * self.G.W + u[ok].long()
        wet.index_put_((idx,), torch.ones(len(idx), device=self.dev), accumulate=True)

    def _far_xy(self, rho):
        """領域の外側に、領域内と同じ密度から距離で指数減衰する密度で xy を配る(境目の段差を無くす)"""
        h, ell = self.box_half, self.far_decay
        R = h + 4 * ell
        n = int(rho * ((2 * R) ** 2 - (2 * h) ** 2))
        if n <= 0:
            return torch.zeros((0, 2), device=self.dev)
        xy = torch.rand(n, 2, device=self.dev) * (2 * R) - R
        d = torch.maximum(torch.abs(xy[:, 0]), torch.abs(xy[:, 1])) - h
        keep = (d > 0) & (torch.rand(n, device=self.dev) < torch.exp(-d / ell))
        return xy[keep]

    def _draw_ripples(self, hits_g, dt):
        G, dev = self.G, self.dev
        H, W = G.H, G.W
        self.rip[:, 2] += dt
        self.rip = self.rip[self.rip[:, 2] < 0.7]
        if hits_g is not None and len(hits_g):
            big = hits_g[hits_g[:, 2] > 1.4]
            n_new = min(len(big), max(1, int(self.ripples_per_s * dt)))
            if n_new > 0 and len(big):
                sel = big[torch.randperm(len(big), device=dev)[:n_new]]
                new = torch.stack([sel[:, 0], sel[:, 1], torch.rand(n_new, device=dev) * dt, (sel[:, 2] / 4.0).clamp(0.3, 1.0)], 1)
                far = self._far_xy(0.4 * n_new / (2 * self.box_half) ** 2)
                if len(far):
                    j = torch.randint(0, len(sel), (len(far),), device=dev)
                    farr = torch.stack([far[:, 0], far[:, 1], torch.rand(len(far), device=dev) * dt, (sel[j, 2] / 4.0).clamp(0.3, 1.0) * 0.8], 1)
                    new = torch.cat([new, farr])
                self.rip = torch.cat([self.rip, new])
                m = min(len(sel), 40)  # 着水の飛沫
                P0 = torch.cat([sel[:m, 0:2], torch.zeros(m, 1, device=dev)], 1)[torch.arange(m, device=dev).repeat_interleave(3)]
                n = len(P0)
                vv = torch.stack([torch.randn(n, device=dev) * 0.35, torch.randn(n, device=dev) * 0.35, torch.rand(n, device=dev) * 0.9 + 0.5], 1)
                self.sp_P, self.sp_v = torch.cat([self.sp_P, P0]), torch.cat([self.sp_v, vv])
                self.sp_l = torch.cat([self.sp_l, torch.full((n,), 0.25, device=dev)])
        R = len(self.rip)
        if R == 0:
            return torch.zeros(H * W, device=dev)
        canv = torch.zeros(H * W, device=dev)
        ca, sa = torch.cos(self.ang), torch.sin(self.ang)
        # 中心の可視判定(傘/人に隠れる波紋は描かない)
        cu, cv, cz, cok = self._visible(torch.cat([self.rip[:, 0:2], torch.zeros(R, 1, device=dev)], 1))
        for scale, base_a in ((1.0, 1.0), (0.55, 0.6)):  # 外輪と内輪
            rad = (0.02 + 0.22 * self.rip[:, 2]) * scale
            px = self.rip[:, 0:1] + rad[:, None] * ca[None]
            py = self.rip[:, 1:2] + rad[:, None] * sa[None]
            pts = torch.stack([px, py, torch.zeros_like(px)], -1).reshape(-1, 3)
            u, v, z = G.proj(pts)
            u, v = u.reshape(R, -1), v.reshape(R, -1)
            fade = (1 - self.rip[:, 2] / 0.7).clamp(0, 1) ** 1.5 * self.rip[:, 3] * base_a * cok.float()
            ok = (torch.isfinite(u) & torch.isfinite(v)).all(dim=1).float()
            u, v = torch.nan_to_num(u), torch.nan_to_num(v)
            inten = (fade * ok)[:, None].expand(R, u.shape[1] - 1).reshape(-1)
            splat_segments(canv, H, W, u[:, :-1].reshape(-1), v[:, :-1].reshape(-1), u[:, 1:].reshape(-1), v[:, 1:].reshape(-1), inten * 0.5)
        return canv

    def _draw_splashes(self, hits_all, dt):
        G, dev = self.G, self.dev
        if hits_all is not None and len(hits_all):
            h = hits_all[:150]
            h = h[torch.arange(len(h), device=dev).repeat_interleave(2)]
            n = len(h)
            vv = torch.stack([torch.randn(n, device=dev) * 0.5, torch.randn(n, device=dev) * 0.5, torch.rand(n, device=dev) * 1.2 + 0.6], 1)
            self.sp_P, self.sp_v = torch.cat([self.sp_P, h]), torch.cat([self.sp_v, vv])
            self.sp_l = torch.cat([self.sp_l, torch.full((n,), 0.22, device=dev)])
        canv = torch.zeros(G.H * G.W, device=dev)
        if len(self.sp_P) == 0:
            return canv
        self.sp_v[:, 2] -= 9.81 * dt
        self.sp_P = self.sp_P + self.sp_v * dt
        self.sp_l = self.sp_l - dt
        k = (self.sp_l > 0) & (self.sp_P[:, 2] > -0.01)
        self.sp_P, self.sp_v, self.sp_l = self.sp_P[k], self.sp_v[k], self.sp_l[k]
        if len(self.sp_P):
            u, v, z, ok = self._visible(self.sp_P)
            splat_points(canv, G.H, G.W, u[ok] - 0.5, v[ok] - 0.5, torch.ones(int(ok.sum().item()), device=dev))
        return canv

    def _draw_rain(self, img, P, V, D):
        G, dev = self.G, self.dev
        H, W = G.H, G.W
        u0, v0, z0, ok = self._visible(P)
        u1, v1, _ = G.proj(P - V * self.shutter)
        ok = ok & torch.isfinite(u1) & torch.isfinite(v1)
        sel = ok.nonzero().squeeze(1)
        if len(sel) == 0:
            return img
        u0, v0, u1, v1, z, D = u0[sel], v0[sel], u1[sel], v1[sel], z0[sel], D[sel]
        coc = self.dof * G.fpix * torch.abs(1.0 / G.zf - 1.0 / z)  # 錯乱円径[px]
        layer = torch.bucketize(coc, self.edges, right=True)  # 0..4
        inten = (D * 1e-3 * G.fpix / z).clamp(0.12, 1.0) * self.rain_gain  # 粒の見かけ幅[px]が1px未満なら暗くする
        canvas = torch.zeros(len(SIGMAS) * H * W, device=dev)
        splat_segments(canvas, H, W, u0, v0, u1, v1, inten, offset=layer * (H * W))
        canvas = canvas.view(len(SIGMAS), 1, H, W)
        total = torch.zeros(H, W, device=dev)
        for li, s in enumerate(SIGMAS):
            total = total + gblur(canvas[li:li + 1], s)[0, 0]
        a = total.clamp(0, 1)[..., None] * self.rain_alpha  # スクリーン合成(明るい背景では目立たない)
        return 255.0 - (255.0 - img) * (1.0 - a)

    def _post(self, img):
        G, dev = self.G, self.dev
        br = (img - 190.0).clamp(min=0)
        img = img + self.bloom * blur3(br, G.H / 90.0)
        x = (img / 255.0).clamp(0, 1)
        x = x + 0.25 * (x * x * (3 - 2 * x) - x)  # フィルム調のS字コントラスト
        x = x * torch.tensor([0.97, 1.0, 1.04], device=dev)  # わずかに寒色(RGB順)
        x = x * (1 - self.vig * G.rad2.clamp(0, 1.5) * 0.7)[..., None]
        out = x * 255.0 + torch.randn(G.H, G.W, 1, device=dev) * self.grain
        return out.clamp(0, 255)

    def frame(self, P, V, D, hits_p, hits_u, hits_g, hits_all, dt):
        """P,V: 雨粒の位置・速度(N,3) / D: 粒径[mm](N,) / hits_*: 衝突位置(K,3) or None。
        すべて torch テンソル(デバイス上)。戻り値: uint8 numpy (H/ss, W/ss, 3)"""
        G, dev = self.G, self.dev
        H, W = G.H, G.W
        img = self.base
        self._accumulate(self.wet_p, hits_p)
        self._accumulate(self.wet_u, hits_u)
        # 人の濡れ(暗化+鏡面ハイライト)
        wp = self.wet_p.view(H, W)
        a = (blur2(wp, 2.0) * 5.0).clamp(0, 0.6)
        spec = (blur2(wp, 0.8) - blur2(wp, 3.5)).clamp(min=0)
        img = img * (1 - 0.45 * a[..., None]) + (spec * 70.0 * a)[..., None]
        # 傘の水玉
        au = (blur2(self.wet_u.view(H, W), 1.5) * 2.0).clamp(0, 1)
        img = img * (1 - 0.25 * au[..., None]) + (G.sparkle * au * 38.0)[..., None]
        # 波紋・飛沫
        rip = blur2(self._draw_ripples(hits_g, dt).view(H, W).clamp(0, 1), 0.8)
        img = img + rip[..., None] * torch.tensor([150.0, 160.0, 175.0], device=dev) * 0.42
        spl = self._draw_splashes(hits_all, dt).view(H, W).clamp(0, 1)[..., None]
        img = img * (1 - 0.65 * spl) + torch.tensor([245.0, 250.0, 255.0], device=dev) * 0.65 * spl
        # 雨(領域内 + 遠景)
        xy = self._far_xy(len(P) / (2 * self.box_half) ** 2)
        if len(xy):
            idx = (torch.rand(len(xy), device=dev) * len(P)).long().clamp(0, len(P) - 1)
            P = torch.cat([P, torch.cat([xy, P[idx, 2:3]], 1)])
            V = torch.cat([V, V[idx]])
            D = torch.cat([D, D[idx]])
        img = self._draw_rain(img, P, V, D)
        out = self._post(img)
        if self.ss > 1:
            out = F.avg_pool2d(out.permute(2, 0, 1)[None], self.ss)[0].permute(1, 2, 0)
        return out.round().clamp(0, 255).to(torch.uint8).cpu().numpy()