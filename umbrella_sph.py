"""
【実験的】Genesis の SPH 流体で、1つの傘角度について雨・飛沫・流れを重く・リアルに可視化する。

位置づけ(重要):
  - 濡れ量の「正確な数値」は umbrella_rain_v2.py(粒子+すり抜け検証済み)が出す。
  - こちらは見た目のリアルさ用。SPH粒子は最小でも約5〜10mm(実際の雨粒は1〜3mm)で、
    Genesisの剛体との衝突も離散判定なので、数値の根拠には使わない。
  - 傘は薄い殻で、そのままではSDFが定義できない。そこで v2 のボクセル殻から
    マーチングキューブで「閉じた衝突用プロキシ」を作り、見た目用モデルとは別に置く。
  - Genesis の SPH/エミッタ/カップラーの API はバージョンで変わりやすい。
    ドローン時と同様、エラーが出たらそのメッセージを教えてください。

実行例(まず v2 で最適角度を出してから):
  uv run python umbrella_sph.py --couple assets/couple.glb --umbrella assets/umbrella.glb --theta 15 --phi 0
"""

import os

import imageio
import numpy as np
import trimesh
from scipy import ndimage

import genesis as gs
from umbrella_rain_v2 import build_assets, build_parser, place, to_np


def proxy_mesh(raw, pitch, fill):
    """ボクセル殻 -> 閉じたメッシュ(SDF・SPH衝突用)。"""
    m, origin = raw
    m = ndimage.binary_dilation(m, iterations=1)
    if fill:
        m = ndimage.binary_fill_holes(m)
    T = np.eye(4)
    T[:3, :3] = np.eye(3) * pitch
    T[:3, 3] = origin
    vg = trimesh.voxel.VoxelGrid(trimesh.voxel.encoding.DenseEncoding(m), transform=T)
    return vg.marching_cubes


def main():
    p = build_parser()
    p.add_argument("--theta", type=float, required=True)
    p.add_argument("--particle", type=float, default=0.01, help="SPH粒子サイズ[m](小さいほど重い)")
    p.add_argument("--emit-per-step", type=int, default=6, help="1ステップあたり発射する雨粒数")
    p.add_argument("--speed", type=float, default=6.0)
    p.add_argument("--seconds", type=float, default=4.0)
    a = p.parse_args()

    gs.init(backend=gs.gpu, logging_level="warning", seed=a.seed)
    A = build_assets(a, gs.device)
    R, quat, H, lift = place(A, a, a.theta)
    if H is None:
        raise SystemExit("この角度は人と干渉して実現できません。")

    cache = os.path.join(a.out, "_cache")
    pitch = a.voxel
    Rn, Hn = R.cpu().numpy(), H.cpu().numpy()

    u_proxy = proxy_mesh(A.u_raw, pitch, fill=False)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rn, Hn
    u_proxy.apply_transform(T)  # 傘ローカル -> ワールド
    p_proxy = proxy_mesh(A.p_raw, pitch, fill=True)
    u_path, p_path = os.path.join(cache, "umbrella_proxy.obj"), os.path.join(cache, "couple_proxy.obj")
    u_proxy.export(u_path)
    p_proxy.export(p_path)

    coupler = None
    for name in ("LegacyCouplerOptions", "CouplerOptions"):
        if hasattr(gs.options, name):
            coupler = getattr(gs.options, name)(rigid_sph=True)
            break
    kw = dict(
        sim_options=gs.options.SimOptions(dt=4e-3, substeps=10),
        sph_options=gs.options.SPHOptions(lower_bound=(-1.5, -1.5, 0.0), upper_bound=(1.5, 1.5, 2.8),
                                          particle_size=a.particle),
        show_viewer=False,
    )
    if coupler is not None:
        kw["coupler_options"] = coupler
    scene = gs.Scene(**kw)

    scene.add_entity(gs.morphs.Plane())
    # 見た目用(衝突なし)
    scene.add_entity(gs.morphs.Mesh(file=os.path.abspath(A.couple_glb), fixed=True, collision=False, parse_glb_with_zup=True))
    scene.add_entity(gs.morphs.Mesh(file=os.path.abspath(A.umb_glb), pos=tuple(float(x) for x in Hn),
                                    quat=tuple(float(x) for x in quat), fixed=True, collision=False,
                                    parse_glb_with_zup=True))
    # 衝突用プロキシ(非表示、閉じたメッシュ)
    for path in (u_path, p_path):
        scene.add_entity(gs.morphs.Mesh(file=os.path.abspath(path), fixed=True, visualization=False, collision=True,
                                        convexify=False))

    emitter = scene.add_emitter(
        material=gs.materials.SPH.Liquid(),
        max_particles=400000,
        surface=gs.surfaces.Default(color=(0.6, 0.8, 1.0, 0.8), vis_mode="particle"),
    )
    cam = scene.add_camera(res=tuple(a.res), pos=tuple(a.cam_pos), lookat=tuple(a.cam_lookat), fov=a.fov, GUI=False)
    scene.build()

    rng = np.random.default_rng(a.seed)
    n_steps = int(a.seconds / 4e-3)
    out = os.path.join(a.out, f"sph_phi{int(a.phi):03d}_theta{int(a.theta):+03d}.mp4")
    writer = imageio.get_writer(out, fps=30, codec="libx264", quality=8)
    every = max(1, round(1 / (30 * 4e-3)))
    direction = np.array([a.wind_x, a.wind_y, -a.speed])
    direction = direction / np.linalg.norm(direction)

    for i in range(n_steps):
        for _ in range(a.emit_per_step):
            emitter.emit(
                pos=np.array([rng.uniform(-1.2, 1.2), rng.uniform(-1.2, 1.2), 2.6]),
                direction=direction,
                speed=a.speed,
                droplet_shape="circle",
                droplet_size=a.particle,
            )
        scene.step()
        if i % every == 0:
            writer.append_data(np.ascontiguousarray(to_np(cam.render(rgb=True)[0])).astype(np.uint8))
    writer.close()
    print(f"保存: {out}")


if __name__ == "__main__":
    main()
