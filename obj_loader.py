import re
import numpy as np


def load_obj_parts(path):
    """OBJ を o/g グループ単位のパーツに分けて読む(numpyのみ)。多角形は扇状に三角形化、負のインデックス対応。
    戻り値: [{"name", "V"(N,3), "F"(M,3), "mtl": 最後のusemtl名 or None}]"""
    verts = []
    groups = {}  # name -> (list of face index arrays, mtl)
    order = []
    cur, mtl = "default", None

    def grp(name):
        if name not in groups:
            groups[name] = {"faces": [], "mtl": None}
            order.append(name)
        return groups[name]

    with open(path, "rb") as f:
        for raw in f:
            if raw[:2] == b"v ":
                p = raw.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif raw[:2] == b"f ":
                nv = len(verts)
                idx = []
                for tok in raw.split()[1:]:
                    i = int(tok.split(b"/")[0])
                    idx.append(i - 1 if i > 0 else nv + i)
                g = grp(cur)
                if mtl is not None:
                    g["mtl"] = mtl
                for k in range(1, len(idx) - 1):  # 扇状三角形化
                    g["faces"].append((idx[0], idx[k], idx[k + 1]))
            elif raw[:2] in (b"o ", b"g "):
                cur = raw[2:].strip().decode("utf-8", "replace") or "default"
                grp(cur)
            elif raw[:7] == b"usemtl ":
                mtl = raw[7:].strip().decode("utf-8", "replace")
    V = np.asarray(verts, dtype=np.float64)
    parts = []
    for name in order:
        F = np.asarray(groups[name]["faces"], dtype=np.int64).reshape(-1, 3)
        if len(F) == 0:
            continue
        used, inv = np.unique(F, return_inverse=True)
        parts.append(dict(name=name, V=V[used], F=inv.reshape(-1, 3), mtl=groups[name]["mtl"]))
    return parts


def mtl_color(name):
    """3ds Max の 'wire_RRRGGGBBB' 形式の材質名から色(0-255)を取り出す。無ければ None。"""
    m = re.match(r"wire_(\d{3})(\d{3})(\d{3})$", name or "")
    return tuple(int(m.group(i)) for i in (1, 2, 3)) if m else None
