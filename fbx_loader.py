"""
fbx_loader.py â å¤é¨ã©ã¤ãã©ãªä¸è¦ã®ãã¤ããªFBXã­ã¼ãã¼(numpy + scipy ã®ã¿)

ãã¤ããªFBX(7.x)ãããéå±¤(è¦ªå­)ã® Lcl/Pre/Post åè»¢ã»å¹³è¡ç§»åã»ã¹ã±ã¼ã«ã»ããããã»
Geometricå¤æãé©ç¨ãããã¯ã¼ã«ãåº§æ¨ã®ä¸è§å½¢ã¡ãã·ã¥ããè¿ããtrimesh ã¯ FBX ãèª­ããªãããç¨æã
æ³¨æ: ãããªã¢ã«/ãã¯ã¹ãã£/ã¢ãã¡ã¼ã·ã§ã³/ã¹ã­ãã³ã°ã¯èª­ã¾ãªã(éçãã¼ãºã®å½¢ç¶ã®ã¿)ã
ASCII FBX ã¯éå¯¾å¿(Blenderç­ã§ãã¤ããªFBXãglbã«å¤æãã¦ãã ãã)ã
"""
import struct
import zlib

import numpy as np

def read_fbx(path):
    d = open(path, 'rb').read()
    ver = struct.unpack_from('<I', d, 23)[0]
    big = ver >= 7500
    pos = 27
    def prop(p):
        t = chr(d[p]); p += 1
        if t in 'YCIFDL':
            fmt = {'Y':'<h','C':'<?','I':'<i','F':'<f','D':'<d','L':'<q'}[t]
            v = struct.unpack_from(fmt, d, p)[0]; return v, p + struct.calcsize(fmt)
        if t in 'fdlib':
            n, enc, clen = struct.unpack_from('<III', d, p); p += 12
            dt = {'f':'<f4','d':'<f8','l':'<i8','i':'<i4','b':'?'}[t]
            raw = d[p:p+clen]; p += clen
            if enc == 1: raw = zlib.decompress(raw)
            return np.frombuffer(raw, dtype=dt, count=n), p
        if t in 'SR':
            n = struct.unpack_from('<I', d, p)[0]; p += 4
            raw = d[p:p+n]; p += n
            return (raw.decode('utf-8', 'replace') if t == 'S' else raw), p
        raise ValueError(t)
    def node(p):
        if big: end, npr, plen, nl = struct.unpack_from('<QQQB', d, p); p += 25
        else:   end, npr, plen, nl = struct.unpack_from('<IIIB', d, p); p += 13
        if end == 0: return None, p
        name = d[p:p+nl].decode(); p += nl
        props = []
        for _ in range(npr):
            v, p = prop(p); props.append(v)
        children = []
        while p < end:
            c, p = node(p)
            if c is None: break
            children.append(c)
        return dict(name=name, props=props, children=children), end
    nodes = []
    while pos < len(d) - 200:
        n, pos2 = node(pos)
        if n is None: break
        nodes.append(n); pos = pos2
    return nodes


from scipy.spatial.transform import Rotation as Rot

_ORDERS = {0: "xyz", 1: "xzy", 2: "yzx", 3: "yxz", 4: "zxy", 5: "zyx"}  # FBX RotationOrder (適用順)


def _name(c):
    s = c["props"][1] if len(c["props"]) > 1 else ""
    return str(s).split("\x00")[0]


def _rot(euler_deg, order=0):
    seq = _ORDERS.get(int(order), "xyz")
    idx = {"x": 0, "y": 1, "z": 2}
    ang = [euler_deg[idx[ch]] for ch in seq]
    return Rot.from_euler(seq, ang, degrees=True).as_matrix()  # 小文字=外因性: 先頭軸から順に適用


def _T(t):
    m = np.eye(4)
    m[:3, 3] = t
    return m


def _R(r, order=0):
    m = np.eye(4)
    m[:3, :3] = _rot(r, order)
    return m


def _S(s):
    return np.diag([s[0], s[1], s[2], 1.0])


def _props(model):
    out = {}
    P = next((x for x in model["children"] if x["name"] == "Properties70"), None)
    if P:
        for x in P["children"]:
            out[x["props"][0]] = [float(v) if not isinstance(v, str) else v for v in x["props"][4:]]
    return out


def load_fbx_parts(path):
    """バイナリFBXから、階層の変換を適用したワールド座標の三角形メッシュ一覧を返す。
    戻り値: [{"name": str, "V": (N,3), "F": (M,3)}], info(dict)"""
    nodes = read_fbx(path)
    obj = next(n for n in nodes if n["name"] == "Objects")
    conn = next(n for n in nodes if n["name"] == "Connections")
    byid = {c["props"][0]: c for c in obj["children"]}
    parent_of, geo_of = {}, {}
    for c in conn["children"]:
        if c["props"][0] != "OO":
            continue
        ch, pa = c["props"][1], c["props"][2]
        if ch in byid and byid[ch]["name"] == "Model" and (pa == 0 or (pa in byid and byid[pa]["name"] == "Model")):
            parent_of[ch] = pa
        if ch in byid and byid[ch]["name"] == "Geometry" and pa in byid and byid[pa]["name"] == "Model":
            geo_of.setdefault(pa, []).append(ch)

    local_cache = {}

    def world(mid):
        if mid in local_cache:
            return local_cache[mid]
        p = _props(byid[mid])
        g = lambda k, d: p.get(k, d)
        order = g("RotationOrder", [0])[0]
        T, R = g("Lcl Translation", [0, 0, 0]), g("Lcl Rotation", [0, 0, 0])
        Sc = g("Lcl Scaling", [1, 1, 1])
        Rp, Sp = g("RotationPivot", [0, 0, 0]), g("ScalingPivot", [0, 0, 0])
        Ro, So = g("RotationOffset", [0, 0, 0]), g("ScalingOffset", [0, 0, 0])
        Rpre, Rpost = g("PreRotation", [0, 0, 0]), g("PostRotation", [0, 0, 0])
        L = (_T(T) @ _T(Ro) @ _T(Rp) @ _R(Rpre, order) @ _R(R, order) @ np.linalg.inv(_R(Rpost, order))
             @ _T(-np.array(Rp)) @ _T(So) @ _T(Sp) @ _S(Sc) @ _T(-np.array(Sp)))
        pa = parent_of.get(mid, 0)
        W = (world(pa) @ L) if pa in byid else L
        local_cache[mid] = W
        return W

    parts = []
    for mid, gids in geo_of.items():
        W = world(mid)
        p = _props(byid[mid])
        G = (_T(p.get("GeometricTranslation", [0, 0, 0])) @ _R(p.get("GeometricRotation", [0, 0, 0]), p.get("RotationOrder", [0])[0])
             @ _S(p.get("GeometricScaling", [1, 1, 1])))
        M = W @ G
        for gid in gids:
            ge = byid[gid]
            V = next(x for x in ge["children"] if x["name"] == "Vertices")["props"][0].reshape(-1, 3)
            I = next(x for x in ge["children"] if x["name"] == "PolygonVertexIndex")["props"][0].astype(np.int64)
            ends = np.nonzero(I < 0)[0]
            starts = np.concatenate([[0], ends[:-1] + 1])
            sizes = ends - starts + 1
            idx = np.where(I < 0, -I - 1, I)
            tris = []
            for s in np.unique(sizes):
                if s < 3:
                    continue
                st = starts[sizes == s]
                for k in range(1, s - 1):
                    tris.append(np.stack([idx[st], idx[st + k], idx[st + k + 1]], 1))
            F = np.vstack(tris) if tris else np.zeros((0, 3), np.int64)
            Vw = V @ M[:3, :3].T + M[:3, 3]
            parts.append(dict(name=_name(byid[mid]), V=Vw, F=F))
    return parts, {}
