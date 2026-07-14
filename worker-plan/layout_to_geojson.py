# Post-traitement géométrique : plan_raw.json (sortie Plane-DUSt3R sérialisée)
# → polygone de pièce top-down → plan.geojson métrique.
# Pur numpy + shapely : testable en local sans GPU (test_layout_to_geojson.py).
#
# Entrée (plan_raw.json) :
#   global_plane_info : [{index, pparam [nx,ny,nz,d], pre, next, left_endpoint, right_endpoint}]
#   floor_pparam / ceiling_pparam : [nx,ny,nz,d] ou []
#   cam_centers : [[x,y,z], ...]   (centres caméra, même repère monde que les plans)
#   scale_factor : float           (unités monde → mètres)
#   scale_mode, image_names : métadonnées recopiées dans le GeoJSON

import argparse
import json
import math

import numpy as np
from shapely.geometry import Polygon

MIN_AREA_M2 = 0.5
COLLINEAR_DEG = 10.0   # sous cet angle entre deux murs, l'intersection est instable
MERGE_DEG = 3.0        # murs consécutifs quasi-colinéaires fusionnés dans le polygone
MIN_VERTEX_DIST = 0.02  # 2 cm


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _oriented_floor(raw, warnings):
    """Plan du sol (n, d) avec n pointant vers le haut (caméras au-dessus)."""
    floor = list(raw.get("floor_pparam") or [])
    ceiling = list(raw.get("ceiling_pparam") or [])
    if not floor and not ceiling:
        raise ValueError("ni sol ni plafond détecté (floor_pparam et ceiling_pparam vides)")
    if not floor:
        # même convention que custom.py : sol = plafond avec normale retournée
        floor = [ceiling[0], -ceiling[1], ceiling[2], ceiling[3]]
        warnings.append("sol non détecté, déduit du plafond")
    n, d = _unit(np.array(floor[:3], float)), float(floor[3])
    cams = np.array(raw["cam_centers"], float)
    if np.mean(cams @ n + d) < 0:
        n, d = -n, -d
    return n, d, ceiling


def _ceiling_height(up, d_floor, ceiling, warnings):
    if not ceiling:
        warnings.append("plafond non détecté, hauteur inconnue")
        return None
    nc, dc = _unit(np.array(ceiling[:3], float)), float(ceiling[3])
    if np.dot(nc, up) < 0:
        nc, dc = -nc, -dc
    # offset d'un plan (n co-orienté sur up) le long de l'axe up : -d * (n·up)
    h = abs(-dc * float(np.dot(nc, up)) - (-d_floor))
    if not 2.0 <= h <= 4.5:
        warnings.append(f"hauteur sous plafond suspecte : {h:.2f} m")
    return h


def _chains(walls):
    """Chaînes de murs via le graphe pre/next. Retourne (chaîne la plus longue, fermée ?)."""
    by_id = {w["index"]: w for w in walls}
    chains = []
    visited = set()
    # chaînes ouvertes : départ = murs sans pre (ou dont le pre est manquant)
    for w in walls:
        if w["index"] in visited or (w["pre"] is not None and w["pre"] in by_id):
            continue
        chain = []
        cur = w
        while cur is not None and cur["index"] not in visited:
            visited.add(cur["index"])
            chain.append(cur)
            cur = by_id.get(cur["next"]) if cur["next"] is not None else None
        chains.append((chain, False))
    # le reste = cycles (pièce fermée)
    for w in walls:
        if w["index"] in visited:
            continue
        chain = []
        cur = w
        while cur["index"] not in visited:
            visited.add(cur["index"])
            chain.append(cur)
            cur = by_id.get(cur["next"])
            if cur is None:
                break
        chains.append((chain, cur is not None and chain and cur["index"] == chain[0]["index"]))
    if not chains:
        raise ValueError("aucun mur dans global_plane_info")
    chains.sort(key=lambda c: len(c[0]), reverse=True)
    if len(chains) > 1:
        return chains[0][0], chains[0][1], [len(c[0]) for c in chains[1:]]
    return chains[0][0], chains[0][1], []


def _wall_line_2d(wall, origin, e1, e2):
    """Droite 2D du mur : n2·p + c = 0 dans le repère (origin, e1, e2)."""
    n = np.array(wall["pparam"][:3], float)
    d = float(wall["pparam"][3])
    n2 = np.array([np.dot(n, e1), np.dot(n, e2)])
    c = float(np.dot(n, origin) + d)
    norm = np.linalg.norm(n2)
    if norm < 1e-9:
        return None, None  # mur horizontal ?! — dégénéré
    return n2 / norm, c / norm


def _project(p, origin, e1, e2):
    p = np.array(p, float)
    return np.array([np.dot(p - origin, e1), np.dot(p - origin, e2)])


def _intersect_2d(w1, w2, origin, e1, e2):
    """Intersection des droites 2D de deux murs ; None si quasi-parallèles."""
    n1, c1 = _wall_line_2d(w1, origin, e1, e2)
    n2, c2 = _wall_line_2d(w2, origin, e1, e2)
    if n1 is None or n2 is None:
        return None
    cross = abs(n1[0] * n2[1] - n1[1] * n2[0])  # sin(angle entre normales)
    if cross < math.sin(math.radians(COLLINEAR_DEG)):
        return None
    A = np.array([n1, n2])
    b = np.array([-c1, -c2])
    return np.linalg.solve(A, b)


def layout_to_geojson(raw, job_id=None):
    warnings = []
    scale = float(raw.get("scale_factor") or 1.0)

    walls_raw = raw["global_plane_info"]
    # mise à l'échelle métrique : endpoints, d des plans, centres caméra
    raw = dict(raw)
    raw["cam_centers"] = (np.array(raw["cam_centers"], float) * scale).tolist()
    walls = []
    for w in walls_raw:
        w = dict(w)
        w["pparam"] = list(w["pparam"][:3]) + [float(w["pparam"][3]) * scale]
        for k in ("left_endpoint", "right_endpoint"):
            if w.get(k) is not None:
                w[k] = (np.array(w[k], float) * scale).tolist()
        walls.append(w)

    def scale_plane(p):
        return [p[0], p[1], p[2], float(p[3]) * scale] if p else p

    raw["floor_pparam"] = scale_plane(raw.get("floor_pparam"))
    raw["ceiling_pparam"] = scale_plane(raw.get("ceiling_pparam"))

    up, d_floor, ceiling = _oriented_floor(raw, warnings)
    # d_floor tel que up·x + d_floor = 0 sur le sol ; offset du sol le long de up = -d_floor
    height = _ceiling_height(up, d_floor, ceiling, warnings)

    chain, closed, dropped = _chains(walls)
    if dropped:
        warnings.append(f"chaînes de murs secondaires ignorées : {dropped}")
    if not closed:
        warnings.append("pièce partiellement filmée : polygone fermé par un segment déduit")

    # repère 2D au sol : origine = centroïde caméras projeté, e1 = mur le plus long
    cams = np.array(raw["cam_centers"], float)
    centroid = cams.mean(axis=0)
    origin = centroid - (np.dot(centroid, up) + d_floor) * up

    def horiz_dir(w):
        if w.get("left_endpoint") is None or w.get("right_endpoint") is None:
            return None, 0.0
        d3 = np.array(w["right_endpoint"], float) - np.array(w["left_endpoint"], float)
        d3 -= np.dot(d3, up) * up
        return d3, float(np.linalg.norm(d3))

    longest = max(chain, key=lambda w: horiz_dir(w)[1])
    d3, ln = horiz_dir(longest)
    e1 = _unit(d3) if ln > 1e-6 else _unit(np.cross(up, [0.0, 0.0, 1.0]))
    e2 = np.cross(up, e1)

    # sommets : intersection des droites 2D de murs consécutifs ;
    # extrémités de chaîne ouverte = endpoints projetés
    verts = []
    inferred_flags = []  # inferred_flags[i] : le mur entre verts[i] et verts[i+1] est déduit
    pairs = list(zip(chain, chain[1:] + [chain[0]])) if closed else list(zip(chain, chain[1:]))
    if not closed:
        p = chain[0].get("left_endpoint")
        verts.append(_project(p, origin, e1, e2) if p is not None else None)
    for w1, w2 in pairs:
        v = _intersect_2d(w1, w2, origin, e1, e2)
        if v is None:
            # murs quasi-colinéaires : milieu des endpoints projetés
            a, b = w1.get("right_endpoint"), w2.get("left_endpoint")
            pts = [_project(p, origin, e1, e2) for p in (a, b) if p is not None]
            v = np.mean(pts, axis=0) if pts else None
            if v is None:
                warnings.append(f"jonction murs {w1['index']}→{w2['index']} indéterminée, ignorée")
                continue
        verts.append(v)
    if not closed:
        p = chain[-1].get("right_endpoint")
        verts.append(_project(p, origin, e1, e2) if p is not None else None)
    verts = [v for v in verts if v is not None]
    if len(verts) < 3:
        raise ValueError(f"seulement {len(verts)} sommets exploitables — plan inutilisable")

    inferred_flags = [False] * len(verts)  # mur i : verts[i] → verts[(i+1) % n]
    if not closed:
        inferred_flags[-1] = True  # segment de fermeture

    # nettoyage : sommets confondus puis murs quasi-colinéaires fusionnés.
    # Le mur flags[i] va de vs[i] à vs[(i+1) % n] ; supprimer le sommet j
    # fusionne les murs i et j en un seul qui garde l'indice i.
    def cleanup(vs, flags, min_dist, max_deg):
        changed = True
        while changed and len(vs) > 3:
            changed = False
            for i in range(len(vs)):
                j = (i + 1) % len(vs)
                near = np.linalg.norm(vs[j] - vs[i]) < min_dist
                collinear = (
                    float(np.dot(_unit(vs[j] - vs[i]), _unit(vs[(j + 1) % len(vs)] - vs[j])))
                    > math.cos(math.radians(max_deg))
                    and flags[i] == flags[j]  # ne pas fondre un mur déduit dans un mur réel
                )
                if near or collinear:
                    merged = flags[i] or flags[j]
                    vs.pop(j)
                    flags.pop(j)
                    flags[i if j > i else i - 1] = merged
                    changed = True
                    break
        return vs, flags

    verts, inferred_flags = cleanup(list(verts), inferred_flags, MIN_VERTEX_DIST, MERGE_DEG)

    poly = Polygon([tuple(v) for v in verts])
    if not poly.is_valid:
        warnings.append("polygone auto-intersectant, réparé par buffer(0)")
        poly = poly.buffer(0)
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        verts = [np.array(c) for c in poly.exterior.coords[:-1]]
        inferred_flags = [False] * len(verts)
    if poly.area < MIN_AREA_M2:
        raise ValueError(f"aire {poly.area:.2f} m² < {MIN_AREA_M2} m² — reconstruction inutilisable")

    # orientation CCW cohérente entre polygone et liste de murs (shoelace)
    n = len(verts)
    signed_area = sum(
        verts[i][0] * verts[(i + 1) % n][1] - verts[(i + 1) % n][0] * verts[i][1] for i in range(n)
    ) / 2.0
    if signed_area < 0:
        verts = list(reversed(verts))
        # nouveau mur k relie v[n-1-k] → v[n-2-k] : c'était le mur (n-k-2) % n
        inferred_flags = [inferred_flags[(n - k - 2) % n] for k in range(n)]
        poly = Polygon([tuple(v) for v in verts])

    def rnd(x):
        return round(float(x), 3)

    coords = [[rnd(v[0]), rnd(v[1])] for v in verts]
    features = [{
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [coords + [coords[0]]]},
        "properties": {
            "kind": "room",
            "area_m2": rnd(poly.area),
            "perimeter_m": rnd(poly.length),
            "ceiling_height_m": rnd(height) if height is not None else None,
            "closed": closed,
            "scale_mode": raw.get("scale_mode"),
            "scale_factor": rnd(scale),
            "n_walls": len(coords),
            "n_frames": len(raw.get("image_names") or raw["cam_centers"]),
            "warnings": warnings,
        },
    }]
    for i in range(len(coords)):
        a, b = coords[i], coords[(i + 1) % len(coords)]
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [a, b]},
            "properties": {
                "kind": "wall",
                "index": i,
                "length_m": rnd(math.dist(a, b)),
                "inferred": bool(inferred_flags[i]),
            },
        })
    return {
        "type": "FeatureCollection",
        "properties": {
            "units": "meters",
            "crs": "local-xy-meters",
            "generator": "plane-dust3r",
            "job_id": job_id,
        },
        "features": features,
    }


def main():
    ap = argparse.ArgumentParser(description="plan_raw.json (Plane-DUSt3R) → plan.geojson")
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--job_id", default=None)
    args = ap.parse_args()
    with open(args.in_json) as f:
        raw = json.load(f)
    geojson = layout_to_geojson(raw, job_id=args.job_id)
    with open(args.out, "w") as f:
        json.dump(geojson, f, indent=2)
    props = geojson["features"][0]["properties"]
    print(f"plan.geojson : {props['n_walls']} murs, {props['area_m2']} m², "
          f"h={props['ceiling_height_m']} m, closed={props['closed']}, warnings={props['warnings']}")


if __name__ == "__main__":
    main()
