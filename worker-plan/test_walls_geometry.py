# Tests locaux (sans GPU) du mode walls : scène synthétique d'une pièce 4×3 m
# dans un MONDE INCLINÉ (attrape toute hypothèse cachée d'axe vertical), avec
# photos de coin (ponts), fermeture de boucle, armoire devant un mur, bruit et
# outliers. python3 worker-plan/test_walls_geometry.py (venv numpy + shapely).

import math

import numpy as np

from walls_geometry import analyze_walls
from layout_to_geojson import layout_to_geojson


def _unit(v):
    return v / np.linalg.norm(v)


UP = _unit(np.array([0.1, 0.97, 0.2]))          # verticale monde inclinée
A1 = _unit(np.array([1.0, 0.0, 0.0]) - np.dot([1.0, 0.0, 0.0], UP) * UP)
A2 = np.cross(UP, A1)
B = np.array([5.0, -2.0, 3.0])                   # origine décalée
ROOM_W, ROOM_D, ROOM_H = 4.0, 3.0, 2.5
CENTER = np.array([2.0, 1.5])
H_IMG, W_IMG = 32, 40                            # grille de "pixels" synthétique

# yaws des 12 photos (° ; 0 = face au mur 0) : 3 photos mur 0, pont, 2 photos
# mur 1, 3 photos mur 2, pont, 1 photo mur 3, retour sur le mur 0 (boucle)
YAWS = [-8, 0, 12, 47, 90, 98, 178, 185, 192, 227, 270, 352]
EXPECTED_BRIDGES = [3, 9]
EXPECTED_GROUPS = [[0, 1, 2, 11], [4, 5], [6, 7, 8], [10]]


def world(x, y, h):
    return B + x * A1 + y * A2 + h * UP


def dir2d(phi_deg):
    p = math.radians(phi_deg)
    return np.array([math.sin(p), -math.cos(p)])  # 0°→mur0 (y=0), 90°→mur1 (x=4)


def wall_point(k, s, rng):
    """Point 2D sur le mur k, abscisse s le long du mur (petites marges)."""
    if k == 0:
        return np.array([0.2 + s * 3.6, 0.0])
    if k == 1:
        return np.array([4.0, 0.2 + s * 2.6])
    if k == 2:
        return np.array([3.8 - s * 3.6, 3.0])
    return np.array([0.0, 2.8 - s * 2.6])


def build_scene(scale=1.0, seed=7):
    rng = np.random.default_rng(seed)
    cam2w_list, pts_list, confs_list, names = [], [], [], []
    for idx, phi in enumerate(YAWS):
        d2 = dir2d(phi)
        fwd = _unit(d2[0] * A1 + d2[1] * A2)
        pos = world(*(CENTER + 0.3 * d2), 1.5) * scale
        y_cam = -UP
        x_cam = np.cross(y_cam, fwd)
        cam2w = np.eye(4)
        cam2w[:3, 0], cam2w[:3, 1], cam2w[:3, 2], cam2w[:3, 3] = x_cam, y_cam, fwd, pos
        cam2w_list.append(cam2w)

        faced = int(round((phi % 360) / 90.0)) % 4
        is_bridge = idx in EXPECTED_BRIDGES
        pts = np.zeros((H_IMG * W_IMG, 3))
        conf = np.full(H_IMG * W_IMG, 2.5)
        for p in range(H_IMG * W_IMG):
            r, c = divmod(p, W_IMG)
            fr, fc = r / H_IMG, c / W_IMG
            u = rng.random()
            if fr < 0.15:                                   # plafond
                xy = CENTER + (rng.random(2) - 0.5) * [3.5, 2.5]
                h = ROOM_H + rng.normal(0, 0.01)
            elif fr > 0.88:                                 # sol
                xy = CENTER + (rng.random(2) - 0.5) * [3.5, 2.5]
                h = rng.normal(0, 0.01)
            elif u < 0.03:                                  # points flous (conf basse)
                xy = CENTER + (rng.random(2) - 0.5) * 2.0
                h = rng.random() * ROOM_H
                conf[p] = 0.5
            elif u < 0.08:                                  # mobilier épars
                xy = CENTER + (rng.random(2) - 0.5) * 2.0
                h = rng.random() * 1.2
            else:                                           # murs
                if is_bridge:                               # coin : mélange 50/50
                    k = faced if rng.random() < 0.5 else (faced + (1 if phi % 90 > 45 else -1)) % 4
                elif fc < 0.2:                              # bords ultra-wide :
                    k = (faced - 1) % 4                     # murs voisins
                elif fc >= 0.8:
                    k = (faced + 1) % 4
                else:
                    k = faced
                s = rng.random()
                h = 0.1 + ((fr - 0.15) / 0.73) * (ROOM_H - 0.2) + rng.normal(0, 0.02)
                xy = wall_point(k, s, rng)
                # armoire à 10 cm devant le mur 0 (25 % de ses points, h ≤ 1.7)
                if k == 0 and rng.random() < 0.25:
                    xy = xy + np.array([0.0, 0.10])
                    h = rng.random() * 1.7
            P = world(xy[0], xy[1], h) + rng.normal(0, 0.015, 3)
            pts[p] = P * scale
        pts_list.append(pts)
        confs_list.append(conf.reshape(H_IMG, W_IMG))
        names.append(f"IMG_{idx:02d}.jpeg")
    return np.stack(cam2w_list), pts_list, confs_list, names


def close(a, b, tol):
    assert abs(a - b) < tol, f"{a} != {b} (±{tol})"


def test_floor_and_up():
    cam2w, pts, confs, names = build_scene()
    raw, report = analyze_walls(cam2w, pts, confs, names)
    n = _unit(np.array(raw["floor_pparam"][:3]))
    ang = math.degrees(math.acos(min(1.0, abs(float(np.dot(n, UP))))))
    assert ang < 2.0, f"normale sol à {ang:.2f}° du vrai up"
    assert report["up_prior_vs_floor_deg"] < 3.0
    assert report["ceiling"]["found"], report["ceiling"]
    close(report["ceiling"]["height_m"], ROOM_H, 0.1)
    print("✓ verticale + sol/plafond RANSAC (monde incliné)")


def test_clusters_and_bridges():
    cam2w, pts, confs, names = build_scene()
    raw, report = analyze_walls(cam2w, pts, confs, names)
    assert report["bridge_photos"] == EXPECTED_BRIDGES, report["bridge_photos"]
    got = [c["photos"] for c in report["clusters"]]
    assert got == EXPECTED_GROUPS, got
    print("✓ clustering yaw : 4 murs, ponts exclus, boucle fusionnée (photo 11 → mur 0)")


def test_walls_robust_to_wardrobe():
    cam2w, pts, confs, names = build_scene()
    raw, report = analyze_walls(cam2w, pts, confs, names)
    w0 = report["walls"][0]
    assert w0["rms_m"] < 0.03, w0
    # le plan fit du mur 0 doit être le MUR (y=0), pas l'armoire (y=0.10) :
    # distance du vrai point de mur au plan < 4 cm
    p = world(2.0, 0.0, 1.2)
    n = np.array(raw["global_plane_info"][0]["pparam"][:3])
    d = raw["global_plane_info"][0]["pparam"][3]
    assert abs(np.dot(n, p) + d) / np.linalg.norm(n) < 0.04, "le fit a préféré l'armoire"
    print("✓ RANSAC mur robuste à l'armoire à 10 cm")


def test_end_to_end():
    cam2w, pts, confs, names = build_scene()
    raw, report = analyze_walls(cam2w, pts, confs, names)
    geo = layout_to_geojson(raw, job_id="walls-test")
    props = geo["features"][0]["properties"]
    assert props["closed"] is True
    assert props["n_walls"] == 4, props
    close(props["area_m2"], ROOM_W * ROOM_D, 0.3)
    close(props["ceiling_height_m"], ROOM_H, 0.1)
    close(props["perimeter_m"], 2 * (ROOM_W + ROOM_D), 0.3)
    assert props["n_doors"] == 0 and props["n_windows"] == 0
    assert report["metric_check"]["passed"], report["metric_check"]
    print("✓ end-to-end plan_raw → layout_to_geojson : 4 murs fermés, 12 m², h 2,5 m")


def test_metric_check_scaled():
    cam2w, pts, confs, names = build_scene(scale=3.0)
    raw, report = analyze_walls(cam2w, pts, confs, names)
    assert not report["metric_check"]["passed"], report["metric_check"]
    assert any("échelle" in w for w in report["warnings"])
    print("✓ metric_check : échelle ×3 détectée comme suspecte")


if __name__ == "__main__":
    test_floor_and_up()
    test_clusters_and_bridges()
    test_walls_robust_to_wardrobe()
    test_end_to_end()
    test_metric_check_scaled()
    print("Tous les tests walls passent.")
