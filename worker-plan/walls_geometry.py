# Géométrie du mode "walls" : photos ordonnées des murs → plan_raw.json.
# Pur numpy (matplotlib importé uniquement dans render_debug) : tout se rejoue
# et se teste en local sans GPU (test_walls_geometry.py).
#
# Entrées (produites par mast3r_walls_infer.py depuis MASt3R-SfM) :
#   cam2w      : (N,4,4) poses caméra→monde (convention OpenCV : x droite,
#                y bas, z avant)
#   pts_list   : liste N × (H·W, 3) pointmaps denses (repère monde)
#   confs_list : liste N × (H, W) confiances
# Sortie : plan_raw au schéma commun de layout_to_geojson (pparam n·x + d = 0,
# endpoints 3D repère monde, liens pre/next réciproques) + rapport de qualité.

import math

import numpy as np

CONF_MIN = 1.5              # confiance dense minimale
YAW_NEW_GROUP_DEG = 40.0    # nouveau mur si écart au yaw moyen du groupe > 40°
YAW_AMBIGUOUS_DEG = 20.0    # photo "de coin" si à < 20° de DEUX moyennes
CENTER_COLS = (0.25, 0.75)  # colonnes centrales (l'ultra-wide voit les voisins sur les bords)
HEIGHT_BAND = (0.20, 0.85)  # bande de hauteur sol-plafond pour le fit mur
LINE_INLIER_M = 0.04        # seuil inlier RANSAC droite (4 cm)
LINE_ITERS = 200
LINE_MIN_SPAN_M = 0.2       # échantillons RANSAC distants d'au moins 20 cm
FLOOR_INLIER_M = 0.03
FLOOR_ITERS = 300
FLOOR_LOW_PCT = 15.0        # points bas = percentile hauteur < 15 %
CEIL_HIGH_PCT = 85.0
UP_MAX_TILT_DEG = 15.0      # normale sol/plafond à < 15° du prior caméras
MIN_PLANE_INLIERS = 2000    # en-dessous, plafond considéré non détecté
ENDPOINT_PCT = (2.0, 98.0)  # extrémités = percentiles des inliers le long de la droite
MAX_FIT_POINTS = 60_000     # sous-échantillonnage par mur avant RANSAC
MAX_UP_POINTS = 150_000     # sous-échantillonnage pour le RANSAC sol/plafond
CAM_HEIGHT_RANGE = (1.0, 2.0)
ROOM_DIAMETER_RANGE = (1.5, 20.0)
MIN_WALL_INLIER_RATIO = 0.35


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _circ_mean(angles):
    return math.atan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))


def _angdiff(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def _subsample(arr, n, rng):
    if len(arr) <= n:
        return arr
    return arr[rng.choice(len(arr), n, replace=False)]


def _ransac_plane(P, up_prior, rng, inlier_m, iters, max_tilt_deg):
    """Plan RANSAC contraint quasi-horizontal (normale près du prior), refit
    total least squares. Retourne (n, d, n_inliers, rms) ou None."""
    if len(P) < 50:
        return None
    cos_tol = math.cos(math.radians(max_tilt_deg))
    best_inl, best_count = None, 0
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        if np.dot(n, up_prior) < 0:
            n = -n
        if np.dot(n, up_prior) < cos_tol:
            continue
        d = -float(np.dot(n, a))
        inl = np.abs(P @ n + d) < inlier_m
        cnt = int(inl.sum())
        if cnt > best_count:
            best_count, best_inl = cnt, inl
    if best_inl is None or best_count < 50:
        return None
    Q = P[best_inl]
    mu = Q.mean(axis=0)
    _, V = np.linalg.eigh(np.cov((Q - mu).T))
    n = _unit(V[:, 0])
    if np.dot(n, up_prior) < 0:
        n = -n
    d = -float(np.dot(n, mu))
    rms = float(np.sqrt(np.mean((Q @ n + d) ** 2)))
    return n, d, best_count, rms


def estimate_up_and_floor(cam2w, pts_list, confs_list, rng, warnings, report):
    """Verticale + plans sol/plafond. Prior = moyenne des 'hauts caméra'
    (cf. gravity_align de mast3r_colmap.py), raffiné par RANSAC sur les points
    bas ; le plafond sert de borne de hauteur s'il est trouvé."""
    up0 = _unit(-cam2w[:, :3, 1].mean(axis=0))
    pts = np.concatenate([
        p[c.reshape(-1) > CONF_MIN] for p, c in zip(pts_list, confs_list)
    ])
    pts = _subsample(pts, MAX_UP_POINTS, rng)
    if len(pts) < 500:
        raise ValueError(f"seulement {len(pts)} points confiants — reconstruction inutilisable")

    h0 = pts @ up0
    low = pts[h0 < np.percentile(h0, FLOOR_LOW_PCT)]
    floor = _ransac_plane(low, up0, rng, FLOOR_INLIER_M, FLOOR_ITERS, UP_MAX_TILT_DEG)
    if floor is None:
        warnings.append("RANSAC sol sans consensus : verticale = prior caméras, sol au percentile bas")
        n = up0
        d = -float(np.percentile(h0, 2.0))
        floor = (n, d, 0, 0.0)
    n_f, d_f, n_inl_f, rms_f = floor

    cams = cam2w[:, :3, 3]
    if float(np.mean(cams @ n_f + d_f)) < 0:  # caméras au-dessus du sol
        n_f, d_f = -n_f, -d_f
    cross_deg = math.degrees(math.acos(min(1.0, abs(float(np.dot(n_f, up0))))))
    if cross_deg > 10.0:
        warnings.append(f"verticale : prior caméras et sol RANSAC divergent de {cross_deg:.1f}°")
    report["up_prior_vs_floor_deg"] = round(cross_deg, 2)
    report["floor"] = {"n_inliers": n_inl_f, "rms_m": round(rms_f, 4)}

    h = pts @ n_f + d_f
    high = pts[h > np.percentile(h, CEIL_HIGH_PCT)]
    ceil = _ransac_plane(high, n_f, rng, FLOOR_INLIER_M, FLOOR_ITERS, UP_MAX_TILT_DEG)
    ceiling_pparam = []
    if ceil is not None and ceil[2] >= MIN_PLANE_INLIERS:
        n_c, d_c, n_inl_c, rms_c = ceil
        ceiling_pparam = [float(n_c[0]), float(n_c[1]), float(n_c[2]), float(d_c)]
        hc = abs(-d_c * float(np.dot(n_c, n_f)) - (-d_f))
        report["ceiling"] = {"found": True, "n_inliers": n_inl_c, "rms_m": round(rms_c, 4),
                             "height_m": round(hc, 3)}
    else:
        hc = float(np.percentile(h, 99.0))
        report["ceiling"] = {"found": False, "height_est_m": round(hc, 3)}
        warnings.append("plafond non détecté (peu visible sur les photos) — bande de hauteur estimée")

    floor_pparam = [float(n_f[0]), float(n_f[1]), float(n_f[2]), float(d_f)]
    return floor_pparam, ceiling_pparam, n_f, float(hc)


def cluster_by_yaw(cam2w, up, warnings, report):
    """Photos → murs par direction de visée horizontale. Segmentation gloutonne
    de la séquence ordonnée + fusion wrap + une passe de réaffectation au yaw
    moyen le plus proche. Les photos 'de coin' sont flaggées mais affectées à
    UN seul groupe (une double affectation injecterait dans le fit du mur A un
    bloc structuré du mur B que le RANSAC pourrait préférer)."""
    fwd = cam2w[:, :3, 2]
    fwd_h = fwd - (fwd @ up)[:, None] * up
    e1 = _unit(fwd_h[0])
    e2 = np.cross(up, e1)
    yaw = np.arctan2(fwd_h @ e2, fwd_h @ e1)
    n = len(yaw)

    groups = [[0]]
    for i in range(1, n):
        m = _circ_mean(yaw[groups[-1]])
        if _angdiff(float(yaw[i]), m) > math.radians(YAW_NEW_GROUP_DEG):
            groups.append([i])
        else:
            groups[-1].append(i)
    if len(groups) > 1 and _angdiff(_circ_mean(yaw[groups[-1]]), _circ_mean(yaw[groups[0]])) \
            <= math.radians(YAW_NEW_GROUP_DEG):
        groups[0] = groups[-1] + groups[0]  # la séquence boucle sur le premier mur
        groups.pop()

    # photos "pont" : groupe d'UNE photo à ~équidistance angulaire de ses deux
    # voisins = photo de coin. Elle sert au matching SfM mais son nuage mélange
    # deux murs : on l'exclut du fit (un faux mur diagonal entrerait dans la
    # chaîne sinon).
    bridges = []
    if len(groups) > 2:
        means = [_circ_mean(yaw[g]) for g in groups]
        kept = []
        for k, g in enumerate(groups):
            if len(g) == 1:
                d_prev = _angdiff(means[k], means[(k - 1) % len(groups)])
                d_next = _angdiff(means[k], means[(k + 1) % len(groups)])
                if abs(d_prev - d_next) < math.radians(YAW_AMBIGUOUS_DEG) \
                        and max(d_prev, d_next) < math.radians(65.0):
                    bridges.extend(g)
                    continue
            kept.append(g)
        groups = kept

    # une passe de réaffectation au yaw moyen le plus proche (photos de coin
    # captées par le mauvais groupe pendant la passe gloutonne)
    means = [_circ_mean(yaw[g]) for g in groups]
    if len(groups) > 1:
        reassigned = [[] for _ in groups]
        for i in range(n):
            if i in bridges:
                continue
            k = int(np.argmin([_angdiff(float(yaw[i]), m) for m in means]))
            reassigned[k].append(i)
        if all(reassigned):  # jamais de groupe vidé
            groups = reassigned
        groups.sort(key=lambda g: min(g))
        means = [_circ_mean(yaw[g]) for g in groups]

    ambiguous = []
    if len(groups) > 1:
        for i in range(n):
            d = sorted(_angdiff(float(yaw[i]), m) for m in means)
            if d[1] - d[0] < math.radians(YAW_AMBIGUOUS_DEG):
                ambiguous.append(i)
    if not 3 <= len(groups) <= 8:
        warnings.append(f"{len(groups)} groupes de murs détectés (attendu 3-8) — vérifier plan_debug.png")
    if bridges:
        warnings.append(f"photos de coin exclues du fit (utiles au matching) : {bridges}")

    report["clusters"] = [
        {"wall": k, "photos": g, "yaw_mean_deg": round(math.degrees(means[k]), 1),
         "ambiguous_photos": [i for i in g if i in ambiguous]}
        for k, g in enumerate(groups)
    ]
    report["bridge_photos"] = bridges
    return groups, e1, e2


def _ransac_line(xy, rng):
    """Droite 2D RANSAC + refit orthogonal. Retourne (n2, c, inliers_mask)."""
    best_inl, best_count = None, 0
    for _ in range(LINE_ITERS):
        i, j = rng.choice(len(xy), 2, replace=False)
        ab = xy[j] - xy[i]
        if np.linalg.norm(ab) < LINE_MIN_SPAN_M:
            continue
        t = _unit(ab)
        n2 = np.array([-t[1], t[0]])
        c = -float(np.dot(n2, xy[i]))
        inl = np.abs(xy @ n2 + c) < LINE_INLIER_M
        cnt = int(inl.sum())
        if cnt > best_count:
            best_count, best_inl = cnt, inl
    if best_inl is None or best_count < 30:
        return None, None, None
    Q = xy[best_inl]
    mu = Q.mean(axis=0)
    _, V = np.linalg.eigh(np.cov((Q - mu).T))
    n2 = _unit(V[:, 0])
    c = -float(np.dot(n2, mu))
    return n2, c, best_inl


def fit_wall_line(group, pts_list, confs_list, floor_n, floor_d, hc, origin, e1, e2, rng):
    """Fit d'un plan vertical sur les points des photos d'un mur.
    Retourne (fit dict, xy retenus, xy rejetés) — xy pour le rendu debug."""
    chunks = []
    for i in group:
        pts = pts_list[i]
        conf = confs_list[i]
        H, W = conf.shape
        col = np.arange(pts.shape[0]) % W
        h = pts @ floor_n + floor_d
        keep = (
            (conf.reshape(-1) > CONF_MIN)
            & (col >= CENTER_COLS[0] * W) & (col < CENTER_COLS[1] * W)
            & (h > HEIGHT_BAND[0] * hc) & (h < HEIGHT_BAND[1] * hc)
        )
        chunks.append(pts[keep])
    P = np.concatenate(chunks) if chunks else np.zeros((0, 3))
    P = _subsample(P, MAX_FIT_POINTS, rng)
    if len(P) < 60:
        return None, np.zeros((0, 2)), np.zeros((0, 2))

    xy = np.stack([(P - origin) @ e1, (P - origin) @ e2], axis=1)
    n2, c, inl = _ransac_line(xy, rng)
    if n2 is None:
        return None, np.zeros((0, 2)), xy

    rms = float(np.sqrt(np.mean((xy[inl] @ n2 + c) ** 2)))
    ratio = float(inl.sum()) / len(xy)

    # plan 3D vertical n·x + d = 0 (n2·xy + c = 0 ⇔ n3·X + (c − n3·origin) = 0)
    n3 = n2[0] * e1 + n2[1] * e2
    d3 = float(c - np.dot(n3, origin))

    # extrémités le long de la droite (percentiles anti-traînards), 3D à mi-hauteur
    t_dir = np.array([-n2[1], n2[0]])
    mu = xy[inl].mean(axis=0)
    s = (xy[inl] - mu) @ t_dir
    s0, s1 = np.percentile(s, ENDPOINT_PCT)
    ends2d = [mu + s0 * t_dir, mu + s1 * t_dir]
    mid_h = 0.5 * hc
    ends3d = [origin + p[0] * e1 + p[1] * e2 + mid_h * floor_n for p in ends2d]

    fit = {
        "pparam": [float(n3[0]), float(n3[1]), float(n3[2]), d3],
        "n2": n2, "c": c,
        "ends2d": ends2d,
        "ends3d": [e.tolist() for e in ends3d],
        "n_points": int(len(xy)),
        "n_inliers": int(inl.sum()),
        "inlier_ratio": round(ratio, 3),
        "rms_m": round(rms, 4),
        "length_m": round(float(s1 - s0), 3),
    }
    return fit, xy[inl], xy[~inl]


def build_walls(fits, warnings, report):
    """Chaîne séquentielle cyclique (ordre horaire des groupes = ordre du
    périmètre), liens pre/next réciproques par construction. right_endpoint du
    mur k = extrémité la plus proche du mur suivant."""
    K = len(fits)
    walls = []
    for k, fit in enumerate(fits):
        nxt = fits[(k + 1) % K]
        mid_next = (nxt["ends2d"][0] + nxt["ends2d"][1]) / 2
        d0 = np.linalg.norm(fit["ends2d"][0] - mid_next)
        d1 = np.linalg.norm(fit["ends2d"][1] - mid_next)
        right, left = (0, 1) if d0 < d1 else (1, 0)
        walls.append({
            "index": k,
            "pparam": fit["pparam"],
            "pre": (k - 1) % K,
            "next": (k + 1) % K,
            "left_endpoint": fit["ends3d"][left],
            "right_endpoint": fit["ends3d"][right],
        })

    angles = []
    for k in range(K):
        n_a = np.array(fits[k]["n2"])
        n_b = np.array(fits[(k + 1) % K]["n2"])
        ang = math.degrees(math.acos(min(1.0, abs(float(np.dot(n_a, n_b))))))
        angles.append(round(ang, 1))
        if ang < 30.0:
            warnings.append(f"murs {k} et {(k + 1) % K} quasi-parallèles ({ang:.0f}°) — jonction instable")
    report["consecutive_angles_deg"] = angles
    report["walls"] = [
        {k: v for k, v in fit.items() if k in
         ("n_points", "n_inliers", "inlier_ratio", "rms_m", "length_m")} | {"index": i}
        for i, fit in enumerate(fits)
    ]
    for i, fit in enumerate(fits):
        if fit["inlier_ratio"] < MIN_WALL_INLIER_RATIO:
            warnings.append(f"mur {i} : inlier_ratio {fit['inlier_ratio']} — fit pollué (meubles ?)")
    return walls


def metric_check(walls, floor_pparam, cam_centers):
    """Échelle MASt3R native plausible ? (même schéma que planedust3r_infer)."""
    n = _unit(np.array(floor_pparam[:3], float))
    d = float(floor_pparam[3])
    cams = np.asarray(cam_centers, float)
    result = {"scale_tested": 1.0, "cam_height_m": round(float(np.mean(np.abs(cams @ n + d))), 3),
              "room_diameter_m": None, "passed": False}
    ends = [np.array(p, float) for w in walls for p in (w["left_endpoint"], w["right_endpoint"])]
    if len(ends) >= 2:
        pts = np.stack(ends)
        result["room_diameter_m"] = round(float(np.linalg.norm(pts.max(0) - pts.min(0))), 3)
    ok_h = CAM_HEIGHT_RANGE[0] <= result["cam_height_m"] <= CAM_HEIGHT_RANGE[1]
    ok_d = result["room_diameter_m"] is None or (
        ROOM_DIAMETER_RANGE[0] <= result["room_diameter_m"] <= ROOM_DIAMETER_RANGE[1]
    )
    result["passed"] = bool(ok_h and ok_d)
    return result


def render_debug(path, wall_xy, rejected_xy, fits, cam2w, up, origin, e1, e2):
    """Rendu top-down : points par mur, droites, caméras + visées."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 9))
    cmap = plt.get_cmap("tab10")
    rng = np.random.default_rng(0)
    for k, xy in enumerate(wall_xy):
        pts = _subsample(xy, 3000, rng)
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=2, color=cmap(k % 10), alpha=0.5,
                       label=f"mur {k}")
    rej = np.concatenate([r for r in rejected_xy if len(r)]) if any(len(r) for r in rejected_xy) else None
    if rej is not None:
        rej = _subsample(rej, 4000, rng)
        ax.scatter(rej[:, 0], rej[:, 1], s=1, color="0.85", zorder=0)
    for k, fit in enumerate(fits):
        a, b = fit["ends2d"]
        ax.plot([a[0], b[0]], [a[1], b[1]], color=cmap(k % 10), lw=3)
        m = (a + b) / 2
        ax.annotate(str(k), m, fontsize=14, fontweight="bold", color=cmap(k % 10))
    cams = cam2w[:, :3, 3]
    cxy = np.stack([(cams - origin) @ e1, (cams - origin) @ e2], axis=1)
    fwd = cam2w[:, :3, 2]
    fwd_h = fwd - (fwd @ up)[:, None] * up
    fxy = np.stack([fwd_h @ e1, fwd_h @ e2], axis=1)
    ax.scatter(cxy[:, 0], cxy[:, 1], color="k", s=25, zorder=5)
    ax.quiver(cxy[:, 0], cxy[:, 1], fxy[:, 0], fxy[:, 1], color="k", width=0.003, zorder=5)
    for i, p in enumerate(cxy):
        ax.annotate(str(i), p, fontsize=7, color="k")
    ax.set_aspect("equal")
    ax.grid(True, lw=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("walls debug — top-down (m)")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def analyze_walls(cam2w, pts_list, confs_list, image_names, debug_path=None, seed=42):
    """Orchestrateur complet : poses + pointmaps → (plan_raw, walls_report)."""
    rng = np.random.default_rng(seed)
    warnings = []
    report = {"n_images": len(image_names)}
    cam2w = np.asarray(cam2w, float)
    pts_list = [np.asarray(p, float).reshape(-1, 3) for p in pts_list]
    confs_list = [np.asarray(c, float) for c in confs_list]

    floor_pparam, ceiling_pparam, up, hc = estimate_up_and_floor(
        cam2w, pts_list, confs_list, rng, warnings, report)
    groups, e1, e2 = cluster_by_yaw(cam2w, up, warnings, report)

    cams = cam2w[:, :3, 3]
    centroid = cams.mean(axis=0)
    origin = centroid - (float(np.dot(centroid, up)) + float(floor_pparam[3])) * up

    fits, wall_xy, rejected_xy, kept_groups = [], [], [], []
    for k, group in enumerate(groups):
        fit, xy_in, xy_out = fit_wall_line(
            group, pts_list, confs_list, up, float(floor_pparam[3]), hc, origin, e1, e2, rng)
        if fit is None:
            warnings.append(f"mur {k} (photos {group}) : fit impossible — ignoré")
            rejected_xy.append(xy_out)
            continue
        fits.append(fit)
        wall_xy.append(xy_in)
        rejected_xy.append(xy_out)
        kept_groups.append(group)
    if len(fits) < 3:
        raise ValueError(f"seulement {len(fits)} murs exploitables — polygone impossible")

    walls = build_walls(fits, warnings, report)
    check = metric_check(walls, floor_pparam, cams)
    report["metric_check"] = check
    if not check["passed"]:
        warnings.append(f"échelle MASt3R suspecte : hauteur caméra {check['cam_height_m']} m")

    if debug_path is not None:
        try:
            render_debug(debug_path, wall_xy, rejected_xy, fits, cam2w, up, origin, e1, e2)
        except Exception as e:  # le debug ne doit jamais faire échouer le job
            warnings.append(f"rendu debug échoué : {e}")

    report["warnings"] = warnings
    plan_raw = {
        "global_plane_info": walls,
        "floor_pparam": floor_pparam,
        "ceiling_pparam": ceiling_pparam,
        "cam_centers": cams.tolist(),
        "image_names": list(image_names),
        "scale_mode": "mast3r",
        "scale_factor": 1.0,
        "metric_check": check,
    }
    return plan_raw, report
