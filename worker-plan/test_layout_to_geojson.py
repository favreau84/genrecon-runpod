# Tests locaux (sans GPU) du post-traitement layout_to_geojson.
# python3 worker-plan/test_layout_to_geojson.py   (venv avec numpy + shapely)
#
# Fixtures dans le repère DUSt3R de plane_merge : Y = axe vertical
# (les normales sol/plafond y sont filtrées sur |n_y| > 0.95).

from layout_to_geojson import layout_to_geojson


def rect_room(scale_factor=1.0, inverted=False, open_chain=False):
    """Pièce rectangulaire 4×3 m, plafond à 2,5 m, murs chaînés A→B→C→D."""
    walls = [
        # A : z=0, de (0,·,0) à (4,·,0)
        {"index": 0, "pparam": [0, 0, 1, 0], "pre": 3, "next": 1,
         "left_endpoint": [0, 1, 0], "right_endpoint": [4, 1, 0]},
        # B : x=4
        {"index": 1, "pparam": [1, 0, 0, -4], "pre": 0, "next": 2,
         "left_endpoint": [4, 1, 0], "right_endpoint": [4, 1, 3]},
        # C : z=3
        {"index": 2, "pparam": [0, 0, 1, -3], "pre": 1, "next": 3,
         "left_endpoint": [4, 1, 3], "right_endpoint": [0, 1, 3]},
        # D : x=0
        {"index": 3, "pparam": [1, 0, 0, 0], "pre": 2, "next": 0,
         "left_endpoint": [0, 1, 3], "right_endpoint": [0, 1, 0]},
    ]
    if open_chain:  # pièce filmée sans le mur D
        walls = walls[:3]
        walls[0]["pre"] = None
        walls[2]["next"] = None
    floor = [0, 1, 0, 0]          # y = 0
    ceiling = [0, -1, 0, 2.5]     # y = 2.5
    if inverted:
        floor = [0, -1, 0, 0]
        ceiling = [0, 1, 0, -2.5]
    return {
        "global_plane_info": walls,
        "floor_pparam": floor,
        "ceiling_pparam": ceiling,
        "cam_centers": [[2.0, 1.5, 1.5], [1.2, 1.4, 1.0], [2.8, 1.6, 2.0]],
        "image_names": ["f1.jpg", "f2.jpg", "f3.jpg"],
        "scale_mode": "metric",
        "scale_factor": scale_factor,
    }


def room_props(geo):
    room = geo["features"][0]
    assert room["properties"]["kind"] == "room"
    return room["properties"], room["geometry"]["coordinates"][0]


def close(a, b, tol=0.02):
    assert abs(a - b) < tol, f"{a} != {b} (±{tol})"


def test_rect_closed():
    geo = layout_to_geojson(rect_room(), job_id="test")
    props, ring = room_props(geo)
    close(props["area_m2"], 12.0)
    close(props["perimeter_m"], 14.0)
    close(props["ceiling_height_m"], 2.5)
    assert props["closed"] is True
    assert props["n_walls"] == 4
    assert ring[0] == ring[-1], "anneau GeoJSON non fermé"
    walls = [f for f in geo["features"] if f["properties"]["kind"] == "wall"]
    assert len(walls) == 4
    assert not any(w["properties"]["inferred"] for w in walls)
    lengths = sorted(w["properties"]["length_m"] for w in walls)
    close(lengths[0], 3.0), close(lengths[-1], 4.0)
    print("✓ pièce rectangulaire fermée 4×3")


def test_open_chain():
    geo = layout_to_geojson(rect_room(open_chain=True))
    props, _ = room_props(geo)
    close(props["area_m2"], 12.0)
    assert props["closed"] is False
    assert any("partiellement" in w for w in props["warnings"])
    walls = [f for f in geo["features"] if f["properties"]["kind"] == "wall"]
    inferred = [w for w in walls if w["properties"]["inferred"]]
    assert len(inferred) == 1, f"1 mur déduit attendu, trouvé {len(inferred)}"
    close(inferred[0]["properties"]["length_m"], 3.0)  # le mur D manquant
    print("✓ chaîne ouverte fermée par un segment déduit")


def test_inverted_normals():
    geo = layout_to_geojson(rect_room(inverted=True))
    props, _ = room_props(geo)
    close(props["area_m2"], 12.0)
    close(props["ceiling_height_m"], 2.5)
    print("✓ normales sol/plafond inversées")


def test_scale_factor():
    geo = layout_to_geojson(rect_room(scale_factor=2.0))
    props, _ = room_props(geo)
    close(props["area_m2"], 48.0, tol=0.1)
    close(props["ceiling_height_m"], 5.0)
    assert any("suspecte" in w for w in props["warnings"])
    print("✓ facteur d'échelle appliqué (aire ×4, warning hauteur)")


def test_ccw_orientation():
    geo = layout_to_geojson(rect_room())
    _, ring = room_props(geo)
    n = len(ring) - 1
    signed = sum(ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1] for i in range(n)) / 2
    assert signed > 0, "polygone non CCW"
    print("✓ orientation CCW")


def skewed_room(deg=4.0):
    """Rectangle 4×3 dont les murs A et C sont tournés de ±deg° : l'axe dominant
    doit les redresser en angles droits."""
    import math
    room = rect_room()
    for idx, sign in ((0, 1), (2, -1)):
        a = math.radians(sign * deg)
        # normale [0,0,1] tournée de a autour de Y (axe vertical de la fixture)
        n = [math.sin(a), 0.0, math.cos(a)]
        w = room["global_plane_info"][idx]
        # le plan passe toujours par le milieu du mur d'origine
        mid = [(l + r) / 2 for l, r in zip(w["left_endpoint"], w["right_endpoint"])]
        d = -(n[0] * mid[0] + n[1] * mid[1] + n[2] * mid[2])
        w["pparam"] = n + [d]
    return room


def test_orthogonalize():
    geo = layout_to_geojson(skewed_room())
    props, ring = room_props(geo)
    close(props["area_m2"], 12.0, tol=0.3)
    assert any("redressé" in w for w in props["warnings"]), props["warnings"]
    for i in range(len(ring) - 1):
        a, b = ring[i], ring[i + 1]
        c = ring[(i + 2) % (len(ring) - 1)] if i + 2 <= len(ring) - 1 else ring[1]
        v1 = [b[0] - a[0], b[1] - a[1]]
        v2 = [c[0] - b[0], c[1] - b[1]]
        dot = abs(v1[0] * v2[0] + v1[1] * v2[1])
        n1 = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
        n2 = (v2[0] ** 2 + v2[1] ** 2) ** 0.5
        assert dot / (n1 * n2) < 0.02, f"angle non droit au sommet {i + 1}"
    # sans ortho : les murs restent penchés
    room = skewed_room()
    room["ortho"] = False
    geo2 = layout_to_geojson(room)
    _, ring2 = room_props(geo2)
    v1 = [ring2[1][0] - ring2[0][0], ring2[1][1] - ring2[0][1]]
    v2 = [ring2[2][0] - ring2[1][0], ring2[2][1] - ring2[1][1]]
    dot = abs(v1[0] * v2[0] + v1[1] * v2[1])
    n = ((v1[0] ** 2 + v1[1] ** 2) * (v2[0] ** 2 + v2[1] ** 2)) ** 0.5
    assert dot / n > 0.03, "ortho=False devrait conserver l'obliquité"
    print("✓ orthogonalisation (angles droits, aire conservée, désactivable)")


def grid_points(x_range, y_range, z_range, n=6):
    pts = []
    for i in range(n):
        for j in range(n):
            pts.append([
                x_range[0] + (x_range[1] - x_range[0]) * i / (n - 1),
                y_range[0] + (y_range[1] - y_range[0]) * j / (n - 1),
                z_range[0] + (z_range[1] - z_range[0]) * (i + j) / (2 * n - 2),
            ])
    return pts


def test_openings():
    room = rect_room()
    room["openings_raw"] = [
        # porte sur le mur A (z=0) : x ∈ [1.0, 1.9], sol → 2.05 m — vue 2 fois
        {"label": "door", "score": 0.8, "img_id": 0,
         "points": grid_points((1.0, 1.9), (0.02, 2.05), (0.0, 0.02))},
        {"label": "door", "score": 0.6, "img_id": 1,
         "points": grid_points((1.05, 1.92), (0.0, 2.0), (0.0, 0.02))},
        # fenêtre sur le mur B (x=4) : z ∈ [1.0, 2.2], allège 0.9 m
        {"label": "window", "score": 0.7, "img_id": 2,
         "points": [[4.0, y, z] for y in (0.9, 1.5, 2.1, 2.25) for z in
                    (1.0, 1.3, 1.6, 1.9, 2.2)]},
        # bruit : trop loin de tout mur → ignoré
        {"label": "window", "score": 0.9, "img_id": 3,
         "points": grid_points((1.8, 2.4), (0.5, 1.5), (1.2, 1.6))},
    ]
    geo = layout_to_geojson(room)
    props, _ = room_props(geo)
    assert props["n_doors"] == 1 and props["n_windows"] == 1, (props["n_doors"], props["n_windows"])
    openings = [f for f in geo["features"] if f["properties"]["kind"] == "opening"]
    assert len(openings) == 2
    door = next(o for o in openings if o["properties"]["opening_type"] == "door")
    win = next(o for o in openings if o["properties"]["opening_type"] == "window")
    close(door["properties"]["width_m"], 0.9, tol=0.15)
    assert door["properties"]["n_views"] == 2
    assert door["properties"]["sill_height_m"] < 0.3
    close(win["properties"]["width_m"], 1.2, tol=0.15)
    close(win["properties"]["sill_height_m"], 0.9, tol=0.2)
    print("✓ ouvertures (porte fusionnée 2 vues, fenêtre avec allège, bruit ignoré)")


def test_opening_raycast():
    """La box 2D d'une porte est construite par projection directe (pose et
    focale connues) ; le lancer de rayons doit retrouver le rectangle exact,
    même si les points 3D ne couvrent qu'une bande partielle de la porte."""
    import numpy as np
    room = rect_room()
    # caméra OpenCV (x droite, y bas, z avant) face au mur A (z=0), Y monde = haut
    C = np.array([1.45, 1.0, 3.2])
    R = np.diag([1.0, -1.0, -1.0])  # colonnes : droite, bas, avant=(0,0,-1)
    f, gw, gh = 300.0, 512, 288
    W, H = 1280, 720
    door = {"x": (1.0, 1.9), "y": (0.0, 2.05)}  # rectangle réel sur le mur z=0

    def project(xw, yw):
        pc = R.T @ (np.array([xw, yw, 0.0]) - C)
        return gw / 2 + f * pc[0] / pc[2], gh / 2 + f * pc[1] / pc[2]

    us, vs = zip(*[project(x, y) for x in door["x"] for y in door["y"]])
    box = [min(us) * W / gw, min(vs) * H / gh, max(us) * W / gw, max(vs) * H / gh]

    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = C
    room["poses"] = [pose.tolist()] * 3
    room["focals"] = [f] * 3
    room["dust3r_size"] = [gw, gh]
    room["openings_raw"] = [{
        "label": "door", "score": 0.9, "img_id": 0,
        "box": box, "img_size": [W, H],
        # points volontairement restreints à une bande (masque de confiance)
        "points": grid_points((1.2, 1.7), (0.9, 1.3), (0.0, 0.02)),
    }]
    geo = layout_to_geojson(room)
    props, _ = room_props(geo)
    assert props["n_doors"] == 1, props
    door_f = next(f for f in geo["features"] if f["properties"]["kind"] == "opening")
    close(door_f["properties"]["width_m"], 0.9, tol=0.05)
    assert door_f["properties"]["sill_height_m"] < 0.1
    close(door_f["properties"]["head_height_m"], 2.05, tol=0.1)
    print("✓ ray-cast : rectangle exact retrouvé depuis la box (bande de points partielle)")


if __name__ == "__main__":
    test_rect_closed()
    test_open_chain()
    test_inverted_normals()
    test_scale_factor()
    test_ccw_orientation()
    test_orthogonalize()
    test_openings()
    test_opening_raycast()
    print("Tous les tests passent.")
