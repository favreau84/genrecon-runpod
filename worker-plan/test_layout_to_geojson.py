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


if __name__ == "__main__":
    test_rect_closed()
    test_open_chain()
    test_inverted_normals()
    test_scale_factor()
    test_ccw_orientation()
    print("Tous les tests passent.")
