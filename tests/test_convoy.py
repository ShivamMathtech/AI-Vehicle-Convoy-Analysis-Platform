from backend.app.analysis.convoy import ConvoyAnalyzer
from backend.app.analysis.motion import direction
from backend.app.interfaces import Det


def mk(i, x, y):
    return Det(i, 7, "truck", .9, [x - 20, y - 10, x + 20, y + 10], (x, y), 800)


def test_moving_trio_forms_persistent_group():
    a = ConvoyAnalyzer(link_dist=150, min_members=3, min_persist=10)
    for f in range(20):
        a.update(f, [mk(1, 100 + 5 * f, 200), mk(2, 180 + 5 * f, 200), mk(3, 260 + 5 * f, 200), mk(4, 900, 700)])
    s = a.summary()
    assert len(s) == 1 and s[0]["member_track_ids"] == [1, 2, 3]
    assert s[0]["direction_consistency"] > .99


def test_image_plane_direction():
    assert direction((0, 0), (10, -10)) == "North-East"
    assert direction((0, 0), (1, 1)) == "Stationary"
