import itertools

import numpy as np

from ..interfaces import BaseAnalyzer
from .motion import direction


class ConvoyAnalyzer(BaseAnalyzer):
    """Graph grouping: edge = centroid distance <= link_dist AND compatible motion.
    Groups = connected components (>= min_members), tracked across frames by member overlap,
    and reported only after persisting >= min_persist processed frames."""

    def __init__(self, link_dist, cos_min=0.5, min_members=3, min_persist=15, vel_window=10, fps=25.0):
        self.d, self.c, self.m, self.p, self.w, self.fps = link_dist, cos_min, min_members, min_persist, vel_window, fps
        self.traj, self.conf, self.groups, self._prev, self._next = {}, {}, {}, {}, 1

    def _vel(self, i):
        t = self.traj[i][-self.w:]
        return np.array([t[-1][1] - t[0][1], t[-1][2] - t[0][2]]) if len(t) > 1 else np.zeros(2)

    def _compatible(self, a, b, eps=2.0):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < eps and nb < eps:
            return True
        if na < eps or nb < eps:
            return False
        return float(a @ b / (na * nb)) >= self.c

    def _match(self, members, used):
        best, score = None, 0.5
        for gid, prev in self._prev.items():
            j = len(members & prev) / len(members | prev)
            if gid not in used and j >= score:
                best, score = gid, j
        if best is None:
            best, self._next = self._next, self._next + 1
        used.add(best)
        return best

    def update(self, fid, dets):
        for t in dets:
            self.traj.setdefault(t.track_id, []).append((fid, *t.centroid))
            self.conf.setdefault(t.track_id, []).append(t.conf)
        pos = {t.track_id: np.array(t.centroid) for t in dets}
        vel = {i: self._vel(i) for i in pos}
        parent = {i: i for i in pos}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in itertools.combinations(pos, 2):
            if np.linalg.norm(pos[a] - pos[b]) <= self.d and self._compatible(vel[a], vel[b]):
                parent[find(a)] = find(b)
        comps = {}
        for i in pos:
            comps.setdefault(find(i), set()).add(i)

        out, used, prev = [], set(), {}
        for mem in comps.values():
            if len(mem) < self.m:
                continue
            gid = self._match(mem, used)
            pts = np.array([pos[i] for i in mem])
            d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
            np.fill_diagonal(d, np.inf)
            spacing = float(d.min(axis=1).mean())
            u = [vel[i] / np.linalg.norm(vel[i]) for i in mem if np.linalg.norm(vel[i]) > 2.0]
            dcons = float(np.linalg.norm(np.sum(u, axis=0)) / len(u)) if u else 1.0
            mv = np.mean([vel[i] for i in mem], axis=0)
            g = self.groups.setdefault(gid, {"first": fid, "members": set(), "frames": 0, "spacing": [], "dc": []})
            g["members"] |= mem
            g["frames"] += 1
            g["last"] = fid
            g["spacing"].append(spacing)
            g["dc"].append(dcons)
            prev[gid] = mem
            out.append({"group_id": gid, "members": sorted(mem), "spacing": round(spacing, 1),
                        "direction_consistency": round(dcons, 3), "direction": direction((0, 0), mv),
                        "confirmed": g["frames"] >= self.p})
        self._prev = prev
        return out

    def summary(self):
        res = []
        for gid, g in self.groups.items():
            if g["frames"] < self.p:
                continue
            dc = float(np.mean(g["dc"]))
            mc = float(np.mean([np.mean(self.conf[i]) for i in g["members"]]))
            res.append({"group_id": gid, "member_track_ids": sorted(g["members"]),
                        "vehicle_count": len(g["members"]), "frames": g["frames"],
                        "duration_s": round((g["last"] - g["first"] + 1) / self.fps, 2),
                        "mean_spacing_px": round(float(np.mean(g["spacing"])), 1),
                        "spacing_variance": round(float(np.var(g["spacing"])), 1),
                        "direction_consistency": round(dc, 3), "group_confidence": round(mc * dc, 3)})
        return res
