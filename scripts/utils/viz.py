"""RViz visualization for the inference node (ROS-dependent).

``Visualizer`` owns all marker construction, split across two MarkerArray topics:
  - ``/nmp/viz/traj``  — trajectory line, waypoints, sub-goal, goal, ego (from inference)
  - ``/nmp/viz/lidar`` — sensing sphere, FOV circles, LiDAR hits (from every cloud frame)

Frame transforms use ``neural_minco.geometry``.
"""

import numpy as np
import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from neural_minco.geometry import body_to_world_numpy

TRAJ_TOPIC = "/nmp/viz/traj"
LIDAR_TOPIC = "/nmp/viz/lidar"

# Warm palette (trajectory kind)
TRAJ_COLOR = (1.0, 0.45, 0.0)
Q_COLOR = (0.0, 0.85, 0.90)
P_END_COLOR = (1.0, 0.10, 0.10)
GOAL_COLOR = (0.90, 0.15, 0.90)
EGO_COLOR = (1.0, 1.0, 1.0)

# Cool / muted palette (lidar kind)
SPHERE_COLOR = (0.40, 0.42, 0.50)
FOV_COLOR = (1.0, 0.82, 0.0)

FRAME_ID = "world"
NS_TRAJ = "nmp_traj"
NS_LIDAR = "nmp_lidar"
LIFETIME = 0.5


class Visualizer:
    """Publish per-frame RViz markers: trajectory (from inference) + lidar sphere/hits (every cloud)."""

    def __init__(self, r_max, fov_min_rad, fov_max_rad):
        self.r_max = r_max
        self.fov_min_rad = fov_min_rad
        self.fov_max_rad = fov_max_rad
        self.traj_pub = rospy.Publisher(TRAJ_TOPIC, MarkerArray, queue_size=1)
        self.lidar_pub = rospy.Publisher(LIDAR_TOPIC, MarkerArray, queue_size=1)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def publish_lidar(self, pts_body, ranges, p, quat, stamp):
        """LiDAR hits as filled 2° cells on the sensing sphere (TRIANGLE_LIST).

        Called on *every* cloud frame, independently of inference.
        pts_body / ranges are in body frame (from range-image backproject).
        The sphere and FOV circles are rotated to the drone's body attitude.
        """
        ma = MarkerArray()
        ma.markers.append(self._sphere_marker(p, quat, stamp, 0))
        ma.markers.append(self._circle_marker(p, quat, stamp, 1, self.fov_min_rad))
        ma.markers.append(self._circle_marker(p, quat, stamp, 2, self.fov_max_rad))
        if len(pts_body):
            ma.markers.append(self._cells_marker(pts_body, ranges, p, quat, stamp, 4))
        self.lidar_pub.publish(ma)

    def publish_traj(self, traj_body, Q, p_end, goal_body, p, quat, stamp):
        """Publish trajectory-kind markers: line, waypoints, endpoint, goal, ego.

        Called when inference produces a new trajectory.
        """
        traj_w = body_to_world_numpy(traj_body, p, quat)
        Q_w = body_to_world_numpy(Q, p, quat)
        p_end_w = body_to_world_numpy(p_end[None], p, quat)[0]
        goal_w = body_to_world_numpy(goal_body[None], p, quat)[0]
        now = rospy.Time.now()

        ma = MarkerArray()
        ma.markers.append(self._traj_marker(traj_w, stamp, 0))
        ma.markers.append(self._spheres_marker(Q_w, stamp, 1, Q_COLOR, 0.20))
        ma.markers.append(self._spheres_marker(p_end_w[None], stamp, 2, P_END_COLOR, 0.25))
        ma.markers.append(self._spheres_marker(goal_w[None], stamp, 3, GOAL_COLOR, 0.25))
        ma.markers.append(self._spheres_marker(p[None], now, 4, EGO_COLOR, 0.30))
        self.traj_pub.publish(ma)

    # ------------------------------------------------------------------
    #  Marker builders
    # ------------------------------------------------------------------

    def _base(self, stamp, id, ns, mtype, scale):
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = FRAME_ID
        m.ns = ns
        m.id = id
        m.type = mtype
        m.action = Marker.ADD
        s = max(float(scale), 1e-6)
        m.scale.x = s
        m.scale.y = s
        m.scale.z = s
        m.pose.orientation.w = 1.0
        m.lifetime = rospy.Duration(LIFETIME)
        return m

    def _traj_marker(self, pts, stamp, id):
        m = self._base(stamp, id, NS_TRAJ, Marker.LINE_STRIP, 0.12)
        for pt in pts:
            m.points.append(Point(x=pt[0], y=pt[1], z=pt[2]))
        m.colors = [ColorRGBA(*TRAJ_COLOR, 1.0)] * len(pts)
        return m

    def _spheres_marker(self, pts, stamp, id, color, scale):
        m = self._base(stamp, id, NS_TRAJ, Marker.SPHERE_LIST, scale)
        m.scale.y = m.scale.z = scale
        c = ColorRGBA(*color, 1.0)
        for pt in pts:
            m.points.append(Point(x=float(pt[0]), y=float(pt[1]), z=float(pt[2])))
            m.colors.append(c)
        return m

    def _sphere_marker(self, p, quat, stamp, id, color=SPHERE_COLOR, alpha=0.20, width=0.03, deg_step=2.0):
        """Wireframe sensing-range sphere at 2° resolution, rotated to the drone's body attitude."""
        r = self.r_max
        n_seg = int(round(360.0 / deg_step))

        # Build vertices in body-aligned frame (north pole = +Z_body, equator = XY_body).
        lat_el = np.deg2rad(np.arange(-90.0 + deg_step, 90.0, deg_step))
        a = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
        lat_circ = np.stack(
            [
                r * np.cos(lat_el)[:, None] * np.cos(a),
                r * np.cos(lat_el)[:, None] * np.sin(a),
                np.full((len(lat_el), n_seg), r * np.sin(lat_el)[:, None]),
            ],
            axis=-1,
        )
        lat_segs = np.concatenate([_polyline_segments(c, closed=True) for c in lat_circ], axis=0)

        lon_az = np.deg2rad(np.arange(0.0, 360.0, deg_step))
        el = np.linspace(-np.pi / 2, np.pi / 2, n_seg // 2 + 1)
        lon_mer = np.stack(
            [
                r * np.cos(el)[None, :] * np.cos(lon_az)[:, None],
                r * np.cos(el)[None, :] * np.sin(lon_az)[:, None],
                np.full((len(lon_az), len(el)), r * np.sin(el)[None, :]),
            ],
            axis=-1,
        )
        lon_segs = np.concatenate([_polyline_segments(m, closed=False) for m in lon_mer], axis=0)

        seg_pts = np.concatenate([lat_segs, lon_segs], axis=0)
        # Rotate body-aligned → world, then translate to drone position.
        seg_pts_w = body_to_world_numpy(seg_pts, p, quat)

        m = self._base(stamp, id, NS_LIDAR, Marker.LINE_LIST, width)
        c = ColorRGBA(*color, alpha)
        m.points.extend(Point(x=float(x), y=float(y), z=float(z)) for x, y, z in seg_pts_w)
        m.colors.extend([c] * len(seg_pts_w))
        return m

    def _circle_marker(self, p, quat, stamp, id, elevation, color=FOV_COLOR, alpha=0.9, width=0.08, n_seg=96):
        """Latitude circle at the given elevation (FOV boundary), rotated to drone attitude."""
        a = np.linspace(0, 2 * np.pi, n_seg)
        pts_body = np.stack(
            [
                self.r_max * np.cos(elevation) * np.cos(a),
                self.r_max * np.cos(elevation) * np.sin(a),
                np.full_like(a, self.r_max * np.sin(elevation)),
            ],
            axis=1,
        )
        pts_w = body_to_world_numpy(pts_body, p, quat)
        m = self._base(stamp, id, NS_LIDAR, Marker.LINE_STRIP, width)
        for pt in pts_w:
            m.points.append(Point(x=float(pt[0]), y=float(pt[1]), z=float(pt[2])))
        m.points.append(m.points[0])
        m.colors = [ColorRGBA(*color, alpha)] * len(m.points)
        return m

    def _cells_marker(self, pts_body, ranges, p, quat, stamp, id, deg=2.0):
        """Each back-projected hit → one filled spherical quad (TRIANGLE_LIST, world frame).

        The quad spans the 2°×2° angular bin of the range image, placed at the measured
        range, so cells tile the sensing sphere between the wireframe lines.
        """
        if len(pts_body) == 0:
            m = self._base(stamp, id, NS_LIDAR, Marker.TRIANGLE_LIST, 1.0)
            return m

        half = np.radians(deg / 2.0)
        x, y, z = pts_body[:, 0], pts_body[:, 1], pts_body[:, 2]
        az = np.arctan2(y, x)
        el = np.arcsin(np.clip(z / np.maximum(ranges, 1e-9), -1.0, 1.0))
        az0, az1 = az - half, az + half
        el0, el1 = np.maximum(el - half, -np.pi / 2 + 0.01), np.minimum(el + half, np.pi / 2 - 0.01)
        r = ranges
        ca0, ca1 = np.cos(az0), np.cos(az1)
        sa0, sa1 = np.sin(az0), np.sin(az1)
        ce0, ce1 = np.cos(el0), np.cos(el1)
        se0, se1 = np.sin(el0), np.sin(el1)

        # Four corners of each cell (body frame)
        v00 = np.stack([r * ce0 * ca0, r * ce0 * sa0, r * se0], axis=1)
        v10 = np.stack([r * ce0 * ca1, r * ce0 * sa1, r * se0], axis=1)
        v11 = np.stack([r * ce1 * ca1, r * ce1 * sa1, r * se1], axis=1)
        v01 = np.stack([r * ce1 * ca0, r * ce1 * sa0, r * se1], axis=1)

        # Two triangles per cell (N*2 triangles, N*6 vertices)
        n = len(ranges)
        tris = np.empty((n * 2, 3, 3), dtype=np.float64)
        tris[0::2, 0] = v00
        tris[0::2, 1] = v10
        tris[0::2, 2] = v11
        tris[1::2, 0] = v00
        tris[1::2, 1] = v11
        tris[1::2, 2] = v01
        tris_w = body_to_world_numpy(tris.reshape(-1, 3), p, quat)

        m = self._base(stamp, id, NS_LIDAR, Marker.TRIANGLE_LIST, 1.0)
        cols = _range_colors(ranges, self.r_max)
        # Each cell → 2 tris × 3 verts = 6 colour entries
        m.colors = [c for c in cols for _ in range(6)]
        m.points = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in tris_w]
        return m


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _polyline_segments(pts, closed=False):
    n = len(pts)
    seg = np.stack([pts[: n - 1], pts[1:n]], axis=1).reshape(-1, 3)
    if closed:
        seg = np.concatenate([seg, pts[[n - 1, 0]]], axis=0)
    return seg


def _range_colors(ranges, r_max):
    """Green (near) → blue (far)."""
    t = np.clip(ranges / r_max, 0.0, 1.0)
    return [ColorRGBA(0.1, float(1.0 - 0.8 * ti), float(0.1 + 0.9 * ti), 1.0) for ti in t]
