"""Shared non-ROS helper modules for scripts/ (imported as `from utils.<mod> import ...`).

`scripts/` is on `sys.path` at runtime, so `utils` resolves here. ROS-bound helpers
(e.g. `viz.Visualizer`) are fine in this package — they are only imported by ROS scripts.
"""
