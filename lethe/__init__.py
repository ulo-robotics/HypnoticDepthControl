"""Lethe: closed-loop propofol control with a learned S4D policy on a physics compensator."""
import os

CACHE = os.environ.get("LETHE_CACHE") or os.path.expanduser("~/.cache/lethe")     # compiled kernels, engine decisions, XLA caches
