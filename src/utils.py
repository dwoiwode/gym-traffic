import numpy as np


def kmh_to_ms(kmh):
    return kmh / 3.6


def ms_to_kmh(ms):
    return ms * 3.6


def zero_to_hundert_in_ms2(s):
    return kmh_to_ms(100) / s


def distance_euler(p, q):
    """Helper function to compute distance between two points."""
    return np.linalg.norm(p - q)


def distance_manhattan(p, q):
    """Helper function to compute distance between two points."""
    return np.sum(np.abs(p - q))
