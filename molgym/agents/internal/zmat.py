from typing import List

import numpy as np


def get_distance(p_i: np.ndarray, p_j: np.ndarray) -> float:
    """
    Compute distance between points i and j

    :param p_i: point i
    :param p_j: point j
    :return: distance
    """
    return np.sqrt(np.sum(np.square(p_i - p_j)))


def get_angle(p_i: np.ndarray, p_j: np.ndarray, p_k: np.ndarray) -> float:
    """
    Compute angle between points i, j, and k

    :param p_i: point i
    :param p_j: point j
    :param p_k: point k
    :return: angle in radians
    """
    rij = p_i - p_j
    rkj = p_k - p_j

    sin_theta = np.linalg.norm(np.cross(rij, rkj))
    cos_theta = np.dot(rij, rkj)
    return np.arctan2(sin_theta, cos_theta)


def get_dihedral(p_i: np.ndarray, p_j: np.ndarray, p_k: np.ndarray, p_l: np.ndarray) -> float:
    """
    Return dihedral between points i, j, k, and l.

    :param p_i: point i
    :param p_j: point j
    :param p_k: point k
    :param p_l: point l
    :return: dihedral angle in radians
    """
    r_ji = p_j - p_i
    r_kj = p_k - p_j
    r_lk = p_l - p_k

    v1 = np.cross(r_ji, r_kj)
    v1 = v1 / np.linalg.norm(v1)

    v2 = np.cross(r_lk, r_kj)
    v2 = v2 / np.linalg.norm(v2)

    m1 = np.cross(v1, r_kj) / np.linalg.norm(r_kj)

    x = np.dot(v1, v2)
    y = np.dot(m1, v2)

    psi = np.arctan2(y, x)
    if psi < 0:
        return -psi - np.pi
    else:
        return np.pi - psi


def position_point(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, distance: float, angle: float,
                   dihedral: float) -> np.ndarray:
    """
    Determine point p in space that is:
        - <distance> far from p2
        - <angle> between p2 and p1
        - <dihedral> between p2, p1, and p0

    :param p0: position for dihedral
    :param p1: position for angle
    :param p2: position for distance
    :param distance: distance between p and v2
    :param angle: angle between p, p2 and p1
    :param dihedral: dihedral angle between p, p2, p1, and p0
    :return: coordinates of p
    """
    x = distance * np.cos(angle)
    y = distance * np.cos(dihedral) * np.sin(angle)
    z = distance * np.sin(dihedral) * np.sin(angle)

    v_a = p1 - p0

    v_b = p2 - p1
    v_b = v_b / np.linalg.norm(v_b)

    c_ab = np.cross(v_a, v_b)
    c_ab = c_ab / np.linalg.norm(c_ab)

    c_ab_b = np.cross(c_ab, v_b)

    return p2 - v_b * x + c_ab_b * y + c_ab * z


def position_atom_helper(
    positions: List[np.ndarray],
    focus: int,
    distance: float,
    angle: float,
    dihedral: float,
) -> np.ndarray:
    if focus > len(positions):
        raise RuntimeError('Focus greater than number of atoms')

    if len(positions) == 0:
        return np.array([0, 0, 0], dtype=np.float)

    focus = positions[focus]
    sorted_positions = sorted(positions, key=lambda p: get_distance(p, focus))

    p_aux_1 = np.array([1, 0, 0], dtype=np.float)
    p_aux_0 = np.array([0, 1, 0], dtype=np.float)

    if len(positions) == 1:
        p2 = sorted_positions[0]
        p1 = p2 + p_aux_1
        p0 = p2 + p_aux_0

    elif len(positions) == 2:
        p2 = sorted_positions[0]
        p1 = sorted_positions[1]
        p0 = p2 + p1 + p_aux_0 + p_aux_1

    else:
        p2 = sorted_positions[0]
        p1 = sorted_positions[1]
        p0 = sorted_positions[2]

    return position_point(p0, p1, p2, distance=distance, angle=angle, dihedral=dihedral)
