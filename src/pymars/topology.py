# -*- coding: utf-8 -*-
"""
Topology detection helpers.

Contains:
- _detect_bonds, _detect_bonds_pbc : bond detection (based on covalent radii)
- detect_topology : returns bonds array shape (nbonds,2)
- count_graphs : returns number of connected components (graphs)
- get_graph_components : return list of components (lists of atom indices)
"""
from typing import Optional, Tuple, List

import numpy as np
import numba

from .periodic_table import COV_RADII, UFF_MAX_COORDINATION
from .units import AtomicUnits

@numba.njit
def _detect_bonds_pbc(radii, coordinates, cell):
    reciprocal_cell = np.linalg.inv(cell).T
    cell = cell.T
    nat = len(radii)
    bond1 = []
    bond2 = []
    distances = []
    for i in range(nat):
        for j in range(i + 1, nat):
            vec = coordinates[i] - coordinates[j]
            vecpbc = reciprocal_cell @ vec
            vecpbc -= np.round(vecpbc)
            vec = cell @ vecpbc
            dist = np.linalg.norm(vec)
            if dist < radii[i] + radii[j] + 0.4 and dist > 0.4:
                bond1.append(i)
                bond2.append(j)
                distances.append(dist)
    return bond1, bond2, distances

@numba.njit
def _detect_bonds(radii, coordinates):
    nat = len(radii)
    bond1 = []
    bond2 = []
    distances = []
    for i in range(nat):
        for j in range(i + 1, nat):
            vec = coordinates[i] - coordinates[j]
            dist = np.linalg.norm(vec)
            if dist < radii[i] + radii[j] + 0.4 and dist > 0.4:
                bond1.append(i)
                bond2.append(j)
                distances.append(dist)
    return bond1, bond2, distances

def detect_topology(species: np.ndarray, coordinates: np.ndarray, cell: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Detects the topology (bond list) of a system based on species and coordinates.
    Returns a np.ndarray of shape [nbonds,2] with the two atom indices for each bond.

    species : array-like int atomic numbers (or indices expected by periodic_table)
    coordinates : (N,3) array in Angstrom
    cell : optional 3x3 array in Angstrom (for PBC)
    """
    radii = (COV_RADII * AtomicUnits.ANG)[species]
    max_coord = UFF_MAX_COORDINATION[species]

    if cell is not None:
        bond1, bond2, distances = _detect_bonds_pbc(radii, coordinates, cell)
    else:
        bond1, bond2, distances = _detect_bonds(radii, coordinates)

    bond1 = np.array(bond1, dtype=np.int32)
    bond2 = np.array(bond2, dtype=np.int32)
    bonds = np.stack((bond1, bond2), axis=1) if bond1.size > 0 else np.zeros((0, 2), dtype=np.int32)

    # coordination counts
    coord = np.zeros(len(species), dtype=np.int32)
    if bonds.shape[0] > 0:
        np.add.at(coord, bonds[:, 0], 1)
        np.add.at(coord, bonds[:, 1], 1)

    if np.all(coord <= max_coord):
        return bonds

    # otherwise filter by distance/coordination heuristics
    distances = np.array(distances, dtype=np.float32)
    radiibonds = radii[bonds]
    req = radiibonds.sum(axis=1)
    rminbonds = radiibonds.min(axis=1)
    sorted_indices = np.lexsort((-distances / req, rminbonds))

    bonds = bonds[sorted_indices, :]
    distances = distances[sorted_indices]

    true_bonds: List[Tuple[int, int]] = []
    for ibond in range(bonds.shape[0]):
        i, j = bonds[ibond]
        ci, cj = coord[i], coord[j]
        mci, mcj = max_coord[i], max_coord[j]
        if ci <= mci and cj <= mcj:
            true_bonds.append((i, j))
        else:
            coord[i] -= 1
            coord[j] -= 1

    if len(true_bonds) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    true_bonds = np.array(true_bonds, dtype=np.int32)
    sorted_indices = np.lexsort((true_bonds[:, 1], true_bonds[:, 0]))
    true_bonds = true_bonds[sorted_indices, :]

    return true_bonds

def count_graphs(species: np.ndarray, coordinates: np.ndarray, cell: Optional[np.ndarray] = None) -> int:
    """
    Count connected components (graphs) in the molecular topology.

    Returns:
        int : number of connected components (1 means single connected molecule)
    """
    bonds = detect_topology(species, coordinates, cell)
    n = len(species)
    if bonds.size == 0:
        return n  # every atom is isolated

    # union-find (disjoint set)
    parent = np.arange(n, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for (i, j) in bonds:
        union(int(i), int(j))

    # count unique roots
    roots = set()
    for i in range(n):
        roots.add(find(i))
    return len(roots)

def get_graph_components(species: np.ndarray, coordinates: np.ndarray, cell: Optional[np.ndarray] = None) -> List[np.ndarray]:
    """
    Return a list of components; each component is an np.ndarray of atom indices.
    """
    bonds = detect_topology(species, coordinates, cell)
    n = len(species)
    if bonds.size == 0:
        return [np.array([i], dtype=np.int32) for i in range(n)]

    parent = np.arange(n, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for (i, j) in bonds:
        union(int(i), int(j))

    comps = {}
    for i in range(n):
        r = find(i)
        comps.setdefault(r, []).append(i)
    return [np.array(c, dtype=np.int32) for c in comps.values()]