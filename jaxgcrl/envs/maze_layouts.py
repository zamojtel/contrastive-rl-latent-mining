"""Custom occupancy grids for the latent-mining experiments.

Coordinate convention:
    grid[row, col]

Cell values:
    0 -- free space
    1 -- wall
"""

import numpy as np


FREE = 0
WALL = 1
MAZE_SIZE = 11


# M0: open arena.
M0 = np.full(
    (MAZE_SIZE, MAZE_SIZE),
    FREE,
    dtype=np.int8,
)

M0[0, :] = WALL
M0[-1, :] = WALL
M0[:, 0] = WALL
M0[:, -1] = WALL


# M1: vertical barrier with a bottom detour.
M1 = M0.copy()
M1[0:9, 5] = WALL


# M2: exact 90-degree rotation of M1.
M2 = np.rot90(M1, k=1).copy()


# M3: cross maze with four narrow passages.
M3 = M0.copy()
M3[1:-1, 5] = WALL
M3[5, 1:-1] = WALL

M3_DOORS = (
    (2, 5),
    (5, 2),
    (5, 8),
    (8, 5),
)

for row, col in M3_DOORS:
    M3[row, col] = FREE


# M4: M1 with one additional doorway.
M4 = M1.copy()
M4[4, 5] = FREE


CUSTOM_MAZE_GRIDS = {
    "m0": M0,
    "m1": M1,
    "m2": M2,
    "m3": M3,
    "m4": M4,
}

CUSTOM_MAZE_NAMES = {
    "m0": "Open arena",
    "m1": "Vertical barrier",
    "m2": "Horizontal barrier",
    "m3": "Cross maze",
    "m4": "Double-door",
}

SHOWCASE_PAIRS = {
    "m0": ((2, 2), (8, 8)),
    "m1": ((4, 4), (4, 6)),
    "m2": ((4, 4), (6, 4)),
    "m3": ((2, 2), (8, 8)),
    "m4": ((4, 4), (4, 6)),
}

EXPECTED_FREE_COUNTS = {
    "m0": 81,
    "m1": 73,
    "m2": 73,
    "m3": 68,
    "m4": 74,
}


def _validate_layouts():
    for maze_id, grid in CUSTOM_MAZE_GRIDS.items():
        if grid.shape != (MAZE_SIZE, MAZE_SIZE):
            raise ValueError(
                f"{maze_id}: invalid shape {grid.shape}"
            )

        if not set(np.unique(grid)).issubset({FREE, WALL}):
            raise ValueError(
                f"{maze_id}: values must be only 0 or 1"
            )

        free_count = int(np.sum(grid == FREE))

        if free_count != EXPECTED_FREE_COUNTS[maze_id]:
            raise ValueError(
                f"{maze_id}: expected "
                f"{EXPECTED_FREE_COUNTS[maze_id]} free cells, "
                f"got {free_count}"
            )

    if not np.array_equal(M2, np.rot90(M1, k=1)):
        raise ValueError(
            "M2 must be an exact rotation of M1"
        )

    if int(np.sum(M1 != M4)) != 1:
        raise ValueError(
            "M1 and M4 must differ by exactly one cell"
        )


_validate_layouts()
