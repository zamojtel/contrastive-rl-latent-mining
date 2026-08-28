import xml.etree.ElementTree as ET

import numpy as np
import pytest

from jaxgcrl.envs.maze_layouts import (
    CUSTOM_MAZE_GRIDS,
    EXPECTED_FREE_COUNTS,
    FREE,
    MAZE_SIZE,
    SHOWCASE_PAIRS,
    WALL,
)
from jaxgcrl.envs.simple_maze import (
    compile_custom_layout,
    make_maze,
)
from jaxgcrl.utils.env import create_env, legal_envs


MAZE_IDS = tuple(CUSTOM_MAZE_GRIDS)
SCALE = 2.0


def reachable_free_cells(grid):
    free_cells = {
        tuple(cell)
        for cell in np.argwhere(grid == FREE)
    }

    pending = [next(iter(free_cells))]
    visited = set()

    while pending:
        row, col = pending.pop()
        cell = (row, col)

        if cell in visited:
            continue

        visited.add(cell)

        for drow, dcol in (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ):
            neighbour = (
                row + drow,
                col + dcol,
            )

            if (
                neighbour in free_cells
                and neighbour not in visited
            ):
                pending.append(neighbour)

    return visited


@pytest.mark.parametrize("maze_id", MAZE_IDS)
def test_grid_geometry_and_connectivity(maze_id):
    grid = CUSTOM_MAZE_GRIDS[maze_id]
    start, goal = SHOWCASE_PAIRS[maze_id]

    assert grid.shape == (MAZE_SIZE, MAZE_SIZE)
    assert set(np.unique(grid)) == {FREE, WALL}

    assert np.all(grid[0, :] == WALL)
    assert np.all(grid[-1, :] == WALL)
    assert np.all(grid[:, 0] == WALL)
    assert np.all(grid[:, -1] == WALL)

    free_count = int(np.sum(grid == FREE))

    assert free_count == EXPECTED_FREE_COUNTS[maze_id]
    assert grid[start] == FREE
    assert grid[goal] == FREE

    assert len(reachable_free_cells(grid)) == free_count


def test_controlled_layout_relations():
    m1 = CUSTOM_MAZE_GRIDS["m1"]
    m2 = CUSTOM_MAZE_GRIDS["m2"]
    m4 = CUSTOM_MAZE_GRIDS["m4"]

    np.testing.assert_array_equal(
        m2,
        np.rot90(m1, k=1),
    )

    differences = np.argwhere(m1 != m4)

    np.testing.assert_array_equal(
        differences,
        np.asarray([[4, 5]]),
    )


@pytest.mark.parametrize("maze_id", MAZE_IDS)
def test_compilation_and_xml(maze_id):
    grid = CUSTOM_MAZE_GRIDS[maze_id]

    compiled_grid, positions = compile_custom_layout(
        grid,
        SCALE,
    )

    np.testing.assert_array_equal(
        compiled_grid,
        np.flipud(grid).T,
    )

    free_rows, free_cols = np.where(grid == FREE)
    expected_positions = (
        np.column_stack(
            (
                free_cols,
                MAZE_SIZE - 1 - free_rows,
            )
        ).astype(np.float32)
        * SCALE
    )

    np.testing.assert_allclose(
        np.asarray(positions),
        expected_positions,
    )

    xml_string, starts, goals = make_maze(
        maze_id,
        SCALE,
    )

    np.testing.assert_allclose(
        np.asarray(starts),
        expected_positions,
    )
    np.testing.assert_allclose(
        np.asarray(goals),
        expected_positions,
    )

    root = ET.fromstring(xml_string)
    walls = [
        geom
        for geom in root.findall(".//geom")
        if geom.attrib.get("name", "").startswith(
            "block_"
        )
    ]

    assert len(walls) == int(np.sum(grid == WALL))


@pytest.mark.parametrize("maze_id", MAZE_IDS)
def test_create_env_registration(maze_id):
    env_name = f"simple_{maze_id}"

    assert env_name in legal_envs

    env = create_env(
        env_name=env_name,
        backend="spring",
        maze_size_scaling=SCALE,
    )

    starts = np.asarray(env.possible_starts)
    goals = np.asarray(env.possible_goals)

    expected_count = EXPECTED_FREE_COUNTS[maze_id]

    assert len(starts) == expected_count
    assert len(goals) == expected_count
    assert np.isclose(starts.min(), 2.0)
    assert np.isclose(starts.max(), 18.0)
