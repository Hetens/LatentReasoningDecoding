## Sudoku metrics and solving logic

This document explains how `sudoku_metrics` and `solve_sudoku` work, what each metric means, and how the Sudoku is solved internally.

### Puzzle representation

- **Input format**: An 81-character string, read left-to-right, row by row.
  - Digits `'1'`–`'9'` are filled cells.
  - `'.'` or `'0'` are empty cells.
  - Optionally, the string may be `puzzle|solution`; in that case only the puzzle part (before the `|`) is used.
- **Internal grid**: Both functions convert the puzzle into a 9×9 `grid`:
  - `grid[r][c] == 0` means the cell is empty.
  - `grid[r][c] in {1, …, 9}` is a fixed or deduced value.
- **Candidates**: For each empty cell, a set `cand[r][c]` holds all digits that are still possible given current assignments and Sudoku rules (no duplicates in any row, column, or 3×3 box).

---

## Metrics produced by `sudoku_metrics`

`sudoku_metrics` both **solves** the puzzle and **counts** how often specific logical techniques are used. It returns a `SudokuMetrics` instance with the following fields.

### `num_givens`

- **Definition**: The number of non-empty cells in the original puzzle.
- **How computed**: Counts characters in the input string that are digits `'1'`–`'9'`.
- **Interpretation**: A lower number of givens generally (but not always) corresponds to a harder puzzle.

### `num_singles` (naked singles)

- **Naked single rule**: If a cell has exactly **one** valid candidate, that candidate must be the value of the cell.
- **How computed**:
  - During solving, whenever the solver finds a cell `(r, c)` with `grid[r][c] == 0` and `len(cand[r][c]) == 1`, it:
    - Sets the cell to that value.
    - Removes that value from the candidates of all peers in the same row, column, and box.
    - Increments `num_singles` by 1.
- **Interpretation**: Counts all placements that were forced simply because only one number could go in that cell.

### `num_hidden_singles`

- **Hidden single rule**: Within a unit (row, column, or box), if a digit can go in exactly **one** cell, that placement is forced, even if that cell currently has multiple candidates.
- **How computed**:
  - For each digit `d` in `{1, …, 9}`:
    - For each 3×3 box, row, and column:
      - Collect all empty cells in that unit that still have `d` as a candidate.
      - If there is exactly one such cell `(r, c)`, the solver sets `grid[r][c] = d`, updates candidates, and increments `num_hidden_singles`.
- **Interpretation**: Counts placements forced by the *unit-level* uniqueness of a digit rather than by a cell having only one candidate.

### `num_naked_pairs`

- **Naked pair rule**: In a unit (row, column, or box), if **exactly two** cells share the **same pair of candidates** `{a, b}`, then:
  - Those two cells must take the values `a` and `b` in some order.
  - Therefore, digits `a` and `b` can be removed from the candidate sets of all *other* cells in that unit.
- **How computed**:
  - For each unit (every row, every column, every box):
    - Find all empty cells with exactly two candidates.
    - For each pair of such cells `(r1, c1)` and `(r2, c2)`:
      - If `cand[r1][c1] == cand[r2][c2] == {a, b}` and **no other cell in the unit** has that same candidate set, treat this as a naked pair.
      - Remove `a` and `b` from the candidates of every other cell in the unit.
      - If at least one candidate is actually removed, increment `num_naked_pairs`.
- **Interpretation**: Measures how often the solver used the pattern “two cells share a unique pair of candidates” to prune other cells.

### `num_hidden_pairs`

- **Hidden pair rule**: In a unit, if two digits `{d1, d2}` appear as candidates in **exactly the same two cells** and nowhere else in that unit, then:
  - Those two cells must be `{d1, d2}` in some order.
  - Any other candidates in those cells can be removed.
- **How computed**:
  - For each unit (row, column, box):
    - Let `empty` be all empty cells in the unit.
    - For each pair of digits `(d1, d2)`:
      - Find cells where `d1` is a candidate (`cells1`) and where `d2` is a candidate (`cells2`).
      - If `cells1` and `cells2` are the same two positions, those two digits form a hidden pair.
      - Remove all other candidates from those two cells.
      - If any candidate is actually removed, increment `num_hidden_pairs`.
- **Interpretation**: Counts times the solver recognized that two digits are “hidden” together in exactly two cells and pruned other possibilities from those cells.

### `num_pointing_pairs_triples`

- **Pointing pair/triple rule**:
  - Look within a 3×3 box for a digit `d`.
  - If all candidates for `d` in that box lie **in the same row** (or **in the same column**), then:
    - Outside of that box, in the same row (or column), `d` cannot appear.
    - So `d` is removed from candidates of those outside cells in the row/column.
- **How computed**:
  - For each box and each digit `d`:
    - Collect all empty cells in the box that have `d` as a candidate.
    - If there are at least 2 such cells and they all share the same row or the same column:
      - Eliminate `d` from other cells in that row/column that are **outside** the box.
      - If any candidates are removed, increment `num_pointing_pairs_triples`.
- **Interpretation**: Counts how often the box structure forced eliminations along a row or column because all occurrences of a digit in that box “point” along a single line.

### `num_box_line_intersections`

- **Box/line reduction rule** (also called line-box interaction):
  - Look at a row (or column) and a digit `d`.
  - If all candidates for `d` in that row (or column) lie within the **same box**, then:
    - `d` cannot appear elsewhere in that box, outside of that row (or column).
    - So `d` is removed from other cells in that box.
- **How computed**:
  - For each row and each digit `d`:
    - Collect empty cells with `d` as a candidate.
    - If there are at least 2 and they all sit in the same 3×3 box:
      - Remove `d` from other cells in that box that are not on the row.
  - The same is done for each column.
  - If any candidate is removed, `num_box_line_intersections` is incremented.
- **Interpretation**: Counts how often the intersection between a row/column and a box allowed additional eliminations.

### `num_guesses` and `num_backtracks`

- **Guessing (backtracking)**:
  - When none of the logical techniques above can make progress and the puzzle is not solved, the solver chooses a still-empty cell and **guesses** one of its candidates.
  - It then recursively continues solving; if a contradiction is reached, it “backtracks” and tries a different value.
- `**num_guesses`**:
  - Incremented every time the solver chooses a value to try for a cell as part of backtracking.
- `**num_backtracks**`:
  - Incremented (by 2) each time a guess leads to a dead end and the solver must revert to the previous state.
  - The factor of 2 mirrors the way QQWing counts rollback operations.
- **Interpretation**:
  - High `num_guesses`/`num_backtracks` indicate a puzzle that cannot be solved purely by the listed logical techniques and requires search.

### `difficulty`

The difficulty is assigned after solving, using QQWing-style thresholds:

- **Expert**:
  - If `num_guesses > 0` (backtracking was required).
- **Intermediate**:
  - If no guesses were needed, but any of:
    - `num_box_line_intersections > 0`
    - `num_pointing_pairs_triples > 0`
    - `num_naked_pairs > 0`
    - `num_hidden_pairs > 0`
- **Easy**:
  - If none of the above, but `num_hidden_singles > 0`.
- **Simple**:
  - If only naked singles were used (`num_singles > 0`) and no more advanced logic.
- **Unknown**:
  - If the puzzle was not solved or no techniques recorded.

### `solution`

- If a full 9×9 grid is successfully filled with digits obeying Sudoku rules, `solution` is set to the 81-character solution string (row-major).
- Otherwise, `solution` remains `None`.

---

## How `sudoku_metrics` solves the puzzle

At a high level, `sudoku_metrics` does:

1. **Initialize candidates**:
  - Every empty cell starts with candidates `{1, …, 9}`.
  - For each given digit, the digit is removed from candidates in the same row, column, and box.
2. **Logical solving loop** (no guessing yet):
  - Repeatedly apply techniques in this order, as long as any of them can make a move:
  1. Naked singles (`num_singles`).
  2. Hidden singles (box, then row, then column; `num_hidden_singles`).
  3. Naked pairs (`num_naked_pairs`).
  4. Pointing pairs/triples (`num_pointing_pairs_triples`).
  5. Box/line intersections (`num_box_line_intersections`).
  6. Hidden pairs (`num_hidden_pairs`).
    ch time a technique places a value or removes candidates, the candidates are updated consistently for all peers.
3. **If the puzzle is solved** after these logical steps:
  - The 81-character solution string is recorded in `solution`, and metrics are finalized.
4. **If not solved and no logical move remains**:
  - The solver resorts to **backtracking search**:
    - Chooses a cell to guess (either with the fewest candidates or first in row-major order, depending on `guess_order`).
    - For each candidate value:
      - Saves the current `grid` and `cand`.
      - Sets the cell to that value and recursively tries to finish solving using a similar loop (including logical techniques).
      - If the recursive attempt fails, it restores the previous state and tries the next candidate.
    - `num_guesses` and `num_backtracks` track how often this search is invoked and undone.

If at least one branch of the backtracking search completes a valid grid, the puzzle is considered solved and the solution is recorded.

---

## How `solve_sudoku` works

`solve_sudoku` is a more direct solver that returns only the solution string (or raises an error) without collecting metrics.

### Steps

1. **Parse input**:
  - Normalize the string (strip spaces, take part before `|`, require length 81).
  - Build a 9×9 `grid` with `0` for empty cells and digits for givens.
2. **Initialize candidates**:
  - Start all cells with candidates `{1, …, 9}`.
  - For each given digit, remove that digit from the candidates of all cells in the same row, column, and box.
3. **Constraint propagation (`propagate`)**:
  - Repeatedly apply:
    - **Naked singles**: Any cell with exactly one candidate is filled.
    - **Hidden singles**: For each unit (row, column, box) and digit `d`, if only one cell in that unit can take `d`, fill that cell.
  - Continue until no more singles can be found.
4. **Backtracking search (`solve`)**:
  - After propagation, scan for the first empty cell:
    - If an empty cell has no candidates, the current branch is inconsistent → backtrack (return `False`).
    - Otherwise, try each candidate `v` for that cell:
      - Save `grid` and `cand`.
      - Place `v` in the cell and recursively call `solve`.
      - If the recursive call succeeds, bubble up success.
      - If it fails, restore the saved state and try the next candidate.
  - If all cells are filled without contradiction, the puzzle is solved.
5. **Return**:
  - If `solve()` finishes successfully, the function returns the final grid as an 81-character solution string.
  - If every branch fails, `solve_sudoku` raises `ValueError("Puzzle has no solution")`.

---

## Summary

- `sudoku_metrics` and `solve_sudoku` both use standard Sudoku constraint propagation and backtracking.
- The metrics in `SudokuMetrics` count how often specific techniques (singles, pairs, pointing/box-line interactions, and guessing) were applied while solving.
- The corrected naked-pair logic ensures that eliminations are only performed when **exactly two cells** in a unit share a given two-digit candidate set, matching the intended Sudoku rule and preventing invalid eliminations.

