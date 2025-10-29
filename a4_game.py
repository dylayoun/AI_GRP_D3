# a4_game.py
"""
Hinger – Task 4: Main game loop that uses Task 1 (State) and Task 3 (Agent).

- Uses a1_state.State for the board & rules
- Uses a3_agent.Agent for AI move selection (minimax/alphabeta)
- Alternates turns; validates & applies moves; detects hinger/draw
- Logs JSON history with per-move coordinates for visual replay
- Marks the hinger move with a "hinger" flag for the viewer
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import a1_state   # State(grid)
import a3_agent   # Agent(sizeTuple).move(state, mode)


# ========= Helpers (camelCase) =========

def agentName(agent: Any, default: str) -> str:
    if agent is None:
        return default
    return getattr(agent, "name", default)


def extractBoardSize(state: a1_state.State) -> Tuple[int, int]:
    """Map your State attributes to (rows, cols)."""
    rows = getattr(state, "row_len", None)
    cols = getattr(state, "col_len", None)
    if rows is None or cols is None:
        grid = getattr(state, "grid", None)
        if isinstance(grid, list) and grid and isinstance(grid[0], list):
            return len(grid), len(grid[0])
        return 4, 5
    return int(rows), int(cols)


def legalMovesRC(state: a1_state.State) -> List[Tuple[int, int]]:
    """Convert State.moves() which returns (r, c, cost) to a list of (r, c)."""
    rc: List[Tuple[int, int]] = []
    if hasattr(state, "moves"):
        for move in state.moves():
            if isinstance(move, (tuple, list)) and len(move) >= 2:
                rc.append((int(move[0]), int(move[1])))
    return rc


def toCoord(move: Any) -> Tuple[Optional[int], Optional[int]]:
    """Parse move to (r, c). Supports (r,c), (r,c,cost), dicts, and 'r,c' strings."""
    if isinstance(move, (tuple, list)) and len(move) >= 2:
        try:
            return int(move[0]), int(move[1])
        except Exception:
            return None, None
    if isinstance(move, dict):
        for a, b in (("row", "col"), ("r", "c")):
            if a in move and b in move:
                try:
                    return int(move[a]), int(move[b])
                except Exception:
                    return None, None
    if isinstance(move, str) and "," in move:
        parts = move.replace(" ", "").split(",")
        if len(parts) == 2:
            try:
                return int(parts[0]), int(parts[1])
            except Exception:
                return None, None
    return None, None


def isValidMove(state: a1_state.State, move: Any) -> bool:
    r, c = toCoord(move)
    if r is None:
        return False
    return (r, c) in set(legalMovesRC(state))


def applyMove(state: a1_state.State, move: Any) -> None:
    r, c = toCoord(move)
    if r is None:
        raise ValueError(f"Cannot apply invalid move: {move!r}")
    if not hasattr(state, "makeMove"):
        raise AttributeError("State.makeMove(r, c) is required.")
    state.makeMove(r, c)


def hingerTriggered(stateBefore: a1_state.State, move: Any) -> bool:
    """
    A 'hinger' is a cell with value 1 whose removal increases the number of regions.
    Simulate the removal on a copy and compare numRegions() before vs after.
    """
    r, c = toCoord(move)
    if r is None:
        return False

    grid = stateBefore.grid
    if not (0 <= r < len(grid) and 0 <= c < len(grid[0])):
        return False
    if grid[r][c] != 1:
        return False

    currRegions = stateBefore.numRegions()
    newGrid = [row[:] for row in grid]
    newGrid[r][c] = 0
    newState = a1_state.State(newGrid)
    newRegions = newState.numRegions()
    return newRegions > currRegions


def isDraw(state: a1_state.State) -> bool:
    """Draw if all counters are removed (all cells are 0)."""
    for row in state.grid:
        for v in row:
            if v != 0:
                return False
    return True


def maybeSaveHistory(path: Optional[str], payload: Dict[str, Any]) -> None:
    if not path:
        return
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print(f"[warn] Failed to save history to {path}: {e}")


# ========= Core gameplay =========

def play(
    state: a1_state.State,
    player1: Optional[a3_agent.Agent],
    player2: Optional[a3_agent.Agent],
    *,
    player1Mode: str = "alphabeta",
    player2Mode: str = "alphabeta",
    timeLimit: float = 60.0,
    saveHistoryPath: Optional[str] = None,
    quiet: bool = False,
) -> Optional[str]:
    """
    Run a complete Hinger game using your State and Agent implementations.
    Returns winner name (str) or None for draw.
    """
    name1 = agentName(player1, "Player1")
    name2 = agentName(player2, "Player2")

    rows, cols = extractBoardSize(state)

    moveHistory: Dict[str, Any] = {
        "meta": {
            "game": "Hinger",
            "players": [name1, name2],
            "time_limit_sec": timeLimit,
            "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "rows": rows,
            "cols": cols,
        },
        "moves": [],
        "result": None,
    }

    if not quiet:
        print("=== Hinger: Game start ===")
        print(state)

    turn = 0
    currentIdx = 0  # 0 -> A, 1 -> B
    gameStart = time.time()

    def endGame(winner: Optional[str], reason: str) -> Optional[str]:
        moveHistory["meta"]["totalTurns"] = turn
        moveHistory["result"] = {
            "winner": winner,
            "reason": reason,
            "durationSec": round(time.time() - gameStart, 3),
        }
        maybeSaveHistory(saveHistoryPath, moveHistory)
        return winner

    while True:
        turn += 1
        playerIs1 = (currentIdx == 0)
        playerAgent = player1 if playerIs1 else player2
        playerName = name1 if playerIs1 else name2
        oppName = name2 if playerIs1 else name1
        mode = player1Mode if playerIs1 else player2Mode

        if not quiet:
            print(f"\nTurn {turn}: {playerName}'s move")

        # choose move
        startTs = time.time()
        if playerAgent is None:
            raw = input("Enter your move (e.g., '2,3'): ").strip()
            move = raw
        else:
            move = playerAgent.move(state, mode)
        endTs = time.time()
        elapsed = endTs - startTs

        # time limit (soft)
        if timeLimit is not None and elapsed > timeLimit:
            if not quiet:
                print(f"{playerName} exceeded {timeLimit:.0f}s ({elapsed:.2f}s). {oppName} wins.")
            r, c = toCoord(move)
            moveHistory["moves"].append({
                "turn": turn, "player": playerName, "move": move, "coord": [r, c] if r is not None else None,
                "start_ts": startTs, "end_ts": endTs, "elapsed_sec": elapsed, "flags": ["timeout"],
            })
            return endGame(oppName, "timeout")

        # legality (from State.moves())
        if not isValidMove(state, move):
            if not quiet:
                print(f"Illegal move by {playerName}: {move!r}. {oppName} wins.")
            r, c = toCoord(move)
            moveHistory["moves"].append({
                "turn": turn, "player": playerName, "move": move, "coord": [r, c] if r is not None else None,
                "start_ts": startTs, "end_ts": endTs, "elapsed_sec": elapsed, "flags": ["illegal"],
            })
            return endGame(oppName, "illegal_move")

        # EXTRA SAFETY: prevent re-playing a cleared square (value==0)
        r, c = toCoord(move)
        if getattr(state, "grid", None) is not None:
            try:
                if state.grid[r][c] == 0:
                    if not quiet:
                        print(f"Illegal move by {playerName}: cell {(r, c)} already empty. {oppName} wins.")
                    moveHistory["moves"].append({
                        "turn": turn, "player": playerName, "move": move, "coord": [r, c],
                        "start_ts": startTs, "end_ts": endTs, "elapsed_sec": elapsed,
                        "flags": ["illegal", "occupied_cell"],
                    })
                    return endGame(oppName, "illegal_move")
            except IndexError:
                # Shouldn't happen if isValidMove passed, but guard anyway
                moveHistory["moves"].append({
                    "turn": turn, "player": playerName, "move": move, "coord": [r, c] if r is not None else None,
                    "start_ts": startTs, "end_ts": endTs, "elapsed_sec": elapsed, "flags": ["illegal"],
                })
                return endGame(oppName, "illegal_move")

        # detect hinger BEFORE applying
        triggered = hingerTriggered(state, move)

        # apply move
        applyMove(state, move)

        # record move
        moveEntry = {
            "turn": turn,
            "player": playerName,
            "move": move,
            "coord": [r, c] if r is not None else None,
            "start_ts": startTs,
            "end_ts": endTs,
            "elapsed_sec": elapsed,
            "flags": [],
        }
        moveHistory["moves"].append(moveEntry)

        if not quiet:
            print(f"→ {playerName} played {move!r}")
            print(state)

        # if hinger: mark the just-recorded move and finish
        if triggered:
            moveEntry["flags"].append("hinger")
            if not quiet:
                print(f"🏆 {playerName} triggered a hinger and wins!")
            return endGame(playerName, "hinger")

        # draw
        if isDraw(state):
            if not quiet:
                print("🤝 Draw — all counters removed.")
            return endGame(None, "draw")

        # next
        currentIdx ^= 1


# ========= Minimal real tester =========

def tester():
    """Runs a short game using your real modules (a1_state + a3_agent)."""
    grid = [
        [1, 1, 0, 0, 2],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
    ]
    state = a1_state.State(grid)
    rows, cols = extractBoardSize(state)

    player1 = a3_agent.Agent((rows, cols), name="player1")
    player2 = a3_agent.Agent((rows, cols), name="player2")

    print("Running Hinger (a1+a3) with alphabeta...")
    winner = play(
        state,
        player1,
        player2,
        player1Mode="alphabeta",
        player2Mode="alphabeta",
        timeLimit=60,
        saveHistoryPath="gameHistory.json",
        quiet=False,
    )
    print(f"\nWinner: {winner}")
    print("History saved to gameHistory.json")


if __name__ == "__main__":
    tester()
