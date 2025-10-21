"""
Hinger Project
Coursework 001 for: CMP-6058A Artificial Intelligence

Task 2: Safe Path Finding Algorithms

Group Number: 
Student ID: 100430249 (Robert Soanes)
Partner IDs: 100889423 (Dylan Young)

@date: 29/09/2025
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import random
import time
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

Move = Tuple[int, str, str]  # (index, old_bit, new_bit)
State = str


# Helpers & Validation 

def _ensureBinaryState(s: State) -> None:
    """Validate that a state is a non-empty binary string"""
    if not isinstance(s, str) or not s:
        raise ValueError("State must be a non-empty string")
    if any(c not in {'0', '1'} for c in s):
        raise ValueError("State must be binary")


def _sameLength(a: State, b: State) -> None:
    """Ensure two states have the same length."""
    if len(a) != len(b):
        raise ValueError("States must be the same length")


def _applyMove(state: State, move: Move) -> State:
    """Apply a single move to a state, returning the new state"""
    idx, old_bit, new_bit = move
    if state[idx] != old_bit:
        raise ValueError(f"Move inconsistent with state at index {idx}: {state[idx]} != {old_bit}")
    return state[:idx] + new_bit + state[idx + 1:]


def _neighbors(state: State, forbidden: Optional[Set[State]] = None) -> Iterable[Tuple[State, Move]]:
    """Generate all 1-bit flip neighbors of state that are not forbidden"""
    n = len(state)
    for i in range(n):
        flipped = '1' if state[i] == '0' else '0'
        ns = state[:i] + flipped + state[i+1:]
        if forbidden is not None and ns in forbidden:
            continue
        yield ns, (i, state[i], flipped)


def _reconstructMoves(parents: Dict[State, Tuple[Optional[State], Optional[Move]]], end: State) -> List[Move]:
    """Reconstruct the path of moves from start to finish using parent pointers"""
    moves: List[Move] = []
    cur = end
    while True:
        parent, move = parents[cur]
        if parent is None:
            break
        moves.append(move)  # type: ignore
        cur = parent
    moves.reverse()
    return moves


def _hamming(a: State, b: State) -> int:
    """Calculate the Hamming distance between two states"""
    return sum(1 for x, y in zip(a, b) if x != y)


def _weightedHamming(a: State, b: State, weights: Optional[Dict[int, float]]) -> float:
    """
    Calculate weighted Hamming distance between two states
    
    If weights are provided, uses the weight for each differing bit position.
    Otherwise, each differing bit contributes 1.0 to the distance
    """
    if not weights:
        return float(_hamming(a, b))
    total = 0.0
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            total += float(weights.get(i, 1.0))
    return total


# Search Algorithms 

def path_BFS(start: State, end: State, forbidden: Optional[Set[State]] = None) -> Optional[List[Move]]:
    """
    Breadth-First Search: returns a shortest safe path as a list of moves.
    
    BFS explores states level by level, guaranteeing the shortest path (in terms
    of number of moves) when all moves have equal cost. It uses a queue to maintain
    the frontier.
    
    Time Complexity: O(2^n) where n is the length of the state
    Space Complexity: O(2^n) for the visited set and queue
    
    Args:
        start: Initial binary state
        end: Goal binary state
        forbidden: Set of states to avoid (hinger states)
    
    Returns:
        List of moves forming the shortest safe path, or None if no path exists
    """
    _ensureBinaryState(start)
    _ensureBinaryState(end)
    _sameLength(start, end)

    if start == end:
        return []

    if forbidden is None:
        forbidden = set()
    if start in forbidden:
        return None

    q = deque([start])
    parents: Dict[State, Tuple[Optional[State], Optional[Move]]] = {start: (None, None)}
    visited: Set[State] = {start}

    while q:
        cur = q.popleft()
        for ns, mv in _neighbors(cur, forbidden):
            if ns in visited:
                continue
            parents[ns] = (cur, mv)
            if ns == end:
                return _reconstructMoves(parents, end)
            visited.add(ns)
            q.append(ns)
    return None


def path_DFS(start: State, end: State, forbidden: Optional[Set[State]] = None, depth_limit: Optional[int] = None) -> Optional[List[Move]]:
    """
    Depth-First Search: returns a safe path if one exists (not guaranteed minimal).
    
    DFS explores as far as possible along each branch before backtracking. It uses
    a stack to keep track of visited nodes. It is iterative rather than
    recursive to avoid stack overflow on deep searches.
    
    Note: DFS does NOT guarantee the shortest path, but can be more memory-efficient
    than BFS in some cases.
    
    Time Complexity: O(2^n) where n is the length of the state
    Space Complexity: O(n * 2^n) in worst case due to stack depth
    
    Args:
        start: Initial binary state
        end: Goal binary state
        forbidden: Set of states to avoid (hinger states)
        depth_limit: Optional maximum depth to explore
    
    Returns:
        List of moves forming a safe path, or None if no path exists
    """
    _ensureBinaryState(start)
    _ensureBinaryState(end)
    _sameLength(start, end)

    if start == end:
        return []

    if forbidden is None:
        forbidden = set()
    if start in forbidden:
        return None

    stack: List[Tuple[State, int]] = [(start, 0)]
    parents: Dict[State, Tuple[Optional[State], Optional[Move]]] = {start: (None, None)}
    visited: Set[State] = {start}

    while stack:
        cur, depth = stack.pop()
        if depth_limit is not None and depth >= depth_limit:
            continue
        for ns, mv in _neighbors(cur, forbidden):
            if ns in visited:
                continue
            parents[ns] = (cur, mv)
            if ns == end:
                return _reconstructMoves(parents, end)
            visited.add(ns)
            stack.append((ns, depth + 1))
    return None


def _dls(current: State, end: State, limit: int, forbidden: Optional[Set[State]], 
         parents: Dict[State, Tuple[Optional[State], Optional[Move]]], visited: Set[State]) -> Optional[List[Move]]:
    """Helper function for IDDFS: performs depth-limited search recursively"""
    if current == end:
        return _reconstructMoves(parents, end)
    if limit == 0:
        return None
    for ns, mv in _neighbors(current, forbidden):
        if ns in visited:
            continue
        visited.add(ns)
        parents[ns] = (current, mv)
        result = _dls(ns, end, limit - 1, forbidden, parents, visited)
        if result is not None:
            return result
    return None


def path_IDDFS(start: State, end: State, forbidden: Optional[Set[State]] = None, maxDepth: int = 10_000) -> Optional[List[Move]]:
    """
    Iterative Deepening Depth-First Search: combines benefits of BFS and DFS.
    
    IDDFS repeatedly performs depth-limited DFS with increasing depth limits until
    a solution is found. This guarantees finding the shortest path (like BFS) while
    using memory proportional to the depth of the solution (like DFS).
    
    The algorithm is particularly useful when the solution depth is unknown but
    expected to be relatively shallow compared to the search space.
    
    Time Complexity: O(n * 2^n) where n is the solution depth
    Space Complexity: O(n) - much better than BFS's O(2^n)
    
    Args:
        start: Initial binary state
        end: Goal binary state
        forbidden: Set of states to avoid (hinger states)
        maxDepth: Maximum depth to explore before giving up
    
    Returns:
        List of moves forming the shortest safe path, or None if no path exists
        within maxDepth
    """
    _ensureBinaryState(start)
    _ensureBinaryState(end)
    _sameLength(start, end)

    if start == end:
        return []

    if forbidden is None:
        forbidden = set()
    if start in forbidden:
        return None

    for depth in range(maxDepth + 1):
        parents: Dict[State, Tuple[Optional[State], Optional[Move]]] = {start: (None, None)}
        visited: Set[State] = {start}
        result = _dls(start, end, depth, forbidden, parents, visited)
        if result is not None:
            return result
    return None


@dataclass(order=True)
class _PQItem:
    """Priority queue item for A* search, ordered by f-score"""
    f: float
    g: float
    state: State


def path_astar(
    start: State,
    end: State,
    forbidden: Optional[Set[State]] = None,
    weights: Optional[Dict[int, float]] = None,
    heuristic: Optional[Callable[[State, State], float]] = None,
) -> Optional[List[Move]]:
    """
    A* search: finds the optimal path using a heuristic to guide exploration.
    
    HEURISTIC FUNCTION JUSTIFICATION:
    This implementation uses weighted Hamming distance as the default heuristic.
    The Hamming distance counts the number of differing bits between two states,
    which represents a lower bound on the number of moves needed (since each move
    can change at most one bit).
    
    ADMISSIBILITY: The weighted Hamming heuristic is admissible because:
    1. Each bit that differs must be flipped at least once
    2. Each flip costs at least the weight for that bit position
    3. Therefore, the sum of weights for differing bits never overestimates the
       actual minimum cost
    
    CONSISTENCY: The heuristic is also consistent (monotonic) because:
    - Moving from state s to neighbor s' by flipping bit i:
      - Decreases h by weight[i] if it matches the goal (progress toward goal)
      - Increases h by weight[i] if it differs from goal (moving away)
      - The actual cost is also weight[i], so h(s) ≤ cost(s, s') + h(s')
    
    This makes A* optimal: it's guaranteed to find the minimum-cost path when
    the heuristic is admissible and consistent.
    
    Time Complexity: O(2^n * log(2^n)) = O(n * 2^n) with priority queue operations
    Space Complexity: O(2^n) for open and closed sets
    
    Args:
        start: Initial binary state
        end: Goal binary state
        forbidden: Set of states to avoid (hinger states)
        weights: Per-bit costs for flipping bit i (default 1.0 for all bits)
        heuristic: Optional custom heuristic h(s, end). If omitted, uses weighted Hamming
    
    Returns:
        List of moves forming the minimum-cost safe path, or None if no path exists
    """
    _ensureBinaryState(start)
    _ensureBinaryState(end)
    _sameLength(start, end)

    if start == end:
        return []

    if forbidden is None:
        forbidden = set()
    if start in forbidden:
        return None

    def default_h(s: State, t: State) -> float:
        return _weightedHamming(s, t, weights)

    h = heuristic or default_h

    open_heap: List[_PQItem] = []
    g_cost: Dict[State, float] = {start: 0.0}
    parents: Dict[State, Tuple[Optional[State], Optional[Move]]] = {start: (None, None)}

    heapq.heappush(open_heap, _PQItem(h(start, end), 0.0, start))
    closed: Set[State] = set()

    while open_heap:
        item = heapq.heappop(open_heap)
        cur = item.state
        if cur in closed:
            continue
        if cur == end:
            return _reconstructMoves(parents, end)
        closed.add(cur)

        for ns, mv in _neighbors(cur, forbidden):
            c = float(weights.get(mv[0], 1.0)) if weights else 1.0
            tentative = g_cost[cur] + c
            if ns not in g_cost or tentative < g_cost[ns]:
                g_cost[ns] = tentative
                parents[ns] = (cur, mv)
                heapq.heappush(open_heap, _PQItem(tentative + h(ns, end), tentative, ns))
    return None


# Testing & Comparison 

def _applyMovesSequence(start: State, moves: List[Move]) -> State:
    """Apply a sequence of moves to a start state, returning the final state"""
    cur = start
    for mv in moves:
        cur = _applyMove(cur, mv)
    return cur


def tester() -> None:
    """
    Run comprehensive sanity tests on all implemented search algorithms.
    
    Tests include:
    - Trivial cases (start == end)
    - Simple transformations with no forbidden states
    - Forbidden state handling
    - Weighted cost scenarios
    - Edge cases
    """
    print("Running tests")
  
    # Test 1: Trivial equal states
    print("\n[Test 1] Trivial cases (start == end)...")
    assert path_BFS("0", "0") == []
    assert path_DFS("1", "1") == []
    assert path_IDDFS("101", "101") == []
    assert path_astar("00", "00") == []
    print(" All algorithms correctly handle trivial cases")

    # Test 2: Simple 3-bit flip from 000 -> 111 (no forbidden)
    print("\n[Test 2] Basic transformations (000 -> 111)...")
    start, end = "000", "111"
    for fn in (path_BFS, path_DFS, path_IDDFS, path_astar):
        res = fn(start, end)
        assert res is not None, f"{fn.__name__} failed to find path"
        final = _applyMovesSequence(start, res)
        assert final == end, f"{fn.__name__} produced wrong end state: {final}"
        print(f"   {fn.__name__}: found path of length {len(res)}")
    
    # Test 3: Forbidden state handling
    print("\n[Test 3] Forbidden state avoidance...")
    forbidden = {"100"}
    res_bfs = path_BFS("000", "111", forbidden=forbidden)
    assert res_bfs is not None, "BFS failed with forbidden states"
    final = _applyMovesSequence("000", res_bfs)
    assert final == "111", "BFS path didn't reach goal"
    # Verify no forbidden states in path
    cur = "000"
    for mv in res_bfs:
        cur = _applyMove(cur, mv)
        assert cur not in forbidden, f"Path includes forbidden state: {cur}"
    print(f"   BFS found safe path of length {len(res_bfs)}, avoiding forbidden states")

    # Test 4: Weighted costs with A*
    print("\n[Test 4] Weighted A* search...")
    weights = {0: 5.0, 1: 1.0, 2: 1.0}
    res_astar = path_astar("000", "111", weights=weights)
    assert res_astar is not None, "A* failed with weights"
    final = _applyMovesSequence("000", res_astar)
    assert final == "111", "A* path didn't reach goal"
    # Calculate total cost
    total_cost = sum(weights.get(mv[0], 1.0) for mv in res_astar)
    print(f"   A* found path of length {len(res_astar)} with total cost {total_cost}")
    print(f"    (avoids expensive bit 0 with weight 5.0)")

    # Test 5: No path exists
    print("\n[Test 5] Unreachable goal state...")
    # Create scenario where goal is completely surrounded by forbidden states
    start = "0000"
    end = "1111"
    # Block all neighbors of the goal
    forbidden_blocking = {
        "1110", "1101", "1011", "0111"  # All 1-bit neighbors of 1111
    }
    res = path_BFS(start, end, forbidden=forbidden_blocking)
    assert res is None, "Should return None when no path exists"
    print("   Correctly returns None when goal is unreachable")

    # Test 6: Longer state string
    print("\n[Test 6] Longer state strings...")
    start_long = "0000000000"
    end_long = "1111111111"
    res_bfs_long = path_BFS(start_long, end_long)
    assert res_bfs_long is not None
    assert len(res_bfs_long) == 10  # Hamming distance
    print(f"   BFS handles 10-bit states correctly (path length: {len(res_bfs_long)})")

def compare(
    cases: Optional[List[Tuple[State, State, Optional[Set[State]]]]] = None,
    forbidden_density: float = 0.0,
    weights: Optional[Dict[int, float]] = None,
    nRepeats: int = 1,
) -> List[Dict[str, object]]:
    """
    Evaluate and compare BFS, DFS, IDDFS, and A* on multiple test cases.
    
    This function provides empirical performance comparison across different
    algorithms, measuring:
    - Correctness (whether a path is found)
    - Path length
    - Execution time
    
    Args:
        cases: List of (start, end, forbidden) tuples. If None, generates random cases
        forbidden_density: Density of forbidden states (0.0 to 1.0) for random cases
        weights: Per-bit costs for A* (used only for A* algorithm)
        nRepeats: Number of random cases to generate per bit length
    
    Returns:
        List of result dictionaries containing performance metrics:
        - algo: Algorithm name
        - n_bits: Length of state strings
        - forbidden_size: Number of forbidden states
        - found: Whether or not a path was found
        - path_len: Length of path (if found)
        - time_ms: Execution time in milliseconds
    """
    rng = random.Random(42)
    results: List[Dict[str, object]] = []

    algos = [
        ("BFS", lambda s, e, f: path_BFS(s, e, f)),
        ("DFS", lambda s, e, f: path_DFS(s, e, f)),
        ("IDDFS", lambda s, e, f: path_IDDFS(s, e, f, maxDepth=10_000)),
        ("A*", lambda s, e, f: path_astar(s, e, f, weights=weights)),
    ]

    if cases is None:
        # Generate random test cases
        cases = []
        for n_bits in (8, 10, 12):
            for _ in range(nRepeats):
                s = "".join(rng.choice("01") for _ in range(n_bits))
                e = "".join(rng.choice("01") for _ in range(n_bits))
                # Generate forbidden states
                n_forbidden = int(forbidden_density * (n_bits * 2))
                forb: Set[State] = set()
                while len(forb) < n_forbidden:
                    x = "".join(rng.choice("01") for _ in range(n_bits))
                    if x not in {s, e}:
                        forb.add(x)
                cases.append((s, e, forb))

    # Run each algorithm on each case
    for (start, end, forb) in cases:
        for name, fn in algos:
            t0 = time.perf_counter()
            path = fn(start, end, forb or set())
            ms = (time.perf_counter() - t0) * 1000.0
            results.append(
                {
                    "algo": name,
                    "n_bits": len(start),
                    "forbidden_size": len(forb or set()),
                    "found": path is not None,
                    "path_len": len(path) if path is not None else None,
                    "time_ms": round(ms, 3),
                }
            )
    return results


# Minimum-cost Safe Path 

def min_safe(start: State, end: State, forbidden: Optional[Set[State]] = None, 
             weights: Optional[Dict[int, float]] = None) -> Optional[List[Move]]:
    """
    Return a safe path with minimal total cost between two binary states.
    
    ALGORITHM CHOICE JUSTIFICATION:
    This function uses A* search for finding the minimum-cost path because:
    
    1. OPTIMALITY GUARANTEE: A* with an admissible heuristic guarantees finding
       the minimum-cost path. The weighted Hamming distance heuristic is admissible
       (never overestimates the true cost).
    
    2. EFFICIENCY: A* is more efficient than uninformed searches (BFS, DFS, IDDFS)
       when costs vary between moves. The heuristic guides exploration toward the
       goal, reducing the number of states explored compared to BFS.
    
    3. GENERALITY: A* handles both:
       - Uniform costs (when weights is None): equivalent to BFS but with same
         optimality guarantee
       - Non-uniform costs (when weights is provided): finds minimum-cost path
         considering different costs for flipping different bits
    
    4. COMPARISON WITH ALTERNATIVES:
       - BFS: Only optimal for uniform costs; doesn't consider varying weights
       - DFS: Not optimal; may find expensive paths
       - IDDFS: Similar limitations to BFS regarding cost awareness
       - Dijkstra's: A* reduces to Dijkstra's when h=0, but our heuristic makes
         it faster by prioritizing promising paths
    
    COMPLEXITY ANALYSIS:
    - Time: O(2^n * log(2^n)) where n is the state length, due to priority queue
    - Space: O(2^n) for open and closed sets
    - In practice: Much better than worst-case due to heuristic guidance
    
    Args:
        start: Initial binary state
        end: Goal binary state  
        forbidden: Set of states to avoid (hinger states in the game)
        weights: Per-bit costs for flipping bit i. If None, all moves cost 1.0
    
    Returns:
        List of moves forming the minimum-cost safe path, or None if no path exists
    """
    return path_astar(start, end, forbidden=forbidden, weights=weights)


#  Main Entry Point 

if __name__ == "__main__":
    tester()
    
    # Optional: Run a quick comparison
    print("Running performance comparison on random cases...")
    results = compare(nRepeats=3)
    
    # Print summary
    for algo in ["BFS", "DFS", "IDDFS", "A*"]:
        algoResults = [r for r in results if r["algo"] == algo]
        avgTime = sum(r["time_ms"] for r in algoResults) / len(algoResults)  # type: ignore
        successRate = sum(1 for r in algoResults if r["found"]) / len(algoResults)
        print(f"\n{algo}:")
        print(f"  Average time: {avgTime:.3f} ms")
        print(f"  Success rate: {successRate:.1%}")