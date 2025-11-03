"""
Hinger Project
Coursework 001 for: CMP-6058A Artificial Intelligence

Task 4: Main Game Loop - Optimized and Enhanced

Group Number:
Student ID: 100430249 (Robert Soanes)
Partner IDs: 100889423 (Dylan Young)

@date: 29/09/2025

This module implements the core gameplay loop for the Hinger game, allowing
two agents (or one agent and a human player) to take turns making moves on a
shared game state.

Features:
- Alternating turns between two players
- Win/Draw/Illegal move detection
- Time limit enforcement per move
- JSON history logging for visual replay
- Comprehensive test suite
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import a1_state
import a3_agent


# ============================================================================
# Type Definitions & Constants
# ============================================================================

Move = Union[Tuple[int, int], Tuple[int, int, int], str, Dict[str, int]]
Coordinate = Tuple[int, int]

class GameResult(Enum):
    """Possible game outcomes."""
    WIN = "win"
    DRAW = "draw"
    ILLEGAL_MOVE = "illegal_move"
    TIMEOUT = "timeout"

@dataclass
class MoveRecord:
    """Structured representation of a single move."""
    turn: int
    player: str
    coord: Optional[Coordinate]
    startTime: float
    endTime: float
    flags: List[str]
    
    @property
    def elapsedSec(self) -> float:
        return self.endTime - self.startTime
    
    def toDict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "turn": self.turn,
            "player": self.player,
            "coord": list(self.coord) if self.coord else None,
            "start_ts": self.startTime,
            "end_ts": self.endTime,
            "elapsed_sec": round(self.elapsedSec, 3),
            "flags": self.flags,
        }


# ============================================================================
# Core Game Logic
# ============================================================================

class HingerGame:
    """
    Main game controller for Hinger.
    
    Manages game state, turn alternation, move validation, and win/draw
    detection. Supports both AI agents and human players.
    """
    
    def __init__(
        self,
        state: a1_state.State,
        player1: Optional[a3_agent.Agent] = None,
        player2: Optional[a3_agent.Agent] = None,
        *,
        player1Name: str = "Player1",
        player2Name: str = "Player2",
        player1Mode: str = "alphabeta",
        player2Mode: str = "alphabeta",
        timeLimit: float = 60.0,
        verbose: bool = True,
    ):
        """
        Initialize a new Hinger game.
        
        Args:
            state: Initial game state
            player1: Agent for player 1 (None = human player)
            player2: Agent for player 2 (None = human player)
            player1Name: Display name for player 1
            player2Name: Display name for player 2
            player1Mode: AI mode for player 1 ("minimax" or "alphabeta")
            player2Mode: AI mode for player 2 ("minimax" or "alphabeta")
            timeLimit: Maximum time per move in seconds
            verbose: Whether to print game progress
        """
        self.state = state
        self.players = [player1, player2]
        self.playerNames = [
            self._getAgentName(player1, player1Name),
            self._getAgentName(player2, player2Name)
        ]
        self.playerModes = [player1Mode, player2Mode]
        self.timeLimit = timeLimit
        self.verbose = verbose
        
        # Game state
        self.currentPlayerIdx = 0
        self.turnCount = 0
        self.moveHistory: List[MoveRecord] = []
        self.gameStartTime = time.time()
        
        # Board dimensions
        self.rows = len(state.grid)
        self.cols = len(state.grid[0]) if state.grid else 0
    
    @staticmethod
    def _getAgentName(agent: Optional[a3_agent.Agent], default: str) -> str:
        """Extract agent name or use default."""
        if agent is None:
            return default
        return getattr(agent, "name", default)
    
    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    # ------------------------------------------------------------------------
    # Move Parsing & Validation
    # ------------------------------------------------------------------------
    
    @staticmethod
    def parseCoordinate(move: Move) -> Optional[Coordinate]:
        """
        Parse various move formats into (row, col) coordinate.
        
        Supports:
        - Tuples: (r, c) or (r, c, cost)
        - Dicts: {"row": r, "col": c} or {"r": r, "c": c}
        - Strings: "r,c"
        
        Returns:
            (row, col) tuple or None if parsing fails
        """
        # Tuple or list format
        if isinstance(move, (tuple, list)) and len(move) >= 2:
            try:
                return int(move[0]), int(move[1])
            except (ValueError, TypeError):
                return None
        
        # Dictionary format
        if isinstance(move, dict):
            for rowKey, colKey in [("row", "col"), ("r", "c")]:
                if rowKey in move and colKey in move:
                    try:
                        return int(move[rowKey]), int(move[colKey])
                    except (ValueError, TypeError):
                        return None
        
        # String format "r,c"
        if isinstance(move, str) and "," in move:
            parts = move.replace(" ", "").split(",")
            if len(parts) == 2:
                try:
                    return int(parts[0]), int(parts[1])
                except ValueError:
                    return None
        
        return None
    
    def getLegalMoves(self) -> List[Coordinate]:
        """Get list of legal (row, col) coordinates for current state."""
        legal = []
        for move in self.state.moves():
            if isinstance(move, (tuple, list)) and len(move) >= 2:
                legal.append((int(move[0]), int(move[1])))
        return legal
    
    def isValidMove(self, move: Move) -> bool:
        """
        Check if a move is valid in the current state.
        
        A move is valid if:
        1. It can be parsed to coordinates
        2. Coordinates are within bounds
        3. The cell contains a counter (value > 0)
        4. It appears in the legal moves list
        """
        coord = self.parseCoordinate(move)
        if coord is None:
            return False
        
        r, c = coord
        
        # Bounds check
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False
        
        # Cell must have a counter
        if self.state.grid[r][c] <= 0:
            return False
        
        # Must be in legal moves
        return coord in self.getLegalMoves()
    
    def isHingerMove(self, coord: Coordinate) -> bool:
        """
        Check if playing at coord would trigger a hinger (win).
        
        A hinger occurs when removing a counter increases the number
        of disconnected regions on the board.
        """
        r, c = coord
        
        # Must be a cell with value 1
        if self.state.grid[r][c] != 1:
            return False
        
        # Count regions before and after
        currentRegions = self.state.numRegions()
        
        # Simulate the move
        testGrid = [row[:] for row in self.state.grid]
        testGrid[r][c] = 0
        testState = a1_state.State(testGrid)
        newRegions = testState.numRegions()
        
        return newRegions > currentRegions
    
    def isDraw(self) -> bool:
        """Check if the game is a draw (all counters removed)."""
        return all(cell == 0 for row in self.state.grid for cell in row)
    
    # ------------------------------------------------------------------------
    # Turn Management
    # ------------------------------------------------------------------------
    
    def getCurrentPlayerName(self) -> str:
        """Get the name of the current player."""
        return self.playerNames[self.currentPlayerIdx]
    
    def getOpponentName(self) -> str:
        """Get the name of the opponent."""
        return self.playerNames[1 - self.currentPlayerIdx]
    
    def getMoveFromPlayer(self) -> Move:
        """
        Get a move from the current player (AI or human).
        
        Returns:
            Move in agent's preferred format
        """
        player = self.players[self.currentPlayerIdx]
        
        if player is None:
            # Human player
            legal = self.getLegalMoves()
            self._log(f"Legal moves: {legal}")
            while True:
                try:
                    userInput = input("Enter your move (row,col): ").strip()
                    coord = self.parseCoordinate(userInput)
                    if coord and coord in legal:
                        return coord
                    print(f"Invalid move. Please choose from: {legal}")
                except (KeyboardInterrupt, EOFError):
                    raise
        else:
            # AI agent
            mode = self.playerModes[self.currentPlayerIdx]
            return player.move(self.state, mode)
    
    def applyMove(self, coord: Coordinate) -> None:
        """Apply a move to the game state."""
        r, c = coord
        self.state.makeMove(r, c)
    
    def switchPlayer(self) -> None:
        """Switch to the other player."""
        self.currentPlayerIdx = 1 - self.currentPlayerIdx
    
    # ------------------------------------------------------------------------
    # Game Loop
    # ------------------------------------------------------------------------
    
    def playTurn(self) -> Optional[Tuple[GameResult, str]]:
        """
        Execute one turn of the game.
        
        Returns:
            (result, winnerName) if game ends, None otherwise
        """
        self.turnCount += 1
        playerName = self.getCurrentPlayerName()
        opponentName = self.getOpponentName()
        
        self._log(f"\n{'='*60}")
        self._log(f"Turn {self.turnCount}: {playerName}'s move")
        self._log(f"{'='*60}")
        if self.verbose:
            print(self.state)
        
        # Get move with timing
        startTime = time.time()
        try:
            move = self.getMoveFromPlayer()
        except (KeyboardInterrupt, EOFError):
            self._log("\nGame interrupted by user.")
            return GameResult.DRAW, None
        endTime = time.time()
        elapsed = endTime - startTime
        
        coord = self.parseCoordinate(move)
        flags = []
        
        # Check time limit
        if elapsed > self.timeLimit:
            self._log(f"{playerName} exceeded time limit "
                     f"({elapsed:.2f}s > {self.timeLimit:.1f}s)")
            self._log(f"{opponentName} wins by timeout!")
            
            flags.append("timeout")
            self.moveHistory.append(MoveRecord(
                self.turnCount, playerName, coord, startTime, endTime, flags
            ))
            return GameResult.TIMEOUT, opponentName
        
        # Validate move
        if not self.isValidMove(move):
            reason = "invalid format"
            if coord:
                r, c = coord
                if not (0 <= r < self.rows and 0 <= c < self.cols):
                    reason = "out of bounds"
                elif self.state.grid[r][c] <= 0:
                    reason = "cell already empty"
                else:
                    reason = "not in legal moves"
            
            self._log(f"Illegal move by {playerName}: {move} ({reason})")
            self._log(f"{opponentName} wins by illegal move!")
            
            flags.append("illegal")
            flags.append(reason.replace(" ", "_"))
            self.moveHistory.append(MoveRecord(
                self.turnCount, playerName, coord, startTime, endTime, flags
            ))
            return GameResult.ILLEGAL_MOVE, opponentName
        
        # Check for hinger before applying move
        isHinger = self.isHingerMove(coord)
        
        # Apply the move
        self.applyMove(coord)
        
        self._log(f"{playerName} played {coord}")
        
        # Record the move
        if isHinger:
            flags.append("hinger")
        self.moveHistory.append(MoveRecord(
            self.turnCount, playerName, coord, startTime, endTime, flags
        ))
        
        # Check for win
        if isHinger:
            self._log(f"{playerName} wins by hinger!")
            return GameResult.WIN, playerName
        
        # Check for draw
        if self.isDraw():
            self._log("Draw - all counters removed!")
            return GameResult.DRAW, None
        
        # Continue game
        self.switchPlayer()
        return None
    
    def play(self) -> Tuple[GameResult, Optional[str]]:
        """
        Run the complete game until termination.
        
        Returns:
            (result, winnerName) where winnerName is None for draws
        """
        self._log("=" * 60)
        self._log("HINGER GAME START")
        self._log("=" * 60)
        self._log(f"Player 1: {self.playerNames[0]}")
        self._log(f"Player 2: {self.playerNames[1]}")
        self._log(f"Time limit: {self.timeLimit}s per move")
        self._log("=" * 60)
        
        while True:
            result = self.playTurn()
            if result:
                gameResult, winner = result
                return gameResult, winner
    
    # ------------------------------------------------------------------------
    # History & Serialization
    # ------------------------------------------------------------------------
    
    def getGameHistory(self) -> Dict[str, Any]:
        """
        Generate complete game history for JSON export.
        
        Returns:
            Dictionary containing metadata, moves, and result
        """
        return {
            "meta": {
                "game": "Hinger",
                "players": self.playerNames,
                "time_limit_sec": self.timeLimit,
                "start_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(self.gameStartTime)
                ),
                "rows": self.rows,
                "cols": self.cols,
                "total_turns": self.turnCount,
            },
            "moves": [move.toDict() for move in self.moveHistory],
            "result": self._formatResult(),
        }
    
    def _formatResult(self) -> Dict[str, Any]:
        """Format the game result for history."""
        if not self.moveHistory:
            return {"winner": None, "reason": "no_moves", "duration_sec": 0}
        
        lastMove = self.moveHistory[-1]
        
        # Determine result
        if "hinger" in lastMove.flags:
            winner = lastMove.player
            reason = "hinger"
        elif "timeout" in lastMove.flags:
            winner = self.getOpponentName()
            reason = "timeout"
        elif "illegal" in lastMove.flags:
            winner = self.getOpponentName()
            reason = "illegal_move"
        elif self.isDraw():
            winner = None
            reason = "draw"
        else:
            winner = None
            reason = "incomplete"
        
        return {
            "winner": winner,
            "reason": reason,
            "duration_sec": round(time.time() - self.gameStartTime, 3),
        }
    
    def saveHistory(self, filepath: Union[str, Path]) -> None:
        """Save game history to JSON file."""
        filepath = Path(filepath)
        try:
            with filepath.open("w", encoding="utf-8") as f:
                json.dump(self.getGameHistory(), f, indent=2)
            self._log(f"Game history saved to {filepath}")
        except Exception as e:
            self._log(f"Failed to save history: {e}")


# ============================================================================
# Convenience Functions
# ============================================================================

def playGame(
    state: a1_state.State,
    player1: Optional[a3_agent.Agent],
    player2: Optional[a3_agent.Agent],
    **kwargs
) -> Optional[str]:
    """
    Convenience function matching the original API.
    
    Returns:
        Winner name or None for draw
    """
    game = HingerGame(state, player1, player2, **kwargs)
    result, winner = game.play()
    
    # Save history if path provided
    if "saveHistoryPath" in kwargs:
        game.saveHistory(kwargs["saveHistoryPath"])
    
    return winner


# ============================================================================
# Comprehensive Test Suite
# ============================================================================

def testCoordinateParsing():
    """Test move coordinate parsing."""
    print("\n" + "="*60)
    print("TEST: Coordinate Parsing")
    print("="*60)
    
    testCases = [
        ((2, 3), (2, 3)),
        ((2, 3, 5), (2, 3)),
        ({"row": 1, "col": 2}, (1, 2)),
        ({"r": 0, "c": 4}, (0, 4)),
        ("2,3", (2, 3)),
        ("invalid", None),
        ({}, None),
    ]
    
    for move, expected in testCases:
        result = HingerGame.parseCoordinate(move)
        status = "[PASS]" if result == expected else "[FAIL]"
        print(f"{status} {move!r:30} -> {result}")
        assert result == expected, f"Expected {expected}, got {result}"
    
    print("[PASS] All coordinate parsing tests passed!")


def testMoveValidation():
    """Test move validation logic."""
    print("\n" + "="*60)
    print("TEST: Move Validation")
    print("="*60)
    
    grid = [
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ]
    state = a1_state.State(grid)
    game = HingerGame(state, None, None, verbose=False)
    
    # Valid moves
    validMoves = [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1), (2, 2)]
    for move in validMoves:
        assert game.isValidMove(move), f"Valid move {move} rejected"
        print(f"[PASS] {move} is valid")
    
    # Invalid moves
    invalidMoves = [
        ((0, 2), "empty cell"),
        ((1, 1), "empty cell"),
        ((5, 5), "out of bounds"),
        ((-1, 0), "negative index"),
    ]
    for move, reason in invalidMoves:
        assert not game.isValidMove(move), f"Invalid move {move} accepted"
        print(f"[PASS] {move} correctly rejected ({reason})")
    
    print("[PASS] All move validation tests passed!")


def testHingerDetection():
    """Test hinger move detection."""
    print("\n" + "="*60)
    print("TEST: Hinger Detection")
    print("="*60)
    
    # Create a board with a known hinger
    grid = [
        [1, 1, 0],
        [0, 1, 0],
        [1, 1, 0],
    ]
    state = a1_state.State(grid)
    game = HingerGame(state, None, None, verbose=False)
    
    # (1, 1) should be a hinger - it connects top and bottom regions
    isHinger = game.isHingerMove((1, 1))
    print(f"Move (1,1) is hinger: {isHinger}")
    
    # Test non-hinger moves
    print(f"Move (0,0) is hinger: {game.isHingerMove((0, 0))}")
    print(f"Move (2,0) is hinger: {game.isHingerMove((2, 0))}")
    
    print("[PASS] Hinger detection test completed!")


def testNormalGame():
    """Test a normal game completion."""
    print("\n" + "="*60)
    print("TEST: Normal Game (AI vs AI)")
    print("="*60)
    
    grid = [
        [1, 1, 0, 0, 2],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
    ]
    state = a1_state.State(grid)
    rows, cols = len(grid), len(grid[0])
    
    player1 = a3_agent.Agent((rows, cols), name="AlphaBot")
    player2 = a3_agent.Agent((rows, cols), name="BetaBot")
    
    game = HingerGame(
        state, player1, player2,
        player1Mode="alphabeta",
        player2Mode="alphabeta",
        timeLimit=60.0,
        verbose=True
    )
    
    result, winner = game.play()
    game.saveHistory("testNormalGame.json")
    
    print(f"\nResult: {result.value}")
    print(f"Winner: {winner or 'Draw'}")
    print("[PASS] Normal game test completed!")
    
    return result, winner


def testIllegalMoveGame():
    """Test game termination on illegal move."""
    print("\n" + "="*60)
    print("TEST: Illegal Move Detection")
    print("="*60)
    
    grid = [
        [1, 1],
        [1, 1],
    ]
    state = a1_state.State(grid)
    
    # Create a "bad agent" that tries to play invalid moves
    class BadAgent:
        def __init__(self):
            self.name = "BadAgent"
            self.callCount = 0
        
        def move(self, state, mode):
            self.callCount += 1
            if self.callCount == 1:
                return (0, 0)  # Legal first move
            else:
                return (0, 0)  # Illegal - already played!
    
    rows, cols = len(grid), len(grid[0])
    badAgent = BadAgent()
    goodAgent = a3_agent.Agent((rows, cols), name="GoodAgent")
    
    game = HingerGame(
        state, badAgent, goodAgent,
        player1Mode="alphabeta",
        player2Mode="alphabeta",
        timeLimit=60.0,
        verbose=True
    )
    
    result, winner = game.play()
    game.saveHistory("testIllegalMove.json")
    
    assert result == GameResult.ILLEGAL_MOVE, "Should end with illegal move"
    assert winner == "GoodAgent", "Good agent should win"
    
    print(f"\nResult: {result.value}")
    print(f"Winner: {winner}")
    print("[PASS] Illegal move detection test passed!")


def testTimeoutGame():
    """Test game termination on timeout."""
    print("\n" + "="*60)
    print("TEST: Timeout Detection")
    print("="*60)
    
    grid = [
        [1, 1],
        [1, 1],
    ]
    state = a1_state.State(grid)
    
    # Create a slow agent
    class SlowAgent:
        def __init__(self):
            self.name = "SlowAgent"
        
        def move(self, state, mode):
            time.sleep(0.15)  # Exceed 0.1s limit
            moves = state.moves()
            return (moves[0][0], moves[0][1]) if moves else (0, 0)
    
    rows, cols = len(grid), len(grid[0])
    slowAgent = SlowAgent()
    fastAgent = a3_agent.Agent((rows, cols), name="FastAgent")
    
    game = HingerGame(
        state, slowAgent, fastAgent,
        timeLimit=0.1,  # Very short limit
        verbose=True
    )
    
    result, winner = game.play()
    game.saveHistory("testTimeout.json")
    
    assert result == GameResult.TIMEOUT, "Should end with timeout"
    assert winner == "FastAgent", "Fast agent should win"
    
    print(f"\nResult: {result.value}")
    print(f"Winner: {winner}")
    print("[PASS] Timeout detection test passed!")


def testDrawGame():
    """Test draw detection."""
    print("\n" + "="*60)
    print("TEST: Draw Detection")
    print("="*60)
    
    # Small board that can end in a draw
    grid = [
        [1, 1],
        [1, 0],
    ]
    state = a1_state.State(grid)
    game = HingerGame(state, None, None, verbose=False)
    
    # Manually play out moves
    game.applyMove((0, 0))
    game.applyMove((0, 1))
    game.applyMove((1, 0))
    
    assert game.isDraw(), "Should be a draw with all cells empty"
    
    print("[PASS] Draw detection test passed!")


def runAllTests():
    """Run comprehensive test suite."""
    print("\n" + "="*60)
    print("HINGER GAME - COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    try:
        # Unit tests
        testCoordinateParsing()
        testMoveValidation()
        testHingerDetection()
        testDrawGame()
        
        # Integration tests
        testNormalGame()
        testIllegalMoveGame()
        testTimeoutGame()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("="*60)
        
        print("\nTest output files generated:")
        print("   - testNormalGame.json")
        print("   - testIllegalMove.json")
        print("   - testTimeout.json")
        print("\nUse viewHinger.py to replay these games visually!")
        
    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n[FAIL] UNEXPECTED ERROR: {e}")
        raise


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    runAllTests()