
# viewer_hinger.py
import json
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

GRID_ROWS_DEFAULT = 8
GRID_COLS_DEFAULT = 8
CELL_SIZE = 56
MARGIN = 30
BG_COLOR = "#111111"
GRID_COLOR = "#444444"
A_COLOR = "#3BA7FF"    # Player1
B_COLOR = "#FF7A7A"    # Player2
HIGHLIGHT_COLOR = "#F8E45C"
ILLEGAL_COLOR = "#E6C229"  # illegal/timeout outline
HINGER_RING = "#FFFFFF"    # white ring for hinger
TEXT_COLOR = "#FFFFFF"
PLAY_DELAY_SEC = 0.5

def parseCoordFromMove(move, rows, cols):
    if isinstance(move, (tuple, list)) and len(move) >= 2:
        try: return int(move[0]), int(move[1])
        except Exception: return None
    if isinstance(move, dict):
        for a,b in (("row","col"),("r","c")):
            if a in move and b in move:
                try: return int(move[a]), int(move[b])
                except Exception: return None
    if isinstance(move, str) and "," in move:
        parts = move.replace(" ", "").split(",")
        if len(parts)==2:
            try: return int(parts[0]), int(parts[1])
            except Exception: return None
    try:
        idx = int(move)
        if 0 <= idx < rows*cols:
            return idx//cols, idx%cols
    except Exception:
        pass
    return None

class HingerViewer(tk.Tk):
    def __init__(self, history):
        super().__init__()
        self.title("Hinger Replay Viewer")
        self.configure(bg=BG_COLOR)

        self.history = history
        meta = history.get("meta", {})
        self.players = meta.get("players", ["Player1","Player2"])
        self.moves = history.get("moves", [])
        self.result = history.get("result", {})
        self.rows = int(meta.get("rows", GRID_ROWS_DEFAULT))
        self.cols = int(meta.get("cols", GRID_COLS_DEFAULT))
        self.delay = PLAY_DELAY_SEC

        w = MARGIN*2 + self.cols * CELL_SIZE
        h = MARGIN*2 + self.rows * CELL_SIZE + 110
        self.canvas = tk.Canvas(self, width=w, height=h, bg=BG_COLOR, highlightthickness=0)
        self.canvas.pack()

        ctrl = tk.Frame(self, bg=BG_COLOR)
        ctrl.pack(pady=6)
        self.btnPrev = tk.Button(ctrl, text="⏮ Prev", command=self.prevMove)
        self.btnPlay = tk.Button(ctrl, text="▶ Play", command=self.togglePlay)
        self.btnNext = tk.Button(ctrl, text="Next ⏭", command=self.nextMove)
        self.btnReset = tk.Button(ctrl, text="⟲ Reset", command=self.resetBoard)
        for b in (self.btnPrev, self.btnPlay, self.btnNext, self.btnReset):
            b.pack(side=tk.LEFT, padx=5)

        self.statusVar = tk.StringVar()
        self.status = tk.Label(self, textvariable=self.statusVar, bg=BG_COLOR, fg=TEXT_COLOR)
        self.status.pack()

        self.currentIndex = 0
        self.occupied = {}  # (r,c) -> "A"/"B"
        self.highlightCell = None
        self._playing = False

        self.drawGrid()
        self.updateStatus()

    def drawGrid(self):
        self.canvas.delete("all")
        x0 = MARGIN
        y0 = MARGIN
        x1 = MARGIN + self.cols * CELL_SIZE
        y1 = MARGIN + self.rows * CELL_SIZE
        self.canvas.create_rectangle(x0, y0, x1, y1, outline=GRID_COLOR)
        for c in range(self.cols + 1):
            x = MARGIN + c * CELL_SIZE
            self.canvas.create_line(x, y0, x, y1, fill=GRID_COLOR)
        for r in range(self.rows + 1):
            y = MARGIN + r * CELL_SIZE
            self.canvas.create_line(x0, y, x1, y, fill=GRID_COLOR)
        legendY = y1 + 24
        pA = self.players[0] if len(self.players)>0 else "Player1"
        pB = self.players[1] if len(self.players)>1 else "Player2"
        self.canvas.create_text(MARGIN, legendY, text=f"{pA} = A", fill=A_COLOR, anchor="w", font=("Arial", 11, "bold"))
        self.canvas.create_text(MARGIN + 180, legendY, text=f"{pB} = B", fill=B_COLOR, anchor="w", font=("Arial", 11, "bold"))
        self.canvas.create_text(MARGIN, legendY + 22, text=f"Illegal/Timeout shown with gold outline; Hinger = white ring", fill=ILLEGAL_COLOR, anchor="w", font=("Arial", 10))
        self.redrawPieces()

    def redrawPieces(self):
        self.canvas.delete("pieces")
        self.canvas.delete("highlight")
        for (r, c), who in self.occupied.items():
            cx = MARGIN + c*CELL_SIZE + CELL_SIZE/2
            cy = MARGIN + r*CELL_SIZE + CELL_SIZE/2
            rad = CELL_SIZE*0.35
            color = A_COLOR if who == "A" else B_COLOR
            self.canvas.create_oval(cx-rad, cy-rad, cx+rad, cy+rad, fill=color, outline="", tags="pieces")
        if self.highlightCell:
            r, c = self.highlightCell
            x0 = MARGIN + c*CELL_SIZE + 2
            y0 = MARGIN + r*CELL_SIZE + 2
            x1 = x0 + CELL_SIZE - 4
            y1 = y0 + CELL_SIZE - 4
            self.canvas.create_rectangle(x0, y0, x1, y1, outline=HIGHLIGHT_COLOR, width=3, tags="highlight")

    def coordInBounds(self, rc):
        if rc is None:
            return None
        r, c = rc
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return r, c
        return None

    def applyMoveVisual(self, idx):
        if idx < 0 or idx >= len(self.moves):
            return
        m = self.moves[idx]
        player = m.get("player", "")
        who = "A" if player == self.players[0] else "B"
        flags = set(m.get("flags", []))

        rc = m.get("coord")
        if rc is None:
            rc = parseCoordFromMove(m.get("move"), self.rows, self.cols)
        rc = self.coordInBounds(rc)
        if rc is None:
            self.statusVar.set(f"Turn {m.get('turn')}: {player} -> {m.get('move')} (no coord)")
            return

        r, c = rc
        # illegal/timeout: show gold outline only
        if "illegal" in flags or "timeout" in flags:
            x0 = MARGIN + c*CELL_SIZE + 3
            y0 = MARGIN + r*CELL_SIZE + 3
            x1 = x0 + CELL_SIZE - 6
            y1 = y0 + CELL_SIZE - 6
            self.canvas.create_rectangle(x0, y0, x1, y1, outline=ILLEGAL_COLOR, width=3, tags="pieces")
            self.highlightCell = (r, c)
            self.redrawPieces()
            return

        # normal disk
        self.occupied[(r, c)] = who
        self.highlightCell = (r, c)
        self.redrawPieces()

        # hinger: overlay white ring
        if "hinger" in flags:
            cx = MARGIN + c*CELL_SIZE + CELL_SIZE/2
            cy = MARGIN + r*CELL_SIZE + CELL_SIZE/2
            inner = CELL_SIZE*0.28
            outer = CELL_SIZE*0.42
            self.canvas.create_oval(cx-outer, cy-outer, cx+outer, cy+outer, outline=HINGER_RING, width=3, tags="pieces")
            self.canvas.create_oval(cx-inner, cy-inner, cx+inner, cy+inner, outline="", fill="", tags="pieces")

    def resetBoard(self):
        self.currentIndex = 0
        self.occupied.clear()
        self.highlightCell = None
        self.drawGrid()
        self.updateStatus()

    def nextMove(self):
        if self.currentIndex < len(self.moves):
            self.applyMoveVisual(self.currentIndex)
            self.currentIndex += 1
            self.updateStatus()
            if self.currentIndex >= len(self.moves):
                self._playing = False
                self.btnPlay.config(text="Play")

    def prevMove(self):
        if self.currentIndex > 0:
            self.currentIndex -= 1
            upto = self.currentIndex
            self.occupied.clear()
            self.highlightCell = None
            for i in range(upto):
                self.applyMoveVisual(i)
            self.redrawPieces()
            self.updateStatus()

    def togglePlay(self):
        self._playing = not self._playing
        self.btnPlay.config(text="Pause" if self._playing else "Play")
        if self._playing:
            self.after(int(self.delay * 1000), self._autoPlayStep)

    def _autoPlayStep(self):
        if not self._playing:
            return
        if self.currentIndex < len(self.moves):
            self.nextMove()
            self.after(int(self.delay * 1000), self._autoPlayStep)
        else:
            self._playing = False
            self.btnPlay.config(text="Play")

    def updateStatus(self):
        msg = f"Move {self.currentIndex}/{len(self.moves)}"
        if self.result:
            winner = self.result.get("winner")
            reason = self.result.get("reason")
            duration = self.result.get("duration_sec")
            turns = self.history.get("meta", {}).get("total_turns")
            if winner is None:
                msg += f" | Result: Draw ({reason}, turns={turns}, {duration}s)"
            else:
                msg += f" | Winner: {winner} ({reason}, turns={turns}, {duration}s)"
        self.statusVar.set(msg)

def loadHistory(path=None):
    if path is None:
        root = tk.Tk()
        root.withdraw()
        fname = filedialog.askopenfilename(
            title="Open game history JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        root.destroy()
        if not fname:
            return None
        path = fname
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    hist = loadHistory(path)
    if not hist:
        messagebox.showerror("Error", "No history selected or file could not be loaded.")
        return
    app = HingerViewer(hist)
    app.mainloop()

if __name__ == "__main__":
    main()
