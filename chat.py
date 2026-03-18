import argparse
import threading
import tkinter as tk
from tkinter import font as tkfont
import torch
from model import TinyLM, ModelConfig

#cli

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str,   default="checkpoints/model.pt")
parser.add_argument("--temp",       type=float, default=0.8)
parser.add_argument("--topk",       type=int,   default=40)
parser.add_argument("--max-tokens", type=int,   default=200)
args = parser.parse_args()

#pallete

BG           = "#080c14"
BG_CHAT      = "#080c14"
BG_MSG_USER  = "#13173a"
BG_INPUT     = "#0d1220"
BG_PANEL     = "#09101f"

FG           = "#c7d2fe"
FG_USER      = "#e0e7ff"
FG_AI        = "#94a3b8"
FG_DIM       = "#3d4f6e"
FG_SYSTEM    = "#3b4f6e"

ACCENT       = "#818cf8"
ACCENT_LIGHT = "#a5b4fc"
ACCENT2      = "#60a5fa"
ACCENT_DIM   = "#2e2a6e"
BORDER       = "#1a2236"
BORDER_FOCUS = "#4338ca"
GREEN        = "#34d399"
RED          = "#f87171"

# gradient colour stops

GH_L = "#130a2e"   # header left
GH_R = "#071428"   # header right
GB_L = "#4c1d95"   # button left
GB_R = "#1d4ed8"   # button right
GS_L = "#3730a3"   # separator left
GS_R = "#1d4ed8"   # separator right

            # torch load

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    import torch_directml
    device = torch_directml.device()
except Exception:
    pass

model = None; vocab = None; inv_vocab = None
encode = None; decode = None; model_loaded = False


def load_model():
    global model, vocab, inv_vocab, encode, decode, model_loaded
    try:
        ck      = torch.load(args.checkpoint, map_location=device)
        vocab   = ck["vocab"]; inv_vocab = ck["inv_vocab"]
        encode  = lambda s: [vocab.get(c, 0) for c in s]
        decode  = lambda l: "".join(inv_vocab.get(i, "?") for i in l)
        model   = TinyLM(ck["config"]).to(device)
        model.load_state_dict(ck["model_state"])
        model.eval(); model_loaded = True
        return True, ck.get("val_loss", "?"), ck.get("step", "?")
    except FileNotFoundError:
        return False, None, None


def generate(prompt):
    if not model_loaded:
        return "Model not loaded.  Run python train.py first."
    enc = encode(prompt)
    idx = torch.tensor([enc], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=args.max_tokens,
                         temperature=args.temp, top_k=args.topk)
    return decode(out[0][len(enc):].tolist())


def _hex_rgb(h):
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _lerp_color(c1, c2, t):
    r1, g1, b1 = _hex_rgb(c1)
    r2, g2, b2 = _hex_rgb(c2)
    return "#{:02x}{:02x}{:02x}".format(
        int(r1 + (r2 - r1) * t),
        int(g1 + (g2 - g1) * t),
        int(b1 + (b2 - b1) * t),
    )


def _h_gradient_image(width, height, c1, c2):
    if width < 1:
        width = 1
    img = tk.PhotoImage(width=width, height=height)
    r1, g1, b1 = _hex_rgb(c1)
    r2, g2, b2 = _hex_rgb(c2)
    cols = [
        "#{:02x}{:02x}{:02x}".format(
            int(r1 + (r2 - r1) * x / max(width - 1, 1)),
            int(g1 + (g2 - g1) * x / max(width - 1, 1)),
            int(b1 + (b2 - b1) * x / max(width - 1, 1)),
        )
        for x in range(width)
    ]
    row = "{" + " ".join(cols) + "}"
    for y in range(height):
        img.put(row, to=(0, y))
    return img

        # canvas styles

class HGradCanvas(tk.Canvas):

    def __init__(self, parent, c1, c2, height, **kw):
        super().__init__(parent, height=height,
                         highlightthickness=0, bd=0, **kw)
        self._c1, self._c2 = c1, c2
        self._bg_img = None
        self._bg_id  = None
        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, _e=None):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 2 or h < 2:
            return
        img = _h_gradient_image(w, h, self._c1, self._c2)
        self._bg_img = img
        if self._bg_id is None:
            self._bg_id = self.create_image(0, 0, anchor="nw", image=img)
        else:
            self.itemconfig(self._bg_id, image=img)
        self.tag_lower(self._bg_id)
        self._on_gradient_ready()

    def _on_gradient_ready(self):
        pass


        # header

class Header(HGradCanvas):
    def __init__(self, parent, **kw):
        super().__init__(parent, c1=GH_L, c2=GH_R, height=58, **kw)
        self._dot_id  = None
        self._stat_id = None

    def _on_gradient_ready(self):
        self.delete("content")
        w = self.winfo_width()

        # Logo glow ring + filled dot
        self.create_oval(22, 16, 42, 36,
                         fill="#200d50", outline=ACCENT, width=1,
                         tags="content")
        self.create_oval(27, 21, 37, 31,
                         fill=ACCENT, outline="", tags="content")
        self.create_oval(29, 23, 35, 29,
                         fill=GH_L, outline="", tags="content")

        # header text
        self.create_text(52, 29, text="DARKGPT",
                         font=_FONTS.get("title", ("TkDefaultFont", 14, "bold")),
                         fill="#dde6ff", anchor="w", tags="content")

        # glow lines
        h = self.winfo_height()
        self.create_line(0, h - 2, w, h - 2, fill="#312e81", tags="content")
        self.create_line(0, h - 1, w, h - 1, fill="#1e1b4b", tags="content")

        # Status
        sx = w - 130
        if self._dot_id is None:
            self._dot_id = self.create_oval(
                sx, 22, sx + 14, 36,
                fill=FG_DIM, outline="", tags="content")
            self._stat_id = self.create_text(
                sx + 22, 29, text="LOADING",
                font=_FONTS.get("small", ("TkDefaultFont", 9, "bold")),
                fill=FG_DIM, anchor="w", tags="content")
        else:
            self.coords(self._dot_id,  sx, 22, sx + 14, 36)
            self.coords(self._stat_id, sx + 22, 29)

    def set_status(self, label, color):
        if self._dot_id and self._stat_id:
            self.itemconfig(self._dot_id,  fill=color)
            self.itemconfig(self._stat_id, text=label, fill=color)


        #gradient

class GradSep(HGradCanvas):
    def __init__(self, parent, **kw):
        super().__init__(parent, c1=GS_L, c2=GS_R, height=2, **kw)


        #gradient

class GradButton(tk.Canvas):
    W, H = 96, 46

    def __init__(self, parent, text="SEND", command=None, **kw):
        super().__init__(parent, width=self.W, height=self.H,
                         highlightthickness=0, bd=0,
                         cursor="hand2", **kw)
        self._text     = text
        self._cmd      = command
        self._hover    = False
        self._disabled = False
        self._img      = None
        self._sheen    = None
        self._draw()
        self.bind("<Enter>",           self._on_enter)
        self.bind("<Leave>",           self._on_leave)
        self.bind("<ButtonRelease-1>", self._on_click)

    def _draw(self):
        self.delete("all")
        w, h = self.W, self.H

        if self._disabled:
            c1, c2, fg = "#1e1b3a", "#1a2740", FG_DIM
        else:
            c1, c2, fg = GB_L, GB_R, "#ffffff"
            if self._hover:
                c1 = _lerp_color(GB_L, "#ffffff", 0.18)
                c2 = _lerp_color(GB_R, "#ffffff", 0.18)

                # effects

        if self._hover and not self._disabled:
            for i in range(5, 0, -1):
                halo = _lerp_color(ACCENT, BG, 1 - i / 7)
                self.create_rectangle(-i, -i, w + i - 1, h + i - 1,
                                      fill="", outline=halo)

        img = _h_gradient_image(w, h, c1, c2)
        self._img = img
        self.create_image(0, 0, anchor="nw", image=img)

        sheen = _h_gradient_image(
            w, 2,
            _lerp_color("#ffffff", c1, 0.7),
            _lerp_color("#ffffff", c2, 0.7),
        )
        self._sheen = sheen
        self.create_image(0, 0, anchor="nw", image=sheen)

        self.create_text(
            w // 2, h // 2 + 1,
            text=self._text,
            font=_FONTS.get("btn", ("TkDefaultFont", 10, "bold")),
            fill=fg,
        )

    def _on_enter(self, _e):
        if not self._disabled:
            self._hover = True
            self._draw()

    def _on_leave(self, _e):
        self._hover = False
        self._draw()

    def _on_click(self, _e):
        if not self._disabled and self._cmd:
            self._cmd()

    def set_state(self, state, label=None):
        self._disabled = state == tk.DISABLED
        if label:
            self._text = label
        self._draw()

        #fonts

_FONTS: dict = {}


def _init_fonts():
    fams = tkfont.families()
    ui = next(
        (f for f in ["Segoe UI", "SF Pro Text", "Helvetica Neue",
                     "Cantarell", "Ubuntu", "DejaVu Sans"]
         if f in fams),
        "TkDefaultFont",
    )
    _FONTS["body"]  = (ui, 11)
    _FONTS["small"] = (ui, 9)
    _FONTS["label"] = (ui, 9, "bold")
    _FONTS["title"] = (ui, 14, "bold")
    _FONTS["btn"]   = (ui, 10, "bold")

class DarkGPTApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("DarkGPT")
        self.root.configure(bg=BG)
        self.root.geometry("980x740")
        self.root.minsize(640, 480)

        self._generating = False
        self._dot_phase  = 0
        self._dot_job    = None

        _init_fonts()
        self._build_ui()
        self._load_model_async()

        # layout styles

    def _build_ui(self):
        self._build_header()
        GradSep(self.root, bg=BG).pack(fill=tk.X)
        self._build_chat()
        GradSep(self.root, bg=BG).pack(fill=tk.X)
        self._build_bottom()
        self._append_system("◈  DarkGPT initialising...\n\n")

    def _build_header(self):
        self.header = Header(self.root, bg=GH_L)
        self.header.pack(fill=tk.X)

    def _build_chat(self):
        from tkinter import scrolledtext
        wrapper = tk.Frame(self.root, bg=BG_CHAT)
        wrapper.pack(fill=tk.BOTH, expand=True)

        self.chat = scrolledtext.ScrolledText(
            wrapper,
            font=_FONTS["body"],
            bg=BG_CHAT, fg=FG,
            insertbackground=ACCENT,
            relief=tk.FLAT, bd=0,
            padx=44, pady=24,
            wrap=tk.WORD,
            state=tk.DISABLED,
            cursor="arrow",
            spacing1=2, spacing3=2,
        )
        self.chat.pack(fill=tk.BOTH, expand=True)

        # scrollbar

        self.chat.vbar.config(
            bg=BG_PANEL, troughcolor=BG_CHAT,
            width=5, relief=tk.FLAT,
            activebackground=BORDER,
            highlightthickness=0, bd=0,
        )

        # Text tags

        t = self.chat
        t.tag_config("user_label",
                     foreground=ACCENT_LIGHT, font=_FONTS["label"],
                     spacing1=16, spacing3=4)
        t.tag_config("user_text",
                     foreground=FG_USER, font=_FONTS["body"],
                     background=BG_MSG_USER,
                     lmargin1=0, lmargin2=0, rmargin=0,
                     spacing1=8, spacing3=10)
        t.tag_config("user_gap",
                     foreground=BG_CHAT, background=BG_CHAT,
                     font=(_FONTS["body"][0], 3))
        t.tag_config("ai_label",
                     foreground=FG_DIM, font=_FONTS["label"],
                     spacing1=16, spacing3=4)
        t.tag_config("ai_text",
                     foreground=FG_AI, font=_FONTS["body"],
                     spacing1=4, spacing3=4)
        t.tag_config("system_text",
                     foreground=FG_SYSTEM, font=_FONTS["small"],
                     spacing1=1, spacing3=1)
        t.tag_config("divider",
                     foreground=BORDER, font=_FONTS["small"],
                     spacing1=10, spacing3=10)

    def _build_bottom(self):
        panel = tk.Frame(self.root, bg=BG_PANEL)
        panel.pack(fill=tk.X, side=tk.BOTTOM)

        # Input row

        row = tk.Frame(panel, bg=BG_PANEL, pady=14)
        row.pack(fill=tk.X, padx=26)

        self._input_border = tk.Frame(
            row, bg=BORDER, padx=1, pady=1, highlightthickness=0
        )
        self._input_border.pack(side=tk.LEFT, fill=tk.X, expand=True)

        inner = tk.Frame(self._input_border, bg=BG_INPUT)
        inner.pack(fill=tk.BOTH)

        self.input_box = tk.Text(
            inner,
            font=_FONTS["body"],
            bg=BG_INPUT, fg=FG_USER,
            insertbackground=ACCENT,
            relief=tk.FLAT, bd=0,
            padx=16, pady=12,
            height=3, wrap=tk.WORD,
        )
        self.input_box.pack(fill=tk.BOTH, expand=True)
        self.input_box.bind("<Return>",       self._on_enter)
        self.input_box.bind("<Shift-Return>", self._on_shift_enter)
        self.input_box.bind(
            "<FocusIn>",
            lambda _e: self._input_border.config(bg=BORDER_FOCUS))
        self.input_box.bind(
            "<FocusOut>",
            lambda _e: self._input_border.config(bg=BORDER))
        self.input_box.focus_set()

        self.send_btn = GradButton(
            row, text="SEND", command=self._send, bg=BG_PANEL
        )
        self.send_btn.pack(side=tk.RIGHT, padx=(14, 0))

        # Animated dots

        self._dots_cv = tk.Canvas(
            panel, height=20, bg=BG_PANEL,
            highlightthickness=0, bd=0
        )
        self._dots_cv.pack(fill=tk.X, padx=30)

        tk.Label(
            panel,
            text="ENTER  send    ·    SHIFT+ENTER  new line",
            font=_FONTS["small"], fg=FG_DIM, bg=BG_PANEL, pady=7,
        ).pack()

    # animation

    def _start_dots(self):
        self._dot_phase = 0
        self._tick_dots()

    def _tick_dots(self):
        if not self._generating:
            self._dots_cv.delete("all")
            return
        c = self._dots_cv
        c.delete("all")
        cx, cy = 38, 10
        for i in range(3):
            phase = (self._dot_phase - i * 3) % 12
            t     = abs(6 - phase) / 6.0
            r     = int(3 + 3 * t)
            color = _lerp_color(ACCENT_DIM, ACCENT2, t)
            c.create_oval(cx + i * 22 - r, cy - r,
                          cx + i * 22 + r, cy + r,
                          fill=color, outline="")
        self._dot_phase = (self._dot_phase + 1) % 12
        self._dot_job   = self.root.after(90, self._tick_dots)

    def _stop_dots(self):
        if self._dot_job:
            self.root.after_cancel(self._dot_job)
            self._dot_job = None
        self._dots_cv.delete("all")

    def _load_model_async(self):
        def _load():
            result = load_model()
            self.root.after(0, lambda: self._on_model_loaded(*result))
        threading.Thread(target=_load, daemon=True).start()

    def _on_model_loaded(self, ok, val_loss, step):
        if ok:
            self._append_system(f"Model loaded \n")
            self._append_system("Type a message and press ENTER.\n\n")
        else:
            self.header.set_status("NO MODEL", RED)
            self._append_system(
                f"Checkpoint not found: {args.checkpoint}\n\n")

    def _on_enter(self, _e):
        self._send()
        return "break"

    def _on_shift_enter(self, _e):
        return None

    def _send(self):
        text = self.input_box.get("1.0", tk.END).strip()
        if not text or not model_loaded or self._generating:
            return
        self._generating = True
        self.input_box.delete("1.0", tk.END)
        self.send_btn.set_state(tk.DISABLED, "···")
        self.header.set_status("THINKING", ACCENT)
        self._start_dots()
        self._append_user(text)

        def _run():
            resp = generate(text)
            self.root.after(0, lambda: self._on_response(resp))
        threading.Thread(target=_run, daemon=True).start()

    def _on_response(self, resp):
        self._generating = False
        self._stop_dots()
        self.header.set_status("READY", GREEN)
        self._append_ai(resp)
        self.send_btn.set_state(tk.NORMAL, "SEND")
        self.input_box.focus_set()

    def _append_user(self, text):
        t = self.chat
        t.config(state=tk.NORMAL)
        t.insert(tk.END, "  YOU\n",          "user_label")
        t.insert(tk.END, "  " + text + "\n", "user_text")
        t.insert(tk.END, "\n",               "user_gap")
        t.config(state=tk.DISABLED)
        t.see(tk.END)

    def _append_ai(self, text):
        t = self.chat
        t.config(state=tk.NORMAL)
        t.insert(tk.END, "  DARKGPT\n",              "ai_label")
        t.insert(tk.END, "  " + text.strip() + "\n", "ai_text")
        t.insert(tk.END, "\n  " + "─" * 54 + "\n\n", "divider")
        t.config(state=tk.DISABLED)
        t.see(tk.END)

    def _append_system(self, text):
        t = self.chat
        t.config(state=tk.NORMAL)
        t.insert(tk.END, "  " + text, "system_text")
        t.config(state=tk.DISABLED)
        t.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.iconbitmap("icon.ico")
    except Exception:
        pass
    app = DarkGPTApp(root)
    root.mainloop()
