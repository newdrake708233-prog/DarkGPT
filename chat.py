import argparse
import threading
import tkinter as tk
from tkinter import scrolledtext
import torch
from model import TinyLM, ModelConfig


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str,   default="checkpoints/model.pt")
parser.add_argument("--temp",       type=float, default=0.8)
parser.add_argument("--topk",       type=int,   default=40)
parser.add_argument("--max-tokens", type=int,   default=200)
args = parser.parse_args()


BG          = "#0a0a0a"
BG_PANEL    = "#111111"
BG_INPUT    = "#1a1a1a"
FG          = "#f0f0f0"
FG_DIM      = "#666666"
FG_USER     = "#ffffff"
FG_AI       = "#cccccc"
ACCENT      = "#ffffff"
BORDER      = "#2a2a2a"
TAG_USER    = "#ffffff"
TAG_AI      = "#888888"
FONT_MONO   = ("Courier New", 11)
FONT_MONO_S = ("Courier New", 9)
FONT_MONO_L = ("Courier New", 13, "bold")
FONT_UI     = ("Courier New", 10)

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    import torch_directml
    device = torch_directml.device()
except Exception:
    pass

model     = None
vocab     = None
inv_vocab = None
encode    = None
decode    = None
model_loaded = False

def load_model():
    global model, vocab, inv_vocab, encode, decode, model_loaded
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        cfg        = checkpoint["config"]
        vocab      = checkpoint["vocab"]
        inv_vocab  = checkpoint["inv_vocab"]
        encode     = lambda s: [vocab.get(c, 0) for c in s]
        decode     = lambda l: "".join(inv_vocab.get(i, "?") for i in l)
        model      = TinyLM(cfg).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        model_loaded = True
        return True, checkpoint.get("val_loss", "?"), checkpoint.get("step", "?")
    except FileNotFoundError:
        return False, None, None

def generate(prompt: str) -> str:
    if not model_loaded:
        return "Model not loaded. Run python train.py first."
    encoded  = encode(prompt)
    idx      = torch.tensor([encoded], dtype=torch.long, device=device)
    output   = model.generate(idx, max_new_tokens=args.max_tokens,
                               temperature=args.temp, top_k=args.topk)
    new_toks = output[0][len(encoded):].tolist()
    return decode(new_toks)

class DarkGPTApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DarkGPT")
        self.root.configure(bg=BG)
        self.root.geometry("900x700")
        self.root.minsize(600, 400)

        self._build_ui()
        self._load_model_async()

    def _build_ui(self):
        # ── Top bar ──
        topbar = tk.Frame(self.root, bg=BG_PANEL, height=48)
        topbar.pack(fill=tk.X, side=tk.TOP)
        topbar.pack_propagate(False)

        tk.Label(
            topbar, text="DARKGPT", font=("Courier New", 14, "bold"),
            fg=FG, bg=BG_PANEL, padx=20
        ).pack(side=tk.LEFT, pady=12)

        self.status_label = tk.Label(
            topbar, text="● LOADING MODEL...", font=FONT_MONO_S,
            fg=FG_DIM, bg=BG_PANEL, padx=20
        )
        self.status_label.pack(side=tk.RIGHT, pady=12)

        # ── Divider ──
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill=tk.X)

        # ── Chat area ──
        self.chat_frame = tk.Frame(self.root, bg=BG)
        self.chat_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        self.chat_display = scrolledtext.ScrolledText(
            self.chat_frame,
            font=FONT_MONO,
            bg=BG,
            fg=FG,
            insertbackground=FG,
            relief=tk.FLAT,
            bd=0,
            padx=32,
            pady=24,
            wrap=tk.WORD,
            state=tk.DISABLED,
            cursor="arrow",
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)

        self.chat_display.tag_config("user_label",  foreground="#ffffff", font=("Courier New", 9, "bold"))
        self.chat_display.tag_config("user_text",   foreground=FG_USER,   font=FONT_MONO)
        self.chat_display.tag_config("ai_label",    foreground=FG_DIM,    font=("Courier New", 9, "bold"))
        self.chat_display.tag_config("ai_text",     foreground=FG_AI,     font=FONT_MONO)
        self.chat_display.tag_config("system_text", foreground=FG_DIM,    font=FONT_MONO_S)
        self.chat_display.tag_config("thinking",    foreground="#444444",  font=FONT_MONO_S)
        self.chat_display.tag_config("divider",     foreground=BORDER,     font=FONT_MONO_S)

        tk.Frame(self.root, bg=BORDER, height=1).pack(fill=tk.X)

        input_frame = tk.Frame(self.root, bg=BG_PANEL, pady=16)
        input_frame.pack(fill=tk.X, side=tk.BOTTOM)

        row = tk.Frame(input_frame, bg=BG_PANEL)
        row.pack(fill=tk.X, padx=24)

        input_container = tk.Frame(row, bg=BG_INPUT, highlightbackground=BORDER,
                                    highlightthickness=1)
        input_container.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.input_box = tk.Text(
            input_container,
            font=FONT_MONO,
            bg=BG_INPUT,
            fg=FG,
            insertbackground=FG,
            relief=tk.FLAT,
            bd=0,
            padx=14,
            pady=10,
            height=3,
            wrap=tk.WORD,
        )
        self.input_box.pack(fill=tk.BOTH, expand=True)
        self.input_box.bind("<Return>",       self._on_enter)
        self.input_box.bind("<Shift-Return>", self._on_shift_enter)
        self.input_box.focus_set()

        self.send_btn = tk.Button(
            row,
            text="SEND",
            font=("Courier New", 10, "bold"),
            fg=BG,
            bg=FG,
            activeforeground=BG,
            activebackground="#cccccc",
            relief=tk.FLAT,
            bd=0,
            padx=20,
            pady=10,
            cursor="hand2",
            command=self._send,
        )
        self.send_btn.pack(side=tk.RIGHT, padx=(12, 0))

        hint = tk.Frame(self.root, bg=BG_PANEL)
        hint.pack(fill=tk.X, side=tk.BOTTOM)
        tk.Label(
            hint,
            text="ENTER to send  ·  SHIFT+ENTER for new line",
            font=FONT_MONO_S,
            fg=FG_DIM,
            bg=BG_PANEL,
            pady=6,
        ).pack()

        self._append_system("Initialising DarkGPT...\n")

    def _load_model_async(self):
        def load():
            ok, val_loss, step = load_model()
            self.root.after(0, lambda: self._on_model_loaded(ok, val_loss, step))
        threading.Thread(target=load, daemon=True).start()

    def _on_model_loaded(self, ok, val_loss, step):
        if ok:
            self.status_label.config(text=f"● READY  |  step {step}  |  loss {val_loss:.4f}", fg="#aaaaaa")
            self._append_system(f"Model loaded — trained for {step} steps (val loss: {val_loss:.4f})\n")
            self._append_system("Type a message and press ENTER to begin.\n\n")
        else:
            self.status_label.config(text="● NO MODEL FOUND", fg="#ff4444")
            self._append_system("No checkpoint found at: " + args.checkpoint + "\n")
            self._append_system("Run  python train.py  first to train your model.\n\n")

    def _on_enter(self, event):
        self._send()
        return "break"

    def _on_shift_enter(self, event):
        return None

    def _send(self):
        text = self.input_box.get("1.0", tk.END).strip()
        if not text or not model_loaded:
            return

        self.input_box.delete("1.0", tk.END)
        self.send_btn.config(state=tk.DISABLED, text="...")

        self._append_user(text)
        def run():
            response = generate(text)
            self.root.after(0, lambda: self._on_response(response))

        threading.Thread(target=run, daemon=True).start()

    def _on_response(self, response):
        self._append_ai(response)
        self.send_btn.config(state=tk.NORMAL, text="SEND")
        self.input_box.focus_set()

    def _append_user(self, text):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "YOU\n", "user_label")
        self.chat_display.insert(tk.END, text + "\n\n", "user_text")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def _append_ai(self, text):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "DARKGPT\n", "ai_label")
        self.chat_display.insert(tk.END, text.strip() + "\n\n", "ai_text")
        self.chat_display.insert(tk.END, "─" * 60 + "\n\n", "divider")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def _append_system(self, text):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, text, "system_text")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.iconbitmap("icon.ico")
    except Exception:
        pass

    app = DarkGPTApp(root)
    root.mainloop()