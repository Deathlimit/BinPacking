import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from packing_env import BinPackingEnv
from test_model import MLPPolicy


class MLPGui:
    def __init__(self, root):
        self.root = root
        self.root.title("MLP Bin Packing Viewer")
        self.root.geometry("1100x700")

        self.env = None
        self.model = None
        self.obs = None
        self.done = True
        self.progress = tk.DoubleVar(value=0)

        self._build_ui()

    def _build_ui(self):
        left = ttk.Frame(self.root, padding=10)
        left.grid(row=0, column=0, sticky=(tk.N, tk.S))
        right = ttk.Frame(self.root, padding=10)
        right.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        ttk.Label(left, text="Environment", font=("Arial", 12, "bold")).grid(
            row=0, column=0, sticky=tk.W
        )
        self.width_var = tk.IntVar(value=100)
        self.num_var = tk.IntVar(value=50)
        self.seed_var = tk.IntVar(value=123)
        ttk.Label(left, text="Canvas Width").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(left, textvariable=self.width_var, width=10).grid(row=1, column=1)
        ttk.Label(left, text="Num Rects").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(left, textvariable=self.num_var, width=10).grid(row=2, column=1)
        ttk.Label(left, text="Seed").grid(row=3, column=0, sticky=tk.W)
        ttk.Entry(left, textvariable=self.seed_var, width=10).grid(row=3, column=1)
        ttk.Button(left, text="Create Env", command=self.create_env).grid(
            row=4, column=0, columnspan=2, pady=5
        )

        ttk.Separator(left).grid(
            row=5, column=0, columnspan=2, sticky=(tk.E, tk.W), pady=10
        )

        ttk.Label(left, text="MLP Model", font=("Arial", 12, "bold")).grid(
            row=6, column=0, sticky=tk.W
        )
        ttk.Button(left, text="Load mlp_imitation.pth", command=self.load_model).grid(
            row=7, column=0, columnspan=2, pady=5
        )

        ttk.Separator(left).grid(
            row=8, column=0, columnspan=2, sticky=(tk.E, tk.W), pady=10
        )

        ttk.Label(left, text="Controls", font=("Arial", 12, "bold")).grid(
            row=9, column=0, sticky=tk.W
        )
        ttk.Button(left, text="Reset", command=self.reset_env).grid(
            row=10, column=0, pady=5
        )
        ttk.Button(left, text="Step", command=self.step_once).grid(
            row=10, column=1, pady=5
        )
        ttk.Button(left, text="Run All", command=self.run_all).grid(
            row=11, column=0, columnspan=2, pady=5
        )
        pb = ttk.Progressbar(left, variable=self.progress, maximum=100)
        pb.grid(row=12, column=0, columnspan=2, sticky=(tk.E, tk.W), pady=5)

        self.figure, self.ax = plt.subplots(1, 1, figsize=(7, 5))
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.status = tk.StringVar(value="Status: Ready")
        ttk.Label(right, textvariable=self.status).pack(anchor=tk.W)

    def create_env(self):
        try:
            self.env = BinPackingEnv(
                canvas_width=self.width_var.get(), num_rects=self.num_var.get()
            )
            self.status.set("Environment created")
        except Exception as e:
            messagebox.showerror("Error", f"Create env failed: {e}")

    def load_model(self):
        if self.env is None:
            messagebox.showwarning("Warning", "Create environment first")
            return
        try:
            self.model = MLPPolicy(input_dim=self.env.observation_space.shape[0])
            self.model.load_state_dict(
                torch.load("mlp_imitation.pth", map_location="cpu")
            )
            self.model.eval()
            self.status.set("MLP model loaded")
        except Exception as e:
            messagebox.showerror("Error", f"Load model failed: {e}")

    def reset_env(self):
        if self.env is None:
            messagebox.showwarning("Warning", "Create environment first")
            return
        try:
            self.obs, _ = self.env.reset(seed=self.seed_var.get())
            self.done = False
            self.progress.set(0)
            self.status.set("Environment reset")
            self._draw()
        except Exception as e:
            messagebox.showerror("Error", f"Reset failed: {e}")

    def step_once(self):
        if self.env is None or self.model is None or self.done:
            return
        with torch.no_grad():
            x = torch.from_numpy(self.obs).unsqueeze(0)
            x_norm = self.model(x).squeeze(0).numpy()
            action = x_norm.astype(np.float32)
            self.obs, reward, self.done, _, info = self.env.step(action)
        self.progress.set(100 * (self.env.current_idx / self.env.num_rects))
        self.status.set(f"Step {self.env.current_idx}/{self.env.num_rects}")
        self._draw()
        if self.done:
            self._final_stats()

    def run_all(self):
        if self.env is None or self.model is None:
            return
        while not self.done:
            self.step_once()
            self.root.update_idletasks()
            self.root.update()

    def _draw(self):
        self.ax.clear()
        self.ax.set_xlim(0, self.env.canvas_width)
        self.ax.set_ylim(0, max(self.env.max_height, 50) * 1.1)
        self.ax.set_title("MLP Packing")
        self.ax.set_xlabel("Width")
        self.ax.set_ylabel("Height")
        for x, y, w, h in self.env.placed_rects:
            self.ax.add_patch(
                plt.Rectangle((x, y), w, h, ec="black", fc="steelblue", alpha=0.8)
            )

        self.ax.axhline(
            y=self.env.ideal_height, color="gold", linestyle="--", label="Ideal"
        )
        self.ax.axhline(
            y=self.env.max_height, color="green", linestyle="-", label="Actual"
        )
        self.ax.legend()
        self.canvas.draw()

    def _final_stats(self):
        m = self.env.get_metrics()
        msg = (
            f"Adjusted Efficiency: {m['efficiency']:.2%}\n"
            f"Raw Efficiency:      {m['efficiency_raw']:.2%}\n"
            f"Area Utilization:    {m['area_utilization']:.2%}\n"
            f"Max Height:          {m['max_height']:.1f}\n"
            f"Ideal Height:        {m['ideal_height']:.1f}"
        )
        messagebox.showinfo("Results", msg)


if __name__ == "__main__":
    root = tk.Tk()
    app = MLPGui(root)
    root.mainloop()
