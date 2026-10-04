from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .io import csv_headers, read_xy_csv, write_fit_csv
from .models import fit_model, fit_best_model, FitResult
from .report import fit_text, savings_text
from .savings import avoided_energy_savings


class ECAMResurrectedApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ECAM Resurrected — Python M&V")
        self.geometry("900x680")
        self.minsize(760, 560)
        self.fit_result: FitResult | None = None

        self.baseline_path = tk.StringVar()
        self.post_path = tk.StringVar()
        self.x_col = tk.StringVar()
        self.y_col = tk.StringVar()
        self.model = tk.StringVar(value="Auto (AICc)")
        self.confidence = tk.DoubleVar(value=0.80)

        self._build()

    def _build(self):
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(8, weight=1)

        ttk.Label(root, text="ECAM Resurrected", font=("TkDefaultFont", 18, "bold")).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(root, text="Independent Python resurrection of ECAM v4 core M&V change-point regression workflows.").grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 10))

        ttk.Label(root, text="Baseline CSV").grid(row=2, column=0, sticky="w")
        ttk.Entry(root, textvariable=self.baseline_path).grid(row=2, column=1, sticky="ew", padx=6)
        ttk.Button(root, text="Browse…", command=self._choose_baseline).grid(row=2, column=2)

        ttk.Label(root, text="Temperature / X").grid(row=3, column=0, sticky="w")
        self.x_combo = ttk.Combobox(root, textvariable=self.x_col, state="readonly")
        self.x_combo.grid(row=3, column=1, sticky="ew", padx=6)

        ttk.Label(root, text="Energy / Y").grid(row=4, column=0, sticky="w")
        self.y_combo = ttk.Combobox(root, textvariable=self.y_col, state="readonly")
        self.y_combo.grid(row=4, column=1, sticky="ew", padx=6)

        opts = ttk.Frame(root)
        opts.grid(row=5, column=0, columnspan=3, sticky="ew", pady=8)
        ttk.Label(opts, text="Model").pack(side="left")
        ttk.Combobox(opts, textvariable=self.model, state="readonly", width=16,
                     values=["Auto (AICc)", "2P", "3P Heating", "3P Cooling", "4P", "5P", "6P"]).pack(side="left", padx=(6, 18))
        ttk.Label(opts, text="Confidence").pack(side="left")
        ttk.Combobox(opts, textvariable=self.confidence, width=8, state="readonly",
                     values=[0.50, 0.60, 0.65, 0.68, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]).pack(side="left", padx=6)
        ttk.Button(opts, text="Fit Baseline", command=self._fit).pack(side="left", padx=12)
        ttk.Button(opts, text="Export Fit CSV…", command=self._export_fit).pack(side="left")

        ttk.Separator(root).grid(row=6, column=0, columnspan=3, sticky="ew", pady=8)
        ttk.Label(root, text="Post-period CSV (optional for avoided-energy savings)").grid(row=7, column=0, sticky="w")
        ttk.Entry(root, textvariable=self.post_path).grid(row=7, column=1, sticky="ew", padx=6)
        button_frame = ttk.Frame(root)
        button_frame.grid(row=7, column=2)
        ttk.Button(button_frame, text="Browse…", command=self._choose_post).pack(side="left")
        ttk.Button(button_frame, text="Savings", command=self._savings).pack(side="left", padx=(6, 0))

        self.output = tk.Text(root, wrap="word", font=("TkFixedFont", 10))
        self.output.grid(row=8, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
        scrollbar = ttk.Scrollbar(root, command=self.output.yview)
        scrollbar.grid(row=8, column=3, sticky="ns")
        self.output.configure(yscrollcommand=scrollbar.set)

        ttk.Label(root, text="Regression uncertainty is conditional on the fitted model/change points and does not include all measurement/model-form uncertainty.", foreground="#555").grid(row=9, column=0, columnspan=3, sticky="w", pady=(8, 0))

    def _choose_baseline(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        self.baseline_path.set(path)
        try:
            headers = csv_headers(path)
            self.x_combo["values"] = headers
            self.y_combo["values"] = headers
            lower = {h.lower(): h for h in headers}
            for key in ("temperature", "temp", "oat", "oa_t", "outdoor_temp"):
                if key in lower:
                    self.x_col.set(lower[key]); break
            for key in ("energy", "kwh", "use", "usage", "load"):
                if key in lower:
                    self.y_col.set(lower[key]); break
            if headers and not self.x_col.get(): self.x_col.set(headers[0])
            if len(headers) > 1 and not self.y_col.get(): self.y_col.set(headers[1])
        except Exception as exc:
            messagebox.showerror("CSV error", str(exc))

    def _choose_post(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.post_path.set(path)

    @staticmethod
    def _model_key(label: str) -> str:
        return {
            "2P": "2p", "3P Heating": "3p_heat", "3P Cooling": "3p_cool",
            "4P": "4p", "5P": "5p", "6P": "6p",
        }.get(label, "auto")

    def _fit(self):
        try:
            x, y = read_xy_csv(self.baseline_path.get(), self.x_col.get(), self.y_col.get())
            key = self._model_key(self.model.get())
            self.fit_result = fit_best_model(x, y) if key == "auto" else fit_model(x, y, key)
            self.output.delete("1.0", "end")
            self.output.insert("end", fit_text(self.fit_result))
        except Exception as exc:
            messagebox.showerror("Fit error", str(exc))

    def _savings(self):
        if self.fit_result is None:
            self._fit()
        if self.fit_result is None:
            return
        try:
            x, y = read_xy_csv(self.post_path.get(), self.x_col.get(), self.y_col.get())
            savings = avoided_energy_savings(self.fit_result, x, y, confidence=float(self.confidence.get()))
            self.output.insert("end", "\n\n" + savings_text(savings))
        except Exception as exc:
            messagebox.showerror("Savings error", str(exc))

    def _export_fit(self):
        if self.fit_result is None:
            messagebox.showinfo("Nothing to export", "Fit a baseline model first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if path:
            write_fit_csv(path, self.fit_result.x, self.fit_result.y, self.fit_result.yhat, self.fit_result.residuals)


def main():
    app = ECAMResurrectedApp()
    app.mainloop()


if __name__ == "__main__":
    main()
