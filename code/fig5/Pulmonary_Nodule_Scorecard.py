from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from scorecard_core import BAND_EN, BLOOD_RULES, RULE_NAME_EN, compute_score, risk_band, risk_rate
BAND_COLOR = {'Low': '#009E73', 'Intermediate': '#E69F00', 'High': '#D55E00'}
DISPLAY_NAME = {'AU_Urea': 'Urea'}

class ScorecardApp(tk.Tk):

    def __init__(self) -> None:
        super().__init__()
        self.title('Pulmonary Nodule Malignancy Risk Scorecard (11 rules)')
        self.resizable(False, False)
        self.entries: dict[str, tk.Entry] = {}
        self._build()

    def _build(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.grid()
        left = ttk.LabelFrame(root, text=' Inputs (blank = not measured) ', padding=10)
        left.grid(row=0, column=0, sticky='n', padx=(0, 10))
        ttk.Label(left, text='— Blood tests —', font=('', 9, 'bold')).grid(row=0, column=0, columnspan=3, sticky='w', pady=(0, 4))
        for i, (name, (lo, hi), pts, unit) in enumerate(BLOOD_RULES, start=1):
            ttk.Label(left, text=DISPLAY_NAME.get(name, name)).grid(row=i, column=0, sticky='w')
            e = ttk.Entry(left, width=10)
            e.grid(row=i, column=1, padx=4)
            self.entries[name] = e
            ref = (f'{lo:g}' if lo is not None else '') + '–' + (f'{hi:g}' if hi is not None else '')
            ttk.Label(left, text=f"{unit}  ref. {ref.strip('–')}  ({pts:+d} pts)", foreground='#666666').grid(row=i, column=2, sticky='w')
        r0 = len(BLOOD_RULES) + 1
        ttk.Label(left, text='— CT features —', font=('', 9, 'bold')).grid(row=r0, column=0, columnspan=3, sticky='w', pady=(8, 4))
        ttk.Label(left, text='Max diameter').grid(row=r0 + 1, column=0, sticky='w')
        e = ttk.Entry(left, width=10)
        e.grid(row=r0 + 1, column=1, padx=4)
        self.entries['diameter'] = e
        ttk.Label(left, text='mm  (≤8: 0  8–15: +4  15–30: −2  >30: +3)', foreground='#666666').grid(row=r0 + 1, column=2, sticky='w')
        self.var_lob = tk.IntVar(value=0)
        self.var_spi = tk.IntVar(value=0)
        ttk.Checkbutton(left, text='Lobulation (+10 pts)', variable=self.var_lob).grid(row=r0 + 2, column=0, columnspan=2, sticky='w', pady=2)
        ttk.Checkbutton(left, text='Spiculation (+6 pts)', variable=self.var_spi).grid(row=r0 + 3, column=0, columnspan=2, sticky='w', pady=2)
        ttk.Button(left, text='CALCULATE SCORE', command=self.on_calc).grid(row=r0 + 4, column=0, columnspan=3, pady=(12, 2), sticky='we')
        right = ttk.LabelFrame(root, text=' Results ', padding=10)
        right.grid(row=0, column=1, sticky='n')
        ttk.Label(right, text='Total score').pack(anchor='w')
        self.lbl_score = ttk.Label(right, text='— pts', font=('', 22, 'bold'))
        self.lbl_score.pack(anchor='w', pady=(0, 6))
        self.lbl_risk = ttk.Label(right, text='Enter values and calculate', font=('', 11, 'bold'))
        self.lbl_risk.pack(anchor='w')
        self.lbl_rate = ttk.Label(right, text='', wraplength=230, justify='left')
        self.lbl_rate.pack(anchor='w', pady=(0, 8))
        ttk.Label(right, text='Rule breakdown:', font=('', 9, 'bold')).pack(anchor='w')
        self.tree = ttk.Treeview(right, columns=('hit', 'pts'), show='headings', height=11)
        self.tree.heading('hit', text='Rule')
        self.tree.heading('pts', text='Points')
        self.tree.column('hit', width=170)
        self.tree.column('pts', width=60, anchor='center')
        self.tree.pack()

    def _parse(self, key: str) -> float | None:
        s = self.entries[key].get().strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            raise ValueError(f"'{key}': '{s}' is not a number")

    def on_calc(self) -> None:
        try:
            blood = {name: self._parse(name) for name, *_ in BLOOD_RULES}
            diameter = self._parse('diameter')
        except ValueError as ex:
            self.lbl_risk.config(text=str(ex), foreground='#D55E00')
            return
        total, details = compute_score(blood, diameter, bool(self.var_lob.get()), bool(self.var_spi.get()))
        rate = risk_rate(total)
        self.lbl_score.config(text=f'{total} pts')
        if rate is None:
            self.lbl_risk.config(text='(validation risk table not found)', foreground='#666666')
            self.lbl_rate.config(text='')
        else:
            band = BAND_EN[risk_band(rate)]
            self.lbl_risk.config(text=f'Risk: {band}', foreground=BAND_COLOR[band])
            self.lbl_rate.config(text=f'Observed malignancy rate at this score in the external validation cohort: ~{rate:.0%}')
        for row in self.tree.get_children():
            self.tree.delete(row)
        for name, hit, got in details:
            self.tree.insert('', 'end', values=(('√ ' if hit else '× ') + RULE_NAME_EN.get(name, name), f'{got:+d}'))
if __name__ == '__main__':
    ScorecardApp().mainloop()
