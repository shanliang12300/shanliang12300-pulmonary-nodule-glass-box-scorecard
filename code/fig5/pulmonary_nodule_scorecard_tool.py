from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from scorecard_core import BLOOD_RULES, DIAMETER_RULES, compute_score, risk_band, risk_rate
BAND_COLOR = {'低危': '#009E73', '中危': '#E69F00', '高危': '#D55E00'}

class ScorecardApp(tk.Tk):

    def __init__(self) -> None:
        super().__init__()
        self.title('肺结节恶性风险评分卡（11 规则）')
        self.resizable(False, False)
        self.entries: dict[str, tk.Entry] = {}
        self._build()

    def _build(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.grid()
        left = ttk.LabelFrame(root, text=' 输入（留空 = 未查/未测） ', padding=10)
        left.grid(row=0, column=0, sticky='n', padx=(0, 10))
        ttk.Label(left, text='—— 血液检查 ——', font=('', 9, 'bold')).grid(row=0, column=0, columnspan=3, sticky='w', pady=(0, 4))
        for i, (name, (lo, hi), pts, unit) in enumerate(BLOOD_RULES, start=1):
            ttk.Label(left, text=f'{name}').grid(row=i, column=0, sticky='w')
            e = ttk.Entry(left, width=10)
            e.grid(row=i, column=1, padx=4)
            self.entries[name] = e
            ref = (f'{lo:g}' if lo is not None else '') + '–' + (f'{hi:g}' if hi is not None else '')
            ttk.Label(left, text=f"{unit}  参考 {ref.strip('–')}  ({pts:+d}分)", foreground='#666666').grid(row=i, column=2, sticky='w')
        r0 = len(BLOOD_RULES) + 1
        ttk.Label(left, text='—— CT 特征 ——', font=('', 9, 'bold')).grid(row=r0, column=0, columnspan=3, sticky='w', pady=(8, 4))
        ttk.Label(left, text='结节最大径').grid(row=r0 + 1, column=0, sticky='w')
        e = ttk.Entry(left, width=10)
        e.grid(row=r0 + 1, column=1, padx=4)
        self.entries['diameter'] = e
        ttk.Label(left, text='mm  (≤8:0  8–15:+4  15–30:−2  >30:+3)', foreground='#666666').grid(row=r0 + 1, column=2, sticky='w')
        self.var_lob = tk.IntVar(value=0)
        self.var_spi = tk.IntVar(value=0)
        ttk.Checkbutton(left, text='分叶征 (+10分)', variable=self.var_lob).grid(row=r0 + 2, column=0, columnspan=2, sticky='w', pady=2)
        ttk.Checkbutton(left, text='毛刺征 (+6分)', variable=self.var_spi).grid(row=r0 + 3, column=0, columnspan=2, sticky='w', pady=2)
        ttk.Button(left, text='计 算 评 分', command=self.on_calc).grid(row=r0 + 4, column=0, columnspan=3, pady=(12, 2), sticky='we')
        right = ttk.LabelFrame(root, text=' 结果 ', padding=10)
        right.grid(row=0, column=1, sticky='n')
        ttk.Label(right, text='总分').pack(anchor='w')
        self.lbl_score = ttk.Label(right, text='— 分', font=('', 22, 'bold'))
        self.lbl_score.pack(anchor='w', pady=(0, 6))
        self.lbl_risk = ttk.Label(right, text='请左侧输入后计算', font=('', 11, 'bold'))
        self.lbl_risk.pack(anchor='w')
        self.lbl_rate = ttk.Label(right, text='')
        self.lbl_rate.pack(anchor='w', pady=(0, 8))
        ttk.Label(right, text='逐条明细：', font=('', 9, 'bold')).pack(anchor='w')
        self.tree = ttk.Treeview(right, columns=('hit', 'pts'), show='headings', height=11)
        self.tree.heading('hit', text='规则')
        self.tree.heading('pts', text='得分')
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
            raise ValueError(f'「{key}」的输入「{s}」不是数字')

    def on_calc(self) -> None:
        try:
            blood = {name: self._parse(name) for name, *_ in BLOOD_RULES}
            diameter = self._parse('diameter')
        except ValueError as ex:
            self.lbl_risk.config(text=str(ex), foreground='#D55E00')
            return
        total, details = compute_score(blood, diameter, bool(self.var_lob.get()), bool(self.var_spi.get()))
        rate = risk_rate(total)
        self.lbl_score.config(text=f'{total} 分')
        if rate is None:
            self.lbl_risk.config(text='（缺少验证队列风险表，仅显示分值）', foreground='#666666')
            self.lbl_rate.config(text='')
        else:
            band = risk_band(rate)
            self.lbl_risk.config(text=f'风险分层：{band}', foreground=BAND_COLOR[band])
            self.lbl_rate.config(text=f'外部验证队列中该分值的实际恶性率约 {rate:.0%}')
        for row in self.tree.get_children():
            self.tree.delete(row)
        for name, hit, got in details:
            self.tree.insert('', 'end', values=(('√ ' if hit else '× ') + name, f'{got:+d}'))
if __name__ == '__main__':
    ScorecardApp().mainloop()
