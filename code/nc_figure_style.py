from __future__ import annotations
import os
import re
from typing import Optional
import matplotlib as mpl
import matplotlib.pyplot as plt
MM = 1.0 / 25.4
SINGLE = 89 * MM
DOUBLE = 183 * MM
C = {'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73', 'red': '#D55E00', 'purple': '#CC79A7', 'sky': '#56B4E9', 'yellow': '#F0E442', 'black': '#000000', 'grey': '#999999'}
MODEL_COLORS = {'RF': C['blue'], 'LR': C['orange'], 'SVC': C['green'], 'LightGBM': C['purple'], 'XGBoost': C['red']}
METHOD_COLORS = {'Boruta': C['green'], 'LASSO': C['orange'], 'mRMR': C['blue'], 'ReliefF': C['purple'], 'PCA': C['red']}
NC_STYLE = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Microsoft YaHei', 'Liberation Sans', 'Noto Sans CJK SC', 'DejaVu Sans'], 'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False, 'axes.facecolor': 'white', 'figure.facecolor': 'white', 'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'legend.frameon': False}

def _best_full_font() -> 'str | None':
    import os as _os
    candidates = ['C:\\Windows\\Fonts\\ARIALUNI.TTF', 'C:\\Windows\\Fonts\\msyh.ttc', 'C:\\Windows\\Fonts\\msyh.ttf', 'C:\\Windows\\Fonts\\simhei.ttf', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', '/System/Library/Fonts/PingFang.ttc']
    for p in candidates:
        if _os.path.exists(p):
            try:
                mpl.font_manager.fontManager.addfont(p)
                from matplotlib.ft2font import FT2Font
                return FT2Font(p).family_name
            except Exception:
                continue
    return None

def apply_style() -> None:
    full = _best_full_font()
    if full:
        fonts = [f for f in NC_STYLE['font.sans-serif'] if f != full]
        NC_STYLE['font.sans-serif'] = [full] + fonts
    mpl.rcParams.update(NC_STYLE)
apply_style()
RENAME = {'CA125_GC': 'CA125', 'LYMPH%': 'LYM%', 'LYMM': 'LYM#', 'DD': 'D-dimer'}

def pub(name: str) -> str:
    abnormal = name.endswith('_abnormal')
    base = name[:-9] if abnormal else name
    base = re.sub('^AU_', '', base)
    base = RENAME.get(base, base)
    return base + (' (abnormal)' if abnormal else '')

def panel_label(ax, letter: str, x: float=-0.18, y: float=1.05) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight='bold', va='top', ha='left')

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def savefig(fig, output_dir: str, filename: str, close: bool=True, formats: tuple[str, ...]=('png', 'pdf')) -> None:
    ensure_dir(output_dir)
    for ext in formats:
        fig.savefig(os.path.join(output_dir, f'{filename}.{ext}'))
    if close:
        plt.close(fig)
    print(f'[figure saved] {os.path.join(output_dir, filename)}')
