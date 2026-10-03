import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'source_data'
OUT = Path(__file__).resolve().parent
COLORS = {'EP': '#0072B2', 'ES': '#D55E00', 'EE': '#009E73', 'surface': '#737373'}
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300})


def load(name):
    return json.loads((DATA / name).read_text())


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight', dpi=170)
    plt.close(fig)
