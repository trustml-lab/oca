import os
import glob
import json
import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

base_dir = "/home/jyj1025/osg-advbandit/snapshots/Shift_Plot_over_alpha-4"
output_dir = os.path.join(base_dir, "plots_over_alpha")
os.makedirs(output_dir, exist_ok=True)

glob_pattern = os.path.join(base_dir, "*-fM2-alpha-*-Hs-*", "fdrs.json")
# glob_pattern = os.path.join(base_dir, "NIPS25-*-Plot-over-alpha-*-fM2-T-30000-alpha-*-Hs-1000-unkT-False", "fdrs.json")
json_paths = sorted(glob.glob(glob_pattern))
alpha_regex = re.compile(r"-fM2-alpha-([0-9\.]+)-Hs-")
# alpha_regex = re.compile(r"-fM2-.*?-alpha-([0-9.]+)-Hs-")

fdr_dict = defaultdict(list)
for p in json_paths:
    m = alpha_regex.search(p)
    if not m:
        continue
    alpha = float(m.group(1))
    with open(p, "r") as f:
        fdrs = json.load(f) 
    fdr_dict[alpha].extend(fdrs)

if not fdr_dict:
    raise RuntimeError("No fdrs.json files found. Check 'base_dir' and glob pattern.")

alphas = sorted(fdr_dict.keys())
fdr_list = [fdr_dict[a] for a in alphas]

pdf_path = os.path.join(output_dir, "violin_over_alpha.pdf")
png_path = os.path.join(output_dir, "violin_over_alpha.png")

with PdfPages(pdf_path) as pdf:
    plt.figure(figsize=(7, 7))
    ax = plt.gca()

    positions = np.arange(1, len(alphas) + 1)
    vparts = plt.violinplot(fdr_list, positions=positions, showmeans=True, showextrema=False)

    for body in vparts['bodies']:
        body.set_facecolor("lightgreen")
        body.set_edgecolor("black")
        body.set_alpha(0.9)

    plt.xlabel(r'$\alpha$', fontsize=14)
    plt.ylabel('FDR', fontsize=14)
    desired_ticks = [round(x, 1) for x in np.arange(0.1, 1.0, 0.1)] 
    tick_pos, tick_labels = [], []
    for idx, a in enumerate(alphas):
        for t in desired_ticks:
            if np.isclose(a, t, atol=1e-8):
                tick_pos.append(positions[idx])
                tick_labels.append(f"{t:.1f}")
                break
    plt.xticks(tick_pos, tick_labels, fontsize=10)
    
    plt.yticks(np.arange(0.0, 1.05, 0.1), fontsize=10)
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, len(alphas) + 1)

    # stochasic
    # if model == 'llama3.1-8B' and 'triviaqa' in dataset_name:
    #     ylim = 0.239
    # elif model == 'gpt-3.5' and 'triviaqa' in dataset_name:
    #     ylim = 0.124
    # elif model == 'llama3.1-8B' and 'nq' in dataset_name:
    #     ylim = 0.425
    # elif model == 'gpt-3.5' and 'nq' in dataset_name:
    #     ylim = 0.294
    
    # shift
    # ylim = 0.210
    ylim = 0.340
    
    # all
    # ylim = 0.239
    
    yl = plt.axhline(y=ylim, color='k', alpha=0.7, linestyle='dotted', label=r'$\mathrm{Err}_{T}$')

    plt.plot(
        [0, len(alphas) + 1],
        [0.0, 1.0],
        linestyle='--', color='k', linewidth=0.8,
        label=r'$y=x$'
    )
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)
    plt.tight_layout()

    exsul_legend = mlines.Line2D(
        [], [], color='lightgreen', marker='o', linestyle='None', markersize=10,
        label=r'$\mathtt{ExSUL}$ (ours)'
    )
    plt.legend(handles=[exsul_legend, yl], loc='upper right', fontsize=10)


    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    pdf.savefig(bbox_inches='tight')

print(f"Saved:\n- {png_path}\n- {pdf_path}")
