import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams

# -------- Global fonts --------
rcParams['font.family'] = 'Times New Roman'
rcParams['axes.unicode_minus'] = False

# -------- Data --------
data = {
    "Setting":[
        "Baseline","-S","-E","-G",
        "-(S&E)","-(S&G)","-(E&G)",
        "-(S&E&G)","-RH","-SH","-VWS"
    ],
    "Track":[125.50231,147.45816,115.91947,127.66226, 82.98410,158.70530, 76.45161,121.72918,117.60029,113.56963,112.72620],
    "Intensity":[1.82207,1.93770,2.35999,1.72963,2.21008,1.73785,1.88997,2.21976,2.02177,2.03526,1.93448],
    "MSLP":[3.93314,4.21291,5.37520,4.03407,4.32208,4.18553,3.95379,5.26765,4.50460,4.33903,4.22148],
    "dU_RI":[3.77341,5.03130,7.10990,4.68275,4.82423,3.89182,4.57132,7.83442,4.57058,4.72289,4.27067]
}
df_all = pd.DataFrame(data).set_index("Setting")

main_settings = [
    "Baseline","-S","-E","-G",
    "-(S&E)","-(S&G)","-(E&G)",
    "-(S&E&G)"
]
sub_settings = ["-RH","-SH","-VWS"]

df_main = df_all.loc[main_settings]
df_sub  = df_all.loc[sub_settings]

# -------- Style: colors and hatches --------
bar_colors  = ['black'] + ['silver']*3 + ['#ADD8E6']*3 + ['#FFB6C1']
bar_hatches = ['']      + ['']*3         + ['//']*3        + ['+']

markers    = ['o', '^', 's']               # RH / SH / VWS
markercol  = ['black', 'black', 'black']
markerlab  = ['-RH', '-SH', '-VWS']
x_offset   = [-0.15, 0.0, 0.15]            # horizontal offsets

metrics      = ["Track","Intensity","MSLP","dU_RI"]
ylabels      = ["RMSE (km)", "RMSE (m s$^{-1}$)", "RMSE (hPa)", "RMSE (m s$^{-1}$)"]
subplot_tags = ['(a) ', '(b) ', '(c) ', '(d) ', '(e) ']

settings = df_main.index.tolist()
x        = np.arange(len(settings))

# Precompute x positions for the three group labels
x1 = np.mean(x[1:4])   # Single-factor: bars 0-3
x2 = np.mean(x[4:7])   # Double-factor: bars 4-6
x3 = x[7]              # Triple-factor: bar 7

# -------- Plotting --------
fig, axes = plt.subplots(5, 1, figsize=(18, 23), constrained_layout=False)

# ----- 1–4 bar charts -----
for i, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
    ax   = axes[i]
    vals = df_main[metric].values

    # Draw bars with thick black edges
    bars = ax.bar(
        x, vals,
        color=bar_colors,
        edgecolor='black', linewidth=4,
        width=0.65
    )
    # Apply hatches
    for j, bar in enumerate(bars):
        bar.set_hatch(bar_hatches[j])

    # Compute y-axis upper bound & text offset
    ymax = max(np.r_[vals, df_sub[metric].values]) * 1.20
    y_text_offset = ymax * 0.02
    ax.set_ylim(0, ymax)

    # Add grouped titles inside the frame
    ax.text(x1, ymax*0.91, "Single-factor ablation",
            ha='center', va='bottom',
            fontsize=18, fontweight='bold', color=bar_colors[2])
    ax.text(x2, ymax*0.91, "Double-factor",
            ha='center', va='bottom',
            fontsize=18, fontweight='bold', color=bar_colors[4])
    ax.text(x3, ymax*0.91, "Triple-factor",
            ha='center', va='bottom',
            fontsize=18, fontweight='bold', color=bar_colors[7])

    # Polymarkers for sub-settings at "-E"
    idx_env = settings.index("-E")
    for dx, mk, col, lab in zip(x_offset, markers, markercol, markerlab):
        y_val = df_sub.loc[lab, metric]
        ax.plot(
            x[idx_env] + dx, y_val,
            marker=mk, color=col,
            markersize=10, linestyle='None',
            label=lab, zorder=5
        )
        ax.text(
            x[idx_env] + dx,
            y_val - y_text_offset,
            lab,
            ha='center', va='top',
            rotation=90,
            fontsize=15,
            color=col
        )

    # Annotate bar-top values
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            v + y_text_offset,
            f"{v:.2f}",
            ha='center', va='bottom',
            fontsize=15
        )

    # Baseline dashed line
    ax.axhline(
        bars[0].get_height(),
        ls='--', color='black', alpha=0.6
    )

    # Title and axis labels
    ax.set_title(
        subplot_tags[i] + metric,
        loc='left', fontsize=20
    )
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(settings, fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(False)

    # Legend (keep only the last three polymarkers)
    handles, labels_leg = ax.get_legend_handles_labels()
    ax.legend(
        handles[-3:], labels_leg[-3:],
        fontsize=16, loc='upper left',
        bbox_to_anchor=(1.01, 1.00),
        borderaxespad=0
    )

# ----- 5 line chart (Relative vs Baseline) -----
base_track     = df_main.loc["Baseline", "Track"]
base_intensity = df_main.loc["Baseline", "Intensity"]

track_rel     = ( df_main["Track"] - base_track)     / base_track
intensity_rel = ( df_main["Intensity"] - base_intensity) / base_intensity

ax = axes[4]
ax.plot(
    x, track_rel, 'o-',
    linewidth=3, markersize=8, label='Track', color='black',
)
ax.plot(
    x, intensity_rel, 's-',
    linewidth=3, markersize=8, label='Intensity', color='#1E90FF',
)

# Thick zero line
ax.axhline(0, color='black', linestyle='--', linewidth=1)

ax.set_xticks(x)
ax.set_xticklabels(settings, fontsize=18)
ax.set_ylabel('Normalized RMSE', fontsize=18)
ax.tick_params(axis='y', labelsize=18)  # enlarge y-axis tick labels
ax.set_title(
    subplot_tags[4] + 'Relative Improvement Compare with Baseline: Track & Intensity',
    loc='left', fontsize=20
)
ax.legend(fontsize=16, loc='upper right', 
          bbox_to_anchor=(1.12, 1.00),
          borderaxespad=0)
ax.grid(True, linestyle='--', alpha=0.3)

# ----- Layout & save -----
plt.tight_layout(rect=[0, 0, 0.95, 0.95])  # leave 5% headroom at top
plt.savefig(
    '/Users/lzh/Desktop/Ablation_polymarker.png',
    dpi=300, bbox_inches='tight'
)
plt.show()


#%%
# Model results comparison
import matplotlib.pyplot as plt
import numpy as np

# --- Data ----------------------------------------------------
metrics = ["Latitude", "Longitude", "Intensity", "MSLP"]
rmse_tcn   = [0.72, 1.38, 2.88, 4.91]
rmse_basic = [0.33, 0.42, 2.39, 4.30]
rmse_owzp  = [0.6038, 1.147, 1.822, 3.933]

mae_deeptc = 6.17
mae_safnet = 4.30
mae_owzp   = 3.12

# --- Figure / Axes ------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5))
ax2 = ax.twinx()

x      = np.arange(len(metrics))
sep    = x[-1] + 1
width  = 0.25

# --- RMSE bars (left axis) -----------------------------------
bars_tcn   = ax.bar(x - width, rmse_tcn,   width,
                    color="lightgrey", edgecolor="black",
                    label="TCN (2018-2021)")
bars_basic = ax.bar(x,         rmse_basic, width,
                    color="orange",    edgecolor="black",
                    label="Basic-Tr. (2018-2021)")
bars_owzp  = ax.bar(x + width, rmse_owzp,  width,
                    color="tab:red",   edgecolor="black",
                    label="OWZP-Tr. (2020-2023)")

# ▲ scatter (RMSE)
rmse_owzp_new = [0.3245, 0.800, 1.79, 3.57]
tri_scatter = ax.scatter(x + width, rmse_owzp_new,
                         marker='^', color='deepskyblue',
                         s=60, zorder=5,
                         label="OWZP-Tr. (2018–2021)")

# --- MAE bars (right axis) -----------------------------------
bars_deeptc   = ax2.bar(sep - width, mae_deeptc, width,
                        color="tab:blue", edgecolor="black",
                        label="DeepTC")
bars_safnet   = ax2.bar(sep,         mae_safnet, width,
                        color="lightgreen", edgecolor="black",
                        label="SAF-Net")
bars_owzp_mae = ax2.bar(sep + width, mae_owzp,   width,
                        color="tab:red", edgecolor="black",
                        label="OWZP-Tr.")

# ◯ ■ scatter (MAE)
mae_owzp_circle = 2.79
mae_owzp_square = 2.69
circ_scatter   = ax2.scatter(sep + width - 0.08, mae_owzp_circle,
                             marker='o', color='violet', edgecolor='k',
                             s=60, zorder=5,
                             label="OWZP-Tr. (DeepTC dataset)")
square_scatter = ax2.scatter(sep + width + 0.08, mae_owzp_square,
                             marker='s', color='gold', edgecolor='k',
                             s=60, zorder=5,
                             label="OWZP-Tr. (SAF-Net dataset)")

# Add triangular scatter (2018–2021)
mae_owzp_tri = 2.
tri2_scatter = ax2.scatter(sep + width, mae_owzp_tri,
                           marker='^', color='deepskyblue', edgecolor='k',
                           s=60, zorder=5,
                           label="OWZP-Tr. (2018–2021)")

# --- Value labels --------------------------------------------
def autolabel(bars, axis, fmt="{:.2f}"):
    for bar in bars:
        h = bar.get_height()
        axis.annotate(fmt.format(h),
                      xy=(bar.get_x() + bar.get_width()/2, h),
                      xytext=(0, 4), textcoords="offset points",
                      ha='center', va='bottom', fontsize=11)

autolabel(bars_tcn, ax); autolabel(bars_basic, ax); autolabel(bars_owzp, ax)
autolabel(bars_deeptc, ax2); autolabel(bars_safnet, ax2); autolabel(bars_owzp_mae, ax2)

# --- Other appearance ----------------------------------------
ax.axvline(sep - 0.5, color="k", linestyle="--")
ax.set_xticks(list(x)+[sep])
ax.set_xticklabels(metrics+["dU_RI"], fontsize=12)
ax.set_ylabel("RMSE", fontsize=14);  ax2.set_ylabel("MAE", fontsize=14)
ax.set_title("Model performance comparison", fontsize=16)
ax.tick_params(axis="y", labelsize=12); ax2.tick_params(axis="y", labelsize=12)
ax.grid(False); ax2.grid(False)

# Left legend (RMSE + RMSE scatter)
handles_left  = [bars_tcn[0], bars_basic[0], bars_owzp[0], tri_scatter]
labels_left   = ["TCN (2018-2021)",
                 "Basic-Tr. (2018-2021)",
                 "OWZP-Tr. (2020-2023)",
                 "OWZP-Tr. (2018–2021)"]
ax.legend(handles_left, labels_left,
          loc="upper left", fontsize=10, frameon=True)

# Right legend (MAE + MAE scatter)
handles_right = [
    bars_deeptc[0], bars_safnet[0], bars_owzp_mae[0],
    circ_scatter, square_scatter, tri2_scatter
]
labels_right  = ["DeepTC",
                 "SAF-Net",
                 "OWZP-Tr.",
                 "OWZP-Tr. (DeepTC dataset)",
                 "OWZP-Tr. (SAF-Net dataset)",
                 "OWZP-Tr. (2018–2021)"]
ax2.legend(handles_right, labels_right,
           loc="center left",
           bbox_to_anchor=(1.03, 0.87),
           fontsize=10, frameon=True)

# --- Layout / save -------------------------------------------
plt.tight_layout(rect=[0, 0, 0.95, 1])
plt.savefig('/Users/lzh/Desktop/bar.png', dpi=300, bbox_inches='tight')
plt.show()
