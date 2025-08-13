#!/usr/bin/env python3
"""
Batch-enabled refactor of dorin_argparse.py
- Supports single specimen (CLI flags or simple JSON config)
- Supports multiple specimens via a `specimens` array in the config, where all
  other parameters (d0, l1, l2, etc.) are shared across the batch.
- Writes ONE CSV summarizing all results (one row per specimen).
- Keeps per-specimen figures & folders the same as before.

Example multi-specimen config (shared params at top):
{
  "d0": 7.93,
  "l1": 15,
  "l2": 25,
  "h_max": 33,
  "l1_th_fix": 0.005,
  "min_factor": 2,
  "d0_auto_factor": 1.02,
  "export_suffix": "standard",
  "base_path": "/home/dorin/Projects/Uniform-Elongation-measurement",
  "specimens": [
    { "specimen_id": "AZK6z2", "specimen_folder": "Eval_top" },
    { "specimen_id": "AMA2x1", "specimen_folder": "Eval_bottom" }
  ]
}

Usage examples:
  # Using multi-specimen config only
  python dorin_argparse_batch.py --config run_multi.json

  # Using config *and* add one-off specimen via CLI
  python dorin_argparse_batch.py --config run_multi.json \
      --specimen-id EXTRA1 --specimen-folder Eval_top

  # Single specimen purely via CLI (no config)
  python dorin_argparse_batch.py \
      --specimen-id AZK6z2 --specimen-folder Eval_top \
      --d0 7.93 --l1 15 --l2 25 --h-max 33 \
      --l1-th-fix 0.005 --min-factor 2 --d0-auto-factor 1.02 \
      --export-suffix standard

  # Choose output CSV path
  python dorin_argparse_batch.py --config run_multi.json --output-csv results_batch.csv
"""

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Any

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import open3d as o3d
from scipy.interpolate import interp1d
from numpy import vectorize
from scipy.signal import savgol_filter


# =============================
# Utility: Data container
# =============================
@dataclass
class RunConfig:
    specimen_id: str
    specimen_folder: str
    d0: float
    l1: float
    l2: float
    h_max: float
    l1_th_fix: float
    min_factor: float
    d0_auto_factor: float
    export_suffix: str
    base_path: str = "/home/dorin/Projects/Uniform-Elongation-measurement"

    @property
    def file_path(self) -> str:
        return os.path.join(self.base_path, self.specimen_id, self.specimen_folder, "specimenFull.ply")

    @property
    def modpath(self) -> str:
        # e.g., /.../SpecimenFolder__standard_
        suffix = f"_{self.export_suffix}" if self.export_suffix else ""
        return os.path.join(self.base_path, self.specimen_id, f"{self.specimen_folder}{suffix}_")


# =============================
# I/O helpers
# =============================

def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_fig(fig, filename_base: str):
    os.makedirs(os.path.dirname(filename_base), exist_ok=True)
    fig.savefig(filename_base + ".pdf")
    fig.savefig(filename_base + ".png", dpi=300)


# =============================
# Original analysis functions (minor cleanups)
# =============================

def load_and_slice_pcd(file_path, y_slice_center=0.0, slice_thickness=0.0001):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    df = pd.DataFrame(points, columns=["X", "Y", "Z"])
    mask = (df["Y"] > y_slice_center - slice_thickness) & (df["Y"] < y_slice_center + slice_thickness)
    return df[mask][["X", "Y", "Z"]].to_numpy().T


def plot_scatter(x, y, xlabel, ylabel, filename_base, marker='x', color='black'):
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.plot(x, y, ls='none', marker=marker, mec=color, ms=2., zorder=2)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%1.0f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%1.3e'))
    ax.grid(which='major', axis='both', linewidth=1., linestyle='-', color='0.6')
    ax.grid(which='minor', axis='both', linewidth=0.5, linestyle='-', color='0.75')
    plt.tight_layout()
    save_fig(fig, filename_base)
    print(f"Figure 2 saved as:\n→ {filename_base}.pdf\n→ {filename_base}.png")


def separate_min_max(x, z):
    x = np.asarray(x)
    z = np.asarray(z)
    return (x[x > 0], z[x > 0]), (x[x <= 0], z[x <= 0])


def compute_specimen_diameter(Z_max, X_max, Z_min, X_min, h_max, modpath):
    # Check if there's enough data
    if len(Z_max) < 2 or len(Z_min) < 2:
        print("Skipping Fig4: Not enough data points for interpolation.")
        return None, None, None

    # Sort and remove duplicate Z values for stability
    def dedup_and_sort(Z, X):
        df = pd.DataFrame({'Z': Z, 'X': X}).sort_values(by='Z')
        df = df.drop_duplicates(subset='Z')  # Keep only first entry per Z
        return df['Z'].values, df['X'].values

    Z_max, X_max = dedup_and_sort(Z_max, X_max)
    Z_min, X_min = dedup_and_sort(Z_min, X_min)

    Edge_max_x_interp = interp1d(Z_max, X_max, kind='linear', bounds_error=False, fill_value="extrapolate")
    Edge_min_x_interp = interp1d(Z_min, X_min, kind='linear', bounds_error=False, fill_value="extrapolate")
    start_x = max(Z_max[0], Z_min[0])
    end_x = min(Z_max[-1], Z_min[-1])
    x_field = np.linspace(start_x, end_x, 1000)
    D_X = Edge_max_x_interp(x_field) - Edge_min_x_interp(x_field)

    x_field_reverse = x_field[::-1]
    D_X_interp = interp1d(x_field_reverse, D_X, kind='linear', bounds_error=False, fill_value="extrapolate")
    D_X = D_X_interp(x_field)

    fig4, ax4 = plt.subplots(figsize=(7, 5), dpi=150, facecolor='white')
    ax4.plot(x_field, D_X, ls='none', marker='x', mec='black', ms=2., zorder=2)
    ax4.set_xlabel("Specimen Height [mm]", fontsize=16)
    ax4.set_ylabel("Specimen Diameter [mm]", fontsize=16, rotation=90)
    ax4.tick_params(axis='both', labelsize=14)
    ax4.set_axisbelow(True)
    ax4.set_xlim(-0.5, h_max)
    ax4.xaxis.set_major_formatter(FormatStrFormatter('%1.3f'))
    ax4.yaxis.set_major_formatter(FormatStrFormatter('%1.2e'))
    ax4.xaxis.set_ticks_position('both')
    plt.tight_layout()
    save_fig(fig4, os.path.join(modpath, 'Fig4'))
    print(f"Figure 4 saved as:\n→ {os.path.join(modpath, 'Fig4')}.pdf\n→ {os.path.join(modpath, 'Fig4')}.png")

    return x_field, D_X, D_X_interp


def eps_ue(l2, l1, d0, D_X_interp):
    d1 = float(D_X_interp(l1))
    d2 = float(D_X_interp(l2))
    d_avg = (d1 + d2) / 2.0
    strain_tangent = (((d0 / d_avg) ** 2) - 1) * 100.
    return d1, d2, strain_tangent


def eps_ue_integral(l2, l1, d0, D_X_interp):
    l0 = l2 - l1
    if abs(l0) < 1e-6:
        print("⚠️ Integral strain skipped: l1 and l2 are too close.")
        return float('nan')

    V0 = math.pi * (d0 ** 2) / 4. * l0

    dx = 0.001
    integrand = 0.
    dxres = 0.
    VolumeResult = 0.
    x = dx / 2.

    while VolumeResult < V0:
        integrand += (float(D_X_interp(l2 - x)) / 2.) ** 2 * dx
        VolumeResult = math.pi * integrand
        x += dx
        dxres += dx

    return (dxres - l0) / l0 * 100.


eps_ue_integral_value = vectorize(eps_ue_integral)


# ---- Savitzky–Golay smoothing helpers ----

def auto_window_size(data_length, divisor=50, min_size=5):
    size = max(int(data_length / divisor), min_size)
    size = 101
    return size


def auto_window_size_for_Derivative(data_length, divisor=50, min_size=5):
    size = max(int(data_length / divisor), min_size)
    size = 201
    return size


def auto_poly_order(window_size):
    if window_size < 7:
        return 1
    elif window_size < 15:
        return 1
    else:
        return 1


def apply_savgol_auto(data, deriv=0):
    data = np.asarray(data)
    window_size = auto_window_size(len(data))
    poly_order = auto_poly_order(window_size)

    if poly_order >= window_size:
        poly_order = max(1, window_size - 2)
    if window_size % 2 == 0:
        window_size += 1

    return savgol_filter(data, window_length=window_size, polyorder=poly_order, deriv=deriv)


def apply_savgol_auto_Derivative(data, deriv=0):
    data = np.asarray(data)
    window_size = auto_window_size_for_Derivative(len(data))
    poly_order = auto_poly_order(window_size)

    if poly_order >= window_size:
        poly_order = max(1, window_size - 2)
    if window_size % 2 == 0:
        window_size += 1

    return savgol_filter(data, window_length=window_size, polyorder=poly_order, deriv=deriv)


def process_diameter_profile_auto(D_X):
    D_X_filtered = apply_savgol_auto(D_X, deriv=0)
    D_X_derivative = apply_savgol_auto(D_X, deriv=1)
    D_X_derivative_smoothed = apply_savgol_auto_Derivative(D_X_derivative, deriv=0)
    return D_X_filtered, D_X_derivative, D_X_derivative_smoothed


def plot_diameter_comparison(x_field, D_x, D_x_filtered, modpath):
    fig10 = plt.figure(figsize=(7, 5), dpi=150, facecolor='white')
    plt.plot(x_field, D_x, ls='-', c='black', zorder=1, label='Raw Diameter')
    plt.plot(x_field, D_x_filtered, ls='--', c='red', zorder=2, label='Filtered Diameter')

    plt.xlabel("Specimen Height [mm]", fontsize=16)
    plt.ylabel("Specimen Diameter [mm]", fontsize=16, rotation=90)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xscale('linear')
    plt.yscale('linear')

    plt.grid(which='major', axis='both', linewidth=1., linestyle='-', color='0.6')
    plt.grid(which='minor', axis='both', linewidth=0.5, linestyle='-', color='0.75')

    plt.legend(fontsize=14, loc='best')
    plt.tight_layout()

    save_fig(fig10, os.path.join(modpath, 'Fig10'))
    print(f"Figure 10 saved as:\n→ {os.path.join(modpath, 'Fig10')}.pdf\n→ {os.path.join(modpath, 'Fig10')}.png")


def plot_derivative_comparison(x_field, D_X_derivative, D_X_derivative_smoothed, modpath):
    fig20, ax = plt.subplots(figsize=(7, 5), dpi=150, facecolor='white')
    ax.plot(x_field, D_X_derivative, ls='-', lw=2, c='black', zorder=2, label='Diameter Derivative')
    ax.plot(x_field, D_X_derivative_smoothed, ls='--', lw=2, c='red', zorder=2, label='Filtered Derivative')
    ax.set_xlabel("Specimen Height [mm]", fontsize=16)
    ax.set_ylabel("Diameter Derivative [mm/mm]", fontsize=16, rotation=90)

    ax.tick_params(axis='both', labelsize=14)
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.set_xlim(-0.5, 30)
    ax.set_ylim(0, 0.025)

    ax.xaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%1.2e'))

    ax.set_axisbelow(True)
    ax.grid(which='major', axis='both', linewidth=1., linestyle='-', color='0.6')
    ax.grid(which='minor', axis='both', linewidth=0.5, linestyle='-', color='0.75')

    ax.legend(loc='best', fontsize=14)
    plt.tight_layout()

    save_fig(fig20, os.path.join(modpath, 'Fig20'))
    print(f"Figure 20 saved as:\n→ {os.path.join(modpath, 'Fig20')}.pdf\n→ {os.path.join(modpath, 'Fig20')}.png")


def justfindl2(x_field, D_X_derivative_smoothed, h_max):
    min_deriv = D_X_derivative_smoothed[0]
    l2 = x_field[0]

    for i in range(1, len(D_X_derivative_smoothed)):
        if D_X_derivative_smoothed[0] < 1e-3:
            print(" D_X Filtered data too low")
            break

        if D_X_derivative_smoothed[i] < min_deriv:
            min_deriv = D_X_derivative_smoothed[i]
            l2 = x_field[i]

        if x_field[i] > h_max:
            break

    return l2, min_deriv


def justfindl1(x_field, D_X_derivative_smoothed, l1_th_fix, h_max):
    l1 = x_field[0]

    for i in range(1, len(D_X_derivative_smoothed)):
        if x_field[i] > h_max:
            break
        if D_X_derivative_smoothed[i] < l1_th_fix:
            l1 = x_field[i]
            return l1

    return None


def l1_var_1(x_field, D_X_derivative_smoothed, h_max, D_X_interp, d0_auto_Factor, min_Factor):
    l2, _ = justfindl2(x_field, D_X_derivative_smoothed, h_max)
    l2_index = np.argmin(np.abs(x_field - l2))
    min_deriv = D_X_derivative_smoothed[l2_index]
    threshold = min_deriv * min_Factor
    sorted_indices = np.argsort(D_X_derivative_smoothed)
    derivatives_sorted = D_X_derivative_smoothed[sorted_indices]
    xfield_sorted = x_field[sorted_indices]
    derivatives_interp = interp1d(derivatives_sorted, xfield_sorted, kind='linear', bounds_error=False, fill_value="extrapolate")
    l1_var_1_val = float(derivatives_interp(threshold))
    d0_auto = float(D_X_interp(l2)) * d0_auto_Factor

    return l1_var_1_val, d0_auto, derivatives_interp


def findl2_variant3(x_field, D_X_filtered, h_max, d0):
    for i in range(1, len(D_X_filtered)):
        if x_field[i] > h_max:
            break
        if D_X_filtered[i] >= d0:
            l2_variant_3 = x_field[i]
            return l2_variant_3
    return None


def findl1_variant3(x_field, D_X_derivative_smoothed, h_max):
    min_deriv = D_X_derivative_smoothed[0]
    l1_var_3 = x_field[0]

    for i in range(1, len(D_X_derivative_smoothed)):
        if x_field[i] > h_max:
            break

        if D_X_derivative_smoothed[0] < 1e-3:
            print(" D_X Filtered data too low")
            break

        if D_X_derivative_smoothed[i] < min_deriv:
            min_deriv = D_X_derivative_smoothed[i]
            l1_var_3 = x_field[i]

    return l1_var_3, min_deriv


def strain_using_Diameter_Curve(d1, d2, d0):
    d_avg = (d1 + d2) / 2.0
    strain_tangent = (((d0 / d_avg) ** 2) - 1) * 100.
    return strain_tangent


def plot_L1_L2(x_field, D_X_derivative_smoothed, l1, l2, h_max, modpath, fig_id, y_min=2e-5, y_max=2e-2):
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150, facecolor='white')
    ax.plot(x_field, D_X_derivative_smoothed, ls='-', lw=2, c='black', zorder=2, label='Diameter Derivative Smoothed')
    ax.plot([l1, l1], [y_min, y_max], ls=':', lw=1, c='black', zorder=1, label='Line L1')
    ax.plot([l2, l2], [y_min, y_max], ls=':', lw=1, c='red', zorder=1, label='Line L2')
    ax.set_xlabel("Specimen Height [mm]", fontsize=16)
    ax.set_ylabel("Diameter Derivative Smoothed [mm/mm]", fontsize=16)

    ax.tick_params(axis='both', labelsize=14)
    ax.set_xscale('linear')
    ax.set_yscale('log')
    ax.set_xlim(0, h_max)

    ax.xaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%1.2e'))

    ax.set_axisbelow(True)
    ax.grid(which='major', axis='both', linewidth=1., linestyle='-', color='0.6')
    ax.grid(which='minor', axis='both', linewidth=0.5, linestyle='-', color='0.75')

    ax.legend(loc='best', fontsize=14)
    plt.tight_layout()

    save_fig(fig, os.path.join(modpath, f"Fig{fig_id}"))
    print(f"Figure {fig_id} saved as:\n→ {os.path.join(modpath, f'Fig{fig_id}')}.pdf\n→ {os.path.join(modpath, f'Fig{fig_id}')}.png")


# =============================
# Orchestration
# =============================

def run(cfg: RunConfig) -> Dict[str, Any]:
    # Validate inputs
    if not os.path.exists(cfg.file_path):
        print(f"ERROR: File not found: {cfg.file_path}")
        return {"specimen_id": cfg.specimen_id, "specimen_folder": cfg.specimen_folder, "error": "file_not_found"}

    XX, YY, ZZ = load_and_slice_pcd(cfg.file_path)

    # Fig 1: 3D scatter
    fig1 = plt.figure(figsize=(7, 5), dpi=150, facecolor='white')
    ax = fig1.add_subplot(111, projection='3d')
    ax.scatter(XX, YY, ZZ, c='black')
    ax.set_xlabel("Specimen Edges [mm]")
    ax.set_ylabel("Cut Coordinate [mm]")
    ax.set_zlabel("Specimen Height [mm]")
    plt.tight_layout()
    save_fig(fig1, os.path.join(cfg.modpath, 'Fig1newone'))
    print(f"Figure 1 saved as:\n→ {os.path.join(cfg.modpath, 'Fig1newone')}.pdf \n→ {os.path.join(cfg.modpath, 'Fig1newone')}.png")

    # Fig 2: scatter ZZ vs XX
    plot_scatter(ZZ, XX, "Specimen Height [mm]", "Specimen Edges [mm]", os.path.join(cfg.modpath, 'Fig2new'))

    # Fig 3: separate min/max
    (X_max, Z_max), (X_min, Z_min) = separate_min_max(XX, ZZ)
    fig3, ax3 = plt.subplots(figsize=(7, 5), dpi=150)
    ax3.plot(Z_max, X_max, ls='none', marker='x', mec='red', ms=2., zorder=2)
    ax3.plot(Z_min, X_min, ls='none', marker='x', mec='orange', ms=2., zorder=2)
    ax3.set_xlabel("Specimen Height [mm]", fontsize=16)
    ax3.set_ylabel("Specimen Edges [mm]", fontsize=16)
    ax3.tick_params(axis='both', labelsize=14)
    ax3.grid(True)
    plt.tight_layout()
    save_fig(fig3, os.path.join(cfg.modpath, 'Fig3new'))
    print(f"Figure 3 saved as:\n→ {os.path.join(cfg.modpath, 'Fig3new')}.pdf\n→ {os.path.join(cfg.modpath, 'Fig3new')}.png")

    # Diameter curve
    x_field, D_X, D_X_interp = compute_specimen_diameter(Z_max, X_max, Z_min, X_min, cfg.h_max, cfg.modpath)

    # Initialize outputs
    d1 = d2 = strain_tangent = strain_integral = None
    strain_tangent_abs = strain_integral_abs = None
    L1_variant_1 = d0_auto = None
    strain_tangent_var1 = strain_integral_var_1 = None
    strain_tangent_var2 = strain_integral_var_2 = None
    l1_variant_3 = l2_variant_3 = strain_tangent_var3 = strain_integral_var_3 = None

    if D_X_interp is not None:
        d1, d2, strain_tangent = eps_ue(cfg.l2, cfg.l1, cfg.d0, D_X_interp)
        strain_integral = eps_ue_integral(cfg.l2, cfg.l1, cfg.d0, D_X_interp)

        print("\n                           +++")
        print("\n                           ")
        print("Results using User Defined Values L1 and L2:")
        print(f'Diameter at l1: {d1:.3f}')
        print(f'Diameter at l2: {d2:.3f}')
        print(f'Uniform Elongation Strain using Tangent Method :{strain_tangent:.3f} %')
        print(f'Uniform Elongation Strain using Integral Method :{strain_integral:.3f} %')

    # Smoothed curves
    D_X_filtered, D_X_derivative, D_X_derivative_smoothed = process_diameter_profile_auto(D_X)

    plot_diameter_comparison(x_field, D_X, D_X_filtered, cfg.modpath)
    plot_derivative_comparison(x_field, D_X_derivative, D_X_derivative_smoothed, cfg.modpath)

    if strain_tangent is not None and strain_integral is not None:
        print(f"Uniform Strain (Tangent): {strain_tangent:.3f} %")
        print(f"Uniform Strain (Integral): {strain_integral:.3f} %")

    print("\n                           +++")
    print("\n     Results using hardcoded threshold for l1 and minimum for l2:")

    l2_abs, min_dia_deriv = justfindl2(x_field, D_X_derivative_smoothed, cfg.h_max)
    l1_abs = justfindl1(x_field, D_X_derivative_smoothed, cfg.l1_th_fix, cfg.h_max)
    if l1_abs is not None:
        print(f'L1 absolute is : {l1_abs:.3f} mm ')
    else:
        print("⚠️ l1 could not be determined.")
    print(f'L2 absolute is : {l2_abs:.3f} mm ')
    print(f'Minimum Diameter Derivative is : {min_dia_deriv:.7f} mm ')

    if D_X_interp is not None and l1_abs is not None:
        d1_abs, d2_abs, strain_tangent_abs = eps_ue(l2_abs, l1_abs, cfg.d0, D_X_interp)
        strain_integral_abs = eps_ue_integral(l2_abs, l1_abs, cfg.d0, D_X_interp)
        print(f'Diameter at L1 absolute is : {d1_abs:.3f} mm ')
        print(f'Diameter at L2 absolute is : {d2_abs:.3f} mm ')
        print(f'Absolute Uniform Strain ( Tangent ) : {strain_tangent_abs:.3f} % ')
        print(f'Absolute Uniform Strain ( Integral ) : {strain_integral_abs:.3f} % ')

    print("\n                           +++")
    print("       Results using Relative threshold for l1 and minimum for l2 :")
    if D_X_interp is not None:
        L1_variant_1, d0_auto, derivatives_interpolation = l1_var_1(
            x_field, D_X_derivative_smoothed, cfg.h_max, D_X_interp, cfg.d0_auto_factor, cfg.min_factor
        )
        print(f'L1 for Variant 1 & 2 is : {L1_variant_1:.3f}')
        print(f'L2 Variant 1 & 2 is : {l2_abs:.3f}')
        print(f'D0_Auto is : {d0_auto:.3f}')

        print("\n                           +++")
        print("       Results for Variant 2 :")
        d1_var1, d2_var1, strain_tangent_var2 = eps_ue(l2_abs, L1_variant_1, cfg.d0, D_X_interp)
        print(f'Relative Uniform Strain ( Tangent ) Var 2: {strain_tangent_var2:.3f} % ')
        print(f'd1 ( Tangent ) Var 1 & 2: {d1_var1:.3f} mm ')
        print(f'd2( Tangent ) Var 1 & 2: {d2_var1:.3f} mm ')

        strain_integral_var_2 = eps_ue_integral(l2_abs, L1_variant_1, cfg.d0, D_X_interp)
        print(f'Relative Uniform Strain ( Integral ) variant 2: {strain_integral_var_2:.3f} % ')

        print("\n                           +++")
        print("       Results for Variant 1 :")
        _, _, strain_tangent_var1 = eps_ue(l2_abs, L1_variant_1, d0_auto, D_X_interp)
        print(f'Relative Uniform Strain ( Tangent ) Variant 1: {strain_tangent_var1:.3f} % ')

        strain_integral_var_1 = eps_ue_integral(l2_abs, L1_variant_1, d0_auto, D_X_interp)
        print(f'Relative Uniform Strain ( Integral ) variant 1: {strain_integral_var_1:.3f} % ')

    print("\n                           +++")
    print("       Results for Variant 3 :")
    l1_variant_3, _ = findl1_variant3(x_field, D_X_derivative_smoothed, cfg.h_max)
    l2_variant_3 = findl2_variant3(x_field, D_X_filtered, cfg.h_max, cfg.d0)
    print(f'L1 for Variant 3 is : {l1_variant_3:.3f}')
    print(f'L2 for Variant 3 is : {l2_variant_3:.3f}' if l2_variant_3 is not None else 'L2 for Variant 3 could not be determined')

    if D_X_interp is not None and l2_variant_3 is not None:
        d1_var3, d2_var3, strain_tangent_var3 = eps_ue(l2_variant_3, l1_variant_3, cfg.d0, D_X_interp)
        print(f'd1 Var 3: {d1_var3:.5f} mm ')
        print(f'd2 Var 3: {d2_var3:.5f} mm ')
        print(f'Relative Uniform Strain ( Tangent ) Variant 3: {strain_tangent_var3:.3f} % ')

        strain_integral_var_3 = eps_ue_integral(l2_variant_3, l1_variant_3, cfg.d0, D_X_interp)
        print(f'Relative Uniform Strain ( Integral ) variant 3: {strain_integral_var_3:.3f} % ')

    # L1/L2 plots
    if 'l1_abs' in locals() and l1_abs is not None:
        plot_L1_L2(x_field, D_X_derivative_smoothed, l1_abs, l2_abs, cfg.h_max, cfg.modpath, fig_id='30')
    if L1_variant_1 is not None:
        plot_L1_L2(x_field, D_X_derivative_smoothed, L1_variant_1, l2_abs, cfg.h_max, cfg.modpath, fig_id='40')
    if l2_variant_3 is not None:
        plot_L1_L2(x_field, D_X_derivative_smoothed, l1_variant_3, l2_variant_3, cfg.h_max, cfg.modpath, fig_id='50')

    # Pack results dict (no file write here; caller will aggregate)
    results = {
        "specimen_id": cfg.specimen_id,
        "specimen_folder": cfg.specimen_folder,
        "d0": cfg.d0,
        "l1": cfg.l1,
        "l2": cfg.l2,
        "h_max": cfg.h_max,
        "l1_th_fix": cfg.l1_th_fix,
        "min_factor": cfg.min_factor,
        "d0_auto_factor": cfg.d0_auto_factor,
        "export_suffix": cfg.export_suffix,
        "error": None,
        "d1_user": float(d1) if d1 is not None else None,
        "d2_user": float(d2) if d2 is not None else None,
        "strain_tangent_user": float(strain_tangent) if strain_tangent is not None else None,
        "strain_integral_user": float(strain_integral) if strain_integral is not None else None,
        "l1_abs": float(l1_abs) if 'l1_abs' in locals() and l1_abs is not None else None,
        "l2_abs": float(l2_abs) if 'l2_abs' in locals() and l2_abs is not None else None,
        "strain_tangent_abs": float(strain_tangent_abs) if strain_tangent_abs is not None else None,
        "strain_integral_abs": float(strain_integral_abs) if strain_integral_abs is not None else None,
        "l1_var1": float(L1_variant_1) if L1_variant_1 is not None else None,
        "l2_var1": float(l2_abs) if 'l2_abs' in locals() and l2_abs is not None else None,
        "strain_tangent_var1": float(strain_tangent_var1) if strain_tangent_var1 is not None else None,
        "strain_integral_var1": float(strain_integral_var_1) if strain_integral_var_1 is not None else None,
        "strain_tangent_var2": float(strain_tangent_var2) if strain_tangent_var2 is not None else None,
        "strain_integral_var2": float(strain_integral_var_2) if strain_integral_var_2 is not None else None,
        "l1_var3": float(l1_variant_3) if l1_variant_3 is not None else None,
        "l2_var3": float(l2_variant_3) if l2_variant_3 is not None else None,
        "strain_tangent_var3": float(strain_tangent_var3) if strain_tangent_var3 is not None else None,
        "strain_integral_var3": float(strain_integral_var_3) if 'strain_integral_var_3' in locals() and strain_integral_var_3 is not None else None
    }

    return results


# =============================
# Argument parsing & batch assembly
# =============================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Uniform elongation analysis from 3D point cloud (single or batch via JSON config)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None, help="Path to JSON config file.")

    # Single-specimen fields (may also be used to append to a batch)
    p.add_argument("--specimen-id", dest="specimen_id", type=str, help="Specimen ID (e.g., AZK6z2)")
    p.add_argument("--specimen-folder", dest="specimen_folder", type=str, help="Folder name (e.g., Eval_top)")

    # Shared parameters
    p.add_argument("--d0", type=float, help="Initial diameter d0 [mm]")
    p.add_argument("--l1", type=float, help="Length l1 [mm]")
    p.add_argument("--l2", type=float, help="Length l2 [mm]")
    p.add_argument("--h-max", dest="h_max", type=float, help="Maximum height to consider [mm]")

    p.add_argument("--l1-th-fix", dest="l1_th_fix", type=float, help="Threshold for l1 detection (absolute)")
    p.add_argument("--min-factor", dest="min_factor", type=float, help="Relative factor for l1 in Variant 1")
    p.add_argument("--d0-auto-factor", dest="d0_auto_factor", type=float, help="Factor for automatic d0 in Variant 1")

    p.add_argument("--export-suffix", dest="export_suffix", type=str, help="Suffix for export naming")
    p.add_argument("--base-path", dest="base_path", type=str, default="/home/dorin/Projects/Uniform-Elongation-measurement", help="Root path to projects")

    p.add_argument("--output-csv", dest="output_csv", type=str, default="results_batch.csv", help="Path/name for the aggregated CSV output")

    return p


def build_runconfigs(args: argparse.Namespace) -> Tuple[List[RunConfig], Dict[str, Any]]:
    """Return a list of RunConfig objects and a dict of the resolved shared params (for logging).

    Supports two config styles:
    A) Shared-params + specimens (each specimen has only id/folder)
    B) Per-specimen params (each specimen object can supply its own d0, l1, l2, ...)
       Any missing keys in a specimen fall back to shared values or CLI.
    """
    shared: Dict[str, Any] = {}

    # Load from config file if provided
    cfg_dict: Dict[str, Any] = {}
    if args.config:
        cfg_dict = read_json(args.config)
        if not isinstance(cfg_dict, dict):
            raise ValueError("Config JSON must contain an object at top-level.")

    # Gather shared parameters (from config first, then CLI overrides)
    for key in [
        "d0", "l1", "l2", "h_max",
        "l1_th_fix", "min_factor", "d0_auto_factor",
        "export_suffix", "base_path"
    ]:
        # precedence: CLI overrides config
        if getattr(args, key) is not None:
            shared[key] = getattr(args, key)
        elif key in cfg_dict:
            shared[key] = cfg_dict[key]

    specimens_from_config = cfg_dict.get("specimens") if isinstance(cfg_dict, dict) else None
    extra_from_cli = (args.specimen_id is not None and args.specimen_folder is not None)

    run_cfgs: List[RunConfig] = []

    if specimens_from_config:
        # Determine if per-specimen params are present
        per_spec_params = [
            "d0", "l1", "l2", "h_max",
            "l1_th_fix", "min_factor", "d0_auto_factor",
            "export_suffix", "base_path"
        ]
        for item in specimens_from_config:
            if not isinstance(item, dict) or "specimen_id" not in item or "specimen_folder" not in item:
                raise SystemExit("Each entry in 'specimens' must be an object with 'specimen_id' and 'specimen_folder'.")
            # Merge specimen-specific params with shared (specimen overrides shared)
            merged: Dict[str, Any] = {k: item.get(k, shared.get(k)) for k in per_spec_params}
            # Validate that required fields are available after merge
            missing = [k for k in per_spec_params if k != "base_path" and merged.get(k) is None]
            if missing:
                raise SystemExit(
                    f"Specimen {item.get('specimen_id')} missing required parameters: {', '.join(missing)}"
                )
            run_cfgs.append(
                RunConfig(
                    specimen_id=item["specimen_id"],
                    specimen_folder=item["specimen_folder"],
                    d0=float(merged["d0"]),
                    l1=float(merged["l1"]),
                    l2=float(merged["l2"]),
                    h_max=float(merged["h_max"]),
                    l1_th_fix=float(merged["l1_th_fix"]),
                    min_factor=float(merged["min_factor"]),
                    d0_auto_factor=float(merged["d0_auto_factor"]),
                    export_suffix=str(merged["export_suffix"]),
                    base_path=str(merged.get("base_path", "/home/dorin/Projects/Uniform-Elongation-measurement"))
                )
            )
        # Optionally append one extra specimen from CLI (uses shared params)
        if extra_from_cli:
            missing_shared = [k for k in [
                "d0", "l1", "l2", "h_max",
                "l1_th_fix", "min_factor", "d0_auto_factor",
                "export_suffix"
            ] if k not in shared]
            if missing_shared:
                raise SystemExit(
                    "Missing required shared parameters for CLI-appended specimen: " + ", ".join(missing_shared)
                )
            run_cfgs.append(RunConfig(specimen_id=args.specimen_id, specimen_folder=args.specimen_folder, **shared))

    else:
        # Single-specimen mode: must have specimen_id & folder from either CLI or config root
        sid = args.specimen_id or cfg_dict.get("specimen_id")
        sfolder = args.specimen_folder or cfg_dict.get("specimen_folder")
        if not sid or not sfolder:
            raise SystemExit("specimen_id and specimen_folder are required (via CLI or config).")
        # Validate shared fields in single mode
        missing_shared = [k for k in [
            "d0", "l1", "l2", "h_max",
            "l1_th_fix", "min_factor", "d0_auto_factor",
            "export_suffix"
        ] if k not in shared]
        if missing_shared:
            raise SystemExit(
                "Missing required parameters: " + ", ".join(missing_shared) +
                "\nProvide them via --config JSON and/or CLI flags."
            )

        run_cfgs.append(RunConfig(specimen_id=sid, specimen_folder=sfolder, **shared))

    return run_cfgs, shared


def write_batch_csv(rows: List[Dict[str, Any]], output_csv_path: str) -> None:
    if not rows:
        print("No rows to write.")
        return
    # Union of keys across all rows to be robust in case of missing values
    fieldnames = sorted({k for row in rows for k in row.keys()})
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True) if os.path.dirname(output_csv_path) else None
    with open(output_csv_path, mode="w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\nBatch results saved to {output_csv_path}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    run_cfgs, shared = build_runconfigs(args)

    # Print shared configuration
    print("Shared parameters (resolved):")
    for k in sorted(shared.keys()):
        print(f"  {k}: {shared[k]}")

    all_rows: List[Dict[str, Any]] = []

    for idx, cfg in enumerate(run_cfgs, start=1):
        print("\n=============================")
        print(f"Running specimen {idx}/{len(run_cfgs)}: {cfg.specimen_id} / {cfg.specimen_folder}")
        print("Resolved specimen config:")
        for k, v in asdict(cfg).items():
            print(f"  {k}: {v}")
        row = run(cfg)
        all_rows.append(row)

    # Write aggregated CSV
    write_batch_csv(all_rows, args.output_csv)

    # Exit code: 0 if no error in any row, else 1
    rc = 0 if all((r.get("error") is None) for r in all_rows) else 1
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
