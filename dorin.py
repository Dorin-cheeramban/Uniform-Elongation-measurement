import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import open3d as o3d
from scipy.interpolate import interp1d
import os
import sys
import math
from numpy import vectorize


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
    plt.savefig(filename_base + '.pdf')
    plt.savefig(filename_base + '.png', dpi=300)
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
    D_x = Edge_max_x_interp(x_field) - Edge_min_x_interp(x_field)
    
    x_field_reverse = x_field[::-1]
    D_x_interp = interp1d(x_field_reverse, D_x, kind='linear', bounds_error=False, fill_value="extrapolate")
    D_x = D_x_interp(x_field)

    fig4, ax4 = plt.subplots(figsize=(7, 5), dpi=150, facecolor='white')
    ax4.plot(x_field, D_x, ls='none', marker='x', mec='black', ms=2., zorder=2)
    ax4.set_xlabel("Specimen Height [mm]", fontsize=16)
    ax4.set_ylabel("Specimen Diameter [mm]", fontsize=16, rotation=90)
    ax4.tick_params(axis='both', labelsize=14)
    ax4.set_axisbelow(True)
    ax4.set_xlim(-0.5, h_max)
    ax4.xaxis.set_major_formatter(FormatStrFormatter('%1.3f'))
    ax4.yaxis.set_major_formatter(FormatStrFormatter('%1.2e'))
    ax4.xaxis.set_ticks_position('both')
    plt.tight_layout()
    fig4.savefig(modpath + 'Fig4.pdf')
    fig4.savefig(modpath + 'Fig4.png', dpi=300)
    print(f"Figure 4 saved as:\n→ {modpath + 'Fig4.pdf'}\n→ {modpath + 'Fig4.png'}")
    
    return x_field,D_x,D_x_interp

def eps_ue(l1, l2, d0, D_x_interp):
    d1 = D_x_interp(l1)
    d2 = D_x_interp(l2)
    d_avg = (d1 + d2) / 2.0
    strain = (((d0 / d_avg) ** 2) - 1) * 100.
    return d1, d2, strain   

def eps_ue_integral(l1, l2, d0, D_x_interp):
    l0 = l2 - l1
    V0 = math.pi * (d0 ** 2) / 4. * l0

    dx = 0.001
    integrand = 0.
    dxres = 0.
    VolumeResult = 0.
    x = dx / 2.

    while VolumeResult < V0:
        integrand += (D_x_interp(l2 - x) / 2.) ** 2 * dx
        VolumeResult = math.pi * integrand
        x += dx
        dxres += dx

    return (dxres - l0) / l0 * 100.

eps_ue_integral = vectorize(eps_ue_integral)

def main():
    if len(sys.argv) != 11:
        print("Usage: script.py SpecimenID SpecimenFolder d0 l1 l2 h_max l1_th_fix min_Factor d0_auto_Factor exportFileSuffix")
        sys.exit(1)

    SpecimenID, SpecimenFolder = sys.argv[1], sys.argv[2]
    d0, l1, l2, h_max = map(float, sys.argv[3:7])
    l1_th_fix, min_Factor, d0_auto_Factor = map(float, sys.argv[7:10])
    exportFileSuffix = '_' + str(sys.argv[10])

    base_path = f'/home/dorin/projects/{SpecimenID}/{SpecimenFolder}/'
    file_path = os.path.join(base_path, 'specimenFull.ply')
    modpath = f'/home/dorin/projects/{SpecimenID}/{SpecimenFolder}_' + exportFileSuffix + '_'

    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        sys.exit(1)

    XX, YY, ZZ = load_and_slice_pcd(file_path)

    fig1 = plt.figure(figsize=(7, 5), dpi=150, facecolor='white')
    ax = fig1.add_subplot(111, projection='3d')
    ax.scatter(XX, YY, ZZ, c='black')
    ax.set_xlabel("Specimen Edges [mm]")
    ax.set_ylabel("Cut Coordinate [mm]")
    ax.set_zlabel("Specimen Height [mm]")
    plt.tight_layout()
    fig1.savefig(modpath + 'Fig1newone.pdf')
    fig1.savefig(modpath + 'Fig1newone.png', dpi=300)
    print(f"Figure 1 saved as:\n→ {modpath + 'Fig1newone.pdf'} \n→ {modpath  + 'Fig1newone.png'}")
    

    plot_scatter(ZZ, XX, "Specimen Height [mm]", "Specimen Edges [mm]", modpath + 'Fig2new')

    (X_max, Z_max), (X_min, Z_min) = separate_min_max(XX, ZZ)
    fig3, ax3 = plt.subplots(figsize=(7, 5), dpi=150)
    ax3.plot(Z_max, X_max, ls='none', marker='x', mec='red', ms=2., zorder=2)
    ax3.plot(Z_min, X_min, ls='none', marker='x', mec='orange', ms=2., zorder=2)
    ax3.set_xlabel("Specimen Height [mm]", fontsize=16)
    ax3.set_ylabel("Specimen Edges [mm]", fontsize=16)
    ax3.tick_params(axis='both', labelsize=14)
    ax3.grid(True)
    plt.tight_layout()
    fig3.savefig(modpath + 'Fig3new.pdf')
    fig3.savefig(modpath + 'Fig3new.png', dpi=300)
    print(f"Figure 3 saved as:\n→ {modpath + 'Fig3new.pdf'}\n→ {modpath + 'Fig3new.png'}")
    

    x_field, D_X, D_X_interp = compute_specimen_diameter(Z_max, X_max, Z_min, X_min, h_max, modpath )
    
    if  D_X_interp is not None:
        d1,d2, strain = eps_ue(l1, l2, d0, D_X_interp)
        strain_integral = eps_ue_integral(l1, l2, d0, D_X_interp)
        print(f'Diameter at l1: {d1:.3f}')
        print(f'Diameter at l2:{d2:.3f}')
        print(f'Uniform Elongation Strain using Tangent Method:{strain:.3f} %')
        print(f'Uniform Elongation Strain using Integeral Method:{strain_integral:.3f} %')
        

    plt.show()
    plt.close()

if __name__ == '__main__':
    main()




