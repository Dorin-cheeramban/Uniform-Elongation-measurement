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
from scipy.signal import savgol_filter 


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
    fig4.savefig(modpath + 'Fig4.pdf')
    fig4.savefig(modpath + 'Fig4.png', dpi=300)
    print(f"Figure 4 saved as:\n→ {modpath + 'Fig4.pdf'}\n→ {modpath + 'Fig4.png'}")
    
    return x_field,D_X,D_X_interp

def eps_ue(l2, l1, d0, D_X_interp):
    d1 = D_X_interp(l1)
    d2 = D_X_interp(l2)
    d_avg = (d1 + d2) / 2.0
    strain_tangent = (((d0 / d_avg) ** 2) - 1) * 100.
    return d1, d2, strain_tangent   

def eps_ue_integral(l2,l1, d0, D_X_interp):
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
        integrand += (D_X_interp(l2 - x) / 2.) ** 2 * dx
        VolumeResult = math.pi * integrand
        x += dx
        dxres += dx
        #print(f"[eps_ue_integral] dxres (effective integrated length): {dxres:.6f} mm")

    return (dxres - l0) / l0 * 100.

eps_ue_integral_value = vectorize(eps_ue_integral)


def auto_window_size(data_length, divisor=50, min_size=5):
    size = max(int(data_length / divisor), min_size)
    size = 101
    return size
    #return size + 1 if size % 2 == 0 else size

def auto_window_size_for_Derivative (data_length, divisor=50, min_size=5):
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


    # Final safety checks
    if poly_order >= window_size:
        poly_order = max(1, window_size - 2)
    if window_size % 2 == 0:
        window_size += 1

    #print(f"[Auto SGFilter] Deriv={deriv}, Window={window_size}, PolyOrder={poly_order}")
    return savgol_filter(data, window_length=window_size, polyorder=poly_order, deriv=deriv)

def apply_savgol_auto_Derivative (data, deriv=0):
    data = np.asarray(data)
    window_size = auto_window_size_for_Derivative(len(data))
    poly_order = auto_poly_order(window_size)

    # Final safety checks
    if poly_order >= window_size:
        poly_order = max(1, window_size - 2)
    if window_size % 2 == 0:
        window_size += 1

    #print(f"[Auto SGFilter] Deriv={deriv}, Window={window_size}, PolyOrder={poly_order}")
    return savgol_filter(data, window_length=window_size, polyorder=poly_order, deriv=deriv)


def process_diameter_profile_auto(D_X):
    D_X_filtered = apply_savgol_auto( D_X, deriv=0)          # Step 1: Smooth diameter
    D_X_derivative = apply_savgol_auto( D_X, deriv=1)        # Step 2: Compute derivative
    D_X_derivative_smoothed = apply_savgol_auto_Derivative(D_X_derivative, deriv=0)  # Step 3: Smooth derivative

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

    plt.legend(fontsize=14, loc='best')  # ✅ Adds the legend box
    plt.tight_layout()

    filename_base = modpath + 'Fig10'
    plt.savefig(filename_base + '.pdf')
    plt.savefig(filename_base + '.png', dpi=300)
    print(f"Figure 10 saved as:\n→ {filename_base}.pdf\n→ {filename_base}.png")


def plot_derivative_comparison(x_field, D_X_derivative,D_X_derivative_smoothed, modpath):
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

    # Format tick labels
    ax.xaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%1.2e'))

    # Grid
    ax.set_axisbelow(True)
    ax.grid(which='major', axis='both', linewidth=1., linestyle='-', color='0.6')
    ax.grid(which='minor', axis='both', linewidth=0.5, linestyle='-', color='0.75')

    ax.legend(loc='best', fontsize=14)
    plt.tight_layout()

    filename_base = modpath + 'Fig20'
    fig20.savefig(filename_base + '.pdf')
    fig20.savefig(filename_base + '.png', dpi=300)
    print(f"Figure 20 saved as:\n→ {filename_base}.pdf\n→ {filename_base}.png")


def justfindl2(x_field, D_X_derivative_smoothed, h_max):
    min_deriv = D_X_derivative_smoothed[0]
    l2 = x_field[0]

    for i in range(1,len(D_X_derivative_smoothed)):
        if D_X_derivative_smoothed[0] < 1e-3 :
            print(" D_X Fileterd data too low")
            break

        # ✅ Update l2 if smaller derivative found (true minimum)
        if D_X_derivative_smoothed[i] < min_deriv:
            min_deriv = D_X_derivative_smoothed[i]
            l2 = x_field[i]

        if x_field[i] > h_max:
            break
    
    return l2, min_deriv

def justfindl1(x_field, D_X_derivative_smoothed,l1_th_fix, h_max):
    l1 = x_field[0]

    for i in range(1,len(D_X_derivative_smoothed)):

        if x_field[i] > h_max:
            break
    
        #✅ Set l1 when first value below threshold (only once)
        if D_X_derivative_smoothed[i] < l1_th_fix:
            l1 = x_field[i]
            return l1

        
    return None
                        


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
        d1,d2, strain_tangent = eps_ue(l2, l1, d0, D_X_interp)
        strain_integral = eps_ue_integral(l2, l1, d0, D_X_interp)

        print(f'Diameter at l1: {d1:.3f}')
        print(f'Diameter at l2: {d2:.3f}')
        print(f'Uniform Elongation Strain using Tangent Method :{strain_tangent:.3f} %')
        print(f'Uniform Elongation Strain using Integeral Method :{strain_integral:.3f} %')
        
    D_X_filtered, D_X_derivative, D_X_derivative_smoothed = process_diameter_profile_auto(D_X)

    plot_diameter_comparison(x_field, D_X, D_X_filtered, modpath)

    plot_derivative_comparison(x_field, D_X_derivative,D_X_derivative_smoothed, modpath)

    if strain_tangent is not None:
        print(f"Uniform Elongation (Tangent): {strain_tangent:.3f} %")
        print(f"Uniform Elongation (Integral): {strain_integral:.3f} %")
    
    #plt.show()
    #plt.close()

    l2_abs, min_dia_deriv = justfindl2(x_field, D_X_derivative_smoothed, h_max)
    print(f'L2 absolute is : {l2_abs} mm ')
    print(f'Minimum Diameter Derivatrive is : {min_dia_deriv} mm ')

    l1_abs = justfindl1(x_field,D_X_derivative_smoothed,l1_th_fix, h_max)
    if l1_abs is not None:
        print(f'L1 absolute is : {l1_abs} mm ')
    else : 
        print("⚠️ l1 could not be determined.")

    if  D_X_interp is not None:
        d1_abs,d2_abs, strain_tangent_abs = eps_ue(l2_abs, l1_abs, d0, D_X_interp)
        strain_integral_abs = eps_ue_integral(l2_abs, l1_abs, d0, D_X_interp)
        print(f'Diameter at L1 absolute is : {d1_abs} mm ')
        print(f'Diameter at L2 absolute is : {d2_abs} mm ')
        print(f'Absolute Uniform Elongation ( Tangent ) : {strain_tangent_abs} mm ')
        print(f'Absolute Uniform Elongation ( Integeral ) : {strain_integral_abs} mm ')

if __name__ == '__main__':
    main()



