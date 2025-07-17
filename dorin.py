#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 12:55:43 2022

@author: Erb, Kmn
"""

# v1: h_max inserted in the last 4 plots instead of fixed values
# v2: Some further printing and output improvement
# v3: l_th export correction
# v4: Slide modifications and automatization
# v5: (�berf�hrt in 11_* f�hr zu fr�h abgeschnittene Proben)
# v6: Anderer Speicherort (eine Ebene hoeher) um Schreibrechteproblematik zu entschaerfen, falls jemand anderes die ply aus makemake erzeugt hat & 
#     in eps_ue_integral Suche von l2 ausgehend nach vorne, so dass Problem behoben, dass nicht genug x-Werte vorliegen
#     Bei der Bestimmung des Minimums des Durchmessers ist dieser nun auf min 1e-3 begrenzt
#     vorne und hinten wird jeweils ein Bereich abgeschnitten, um geometrische Ausreiser zu entfernen
# v9: Threshold an zwei Stellen "if D_x_filtered_deriv_filtered[i]<1e-4:" auf 1e-4 geaendert (fuer Auswertung der WZV sZW ..)

#from pylab import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np  # if you use numpy functions

from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
import pandas as pd
import open3d as o3d
import os
import locale
from scipy.signal import savgol_filter
import sys


#rc('text', usetex=True)
#rc('font', family='serif', serif='charter')
#rc('text.latex', preamble=r'\usepackage{amstext}'
                           #r"\usepackage[squaren]{SIunits}"
                           #r"\usepackage[bitstream-charter]{mathdesign}"
                           #r"\usepackage{setspace}"
                           #r"\usepackage{nicefrac}"
                           #r"\doublespacing"
                           #r"\setlength{\parindent}{0pt}")
#serif='charter' <-- Text TUD Schriftart
#SIUnits <-- F�r Celsius usw. http://computer.wer-weiss-was.de/textverarbeitung/banale_frage_zu_latex-1874179.html
#amstext <-- \text{} in Formeln m�glich
#bitstream-charter <-- TUD Formel-Schriftarten
#setspace, onehalfspaceing <-- Abstand bei Zeilenumbruch
#setlength <-- Erste Zeile nicht einger�ckt
#locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
###
# Load Meshlab lib
#os.system('source /opt/python/3.10_latest/bin/activate')


SpecimenID=str(sys.argv[1])  #'AZK6z3'
SpecimenFolder=str(sys.argv[2])  #'Eval_top'

d0 = float(sys.argv[3])  #8.15 #mm
l1 = float(sys.argv[4])  #20.0 #mm
l2 = float(sys.argv[5])  #30.0 #mm
h_max = float(sys.argv[6])  #36.0 #mm

l1_th_fix=float(sys.argv[7])  #5e-3 dD/dh threshhold
min_Factor=float(sys.argv[8])  #2 dD/dh threshhold
d0_auto_Factor=float(sys.argv[9])  #1 Factor between du2 und d0

exportFileSuffix='_'+str(sys.argv[10])    #standard'

#/Users/admin/Projects

#path='/mnt/KB-H/TV/02_Forschungsvorhaben/D56-RobusteBruchverformungskennwerte/04_Daten/Scans/0001_Test_GUI_V1/'+SpecimenID+'/'+SpecimenFolder+'/'
path='/Users/admin/Projects/'+SpecimenID+'/'+SpecimenFolder+'/'

#modpath='/mnt/KB-H/TV/02_Forschungsvorhaben/D56-RobusteBruchverformungskennwerte/04_Daten/Scans/0001_Test_GUI_V1/'+SpecimenID+'/'+SpecimenFolder+'_'
modpath='/Users/admin/Projects/'+SpecimenID+'/'+SpecimenFolder+'_'


file= 'specimenFull.ply'


####
# Convert 3D to 2D hull



File = path + file

if not os.path.exists(File):
    print(f"ERROR: File not found: {File}")
    sys.exit(1)
    
pcd_read = o3d.io.read_point_cloud(File)  #pcd_read is an object of type open3d.geometry.PointCloud. It holds the point cloud data you just loaded from your .ply file using Open3D’s read_point_cloud() function.

points_matrix = np.array(pcd_read.points)   #Now, to work with these points using NumPy or Pandas, you need them in a NumPy array format. asarray() converts this Open3D Vector3dVector into a standard NumPy array
# plt.plot(points_matrix[:,2],points_matrix[:,1])

# X=pd.DataFrame(points_matrix)*10**3
X=pd.DataFrame(points_matrix)  #points_matrix at this point is a NumPy array with shape (N, 3) — meaning N points, each with 3 values: [x, y, z]. Now we convert this NumPy array to a Pandas DataFrame named X.

print(X[1].min(), X[1].max())


print(min(X[1])) #X[1] accesses column 1 of DataFrame X, which contains all y-values of the points. min(X[1]) finds the minimum y-coordinate value in your point cloud. Then it prints this minimum y-value.





#3D View
d3_View = 0  #If d3_View is set to 1, it shows the full 3D point cloud interactively using Open3D’s viewer. Here it's 0, so it skips this step.
if d3_View == 1:
    o3d.visualization.draw_geometries([pcd_read])

#select region for example y_values=const.  #Select a horizontal 2D "slice" of points at a specific Y-coordinate

#value=0  #ou're choosing a Y-value = 228 (your "cutting plane"). intervall defines a small range around 228 (like a thin horizontal band).
#value=228  

value = 0

intervall=0.0001



value_plus=value+intervall  #So the band lies between 227.9999 and 228.0001.
value_minus=value-intervall

XX=X[0][(X[1]>value_minus) & (X[1]<value_plus)]  #This filters all the points whose Y-coordinate falls inside that band. Then extracts their corresponding X, Y, and Z values.
YY=X[1][(X[1]>value_minus) & (X[1]<value_plus)]
ZZ=X[2][(X[1]>value_minus) & (X[1]<value_plus)]

#XX=X[0]
#YY=X[1]
#ZZ=X[2]



print(XX)
print(ZZ)
XX=XX.to_numpy()  #So that Matplotlib can easily handle them for plotting
YY=YY.to_numpy()
ZZ=ZZ.to_numpy()




#plot 3D
#Figure 1
fig1=plt.figure(1,figsize=(7,5), dpi=150, facecolor='white')  #✅ This creates a new figure window for the plot.1 is the figure number.figsize=(7,5) sets the size of the figure in inches.dpi=150 means the figure resolution is 150 dots per inch. facecolor='white' makes the background white.

ax = fig1.add_subplot(111, projection='3d') #✅ This creates a 3D axis system inside the figure. axes(projection='3d') tells Matplotlib we want a 3D plot.
ax.scatter(XX,YY,ZZ,c='black') #✅ This creates a 3D scatter plot on those axes. XX, YY, ZZ are arrays of the x, y, z coordinates of your points.c='black' makes the points black.
ax.set_xlabel(r'Specimen Edges $\ h$ $[\text{mm}]$') #✅ These set the axis labels for X, Y, and Z axes. The r'...' tells Python this is a raw string — good for LaTeX-style math text like $\ h$.The labels indicate physical meanings and units for each axis.
ax.set_ylabel(r'Cut Coordinate $\ h$ $[\text{mm}]$')
ax.set_zlabel(r'Specimen Height $\ h$ $[\text{mm}]$')
plt.tight_layout() #✅ This automatically adjusts the figure’s layout to ensure labels and titles fit nicely without overlapping.
Filename=sys.argv[0]
Filename=modpath + r'Fig1.pdf'    #Better way : 
                                                    #Filename_pdf = modpath + 'Fig1.pdf' 
                                                    #Filename_png = modpath + 'Fig1.png'
                                                    #fig1.savefig(Filename_pdf)
                                                    #fig1.savefig(Filename_png)
Filename1_pdf = modpath + 'Fig1.pdf'
Filename1_png = modpath + 'Fig1.png'

fig1.savefig(Filename1_pdf)
fig1.savefig(Filename1_png, dpi=300)
print(f"Figure 1 saved as:\n→ {Filename1_pdf}\n→ {Filename1_png}")

  #✅ These two lines assign filenames to save your figure. modpath is a string containing the path where you want to save the file.
plt.show()
sys.exit()


