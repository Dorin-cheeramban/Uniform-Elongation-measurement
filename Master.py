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
# v5: (überführt in 11_* führ zu früh abgeschnittene Proben)
# v6: Anderer Speicherort (eine Ebene hoeher) um Schreibrechteproblematik zu entschaerfen, falls jemand anderes die ply aus makemake erzeugt hat & 
#     in eps_ue_integral Suche von l2 ausgehend nach vorne, so dass Problem behoben, dass nicht genug x-Werte vorliegen
#     Bei der Bestimmung des Minimums des Durchmessers ist dieser nun auf min 1e-3 begrenzt
#     vorne und hinten wird jeweils ein Bereich abgeschnitten, um geometrische Ausreiser zu entfernen
# v9: Threshold an zwei Stellen "if D_x_filtered_deriv_filtered[i]<1e-4:" auf 1e-4 geaendert (fuer Auswertung der WZV sZW ..)

from pylab import *
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
import pandas as pd
import open3d as o3d
import os
import locale
from scipy.signal import savgol_filter


rc('text', usetex=True)
rc('font', family='serif', serif='charter')
rc('text.latex', preamble=r'\usepackage{amstext}'
                           #r"\usepackage[squaren]{SIunits}"
                           #r"\usepackage[bitstream-charter]{mathdesign}"
                           r"\usepackage{setspace}"
                           #r"\usepackage{nicefrac}"
                           r"\doublespacing"
                           r"\setlength{\parindent}{0pt}")
#serif='charter' <-- Text TUD Schriftart
#SIUnits <-- Für Celsius usw. http://computer.wer-weiss-was.de/textverarbeitung/banale_frage_zu_latex-1874179.html
#amstext <-- \text{} in Formeln möglich
#bitstream-charter <-- TUD Formel-Schriftarten
#setspace, onehalfspaceing <-- Abstand bei Zeilenumbruch
#setlength <-- Erste Zeile nicht eingerückt
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



path='/mnt/KB-H/TV/02_Forschungsvorhaben/D56-RobusteBruchverformungskennwerte/04_Daten/Scans/0001_Test_GUI_V1/'+SpecimenID+'/'+SpecimenFolder+'/'
modpath='/mnt/KB-H/TV/02_Forschungsvorhaben/D56-RobusteBruchverformungskennwerte/04_Daten/Scans/0001_Test_GUI_V1/'+SpecimenID+'/'+SpecimenFolder+'_'

file= 'specimenFull.ply'



####
# Convert 3D to 2D hull



File = path + file
pcd_read = o3d.io.read_point_cloud(File)

points_matrix = asarray(pcd_read.points)
# plt.plot(points_matrix[:,2],points_matrix[:,1])

# X=pd.DataFrame(points_matrix)*10**3
X=pd.DataFrame(points_matrix)
print(min(X[1]))

#3D View
d3_View = 0
if d3_View == 1:
    o3d.visualization.draw_geometries([pcd_read])

#select region for example y_values=const.

#value=0
value=228
intervall=0.0001



value_plus=value+intervall
value_minus=value-intervall

XX=X[0][(X[1]>value_minus) & (X[1]<value_plus)]
YY=X[1][(X[1]>value_minus) & (X[1]<value_plus)]
ZZ=X[2][(X[1]>value_minus) & (X[1]<value_plus)]

print(XX)
print(ZZ)
XX=XX.to_numpy()
YY=YY.to_numpy()
ZZ=ZZ.to_numpy()

#plot 3D
#Figure 1
fig1=plt.figure(1,figsize=(7,5), dpi=150, facecolor='white')

ax = axes(projection ='3d')
ax.scatter(XX,YY,ZZ,c='black')
xlabel(r'Specimen Edges $\ h$ $[\text{mm}]$')
ylabel(r'Cut Coordinate $\ h$ $[\text{mm}]$')
ax.set_zlabel(r'Specimen Height $\ h$ $[\text{mm}]$')
tight_layout()
Filename=sys.argv[0]
Filename=modpath + r'Fig1.pdf'
Filename=modpath + r'Fig1.png'


savefig(Filename)




#2D zx plane
fig2=plt.figure(2,figsize=(7,5), dpi=150, facecolor='white')
#plot(ZZ, XX, marker='x'  )

plot(ZZ,XX,ls='none',marker='x',mec='black',ms=2.,zorder=2)
#plot(h_x,D_x_filtered_deriv,ls='--',c='red', zorder=2)
#plot(NA_2,deps_eq_design,ls='none',marker='^',mfc='white',mec='red',ms=10.,fillstyle='full', zorder=2, label=r"Design RO")


xmin=0.
xmax=20.

ymin=-2e-2
ymax=2e-2

xlabel(r"Specimen Height $\ h$ $[\text{mm}]$", fontsize=16)
ylabel(r"\begin{center} $\displaystyle \text{Specimen Edges} \ [\text{mm}]$ \end{center}", fontsize=16, rotation=90)


yscale('linear')
xscale('linear')

xticks(fontsize=14.)
yticks(fontsize=14.)


gca().set_axisbelow(True)

#gca().set_xlim(xmin,xmax)
#gca().set_ylim(ymin,ymax)

x_major_formatter = FormatStrFormatter('%1.3f')
y_major_formatter = FormatStrFormatter('%1.2e')

gca().xaxis.set_major_formatter(x_major_formatter)
gca().xaxis.set_ticks_position('both')

gca().yaxis.set_major_formatter(y_major_formatter)

gca().grid(which='major', axis='x', linewidth=1., linestyle='-', color='0.6')
gca().grid(which='minor', axis='x', linewidth=0.5, linestyle='-', color='0.75')
gca().grid(which='major', axis='y', linewidth=1., linestyle='-', color='0.6')
gca().grid(which='minor', axis='y', linewidth=0.5, linestyle='-', color='0.75')

#gca().text((xmax-xmin)*0.54-(xmax+xmin)/2.,(ymax-ymin)*0.35+(ymax+ymin)/2., BoxText, size=16, bbox={'facecolor':'white', 'pad':20, 'lw':1})

tight_layout()
Filename=sys.argv[0]
Filename=modpath + r'Fig2.pdf'
Filename=modpath + r'Fig2.png'


savefig(Filename)


###
# Devide in Max min

X_max=[]
Z_max=[]
X_min=[]
Z_min=[]
for i in arange(0,size(XX)):
  if XX[i]>0:
    Z_max.append(ZZ[i])
    X_max.append(XX[i])
  else:
    Z_min.append(ZZ[i])
    X_min.append(XX[i])



fig3=plt.figure(3,figsize=(7,5), dpi=150, facecolor='white')
#plot(ZZ, XX, marker='x'  )

plot(Z_max,X_max,ls='none',marker='x',mec='red',ms=2.,zorder=2)
plot(Z_min,X_min,ls='none',marker='x',mec='orange',ms=2.,zorder=2)

#plot(h_x,D_x_filtered_deriv,ls='--',c='red', zorder=2)
#plot(NA_2,deps_eq_design,ls='none',marker='^',mfc='white',mec='red',ms=10.,fillstyle='full', zorder=2, label=r"Design RO")


xmin=0.
xmax=20.

ymin=-2e-2
ymax=2e-2

xlabel(r"Specimen Height $\ h$ $[\text{mm}]$", fontsize=16)
ylabel(r"\begin{center} $\displaystyle \text{Specimen Edges} \ [\text{mm}]$ \end{center}", fontsize=16, rotation=90)


yscale('linear')
xscale('linear')

xticks(fontsize=14.)
yticks(fontsize=14.)


gca().set_axisbelow(True)

#gca().set_xlim(xmin,xmax)
#gca().set_ylim(ymin,ymax)

x_major_formatter = FormatStrFormatter('%1.3f')
y_major_formatter = FormatStrFormatter('%1.2e')

gca().xaxis.set_major_formatter(x_major_formatter)
gca().xaxis.set_ticks_position('both')

gca().yaxis.set_major_formatter(y_major_formatter)

gca().grid(which='major', axis='x', linewidth=1., linestyle='-', color='0.6')
gca().grid(which='minor', axis='x', linewidth=0.5, linestyle='-', color='0.75')
gca().grid(which='major', axis='y', linewidth=1., linestyle='-', color='0.6')
gca().grid(which='minor', axis='y', linewidth=0.5, linestyle='-', color='0.75')

#gca().text((xmax-xmin)*0.54-(xmax+xmin)/2.,(ymax-ymin)*0.35+(ymax+ymin)/2., BoxText, size=16, bbox={'facecolor':'white', 'pad':20, 'lw':1})

tight_layout()
Filename=sys.argv[0]
Filename=modpath + r'Fig3.pdf'
Filename=modpath + r'Fig3.png'


savefig(Filename)




###
# Extract Diameter
Edge_max_x_interp=interp1d(Z_max,X_max)
Edge_min_x_interp=interp1d(Z_min,X_min)

start_x=max(Z_max[0],Z_min[0])
end_x=min(Z_max[-1],Z_min[-1])

x_field=linspace(start_x,end_x,1000)

D_x = Edge_max_x_interp(x_field) - Edge_min_x_interp(x_field)


x_field_reverse=[]
#D_x_reverse=[]
i_max=size(x_field)

for i in arange(0,i_max):
   j=i_max-(i+1)
   x_field_reverse.append(x_field[j])
   #D_x_reverse.append(D_x[j])
   
D_x_interp=interp1d(x_field_reverse,D_x)
D_x=D_x_interp(x_field)



fig4=plt.figure(4,figsize=(7,5), dpi=150, facecolor='white')
#plot(ZZ, XX, marker='x'  )

plot(x_field,D_x,ls='none',marker='x',mec='black',ms=2.,zorder=2)

#plot(h_x,D_x_filtered_deriv,ls='--',c='red', zorder=2)
#plot(NA_2,deps_eq_design,ls='none',marker='^',mfc='white',mec='red',ms=10.,fillstyle='full', zorder=2, label=r"Design RO")


xmin=-0.5
xmax=h_max

ymin=-2e-2
ymax=2e-2

xlabel(r"Specimen Height $\ h$ $[\text{mm}]$", fontsize=16)
ylabel(r"\begin{center} $\displaystyle \text{Specimen Diameter} \ [\text{mm}]$ \end{center}", fontsize=16, rotation=90)


yscale('linear')
xscale('linear')

xticks(fontsize=14.)
yticks(fontsize=14.)


gca().set_axisbelow(True)

gca().set_xlim(xmin,xmax)
#gca().set_ylim(ymin,ymax)

x_major_formatter = FormatStrFormatter('%1.3f')
y_major_formatter = FormatStrFormatter('%1.2e')

gca().xaxis.set_major_formatter(x_major_formatter)
gca().xaxis.set_ticks_position('both')

gca().yaxis.set_major_formatter(y_major_formatter)

#gca().grid(which='major', axis='x', linewidth=1., linestyle='-', color='0.6')
#gca().grid(which='minor', axis='x', linewidth=0.5, linestyle='-', color='0.75')
#gca().grid(which='major', axis='y', linewidth=1., linestyle='-', color='0.6')
#gca().grid(which='minor', axis='y', linewidth=0.5, linestyle='-', color='0.75')

#gca().text((xmax-xmin)*0.54-(xmax+xmin)/2.,(ymax-ymin)*0.35+(ymax+ymin)/2., BoxText, size=16, bbox={'facecolor':'white', 'pad':20, 'lw':1})

tight_layout()
Filename=sys.argv[0]
Filename=modpath + r'Fig4.pdf'
Filename=modpath + r'Fig4.png'

savefig(Filename)

#show()




#export
#result = pd.concat(ZZ,XX,axis=1) 
#pd.DataFrame([x_field,D_x]).to_csv(path + r'export.csv' , header=True, index=None, sep=',', line_terminator='\n', decimal='.') 
#[x_field,D_x].tofile(path + r'export.csv', sep = ',')


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
####
# Evaluation
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


####
# Cut the data at h_max
for i in arange(0,size(x_field)):
   if x_field[i]>h_max:
      break

h_x=x_field[:i]
D_x=D_x[:i]


def eps_ue(l1,l2,d0):
    return (((d0/((D_x_interp(l1)+D_x_interp(l2))/2.))**2.)-1)*100.


print('Diameter at l1:', D_x_interp(l1))
print('Diameter at l2:', D_x_interp(l2))




###
# New Method
def eps_ue_integral(l1,l2,d0):
    l0=l2-l1 #mm
    V0 = math.pi*d0**2/4.*l0

    dx=0.001 #mm
    integrand=0.
    dxres=0.
    VolumeResult=0
    x=dx/2.

    while VolumeResult < V0:
        integrand+=(D_x_interp(l2-x)/2.)**2*dx
        VolumeResult=math.pi*integrand
        x+=dx
        dxres+=dx

    return (dxres-l0)/l0*100.
eps_ue_integral=vectorize(eps_ue_integral)

print('+++')
print('Results using user-defined l1 and l2')
print('l1: ', l1)
print('l2: ', l2)


print('eps_ue_tangent: ', eps_ue(l1,l2,d0))
print('eps_ue_integral: ', eps_ue_integral(l1,l2,d0))




## Savitzky Golay Filter - Einmal

## Define Window Size
DataValues=int(size(D_x)/50.)
if DataValues%2==0:  #WindowSize must be odd
    WindowSize=DataValues+1
else:
    WindowSize=DataValues

## Define Polynomial Order
if WindowSize<3+2:
    PolyOrder=1
elif WindowSize<5+2:
    PolyOrder=3
else:
    PolyOrder=5

#PolyOrder=3
print('PolyOrder', PolyOrder)
print('WindowSize', WindowSize) 
    
WindowSize=101
PolyOrder=1

sg_out=savgol_filter(array(D_x), WindowSize, PolyOrder) # Last Two Values: window size, polynomial order

D_x_filtered=sg_out
#print(D_x_filtered)


sg_out=savgol_filter(array(D_x), WindowSize, PolyOrder,1) # Last Two Values: window size, polynomial order

D_x_filtered_deriv=sg_out
#print(D_x_filtered_deriv)



## Savitzky Golay Filter - Einmal

## Define Window Size
DataValues=int(size(D_x_filtered_deriv)/50.)
if DataValues%2==0:  #WindowSize must be odd
    WindowSize=DataValues+1
else:
    WindowSize=DataValues

## Define Polynomial Order
if WindowSize<3+2:
    PolyOrder=1
elif WindowSize<5+2:
    PolyOrder=3
else:
    PolyOrder=5

#PolyOrder=3
print('PolyOrder', PolyOrder)
print('WindowSize', WindowSize) 
    
WindowSize=201
PolyOrder=1

sg_out=savgol_filter(array(D_x_filtered_deriv), WindowSize, PolyOrder) # Last Two Values: window size, polynomial order

D_x_filtered_deriv_filtered=sg_out





#Figure 10
fig10=plt.figure(10,figsize=(7,5), dpi=150, facecolor='white')
plot(h_x,D_x,ls='-',c='black', zorder=1)
plot(h_x,D_x_filtered,ls='--',c='red', zorder=2)
#plot(NA_2,deps_eq_design,ls='none',marker='^',mfc='white',mec='red',ms=10.,fillstyle='full', zorder=2, label=r"Design RO")

#legend(loc=2, fontsize=14.)

#BoxText=r"\textbf{Spitzenwerte: uA16dk82} \\ Lastabfall"

xmin=0.
xmax=h_max

ymin=0
ymax=12.

xlabel(r"Specimen Height $\ h$ $[\text{mm}]$", fontsize=16)
ylabel(r"\begin{center} $\displaystyle \text{Specimen Diamater} \ d \ [\text{mm}]$ \end{center}", fontsize=16, rotation=90)


yscale('linear')
xscale('linear')

xticks(fontsize=14.)
yticks(fontsize=14.)


gca().set_axisbelow(True)

#gca().set_xlim(xmin,xmax)
#gca().set_ylim(ymin,ymax)

x_major_formatter = FormatStrFormatter('%1.3f')
y_major_formatter = FormatStrFormatter('%1.3f')

gca().xaxis.set_major_formatter(x_major_formatter)
gca().xaxis.set_ticks_position('both')

gca().yaxis.set_major_formatter(y_major_formatter)

gca().grid(which='major', axis='x', linewidth=1., linestyle='-', color='0.6')
gca().grid(which='minor', axis='x', linewidth=0.5, linestyle='-', color='0.75')
gca().grid(which='major', axis='y', linewidth=1., linestyle='-', color='0.6')
gca().grid(which='minor', axis='y', linewidth=0.5, linestyle='-', color='0.75')

#gca().text((xmax-xmin)*0.54-(xmax+xmin)/2.,(ymax-ymin)*0.35+(ymax+ymin)/2., BoxText, size=16, bbox={'facecolor':'white', 'pad':20, 'lw':1})

tight_layout()
Filename=sys.argv[0]
Filename=modpath + r'Fig10.pdf'

savefig(Filename)





#Figure 20
fig20=plt.figure(20,figsize=(7,5), dpi=150, facecolor='white')
#plot(h_x,D_x_filtered_deriv_filtered,ls='-',c='black', zorder=1)
plot(h_x,D_x_filtered_deriv,ls='-',lw=2,c='black', zorder=2)
#plot(NA_2,deps_eq_design,ls='none',marker='^',mfc='white',mec='red',ms=10.,fillstyle='full', zorder=2, label=r"Design RO")

#legend(loc=2, fontsize=14.)

#BoxText=r"\textbf{Spitzenwerte: uA16dk82} \\ Lastabfall"

xmin=-0.5
xmax=18.

ymin=-2e-2
ymax=2e-2

xlabel(r"Specimen Height $\ h$ $[\text{mm}]$", fontsize=16)
ylabel(r"\begin{center} $\displaystyle \text{Specimen Diameter Derivative} \ \frac{\mathrm{d} d}{\mathrm{d} h} \ [\text{mm} \slash \text{mm}]$ \end{center}", fontsize=16, rotation=90)


yscale('linear')
xscale('linear')

xticks(fontsize=14.)
yticks(fontsize=14.)


gca().set_axisbelow(True)

gca().set_xlim(xmin,xmax)
#gca().set_ylim(ymin,ymax)

x_major_formatter = FormatStrFormatter('%1.3f')
y_major_formatter = FormatStrFormatter('%1.2e')

gca().xaxis.set_major_formatter(x_major_formatter)
gca().xaxis.set_ticks_position('both')

gca().yaxis.set_major_formatter(y_major_formatter)

#gca().grid(which='major', axis='x', linewidth=1., linestyle='-', color='0.0')
#gca().grid(which='minor', axis='x', linewidth=0.5, linestyle='-', color='0.0')
#gca().grid(which='major', axis='y', linewidth=1., linestyle='-', color='0.0')
#gca().grid(which='minor', axis='y', linewidth=0.5, linestyle='-', color='0.0')

#gca().text((xmax-xmin)*0.54-(xmax+xmin)/2.,(ymax-ymin)*0.35+(ymax+ymin)/2., BoxText, size=16, bbox={'facecolor':'white', 'pad':20, 'lw':1})

tight_layout()
Filename=sys.argv[0]
Filename=modpath + r'Fig20.pdf'
Filename=sys.argv[0]
Filename=modpath + r'Fig20.png'
savefig(Filename)






##
# Find "Robuste Stuetzstellen" l1 und l2, Option 1 ---> OLD

min_help=D_x_filtered_deriv_filtered[0]
h_pos_min=h_x[0]
h_pos_th=h_x[0]

find=False

for i in arange(1,size(D_x_filtered_deriv_filtered)):
    if D_x_filtered_deriv_filtered[i]<1e-3:
        break
    if D_x_filtered_deriv_filtered[i]<min_help:
        min_help=D_x_filtered_deriv_filtered[i]
        h_pos_min=h_x[i]
    if D_x_filtered_deriv_filtered[i]<l1_th_fix and find==False:
        find=True
        h_pos_th=h_x[i]
    if h_x[i]>h_max:
        break

l2_abs=h_pos_min
l1_abs=h_pos_th
D_x_filtered_deriv_filtered_min=min_help

print('+++')
print('Results using hard coded threshold for l1 and abs minimum for l2')
print('l1: ', l1_abs)
print('l2: ', l2_abs)


print('eps_ue_tangent: ', eps_ue(l1_abs,l2_abs,d0))
print('eps_ue_integral: ', eps_ue_integral(l1_abs,l2_abs,d0))

print('minimum Diameter Derivative', min_help)


##
# Find "Robuste Stuetzstellen" l1 und l2, Option 2  -> VARIANT 1 & 2

min_help=D_x_filtered_deriv_filtered[0]
h_pos_min=h_x[0]
h_pos_th=h_x[0]
l1_th=D_x_filtered_deriv_filtered_min*min_Factor
find=False

for i in arange(1,size(D_x_filtered_deriv_filtered)):
    if D_x_filtered_deriv_filtered[i]<1e-4:
        break
    if D_x_filtered_deriv_filtered[i]<min_help:
        min_help=D_x_filtered_deriv_filtered[i]
        h_pos_min=h_x[i]
    if D_x_filtered_deriv_filtered[i]<l1_th and find==False:
        find=True
        h_pos_th=h_x[i]
    if h_x[i]>h_max:
        break

l2_rel=h_pos_min
l1_rel=h_pos_th

print('+++')
print('Results using relative threshold factor 2 of minimum for l1 and abs minimum for l2')
print('l1: ', l1_rel)
print('l2: ', l2_rel)

#!! Variant 2 Results
print('eps_ue_tangent: ', eps_ue(l1_rel,l2_rel,d0))
print('eps_ue_integral: ', eps_ue_integral(l1_rel,l2_rel,d0))

d0_auto=D_x_interp(l2_rel)*d0_auto_Factor
print('d0_auto ', d0_auto)

#!! Variant 1 Results
print('+++')
print('Results using relative threshold factor 2 of minimum for l1 and abs minimum for l2 and d0 Auto')
print('eps_ue_tangent: ', eps_ue(l1_rel,l2_rel,d0_auto))
print('eps_ue_integral: ', eps_ue_integral(l1_rel,l2_rel,d0_auto))



##
# Find "Robuste Stuetzstellen" l1 und l2, Option 3 (l1 Auto) -> Variant 3




##
# find l2 via d0
l2_rel_d0=h_x[-1]

for i in arange(1,size(h_x)):
    if D_x_filtered[i]>=d0:
        break
l2_rel_d0=h_x[i]

##
# find l1 via derivative
min_help=D_x_filtered_deriv_filtered[0]
h_pos_min=h_x[0]
h_pos_th=h_x[0]
l1_th=D_x_filtered_deriv_filtered_min*min_Factor
find=False

for i in arange(1,size(D_x_filtered_deriv_filtered)):
    if D_x_filtered_deriv_filtered[i]<1e-4:
        break
    if D_x_filtered_deriv_filtered[i]<min_help:
        min_help=D_x_filtered_deriv_filtered[i]
        h_pos_min=h_x[i]
    if D_x_filtered_deriv_filtered[i]<l1_th and find==False:
        find=True
        h_pos_th=h_x[i]
    if h_x[i]>h_max:
        break
l1_rel_d0=h_pos_min
#l1_rel_d0=h_pos_th

print('+++')
print('Results using relative threshold and diameter d0 criterion')
print('l1: ', l1_rel_d0)
print('l2: ', l2_rel_d0)

print('eps_ue_tangent: ', eps_ue(l1_rel_d0,l2_rel_d0,d0))
print('eps_ue_integral: ', eps_ue_integral(l1_rel_d0,l2_rel_d0,d0))


###
# Export to File
f = open(modpath+SpecimenFolder+"_export"+exportFileSuffix+".csv", "w")
f.write(SpecimenID+";"+SpecimenFolder+";\n")
f.write('+++;;\n')
f.write("Settings:;;\n")
f.write('d0 = ;'+str(d0)+';mm\n')
f.write('l1 = ;'+str(l1)+';mm\n')
f.write('l2 = ;'+str(l2)+';mm\n')
f.write('h_max = ;'+str(h_max)+';mm\n')
f.write('l1_th_fix = ;'+str(l1_th_fix)+';-\n')
f.write('min_Factor = ;'+str(min_Factor)+';-\n')
f.write('+++;;\n')
f.write('Results using absolute user-defined values for l1 and l2;;\n')
f.write('l1_user_defined: ;'+str(l1)+';mm\n')
f.write('l2_user_defined: ;'+str(l2)+';mm\n')
f.write('Diameter at l1: ;'+str(D_x_interp(l1))+';mm\n')
f.write('Diameter at l2: ;'+str(D_x_interp(l2))+';mm\n')
f.write('eps_ue_tangent: ;'+str(eps_ue(l1,l2,d0))+';%\n')
f.write('eps_ue_integral: ;'+str(eps_ue_integral(l1,l2,d0))+';%\n')
f.write('+++;;\n')
f.write('Results using hard coded threshold for l1 and abs minimum for l2;;\n')
f.write('l1_abs: ;'+str(l1_abs)+';mm\n')
f.write('l2_abs: ;'+str(l2_abs)+';mm\n')
f.write('Minimum Diameter Derivative;'+str(D_x_filtered_deriv_filtered_min)+';-\n')
f.write('eps_ue_tangent_abs: ;'+str(eps_ue(l1_abs,l2_abs,d0))+';%\n')
f.write('eps_ue_integral_abs: ;'+str(eps_ue_integral(l1_abs,l2_abs,d0))+';%\n')
f.write('+++;;\n')
f.write('Results using relative threshold of minimum for l1 and abs minimum for l2;;\n')
f.write('l1_rel: ;'+str(l1_rel)+';mm\n')
f.write('l2_rel: ;'+str(l2_rel)+';mm\n')
f.write('eps_ue_tangent_rel: ;'+str(eps_ue(l1_rel,l2_rel,d0))+';%\n')
f.write('eps_ue_integral_rel: ;'+str(eps_ue_integral(l1_rel,l2_rel,d0))+';%\n')
f.write('D_x_filtered_deriv_filtered_min: ;'+str(D_x_filtered_deriv_filtered_min)+';-\n')
f.write('+++;;\n')
f.write('Results using relative threshold of minimum for l1 and abs minimum for l2 with auto d0;;\n')
f.write('d0_auto_Factor = ;'+str(d0_auto_Factor)+';-\n')
f.write('Auto_d0_Value: ;'+str(d0_auto)+';mm\n')
f.write('eps_ue_tangent_rel_d0Auto: ;'+str(eps_ue(l1_rel,l2_rel,d0_auto))+';%\n')
f.write('eps_ue_integral_re_d0Auto: ;'+str(eps_ue_integral(l1_rel,l2_rel,d0_auto))+';%\n')
f.write('+++;;\n')
f.write('eps_ue_tangent_rel_d0Crit: ;'+str(eps_ue(l1_rel_d0,l2_rel_d0,d0))+';%\n')
f.write('eps_ue_integral_rel_d0Crit: ;'+str(eps_ue_integral(l1_rel_d0,l2_rel_d0,d0))+';%\n')
f.write("height;Diameter;\n")
f.write("mm;mm;\n")
for i in arange(0,size(h_x)):
  f.write(str(h_x[i])+";"+str(D_x[i])+";\n")
f.close()







#Figure 30
fig30=plt.figure(30,figsize=(7,5), dpi=150, facecolor='white')
plot(h_x,D_x_filtered_deriv_filtered,ls='-',c='black', zorder=1)
plot([l1_abs,l1_abs],[ymin,ymax],ls=':',lw=1,c='black', zorder=1)
plot([l2_abs,l2_abs],[ymin,ymax],ls=':',lw=1,c='black', zorder=1)
#plot(h_x,D_x_filtered_deriv,ls='--',c='red', zorder=2)
#plot(NA_2,deps_eq_design,ls='none',marker='^',mfc='white',mec='red',ms=10.,fillstyle='full', zorder=2, label=r"Design RO")

#legend(loc=2, fontsize=14.)

#BoxText=r"\textbf{Spitzenwerte: uA16dk82} \\ Lastabfall"

xmin=0.
xmax=h_max

ymin=2e-5
ymax=2e-2

xlabel(r"Specimen Height $\ h$ $[\text{mm}]$", fontsize=16)
ylabel(r"\begin{center} $\displaystyle \text{Derivative of Specimen Diamater} \ \frac{\mathrm{d} d}{\mathrm{d} h} \ [\text{mm} \slash \text{mm}]$ \end{center}", fontsize=16, rotation=90)


yscale('log')
xscale('linear')

xticks(fontsize=14.)
yticks(fontsize=14.)


gca().set_axisbelow(True)

gca().set_xlim(xmin,xmax)
gca().set_ylim(ymin,ymax)

x_major_formatter = FormatStrFormatter('%1.3f')
y_major_formatter = FormatStrFormatter('%1.2e')

gca().xaxis.set_major_formatter(x_major_formatter)
gca().xaxis.set_ticks_position('both')

gca().yaxis.set_major_formatter(y_major_formatter)

gca().grid(which='major', axis='x', linewidth=1., linestyle='-', color='0.6')
gca().grid(which='minor', axis='x', linewidth=0.5, linestyle='-', color='0.75')
gca().grid(which='major', axis='y', linewidth=1., linestyle='-', color='0.6')
gca().grid(which='minor', axis='y', linewidth=0.5, linestyle='-', color='0.75')

#gca().text((xmax-xmin)*0.54-(xmax+xmin)/2.,(ymax-ymin)*0.35+(ymax+ymin)/2., BoxText, size=16, bbox={'facecolor':'white', 'pad':20, 'lw':1})

tight_layout()
Filename=sys.argv[0]
Filename=modpath + r'Fig30.pdf'

savefig(Filename)





#Figure 40
fig40=plt.figure(40,figsize=(7,5), dpi=150, facecolor='white')
plot(h_x,D_x_filtered_deriv_filtered,ls='-',c='black', zorder=1)
plot([l1_rel,l1_rel],[ymin,ymax],ls=':',lw=1,c='black', zorder=1)
plot([l2_rel,l2_rel],[ymin,ymax],ls=':',lw=1,c='black', zorder=1)
#plot(h_x,D_x_filtered_deriv,ls='--',c='red', zorder=2)
#plot(NA_2,deps_eq_design,ls='none',marker='^',mfc='white',mec='red',ms=10.,fillstyle='full', zorder=2, label=r"Design RO")

#legend(loc=2, fontsize=14.)

#BoxText=r"\textbf{Spitzenwerte: uA16dk82} \\ Lastabfall"

xmin=0.
xmax=h_max

ymin=2e-5
ymax=2e-2

xlabel(r"Specimen Height $\ h$ $[\text{mm}]$", fontsize=16)
ylabel(r"\begin{center} $\displaystyle \text{Derivative of Specimen Diamater} \ \frac{\mathrm{d} d}{\mathrm{d} h} \ [\text{mm} \slash \text{mm}]$ \end{center}", fontsize=16, rotation=90)


yscale('log')
xscale('linear')

xticks(fontsize=14.)
yticks(fontsize=14.)


gca().set_axisbelow(True)

gca().set_xlim(xmin,xmax)
gca().set_ylim(ymin,ymax)

x_major_formatter = FormatStrFormatter('%1.3f')
y_major_formatter = FormatStrFormatter('%1.2e')

gca().xaxis.set_major_formatter(x_major_formatter)
gca().xaxis.set_ticks_position('both')

gca().yaxis.set_major_formatter(y_major_formatter)

gca().grid(which='major', axis='x', linewidth=1., linestyle='-', color='0.6')
gca().grid(which='minor', axis='x', linewidth=0.5, linestyle='-', color='0.75')
gca().grid(which='major', axis='y', linewidth=1., linestyle='-', color='0.6')
gca().grid(which='minor', axis='y', linewidth=0.5, linestyle='-', color='0.75')

#gca().text((xmax-xmin)*0.54-(xmax+xmin)/2.,(ymax-ymin)*0.35+(ymax+ymin)/2., BoxText, size=16, bbox={'facecolor':'white', 'pad':20, 'lw':1})

tight_layout()
Filename=sys.argv[0]
Filename=modpath + r'Fig40.pdf'

savefig(Filename)





#Figure 50
fig50=plt.figure(50,figsize=(7,5), dpi=150, facecolor='white')
plot(h_x,D_x_filtered_deriv_filtered,ls='-',c='black', zorder=1)
plot([l1_rel_d0,l1_rel_d0],[ymin,ymax],ls=':',lw=1,c='black', zorder=1)
plot([l2_rel_d0,l2_rel_d0],[ymin,ymax],ls=':',lw=1,c='black', zorder=1)
#plot(h_x,D_x_filtered_deriv,ls='--',c='red', zorder=2)
#plot(NA_2,deps_eq_design,ls='none',marker='^',mfc='white',mec='red',ms=10.,fillstyle='full', zorder=2, label=r"Design RO")

#legend(loc=2, fontsize=14.)

#BoxText=r"\textbf{Spitzenwerte: uA16dk82} \\ Lastabfall"

xmin=0.
xmax=h_max

ymin=2e-5
ymax=2e-2

xlabel(r"Specimen Height $\ h$ $[\text{mm}]$", fontsize=16)
ylabel(r"\begin{center} $\displaystyle \text{Derivative of Specimen Diamater} \ \frac{\mathrm{d} d}{\mathrm{d} h} \ [\text{mm} \slash \text{mm}]$ \end{center}", fontsize=16, rotation=90)


yscale('log')
xscale('linear')

xticks(fontsize=14.)
yticks(fontsize=14.)


gca().set_axisbelow(True)

gca().set_xlim(xmin,xmax)
gca().set_ylim(ymin,ymax)

x_major_formatter = FormatStrFormatter('%1.3f')
y_major_formatter = FormatStrFormatter('%1.2e')

gca().xaxis.set_major_formatter(x_major_formatter)
gca().xaxis.set_ticks_position('both')

gca().yaxis.set_major_formatter(y_major_formatter)

gca().grid(which='major', axis='x', linewidth=1., linestyle='-', color='0.6')
gca().grid(which='minor', axis='x', linewidth=0.5, linestyle='-', color='0.75')
gca().grid(which='major', axis='y', linewidth=1., linestyle='-', color='0.6')
gca().grid(which='minor', axis='y', linewidth=0.5, linestyle='-', color='0.75')

#gca().text((xmax-xmin)*0.54-(xmax+xmin)/2.,(ymax-ymin)*0.35+(ymax+ymin)/2., BoxText, size=16, bbox={'facecolor':'white', 'pad':20, 'lw':1})

tight_layout()
Filename=sys.argv[0]
Filename=modpath + r'Fig50.pdf'

savefig(Filename)


#Figure 60
fig60=plt.figure(60,figsize=(7,5), dpi=150, facecolor='white')
plot(h_x,D_x_filtered_deriv_filtered,ls='-',c='black', zorder=1)
#plot([l1_rel_d0,l1_rel_d0],[ymin,ymax],ls=':',lw=1,c='black', zorder=1)
#plot([l2_rel_d0,l2_rel_d0],[ymin,ymax],ls=':',lw=1,c='black', zorder=1)
#plot(h_x,D_x_filtered_deriv,ls='--',c='red', zorder=2)
#plot(NA_2,deps_eq_design,ls='none',marker='^',mfc='white',mec='red',ms=10.,fillstyle='full', zorder=2, label=r"Design RO")

#legend(loc=2, fontsize=14.)

#BoxText=r"\textbf{Spitzenwerte: uA16dk82} \\ Lastabfall"

xmin=0.
xmax=h_max

ymin=2e-5
ymax=2e-2

xlabel(r"Specimen Height $\ h$ $[\text{mm}]$", fontsize=16)
ylabel(r"\begin{center} $\displaystyle \text{Derivative of Specimen Diamater} \ \frac{\mathrm{d} d}{\mathrm{d} h} \ [\text{mm} \slash \text{mm}]$ \end{center}", fontsize=16, rotation=90)


yscale('log')
xscale('linear')

xticks(fontsize=14.)
yticks(fontsize=14.)


gca().set_axisbelow(True)

gca().set_xlim(xmin,xmax)
gca().set_ylim(ymin,ymax)

x_major_formatter = FormatStrFormatter('%1.3f')
y_major_formatter = FormatStrFormatter('%1.2e')

gca().xaxis.set_major_formatter(x_major_formatter)
gca().xaxis.set_ticks_position('both')

gca().yaxis.set_major_formatter(y_major_formatter)

gca().grid(which='major', axis='x', linewidth=1., linestyle='-', color='0.6')
gca().grid(which='minor', axis='x', linewidth=0.5, linestyle='-', color='0.75')
gca().grid(which='major', axis='y', linewidth=1., linestyle='-', color='0.6')
gca().grid(which='minor', axis='y', linewidth=0.5, linestyle='-', color='0.75')

#gca().text((xmax-xmin)*0.54-(xmax+xmin)/2.,(ymax-ymin)*0.35+(ymax+ymin)/2., BoxText, size=16, bbox={'facecolor':'white', 'pad':20, 'lw':1})

tight_layout()
Filename=sys.argv[0]
Filename=modpath + r'Fig60.pdf'
Filename=modpath + r'Fig60.png'

savefig(Filename)
#show()
