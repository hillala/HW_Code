#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 15:20:02 2017

@author: hilla
"""
# nececerry mudols:
from __future__ import division, print_function
import numpy as np 
from sklearn import mixture
import sys
#sys.path.append("/home/hilla/Gabriel/ncRNA/HW_Code")
sys.path.append("/groups/steen/hilla/Amanda_SMA/HW")
from detect_peaks import detect_peaks
import hw_algorithm as hw

#parameters:
#Inputfile='/home/hilla/Gabriel/ncRNA/Amanda_SMA/data/Het_P10/chr10_dir_b.txt'
#Outputfile='/home/hilla/Gabriel/ncRNA/Amanda_SMA/Output/Het_P10/chr10_dir_bin_20.txt'
#Lengthfile='/home/hilla/Gabriel/ncRNA/martins/chrLength.txt'   
#Namefile='/home/hilla/Gabriel/ncRNA/HW_Code/ChrName.txt'
Inputfile=sys.argv[1]
Outputfile=sys.argv[2]
Lengthfile=sys.argv[3]
Namefile=sys.argv[4]
L_bin = sys.argv[5]
L_min = sys.argv[6]
L_max=sys.argv[7]
FDR_Point=sys.argv[8]
FDR_Region=sys.argv[9]
Gene_Num=sys.argv[10]

#L_bin=20
#L_min=2
#L_max=16
#FDR_Point=0.05
#FDR_Region=0.05
#Gene_Num=1

chr_length= [int(l.rstrip('\n')) for l in open(Lengthfile)] 
chr_name= [name.rstrip('\n') for name in open(Namefile)]
Length=chr_length[Gene_Num-1]
# insides:
BinNum=int(np.floor(Length/L_bin))
MinDistance=1 #np.power(2, L_min)
MinLength=1 #np.power(2, L_min)
BreakPoints=[]
RBreakPoints=[]
VInitial_Locations=[]
VFinal_Locations=[]
Areas=[]
RAreas=[]

flag=1

BreakPoints=[]
flag=1
count=0

Rj=hw.ReadFile2StreandPairedEnd(Inputfile,Length,L_bin,BinNum)

while flag==1:
    for L in xrange(L_min,L_max):
        Range=np.power(2, L)
        Hij=hw.HaarWavelet(L,Range,Rj,BinNum)
        Loc=np.sort(hb.ExtremaLocation(Hij,FDR_Point))+Range
        #Loc=np.sort(hw.ExtremaLocation(Hij,FDR_Point))+Range
        BreakPoints=hw.CombinedPoints(BreakPoints,Loc,Range)
        BreakPoints.astype(int)
        print(L)
    Dens=hw.AverageLogDensity(Rj,BreakPoints)
    Dens,Initial_Locations,Final_Locations=hw.RemoveZeroReads(Dens,BreakPoints)
    samples=Dens.reshape(len(Dens), 1)
    Prob=hw.PredictRegion_GMM_3(Dens.reshape(len(Dens), 1))
    #Prob=hw.PredictRegion_GMM_2(Dens.reshape(len(Dens), 1))

    if np.sum(Prob<FDR_Region)<1:
        break
    else:

        Dens,Initial_Locations,Final_Locations=hw.TranslationProbability(Prob,FDR_Region,Dens,Initial_Locations,Final_Locations)
        VInitial_Locations,VFinal_Locations=hw.CombinedIterations(VInitial_Locations,VFinal_Locations,Initial_Locations,Final_Locations)
        VInitial_Locations,VFinal_Locations=hw.CombinedAreas(VInitial_Locations,VFinal_Locations,MinDistance)

        Rj=hw.RemovedUsedReads(Initial_Locations,Final_Locations,Rj)

    count=count+1
    print(count)
    print(len(Dens))

Rj=hw.ReadFile2StreandPairedEnd(Inputfile,Length,L_bin,BinNum)
    
Out=np.vstack((VInitial_Locations, VFinal_Locations)).T 
np.concatenate((VInitial_Locations, VFinal_Locations), axis=0)
np.savetxt(Outputfile, Out, fmt='%i,')
np.mean() for i in range(len(VInitial_Locations))
np.mean(Initial_Locations[1:]-Final_Locations[:-1])

def ExtremaLocation (Serias,FDR_Point):
    ind1 = detect_peaks(Serias,mpd=150,threshold=0.0001)
    ind2 = detect_peaks(Serias,mpd=100,threshold=0.0001,valley=True)
#    ind1 = detect_peaks(Serias,mpd=1,threshold=0.0001)
#    ind2 = detect_peaks(Serias,mpd=1,threshold=0.0001,valley=True)
    ind=np.sort(np.append(ind1,ind2)) 
 #   ExtLoc=np.append(argrelextrema(Serias, np.greater), argrelextrema(Serias, np.less))
 #   a=np.stack((np.abs(Serias[ind]-Serias[ind+1]),np.abs(Serias[ind]-Serias[ind-1])))
 #   a=np.abs(a)
 #   MaxChange=a.max(0)
    ExtVal=np.abs(Serias[ind])
    #IndExt=np.argsort(ExtVal)
##    Vec=np.argsort(np.abs(Serias[ExtLoc]))
#    Loc=ExtLoc[IndExt[-int(np.floor(FDR_Point*len(IndExt))):]]
#    Serias=Hij
#    MaxChange=np.abs(Serias[ind])
    IndExt=np.argsort(ExtVal)
    Loc=ind[IndExt[-int(np.floor(FDR_Point*len(ind))):]]
    return Loc 

def ExtremaLocation (Serias,FDR_Point):
    ind1 = detect_peaks(Serias,mpd=150,threshold=0.0001)
    ind2 = detect_peaks(Serias,mpd=150,threshold=0.0001,valley=True)
    ind=np.sort(np.append(ind1,ind2))   
    a=np.stack((np.abs(Serias[ind]-Serias[ind+1]),np.abs(Serias[ind]-Serias[ind-1])))
    a=np.abs(a)
    MaxChange=a.max(0)
    IndExt=np.argsort(MaxChange)
    Loc=ind[IndExt[-int(np.floor(FDR_Point*len(ind))):]]
    return Loc