#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 16:27:44 2017

@author: hilla
"""

from __future__ import division, print_function
import numpy as np 
#from scipy.signal import argrelextrema
from sklearn import mixture
from detect_peaks import detect_peaks
#from detect_peaks import _plotb 


def HaarWavelet(L,Range,Signal,BinNum):
   """ calculate Haar wavelet coefficents
   calculate the  Harr wavelete coefficent for a given window size
   Parameters
       L: window size parameter (size of 2**L)
       Range: 2**L
       Signal: the log density for every bin
       BinNum: number of total bin in the signal
   Return
       1-D array in the size of the number of windows exist in the signal
       number of bins in the signal / window size
   """
   S1=np.sum(Signal[0:Range])
   S2=np.sum(Signal[Range:2*Range])
   Coeff=1/(np.sqrt(np.power(2, L+1)))
   Hij=np.zeros(BinNum-2*Range)
   Hij[0]=Coeff*(S2-S1)
   for loc in xrange(Range,BinNum-Range-1):
       S1=S1-Signal[loc-Range]+Signal[loc]
       S2=S2-Signal[loc]+Signal[loc+Range]
       Hij[loc-Range+1]=Coeff*(S2-S1)
   return Hij

def ExtremLocation_Val (Serias,FDR_Point,Min_Dist,Min_Val):
    """
    finds extermum points from the Haar wavlet coefficents, based on the
    absulot value of the points
    Parameters
        Serias: HW coefficents
        FDR_Points: the fraction of the extermum points selected
        Min_Dist: the minimum distance in bins between extermum points
        Min_Val: minimum value for extermum point
    Return
        1D array of extermom points with the size of fraction of FDR_Point
        from the total selected points
    """
    ind1 = detect_peaks(Serias,mpd=Min_Dist,threshold=Min_Val)
    ind2 = detect_peaks(Serias,mpd=Min_Dist,threshold=Min_Val,valley=True)
    ind=np.sort(np.append(ind1,ind2))   
    MaxChange=np.abs(Serias[ind])
    IndExt=np.argsort(MaxChange)
    Loc=ind[IndExt[-int(np.floor(FDR_Point*len(ind))):]]
    return Loc 


def ExtremLocation_ValChange (Serias,FDR_Point,Min_Dist,Min_Val):
    """
    finds extermum points from the Haar wavlet coefficents, based on the
    difference between the points value and the value of it nearest neighbors 
    Parameters
        Serias: HW coefficents
        FDR_Points: the fraction of the extermum points selected
        Min_Dist: the minimum distance in bins between extermum points
        Min_Val: minimum value for extermum point
    Return
        1D array of extermom points with the size of fraction of FDR_Point
        from the total selected points
    """
    ind1 = detect_peaks(Serias,mpd=Min_Dist,threshold=Min_Val)
    ind2 = detect_peaks(Serias,mpd=Min_Dist,threshold=Min_Val,valley=True)
    ind=np.sort(np.append(ind1,ind2))   
    a=np.stack((np.abs(Serias[ind]-Serias[ind+1]),np.abs(Serias[ind]-Serias[ind-1])))
    a=np.abs(a)
    MaxChange=a.max(0)
    IndExt=np.argsort(MaxChange)
    Loc=ind[IndExt[-int(np.floor(FDR_Point*len(ind))):]]
    return Loc 

def ExtremLocation_Random (Serias,FDR_Point,Min_Dist,Min_Val):
    """
    finds extermum points from the Haar wavlet coefficents, select points randomly
        Serias: HW coefficents
        FDR_Points: the fraction of the extermum points selected
        Min_Dist: the minimum distance in bins between extermum points
        Min_Val: minimum value for extermum point
    Return
        1D array of extermom points with the size of fraction of FDR_Point
        from the total selected points
    """
    ind1 = detect_peaks(Serias,mpd=Min_Dist,threshold=Min_Val)
    ind2 = detect_peaks(Serias,mpd=Min_Dist,threshold=Min_Val,valley=True)
    ind=np.sort(np.append(ind1,ind2))   
    Loc=ind[IndExt[-int(np.floor(FDR_Point*len(ind))):]]
    return Loc 

def CombinedPoints (Points,Locations,Range):
    """
    combineds extermum points selected in specific window with points that 
    selected earlier. If new selected points are too close to points from earlier
    iterations those points are dismissed.
    Parameters
        Points: extermum points selected in previouse iterations (1D array)
        Locations: extermum points selected in corrent iterations (1D array)
        Range: minimum allow distance between corrent and previouse points 
    Return
        1D array of the combination points 
    """
    Locations=np.sort(Locations)
    ind=np.diff(Locations)>Range
    Locations=Locations[np.append(0<1,ind)]
    Val=np.append(Points, Locations)
    Vflag=np.append(np.ones((len(Points),), dtype=bool),np.zeros((len(Locations),), dtype=bool))
    Vec=np.argsort(Val)
    Val=Val[Vec]
    Vflag=Vflag[Vec]
    ind=np.diff(Val)>Range
    Points=np.unique(np.append(Val[np.append(ind,0<1)|Vflag],Val[np.append(0<1,ind)|Vflag]))
    return Points

def RemoveZeroReads (Dens,Points):
    """
    Removed selected regions with zero log density
    Parameters
        Dens: avearge log density of each area (1D array) 
        Points: extermum points selected in corrent iterations (1D array)
    Return
        Dens: Areas log density (1D array)
        Initial_Locations: the location of the start bin of each area (1D array)
        Final_Locations: the location of the end bin of each area (1D array)
    """
    Final_Locations=Points[1:]
    Initial_Locations=Points[:-1]
    ind=Dens>0
    Final_Locations=Final_Locations[ind]
    Initial_Locations=Initial_Locations[ind]
    Dens=Dens[ind]
    return [Dens,Initial_Locations,Final_Locations]


def RemoveLowReads(Dens,Points,Min_Val_Remove):
    """
    Removed selected regions with low log density
    Parameters
        Dens: avearge log density of each area (1D array)  
        Points: extermum points selected in corrent iterations (1D array)
        Min_Val_Remove: minimum log density
    Return
        Dens: Areas log density (1D array)
        Initial_Locations: the location of the start bin of each area (1D array)
        Final_Locations: the location of the end bin of each area (1D array)
    """
    Final_Locations=Points[1:]
    Initial_Locations=Points[:-1]
    ind=Dens>Min_Val_Remove
    Final_Locations=Final_Locations[ind]
    Initial_Locations=Initial_Locations[ind]
    Dens=Dens[ind]
    return [Dens,Initial_Locations,Final_Locations]


def AverageLogDensity (LogDens,Points):
    """
    Calculte the average density for each selected area
    Parameters
        LogDens: avearge log density of each bin (1D array)  
        Points: selected extermum points (1D array)
    Return
        AveLogDens: average log density for each selected regions, 
        over all the bins in the regions(1D array)
    """
    Points=np.sort(np.array(Points,dtype=int))
    AveLogDens=np.array([np.mean(LogDens[Points[ind]:Points[ind+1]]) for ind in xrange(len(Points)-1)])
    return AveLogDens

def PredictRegion_GMM_3(samples):
    """
    create a 3 Gaussian model fit for the average density of the regions
    Parameters
        samples: avearge log density of each region (1D array)  
    Return
        Prob: the probabilty of region to belong to the Gaussian
        of the lowest average density (1D array)
    """
    gmix = mixture.GMM(n_components=3, covariance_type='full')
    gmix.fit(samples)
    LowestCompunent=np.argsort(np.squeeze(gmix.means_))
    Prob=gmix.predict_proba(samples)
    return Prob[:,LowestCompunent[0]]

def PredictRegion_GMM_3_highest(samples):
    """
    create a 3 Gaussian model fit for the average density of the regions
    Parameters
        samples: avearge log density of each region (1D array)  
    Return
        Prob: the probabilty of region to belong to the Gaussian
        of the highest average density (1D array)
    """
    gmix = mixture.GMM(n_components=3, covariance_type='full')
    gmix.fit(samples)
    HighestCompunent=np.argsort(np.squeeze(gmix.means_))
    Prob=gmix.predict_proba(samples)
    return Prob[:,HighestCompunent[2]]

def PredictRegion_GMM_2(samples):
    """
    create a 2 Gaussian model fit for the average density of the regions
    Parameters
        samples: avearge log density of each region (1D array)  
    Return
        Prob: the probabilty of region to belong to the Gaussian
        of the lowest average density (1D array)
    """
    gmix = mixture.GMM(n_components=2, covariance_type='full')
    gmix.fit(samples)
    LowestCompunent=np.argsort(np.squeeze(gmix.means_))
    Prob=gmix.predict_proba(samples)
    return Prob[:,LowestCompunent[0]]

def PredictRegion_GMM_2_highest(samples):
    """
    create a 2 Gaussian model fit for the average density of the regions
    Parameters
        samples: avearge log density of each region (1D array)  
    Return
        Prob: the probabilty of region to belong to the Gaussian
        of the highest average density (1D array)
    """
    gmix = mixture.GMM(n_components=2, covariance_type='full')
    gmix.fit(samples)
    HighestCompunent=np.argsort(np.squeeze(gmix.means_))
    Prob=gmix.predict_proba(samples)
    return Prob[:,HighestCompunent[1]]

def TranslationProbability_highest(Prob,FDR,Dens,Initial_Locations,Final_Locations):
    """
   returns location and density of areas that belongs to the highest Gaussian
   Parameters
        Prob: he probabilty of each region to belongs to the lowest Gaussian (1D array)
        FDR: The threshold value of belonging to the highest Gaussian
        Dens: Density of areas (1D array)
        Initial_Locations: start bin of each area (1D array)
        Final_Locations: end bin of each area (1D array)
    Return
        Dens: density of selected regions
        Initial_Locations: start bin of selected regions (1D array)
        Final_Locations: end bin of selected regions (1D array)  
    """
    ind=Prob>FDR
    Initial_Locations=Initial_Locations[ind]
    Final_Locations=Final_Locations[ind]
    Dens=Dens[ind]
    return [Dens,Initial_Locations,Final_Locations]


def TranslationProbability(Prob,FDR,Dens,Initial_Locations,Final_Locations):
    """
   returns location and density of areas that belongs to the lowest Gaussian
   Parameters
        Prob: The probabilty of each region to belongs to the lowest Gaussian (1D array)
        FDR: The threshold value of belonging to the highest Gaussian
        Dens: Density of areas (1D array)
        Initial_Locations: start bin of each area (1D array)
        Final_Locations: end bin of each area (1D array)
    Return
        Dens: density of selected regions
        Initial_Locations: start bin of selected regions (1D array)
        Final_Locations: end bin of selected regions (1D array)  
    """
    ind=Prob<FDR
    Initial_Locations=Initial_Locations[ind]
    Final_Locations=Final_Locations[ind]
    Dens=Dens[ind]
    return [Dens,Initial_Locations,Final_Locations]

def CombinedIterations(VInitial_Points,VFinal_Points,Initial_Locations,Final_Locations):
    """
   combined regions that selectecd in this iteration with regions that selected
   in previouse iterations 
   Parameters
        VInitial_Points: start points of areas from previouse iterations (1D array)
        VFinal_Points: end points of areas from previouse iterations (1D array)
        Initial_Locations: start points of areas from corrent iteration (1D array)
        Final_Locations: end points of areas from corrent iteration (1D array)
    Return
        Dens: density of selected regions
        Initial_Locations: start points of areas from every iteration (1D array)
        Final_Locations: end points of areas from every iteration (1D array)  
    """
    VInitial_Points=np.append(VInitial_Points,Initial_Locations)
    VFinal_Points=np.append(VFinal_Points,Final_Locations)
    ind=np.argsort(VInitial_Points)
    VInitial_Points=VInitial_Points[ind]
    VFinal_Points=VFinal_Points[ind]
    return [VInitial_Points,VFinal_Points]
    

def CombinedAreas(Initial_Locations,Final_Locations,MinDistance):
    """
   Unite areas that are close to each other
   Parameters
        Initial_Locations: areas start points (1D array)
        Final_Locations: areas end points (1D array)
        MinDistance: minimum allow distance (in bins) between regions
    Return
        Initial_Locations: combined areas start point (1D array)
        Final_Locations: combined areas end points (1D array)  
    """
    ind=(Initial_Locations[1:]-Final_Locations[:-1])>MinDistance
    while np.sum(ind)<len(ind):
        Initial_Locations=Initial_Locations[np.append(0<1,ind)]
        Final_Locations=Final_Locations[np.append(ind,0<1)]
        ind=(Initial_Locations[1:]-Final_Locations[:-1])>MinDistance
    return (Initial_Locations,Final_Locations)

def RemoveShourtAreas(VInitial_Locations,VFinal_Locations,MinLength):
    """
   delete areas that are too shorth
   Parameters
        VInitial_Points: start points of areas (1D array)
        VFinal_Points: end points of areas  (1D array)
        MinLength: minimun length of an area in number of bins
    Return
        VInitial_Locations: start points of areas  (1D array)
        VFinal_Locations: end points of areas (1D array)  
    """
    ind=(VFinal_Locations-VInitial_Locations)>MinLength
    return VInitial_Locations[ind],VFinal_Locations[ind]

def RemovedUsedReads(Initial_Locations,Final_Locations,R):
   """
   remove the reads from areas that were selected as transcribed region in the
   corrent iteration
   Parameters
        Initial_Points: start points of selected regions (1D array)
        Final_Points: end points of selected regipons (1D array)
        R: density of each bin (1D array)
   Return
      R: revised density of each bin (1D array)
   """
   Initial_Locations=np.array(Initial_Locations,dtype=int)
   Final_Locations=np.array(Final_Locations,dtype=int)
   for j in xrange(len(Initial_Locations)):
       R[Initial_Locations[j]:Final_Locations[j]]=0 
   return R

def ReadFile2Streand(fileStreand,Length,L_bin,BinNum):
    """
   Read list of reads start and end points, for the apropriate 
   base pair location on the streand vector
   Parameters
        fileStreand: the file name from which to upload the reads locations
        Length: chromosome length
        BinNum: number of bins for the streand
    Return
       array with the average log density of each bin in the streand (1D array)
    """
    Streand=np.zeros((Length,), dtype=int)   
    G = [[int(item) for item in line.split()] for line in open(fileStreand)]
    G1,G2=np.array(zip(*G),dtype=np.int)    
    for i in xrange(0,len(G1)):
        Streand[G1[i]-1:G1[i]-1+G2[i]]+=1    
    Split=np.array_split(Streand[0:BinNum*L_bin],BinNum)
    Read_Denst =np.array(map(np.mean, Split))
    return np.log(1+Read_Denst)  

def ReadFile2StreandPairedEnd(fileStreand,Length,L_bin,BinNum):
    """
   Read list of reads start and end points, for the apropriate 
   base pair location on the streand vector
   Parameters
        fileStreand: the file name from which to upload the reads locations
        Length: chromosome length
        BinNum: number of bins for the streand
    Return
       array with the average log density of each bin in the streand (1D array)
    """
    Streand=np.zeros((Length,), dtype=int)   
    G = [[int(item) for item in line.split()] for line in open(fileStreand)]
    G1,G2=np.array(zip(*G),dtype=np.int)    
    for i in xrange(0,len(G1)):
        Streand[G1[i]-1:G2[i]-1]+=1    
    Split=np.array_split(Streand[0:BinNum*L_bin],BinNum)
    Read_Denst =np.array(map(np.mean, Split))
    return np.log(1+Read_Denst) 

def WriteOutputFile(fileID,initial,final,chromosom,direction):
    if direction=='dir':
        d='+'
    else:
        d='-'
    fileID.write('>%s/t%s/t%s/t%s\n' % \
                                  (chromosom,initial,final,d))
                          
                               
def FineTunning(Signal1,Initial_Locations,Final_Locations,Range1,L_bin):
    """
   select the start and end point of transcribed regions 
   Parameters
        Signal1: reads density of every bp in the region (1D array)
        Initial_Locations: initial start point of the region
        Final_Locations: end point of the region
        Range1: the number of bp from which to select a potential new start/end point
        L_bin: bin size
    Return
        Initial_Locations: the new regions start point
        Final_Locations: the new region end point
    """
    Initial_Locations1=np.zeros(len(Initial_Locations))
    Final_Locations1=np.zeros(len(Final_Locations))
    for k in range(len(Initial_Locations)):
        #R=np.floor(0.1*(Final_Locations[k]-Initial_Locations[k]))
        Signal=Signal1[Initial_Locations[k]-Range1/2:Initial_Locations[k]+Range1/2]
        val=HaarWavelet(np.sqrt(L_bin),L_bin,Signal,Range1)
        ind1 = detect_peaks(val,mpd=1,threshold=0)
        Initial_Locations1[k]=ind1[0]+Initial_Locations[k]-Range1/2
        Signal=Signal1[Final_Locations[k]-Range1/2:Final_Locations[k]+Range1/2]
        ind1=detect_peaks(val,mpd=1,threshold=0,valley=True)
        Final_Locations1[k]=ind1[0]+Final_Locations[k]-Range1/2
    return Initial_Locations,Final_Locations
    
def ReadLocationFile(file_name):
    """
   read file that contain regions location on chromosom , and sort regions by location
        file_name: the file name
    Return
        IntAnn: regions start locations (1D array) 
        FinalAnn: regions end locations (1D array) 
    """
    G = [[int(item) for item in line.split()] for line in open(file_name)]  
    if len(G):
        IntAnn,FinalAnn=np.array(zip(*G),dtype=np.int)
        ind=np.argsort(IntAnn)
        FinalAnn=FinalAnn[ind]
        IntAnn=IntAnn[ind]
    else:
        FinalAnn=[]
        IntAnn=[]
    return [IntAnn,FinalAnn]

def ReadAnnotation(Annfile):
    """
   read file that contain regions location on chromosom , and sort regions by location
   Parameters:
        file_name: the file name
    Return
        IntAnn: regions start locations (1D array) 
        FinalAnn: regions end locations (1D array) 
    """
    G = [[int(item.replace(",", "")) for item in line.split()] for line in open(Annfile)]
    if len(G)>0:
        IntAnn,FinalAnn=np.array(zip(*G),dtype=np.int)
        ind=np.argsort(IntAnn)
        FinalAnn=FinalAnn[ind]
        IntAnn=IntAnn[ind]
    else:
        FinalAnn=[]
        IntAnn=[]
    return [IntAnn,FinalAnn]

def CalcOverlapGeneTranscript(IntAnn,FinalAnn,TranscripEnd,TranscripInt,transInd):
    """
   calcolate the overlap between transcribed regions to annotated regions. returns the
   index of overlap annotated region and overlap rate (% of common bp). Return a 
   list with all the overlaps in the annotation and overlap coantity even if more 1.
   Parameters:
        IntAnn: start location of annotation  (1D array) 
        FinalAnn: end location of annotation (1D array) 
        TranscripEnd: end location of detected transcribed region  
        TranscripInt: start location of detected transcribed region 
        transInd : index of the transcribed regions
    Return:
        GeneIndex: index of annotated regions that are overlap with the area 
        overlap_gene_trans : the % of bp overlap in the annotation region
        overlap_trans_gene: the % of bp overlap in the transcribed region
        TransIndex index of the input transcribed region
    """
    ind=np.logical_not(np.logical_or(TranscripEnd<=IntAnn,TranscripInt>=FinalAnn))
    Vec=np.arange(len(IntAnn))
    GeneIndex=Vec[ind]
    GeneInt=IntAnn[ind]
    GeneEnd=FinalAnn[ind]
    GeneNum=len(GeneInt)
    overlap_gene_trans=np.zeros(GeneNum)
    overlap_trans_gene=np.zeros(GeneNum)
    TransIndex=np.ones(GeneNum)*transInd
    II=TranscripInt  
    FF=TranscripEnd
    for ind in range(GeneNum):   
        I=GeneInt[ind]
        F=GeneEnd[ind]
        GeneLength=GeneEnd[ind]- GeneInt[ind]
        TranseLength=TranscripEnd-TranscripInt 
        if II-I<0:
            if FF-F>0:
                overlap_gene_trans[ind]=1
                overlap_trans_gene[ind]=GeneLength/TranseLength
            else:
                overlap_gene_trans[ind]=(FF-I)/GeneLength
                overlap_trans_gene[ind]=(FF-I)/TranseLength
        else:
            if F-FF>0:
                overlap_gene_trans[ind]=TranseLength/GeneLength
                overlap_trans_gene[ind]=1
            else:
                overlap_gene_trans[ind]=(F-II)/GeneLength
                overlap_trans_gene[ind]=(F-II)/TranseLength 
            
    return (GeneIndex,overlap_gene_trans,overlap_trans_gene,TransIndex)

def IsAreaInGene(index,IntAnn,FinalAnn,IntArea,FinalArea,Min_Dist):
    """
   cheack wether a given transcribed area belong to 1 or more annotated region
   Parameters:
        index: index of transcribeds region     
        IntAnn: start locations of annotation (1D array) 
        FinalAnn: end locations of annotation (1D array) 
        IntArea: start locations of detected transcribed regions (1D array)  
        FinalArea: end locations of detected transcribed regions (1D array)
        Min_Dist: minimum required distance between annotated region
        and transcribed regions
    Return:
        boolean 'TRUE'- the transcribed area belong to at least 1 transcribed area
        'FALSE' the trnascribed area doesn't belong to any annotated region
    """
    B=np.logical_or((IntAnn-FinalArea[index])>Min_Dist,(IntArea[index]-FinalAnn)>Min_Dist)    
    return len(B)==np.sum(B) 

def LoopOverlap(Int,Final,IntArea,FinalArea):
    """
   return all the overlap annotated and transcribed regions and overlap rate.
   Parameters:
        Int: start points annotated regions (1D array)     
        Final: end points annotated regions (1D array) 
        IntArea: start points transcribed areas (1D array) 
        FinalArea: end points transcribed areas (1D array)
    Return:
        CC: list of array with annotated regions index, overlap between annotated regions
        and transcribed regions and vice versa, and the index
        of overlap transcribed regions
    """
    CC=[CalcOverlapGeneTranscript(Int,Final,FinalArea[index],IntArea[index],index) \
    for index in range(len(IntArea)) \
    if np.sum(np.logical_not(np.logical_or(FinalArea[index]<=Int,IntArea[index]>=Final)))>0]
    return CC

def MultipleBinSize(List,L_bin):
    """
   multiple a list variables with integer
    """
    return [val * L_bin for val in List]

def FindExactPoint(Serias):
    """
   return the index of the highest value in a numpy array 
    """
    return np.argmax(Serias)


def Load2Streand(location1,location2,Length):
    """
   Allocate number of reads to each pb in the streand array
   Parameters:
      location1: starts read bp locations (1D array)
      location2:ends reads bp locations (1D array)
      Length: number of total bp in the streand
   Return:
      Streand: array of the number of reads in each bp (1D array)
    """
    Streand=np.zeros((Length,), dtype=int)
    for i in xrange(0,len(location1)):
        Streand[location1[i]-1:location2[i]-1]+=1 
    return Streand

def Splite2Bin(Streand,L_bin,BinNum):
    """
   Allocate number of reads to each pb in the streand array
   Parameters:
      Streand: array of number of reads for each bp (1D array)
      L_bin:bin size (in bp's) (1D array)
      BinNum: total number of bins in the streand
   Return:
      Read_Denst: array of the average log density for each bin(1D array)
    """
    Split=np.array_split(Streand[0:BinNum*L_bin],BinNum)
    Read_Denst =np.array(map(np.mean, Split))
    return Read_Denst 

def RPKM(loc1,loc2,Streand,length):
    """
  calculated RPKM for 
   Parameters:
      loc1: start location of a given region
      loc2: end location of a given region
      Streand: number of reads for each bp in the straend (1D array)
      length: number of bp in the streand
   Return:
      RPKM result for the given region
    """
    return np.mean(Streand[loc1:loc2])*(10**9)/length

def LogDens(Streand,L_bin,BinNum):
    """
  calculated RPKM for 
   Parameters:
      Streand: number of reads for each bp in the straend (1D array)
      L_bin: ebin size in bp's
      BinNum: total number of bins in the streand
   Return:
      log density of each bin in the streand (1D array)
    """
    Split=np.array_split(Streand[0:BinNum*L_bin],BinNum)
    Read_Denst =np.array(map(np.mean, Split))
    return np.log(1+Read_Denst) 

def OpenCompliteInputFile(file_name):
    mapping_info=[name.rstrip('\n').split(" ") for name in open(file_name)]
    return mapping_info

def HWComplite(L_min,L_max,region,BinNum2,FDR_Point,Min_Dist,Min_Val,FDR_Region):
    """
  located the transcribed region based on HW algorithem 
   Parameters:
      L_min: Minimum windows size (2**L_Min)
      L_max: Maximum window size (2**L_ma)
      region: the genomic region from which to detect transcribed regions (1D array)
      BinNum2: number of reads for each bp in the region (1D array)
      FDR_Point: fraction of selected extermum points
      Min_Dist: minimum distance between an extermum points
      Min_Val: minimum value for an extermum point
      FDR_Regions: probabilty of not belonging to the lowest 
      Guassian distribution of densities
   Return:
      VInitial_Locations: start locations of transcribed areas (1D array)
      VFinal_Locations:  start locations of transcribed areas (1D array) 
    """
    BreakPoints=[]
    VInitial_Locations=[]
    VFinal_Locations=[]
    flag=1
    while flag==1:
        " collect extermum points from different window sizes from L_min-L_max"
        for L in xrange(L_min,L_max):
            " number of bins per window"
            Range=np.power(2, L)
            " HW coefficent for every bin"
            if BinNum2>2*Range:
                Hij=HaarWavelet(L,Range,region,BinNum2)
                " select extermum points and sort them by value, select FDR fraction"
                "of the points" 
                Loc=np.sort(ExtremLocation_Val(Hij,FDR_Point,Min_Dist,Min_Val))+Range
                if len(Loc)>0:
                    BreakPoints=CombinedPoints(BreakPoints,Loc,Range)
                BreakPoints.astype(int)
            else:
                break
        " calculate average log density for the areas that defined"
        "by the selected break points"
        Dens=AverageLogDensity(region,BreakPoints)
        " remove areas with no reads"
        Dens,Initial_Locations,Final_Locations=RemoveZeroReads(Dens,BreakPoints)
        " algorithem will stop if less from 3 areas exists"
        if len(Dens)>2:
            " calculate the probabilty of each area of belong to the lowest Gaussian"
            Prob=PredictRegion_GMM_3(Dens.reshape(len(Dens), 1))
            " algorithem will stop when there are no regions that arent"
            "belong to the lowest Gaussian"
            if np.sum(Prob<FDR_Region)<1:
                flag=0
            else:
                " select area not belong to the lowest Gaussian"
                Dens,Initial_Locations,Final_Locations=TranslationProbability(Prob,FDR_Region,Dens,Initial_Locations,Final_Locations)
                " combined selected areas with areas from previouse iterations"
                VInitial_Locations,VFinal_Locations=CombinedIterations(VInitial_Locations,VFinal_Locations,Initial_Locations,Final_Locations)
                " remove from the streands reads rom areas that were selecte"
                " at this iteration"
                region=RemovedUsedReads(Initial_Locations,Final_Locations,region)    
    #        count=count+1
    #        print(count)
    #        print(len(VInitial_Locations))
        else:
            flag=0
    return VInitial_Locations,VFinal_Locations

def HWCompliteMethod(L_min,L_max,region,BinNum,FDR_Point,Min_Dist,Min_Val,FDR_Region,method):
    """
  located the transcribed region based on HW algorithem 
   Parameters:
      L_min: Minimum windows size (2**Lod_Min)
      L_max: Maximum window size (2**L_ma)
      region: the genomic region from which to detect transcribed regions (1D array)
      BinNum2: number of reads for each bp in the region (1D array)
      FDR_Point: fraction of selected extermum points
      Min_Dist: minimum distance between an extermum points
      Min_Val: minimum value for an extermum point
      FDR_Regions: probabilty of not belonging to the lowest 
      Guassian distribution of densities
   Return:
      VInitial_Locations: start locations of transcribed areas (1D array)
      VFinal_Locations:  start locations of transcribed areas (1D array) 
    """
    BreakPoints=[]
    VInitial_Locations=[]
    VFinal_Locations=[]
    flag=1
    while flag==1:
        " collect extermum points from different window sizes from L_min-L_max"
        for L in xrange(L_min,L_max):
            " number of bins per window"
            Range=np.power(2, L)
            " HW coefficent for every bin"
            if BinNum>2*Range:
                Hij=HaarWavelet(L,Range,region,BinNum)
                " select extermum points and sort them by value, select FDR fraction"
                "of the points"
                if method=='m':
                    Loc=np.sort(ExtremLocation_Val(Hij,FDR_Point,Min_Dist,Min_Val))+Range
                else if method=='n':
                    Loc=np.sort(ExtremLocation_ValChange(Serias,FDR_Point,Min_Dist,Min_Val))+Range
                    else:
                        Loc=np.sort(ExtremLocation_Random(Hij,FDR_Point,Min_Dist,Min_Val))+Range
                if len(Loc)>0:
                    BreakPoints=CombinedPoints(BreakPoints,Loc,Range)
                BreakPoints.astype(int)
            else:
                break
        " calculate average log density for the areas that defined"
        "by the selected break points"
        Dens=AverageLogDensity(region,BreakPoints)
        " remove areas with no reads"
        Dens,Initial_Locations,Final_Locations=RemoveZeroReads(Dens,BreakPoints)
        " algorithem will stop if less from 3 areas exists"
        if len(Dens)>2:
            " calculate the probabilty of each area of belong to the lowest Gaussian"
            Prob=PredictRegion_GMM_3(Dens.reshape(len(Dens), 1))
            " algorithem will stop when there are no regions that arent"
            "belong to the lowest Gaussian"
            if np.sum(Prob<FDR_Region)<1:
                flag=0
            else:
                " select area not belong to the lowest Gaussian"
                Dens,Initial_Locations,Final_Locations=TranslationProbability(Prob,FDR_Region,Dens,Initial_Locations,Final_Locations)
                " combined selected areas with areas from previouse iterations"
                VInitial_Locations,VFinal_Locations=CombinedIterations(VInitial_Locations,VFinal_Locations,Initial_Locations,Final_Locations)
                " remove from the streands reads rom areas that were selecte"
                " at this iteration"
                region=RemovedUsedReads(Initial_Locations,Final_Locations,region)    
    #        count=count+1
    #        print(count)
    #        print(len(VInitial_Locations))
        else:
            flag=0
    return VInitial_Locations,VFinal_Locations