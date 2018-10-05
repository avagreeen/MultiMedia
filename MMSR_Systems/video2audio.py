#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 21:30:07 2018

@author: Fenglu Xu

"""
# In[]:
import os
os.chdir('/Users/xuecho/Desktop/multi') 
fileDir = os.path.dirname(os.path.realpath('__file__'))
print fileDir

moive_files_dir = fileDir + '/LIRIS-ACCEDE-data/data'
audio_files_dir = fileDir + '/LIRIS-ACCEDE-data/audio/'
print moive_files_dir

files = os.listdir(moive_files_dir)  
    
# In[]:
from moviepy.editor import *
for file in files:
     audioclip = AudioFileClip(moive_files_dir + '/' + file)
     name = file.split('.')
     audioclip.write_audiofile(audio_files_dir+name[0]+'.mp3')
     
  # In[]:
from yaafelib import FeaturePlan, Engine, AudioFileProcessor
import numpy 

def init():
    global engine
    fp = FeaturePlan(sample_rate=44100, resample=True, time_start=0,time_limit=20)           
    
    fp.addFeature("loudness: Loudness") 
    fp.addFeature("perceptualSharpness: PerceptualSharpness")
    fp.addFeature("perceptualSpread: PerceptualSpread")
    fp.addFeature("obsi: OBSI")  
    fp.addFeature("obsir: OBSIR")
    
    df = fp.getDataFlow()
    engine = Engine()                  # Engine setup
    engine.load(df)

    return 'initialization'

def startEngine(path):
    global afp, features
    afp = AudioFileProcessor()
    afp.processFile(engine,path)       

    features = engine.readAllOutputs() # matrix of all features

    return 'extracted'

#MFCC
def getMFCC():
    mfcc = features.get('mfcc')
    mfccMean=mfcc.mean(axis=0)       # mean
    mfccMean=mfccMean.reshape(-1,)
    mfccVar=mfcc.var(axis=0)         # variance
    mfccVar=mfccVar.reshape(-1,)
    return mfcc

def getLoudness():
    loudness = features.get('loudness')
    if len(loudness)==0:
        return 0,0
    loudnessMax=loudness.max(axis=0)       
    loudnessMax=loudnessMax.reshape(-1,)
    loudnessMin=loudness.min(axis=0)       
    loudnessMin=loudnessMin.reshape(-1,)
    return min(loudnessMin),max(loudnessMax)

def getPerceptualSharpness():
    perceptualSharpness = features.get('perceptualSharpness')
    if len(perceptualSharpness)==0:
        return 0,0
    perceptualSharpnessMax=perceptualSharpness.max(axis=0)       
    perceptualSharpnessMax=perceptualSharpnessMax.reshape(-1,)
    perceptualSharpnessMin=perceptualSharpness.min(axis=0)       
    perceptualSharpnessMin=perceptualSharpnessMin.reshape(-1,)
    return min(perceptualSharpnessMin),max(perceptualSharpnessMax)


def getPerceptualSpread():
    perceptualSpread = features.get('perceptualSpread')
    if len(perceptualSpread)==0:
        return 0,0
    perceptualSpreadMax=perceptualSpread.max(axis=0)       
    perceptualSpreadMax=perceptualSpreadMax.reshape(-1,)
    perceptualSpreadMin=perceptualSpread.min(axis=0)       
    perceptualSpreadMin=perceptualSpreadMin.reshape(-1,)
    return min(perceptualSpreadMin), max(perceptualSpreadMax)

def getOBSI():
    obsi = features.get('obsi')
    if len(obsi)==0:
        return 0,0
    obsiMax=obsi.max(axis=0)       
    obsiMax=obsiMax.reshape(-1,)
    obsiMin=obsi.min(axis=0)       
    obsiMin=obsiMin.reshape(-1,)
    return min(obsiMin), max(obsiMax)


def getOBSIR():
    obsir = features.get('obsir')
    if len(obsir)==0:
        return 0,0
    obsirMax=obsir.max(axis=0)       
    obsirMax=obsirMax.reshape(-1,)
    obsirMin=obsir.min(axis=0)       
    obsirMin=obsirMin.reshape(-1,)
    return min(obsirMin),max(obsirMax)

def main():
    files = os.listdir(audio_files_dir)  
    
    text_file = open("Output.txt", "w")
    text_file.write("audioID,loudnessMin, loudnessMax, perceptualSharpnessMin, perceptualSharpnessMax, perceptualSpreadMin, perceptualSpreadMax, obsiMin, obsiMax,obsirMin, obsirMax")#

    count = 0
    for file in files:
        audio_path = audio_files_dir + file   
        init()
        startEngine(audio_path)
        name = file.split('.')
        obsirMin,obsirMax = getOBSIR()
        obsiMin,obsiMax = getOBSI()
        perceptualSpreadMin,perceptualSpreadMax = getPerceptualSpread()
        perceptualSharpnessMin, perceptualSharpnessMax = getPerceptualSharpness()
        loudnessMin, loudnessMax =getLoudness()
        line = [name[0],loudnessMin,loudnessMax, perceptualSharpnessMin, perceptualSharpnessMax, perceptualSpreadMin, perceptualSpreadMax,obsiMin, obsiMax,obsirMin, obsirMax]#
        text_file.write('\n')
        text_file.write(str(line))
        count += 1
        print count
    text_file.close()
    
if __name__ == '__main__':
    main()
