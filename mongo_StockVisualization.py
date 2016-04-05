# -*- coding: utf-8 -*-
"""
Created on Tue Apr 05 11:23:17 2016

@author: Jiansen
"""
import numpy as np
import pandas as pd
import datetime
import time
import os
from bson.objectid import ObjectId
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pymongo as pm
from pymongo import MongoClient

# filename and path setup for the exported pictures
path = 'C:/Users/Jiansen/Documents/Finance/a_stock_intra/visualization'
filename = 'square_plot_for SH6000316 tickVis_040502.pdf'
filename = os.path.join(path, filename)

def plotfun(lis1,lis2,label0,title0,xlab0,ylab0):
    plotx= plt.plot(lis1,lis2,'r',label=label0)
    plt.title(title0)
    plt.xlabel(xlab0)
    plt.ylabel(ylab0)
    return plotx
    
    

#connect the database

client = MongoClient('localhost',27017)
print client
db= client.stocks
SH600516=db.SH600516

#test the connection
print SH600516.count()
print SH600516.find_one({"vol":{"$gt":200000}})
s1= SH600516.find_one()
time0=s1.get("time")
print time0

#-------------------------
def tick_dateS(datex,key):
    volume2= pd.Series()
    sec2= pd.Series()
    for doc0 in SH600516.find({"date":datex}):
        volume2 = np.append(volume2,doc0.get(key))
        sec2 = np.append(sec2,doc0.get("sec"))
    return [sec2,volume2]

def tick_plotS(datex,key):
    volx=tick_dateS(datex,key)
    plotfun(volx[0],volx[1],key,"stock "+key+" on "+datex,"time",key)
    
def tick_histS(datex,key):
    plt.figure(num=None, figsize=(8,6), dpi=80, facecolor='w', edgecolor='k')
    volx=tick_dateS(datex,key)[1]
    plt.hist(volx)
    plt.title('histogram of '+key+" on "+datex,fontsize=20)
    plt.show()
    
def tick_dateS2(datex,key,m,n):
    volume2= pd.Series()
    sec2= pd.Series()
    for doc0 in SH600516.find({"date":datex})[m:n]:
        volume2 = np.append(volume2,doc0.get(key))
        sec2 = np.append(sec2,doc0.get("sec"))
    return [sec2,volume2]

def tick_plotS2(datex,key,m,n):
    volx=tick_dateS2(datex,key,m,n)
    plotfun(volx[0],volx[1],key,"stock "+key+" on "+datex,"time",key)
    
def tick_histS2(datex,key,m,n):
    plt.figure(num=None, figsize=(8,6), dpi=80, facecolor='w', edgecolor='k')
    volx=tick_dateS2(datex,key,m,n)[1]
    plt.hist(volx)
    plt.title('histogram of '+key+" on "+datex,fontsize=20)
    plt.show()
    
#---------------------------------------    
def vol_dateS(datex):
    volume2= pd.Series()
    sec2= pd.Series()
    for doc0 in SH600516.find({"date":datex}):
        volume2 = np.append(volume2,doc0.get("vol"))
        sec2 = np.append(sec2,doc0.get("sec"))
    return [sec2,volume2]

def volume_plotS(datex):
    volx=vol_dateS(datex)
    plotfun(volx[0],volx[1],"volume","stock trading volume on "+datex,"time","volume")
    
def lobRatio_dateS(datex):
    ratio= pd.Series()
    sec2 = pd.Series()
    for doc0 in SH600516.find({"date":datex}):
        if (doc0.get("bc1")!=0) and (doc0.get("sc1")!=0) :
            buy = sum(map(doc0.get,["bc1","bc2","bc3","bc4","bc5"]))
            sell= sum(map(doc0.get,["sc1","sc2","sc3","sc4","sc5"]))
            b_over_s= float(buy)/float(sell)
            ratio=np.append(ratio,b_over_s)
            sec2 = np.append(sec2,doc0.get("sec"))
    return [sec2,ratio]
def lob_plotS(datex):
    lob=lobRatio_dateS(datex)
    plotfun(lob[0],lob[1],"LOB","stock LOB visualization for SH600516 on "+datex,"time","sum(bc)/sum(sc)")
    
#-------------------------    
def val2col(value):
    unit=10000
    scale = 200
    if value <= 5*unit:
        scale = 150
    elif value <= 10*unit:
        scale = 120
    elif value <= 15*unit:
        scale = 90
    elif value <= 20*unit:
        scale = 40
    else:
        scale = 0
    return scale

def str2col(str,value):
    key1 = str[0]
    key2 = str[2]
    #position = 0
    scale = val2col(value)
    sign = 0
    if key1== 'b':
        sign = 1
        col= '#%02x%02x%02x' % (255,scale,scale) 
    elif key1=='s':
        sign = -1
        col= '#%02x%02x%02x' % (scale,255,scale)
    return [col,sign,float(key2)]

def draw_square(width,height,gap,i0,color,sign,j0):
    if sign == 1:
        squ=patches.Rectangle(
            (i0*width, gap+0.5+(j0-1)*height), width, height,
            facecolor=color,edgecolor= 'none'       # Default
        )    
    elif sign == -1:
        squ=patches.Rectangle(
            (i0*width, -gap-j0*height+0.5), width, height,
            facecolor=color,edgecolor= 'none'       # Default
        )   
    return squ

def market_dateS(datex,width,height,gap,u0,k):
    bcStr = ["bc1","bc2","bc3","bc4","bc5"]
    scStr = ["sc1","sc2","sc3","sc4","sc5"]
    #buyStr =  ["buy1","buy2","buy3","buy4","buy5"]
    #sellStr =  ["sale1","sale2","sale3","sale4","sale5"]
    squareLis = []
    #u0 = 180 # start point with sc2&bc2 !=0
    pos = 0
    for doc0 in SH600516.find({"date":datex})[u0:u0+k]:
        if (doc0.get("bc1")!=0) and (doc0.get("sc1")!=0):
            buyV = map(doc0.get,bcStr)
            sellV= map(doc0.get,scStr)
            for i0 in np.arange(0,5,1):
                colB = str2col(bcStr[i0],buyV[i0])
                colS = str2col(scStr[i0],sellV[i0])
                squareLis.append(draw_square(width,height,gap,pos,colB[0],colB[1],colB[2]))
                squareLis.append(draw_square(width,height,gap,pos,colS[0],colS[1],colS[2]))
            print "succeed!"
        else:
            print "bc1,sc1 are zero!"
        pos = pos+1
    return squareLis

#-------------------------
def draw_squareP(width,height,i0,color,j0):
    squ=patches.Rectangle(
        (i0*width, 0.5+j0*height), width, height,
        facecolor=color,edgecolor= 'none'       # Default
        )
    return squ

def market_dateSP(datex,width,height,u0,k):
    bcStr = ["bc1","bc2","bc3","bc4","bc5"]
    scStr = ["sc1","sc2","sc3","sc4","sc5"]
    buyStr =  ["buy1","buy2","buy3","buy4","buy5"]
    sellStr =  ["sale1","sale2","sale3","sale4","sale5"]
    squareLis = []
    #u0 = 180 # start point with sc2&bc2 !=0
    pos = 0
    iniP = 0.0
    for doc0 in SH600516.find({"date":datex})[u0:u0+k]:
        if (doc0.get("bc2")!=0) and (doc0.get("sc2")!=0):
            if pos ==0:
                iniP = doc0.get("buy1")             
            buyV = map(doc0.get,bcStr)
            sellV= map(doc0.get,scStr)
            buyP = map(doc0.get,buyStr)
            sellP= map(doc0.get,sellStr)
            for i0 in np.arange(0,5,1):
                colB = str2col(bcStr[i0],buyV[i0])
                colS = str2col(scStr[i0],sellV[i0])
                coorY1= (buyP[i0]-iniP)/0.01
                coorY2= (sellP[i0]-iniP)/0.01
                squareLis.append(draw_squareP(width,height,pos,colB[0],coorY1))
                squareLis.append(draw_squareP(width,height,pos,colS[0],coorY2))
            print "succeed!"
            pos = pos+1
        else:
            print "bc1,sc1 are zero!"
    return squareLis

#-------------------------
def textValue(ax,width,height,pos1,pos2,num,fsize):
    ax.text(width/2.0+pos1*width,0.5+height/2.0+pos2*height,str(num),
    verticalalignment='center', horizontalalignment='center',
    color='black', fontsize=fsize)

def markVal_dateSP(ax,datex,width,height,u0,k,fsize):
    bcStr = ["bc1","bc2","bc3","bc4","bc5"]
    scStr = ["sc1","sc2","sc3","sc4","sc5"]
    buyStr =  ["buy1","buy2","buy3","buy4","buy5"]
    sellStr =  ["sale1","sale2","sale3","sale4","sale5"]
    #u0 = 180 # start point with sc2&bc2 !=0
    pos = 0
    iniP = 0.0
    for doc0 in SH600516.find({"date":datex})[u0:u0+k]:
        if (doc0.get("bc2")!=0) and (doc0.get("sc2")!=0):
            if pos ==0:
                iniP = doc0.get("buy1")             
            buyV = map(doc0.get,bcStr)
            sellV= map(doc0.get,scStr)
            buyP = map(doc0.get,buyStr)
            sellP= map(doc0.get,sellStr)
            volume= doc0.get("vol")
            for i0 in np.arange(0,5,1):
                coorY1= (buyP[i0]-iniP)/0.01
                coorY2= (sellP[i0]-iniP)/0.01
                textValue(ax,width,height,pos,coorY1,buyV[i0],fsize)
                textValue(ax,width,height,pos,coorY2,sellV[i0],fsize)
                if i0 ==4:
                    textValue(ax,width,height,pos,coorY2+1,volume,fsize)
            print "text succeed!"
            pos = pos+1
        else:
            print "bc2,sc2 are zero!"

#-------------------------
def plotLine(width,height,pos1,pos2,color):
    plt.plot((pos1*width,pos1*width+width), (0.5+pos2*height,0.5+pos2*height), color)

def priceLine_dateSP(datex,width,height,u0,k,color):
    #u0 = 180 # start point with sc2&bc2 !=0
    pos = 0
    iniP = 0.0
    volumeX = 0.0
    priceX= 0.0
    for doc0 in SH600516.find({"date":datex})[u0:u0+k]:
        if (doc0.get("bc2")!=0) and (doc0.get("sc2")!=0):
            vol_1= float(doc0.get("vol"))
            pri_1 = doc0.get("price")            
            if pos ==0:
                iniP = doc0.get("buy1")
                coorY= (pri_1-iniP)/0.01
                plotLine(width,height,pos,coorY,color)
            else:
                priceMA2= np.average([priceX,pri_1], weights=[float(volumeX),float(vol_1)])
                coorY= (priceMA2-iniP)/0.01
                plotLine(width,height,pos,coorY,color)                         
            print "price line succeed!"
            pos = pos+1
            volumeX = vol_1
            priceX = pri_1
        else:
            print "bc2,sc2 are zero!"
#-------------------
# test codes
# initialization 
fig = plt.figure(figsize=(20, 15), dpi=240, facecolor='w', edgecolor='k')
ax = fig.add_subplot(111, aspect='equal')
width=0.1
height= 0.05/2.5
datex = "2013-03-18"
u0 = 700
fsize=10
k=10
sqPatch = market_dateSP(datex,width,height,u0,k)
for p in sqPatch:
    ax.add_patch(p)
plt.title('visualization of partial SH600316 tickData on '+datex+'with u0 = '+str(u0),fontsize=18)
markVal_dateSP(ax,datex,width,height,u0,k,fsize)
priceLine_dateSP(datex,width,height,u0,k,'b-')
plt.show()
            
#save the fig
fig.savefig(filename,format='pdf',dpi=1000)