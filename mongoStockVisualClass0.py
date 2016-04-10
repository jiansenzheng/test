# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 17:40:18 2016

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
filename = 'square_plot_for SH6000316 tickVis_041007.pdf'
filename = os.path.join(path, filename)

# general functions
def plotfun(lis1,lis2,label0,title0,xlab0,ylab0):
    plotx= plt.plot(lis1,lis2,'r',label=label0)
    plt.title(title0)
    plt.xlabel(xlab0)
    plt.ylabel(ylab0)
    return plotx
    
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
class basic_visualize(object):
    def __init__(self,collection,datex,key):
        self.collect=collection
        self.datex=datex
        self.key= key

    def tick_dateS(self):
        volume2= pd.Series()
        sec2= pd.Series()
        for doc0 in self.collect.find({"date":self.datex}):
            volume2 = np.append(volume2,doc0.get(self.key))
            sec2 = np.append(sec2,doc0.get("sec"))
        return [sec2,volume2]
    
    def tick_plotS(self):
        volx=self.tick_dateS()
        plotfun(volx[0],volx[1],self.key,"stock "+self.key+" on "+self.datex,"time",self.key)
        
    def tick_histS(self):
        plt.figure(num=None, figsize=(8,6), dpi=80, facecolor='w', edgecolor='k')
        volx=self.tick_dateS()[1]
        plt.hist(volx)
        plt.title('histogram of '+self.key+" on "+self.datex,fontsize=20)
        plt.show()
        
    def tick_dateS2(self,m,n):
        volume2= pd.Series()
        sec2= pd.Series()
        for doc0 in self.collect.find({"date":self.datex})[m:n]:
            volume2 = np.append(volume2,doc0.get(self.key))
            sec2 = np.append(sec2,doc0.get("sec"))
        return [sec2,volume2]
    
    def tick_plotS2(self,m,n):
        volx=self.tick_dateS2(m,n)
        plotfun(volx[0],volx[1],self.key,"stock "+self.key+" on "+self.datex,"time",self.key)
        
    def tick_histS2(self,m,n):
        plt.figure(num=None, figsize=(8,6), dpi=80, facecolor='w', edgecolor='k')
        volx=self.tick_dateS2(self.datex,self.key,m,n)[1]
        plt.hist(volx)
        plt.title('histogram of '+self.key+" on "+self.datex,fontsize=20)
        plt.show()
          
    def vol_dateS(self):
        volume2= pd.Series()
        sec2= pd.Series()
        for doc0 in self.collect.find({"date":self.datex}):
            volume2 = np.append(volume2,doc0.get("vol"))
            sec2 = np.append(sec2,doc0.get("sec"))
        return [sec2,volume2]
    
    def volume_plotS(self):
        volx= self.vol_dateS()
        plotfun(volx[0],volx[1],"volume","stock trading volume on "+self.datex,"time","volume")
        
    def lobRatio_dateS(self):
        ratio= pd.Series()
        sec2 = pd.Series()
        for doc0 in self.collect.find({"date":self.datex}):
            if (doc0.get("bc1")!=0) and (doc0.get("sc1")!=0) :
                buy = sum(map(doc0.get,["bc1","bc2","bc3","bc4","bc5"]))
                sell= sum(map(doc0.get,["sc1","sc2","sc3","sc4","sc5"]))
                b_over_s= float(buy)/float(sell)
                ratio=np.append(ratio,b_over_s)
                sec2 = np.append(sec2,doc0.get("sec"))
        return [sec2,ratio]
        
    def lob_plotS(self):
        lob=self.lobRatio_dateS()
        plotfun(lob[0],lob[1],"LOB","stock LOB visualization for "+ " on "+self.datex,"time","sum(bc)/sum(sc)")
        
#------------------------------------ ----------- ----------- -----------     
class tick_visualize(object):
    def __init__(self,collection,datex,ax,width,height,fsize):
        self.collect=collection
        self.datex=datex
        self.ax= ax
        self.width = width
        self.height=height
        self.fsize=fsize

    def draw_square(self,gap,i0,color,sign,j0):
        if sign == 1:
            squ=patches.Rectangle(
                (i0*self.width, gap+0.5+(j0-1)*self.height), self.width, self.height,
                facecolor=color,edgecolor= 'none'       # Default
            )    
        elif sign == -1:
            squ=patches.Rectangle(
                (i0*self.width, -gap-j0*self.height+0.5), self.width, self.height,
                facecolor=color,edgecolor= 'none'       # Default
            )   
        return squ
    
    def market_dateS(self,gap,u0,k):
        bcStr = ["bc1","bc2","bc3","bc4","bc5"]
        scStr = ["sc1","sc2","sc3","sc4","sc5"]
        #buyStr =  ["buy1","buy2","buy3","buy4","buy5"]
        #sellStr =  ["sale1","sale2","sale3","sale4","sale5"]
        squareLis = []
        #u0 = 180 # start point with sc2&bc2 !=0
        pos = 0
        for doc0 in self.collect.find({"date":self.datex})[u0:u0+k]:
            if (doc0.get("bc1")!=0) and (doc0.get("sc1")!=0):
                buyV = map(doc0.get,bcStr)
                sellV= map(doc0.get,scStr)
                for i0 in np.arange(0,5,1):
                    colB = str2col(bcStr[i0],buyV[i0])
                    colS = str2col(scStr[i0],sellV[i0])
                    squareLis.append(self.draw_square(gap,pos,colB[0],colB[1],colB[2]))
                    squareLis.append(self.draw_square(gap,pos,colS[0],colS[1],colS[2]))
                print "succeed!"
            else:
                print "bc1,sc1 are zero!"
            pos = pos+1
        return squareLis
    
    #-------------------------
    def draw_squareP(self,i0,color,j0):
        squ=patches.Rectangle(
            (i0*self.width, 0.5+j0*self.height), self.width, self.height,
            facecolor=color,edgecolor= 'none'       # Default
            )
        return squ
    
    def market_dateSP2(self,u0,k):
        bcStr = ["bc1","bc2","bc3","bc4","bc5"]
        scStr = ["sc1","sc2","sc3","sc4","sc5"]
        buyStr =  ["buy1","buy2","buy3","buy4","buy5"]
        sellStr =  ["sale1","sale2","sale3","sale4","sale5"]
        squareLis = []
        #u0 = 180 # start point with sc2&bc2 !=0
        pos = 0
        iniP = 0.0
        for doc0 in self.collect.find({"date":self.datex})[u0:u0+k]:
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
                    squareLis.append(self.draw_squareP(pos+1,colB[0],coorY1))
                    squareLis.append(self.draw_squareP(pos+1,colS[0],coorY2))
                print "succeed!"
                pos = pos+1
            else:
                print "bc1,sc1 are zero!"
        return squareLis
    
    #-------------------------
    def textValue(self,pos1,pos2,num):
        ax.text(self.width*(0.5+pos1),0.5+self.height*(0.5+pos2),str(num),
        verticalalignment='center', horizontalalignment='center',
        color='black', fontsize=self.fsize)
    
    def markVal_dateSP(self,u0,k):
        bcStr = ["bc1","bc2","bc3","bc4","bc5"]
        scStr = ["sc1","sc2","sc3","sc4","sc5"]
        buyStr =  ["buy1","buy2","buy3","buy4","buy5"]
        sellStr =  ["sale1","sale2","sale3","sale4","sale5"]
        #u0 = 180 # start point with sc2&bc2 !=0
        pos = 0
        iniP = 0.0
        for doc0 in self.collect.find({"date":self.datex})[u0:u0+k]:
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
                    self.textValue(pos,coorY1,buyV[i0])
                    self.textValue(pos,coorY2,sellV[i0])
                    if i0 ==4:
                        self.textValue(pos,coorY2+1,volume)
                print "text succeed!"
                pos = pos+1
            else:
                print "bc2,sc2 are zero!"
    
    def markVal_dateSP_hms(self,u0,k):
        bcStr = ["bc1","bc2","bc3","bc4","bc5"]
        scStr = ["sc1","sc2","sc3","sc4","sc5"]
        buyStr =  ["buy1","buy2","buy3","buy4","buy5"]
        sellStr =  ["sale1","sale2","sale3","sale4","sale5"]
        #u0 = 180 # start point with sc2&bc2 !=0
        pos = 0
        iniP = 0.0
        for doc0 in self.collect.find({"date":self.datex})[u0:u0+k]:
            if (doc0.get("bc2")!=0) and (doc0.get("sc2")!=0):
                buyV = map(doc0.get,bcStr)
                sellV= map(doc0.get,scStr)
                buyP = map(doc0.get,buyStr)
                sellP= map(doc0.get,sellStr)
                if pos ==0:
                    iniP = doc0.get("buy1")
                    for i0 in np.arange(0,5,1):
                        coorY1= (buyP[i0]-iniP)/0.01
                        coorY2= (sellP[i0]-iniP)/0.01
                        self.textValue(0,coorY1,buyP[i0])
                        self.textValue(0,coorY2,sellP[i0])
                        print "successful insertion!"
                volume= doc0.get("vol")
                hms = doc0.get("time").split(' ')[1]
                for i0 in np.arange(0,5,1):
                    coorY1= (buyP[i0]-iniP)/0.01
                    coorY2= (sellP[i0]-iniP)/0.01
                    self.textValue(pos+1,coorY1,buyV[i0])
                    self.textValue(pos+1,coorY2,sellV[i0])
                    if i0 ==4:
                        self.textValue(pos+1,coorY2+1,volume)
                        self.textValue(pos+1,coorY1-2,hms)
                print "text succeed!"
                pos = pos+1
            else:
                print "bc2,sc2 are zero!"
    
    #-------------------------
    def plotLine(self,pos1,pos2,color):
        plt.plot((pos1*self.width,(pos1+1.0)*self.width), 
                 (0.5+(pos2+0.5)*self.height,0.5+(pos2+0.5)*self.height), color)
    
    def priceLine_dateSP2(self,u0,k,color):
        #u0 = 180 # start point with sc2&bc2 !=0
        pos = 0
        iniP = 0.0
        volumeX = 0.0
        priceX= 0.0
        for doc0 in self.collect.find({"date":self.datex})[u0:u0+k]:
            if (doc0.get("bc2")!=0) and (doc0.get("sc2")!=0):
                vol_1= float(doc0.get("vol"))
                pri_1 = doc0.get("price")            
                if pos ==0:
                    iniP = doc0.get("buy1")
                    coorY= (pri_1-iniP)/0.01
                    self.plotLine(pos+1,coorY,color)
                else:
                    priceMA2= np.average([priceX,pri_1], weights=[float(volumeX),float(vol_1)])
                    coorY= (priceMA2-iniP)/0.01
                    self.plotLine(pos+1,coorY,color)                         
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
plt.axis('off')

width=0.1
height= 0.05/2.5
datex = "2013-03-18"
u0 = 600
fsize=10
k=10
#create the class
collection= SH600516
name=collection.name
classTick= tick_visualize(collection,datex,ax,width,height,fsize)
#start plotting
sqPatch = classTick.market_dateSP2(u0,k)
for p in sqPatch:
    ax.add_patch(p)

classTick.markVal_dateSP_hms(u0,k)
classTick.priceLine_dateSP2(u0,k,'b-')
plt.title('visualization of partial '+str(name)+' tickData on '+datex+'with u0 = '+str(u0),fontsize=18)

plt.show()
fig.savefig(filename,format='pdf',dpi=1000)

plt.close(fig)
#plot by the class basic_visualize()
fig1 = plt.figure(figsize=(20, 15), dpi=240, facecolor='w', edgecolor='k')
ax1 = fig.add_subplot(111, aspect='equal')
classBasic=basic_visualize(collection,datex,'vol')
classBasic.tick_plotS()
plt.title('trading volume for '+str(name)+' on '+datex)