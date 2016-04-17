# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 08:59:04 2016

@author: Jiansen
"""

%matplotlib inline
import numpy as np
import pandas as pd
import datetime
import time
import os
from bson.objectid import ObjectId
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn import linear_model

import pymongo as pm
from pymongo import MongoClient


#connect the database

client = MongoClient('localhost',27017)
print client
db= client.stocks
SZ002500=db.SZ002500

#test the connection
print SZ002500.count()
print SZ002500.find_one({"vol":{"$gt":200000}})
s1= SZ002500.find_one()
time0=s1.get("time")
print time0


# filename and path setup for the exported pictures
path = 'C:/Users/Jiansen/Documents/Finance/a_stock_intra/visualization/SZ002500'
filename = 'square_plot_for SZ002500 tickVis_041201.pdf'
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
# spread data/plot/histogram methods defines on April 12th 2016        
    def spread_date(self,key1,key2):
        spread=pd.Series()
        sec2= pd.Series()
        for doc0 in self.collect.find({"date":self.datex}):
            spread = np.append(spread,float(doc0.get(key1))-float(doc0.get(key2)))
            sec2 = np.append(sec2,doc0.get("sec"))  
        return [sec2,spread]
    
    def spread_plot(self,key1,key2):
        sp=self.spread_date(key1,key2)
        plotfun(sp[0],sp[1],str(key1)+"—"+str(key2),"stock "+str(key1)+"—"+str(key2)+" on "+self.datex,"time","price difference")
    
    def spread_hist(self,key1,key2):
        plt.figure(num=None, figsize=(8,6), dpi=80, facecolor='w', edgecolor='k')
        sp=self.spread_date(key1,key2)[1]
        plt.hist(sp)
        #plt.title('histogram of '+key1+"——"+key2+" on "+self.datex,fontsize=20)
        plt.title('histogram of '+self.key+" on "+self.datex,fontsize=20)
        plt.show()

    def spread_date2(self,key1,key2,m,n):
        spread=pd.Series()
        sec2= pd.Series()
        for doc0 in self.collect.find({"date":self.datex})[m:n]:
            spread = np.append(spread,float(doc0.get(key1))-float(doc0.get(key2)))
            #spread = np.append(spread,doc0.get(key1))
            sec2 = np.append(sec2,doc0.get("sec"))  
        return [sec2,spread]
            
    def spread_plot2(self,key1,key2,m,n):
        sp=self.spread_date2(key1,key2,m,n)
        #plotfun(sp[0],sp[1],key1+"——"+key2,"stock "+key1+"——"+key2+" on "+self.datex,"time","price difference")
        plotfun(sp[0],sp[1],self.key,"stock "+self.key+" on "+self.datex,"time","price difference")
    
    def spread_hist2(self,key1,key2,m,n):
        plt.figure(num=None, figsize=(8,6), dpi=80, facecolor='w', edgecolor='k')
        sp=self.spread_date2(key1,key2,m,n)[1]
        plt.hist(sp)
        plt.title('histogram of '+self.key+" on "+self.datex,fontsize=20)
        plt.show()
# compute the liquidity index based on spread
    def liqSpr_date(self):
        liq=pd.Series()
        sec2= pd.Series()
        key1= ['sale1','buy1']
        key2= ['bc1','sc1']
        for doc0 in self.collect.find({"date":self.datex}):
            if (doc0.get(key2[0])!=0) and (doc0.get(key2[1])!=0) and (doc0.get(key1[0])!=doc0.get(key1[1])):
                order=float(doc0.get(key2[0]))+float(doc0.get(key2[1]))
                sp=float(doc0.get(key1[0]))-float(doc0.get(key1[1])) 
                liq= np.append(liq,order/sp/100.0)
                sec2 = np.append(sec2,doc0.get("sec"))  
        return [sec2,liq]
    
    def liqSpr_plot(self):
        sp=self.liqSpr_date()
        plotfun(sp[0],sp[1],"spread liquidity","stock spread based Liquidity on "+self.datex,"time","liquidity")
    
    def liqSpr_hist(self):
        #plt.figure(num=None, figsize=(8,6), dpi=80, facecolor='w', edgecolor='k')
        sp=self.liqSpr_date()[1]
        plt.hist(sp)
        plt.title('histogram of  on stock spread based Liquidity'+self.datex,fontsize=20)
        
    def liqSprVol_date(self):
        liq=pd.Series()
        sec2= pd.Series()
        volume = pd.Series()
        key1= ['sale1','buy1']
        key2= ['bc1','sc1','vol']
        for doc0 in self.collect.find({"date":self.datex}):
            if (doc0.get(key2[0])!=0) and (doc0.get(key2[1])!=0) and (doc0.get(key1[0])!=doc0.get(key1[1])):
                order=float(doc0.get(key2[0]))+float(doc0.get(key2[1]))
                sp=float(doc0.get(key1[0]))-float(doc0.get(key1[1])) 
                vol=float(doc0.get(key2[2])/100.0)
                liq= np.append(liq,order/sp/100.0)
                sec2 = np.append(sec2,doc0.get("sec"))  
                volume= np.append(volume,vol)
        return [sec2,liq,volume]
        
    def liqSprVol_plot(self,ax1):
        sp=self.liqSprVol_date()
        plotx= ax1.plot(sp[0],sp[1],'r-',label="spread liquidity")
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('liquidity', color='r',fontsize=16)
        ax2 = ax1.twinx()
        ax2.plot(sp[0],sp[2],'g--')
        ax2.set_ylabel('volume', color='g',fontsize=16)
        #plotfun(sp[0],sp[1],"spread liquidity","stock spread based Liquidity on "+self.datex,"time","liquidity")
#-----------------------------
    def liqSprVol_date2(self,m,n):
        liq = pd.Series()
        sec2= pd.Series()
        volume = pd.Series()
        key1= ['sale1','buy1']
        key2= ['bc1','sc1','vol']
        for doc0 in self.collect.find({"date":self.datex})[m:n]:
            if (doc0.get(key2[0])!=0) and (doc0.get(key2[1])!=0) and (doc0.get(key1[0])!=doc0.get(key1[1])):
                order=float(doc0.get(key2[0]))+float(doc0.get(key2[1]))
                sp=float(doc0.get(key1[0]))-float(doc0.get(key1[1])) 
                vol=float(doc0.get(key2[2])/100.0)
                liq= np.append(liq,order/sp/100.0)
                sec2 = np.append(sec2,doc0.get("sec"))  
                volume= np.append(volume,vol)
        return [sec2,liq,volume]
        
    def liqSprVol_plot2(self,ax1,m,n):
        sp=self.liqSprVol_date2(m,n)
        plotx= ax1.plot(sp[0],sp[1],'r-',label="spread liquidity")
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('liquidity', color='r',fontsize=16)
        ax2 = ax1.twinx()
        ax2.plot(sp[0],sp[2],'g--')
        ax2.set_ylabel('volume', color='g',fontsize=16)
#------------------------------------ ----------- -------
    def range_date(self):
        amplitudeB= pd.Series()
        amplitudeS= pd.Series()
        sec2= pd.Series()
        volume = pd.Series()
        price = pd.Series()
        buyStr =  ["buy1","buy2","buy3","buy4","buy5"]
        sellStr =  ["sale1","sale2","sale3","sale4","sale5"]
        for doc0 in self.collect.find({"date":self.datex}):
            if (doc0.get("buy5")!=0) and (doc0.get("sale5")!=0):
                rangeB= np.std(map(doc0.get,buyStr))
                rangeS = np.std(map(doc0.get,sellStr))
                amplitudeB= np.append(amplitudeB,rangeB)
                amplitudeS= np.append(amplitudeS,rangeS)
                vol =float(doc0.get("vol")/100.0)
                volume= np.append(volume,vol)
                price = np.append(price,doc0.get("price"))
                sec2 = np.append(sec2,doc0.get("sec"))  
        return [sec2,amplitudeB,amplitudeS,price,volume]
    
    def range_plot(self,ax1):
        sp=self.range_date()
        plotx= ax1.plot(sp[0],sp[1],'r-',label="Buy Cap")
        ploty= ax1.plot(sp[0],sp[2],'g-',label="Sell Cap")
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('amplitude', color='k',fontsize=16)
        ax2 = ax1.twinx()
        ax2.plot(sp[0],sp[3],'b--')
        ax2.set_ylabel('traded price', color='b',fontsize=16)
#------------------------------------------------------
    def range_date2(self,m,n):
        amplitudeB= pd.Series()
        amplitudeS= pd.Series()
        sec2= pd.Series()
        volume = pd.Series()
        price = pd.Series()
        buyStr =  ["buy1","buy2","buy3","buy4","buy5"]
        sellStr =  ["sale1","sale2","sale3","sale4","sale5"]
        for doc0 in self.collect.find({"date":self.datex})[m:n]:
            if (doc0.get("buy5")!=0) and (doc0.get("sale5")!=0):
                rangeB= np.std(map(doc0.get,buyStr))
                rangeS = np.std(map(doc0.get,sellStr))
                amplitudeB= np.append(amplitudeB,rangeB)
                amplitudeS= np.append(amplitudeS,rangeS)
                vol =float(doc0.get("vol")/100.0)
                volume= np.append(volume,vol)
                price = np.append(price,doc0.get("price"))
                sec2 = np.append(sec2,doc0.get("sec"))  
        return [sec2,amplitudeB,amplitudeS,price,volume]
    
    def range_plot2(self,ax1,m,n):
        sp=self.range_date2(m,n)
        plotx= ax1.plot(sp[0],sp[1],'r-',label="Buy Cap")
        ploty= ax1.plot(sp[0],sp[2],'g-',label="Sell Cap")
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('amplitude', color='k',fontsize=16)
        ax2 = ax1.twinx()
        ax2.plot(sp[0],sp[3],'b--')
        ax2.set_ylabel('traded price', color='b',fontsize=16)

#------------------------------------ ----------- -------        
    def capPrice_date(self):
        sec2= pd.Series()
        priceB= pd.Series()
        priceS= pd.Series()
        price = pd.Series()
        #volume = pd.Series()
        buyStr =  ["buy1","buy2","buy3","buy4","buy5"]
        sellStr =  ["sale1","sale2","sale3","sale4","sale5"]
        bcStr = ["bc1","bc2","bc3","bc4","bc5"]
        scStr = ["sc1","sc2","sc3","sc4","sc5"]
        for doc0 in self.collect.find({"date":self.datex}):
            if (doc0.get("buy5")!=0) and (doc0.get("sale5")!=0):
                capB = map(doc0.get,buyStr)
                volB = np.sqrt(np.array(map(doc0.get,bcStr)))
                capS = map(doc0.get,sellStr)
                volS = np.sqrt(np.array(map(doc0.get,scStr)))
                priceB = np.append(priceB,np.average(capB,weights=volB))
                priceS = np.append(priceS,np.average(capS,weights=volS))
                price = np.append(price,doc0.get("price"))
                sec2 = np.append(sec2,doc0.get("sec"))  
        return [sec2,priceB,priceS,price]
    
    def capPrice_plot(self,ax1):
        sp=self.capPrice_date()
        plotx= ax1.plot(sp[0],sp[1],'r-',label="Buy Cap")
        ploty= ax1.plot(sp[0],sp[2],'g-',label="Sell Cap")
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('size weighted order price', color='k',fontsize=16)
        ax2 = ax1.twinx()
        ax2.plot(sp[0],sp[3],'b--')
        ax2.set_ylabel('traded price', color='b',fontsize=16)

#--------------sqrt(volume) averaged order price---------------------- ----------- -------        
    def capPrice_date2(self,m,n):
        sec2= pd.Series()
        priceB= pd.Series()
        priceS= pd.Series()
        price = pd.Series()
        #volume = pd.Series()
        buyStr =  ["buy1","buy2","buy3","buy4","buy5"]
        sellStr =  ["sale1","sale2","sale3","sale4","sale5"]
        bcStr = ["bc1","bc2","bc3","bc4","bc5"]
        scStr = ["sc1","sc2","sc3","sc4","sc5"]
        for doc0 in self.collect.find({"date":self.datex})[m:n]:
            if (doc0.get("buy5")!=0) and (doc0.get("sale5")!=0):
                capB = map(doc0.get,buyStr)
                volB = np.sqrt(np.array(map(doc0.get,bcStr)))
                capS = map(doc0.get,sellStr)
                volS = np.sqrt(np.array(map(doc0.get,scStr)))
                priceB = np.append(priceB,np.average(capB,weights=volB))
                priceS = np.append(priceS,np.average(capS,weights=volS))
                price = np.append(price,doc0.get("price"))
                sec2 = np.append(sec2,doc0.get("sec"))  
        return [sec2,priceB,priceS,price]
    
    def capPrice_plot2(self,ax1,m,n):
        sp=self.capPrice_date2(m,n)
        plotx= ax1.plot(sp[0],sp[1],'r-',label="Buy Cap")
        ploty= ax1.plot(sp[0],sp[2],'g-',label="Sell Cap")
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('size weighted order price', color='k',fontsize=16)
        ax2 = ax1.twinx()
        ax2.plot(sp[0],sp[3],'b--')
        ax2.set_ylabel('traded price', color='b',fontsize=16)

#------------volume averaged order price------------------------ ----------- -------        
    def capPrice_date2S(self,m,n):
        sec2= pd.Series()
        priceB= pd.Series()
        priceS= pd.Series()
        price = pd.Series()
        #volume = pd.Series()
        buyStr =  ["buy1","buy2","buy3","buy4","buy5"]
        sellStr =  ["sale1","sale2","sale3","sale4","sale5"]
        bcStr = ["bc1","bc2","bc3","bc4","bc5"]
        scStr = ["sc1","sc2","sc3","sc4","sc5"]
        for doc0 in self.collect.find({"date":self.datex})[m:n]:
            if (doc0.get("buy5")!=0) and (doc0.get("sale5")!=0):
                capB = map(doc0.get,buyStr)
                volB = np.array(map(doc0.get,bcStr))
                capS = map(doc0.get,sellStr)
                volS = np.array(map(doc0.get,scStr))
                priceB = np.append(priceB,np.average(capB,weights=volB))
                priceS = np.append(priceS,np.average(capS,weights=volS))
                price = np.append(price,doc0.get("price"))
                sec2 = np.append(sec2,doc0.get("sec"))  
        return [sec2,priceB,priceS,price]
    
    def capPrice_plot2S(self,ax1,m,n):
        sp=self.capPrice_date2S(m,n)
        plotx= ax1.plot(sp[0],sp[1],'r-',label="Buy Cap")
        ploty= ax1.plot(sp[0],sp[2],'g-',label="Sell Cap")
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('size weighted order price', color='k',fontsize=16)
        ax2 = ax1.twinx()
        ax2.plot(sp[0],sp[3],'b--')
        ax2.set_ylabel('traded price', color='b',fontsize=16)
        

#------------volume averaged order price difference------------------------ ----------- -------        
    def capPriceDif_date2S(self,m,n):
        sec2= pd.Series()
        priceDif= pd.Series()
        price = pd.Series()
        #volume = pd.Series()
        buyStr =  ["buy1","buy2","buy3","buy4","buy5"]
        sellStr =  ["sale1","sale2","sale3","sale4","sale5"]
        bcStr = ["bc1","bc2","bc3","bc4","bc5"]
        scStr = ["sc1","sc2","sc3","sc4","sc5"]
        for doc0 in self.collect.find({"date":self.datex})[m:n]:
            if (doc0.get("buy5")!=0) and (doc0.get("sale5")!=0):
                capB = map(doc0.get,buyStr)
                volB = np.array(map(doc0.get,bcStr))
                capS = map(doc0.get,sellStr)
                volS = np.array(map(doc0.get,scStr))
                capDif = np.average(capB,weights=volB)- np.average(capS,weights=volS)
                priceDif = np.append(priceDif,capDif) 
                price = np.append(price,doc0.get("price"))
                sec2 = np.append(sec2,doc0.get("sec"))  
        return [sec2,priceDif,price]
    
    def capPriceDif_plot2S(self,ax1,m,n):
        sp=self.capPriceDif_date2S(m,n)
        plotx= ax1.plot(sp[0],sp[1],'c-',label="Size Weighted Cap Price Difference")
        plt.legend(loc='best')
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('Buy&Sell order price difference', color='k',fontsize=16)
        ax2 = ax1.twinx()
        ax2.plot(sp[0],sp[2],'b--')
        ax2.set_ylabel('traded price', color='b',fontsize=16)
#-------------------------------------------------------
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
                    if vol_1==0.0:
                        priceMA2= pri_1                        
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
                
#----class3-----
#----------a sub class for visualization------------

class capVisualWPrice(basic_visualize,tick_visualize):
    def __init__(self,collection,datex,width,height,fsize):
        self.collect=collection
        self.datex=datex
        self.width = width
        self.height=height
        self.fsize=fsize
    
    def visual(self,u0,k,filename):
        name=self.collect.name
        fig = plt.figure(1,figsize=(16, 12), dpi=240, facecolor='w', edgecolor='k')
        ax1 = fig.add_subplot(211)
        plt.sca(ax1)
        classBasic = basic_visualize(self.collect,self.datex,'buy1')
        classBasic.capPrice_plot2S(ax1,u0,u0+k)
        plt.title('SizeWeighted Order Price for Buy&Sell Cap and traded Price for '+str(name)+' on '+self.datex,
                  fontsize=20)
        #----
        ax = fig.add_subplot(212)
        classTick= tick_visualize(self.collect,self.datex,ax,self.width,self.height,self.fsize)
        plt.sca(ax)
        plt.axis('off')
        sqPatch = classTick.market_dateSP2(u0,k)
        for p in sqPatch:
            ax.add_patch(p)
        classTick.priceLine_dateSP2(u0,k,'b-')
        plt.title('visualization of partial '+str(name)+' tickData on '+datex+'with u0 = '+str(u0),fontsize=18)
        plt.show()
        fig.savefig(filename,format='pdf',dpi=1000)
        plt.close(fig) 
  
    def regCapWPrice(self,u0,k,a,b,c,dt):
        classBasic = basic_visualize(self.collect,self.datex,'buy1')
        #data
        # a=400,b=1200,c=500
        BSwp=classBasic.capPrice_date2S(u0,u0+k)
        buySell=BSwp[1:3]
        tPrice = BSwp[3]
        length= len(np.array(buySell).transpose())
        buySell_train= np.array(buySell).transpose()[a:b]
        buySell_test =  np.array(buySell).transpose()[length-(c+50):length-50]
        tPrice_train =  np.array(tPrice)[(a+dt):(b+dt)]
        tPrice_test  =   np.array(tPrice)[(length-(c+50)+dt):(length-50+dt)]
        
        #regr.predict(np.array(buySell).transpose()[-2])-np.array(tPrice)[-2]
        
        #buySell_train= buySell[:100]
        #buySell_test =  buySell[-(c+100):-(c)]
        #tPrice_train =  tPrice[a+dt:b+dt]
        #tPrice_test  =   tPrice[-(c+100):-(c)]        
        regr = linear_model.LinearRegression()
        regr.fit(buySell_train, tPrice_train)
        # The coefficients
        coeff= regr.coef_
        print 'Coefficients: \n=', coeff
        # The mean square error
        print("Residual sum of squares: %.5f"
              % np.mean((regr.predict(buySell_test) - tPrice_test) ** 2))
        # Explained variance score: 1 is perfect prediction
        rsq= regr.score(buySell_test,tPrice_test)
        print('Variance score: %.5f' % rsq)
        #plt.scatter(buySell_test,tPrice_test,color='black')
        fig = plt.figure(1,figsize=(16, 12), dpi=240, facecolor='w', edgecolor='k')
        plt.plot(tPrice_test,regr.predict(buySell_test) , 'bo',
                 linewidth=2)
        plt.xticks(())
        plt.yticks(())
        plt.show()
        return np.array([coeff,rsq])
        
        
#-----------test code-------------------

height= 0.05/2.5
datex = "2015-11-11"
u0 = 1000
fsize=10
k=100
width=1.0/k
collection= SZ002500
path = 'C:/Users/Jiansen/Documents/Finance/a_stock_intra/visualization/SZ002500'
filename = 'square_CapSeries_for SZ002500 tickVis_041406.pdf'
filename = os.path.join(path, filename)
capVisPClass= capVisualWPrice(collection,datex,width,height,fsize)
capVisPClass.visual(u0,k,filename)

#------multivariate regression-----------
a=0
b=300
c=100
rsqLis=pd.Series()
for dt in np.arange(0,21,1):
    rsq0=capVisPClass.regCapWPrice(u0,k,a,b,c,dt)[1]
    rsqLis=np.append(rsqLis,rsq0)
    dtLiss=np.arange(0,21,1)
# get the R^2 plot
plotfun(dtLis,rsqLis,"R Square","R Square for different dt","dt","R^2")