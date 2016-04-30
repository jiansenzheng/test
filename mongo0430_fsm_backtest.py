# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 16:46:54 2016

@author: Jiansen
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 23:24:09 2016

@author: Jiansen
"""
#%matplotlib inline
import random
import numpy as np
import pandas as pd
import datetime
import time
import timeit
import os
from bson.objectid import ObjectId
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn import linear_model
from sklearn.linear_model import Ridge
import statsmodels.api as sm

import pymongo as pm
from pymongo import MongoClient


#filename = 'C:/Users/Jiansen/spyder_pro/RL/cat_mouse/testPrice2_2015-12-18.csv'
#price=np.array(pd.read_csv(filename)).transpose()[1]
#returns= 100*np.diff(np.log(price))

client = MongoClient('localhost',27017)
print client
db= client.stocks

SZ002500=db.SZ002500

#global length


#test the connection
#print SZ002500.count()
#print SZ002500.find_one({"vol":{"$gt":200000}})
s1= SZ002500.find_one().get("time")
print s1



#-----------------
def moment():
    return time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time()))

#----------------some global functions---------------------
def plotfun(lis1,lis2,label0,title0,xlab0,ylab0):
    plt.plot(lis1,lis2,'ro-',label=label0)
    plt.title(title0)
    plt.xlabel(xlab0)
    plt.ylabel(ylab0)
    plt.show()

def selectPrice(p):
    n0= len(p)
    plateau= pd.Series()
    up= pd.Series()
    down= pd.Series()
    for i0 in np.arange(1,n0,1):
        if p[i0]==p[i0-1]:
            plateau = np.append(plateau,p[i0])
        elif p[i0]>p[i0-1]:
            up = np.append(up,100.0*np.log(p[i0]/p[i0-1]))
        elif p[i0]<p[i0-1]:
            down = np.append(down,100.0*np.log(p[i0]/p[i0-1]))
    return np.array([up,down,plateau])

#-----------------------------------
class base_fsm(object):
    def enter_state(self, obj):
        raise NotImplementedError() 
    def exec_state(self, obj, param): # add new DOF
        raise NotImplementedError() 
    def exit_state(self, obj):
        raise NotImplementedError()
 
class Trader(object):    
    def __init__(self):
        self.position=pd.Series(0)
        #self.invested = 0
        self.tradeP1 =pd.Series()
        self.direction=pd.Series()
        self.tradeP2 =pd.Series()
        self.PL= pd.Series()
        self.lastTradeP1=[]
        self.momentO=pd.Series()
        self.momentC=pd.Series()
        self.amount= 10.0
        self.cash = pd.Series(200000.0)
        self.security=pd.Series(500.0)
        
    def sell(self,label):
        global slip
        if i0>6:
            self.position=np.append(self.position,-1)
            if label==1:
                self.lastTradeP1=[-1,price[-1]]
                self.direction=np.append(self.direction,-1)
                self.tradeP1= np.append(self.tradeP1, price[-1]*(1.0-slip/2.0))
                self.momentO=np.append(self.momentO,sec)
                #self.cash=np.append(self.cash, self.cash[-1]-self.tradeP1[-1]*self.amount*100)
                #self.security=np.append(self.security,self.security[-1]-self.amount)
            else:
                pass
        else:
            self.position=np.append(self.position,0)      
            #print "not enough data"
        
    def buy(self,label):
        global slip
        if i0>6:
            self.position=np.append(self.position,1)
            if label == 1:
                self.lastTradeP1=[1,price[-1]]
                self.direction=np.append(self.direction,1)        
                self.tradeP1=np.append(self.tradeP1, price[-1]*(1.0+slip/2.0))
                self.momentO=np.append(self.momentO,sec)
                #self.cash=np.append(self.cash, self.cash[-1]-self.tradeP1[-1]*self.amount)
                #self.security=np.append(self.security,self.security[-1]-self.amount)
            else:
                pass
        else:
            self.position=np.append(self.position,0) 
            
    def empty(self,cut):
        global transact,slip,sec
        if i0>6:
            self.position=np.append(self.position,0)
            if cut==1:
                self.tradeP2=np.append(self.tradeP2,price[-1])
                pl= self.tradeP2[-1]-self.tradeP1[-1]
                pl=self.direction[-1]*pl-(transact+slip/2.0)*self.tradeP2[-1]
                pl = pl*self.amount*100.0
                self.momentC=np.append(self.momentC,sec)
                self.PL=np.append(self.PL, pl)
            else:
                pass
        else:
            pass
    
    def signal(self):
        pass    
    
    def asset(self):
        ix=0
        if ix!=0:
            plCumu=np.cumsum(self.PL)
            print "accumulate profit:\n ",plCumu,"\n"
            label0="profit"
            title0= "accumulate profit"
            plotfun(self.momentC,plCumu,label0,title0,"trading time","amount")
        else:
            pass
        
    def attach_fsm(self, state, fsm):
        self.fsm = fsm
        self.curr_state = state
                 
    def change_state(self, new_state, new_fsm,label):
        self.curr_state = new_state
        self.fsm.exit_state(self)
        self.fsm = new_fsm
        self.fsm.enter_state(self)
        self.fsm.exec_state(self,label)

    def keep_state(self,label):
        self.fsm.exec_state(self,label)    
    
              
class short_fsm(base_fsm):
    def enter_state(self, obj):
        pass
    def exec_state(self, obj,label):
        obj.sell(label)
        obj.signal()
    def exit_state(self, obj):
        pass
              
              
class long_fsm(base_fsm):
    def enter_state(self, obj):
        pass        
    def exec_state(self, obj,label):
        obj.buy(label)
        obj.signal()
    def exit_state(self, obj):
        pass        
        
class empty_fsm(base_fsm):
    def enter_state(self, obj):
        pass
    def exec_state(self, obj, cut):
        obj.empty(cut)
        if cut==1:
            obj.asset()
        else:
            pass
    def exit_state(self, obj):
        pass
        
class fsm_mgr(object):
    def __init__(self):
        self._fsms = {}
        self._fsms[0] = empty_fsm()
        self._fsms[1] = long_fsm()  
        self._fsms[2] = short_fsm()
    def get_fsm(self, state):
        return self._fsms[state]       
    def frame(self, objs, state):
        global i0,finalMoment,sl,tp
        for obj in objs:
            if i0<finalMoment:  #intraday setup
                if obj.curr_state==0:
                    if state == obj.curr_state:
                        obj.keep_state(0)
                    else: #open position
                        obj.change_state(state, self._fsms[state],1)
                else: # or remove the position if state changes #SL&TP
                    exposure= obj.direction[-1]*(price[-1]-obj.tradeP1[-1])
                    if exposure<-sl*obj.tradeP1[-1] or exposure> tp*obj.tradeP1[-1]:
                        obj.change_state(0, self._fsms[0], 1) #close position
                    else:
                        if state == obj.curr_state or state ==0: #modification
                            obj.keep_state(0)
                        else:  #close position #important codes
                            obj.change_state(0, self._fsms[0],1)
            else:
                if obj.curr_state!=0:
                    obj.change_state(0, self._fsms[0],1)
                else:
                    obj.keep_state(0)
#----------
#----------                    
class strategy(Trader): 
    def __init__(self):
        pass
    def simpleStra(self,doc0):
        global i0,price
        price=np.append(price,doc0.get("price"))
        pstate = 0
        if i0>6:
            if price[-1]>price[-4]>price[-7]:
                pstate= 1
            elif price[-1]<price[-4]<price[-7]:
                pstate = 2
            else:
                pstate = 0           
            return pstate
        else:
            return pstate
            
    def slopeStra(self,doc0,seriesLen,thres1,thres2):
        amp = lambda lis: sum(lis)/np.std(lis)
        global pos,price,indexP
        pstate = 0
        if (doc0.get("buy5")!=0) and (doc0.get("sale5")!=0):
            price = np.append(price,doc0.get("price"))
            indexP = np.append(indexP,sec)
            pos=pos+1
            if pos>=seriesLen+1:
                series= selectPrice(price[-seriesLen:])
                upA = amp(series[0])
                downA= amp(series[1])
                fall= abs(downA/upA)*(seriesLen-len(series[2]))/seriesLen
                #fallS= np.append(fallS, )
                #sec2 = np.append(sec2,doc0.get("sec"))             
                if fall>thres1: 
                    pstate= 2
                elif fall<thres2:
                    pstate = 1
                else:
                    pstate = 0            
                return pstate
            else:
                return pstate
        else:
            return pstate
# strategy based on HP filter_indicator and LOB information          
    def slopeStraHP(self,doc0,seriesLen,thres1,thres2,thres3,thres4):
        amp = lambda lis: sum(lis)/np.std(lis)
        global pos,price,indexP
        bcStr = ["bc1","bc2","bc3","bc4","bc5"]
        scStr = ["sc1","sc2","sc3","sc4","sc5"]
        #buyStr =  ["buy1","buy2","buy3","buy4","buy5"]
        #sellStr =  ["sale1","sale2","sale3","sale4","sale5"]
        buySize = map(doc0.get,bcStr)
        sellSize  = map(doc0.get,scStr)
        pstate = 0
        if (doc0.get("buy5")!=0) and (doc0.get("sale5")!=0):
            price = np.append(price,doc0.get("price"))
            indexP = np.append(indexP,sec)
            pos=pos+1
            if pos>=seriesLen+1:
                tradex = Trader()
                position = tradex.position.iloc[-1]
                priceX= price[-seriesLen:]
                cycle, trend = sm.tsa.filters.hpfilter(priceX,2) 
                series= selectPrice(trend)
                upA = amp(series[0])
                downA= amp(series[1])
                fall= np.sqrt(abs(downA/upA)*(seriesLen-len(series[2]))/seriesLen)
                #fallS= np.append(fallS, )
                #sec2 = np.append(sec2,doc0.get("sec"))
                if position==0:
                    if fall>thres1 and np.amax(sellSize)>1.0*np.amax(buySize): 
                        pstate= 2
                    elif fall<thres2 and np.amax(buySize)>1.0*np.amax(sellSize):
                        pstate = 1
                    else:
                        pstate = 0   
                elif position==1:
                    if np.sum(sellSize)>thres3*np.sum(buySize):
                        pstate =2
                    elif np.sum(sellSize)<thres4*np.sum(buySize):
                        pstate =1
                    else:
                        pstate =0
                else:
                    if np.sum(buySize)>thres3*np.sum(sellSize):
                        pstate =2
                    elif np.sum(buySize)<thres4*np.sum(sellSize):
                        pstate =1
                    else:
                        pstate =0
                return pstate
            else:
                return pstate
        else:
            return pstate         
            
#-----------------
    def slopeStraHPVV(self,doc0,seriesLen,thres1,thres2,thres3,thres4):
        amp = lambda lis: abs(sum(lis))/np.std(lis)
        global pos,price,indexP,volume,iniV
        buyStr =  ["buy1","buy2","buy3","buy4","buy5"]
        sellStr =  ["sale1","sale2","sale3","sale4","sale5"]
        buySize = map(doc0.get,buyStr)
        sellSize  = map(doc0.get,sellStr)
        pstate = 0
        if (doc0.get("buy5")!=0) and (doc0.get("sale5")!=0):
            price = np.append(price,doc0.get("price"))  
            if len(volume)==0:
                iniV= float(doc0.get("vol"))
                volume = np.append(volume,1.0)
            else:
                volume = np.append(volume,float(doc0.get("vol"))/iniV)
            indexP = np.append(indexP,sec)
            pos=pos+1
            if pos>=seriesLen+1:
                tradex = Trader()
                position = tradex.position.iloc[-1]
                priceX= price[-seriesLen:]
                volumeX= volume[-seriesLen:]
                cycle, trend = sm.tsa.filters.hpfilter(priceX,2) 
                volat = np.std(100*np.diff(np.log(trend)))
                series= selectPrice(trend)
                upA = np.log(amp(series[0]))
                downA= np.log(amp(series[1]))
                #fall= abs(downA/upA)*(seriesLen-len(series[2]))/seriesLen
                #fallS= np.append(fallS, )
                #sec2 = np.append(sec2,doc0.get("sec"))
                if position==0:
                    if np.mean(volumeX)> thres3 and volat>thres4: #filtering
                        if upA>thres1 and downA<thres2 and np.amax(sellSize)<1.0*np.amax(buySize):
                            pstate = 1
                        elif upA<thres2 and downA>thres1 and np.amax(sellSize)>1.0*np.amax(buySize):
                            pstate = 2
                        else:
                            pstate = 0
                    else:
                        pstate =0
                elif position==1:
                    if np.sum(sellSize)>0.8*np.sum(buySize):
                        pstate =2
                    elif np.sum(sellSize)<0.5*np.sum(buySize):
                        pstate =1
                    else:
                        pstate =0
                else:
                    if np.sum(buySize)>0.8*np.sum(sellSize):
                        pstate =2
                    elif np.sum(buySize)<0.5*np.sum(sellSize):
                        pstate =1
                    else:
                        pstate =0
                return pstate
            else:
                return pstate
        else:
            return pstate    
            
    def slopeStraHPVV2(self,doc0,seriesLen,thres1,thres2,thres3,thres4):
        amp = lambda lis: abs(sum(lis))/np.std(lis)
        global pos,price,indexP,volume,iniV,sec
        bcStr = ["bc1","bc2","bc3","bc4","bc5"]
        scStr = ["sc1","sc2","sc3","sc4","sc5"]
        pstate = 0
        if (doc0.get("buy5")!=0) and (doc0.get("sale5")!=0):
            buySize = map(doc0.get,bcStr)
            sellSize  = map(doc0.get,scStr)
            price = np.append(price,doc0.get("price"))  
            if len(volume)==0:
                if doc0.get("vol")!=0:
                    iniV= float(doc0.get("vol"))
                    volume = np.append(volume,1.0)
                else:
                    pass
            else:
                if doc0.get("vol")!=0:
                    volume = np.append(volume,float(doc0.get("vol"))/iniV)
                else:
                    pass
            indexP = np.append(indexP,sec)
            pos=pos+1
            if pos>=seriesLen+1:
                #tradex = Trader()
                #position = tradex.position.iloc[-1]
                priceX= price[-seriesLen:]
                if len(volume)>seriesLen:
                    volumeX= volume[-seriesLen:]
                    cycle, trend = sm.tsa.filters.hpfilter(priceX,2) 
                    volat = np.std(100*np.diff(np.log(trend)))
                    series= selectPrice(trend)
                    upA = np.log(amp(series[0]))
                    downA= np.log(amp(series[1]))
                    #fall= abs(downA/upA)*(seriesLen-len(series[2]))/seriesLen
                    #fallS= np.append(fallS, )
                    #sec2 = np.append(sec2,doc0.get("sec"))
                    if np.mean(volumeX)> thres3 and volat>thres4: #filtering
                        if upA>thres1 and downA<thres2 and np.amax(sellSize)<1.0*np.amax(buySize):
                            pstate = 1
                        elif upA<thres2 and downA>thres1 and np.amax(sellSize)>1.0*np.amax(buySize):
                            pstate = 2
                        else:
                            pstate = 0
                    else:
                        pstate =0
                    return pstate
                else:
                    return pstate
            else:
                return pstate
        else:
            return pstate             
###----       
    def slopeStraHPVV3(self,doc0,seriesLen,thres1,thres2,thres3,thres4):
        amp = lambda lis: abs(sum(lis))/np.std(lis)
        global pos,price,indexP,volume,iniV,sec
        buyStr =  ["buy1","buy2","buy3","buy4","buy5"]
        sellStr =  ["sale1","sale2","sale3","sale4","sale5"]
        buySize = map(doc0.get,buyStr)
        sellSize  = map(doc0.get,sellStr)
        pstate = 0
        if (doc0.get("buy5")!=0) and (doc0.get("sale5")!=0):
            price = np.append(price,doc0.get("price"))  
            if len(volume)==0:
                if doc0.get("vol")!=0:
                    iniV= float(doc0.get("vol"))
                    volume = np.append(volume,1.0)
                else:
                    pass
            else:
                if doc0.get("vol")!=0:
                    volume = np.append(volume,float(doc0.get("vol"))/iniV)
                else:
                    pass
            indexP = np.append(indexP,sec)
            pos=pos+1
            if pos>=seriesLen+1:
                #tradex = Trader()
                #position = tradex.position.iloc[-1]
                priceX= price[-seriesLen:]
                if len(volume)>seriesLen:
                    volumeX= volume[-seriesLen:]
                    cycle, trend = sm.tsa.filters.hpfilter(priceX,2) 
                    volat = np.std(100*np.diff(np.log(trend)))
                    series= selectPrice(trend)
                    upA = np.log(amp(series[0]))
                    downA= np.log(amp(series[1]))
                    fall= np.sqrt(abs(downA/upA)*(seriesLen-len(series[2]))/seriesLen)
                    #fallS= np.append(fallS, )
                    #sec2 = np.append(sec2,doc0.get("sec"))
                    if np.mean(volumeX)> thres3 and volat>thres4: #filtering
                        if upA>thres1 and downA<thres2 and np.amax(sellSize)<1.0*np.amax(buySize):
                            pstate = 1
                        elif upA<thres2 and downA>thres1 and np.amax(sellSize)>1.0*np.amax(buySize):
                            pstate = 2
                        else:
                            pstate = 0
                    else:
                        pstate =0
                    return pstate
                else:
                    return pstate
            else:
                return pstate
        else:
            return pstate          
            
    def slopeStraHPM(self,doc0,seriesLen,k,thres1,thres2,thres3,thres4):
        amp = lambda lis: sum(lis)/np.std(lis)
        global pos,price,indexP,fallS
        bcStr = ["bc1","bc2","bc3","bc4","bc5"]
        scStr = ["sc1","sc2","sc3","sc4","sc5"]
        #buyStr =  ["buy1","buy2","buy3","buy4","buy5"]
        #sellStr =  ["sale1","sale2","sale3","sale4","sale5"]
        pstate = 0
        if (doc0.get("buy5")!=0) and (doc0.get("sale5")!=0):
            price = np.append(price,doc0.get("price"))
            buySize = map(doc0.get,bcStr)
            sellSize  = map(doc0.get,scStr)
            indexP = np.append(indexP,sec)
            pos=pos+1
            if pos>=seriesLen+1:
                tradex = Trader()
                position = tradex.position.iloc[-1]
                priceX= price[-seriesLen:]
                cycle, trend = sm.tsa.filters.hpfilter(priceX,2) 
                series= selectPrice(trend)
                upA = amp(series[0])
                downA= amp(series[1])
                fall0= np.sqrt(abs(downA/upA)*(seriesLen-len(series[2]))/seriesLen)
                fallS= np.append(fallS, fall0)
                if len(fallS)>k:
                    fall= np.mean(fallS[-k:])
                #fallS= np.append(fallS, )
                #sec2 = np.append(sec2,doc0.get("sec"))
                    if position==0:
                        if fall>thres1 and sum(buySize[0:2])>sum(sellSize):
                        #if fall>thres1 and np.amax(buySize)>1.0*np.amax(sellSize): 
                            pstate= 1
                        elif fall<thres2 and sum(sellSize[0:2])>sum(buySize):
                            #fall<thres2 and np.amax(sellSize)>1.0*np.amax(buySize):
                            pstate = 2
                        else:
                            pstate = 0   
                    elif position==1:
                        #if fall<thres1 and np.sum(sellSize)>thres3*np.sum(buySize):
                        if fall<thres1 and np.sum(sellSize[0:4])>thres3*np.sum(buySize[0:2]):
                            pstate =2
                        elif fall>thres1 and thres4*sum(buySize[0:2])>sum(sellSize[0:4]):
                        #elif fall>thres1 and np.sum(sellSize)<thres4*np.sum(buySize):    
                            pstate =1
                        else:
                            pstate =0
                    else:
                        if fall<thres2 and np.sum(buySize[0:4])<thres3*np.sum(sellSize[0:2]):
                        #if fall<thres2 and np.sum(buySize)<thres3*np.sum(sellSize):
                            pstate =2
                        elif fall>thres2 and np.sum(buySize[0:4])>thres4*np.sum(sellSize[0:2]):
                        #elif fall>thres2 and np.sum(buySize)>thres4*np.sum(sellSize):
                            pstate =1
                        else:
                            pstate =0
                    return pstate
                else:
                    return pstate
            else:
                return pstate
        else:
            return pstate         
        
    def slopeStraHPMLo(self,doc0,seriesLen,k,thres1,thres2,thres3,thres4):
        amp = lambda lis: sum(lis)/np.std(lis)
        global pos,price,indexP,fallS
        bcStr = ["bc1","bc2","bc3","bc4","bc5"]
        scStr = ["sc1","sc2","sc3","sc4","sc5"]
        #buyStr =  ["buy1","buy2","buy3","buy4","buy5"]
        #sellStr =  ["sale1","sale2","sale3","sale4","sale5"]
        pstate = 0
        if (doc0.get("buy5")!=0) and (doc0.get("sale5")!=0):
            price = np.append(price,doc0.get("price"))
            buySize = map(doc0.get,bcStr)
            sellSize  = map(doc0.get,scStr)
            indexP = np.append(indexP,sec)
            pos=pos+1
            if pos>=seriesLen+1:
                tradex = Trader()
                position = tradex.position.iloc[-1]
                priceX= price[-seriesLen:]
                cycle, trend = sm.tsa.filters.hpfilter(priceX,2) 
                series= selectPrice(trend)
                upA = amp(series[0])
                downA= amp(series[1])
                fall0= np.sqrt(abs(downA/upA)*(seriesLen-len(series[2]))/seriesLen)
                fallS= np.append(fallS, fall0)
                
                if len(fallS)>k:
                    fall= np.mean(fallS[-k:])
                    down= np.amax(trend)-price[-1]
                #fallS= np.append(fallS, )
                #sec2 = np.append(sec2,doc0.get("sec"))
                    if position==0:
                        if fall>thres1 and sum(buySize[0:2])>sum(sellSize) and down>0.03:
                        #if fall>thres1 and sum(buySize[0:2])>sum(sellSize):
                        #if fall>thres1 and np.amax(buySize)>1.0*np.amax(sellSize): 
                            pstate= 1
                        else:
                            pstate = 0   
                    else:
                        #if fall<thres1 and np.sum(sellSize)>thres3*np.sum(buySize):
                        if fall<thres1 and np.sum(sellSize[0:4])>thres3*np.sum(buySize):
                            pstate =2
                        elif fall>thres1 and thres4*sum(buySize[0:2])>sum(sellSize[0:4]):
                        #elif fall>thres1 and np.sum(sellSize)<thres4*np.sum(buySize):    
                            pstate =1
                        else:
                            pstate =0
                    return pstate
                else:
                    return pstate
            else:
                return pstate
        else:
            return pstate         
        
    def slopeStraHPMLoM(self,doc0,seriesLen,k,thres1,thres2,thres3,thres4):
        amp = lambda lis: sum(lis)/np.std(lis)
        global pos,price,indexP,fallS
        bcStr = ["bc1","bc2","bc3","bc4","bc5"]
        scStr = ["sc1","sc2","sc3","sc4","sc5"]
        #buyStr =  ["buy1","buy2","buy3","buy4","buy5"]
        #sellStr =  ["sale1","sale2","sale3","sale4","sale5"]
        pstate = 0
        if (doc0.get("buy5")!=0) and (doc0.get("sale5")!=0):
            price = np.append(price,doc0.get("price"))
            buySize = map(doc0.get,bcStr)
            sellSize  = map(doc0.get,scStr)
            indexP = np.append(indexP,sec)
            pos=pos+1
            if pos>=seriesLen+1:
                tradex = Trader()
                position = tradex.position.iloc[-1]
                priceX= price[-seriesLen:]
                cycle, trend = sm.tsa.filters.hpfilter(priceX,2) 
                series= selectPrice(trend)
                upA = amp(series[0])
                downA= amp(series[1])
                fall0= np.sqrt(abs(downA/upA)*(seriesLen-len(series[2]))/seriesLen)
                fallS= np.append(fallS, fall0)
                
                if len(fallS)>k:
                    fall= np.mean(fallS[-k:])
                    down= np.amax(trend)-price[-1]
                    if position==0:
                        if fall>thres1 and sum(buySize[0:2])>sum(sellSize) and down>thres3:
                            pstate= 1
                        else:
                            pstate = 0   
                    else:
                        if fall<thres2 and sum(sellSize[0:2])>sum(buySize):
                            pstate =2
                        elif fall>thres2 and sum(buySize[0:2])>sum(sellSize):
                            pstate =1
                        else:
                            pstate =0
                    return pstate
                else:
                    return pstate
            else:
                return pstate
        else:
            return pstate         
        
            
#----------
#self.thres1=2.5   #thresholds for strategies
#self.thres2=2.4
#self.thres3=1.0
#self.thres4=0.06
#----------
 
class World(strategy):  #inherit from strategy class
    def init(self,datex,stra,collection,thres1,thres2,thres3,thres4):
        self._traders = []
        self._fsm_mgr = fsm_mgr()
        self.__init_trader()
        self.datex = datex
        self.stra=stra
        self.collection=collection
        global i0,N0,pos,transact,price,indexP,volume,slip,sec,fallS
        global finalMoment,sl,tp
        N0=200
        i0=0
        pos=0
        transact=2.6/1000.0
        slip= 2.46/1000 #fixing slippage
        price=pd.Series()  #moving price series
        volume=pd.Series()
        indexP=pd.Series()
        fallS=pd.Series()
        #seconds=pd.Series()
        finalMoment=3900  #close position and stop opening after this moment
        sl=0.0050      #StopLoss Ratio
        tp=0.0025      #TakeProfit Ratio  
        self.thres1=thres1   #thresholds for strategies
        self.thres2=thres2
        self.thres3=thres3
        self.thres4=thres4
        
    def __init_trader(self):
        for i in xrange(1):   
            tmp = Trader()
            tmp.attach_fsm(0, self._fsm_mgr.get_fsm(0))
            self._traders.append(tmp) 
    def __frame(self,doc,pool):
        #global thres1,thres2
        if   pool==1:
            self._fsm_mgr.frame(self._traders, strategy().simpleStra(doc))
        elif pool==2:
            self._fsm_mgr.frame(self._traders, strategy().slopeStra(doc,36,self.thres1,self.thres2))
        elif pool==3:
            self._fsm_mgr.frame(self._traders, strategy().slopeStraHP(doc,36,self.thres1,self.thres2,
                                                                     self.thres3,self.thres4))
        elif pool==4:
            self._fsm_mgr.frame(self._traders, strategy().slopeStraHPVV(doc,36,self.thres1,self.thres2,
                                self.thres3,self.thres4))            
        elif pool==5:
            self._fsm_mgr.frame(self._traders, strategy().slopeStraHPVV2(doc,36,
                                self.thres1,self.thres2,self.thres3,self.thres4))   
        elif pool==6:
            self._fsm_mgr.frame(self._traders, strategy().slopeStraHPM(doc,24,12,
                                self.thres1,self.thres2,self.thres3,self.thres4)) 
        elif pool==7:
            self._fsm_mgr.frame(self._traders, strategy().slopeStraHPMLo(doc,24,12,
                                self.thres1,self.thres2,self.thres3,self.thres4)) 
        elif pool==8:
            self._fsm_mgr.frame(self._traders, strategy().slopeStraHPMLoM(doc,24,12,
                                self.thres1,self.thres2,self.thres3,self.thres4)) 
        else:
            self._fsm_mgr.frame(self._traders, strategy().simpleStra(doc))
    def run(self):
        global i0,length,sec
        length = self.collection.find({"date":self.datex}).count()
        #while i0<2000:        
        while i0<length-1-N0-1:
            doc0 = self.collection.find({"date":self.datex})[N0+i0]
            sec = doc0.get("sec")
            self.__frame(doc0,self.stra)  #choose one strategy for backtest
            #time.sleep(0.01)
            #print "current seconds= ", doc0.get("sec"),"\n"
            i0=i0+1
#----------------------backtest analysis---------------------
    def addPL(self):
        global totalPL
        if len(self._traders)==1:
            tradex=self._traders[0]
            totalPL= np.append(totalPL,tradex.PL)
            plCumu=np.cumsum(tradex.PL)
            print "accumulate PL for this day :\n ",plCumu,"\n"
            
    def addPLx(self):
        global totalPL
        if len(self._traders)==1:
            tradex=self._traders[0]
            totalPL= np.append(totalPL,tradex.PL)   
            
    def performance(self):
        global path
        if len(self._traders)==1:
            tradex=self._traders[0]
            plCumu=np.cumsum(tradex.PL)
            print "accumulate profit  :\n ",plCumu,"\n"
            label0="profit or loss"
            title0= "accumulate P&L"+self.datex
            fig=plt.figure(1,figsize=(16, 12), dpi=4800, facecolor='w', edgecolor='k')
            plt.plot(tradex.momentC,plCumu,'go-',label=label0)
            plt.title(title0)
            plt.legend(loc='best')
            plt.xlabel("trading time")
            plt.ylabel("amount")
            plt.show()
            thres1 =  self.thres1
            thres2 = self.thres2
            thres3 =  self.thres3
            filename = 'accPL_HPLOB_SZ002500'+self.datex+'_s1_'+str(thres1)+'_s2_'+str(thres2)+'_s3_'+str(thres3)+'.pdf'
            filename = os.path.join(path, filename)
            fig.savefig(filename)
            plt.close(fig)
            
    def tradeVisual(self):
        global price,indexP
        global path
        if len(self._traders)==1:
            tradex=self._traders[0]
            openTime= tradex.momentO
            openPrice = tradex.tradeP1
            closeTime = tradex.momentC
            closePrice= tradex.tradeP2
            longShort = tradex.direction
            print "position signal: ",tradex.position
            print "length of open: ", len(openPrice), "\n"
            print "length of close: ", len(closePrice) , "\n"
            fig1=plt.figure(1,figsize=(16, 12), dpi=4800, facecolor='w', edgecolor='k')
            ax1 = fig1.add_subplot(111)
            label0="Price"
            ax1.plot(indexP,price,'b-',label=label0)
            ax1.plot(openTime,openPrice,'g^')
            ax1.plot(closeTime,closePrice,'rs')
            ax1.set_xlabel('time(#slice)')
            ax1.set_ylabel('trading price', color='r',fontsize=16)
            ax2 = ax1.twinx()
            ax2.plot(openTime,longShort,'co')
            for ls in np.arange(0,len(openTime),1):
                ax2.plot((openTime[ls], openTime[ls]), (-0.97, 0.97), 'c--')
            ax2.set_ylabel('trading direction', color='c',fontsize=16)
            ax2.set_ylim([-2.0,2.0])
            title1= "Open and Close Position(long and short) SZ002500"+self.datex
            plt.title(title1,fontsize=20)
            plt.grid(True)
            plt.legend(loc='best')
            thres1 =  self.thres1
            thres2 = self.thres2
            thres3 =  self.thres3
            plt.show()
            filename1 = 'vis_HPLOB_SZ002500'+self.datex+'_s1_'+str(thres1)+'_s2_'+str(thres2)+'_s3_'+str(thres3)+'.pdf'
            filename1 = os.path.join(path, filename1)
            fig1.savefig(filename1,format='pdf',dpi=4800)
            plt.close(fig1)

    def tradeHist(self):
        global path
        if len(self._traders)==1: 
            fig2=plt.figure(num=None, figsize=(16,12), dpi=4800, facecolor='w', edgecolor='k')
            tradex=self._traders[0]
            pl=tradex.PL
            win=0   #compuate winratio and profit ratio
            profit=pd.Series()
            loss= pd.Series()
            PLRatio= 0
            if len(pl)>0:
                for plx in pl:
                    if plx>0:
                        win=win+1
                        profit=np.append(profit,plx)
                    else:
                        loss=np.append(loss,plx)
                winratio=win/len(pl)
                print "win ratio is ",winratio,"\n"
                if len(loss)>0:
                    if len(profit)>0:
                        PLRatio= sum(profit)/sum(loss)
                        print "profit ratio is ",PLRatio,"\n"
                    else:
                        print "total loss is ",sum(loss),"\n"
                else:
                    print "total profit is ",sum(profit),"\n"  
                if len(pl)>1:
                    thres1 =  self.thres1
                    thres2 =  self.thres2
                    thres3 =  self.thres3
                    #label0= "parameters_th1_{0}_th2_{1}_th3_{2}_th4_0.05".format(thres1,thres2,thres3)
                    plt.hist(pl,facecolor='g')
                    plt.title('histogram of Profit&Loss'+" on "+ self.datex,fontsize=20)
                    plt.xlabel("P&L Value")
                    plt.ylabel("Frequency")
                    plt.legend(loc='best')
                    plt.grid(True)
                    plt.show()
                    filename2 = 'hist_HPLOB_SZ002500'+self.datex+'_s1_'+str(thres1)+'_s2_'+str(thres2)+'_s3_'+str(thres3)+'.pdf'
                    filename2 = os.path.join(path, filename2)
                    fig2.savefig(filename2,format='pdf',dpi=4800)
                    plt.close(fig2)
            else:
                print "no trading today! \n"
        else:
            pass      
#----------   
#datex =  "2015-12-01"
collection=SZ002500
dateCol= ["2015-11-24","2015-11-25","2015-11-26","2015-11-27","2015-11-30",
          "2015-12-01","2015-12-02","2015-12-03","2015-12-04","2015-12-07",
          "2015-12-08","2015-12-09","2015-12-10"]
dfPL=pd.DataFrame([],columns=["thre1","thre2","thre3","thre4","totalPL"])
path = 'C:/Users/Jiansen/Documents/Finance/a_stock_intra/visualization/Backtest/SZ002500/trials/test0430_SZ002500_04'

if __name__ == "__main__":
    for thres1 in np.arange(2.3,2.7,0.4):
        for thres2 in np.arange(1.5,1.9,0.4):
            for thres3 in np.arange(0.8,1.2,0.4):
                for thres4 in np.arange(0.4,0.8,0.4):
                    global totalPL
                    totalPL=pd.Series()
                    for date in dateCol:
                        start=timeit.default_timer()
                        world = World()
                        #world.init(date,3,collection,thres1,thres2,thres3,0.05)
                        world.init(date,8,collection,thres1,thres2,thres3,thres4)
                        world.run()
                        world.addPLx()
                        world.performance()
                        world.tradeVisual()
                        world.tradeHist()
                        stop=timeit.default_timer()
                        print "date: ", date,"\n"
                        print "total time consumed", stop-start
                    plx= sum(totalPL)
                    print "total PL", plx,"\n"
                    print "current time",moment()
                    df0= pd.DataFrame([[thres1,thres2,thres3,thres4,plx]],columns=["thre1","thre2","thre3","thre4","totalPL"])
                    dfPL=dfPL.append(df0,ignore_index=True)
                    print "parameters_th1{0}_th2{1}_th3{2}_th4{3}".format(thres1,thres2,thres3,thres4)
                    if len(totalPL)>3:
                        print "sharpe ratio for PL ", np.mean(totalPL)/np.std(totalPL)
                        label0= "parameters_th1_{0}_th2_{1}_th3_{2}_th4{3}".format(thres1,thres2,thres3,thres4)
                        title0= "accumulate P&L for SZ002500"
                        fig=plt.figure(1,figsize=(16, 12), dpi=4800, facecolor='w', edgecolor='k')
                        plt.plot(np.arange(0,len(totalPL),1),totalPL,'ro-',label=label0)
                        plt.legend(loc='best')
                        plt.title(title0)
                        plt.xlabel("trading time")
                        plt.ylabel("total PL amount")
                        plt.show()
                        filename="test"+"thres1_"+str(thres1)+"thres2_"+str(thres2)+"thres3_"+str(thres3)
                        filename=filename+"thres4_"+str(thres4)+".pdf"
                        filename=os.path.join(path, filename)
                        fig.savefig(filename)
                        plt.close(fig)   
    path0=path                
    filename0="PL_backtest_SZ002500_043004.csv"
    filename0=os.path.join(path0, filename0)
    dfPL.to_csv(filename0,index=False)   