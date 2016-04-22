# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 10:12:05 2016

@author: Jiansen
"""

import time
import random
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
global collection
collection=SZ002500

#test the connection
print SZ002500.count()
print SZ002500.find_one({"vol":{"$gt":200000}})
s1= SZ002500.find_one()
time0=s1.get("time")
print time0

global i0,N0,pos
N0=200
i0=0
pos=0


global price
price=pd.Series()
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
# print price[29]<price[29-3]<price[29-6]
class base_fsm(object):
    def enter_state(self, obj):
        raise NotImplementedError() 
    def exec_state(self, obj, param): # add new DOF
        raise NotImplementedError() 
    def exit_state(self, obj):
        raise NotImplementedError()
 
class Trader(object):    
    def __init__(self):
        self.position=pd.Series()
        self.invested = 0
        self.tradeP1 =pd.Series()
        self.tradeP2 =pd.Series()
        self.PL= pd.Series()
        self.lastTradeP1=[]
        self.moment=pd.Series()
        
    def sell(self,label):
        print "sell!!!"
        if i0>6:
            s1=np.arange(i0-6,i0+1,1)
            plt.plot(s1,price[s1],'g-',linewidth=2.5)
            plt.show()
            self.position=np.append(self.position,-1)
            if label==1:
                self.lastTradeP1=[-1,price[i0]]
                self.tradeP1=np.append(self.tradeP1,self.lastTradeP1)
            else:
                pass
        else:
            self.position=np.append(self.position,0)      
            print "not enough data"
        
    def buy(self,label):
        print "buy!!!"
        if i0>6:        
            s1=np.arange(i0-6,i0+1,1)
            plt.plot(s1,price[s1],'r-',linewidth=2.5)
            plt.show()
            self.position=np.append(self.position,1)
            if label ==1:
                self.lastTradeP1=[1,price[i0]]
                self.tradeP1=np.append(self.tradeP1,self.lastTradeP1)
            else:
                pass
        else:
            self.position=np.append(self.position,0)    
            print "not enough data"
            
    def empty(self,cut):
        print "empty!!!"
        if i0>7:        
            s1=np.arange(i0-7,i0,1)
            plt.plot(s1,returns[s1],'c-',linewidth=3.0)
            plt.show()
            self.position=np.append(self.position,0)
            if cut==1:
                self.tradeP2=np.append(self.tradeP2,price[i0])
                pl=self.lastTradeP1[0]*(price[i0]-self.lastTradeP1[1])
                self.moment=np.append(self.moment,i0)
                self.PL=np.append(self.PL,pl)
            else:
                pass
        else:
            print "not enough data"
    
    def signal(self):
        print "signal=,",self.position
    
    def asset(self):
        plCumu=np.cumsum(self.PL)
        print "accumulate profit:\n ",plCumu,"\n"
        label0="profit"
        title0= "accumulate profit"
        plotfun(self.moment,plCumu,label0,title0,"trading time","amount")
      
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
        print "Trader%s enter price down state!"%(id(obj))
    def exec_state(self, obj,label):
        print "Trader%s in price down state!"%(id(obj))
        obj.sell(label)
        obj.signal()
    def exit_state(self, obj):
        print "Trader%s exit price down state!"%(id(obj))
              
              
class long_fsm(base_fsm):
    def enter_state(self, obj):
        print "Trader%s enter price up state!"%(id(obj))
    def exec_state(self, obj,label):
        print "Trader%s in price up state!"%(id(obj))
        obj.buy(label)
        obj.signal()
    def exit_state(self, obj):
        print "Trader%s exit price up state!"%(id(obj))
        
class empty_fsm(base_fsm):
    #def __init__(self,cut):
     #   self.cut=cut
    def enter_state(self, obj):
        print "Trader%s enter empty state!"%(id(obj))
    def exec_state(self, obj, cut):
        print "Trader%s in empty state!"%(id(obj))
        obj.empty(cut)
        obj.signal()
        if cut==1:
            obj.asset()
    def exit_state(self, obj):
        print "Trader%s exit empty state!"%(id(obj))
        
class fsm_mgr(object):
    def __init__(self):
        self._fsms = {}
        self._fsms[0] = empty_fsm()
        self._fsms[1] = long_fsm()  
        self._fsms[2] = short_fsm()
    def get_fsm(self, state):
        return self._fsms[state]       
    def frame(self, objs, state):
        for obj in objs:
            if obj.curr_state==0:
                if state == obj.curr_state:
                    obj.keep_state(0)
                else: #open position
                    obj.change_state(state, self._fsms[state],1)
            else: # or remove the position if state changes
                if state == obj.curr_state:
                    obj.keep_state(0)
                else:  #close position
                    obj.change_state(state, self._fsms[0],1)
#----------
#----------                    
class strategy():  
    def simpleStra(self,doc0):
        global i0,price
        price=np.append(price,doc0.get("price"))
        pstate = 0
        if i0>6:
            if price[i0]>price[i0-3]>price[i0-6]:
                pstate= 1
                print "price up!"
            elif price[i0]<price[i0-3]<price[i0-6]:
                pstate = 2
                print "price down!"
            else:
                pstate = 0
                print "nosiy price!"            
            return pstate
        else:
            print "not enough data"
            return pstate
            
    def slopeStra(self,doc0,seriesLen,thres1,thres2):
        amp = lambda lis: sum(lis)/np.std(lis)
        global pos,price
        pstate = 0
        if (doc0.get("buy5")!=0) and (doc0.get("sale5")!=0):
            price = np.append(price,doc0.get("price"))
            if pos>=seriesLen:
                series= selectPrice(price[-seriesLen:])
                upA = amp(series[0])
                downA= amp(series[1])
                fall= abs(downA/upA)*(seriesLen-len(series[2]))/seriesLen
                #fallS= np.append(fallS, )
                #sec2 = np.append(sec2,doc0.get("sec"))             
                if fall>thres1: 
                    pstate= 1
                    print "price will go up!"
                elif fall<thres2:
                    pstate = 2
                    print "price will go down!"
                else:
                    pstate = 0
                    print "nosiy price!"            
                return pstate
            pos=pos+1
            print "not enough data for slopeStra-IN"
            return pstate
        else:
            print "not data for slopeStra"
            return pstate
#----------
#----------
 
class World(strategy):  #inherit from strategy class
    def init(self,datex):
        self._traders = []
        self._fsm_mgr = fsm_mgr()
        self.__init_trader()
        self.datex = datex
    def __init_trader(self):
        for i in xrange(1):   
            tmp = Trader()
            tmp.attach_fsm(0, self._fsm_mgr.get_fsm(0))
            self._traders.append(tmp) 
    def __frame(self,doc,pool):
        if pool==1:
            self._fsm_mgr.frame(self._traders, strategy().simpleStra(doc))
        elif pool==2:
            global thres1,thres2
            self._fsm_mgr.frame(self._traders, strategy().slopeStra(doc,36,thres1,thres2))
        else:
            self._fsm_mgr.frame(self._traders, strategy().simpleStra(doc))
    def run(self):
        global i0
        while True:
            global collection
            doc0= collection.find({"date":self.datex})[N0+i0]
            self.__frame(doc0,2)  #choose the second strategy
            time.sleep(0.5)
            print "i0=",i0
            print "pos=",pos
            i0=i0+1
       
datex =  "2015-12-18"
thres1=6.0
thres2=1.0 
      
if __name__ == "__main__":
    world = World()
    world.init(datex)
    world.run()