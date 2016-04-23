# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 21:28:39 2016

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
#global length


#test the connection
print SZ002500.count()
print SZ002500.find_one({"vol":{"$gt":200000}})
s1= SZ002500.find_one()
time0=s1.get("time")
print time0

global i0,N0,pos,transact,price
N0=200
i0=0
pos=0
transact=2.6/1000.0
slip= 2.46/1000 #fixing slippage
price=pd.Series()  #moving price series
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
        #print "sell_order!!!"
        global slip
        if i0>6:
            self.position=np.append(self.position,-1)
            if label==1:
                self.lastTradeP1=[-1,price[-1]]
                self.direction=np.append(self.direction,-1)
                self.tradeP1= np.append(self.tradeP1, price[-1]*(1.0-slip/2.0))
                self.momentO=np.append(self.momentO,i0)
                self.cash=np.append(self.cash, self.cash.iloc[-1]-self.tradeP1[-1]*self.amount)
                #self.cash=np.append(self.cash, self.cash.iloc[-1]-self.amount)    
                self.security=np.append(self.security,self.security.iloc[-1]-self.amount)
                print "open short position!\n"
            else:
                pass
        else:
            self.position=np.append(self.position,0)      
            print "not enough data"
        
    def buy(self,label):
        #print "buy_order!!!"
        global slip
        if i0>6:
            self.position=np.append(self.position,1)
            if label == 1:
                self.lastTradeP1=[1,price[-1]]
                self.direction=np.append(self.direction,1)        
                self.tradeP1=np.append(self.tradeP1, price[-1]*(1.0+slip/2.0))
                self.momentO=np.append(self.momentO,i0)
                self.cash=np.append(self.cash, self.cash.iloc[-1]-self.tradeP1[-1]*self.amount)
                self.security=np.append(self.security,self.security.iloc[-1]-self.amount)
                print "open long position!\n"
            else:
                pass
        else:
            self.position=np.append(self.position,0)    
            print "not enough data"
            
    def empty(self,cut):
        print "empty!!!"
        global transact,slip
        if i0>6:
            self.position=np.append(self.position,0)
            if cut==1:
                self.tradeP2=np.append(self.tradeP2,price[-1])
                pl=self.direction[-1]*(price[-1]-self.tradeP1[-1])-(transact+slip/2.0)*price[-1]
                pl = pl*self.amount*100.0
                self.momentC=np.append(self.momentC,i0)
                self.PL=np.append(self.PL, pl)
                print "close_order!!\n"
            else:
                pass
        else:
            print "not enough data\n"
    
    def signal(self):
        print "signal=,",self.position
    
    def asset(self):
        plCumu=np.cumsum(self.PL)
        print "accumulate profit:\n ",plCumu,"\n"
        label0="profit"
        title0= "accumulate profit"
        plotfun(self.momentC,plCumu,label0,title0,"trading time","amount")
      
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
        print "Trader%s enter short position state!\n"%(id(obj))
    def exec_state(self, obj,label):
        print "Trader%s in short position state!\n"%(id(obj))
        obj.sell(label)
        obj.signal()
    def exit_state(self, obj):
        print "Trader%s exit short position state!\n"%(id(obj))
              
              
class long_fsm(base_fsm):
    def enter_state(self, obj):
        print "Trader%s enter long position state!\n"%(id(obj))
    def exec_state(self, obj,label):
        print "Trader%s in long position state!\n"%(id(obj))
        obj.buy(label)
        obj.signal()
    def exit_state(self, obj):
        print "Trader%s exit long position state!\n"%(id(obj))
        
class empty_fsm(base_fsm):
    #def __init__(self,cut):
     #   self.cut=cut
    def enter_state(self, obj):
        print "Trader%s enter empty position state!\n"%(id(obj))
    def exec_state(self, obj, cut):
        print "Trader%s in empty position state!\n"%(id(obj))
        obj.empty(cut)
        #obj.signal()
        if cut==1:
            print "show asset curve: \n"
            obj.asset()
        else:
            pass
    def exit_state(self, obj):
        print "Trader%s exit empty position state!\n"%(id(obj))
        
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
                if state == obj.curr_state or state ==0: #modification
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
            if price[-1]>price[-4]>price[-7]:
                pstate= 1
                print "price up! \n"
            elif price[-1]<price[-4]<price[-7]:
                pstate = 2
                print "price down!\n"
            else:
                pstate = 0
                print "nosiy price!\n"            
            return pstate
        else:
            print "not enough data\n"
            return pstate
            
    def slopeStra(self,doc0,seriesLen,thres1,thres2):
        amp = lambda lis: sum(lis)/np.std(lis)
        global pos,price
        pstate = 0
        if (doc0.get("buy5")!=0) and (doc0.get("sale5")!=0):
            price = np.append(price,doc0.get("price"))
            pos=pos+1
            if pos>=seriesLen+1:
                series= selectPrice(price[-seriesLen:])
                upA = amp(series[0])
                downA= amp(series[1])
                fall= abs(downA/upA)*(seriesLen-len(series[2]))/seriesLen
                #fallS= np.append(fallS, )
                #sec2 = np.append(sec2,doc0.get("sec"))             
                if fall>thres1: 
                    pstate= 1
                    print "price will go up!\n"
                elif fall<thres2:
                    pstate = 2
                    print "price will go down!\n"
                else:
                    pstate = 0
                    print "nosiy price!\n"            
                return pstate
            else:
                print "not enough data for slopeStra-IN\n"
                return pstate
            #pos=pos+1
        else:
            print "not data for slopeStra\n"
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
        global i0,length
        length = collection.find({"date":self.datex}).count()
        while i0<2000:        
        #while i0<length-1-N0-1:
            global collection
            doc0 = collection.find({"date":self.datex})[N0+i0]
            self.__frame(doc0,2)  #choose the second strategy
            time.sleep(0.1)
            print "i0=",i0,"\n"
            print "current seconds= ", doc0.get("sec"),"\n"
            print "pos=",pos
            i0=i0+1
#----------------------backtest analysis---------------------
    def performance(self):
        global filename
        if len(self._traders)==1:
            tradex=self._traders[0]
            plCumu=np.cumsum(tradex.PL)
            print "accumulate profit  :\n ",plCumu,"\n"
            label0="profit or loss"
            title0= "accumulate P&L"
            fig=plt.figure(1,figsize=(16, 12), dpi=4800, facecolor='w', edgecolor='k')
            plt.plot(tradex.momentC,plCumu,'go-',label=label0)
            plt.title(title0)
            plt.xlabel("trading time")
            plt.ylabel("amount")
            fig.savefig(filename)
            plt.close(fig)
            
    def tradeVisual(self):
        if len(self._traders)==1:
            tradex=self._traders[0]
            openTime= tradex.momentO
            openPrice= tradex.tradeP1
            closeTime=tradex.momentC
            closePrice= tradex.tradeP2
            longShort= tradex.direction
            fig1=plt.figure(1,figsize=(16, 12), dpi=4800, facecolor='w', edgecolor='k')
            ax1 = fig1.add_subplot(111)
            ax1.plot(openTime,openPrice,'g^')
            ax1.plot(closeTime,closePrice,'rs')
            ax1.set_xlabel('time(#slice)')
            ax1.set_ylabel('trading price', color='r',fontsize=16)
            ax2 = ax1.twinx()
            ax2.plot(openTime,longShort,'co')
            ax2.set_ylabel('trading direction', color='c',fontsize=16)
            plt.show()
            fig1.savefig(filename1,format='pdf',dpi=4800)
            plt.close(fig1)
            
            
            
#----------------------------------
            
global path,filename
path = 'C:/Users/Jiansen/Documents/Finance/a_stock_intra/visualization/Backtest/SZ002500/trials'
filename = 'accuPL_SZ002500_2015-12-08_042302.pdf'
filename = os.path.join(path, filename)
filename1 = 'accuPL_SZ002500_2015-12-08_042303.pdf'
filename1 = os.path.join(path, filename1)

       
datex =  "2015-12-18"
thres1=6.0
thres2=1.0 
      
if __name__ == "__main__":
    world = World()
    world.init(datex)
    world.run()
    world.performance()
    world.tradeVisual()
    