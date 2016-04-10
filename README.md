               python scripts for the visualization of stocks tick data
 
 Requires: python 2.7.10, pymongo 3.2.2, pandas 0.16.2, numpy 1.10.1
  
  
----------------------------------
I.  overview
II. code example
----------------------------------

-----------------------------------
I overview
----------------------------------
  
   The visualization of stocks tick data has been enabled by these two python scripts,
mongo_StockVisualization.py and mongoStockVisualClass0.py, where the latter 
is a reconstruction of the former, providing a general interface to visualize 
the order_size/volume/trading price/ for each tick shot in a certain stock 
database based on mongoDB technique. 
   Two classes are defined in mongoStockVisualClass0.py,
1. class basic_visualize()
 # for a time series plot of the order_size_at_certain_price/
 volume/trading price/ etc.
2. class tick_visualize()
 visualize the tick data for a certain trading day, and the order_size are
 color encoded, 
with the blue line denoting the moving average traded price of two neighbouring 
tick shots.

(to continue...)
ps: We create the indices for the database by another python script

-----------------------------------
II Code examples
----------------------------------

Code examples detailing how the use of the codes:
----------------initialize-----------------------------------------------------
# initialize the parameters for visualization
width=0.1
height= 0.05/2.5
datex = "2013-03-18"
u0 = 600
fsize=10
k=10
collection= SH600516
-----Example 1-----------------------------------------------------------------
fig = plt.figure(figsize=(20, 15), dpi=240, facecolor='w', edgecolor='k')
ax = fig.add_subplot(111, aspect='equal')
plt.axis('off')
#create the class
name=collection.name
classTick= tick_visualize(collection,datex,ax,width,height,fsize)
#start plotting
sqPatch = classTick.market_dateSP2(u0,k)
for p in sqPatch:
    ax.add_patch(p)

classTick.markVal_dateSP_hms(u0,k)
classTick.priceLine_dateSP2(u0,k,'b-')
plt.title('visualization of partial '+str(name)+' tickData on '+datex+'with 
u0 = '+str(u0),fontsize=18)

plt.show()
fig.savefig(filename,format='pdf',dpi=1000)

plt.close(fig)
-----------------Example 2------------------------------------------------------
#plot by the class basic_visualize()
fig1 = plt.figure(figsize=(20, 15), dpi=240, facecolor='w', edgecolor='k')
ax1 = fig.add_subplot(111, aspect='equal')
classBasic=basic_visualize(collection,datex,'vol')
classBasic.tick_plotS()
plt.title('trading volume for '+str(name)+' on '+datex)
plt.show()
plt.close(fig)
------------------------------------------------------------------

Feedback: jiansendhu@126.com, j.zheng@uu.nl