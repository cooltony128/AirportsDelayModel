import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import keras

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics 

from operator import attrgetter
from datetime import datetime,timedelta
from pytz import timezone
from time import time
from scipy import stats
from scipy.ndimage.interpolation import shift
from itertools import chain
from collections import deque
from IPython.display import clear_output
from math import sqrt
from statsmodels.tools.eval_measures import mse
from statsmodels.tools.eval_measures import meanabs as mae


def test_index_gen(time_stamp_threshhold = '2008-01-01 00:00:00-08:00',test_time_num = 1800, test_airport_num = 60):
    idx2airport=pd.read_csv("idx2airport.csv",index_col=0)['0'].to_dict()
    idx2time_stamp=pd.read_csv("idx2time_stamp.csv",index_col=0)['0'].to_dict()
    time_stamp2idx=pd.read_csv("time_stamp2idx.csv",index_col=0)['0'].to_dict()
    random.seed(4)
    test_airport_index = random.sample(idx2airport.keys(), k=test_airport_num)
    test_date_index = random.sample(list(idx2time_stamp.keys())[time_stamp2idx[time_stamp_threshhold]:], k=test_time_num)
    return test_date_index,test_airport_index

def rwse_eval(pred_data, test_date_index, test_airport_index):
    arr_sche = pd.read_csv("ArrTotalFlights.csv",index_col=0)
    dep_sche = pd.read_csv("DepTotalFlights.csv",index_col=0)
    DelayRatio = pd.read_csv("DelayRatio.csv",index_col=0)
    p = DelayRatio.fillna(0).iloc[test_date_index, test_airport_index]
    diff = p - pred_data
    numerator = 0
    denominator = 0
    
    for i in test_airport_index:
        for j in test_date_index:
            weight_wae = np.abs(arr_sche[str(i)].values[j]) + np.abs(arr_sche[str(i)].values[j])
            numerator +=  (np.abs(diff[str(i)].loc[j])**2) * (weight_wae**2)
            
    for i in test_airport_index:
        for j in test_date_index:
            denominator += ((np.abs(arr_sche[str(i)].values[j]) + np.abs(arr_sche[str(i)].values[j])))**2
            
    rwse = float(sqrt(numerator/denominator))
    return rwse

def wae_eval(pred_data,test_date_index,test_airport_index):
    arr_sche = pd.read_csv("ArrTotalFlights.csv",index_col=0)
    dep_sche = pd.read_csv("DepTotalFlights.csv",index_col=0)
    DelayRatio = pd.read_csv("DelayRatio.csv",index_col=0)
    p = DelayRatio.fillna(0).iloc[test_date_index, test_airport_index]
    diff = p - pred_data
    numerator = 0
    denominator = 0
        
    for i in test_airport_index:
        for j in test_date_index:
            weight_wae = np.abs(arr_sche[str(i)].values[j]) + np.abs(arr_sche[str(i)].values[j])
            numerator +=  np.abs(diff[str(i)].loc[j]) * weight_wae
            
    for i in test_airport_index:
        for j in test_date_index:
            denominator += (np.abs(arr_sche[str(i)].values[j]) + np.abs(arr_sche[str(i)].values[j]))
            
    wae = float(numerator/denominator)
    return wae

def model_evaluation(pred_data, test_date_index, test_airport_index):
    DelayRatio=pd.read_csv("DelayRatio.csv",index_col=0)
    
    mae_score = np.mean(mae(DelayRatio.fillna(0).iloc[test_date_index, test_airport_index],pred_data,axis=0))
    print ('mae metric: ',mae_score)
    
    rmse_score = np.mean(mse(DelayRatio.fillna(0).iloc[test_date_index, test_airport_index],pred_data,axis=0))**0.5
    print ('rmse metric: ',rmse_score)
    
    wae_score = wae_eval(pred_data, test_date_index, test_airport_index)
    print ('wae metric: ', wae_score)
    
    rwse_score = rwse(pred_data, test_date_index, test_airport_index)
    print ('rwse metric: ', rwse_score)

    DelayFlights = pd.read_csv("ArrDelayFlights.csv",index_col=0)+pd.read_csv("DepDelayFlights.csv",index_col=0)
    TotalFlights = pd.read_csv("ArrTotalFlights.csv",index_col=0)+pd.read_csv("DepTotalFlights.csv",index_col=0)
    
    w_pre_data = TotalFlights.iloc[test_date_index, test_airport_index]*pred_data
    #display(w_pre_data)
#     w_mae_score = np.mean(mae(DelayFlights.iloc[test_date_index, test_airport_index],w_pre_data,axis=0))
#     print ('w_mae metric: ',w_mae_score)
    
#     w_rmse_score = np.mean(mse(DelayFlights.iloc[test_date_index, test_airport_index],w_pre_data,axis=0))**0.5
#     print ('w_rmse metric: ',w_rmse_score)
    return 


class GraphFlow:
    #================= 08/30/19 maintained by Jiayin Guo=============
    #================= 08/16/19 maintained by Jiayin Guo=============
    def __init__(self,idx2airport,airport2idx,idx2time_stamp,time_stamp2idx,
                 ArrTotalFlights,DepTotalFlights,ArrDelayFlights,DepDelayFlights,
                 pre_data,G,dt,grid = None ,start_time = None,end_time = None,DelayRatio=None):
        self.idx2airport = idx2airport
        self.airport2idx = airport2idx
        self.idx2time_stamp = idx2time_stamp
        self.time_stamp2idx = time_stamp2idx
        self.ArrTotalFlights = ArrTotalFlights
        self.DepTotalFlights = DepTotalFlights
        self.ArrDelayFlights = ArrDelayFlights
        self.DepDelayFlights = DepDelayFlights
        self.G = G
        self.pre_data=pre_data
        self.dt = dt
        
        #=========== self.grid=============
        if grid is not None:
            self.grid = grid
        elif (start_time is not None) & (end_time is not None):
            self.grid = pd.date_range(start_time,end_time,freq=dt,tz=timezone('America/Los_Angeles'))
        else:
            raise
        #=========== self.DelayRatio=============
        if DelayRatio is not None:
            self.DelayRatio = DelayRatio
        else:
            self.DelayRatio = pd.DataFrame( data = (ArrDelayFlights.values + DepDelayFlights.values)/
                                           (ArrTotalFlights.values + DepTotalFlights.values),
                                           index=DepTotalFlights.index,
                                           columns=DepTotalFlights.columns)
            
        #=================self.RealDelayRatio,RealTotalFlights,RealDelayFlights====
        self.RealDelayRatio = self.DelayRatio.rename(index = self.idx2time_stamp).rename(columns = self.idx2airport)
        self.RealArrDelayFlights = self.ArrDelayFlights.rename(index = self.idx2time_stamp).rename(columns = self.idx2airport)
        self.RealDepDelayFlights = self.DepDelayFlights.rename(index = self.idx2time_stamp).rename(columns = self.idx2airport)
        self.RealArrTotalFlights = self.ArrTotalFlights.rename(index = self.idx2time_stamp).rename(columns = self.idx2airport)
        self.RealDepTotalFlights = self.DepTotalFlights.rename(index = self.idx2time_stamp).rename(columns = self.idx2airport)
        return
          
    def draw_network_attr(self, nodes_attr = None , edges_attr = 'weight' , size = 6 , with_pos = True):
        plt.figure(1,figsize=(size,size))
        G=self.to_undir_G(self.G)
        if nodes_attr == 'TimeModifyer':
            labels={node:value.total_seconds()/3600 for node,value in nx.get_node_attributes(G,nodes_attr).items()}
        elif nodes_attr =='time_zone':
            labels={node:value.split('/')[1] for node,value in nx.get_node_attributes(G,nodes_attr).items()}
        elif nodes_attr is None:
            labels=None
        else:
            raise  
       
        pos=nx.get_node_attributes(G,'pos')
        if not with_pos:
            pos=nx.spring_layout(G)
        nx.draw(G,pos=pos,labels = labels)
        nx.draw_networkx_edge_labels(G,pos=pos,edge_labels=nx.get_edge_attributes(G,edges_attr))
        plt.show()
        return

    def real_format(self,df):
        return df.rename(columns = self.idx2time_stamp).rename(index = self.idx2airport)
    
    def slice(self,start_time,end_time):
        idx2airport = self.idx2airport
        airport2idx = self.airport2idx
        idx2time_stamp = self.idx2time_stamp
        time_stamp2idx =self. time_stamp2idx
        dt = self.dt
        grid =  pd.date_range(start_time,end_time,freq=dt,tz=timezone('America/Los_Angeles'))
        G = self.G
        
        t= time()

        # 1 generate new pre_data
        

        pre_data = self.pre_data[(self.pre_data.AbsArrTime >= str(grid[0])) &
                                 (self.pre_data.AbsDepTime <= str(grid[-1]))]
        
        print("===============pre_data generated: ", t-time())
        
        # 2 use pre_data to generate G

        temp = pre_data.groupby(['Origin','Dest'],as_index=False)['Year'].count().rename(columns = {"Year": "weight"})
        nx.set_edge_attributes(G, temp.set_index(['Origin','Dest']).to_dict('index'))
        
        # 3 use pre_data, G to generate TotalFlights, DelayFlights
        [ArrTotalFlights, ArrDelayFlights, DepTotalFlights , DepDelayFlights] = map(
            lambda x:  x[(x.index >= self.time_stamp2idx[grid[0]]) &
                         (x.index <= self.time_stamp2idx[grid[-1]])],
            [self.ArrTotalFlights, self.ArrDelayFlights, self.DepTotalFlights , self.DepDelayFlights] )
        
        return GraphFlow(idx2airport = idx2airport,
                         airport2idx = airport2idx,
                         idx2time_stamp = idx2time_stamp,
                         time_stamp2idx = time_stamp2idx,
                         ArrTotalFlights = ArrTotalFlights,
                         DepTotalFlights = DepTotalFlights,
                         ArrDelayFlights = ArrDelayFlights,
                         DepDelayFlights = DepDelayFlights,
                         pre_data = pre_data,
                         G = G,
                         dt = dt,
                         grid = grid)
        

    def sub_graph_flow(self, start_time = None , end_time = None ,sub_nodes = None, edges = None , verbose = False):
        idx2airport = self.idx2airport
        airport2idx = self.airport2idx
        idx2time_stamp = self.idx2time_stamp
        time_stamp2idx =self. time_stamp2idx
        dt = self.dt
        if start_time is None:
            start_time = str(self.grid[0])[:-6]
            
        if end_time is None:
            end_time = str(self.grid[-1])[:-6]
            
        grid =  pd.date_range(start_time,end_time,freq=dt,tz=timezone('America/Los_Angeles'))
        
        # -1 determine if a faster algrithm apllies
        if (sub_nodes is None) & (edges is None):
            
            return self.slice(start_time,end_time)
        
        if (sub_nodes is not None) & ((edges is not None)):
            print('TODO: sub_nodes and edges can not both None ')
            raise
        t=time()    
        # 0 generate edges
        if edges is None:
            edges = set(self.G.subgraph(sub_nodes).copy().edges)
        print("==============edges generated: ", t-time())
        # 1 generate new pre_data
        

        pre_data_temp = self.pre_data[(self.pre_data.AbsArrTime >= str(grid[0])) &
                                 (self.pre_data.AbsDepTime <= str(grid[-1]))]
        
        pre_data = pre_data_temp.groupby(['Origin','Dest']).filter(lambda x: x.name in edges) 
        print("===============pre_data generated: ", t-time())
        
        # 2 use pre_data to generate G

            
        G = self.G.edge_subgraph(edges).copy()
        temp = pre_data.groupby(['Origin','Dest'],as_index=False)['Year'].count().rename(columns = {"Year": "weight"})
        nx.set_edge_attributes(G, temp.set_index(['Origin','Dest']).to_dict('index'))
        #print(dict(G.edges))
        
        print("===============G generated: ", t-time())
            

        # 3 use pre_data, G to generate TotalFlights, DelayFlights
        counter=0
        attrs=['ArrTotalFlights','ArrDelayFlights','DepTotalFlights','DepDelayFlights']
        dfs={x:[] for x in attrs}
        ans={}
        if (len(airport2idx)>0) & (len(time_stamp2idx)>0) & (len(idx2airport)>0) & (len(idx2time_stamp)>0) :
            for airport in G.nodes:
                print("=========Testing ",airport,"==========airports remains :", len(G.nodes)-counter,'========time lasted so far: ',time()-t)
                temp_raw=fun_1_delay_rate(airport,grid=grid,dt=dt,pre_data=pre_data,G=G)
                for attr in attrs:
                    temp=temp_raw[attr]
                    temp.index=map(lambda x: time_stamp2idx[x], temp.index)
                    temp.name=airport2idx[airport]
                    dfs[attr].append(temp)
                counter+=1
                
            for attr in attrs:   
                ans[attr]=pd.concat(dfs[attr],axis=1).sort_index(axis=1)    

            print('======================Total time:',time()-t)
            
            
            ArrTotalFlights = ans['ArrTotalFlights']
            ArrTotalFlights.index=pd.to_numeric(ArrTotalFlights.index)
            ArrTotalFlights.columns=pd.to_numeric(ArrTotalFlights.columns)

            ArrDelayFlights = ans['ArrDelayFlights']
            ArrDelayFlights.index=pd.to_numeric(ArrDelayFlights.index)
            ArrDelayFlights.columns=pd.to_numeric(ArrDelayFlights.columns)

            DepTotalFlights = ans['DepTotalFlights']
            DepTotalFlights.index=pd.to_numeric(DepTotalFlights.index)
            DepTotalFlights.columns=pd.to_numeric(DepTotalFlights.columns)

            DepDelayFlights = ans['DepDelayFlights']
            DepDelayFlights.index=pd.to_numeric(DepDelayFlights.index)
            DepDelayFlights.columns=pd.to_numeric(DepDelayFlights.columns)
        else:
            print('At least one dict is empty')
            print(len(airport2idx) ,len(time_stamp2idx) ,len(idx2airport) ,len(idx2time_stamp))
            raise
        if not verbose:
            clear_output()
        return GraphFlow(idx2airport = idx2airport,
                         airport2idx = airport2idx,
                         idx2time_stamp = idx2time_stamp,
                         time_stamp2idx = time_stamp2idx,
                         ArrTotalFlights = ArrTotalFlights,
                         DepTotalFlights = DepTotalFlights,
                         ArrDelayFlights = ArrDelayFlights,
                         DepDelayFlights = DepDelayFlights,
                         pre_data = pre_data,
                         G = G,
                         dt = dt,
                         grid = grid)
    
    def quotient_graph_flow(self, sub_grid = None,nodes_groups = None):
        # TODO
        return
      
    def describe(self):
        print('dt :', self.dt)
        print('start time:', self.grid[0])
        print('end time:', self.grid[-1])
        print('All data frame has shape:', self.ArrTotalFlights.shape)
        print("head of RealArrTotalFlights: ")
        display(self.RealArrTotalFlights.head(3))
        print("head of pre_data: ")
        display(self.pre_data.head(3))
        
        return
    def export_GF(self):
        #TODO
        return
    @classmethod
    def import_GF(cls,dt = timedelta(seconds=3600)):
        idx2airport=pd.read_csv("idx2airport.csv",index_col=0)['0'].to_dict()
        airport2idx={y:x for x,y in idx2airport.items()}
        idx2time_stamp=pd.to_datetime(pd.read_csv("idx2time_stamp.csv",index_col=0)['0'],utc = True).dt.tz_convert('America/Los_Angeles').to_dict()
        time_stamp2idx={y:x for x,y in idx2time_stamp.items()}
        
        ArrTotalFlights=pd.read_csv("ArrTotalFlights.csv",index_col=0)
        ArrTotalFlights.index=pd.to_numeric(ArrTotalFlights.index)
        ArrTotalFlights.columns=pd.to_numeric(ArrTotalFlights.columns)
        
        ArrDelayFlights=pd.read_csv("ArrDelayFlights.csv",index_col=0)
        ArrDelayFlights.index=pd.to_numeric(ArrDelayFlights.index)
        ArrDelayFlights.columns=pd.to_numeric(ArrDelayFlights.columns)
        
        DepTotalFlights=pd.read_csv("DepTotalFlights.csv",index_col=0)
        DepTotalFlights.index=pd.to_numeric(DepTotalFlights.index)
        DepTotalFlights.columns=pd.to_numeric(DepTotalFlights.columns)
        
        DepDelayFlights=pd.read_csv("DepDelayFlights.csv",index_col=0)
        DepDelayFlights.index=pd.to_numeric(DepDelayFlights.index)
        DepDelayFlights.columns=pd.to_numeric(DepDelayFlights.columns)
        

        
        pre_data=pd.read_csv("pre_data.csv",index_col=0)
        
        G=nx.DiGraph()
        G.add_edges_from(pd.read_csv('graph_edges.csv',index_col=0).apply(lambda x:(x.source,x.target,{'Distance':x.Distance,'weight':x.weight}),axis=1))
        nx.set_node_attributes(G,pd.read_csv('graph_nodes.csv',index_col=0).apply(
            lambda x:{'pos':tuple(map(float,x.pos[1:-1].split(','))),'weight':x.time_zone},axis=1).to_dict())
        start_time = str(list(time_stamp2idx.keys())[0])[:-6]
        end_time = str(list(time_stamp2idx.keys())[-1])[:-6]
        

        return cls(idx2airport = idx2airport,
                   airport2idx = airport2idx,
                   idx2time_stamp = idx2time_stamp,
                   time_stamp2idx = time_stamp2idx,
                   ArrTotalFlights = ArrTotalFlights,
                   DepTotalFlights = DepTotalFlights,
                   ArrDelayFlights = ArrDelayFlights,
                   DepDelayFlights = DepDelayFlights,
                   pre_data = pre_data,
                   G = G,
                   dt = dt,
                   grid = None ,
                   start_time = start_time,
                   end_time = end_time,
                   DelayRatio=None)
    @classmethod
    def to_undir_G(cls,G):
        G_new=G.to_undirected()

        for edge in G_new.edges:
            G_new.edges[edge]['Distance'] = 0
            G_new.edges[edge]['weight'] = 0
            if edge in G.edges:
                G_new.edges[edge]['Distance']+=G.edges[edge]['Distance'] 
                G_new.edges[edge]['weight'] += G.edges[edge]['weight']
            if swap(edge) in G.edges:
                G_new.edges[edge]['Distance']+=G.edges[swap(edge)]['Distance'] 
                G_new.edges[edge]['weight'] += G.edges[swap(edge)]['weight']
            G_new.edges[edge]['Distance']/=2
            #G_new.edges[edge]['TotalNumberinRange'] = G.edges[edge]['TotalNumberinRange'] + G.edges[swap(edge)]['TotalNumberinRange']
        return G_new
    
   # analysis_1 = my_analysis_1

def swap(edge):
    return (edge[1],edge[0])
def fun_1_ts_gen(data,G,airport,verbose=False):
    t=time()
    depdf=data[data.Origin == airport].set_index('AbsCRSDepTime').rename_axis(None)
    depdf['Leaving']=[True]*len(depdf)
    print('AbsCRSDepTime columnn generated:',time()-t)
    
    arrdf=data[data.Dest == airport].set_index('AbsCRSArrTime').rename_axis(None)
    arrdf['Leaving']=[False]*len(arrdf)
    print('AbsCRSArrTime columnn generated:',time()-t)
    
    res=pd.concat([depdf,arrdf],sort=False).sort_index(axis = 0)
    print('data concated:',time()-t)
    
    if not verbose:
        columns=[ 'Year',  'Month',  'DayofMonth' , 'AirTime' , 'TaxiIn' , 'TaxiOut']
        print('fun_1_ts_gen ended:',time()-t)
        return res.drop(columns=columns)
    print('fun_1_ts_gen ended:',time()-t)
    return res

def fun_1_delay_rate(airport,grid,dt,pre_data,G):
    t=time()
    #pre_data=pd.read_csv("Data/pre_timeseries.csv")
    print('loading finished:' ,time()-t)
    
    
    data=fun_1_ts_gen(pre_data,G,airport)
    
    arr_data = data[~data.Leaving]
    dep_data = data[data.Leaving]
    if arr_data.empty:
        arrdelay_data = arr_data.copy()
    else:    
        arrdelay_data=arr_data[(arr_data.ArrDelay>15) ]
    
    if dep_data.empty:
        depdelay_data = dep_data.copy()
    else:    
        depdelay_data=dep_data[(dep_data.DepDelay>15) ]
    
    des_df_pairs = [['ArrTotalFlights',arr_data],
                      ['DepTotalFlights',dep_data],
                      ['ArrDelayFlights',arrdelay_data],
                      ['DepDelayFlights',depdelay_data]]                    
    print('des_df_pairs generated:' ,time()-t)
    seriess = []
    for des, df in des_df_pairs:
        if df.empty:
            print(des+' data is empty')
            seriess.append(pd.Series(data=[0]*len(grid),index=grid,name=des))
        else:
            seriess.append(fun_1_counter(pd.to_datetime(df.index),grid,dt).rename(des))
    
    res=pd.concat(seriess,axis=1)
    print('counter finishded:' ,time()-t)
    res['DelayRatio']=(res.ArrDelayFlights + res.DepDelayFlights )/ (res.DepTotalFlights + res.ArrTotalFlights)
    
    print('DelayRatio finishded:' ,time()-t)
   
    return  res

def fun_1_counter(ts_raw_data,grid,dt):# covolution of delta)[t,t+dt) for t in grid
    t=time()
    if ts_raw_data.empty:
        raise
    if grid.empty:
        raise
    covolution=[]
    interval=deque()
    r_idx=0
    for ts in grid:
        while r_idx<len(ts_raw_data) and ts_raw_data[r_idx]<ts+dt :
            interval.append(ts_raw_data[r_idx])
            r_idx+=1
        
        while interval and interval[0]< ts+dt-pd.to_timedelta(grid.freq):
            interval.popleft()
        covolution.append(len(interval))
    print('counter finished :',time()-t)
    return pd.Series(covolution,index=grid)






class StaticModel:
    
    def __init__(self,gf):
        

        self.gf = gf
        self.data = gf.pre_data.set_index(pd.to_datetime(gf.pre_data.AbsCRSDepTime,utc = True)).sort_index().copy()
        #display(self.data.head(1))
        
        

        self.time_attrs = ['month','day','hour','minute','dayofweek','quarter','is_month_start','is_month_end','is_quarter_start','is_quarter_end']
        self.airport_attrs =['weight','Distance']
        self.delay_reasons = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay','DepDelay','ArrDelay', ]

        #config vectorize, layout
        self.vectorize, self.layout = self.vectorize_gen(time_attrs = self.time_attrs,
                                          airport_attrs = self.airport_attrs,
                                          delay_reasons = self.delay_reasons)
        # config models
        self.models = pd.Series(data = [self.NN_model,self.LR_model,self.DT_model] , index =['NN_model','LR_model','DT_model'] )

        # config sig_indexs
        self.index_range = range(129,193)
        self.sig_indexs = pd.Series(data =[[bool(num & (1<<n)) for n in range(8)] for num in self.index_range] ,index = self.index_range)
        #display(self.sig_indexs.iloc[[0,-1]])
    
    def describe(self):
        self.gf.describe()
        display('time sig of fold 1: ',self.time_attrs)
        display('airport sig of fold 2: ',self.airport_attrs)
        display('delay_reasons : ',self.delay_reasons)
        display('layout for sigs: ',self.layout)
        display('models considered: ',self.models)
        display('sig combinition considered: ',self.sig_indexs.iloc[[0,-1]])

            

    def fit(self,edges,pred_for_arrdelay = True, normalized = True, verbose = False):
        
        
        t= time()
        temp = self.data.groupby(['Origin','Dest']).filter(lambda x: x.name in edges)
        print('sample_data is ready: ',time()-t)
        self.df = temp.fillna(0).apply(self.vectorize,axis = 1)
        print('sample_df is ready',time()-t)
        display(self.df.head(1))

        if len(self.df)<100:
            print('data size too small')
            raise
            
        edges_info = '_'.join(list(map(lambda x: x[0]+'2'+x[1],edges))[:5])
        
        name = ''
        if pred_for_arrdelay:
            name+='arrdelay'
        else:
            name+='depdelay'
        if  normalized:
            name+= '_normalized'
        print('report_'+edges_info+'_'+name)

        t= time()

        res = self.report(models = self.models,
                     sig_indexs = self.sig_indexs,
                     layout =self.layout,
                     total_df = self.df,
                     pred_for_arrdelay = pred_for_arrdelay,
                     normalized = normalized)
        
        if not verbose:
            clear_output()
            
        print('total time: ',time()-t)
        res.to_csv('report_'+edges_info+'_'+name+'.csv')
        return res
    
    def predict(self, recordS):
        pass
        
        
    @staticmethod
    def time_sig_fold_1_gen(time_attrs = [], prefix=''):
        def time_sig_fold_1(time_index_str):
            return pd.Series(data = attrgetter(*time_attrs)(pd.to_datetime(time_index_str)),
                             index = map(lambda x: prefix+x,time_attrs ))
        return time_sig_fold_1
    @staticmethod
    def time_sig_fold_2_gen(prefix = ''):
        def time_sig_fold_2(time_1,time_2):
            return pd.Series(data = (pd.to_datetime(time_2,utc = True)-pd.to_datetime(time_1,utc = True)).seconds//60,
                             index = [prefix+'Diff'])
        return time_sig_fold_2
    @staticmethod
    def airport_sig_fold_1_gen(gf , prefix=''):
        def airport_sig_fold_1(airport):
            return pd.Series(data = gf.airport2idx[airport],
                             index = [prefix+'iata'])
        return airport_sig_fold_1

    @staticmethod
    def airport_sig_fold_2_gen(gf, airport_attrs =['weight','Distance'], prefix=''):
        def airport_sig_fold_2(origin,dest):
            return pd.Series(data = [ gf.G.edges[(origin,dest)][x] for x in airport_attrs],
                             index = map(lambda x: prefix+x,airport_attrs ))
        return airport_sig_fold_2
    @staticmethod
    def delay_sig_gen( delay_reasons =[]):
        def delay_sig(df):
            return df[delay_reasons]
        return delay_sig
    @staticmethod
    def is_delay_sig_gen():
        def is_delay_sig(df):
            return df[['DepDelay','ArrDelay']]>15
        return is_delay_sig
 
    def vectorize_gen(self,sig_index = [True]*6,time_attrs = None,time_attrs_fold_2 = None,airport_attrs = None,delay_reasons = None):
        layout = []

        time_sig_fold_1_dep = self.time_sig_fold_1_gen(time_attrs = time_attrs, prefix = 'dep')
        layout.append(len(time_attrs))

        time_sig_fold_1_arr = self.time_sig_fold_1_gen(time_attrs = time_attrs, prefix = 'arr')
        layout.append(len(time_attrs))

        time_sig_fold_2 = self.time_sig_fold_2_gen()
        layout.append(1)

        airport_sig_fold_1_dep = self.airport_sig_fold_1_gen(self.gf , prefix='dep')
        layout.append(1)

        airport_sig_fold_1_arr = self.airport_sig_fold_1_gen(self.gf , prefix='arr')
        layout.append(1)

        airport_sig_fold_2 = self.airport_sig_fold_2_gen(self.gf, airport_attrs = airport_attrs, prefix='')
        layout.append(len(airport_attrs))

        delay_sig = self.delay_sig_gen( delay_reasons = delay_reasons)
        layout.append(len(delay_reasons))

        is_delay_sig =self.is_delay_sig_gen()
        layout.append(2)

        def vectorize(df):
            #print(df.shape)
            return pd.concat(pd.Series (data = [time_sig_fold_1_dep(df.AbsCRSDepTime),
                                                time_sig_fold_1_arr(df.AbsCRSArrTime),
                                                time_sig_fold_2(df.AbsCRSDepTime,df.PreAbsArrTime),
                                                airport_sig_fold_1_dep(df.Origin),
                                                airport_sig_fold_1_arr(df.Dest),
                                                airport_sig_fold_2(df.Origin,df.Dest),
                                                delay_sig(df),
                                                is_delay_sig(df)
                                               ]).values)
            return 
        return vectorize, layout
    
    # Make one -hot encoder
    @staticmethod
    def one_hot_encode_object_array(arr):
        uniques, ids = np.unique(arr, return_inverse=True)
        return np_utils.to_categorical(ids, len(uniques))

    @classmethod
    def NN_model(cls,X,y):

        # Create tain and test data
        train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.7, random_state=0)

        train_y_ohe = cls.one_hot_encode_object_array(train_y)
        test_y_ohe = cls.one_hot_encode_object_array(test_y)
        
        #print(train_X.shape, train_y_ohe.shape)
        model = Sequential()
        model.add(Dense(50, input_shape=(X.shape[1],)))
        model.add(Activation('relu'))
        model.add(Dense(30))
        model.add(Activation('relu'))
        model.add(Dense(test_y_ohe.shape[1]))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

        # Actual modelling

        model.fit(train_X, train_y_ohe, verbose=0, batch_size=1)
        score, accuracy = model.evaluate(test_X, test_y_ohe, batch_size=16, verbose=0)
        accuracy = max(1-accuracy,accuracy)
        print("Test fraction correct (NN-Score) = {:.2f}".format(score))
        print("Test fraction correct (NN-Accuracy) = {:.2f}".format(accuracy))
        return score, accuracy,model

    @classmethod
    def LR_model(cls,X,y):
        # Create tain and test data
        train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.7, random_state=0)

        train_y_ohe = cls.one_hot_encode_object_array(train_y)
        test_y_ohe = cls.one_hot_encode_object_array(test_y)

        model = Sequential()
        model.add(Dense(test_y_ohe.shape[1], input_shape=(X.shape[1],)))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

        # Actual modelling
        model.fit(train_X, train_y_ohe, verbose=0, batch_size=1)
        score, accuracy = model.evaluate(test_X, test_y_ohe, batch_size=16, verbose=0)
        accuracy = max(1-accuracy,accuracy)

        print("Test fraction correct (LR-Score) = {:.2f}".format(score))
        print("Test fraction correct (LR-Accuracy) = {:.2f}".format(accuracy))
        return score, accuracy , model

    @staticmethod
    def DT_model(X,y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1)

        model = DecisionTreeClassifier()
        model = model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        score = accuracy
        print("Test fraction correct (DT-Score) = {:.2f}".format(score))
        print("Test fraction correct (DT-Accuracy) = {:.2f}".format(accuracy))

        return score, accuracy , model
    
    @staticmethod
    def RF_model(X,y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1)

        model = DecisionTreeClassifier()
        model = model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        score = accuracy
        print("Test fraction correct (RF-Score) = {:.2f}".format(score))
        print("Test fraction correct (RF-Accuracy) = {:.2f}".format(accuracy))

        return score, accuracy , model
    
    @staticmethod
    def report(models, sig_indexs, layout,total_df,pred_for_arrdelay = True, normalized = False): 
    
        if pred_for_arrdelay:
            pred = -1
        else:
            pred = -2

        data = np.full((len(models),len(sig_indexs)),0.)
        for r, model in enumerate(models.values):
            for c, sig_index in enumerate(sig_indexs.values):
                new_layout = np.array(layout)[sig_index]
                col_index = list(chain.from_iterable(np.array([(range(x,y)) for x,y in zip(np.cumsum(shift(layout,1,cval = 0)),
                                                                               np.cumsum(layout))]
                                                 )[sig_index]))
                print('r: ',r,'c: ',c,'col_index ',col_index)
                array = total_df.iloc[:,col_index].values.astype(int)
                X = array[:, 0:-2]
                if normalized:
                    X = np.nan_to_num(stats.zscore(X,axis = 0))

                y = array[:, pred]
                data[r,c] = model(X,y)[1]
        return pd.DataFrame(data = data, index = models.index, columns = sig_indexs.index)



