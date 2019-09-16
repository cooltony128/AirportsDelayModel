# GraphFlow

A visualization and analysis module based on neworkx and pandas

check [here](https://github.com/jyguo1729/AirportsDelayModel/blob/master/Demo.ipynb) for demos.

## Getting Started

### Prerequisites
* networkx 
* pandas
* matplotlib
* statsmodels
* IPython
* pytz
### Installing

Dowload the newest release [here](https://drive.google.com/file/d/1ISxgHh9LJaLB8z2IXwncf__86iTCxnz6/view?usp=sharing) to the root directory.

## Running the tests
generate a top `GraphFlow` object `gf`
```
from graphflow import GraphFlow
from datetime import timedelta
import pandas as pd
import numpy as np

dt = timedelta(seconds=3600)
gf = GraphFlow.import_GF(dt)
```
describe a  `GraphFlow` object
```
gf.describe()
```
![aaa](https://lh3.googleusercontent.com/OyfxePTHlYIq5ldmRzAhh-kes74via27Tr6_Vu2jfMnri1sSzMCZhS9MHeDkXGpNApp2k-vSLzJfsw)

visualize a `GraphFlow` object:
```
gf.draw_network_attr(with_pos = True)
```
![enter image description here](https://lh3.googleusercontent.com/mbS1riOinEs-F1MAK7htqsb6GsGNnMsqyVLbEpFadgRToaVc9i7_HXCN8gsWiNlCWqqwvWUMcIpnwA)
given nodes and time range  generate  a new `GraphFlow` object
```
sub_nodes = ['HNL','LAS','LAX','OGG']
sub_gf=gf.sub_graph_flow(start_time = '2007-02-01 01:00:00',end_time = '2007-03-01 01:00:00',sub_nodes = sub_nodes )
```
![enter image description here](https://lh3.googleusercontent.com/BGutlh3cQAL6yfaBxD927m3CRFt9UU69PucxB-Wgx-YXppWTGxnm1ZJjP4GHymswTVKJcfqq_U5FRw)
![enter image description here](https://lh3.googleusercontent.com/T0R8uf3UFOxcL8w46Uti4EXfgFJr_d44ui9OIlCpccCVABS9zbUs3a11OCvTMn-k1-MJ00n8QZmwYg)

given edges and time range  generate  a new `GraphFlow` object
```
sub_gf = gf.sub_graph_flow('2007-02-01 01:00:00','2007-03-01 01:00:00',
                           edges =  (set(gf.G.out_edges('LAX'))|set(gf.G.in_edges('LAX'))))
```
![enter image description here](https://lh3.googleusercontent.com/6vEPlkbiEQ_ZVUbQXtIcrTdDe0KNLEE_gKRu11l2kNETJnbLnVPBDz-AgUnvmj5DJW0TyF1OSBzFpw)
![enter image description here](https://lh3.googleusercontent.com/jBABA7rioY7eLJtvvHSkecgZMDWEwxYKzcG_vT3k3PVm9Imj9iALFvUy0LMVWvdyme2TIwv9aGBl4Q)
if no edges or nodes are provieded, time range only can also generate  a new `GraphFlow` object
```
sub_gf=gf.sub_graph_flow('2007-02-01 01:00:00','2007-03-01 01:00:00')
```
![enter image description here](https://lh3.googleusercontent.com/kCo-jo1BVwOy4fCt93huGIQlcyzyBsD_A2sesDTMitpHP93kjYHHIMSWiJ8MDBkM1X8UR727vEvz1w)
![enter image description here](https://lh3.googleusercontent.com/L9b6jjoxzpGI7FZOWd7G9SdoRbKAk3LBQZNQDONOCa-aOMypO5sGNPmlIz2c1ETHfhvqZsnnju1ITA)

generate test set for models to predict and provide evaluation function
```
from graphflow import test_index_gen,model_evaluation
test_date_index,test_airport_index = test_index_gen()

# sample input 
pred_data = np.random.random((len(test_date_index),len(test_airport_index)))

# evluate the prediction
model_evaluation(pred_data, test_date_index, test_airport_index)
```
```
mae metric:  0.46644295778469996
rmse metric:  0.5473421547099921
```

## Documentation

This module needs some files to initialize: 
files list:
* graphflow.py
* pre_data.csv
* ArrTotalFlights.csv
* ArrDelayFlights.csv
* DepTotalFlights.csv
* DepDelayFlights.csv
* DelayRatio.csv
* airport2idx.csv
* time_stamp2idx.csv
* graph_edges.csv
* graph_nodes.csv
### graphflow.GraphFlow
_class_ `graphflow.GrapgFlow(idx2airport,airport2idx,idx2time_stamp,time_stamp2idx,
                 ArrTotalFlights,DepTotalFlights,ArrDelayFlights,DepDelayFlights,
                 pre_data,G,dt,grid = None ,start_time = None,end_time = None,DelayRatio=None)`

_Attributes_

| Attributes |Type| Description |
| --- | ----------- |----|
|dt| pd.TimeDelta  |`1D` or `1H` so far|
|G|nx.DiGraph| `G.edges` having `weight` attribution as the total flights in  time range of `grid`  and with `Distance` attribution as the distance between two nodes. `G.nodes`nodes has `pos` attribution as the real position of a nodes and `weight` attribution as the timezone information of type string |
| pre_data | pd.DataFrame |provide pre-cleaned raw data time range of `grid` |
| grid | pd.DatetimeIndex |with `freq` being `dt` |
| idx2airport | dict |`idx2airport[i]` return  the iata name for airport index ` i ` |
|airport2idx | dict |inverse to `idx2airport`|
|idx2time_stamp | dict |`idx2time_stamp[t]` return  the timestamp for time index ` t `|
|time_stamp2idx | dict |inverse to `idx2airport`|
| ArrTotalFlights |  pd.DataFrame |`ArrTotalFlights[t,i]` is number of flights with **scheduled arrive** time between [`idx2time_stamp[t]`,`idx2time_stamp[t] + dt`) at airport `idx2airport[i]`|
| ArrDelayFlights | pd.DataFrame | `ArrDelayFlights[t,i]` is number of arrive delayed flights with **scheduled arrive** time between between [`idx2time_stamp[t]`,`idx2time_stamp[t] + dt)` at airport `idx2airport[i]` |
| DepTotalFlights |  pd.DataFrame |`DepTotalFlights[t,i]` is number of flights with **scheduled departure** time between [`idx2time_stamp[t]`,`idx2time_stamp[t] + dt`) at airport `idx2airport[i]`|
| DepDelayFlights | pd.DataFrame | `DepDelayFlights[t,i]` is number of departure delayed flights with **scheduled departure** time between between [`idx2time_stamp[t]`,`idx2time_stamp[t] + dt)` at airport `idx2airport[i]` |
| RealDepTotalFlights | pd.DataFrame |with same value as `DepTotalFlights` but with `columns` translated by `idx2airport` and `index` translated by `idx2time_stamp` |
| RealArrTotalFlights | pd.DataFrame |with same value as `ArrTotalFlights` but with `columns` translated by `idx2airport` and `index` translated by `idx2time_stamp` |
| RealDepDelayFlights | pd.DataFrame |with same value as `DepDelayFlights` but with `columns` translated by `idx2airport` and `index` translated by `idx2time_stamp` |
| RealArrDelayFlights | pd.DataFrame |with same value as `ArrDelayFlights` but with `columns` translated by `idx2airport` and `index` translated by `idx2time_stamp` |
| RealDelayRatio | pd.DataFrame |with same value as `DelayRatio` but with `columns` translated by `idx2airport` and `index` translated by `idx2time_stamp` |









_Methods_

| Methods | Description |
| --- | ----------- |
| describe() |  describe the current GraphFlow obeject|
| real_format(df) |  convert a `DataFrame` object `df`into columns by times stamps and index by airports names|
| draw_network_attr(nodes_attr= None,edges_attr='weight',size=10,with_pos=True) |  draw undirected graph `fun_1_to_undir_G(G)` try several times to get a better display. If `with_pos` is `Fl=False` position of nodes will be plotted in random|
| sub_graph_flow(start_time = None,end_time = None,sub_nodes = None, edges = None)|  TODO|
### graphflow.GraphFlow.describe
`GraphFlow.describe()`

_parameters_`None`

 _return_ `None`
 
 _description_
 describe the current GraphFlow obeject.


### graphflow.GraphFlow.real_format
`GraphFlow.real_format(df)`

 _parameters_ TODO
 
_return_ `pd.DataFrame`

_description_
 convert a `DataFrame` object `df`into columns by times stamps and index by airports names
### graphflow.GraphFlow.draw_network_attr
`GraphFlow.draw_network_attr(self, nodes_attr = None , edges_attr = 'weight' , size = 6 , with_pos = True)`

_parameters_ TODO

 _return_ `None`
 
_description_
draw undirected graph `fun_1_to_undir_G(G)` try several times to get a better display. If `with_pos` is `Fl=False` position of nodes will be plotted in random

### graphflow.GraphFlow.sub_graph_flow
`GraphFlow.sub_graph_flow(start_time = None,end_time = None,sub_nodes = None, edges = None)`

_parameters_ TODO

_return_  `graphflow.GraphFlow`

_description_
Doing
* Generate a sub  graph flow. In this case all 'DataFrame' including `TotalFlights`,`DelayFlights`,`DelayRatio` will be re-computed as sub_nodes will build a smaller GraphFlow.  and `G` will be updated as well

* If `sub_nodes` is `None` and `edges` is `None` , only weight of `G` will be recomputed, `pre_data` and `TotalFlights`,`DelayFlights`,`DelayRatio`will be sliced accordingly, which is fast. 

* If `sub_nodes` is `None` and `edges` is not `None`, generated a `GraphFlow` object with given `edges`. all data `G` `pre_data` , `TotalFlights`,`DelayFlights`,`DelayRatio` will be recomputed. It may be slower

### graphflow.test_index_gen
`test_index_gen(time_stamp_threshhold = '2008-01-01 00:00:00-08:00',test_time_num = 1800, test_airport_num = 60)`

_parameters_ TODO

_return_ `test_date_index,test_airport_index`

_description_
* `test_date_index` :list of lenth M
element is time index at which model predicts the delay ratio. In our case, all elements `tidx in test_date_index`' satisfy `idx2time_stamp[tidx]` after `2018-01-01 00:00:00`

* `test_airport_index` :list of lenth B
element is airport index at which model predicts the delay ratio
 `model_evaluation(pred_data)`. 

### graphflow.model_evaluation

`model_evaluation(pred_data, test_date_index, test_airport_index)`

_parameters_
*  `pred_data`: `np.ndarray`
`pred_data[t,i]` gives the predicted delay ratio for time index `test_date_index[t]`
 and `test_airport_index[i]`
 
_return_  `None`


_description_
generate the mae , rmse, wae, rwse score of the prediction ``pred_data``.


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Authors

* **Jiayin Guo** - *Initial work* - [MyGithub](https://github.com/jyguo1729)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.


## Deployment

* TODO 

## Built With

* TODO
## Contributing

* TODO

## Versioning

* TODO






## Acknowledgments
* TODO

