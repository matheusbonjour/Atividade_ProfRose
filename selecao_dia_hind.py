

import matplotlib as mpl 
import matplotlib.pyplot as plt
import numpy as np 
import math as mat
import scipy.io 
import xarray as xr     
import matplotlib
import sys, glob
import cartopy, cartopy.crs as ccrs 
from sklearn.linear_model import LinearRegression
from scipy import signal
import cmocean.cm as cm
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from eofs.xarray import Eof
from eofs.multivariate.standard import MultivariateEof
from mpl_toolkits.axes_grid1 import AxesGrid


import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature, NaturalEarthFeature
#from estatistica_semplot import index_plot

index_plot = mtempo99.index

ds_u = xr.open_dataset('/p1-nemo/mbonjour/novos_wt_dados/30anos_1000hpa/u1000_carol_9120.nc',decode_times=True)
ds_v = xr.open_dataset('/p1-nemo/mbonjour/novos_wt_dados/30anos_1000hpa/v1000_carol_9120.nc',decode_times=True)
ds_hgt = xr.open_dataset('/p1-nemo/mbonjour/novos_wt_dados/30anos_1000hpa/hgt1000_carol_9120.nc',decode_times=True)


# Criando dataset para tirar tendência 
dsT_hgt = ds_hgt['z'].sel(time=slice("1991-01-01T00:00:00","2020-12-31T23:00:00")).squeeze()
dsT_u = ds_u['u'].sel(time=slice("1991-01-01T00:00:00","2020-12-31T23:00:00")).squeeze()
dsT_v = ds_v['v'].sel(time=slice("1991-01-01T00:00:00","2020-12-31T23:00:00")).squeeze()

# Criando dataset para tirar sazonalidade 
dsS_hgt = ds_hgt['z'].sel(time=slice("1991-01-01T00:00:00","2020-12-31T23:00:00")).squeeze() 
dsS_u = ds_u['u'].sel(time=slice("1991-01-01T00:00:00","2020-12-31T23:00:00")).squeeze()
dsS_v = ds_v['v'].sel(time=slice("1991-01-01T00:00:00","2020-12-31T23:00:00")).squeeze()


# Extraindo Lat e Lon dos dados 
lat = ds_hgt['latitude'].squeeze()
lon = ds_hgt['longitude'].squeeze()
# Extraindo Tempo 
tempo = ds_hgt['time'].squeeze()

tempo_novo = pd.date_range("1991-01-01","2020-12-31", freq='D')

#dsS_hgt['time'] = tempo_novo 
#dsS_u['time'] = tempo_novo 
#dsS_v['time'] = tempo_novo 



#dsS2_hgt = dsS_hgt.sel(time=dia_escolhido)

# SEL COMPOSITES
dshgt = dsS_hgt.sel(time=index_plot)
dsu = dsS_u.sel(time=index_plot)
dsv = dsS_v.sel(time=index_plot)

hgtmean = dshgt.mean("time")
vmean = dsv.mean("time")
umean = dsu.mean("time")


dsf_hgt = hgtmean.values
dsf_u = umean.values
dsf_v = vmean.values


dsf_hgt = hgtmean.values

sp=8
hd=4
scale1 = 120
extent = [-60, -25, -45, -10] 
dia_escolhido = index_plot





    
fig, ax = plt.subplots(1,3,figsize=(12,8),constrained_layout = True,subplot_kw=dict(projection=ccrs.PlateCarree()))


#fig = plt.figure(figsize=(8,8))
img_extent = [extent[0], extent[2], extent[1], extent[3]] # [min. lon, max. lon, min. lat, max. lat]
#ax = plt.axes(projection=ccrs.PlateCarree())
data_min = dsf_hgt.min()
data_max = dsf_hgt.max()
interval = 80
levels = np.arange(data_min,data_max,interval)
Nt = len(tempo) 

i=0


for i in range(len(index_plot)):


    cs = ax.flat[i].contourf(lon, lat, dsf_hgt[i,:,:], levels=levels, extend='both',cmap=cm.balance)

    ax.flat[i].coastlines(resolution='50m', color='black', linewidth=0.8)
    ax.flat[i].add_feature(cartopy.feature.BORDERS, edgecolor='black', linewidth=0.5)
    
    
    
    gl = ax.flat[i].gridlines(crs=ccrs.PlateCarree(), color='gray', alpha=1.0, linestyle='--', linewidth=0.25, xlocs=np.arange(-180, 180, 5), ylocs=np.arange(-90, 90, 5), draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False 
    gl.ylabel_style = {'size':16, 'weight':'bold'}
    gl.xlabel_style = {'size':16, 'weight':'bold'}

    if i == 1:
        gl.left_labels = False

    if i == 2: 
        gl.left_labels = False
    #cs = ax.flat[i].contourf(lon, lat, dsf_hgt[i,::sp,::sp], levels=levels, extend='both',cmap='RdBu_r')
    ww = ax.flat[i].quiver(lon[::sp], lat[::sp], dsf_u[i,::sp,::sp], dsf_v[i,::sp,::sp],headwidth=hd, headlength=hd, headaxislength=hd, scale=scale1)
    #plt.title(pd.to_datetime(str(index_plot)).strftime("%Y_%b_%d"), fontsize=16, fontweight='bold')
    ax.flat[i].set_title(pd.to_datetime(str(index_plot[i])).strftime("%Y_%b_%d_%H"), fontsize=16, fontweight='bold')

cb=fig.colorbar(cs, ax=ax.flat[:], shrink=0.6, aspect=12) 
cb.set_label('Altura do Geopotencial [m]', fontsize=16, fontweight='bold') 

#fig.tight_layout()
#cb.tight_layout()
plt.savefig(f'NOVO_Hora_{pd.to_datetime(str(index_plot[0])).strftime("%Y_%b_%d_%H")}',dpi=300)

plt.show()

