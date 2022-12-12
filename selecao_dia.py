

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
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from kneed import KneeLocator
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


index_plot = btempo01.index


dshgt = dsS_hgt.sel(time=index_plot)
dsu = dsS_u.sel(time=index_plot)
dsv = dsS_v.sel(time=index_plot)



dsf_hgt = dshgt.values
dsf_u = dsu.values
dsf_v = dsv.values


hgtmean = dshgt.mean("time")
vmean = dsv.mean("time")
umean = dsu.mean("time")


dsf_hgt = hgtmean.values
dsf_u = umean.values
dsf_v = vmean.values



sp=10
hd=4
scale1 = 100
extent = [-60, -25, -45, -10] 
dia_escolhido = index_plot



fig = plt.figure(figsize=(8,8))
img_extent = [extent[0], extent[2], extent[1], extent[3]] # [min. lon, max. lon, min. lat, max. lat]
ax = plt.axes(projection=ccrs.PlateCarree())
data_min = -1500
data_max = 1500
interval = 100
levels = np.arange(data_min,data_max,interval)
Nt = len(tempo) 
ax.coastlines(resolution='50m', color='black', linewidth=0.8)
ax.add_feature(cartopy.feature.BORDERS, edgecolor='black', linewidth=0.5)
gl = ax.gridlines(crs=ccrs.PlateCarree(), color='gray', alpha=1.0, linestyle='--', linewidth=0.25, xlocs=np.arange(-180, 180, 5), ylocs=np.arange(-90, 90, 5), draw_labels=True)
gl.top_labels = False
gl.right_labels = False 
i=0
cs = ax.contourf(lon, lat, dsf_hgt[:,:], levels=levels, extend='both',cmap='RdBu_r')
ww = ax.quiver(lon[::sp], lat[::sp], dsf_u[::sp,::sp], dsf_v[::sp,::sp],headwidth=hd, headlength=hd, headaxislength=hd, scale=scale1)
cb=fig.colorbar(cs, ax=ax, shrink=0.8, aspect=20) 
cb.set_label('Altura do Geopotencial [m]',labelpad=-7) 




plt.savefig('composites01',dpi=300)

plt.show()