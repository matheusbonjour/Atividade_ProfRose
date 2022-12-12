
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


index_plot = mtempo95.index


dshgt = dsS_hgt.sel(time=index_plot)
dsu = dsS_u.sel(time=index_plot)
dsv = dsS_v.sel(time=index_plot)


monthzada = dshgt.time.dt.month
month_length = dshgt.time.dt.days_in_month

weights = (
    month_length.groupby("time.season") / month_length.groupby("time.season").sum()
)

ds_w_hgt = dshgt.groupby("time.season").mean(dim="time")
ds_w_v = dsv.groupby("time.season").mean(dim="time")
ds_w_u = dsu.groupby("time.season").mean(dim="time")

ds_w_hgt = (dshgt*weights).groupby("time.season").sum(dim="time")
ds_w_v = (dsv*weights).groupby("time.season").sum(dim="time")
ds_w_u = (dsu*weights).groupby("time.season").sum(dim="time")


    
fig, ax = plt.subplots(2,2,figsize=(10,10),subplot_kw=dict(projection=ccrs.PlateCarree()))
#xtick1 = np.arange(lon.min(),lon.max(),5)
#ytick1 = np.arange(lat.max(),lat.min(),5)
#fig = plt.figure(figsize=(8,8))
img_extent = [extent[0], extent[2], extent[1], extent[3]] 
#ax = plt.axes(projection=ccrs.PlateCarree())
data_min = ds_w_hgt.min()
data_max = ds_w_hgt.max()
interval = 80
levels = np.arange(data_min,data_max,interval)
Nt = len(tempo) 
tags = list('abcdefghijklmnopqrs')

for i, seas in zip(range(len(ds_w_hgt.season.values)),ds_w_hgt.season.values):

    dsf_hgt = ds_w_hgt.sel(season=seas).squeeze().values
    dsf_u = ds_w_u.sel(season=seas).squeeze().values
    dsf_v = ds_w_v.sel(season=seas).squeeze().values


    sp=8
    hd=4
    scale1 = 100
    extent = [-60, -25, -45, -10] 



    cs = ax.flat[i].contourf(lon, lat, dsf_hgt[:,:], levels=levels, extend='both',cmap=cm.balance)

    ax.flat[i].coastlines(resolution='50m', color='black', linewidth=0.8)
    ax.flat[i].add_feature(cartopy.feature.BORDERS, edgecolor='black', linewidth=0.5)
    
    
    
    gl = ax.flat[i].gridlines(crs=ccrs.PlateCarree(), color='gray', alpha=1.0, linestyle='--', linewidth=0.25, xlocs=np.arange(-180, 180, 5), ylocs=np.arange(-90, 90, 5), draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False 
    gl.ylabel_style = {'size':14, 'weight':'bold'}
    gl.xlabel_style = {'size':14, 'weight':'bold'}
    #gl.xlocator = mticker.FixedLocator(xtick1)
    #gl.ylocator = mticker.FixedLocator(ytick1) 
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER


    if i == 1:
        gl.left_labels = False

    if i == 3: 
        gl.left_labels = False


    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    

    ww = ax.flat[i].quiver(lon[::sp], lat[::sp], dsf_u[::sp,::sp], dsf_v[::sp,::sp],headwidth=hd, headlength=hd, headaxislength=hd, scale=scale1)

    ax.flat[i].set_title(f'{seas}', fontsize=16, fontweight='bold')
    plt.text(0, 1, tags[i], 
            transform=ax.flat[i].transAxes, 
            va='bottom', 
            fontsize=plt.rcParams['font.size']*2, 
            fontweight='bold',
            color='r')
    plt.subplots_adjust(hspace=0.0)


cb=fig.colorbar(cs, ax=ax.flat[:], shrink=0.6, aspect=12) 
cb.set_label('Altura do Geopotencial [m]', fontsize=16, fontweight='bold') 

#fig.tight_layout()
#cb.tight_layout()
plt.savefig('seasons_mtempo95_weightened3',dpi=300)

plt.show()



