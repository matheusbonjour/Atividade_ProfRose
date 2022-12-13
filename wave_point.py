
import os
import subprocess
import matplotlib
from pathlib import Path
from netCDF4 import Dataset
import numpy as np
import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from geopy.distance import geodesic
from math import sin, cos, sqrt, atan2, radians
import pandas as pd
import datetime as dt
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature 
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER





wave_file='/p1-nemo/mbonjour/wt_dados/swh_wdir_2015_2016.nc'
wave_file2 = '/p1-nemo/mbonjour/wt_dados/vtpk_2015_2016.nc'


ds = xr.open_dataset(wave_file).resample(time='1D').mean('time')
dsw2 = xr.open_dataset(wave_file2).resample(time='1D').mean('time')



hs = ds.VHM0.sel(time=slice("2015-01-01T18:00:00","2015-01-31T18:00:00")).squeeze() 
dpw = ds.VMDR.sel(time=slice("2015-01-01T18:00:00","2015-01-31T18:00:00")).squeeze() 
tpk = dsw2.VTPK.sel(time=slice("2015-01-01T18:00:00","2015-01-31T18:00:00")).squeeze() 


def vel_conv(vel,dir):
  if dir <= 90:
    u = vel*np.sin(np.radians(dir))
    v = vel*np.cos(np.radians(dir))
  if dir > 90 and dir <=180:
    dir=dir-90
    u = vel*np.cos(np.radians(dir))
    v = -vel*np.sin(np.radians(dir))
  if dir > 180 and dir <=270:
    dir=dir-180
    u = -vel*np.sin(np.radians(dir))
    v = -vel*np.cos(np.radians(dir))
  if dir > 270 and dir <=360:
    dir=dir-270
    u = -vel*np.cos(np.radians(dir))
    v = vel*np.sin(np.radians(dir))
  return(u,v) 


time=ds.time


if True:
    dp2=np.array(dp.copy())
    ol=np.where((dp>=0) & (dp<180))
    dp2[ol]=dp2[ol] + 180
    ol=np.where((dp>=180) &  (dp<360))
    dp2[ol]=dp2[ol] - 180
    ol=np.where(np.isnan(dp2))
    dp2[ol]=0
    uw=np.zeros(dp.shape)
    vw=np.zeros(dp.shape) 
    for i in range(uw.shape[0]):
      for j in range(uw.shape[1]):
        for k in range(uw.shape[2]):
          uw[i,j,k], vw[i,j,k] = vel_conv(1,dp2[i,j,k])



lat = ds.latitude.values
lon = ds.longitude.values
lon, lat = np.meshgrid(lon, lat)

tmin = [0]
tmax = [3]
id = 0 
levels = np.linspace(min(tmin),max(tmax),10)


U=uw[id,:]
V=vw[id,:]
magnitude=hs.values[id,:]

plt.close(fig)

long_min = -60
long_max = -25
lat_min = -45
lat_max = -10


sp=10
hd=4
scale1 = 100
extent = [-60, -25, -45, -10] 


fig = plt.figure(figsize=(8,8))
img_extent = [long_min, long_max, lat_min, lat_max] # [min. lon, max. lon, min. lat, max. lat]
ax = plt.axes(projection=ccrs.PlateCarree())

ax.coastlines(resolution='50m', color='black', linewidth=0.8)
ax.add_feature(cartopy.feature.BORDERS, edgecolor='black', linewidth=0.5)

lat_pboia = df_daily['Latitude'][0]
lon_pboia = df_daily['Longitude'][0]





ax.plot([lon_pboia],[lat_pboia],
         color='black', linewidth=5, marker='o',markersize=5, zorder=10,label = 'Boia Santos',linestyle='None',
         transform=ccrs.PlateCarree(),
         )

ax.legend()
Nt = len(tempo) 
#ax.coastlines(resolution='10m', color='black', linewidth=0.8)
#ax.add_feature(cartopy.feature.BORDERS, edgecolor='black', linewidth=0.5)

#ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='face', facecolor='0.75'))
gl = ax.gridlines(crs=ccrs.PlateCarree(), color='gray', alpha=1.0, linestyle='--', linewidth=0.25, xlocs=np.arange(-180, 180, 5), ylocs=np.arange(-90, 90, 5), draw_labels=True)
gl.top_labels = False
gl.right_labels = False 
gl.ylabel_style = {'size':16, 'weight':'bold'}
gl.xlabel_style = {'size':16, 'weight':'bold'}
i=0
cs = ax.contourf(lon, lat, magnitude[:,:], levels=levels, extend='both',cmap='RdBu_r')
#ww = ax.quiver(lon[::sp], lat[::sp], U[::sp,::sp], V[::sp,::sp],headwidth=hd, headlength=hd, headaxislength=hd, scale=scale1)
cb=fig.colorbar(cs, ax=ax, shrink=0.8, aspect=20) 
cb.set_label('Altura Significativa de Onda [m]') 
#states = cfeature.NaturalEarthFeature(category='cultural', scale='10m', facecolor='none', lw=0.5, name='admin_1_states_provinces_shp')
#ax.add_feature(states, edgecolor='0.5', zorder=2) 


plt.savefig('Ondas_ponto.png',dpi=300)







