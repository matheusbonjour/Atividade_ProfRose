


dshgt = dsS_hgt.sel(time=index_plot)
dsu = dsS_u.sel(time=index_plot)
dsv = dsS_v.sel(time=index_plot)



dsf_hgt = dshgt.values
dsf_u = dsu.values
dsf_v = dsv.values
tempo = dshgt.time.squeeze()
lat = dshgt['latitude'].squeeze()
lon = dshgt['longitude'].squeeze()
extent = [-60, -25, -45, -10] 


hgtmean = dshgt.mean("time")
vmean = dsv.mean("time")
umean = dsu.mean("time")


sp=8
hd=4
scale1 = 120

fig = plt.figure(figsize=(10,8))
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
cs = ax.contourf(lon, lat, dsf_hgt[i,:,:], levels=levels, extend='both',cmap='RdBu_r')
ww = ax.quiver(lon[::sp], lat[::sp], dsf_u[i,::sp,::sp], dsf_v[i,::sp,::sp],headwidth=hd, headlength=hd, headaxislength=hd, scale=scale1)
cb=fig.colorbar(cs, ax=ax, shrink=0.8, aspect=20) 
cb.set_label('Altura do Geopotencial [m]',labelpad=-7) 
plt.title(pd.to_datetime(str(tempo[i].values)).strftime("%Y_%b_%d"), fontsize=16, fontweight='bold')


def animate(i):

    

      
    ax.coastlines(resolution='50m', color='black', linewidth=0.8)
    ax.add_feature(cartopy.feature.BORDERS, edgecolor='black', linewidth=0.5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), color='gray', alpha=1.0, linestyle='--', linewidth=0.25, xlocs=np.arange(-180, 180, 5), ylocs=np.arange(-90, 90, 5), draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False 
    cs = ax.contourf(lon, lat, dsf_hgt[i,:,:], levels=levels, extend='both',cmap='RdBu_r')
    
    ww = ax.quiver(lon[::sp], lat[::sp], dsf_u[i,::sp,::sp], dsf_v[i,::sp,::sp],headwidth=hd, headlength=hd, headaxislength=hd, scale=scale1)
    ax.quiverkey(ww, 0.85, 0.87,0.1,r'$0.1 \frac{m}{s}$', labelpos='E', coordinates='figure')
    plt.title(pd.to_datetime(str(tempo[i].values)).strftime("%Y_%b_%d"), fontsize=16, fontweight='bold')
    
anim = FuncAnimation(fig, animate, interval = 100, frames = Nt)

plt.tight_layout()
plt.show()