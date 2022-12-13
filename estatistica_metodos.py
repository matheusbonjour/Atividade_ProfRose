import pandas as pd
import xarray as xr
import numpy as np
import math as m

import matplotlib.pyplot as plt 
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

from scipy import signal

from statsmodels.graphics.tsaplots import plot_acf


# Leitura do dado
df = pd.read_csv('waverys_1993_2019_ptoSantos.txt', sep='\s+', header=None ,names=['Data', 'Tempo', 'Latitude', 'Longitude', 'Hs'])


# Criando novo indice com as datas e horas
df['Id'] = df['Data']+ ' '+df['Tempo']
df = df.set_index(['Id'])
df.index=pd.to_datetime(df.index)

# Fazendo média diária
df_daily = df.resample('D').mean()


# Plot simples da serie temporal #
#ax1 = df_daily.plot(y='Hs', color='#A1456D', figsize=(12,8))

fig1, ax1 = plt.subplots(figsize=(12,8))
ax1.plot(df_daily.index, df_daily['Hs'], color='#A1456D')

# Configurando eixos # 
ax1.set_ylabel("Hs (m)", fontsize=16, fontweight='bold')
ax1.set_xlabel("Tempo", fontsize=16,  fontweight='bold')

# Media de Hs 
media = df_daily['Hs'].mean()
std1 = df_daily['Hs'].std() 
print(f'A média de Hs é {media}')
print(f'O desvio padrão é {std1}')
stdmd1 = media +std1 
stdmd2 = media + 2*std1
stdmd3 = media + 3*std1
stdmd1n = media - std1
ax1.set_facecolor('#FFF8D4')
ax1.grid(color='grey')
ax1.axhline(y=media, linewidth=4,color='r',label='Media')
ax1.axhline(y=stdmd1, linewidth=3, color='green',label='Desvio Padrão')
ax1.axhline(y=stdmd1n, linewidth=3, color='green')
ax1.axhline(y=stdmd3, linewidth=4, color='k', label="3x Desvio")
ax1.legend(fontsize=16)
#ax1.set_yticks(fontsize=20)
ax1.xaxis.set_tick_params(labelsize=16)
ax1.yaxis.set_tick_params(labelsize=16)
plt.savefig('Fig1_metodos')
#plt.show()
plt.close()


# obtendo tendencia
Hs_ori = df_daily['Hs']
#Hs_ori = Hs_ori.rename(columns = {0: 'Hs'})
Hs_trend = signal.detrend(Hs_ori)

Hs_trend = pd.DataFrame(Hs_trend)
Hs_trend.index = Hs_ori.index
Hs_trend = Hs_trend.rename(columns = {0: 'Hs'})
# Retirando tendencia
Hs_detrended = Hs_ori - Hs_trend['Hs']

df_daily['Trend'] = Hs_trend['Hs']+media
df_daily['Detrend'] = Hs_detrended 


fig2, ax2 = plt.subplots(1,figsize=(12,8))

ax2.plot(Hs_ori,label='Serie original', color='#A1456D')
ax2.plot(Hs_trend['Hs']+ media, label='Serie sem tendência', color='#B0771B')
ax2.plot(Hs_detrended,label='Tendência',color ='k')
ax2.set_facecolor('#FFF8D4')
ax2.grid(color='grey')
ax2.legend(fontsize=16)
#ax1.set_yticks(fontsize=20)
ax2.xaxis.set_tick_params(labelsize=16)
ax2.yaxis.set_tick_params(labelsize=16)
ax2.set_xlim(Hs_ori.index[0], Hs_ori.index[-1])
# Configurando eixos # 
ax2.set_ylabel("Hs (m)", fontsize=16, fontweight='bold')
ax2.set_xlabel("Tempo", fontsize=16, fontweight='bold')

plt.savefig('Fig2_metodos')
plt.close()


media_diaria = pd.DataFrame(data = {'Id': pd.date_range(start = '20160101', end = '20161231', freq = '1D'), 'Hs': list(Hs_ori.groupby([Hs_ori.index.month, df_daily['Hs'].index.day]).mean())})
#media_diaria = pd.DataFrame(data= {'serie': pd.date_range(start='20120101', end = '20121231', freq='1D'), 'We':list(Wener_ori.groupby([Wener_ori.index.month, df_daily['We'].index.day]).mean())})
media_diaria = media_diaria.drop(59)

media_diaria = media_diaria.set_index('Id')

fig3, ax3 = plt.subplots(figsize=(14,8))

ax3.plot(media_diaria, label='Média diária de todos os anos', linewidth=3, color='#A1456D')

ax3.set_facecolor('#FFF8D4')
ax3.grid(color='grey')
#ax3.axhline(y=media, linewidth=4,color='r',label='Media')
#ax3.axhline(y=stdmd1, linewidth=3, color='green',label='Desvio Padrão')
#ax3.axhline(y=stdmd1n, linewidth=3, color='green')
#ax3.axhline(y=stdmd3, linewidth=4, color='k', label="3x Desvio")
ax3.legend(fontsize=16)
#ax1.set_yticks(fontsize=20)
ax3.xaxis.set_tick_params(labelsize=16)
ax3.yaxis.set_tick_params(labelsize=16)
ax3.set_xlim(media_diaria.index[0], media_diaria.index[-1])
date_form = DateFormatter("%b")
ax3.xaxis.set_major_formatter(date_form)
# Configurando eixos # 
ax3.set_ylabel("Hs (m)",fontsize=16, fontweight='bold')
ax3.set_xlabel("Tempo",fontsize=16, fontweight='bold')


plt.savefig('Fig3_metodos')
plt.close()



nf = len(media_diaria)

tf = np.arange(1,nf+1)
wf = 2* m.pi / nf
argf = wf*tf
#print(f't = {tf} \n w ={wf} \n wt = argf')

def calc_phi(A,B):
   phi = np.arctan((B/A))
   return phi



svfA1 = []
svfB1 = []
for i in range(len(tf)):

  Af1 = media_diaria['Hs'][i] * np.cos((argf[i]))
  
  Bf1 = media_diaria['Hs'][i] * np.sin((argf[i]))
  
  svfA1.append(Af1)
  svfB1.append(Bf1)


Af1f = sum(svfA1)*(2/nf)
Bf1f = sum(svfB1)*(2/nf)

Cf1novo = ((Af1f**2)+(Bf1f**2))**0.5

phi_funcf = calc_phi(Af1f,Bf1f) + np.pi

argf2 = wf*tf - phi_funcf
argf = wf*tf

cicloano = []

for i in range(len(tf)):

  #calcdphi = m.cos(np.degrees(arg2[i]))

  calcdphic1f_reg = Cf1novo * m.cos((argf2[i]))
  cicloano.append(calcdphic1f_reg)

df_cicloano = pd.DataFrame(cicloano)
df3 = media_diaria.reset_index()
# Plot simples da serie temporal #


fig4, ax4 = plt.subplots(2,1,figsize=(18,10))
ax4[0].plot(media_diaria['Hs'],label='Hs Médio',linewidth=5,color='#A1456D')


# Configurando eixos # 
ax4[0].set_ylabel("Hs (m)")
#ax5.set_xlabel("Climatologia diaria")
ax4[0].set_title("Climatologia diária e Harmônico Fundamental", fontsize=14)


# Configurando eixos # 
ax4[0].set_ylabel("Hs (m)",fontsize=16, fontweight='bold')
#ax4[0].set_xlabel("Tempo",fontsize=16, fontweight='bold')

mf1 = media_diaria['Hs'].mean()
ciclo_anual = df_cicloano + mf1
ciclo_anual = ciclo_anual.rename(columns = {0: 'Hs'})
ciclo_anual.index = media_diaria.index

ax4[0].plot(ciclo_anual, label ='Harmonico1',linewidth=5,color='#FA705D')

leg = ax4[0].legend()


anoma_mean  = media_diaria - ciclo_anual
tags = list('abcdefgh')


ax4[1].plot(anoma_mean,label='Anomalia (harmônico - climatologia)',linewidth=5,color='#A1456D')
#ax4[1].set_title('Climatologia - Harmônico fundamental',fontsize=14)
ax4[1].set_ylabel("Hs (m)",fontsize=16, fontweight='bold')
# Configurando eixos # 
ax4[1].set_xlabel("Tempo",fontsize=16, fontweight='bold')

#fig3, ax3 = plt.subplots(figsize=(12,8))

#ax4[].plot(media_diaria, label='Média diária de todos os anos', linewidth=3, color='#A1456D')


for cfg in range(len(ax4)):
  ax4[cfg].set_facecolor('#FFF8D4')
  ax4[cfg].grid(color='grey')
  ax4[cfg].legend(fontsize=16)
  #ax1.set_yticks(fontsize=20)
  ax4[cfg].xaxis.set_tick_params(labelsize=16)
  ax4[cfg].yaxis.set_tick_params(labelsize=16)
  ax4[cfg].set_xlim(media_diaria.index[0], media_diaria.index[-1])
  date_form = DateFormatter("%b")
  ax4[cfg].xaxis.set_major_formatter(date_form)
  plt.text(0, 1, tags[cfg], 
             transform=ax4[cfg].transAxes, 
             va='bottom', 
             fontsize=plt.rcParams['font.size']*2.5, 
             fontweight='bold')  



plt.savefig('Fig4_metodos')
plt.close()


ciclo_anual = df_cicloano + mf1

cau = pd.concat([ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual])
plot_var10 = df_daily['Hs'].drop(index=['1996-02-29', '2000-02-29', '2004-02-29','2008-02-29', '2012-02-29','2016-02-29'])
plot_var10 = pd.DataFrame(plot_var10)

cau = cau.rename(columns = {0: 'Hs'})
cau.index = plot_var10.index
plot_var15 = plot_var10.rename(columns = {0: 'Hs'})
anoma = plot_var10 - cau


fig_anoma, axnoma = plt.subplots(figsize=(14,8))

axnoma.plot(anoma, label='Anomalia de HS', linewidth=3, color='#A1456D')


axnoma.set_ylabel("Hs (m)",fontsize=16, fontweight='bold')
# Configurando eixos # 
axnoma.set_xlabel("Tempo",fontsize=16, fontweight='bold')

axnoma.set_facecolor('#FFF8D4')
axnoma.grid(color='grey')
axnoma.legend(fontsize=16)

axnoma.xaxis.set_tick_params(labelsize=16)
axnoma.yaxis.set_tick_params(labelsize=16)
axnoma.set_xlim(anoma.index[0], anoma.index[-1])


plt.savefig('Fig5_metodos')
plt.close()





def pesos_lowpass(janela, cutoff):
#  meiuca = ((janela - 1) // 2 ) + 1
#  npesos = 2 * meiuca + 1
#  pesos = np.zeros([npesos])
#  meio = npesos // 2
#  pesos[meio] = 2 * cutoff
#  k = np.arange(1., (meio*2))

  order = ((janela - 1) // 2 ) + 1
  nwts = 2 * order + 1
  w = np.zeros([nwts])
  n = nwts // 2
  w[n] = 2 * cutoff 
  k = np.arange(1., n)
  #print(f'k = {k}')
  # Pra calcular Wk la de cima, vamos separar o produto 
  sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
  firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
  #print(f'wp1 = {firstfactor}; wp2 = {sigma}')
  # do meio pra tras
  w[n-1:0:-1] = firstfactor * sigma
  w[n+1:-1] = firstfactor * sigma
  #w[n] = firstfactor[n-2] * sigma[n-2]
  #w[1:-1] = firstfactor * sigma
  # do meio pra frente
  #w[1:-1] = firstfactor * sigma
  print(f'\n w = {w[n]}')
  return w[1:-1]




dadoxr = xr.DataArray(anoma)
#ax = dado1.plot()

janela= 30
cutoff=1/10

peso365 = pesos_lowpass(janela, cutoff)

pesolp365 = xr.DataArray(peso365, dims = ['janela'])

HSxr = dadoxr

#coords=[pd.date_range(dadoxr['Id'])]

# Aplicando filtro lowpass
lowpass = HSxr.rolling(Id= len(peso365), center = True).construct('janela').dot(pesolp365)

mean = anoma.mean()


lowpandas = lowpass.to_pandas()
highpandas = anoma - lowpandas
highpandas = highpandas + mean 

#highpass = HSxr - lowpass
#highpass = highpass + mean

fig, axpass = plt.subplots(3,1,figsize=(16, 14))


axpass[0].plot(anoma,color='#A1456D')
axpass[1].plot(lowpandas,color='#A1456D')
axpass[2].plot(highpandas, color='#A1456D')


axpass[0].set_title('Serie Original',fontsize=16)
axpass[1].set_title('LowPass')
axpass[2].set_title('HighPass')



index_plot = highpandas['Hs'].idxmax()
axpass[2].scatter(index_plot,highpandas['Hs'].max(),color='k')


starty, endy = axpass[0].get_ylim()
#startx, endx = axpass[1].get_xlim()
tags = list('abcdefgh')


Fperc99 = np.percentile(highpandas['Hs'].dropna(),99)
Fperc95 = np.percentile(highpandas['Hs'].dropna(),95)
Fperc01 = np.percentile(highpandas['Hs'].dropna(),1)
Fperc05 = np.percentile(highpandas['Hs'].dropna(),5)



#mhigh = highpandas['Hs'].mean()
#stdhigh = highpandas['Hs'].std()  
#
#std2high = 2*stdhigh
#std3high = 3*stdhigh 
#perc95 = std2high + mhigh
#perc99 = std3high + mhigh
#perc05 = mhigh - std2high
#perc01 = mhigh - std3high  

btempo05 = highpandas['Hs'][highpandas['Hs'] <= Fperc05]
btempo01 = highpandas['Hs'][highpandas['Hs'] <= Fperc01]
mtempo95 = highpandas['Hs'][highpandas['Hs'] >= Fperc95]
mtempo99 = highpandas['Hs'][highpandas['Hs'] >= Fperc99]


axpass[2].axhline(y=Fperc99, linewidth=4, color='green',label='Percentil 99%')
axpass[2].axhline(y=Fperc01, linewidth=4, color='red', label='Percentil 1%')
axpass[2].legend() 


for cfg in range(len(axpass)):
  axpass[cfg].set_facecolor('#FFF8D4')
  axpass[cfg].grid(color='grey')
  #axpass[cfg].legend(fontsize=16)
  #ax1.set_yticks(fontsize=20)
  axpass[cfg].xaxis.set_tick_params(labelsize=16)
  axpass[cfg].yaxis.set_tick_params(labelsize=16)
  axpass[cfg].set_xlim(anoma.index[0], anoma.index[-1])
  #date_form = DateFormatter("%b")
  #axpass[cfg].xaxis.set_major_formatter(date_form)
  
  axpass[cfg].yaxis.set_ticks(np.arange(starty, endy, 1))
  axpass[cfg].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
  axpass[cfg].set_ylim(anoma['Hs'].min(), anoma['Hs'].max())
  axpass[cfg].xaxis.set_major_locator(mdates.YearLocator(2))   #to get a tick every 15 minutes
  axpass[cfg].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
  #axpass[cfg].yaxis.set_ticks(pd.date_time())
  plt.text(0, 1, tags[cfg], 
             transform=axpass[cfg].transAxes, 
             va='bottom', 
             fontsize=plt.rcParams['font.size']*2.5, 
             fontweight='bold')  


plt.savefig('Fig62_metodos')
plt.close()




