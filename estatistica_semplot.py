import pandas as pd
import xarray as xr
import numpy as np
import math as m

import matplotlib.pyplot as plt 
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

from scipy import signal




# Leitura do dado
df = pd.read_csv('waverys_1993_2019_ptoSantos.txt', sep='\s+', header=None ,names=['Data', 'Tempo', 'Latitude', 'Longitude', 'Hs'])


# Criando novo indice com as datas e horas
df['Id'] = df['Data']+ ' '+df['Tempo']
df = df.set_index(['Id'])
df.index=pd.to_datetime(df.index)

# Fazendo média diária
df_daily = df.resample('D').mean()


# Media de Hs 
media = df_daily['Hs'].mean()
std1 = df_daily['Hs'].std() 

stdmd1 = media +std1 
stdmd2 = media + 2*std1
stdmd3 = media + 3*std1
stdmd1n = media - std1


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


media_diaria = pd.DataFrame(data = {'Id': pd.date_range(start = '20160101', end = '20161231', freq = '1D'), 'Hs': list(Hs_ori.groupby([Hs_ori.index.month, df_daily['Hs'].index.day]).mean())})
#media_diaria = pd.DataFrame(data= {'serie': pd.date_range(start='20120101', end = '20121231', freq='1D'), 'We':list(Wener_ori.groupby([Wener_ori.index.month, df_daily['We'].index.day]).mean())})
media_diaria = media_diaria.drop(59)

media_diaria = media_diaria.set_index('Id')



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


mf1 = media_diaria['Hs'].mean()
ciclo_anual = df_cicloano + mf1
ciclo_anual = ciclo_anual.rename(columns = {0: 'Hs'})
ciclo_anual.index = media_diaria.index

anoma_mean  = media_diaria - ciclo_anual

#ciclo_anual = df_cicloano + mf1

cau = pd.concat([ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual, ciclo_anual])
plot_var10 = df_daily['Hs'][~((df_daily.index.month ==2) & (df_daily.index.day ==29))]




#plot_var10 = pd.DataFrame(plot_var10)
plot_var10 = pd.DataFrame(plot_var10)
cau = cau.rename(columns = {0: 'Hs'})
cau.index = plot_var10.index
plot_var10 = plot_var10.rename(columns = {0: 'Hs'})
anoma = plot_var10 - cau







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
lowpass = HSxr.rolling(Id=len(peso365), center = True).construct('janela').dot(pesolp365)

mean = anoma.mean()


lowpandas = lowpass.to_pandas()
highpandas = anoma - lowpandas
highpandas['Hs'] = highpandas + mean 



#highpass = HSxr - lowpass
#highpass = highpass + mean

index_plot2=[]
index_plot2.append([highpandas['Hs'].shift(-2).idxmax(),highpandas['Hs'].shift(-1).idxmax(), highpandas['Hs'].idxmax()])

index_plot3 = []
index_plot3.append([highpandas['Hs'].idxmax() - pd.Timedelta(12,'h'), highpandas['Hs'].idxmax() - pd.Timedelta(6,'h'),highpandas['Hs'].idxmax()])

bb95 = np.percentile(highpandas['Hs'].dropna(),99)

mhigh = highpandas['Hs'].mean()
stdhigh = highpandas['Hs'].std()  

std2high = 2*stdhigh
std3high = 3*stdhigh 
perc95 = std2high + mhigh
perc99 = std3high +mhigh
perc05 = mhigh - std2high
perc01 = mhigh - std3high  

btempo05 = highpandas['Hs'][highpandas['Hs'] <= perc05]
btempo01 = highpandas['Hs'][highpandas['Hs'] <= perc01]
mtempo95 = highpandas['Hs'][highpandas['Hs'] >= perc95]
mtempo99 = highpandas['Hs'][highpandas['Hs'] >= perc99]