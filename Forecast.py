from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data=pd.read_excel('data.xlsx', skipfooter=2)

#func=lambda date: pd.datetime.strptime('%Y%m')
data.Year.astype(int)
data.replace({'Month':{'Jan':1,
'Feb':2, 
'Mar' :3,
'Apr' :4,
'May':5, 
'Jun' :6,
'Jul' :7,
'Aug' :8,
'Sep' :9,
'Oct' :10,
'Nov' :11,
'Dec' :12}},inplace=True)
data.Month.astype(int)
data['time']=pd.to_datetime((data.Year*100+data.Month).apply(str), format='%Y%m')

#data.set_index(t1, inplace=True)
data.drop(['Year','Month'], axis=1, inplace=True)
data.columns = ['assumption','time']

df = pd.DataFrame({'year': np.array(4*[2018]),'month': np.array(range(1,5)), 'day':np.array(4*[1])})
df=pd.to_datetime(df)

a=np.array(range(36,40))

model = ExponentialSmoothing(np.asarray(data['assumption']), seasonal='mul', seasonal_periods=12).fit()
pred = model.predict(start=a[0], end=a[-1])

plt.plot(data['time'], data['assumption'], label='Data')
plt.plot(df, pred, label='Forecast')
plt.legend(loc='best')


fig= plt.figure()
axes=fig.add_subplot(111)

axes.plot(data['time'], data['assumption'], label='Data')
axes.plot(df, pred, label='Forecast')
axes.legend(loc='upper left', fontsize=7)

fig.autofmt_xdate()

fig.suptitle('Monthly Consumption of Type A Medicine', fontsize=20)
plt.xlabel('Year-Month', fontsize=16)
plt.ylabel('Consumtion', fontsize=16)
plt.show()
fig.savefig('Forecast.jpeg')




#
#
#df = pd.DataFrame({'year': [2018,2018,2018,2018,2018,2018,2018,2018,2018,2018,2018,2018],'month': [1,2,3,4,5,6,7,8,9,10,11,12], 'day':[1,1,1,1,1,1,1,1,1,1,1,1]})
#df=pd.to_datetime(df)
#
#model = ExponentialSmoothing(np.asarray(data['assumption']), seasonal='mul', seasonal_periods=12).fit()
#pred = model.predict(start=a[0], end=a[-1])
#
#plt.plot(data['time'], data['assumption'], label='Data')
##plt.plot(test['time'], test['assumption'], label='Test')
#plt.plot(df, pred, label='Predicted')
#plt.legend(loc='best')
#
#a=np.array(range(36,48))
#df.set_index(a, inplace=True)
#df=df.to_frame
