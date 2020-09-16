import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.stats import kstest
from tqdm.notebook import tqdm

#S&P500 components including the index ticker 
stocks = pd.read_excel('SP500.xlsx')['Stock'].tolist()
data = yf.download(stocks)['Adj Close']
returns = np.log(data/data.shift(1))
stats=pd.DataFrame(columns=['Kurt','Skew','An.Mean','An.Std'],index=data.columns)
for c in stats.index:
    stats['Kurt'].loc[c]=kurtosis(returns[c].dropna())
    stats['Skew'].loc[c]=skew(returns[c].dropna())
    stats['An.Mean'].loc[c]=returns[c].dropna().mean()*252
    stats['An.Std'].loc[c]=returns[c].dropna().std()*np.sqrt(252)
    
#Fitting function
def distr_fit(df,c,distribution):
    clean_data=df[c].dropna().values
    dist = getattr(scipy.stats, distribution)
    param = dist.fit(clean_data)
    try:
        p_value=kstest(clean_data,distribution,args=param)[1]
    except:
        p_value=np.nan
    return p_value

#Performing a KS normality test
stats['Norm test']=None
for c in stats.index:
    stats['Norm test'].loc[c]=distr_fit(returns,c,'norm')
    
#Testing 99 scipy's distributions for all S&P500 constituents
distr=pd.read_excel('distributions.xlsx')['Distribution'].to_list()
distr_test=pd.DataFrame(columns=distr,index=returns.columns)
for d in tqdm(distr_test.columns):
    for c in distr_test.index:
        distr_test[d].loc[c]=distr_fit(returns,c,d)
    
