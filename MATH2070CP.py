#%%
import numpy as np
import scipy as sp
import scipy.stats as sps
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
# from matplotlib.collections import LineCollection
# 'exec(%matplotlib inline)' 



# %% Setup
data = pd.read_csv(r"/Users/tonygong/Library/CloudStorage/OneDrive-Personal/Study/USYD/Previous Courses Projects/MATH2070 project/Country Indices.csv", index_col = 0)

#%%
data_1 = data.loc['1/03/2007':]
data_1 = data_1.iloc[0:]/data_1.iloc[0] * 100
data_1.columns = data_1.columns.str.replace(' ', '')
data_1.reset_index(inplace=True)
#data_1['Date'] = pd.to_datetime(data_1['Date'], format='%d/%m/%Y').dt.date


#%%
# gfc blue
data_1[data_1['Date']=='31/05/2010']
data_sub1 = data_1.iloc[0:848]
data_sub1

# GFC Peak as the period from 02/09/2008 – 01/06/2009. yellow
data_1[data_1['Date']=='1/06/2009']
data_sub2 = data_1.iloc[392:588]
data_sub2

# (INTERIM) 01/06/2010 – 10/03/2020 green
s= data_1[data_1['Date']=='1/06/2010'].index.values.astype(int)[0]
e= data_1[data_1['Date']=='10/03/2020'].index.values.astype(int)[0]
data_sub3 = data_1.iloc[s-1:e+1]
data_sub3

# COVID-19 (Orange), 11/03/2020 31/08/2020 orange 
s= data_1[data_1['Date']=='11/03/2020'].index.values.astype(int)[0]
e= data_1[data_1['Date']=='31/08/2020'].index.values.astype(int)[0]
data_sub4 = data_1.iloc[s-1:e+1]
data_sub4

# COVID-19 Peak) as the period from 11/03/2020 – 29/05/2020 red 
s= data_1[data_1['Date']=='11/03/2020'].index.values.astype(int)[0]
e= data_1[data_1['Date']=='29/05/2020'].index.values.astype(int)[0]
data_sub5 = data_1.iloc[s-1:e+1]
data_sub5

data_sub_forlog = data.loc['31/05/2010':'10/03/2020'].reset_index()

#%%
data_sub1 = data_sub1.set_index('Date')
data_sub1.index = pd.to_datetime(data_sub1.index, format='%d/%m/%Y')

data_sub2 = data_sub2.set_index('Date')
data_sub2.index = pd.to_datetime(data_sub2.index, format='%d/%m/%Y')

data_sub3 = data_sub3.set_index('Date')
data_sub3.index = pd.to_datetime(data_sub3.index, format='%d/%m/%Y')

data_sub4 = data_sub4.set_index('Date')
data_sub4.index = pd.to_datetime(data_sub4.index, format='%d/%m/%Y')

data_sub5 = data_sub5.set_index('Date')
data_sub5.index = pd.to_datetime(data_sub5.index, format='%d/%m/%Y')

data_sub_forlog = data_sub_forlog.set_index('Date')
data_sub_forlog.index = pd.to_datetime(data_sub_forlog.index, format='%d/%m/%Y')


#%% 1.(ii)
plt.plot(data_sub1, color = 'blue', label="GFC" )
plt.plot(data_sub2, color = 'yellow',label="GFC Peak")
plt.plot(data_sub3, color = 'green', label="INTERIM")
plt.plot(data_sub4, color = 'orange', label="COVID-19")
plt.plot(data_sub5, color = 'red', label="COVID-19 Peak")

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.xticks(rotation=45)
#plt.title("Rebased index values of each country from 2007 to 2020")
plt.savefig('one_ii.png')
plt.show()

#%%
# 1 (iii)
returns_1 = pd.DataFrame()
for indices in data_sub1:
    if indices not in returns_1:
        returns_1[indices] = np.log(data_sub1[indices]).diff()
        
returns_2 = pd.DataFrame()
for indices in data_sub2:
    if indices not in returns_2:
        returns_2[indices] = np.log(data_sub2[indices]).diff()
        
        
returns_1000 = pd.DataFrame()
for indices in data_sub_forlog:
    if indices not in returns_1000:
        returns_1000[indices] = np.log(data_sub_forlog[indices]).diff()
                
returns_3 = pd.DataFrame()
for indices in data_sub3:
    if indices not in returns_3:
        returns_3[indices] = np.log(data_sub3[indices]).diff()
        
returns_4 = pd.DataFrame()
for indices in data_sub4:
    if indices not in returns_4:
        returns_4[indices] = np.log(data_sub4[indices]).diff()
        
returns_5 = pd.DataFrame()
for indices in data_sub5:
    if indices not in returns_5:
        returns_5[indices] = np.log(data_sub5[indices]).diff()

#%%
# check
# returns_5
# returns_5_full = returns_5.drop(pd.to_datetime('2020-03-10'))
# returns_5_full

# ggg = np.array(returns_5_full)
# ggg_trans = ggg.transpose()
# r_g = ggg_trans.mean(1)
# r_g_trans = [r_g]
# r_g = np.transpose(r_g_trans)
# r_g

#plt.matshow(returns_1.corr())

# f = plt.figure(figsize=(19, 15))
# plt.matshow(returns_1.corr(), fignum=f.number)
# plt.xticks(range(returns_1.shape[1]), returns_1.columns, fontsize=14, rotation=45)
# plt.yticks(range(returns_1.shape[1]), returns_1.columns, fontsize=14)
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=14)
# plt.title('Correlation Matrix', fontsize=16);


rs = np.random.RandomState(0)
df = pd.DataFrame(rs.rand(10, 10))
corr = returns_1.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)

rs = np.random.RandomState(0)
df = pd.DataFrame(rs.rand(10, 10))
corr_2 = returns_2.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)

rs = np.random.RandomState(0)
df = pd.DataFrame(rs.rand(10, 10))
corr_3 = returns_3.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)

rs = np.random.RandomState(0)
df = pd.DataFrame(rs.rand(10, 10))
corr_4 = returns_4.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)

rs = np.random.RandomState(0)
df = pd.DataFrame(rs.rand(10, 10))
corr_5 = returns_5.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)

#%%
data_name = data.columns.tolist()
countries = []
for index in range(0,20, 1):
    i=0
    
    while data_name[index][i] != " ":
      i = i + 1
    countries.append(data_name[index][0:i])

new_countries = countries

#%%
returns_1.columns = new_countries

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize = (16,16))
h = sns.heatmap(returns_1.corr(), cmap='coolwarm', annot=True, cbar=False)
h.set_yticklabels(h.get_yticklabels(), rotation = 0)
h.xaxis.tick_top()
# plt.savefig('GFC_cor.png', bbox_inches='tight', pad_inches=0.0)

#%%
returns_2.columns = new_countries

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize = (16,16))
h = sns.heatmap(returns_2.corr(), cmap='coolwarm', annot=True, cbar=False)
h.set_yticklabels(h.get_yticklabels(), rotation = 0)
h.xaxis.tick_top()
# plt.savefig('GFC_peak.png', bbox_inches='tight', pad_inches=0.0)

#%%
returns_3.columns = new_countries

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize = (16,16))
h = sns.heatmap(returns_3.corr(), cmap='coolwarm', annot=True, cbar=False)
h.set_yticklabels(h.get_yticklabels(), rotation = 0)
h.xaxis.tick_top()
# plt.savefig('INTERIM.png', bbox_inches='tight', pad_inches=0.0)

#%%
returns_4.columns = new_countries

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize = (16,16))
h = sns.heatmap(returns_4.corr(), cmap='coolwarm', annot=True, cbar=False)
h.set_yticklabels(h.get_yticklabels(), rotation = 0)
h.xaxis.tick_top()
# plt.savefig('COVID_19.png', bbox_inches='tight', pad_inches=0.0)

#%%
returns_5.columns = new_countries

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize = (16,16))
h = sns.heatmap(returns_5.corr(), cmap='coolwarm', annot=True, cbar=False)
h.set_yticklabels(h.get_yticklabels(), rotation = 0)
h.xaxis.tick_top()
# plt.savefig('COVID_19_peak.png', bbox_inches='tight', pad_inches=0.0)


#%%
# 1 (iv)
t_1 = np.array(returns_1.corr()).flatten()
t_2 = np.array(returns_2.corr()).flatten()
t_3 = np.array(returns_3.corr()).flatten()
t_4 = np.array(returns_4.corr()).flatten()
t_5 = np.array(returns_5.corr()).flatten()

my_list = [t_1,t_2,t_3,t_4,t_5]

for i in range(5):
        plot = plt.hist(x= my_list[i], bins=20, rwidth=0.9, color='#607c8e')
        plt.xlabel('Correlation Coefficients')
        plt.ylabel('Count')
        plt.grid(axis='y', alpha=0.75)
        plt.show()

#%%
# 2 (i)
returns_3_full = returns_3.drop(pd.to_datetime('2010-05-31'))


#%%
returns_3_full_mean_annualized_trans = np.array([returns_3_full.mean()])
returns_3_full_mean_annualized = np.transpose(returns_3_full_mean_annualized_trans)
returns_3_full_variance_annualized = returns_3_full.cov()
returns_3_full_std_annualized = (returns_3_full_variance_annualized)**0.5
returns_3_full_variance_annualized_inverse = np.linalg.inv(returns_3_full_variance_annualized)
returns_3_full_std_annualized_diag = np.array(returns_3_full_std_annualized).diagonal()

#%%
ones_array = np.ones( (20, 1), dtype=np.int32 )
ones_array_trans = np.transpose(ones_array)

#%%
a = np.dot(ones_array_trans, returns_3_full_variance_annualized_inverse)
a = np.dot(a, ones_array)

b = np.dot(ones_array_trans, returns_3_full_variance_annualized_inverse)
b = np.dot(b, returns_3_full_mean_annualized)

c = np.dot(returns_3_full_mean_annualized_trans, returns_3_full_variance_annualized_inverse)
c = np.dot(c, returns_3_full_mean_annualized)

d=np.dot(a, c)
d = np.subtract(d, b**2)

alpha = (1/a) * returns_3_full_variance_annualized_inverse
alpha = np.dot(alpha, ones_array)

beta_real = np.dot(returns_3_full_variance_annualized_inverse, np.subtract(returns_3_full_mean_annualized, (b/a)*ones_array))

#%%
# 2.(i)
x_weight  = np.add(alpha, 0.2*beta_real)
money_invested = 1000000 * x_weight
miu_star = (b+d*0.2)/a
varian = (1+d*0.2**2)/a
sigma_star = np.sqrt(varian)

#%%
# np.savetxt("money invested.csv", money_invested, delimiter=",")
#%%
# 2.(ii)
hundred_ones = np.ones( (100, 1), dtype=np.int32 )
t = np.transpose(np.array([np.linspace(-0.4, 0.4, 100)]))
miu_portfolio = np.add(b*hundred_ones,d*t)/a

# g1 = np.subtract(miu_portfolio, (b/a)*hundred_ones)

# varian_portfolio = np.add((1/a)*hundred_ones, (d/a)*g1**2)

std_portfolio = np.sqrt(np.add(hundred_ones, d*t**2)/a)

#%%
t_2 = 0.2
z=np.add(-t_2*miu_star, 1/2*varian)
sigma = np.linspace(-0.01,0.05,20)
indifference_curveones = np.ones( (len(sigma), 1), dtype=np.int32 )
mu_comp = np.subtract(np.transpose([0.5*sigma**2]) , z*indifference_curveones)/t_2

#%%
num_assets = len(x_weight)

#%% 
p_weights = []
p_vol=[]
port_count = 0
one_array_loop = np.transpose(np.ones( (20, 1), dtype=np.int32 ))
while port_count < 1000:
    weights = np.random.random(num_assets)
    weights = weights/np.sum(weights)
    weights = weights.reshape((20, 1))
    # print(weights)
    # print(weights.transpose().shape, returns_3_full_variance_annualized.shape) 
    varian_port_loop = np.dot(weights.transpose(), returns_3_full_variance_annualized)
    # print(varian_port_loop)
    varian_port_loop = np.dot(varian_port_loop, weights)
    sigma_i = np.sqrt(varian_port_loop)
    # print(sigma_i)
    if np.all([np.abs(weights) < 20*one_array_loop]) and (sigma_i < 0.1): 
        p_weights.append(weights) 
        p_vol.append(sigma_i)
        port_count = port_count + 1
        

#%%
p_miustar = []
for index in range (0,1000,1):
    p_miu_star_loop = np.dot(returns_3_full_mean_annualized_trans, p_weights[index])
    p_miustar.append(p_miu_star_loop)

#%%
t_EF = np.transpose(np.array([np.linspace(0, 0.4, 100)]))
miu_portfolio_EF = np.add(b*hundred_ones,d*t_EF)/a
std_portfolio_EF = np.sqrt(np.add(hundred_ones, d*t_EF**2)/a)

#%% 2.(iii and iv)
plt.scatter(returns_3_full_std_annualized_diag, returns_3_full_mean_annualized, label="Country Index")
plt.plot(std_portfolio, miu_portfolio, label="MVF")
plt.plot(sigma, mu_comp, label="Indifference curve with t=0.2")
plt.plot(std_portfolio_EF, miu_portfolio_EF, label="EF", color = 'red')
plt.plot(np.sqrt(varian), miu_star,'^', label = "Optimal portfolio")
plt.scatter(p_vol, p_miustar, label = "1000 feasible portfolios")
plt.ylabel(r'$\mu$')
plt.xlabel(r'$\sigma$')
plt.legend()
# plt.savefig('miusigma.png')
plt.show()
 
#%%
# 3(i)
ji = alpha.tolist()[0]
ji_2 = beta_real.tolist()[0]
from sympy import *
init_printing()
var('z')

left_ineq= []
right_ineq = []
combo_ineq = []

for k in range (0,20,1):
    lkjj = solve_univariate_inequality(alpha.tolist()[k][0]+z*beta_real.tolist()[k][0] < 0 , z, relational=False)
    left = round(lkjj.left, 2)
    right = round(lkjj.right, 2)
    combo = np.array([[left], [right]])
    #print(round(lkjj.right, 2))
    # left_ineq.append(round(lkjj.left, 2))
    # right_ineq.append(round(lkjj.right, 2))
    combo_ineq.append(combo)
    #print(lkjj)

float_interbal = []
for k in range (0,20,1):
    arr = combo_ineq[k]
    float_arr = np.vstack(arr.astype(np.float))
    float_interbal.append(float_arr)

#%%
float_interval = []
for k in range (0,20,1):
    whatisthis = np.round(float_interbal[k],4)
    float_interval.append(whatisthis)

#%%
for k in range(0,20,1):
    float_interval[k] = float_interval[k].flatten()
 
 #%%
float_interval = np.array(float_interval)
# float_interval = float_interval.transpose(2,0,1).reshape(20,-1)   

#%%
# np.savetxt("trange_ss.csv", float_interval, delimiter=",")

#%%
# 3(ii)
returns_4_full = returns_4.drop(pd.to_datetime('2020-03-10'))


#%%
returns_4_full_mean_annualized_trans = np.array([returns_4_full.mean()])
returns_4_full_mean_annualized = np.transpose(returns_4_full_mean_annualized_trans)
returns_4_full_variance_annualized = returns_4_full.cov()
returns_4_full_std_annualized = (returns_4_full_variance_annualized)**0.5
returns_4_full_variance_annualized_inverse = np.linalg.inv(returns_4_full_variance_annualized)
returns_4_full_std_annualized_diag = np.array(returns_4_full_std_annualized).diagonal()

#%%
ones_array = np.ones( (20, 1), dtype=np.int32 )
ones_array_trans = np.transpose(ones_array)

#%%
a_covi = np.dot(ones_array_trans, returns_4_full_variance_annualized_inverse)
a_covi = np.dot(a_covi, ones_array)

b_covi = np.dot(ones_array_trans, returns_4_full_variance_annualized_inverse)
b_covi = np.dot(b_covi, returns_4_full_mean_annualized)

c_covi = np.dot(returns_4_full_mean_annualized_trans, returns_4_full_variance_annualized_inverse)
c_covi = np.dot(c_covi, returns_4_full_mean_annualized)

d_covi =np.dot(a_covi, c_covi)
d_covi = np.subtract(d_covi, b_covi**2)

alpha_covi = (1/a_covi) * returns_4_full_variance_annualized_inverse
alpha_covi = np.dot(alpha_covi, ones_array)

beta_real_covi = np.dot(returns_4_full_variance_annualized_inverse, np.subtract(returns_4_full_mean_annualized, (b_covi/a_covi)*ones_array))

#%%

ji_covid = alpha_covi.tolist()[0]
ji_2_covid = beta_real_covi.tolist()[0]
from sympy import *
init_printing()
var('z')

left_ineq_covi= []
right_ineq_covi = []
combo_ineq_covi = []

for k in range (0,20,1):
    lkjj = solve_univariate_inequality(alpha_covi.tolist()[k][0]+z*beta_real_covi.tolist()[k][0] < 0 , z, relational=False)
    left = round(lkjj.left, 2)
    right = round(lkjj.right, 2)
    combo = np.array([[left], [right]])
    #print(round(lkjj.right, 2))
    # left_ineq.append(round(lkjj.left, 2))
    # right_ineq.append(round(lkjj.right, 2))
    combo_ineq_covi.append(combo)
    #print(lkjj)

float_interbal_covid = []
for k in range (0,20,1):
    arr = combo_ineq_covi[k]
    float_arr = np.vstack(arr.astype(np.float))
    float_interbal_covid.append(float_arr)
    
    
#%%
float_interval_covid = []
for k in range (0,20,1):
    whatisthis = np.round(float_interbal_covid[k],4)
    float_interval_covid.append(whatisthis)

#%%
for k in range(0,20,1):
    float_interval_covid[k] = float_interval_covid[k].flatten()
 
 #%%
float_interval_covid = np.array(float_interval_covid)

#%%
# float_interval_covid = float_interval_covid.transpose(2,0,1).reshape(20,-1)   
# np.savetxt("trange_co.csv", float_interval_covid, delimiter=",")

#%%
# 4 (i)
from pandas import DataFrame
df_shortselling = DataFrame (float_interval,columns=['left', 'right'])
markets = []
for col in returns_3_full_variance_annualized.columns: 
    markets.append(col)


df_shortselling['Market'] = markets
#%%
r0 = 0.0025/250
returns_3_full_mean_annualized_tilde = np.subtract(returns_3_full_mean_annualized, r0*ones_array)

x_weight_tilde = 0.2 * np.dot(returns_3_full_variance_annualized_inverse, returns_3_full_mean_annualized_tilde)
x_weight_0 = 1- np.dot(np.transpose(x_weight_tilde), ones_array)

# np.savetxt("x_weight_tilde.csv", x_weight_tilde, delimiter=",")
# np.savetxt("x_weight_0.csv", x_weight_0, delimiter=",")


#%%
# GDP_all_countries = pd.read_csv('GDP_all_countries.csv')

#%%
# countrycode = GDP_all_countries.iloc[:,0:2]
# GDP_all_countries = GDP_all_countries.set_index(['Country Name'])


#%%

# for x in countries:
#     if x == "US":
#         x = "United States"
#     if x == "UK":
#         x = "United Kingdom"
#     if x == "Saudi":
#         x = "Saudi Arabia"
#     if x == "Russia":
#         x = "Russian Federation"
#     if x == "Korea":
#         x = "Korea, Rep."
#     new_countries.append(x)


#%%
# country_df = DataFrame (new_countries,columns=['Country'])
# country_df.to_csv("country_names.csv", index=False)

#%%
# target_firms = GDP_all_countries.loc[new_countries, :]
# for c in new_countries:    
#     if c in GDP_all_countries.index:
#         print("yes")
#     else:
#         print("No", c)

#%%
# target_firms = target_firms.reset_index()
# target_firms = target_firms.drop(columns='Country Code')
# target_firms = target_firms.T
# target_firms = target_firms.rename(columns=target_firms.iloc[0])
# target_firms = target_firms.drop(target_firms.index[0])
# target_firms = target_firms.iloc[0:]/target_firms.iloc[0] * 100

#%%
# target_firms.index = pd.to_datetime(target_firms.index, format='%Y')
#%%
# target_firms = target_firms.apply(pd.to_numeric)
# returns_gdp = pd.DataFrame()
# for indices in target_firms:
#     if indices not in returns_gdp:
#         returns_gdp[indices] = np.log(target_firms[indices]).diff()
# returns_gdp = returns_gdp.drop(returns_gdp.index[0])

#%%

# returns_gdp_mean_tran = np.array([returns_gdp.mean()])
# returns_gdp_mean = np.transpose(returns_gdp_mean_tran)
# returns_gdp_variance = returns_gdp.cov().to_numpy()
# # returns_gdp_std = (returns_gdp_variance)**0.5
# returns_gdp_variance_inverse = np.linalg.inv(returns_gdp_variance)

# # returns_3_full_std_annualized_diag = np.array(returns_3_full_std_annualized).diagonal()



#%%
# one_array_20 = np.ones((20, 1), dtype=np.int32)
# one_array_20_tran = np.transpose(one_array_20)
# a_gdp = np.dot(one_array_20_tran, returns_gdp_variance_inverse)
# a_gdp = np.dot(a_gdp, one_array_20)


# def isPSD(A, tol=1e-8):
#   E = np.linalg.eigvalsh(A)
#   return np.all(E > -tol)

# isPSD(returns_gdp_variance)

#%%
#4(iii)
quarterly = pd.read_csv('quarterly.csv')
annual_temp = pd.read_csv('annual_sau.csv')
#%%
quarterly = quarterly.set_index(['DATE'])
quarterly.index = pd.to_datetime(quarterly.index, format='%d/%m/%Y')

#%%
returns_quaterly_withoutsaud = pd.DataFrame()
for indices in quarterly:
    if indices not in returns_quaterly_withoutsaud:
        returns_quaterly_withoutsaud[indices] = np.log(quarterly[indices]).diff()
# returns_gdp = returns_gdp.drop(returns_gdp.index[0])

#%%
annual_temp = annual_temp.set_index(['DATE'])
annual_temp.index = pd.to_datetime(annual_temp.index, format='%d/%m/%Y')

#%%
returns_annualy_withoutsaud = pd.DataFrame()
for indices in annual_temp:
    if indices not in returns_annualy_withoutsaud:
        returns_annualy_withoutsaud[indices] = np.log(annual_temp[indices]).diff()
# returns_gdp = returns_gdp.drop(returns_gdp.index[0])

#%%
# returns_annualy_withoutsaud.to_csv('output.csv')

#%%
# returns_quaterly_withoutsaud.to_csv('returns_quaterly_withoutsaud.csv')

#%%
returns_quaterly_full = pd.read_csv('returns_quaterly_full.csv')
#%%
returns_quaterly_full_mean_tran = np.array([returns_quaterly_full.mean()])
returns_quaterly_full_mean = np.transpose(returns_quaterly_full_mean_tran)
returns_quaterly_full_variance = returns_quaterly_full.cov()
returns_quaterly_full_std = (returns_quaterly_full_variance)**0.5
returns_quaterly_full_variance_inv = np.linalg.inv(returns_quaterly_full_variance)
# returns_3_full_std_annualized_diag = np.array(returns_3_full_std_annualized).diagonal()

#%%
one_array_20 = np.ones((20, 1), dtype=np.int32)
one_array_20_tran = np.transpose(one_array_20)
a_gdp = np.dot(one_array_20_tran, returns_quaterly_full_variance_inv)
a_gdp = np.dot(a_gdp, one_array_20)

b_gdp = np.dot(one_array_20_tran, returns_quaterly_full_variance_inv)
b_gdp = np.dot(b_gdp, returns_quaterly_full_mean)

c_gdp = np.dot(returns_quaterly_full_mean_tran, returns_quaterly_full_variance_inv)
c_gdp = np.dot(c_gdp, returns_quaterly_full_mean)

d_gdp = np.dot(a_gdp, c_gdp)
d_gdp = np.subtract(d_gdp, b_gdp**2)

alpha_gdp = (1/a_gdp) * returns_quaterly_full_variance_inv
alpha_gdp = np.dot(alpha_gdp, one_array_20)

beta_gdp = np.dot(returns_quaterly_full_variance_inv, np.subtract(returns_quaterly_full_mean, (b_gdp/a_gdp)*one_array_20))

x_weight_gdp =np.add(alpha_gdp, beta_gdp * 0.2)

r_quarterly = r0*250/4

miu_gdp = (c_gdp-b_gdp*r_quarterly)/(b_gdp-a_gdp*r_quarterly)
sigma_gdp_0 = np.sqrt(((a_gdp/d_gdp) *(r_quarterly - b_gdp/a_gdp)**2 + 1/a_gdp))

sigma_gdp = (sigma_gdp_0*np.sqrt(d_gdp))/(b_gdp-a_gdp*r_quarterly)

#%%
p0_sigma = 0
p0_miu = 0.0025/250

#%%
miu_tan = (c-b*r0)/(b-a*r0)
sigma_0 = np.sqrt(a/d*(r0-b/a)**2+1/a)
sigma_tan = (sigma_0*np.sqrt(d))/(b-a*r0)
t_tan = 1/(b-a*r0)

t_cml = np.transpose(np.array([np.linspace(0, 2, 100)]))
miu_cml = r0+d*sigma_0**2*t_cml
sigma_cml = (miu_cml-r0)/(sigma_0 *np.sqrt(d))

#%%
t_EF = np.transpose(np.array([np.linspace(-0.4,1, 1000)]))
hundred_ones = np.ones( (100, 1), dtype=np.int32 )
miu_cml_EF = r0+d*sigma_0**2*t_EF
sigma_EF = np.sqrt((a*miu_cml_EF**2-2*b*miu_cml_EF+c)/d)

t_EF_red = np.transpose(np.array([np.linspace(0,1, 1000)]))
miu_cml_EF_red = r0+d*sigma_0**2*t_EF_red
sigma_EF_red = np.sqrt((a*miu_cml_EF_red**2-2*b*miu_cml_EF_red+c)/d)


#%%
#5
plt.scatter(p0_sigma, p0_miu, label ="$P_0$")
plt.plot(sigma_cml, miu_cml, label = "CML")
plt.plot(sigma_EF, miu_cml_EF, label = "MVF")
plt.plot(sigma_EF_red, miu_cml_EF_red, color = 'red', label = "EF")
plt.scatter(sigma_tan, miu_tan, label = "Tangency portfolio")
plt.plot(sigma_gdp/(250/4), miu_gdp/(250/4),'^', label = "Market Portfolio")
plt.legend()
plt.ylabel(r'$\mu$')
plt.xlabel(r'$\sigma$')
# plt.savefig('CML_other.png')


# plt.scatter(returns_3_full_std_annualized_diag, returns_3_full_mean_annualized)
# plt.plot(std_portfolio, miu_portfolio)
# plt.plot(sigma, mu_comp)
# #plt.plot(np.sqrt(varian), miu_star,'^')
# plt.scatter(p_vol, p_miustar)
# plt.title("Unrestricted feasible set")

# #plt.legend("optimal portfolio") repectively rename
# plt.show()

#%%
investment_100 = np.round(x_weight_tilde*100, 1)
# np.savetxt("investment_100.csv", investment_100, delimiter=",")
#%%
t_sml = np.transpose(np.array([np.linspace(-2, 2, 100)]))
# miu_market = (b+d*t_sml/a)
miu_market = miu_gdp/(250/4)
beta_sml = (r0 + d*sigma_0**2*t_sml-r0)/(miu_market-r0)
# miu_sml = np.add(r0*t_sml, np.dot(np.transpose(beta_sml), np.subtract(miu_market, r0*t_sml)))

#%%
beta_countries=[]
for k in range(0,20,1):
    beta = (returns_3_full_mean_annualized[k]-r0)/(miu_gdp/(250/4)-r0)
    beta_countries.extend([beta])
    
beta_countries = np.array(beta_countries)
beta_countries = beta_countries.transpose(2,0,1).reshape(20,-1)
# np.savetxt("beta_countries.csv", beta_countries, delimiter=",")
# np.savetxt("returns_3_full_mean_annualized.csv", returns_3_full_mean_annualized, delimiter=",")

#%%
beta_sml = np.linspace(-1,10,20)
miu_sml = np.add(np.transpose(beta_sml * (miu_gdp/(250/4) -r0)), r0 * one_array_20)
tan_miub = (miu_tan-r0) / (miu_gdp/(250/4) -r0)
gdp_b = 1
cash_fund_b =  0
miu_star_b = (miu_star-r0) / (miu_gdp/(250/4) -r0)

#%%
#6
plt.scatter(beta_countries, returns_3_full_mean_annualized, label = "20 Countries")
plt.plot(beta_sml, miu_sml, label = "SML")
plt.scatter(tan_miub, miu_tan, color = 'red', label = "Tangency Portfolio")
plt.scatter(gdp_b, miu_gdp/(250/4), color = 'black', label = "Market Portfolio")
plt.scatter(cash_fund_b, r0, label = "Riskless Cash Fund")
plt.scatter(miu_star_b, miu_star, label = "Optimal Portfolio without riskless asset")
plt.xlabel(r'$\beta$')
plt.ylabel('Expected return')
plt.legend(loc='upper right', bbox_to_anchor=(0.57, 1), prop={'size':8})
# plt.savefig("sml.png")


# miu_market = (b+d*t_sml/a)
# beta_countries = np.subtract(, r0*one_array_20)/np.subtract(miu_gdp*one_array_20 , r0*one_array_20)

#%%
#returns_quaterly_full = returns_quaterly_full.set_index('DATE')

market_return_quaterly = np.dot(returns_quaterly_full, x_weight_gdp)
















# #%%
# covMatrix = returns_3_full.cov()
# covMatrix_inv = np.linalg.inv(covMatrix) 

# ones_array = np.ones( (20, 1), dtype=np.int32 )
# ones_array_trans = np.transpose(ones_array)
# np.shape(ones_array)

# a= np.dot(ones_array_trans, covMatrix_inv)
# a=np.dot(a, ones_array)
# a

# sss = np.array(returns_3_full)
# sss_trans = sss.transpose()
# r = sss_trans.mean(1)
# r_trans = [r]
# r = np.transpose(r_trans)
# r
# r = (r+1)**250-1

# b_med = np.dot(ones_array_trans, covMatrix_inv)
# b= np.dot(b_med, r)
# b


# c= np.dot(r_trans, (np.dot(covMatrix_inv, r)))
# c

# d = a*c-b**2
# d


# alpha = 1/a * np.dot(covMatrix_inv, ones_array)
# alpha

# med_0 = (b/a) * ones_array
# med = np.subtract(r , med_0)
# beta = np.dot(covMatrix_inv, med)


# x = alpha+beta*0.2
# np.shape(x)

# x= np.add(alpha, beta*0.2)
# x



# u_star = (b+d*0.2)/a
# u_star

# variance_star = (1+d*0.2**2)/a
# variance_star

# variance = np.var(sss_trans, axis =1 )

# variance = np.transpose([variance])
# sd = variance**(0.5)*np.sqrt(250)
# sd



# r

# assets = pd.concat([pd.DataFrame(r), pd.DataFrame(sd)], axis=1)
# assets.columns = ['Returns', 'Volatility']
# assets

# p_ret = []
# p_vol = []
# p_weights = []

# num_assets = 20
# num_portfolios = 1000

# rr = np.mean(r, axis = 1)
# for portfolio in range(num_portfolios):
#     weights = np.random.random(num_assets)
#     weights = weights/np.sum(weights)
#     p_weights.append(weights)
#     returns = np.dot(weights, rr)
#     p_ret.append(returns)
#     var = covMatrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()
#     sd = np.sqrt(var)
#     ann_sd = sd*np.sqrt(250)
#     p_vol.append(ann_sd)

# data = {'Returns':p_ret, 'Volatility':p_vol}

# for counter, symbol in enumerate(returns_3_full.columns.tolist()):
#     data[symbol+' weight'] = [w[counter] for w in p_weights]

# portfolios = pd.DataFrame(data)
# portfolios

# portfolios.plot.scatter(x='Volatility', y='Returns', grid=True)





# a_3

# a_3 = 1/a
# a_1 = a/d
# a_2 = b/a
# y=np.linspace(-0.1,0.1)
# x=np.linspace(-0.1,0.1)
# y = np.sqrt((x**2-0.0000286)*0.00619539)+0.00014524
# #y = np.sqrt(2.85548493e-05+161.41034254*(x-0.00014524)**2)
# plt.plot(y,x)
# plt.plot(y,-x)
# plt.scatter(sd, r)
# plt.show()




# plt.show()


# np.shape(points)
