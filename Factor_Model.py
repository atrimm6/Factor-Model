import numpy as np
from pandas import Series, DataFrame
import pandas as pd

#Read in csv files

df_X = pd.read_csv("X.csv")
df_r = pd.read_csv("r.csv")
df_r.index.name = 'stocks'; df_r.columns.name = 'weekly returns'

#Check dataframes

print df_r.head()
print df_X.head()

#Fill in missing values by filling forward, then filling backward

df_r=df_r.transpose()
df_r=df_r.fillna(method="ffill")
df_r=df_r.fillna(method="bfill")
df_r=df_r.transpose()

#Check filled in dataframe

print df_r.head()

def get_weekly_returns():
	#Returns an list of the returns of the N stocks for each week
	weeks = list(df_r.columns.values)
	weekly_returns = []
	for i in weeks:
		weekly_returns.append(df_r[i].values)
	return weekly_returns

def get_reg_coeffs(r):
	#Performs linear regression for each stock
	reg_coeffs = []
	for i in range(len(r)):
		X=df_X.ix[i].values
		reg_coeffs.append(1/np.dot(X,X.T)*X*r[i])
	return reg_coeffs

def get_fs(reg_coeffs):
	#Gets the fs at time t
	fs = []
	for i in range(len(reg_coeffs[0])-1):
		X = []
		for k in range(len(reg_coeffs)):
			X.append(reg_coeffs[k][i])
		fs.append(np.mean(X))
	return fs

def get_us(reg_coeffs):
	#Gets the us at time t
	us = []
	for i in range(len(reg_coeffs)):
		us.append(reg_coeffs[i][-1])
	return us

def get_cov_mat(arr):
#Computes F_{kl}s
	covariance_matrix = []
	for i in range(len(arr)):
		ith_row = []
		for j in range(len(arr)):
			ith_row.append(np.mean(df_r.ix[i].values*df_r.ix[j].values)-np.mean(df_r.ix[i].values)*np.mean(df_r.ix[j].values))
		covariance_matrix.append(ith_row)
	covariance_matrix = np.array(covariance_matrix)
	return covariance_matrix

def get_specific_risk(arr):
#Computes \Delta_{ii}s
	specific_risk = []
	for i in range(len(arr)):
		specific_risk.append(np.mean(df_r.ix[i].values*df_r.ix[i].values)-np.mean(df_r.ix[i].values)*np.mean(df_r.ix[i].values))
	specific_risk = np.array(specific_risk)
	return specific_risk

def main():
#Computes f_{k,t} s, u_{i,t} s, F_{kl} s, and \Delta_{ii} s and stores them in csv files
	r = get_weekly_returns()
	weekly_fs = []
	weekly_us = []
	for i in range(len(r)):
		coeffs = get_reg_coeffs(r[i])
		weekly_fs.append(get_fs(coeffs))
		weekly_us.append(get_us(coeffs))
	weekly_fs = np.array(weekly_fs)
	weekly_us = np.array(weekly_us)
	df_fs = DataFrame(weekly_fs.T, columns=df_r.columns.values)
	df_us = DataFrame(weekly_us.T, columns=df_r.columns.values)
	df_Fs = DataFrame(get_cov_mat(df_fs['2009Jan02'].values))
	df_deltas = DataFrame(get_specific_risk(df_us['2009Jan02'].values))
	df_fs.to_csv('df_fs.csv')
	df_us.to_csv('df_us.csv')
	df_Fs.to_csv('df_big_Fs.csv')
	df_deltas = df_deltas.to_csv('df_deltas.csv')

if __name__ == "__main__":
	main()