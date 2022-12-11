# Import library
import pandas  as pd #Data manipulation
import numpy as np #Data manipulation
#numpy is a library that is mostly used for mathematical implimentation 
#pandas is more off towards data analysis
import matplotlib.pyplot as plt # Visualization
import seaborn as sns #Visualization
plt.rcParams['figure.figsize'] = [8,5]
#the above command is used to provide the figure size
plt.rcParams['font.size'] =14
#the above command is used to provide the fontsize
#the below command is used to provide the fontweight
plt.rcParams['font.weight']= 'bold'
plt.style.use('seaborn-whitegrid')
#the above statement may look intitmidating but is a very basic and
# dumb thing to be scare of .It is a theme that will appear at the back 
# of the plots that we construct
# Import dataset
#path ='dataset/'
df = pd.read_csv('insurance.csv')
#The above statement is used to read the csv file
print('\nNumber of rows and columns in the data set: ',df.shape)
#df.shape returns a tupple of (no. of rows,no. of columns)
print('')
#df.head(n),df.tail(n) returns the first n or last n lines of the dataset 
#if the () is emplty it returns the first 5 rows of the dataset.
df.head()
# for our visualization purpose will fit line using seaborn library only for bmi as independent variable 
#and charges as dependent variable"""
#The below statement is used to plot the vlues using the seaborn lib
sns.lmplot(x='bmi',y='charges',data=df,aspect=2,height=6)
#here aspect and height are scalar quatities .height is height of each facet in inches.
#aspect is ratio of each facet so that aspect*height gives the width of each facet in inches
plt.xlabel('Boby Mass Index$(kg/m^2)$: as Independent variable')
plt.ylabel('Insurance Charges: as Dependent variable')
plt.title('Charge Vs BMI');
df.describe()
#the above statement is a very powerful statement . it gives us the count ,mean,std,min,25%,50%,75% and max value of each parameter in the columns of the dataset
#mind it that the above data is just for columns with numerical data , not the boolean values,strings ,etc
#Now below we are looking for null vlues in the dataset
plt.figure(figsize=(12,4))#giving dimentions
sns.heatmap(df.isnull(),cbar=False,cmap='viridis',yticklabels=False)#plots rectangular data as colour encoded matrix
#cbar is optional , it tells whether to draw a colorbar
#last argument is used to denote whether to print the column names or not
plt.title('Missing value in the dataset');
# correlation plot
corr = df.corr()
sns.heatmap(corr, cmap = 'Wistia', annot= True);
#the above statement plots the correlation between the various dependant and independant variables
f= plt.figure(figsize=(12,4))

ax=f.add_subplot(121)
#121 means top left
sns.distplot(df['charges'],bins=50,color='r',ax=ax)
ax.set_title('Distribution of insurance charges')
#122 means top right
ax=f.add_subplot(122)
sns.distplot(np.log10(df['charges']),bins=40,color='b',ax=ax)
ax.set_title('Distribution of insurance charges in $log$ sacle')
ax.set_xscale('log');

f = plt.figure(figsize=(14,6))
ax = f.add_subplot(121)
sns.violinplot(x='sex', y='charges',data=df,palette='Wistia',ax=ax)
ax.set_title('Violin plot of Charges vs sex')

ax = f.add_subplot(122)
sns.violinplot(x='smoker', y='charges',data=df,palette='magma',ax=ax)
ax.set_title('Violin plot of Charges vs smoker');
#
plt.figure(figsize=(14,6))
sns.boxplot(x='children', y='charges',hue='sex',data=df,palette='rainbow')
plt.title('Box plot of charges vs children');
df.groupby('children').agg(['mean','min','max'])['charges']
plt.figure(figsize=(14,6))
sns.violinplot(x='region', y='charges',hue='sex',data=df,palette='rainbow',split=True)
plt.title('Violin plot of charges vs children');

f = plt.figure(figsize=(14,6))
ax = f.add_subplot(121)
sns.scatterplot(x='age',y='charges',data=df,palette='magma',hue='smoker',ax=ax)
ax.set_title('Scatter plot of Charges vs age')

ax = f.add_subplot(122)
sns.scatterplot(x='bmi',y='charges',data=df,palette='viridis',hue='smoker')
ax.set_title('Scatter plot of Charges vs bmi')
plt.savefig('sc.png');
plt.plot()
plt.show()
#Types and need for encoding :
#encoding is done because the machine cannot understand the categorical data, it needs to be transformed into a numerical form so that the computer can form a model over it
#1.label encoding =>refers to transforming the word labels into numerical form so that the algorithms can understand how to operate on them.
#2.one hot encoding=>is a representation of categorical variable as binary vectors.
#dummy variable trap
#usually one hot encoding is used !!
# Dummy variable
categorical_columns = ['sex','children', 'smoker', 'region']
df_encode = pd.get_dummies(data = df, prefix = 'OHE', prefix_sep='_',
               columns = categorical_columns,
               drop_first =True,
              dtype='int8')
              # Lets verify the dummay variable process
print('Columns in original data frame:\n',df.columns.values)
print('\nNumber of rows and columns in the dataset:',df.shape)
print('\nColumns in data frame after encoding dummy variable:\n',df_encode.columns.values)
print('\nNumber of rows and columns in the dataset:',df_encode.shape)
from scipy.stats import boxcox
y_bc,lam, ci= boxcox(df_encode['charges'],alpha=0.05)

#df['charges'] = y_bc  
# it did not perform better for this model, so log transform is used
## Log transform
df_encode['charges'] = np.log(df_encode['charges'])
from sklearn.model_selection import train_test_split
X = df_encode.drop('charges',axis=1) # Independet variable
y = df_encode['charges'] # dependent variable

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=23)
# Step 1: add x0 =1 to dataset
X_train_0 = np.c_[np.ones((X_train.shape[0],1)),X_train]
X_test_0 = np.c_[np.ones((X_test.shape[0],1)),X_test]

# Step2: build model
theta = np.matmul(np.linalg.inv( np.matmul(X_train_0.T,X_train_0) ), np.matmul(X_train_0.T,y_train)) 

# The parameters for linear regression model
parameter = ['theta_'+str(i) for i in range(X_train_0.shape[1])]
columns = ['intersect:x_0=1'] + list(X.columns.values)
parameter_df = pd.DataFrame({'Parameter':parameter,'Columns':columns,'theta':theta})
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train) # Note: x_0 =1 is no need to add, sklearn will take care of it.

#Parameter
sk_theta = [lin_reg.intercept_]+list(lin_reg.coef_)
parameter_df = parameter_df.join(pd.Series(sk_theta, name='Sklearn_theta'))
parameter_df

y_pred_norm =  np.matmul(X_test_0,theta)

#Evaluvation: MSE
J_mse = np.sum((y_pred_norm - y_test)**2)/ X_test_0.shape[0]

# R_square 
sse = np.sum((y_pred_norm - y_test)**2)
sst = np.sum((y_test - y_test.mean())**2)
R_square = 1 - (sse/sst)
print('The Mean Square Error(MSE) or J(theta) is: ',J_mse)
print('R square obtain for normal equation method is :',R_square)

# sklearn regression module
y_pred_sk = lin_reg.predict(X_test)

#Evaluvation: MSE
from sklearn.metrics import mean_squared_error
J_mse_sk = mean_squared_error(y_pred_sk, y_test)

# R_square
R_square_sk = lin_reg.score(X_test,y_test)
print('The Mean Square Error(MSE) or J(theta) is: ',J_mse_sk)
print('R square obtain for scikit learn library is :',R_square_sk)

# Check for Linearity
f = plt.figure(figsize=(14,5))
ax = f.add_subplot(121)
sns.scatterplot(y_test,y_pred_sk,ax=ax,color='r')
ax.set_title('Check for Linearity:\n Actual Vs Predicted value')

# Check for Residual normality & mean
ax = f.add_subplot(122)
sns.distplot((y_test - y_pred_sk),ax=ax,color='b')
ax.axvline((y_test - y_pred_sk).mean(),color='k',linestyle='--')
ax.set_title('Check for Residual normality & mean: \n Residual eror');

# Check for Multivariate Normality
# Quantile-Quantile plot 
f,ax = plt.subplots(1,2,figsize=(14,6))
import scipy as sp
_,(_,_,r)= sp.stats.probplot((y_test - y_pred_sk),fit=True,plot=ax[0])
ax[0].set_title('Check for Multivariate Normality: \nQ-Q Plot')

#Check for Homoscedasticity
sns.scatterplot(y = (y_test - y_pred_sk), x= y_pred_sk, ax = ax[1],color='r') 
ax[1].set_title('Check for Homoscedasticity: \nResidual Vs Predicted');

# Check for Multicollinearity
#Variance Inflation Factor
VIF = 1/(1- R_square_sk)
VIF