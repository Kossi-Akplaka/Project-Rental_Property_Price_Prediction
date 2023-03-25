#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set the number of columns the notebook can display
pd.set_option('display.max_columns', 200)


# # Load the data

# In[2]:


df = pd.read_csv(r"C:\Users\akpla\Downloads\archive (6)\housing.csv")


# In[3]:


df.head(5)


# # Data Understanding

# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df.region.value_counts()


# In[7]:


df.state.value_counts().head(5)


# In[8]:


df.type.unique()


# In[9]:


# Drop features 
df1=df.drop(['id', 'url',
    #'region', 
    'region_url', 
    #'price', 'type', 'sqfeet', 'beds',
    #   'baths', 'cats_allowed', 'dogs_allowed',
    'smoking_allowed',
       'wheelchair_access',
    'electric_vehicle_charge', 
    #'comes_furnished',
       'laundry_options', 
    #'parking_options',
    'image_url', 'description', 'lat',
       'long',
             #'state'
], axis = 'columns').copy()


# In[10]:


df1[10000:10005]


# In[11]:


df1.parking_options.unique()


# # Data Cleaning

# In[12]:


df1.dtypes


# In[13]:


df2 = df1[['region', 'type', 'sqfeet', 'beds', 'baths', 'cats_allowed',
       'dogs_allowed', 'comes_furnished', 'parking_options', 'state', 'price']].copy()


# In[14]:


# Check for NA values
df2.isna().sum()


# In[15]:


df2['parking_options']=df2.parking_options.fillna('no parking')
df2.isna().sum()


# # Features Engineering

# In[16]:


# create a new column pets_allowed in a pandas DataFrame that takes the value 1 if both dogs_allowed and cats_allowed are 1
df2['pets_allowed'] = (df2['cats_allowed'] & df2['dogs_allowed']).astype(int)
df3 = df2.drop(['cats_allowed', 'dogs_allowed'], axis = 'columns')


# In[17]:


df3.head(10)


# In[18]:


df3.parking_options.unique()


# In[19]:


is_a_parking = ['carport', 'attached garage', 'off-street parking', 'detached garage', 'valet parking']
df3.parking_options=df3.parking_options.apply(lambda x: 1 if x in is_a_parking else 0)


# In[20]:


df3.head(10)


# In[21]:


df3.region.nunique()


# In[22]:


# cleaning the region variable
'''
name after foward slash
2 capital letters after space
'''
def region_foward_slash(x):
    tokens = x.split('/')
    return tokens[0].strip()
df3.region = df3.region.apply(lambda x: region_foward_slash(x))


# In[23]:


def strip_last_two_uppercase_chars(string):
    if string[-2:].isupper():
        return string[:-2].strip()
    return(string)
df3.region = df3.region.apply(lambda x: strip_last_two_uppercase_chars(x))


# In[24]:


df3.region= df3.region.apply(lambda x: 'west virginia' if x=='west virginia (old)' else x)
df3.query('region=="west virginia (old)"')


# In[25]:


df3.region.unique()[:10]


# In[26]:


df.state.unique()


# # Dimensionality reduction

# In[27]:


df3.head(10)


# In[28]:


df3.type.value_counts()


# In[29]:


type_list = df3.type.value_counts()[df3.type.value_counts() < 15885 ]


# In[30]:


df3.type=df3.type.apply(lambda x:'other' if x in type_list else x)


# In[31]:


df3.type.value_counts()


# # Feature Understanding

# In[32]:


df3.describe()


# In[33]:


# Plot the numerical variables
get_ipython().run_line_magic('matplotlib', 'inline')
df3[['price','sqfeet']].hist(figsize=(20,10))
plt.plot()


# This chart shows that the price and the square footage of some rentals home are 0.
# According to zonning, The minimum square footage for a house is 120 square feet, 
# and at least one room must be habitable so we'll need to remove those outliers in the data.

# In[34]:


df3.beds.value_counts()


# In[35]:


df.baths.value_counts()


# Most home have at most 3 bedrooms and 3 bathrooms so we can remove the rentals that have more than 4 bedrooms and 4 baths
# A 4 bedroom average size is around 2000 square feet for small home and 5000 square feet for big home so we can remove rental 
# properties that have more than 5000 square feet.

# # Outliers removal

# In[36]:


# Square feet outliers removal
df3.query('sqfeet<120 or sqfeet>5000')


# In[37]:


df4 = df3.query('sqfeet>=120 and sqfeet<=5000')
df4.shape


# In[38]:


# remove home with more than 4 bedrooms and 4 bathrooms
df4.query('beds>4 or baths>4')


# In[39]:


df5 = df4.query('beds<=4 and baths<=4')
df5.shape


# In[40]:


df5.query('beds == 0 and baths ==0')


# In[41]:


# remove the price outliers by region using one standard deviation
def remove_price_outliers(df5):
    df_out = pd.DataFrame()
    for key, subdf in df5.groupby('region'):
        m = np.mean(subdf.price)
        st = np.std(subdf.price)
        reduced_df = subdf[(subdf.price>(m-st)) & (subdf.price<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out

df6=remove_price_outliers(df5)
df6.shape


# In[42]:


df6.price.describe()


# In[43]:


q_high,q_low = df6.price.quantile(0.999),df6.price.quantile(0.001)
q_high,q_low 


# - 99.8% of the price of rentals properties in the data falls between 150 and 4395. 
# - We can remove the prices that areless than 100 and more than 5000

# In[44]:


df7 = df6.query('price>100 and price<5000').copy()


# In[45]:


get_ipython().run_line_magic('matplotlib', 'inline')
df7[['price','sqfeet']].hist(figsize=(20,10))
plt.plot()


# - The price and sqfeet data are normally distributed.
# - Most rentals houses prices are between 500 and 2000

# # Features Relationship

# In[46]:


df7.parking_options = df7.parking_options.astype(int)
df7.parking_options.dtypes
df7.head()


# In[47]:


# create a new figure to hold the subplots
fig, axes = plt.subplots(nrows=len(df7['type'].unique()), figsize=(5,20))

# loop through each unique state and create a scatterplot on its own subplot
for i, (key, subdf) in enumerate(df7.groupby('type')):
    sns.scatterplot(x=subdf.sqfeet, y=subdf.price, ax=axes[i])
    axes[i].set_title(key)  # set the title to the name of the state
    
# display the plots
plt.tight_layout()
plt.show()


# # Data Correlation

# In[48]:


correlation = df7.corr()
sns.heatmap(correlation, annot=True)


# Bedrooms are Sqfeet are highly correlated, Baths and Bedrooms are also positively correlated

# # Label Encoding with OneHotEncoding

# In[49]:


df7.head()


# In[50]:


dummies1 = pd.get_dummies(df7.region)
dummies2 = pd.get_dummies(df7.type)
dummies3 = pd.get_dummies(df7.state)
df8 = pd.concat([df7,dummies1,dummies2,dummies3], axis = 'columns').drop(['region','type','state'], axis = 'columns')


# In[51]:


df8.head()


# In[52]:


df8.shape


# # Build model 

# In[53]:


# Splitting the data into features X and target variable Y
X = df8.drop('price', axis = 'columns')
y = df8.price
# Scale the features X so that no single feature dominates the learning algorithm
from sklearn.preprocessing import StandardScaler
X_scaled =StandardScaler().fit_transform(X)


# In[54]:


# Splitting the data into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size = 0.2)


# In[55]:


# Linear Regression Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train,y_train)
lr.score(X_test,y_test)


# In[56]:


# Decision Tree regressor
from sklearn.tree import DecisionTreeRegressor
d_regressor= DecisionTreeRegressor().fit(X_train, y_train)
d_regressor.score(X_test, y_test)


# # Use K-Fold to cross validate our linear model

# In[ ]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
score = cross_val_score(DecisionTreeRegressor(), X,y,cv=ShuffleSplit(n_splits=5, test_size=0.2, random_state=0))
score


# In[59]:


score.mean()


# # Find the optimal model using GridSearchCV

# This took a very long time

# In[ ]:


# import warnings filter
'''from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2]
                
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        },
        'RandomForestRegressor':{
            'model': RandomForestRegressor(),
            'params':{
                'bootstrap':['True','False']
            }
        }
        
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])
find_best_model_using_gridsearchcv(X,y)'''


# In[62]:


# Save the model from future use
import pickle
pickle.dump(d_regressor, open('model.pckl', 'wb'))


# In[ ]:





# In[ ]:




