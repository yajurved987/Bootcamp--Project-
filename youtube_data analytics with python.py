#!/usr/bin/env python
# coding: utf-8

# # Importing Needed Packages

# In[58]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams['figure.figsize'] = 10,8


# # Reading Data from Dataset (train.csv)

# In[59]:


df =pd.read_csv("Downloads/train.csv")


# In[60]:


df


# In[61]:


df.head()  #it shows the first 5 rows


# # Data Exploration

# In[62]:


# checking no of rows and columns
df.shape 


# In[63]:


#checking  given what type of data of the columns
df.info()   


# In[64]:


# summarize the data
df.describe()


# In[65]:


#checking  whether it contains null values in the given data
df.isnull().any() 


# In[66]:


#generating  the histograms to analyse the data
import seaborn as sns
sns.set()
df.hist(bins = 10, figsize = (10,10), grid = True); 
plt.show() 


# In[67]:


df.describe().columns #it gives the continuous type of columns


# In[68]:


df.isnull().sum()


# In[69]:


x =  df.isnull().sum()
drop_col = x[x>(35/100 * df.shape[0])]
drop_col


# In[70]:


drop_col.index


# In[71]:


df["Embarked"].describe()


# In[72]:


df["Embarked"].fillna("s",inplace=True)


# In[73]:


df['Age'] = df['Age'].fillna(df['Age'].mean())


# In[74]:


df.isnull().sum()


# In[75]:


#relation between the columns in the given data
df.corr()   


# sibsp: Number of siblings/spouse Aboard
# 
# parch : Number of parents /children Aboard
# 
# so we can make a new column family_size by combining these two columns

# In[76]:


#boxplotlib - it helps to understand the relation between the data
import seaborn as sb
sb.heatmap(df[['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']])
plt.show()


# In[77]:


df["familySize"] = df['SibSp']+df["Parch"]
df.drop(['SibSp','Parch'],axis =1, inplace = True)
df.corr()


# FamilySize is the ship doesnot have much coorelance with survival rate
# 
# Lets check if we wheather the person was alone or not can affect the survival rate

# In[78]:


df['alone'] = [0 if df["familySize"][i]>0 else 1 for i in df.index] 
df.head()


# In[79]:


df.groupby(['alone'])['Survived'].mean()


# If the person is alone he /she has less chance of surviving

# the reason might be the person who is travelling with his family might be beloning to rich class and might be prioritized over other.

# In[80]:


df[['alone','Fare']].corr()


# so we can see if the person was not alone , the change the ticket price is high.

# In[81]:


df['Sex']=[0 if df["Sex"][i]=='male'   else 1 for i in df.index] #1 for F , 0 for Male
df.groupby(["Sex"])["Survived"].mean()

it shows women were prioritized over men 
# In[82]:


df.groupby(['Embarked'])['Survived'].mean()


# CONCLUSIONS
# 
# Female passengers were prioritized over men
# 
# People with high class or rich people have higher survival rate than others. The hierarichy might have been followed while savings the passengers.
# 
# Passengers travelling with their family have higher survival rate.
# 
# Passengers who boarded the ship at Cherbourg, survived more in proportion then the others.

# In[88]:


sb.barplot( x=df["Pclass"],y=df["Survived"])


# the more no of 1st class people servived

# In[92]:


sb.boxplot( data =df,x=df["Pclass"],y=df["Age"])


# here we can see the outliers in 2 and 3 class

# In[ ]:




