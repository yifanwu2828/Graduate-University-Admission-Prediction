#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# configure pd to view all columns and rows in df.head()
pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.max_rows', None)


# In[3]:


# raw dataset or unprocessed dataset
data_path = '../Data/original_data.csv'
dataset = pd.read_csv(data_path)
## print shape of dataset with rows and columns
print(dataset.shape)
n_sample = dataset.shape[0]
# raw dataset 
dataset.head()


# ### Numerical Variables

# In[4]:


# check if dataset is balanced

# balanced 52% vs 48%
accept = len(dataset[dataset['admit']==1])
print("postive examples:", accept, round(accept/n_sample,4),)
reject = len(dataset[dataset['admit']==0])
print("negative examples:", reject, round(reject/n_sample,4))
# no missing value in target variable "admit"
print((accept+reject)/n_sample)


# In[5]:


# drop columns
dataset = dataset.drop(["userName","userProfileLink",
                        "major","specialization",
                        "program","department",
                        "toeflEssay","greA",
                       "gmatA","gmatQ","gmatV", "ugCollege",], axis=1)
dataset.head()


# In[6]:


zerogpa = dataset[dataset.cgpa==0]
zerogpa.shape


# In[7]:


termAndYears = dataset['termAndYear']
# dataset.drop(['termAndYear'])
dataset['year'] = [int(term.split(' - ')[1]) if isinstance(term, str) and len(term.split(' - '))==2 else np.nan for term in termAndYears]
dataset = dataset.drop(["termAndYear"], axis=1)


# In[8]:


import requests
import json

renamedUnis = {'Virginia Tech': 'Virginia Polytechnic Institute and State University',
               'University of Wisconsin--Madison': 'University of Wisconsin Madison',
               'Texas A&M University College Station': 'Texas A and M University College Station',
               'Stony Brook University SUNY': 'SUNY Stony Brook',
               'University at Buffalo SUNY': 'SUNY Buffalo',
               'Rutgers, The State University of New Jersey New Brunswick': 'Rutgers University New Brunswick/Piscataway',
               'Purdue University West Lafayette': 'Purdue University',
               'Ohio State University': 'Ohio State University Columbus'
              }

def get_rankings(amount, printList = False):    '''
    Returns list of top engineering graduate schools from US News rankings in order from highest to lowest ranking.
    
    params:
    amount (int): amount of top rated schools to add to list
    printList (bool): if true, prints list of rankings
    '''
    assert(isinstance(amount, int) and int>=0 and int <=218)
    assert(isinstance(printList, bool))
    
    rankings = []
    headers = {
              'authority': 'www.usnews.com',
              'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36',
              }
    iter = 1
    page = 1
    while(amount>0):        
        resp = requests.get('https://www.usnews.com/best-graduate-schools/api/search?program=top-engineering-schools&specialty=eng&_page='+str(page), headers=headers)
        page += 1
        for item in json.loads(resp.text)['data']['items']:
            if (amount > 0):
                uni = item['name'].split(' (')[0].replace('--', ' ')
                if uni in renamedUnis:
                    uni = renamedUnis[uni]
                if printList: print('{}. {}'.format(iter, uni))
                rankings.append(uni)
                iter += 1
                amount -= 1
            else:
                break
    
    return rankings

rankings = get_rankings(218, True)
notFoundTargets = []
print()
for target in dataset['univName']:
    if target not in rankings and target not in notFoundTargets:
        notFoundTargets.append(target)
        
assert(len(notFoundTargets) == 0)
sorter = dict(zip(rankings, range(1, len(rankings)+1)))
dataset['targetRank'] = dataset['univName'].map(sorter)
dataset.head(20)


# In[9]:


# GRE score conversion table
score_table = pd.read_csv('../Data/score.csv')
score_table.set_index(['old'],inplace=True)
score_table.head()


# In[10]:


def greConversion(dataset, feature):
    '''
    covert old GRE socre ot new GRE scroe
    '''
    assert isinstance (dataset, pd.DataFrame)
    assert isinstance (feature, str)
    gre_score = list(dataset[feature])
    for i in range(len(gre_score)):
        if gre_score[i] > 170:
            try:
                if feature =='greV':
                    gre_score[i]=score_table['newV'][gre_score[i]]
                elif feature == 'greQ':
                    gre_score[i]=score_table['newQ'][gre_score[i]]
            except:
                continue
    
    return gre_score


# In[11]:


# perform greConversion
dataset['greV'] = greConversion(dataset, 'greV')
dataset['greQ'] = greConversion(dataset, 'greQ')


# In[12]:


# view dataset afer greConversion 
#dataset.head(5)


# In[13]:


# sanity check. we are not remove too many samples
a = dataset[dataset['greV'] > 170]
b = dataset[dataset['greQ'] > 170]
print("num of abnormal score of greV", len(a))
print("num of abnormal score of greQ", len(b))


# In[14]:


# Filter out candidates with greV or greQ > 170 after gre_Conversion
dataset = dataset[(dataset['greV'] <= 170) & (dataset['greV'] >= 130)]
dataset = dataset[(dataset['greQ'] <= 170) & (dataset['greQ'] >= 130)]
print(dataset.shape)
print("remain data percentage:",dataset.shape[0]/n_sample)


# In[15]:


dataset['greQ'].describe()


# In[16]:


##################################################################################################################


# In[17]:


def cgpa_conversion(dataset, cgpa="cgpa", cgpaScale="topperCgpa"):
    '''
    convert cgpa wrt topperCgpa
    :return: cgpa
    :type list
    '''
    assert isinstance (dataset, pd.DataFrame)
    assert isinstance (cgpa, str)
    assert isinstance (cgpaScale, str)
    cgpa = dataset[cgpa].tolist()
    cgpaScale = dataset[cgpaScale].tolist()
    for i in range(len(cgpa)):
        if cgpaScale[i] != 0 and cgpaScale[i] is not np.nan: 
            cgpa[i] = cgpa[i]/ cgpaScale[i]
            # Ratio should <=1, otherwise set to nan
            if cgpa[i] >1:
                cgpa[i]= np.nan
            elif cgpa[i]==0:
                cgpa[i]= np.nan
                
        else:
            cgpa[i]= np.nan
            
    return cgpa


# In[18]:


dataset['cgpaRatio'] = cgpa_conversion(dataset, cgpa="cgpa", cgpaScale="topperCgpa")

# drop cgpa and topperCgpa using cgpaRatio instead
dataset = dataset.drop(['cgpa','topperCgpa','cgpaScale'],axis =1)


# In[19]:


print(len(dataset))
# percent of data remain after filter gre and cgpu.
# Note we do not drop any value during cgpa_conversion
print(len(dataset)/n_sample)
#dataset.head()


# In[20]:


# Deal with journalPubs
dataset.journalPubs.unique()


# In[21]:


# element in dataset['journalPubs'] has different type
dataset['journalPubs'] = dataset['journalPubs'].astype(str)
for i, element, in enumerate(dataset['journalPubs']):
    #print(type(element))
    #print(i)
    if len(element)>3:
        dataset['journalPubs'].iloc[i] = 0


# In[22]:


# convert to int64
dataset['journalPubs'] = dataset['journalPubs'].astype('int64') 
dataset.journalPubs.unique()


# In[23]:


#############################################################################################################


# In[24]:


# Deal with confPubs
dataset.confPubs.unique()


# In[25]:


dataset['confPubs'] = dataset['confPubs'].astype(str)
for i, element, in enumerate(dataset['confPubs']):
    
    if len(str(element))>3:
        dataset['confPubs'].iloc[i] = 0


# In[26]:


# convert to int64
dataset['confPubs'] = dataset['confPubs'].astype('int64')
dataset.confPubs.unique()


# In[27]:


#############################################################################################################


# In[28]:


dataset = dataset.drop(["toeflScore"], axis=1)


# In[29]:


print(dataset.isnull().any())
print('\n')
print(dataset.isnull().sum())


# In[30]:


################################################################################################################


# ### Missing Values in Numerical Features

# In[31]:


# make a list of features which has missing values
features_with_na=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 and dataset[feature].dtypes!='O']
# print the feature name and the percentage of missing values
for feature in features_with_na:
    print(feature, '\t\t',np.round(dataset[feature].isnull().mean(), 4) *100, '\t','% missing values')


# In[32]:


dataset["cgpaRatio"].fillna(dataset["cgpaRatio"].mean(),inplace=True)
dataset["internExp"].fillna(dataset["internExp"].mean(),inplace=True)


# In[33]:


dataset.describe()


# In[34]:


print(dataset.shape)
print(dataset.shape[0]/n_sample)
print('\n')
print(dataset.isnull().any())


# In[35]:


########################################################################################################


# ### Categorical Features


# In[36]:


# rearrange columns 
dataset = dataset[['cgpaRatio',
                   'greV','greQ',
                   'researchExp', 'industryExp','internExp',
                   'journalPubs','confPubs',
                   'univName',
                   'year', 'targetRank',
                   'admit'
                  ]]
print(dataset.shape)
dataset.head()


# In[37]:


#####################################################################################################################


# ### sanity check

# In[38]:


dataset.columns


# In[39]:


dataset.info()


# In[40]:


error = dataset[dataset.cgpaRatio<0.1]
print(error.shape[0])
error.head(20)


# In[41]:


# cgpaRatio

num_unique = len(dataset.cgpaRatio.unique()) 

for i in dataset.cgpaRatio.unique():
    assert isinstance(i,float)
    assert 0<i<=1
# print(num_unique)
# print(dataset.cgpaRatio.unique())


# In[42]:


# greV
num_unique = len(dataset.greV.unique())
for i in dataset.greV.unique():
    assert isinstance(i,float)
    assert 130<=i<=170
# print(num_unique)    
# print(sorted(dataset.greV.unique().astype('int16')))


# In[43]:


# greQ
num_unique = len(dataset.greQ.unique())
for i in dataset.greQ.unique():
    assert isinstance(i,float)
    assert 130<=i<=170
# print(num_unique)    
# print(sorted(dataset.greQ.unique().astype('int16')))


# In[44]:


# researchExp
num_unique = len(dataset.researchExp.unique())
for i in dataset.researchExp.unique():
    assert(isinstance(i,np.int64))
    assert i>=0
# print(num_unique)    
# print(sorted(dataset.researchExp.unique().astype('int16')))


# In[45]:


# industryExp
num_unique = len(dataset.industryExp.unique())
for i in dataset.industryExp.unique():
    assert(isinstance(i,np.int64))
    assert i>=0
# print(num_unique)    
# print(sorted(dataset.industryExp.unique().astype('int16')))


# In[46]:


# internExp
num_unique = len(dataset.internExp.unique())
for i in dataset.internExp.unique():
    assert(isinstance(i,np.float64))
    assert i>=0
# print(num_unique)    
# print(sorted(dataset.internExp.unique().astype('int16')))


# In[47]:


# journalPubs
num_unique = len(dataset.journalPubs.unique())
for i in dataset.journalPubs.unique():
    assert(isinstance(i,np.int64))
    assert i>=0
# print(num_unique)    
# print(sorted(dataset.journalPubs.unique().astype('int16')))


# In[48]:


# confPubs
num_unique = len(dataset.confPubs.unique())
for i in dataset.confPubs.unique():
    assert(isinstance(i,np.int64))
    assert i>=0
# print(num_unique)    
# print(sorted(dataset.confPubs.unique().astype('int16')))


# In[49]:


# targetRank
num_unique = len(dataset.targetRank.unique())
for i in dataset.targetRank.unique():
    assert(isinstance(i,np.int64))
    assert i>=0
# print(num_unique)    
# print(sorted(dataset.targetRank.unique().astype('int16')))


# In[50]:


dataset["year"]= dataset['year'][(dataset['year']>1990) & (dataset['year']<2020)]


# In[51]:


dataset["year"].fillna(dataset["year"].median(), inplace=True)


# In[52]:


# year
num_unique = len(dataset.year.unique())
# for i in dataset.year.unique():
#     assert(isinstance(i,np.int64))
#     assert i>=0
# print(num_unique)    
# print(dataset.year.unique())


# In[53]:


dataset.isnull().any()


# In[54]:


dataset.info()


# In[55]:


dataset.describe()


# In[56]:


print("dataset shape:",dataset.shape)
print("%data_remaining:",dataset.shape[0]/n_sample*100)
dataset.head()


# ## Output to .csv

# In[57]:


SAVE_CSV = False
if SAVE_CSV:
    dataset.to_csv('clean_data.csv', index=False)

