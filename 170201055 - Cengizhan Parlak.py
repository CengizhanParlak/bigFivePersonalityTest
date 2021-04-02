#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import tree
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from dtreeviz.trees import dtreeviz

#Lojistik Regresyon ve Karar Ağacı kullanarak ilk 50 sütundaki sorulara verilen cevaplardan ülke tahmininin yapılması


# In[2]:


data_set = pd.read_csv('C:\\Users\\cengi\\OneDrive\\Masaüstü\\data-final.csv', sep='\t')


# In[3]:


print("Veri seti:", data_set.shape)


# In[4]:


print('Katılımcı sayısı: ', len(data_set))
data_set.head()


# In[5]:


print('Eksik değeler var mı? ', data_set.isnull().values.any())
print('Ne kadar eksik değer var? ', data_set.isnull().values.sum())
data_set.dropna(inplace=True)
print('Eksik değerlerin olduğu satırları sildikten sonraki katılımcı sayısı: ', len(data_set))


# In[6]:


start_rows = len(data_set)
data_set = data_set.replace(0, np.nan).dropna(axis=0).reset_index(drop=True)
remove_rows = start_rows - len(data_set)


# In[7]:


pos_questions = [ # pozitif sorular: karakter özelliğine + etki eder
    'EXT1','EXT3','EXT5','EXT7','EXT9',                       # 5 Dışadönüklük
    'EST1','EST3','EST5','EST6','EST7','EST8','EST9','EST10', # 8 Nevrotiklik
    'AGR2','AGR4','AGR6','AGR8','AGR9','AGR10',               # 6 Uyumluluk
    'CSN1','CSN3','CSN5','CSN7','CSN9','CSN10',               # 6 Sorumluluk
    'OPN1','OPN3','OPN5','OPN7','OPN8','OPN9','OPN10',        # 7 Deneyime Açıklık
]
neg_questions = [ # negatif sorular: karakter özelliğine - etki eder
    'EXT2','EXT4','EXT6','EXT8','EXT10', # 5 Dışadönüklük
    'EST2','EST4',                       # 2 Nevrotiklik
    'AGR1','AGR3','AGR5','AGR7',         # 4 Uyumluluk
    'CSN2','CSN4','CSN6','CSN8',         # 4 Sorumluluk
    'OPN2','OPN4','OPN6',                # 3 Deneyime Açıklık
]


# In[8]:


# Katılımcıların ülke dağılımları
countries = pd.DataFrame(data_set['country'].value_counts())
countries_5000 = countries[countries['country'] >= 5000]
plt.figure(figsize=(15,5))
sns.barplot(data=countries_5000, x=countries_5000.index, y='country')
plt.title('5000 ve Üzeri Katılımcısı Olan Ülkeler')
plt.ylabel('Katılımcılar');


# In[9]:


answer_data = data_set.iloc[:,0:50]
answer_data = answer_data.astype(int)


# In[10]:


answer_data['country'] = data_set['country']


# In[11]:


data_set[pos_questions] = data_set[pos_questions].replace({1:-2, 2:-1, 3:0, 4:1, 5:2})
data_set[neg_questions] = data_set[neg_questions].replace({1:2, 2:1, 3:0, 4:-1, 5:-2})
cols = pos_questions + neg_questions
data_set = data_set[sorted(cols)]
data_set.head()


# In[12]:


personality_traits = ["EXT", "AGR", "CSN", "EST", "OPN"]
answer_columns = [trait + str(number) for trait in personality_traits for number in range(1, 11)]
print(answer_columns)
trait_labels = ['Extroversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']

for trait in personality_traits:
    personality_traits_cols = sorted([col for col in data_set.columns if trait in col and '_E' not in col])
    data_set[trait] = data_set[personality_traits_cols].sum(axis=1)
data_set[personality_traits].head()


# In[ ]:


fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(18,9))
plt.subplots_adjust(left=None, bottom=None, right=None, top=1.3, wspace=None, hspace=None)
row = -1; col = 2
for i, (trait, label) in enumerate(zip(personality_traits, trait_labels)):
    if not i % 2:
        row += 1
    if not i % 2:
        col -= 2
    i += col
    sns.distplot(data_set[trait], ax=axs[row][i], axlabel='', kde=False, bins=40).set_title(label, pad=10)
fig.delaxes(axs[2][1])


# In[ ]:


sns.pairplot(data_set[personality_traits].rename(columns={k:v for k, v in zip(personality_traits, trait_labels)}).sample(250), diag_kind="kde", kind="reg", markers=".");


# In[ ]:


for col in answer_data.columns:
    answer_data[col] = answer_data[col].astype('category').cat.codes


# In[ ]:


corr_data = pd.DataFrame(answer_data.corr()['country'][:])


# In[ ]:


corr_data = corr_data.reset_index()


# In[ ]:


top_correlation = corr_data.sort_values('country', ascending=False).head(10)['index'].to_list()


# In[ ]:


least_correlation = corr_data.sort_values('country', ascending=False).tail(5)['index'].to_list()


# In[ ]:


correlation_data = answer_data[top_correlation+least_correlation]


# In[ ]:


target_data = answer_data['country']


# In[ ]:


var_train, var_test, res_train, res_test = train_test_split(correlation_data, target_data, test_size = 0.3)


# In[ ]:


logistic_reg = LogisticRegression(random_state=0, max_iter=100).fit(var_train, res_train)


# In[ ]:


prediction = logistic_reg.predict(var_test)


# In[ ]:


accuracy_score(res_test, prediction)


# In[ ]:


decision_tree = tree.DecisionTreeClassifier()
decision_tree = decision_tree.fit(var_train, res_train)


# In[ ]:


decision_prediction = decision_tree.predict(var_test)


# In[ ]:


accuracy_score(res_test, decision_prediction)


# In[ ]:


X = answer_data
y = answer_data['country']


# In[ ]:


regr = DecisionTreeRegressor(max_depth=4, random_state=1234)
model = regr.fit(X, y)


# In[ ]:


text_representation = tree.export_text(regr)
print(text_representation)


# In[ ]:




