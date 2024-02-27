#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy.stats as sstats
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from xgboost import XGBClassifier


# In[2]:


results0 = {}
results1 = {}


# In[3]:


df = pd.read_csv("C:\\Users\\Lenovo\\Downloads\\disabled_C.csv")
total_records = df.shape[0]
total_records
df.info()


# In[4]:


missing = df.isnull().sum()
percent = missing / total_records
pd.concat([missing.sort_values(ascending=False), percent.sort_values(
          ascending=False)], axis=1, keys=['Missing', 'Percent of total records'])


# In[5]:


df.drop(columns=["comments", "state"], inplace=True)


# In[6]:


df['children_disabled'].value_counts(dropna=False)


# In[7]:


df['children_disabled'].replace([None], 'Don\'t know', inplace=True)
df['children_disabled'].value_counts(dropna=False)


# In[8]:


df['self_employed'].value_counts(dropna=False)


# In[9]:


df.drop(df[df['self_employed'].isnull()].index, inplace=True)
df['self_employed'].value_counts(dropna=False)


# In[10]:


total = df['Country'].value_counts()[:10]
percent_of_all = total / total_records
pd.concat([total, percent_of_all.sort_values(ascending=False)],
          axis=1, keys=['No of records', 'Percent of total'])


# In[11]:


import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("C:\\Users\\Lenovo\\Downloads\\disabled_C.csv")
total = df['Country'].value_counts()[:10]
plt.figure(figsize=(22,5))
total = total.to_frame()
print(total)
total.plot.bar() 


# In[12]:


df.drop(columns=["Country"], inplace=True)


# In[13]:


df.drop(columns=["Timestamp"], inplace=True)


# In[14]:


df['Age'].unique()


# In[15]:


print("Total number of records with age < 0 or > 100 is: " +
              str(len(list(filter(lambda x: x < 0 or x > 100, df['Age'].tolist())))))


# In[16]:


df.drop(df[df['Age'] < 0].index, inplace=True)
df.drop(df[df['Age'] > 100].index, inplace=True)

plt.figure(figsize=(10,5))
sns.histplot(df["Age"], kde=True)
plt.title("Distribuition of Age")
plt.xlabel("Age")


# In[17]:


df['Gender'].unique()


# In[18]:


df['Gender'].replace(['Female', 'female', 'Cis Female', 'Woman', 'Femake', 'woman', 'Female ',
                     'cis-female/femme', 'Female (cis)', 'femail', 'f', 'F'],
                     "female", inplace=True)

df['Gender'].replace(['M', 'm', 'Male', 'male', 'maile', 'Cis Male', 'Mal', 'Male (CIS)',
                     'Make', 'Male ', 'Man', 'msle', 'Mail', 'cis male', 'Malr', 'Cis Man'],
                     "male", inplace=True)

df['Gender'].replace(['Trans woman', 'Female (trans)', 'Male-ish', 'Trans-female', 'something kinda male?',
                          'queer/she/they', 'non-binary', 'Nah', 'Enby', 'fluid', 'Genderqueer', 'Androgyne',
                          'Agender', 'Guy (-ish) ^_^', 'male leaning androgynous', 'Neuter', 'queer',
                          'A little about you', 'ostensibly male, unsure what that really means'],
                         "queer/other", inplace=True)
df['Gender'].value_counts()


# In[19]:


for column in df:
    df[column] = df[column].astype('category').cat.codes


# In[20]:


normalized_df = df

for column in df:
    c = df[column]
    normalized_df[column] = (c - c.min(axis=0)) / c.max(axis=0)


# In[21]:


def split_data(question_no, df, n_train):
    features = ['Age', 'Gender', 'self_employed','type_diablity', 'children_disabled','family_history',
                'work_interfere', 'no_employees', 'remote_work',
                'tech_company', 'benefits', 'care_options', 'wellness_program',
                'seek_help', 'anonymity', 'leave', 'phys_health_consequence',
                'coworkers', 'supervisor', 'mental_health_interview',
                'phys_health_interview', 'mental_vs_physical',
                'obs_consequence']

    y_label = None

    # Should you seek help
    if question_no == 0:
        features.append('mental_health_consequence')
        y_label = 'treatment'

    # Is this good workplace for my mental health
    elif question_no == 1:
        features.append('treatment')
        y_label = 'mental_health_consequence'

    X = df[features].to_numpy()
    y = df[y_label].to_numpy()

    no_records = len(y)
    n_train = int(no_records * 0.85)
    X_train, X_test, y_train, y_test = X[:n_train], \
        X[n_train + 1:], y[:n_train], y[n_train + 1:]

    return {"X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test}


# In[22]:


# calc correlation matrix
corr = df.corr()

# display correlation matrix
plt.figure(figsize=(10,10))
sns.heatmap(corr, vmax=1, square=True)
plt.show()


# In[41]:


question_no = 0
data_after_split = split_data(question_no, df, 0)

X_train = torch.tensor(data_after_split["X_train"])
print("this is x_train",X_train.shape)
y_train = torch.tensor(data_after_split["y_train"])
X_test = torch.tensor(data_after_split["X_test"])
y_test = torch.tensor(data_after_split["y_test"])


# In[42]:


class Regression_model(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Regression_model, self).__init__()
        self.linear = nn.Linear(
            input_shape,
            output_shape,
            dtype=torch.float64)

    def forward(self, x):
        if question_no == 0:
            y_pred = torch.sigmoid(self.linear(x))
        else:
            y_pred = self.linear(x)
        return y_pred

    def predict(self, x):
        preds = self.forward(x)
        if (question_no == 0):
            return torch.round(preds).flatten()
        else:
            return torch.argmax(preds, dim=1)


# In[53]:


def calc_accuracy(model):
    predictions = model.predict(X_test)
    print (predictions)
    if question_no == 0:
        sum = torch.sum(predictions == y_test)
    else:
        sum = torch.sum(predictions == y_test*2)

    accuracy = sum/X_test.shape[0]
    return accuracy



# In[44]:


epochs = 20000
input_dim = 24 #this was 22 before, but we have 24 columns or
# the size of X_train is [1065,24]
if question_no == 0:
    output_dim = 1  # Single binary output
else:
    output_dim = 3 # three outputs for second question

lrs = [0.001, 0.01, 0.02, 0.1]  # learning rates


# In[45]:


model = Regression_model(input_dim, output_dim)
criterion = torch.nn.CrossEntropyLoss()


# In[46]:


def calc_logistic_regression(no_question):
    accs = {}
    for lr in lrs:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        for epoch in range(epochs):
            x = X_train
            labels = y_train
            optimizer.zero_grad()
            outputs = model(x)
            if question_no == 0:
                #loss = criterion(outputs.squeeze(), labels.unsqueeze(1))
                loss = criterion(torch.squeeze(outputs), labels)
            else:
                # labels must be type Long
                #loss = criterion(outputs.squeeze(), labels)
                loss = criterion(torch.squeeze(outputs), (labels*2).long())
            # exit(0)

            loss.backward()
            optimizer.step()

            # if epoch % 2500 == 0:
            #     print(f'{epoch=}', "loss:", loss.item())
        acc = calc_accuracy(model).item()
        print("learning rate:", lr, "accuracy:", round(acc * 100, 3), '%')
        accs[lr] = acc
    return max(accs.values())


# In[48]:


best_acc = calc_logistic_regression(question_no)


# In[33]:


question_no = 1
data_after_split = split_data(question_no, df, 0)

X_train = torch.tensor(data_after_split["X_train"])
y_train = torch.tensor(data_after_split["y_train"])
X_test = torch.tensor(data_after_split["X_test"])
y_test = torch.tensor(data_after_split["y_test"])


# In[34]:


if question_no == 0:
    output_dim = 1  # Single binary output
else:
    output_dim = 3 # three outputs for second question


# In[36]:


model = Regression_model(input_dim, output_dim)


# In[37]:


best_acc = calc_logistic_regression(question_no)


# In[38]:


results1["LogisticRegression"] = best_acc

