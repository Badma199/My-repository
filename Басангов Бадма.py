#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


# In[ ]:


n = 100_000_000
np.random.seed(0)
np.random.seed(0)
data = {
    'numeric': np.random.rand(n),  
    # Числовые данные
    'datetime': [datetime.now() - timedelta(days=random.randint(0, 365*3)) for _ in range(n)],  
    # Дата в диапазоне 3 лет
    'string': np.random.choice(['apple', 'banana', 'cherry', 'date', 'empty'], n)  
}

df = pd.DataFrame(data)


# In[ ]:


df = pd.concat([df, df.sample(frac=0.1, random_state=1)]).reset_index(drop=True)

# Сохраняем в CSV
df.to_csv('dataset.csv', index=False)


# In[ ]:


def remove_na(df):
    return df.dropna()

def remove_duplicates(df):
    return df.drop_duplicates()

def clean_strings(df):
    df['string'] = df['string'].apply(lambda x: np.nan if not any(char.isdigit() for char in x) else x)
    return df

def remove_time_range(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df[~df['datetime'].dt.hour.isin([1, 2, 3])]

# Обработанные данные
df.to_csv('cleaned_dataset.csv', index=False)


# In[ ]:




df['hour'] = df['datetime'].dt.hour


# In[ ]:


# Расчет метрик
metrics = df.groupby('hour').agg(
    unique_strings=('string', 'nunique'),
    mean_numeric=('numeric', 'mean'),
    median_numeric=('numeric', 'median')
).reset_index()

print(metrics)


# In[ ]:


df = df.merge(metrics, on='hour', how='left')


# In[ ]:


df


# In[ ]:


#### Гистограмма и 95% доверительный интервал:
import matplotlib.pyplot as plt
from scipy import stats

# Гистограмма
plt.hist(df['numeric'], bins=50)
plt.title('Гистограмма для numeric')
plt.xlabel('Значение')
plt.ylabel('Частота')
plt.show()

# 95% доверительный интервал
mean = df['numeric'].mean()
sem = stats.sem(df['numeric'])  # стандартная ошибка среднего
conf_interval = stats.norm.interval(0.95, loc=mean, scale=sem)
print(f"95% доверительный интервал: {conf_interval}")


# In[ ]:


#### Среднее значение по месяцам:
df['month'] = df['datetime'].dt.to_period('M')

monthly_mean = df.groupby('month')['numeric'].mean().reset_index()

plt.plot(monthly_mean['month'].astype(str), monthly_mean['numeric'])
plt.title('Среднее значение numeric по месяцам')
plt.xlabel('Месяц')
plt.ylabel('Среднее значение')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


import seaborn as sns
from collections import Counter

# Частотность символов в строках
string_series = df['string'].dropna()


# In[ ]:




