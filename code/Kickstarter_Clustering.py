#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 20:26:21 2021

@author: hiradnourbakhsh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 13:09:33 2021

@author: hiradnourbakhsh
"""

####### Part II: Unsupervised Learning (Clustering Model) ###################

# importing dataset 

df = pd.read_excel('/Users/hiradnourbakhsh/Desktop/INSY 662/Individual Project/Kickstarter.xlsx')

# visualizing data
df.count()

df.columns

############# Data Preprocessing #################

# drop column which has almost entirely all missing values

df = df.drop(columns = ['launch_to_state_change_days'])

# removing disable_communication (unary variable)

df = df.drop(columns = ['disable_communication'])

# removing states other than successful or failed

df = df[(df.state != 'canceled') & (df.state != 'live') & (df.state != 'suspended')]

# remove deadline, state_changed_at, created_at, and launched_at variables
df = df.drop(columns = ['deadline'])
df = df.drop(columns = ['state_changed_at'])
df = df.drop(columns = ['created_at'])
df = df.drop(columns = ['launched_at'])

# remove project_id and name
df=  df.drop(columns = ['project_id', 'name'])

########### Building clustering Model: static_usd_rate and backers_count ####################

list(df.columns)

X = df[['static_usd_rate', 'backers_count']]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 5)
model = kmeans.fit(X_std)
labels = model.predict(X_std)

from matplotlib import pyplot

pyplot.scatter(df['static_usd_rate'], df['backers_count'], c = labels, cmap = 'rainbow')

############### Elbow Method: Model Inertia Graph ################

X = df[['static_usd_rate', 'backers_count']]

from sklearn.cluster import KMeans
withinss = []
for i in range(2,8):
    kmeans = KMeans(n_clusters = i)
    model = kmeans.fit(X)
    withinss.append(model.inertia_)
    
from matplotlib import pyplot as plt
plt.plot([2,3,4,5,6,7], withinss)

# decrease in inertia becomes insignificant at 5 clusters

#################### Silhouette Score: Individual Clusters ############################3

import numpy as np
    
X = df[['static_usd_rate', 'backers_count']]

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 5)
model = kmeans.fit(X)
labels = model.labels_

from sklearn.metrics import silhouette_samples
silhouette = silhouette_samples(X, labels)

sdf = pd.DataFrame({'label': labels, 'silhouette': silhouette})

print('Average Silhouette Score for Cluster 0: ',np.average(sdf[sdf['label'] == 0].silhouette))

print('Average Silhouette Score for Cluster 1: ',np.average(sdf[sdf['label'] == 1].silhouette))

print('Average Silhouette Score for Cluster 2: ',np.average(sdf[sdf['label'] == 2].silhouette))

print('Average Silhouette Score for Cluster 3: ',np.average(sdf[sdf['label'] == 3].silhouette))

print('Average Silhouette Score for Cluster 4: ',np.average(sdf[sdf['label'] == 4].silhouette))


################ Calculating average silhouette score of entire dataset ###################

from sklearn.metrics import silhouette_score
silhouette_score(X, labels)

# Optimal K
# model with n_clusters = 2 generates best average silhouette score
# however, we will use 5 clusters because it represents reality of data better according to plot
from sklearn.metrics import silhouette_score
for i in range (2,8):    
    kmeans = KMeans(n_clusters=i)
    model = kmeans.fit(X)
    labels = model.labels_
    print(i,':',np.average(silhouette_score(X,labels)))

#################### Pseudo F statistic #########################

X = df[['static_usd_rate', 'backers_count']]

from sklearn.cluster import KMeans

for i in range(2,6):
    kmeans = KMeans(n_clusters=i)
    model = kmeans.fit(X)
    labels = model.labels_
        
    from sklearn.metrics import calinski_harabasz_score
    score = calinski_harabasz_score(X, labels)
    score
        
    from scipy.stats import f
    df1 = 3 # df1 = k-1
    df2 = 31366 # df2 = n-k
    pvalue = 1-f.cdf(score, df1, df2)
    print('k = ', i, '\n p-value = ', pvalue)

# n_clusters = 2 to 5 yield the exact same p value 1.1102230246251565e-16


