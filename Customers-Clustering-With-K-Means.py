# K-means clustering algorithm to group customers of a retail store based on their purchase history.

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import warnings
warnings.filterwarnings('ignore')

for dirname, _, filenames in os.walk(r'C:\Users\Satoshi\Desktop\Data\prodigy-2'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

Market_Basket = pd.read_csv(r'C:\Users\Satoshi\Desktop\Data\prodigy-2Mall_Customers.csv')

print(Market_Basket.head())
print(Market_Basket.shape)

print(Market_Basket.columns)
print(Market_Basket.info())
print(Market_Basket.Gender.value_counts())

print(Market_Basket.Age.min())
print(Market_Basket[Market_Basket['Age']==18])

print(Market_Basket.Age.max())
print(Market_Basket[Market_Basket['Age']==70])

print(Market_Basket['Annual Income (k$)'].min())
print(Market_Basket[Market_Basket['Annual Income (k$)']==15])

print(Market_Basket['Annual Income (k$)'].max())
print(Market_Basket[Market_Basket['Annual Income (k$)']==137])

sns.set_theme(style="ticks", color_codes=True)
sns.color_palette("rocket")

print(sns.displot(Market_Basket, x = 'Age',hue='Gender', kind='kde'))
print(sns.displot(Market_Basket, x = 'Annual Income (k$)',hue='Gender', kind='hist'))

print(sns.catplot(x = 'Age', y='Annual Income (k$)', hue='Gender',kind='point', data=Market_Basket))


# Data Preprocessing
Market_Basket.isnull().sum()

# ### Label Encoding
Market_Basket.info()
laencoder = LabelEncoder()
Market_Basket['Gender'] = laencoder.fit_transform(Market_Basket['Gender'])

# Standard Scaler
X = Market_Basket.drop(['CustomerID'], axis=1)

SC = StandardScaler()
MarkBas_X = SC.fit_transform(X)

# KMeans Model

# In this model, we have considered 6 states and obtained the number of clusters and the clustering model.

# 1. Using Gender and Spending Score

MarkBas_X_1 = MarkBas_X[:,[0,3]]

wcss_1 = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    #k-means++ is an algorithm for choosing the initial values (or "seeds") for the k-means clustering algorithm.
    kmeans.fit(MarkBas_X_1)
    wcss_1.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss_1)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS_1')
plt.show()

kmeans_1 = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans_1 = kmeans_1.fit_predict(MarkBas_X_1)


plt.scatter(MarkBas_X_1[y_kmeans_1 == 0, 0], MarkBas_X_1[y_kmeans_1 == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(MarkBas_X_1[y_kmeans_1 == 1, 0], MarkBas_X_1[y_kmeans_1 == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(MarkBas_X_1[y_kmeans_1 == 2, 0], MarkBas_X_1[y_kmeans_1 == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(MarkBas_X_1[y_kmeans_1 == 3, 0], MarkBas_X_1[y_kmeans_1 == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(kmeans_1.cluster_centers_[:, 0], kmeans_1.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Gender')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


MarkBas_X_2 = MarkBas_X[:,[1,3]]

wcss_2 = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    #k-means++ is an algorithm for choosing the initial values (or "seeds") for the k-means clustering algorithm.
    kmeans.fit(MarkBas_X_2)
    wcss_2.append(kmeans.inertia_)


plt.plot(range(1, 11), wcss_2)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS_2')
plt.show()


kmeans_2 = KMeans(n_clusters = 6, init = 'k-means++', random_state = 42)
y_kmeans_2 = kmeans_2.fit_predict(MarkBas_X_2)


plt.scatter(MarkBas_X_2[y_kmeans_2 == 0, 0], MarkBas_X_2[y_kmeans_2 == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(MarkBas_X_2[y_kmeans_2 == 1, 0], MarkBas_X_2[y_kmeans_2 == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(MarkBas_X_2[y_kmeans_2 == 2, 0], MarkBas_X_2[y_kmeans_2 == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(MarkBas_X_2[y_kmeans_2 == 3, 0], MarkBas_X_2[y_kmeans_2 == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(MarkBas_X_2[y_kmeans_2 == 4, 0], MarkBas_X_2[y_kmeans_2 == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(MarkBas_X_2[y_kmeans_2 == 5, 0], MarkBas_X_2[y_kmeans_2 == 5, 1], s = 100, c = 'black', label = 'Cluster 6')
plt.scatter(kmeans_2.cluster_centers_[:, 0], kmeans_2.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# 3. Using Annual Income and Spending Score

MarkBas_X_3 = MarkBas_X[:,[2,3]]

wcss_3 = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    #k-means++ is an algorithm for choosing the initial values (or "seeds") for the k-means clustering algorithm.
    kmeans.fit(MarkBas_X_3)
    wcss_3.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss_3)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS_3')
plt.show()


kmeans_3 = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans_3 = kmeans_3.fit_predict(MarkBas_X_3)


plt.scatter(MarkBas_X_3[y_kmeans_3 == 0, 0], MarkBas_X_3[y_kmeans_3 == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(MarkBas_X_3[y_kmeans_3 == 1, 0], MarkBas_X_3[y_kmeans_3 == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(MarkBas_X_3[y_kmeans_3 == 2, 0], MarkBas_X_3[y_kmeans_3 == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(MarkBas_X_3[y_kmeans_3 == 3, 0], MarkBas_X_3[y_kmeans_3 == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(MarkBas_X_3[y_kmeans_3 == 4, 0], MarkBas_X_3[y_kmeans_3 == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans_3.cluster_centers_[:, 0], kmeans_3.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# 4. Using Gender and Annual Income and Spending Score

MarkBas_X_4 = MarkBas_X[:,[0,2,3]]

wcss_4 = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    #k-means++ is an algorithm for choosing the initial values (or "seeds") for the k-means clustering algorithm.
    kmeans.fit(MarkBas_X_4)
    wcss_4.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss_4)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS_4')
plt.show()

kmeans_4 = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans_4 = kmeans_4.fit_predict(MarkBas_X_4)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting the clusters
for i in range(4):
    ax.scatter(MarkBas_X_4[y_kmeans_4 == i, 0], MarkBas_X_4[y_kmeans_4 == i, 1], MarkBas_X_4[y_kmeans_4 == i, 2], s=100, label=f'Cluster {i + 1}')

# Plotting the centroids
ax.scatter(kmeans_4.cluster_centers_[:, 0], kmeans_4.cluster_centers_[:, 1], kmeans_4.cluster_centers_[:, 2],
           s=300, c='yellow', label='Centroids')

ax.set_title('Clusters of customers')
ax.set_xlabel('Gender')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
ax.legend()

plt.show()


# 5. Using Gender and Age and Spending Score
MarkBas_X_5 = MarkBas_X[:,[0,1,3]]

wcss_5 = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    #k-means++ is an algorithm for choosing the initial values (or "seeds") for the k-means clustering algorithm.
    kmeans.fit(MarkBas_X_5)
    wcss_5.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss_5)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS_4')
plt.show()

kmeans_5 = KMeans(n_clusters = 6, init = 'k-means++', random_state = 42)
y_kmeans_5 = kmeans_5.fit_predict(MarkBas_X_5)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting the clusters
for i in range(6):
    ax.scatter(MarkBas_X_5[y_kmeans_5 == i, 0], MarkBas_X_5[y_kmeans_5 == i, 1], MarkBas_X_5[y_kmeans_5 == i, 2], s=100, label=f'Cluster {i + 1}')

# Plotting the centroids
ax.scatter(kmeans_5.cluster_centers_[:, 0], kmeans_5.cluster_centers_[:, 1], kmeans_5.cluster_centers_[:, 2],
           s=300, c='yellow', label='Centroids')

ax.set_title('Clusters of customers')
ax.set_xlabel('Gender')
ax.set_ylabel('Age')
ax.set_zlabel('Spending Score (1-100)')
ax.legend()
plt.show()

# 6. Using Age and Annual Income and Spending Score

MarkBas_X_6 = MarkBas_X[:,[1,2,3]]

wcss_6 = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    #k-means++ is an algorithm for choosing the initial values (or "seeds") for the k-means clustering algorithm.
    kmeans.fit(MarkBas_X_6)
    wcss_6.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss_6)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS_4')
plt.show()

kmeans_6 = KMeans(n_clusters = 6, init = 'k-means++', random_state = 42)
y_kmeans_6 = kmeans_6.fit_predict(MarkBas_X_6)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting the clusters
for i in range(6):
    ax.scatter(MarkBas_X_6[y_kmeans_6 == i, 0], MarkBas_X_6[y_kmeans_6 == i, 1], MarkBas_X_6[y_kmeans_6 == i, 2], s=100, label=f'Cluster {i + 1}')

# Plotting the centroids
ax.scatter(kmeans_6.cluster_centers_[:, 0], kmeans_6.cluster_centers_[:, 1], kmeans_6.cluster_centers_[:, 2],
           s=300, c='yellow', label='Centroids')

ax.set_title('Clusters of customers')
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income')
ax.set_zlabel('Spending Score (1-100)')
ax.legend()

plt.show()

