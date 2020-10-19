#kmeans on text vader svns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn import preprocessing
from mpl_toolkits import mplot3d
import seaborn as sns

X= pd.read_csv('vader_dev.csv')
print(X)

plt.figure(1)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=500, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')


kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=500, n_init=10, random_state=0)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
print(y_kmeans)

np.savetxt("Y_vader_dev.csv", y_kmeans)

plt.figure(1)
plt.style.use('ggplot')
sns.scatterplot(X['pos'], X['neu'], hue=y_kmeans, s=50, palette=sns.color_palette("Set1", n_colors=3))
centers = kmeans.cluster_centers_
sns.scatterplot(centers[:, 0], centers[:, 1], color='black',marker='+', s=200)

plt.figure(2)
plt.style.use('ggplot')
sns.scatterplot(X['neg'], X['neu'], hue=y_kmeans, s=50, palette=sns.color_palette("Set1", n_colors=3))
centers = kmeans.cluster_centers_
sns.scatterplot(centers[:, 0], centers[:, 1], color='black',marker='+', s=200)

plt.figure(3)
plt.style.use('ggplot')
sns.scatterplot(X['pos'], X['neg'], hue=y_kmeans, s=50, palette=sns.color_palette("Set1", n_colors=3))
centers = kmeans.cluster_centers_
sns.scatterplot(centers[:, 0], centers[:, 1], color='black',marker='+', s=200)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X['pos'],X['neu'],X['neg'], c=y_kmeans, s=50, cmap='viridis')
ax.set_facecolor('xkcd:white')
centers = kmeans.cluster_centers_
ax.scatter3D(centers[:, 0], centers[:, 1], centers[:, 2], c='black', s=200, alpha=0.5)
plt.show()
