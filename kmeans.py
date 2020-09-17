import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn import preprocessing
from mpl_toolkits import mplot3d
import seaborn as sns
data = np.load('X_train.npy')
X= pd.DataFrame(data)
X = preprocessing.normalize(X)
#print(nor_X)

# plt.figure(1)
# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')

kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
# print(y_kmeans.shape)


y= y_kmeans.reshape((len(y_kmeans),1))
y=np.around(y, decimals=0)
#print(y)

X_dist = kmeans.transform(X)
X_dist=np.around(X_dist, decimals=3)
print("\n DISTANCE FROM CLUSTER CENTRE\n")
print(X_dist)
print(X_dist.shape)

scores=1- X_dist
scores=np.around(scores, decimals=3)
print("\nAUDIO SVNS\n")
print(scores)
print(scores.shape)
dist_label= np.concatenate((scores,y),axis=1)
np.savetxt("train_audio_svns.csv", dist_label, delimiter=",", fmt='%.3f')

plt.figure(2)
plt.style.use('ggplot')
sns.scatterplot(X[:, 0], X[:, 1], hue=y_kmeans, s=50, palette=sns.color_palette("Set1", n_colors=3))
centers = kmeans.cluster_centers_
sns.scatterplot(centers[:, 0], centers[:, 1], color='black',marker='+', s=200)


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X[:, 0], X[:, 1], X[:,2], c=y_kmeans, s=50, cmap='viridis')
ax.set_facecolor('xkcd:white')
centers = kmeans.cluster_centers_
ax.scatter3D(centers[:, 0], centers[:, 1], centers[:, 2], c='black', s=200, alpha=0.5)

plt.show()