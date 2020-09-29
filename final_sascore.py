import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn import preprocessing
from mpl_toolkits import mplot3d
import seaborn as sns
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

X_text= pd.read_csv('vader_train.csv')
X_audio= pd.read_csv('train_audio_svns.csv')
X_final=pd.DataFrame() 
X_final['C1']=(X_audio['C1']+X_text['pos'])/2
X_final['C2']=(X_audio['C2']+X_text['neu'])/2
X_final['C3']=(X_audio['C3']+X_text['neg'])/2
print(X_text.info())
print(X_audio.info())
print(X_final)

#KMMEANS CLUSTERING
# kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=500, n_init=10, random_state=0)
# kmeans.fit(X_final)
# y_kmeans = kmeans.predict(X_final)
# print(y_kmeans)

# y= y_kmeans.reshape((len(y_kmeans),1))
# y=np.around(y, decimals=0)
# np.savetxt("combined_train_svns.csv", np.concatenate((X_final,y),axis=1), delimiter=",", fmt='%.3f')

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(X_final['C1'],X_final['C2'],X_final['C3'], c=y_kmeans, s=50, cmap='viridis')
# centers = kmeans.cluster_centers_
# ax.scatter3D(centers[:, 0], centers[:, 1], centers[:, 2], c='black', s=200, alpha=0.5)
# plt.show()

#HIERARCHICAL AGGLOMETRIC CLUSTERING
# plt.figure(figsize=(10, 7))  
# plt.title("Dendrograms")  
# dend = shc.dendrogram(shc.linkage(X_final, method='ward'))
# plt.show()

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')  
cluster.fit_predict(X_final)

plt.figure(figsize=(10, 7))  
plt.scatter(X_final['C1'],X_final['C2'], c=cluster.labels_, cmap='rainbow')

plt.figure(figsize=(10, 7))  
plt.scatter(X_final['C2'],X_final['C3'], c=cluster.labels_, cmap='rainbow')

plt.figure(figsize=(10, 7))  
plt.scatter(X_final['C1'],X_final['C3'], c=cluster.labels_, cmap='rainbow')
plt.show()