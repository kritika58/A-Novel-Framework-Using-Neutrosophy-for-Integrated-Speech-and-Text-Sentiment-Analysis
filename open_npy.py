import numpy as np
import pandas as pd
from sklearn import preprocessing
data = np.load('X_train.npy')
X= pd.DataFrame(data)
print("-----------TRAIN-CLEAN-100-EXTRACTED FEATURES---------------")
print(X)
print(X.shape)
normalized_X = preprocessing.normalize(X)
print("-----------TRAIN-CLEAN-100-EXTRACTED FEATURES AFTER NORMALISING---------------")
print(normalized_X)
print(normalized_X.shape)
