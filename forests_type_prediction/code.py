import pandas as pd
import time
import numpy as np
df = pd.read_csv("train.csv")

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

y = df.loc[:,'Cover_Type']
X = df.drop(['Id','Cover_Type'],axis=1)

#features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']

'''X_yellowbrick = X_train[features].as_matrix()
y_yellowbrick = y_train.as_matrix()'''

features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3']

X_train1 = X[:11000]
y_train1 = y[:11000]

X_test1 = X[11000:]
y_test1 = y[11000:]

X_train2 = X.loc[:10999,features]
y_train2 = y[:11000]

X_test2 = X.loc[11000:,features]
y_test2 = y[11000:]

clf = RandomForestClassifier(n_estimators=400,criterion='entropy',max_depth=None, min_samples_leaf=1,min_samples_split=2)

import matplotlib.pyplot as plt
from yellowbrick.features.importances import FeatureImportances

fig = plt.figure()
ax = fig.add_subplot()

start_time = time.time()

viz = FeatureImportances(clf, ax=ax,relative=False)
viz.fit(X_train2, y_train2)
viz.poof()

elapsed_time = time.time() - start_time
print(elapsed_time)
