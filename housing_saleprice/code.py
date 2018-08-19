import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import KFold
from IPython.display import HTML, display
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("train.csv")
train = df
actual_test_data = pd.read_csv("test.csv")

quantitative = [f for f in train.columns if train.dtypes[f] != 'object']
#print(quantitative)
quantitative.remove('SalePrice')
quantitative.remove('Id')
qualitative = [f for f in train.columns if train.dtypes[f] == 'object']
#print(train.shape)

for c in qualitative:
    train[c] = train[c].astype('category')
    actual_test_data[c] = actual_test_data[c].astype('category')
    if train[c].isnull().any():
        train[c] = train[c].cat.add_categories(['MISSING'])
        train[c] = train[c].fillna('MISSING')
    if actual_test_data[c].isnull().any():
        actual_test_data[c] = actual_test_data[c].cat.add_categories(['MISSING'])
        actual_test_data[c] = actual_test_data[c].fillna('MISSING')

qual_encoded = []

for feature in qualitative:  
    ordering = pd.DataFrame()
    ordering['val'] = train[feature].unique()
    #print(ordering['val'])
    ordering.index = ordering.val
    #print(ordering.index)
    #print(frame[[feature, 'SalePrice']].groupby(feature).mean())
    ordering['spmean'] = train[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']
    #print(ordering['spmean'])
    ordering = ordering.sort_values('spmean')
    #print(ordering)
    ordering['ordering'] = range(1, ordering.shape[0]+1)
    #print(ordering)
    ordering = ordering['ordering'].to_dict()
    #print(ordering)
    
    for cat, o in ordering.items():
        #print(frame.loc[:,'Street'])
        train.loc[train[feature] == cat, feature+'_E'] = o
        actual_test_data.loc[actual_test_data[feature] == cat, feature+'_E'] = o

    qual_encoded.append(feature+'_E')
    
features = quantitative + qual_encoded
#print(len(features))
X =  train[features].fillna(0.)
X_actual = actual_test_data[features].fillna(0.)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

stdSc = StandardScaler()
X.loc[:, quantitative] = stdSc.fit_transform(X.loc[:, quantitative])
X_actual.loc[:, quantitative] = stdSc.fit_transform(X_actual.loc[:, quantitative])

Y = train['SalePrice']
X_train = X[:1200]
y_train = Y[:1200]
X_test = X[1200:]
y_test = Y[1200:]

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

#clf1 = GradientBoostingRegressor(n_estimators=1500, learning_rate=0.02,max_depth=5, random_state=0).fit(X_train,y_train)
#pred1 = clf1.predict(X_test)
#score1 = r2_score(y_test.values,pred1)
#print(score1)
#print(pred1)


#from sklearn.ensemble import ExtraTreesRegressor

#clf2 = ExtraTreesRegressor(n_estimators=300, max_depth=4,min_samples_split=2, random_state=0).fit(X_train,y_train)
#pred2 = clf2.predict(X_test)
#score2 = r2_score(y_test.values,pred2)

'''from sklearn.ensemble import RandomForestClassifier
clf3 = RandomForestClassifier(n_estimators=500, max_depth=4, random_state=0).fit(X_train,y_train)
pred3 = clf3.predict(X_test)
score3 = r2_score(y_test.values,pred3)

print(score1,score2,score3)'''

from autosklearn.regression import AutoSklearnRegressor

rg = AutoSklearnRegressor(time_left_for_this_task=1200, per_run_time_limit=120, initial_configurations_via_metalearning=25, ensemble_size=50, ensemble_nbest=50)
rg.fit(X,Y)
pred = rg.predict(X_actual)
#print(r2_score(y_test,pred))

'''from tpot import TPOTRegressor
tpot = TPOTRegressor(generations=20, population_size=30, verbosity=2,max_time_mins=15)
tpot.fit(X, Y)
pred = tpot.predict(X_actual)'''
#log1 = np.log(pred)
#log2 = np.log(y_test)
#from sklearn.metrics import mean_squared_error
#print(sqrt(mean_squared_error(log2,log1)))
#print(tpot.score(X_test, y_test))

#from sklearn.model_selection import cross_val_score
#from sklearn.metrics import mean_squared_error, make_scorer
#scorer = make_scorer(mean_squared_error, greater_is_better = False)

'''lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                          0.3, 0.6, 1,1.5,2.0,3.0,5.0,10.0], 
                max_iter = 500000, cv = 10)
lasso.fit(X, Y)
pred = lasso.predict(X_actual)'''

#print("Lasso RMSE on Test set :", rmse_cv_test(lasso).mean())

subms = pd.DataFrame(pred,columns=['SalePrice'])
#print(subms)

lista = []
for i in range(1461,2920):
    lista.append(i)

subms['Id'] = lista
final_sol = subms[subms.columns[::-1]]
ans = final_sol
print(ans)
ans.to_csv('submission.csv',index=False,columns=['Id','SalePrice'])