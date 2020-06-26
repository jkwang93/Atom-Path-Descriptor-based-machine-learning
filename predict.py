# _*_ encoding: utf-8 _*_
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error

# from V5.V5_test_One_hot import Preprocessing_One_hot
from v1_v7.V6.V6_new_feature_engineering_One_hot import Preprocessing_One_hot
__author__ = 'wjk'
__date__ = '2019/10/7 9:16'
import pandas as pd
import numpy as np


atom = 'H'
data = pd.read_csv('../test_V6_DDEC_atom_type.csv')
test_num = data.iloc[:,0].value_counts()[atom].tolist()

atom_type_num, bond_type_num, atom_type, bond_type, X_Distance, y = Preprocessing_One_hot(
    'V6_DDEC_atom_type.csv', 'V6_DDEC_bond_type.csv', 'V6_DDEC_charge.csv', 'V6_DDEC_distance.csv', atom, 1000000)

X_test = np.hstack((atom_type[-test_num:-1,:], bond_type[-test_num:-1,:], X_Distance[-test_num:-1,:]))
y_test = y[-test_num:-1,:]

rf= joblib.load(atom+'.model')
print('load finish')

# rf.fit(X, y.ravel())

rf_predict = rf.predict(X_test)
MSE = mean_squared_error(y_test, rf_predict)
RMSE = np.sqrt(mean_squared_error(y_test, rf_predict))

print(atom+'rf_predict回归精度：', RMSE, '/n', MSE)
# prediction_list.extend(rf_predict)
# test_list.extend(y_test)
#
# Y = pd.DataFrame(prediction_list)
# X = pd.DataFrame(test_list)
# pre_result = pd.concat([X, Y], axis=1)
# pre_result.columns = ['test', 'predict']
# pre_result.to_csv(i + '_results.csv', mode='a', header=0, index=False, sep=',')
print('end!')