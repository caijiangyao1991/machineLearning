import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
#TODO 训练GBDT
#本文使用lightgbm包来训练我们的GBDT模型，训练共100棵树，每棵树有64个叶子结点。
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
NUMERIC_COLS = [
    "ps_reg_01", "ps_reg_02", "ps_reg_03",
    "ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15",
]

print(df_test.head(10))
y_train = df_train['target']  # training label
y_test = df_test['target']  # testing label
X_train = df_train[NUMERIC_COLS]  # training dataset
X_test = df_test[NUMERIC_COLS]  # testing dataset

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 64,
    'num_trees': 100,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
# number of leaves,will be used in feature transformation
num_leaf = 64

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=lgb_train)

print('Save model...')
# save model to file
gbm.save_model('model.txt')

print('Start predicting...')
# predict and get data on leaves, training data

#TODO 特征转换
#在训练得到100棵树之后，我们需要得到的不是GBDT的预测结果，而是每一条训练数据落在了每棵树的哪个叶子结点上，因此需要使用下面的语句：
y_pred = gbm.predict(X_train, pred_leaf=True)
#打印上面结果的输出，可以看到shape是(8001,100)，即训练数据量*树的棵树
print(np.array(y_pred).shape)
print(y_pred[0])

#然后我们需要将每棵树的特征进行one-hot处理，如前面所说，假设第一棵树落在43号叶子结点上，
# 那我们需要建立一个64维的向量，除43维之外全部都是0。因此用于LR训练的特征维数共num_trees * num_leaves。
print('Writing transformed training data')
transformed_training_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf],
                                       dtype=np.int64)  # N * num_tress * num_leafs
for i in range(0, len(y_pred)):
    temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[I])
    transformed_training_matrix[i][temp] += 1
#当然，对于测试集也要进行同样的处理:
y_pred = gbm.predict(X_test, pred_leaf=True)
print('Writing transformed testing data')
transformed_testing_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf], dtype=np.int64)
for i in range(0, len(y_pred)):
    temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[I])
    transformed_testing_matrix[i][temp] += 1

#TODO LR训练
#然后我们可以用转换后的训练集特征和label训练我们的LR模型，并对测试集进行测试：
lm = LogisticRegression(penalty='l2',C=0.05) # logestic model construction
lm.fit(transformed_training_matrix,y_train)  # fitting the data
y_pred_test = lm.predict_proba(transformed_testing_matrix)   # Give the probabilty on each label
#我们这里得到的不是简单的类别，而是每个类别的概率。

#TODO 效果评价
#在Facebook的paper中，模型使用NE(Normalized Cross-Entropy)，进行评价
NE = (-1) / len(y_pred_test) * sum(((1+y_test)/2 * np.log(y_pred_test[:,1]) +  (1-y_test)/2 * np.log(1 - y_pred_test[:,1])))
print("Normalized Cross Entropy " + str(NE))

#现在的GBDT和LR的融合方案真的适合现在的大多数业务数据么？现在的业务数据是什么？
# 是大量离散特征导致的高维度离散数据。而树模型对这样的离散特征，是不能很好处理的，要说为什么，因为这容易导致过拟合。


