#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = "SimHei" #解决中文乱码问题
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.neighbors import KNeighborsRegressor


# In[3]:


df_train = pd.read_csv(r'D:\桌面\学校\大数据天猫\data_format1\train_format1.csv')
df_test = pd.read_csv(r'D:\桌面\学校\大数据天猫\data_format1\test_format1.csv')
user_info = pd.read_csv(r'D:\桌面\学校\大数据天猫\data_format1\user_info_format1.csv')
user_log = pd.read_csv(r'D:\桌面\学校\大数据天猫\data_format1\user_log_format1.csv')


# In[4]:


print(df_test.shape,df_train.shape)
print(user_info.shape,user_log.shape)


# In[5]:


user_info.info()
user_info.head(10)


# In[6]:


user_info['age_range'].replace(0.0,np.nan,inplace=True)
user_info['gender'].replace(2.0,np.nan,inplace=True)
user_info.info()


# In[7]:


fig = plt.figure(figsize = (10, 6))
x = np.array(["NULL","<18","18-24","25-29","30-34","35-39","40-49",">=50"])
#<18岁为1；[18,24]为2； [25,29]为3； [30,34]为4；[35,39]为5；[40,49]为6； > = 50时为7和8
y = np.array([user_info[user_info['age_range'] == -1]['age_range'].count(),
             user_info[user_info['age_range'] == 1]['age_range'].count(),
             user_info[user_info['age_range'] == 2]['age_range'].count(),
             user_info[user_info['age_range'] == 3]['age_range'].count(),
             user_info[user_info['age_range'] == 4]['age_range'].count(),
             user_info[user_info['age_range'] == 5]['age_range'].count(),
             user_info[user_info['age_range'] == 6]['age_range'].count(),
             user_info[user_info['age_range'] == 7]['age_range'].count() + user_info[user_info['age_range'] == 8]['age_range'].count()])
plt.bar(x,y,label='人数')
plt.legend()
plt.title('用户年龄分布')


# In[8]:


sns.countplot(x = 'age_range', order = [-1,1,2,3,4,5,6,7,8], data = user_info)
plt.title('用户年龄分布')


# In[9]:


sns.countplot(x='gender',order = [-1,0,1],data = user_info)
plt.title('用户性别分布')


# In[10]:


sns.countplot(x = 'age_range', order = [-1,1,2,3,4,5,6,7,8],hue= 'gender',data = user_info)
plt.title('用户性别年龄分布')


# In[11]:


user_log.head()


# In[12]:


user_log.isnull().sum(axis=0)


# In[13]:


#user_log = user_log.dropna()
user_log.isnull().sum(axis=0)


# In[14]:


user_log.info()


# In[15]:


df_train.head(10)


# In[16]:


df_train.info()


# In[17]:


user_log['time_stamp'].hist(bins = 9)


# In[18]:


sns.countplot(x = 'action_type', order = [0,1,2,3],data = user_log)


# In[19]:


#以下是特征工程
df_train[df_train['label'] == 1]


# In[20]:


user_log[(user_log['user_id'] == 34176) & (user_log['seller_id'] == 3906)]


# In[21]:


df_train.head()


# In[22]:


user_info.head()


# In[23]:


user_log.head()


# In[24]:


df_train = pd.merge(df_train,user_info,on="user_id",how="left")
df_train.head()


# In[25]:


total_logs_temp = user_log.groupby([user_log["user_id"],user_log["seller_id"]]).count().reset_index()[["user_id","seller_id","item_id"]]


# In[26]:


total_logs_temp.head(10)


# In[27]:


total_logs_temp.rename(columns={"seller_id":"merchant_id","item_id":"total_logs"},inplace=True)
total_logs_temp.head()


# In[28]:


df_train = pd.merge(df_train,total_logs_temp,on=["user_id","merchant_id"],how="left")
df_train.head()


# In[29]:


unique_item_ids_temp = user_log.groupby([user_log["user_id"],user_log["seller_id"],user_log["item_id"]]).count().reset_index()[["user_id","seller_id","item_id"]]


# In[30]:


unique_item_ids_temp.head(10)


# In[31]:


unique_item_ids_temp1 = unique_item_ids_temp.groupby([unique_item_ids_temp["user_id"],unique_item_ids_temp["seller_id"]]).count().reset_index()


# In[32]:


unique_item_ids_temp1.head(10)


# In[33]:


unique_item_ids_temp1.rename(columns={"seller_id":"merchant_id","item_id":"unique_item_ids"},inplace=True)


# In[34]:


unique_item_ids_temp1.head(10)


# In[35]:


df_train = pd.merge(df_train,unique_item_ids_temp1,on=["user_id","merchant_id"],how="left")
df_train.head()


# In[36]:


categories_temp = user_log.groupby([user_log["user_id"],user_log["seller_id"],user_log["cat_id"]]).count().reset_index()[["user_id","seller_id","cat_id"]]
categories_temp.head(20)


# In[37]:


categories_temp1 = categories_temp.groupby([categories_temp["user_id"],categories_temp["seller_id"]]).count().reset_index()
categories_temp1.head(10)


# In[38]:


categories_temp1.rename(columns={"seller_id":"merchant_id","cat_id":"categories"},inplace=True)
categories_temp1.head(10)


# In[39]:


df_train = pd.merge(df_train,categories_temp1,on=["user_id","merchant_id"],how="left")
df_train.head(10)


# In[40]:


browse_days_temp = user_log.groupby([user_log["user_id"],user_log["seller_id"],user_log["time_stamp"]]).count().reset_index()[["user_id","seller_id","time_stamp"]]
browse_days_temp.head(10)


# In[41]:


browse_days_temp1 = browse_days_temp.groupby([browse_days_temp["user_id"],browse_days_temp["seller_id"]]).count().reset_index()
browse_days_temp1.head(10)


# In[42]:


browse_days_temp1.rename(columns={"seller_id":"merchant_id","time_stamp":"browse_days"},inplace=True)
browse_days_temp1.head(10)


# In[43]:


df_train = pd.merge(df_train,browse_days_temp1,on=["user_id","merchant_id"],how="left")
df_train.head(10)


# In[44]:


one_clicks_temp = user_log.groupby([user_log["user_id"],user_log["seller_id"],user_log["action_type"]]).count().reset_index()[["user_id","seller_id","action_type","item_id"]]
one_clicks_temp.head(10)


# In[45]:


one_clicks_temp.rename(columns={"seller_id":"merchant_id","item_id":"times"},inplace=True)
one_clicks_temp.head(10)


# In[46]:


one_clicks_temp["one_clicks"] = one_clicks_temp["action_type"] == 0
one_clicks_temp["one_clicks"] = one_clicks_temp["one_clicks"] * one_clicks_temp["times"]
one_clicks_temp.head(10)


# In[47]:


one_clicks_temp["shopping_carts"] = one_clicks_temp["action_type"] == 1
one_clicks_temp["shopping_carts"] = one_clicks_temp["shopping_carts"] * one_clicks_temp["times"]
one_clicks_temp.head(10)


# In[48]:


one_clicks_temp["purchase_times"] = one_clicks_temp["action_type"] == 2
one_clicks_temp["purchase_times"] = one_clicks_temp["purchase_times"] * one_clicks_temp["times"]
one_clicks_temp.head(10)


# In[49]:


one_clicks_temp["favourite_times"] = one_clicks_temp["action_type"] == 3
one_clicks_temp["favourite_times"] = one_clicks_temp["favourite_times"] * one_clicks_temp["times"]
one_clicks_temp.head(10)


# In[50]:


four_features = one_clicks_temp.groupby([one_clicks_temp["user_id"],one_clicks_temp["merchant_id"]]).sum().reset_index()
four_features.head(10)


# In[51]:


four_features = four_features.drop(["action_type","times"], axis=1)
df_train = pd.merge(df_train,four_features,on=["user_id","merchant_id"],how="left")
df_train.head(10)


# In[52]:


#特征建立完毕，缺省值处理
df_train.info()


# In[53]:


df_train.isnull().sum(axis=0)


# In[54]:


df_train = df_train.fillna(method='ffill')


# In[55]:


df_train.info()


# In[56]:


plt.style.use('ggplot')
sns.countplot(x = 'age_range', order = [1,2,3,4,5,6,7,8],hue= 'gender',data = df_train)
plt.title('训练集用户性别年龄分布')


# In[57]:


colnm = df_train.columns.tolist()
print(colnm)
plt.figure(figsize = (5, 4))
color = sns.color_palette()

df_train[colnm[5]].hist(range=[0,80],bins = 80,color = color[1])
plt.xlabel(colnm[5],fontsize = 12)
plt.ylabel('用户数')


# In[58]:


df_train[colnm[6]].hist(range=[0,40],bins = 40,color = color[1])
plt.xlabel(colnm[6],fontsize = 12)
plt.ylabel('用户数')


# In[59]:


df_train[colnm[7]].hist(range=[0,10],bins = 10,color = color[1])
plt.xlabel(colnm[7],fontsize = 12)
plt.ylabel('用户数')


# In[60]:


df_train[colnm[8]].hist(range=[0,10],bins = 10,color = color[1])
plt.xlabel(colnm[8],fontsize = 12)
plt.ylabel('用户数')


# In[61]:


df_train[colnm[9]].hist(range=[0,50],bins = 50,color = color[1])
plt.xlabel(colnm[9],fontsize = 12)
plt.ylabel('用户单击次数统计')


# In[62]:


df_train[colnm[9]].hist(range=[0,50],bins = 50,color = color[1])
plt.xlabel(colnm[9],fontsize = 12)
plt.ylabel('用户单击次数统计')


# In[63]:


df_train[colnm[11]].hist(range=[0,6],bins = 7,color = color[1])
plt.xlabel(colnm[11],fontsize = 12)
plt.ylabel("用户数")


# In[64]:


df_train[colnm[12]].hist(range=[0,6],bins = 6,color = color[1])
plt.xlabel(colnm[12],fontsize = 12)
plt.ylabel("用户数")


# In[65]:


sns.set_style("dark")

plt.figure(figsize = (10,8))
colnm = df_train.columns.tolist()[2:13]
mcorr = df_train[colnm].corr()
# np.zero_like的意思就是生成一个和你所给数组a相同shape的全0数组。
mask = np.zeros_like(mcorr, dtype=np.bool)
# np.triu_indices_from()返回方阵的上三角矩阵的索引
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True,fmt='0.2f')
# 相关性好像不大，可是日志里确实也没啥可以用的其他特征了啊


# In[66]:


#模型构建与调参
#罗切斯特模型


# In[67]:


Y = df_train['label']
X = df_train.drop(['user_id','merchant_id','label'],axis = 1)
X.head(10)


# In[68]:


Y.head(10)


# In[69]:


X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.25,random_state = 10)


# In[70]:


Logit = LogisticRegression(solver='liblinear')
Logit.fit(X_train, y_train)
Predict = Logit.predict(X_test)
Predict_proba = Logit.predict_proba(X_test)
print(Predict[0:20])
print(Predict_proba[:])
Score = accuracy_score(y_test, Predict)
Score
# 一般的准确率验证方法


# In[71]:


print("lr.coef_: {}".format(Logit.coef_))
print("lr.intercept_: {}".format(Logit.intercept_))
# 截距与斜率


# In[72]:


#初始化逻辑回归算法
LogRegAlg=LogisticRegression(random_state=1,solver='liblinear')
re = LogRegAlg.fit(X,Y)
#使用sklearn库里面的交叉验证函数获取预测准确率分数
scores = model_selection.cross_val_score(LogRegAlg,X,Y,cv=3)
#使用交叉验证分数的平均值作为最终的准确率
print("准确率为: ",scores.mean())


# In[73]:


#k近邻
# 模型实例化，并将邻居个数设为3 
reg = KNeighborsRegressor(n_neighbors=1000)
# 利用训练数据和训练目标值来拟合模型 
reg.fit(X_train, y_train)
print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test)))


# In[76]:


#决策树
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=4,random_state=0) 
tree.fit(X_train, y_train)
Predict_proba = tree.predict_proba(X_test)
print(Predict_proba[:])
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train))) 
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))


# In[78]:


from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot", class_names=["0","1"], feature_names=X.columns.tolist(), impurity=False, filled=True)
# 我们可以利用 tree 模块的 export_graphviz 函数来将树可视化。这个函数会生成一 个 .dot 格式的文件，这是一种用于保存图形的文本文件格式。
# 设置为结点添加颜色 的选项，颜色表示每个结点中的多数类别，同时传入类别名称和特征名称，这样可以对 树正确标记


# In[81]:


import graphviz
with open("tree.dot") as f: 
    dot_graph = f.read() 
graphviz.Source(dot_graph)


# In[82]:


print("Feature importances:\n{}".format(tree.feature_importances_))


# In[83]:


plt.barh(X.columns.tolist(),height=0.5,width=tree.feature_importances_,align="center")


# In[84]:


#随机森林
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=10, random_state=2) 
forest.fit(X_train, y_train)
Predict_proba = forest.predict_proba(X_test)
print(Predict_proba[:])
print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train))) 
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))


# In[85]:


plt.barh(X.columns.tolist(),height=0.5,width=forest.feature_importances_,align="center")


# In[86]:


#梯度回升回归树
from sklearn.ensemble import GradientBoostingClassifier
gbrt = GradientBoostingClassifier(random_state=0) 
gbrt.fit(X_train, y_train)
Predict_proba = gbrt.predict_proba(X_test)
print(Predict_proba[:])
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))


# In[87]:


plt.barh(X.columns.tolist(),height=0.5,width=gbrt.feature_importances_,align="center")


# In[88]:


#多层感知机
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='lbfgs', activation='relu',alpha=0.1,random_state=0,hidden_layer_sizes=[10,10]).fit(X_train, y_train)
Predict = mlp.predict(X_test)
Predict_proba = mlp.predict_proba(X_test)
print(Predict_proba[:])
Score = accuracy_score(y_test, Predict)
print(Score)


# In[89]:


plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(10), X.columns.tolist()) 
plt.xlabel("Columns in weight matrix") 
plt.ylabel("Input feature") 
plt.colorbar()
# 显示了连接输入和第一个隐层之间的权重。图中的行对应 10个输入特征，列对应 10个隐单元。


# In[90]:


# 原始数据预处理之缩放
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[91]:


X_train = X_train[X_train.columns.tolist()].astype(float)
X_test = X_test[X_test.columns.tolist()].astype(float)


# In[92]:


scaler.fit(X_train)


# In[93]:


# 变换数据
X_train_scaled = scaler.transform(X_train)


# In[94]:


X_test_scaled = scaler.transform(X_test)


# In[95]:


mlp1 = MLPClassifier(solver='lbfgs', random_state=0,hidden_layer_sizes=[10]).fit(X_train_scaled, y_train)
Predict = mlp1.predict(X_test)
Score = accuracy_score(y_test, Predict)
print(Score)


# In[96]:


#实践预测
df_test.head()


# In[97]:


df_test = pd.merge(df_test,user_info,on="user_id",how="left")


# In[98]:


df_test = pd.merge(df_test,total_logs_temp,on=["user_id","merchant_id"],how="left")


# In[99]:


df_test = pd.merge(df_test,unique_item_ids_temp1,on=["user_id","merchant_id"],how="left")


# In[100]:


df_test = pd.merge(df_test,categories_temp1,on=["user_id","merchant_id"],how="left")


# In[101]:


df_test = pd.merge(df_test,browse_days_temp1,on=["user_id","merchant_id"],how="left")


# In[102]:


df_test = pd.merge(df_test,four_features,on=["user_id","merchant_id"],how="left")


# In[103]:


df_test = df_test.fillna(method='bfill')
df_test = df_test.fillna(method='ffill')
# 缺失值向后填充


# In[104]:


df_test.head(10)


# In[105]:


df_test.isnull().sum(axis=0)


# In[106]:


X1 = df_test.drop(['user_id','merchant_id','prob'],axis = 1)


# In[107]:


X1.head(10)


# In[108]:


#罗切Predict_proba = Logit.predict_proba(X1)斯特模型
Predict_proba = Logit.predict_proba(X1)
df_test["Logit_prob"] = Predict_proba[:,1]
Predict_proba[0:10]


# In[109]:


df_test.head(10)


# In[111]:


Predict_proba = tree.predict_proba(X1)
df_test["Tree_prob"] = Predict_proba[:,1]
Predict_proba[0:10]


# In[112]:


df_test.head(10)


# In[113]:


#随机森林
Predict_proba = forest.predict_proba(X1)
df_test["Forest_prob"] = Predict_proba[:,1]
Predict_proba[0:10]


# In[114]:


df_test.head(10)


# In[118]:


#梯度提升回归树


# In[120]:


Predict_proba[0:10]Predict_proba = gbrt.predict_proba(X1)


# In[121]:


df_test["Gbrt_prob"] = Predict_proba[:,1]
Predict_proba[0:10]


# In[122]:


df_test.head(10)


# In[123]:


#多层感知机
Predict_proba = mlp.predict_proba(X1)
df_test["mlp_prob"] = Predict_proba[:,1]
Predict_proba[0:10]


# In[124]:


df_test.head(10)


# In[126]:


choose = ["user_id","merchant_id","mlp_prob"]
res = df_test[choose]
res.rename(columns={"mlp_prob":"prob"},inplace=True)
print(res.head(10))
res.to_csv(path_or_buf = r"D:\桌面\prediction.csv",index = False)


# In[ ]:




