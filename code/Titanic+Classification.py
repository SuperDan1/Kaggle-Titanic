
# coding: utf-8

# # 泰坦尼克号预测问题

# 泰坦尼克号生存分类经典又兼具备趣味性的Kaggle案例[泰坦尼克号存活预测问题](https://www.kaggle.com/c/titanic)。
# 比赛说明：泰坦尼克号的沉没是历史上最臭名昭着的沉船之一。1912年4月15日，在首航期间，泰坦尼克号撞上一座冰山后沉没，2224名乘客和机组人员中有1502人遇难。这一耸人听闻的悲剧震撼了国际社会，导致了更好的船舶安全条例。
# 沉船导致生命损失的原因之一是乘客和船员没有足够的救生艇。虽然幸存下来的运气有一些因素，但有些人比其他人更有可能生存，比如妇女，儿童和上层阶级。
# 在这个挑战中，我们要求你完成对什么样的人可能生存的分析。特别是，我们要求你运用机器学习的工具来预测哪些乘客幸存下来的悲剧。

# ## 前言
# 做机器学习的基本过程
#      * 先做一个baseline的model，再进行后续的分析步骤，一步步提高。后续包括分析        模型现在的状态(欠/过拟合)，分析我们使用的feature的作用大小，进行feature        selection，以及现在模型下的bad case和产生的原因等等。
#      * 要重视对数据的认识
#      * 数据中的特殊点/离群点的分析和处理很重要
#      * 特征工程在很多Kaggle的场景下，甚至比模型本身还重要
#      * 最后要做模型融合

# ## Step1 检视数据源

# In[1]:


import pandas as pd
import numpy as np
#导入数据
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
train


# PassengerId :乘客ID
# Survived    :是否存活(1幸存 0死亡)
# Pclass      :乘客等级(1/2/3等舱位)
# Name        :乘客姓名
# Sex         :性别
# Age         :年龄
# SibSp       :堂兄弟/妹个数
# Parch       :父母与小孩个数
# Ticket      :船票信息(标号)
# Fare        :票价
# Cabin       :客舱
# Embarked    :登船的港口

# In[2]:


#查看数据信息
print train.info()


# In[3]:


print test.info()


# In[4]:


#info显示特征的数量信息 describe显示特征的均值，最大值这些分布信息
train.describe()


# ### 初步检视乘客各属性分布

# In[5]:


import matplotlib.pyplot as plt

#使绘图中显示中文
#from pylab import *  
#mpl.rcParams['font.sans-serif'] = ['SimHei']  

fig = plt.figure(figsize=(30,15))               #规定图片大小         
fig.set(alpha=0.2)                              # 设定图表颜色alpha参数

plt.subplot2grid((2,3),(0,0))                   # 在一张大图里分列几个小图
train.Survived.value_counts().plot(kind='bar')  # 画存活的柱状图 
plt.title(u"获救情况 (1为获救)") # 标题
plt.ylabel(u"人数")  

plt.subplot2grid((2,3),(0,1))
train.Pclass.value_counts().plot(kind="bar")
plt.ylabel(u"人数")
plt.title(u"乘客等级分布")

plt.subplot2grid((2,3),(0,2))
plt.scatter(train.Survived,train.Age)       #scatter绘画离散图
plt.ylabel(u"年龄")                         # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y') 
plt.title(u"按年龄看获救分布 (1为获救)")


plt.subplot2grid((2,3),(1,0), colspan=2)        #表示格的列的跨度为2
train.Age[train.Pclass == 1].plot(kind='kde')   #表示绘画密度图
train.Age[train.Pclass == 2].plot(kind='kde')
train.Age[train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"年龄")# plots an axis lable
plt.ylabel(u"密度") 
plt.title(u"各等级的乘客年龄分布")
plt.legend((u'头等舱', u'2等舱',u'3等舱'),loc='best') # sets our legend for our graph.


plt.subplot2grid((2,3),(1,2))
train.Embarked.value_counts().plot(kind='bar')
plt.title(u"各登船口岸上船人数")
plt.ylabel(u"人数")  
fig.tight_layout()
plt.show()



# 所以我们在图上可以看出来，被救的人300多点，不到半数；被救的人中在3客舱的最多；遇难和获救的人年龄似乎跨度都很广；3个不同的舱年龄总体趋势似乎也一致，2/3等舱乘客20岁多点的人最多，1等舱40岁左右的最多(→_→似乎符合财富和年龄的分配哈)；登船港口人数按照S、C、Q递减，而且S远多于另外俩港口。
# 
# 这个时候我们可能会有一些想法了： 
# * 不同舱位/乘客等级可能和财富/地位有关系，最后获救概率可能会不一样 
# * 年龄对获救概率也一定是有影响的，毕竟前面说了，副船长说『小孩和女士先走』
# * 和登船港口是不是有关系呢？也许登船港口不同，人的出身地位不同？

# ### 属性与获救结果的关联性

# In[6]:


#看看各乘客等级的获救情况
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_0 = train.Pclass[train.Survived == 0].value_counts()
Survived_1 = train.Pclass[train.Survived == 1].value_counts()
df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True)        #stacked作用如下，在每个等级显示存活与死亡
plt.title(u"各乘客等级的获救情况")
plt.xlabel(u"乘客等级") 
plt.ylabel(u"人数") 
plt.show()


# 可以看出1等舱的获救比例是最高的，3等舱获救比例是最低的。说明存活与否是与财富有关的。

# In[7]:


#看看各性别的获救情况
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_m = train.Survived[train.Sex == 'male'].value_counts()
Survived_f = train.Survived[train.Sex == 'female'].value_counts()
df=pd.DataFrame({u'男性':Survived_m, u'女性':Survived_f})
df.plot(kind='bar', stacked=True)
plt.title(u"按性别看获救情况")
plt.xlabel(u"性别") 
plt.ylabel(u"人数")
plt.show()


# 可以看出，女性存活比例是很高的，所以性别也是一个重要因素

# In[8]:


#然后我们再来看看各种舱级别情况下各性别的获救情况
fig=plt.figure(figsize=(20,10))
fig.set(alpha=0.65) # 设置图像透明度，无所谓
plt.title(u"根据舱等级和性别的获救情况")

ax1=fig.add_subplot(141)
train.Survived[train.Sex == 'female'][train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')
ax1.set_xticklabels([u"获救", u"未获救"], rotation=0)
ax1.legend([u"女性/高级舱"], loc='best')

ax2=fig.add_subplot(142, sharey=ax1)
train.Survived[train.Sex == 'female'][train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"女性/低级舱"], loc='best')

ax3=fig.add_subplot(143, sharey=ax1)
train.Survived[train.Sex == 'male'][train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"男性/高级舱"], loc='best')

ax4=fig.add_subplot(144, sharey=ax1)
train.Survived[train.Sex == 'male'][train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"男性/低级舱"], loc='best')

plt.show()


# 与之前的分析基本相似。高级舱中女性的存活率相当高，而低级舱的存活率基本是一半一半。高级舱和低级舱男性的存活率都比较低，但是低级舱更低。

# In[9]:


#查看各港口的存活情况
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_0 = train.Embarked[train.Survived == 0].value_counts()
Survived_1 = train.Embarked[train.Survived == 1].value_counts()
df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各登录港口乘客的获救情况")
plt.xlabel(u"登录港口") 
plt.ylabel(u"人数") 

plt.show()


# 可以看出S港口的获救人数最多，C港口的获救比例更高。

# In[10]:


#查看堂兄弟对存活的影响
g = train.groupby(['SibSp','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
df


# In[11]:


#查看父母、子女对存活的影响
g = train.groupby(['Parch','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
df


# 可以看出一个反常的现象，堂兄弟/妹越多反而死亡率越高；父母/子女越多，存活率基本上越高，最高5成左右

# In[12]:


#提取出对预测有利的特征
selected_features = ['Pclass', 'Sex', 'Age','Embarked','SibSp', 'Parch','Fare']

X_train = train[selected_features]
X_test =  test[selected_features]
Y_train = train['Survived']
all_df = pd.concat((X_train, X_test), axis=0)
all_df.shape


# ## Step2 特征工程(feature engineering)

# 对数据进行预处理，为后面的建模做准备。预处理主要包括对缺失值的补全和将object类型数据进行one-hot编码。
# 遇到缺失值的情况，有几种常见的处理方式：
#      * 如果缺失的样本占总数比例极高，可能就舍弃这个特征了。作为特征加入反而可
#        能引入噪声，影响最后的结果。
#      * 若缺失的样本适中，且该属性非连续值属性，可以把NaN作为一个新类别，加到类              别特征中
#      * 若缺失样本适中，且该属性为连续值属性，有时候考虑给定一个步长，把它离散              化，再如上把NaN作为一个特征值加入
#      * 缺失样本不多的情况下，也可以根据现有的值，拟合以下数据，补充。
#      * 缺失样本不多的情况下

# ### 处理缺失值

# 这里尝试用scikit-learn中的RandomForest来拟合缺失的年龄数据。RandomForest是一个用在原始数据中做不同采样，建立多颗Decision Tree，再进行average等等来降低过拟合现象，提高结果的机器学习算法。

# In[13]:


#补充缺失值
all_df.isnull().sum().sort_values(ascending=False).head()


# In[15]:


from sklearn.ensemble import RandomForestRegressor

#使用 RandomForestClassifier 填补缺失的年龄属性

# 把已有的数值型特征取出来丢进Random Forest Regressor中
age_df = all_df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

# 乘客分成已知年龄和未知年龄两部分
known_age = age_df[age_df.Age.notnull()].as_matrix()
unknown_age = age_df[age_df.Age.isnull()].as_matrix()

# y即目标年龄
y = known_age[:, 0]

# X即特征属性值
X = known_age[:, 1:]

# fit到RandomForestRegressor之中
rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
rfr.fit(X, y)
# 用得到的模型进行未知年龄结果预测
predictedAges = rfr.predict(unknown_age[:, 1::])
# 用得到的预测结果填补原缺失数据
all_df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 


# In[ ]:


mean_column1=all_df.mean()


# In[ ]:


all_df=X_train.fillna(mean_column1)


# In[ ]:


print all_df['Embarked'].value_counts()


# In[ ]:


#对于Embarked这种类型的特征，我们使用出现频率最高的特征值来填充，这是相对可以减少引入误差的一种方法
all_df['Embarked'].fillna('S', inplace=True)
all_df.isnull().sum().sum()


# ### 变换某些数值型变量的形式，独热码编码

# In[ ]:


#pd.get_dummies(X_train['Cabin'], prefix='Cabin').head()


# In[ ]:


all_df = pd.get_dummies(all_df)

all_df.head()


# ## 建立模型，进行预测

# In[ ]:


#通过行号进行索引
X_train_dummy = all_df.loc[X_train.index]
X_test_dummy = all_df.loc[X_test.index]
X_train_dummy.shape, X_test_dummy.shape


# In[ ]:


from xgboost import XGBClassifier
from sklearn.cross_validation import cross_val_score
params = range(1,10,1)
test_scores = []
score_max=0
for param in params:
    clf = XGBClassifier(max_depth=param)
    test_score = cross_val_score(clf, X_train_dummy, Y_train, cv=10, scoring='accuracy')
    test_scores.append(np.mean(test_score))
    if test_score.mean() > score_max:
        score_max=test_score.mean()
print score_max


# In[ ]:


plt.plot(params, test_scores)
plt.title("max_depth vs CV Error");


# In[ ]:


clf = XGBClassifier(max_depth=5)
clf.fit(X_train_dummy, Y_train)
Y_predict = clf.predict(X_test_dummy)
submission_df = pd.DataFrame(data= {'PassengerId' : test['PassengerId'], 'Survived': Y_predict})


# In[ ]:


submission_df.to_csv('../result/result.csv',header=['PassengerId','Survived'],index=False)

