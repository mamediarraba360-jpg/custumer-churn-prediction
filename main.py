import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.conftest import object_dtype
import seaborn as sns
data_1=pd.read_csv("Record (1).csv")
data_2=pd.read_csv('Client (1).csv')
data=pd.merge(data_1,data_2 ,on='Customer_ID')
print(data.head())
print(data.info())
data_nan=data.isnull().sum()
print(data_nan)
print(data.drop(columns=["numbcars","forgntvl","rv","HHstatin","dwllsize","dwlltype","truck" ],inplace=True))
print(data.info())
data_number=data.select_dtypes(include="number")
mat_corr=data_number.corr()
print(mat_corr)
seuil = 0.85
corr_pairs=mat_corr.abs().unstack().sort_values(ascending=False)
corr_pairs=corr_pairs[(corr_pairs<1.0) &(corr_pairs>seuil)]
print(corr_pairs)
data_delete=data.drop(columns=["totcalls","adjmou","totmou","totrev","adjrev","avg6rev","avg6mou","avg6qty",
"mou_Mean","adjqty"],inplace=True)
print(data['churn'].value_counts())
print(data['churn'].value_counts(normalize=True))
corr_churn=data.corr(numeric_only=True)["churn"].sort_values(ascending=True)
print(corr_churn)
print(data.describe(include='all'))
asymetrie=data_number.skew().sort_values(ascending=False)
print(asymetrie)
top_asymetrie=asymetrie.abs().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,6))
sns.barplot(x=top_asymetrie,y=top_asymetrie.index)
plt.xlabel('valeur absolue de lasymetrie')
plt.title('top 10 des variables les plus asymetriques')
plt.show()
print(data['blck_dat_Mean'].describe())
print((data['blck_dat_Mean']==0).mean()*100)
print(data.groupby('marital')['churn'].mean().sort_values(ascending=False))
stayed=data[data['churn']==0]['eqpdays']
churned=data[data['churn']==1]['eqpdays']
print(f'Mean equipment age (stayed):  {stayed.mean():.1f} days')
print(f'Mean equipment age (churned): {churned.mean():.1f} days')
plt.figure(figsize=(10,7))
plt.hist(stayed,bins=30,alpha=0.5,label='stayed')
plt.hist(churned,bins=30,alpha=0.5,label='churned')
plt.xlabel('nombre de jour danciennté de l equipement ')
plt.ylabel('nombre de consommateur')
plt.title('age de l équipement entre ceux qui partent et restent')
plt.legend()
plt.show()
churn_area=data[data['churn']==1].groupby('area').size().sort_values(ascending=True)
plt.figure(figsize=(12,8))
plt.barh(churn_area.index,churn_area.values)
plt.xlabel('nombre de désabonné')
plt.ylabel('zone géographique')
plt.title('nombre des desabonnées par position géographique')
plt.show()
print(data["area"].value_counts())
ny_client=data[data["area"]=="NEW YORK CITY AREA"]
other_client=data[data["area"]!="NEW YORK CITY AREA"]
comparaison= pd.DataFrame({"NEW YORK CITY AREA":ny_client[["income","months","avgrev","avg3rev","hnd_price","eqpdays"]].mean(),
"Autres_zones":other_client[["income","months","avgrev","avg3rev","hnd_price","eqpdays"]].mean()})
print(comparaison)
lg_client=data[data["area"]=="LOS ANGELES AREA"]
autre_zone=data[data["area"]!="LOS ANGELES AREA"]
comparaison2=pd.DataFrame({"LOS ANGELES AREA": lg_client[["income","months","avgrev","avg3rev","hnd_price","eqpdays"]].mean(),
"autre_zone":autre_zone[["income","months","avgrev","avg3rev","hnd_price","eqpdays"]].mean()})
print(comparaison2)
print(ny_client.groupby("dualband")["churn"].mean())
print(ny_client.groupby("refurb_new")['churn'].mean())
print(ny_client.groupby("crclscod")["churn"].mean())
ny_client = ny_client.copy()
ny_client["income_cat"] = pd.cut(
    ny_client["income"],
    bins=[0, 25000, 50000, 75000, 100000, float("inf")],
    labels=["0-25k", "25-50k", "50-75k", "75-100k", "100k+"])
analyse_income = ny_client.groupby("income_cat", observed=True).agg(
    nombre_clients=("churn", "count"),
    taux_churn=("churn", "mean"))
analyse_income["taux_churn"] *= 100
variables = ["income", "months","avgrev","avg3rev","hnd_price","eqpdays"]
comparaison_churn = ny_client.groupby("churn")[variables].mean()
print(comparaison_churn)
print(analyse_income)
data_delete=data.drop(columns=["Customer_ID"])
from sklearn.preprocessing import LabelEncoder
object_cols = data.select_dtypes(include='object').columns.tolist()
print('Columns to encode:', object_cols)
for col in object_cols:
    encoder = LabelEncoder()
    data[col] = encoder.fit_transform(data[col].astype(str))
x=data.drop(columns=["churn"])
y=data["churn"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from xgboost  import  XGBClassifier
model=XGBClassifier(n_estimators=300,learning_rate=0.05,max_depth=6,eval_metric='logloss',random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn import metrics
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
precision=metrics.precision_score(y_test,y_pred)
recall=metrics.recall_score(y_test,y_pred)
print("accuracy:",accuracy)
print("precision:",precision)
print("recall:",recall)
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
cm=metrics.confusion_matrix(y_test,y_pred)
dispo=ConfusionMatrixDisplay(confusion_matrix=cm)
dispo.plot()
plt.show()
importance = pd.Series(
    model.feature_importances_,
    index=x.columns).sort_values(ascending=False)
print(importance.head(10))























