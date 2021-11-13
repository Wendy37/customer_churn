import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator

df = pd.read_csv("Customer_Churn.csv")
df = df.dropna(axis=0)  # handling missing data, case-wise deletion
y = df.iloc[:, -1]
y = pd.get_dummies(y)
y = y.drop('STAY', axis=1)  # y
df = df.drop('LEAVE', axis=1)
df = pd.get_dummies(df)  # x
sc = StandardScaler()
sc.fit(df)
scaled_data_array = sc.transform(df)
scaled_data = pd.DataFrame(scaled_data_array, columns=df.columns)

ks = range(1, 11)  # varying cluster values from 1 to 10
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(scaled_data)
    inertias.append(model.inertia_)

kl = KneeLocator(range(1, 11), inertias, curve="convex", direction="decreasing")
print('elbow', kl.elbow)
# Plot ks vs inertias, to apply elbow method
# plt.plot(ks, inertias, '-o')
# plt.xlabel('number of clusters (k)')
# plt.ylabel('SSE')
# plt.xticks(ks)
# plt.show()
model = KMeans(n_clusters=6)
scaled_data['CLUSTER'] = model.fit_predict(scaled_data)
centroids = model.cluster_centers_
result = scaled_data.groupby('CLUSTER').mean()
frame = pd.concat([df, scaled_data['CLUSTER']], axis=1)
df_cluster_summary = frame.groupby('CLUSTER').describe().T.reset_index()
summary = {'count':[]}
count = df_cluster_summary.loc[df_cluster_summary['level_1'] == 'count']
summary = pd.DataFrame(summary)
summary['count'] = count.iloc[0, :8]
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
  print(summary)