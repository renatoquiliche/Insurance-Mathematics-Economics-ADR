#!/usr/bin/env python
# coding: utf-8

# # Data sources: 
# https://www.kaggle.com/datasets/teertha/ushealthinsurancedataset
# # License:
# https://creativecommons.org/publicdomain/zero/1.0/

# # About Dataset
# ## Context
# The venerable insurance industry is no stranger to data driven decision making. Yet in today's rapidly transforming digital landscape, Insurance is struggling to adapt and benefit from new technologies compared to other industries, even within the BFSI sphere (compared to the Banking sector for example.) Extremely complex underwriting rule-sets that are radically different in different product lines, many non-KYC environments with a lack of centralized customer information base, complex relationship with consumers in traditional risk underwriting where sometimes customer centricity runs reverse to business profit, inertia of regulatory compliance - are some of the unique challenges faced by Insurance Business.
# 
# Despite this, emergent technologies like AI and Block Chain have brought a radical change in Insurance, and Data Analytics sits at the core of this transformation. We can identify 4 key factors behind the emergence of Analytics as a crucial part of InsurTech:
# 
# - Big Data: The explosion of unstructured data in the form of images, videos, text, emails, social media
# - AI: The recent advances in Machine Learning and Deep Learning that can enable businesses to gain insight, do predictive analytics and build cost and time - efficient innovative solutions
# - Real time Processing: Ability of real time information processing through various data feeds (for ex. social media, news)
# - Increased Computing Power: a complex ecosystem of new analytics vendors and solutions that enable carriers to combine data sources, external insights, and advanced modeling techniques in order to glean insights that were not possible before.
# 
# This dataset can be helpful in a simple yet illuminating study in understanding the risk underwriting in Health Insurance, the interplay of various attributes of the insured and see how they affect the insurance premium.
# 
# ## Content
# This dataset contains 1338 rows of insured data, where the Insurance charges are given against the following attributes of the insured: Age, Sex, BMI, Number of Children, Smoker and Region. There are no missing or undefined values in the dataset.

# # Variables
# - Charges: Individual medical costs billed by health insurance.
# - smoker: Smoker / Non - smoker
# - Children: Number of children covered by health insurance / Number of dependents
# - bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to size
# - sex: Insurance contractor gender, female / male
# - age: Age of primary beneficiary

# In[1]:


import pandas as pd
import seaborn as sns


#from sklearn.neighbors import DistanceMetric
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

import os
#os.chdir(r"D:\IME Q2 - ADR Artigo\Insurance-Mathematics-Economics-ADR-")

# Local directory on LENOVO LAPTOP
#os.chdir(r"C:\Users\xx-re\OneDrive\Documentos\ADR\Insurance-Mathematics-Economics-ADR")


# In[49]:


os.getcwd()


# In[2]:


#Read the data from GitHub directly
url = "https://github.com/renatoquiliche/Insurance-Mathematics-Economics-ADR/blob/main/Databases/insurance.csv?raw=true"

data_insurance = pd.read_csv(url)


# In[3]:


#Missing values computation
data_insurance.info()
#Data nao tem missing values


# In[4]:


#Pre-processing for FAMD

categorical = pd.get_dummies(data_insurance[["sex","smoker", "region"]]) 
numerical = data_insurance[["age","bmi","children", "charges"]]

scaler = RobustScaler()

#Escalamos la data numerica a la escala de la binaria
numerical_data_scaled = pd.DataFrame(scaler.fit_transform(numerical), columns=["age","bmi","children", "charges"])

#Unimos los dos tipos de data
final_data_scaled = pd.concat([numerical_data_scaled, categorical], axis=1)

final_data_scaled.shape


# # Factor analysis for mixed-data
# Input 12 features,
# Output 2 features

# In[5]:


#FAMD Algorithm PCA on scaled data
pca = PCA(n_components=2)
pca.fit(final_data_scaled)

#print(pca.explained_variance_ratio_)
print("Total Varianza Explicada por los dos primeros componentes principales:", f"{pca.explained_variance_ratio_.sum():.2%}")

#Guardamos los componentes principales
components = pd.DataFrame(pca.transform(final_data_scaled), columns=['PC1','PC2'])


# In[6]:


#FAMD Results
loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=final_data_scaled.columns)
print("FAMD loadings: ")
print(loadings)


# # Cluster estimation and evaluation with metrics

# In[7]:


plt.style.use('bmh')


# In[8]:


wcss = []
sil = []
db = []

for i in range(2,11):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
    kmeans.fit(components)
    wcss.append(kmeans.inertia_)
    sil.append(silhouette_score(components, kmeans.labels_))
    db.append(davies_bouldin_score(components, kmeans.labels_))


# #### Within Cluster Sum of Squares (Inertia)

# In[9]:


# WCSS Graph
plt.plot(range(2,11), wcss, color="steelblue")
#plt.title('The Elbow Method')
plt.xlabel('No of clusters')
plt.ylabel('Within Cluster Sum of Squares')
plt.axvline(4, color="indianred") # vertical
#plt.savefig(path_fig+"\Elbowmethod.png")
plt.show()


# #### Average Silhouette Coefficient

# In[10]:


# SC Graph
plt.plot(range(2,11), sil, color="steelblue")
#plt.title('The Elbow Method')
plt.xlabel('No of clusters')
plt.ylabel('Silhouette Coefficient')
plt.axvline(4, color="indianred") # vertical
#plt.savefig(path_fig+"\Elbowmethod.png")
plt.show()


# #### Davies-Bouldin Index

# In[11]:


# DB Graph
plt.plot(range(2,11), db, color="steelblue")
plt.title('The Elbow Method')
plt.xlabel('No of clusters')
plt.ylabel('Davies-Bouldin Index')
plt.axvline(4, color="indianred") # vertical
#plt.savefig(path_fig+"\Elbowmethod.png")
plt.show()


# # Cluster visualization

# In[12]:


# Cluster label estimation
from tkinter import font


kmeans4 = KMeans(n_clusters= 4, init='k-means++', random_state=0)
kmeans4.fit(components)

# Cluster label estimation
kmeans2 = KMeans(n_clusters= 2, init='k-means++', random_state=0)
kmeans2.fit(components)

# Cluster label estimation
kmeans8 = KMeans(n_clusters= 8, init='k-means++', random_state=0)
kmeans8.fit(components)

#data_insurance = pd.concat([data_insurance, components, pd.Series(kmeans.labels_, name="Cluster")], axis=1)


data_final = pd.concat([data_insurance, components
                        , pd.Series(kmeans4.labels_, name="Cluster4")
                        , pd.Series(kmeans2.labels_, name="Cluster2")
                        , pd.Series(kmeans8.labels_, name="Cluster8")], axis=1)

fig, (ax1, ax2) = plt.subplots(2, 1)

fig.set_size_inches(15, 7)

plt.subplots_adjust(top=1.5)
sns.scatterplot(ax=ax1, data=data_final, x="PC1", y="PC2", hue="Cluster4", palette="Set1")
ax1.set_title('K-Means (K=4)', fontsize=20)

sns.scatterplot(ax=ax2, data=data_final, x="PC1", y="PC2", hue="Cluster2", palette="Set1")
ax2.set_title('K-Means (K=2)', fontsize=20)


# In[13]:


b = plt.figure(figsize=(15,7))
b = sns.scatterplot(data=data_final, x="PC1", y="PC2", hue="smoker", palette="Set1")
b.axes.set_title('Data points by smoker category',fontsize=20)

plt.setp(b.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(b.get_legend().get_title(), fontsize='20') # for legend title


# We observe that the binary variable smoker is dividing the multi-dimensional feature space into two clusters by itself. This could be tested against the results of a K-means algorithm applied to the principal components. The results suggest a high correlation between clustering output and collected labels for smoking individuals.

# In[14]:


from sklearn.metrics.cluster import adjusted_rand_score

ARS = adjusted_rand_score(data_final["smoker"], data_final["Cluster2"])

print("Adjusted Rand-Score", f"{ARS:.2%}")

from sklearn.metrics import jaccard_score

JS = jaccard_score(pd.get_dummies(data_final["smoker"], drop_first=True), data_final["Cluster2"])

print("Jaccard Index", f"{JS:.2%}")


# ### Hopkins statistic to test for the existence of clusters
# Based on null-hypothesis that the distribution of data is uniform

# In[15]:


from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
from math import isnan
import random
#random.seed(0)

# Defining the Hopkins test function    
def hopkins(X):
    d = X.shape[1]
    #d = len(vars) # columns
    n = len(X) # rows
    m = int(0.1 * n) # heuristic from article [1]
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print (ujd, wjd)
        H = 0
 
    return H

print("Hopkins statistic:", f"{hopkins(components):.2%}", "is significant at 99% for the range (75-100%)")


# In[16]:


Cluster4 = data_final["Cluster4"]
Cluster2 = data_final["Cluster2"]

# K-Means K=4

charges_k4c0 = data_final.charges.values[Cluster4==0]
charges_k4c1 = data_final.charges.values[Cluster4==1]
charges_k4c2 = data_final.charges.values[Cluster4==2]
charges_k4c3 = data_final.charges.values[Cluster4==3]

# K-Means K=2

charges_k2c0 = data_final.charges.values[Cluster2==0]
charges_k2c1 = data_final.charges.values[Cluster2==1]


# ### Means t-test to search for clusters based on given categories

# In[17]:


smoker = data_insurance["smoker"]
age = data_insurance["age"]
sex = data_insurance["sex"]
bmi = data_insurance["bmi"]
children = data_insurance["children"]
region = data_insurance["region"]

#variable smoker
charges_sy = data_insurance.charges.values[smoker=="yes"]
charges_sn = data_insurance.charges.values[smoker=="no"]

#other variables
charges_age1 = data_insurance.charges.values[age>=60]
charges_age2 = data_insurance.charges.values[age<60]

charges_sexm = data_insurance.charges.values[sex=="male"]
charges_sexf = data_insurance.charges.values[sex=="female"]

charges_bmi1 = data_insurance.charges.values[bmi >= 30] #obesity
charges_bmi2 = data_insurance.charges.values[bmi < 30]

charges_children1 = data_insurance.charges.values[children < 1] #No children
charges_children2 = data_insurance.charges.values[children >= 1] #One children or more

charges_rse = data_insurance.charges.values[region == "southeast"]
charges_rsw = data_insurance.charges.values[region == "southwest"]
charges_rne = data_insurance.charges.values[region == "northwest"]
charges_rnw = data_insurance.charges.values[region == "northeast"]


# In[18]:


from scipy import stats

#t-test unequal variance between two arrays
def ttest_different_means(a,b):
    res = stats.ttest_ind(a, b, equal_var=False)
    if res[1]<0.05:
        passed="Yes"
    else:
        passed="No"
    return [res[0],passed]


# In[19]:


arr = {"Statistic": [], "Passed 95%": []}
j = 0
for i in ["Statistic", "Passed 95%"]:
    arr[i].append(ttest_different_means(charges_sy, charges_sn)[j])
    arr[i].append(ttest_different_means(charges_age1, charges_age2)[j])
    arr[i].append(ttest_different_means(charges_sexm, charges_sexf)[j])
    arr[i].append(ttest_different_means(charges_bmi1, charges_bmi2)[j])
    arr[i].append(ttest_different_means(charges_children1, charges_children2)[j])
    
    arr[i].append(ttest_different_means(charges_rse,charges_rsw)[j])
    arr[i].append(ttest_different_means(charges_rse,charges_rne)[j])
    arr[i].append(ttest_different_means(charges_rse,charges_rnw)[j])

    arr[i].append(ttest_different_means(charges_rsw,charges_rne)[j])
    arr[i].append(ttest_different_means(charges_rsw,charges_rnw)[j])
    
    arr[i].append(ttest_different_means(charges_rne,charges_rnw)[j])
    # Cluster K=2
    arr[i].append(ttest_different_means(charges_k2c1,charges_k2c0)[j])
    
    # Cluster K=4
    arr[i].append(ttest_different_means(charges_k4c0,charges_k4c1)[j])
    arr[i].append(ttest_different_means(charges_k4c0,charges_k4c2)[j])
    arr[i].append(ttest_different_means(charges_k4c0,charges_k4c3)[j])
    
    arr[i].append(ttest_different_means(charges_k4c1,charges_k4c2)[j])
    arr[i].append(ttest_different_means(charges_k4c1,charges_k4c3)[j])
    
    arr[i].append(ttest_different_means(charges_k4c2,charges_k4c3)[j])
    
    j = 1


# In[20]:


index= ["Smoker vs Non-smoker", "Old(60+) vs Adults (60-)", "Males vs Females", "Obesity vs Rest",
        "Without vs With Children", "Southeast vs Southweast", "Southeast vs Northeast", "Southeast vs Northweast",
        "Southweast vs Northeast", "Southweast vs Northweast", "Northeast vs Northweast", "Cluster(K=2) 1 vs Cluster(K=2) 0", 
        "Cluster(K=4) 0 vs Cluster(K=4) 1", "Cluster(K=4) 0 vs Cluster(K=4) 2", "Cluster(K=4) 0 vs Cluster(K=4) 3",
        "Cluster(K=4) 1 vs Cluster(K=4) 2", "Cluster(K=4) 1 vs Cluster(K=4) 3",
        "Cluster(K=4) 2 vs Cluster(K=4) 3",]

#import os
#os.chdir(r"D:\IME Q2 - ADR Artigo\Insurance-Mathematics-Economics-ADR-")

pd.DataFrame(arr, index=index).to_excel(r"..\Results\Mean t-tests.xlsx")
data_final.to_excel(r"..\Databases\Database.xlsx")
data_final.to_csv(r"..\Databases\Database.csv", index=False)



print("t-test for differences in variable Charges, with unequal variance between samples:")
print(pd.DataFrame(arr, index=index))


# In[21]:


os.getcwd()


# ### Data analyitics supporting multiple insurance contracts based on categories

# In[22]:


#other variables
from math import inf
import seaborn as sns
import matplotlib.pyplot as plt

data_final["Age"] = pd.cut(data_final.age,bins=[0, 60, inf],labels=["Adult (60-)", "Old adult (60+)"])

data_final["BMI"] = pd.cut(data_final.bmi,bins=[0, 30, inf],labels=["Non-obesity(BMI<30)", "Obesity (BMI>30)"])

data_final["smoker"] = data_final["smoker"].astype("category")



sns.set()
g = sns.catplot(y="charges", x="smoker", col="BMI", data=data_final, kind="boxen", row="Age")
(g.set_axis_labels("", "Indemnizations ($)")
  .set_xticklabels(["Non smoker", "Smoker"])
  .despine(left=True))


# 
# # Feature engineering

# In[23]:


import warnings
warnings.filterwarnings("ignore")

input_simulation = data_final[["charges", "Age", "BMI", "smoker"]]

input_simulation["smoker"] = input_simulation["smoker"].str.replace('no','Non - Smoker')
input_simulation["smoker"] = input_simulation["smoker"].str.replace('yes','Smoker')

input_simulation["Multiple Contracts"] = input_simulation["Age"].astype(str) + " AND " + input_simulation["BMI"].astype(str) + " AND " + input_simulation["smoker"].astype(str) 

input_simulation["Multiple Contracts"].value_counts()


# In[24]:


b = plt.figure(figsize=(15,7))
sns.set()
g = sns.histplot(x="charges", hue="Multiple Contracts", data=input_simulation)


# In[25]:


b = plt.figure(figsize=(15,7))
sns.set()
g = sns.kdeplot(x="charges", hue="Multiple Contracts", data=input_simulation, fill=True, common_norm=False)


# In[26]:


b = plt.figure(figsize=(15,7))
sns.set()
g = sns.boxenplot(x="charges", y="Multiple Contracts", data=input_simulation)


# In[27]:


b = plt.figure(figsize=(15,7))
sns.set()
g = sns.boxplot(x="charges", y="Multiple Contracts", data=input_simulation)


# In[28]:


print("Mean charges by combinations of three strongest categories: ")
print(data_final.groupby(['smoker', 'Age', 'BMI'])["charges"].mean())

print("Frequency of sample by combinations of three strongest categories: ")
print(data_final.groupby(['smoker', 'Age', 'BMI'])["charges"].count())

print("Frequency of sample by combinations of three strongest categories (%): ")
print(data_final.groupby(['smoker', 'Age', 'BMI'])["charges"].count()/data_final.shape[0]*100)


# In[29]:


from sklearn.preprocessing import OrdinalEncoder

ord_enc = OrdinalEncoder()

input_simulation["Cluster K=8"] = ord_enc.fit_transform(input_simulation[["Multiple Contracts"]]).astype(int)
input_simulation["Cluster K=2"] = ord_enc.fit_transform(input_simulation[["smoker"]]).astype(int)


# In[30]:


input_simulation


# In[31]:


input_simulation.to_csv("../Databases/contracts.csv", index=False)


# # Test for convergence of SAA
# 
# ![image.png](attachment:image.png)
# 
# ![image-2.png](attachment:image-2.png)

# # Test SAA

# **One of the classic approaches for assessing solution quality in
# optimization is to bound the candidate solution’s optimality gap. If the bound on the
# optimality gap is sufficiently small, then the candidate solution is of high quality.**
# 
# 
# The optimality gap (estimator): $gap_N(\hat{x})=f_N(\hat{x})-V_N$
# 
# 
# - $f_N$: The variable term of optimality gap, measures the state of the solution at $N=N^*$, the average of the values of objective function at $X^{*kN}$ for $K$ simulations
# - $X_N$: Candidate solution with N scenarios
# - $V_{N_o}$: Value of sample problem, that is an estimate of $V^*$ (true problem solution), where $E[V_{N_o}] \leq V^*$ is the bias that create bias in the optimality gap estimator, this bias converge to $0$ when $N_o \rightarrow \infty$
# 
# Computing the optimality gap, and the $(1-\alpha)$ confidence interval
# 
# 
# 1. Obtain a point estimator for $f_N$
#     - Given that in SAA $\Omega$ scenarios are sampled from uniform distribution with random seed $k$, the uncertainty in the model is defined as $\xi^{kN}$ that produces an optimal solution $X^{*kN}$ for the random seed $k$ and a fixed number of scenarios $N$
#     - Multiple replications procedure (MRP) implies generating $K$ solutions for $K$ Monte Carlo simulations, for each random seed $k$ where $X^{*kN}$ is the optimal solution for 'batch' $k$
#     - Compute the average of MRP solutions: $\frac{1}{K} \sum^{K}_k f(X^{*kN})$, then use this to compute $f_N$ and thus the **optimality gap estimator**
#     - Compute the sample variance of the optimality gap estimator $S^2_{gap} = \frac{1}{K-1} \sum^{K}_k [gap^{K}_N-\overline{gap}]^2$
#     - Compute the $(1-\alpha)\%$ confidence interval for the optimality gap: $[0, \overline{gap}+ \frac{Z_{\alpha}S_{gap}}{\sqrt{K}} ]$
#     - Fix $N_o$ as a large number and $N$ as a feasible number to asses quality of solution for MRP with $K$ replications
#     - The test requires a lighter computational burden than the graph-based convergence assesment 

# # Silhouette and Davies-Bouldin for categorical clusters vs K-means algorithm

# ![image.png](attachment:image.png)

# In[32]:


K_Means = {
  "K": np.arange(2,len(sil)+2),
  "Silhouette score": sil,
  "Davis-Bouldin Index": db
}

pd.DataFrame(K_Means)


# In[33]:


from termcolor import colored

print(colored('Silhouette score K-Means (K=2):', 'red', attrs=['bold']), round(silhouette_score(components, data_final["Cluster2"]),2))
print(colored('Silhouette score Smoker (K=2):', 'red', attrs=['bold']), round(silhouette_score(components, input_simulation["Cluster K=2"]),2))

print(colored('Silhouette score K-Means (K=8):', 'red', attrs=['bold']), round(silhouette_score(components, data_final["Cluster8"]),2))
print(colored('Silhouette score Multiple Contracts (K=8):', 'red', attrs=['bold']), round(silhouette_score(components, input_simulation["Cluster K=8"]),2))


print(colored('Davies-Bouldin score K-Means (K=2):', 'green', attrs=['bold']), round(davies_bouldin_score(components, data_final["Cluster2"]),2))
print(colored('Davies-Bouldin score Smoker (K=2):', 'green', attrs=['bold']), round(davies_bouldin_score(components, input_simulation["Cluster K=2"]),2))

print(colored('Davies-Bouldin score K-Means (K=8):', 'green', attrs=['bold']), round(davies_bouldin_score(components, data_final["Cluster8"]),2))
print(colored('Davies-Bouldin score Multiple Contracts (K=8):', 'green', attrs=['bold']), round(davies_bouldin_score(components, input_simulation["Cluster K=8"]),2))


# ### Silhouette
# 
# ![image.png](attachment:image.png)
# 
# ### Davies-Bouldin
# 
# ![image-2.png](attachment:image-2.png)
# 
# The idea is to maximize silhouette score and minimize Davies-Bouldin score for better defined clusters
# Results:
# 
# - The difference between algorithm and category smoker is neglictible for both SC and DBI for data segmentation into two clusters
# - The difference increases when we get eight groups, the disadvantages of algoritmic clustering methods are the following:
#     - There are no more discrete cutoffs based on categories, the decision-making would change. In the K-means result, the rules would be more complex, and the expected gain from following such rules would be HIGHER OR LOWER?
#     - In clusters given by categories of BMI, Smoker and Age, it is easier to segment the sample, but clusters have greater variance and this could affect decision-making.
#     
# ### Is important to test to this point which approach would lead to better results regarding insurance company decision-making in underwritting    
