
# coding: utf-8

# In[60]:


# Load the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas
import sklearn.tree as tree
from scipy import linalg as LA2
from numpy import linalg as LA
from sklearn.neighbors import KNeighborsClassifier


# In[61]:


trainData = pandas.read_csv('Fraud.csv',
                             delimiter=',')


# In[62]:


# Examine a portion of the data frame
print(trainData)


# In[63]:


# Put the descriptive statistics into another dataframe
trainData_descriptive = trainData.describe()
print(trainData_descriptive)


# In[64]:


trainData[trainData['FRAUD'] == 1].count()


# In[65]:


per = (1189/5960)*100


# ## Question 3a

# In[66]:


print('percent of investigations are found to be fraudulent', per)


# ## Question 3b

# In[67]:




# xData = trainData[trainData['FRAUD'] == 0]
# xData['x'].head()
labels = ('Fraudlent 0', 'Fraudlent 1')
Frd1 = trainData[trainData['FRAUD'] == 1]
Frd0 = trainData[trainData['FRAUD'] == 0]
# print(Frd1)
plt.boxplot([Frd0['TOTAL_SPEND'], Frd1['TOTAL_SPEND']], notch=False, sym='+', vert=False, whis=1.5,
        positions=None, widths=None, patch_artist=True,
        bootstrap=None, usermedians=None, conf_intervals=None)
plt.title("Boxplot of Fraud for TOTAL_SPEND")
plt.suptitle("")
plt.yticks(np.arange(len(labels))+1,labels)
plt.xlabel("TOTAL_SPEND")
plt.ylabel("")
plt.grid(True)
plt.show()


# In[25]:


labels = ('Fraudlent 0', 'Fraudlent 1')
Frd1 = trainData[trainData['FRAUD'] == 1]
Frd0 = trainData[trainData['FRAUD'] == 0]
# print(Frd1)
plt.boxplot([Frd0['DOCTOR_VISITS'], Frd1['DOCTOR_VISITS']], notch=False, sym='+', vert=False, whis=1.5,
        positions=None, widths=None, patch_artist=True,
        bootstrap=None, usermedians=None, conf_intervals=None)
plt.title("Boxplot of Fraud for DOCTOR_VISITS")
plt.suptitle("")
plt.yticks(np.arange(len(labels))+1,labels)
plt.xlabel("DOCTOR_VISITS")
plt.ylabel("")
plt.grid(True)
plt.show()


# In[26]:


labels = ('Fraudlent 0', 'Fraudlent 1')
Frd1 = trainData[trainData['FRAUD'] == 1]
Frd0 = trainData[trainData['FRAUD'] == 0]
# print(Frd1)
plt.boxplot([Frd0['NUM_CLAIMS'], Frd1['NUM_CLAIMS']], notch=False, sym='+', vert=False, whis=1.5,
        positions=None, widths=None, patch_artist=True,
        bootstrap=None, usermedians=None, conf_intervals=None)
plt.title("Boxplot of Fraud for NUM_CLAIMS")
plt.suptitle("")
plt.yticks(np.arange(len(labels))+1,labels)
plt.xlabel("NUM_CLAIMS")
plt.ylabel("")
plt.grid(True)
plt.show()


# In[27]:


labels = ('Fraudlent 0', 'Fraudlent 1')
Frd1 = trainData[trainData['FRAUD'] == 1]
Frd0 = trainData[trainData['FRAUD'] == 0]
# print(Frd1)
plt.boxplot([Frd0['MEMBER_DURATION'], Frd1['MEMBER_DURATION']], notch=False, sym='+', vert=False, whis=1.5,
        positions=None, widths=None, patch_artist=True,
        bootstrap=None, usermedians=None, conf_intervals=None)
plt.title("Boxplot of Fraud for MEMBER_DURATION")
plt.suptitle("")
plt.yticks(np.arange(len(labels))+1,labels)
plt.xlabel("X")
plt.ylabel("")
plt.grid(True)
plt.show()


# In[68]:


labels = ('Fraudlent 0', 'Fraudlent 1')
Frd1 = trainData[trainData['FRAUD'] == 1]
Frd0 = trainData[trainData['FRAUD'] == 0]
# print(Frd1)
plt.boxplot([Frd0['OPTOM_PRESC'], Frd1['OPTOM_PRESC']], notch=False, sym='+', vert=False, whis=1.5,
        positions=None, widths=None, patch_artist=True,
        bootstrap=None, usermedians=None, conf_intervals=None)
plt.title("Boxplot of Fraud for OPTOM_PRESC")
plt.suptitle("")
plt.yticks(np.arange(len(labels))+1,labels)
plt.xlabel("OPTOM_PRESC")
plt.ylabel("")
plt.grid(True)
plt.show()


# In[69]:




labels = ('Fraudlent 0', 'Fraudlent 1')
Frd1 = trainData[trainData['FRAUD'] == 1]
Frd0 = trainData[trainData['FRAUD'] == 0]
# print(Frd1)
plt.boxplot([Frd0['NUM_MEMBERS'], Frd1['NUM_MEMBERS']], notch=False, sym='+', vert=False, whis=1.5,
        positions=None, widths=None, patch_artist=True,
        bootstrap=None, usermedians=None, conf_intervals=None)
plt.title("Boxplot of Fraud for NUM_MEMBERS")
plt.suptitle("")
plt.yticks(np.arange(len(labels))+1,labels)
plt.xlabel("NUM_MEMBERS")
plt.ylabel("")
plt.grid(True)
plt.show()


# ## Question 3c

# In[30]:


orthx = LA2.orth(trainData)
print("The orthonormalize x = \n", orthx)
orthx.shape


# In[31]:


# print("Input Matrix = \n", mat[0])


# In[32]:


extractedData = pandas.read_csv('Fraud.csv', delimiter=',',usecols=['TOTAL_SPEND', 'DOCTOR_VISITS', 'NUM_CLAIMS', 'MEMBER_DURATION', 'OPTOM_PRESC' , 'NUM_MEMBERS' ])


# In[33]:


mat = np.matrix(extractedData)


# In[34]:


print(mat)


# In[35]:


orthx = LA2.orth(mat)
print("The orthonormalize x = \n", orthx)
orthx.shape


# In[36]:


check = orthx.transpose().dot(orthx)
print("Also Expect an Identity Matrix = \n", check)


# In[37]:


xtx = mat.transpose() * mat
print("t(x) * x = \n", xtx)


# In[38]:


# Eigenvalue decomposition
evals, evecs = LA.eigh(xtx)
print("Eigenvalues of x = \n", evals)
print("Eigenvectors of x = \n",evecs)


# In[39]:


print(evals.shape)
evecs.shape


# In[40]:


# Here is the transformation matrix
transf = evecs * LA.inv(np.sqrt(np.diagflat(evals)));
print("Transformation Matrix = \n", transf)


# In[41]:


# Here is the transformed X
transf_x = mat * transf;
print("The Transformed x = \n", transf_x)


# In[42]:


# Check columns of transformed X
xtx = transf_x.transpose() * transf_x;
print("Expect an Identity Matrix = \n", xtx)


# In[43]:


fd_wIndex = trainData.set_index("CASE_ID")

print(fd_wIndex)


# In[44]:


trainData = fd_wIndex[['TOTAL_SPEND', 'DOCTOR_VISITS', 'NUM_CLAIMS', 'MEMBER_DURATION', 'OPTOM_PRESC' , 'NUM_MEMBERS']]


# In[45]:


print(trainData)


# In[46]:


target = fd_wIndex['FRAUD']
print(target)


# ## Question 3d

# In[47]:


neigh = KNeighborsClassifier(n_neighbors=5 , algorithm = 'brute', metric = 'euclidean')
nbrs = neigh.fit(transf_x, target)
sc = neigh.score(transf_x, target, sample_weight=None)


# In[48]:


print(sc)


# In[49]:


# See the classification probabilities
class_prob = nbrs.predict_proba(trainData)
for i in class_prob:
    print(class_prob)


# ## Ques 3e

# In[50]:


print(trainData)


# In[51]:


x = np.matrix(trainData.values)
print(x)

xtx = x.transpose() * x
print("t(x) * x = \n", xtx)


# In[52]:


# Eigenvalue decomposition
evals, evecs = np.linalg.eigh(xtx)
print("Eigenvalues of x = \n", evals)
print("Eigenvectors of x = \n",evecs)


# In[53]:


# Here is the transformation matrix
transf = evecs * np.linalg.inv(np.sqrt(np.diagflat(evals)));
print("Transformation Matrix = \n", transf)


# In[54]:


# Here is the transformed X
transf_x = x * transf;
print("The Transformed x = \n", transf_x)


# In[55]:


# Check columns of transformed X
xtx = transf_x.transpose() * transf_x;
print("Expect an Identity Matrix = \n", xtx)


# ## Question 3 f

# In[56]:


focal = [[7500, 15, 3, 127, 2, 2]]  


# In[57]:


nbrs.predict(focal*transf)


# In[58]:


nbrs.predict_proba(focal)

