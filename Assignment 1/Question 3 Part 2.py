
# coding: utf-8

# In[4]:


# Load the necessary libraries
import numpy
import pandas
from sklearn.neighbors import NearestNeighbors as kNN




# In[5]:


fd = pandas.read_csv('D:\MCS\Sem 4\ML\Assignment 1\Fraud.csv',
                       delimiter=',')

fd_wIndex = fd.set_index("CASE_ID")


# In[6]:


# print(trainData)
# Specify the kNN
kNNSpec = kNN(n_neighbors = 5, algorithm = 'brute', metric = 'euclidean')


# In[7]:


# Specify the training data
trainData = fd_wIndex[['TOTAL_SPEND', 'DOCTOR_VISITS', 'NUM_CLAIMS', 'MEMBER_DURATION', 'OPTOM_PRESC' , 'NUM_MEMBERS']]
trainData.describe()


# In[8]:


# Build nearest neighbors
nbrs = kNNSpec.fit(trainData)
distances, indices = nbrs.kneighbors(trainData)


# ## Question 3 E 1st part

# In[9]:


# Find the nearest neighbors of these focal observations       
focal = [[7500, 15, 3, 127, 2, 2]]     # Mercedes-Benz_271

myNeighbors = nbrs.kneighbors(focal, return_distance = False)
print("My Neighbors = \n", myNeighbors)



# In[10]:


# Orthonormalized the training data
print(trainData)
x = numpy.matrix(trainData.values)
print(x)

xtx = x.transpose() * x
print("t(x) * x = \n", xtx)



# In[11]:


# Eigenvalue decomposition
evals, evecs = numpy.linalg.eigh(xtx)
print("Eigenvalues of x = \n", evals)
print("Eigenvectors of x = \n",evecs)



# In[12]:


# Here is the transformation matrix
transf = evecs * numpy.linalg.inv(numpy.sqrt(numpy.diagflat(evals)));
print("Transformation Matrix = \n", transf)



# In[13]:


# Here is the transformed X
transf_x = x * transf;
print("The Transformed x = \n", transf_x)



# In[14]:


# Check columns of transformed X
xtx = transf_x.transpose() * transf_x;
print("Expect an Identity Matrix = \n", xtx)



# In[15]:


nbrs = kNNSpec.fit(transf_x)
distances, indices = nbrs.kneighbors(transf_x)



# ## Question 3 E 2nd part

# In[16]:


# Find the nearest neighbors of these focal observations       
focal = [[7500, 15, 3, 127, 2, 2]]      # Mercedes-Benz_271

transf_focal = focal * transf;

myNeighbors_t = nbrs.kneighbors(transf_focal, return_distance = False)
print("My Neighbors = \n", myNeighbors_t)

