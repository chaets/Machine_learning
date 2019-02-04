
# coding: utf-8

# In[2]:


# Load the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas
import sklearn.tree as tree
from numpy import percentile


# In[3]:


trainData = pandas.read_csv('NormalSample.csv',
                             delimiter=',', usecols = ['x'])
Data = pandas.read_csv('NormalSample.csv')


# In[4]:


# Examine a portion of the data frame
print(trainData)


# ## Ques 1c

# In[5]:


# Put the descriptive statistics into another dataframe
trainData_descriptive = trainData.describe()
print(trainData_descriptive)
print(trainData.describe())


# In[7]:


# Visualize the histogram of the x variable
trainData.hist(column='x', bins=400)
plt.title("Histogram of x")
plt.xlabel("x")
plt.ylabel("Number of Observations")
plt.xticks(np.arange(26,36,step=1))
plt.grid(axis="x")
plt.show()


# # Q1 : According to Izenman (1991) method, what is the recommended bin-width for the histogram of x?

# In[8]:


bin_fd = np.histogram_bin_edges(trainData, bins='fd', range=(26.30, 35.40))
print(bin_fd)


# ## Bin Width Calculation = 0.38461538

# In[9]:


np.diff(bin_fd)


# In[10]:


bin_fd


# In[12]:


# Visualize the histogram of the x variable

trainData.hist(column='x', bins=bin_fd)
plt.title("Histogram of x")
plt.xlabel("x")
plt.ylabel("Number of Observations")
plt.xticks(np.arange(26,36,step=0.3846*5))
plt.grid(axis="x")
plt.show()


# ## 1b) What are the minimum and the maximum values of the field x?
# ### Ans:- min      26.300000
# ### max      35.400000

# In[13]:


a = 26
b = 36
bin_list = [0.1, 0.5, 1, 2 ]


# In[15]:


# Visualize the histogram of the x variable
bin_fd = np.histogram_bin_edges(trainData, bins='auto', range=(26, 36))
trainData.hist(column='x', bins=bin_fd)
plt.title("Histogram of x")
plt.xlabel("x")
plt.ylabel("Number of Observations")
plt.xticks(np.arange(26,36,step=1))
plt.grid(axis="x")
plt.show()


# ## Question 1 d to 1 g

# In[18]:


# Visualize the histogram of the x variable
for i in bin_list:
    trainData.hist(column='x', bins=np.arange(a, b + i, i))
    plt.title("Histogram of x")
    plt.xlabel("x")
    plt.ylabel("Number of Observations")
    plt.xticks(np.arange(26,36,step=1))
    plt.grid(axis="x")
    print('For h = ', i)
    plt.show()
    
    


# ## Question 2

# In[19]:


# calculate a 5-number summary
# calculate quartiles
quartiles = percentile(trainData, [25, 50, 75])
print(quartiles)
trainData.describe()


# In[20]:


# calculate a 5-number summary
# calculate quartiles
def qurt(trainData):
    quartiles = percentile(trainData, [25, 50, 75])
    return(quartiles)


# In[21]:


# calculate min/max
data_min, data_max = trainData.min(), trainData.max()
print(data_min[0], data_max[0])


# In[22]:


B=plt.boxplot(trainData)
[item.get_ydata() for item in B['whiskers']]


# In[23]:


def iqras(trainData):
    median = np.median(trainData)
    upper_quartile = np.percentile(trainData, 75)
    lower_quartile = np.percentile(trainData, 25)

    iqr = upper_quartile - lower_quartile
    upper_whisker = trainData[trainData<=upper_quartile+1.5*iqr].max()
    lower_whisker = trainData[trainData>=lower_quartile-1.5*iqr].min()
    return (iqr, upper_whisker, lower_whisker)


# In[24]:


print(iqras(trainData))


# In[25]:


trainData.boxplot(column='x', vert = False)
plt.title("Boxplot of X for both groups")
plt.suptitle("")
plt.xlabel("X")
plt.ylabel("")
plt.grid(axis="y")
plt.show()


# In[26]:


Data.head()


# ## for Group 0

# In[27]:


xData = Data[Data['group'] == 0]
xData['x'].head()
xData['x'].describe()


# In[28]:


qurt(xData['x'])


# In[29]:


iqras(xData['x'])


# In[30]:


# Put the descriptive statistics into another dataframe
trainData_descr = xData['x'].describe()
print(trainData_descr)


# In[31]:


xData.boxplot(column='x', vert = False)
plt.title("Boxplot of X for group 0")
plt.suptitle("")
plt.xlabel("X")
plt.ylabel("")
plt.grid(axis="y")
plt.show()


# In[32]:


xData.head()
xData['x'].head()
print(iqras(xData['x']))


# In[33]:


xData1 = Data[Data['group'] == 1]
xData1['x'].head()
trainData_descr1 = xData1['x'].describe()
print(trainData_descr1)


# In[34]:


iqras(xData1['x'])


# ## Question 2 c

# In[44]:


xData1.boxplot(column='x', vert = False)
plt.title("Boxplot of X for group 1")
plt.suptitle("")
plt.xlabel("X")
plt.ylabel("")
plt.grid(axis="y")
plt.show()


# In[36]:


xData1.head()
xData1['x'].head()
print(iqras(xData1['x']))


# ## Quest 2 d

# In[42]:


labels = ('All Data', 'Group 0', 'Group 1')

# plt.boxplot([trainData['x'],xData['x'],xData1['x']], vert = False)
plt.boxplot([trainData['x'],xData['x'],xData1['x']], notch=False, sym='+', vert=False, whis=1.5,
        positions=None, widths=None, patch_artist=True,
        bootstrap=None, usermedians=None, conf_intervals=None)
plt.title("Boxplot of X for group 0")
plt.suptitle("")
plt.xlabel("X")
plt.ylabel("")
plt.yticks(np.arange(len(labels))+1,labels)
plt.grid(True)
plt.show()

