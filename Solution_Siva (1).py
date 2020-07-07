#!/usr/bin/env python
# coding: utf-8

# Hi team,
# 
# As per the problem satatement given ,i would like to provide my perspective on how to go with the input coulumn with respect 
# to the problem statement.

# In[3]:


##1.--->First part of the question being to the data type of the column.
#Sol: To just provide .info() command to know data types of entire columns in the data or .describe() and .dtype by pandas.
Ex:
    df_train.info()
    df_train["Example"].dtype ##through pandas (import pandas as pd)


# In[7]:


##2

import pandas as pd
import numpy as np
def data_information(df, id_cols): ## id_cols-1.This includes the columns that does have any significance like on IDV like ref id,index etc and IDV    ## Removing ID columns
    df = df.drop(columns=id_cols)

    ## Empty Data Frame
    data_info = pd.DataFrame(np.random.randn(0, 12) * 0,
                             columns=['No. of Observations (Nrow)',
                                      'No. of Variables (Ncol)',
                                      'No. of Numeric Variables',
                                      'No. of Factor Variables',
                                      'No. of Categorical Variables',
                                      'No. of Logical Variables',
                                      'No. of Date Variables',
                                      'No. of Zero variance Variables (Uniform)',
                                      '% of Variables having complete cases',
                                      '% of Variables having <=50% missing cases',
                                      '% of Variables having >50% missing cases',
                                      '% of Variables having >90% missing cases'])

    ## Data Inofrmation
    data_info.loc[0, 'No. of Observations (Nrow)'] = df.shape[0]
    data_info.loc[0, 'No. of Variables (Ncol)'] = df.shape[1]
    data_info.loc[0, 'No. of Numeric Variables'] = df._get_numeric_data().shape[1]
    data_info.loc[0, 'No. of Factor Variables'] = df.select_dtypes(include='category').shape[1]
    data_info.loc[0, 'No. of Logical Variables'] = df.select_dtypes(include='bool').shape[1]
    data_info.loc[0, 'No. of Categorical Variables'] = df.select_dtypes(include='object').shape[1]
    data_info.loc[0, 'No. of Date Variables'] = df.select_dtypes(include='datetime64').shape[1]
    data_info.loc[0, 'No. of Zero variance Variables (Uniform)'] = df.loc[:, df.apply(pd.Series.nunique) == 1].shape[1]

    null_per = pd.DataFrame(df.isnull().sum()/df.shape[0])
    null_per.columns = ['null_per']

    data_info.loc[0, '% of Variables having complete cases'] = null_per[null_per.null_per == 0].shape[0] * 100 /                                                                df.shape[1]
    data_info.loc[0, '% of Variables having <=50% missing cases'] = null_per[null_per.null_per <= 0.50].shape[0] * 100 /                                                                     df.shape[1]
    data_info.loc[0, '% of Variables having >50% missing cases'] = null_per[null_per.null_per > 0.50].shape[0] * 100 /                                                                    df.shape[1]
    data_info.loc[0, '% of Variables having >90% missing cases'] = null_per[null_per.null_per > 0.90].shape[0] * 100 /                                                                    df.shape[1]

    ## Transposing 
    data_info = data_info.transpose()
    data_info.columns = ['Value']
    data_info['Value'] = data_info['Value'].astype(int)

    return data_info

2.##Request you to run the above code with any dataset.

-->Type of variable and its count are obatined using above code.
-->Further codes can be writen to categorize it as SSN, Phone, FirstName, Lastname, Age, quantity etc based on size of content 
   present inside it or even using describe function would do the job (describe() helps in showing the mean,median,mode).
-->Visually lot of analysis can be done with respect to the given problem statement as unrealistic values in case of phone        numbers,quantity catches attention.
# In[ ]:


3.## Sensitivity (SSN/Passport numbers have high sensitive vs reference number - define high if high sensitivity , medium and low)
-->If its the case of finding the sensitivity within the independent variables, correlation matrix can be used.It defines how 
   each variable relates itself with one another.
-->Covariance yield better results only if the varibles are scaled first.
-->High correlation value with in the independent varibles signifies low sensitivity with the dependent varible as high 
    value resonates similar relationship(or direct relationship).
--> .corr() can be used directly on the dataset or correlation heatmap can be drawn using seaborn.

##Sensitivity in general is used to evaluate the results of classification model.
-->Its generally drawn from confusion matrix as part of model evalution process done at the end.
-->Its calculated in terms True positive rate and False positive rate.
-->Further the model can be fine tuned based on the sensitivity value by rejecting few columns if needed to improve the accuracy of 
   test data.

