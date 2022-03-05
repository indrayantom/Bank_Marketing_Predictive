# ✨ Bank Marketing : Predictive Analysis ✨ 
The data is related to direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe to a term deposit. Link for the dataset : https://www.kaggle.com/henriqueyamahata/bank-marketing?select=bank-additional-names.txt

![python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Microsoft PowerPoint](https://img.shields.io/badge/Microsoft_PowerPoint-B7472A?style=for-the-badge&logo=microsoft-powerpoint&logoColor=white)


Just in case you are not aware, the .ipynb file contains both the codes and the explanation, which you can also see easily [here](https://indrayantom.github.io/Bank_Marketing_Predictive/). Since this work was my final project in a Data Science Bootcamp, I also provide a Google Slide Presentation to summarize all the procedures and findings, see it [here](https://docs.google.com/presentation/d/1XxfgQliJreu22A_ZNEC0bhTklE0TXCUiNk4umcbErE0/edit?usp=sharing).

## Objectives 
Nowadays, banks can make money with a lot of different ways, even though at the core they are still considered as lenders. Generally, they make money by borrowing it from the depositors, who are compensated later with a certain interest rate and security for their funds. Then, the borrowed money will be lent out to borrowers who need it at the moment. The borrowers however, are charged with higher interest rate than what is paid to the depositors. The difference between interest rate paid and interest rate received is often called as interest rate spead, where the banks gain profit from.

The main object of this research is a certain Portuguese bank institution who was trying to collect money from the depositors through direct marketing campaigns. In general, direct marketing campaign requires in-house or out-sourced call centres. Even though the information of sales cost are not provided, several articles said that it could put a considerable strain on the expense ratio of the product. In this case, the sales team of the bank contacted about 40000 customers randomly, while only 11% (around 4500) of them were willing to deposit their money.

By assuming one direct marketing call for one customer costs the company 2 dollars and profits them 50  dollars. It can be said that from 2000 customers contacted, the bank will gain profit of 7000 dollars in total, knowing that 11 in every 100 random calls result in successful sales. However, the bank soughted for some approaches that will help them conduct more effective marketing campaign with better conversion rate, and machine learning is one of the answers.

**Business Objective**

Providing more detail to previous statement, this research is carried out to identify customers who are more likely to subscribe and build a machine learning model that is be able to predict the probability of a certain customer become a depositor, in order to help the sales team conducting more effective marketing campaign or higher profit. Keep in mind that based on the previous example, 10% increase in customer rate will be followed by almost 16% increase in profit. Conversion rate is used as the key metrics related to the evaluation of machine learning model.

## Libraries
Libraries such as [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [matplotlib](https://matplotlib.org/), and [seaborn](https://seaborn.pydata.org/) are the most commonly used in the analysis. However, I also used [sklearn](https://scikit-learn.org/stable/) to conduct the predictive analysis with some classification models.
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
%matplotlib inline
%config InlineBackend.figure_format = 'svg'
import warnings
warnings.filterwarnings('ignore')
import sweetviz as sv

from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, roc_auc_score, classification_report,f1_score,precision_recall_curve,roc_curve
from sklearn.model_selection import cross_val_score
from imblearn import over_sampling,under_sampling
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

from pycaret.classification import *

import shap
from sklearn.metrics import roc_curve
```

## Result Preview
By Cross Validation and Hyperparameter tuning, it is found that Light Gradient Boosting Machine gives the best result compared to other models with AUC 80% , F1 44% and training time 0.25s
![shapp_project](https://user-images.githubusercontent.com/92590596/156873400-f4e34d86-77e9-4461-b92d-70f811222be3.png)

From SHAP value plot above, one can see that the results is quite consistent to the previous results we obtain in EDA process. The variables are sorted descendingly based on their contribution, which color indicates the value level of the feature (red means high, blue means low -> 100 in age will be labelled by red, meanwhile 18 is labelled by blue) and SHAP value measures the impact of that value level to the outcomes, for example, if the age is positively correlated to the subscription probability, red will be on the chart's positive side and blue in the negative side.

- From the plot, 3 features with the most contribution are sos-con variables nr.employed, emp.var.rate and euribor3m. The plot depicts that when nr.employed are pretty low, the subscription probability increases and otherwise. The insights of euribor3m and emp.var.rate are less clear, even though we can understand higher emp.var.rate means lower probability and lower euribor3m means higher probability in overall.

- Younger and older clients tend to have higher probability to subscribe which result is similar to the previous EDA analysis. As well as poutcome and campaign.

- Lower value of consumer price index and confidence index results in higher probability to subscribe. 

- May (month) and Monday (day) give smaller subscription probability.

