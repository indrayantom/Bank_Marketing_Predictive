# ✨ Bank Marketing : Predictive Analysis ✨ 
The data is related to direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe to a term deposit. Link for the dataset : https://www.kaggle.com/henriqueyamahata/bank-marketing?select=bank-additional-names.txt

![python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Microsoft PowerPoint](https://img.shields.io/badge/Microsoft_PowerPoint-B7472A?style=for-the-badge&logo=microsoft-powerpoint&logoColor=white)


Just in case you didn't know, the telcoPredictive_Indra.ipynb file contains both the codes and the explanation. In addition, the professional writing of the analysis is also available in a .pdf file. You also can view the Google Collab docs [here](https://colab.research.google.com/drive/1_DIwM4A7kMZOEInNVh2GWwKaJ5IhoBFT#scrollTo=bNXeBG2UykdD) . Feel free to download and clone this repo 😃😉.

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
By Hyperparameter Tuning and Decision threshold adjusting, it is found that Logistic Regression gives the best result compared to KNN and Random Forest in terms of AUC and F1 score.
![Fi](https://user-images.githubusercontent.com/92590596/156796504-440be765-e057-48a9-b559-ca07f7849550.jpg)

After being tuned with GridSearchCV method and adjusted to 0.3 decision/probability threshold, the improvement becomes significant compared to the default model as the Recall and F1 score are increased to 76\% (**+23\%**) and 63\% (**+5\%**) respectively . With an AUC score of 83\% (**+0\%**), those metrics are successfully increased without a lot of reduction in accuracy, only **3\%** lower from the default model with the score of 77\%.

Futhermore, the feature importance (coefficient) of the Logistic regression model can be seen on above figure. Notice that the coefficients are both positive and negative. It can be elaborated as the predictor of Class 1 (Churn Yes) has positive coefficient whereas the predictor of Class 0 (Churn No) has negative coefficient. Overall, it is evident that the graph  is already in accordance to the result of EDA project carried out before on similar dataset . Contract, tenure, InternetService, PaymentMethod and some additional internet services such as TechSupport and Streaming are considered as the key features on which the business strategists should focus to improve the level of satisfaction and retent the customer. Read the EDA [here](https://github.com/indrayantom/telco_custmer_dea).

