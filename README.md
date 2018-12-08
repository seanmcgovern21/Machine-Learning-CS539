

<p style="text-align: center;">   <b>Team Members </b></p>
<p style="text-align: center;">   Bhon Bunnag, Sean McGovern, Ying Fang, Mengdi Li, Jidapa Thadajarassiri </p>


<!---//Motivation Paragraph  //////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////--->

<h1 align="center">  Motivation </h1>

<p style="text-indent :5em;" > </p>The 'Google Analytics Customer Revenue Prediction'  is a Kaggle competition to predict the revenue generated per customer from data of the Google Merchandise Store (GStore). The data presents us with a skewed target variable, where only a small number of customer visits generate non-zero revenue. Some customers may also visit the GStore multiple times, which produces sequential data. State of the art algorithms such as linear regression and regression trees are insufficient for predicting skewed and sequential data. As such, we propose a joint classification-regression technique, which is more robust against skewed data. Recurrent Neural Networks (RNN) will be integrated into the proposed system to handle sequential data.
Business owners will obviously find this joint model useful to analyze customer generated revenue. Furthermore, this model can be generalized to be used for any sequential data with skewed target variable. 



<!---//Problem Statement and Challenge Paragraph ///////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////--->

<h1 align="center"> Problem Statement and Challenge </h1>

We would like to implement machine learning systems that accurately predicts customer generated revenue. 
The dataset being used in very skewed. The data set also contains recursive data instances. 


<!---//   Training DATA   /////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////--->
<h1 align="center">  Training Data  </h1>

The dataset is provided by Kaggle competition. There are 903,653 visiting records with 55 features of visiting information, such as visitDate, visitorID and visitNumber. The records are from 2016-08-01 to 2017-08-01. A visitor corresponds to one or many visiting records, which produces sequential data. Among the useful 33 features, besides the 4 ID and 2 datetime features, there are 4 numerical features and 23 categorical features. The target variable is “totals.transactionRevenue”. It is noticeable that only 11,515 visiting records (<1.3%) of the dataset contains non-zero value.



<div class="row">
  <div class="column">
    <img src="https://raw.githubusercontent.com/seanmcgovern21/Machine-Learning-CS539/master/images/data_visit_based.png" width="425" height="auto" >
  </div>
  <div class="column">
   <img src="https://raw.githubusercontent.com/seanmcgovern21/Machine-Learning-CS539/master/images/data_customer_based.png" width="425" height="auto">
  </div>
</div>



<!---//Visit Based Model /////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////--->
<h1 align="center">  Visit-based Model   </h1>


<h2> State of the Art </h2>

Linear Regression is a classic state of the art algorithm for predicting real numerical target variables. However, linear regression will produce high bias, and not suitable for the dataset if the ground truth relationship in the dataset is non-linear. Polynomial  regression will solve these  issues, but may lead to overfitting. Decision Tree is also another usable state of the art algorithm for this task. Given that both categorical and numerical features are present in the dataset, the decision tree may be more suitable than Linear/Polynomial regression. Additionally, this algorithm also performs feature selection automatically. 

<!---//Linear Regression Code //////////////////////////////////////////////////////////////////////////////--->
<h3>Linear Regression Code</h3>

<div style="height:210px;width:850px;overflow:auto;">
<pre><code class="python">
	
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

train_mse = []
train_rmse = []
val_mse = []
val_rmse = []

for i in range(fold):
    print('\n\nfold:', i)
    val = processed_train_df[processed_train_df['fullVisitorId'].isin(id_cv[i])]
    train = processed_train_df[~processed_train_df['fullVisitorId'].isin(id_cv[i])]
    x_tr = train.iloc[:,2:]
    y_tr = train.iloc[:,1]
    log_y_tr = np.log1p(y_tr)
    x_val = val.iloc[:,2:]
    y_val = val.iloc[:,1]
    log_y_val = np.log1p(y_val)
    
    # --- INSERT YOUR MODEL -----
    model = LinearRegression().fit(x_tr, log_y_tr)
    log_y_tr_pred = model.predict(x_tr)
    # ---------------------------
    
    log_y_tr_pred = [0 if i < 0 else i for i in log_y_tr_pred]
    log_y_val_pred = model.predict(x_val)
    log_y_val_pred = [0 if i < 0 else i for i in log_y_val_pred]
    
    mse_tr, mse_val = getMse(x_tr, train, val, log_y_tr_pred, log_y_val_pred)
    train_mse.append(mse_tr)
    train_rmse.append(np.sqrt(mse_tr))
    val_mse.append(mse_val)
    val_rmse.append(np.sqrt(mse_val))


print('\n\nAverage:')
print('train_mse_5fold', np.mean(train_mse))
print('train_rmse_5fold', np.mean(train_rmse))
print('val_mse_5fold', np.mean(val_mse))
print('val_rmse_5fold', np.mean(val_rmse))

</code></pre>
</div>


<!---//Polynomial Regression Code //////////////////////////////////////////////////////////////////////////////--->
<h3>Polynomial Regression Code </h3>


<div style="height:210px;width:850px;overflow:auto;">
<pre><code class="python">
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

train_mse = []
train_rmse = []
val_mse = []
val_rmse = []

for i in range(fold):
    print('\n\nfold:', i)
    val = processed_train_df[processed_train_df['fullVisitorId'].isin(id_cv[i])]
    train = processed_train_df[~processed_train_df['fullVisitorId'].isin(id_cv[i])]
    x_tr = train.iloc[:,2:]
    y_tr = train.iloc[:,1]
    log_y_tr = np.log1p(y_tr)
    x_val = val.iloc[:,2:]
    y_val = val.iloc[:,1]
    log_y_val = np.log1p(y_val)
    
    # --- INSERT YOUR MODEL -----
    model_pipeline = Pipeline([('poly',PolynomialFeatures(degree=2)),
                  ('linear', LinearRegression(fit_intercept=False))])
    model = model_pipeline.fit(x_tr, log_y_tr)
    log_y_tr_pred = model.predict(x_tr)
    # ---------------------------
    
    log_y_tr_pred = [0 if i < 0 else i for i in log_y_tr_pred]
    log_y_val_pred = model.predict(x_val)
    log_y_val_pred = [0 if i < 0 else i for i in log_y_val_pred]
    
    mse_tr, mse_val = getMse(x_tr, train, val, log_y_tr_pred, log_y_val_pred)
    train_mse.append(mse_tr)
    train_rmse.append(np.sqrt(mse_tr))
    val_mse.append(mse_val)
    val_rmse.append(np.sqrt(mse_val))


print('\n\nAverage:')
print('train_mse_5fold', np.mean(train_mse))
print('train_rmse_5fold', np.mean(train_rmse))
print('val_mse_5fold', np.mean(val_mse))
print('val_rmse_5fold', np.mean(val_rmse))


</code></pre>
</div>

<!---//Regression Tree Code //////////////////////////////////////////////////////////////////////////////--->
<h3>Regression Tree Code</h3>


<div style="height:210px;width:850px;overflow:auto;">
<pre><code class="python">

from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

train_mse = []
train_rmse = []
val_mse = []
val_rmse = []

for i in range(fold):
    print('\n\nfold:', i)
    val = processed_train_df[processed_train_df['fullVisitorId'].isin(id_cv[i])]
    train = processed_train_df[~processed_train_df['fullVisitorId'].isin(id_cv[i])]
    x_tr = train.iloc[:,2:]
    y_tr = train.iloc[:,1]
    log_y_tr = np.log1p(y_tr)
    x_val = val.iloc[:,2:]
    y_val = val.iloc[:,1]
    log_y_val = np.log1p(y_val)
    
    # --- INSERT YOUR MODEL -----
    model = DecisionTreeRegressor(max_depth=10)
    model.fit(x_tr, log_y_tr)
    log_y_tr_pred = model.predict(x_tr)
    # ---------------------------
    
    log_y_tr_pred = [0 if i < 0 else i for i in log_y_tr_pred]
    log_y_val_pred = model.predict(x_val)
    log_y_val_pred = [0 if i < 0 else i for i in log_y_val_pred]
    
    mse_tr, mse_val = getMse(x_tr, train, val, log_y_tr_pred, log_y_val_pred)
    train_mse.append(mse_tr)
    train_rmse.append(np.sqrt(mse_tr))
    val_mse.append(mse_val)
    val_rmse.append(np.sqrt(mse_val))


print('\n\nAverage:')
print('train_mse_5fold', np.mean(train_mse))
print('train_rmse_5fold', np.mean(train_rmse))
print('val_mse_5fold', np.mean(val_mse))
print('val_rmse_5fold', np.mean(val_rmse))   

</code></pre>
</div>


<!---// Proposed Model Code //////////////////////////////////////////////////////////////////////////////--->
<h2 align="left"> Proposed Model - Pre-classified Regression </h2>
<p align="center">
<img src="https://raw.githubusercontent.com/seanmcgovern21/Machine-Learning-CS539/master/images/Pre-classified%20Regression.png"  width="500" height="auto">
</p>


<!---//Customer Based Model /////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////--->
<h1 align="center">  Customer-based Model  </h1>

<h2 align="left"> State of the Art </h2>
Recurrent neural network (RNN)

<h2 align="left"> Proposed Model - Weighted Classified Subnetwork for Regressionv</h2>
This system contains 2 parts: the main-network and the sub-network. We applied RNN model in the main-network and the goal is to predict generated revenue from customers. This predicted revenue is weighted by the output from sub-network which is the probability of customer spending. In order to get the probability of customer spending, we again applied RNN model but we added sigmoid function in the last layer of the sub-network. The total loss of this proposed method is the sum of log_loss in sub-network and MSE_loss in main-network.
<img src="https://raw.githubusercontent.com/seanmcgovern21/Machine-Learning-CS539/master/images/Weighted%20Classified%20%20Subnetwork%20for%20Regression.png"  width="500" height="auto">

<h2 align="center"> Proposed Method </h2>

![Figure 2: Weighted Classified Subnetwork for Regression](https://github.com/seanmcgovern21/Machine-Learning-CS539/tree/master/Images/sub-network_classification.png)

Our proposed systems combine the knowledge from classification, which differentiate revenue and non-revenue generating customers, into the regression model. We believe this system will be more robust against skewed data. We first propose a simple straightforward system that runs a classifier before running the regression model i.e. Pre-classified Regression, and a more complex system that also handles sequential data i.e. Weighted Classified Subnetwork for Regression.
Pre-classified Regression
The first proposed idea is to apply a classification model before a linear regression model. The classification model is used to predict whether or not a customer will generate revenue. If the revenue is predicted as non-zero, the sample will not enter our regression model. By doing this, the large number of zero-revenue samples will not impact the training process of the regression model. For classification, we plan to perform Logistic Regression, Decision Tree, KNN and Support Vector Machine on the training set separately and choose the algorithm with the best performance. For regression, we plan to apply the models we mentioned in the state-of-the-art section. The Pre-classified Regression will be implemented using Scikit-learn in Python.

Figure 1: Pre-classified Regression
Weighted Classified Subnetwork for Regression 
The general idea of this system is shown in figure 2. Sequential information will be captured by implementing RNN in the main-network while the skewed target will be tackled by implementing logistic regression in the sub-network. The sub-network aims to separate revenue generating customers from non-revenue generating customers by computing the probability of spending of each customer. In the main-network, sequential information of customer visits is fed into the system and the revenue generated per customer (final predicted value) will be weighted by the output from sub-network. The system will be implemented using Pytorch in Python.

Figure 2: Weighted Classified Subnetwork for Regression



<!---//Results: Visit Based Model /////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////--->
<h1 align="center"> Results: Visit Based Model </h1>

<h2 align="center"> Evaluation </h2>

Performance of our proposed methods will be compared to the state-of-the-art methods using Root-Mean-Square Error (RMSE) which is defined as
<img src="https://raw.githubusercontent.com/seanmcgovern21/Machine-Learning-CS539/master/images/RMSE.png" width="500" height="auto">


<h2 align="center"> Results </h2>

<img src="https://raw.githubusercontent.com/seanmcgovern21/Machine-Learning-CS539/master/images/result_visit_based/result_summary_visit-based.png" width="500" height="auto">




<!---//Results: Customer Based Model /////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////--->
<h1 align="center"> Results: Customer Based Model </h1>

<h2 align="center"> Evaluation </h2>


Performance of our proposed methods will be compared to the state-of-the-art methods (section 4.2.) using Root-Mean-Square Error (RMSE) which is defined as
RMSE = 1ni=1n(yi-yi)2
where 	nis the number of customers, 
yi=log(customer revenue+1),
yi=log(predicted customer revenue+1).

<h2 align="center"> Results </h2>

<img src="https://raw.githubusercontent.com/seanmcgovern21/Machine-Learning-CS539/master/images/result_customer_based/result_summary_customer-based.png" width="500" height="auto">






<!---//   Reference ////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////--->


<h2 align="center"> 
Reference
</h2>

Competition Website: https://www.kaggle.com/c/ga-customer-revenue-prediction/ 
[1] Li, Yanghao et al. “Online Human Action Detection using Joint Classification-Regression Recurrent Neural Networks.” ECCV (2016). https://arxiv.org/abs/1604.05633
[2] Lathuilière, Stéphane et al. “A Comprehensive Analysis of Deep Regression.” CoRR abs/1803.08450 (2018): n. pag. https://arxiv.org/pdf/1803.08450.pdf 

[3] Cooper, Robin, et al. "Profit priorities from activity-based costing." Harvard business review 69.3 (1991): 130-135.

[4] Flach, Peter. (2012). Machine Learning The Art and Science of Algorithms that Make Sense of Data. Cambridge, United Kingdom: Cambridge University Press. 





<!---//   Acknoledgements  ////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////--->
<h2 align="center">  Acknowledgements   </h2>	
	
We would like to thank the Professor for providing a great course in Machine Learning and we are thankful for the opportunity to complete a challenging project together. 




