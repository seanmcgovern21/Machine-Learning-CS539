<p style="text-align: center;">   <b>Team Members </b></p>
<p style="text-align: center;">   Bhon Bunnag, Sean McGovern, Ying Fang, Mengdi Li, Jidapa Thadajarassiri </p>

<h2 align="center"> 
Introduction
</h2>

The 'Google Analytics Customer Revenue Prediction'  is a Kaggle competition to predict the revenue generated per customer from data of the Google Merchandise Store (GStore). The data presents us with a skewed target variable, where only a small number of customer visits generate non-zero revenue. Some customers may also visit the GStore multiple times, which produces sequential data. State of the art algorithms such as linear regression and regression trees are insufficient for predicting skewed and sequential data. As such, we propose a joint classification-regression technique, which is more robust against skewed data. Recurrent Neural Networks (RNN) will be integrated into the proposed system to handle sequential data.
Business owners will obviously find this joint model useful to analyze customer generated revenue. Furthermore, this model can be generalized to be used for any sequential data with skewed target variable. 





<!---//Objective Paragraph /////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////--->

<h2 align="center"> 
Objective
</h2>

We would like to implement machine learning systems that accurately predicts customer generated revenue. 

<h2 align="center"> 
Challenge
</h2>

The dataset being used in very skewed. The data set also contains recursive data instances. 

<h2 align="center"> 
Data
</h2>


<h2 align="center"> 
Visit-based Model
</h2>

<h3 align="left"> 
State of the Art
</h3>

Linear Regression is a classic state of the art algorithm for predicting real numerical target variables. However, linear regression will produce high bias, and not suitable for the dataset if the ground truth relationship in the dataset is non-linear. Polynomial  regression will solve these  issues, but may lead to overfitting.  
Decision Tree is also another usable state of the art algorithm for this task. Given that both categorical and numerical features are present in the dataset, the decision tree may be more suitable than Linear/Polynomial regression. Additionally, this algorithm also performs feature selection automatically. 

<h3 align="left"> 
Proposed Model - Pre-classified Regression
</h3>


<h2 align="center"> 
Customer-based Model
</h2>

<h3 align="left"> 
State of the Art
</h3>

<h3 align="left"> 
Proposed Model - Weighted Classified Subnetwork for Regression
</h3>




<h2 align="center"> 
Evaluation
</h2>

Performance of our proposed methods will be compared to the state-of-the-art methods (section 4.2.) using Root-Mean-Square Error (RMSE) which is defined as
<img src="../images/RMSE.png">


<h2 align="center"> 
Results
</h2>

<h3 align="left"> 
Visit-based Model
</h3>

<h3 align="left"> 
Customer-based Model
</h3>













<h2 align="center"> 
State of the Art
</h2>

Linear Regression is a classic state of the art algorithm for predicting real numerical target variables. However, linear regression will produce high bias, and not suitable for the dataset if the ground truth relationship in the dataset is non-linear. Polynomial  regression will solve these  issues, but may lead to overfitting.  
	Decision Tree is also another usable state of the art algorithm for this task. Given that both categorical and numerical features are present in the dataset, the decision tree may be more suitable than Linear/Polynomial regression. Additionally, this algorithm also performs feature selection automatically. 

##Linear Regression

'<https://github.com/seanmcgovern21/Machine-Learning-CS539/blob/master/gg_analytics-RNN.ipynb>'


<h2 align="center"> 
Proposed Method
</h2>
![Figure 2: Weighted Classified Subnetwork for Regression](https://github.com/seanmcgovern21/Machine-Learning-CS539/tree/master/Images/sub-network_classification.png)

Our proposed systems combine the knowledge from classification, which differentiate revenue and non-revenue generating customers, into the regression model. We believe this system will be more robust against skewed data. We first propose a simple straightforward system that runs a classifier before running the regression model i.e. Pre-classified Regression, and a more complex system that also handles sequential data i.e. Weighted Classified Subnetwork for Regression.
Pre-classified Regression
The first proposed idea is to apply a classification model before a linear regression model. The classification model is used to predict whether or not a customer will generate revenue. If the revenue is predicted as non-zero, the sample will not enter our regression model. By doing this, the large number of zero-revenue samples will not impact the training process of the regression model. For classification, we plan to perform Logistic Regression, Decision Tree, KNN and Support Vector Machine on the training set separately and choose the algorithm with the best performance. For regression, we plan to apply the models we mentioned in the state-of-the-art section. The Pre-classified Regression will be implemented using Scikit-learn in Python.

Figure 1: Pre-classified Regression
Weighted Classified Subnetwork for Regression 
The general idea of this system is shown in figure 2. Sequential information will be captured by implementing RNN in the main-network while the skewed target will be tackled by implementing logistic regression in the sub-network. The sub-network aims to separate revenue generating customers from non-revenue generating customers by computing the probability of spending of each customer. In the main-network, sequential information of customer visits is fed into the system and the revenue generated per customer (final predicted value) will be weighted by the output from sub-network. The system will be implemented using Pytorch in Python.

Figure 2: Weighted Classified Subnetwork for Regression



<h2 align="center"> 
Evaluation
</h2>


Performance of our proposed methods will be compared to the state-of-the-art methods (section 4.2.) using Root-Mean-Square Error (RMSE) which is defined as
RMSE = 1ni=1n(yi-yi)2
where 	nis the number of customers, 
yi=log(customer revenue+1),
yi=log(predicted customer revenue+1).


<h2 align="center"> 
Reference
</h2>

Competition Website: https://www.kaggle.com/c/ga-customer-revenue-prediction/ 
[1] Li, Yanghao et al. “Online Human Action Detection using Joint Classification-Regression Recurrent Neural Networks.” ECCV (2016). https://arxiv.org/abs/1604.05633
[2] Lathuilière, Stéphane et al. “A Comprehensive Analysis of Deep Regression.” CoRR abs/1803.08450 (2018): n. pag. https://arxiv.org/pdf/1803.08450.pdf 

[3] Cooper, Robin, et al. "Profit priorities from activity-based costing." Harvard business review 69.3 (1991): 130-135.

[4] Flach, Peter. (2012). Machine Learning The Art and Science of Algorithms that Make Sense of Data. Cambridge, United Kingdom: Cambridge University Press. 




<h2 align="center"> 
Acknowledgements
</h2>	
	
We would like to thank the Professor for providing a great course in Machine Learning and we are thankful for the opportunity to complete a challenging project together. 



You can use the [editor on GitHub](https://github.com/seanmcgovern21/ML_CS539/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

