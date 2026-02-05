Mobile Price Prediction using Machine Learning

 Project Overview

The Mobile Price Prediction project aims to predict the price range of a mobile phone based on its technical specifications such as RAM, battery power, internal memory, camera quality, screen size, and connectivity features.

This is a supervised machine learning classification problem where the target variable is the price range.

Objective

Analyze mobile phone features.

Build a machine learning model to predict price category.

Compare multiple ML algorithms.

Select the best-performing model.

Enable predictions on new unseen mobile data.



Problem Type

Machine Learning Type: Supervised Learning

Task: Classification

Target Variable: price_range

0 → Low cost

1 → Medium cost

2 → High cost


3 → Very high cost



Dataset Description

The dataset contains various mobile phone specifications.

Feature	Description

battery_ power	Battery Capacity in mAh

blue	Has Bluetooth or not

clock_speed	Processor speed

dual_sim	Has dual sim support or not

fc	Front camera megapixels

four_g	Has 4G or not

int_memory	Internal Memory in GB

m_deep	Mobile depth in cm.

mobile_wt	: Weight in gm 

n_cores	Processor Core Count

pc	Primary Camera megapixels

px_height	Pixel Resolution height

px_width	Pixel Resolution width

ram	Ram in MB

sc_h	Mobile Screen height in cm

sc_w	Mobile Screen width in cm

talk_time	Time a single battery charge will last. In hours.

three_g	Has 3G or not

touchscreen	Has touch screen or not

Wi-Fi	Has Wi-Fi or not





Price_range: This is the target

○ 0 = low cost

○ 1 = medium cost

○ 2 = high cost

○ 3 = very high cost

Tech Stack

•	Programming Language: Python

•	Libraries:

o	NumPy

o	Pandas

o	Matplotlib

o	Seaborn

o	Scikit-learn
•	IDE / Tools:

o	Google Colab



Imported the dependencies.

Loaded the dataset to pandas dataframe.

Exploratory Data Analysis (EDA): - Performed the EDA to find the datatypes, null values distribution and summary statistics of the data.

Data understanding and cleaning

Getting the information of dataset: -That is checking the datatypes, finding null values in dataset.

After understanding the data, I found that all the columns were numerical.

Finding null values sum from all columns respectively. There were no null values in the dataset.

Correlation among the Numerical columns: - To find the correlation among the numerical features analyzing the correlation I found ram has very correlation (~0.91) with price_range. Also, I found some of the features which are least correlated with price_range like m_dep, clock_speed, n_cores, dual_sim, wifi, touch_screen, mobile_wt so we can drop them in future.



Checking the distribution of target column: uniformly distributed

For finding the outliers I plotted the box plots between different features.


ram vs price_range

Battery power vs price range

Then I found the correlation heatmap and took some of the observations from it: This plot shows us the strength of the ram feature

From the correlation heat map I found: -

RAM, battery power, and px_width have strong influence on mobile price.

Splitting the data into Features and target

Splitting the data into training and testing

We are splitting data in such a way so that for training we take 80% of data and for testing we took 20% of data.

Data Preprocessing

Performing standardization:- This is done to bring the data to a common scale

Feature scaling using StandardScaler

Machine Learning Models Used

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

Models used	Accuracy	Precision	F1 score	Recall

Logistic Regression	97.25%	 1.00	0.99	0.98

		0.95	0.97	1.00

		0.95	0.95	0.95

		0.98	0.97	0.96

Decision Tree Classifier	85.25%	0.90	0.92	0.90

		0.79	0.82	0.85

		0.79	0.76	0.74

		0.87	0.89	0.91

Random Forest Classifier	90%	0.96	0.95	0.94

		0.88	0.88	0.88

		0.82	0.84	0.87

		0.94	0.92	0.90

Models used and Evaluation Metrics











Confusion Matrix for Models trained: - For Mobile Price Prediction, confusion matrix  tells us:

How often the model correctly or incorrectly predicted the price range (0, 1, 2, 3).



Performed hyperparameter tuning: - To find the best set of parameters using GridSearchCV 

And I found the best model having parameters as: -

'max_depth': None, 


'min_samples_split': 2,

 'n_estimators': 200

}



With hyperparameter tuning also I found the accuracy equals 90%.

 Best Model



 Logistic Regression: -





•	High accuracy


•	Robust to overfitting

•	Handles feature importance well

Later, I found the important features: - Features with low correlation and low importance were removed to reduce noise and improve model performance.

 

 Prediction on New Data: - I evaluated both Logistic Regression and Random Forest. Although Random Forest is more complex, Logistic Regression achieved higher accuracy due to strong relationships in the data, so I selected it as the final model.




Now, the trained model can predict the price range for new mobile specifications by passing feature values in the same format as training data.





For the given problem I found that the Logistic Regression is giving me the best accuracy, so I selected it as my best model.

Best Model: - Logistic Regression

Accuracy: - 97%

Key Feature that affects price range is: - RAM

•	RAM is the most influential feature in determining the mobile price.

•	Logistic Regression achieved the highest accuracy.

•	The model can help classify mobile phones into price categories effectively




