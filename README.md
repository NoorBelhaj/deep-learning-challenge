# Project Scope
This Jupyter Notebook is designed to showcase the process of building a deep neural network model for binary classification using TensorFlow and Keras. The main objective is to predict whether a charitable organization will be successful (IS_SUCCESSFUL) based on various features from the dataset.

# Preprocessing
Importing Dependencies: The necessary libraries and modules are imported, including pandas, tensorflow, train_test_split from sklearn.model_selection, and StandardScaler from sklearn.preprocessing.

Reading the Data: The dataset (charity_data.csv) is read from the provided URL using pd.read_csv() and stored in the application_df DataFrame.

# Data Clean up and preparartion for ML model
Dropping Non-Beneficial Columns: The columns 'EIN' and 'NAME' are identified as non-beneficial for the prediction task and dropped from the DataFrame.

Counting Unique Values: The number of unique values in each column of new_application_df is determined using nunique().

Binning Application Types: The values of the 'APPLICATION_TYPE' column are examined to identify which types have low occurrence (less than 500 instances). These low-occurring types are replaced with the label "Other" in the DataFrame.

Binning Classification Types: Similar to the previous step, the 'CLASSIFICATION' column is examined to identify low-occurring types (less than 1000 instances) and replace them with the label "Other" in the DataFrame.

Converting Categorical Data to Numeric: The categorical data in the DataFrame is converted to numeric format using pd.get_dummies(), creating new columns for each category with binary values.

Final Step is Splitting Data: The preprocessed data is split into features (X) and the target (y) arrays. Then, the data is further split into training and testing datasets using train_test_split.

# ML process start
## Compile, Train, and Evaluate the Model
Model Architecture: A deep neural network (DNN) model is defined using TensorFlow and Keras. It consists of an input layer with 43 units (features), two hidden layers with 80 and 30 units respectively, and an output layer with 1 unit using the sigmoid activation function for binary classification.
'nn.compile() is a method used to configure the learning process before training the model. It sets up the necessary components for training, including the loss function, optimizer, and evaluation metrics. Let's break down the arguments of nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]):

loss="binary_crossentropy":
The loss parameter specifies the loss function used to measure how well the neural network's predictions match the actual target values during training. In this case, "binary_crossentropy" is used, which is a common loss function for binary classification tasks. It is suitable when you have two classes and the network aims to predict a probability value between 0 and 1 for each class independently.

optimizer="adam":
The optimizer parameter determines the optimization algorithm used to update the neural network's weights during training. "Adam" is a popular optimization algorithm that combines the ideas of both the AdaGrad and RMSprop optimizers. It adapts the learning rates for each parameter during training and is efficient in handling large datasets and complex models.

metrics=["accuracy"]:
The metrics parameter defines the evaluation metrics that you want to monitor during training. In this case, "accuracy" is used as the metric. Accuracy is a common evaluation metric for classification tasks, indicating the proportion of correct predictions made by the model on the validation or test set.


# Report & MOdel Analysis
Overview of the analysis: The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. Alphabet Soup provided a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. The objective is is build a model that will predict if a project reqesting funding will be successful.

# Data Preprocessing

1. What variable(s) are the target(s) for your model? 
    - IS_SUCCESSFUL
2. What variable(s) are the features for your model? 
    - APPLICATION_TYPE	
    - AFFILIATION	
    - CLASSIFICATION	
    - USE_CASE	
    - ORGANIZATION	
    - STATUS	
    - INCOME_AMT	
    - SPECIAL_CONSIDERATIONS	
    - ASK_AMT	
3. What variable(s) should be removed from the input data because they are neither targets nor features? 
    - 'EIN' and 
    - 'NAME'

4. Compiling, Training, and Evaluating the Model
Model Architecture: A deep neural network (DNN) model is defined using TensorFlow and Keras. It consists of an input layer with 43 units (features), two hidden layers with 80 and 30 units respectively, and an output layer with 1 unit using the sigmoid activation function for binary classification.
'nn.compile() is a method used to configure the learning process before training the model. It sets up the necessary components for training, including the loss function, optimizer, and evaluation metrics. Let's break down the arguments of nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]):

loss="binary_crossentropy":
The loss parameter specifies the loss function used to measure how well the neural network's predictions match the actual target values during training. In this case, "binary_crossentropy" is used, which is a common loss function for binary classification tasks. It is suitable when you have two classes and the network aims to predict a probability value between 0 and 1 for each class independently.

optimizer="adam":
The optimizer parameter determines the optimization algorithm used to update the neural network's weights during training. "Adam" is a popular optimization algorithm that combines the ideas of both the AdaGrad and RMSprop optimizers. It adapts the learning rates for each parameter during training and is efficient in handling large datasets and complex models.

metrics=["accuracy"]:
The metrics parameter defines the evaluation metrics that you want to monitor during training. In this case, "accuracy" is used as the metric. Accuracy is a common evaluation metric for classification tasks, indicating the proportion of correct predictions made by the model on the validation or test set.

# Testing 
How many neurons, layers, and activation functions did you select for your neural network model, and why?
Were you able to achieve the target model performance?
What steps did you take in your attempts to increase model performance?

In the present case, the values have been changed over few iterations to find the best suited model with best loss and aaccuracy values.
Variable are First and second Layer inputs:

L1 Input	L2 Input	Loss	    Accuracy
130	        64	        0.5649	    0.7327
130	        8	        0.5521	    0.7347
130	        4	        0.5470	    0.7332
90	        36	        0.5552	    0.7326
80	        30	        0.5547	    0.7327
64	        8	        0.5507	    0.7352
43	        8	        0.5500	    0.7354
32	        4	        0.5482	    0.7314
30	        15	        0.5478	    0.7350
16	        8	        0.5475	    0.7352


# Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.
It appears that increasing the size of the hidden layer (Layer 2) does not necessarily result in significant improvements in accuracy. The accuracy values for different configurations are relatively close to each other. 
The loss values are also similar across different configurations, indicating that the model's performance is relatively stable. 
The only option I would eventually explore is the income amount scaling, it has been scalled as gategoricalcal values. It would be interesting to scale the data and give and see if there is any correlation with success rate.
