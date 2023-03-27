# ml-python-challenge
A simple code challenge in Python applied to a Machine Learning problem

## Code Challenge description
The challenge requests to create Machine Learning model, according to dataset provided (file dataset.csv) which contains a list of ships and related features. The model should be able to predict 'crew' when receives input values.
dataset.csv file has the fields below:
- Ship_name
- Cruise_line
- Age
- Tonnage
- passengers
- length
- cabins
- passenger_density
- crew

## Solution
The solution is a Regression Analysis Model (Linear Regression, Polynomial Regression, Random Forests, etc.) because of continuous variables. We build the solutions on three steps:
   1. data analysis focusing on find features correlated to target variable
   2. build the simplest model (es. Linear Regression) and evaluate it, if the result is not so good
   3. build more complex model (es. Random Forests) and evaluate it
   
### Data Analysis
For this step we can use:
- scatterplot matrix, to select correlated variables
- covariance matrix, to evaluate the best correlated variable (for a first linear regression model)

Running covariance_matrix.py python code we can plot the two images below:

![image1](https://user-images.githubusercontent.com/107040849/227962887-575f3965-4743-4be4-8c4d-0cf2d889170e.png)

looking on the first image we see that we have to investigate “Tonnage” and “passengers” variables;

![image2](https://user-images.githubusercontent.com/107040849/227963222-fa7002ee-efa8-43d0-97e9-a06090d56d03.png)

while looking on the second image we see that we have to investigate “cabins” and “length” variables too: the four variable seems to be correlated to crew (target variable).

Using covariance matrix:

![image3](https://user-images.githubusercontent.com/107040849/227963694-f8a55f49-36bd-49be-a163-0f76f4a395e8.png)

we see that passengers variable is the candidate to evaluate a Linear Regression Model.

### Linear Regression
Linear Regression Model is the simplest one and it is very similar to Adaptive Linear Neuron Model. We can use the implementation in Scikit-Learn library which has an optimized implementation (file sklearn.linear.py).

For evaluation we use two measures:

1. mean squared error
2. coefficient of determination

below the results:

![image4](https://user-images.githubusercontent.com/107040849/227964517-9c7ae7a0-5e2e-4308-89b1-330c46c14cc0.png)

0.93 for test dataset is already a good result.

### Random Forests
We investigated also Random Forests. This model is based on decision tree (random_forest.py).

below the resutls:

![image5](https://user-images.githubusercontent.com/107040849/227965176-6df0f86a-d6a0-4b56-bc05-8e0851ccd751.png)

0.98 for train dataset and 0.95 for test dataset is the best results.

The source code is taken from:

preferred-citation:
  type: book
  authors: "Raschka Sebastian"
    website: "https://sebastianraschka.com/news/2019-news/"
