#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 12:13:53 2019

@author: josalop
"""
from pyspark.sql import SparkSession
import pandas as pd
import seaborn as sns

spark = SparkSession \
    .builder \
    .appName("Python Spark Boston Housing MLLib predictions") \
    .getOrCreate()


# Spark load data
house_df = spark.read.load("data/bostonHousingWithHeaders.csv",
                     format="csv", sep=",", inferSchema="true", header="true")
house_df.take(1)
house_df.cache()
house_df.printSchema()


# Descriptive analytics
house_df.describe().toPandas()

house_pd = house_df.sample(withReplacement=False, fraction=1.0).toPandas()
sns.pairplot(house_pd)

import matplotlib.pyplot as plt
import seaborn as sns
clr = ['blue', 'green', 'red'] #different colors for different subplot
fig,  ax = plt.subplots(ncols=3,figsize=(15,3)) #create three empty subplots

#select three features and check them
for i, var in enumerate(['RM', 'LSTAT', 'PTRATIO']):
    sns.distplot(house_pd[var],  color = clr[i], ax=ax[i])
    ax[i].axvline(house_pd[var].mean(), color=clr[i], linestyle='solid', linewidth=2)
    ax[i].axvline(house_pd[var].median(), color=clr[i], linestyle='dashed', linewidth=2)
plt.show()


#check hwo the features are related to the target value
fig, ax = plt.subplots(ncols=3,figsize=(15,3))

for i, var in enumerate(['RM', 'LSTAT', 'PTRATIO']):
    sns.regplot(house_pd[var], house_pd['MEDV'], color=clr[i], ax=ax[i])
    ax[i].set(ylim=(0, None))
plt.show()

 
#display the correlation between the different features
# boston_df.corr() calculates the correlation matrix between the features
fig,ax= plt.subplots(figsize = (10,10))
sns.heatmap(house_pd.corr(), cmap = "coolwarm",  square=True, annot=True)
plt.show()



from pyspark.ml.feature import VectorAssembler
ftr_names = list(house_pd.columns[:-1])
vectorAssembler = VectorAssembler(inputCols = ftr_names, outputCol = 'features')
vhouse_df = vectorAssembler.transform(house_df)
vhouse_df = vhouse_df.select(['features', 'MEDV'])
vhouse_df.show(3)


# Split 
splits = vhouse_df.randomSplit([0.7, 0.3])
train_df = splits[0]
test_df = splits[1]

# Linear regression
from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol = 'features', labelCol='MEDV', maxIter=10, regParam=0.3
                      , elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))


# Use model summary to get some model statistics
trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

train_df.describe().show()


# Test set evaluation
lr_predictions = lr_model.transform(test_df)
lr_predictions.select("prediction","MEDV","features").show(5)
from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="MEDV",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))


test_result = lr_model.evaluate(test_df)
print("Root Mean Squared Error (RMSE) on test data = %.4f" % test_result.rootMeanSquaredError)


print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()



# Decision tree regression
from pyspark.ml.regression import DecisionTreeRegressor
dt = DecisionTreeRegressor(featuresCol ='features', labelCol = 'MEDV')
dt_model = dt.fit(train_df)
dt_predictions = dt_model.transform(test_df)

# Metrics

dt_evaluator = RegressionEvaluator(
    labelCol="MEDV", predictionCol="prediction", metricName="rmse")
rmse = dt_evaluator.evaluate(dt_predictions)
print("Root Mean Squared Error (RMSE) on test data = %.4f" % rmse)



# Gradient Boosted Trees
from pyspark.ml.regression import GBTRegressor
gbt = GBTRegressor(featuresCol = 'features', labelCol = 'MEDV', maxIter=10)
gbt_model = gbt.fit(train_df)
gbt_predictions = gbt_model.transform(test_df)
gbt_predictions.select('prediction', 'MEDV', 'features').show(5)


metric = 'r2'
gbt_evaluator = RegressionEvaluator( labelCol="MEDV", predictionCol="prediction")\
                                    .setMetricName(metric)
performance = gbt_evaluator.evaluate(gbt_predictions)
print("Performance metric (%s) on test data = %.4f" % (metric, performance))