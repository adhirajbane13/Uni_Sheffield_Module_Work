from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler,StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from sklearn.datasets import fetch_openml
import pandas as pd

# Initialize Spark Session for 4 cores
spark = SparkSession.builder \
    .master("local[4]") \
    .appName("Logistic and Generalized Linear Regression with Regularization") \
    .getOrCreate()

# Load and convert dataset
data = fetch_openml(data_id=41214, as_frame=True).data
df = spark.createDataFrame(data)

# Pre-processing
df = df.withColumn("hasClaim", when(col("ClaimNb") > 0, 1).otherwise(0))

# Feature engineering pipeline
categoricalColumns = ['Area', 'VehPower', 'VehBrand', 'VehGas', 'Region']
stages = []

# Index and encode categorical columns
for categoricalCol in categoricalColumns:
    indexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
    encoder = OneHotEncoder(inputCols=[indexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [indexer, encoder]

# Assemble numeric columns
numericCols = ['Exposure', 'VehAge', 'DrivAge', 'BonusMalus', 'Density']
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features1")
scaler = StandardScaler(inputCol="features1", outputCol="features")
stages += [assembler, scaler]

# Under-sample the dataset to address class imbalance
seed = 24073  # Define the seed for reproducibility
train = df.sampleBy("hasClaim", fractions={0: 0.7, 1: 0.7}, seed=seed)
test = df.subtract(train)
# Split the dataset into training and test sets
#train, test = train_df.randomSplit([0.7, 0.3], seed=24073)

# Sample 10% from the training set for cross-validation
#train_sample = train.sample(False, 0.1, seed=24073)
train_sample = train.sampleBy("hasClaim", fractions={0: 0.1, 1: 0.1}, seed=seed)

# Setup Generalized Linear Regression with cross-validation
glr = GeneralizedLinearRegression(labelCol="ClaimNb", family="poisson", link="log", featuresCol="features")
glrParamGrid = ParamGridBuilder() \
    .addGrid(glr.regParam, [0.001, 0.01, 0.1, 1, 10]) \
    .build()
stages_glr = stages + [glr]
glrPipeline = Pipeline(stages=stages_glr)
glrCrossval = CrossValidator(estimator=glrPipeline,
                             estimatorParamMaps=glrParamGrid,
                             evaluator=RegressionEvaluator(labelCol="ClaimNb", metricName="rmse"),
                             numFolds=5)
glrModel = glrCrossval.fit(train_sample)
# Extract best parameters from the cross-validation model
best_glr_params = glrModel.bestModel.stages[-1].extractParamMap()

# Instantiate a new Generalized Linear Regression model with the best parameters
final_glr = GeneralizedLinearRegression(
    labelCol="ClaimNb", 
    featuresCol="features", 
    family="poisson", 
    link="log", 
    regParam=best_glr_params[glr.regParam]  
)

# Create a new pipeline with the same stages but replace the GLR model
final_glr_pipeline = Pipeline(stages=stages + [final_glr])

# Fit the model on the full training set
final_glr_model = final_glr_pipeline.fit(train)

# Transform the test set
final_glr_predictions = final_glr_model.transform(test)
glrEvaluator = RegressionEvaluator(labelCol="ClaimNb", metricName="rmse")
final_glr_rmse = glrEvaluator.evaluate(final_glr_predictions)

# Retrieve coefficients if needed
final_glr_coefficients = str(final_glr_model.stages[-1].coefficients)

# Setup Logistic Regression with L1 regularization (Lasso)
lrL1 = LogisticRegression(labelCol="hasClaim", featuresCol="features", elasticNetParam=1.0)
lrL1ParamGrid = ParamGridBuilder() \
    .addGrid(lrL1.regParam, [0.001, 0.01, 0.1, 1, 10]) \
    .build()
stages_l1lr = stages + [lrL1]
lrL1Pipeline = Pipeline(stages=stages_l1lr)
lrL1Crossval = CrossValidator(estimator=lrL1Pipeline,
                              estimatorParamMaps=lrL1ParamGrid,
                              evaluator=MulticlassClassificationEvaluator(labelCol="hasClaim",metricName="accuracy"),
                              numFolds=5)
lrL1Model = lrL1Crossval.fit(train_sample)
# Extract best parameters from the L1 logistic regression cross-validation model
best_lrL1_params = lrL1Model.bestModel.stages[-1].extractParamMap()

# Instantiate a new Logistic Regression model with the best parameters and L1 regularization
final_lrL1 = LogisticRegression(
    labelCol="hasClaim", 
    featuresCol="features", 
    elasticNetParam=1.0,  # L1 regularization
    regParam=best_lrL1_params[lrL1.regParam]  # for example
)

# Create a new pipeline with the same stages but replace the Logistic Regression model
final_lrL1_pipeline = Pipeline(stages=stages + [final_lrL1])

# Fit the model on the full training set
final_lrL1_model = final_lrL1_pipeline.fit(train)

# Transform the test set
final_lrL1_predictions = final_lrL1_model.transform(test)
lrL1Evaluator = MulticlassClassificationEvaluator(labelCol="hasClaim",metricName="accuracy")
final_lrL1_accuracy = lrL1Evaluator.evaluate(final_lrL1_predictions)

# Retrieve coefficients if needed
final_lrL1_coefficients = str(final_lrL1_model.stages[-1].coefficients)


# Setup Logistic Regression with L2 regularization (Ridge)
lrL2 = LogisticRegression(labelCol="hasClaim", featuresCol="features", elasticNetParam=0.0)
lrL2ParamGrid = ParamGridBuilder() \
    .addGrid(lrL2.regParam, [0.001, 0.01, 0.1, 1, 10]) \
    .build()
stages_l2lr = stages + [lrL2]
lrL2Pipeline = Pipeline(stages=stages_l2lr)
lrL2Crossval = CrossValidator(estimator=lrL2Pipeline,
                              estimatorParamMaps=lrL2ParamGrid,
                              evaluator=MulticlassClassificationEvaluator(labelCol="hasClaim",metricName="accuracy"),
                              numFolds=5)
lrL2Model = lrL2Crossval.fit(train_sample)
# Extract best parameters from the L2 logistic regression cross-validation model
best_lrL2_params = lrL2Model.bestModel.stages[-1].extractParamMap()

# Instantiate a new Logistic Regression model with the best parameters and L2 regularization
final_lrL2 = LogisticRegression(
    labelCol="hasClaim", 
    featuresCol="features", 
    elasticNetParam=0.0,  # L2 regularization
    regParam=best_lrL2_params[lrL2.regParam]  # for example, ensure correct parameter extraction
)

# Create a new pipeline with the same stages but replace the Logistic Regression model
final_lrL2_pipeline = Pipeline(stages=stages + [final_lrL2])

# Fit the model on the full training set
final_lrL2_model = final_lrL2_pipeline.fit(train)

# Transform the test set
final_lrL2_predictions = final_lrL2_model.transform(test)
lrL2Evaluator = MulticlassClassificationEvaluator(labelCol="hasClaim",metricName="accuracy")
final_lrL2_accuracy = lrL2Evaluator.evaluate(final_lrL2_predictions)

# Retrieve coefficients if needed
final_lrL2_coefficients = str(final_lrL2_model.stages[-1].coefficients)


# Output results to a text file
output = f"Generalized Linear Regression RMSE: {final_glr_rmse}\nGLR Coefficients: {final_glr_coefficients}\n"
output += f"Logistic Regression L1 Accuracy: {final_lrL1_accuracy*100:0.4f}%\nLR L1 Coefficients: {final_lrL1_coefficients}\n"
output += f"Logistic Regression L2 Accuracy: {final_lrL2_accuracy*100:0.4f}%\nLR L2 Coefficients: {final_lrL2_coefficients}\n"

output_file_path = "../Output/Q2_output.txt"
with open(output_file_path, "w") as file:
    file.write(output)

# Stop Spark session
spark.stop()
