from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, MultilayerPerceptronClassifier
from pyspark.ml.feature import VectorAssembler, PCA, StandardScaler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Initialize Spark Session
spark = SparkSession.builder.master("local[*]").appName("Higgs Boson Detection").config("spark.executor.memory", "15g").config("spark.driver.memory", "20g").getOrCreate()

# Load data
df = spark.read.csv("../Data/HIGGS.csv", inferSchema=True, header=False)

#ds_df = df.sampleBy("_c0", fractions={0: 0.1, 1: 0.1}, seed=42)

# Assembling features
assembler = VectorAssembler(inputCols=df.columns[1:], outputCol="raw_features")

# Apply PCA
feat_no = 28
#pca = PCA(k=feat_no, inputCol="raw_features", outputCol="features")

scaler = StandardScaler(inputCol="raw_features", outputCol="features")

#pipeline_stages = [assembler, pca]
pipeline_stages = [assembler, scaler]

# Splitting the data
(train_data, test_data) = df.randomSplit([0.8, 0.2], seed=42)

sampled_data = df.sampleBy("_c0", fractions={0: 0.01, 1: 0.01}, seed=42)
(train_sample, test_sample) = sampled_data.randomSplit([0.8, 0.2], seed=42)

# Define classifiers
rf = RandomForestClassifier(labelCol="_c0", featuresCol="features")
gbt = GBTClassifier(labelCol="_c0", featuresCol="features")
mlp = MultilayerPerceptronClassifier(labelCol="_c0", featuresCol="features")

# Parameter grid setup for each classifier
rf_paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [1,5,10]) \
    .addGrid(rf.maxDepth, [1,5, 10]) \
    .addGrid(rf.maxBins, [10,20,50]) \
    .build()

gbt_paramGrid = ParamGridBuilder() \
    .addGrid(gbt.maxIter, [10,20,30]) \
    .addGrid(gbt.maxDepth, [1,5,10]) \
    .addGrid(gbt.stepSize, [0.1,0.2,0.05]) \
    .build()

input_layer = feat_no
hidden_layer1 = 5
hidden_layer2 = 5
hidden_layer3 = 10
output_layer = 2

mlp_paramGrid = ParamGridBuilder() \
    .addGrid(mlp.layers, [
        [input_layer, hidden_layer1, output_layer],
        [input_layer, hidden_layer1, hidden_layer2, output_layer],
        [input_layer, hidden_layer1, hidden_layer2, hidden_layer3, output_layer]
    ]) \
    .addGrid(mlp.maxIter, [20,50,100]) \
    .addGrid(mlp.blockSize, [5,10,20]) \
    .build()

# Evaluators for AUC and accuracy
auc_evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="_c0", metricName="areaUnderROC")
acc_evaluator = MulticlassClassificationEvaluator(labelCol="_c0", predictionCol="prediction", metricName="accuracy")

# Cross-validation setup for each model
rf_pipeline = Pipeline(stages=pipeline_stages + [rf])
gbt_pipeline = Pipeline(stages=pipeline_stages + [gbt])
mlp_pipeline = Pipeline(stages=pipeline_stages + [mlp])

rf_cv = CrossValidator(estimator=rf_pipeline, estimatorParamMaps=rf_paramGrid, evaluator=auc_evaluator, numFolds=5)
gbt_cv = CrossValidator(estimator=gbt_pipeline, estimatorParamMaps=gbt_paramGrid, evaluator=auc_evaluator, numFolds=5)
mlp_cv = CrossValidator(estimator=mlp_pipeline, estimatorParamMaps=mlp_paramGrid, evaluator=auc_evaluator, numFolds=5)

# Model training and selection
best_rf = rf_cv.fit(train_sample)
best_gbt = gbt_cv.fit(train_sample)
best_mlp = mlp_cv.fit(train_sample)

rf_best = best_rf.bestModel.stages[-1].extractParamMap()
gbt_best = best_gbt.bestModel.stages[-1].extractParamMap()
mlp_best = best_mlp.bestModel.stages[-1].extractParamMap()

sample_rf_predictions = best_rf.transform(test_sample)
sample_gbt_predictions = best_gbt.transform(test_sample)
sample_mlp_predictions = best_mlp.transform(test_sample)

sample_rf_auc = auc_evaluator.evaluate(sample_rf_predictions)
sample_gbt_auc = auc_evaluator.evaluate(sample_gbt_predictions)
sample_mlp_auc = auc_evaluator.evaluate(sample_mlp_predictions)
sample_rf_accuracy = acc_evaluator.evaluate(sample_rf_predictions)
sample_gbt_accuracy = acc_evaluator.evaluate(sample_gbt_predictions)
sample_mlp_accuracy = acc_evaluator.evaluate(sample_mlp_predictions)


final_rf = RandomForestClassifier(labelCol="_c0", featuresCol="features",numTrees=rf_best[rf.numTrees], maxDepth=rf_best[rf.maxDepth],maxBins=rf_best[rf.maxBins])

final_gbt = GBTClassifier(labelCol="_c0", featuresCol="features",maxIter=gbt_best[gbt.maxIter],maxDepth=gbt_best[gbt.maxDepth],stepSize=gbt_best[gbt.stepSize])

final_mlp = MultilayerPerceptronClassifier(labelCol="_c0", featuresCol="features",layers=mlp_best[mlp.layers],maxIter=mlp_best[mlp.maxIter],blockSize=mlp_best[mlp.blockSize])

rf_bestparams = {
    "numTrees": rf_best[rf.numTrees],
    "maxDepth" : rf_best[rf.maxDepth],
    "maxBins" : rf_best[rf.maxBins]
}

gbt_bestparams = {
    "maxIter" : gbt_best[gbt.maxIter],
    "maxDepth" : gbt_best[gbt.maxDepth],
    "stepSize" : gbt_best[gbt.stepSize]
}

mlp_bestparams = {
    "layers" : mlp_best[mlp.layers],
    "maxIter" : mlp_best[mlp.maxIter],
    "blockSize" : mlp_best[mlp.blockSize]
}

rf_pipeline1 = Pipeline(stages=pipeline_stages + [final_rf])
gbt_pipeline1 = Pipeline(stages=pipeline_stages + [final_gbt])
mlp_pipeline1 = Pipeline(stages=pipeline_stages + [final_mlp])

rf_bestmodel = rf_pipeline1.fit(train_data)
gbt_bestmodel = gbt_pipeline1.fit(train_data)
mlp_bestmodel = mlp_pipeline1.fit(train_data)

# Evaluation on test data
rf_predictions = rf_bestmodel.transform(test_data)
gbt_predictions = gbt_bestmodel.transform(test_data)
mlp_predictions = mlp_bestmodel.transform(test_data)

rf_auc = auc_evaluator.evaluate(rf_predictions)
gbt_auc = auc_evaluator.evaluate(gbt_predictions)
mlp_auc = auc_evaluator.evaluate(mlp_predictions)
rf_accuracy = acc_evaluator.evaluate(rf_predictions)
gbt_accuracy = acc_evaluator.evaluate(gbt_predictions)
mlp_accuracy = acc_evaluator.evaluate(mlp_predictions)

# Write results
with open("../Output/Q3_output.txt", "w") as file:
    file.write(f"Sample Data Random Forest AUC: {sample_rf_auc}, Accuracy: {sample_rf_accuracy}\n")
    file.write(f"Sample Data Gradient Boosting Trees AUC: {sample_gbt_auc}, Accuracy: {sample_gbt_accuracy}\n")
    file.write(f"Sample Data Neural Network AUC: {sample_mlp_auc}, Accuracy: {sample_mlp_accuracy}\n\n")
    file.write(f"Random Forest Best Hyper-Parameters:\n {rf_bestparams}\n")
    file.write(f"Gradient Boosting Trees Best Hyper-Parameters:\n {gbt_bestparams}\n")
    file.write(f"Neural Network Best Hyper-Parameters:\n {mlp_bestparams}\n\n")
    
    file.write(f"Random Forest AUC: {rf_auc}, Accuracy: {rf_accuracy}\n")
    file.write(f"Gradient Boosting Trees AUC: {gbt_auc}, Accuracy: {gbt_accuracy}\n")
    file.write(f"Neural Network AUC: {mlp_auc}, Accuracy: {mlp_accuracy}\n")

# Print the results
print(f"Sample Random Forest AUC: {sample_rf_auc}, Accuracy: {sample_rf_accuracy}\n")
print(f"Sample Gradient Boosting Trees AUC: {sample_gbt_auc}, Accuracy: {sample_gbt_accuracy}\n")
print(f"Sample Neural Network AUC: {sample_mlp_auc}, Accuracy: {sample_mlp_accuracy}\n\n")
print(f"Random Forest Best Hyper-Parameters:\n {rf_bestparams}\n")
print(f"Gradient Boosting Trees Best Hyper-Parameters:\n {gbt_bestparams}\n")
print(f"Neural Network Best Hyper-Parameters:\n {mlp_bestparams}\n\n")
print(f"Random Forest AUC: {rf_auc}, Accuracy: {rf_accuracy}")
print(f"Gradient Boosting Trees AUC: {gbt_auc}, Accuracy: {gbt_accuracy}")
print(f"Neural Network AUC: {mlp_auc}, Accuracy: {mlp_accuracy}")

# Cleanup
spark.stop()
