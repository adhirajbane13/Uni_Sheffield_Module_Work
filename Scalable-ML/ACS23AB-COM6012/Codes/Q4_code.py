import matplotlib
matplotlib.use('Agg')
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.window import Window
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import avg, col, explode
from pyspark.sql.functions import split as sql_split
import numpy as np

# Initialize Spark Session
spark = SparkSession.builder \
        .master("local[64]") \
        .appName("Movie Recommendation System") \
        .config("spark.executor.memory", "15g") \
        .config("spark.driver.memory", "20g") \
        .config("spark.local.dir","/mnt/parscratch/users/acs23ab") \
        .getOrCreate()


#Part A
# Load data
rating_data = spark.read.load('../Data/ml-20m/ratings.csv', format='csv', inferSchema="true", header="true").cache()

# Sort data by timestamp
sorted_data = rating_data.orderBy('timestamp', ascending=True).cache()

# Define window for ranking data based on timestamp
window_spec = Window.orderBy(rating_data['timestamp'].asc())
# Compute percentile rank
ranked_data = rating_data.withColumn("percent_rank", F.percent_rank().over(window_spec))
ranked_data.show(20, False)

# Splitting data into different training sets and corresponding test sets
train_40 = ranked_data.filter(ranked_data["percent_rank"] < 0.4).cache()
test_40 = ranked_data.filter(ranked_data["percent_rank"] >= 0.4).cache()

train_60 = ranked_data.filter(ranked_data["percent_rank"] < 0.6).cache()
test_60 = ranked_data.filter(ranked_data["percent_rank"] >= 0.6).cache()

train_80 = ranked_data.filter(ranked_data["percent_rank"] < 0.8).cache()
test_80 = ranked_data.filter(ranked_data["percent_rank"] >= 0.8).cache()

# ALS model configuration using last 5 digits of Student No.
als_model = ALS(userCol="userId", itemCol="movieId", seed=24703, coldStartStrategy="drop")

# Evaluators for performance metrics
rmse_evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
mse_evaluator = RegressionEvaluator(metricName="mse", labelCol="rating", predictionCol="prediction")
mae_evaluator = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")

def evaluate_performance(training_data, testing_data, als_instance):
    # Fit the ALS model and evaluate performance
    model = als_instance.fit(training_data)
    predictions = model.transform(testing_data)
    rmse = rmse_evaluator.evaluate(predictions)
    mse = mse_evaluator.evaluate(predictions)
    mae = mae_evaluator.evaluate(predictions)
    
    return {
        "RMSE": rmse,
        "MSE": mse,
        "MAE": mae
    }

# Evaluate ALS model with different training sets
results_als1 = [evaluate_performance(train_40, test_40, als_model),
                evaluate_performance(train_60, test_60, als_model),
                evaluate_performance(train_80, test_80, als_model)]

# Enhancing ALS model by adjusting the rank
als_model_enhanced = als_model.setRank(15)
results_als2 = [evaluate_performance(train_40, test_40, als_model_enhanced),
                evaluate_performance(train_60, test_60, als_model_enhanced),
                evaluate_performance(train_80, test_80, als_model_enhanced)]

print('ALS1 model metrics:')
print(results_als1)
print('ALS2 model metrics:')
print(results_als2)

# Plot comparison of model performances
fig, ax = plt.subplots(figsize=(12, 8))
labels = ['40% Training', '60% Training', '80% Training']
metrics_labels = ['RMSE', 'MSE', 'MAE']
x1 = np.arange(len(labels))
wid = 0.2

# Plot metrics for both ALS configurations
colors_als1 = ['blue', 'green', 'red']  # Colors for ALS1 metrics
colors_als2 = ['cyan', 'lightgreen', 'salmon']  # Colors for ALS2 metrics
for i, metric_label in enumerate(metrics_labels):
    als1_metrics = [metric[metric_label] for metric in results_als1]
    als2_metrics = [metric[metric_label] for metric in results_als2]
    rects1 = ax.bar(x1 - wid/2 + i * wid/3, als1_metrics, wid/3, label=f'ALS1 {metric_label}', color=colors_als1[i])
    rects2 = ax.bar(x1 + wid/2 + i * wid/3, als2_metrics, wid/3, label=f'ALS2 {metric_label}', color=colors_als2[i], alpha=0.7)
    
    # Annotate the bars for ALS1
    for rect in rects1:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 4)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    # Annotate the bars for ALS2
    for rect in rects2:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 4)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

ax.set_xlabel('Training Data Split')
ax.set_ylabel('Metric Values')
ax.set_title('Performance of ALS Settings Across Different Splits')
ax.set_xticks(x1)
ax.set_xticklabels(labels)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.savefig("../Output/Q4_A.png")
plt.show()
    

#Part B

model_40 = als_model_enhanced.fit(train_40)
predictions_40 = model_40.transform(test_40)
model_60 = als_model_enhanced.fit(train_60)
predictions_60 = model_60.transform(test_60)
model_80 = als_model_enhanced.fit(train_80)
predictions_80 = model_80.transform(test_80)

# Function to extract user features from ALS model setting 2
def get_user_features(model):
    user_factors = model.userFactors.collect()
    df = spark.createDataFrame(user_factors, ["userid", "features"])
    features_rdd = df.rdd.map(lambda row: (row[0], Vectors.dense(row[1])))
    new_df = spark.createDataFrame(features_rdd, schema=['userid', 'features'])
    return new_df

# Retrieve user features vectors for each ALS model
dfFeatureVec1 = get_user_features(model_40)
dfFeatureVec2 = get_user_features(model_60)
dfFeatureVec3 = get_user_features(model_80)

movie_data = spark.read.load('../Data/ml-20m/movies.csv', format = 'csv', inferSchema = "true", header = "true").cache()

# Initialize the KMeans model using last 5 digits of Student No.
kmeans = KMeans(k=25, seed=24073)  # using a specific seed

# Apply KMeans to the user feature vectors
kmeans_model1 = kmeans.fit(dfFeatureVec1.select('features'))
transformed1 = kmeans_model1.transform(dfFeatureVec1.select('features'))
joined_data1 = dfFeatureVec1.join(transformed1, ["features"], "leftouter").cache()

# Summarize cluster sizes and identify the largest cluster
cluster_sizes1 = kmeans_model1.summary.clusterSizes
sorted_sizes1 = sorted(cluster_sizes1, reverse=True)
largest_cluster_index1 = cluster_sizes1.index(max(cluster_sizes1))
largest_cluster_data1 = joined_data1.filter(col("prediction") == largest_cluster_index1).cache()

# Repeat the process for other ALS models
kmeans_model2 = kmeans.fit(dfFeatureVec2.select('features'))
transformed2 = kmeans_model2.transform(dfFeatureVec2.select('features'))
joined_data2 = dfFeatureVec2.join(transformed2, ["features"], "leftouter").cache()

cluster_sizes2 = kmeans_model2.summary.clusterSizes
sorted_sizes2 = sorted(cluster_sizes2, reverse=True)
largest_cluster_index2 = cluster_sizes2.index(max(cluster_sizes2))
largest_cluster_data2 = joined_data2.filter(col("prediction") == largest_cluster_index2).cache()

kmeans_model3 = kmeans.fit(dfFeatureVec3.select('features'))
transformed3 = kmeans_model3.transform(dfFeatureVec3.select('features'))
joined_data3 = dfFeatureVec3.join(transformed3, ["features"], "leftouter").cache()

cluster_sizes3 = kmeans_model3.summary.clusterSizes
sorted_sizes3 = sorted(cluster_sizes3, reverse=True)
largest_cluster_index3 = cluster_sizes3.index(max(cluster_sizes3))
largest_cluster_data3 = joined_data3.filter(col("prediction") == largest_cluster_index3).cache()

# Data for plotting
splits = ['Split 1(40%)', 'Split 2(60%)', 'Split 3(80%)']
cluster_data = [sorted_sizes1[:5], sorted_sizes2[:5], sorted_sizes3[:5]]

# Plotting
fig, ax = plt.subplots()
colors = ['red', 'green', 'blue']
for i, (data, split) in enumerate(zip(cluster_data, splits)):
    # Plotting each split with a unique color
    ax.plot(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'], data, label=split, color=colors[i], marker='x')

    # Annotating each point on the plot
    for x, y in zip(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'], data):
        ax.annotate(f'{y}',  # This is the text to display
                    xy=(x, y),  # These are the coordinates to position the text
                    textcoords="offset points",  # How to position the text
                    xytext=(0,10),  # Distance from text to points (x,y)
                    ha='center')  # Horizontal alignment can be left, right or center

ax.set_ylabel('Number of Users')
ax.set_title('Top 5 Cluster Sizes Across Three Splits')
ax.legend()
plt.savefig("../Output/Q4_B.png")
plt.show()

# Function to analyze and display top genres for a training split
def analyze_top_genres(largest_cluster_data, train_data, movie_data):
    user_ids = [row['userid'] for row in largest_cluster_data.select('userid').distinct().collect()]
    ratings_filtered = train_data.filter(col('userId').isin(user_ids))
    movie_avg_ratings = ratings_filtered.groupBy('movieId').agg(avg('rating').alias('avg_rating'))
    top_movies = movie_avg_ratings.filter(col('avg_rating') >= 4).join(movie_data, "movieId")
    genre_data = top_movies.withColumn('genre', explode(sql_split('genres', '\|'))).groupBy('genre').count()
    return genre_data.orderBy(col('count').desc()).limit(10)

# Writing to .txt file
output_file_path = "../Output/Q4_Output.txt"
with open(output_file_path, "w") as file:
    header = f"{'Setting':<15}{'Split':<10}{'RMSE':<10}{'MSE':<10}{'MAE':<10}\n"
    file.write(header)
    for i, results in enumerate(results_als1):
        line = f"{'ALS Setting 1':<15}{40 + 20*i}%{'':<5}{results['RMSE']:<10.4f}{results['MSE']:<10.4f}{results['MAE']:<10.4f}\n"
        file.write(line)
    for i, results in enumerate(results_als2):
        line = f"{'ALS Setting 2':<15}{40 + 20*i}%{'':<5}{results['RMSE']:<10.4f}{results['MSE']:<10.4f}{results['MAE']:<10.4f}\n"
        file.write(line)
    file.write('\n')
    file.write('Top 5 Cluster Sizes Across Three Splits\n')
    # Define a format string with fixed column widths
    header_format = "{:<12}{:>12}{:>12}{:>12}{:>12}{:>12}\n"
    file.write(header_format.format("Split", "Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5"))

    # Writing data with the same format to align columns
    for index, sizes in enumerate(cluster_data, start=1):
        # Prepare a list starting with the split label followed by the sizes
        line_data = [f"Split {index}({40 + 20 * (index - 1)}%)"] + sizes
        formatted_line = header_format.format(*line_data)
        file.write(formatted_line)

    file.write('\nTop 10 Genres for Each Split\n')
    
    # Prepare to handle genres for each split
    for i, data in enumerate([largest_cluster_data1, largest_cluster_data2, largest_cluster_data3], start=1):
        # Get the corresponding training dataset
        train_data = locals()[f'train_{40 + 20 * (i - 1)}']
        genres_df = analyze_top_genres(data, train_data, movie_data)
        
        file.write(f'\nTop Genres for Split {i}:\n')
        header_format = "{:<20}{:>10}\n"  # 20 chars wide for genre, 10 chars for count
        file.write(header_format.format("Genre", "Count"))
        
        # Write each genre and its count using formatted strings
        for row in genres_df.collect():
            file.write(header_format.format(row['genre'], row['count']))

# Stop Spark session
spark.stop()
