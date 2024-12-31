
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,regexp_extract, desc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Initialize SparkSession
spark = SparkSession.builder.appName("NASA Log Analysis").getOrCreate()

# Path to your uploaded data file
log_file_path = "../Data/NASA_access_log_Jul95.gz"

# Read the compressed log file
log_df = spark.read.text(log_file_path)

log_regex = r'^(\S+).*\[(\d{2})/(\w{3})/(\d{4}):(\d{2}):(\d{2}):(\d{2}) -\d{4}\]'

# Extracting information
log_df = log_df.withColumn('host', regexp_extract('value', log_regex, 1))\
               .withColumn('day', regexp_extract('value', log_regex, 2).cast('int'))\
               .withColumn('month', regexp_extract('value', log_regex, 3))\
               .withColumn('year', regexp_extract('value', log_regex, 4).cast('int'))\
               .withColumn('hour', regexp_extract('value', log_regex, 5).cast('int'))

# Assuming the log format and that you'll parse it to extract the host,
# here's an example of filtering requests by country domain (actual parsing depends on log format)
# Example to filter by German (.de) domain
de_requests = log_df.filter(col('host').endswith(".de")).count()
ca_requests = log_df.filter(col('host').endswith(".ca")).count()
sg_requests = log_df.filter(col('host').endswith(".sg")).count()

print(f"Total requests from Germany (.de): {de_requests}")
print(f"Total requests from Canada (.ca): {ca_requests}")
print(f"Total requests from Singapore (.sg): {sg_requests}")

output_file_path = "../Output/Q1_output.txt"

counts = {
    "Germany (.de)": de_requests,
    "Canada (.ca)": ca_requests,
    "Singapore (.sg)": sg_requests
}


with open(output_file_path, "w") as f:
    for country, count in counts.items():
        f.write(f"{country}: {count}\n")


# Open the output file in write mode
with open(output_file_path, "a") as output_file:
    # Define the country codes
    countries = ['de', 'ca', 'sg']
    for country in countries:
        filtered_df = log_df.filter(col('host').endswith(f".{country}"))
        
        # Count unique hosts
        unique_hosts_count = filtered_df.select('host').distinct().count()
        
        # Find the top 9 most frequent hosts
        top_hosts = filtered_df.groupBy('host').count().orderBy(desc('count')).limit(9).collect()
        
        # Write the results for the country to the output file
        output_file.write(f"Country: {country.upper()}, Unique Hosts: {unique_hosts_count}\n")
        output_file.write("Top 9 Hosts:\n")
        
        for row in top_hosts:
            output_file.write(f"{row['host']}, Requests: {row['count']}\n")
        
        # Add a separator line between countries for readability
        output_file.write("-" * 40 + "\n")



countries = ['Germany', 'Canada', 'Singapore']
requests = [de_requests, ca_requests, sg_requests]

# Creating the bar chart
plt.figure(figsize=(10, 6))
plt.bar(countries, requests, color=['black', 'red', 'green'])
plt.title('Total Number of Requests by Country')
for i in range(3):
    plt.text(countries[i],requests[i],requests[i],ha = 'center',va = 'bottom')
plt.xlabel('Country')
plt.ylabel('Number of Requests')
plt.show()

#Storing the plot
plt.savefig("../Output/Q1_A.png")


total_requests = {
    "de": de_requests,
    "ca": ca_requests,
    "sg": sg_requests
}

countries = {
    "de":"Germany",
    "ca": "Canada",
    "sg":"Singapore"
}


k = 1
plt.figure(figsize=(15,20))
for country in countries:
    # Calculate total requests handled by the top 9 hosts
    filtered_df = log_df.filter(col('host').endswith(f".{country}"))
        
    # Count unique hosts
    unique_hosts_count = filtered_df.select('host').distinct().count()
    
    # Find the top 9 most frequent hosts
    top_hosts = filtered_df.groupBy('host').count().orderBy(desc('count')).limit(9).collect()
    top_hosts_total = sum([row['count'] for row in top_hosts])
    
    # Calculate the rest
    rest = total_requests[country] - top_hosts_total
    
    h = []
    for i in range(1,10):
        h.append(f'host{i}')
    
    # Prepare data for plotting
    labels = h + ['Rest']
    sizes = [row['count'] for row in top_hosts] + [rest]
    percentages = [size / total_requests[country] * 100 for size in sizes]
    
    # Plot
    #plt.figure(figsize=(10, 8))
    #plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    #plt.title(f'Request Distribution for {country.upper()}')
    #plt.show()
    x_positions = np.arange(len(labels))
    
    # Create the bar chart
    plt.subplot(3,1,k)
    k += 1
    plt.bar(x_positions, percentages, color='blue')
    for i in range(10):
        plt.text(list(x_positions)[i], percentages[i],f'{percentages[i]:0.2f}%',ha = 'center',va = 'bottom')
    
    # Add title and labels with the country's full name
    plt.title(f'Percentage of Requests by Host for {countries[country]}')
    plt.xlabel('Hosts')
    plt.ylabel('Percentage of Total Requests')
    
    # Replace the x-axis labels with the host names, rotating them for better readability
    plt.xticks(x_positions, labels, rotation=45, ha='right')

plt.show()
#Storing the plot
plt.savefig("../Output/Q1_C.png")

k = 1
plt.figure(figsize=(15,20))
for country in countries:
    # Calculate total requests handled by the top 9 hosts
    filtered_df = log_df.filter(col('host').endswith(f".{country}"))
    
    # Find the most frequent hosts
    top_hosts = filtered_df.groupBy('host').count().orderBy(desc('count')).limit(1).collect()
    df = log_df.filter(col('host') == top_hosts[0]['host'])
    visits_by_day_hour = df.groupBy('day', 'hour').count()
    visits_pd = visits_by_day_hour.toPandas()
    pivoted_df = visits_pd.pivot(index='hour', columns='day', values='count').fillna(0)
    pivoted_df = pivoted_df.sort_index(ascending=False)
    plt.subplot(3,1,k)
    k += 1
    sns.heatmap(pivoted_df, annot=True, fmt=".0f", cmap="YlGnBu", cbar=True, linewidths=.5)
    plt.title(f'Heatmap of Visits by Day and Hour for {countries[country]}', fontsize=20)
    plt.xlabel('Day of Month', fontsize=16)
    plt.ylabel('Hour of Day', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

plt.show()
#Storing the plot
plt.savefig("../Output/Q1_D.png")
spark.stop()
