import os
import json
from datetime import datetime
from dotenv import load_dotenv
from google.cloud import storage
import re
from collections import defaultdict
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType
from pyspark.sql.functions import col, when
from src.youtube_api import youtube_api

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("YouTube Comments Updater") \
    .config("spark.sql.parquet.compression.codec", "snappy") \
    .getOrCreate()

# Define schema for comments
comment_schema = StructType([
    StructField("video_id", StringType(), True),
    StructField("video_title", StringType(), True),
    StructField("author", StringType(), True),
    StructField("text", StringType(), True),
    StructField("like_count", IntegerType(), True),
    StructField("published_at", TimestampType(), True)
])

# Load environment variables
load_dotenv()

# Configuration
GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME')
GCS_DATA_PATH = os.getenv('GCS_DATA_PATH')
GCS_COMMENTS_PATH = os.getenv('GCS_COMMENTS_PATH')

# Initialize GCS client
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET_NAME)

def is_meaningful_comment(text):
    """Check if a comment contains meaningful content."""
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    cleaned_text = ' '.join(cleaned_text.split())
    
    if not cleaned_text:
        return False
    
    if re.match(r'^\d+$', cleaned_text.replace(' ', '')):
        return False
    
    words = [w for w in cleaned_text.split() if len(w) > 2]
    if len(words) < 3:
        return False
    
    meaningful_words = [w for w in words if len(w) > 2 and not w.isdigit()]
    if not meaningful_words:
        return False
    
    return True

def fetch_new_comments(video_id, video_title, after_timestamp=None):
    """Fetch new comments for a video after the specified timestamp."""
    comments = []

    try:
        request = youtube_api.get_client().commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText",
            order="time"
        )

        while request:
            response = youtube_api.execute_with_fallback(request)
            
            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]
                comment_time = comment.get("publishedAt")
                
                # If we have a timestamp filter and this comment is older, stop
                if after_timestamp and comment_time <= after_timestamp:
                    return comments
                
                text = comment.get("textDisplay", "")
                if is_meaningful_comment(text):
                    comments.append({
                        "video_id": video_id,
                        "video_title": video_title,
                        "author": comment.get("authorDisplayName"),
                        "text": text,
                        "like_count": comment.get("likeCount", 0),
                        "published_at": comment_time
                    })

            request = youtube_api.get_client().commentThreads().list_next(request, response) if "nextPageToken" in response else None

    except Exception as e:
        print(f"Error fetching comments for video {video_id}: {str(e)}")
    
    return comments

def save_comments_to_parquet(comments_df):
    """Save comments DataFrame to Parquet format in GCS."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"gs://{GCS_BUCKET_NAME}/{GCS_COMMENTS_PATH}/{timestamp}/comments.parquet"
    
    # Save as Parquet
    comments_df.write.mode("overwrite") \
        .partitionBy("video_id") \
        .parquet(output_path)
    
    print(f"Saved {comments_df.count()} comments to {output_path}")
    return output_path

def get_latest_comments_path():
    """Find the latest comments.parquet directory in GCS."""
    prefix = f"{GCS_COMMENTS_PATH}/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    
    # Filter for parquet directories
    parquet_dirs = set()
    for blob in blobs:
        if "comments.parquet" in blob.name:
            dir_path = os.path.dirname(blob.name)
            parquet_dirs.add(dir_path)
    
    if not parquet_dirs:
        return None
    
    # Get the latest directory
    latest_dir = sorted(parquet_dirs, reverse=True)[0]
    return f"gs://{GCS_BUCKET_NAME}/{latest_dir}/comments.parquet"

def main():
    # Get the latest comments parquet file
    latest_comments_path = get_latest_comments_path()
    if not latest_comments_path:
        print("No existing comments found. Please run get_comments.py first.")
        exit(1)
    
    print(f"Reading existing comments from: {latest_comments_path}")
    
    # Read existing comments
    existing_comments_df = spark.read.parquet(latest_comments_path)
    print(f"Loaded {existing_comments_df.count()} existing comments")
    
    # Get unique videos and their latest comment timestamps
    videos_df = existing_comments_df.groupBy("video_id", "video_title") \
        .agg({"published_at": "max"}) \
        .withColumnRenamed("max(published_at)", "latest_comment")
    
    # Fetch new comments for each video
    all_new_comments = []
    
    for row in videos_df.collect():
        video_id = row.video_id
        video_title = row.video_title
        latest_timestamp = row.latest_comment
        
        print(f"\nFetching new comments for: {video_title}")
        new_comments = fetch_new_comments(video_id, video_title, latest_timestamp)
        all_new_comments.extend(new_comments)
        print(f"Fetched {len(new_comments)} new comments")
    
    if all_new_comments:
        # Convert new comments to DataFrame
        new_comments_df = spark.createDataFrame(all_new_comments, schema=comment_schema)
        
        # Combine existing and new comments
        combined_df = existing_comments_df.union(new_comments_df)
        
        # Remove duplicates and clean
        final_df = combined_df.dropDuplicates(['video_id', 'text']) \
            .filter(col('text').isNotNull()) \
            .cache()
        
        # Save updated comments
        output_path = save_comments_to_parquet(final_df)
        print(f"\nAll comments saved to: {output_path}")
        
        # Show statistics
        print("\nDataset Statistics:")
        print(f"Total comments: {final_df.count()}")
        print(f"New comments added: {new_comments_df.count()}")
        print("\nComments per video:")
        final_df.groupBy('video_title').count().show(truncate=False)
    else:
        print("\nNo new comments found.")
    
    # Clean up Spark session
    spark.stop()

if __name__ == "__main__":
    main()
