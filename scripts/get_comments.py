import json
import os
from datetime import datetime
import shutil
import googleapiclient.discovery
from langdetect import detect, LangDetectException
import emoji
import logging

# Set up logging
logging.basicConfig(
    filename=os.path.join('trailer_data', 'comments_stats.log'),
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# YouTube API configuration
API_KEY = 'AIzaSyD1crl0SKE08IeH_D4of5IKjZsQaGMtt1o'
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

def youtube_client():
    if not API_KEY:
        raise ValueError("YouTube API key not found in environment variables")
    return googleapiclient.discovery.build(
        YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY
    )

def is_english_or_emoji(text):
    """Check if text is in English or contains emojis."""
    # Check for emojis
    if emoji.emoji_count(text) > 0:
        return True
    
    # Check language
    try:
        detected_lang = detect(text)
        return detected_lang == 'en'
    except LangDetectException:
        return False

def get_latest_directory(base_dir):
    """Find the latest timestamp directory in the given base directory."""
    if not os.path.exists(base_dir):
        return None
    
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not dirs:
        return None
    
    # Sort directories by name (timestamp) in descending order
    latest_dir = sorted(dirs, reverse=True)[0]
    return os.path.join(base_dir, latest_dir)

def get_existing_comments(comments_file):
    """Load existing comments from a file and return the latest comment timestamp."""
    if not os.path.exists(comments_file):
        return [], None
    
    with open(comments_file, 'r') as f:
        comments = json.load(f)
        
    if not comments:
        return [], None
    
    # Get the latest comment timestamp
    latest_timestamp = max(comment.get('published_at', '') for comment in comments)
    return comments, latest_timestamp

def fetch_comments(video_id, max_comments=200000, after_timestamp=None):
    """Fetch comments for a video, optionally after a specific timestamp."""
    youtube = youtube_client()
    comments = []
    total_comments = 0
    english_comments = 0

    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText",
            order="time"  # Get newest comments first
        )

        while request and len(comments) < max_comments:
            response = request.execute()
            
            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]
                comment_time = comment.get("publishedAt")
                comment_text = comment.get("textDisplay", "")
                total_comments += 1
                
                # If we have a timestamp filter and this comment is older, stop
                if after_timestamp and comment_time <= after_timestamp:
                    continue
                
                # Filter for English or emoji comments
                if is_english_or_emoji(comment_text):
                    english_comments += 1
                    comments.append({
                        "author": comment.get("authorDisplayName"),
                        "text": comment_text,
                        "like_count": comment.get("likeCount"),
                        "published_at": comment_time,
                    })

            if len(comments) >= max_comments:
                break

            # Check for pagination
            if "nextPageToken" in response:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100,
                    textFormat="plainText",
                    pageToken=response["nextPageToken"],
                    order="time"
                )
            else:
                break

        # Log statistics
        logging.info(f"Video {video_id}: Total comments: {total_comments}, English/Emoji comments: {english_comments}")
        return comments

    except googleapiclient.errors.HttpError as e:
        if "commentsDisabled" in str(e):
            logging.warning(f"Comments are disabled for video ID {video_id}")
            print(f"Comments are disabled for video ID {video_id}")
            return None
        elif "videoNotFound" in str(e):
            logging.warning(f"Video not found: {video_id}")
            print(f"Video not found: {video_id}")
            return None
        else:
            logging.error(f"Error fetching comments for video ID {video_id}: {e}")
            print(f"Error fetching comments for video ID {video_id}: {e}")
            return None
    
    return comments

def process_video_comments(video_data, output_dir, latest_comments_dir=None):
    """Process comments for a single video, handling both new and existing comments."""
    video_id = video_data.get("video_id")
    video_title = video_data.get("title")
    
    if not video_id or not video_title:
        return
    
    print(f"Processing comments for video: {video_title} (ID: {video_id})")
    logging.info(f"Starting to process comments for video: {video_title} (ID: {video_id})")
    
    # Create sanitized filename
    sanitized_title = "".join(c for c in video_title if c.isalnum() or c.isspace()).replace(" ", "_")
    output_filename = f"{sanitized_title}_{video_id}_comments.json"
    output_path = os.path.join(output_dir, output_filename)
    
    # Check for existing comments if we have a latest directory
    existing_comments = []
    latest_timestamp = None
    
    if latest_comments_dir:
        existing_file = os.path.join(latest_comments_dir, output_filename)
        if os.path.exists(existing_file):
            existing_comments, latest_timestamp = get_existing_comments(existing_file)
    
    # Fetch new comments
    new_comments = fetch_comments(video_id, after_timestamp=latest_timestamp)
    
    # Skip if comments are disabled or other error occurred
    if new_comments is None:
        return
    
    # Combine existing and new comments
    all_comments = existing_comments + new_comments
    
    # Save combined comments
    with open(output_path, 'w') as f:
        json.dump(all_comments, f, indent=4)
    
    # Log statistics
    logging.info(f"Video {video_title}: Total saved comments: {len(all_comments)} (New: {len(new_comments)}, Existing: {len(existing_comments)})")
    print(f"Saved {len(all_comments)} comments ({len(new_comments)} new) to {output_path}")

def process_comments(data_dir):
    """Process comments for all videos in the data directory."""
    # Set up directories
    comments_base_dir = os.path.join("trailer_data", "comments")
    os.makedirs(comments_base_dir, exist_ok=True)
    
    # Get latest comments directory
    latest_comments_dir = get_latest_directory(comments_base_dir)
    
    # Create new directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(comments_base_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each JSON file in the data directory
    for json_file in os.listdir(data_dir):
        if not json_file.endswith(".json"):
            continue
            
        json_path = os.path.join(data_dir, json_file)
        try:
            with open(json_path, 'r') as f:
                videos = json.load(f)
            
            for video in videos:
                process_video_comments(video, output_dir, latest_comments_dir)
                
        except json.JSONDecodeError:
            print(f"Error decoding JSON file: {json_path}")
            continue
    
    # Clean up old directories
    print("Cleaning up old comment directories...")
    for dir_name in os.listdir(comments_base_dir):
        dir_path = os.path.join(comments_base_dir, dir_name)
        if os.path.isdir(dir_path) and dir_path != output_dir:
            try:
                shutil.rmtree(dir_path)
                print(f"Deleted old directory: {dir_path}")
            except Exception as e:
                print(f"Error deleting directory {dir_path}: {e}")
    
    return output_dir

if __name__ == "__main__":
    # Get the latest data directory from trailer_data/data
    data_base_dir = os.path.join("trailer_data", "data")
    latest_data_dir = get_latest_directory(data_base_dir)
    
    if not latest_data_dir:
        print("No data directory found. Please run data_collection.py first.")
        exit(1)
    
    print(f"Processing data from: {latest_data_dir}")
    output_dir = process_comments(latest_data_dir)
    print(f"All comments saved in: {output_dir}")
