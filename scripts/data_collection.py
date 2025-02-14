import pandas as pd
from datetime import datetime, timedelta
import json
import os
import googleapiclient.discovery

# YouTube API configuration
API_KEY = 'your-api-key'
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# Target studios
ALLOWED_STUDIOS = [
    "Warner Bros.", "DC", "Marvel Entertainment", "Netflix", "Sony Pictures Entertainment",
    "20th Century Studios", "Universal Pictures", "Paramount Pictures", "Disney", "Pixar",
    "Lionsgate Movies", "Legendary", "New Line Cinema"
]

# Target keywords
SEARCH_KEYWORDS = [
    "official trailer", "teaser trailer", "official teaser", "movie trailer", "new trailer", "film trailer",
    "teaser clip", "upcoming movie", "official clip"
]

# Create a YouTube API client
def youtube_client():
    if not API_KEY:
        raise ValueError("YouTube API key not found in environment variables")
    return googleapiclient.discovery.build(
        YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY
    )

# Filter search results
def filter_videos(videos, allowed_studios, topn):
    """
    Filters videos based on studio, ensures only unique trailers are selected,
    and prioritizes the most viewed video if duplicates are found.
    """
    filtered_videos = []
    unique_movies = {}

    for video in videos:
        # Check if the uploader is in the allowed studios list
        channel_title = video.get("snippet", {}).get("channelTitle", "").strip()
        if channel_title not in allowed_studios:
            continue

        # Check for duplicate trailers, prioritize by view count
        title = video.get("snippet", {}).get("title", "").lower()
        view_count = int(video.get("statistics", {}).get("viewCount", 0))

        # Use the movie title as the unique key (split by "trailer")
        movie_key = title.split("trailer")[0].strip()
        if movie_key not in unique_movies or view_count > unique_movies[movie_key]["view_count"]:
            unique_movies[movie_key] = {
                "video_id": video.get("id"),
                "title": title,
                "published_at": video.get("snippet", {}).get("publishedAt", ""),
                "view_count": view_count,
                "like_count": int(video.get("statistics", {}).get("likeCount", 0)),
                "comment_count": int(video.get("statistics", {}).get("commentCount", 0)),
                "video_url": f"https://www.youtube.com/watch?v={video.get('id')}",
                "channel_title": channel_title,
            }

    # Sort by view count and return top N videos
    sorted_videos = sorted(unique_movies.values(), key=lambda x: x["view_count"], reverse=True)
    return sorted_videos[:topn]

# Fetch and process trailers
def fetch_movie_trailers(period_in_days, topn=50):
    """
    Fetches trailers published within the last `period_in_days` and filters them.
    """
    youtube = youtube_client()
    all_videos = []

    # Calculate the starting date
    start_date = (datetime.utcnow() - timedelta(days=period_in_days)).isoformat() + "Z"
    next_page_token = None

    for keyword in SEARCH_KEYWORDS:
        while True:
            # Search for videos with the given keyword and date filter
            request = youtube.search().list(
                q=keyword,
                part="snippet",
                type="video",
                maxResults=50,  # Reduced to avoid excessive quota usage per request
                publishedAfter=start_date,
                order="viewCount",
                pageToken=next_page_token
            )
            response = request.execute()

            # Get video IDs from search results
            video_ids = [item["id"]["videoId"] for item in response.get("items", [])]
            if not video_ids:
                break

            # Fetch video details (statistics and snippet)
            video_request = youtube.videos().list(
                part="snippet,statistics",
                id=",".join(video_ids)
            )
            video_response = video_request.execute()

            # Add video details to the list
            all_videos.extend(video_response.get("items", []))

            # Check for next page token
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

    # Filter videos based on allowed studios and other criteria
    filtered_videos = filter_videos(all_videos, ALLOWED_STUDIOS, topn)
    return filtered_videos

# Main function to run the script
if __name__ == "__main__":
    # Create base data directory if it doesn't exist
    base_data_dir = os.path.join("trailer_data", "data")
    
    # Clean up existing directories
    if os.path.exists(base_data_dir):
        for item in os.listdir(base_data_dir):
            item_path = os.path.join(base_data_dir, item)
            if os.path.isdir(item_path):
                for file in os.listdir(item_path):
                    os.remove(os.path.join(item_path, file))
                os.rmdir(item_path)
    
    # Create the base directory
    os.makedirs(base_data_dir, exist_ok=True)
    
    # Create timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_data_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    # Fetch trailers for the last 90 days only
    periods = {
        "last_90_days": 90
    }
    
    for period_name, days in periods.items():
        print(f"Fetching trailers for the {period_name}...")
        trailers = fetch_movie_trailers(days, topn=100)

        # Save the results as a JSON file in the timestamped directory
        output_file = os.path.join(run_dir, f"filtered_movie_trailers_{period_name}.json")
        with open(output_file, "w") as f:
            json.dump(trailers, f, indent=4)

        print(f"Saved {len(trailers)} trailers to {output_file}.")
