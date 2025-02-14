import os
import json
from glob import glob
from datetime import datetime
from collections import Counter
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model Configuration
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MODEL_DIR = os.path.join(os.getcwd(), "saved_model")

if os.path.exists(MODEL_DIR):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

model = model.to(device)

# Directory Configuration
COMMENTS_DIR = os.path.join(os.getcwd(), "trailer_data", "comments")
RESULTS_DIR = os.path.join(os.getcwd(), "trailer_data", "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

def get_latest_analysis_results():
    """Get the most recent sentiment analysis results."""
    result_files = glob(os.path.join(RESULTS_DIR, "sentiment_analysis_*.json"))
    if not result_files:
        return None, None
    
    latest_file = max(result_files)
    with open(latest_file, 'r', encoding='utf-8') as f:
        latest_results = json.load(f)
    
    # Extract timestamp from filename
    timestamp = latest_file.split('_')[-1].replace('.json', '')
    return latest_results, timestamp

def get_comment_date(comment):
    """Extract datetime from comment published_at field."""
    try:
        return datetime.strptime(comment['published_at'], "%Y-%m-%dT%H:%M:%SZ")
    except (KeyError, ValueError):
        # Print the problematic comment for debugging
        print(f"Warning: Could not parse date for comment: {comment}")
        return datetime.min

def analyze_comments_batch(comments, batch_size=16, max_length=512):
    valid_texts = [comment['text'] for comment in comments if comment and comment.get('text')]
    results = []
    
    for i in tqdm(range(0, len(valid_texts), batch_size), desc="Analyzing comments"):
        batch_texts = valid_texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(scores, dim=1).cpu().numpy()
            confidences = scores.max(dim=1).values.cpu().numpy()

        for text, pred, conf in zip(batch_texts, predictions, confidences):
            results.append({
                'text': text,
                'sentiment': sentiment_map[pred],
                'confidence': float(conf)
            })
    
    return results

def merge_sentiment_stats(old_stats, new_stats):
    """Merge old and new sentiment statistics."""
    if not old_stats:
        return new_stats
    
    total = old_stats['total_analyzed'] + new_stats['total_analyzed']
    merged_distribution = {}
    
    # Merge sentiment distributions
    for sentiment in set(old_stats['sentiment_distribution'].keys()) | set(new_stats['sentiment_distribution'].keys()):
        old_count = old_stats['sentiment_distribution'].get(sentiment, {'count': 0})['count']
        new_count = new_stats['sentiment_distribution'].get(sentiment, {'count': 0})['count']
        merged_count = old_count + new_count
        merged_distribution[sentiment] = {
            'count': merged_count,
            'percentage': (merged_count / total) * 100
        }
    
    # Calculate new average confidence
    old_total_conf = old_stats['average_confidence'] * old_stats['total_analyzed']
    new_total_conf = new_stats['average_confidence'] * new_stats['total_analyzed']
    avg_confidence = (old_total_conf + new_total_conf) / total
    
    return {
        'total_analyzed': total,
        'sentiment_distribution': merged_distribution,
        'average_confidence': avg_confidence
    }

def get_sentiment_stats(sentiments):
    if not sentiments:
        return None
    
    sentiment_counts = Counter(item['sentiment'] for item in sentiments)
    total = len(sentiments)
    
    return {
        'total_analyzed': total,
        'sentiment_distribution': {
            sentiment: {
                'count': count,
                'percentage': (count / total) * 100
            }
            for sentiment, count in sentiment_counts.items()
        },
        'average_confidence': sum(item['confidence'] for item in sentiments) / total
    }

def main():
    try:
        # Get previous analysis results
        previous_results, previous_timestamp = get_latest_analysis_results()
        
        # Find the most recent timestamp folder
        timestamp_folders = glob(os.path.join(COMMENTS_DIR, "*"))
        if not timestamp_folders:
            raise ValueError(f"No timestamp folders found in {COMMENTS_DIR}")
        
        latest_folder = max(timestamp_folders)
        comment_files = glob(os.path.join(latest_folder, "*.json"))
        
        if not comment_files:
            raise ValueError(f"No comment files found in {latest_folder}")
        
        video_sentiments = {}

        for comment_file in comment_files:
            video_title = os.path.basename(comment_file).rsplit('_comments.json', 1)[0]
            print(f"\nProcessing comments for: {video_title}")

            with open(comment_file, 'r', encoding='utf-8') as f:
                current_comments = json.load(f)
            
            # If this is a new video or we don't have previous results, analyze all comments
            if not previous_results or video_title not in previous_results:
                print(f"New video detected: {video_title}")
                all_sentiments = analyze_comments_batch(current_comments)
                sentiment_stats = get_sentiment_stats(all_sentiments)
                
                video_sentiments[video_title] = {
                    'total_comments': len(current_comments),
                    'sentiment_stats': sentiment_stats,
                    'last_analyzed': datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'last_comment_date': max(get_comment_date(comment) for comment in current_comments).strftime("%Y-%m-%dT%H:%M:%SZ")
                }
                continue
            
            # For existing videos, find new comments using date comparison
            last_analyzed_date = datetime.strptime(
                previous_results[video_title].get('last_comment_date', "1970-01-01T00:00:00Z"),
                "%Y-%m-%dT%H:%M:%SZ"
            )
            
            # Filter new comments based on publishedAt date
            new_comments = [
                comment for comment in current_comments 
                if get_comment_date(comment) > last_analyzed_date
            ]
            
            if not new_comments:
                print(f"No new comments for: {video_title}")
                video_sentiments[video_title] = previous_results[video_title]
                continue
            
            print(f"Analyzing {len(new_comments)} new comments for: {video_title}")
            
            new_sentiments = analyze_comments_batch(new_comments)
            new_stats = get_sentiment_stats(new_sentiments)
            
            # Merge with previous results
            merged_stats = merge_sentiment_stats(
                previous_results[video_title]['sentiment_stats'],
                new_stats
            )
            
            video_sentiments[video_title] = {
                'total_comments': len(current_comments),
                'sentiment_stats': merged_stats,
                'last_analyzed': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'last_comment_date': max(get_comment_date(comment) for comment in current_comments).strftime("%Y-%m-%dT%H:%M:%SZ")
            }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(RESULTS_DIR, f"sentiment_analysis_{timestamp}.json")

        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(video_sentiments, f, ensure_ascii=False, indent=2)

        print(f"\nAnalysis completed! Results saved to: {output_filename}")
    except Exception as e:
        print(f"An error occurred during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
