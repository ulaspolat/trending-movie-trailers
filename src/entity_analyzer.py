import os
import json
from glob import glob
from datetime import datetime
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model Configuration
MODEL_NAME = "dslim/bert-base-NER"
MODEL_DIR = os.path.join(os.getcwd(), "saved_model_ner")

if os.path.exists(MODEL_DIR):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

model = model.to(device)

# Directory Configuration
COMMENTS_DIR = os.path.join(os.getcwd(), "trailer_data", "comments")
RESULTS_DIR = os.path.join(os.getcwd(), "trailer_data", "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def get_latest_analysis_results():
    """Get the most recent entity analysis results."""
    result_files = glob(os.path.join(RESULTS_DIR, "entity_analysis_*.json"))
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
        print(f"Warning: Could not parse date for comment: {comment}")
        return datetime.min

def analyze_entities_batch(comments, batch_size=16, max_length=512):
    """Process a batch of comments and extract entities."""
    valid_texts = [comment['text'] for comment in comments if comment and comment.get('text')]
    all_entities = []
    
    for i in tqdm(range(0, len(valid_texts), batch_size), desc="Analyzing entities"):
        batch_texts = valid_texts[i:i + batch_size]
        
        # Tokenize and get predictions
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, 
                         max_length=max_length, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
            
        # Convert predictions to entities
        for text, tokens, preds in zip(batch_texts, 
                                     inputs['input_ids'], 
                                     predictions):
            current_entity = []
            current_type = None
            
            # Convert token IDs back to text and align with predictions
            token_texts = tokenizer.convert_ids_to_tokens(tokens)
            
            for token, pred in zip(token_texts, preds):
                # Skip special tokens and padding
                if token in ['[PAD]', '[CLS]', '[SEP]', '<s>', '</s>', '<pad>']:
                    continue
                    
                if token.startswith("##"):
                    if current_entity:
                        current_entity.append(token[2:])
                    continue
                
                pred_label = model.config.id2label[pred.item()]
                
                if pred_label.startswith("B-"):
                    # Save previous entity if exists
                    if current_entity:
                        entity_text = ''.join(current_entity).lower().strip()
                        if entity_text and not any(special in entity_text for special in ['[pad]', '[cls]', '[sep]']):
                            all_entities.append({
                                'text': entity_text,
                                'type': current_type
                            })
                    # Start new entity
                    current_entity = [token]
                    current_type = pred_label[2:]
                elif pred_label.startswith("I-") and current_entity:
                    current_entity.append(token)
                else:
                    # Save previous entity if exists
                    if current_entity:
                        entity_text = ''.join(current_entity).lower().strip()
                        if entity_text and not any(special in entity_text for special in ['[pad]', '[cls]', '[sep]']):
                            all_entities.append({
                                'text': entity_text,
                                'type': current_type
                            })
                    current_entity = []
                    current_type = None
            
            # Save last entity if exists
            if current_entity:
                entity_text = ''.join(current_entity).lower().strip()
                if entity_text and not any(special in entity_text for special in ['[pad]', '[cls]', '[sep]']):
                    all_entities.append({
                        'text': entity_text,
                        'type': current_type
                    })
    
    return all_entities

def get_entity_stats(entities):
    """Calculate statistics for extracted entities."""
    if not entities:
        return None
    
    entity_counter = Counter()
    entity_types = {}
    type_counter = Counter()
    
    for entity in entities:
        entity_text = entity['text']
        entity_type = entity['type']
        
        # Skip any remaining special tokens
        if any(special in entity_text for special in ['[pad]', '[cls]', '[sep]', '<s>', '</s>', '<pad>']):
            continue
            
        entity_counter[entity_text] += 1
        entity_types[entity_text] = entity_type
        type_counter[entity_type] += 1
    
    total = sum(entity_counter.values())
    
    # Get most popular entities across all types
    most_popular_entities = [
        {
            'text': text,
            'type': entity_types[text],
            'count': count,
            'percentage': (count / total) * 100
        }
        for text, count in entity_counter.most_common(10)
    ]
    
    # Get top entities by type
    entities_by_type = {}
    for entity_type in type_counter:
        type_entities = [
            {'text': text, 'count': count}
            for text, count in entity_counter.items()
            if entity_types[text] == entity_type
        ]
        entities_by_type[entity_type] = sorted(
            type_entities,
            key=lambda x: x['count'],
            reverse=True
        )[:10]  # Top 10 entities per type
    
    return {
        'total_entities': total,
        'most_popular_entities': most_popular_entities,  # New section for most popular entities
        'entity_type_distribution': {
            etype: {
                'count': count,
                'percentage': (count / total) * 100
            }
            for etype, count in type_counter.items()
        },
        'entities_by_type': entities_by_type
    }

def merge_entity_stats(old_stats, new_stats):
    """Merge old and new entity statistics."""
    if not old_stats:
        return new_stats
    
    total = old_stats['total_entities'] + new_stats['total_entities']
    
    # Merge entity type distributions
    merged_distribution = {}
    all_types = set(old_stats['entity_type_distribution'].keys()) | \
                set(new_stats['entity_type_distribution'].keys())
    
    for etype in all_types:
        old_count = old_stats['entity_type_distribution'].get(etype, {'count': 0})['count']
        new_count = new_stats['entity_type_distribution'].get(etype, {'count': 0})['count']
        merged_count = old_count + new_count
        merged_distribution[etype] = {
            'count': merged_count,
            'percentage': (merged_count / total) * 100
        }
    
    # Merge entities by type
    merged_entities = {}
    for etype in all_types:
        old_entities = old_stats['entities_by_type'].get(etype, [])
        new_entities = new_stats['entities_by_type'].get(etype, [])
        
        # Combine and sum up entities
        entity_counts = Counter()
        entity_types = {}
        
        for e in old_entities + new_entities:
            entity_counts[e['text']] += e['count']
            if etype not in entity_types:
                entity_types[e['text']] = etype
        
        # Get top 10
        merged_entities[etype] = [
            {'text': text, 'count': count}
            for text, count in entity_counts.most_common(10)
        ]
    
    # Merge most popular entities
    all_entities = Counter()
    all_entity_types = {}
    
    # Combine old and new most popular entities
    for e in old_stats.get('most_popular_entities', []) + new_stats.get('most_popular_entities', []):
        all_entities[e['text']] += e['count']
        all_entity_types[e['text']] = e['type']
    
    most_popular = [
        {
            'text': text,
            'type': all_entity_types[text],
            'count': count,
            'percentage': (count / total) * 100
        }
        for text, count in all_entities.most_common(10)
    ]
    
    return {
        'total_entities': total,
        'most_popular_entities': most_popular,
        'entity_type_distribution': merged_distribution,
        'entities_by_type': merged_entities
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
        
        video_entities = {}

        for comment_file in comment_files:
            video_title = os.path.basename(comment_file).rsplit('_comments.json', 1)[0]
            print(f"\nProcessing comments for: {video_title}")

            with open(comment_file, 'r', encoding='utf-8') as f:
                current_comments = json.load(f)
            
            # If this is a new video or we don't have previous results, analyze all comments
            if not previous_results or video_title not in previous_results:
                print(f"New video detected: {video_title}")
                all_entities = analyze_entities_batch(current_comments)
                entity_stats = get_entity_stats(all_entities)
                
                video_entities[video_title] = {
                    'total_comments': len(current_comments),
                    'entity_stats': entity_stats,
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
                video_entities[video_title] = previous_results[video_title]
                continue
            
            print(f"Analyzing {len(new_comments)} new comments for: {video_title}")
            
            new_entities = analyze_entities_batch(new_comments)
            new_stats = get_entity_stats(new_entities)
            
            # Merge with previous results
            merged_stats = merge_entity_stats(
                previous_results[video_title]['entity_stats'],
                new_stats
            )
            
            video_entities[video_title] = {
                'total_comments': len(current_comments),
                'entity_stats': merged_stats,
                'last_analyzed': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'last_comment_date': max(get_comment_date(comment) for comment in current_comments).strftime("%Y-%m-%dT%H:%M:%SZ")
            }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(RESULTS_DIR, f"entity_analysis_{timestamp}.json")

        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(video_entities, f, ensure_ascii=False, indent=2)

        print(f"\nAnalysis completed! Results saved to: {output_filename}")
    except Exception as e:
        print(f"An error occurred during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 