!pip install bertopic nltk spacy textblob
!python -m spacy download en_core_web_sm

import os
import json
from glob import glob
import logging
from datetime import datetime
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import shutil
import traceback
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
import re
from google.colab import drive
import nltk
import spacy
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer

# Google Drive'ı bağla
drive.mount('/content/drive')

# NLTK verilerini indir
nltk.download('stopwords')
nltk.download('punkt')

# Ana dizinleri Google Drive'a göre ayarla
BASE_DIR = '/content/drive/MyDrive/trailer_data'
COMMENTS_DIR = os.path.join(BASE_DIR, "comments")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

# Model yolları
SENTENCE_MODEL_DIR = os.path.join(MODELS_DIR, "sentence_transformer")

# Klasörleri oluştur
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TopicSummarizer:
    def __init__(self, min_comments=5000):
        self.min_comments = min_comments
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nlp = spacy.load('en_core_web_sm')
        
        # Custom stopwords
        self.stop_words = set(stopwords.words('english'))
        custom_stops = {
            'netflix', 'watch', 'watching', 'show', 'video', 'comment', 'comments',
            'like', 'really', 'would', 'could', 'think', 'know', 'want', 'make',
            'just', 'well', 'even', 'actually', 'now', 'one', 'im', 'ive', 'get',
            'got', 'going', 'say', 'said', 'way', 'much', 'many', 'lot', 'thing',
            'things', 'trailer', 'youtube', 'channel', 'subscribe', 'video', 'videos'
        }
        self.stop_words.update(custom_stops)
        
        # Initialize models
        logger.info("Initializing models...")
        self.sentence_model = SentenceTransformer('all-mpnet-base-v2', device=self.device)
        
        # CountVectorizer setup
        self.vectorizer_model = CountVectorizer(
            stop_words=list(self.stop_words),
            min_df=5,
            max_df=0.7,
            ngram_range=(1, 2)
        )
        
        # Topic model setup
        self.topic_model = BERTopic(
            embedding_model=self.sentence_model,
            vectorizer_model=self.vectorizer_model,
            min_topic_size=50,
            nr_topics=5,
            verbose=True
        )
        
        logger.info("Models initialized successfully")
    
    def clean_text(self, text):
        """Basic text cleaning"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def get_sentiment(self, comments):
        """Get overall sentiment of comments"""
        sentiments = []
        for comment in comments[:20]:  # Analyze first 20 comments
            blob = TextBlob(comment)
            sentiments.append(blob.sentiment.polarity)
        
        avg_sentiment = np.mean(sentiments)
        if avg_sentiment < -0.1:
            return "Criticism"
        elif avg_sentiment > 0.1:
            return "Positive Discussion"
        return "Discussion"
    
    def create_topic_title(self, keywords, comments):
        """Create clear and informative topic title"""
        # Get top 3 most important keywords
        main_keywords = [word.title() for word in keywords[:3]]
        
        # Get sentiment
        sentiment_type = self.get_sentiment(comments)
        
        # Create title based on keywords and sentiment
        if len(main_keywords) >= 2:
            title = f"{sentiment_type} about {' and '.join(main_keywords)}"
        else:
            title = f"{sentiment_type} about {main_keywords[0]}"
            
        return title
    
    def process_video_comments(self, comments):
        try:
            if not isinstance(comments, list) or len(comments) < self.min_comments:
                logger.info(f"Skipping video with insufficient comments (minimum: {self.min_comments})")
                return None

            # Clean and prepare comments
            processed_comments = []
            original_comments = []
            
            for comment in comments:
                if isinstance(comment, dict) and comment.get('text'):
                    cleaned = self.clean_text(comment['text'])
                    if len(cleaned.split()) >= 3:
                        processed_comments.append(cleaned)
                        original_comments.append(comment['text'])

            if len(processed_comments) < self.min_comments:
                return None

            # Extract topics
            topics, _ = self.topic_model.fit_transform(processed_comments)
            
            # Get topic information
            topic_info = self.topic_model.get_topic_info()
            topic_info = topic_info[topic_info['Topic'] != -1].head(5)
            
            # Process each topic
            processed_topics = []
            for _, row in topic_info.iterrows():
                topic_id = row['Topic']
                topic_size = row['Count']
                
                # Get topic keywords
                topic_keywords = [term for term, _ in self.topic_model.get_topic(topic_id)]
                
                # Get topic comments
                topic_indices = [i for i, t in enumerate(topics) if t == topic_id]
                topic_comments = [original_comments[i] for i in topic_indices]
                
                # Create topic title
                topic_title = self.create_topic_title(topic_keywords, topic_comments)
                
                processed_topics.append({
                    'topic_title': topic_title,
                    'comment_count': topic_size,
                    'keywords': topic_keywords[:5],
                    'sample_comments': topic_comments[:5]
                })
            
            # Sort topics by comment count
            processed_topics.sort(key=lambda x: x['comment_count'], reverse=True)
            
            return {
                'total_comments': len(processed_comments),
                'topics': processed_topics
            }
            
        except Exception as e:
            logger.error(f"Error processing video comments: {str(e)}")
            logger.error(traceback.format_exc())
            return None

def main():
    try:
        logger.info("Starting topic analysis...")
        comments_path = max(glob(os.path.join(COMMENTS_DIR, "*")))
        
        summarizer = TopicSummarizer(min_comments=5000)
        comment_files = glob(os.path.join(comments_path, "*_comments.json"))
        
        results = {}
        for comment_file in tqdm(comment_files, desc="Processing videos"):
            try:
                video_title = os.path.basename(comment_file).replace("_comments.json", "")
                with open(comment_file, 'r', encoding='utf-8') as f:
                    comments = json.load(f)
                video_results = summarizer.process_video_comments(comments)
                if video_results:
                    results[video_title] = video_results
            except Exception as e:
                logger.error(f"Error processing {video_title}: {str(e)}")
                continue
        
        if results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(RESULTS_DIR, f"topic_analysis_{timestamp}.json")
            
            final_results = {
                'video_topics': results,
                'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_videos_analyzed': len(results)
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Results saved to: {output_path}")
            
    except Exception as e:
        logger.error(f"Main execution error: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
