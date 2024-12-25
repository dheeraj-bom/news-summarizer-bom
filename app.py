# app.py
from flask import Flask, request, jsonify, render_template, redirect, url_for
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from heapq import nlargest
from collections import defaultdict
import requests

app = Flask(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class NewsSummarizer:
    def __init__(self):
        self.stopwords = set(stopwords.words('english') + list(punctuation))
        
    def get_word_frequency(self, text):
        word_freq = defaultdict(int)
        for word in word_tokenize(text.lower()):
            if word not in self.stopwords:
                word_freq[word] += 1
        return word_freq
    
    def score_sentences(self, sentences, word_freq):
        sentence_scores = defaultdict(int)
        for sentence in sentences:
            for word in word_tokenize(sentence.lower()):
                if word in word_freq:
                    sentence_scores[sentence] += word_freq[word]
        return sentence_scores
    
    def summarize_text(self, text, num_sentences=10):
        if not text:
            return ""
        sentences = sent_tokenize(text)
        word_freq = self.get_word_frequency(text)
        sentence_scores = self.score_sentences(sentences, word_freq)
        summary_sentences = nlargest(min(num_sentences, len(sentences)), 
                                   sentence_scores, key=sentence_scores.get)
        summary_sentences.sort(key=sentences.index)
        return ' '.join(summary_sentences)

summarizer = NewsSummarizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize_news():
    try:
        text = request.form.get('text', '')
        if not text:
            return render_template('index.html', error="Please provide text to summarize")
        
        # Calculate target summary length (aim for 300-400 words)
        target_sentences = max(3, len(sent_tokenize(text)) // 4)
        summary = summarizer.summarize_text(text, num_sentences=target_sentences)
        
        # Count words
        summary_word_count = len(word_tokenize(summary))
        original_word_count = len(word_tokenize(text))
        
        return render_template('summary.html',
                             original_text=text,
                             summary=summary,
                             summary_word_count=summary_word_count,
                             original_word_count=original_word_count)
    
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)