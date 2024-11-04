import os
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import pandas as pd

# Ensure you have the necessary NLTK data files for tokenization
nltk.download('punkt_tab')


def load_datasets(directory):
    """Load all CSV files in the directory, returning a list of dataframes."""
    dataframes = []
    for f in os.listdir(directory):
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(directory, f))
            dataframes.append((f, df))
    return dataframes

def tokenize_text(text):
    """Tokenize the given text into words and sentences."""
    words = word_tokenize(text)
    sents = sent_tokenize(text)
    return words, sents

def count_words(words):
    """Count the frequency of each word in the word list."""
    return Counter(words)

def count_sents(sents):
    """Count the number of sentences."""
    return len(sents)

def analyze_text(text):
    """Analyze the given text by tokenizing, counting words, and counting sentences."""
    words, sents = tokenize_text(text)
    word_freq = count_words(words)
    num_sents = count_sents(sents)
    return words, sents, word_freq, num_sents

def analyze_datasets(directory):
    """Analyze each dataset in the directory and also combine results for cumulative analysis."""
    datasets = load_datasets(directory)
    cumulative_word_freq = Counter()
    total_sent_count = 0
    
    for filename, df in datasets:
        if 'subject' in df.columns:  
            text_data = " ".join(df["subject"].dropna().tolist())
            words, sents, word_freq, num_sents = analyze_text(text_data)
            
            print(f"\nAnalysis for {filename}:")
            print(f" - Total words: {len(words)}")
            print(f" - Unique words: {len(word_freq)}")
            print(f" - Total sentences: {num_sents}")
            print(f" - Top 5 words: {word_freq.most_common(5)}")
            
            # Update cumulative analysis
            cumulative_word_freq.update(word_freq)
            total_sent_count += num_sents
        else:
            print(f"\nSkipping {filename} - No 'subject' column found.")
    
    # Cumulative analysis summary
    print("\nCumulative Analysis Across All Datasets:")
    print(f" - Total unique words: {len(cumulative_word_freq)}")
    print(f" - Total sentences: {total_sent_count}")
    print(f" - Top 10 words: {cumulative_word_freq.most_common(10)}")

if __name__ == "__main__":
    directory = "Cleaned_datasets"
    analyze_datasets(directory)
