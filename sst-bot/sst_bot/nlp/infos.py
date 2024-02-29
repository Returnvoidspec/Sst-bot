from transformers import BartTokenizer, BartForConditionalGeneration
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from collections import Counter
from string import punctuation
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
import spacy


class TextSummarizer:
    def __init__(self, sst_text: str, number_people: int = 2, model_name: str = "philschmid/bart-large-cnn-samsum"):
        self.sst_text = sst_text
        self.number_people = number_people
        try:
            self.tokenizer = BartTokenizer.from_pretrained(model_name)
            self.model = BartForConditionalGeneration.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.tokenizer = None
            self.model = None

    def summarize_text(self, text: str) -> str:
        if not self.model or not self.tokenizer:
            return "Model loading failed. Summarization unavailable."
        try:
            # Encodage du texte d'entrée
            inputs = self.tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)

            # Génération du résumé
            summary_ids = self.model.generate(inputs['input_ids'],
                                              num_beams=4,
                                              max_length=75,
                                              min_length=25,
                                              length_penalty=2.0,
                                              no_repeat_ngram_size=3,
                                              early_stopping=True)
            return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error during summarization: {e}")
            return "Summarization failed."

    def get_summary(self) -> str:
        return self.summarize_text(self.sst_text)


def convert_to_label(compound_score):
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'


def sentiments_analysis(speaker_sentences: [[str]]):
    analyzer = SentimentIntensityAnalyzer()
    speakers_compound_scores = []
    overall_compound_scores = []

    # Iterate over each speaker's sentences
    for sentences in speaker_sentences:
        speaker_scores = [analyzer.polarity_scores(sentence)['compound'] for sentence in sentences]
        speakers_compound_scores.append(speaker_scores)
        overall_compound_scores.extend(speaker_scores)

    # Calculate average compound score and sentiment label for each speaker
    speaker_sentiments = []
    for scores in speakers_compound_scores:
        if scores:  # Ensure there are scores to average
            average_score = sum(scores) / len(scores)
            speaker_sentiments.append(convert_to_label(average_score))
        else:
            speaker_sentiments.append('neutral')  # Default to neutral if no sentences

    # Calculate overall sentiment label
    overall_compound = sum(overall_compound_scores) / len(overall_compound_scores) if overall_compound_scores else 0
    overall_label = convert_to_label(overall_compound)

    return speaker_sentiments, overall_label


class TextSentiment:
    def __init__(self, sst_text: str, number_people: int = 2):
        self.sst_text = sst_text
        self.number_people = number_people

    def get_speaker_sentences(self):
        speaker_sentences = []
        for i in range(self.number_people):
            try:
                pattern = re.compile(rf"SPEAKER_{'0' if i < 10 else ''}{i} :  (.+)")
                sentences = pattern.findall(self.sst_text)
                speaker_sentences.append(sentences)
            except re.error as e:
                print(f"Regex error: {e}")
                speaker_sentences.append([])
        return speaker_sentences

    def get_sentiments(self):
        speaker_sentences = self.get_speaker_sentences()
        return sentiments_analysis(speaker_sentences)


class TextAnalysis:
    def __init__(self, sst_text: str, number_people: int = 2):
        self.sst_text = sst_text
        self.number_people = number_people
        self.summarizer = TextSummarizer(sst_text, number_people)
        self.sentiment = TextSentiment(sst_text, number_people)

    def get_relevance_words(self):
        # Load the English tokenizer, tagger, parser, NER, and word vectors
        nlp = spacy.load("en_core_web_sm")

        # Process the text
        doc = nlp(self.sst_text)

        # Filter tokens which are not stop words, punctuation, and are nouns, proper nouns, or adjectives
        relevant_words = {token.text.lower() for token in doc if
                          token.pos_ in ["NOUN", "PROPN", "ADJ"] and not token.is_stop and not token.is_punct}

        return relevant_words

    def get_speaker_relevant_word_counts(self):
        # Extract relevant words
        relevant_words = self.get_relevance_words()

        # Get sentences per speaker
        speaker_sentences = self.sentiment.get_speaker_sentences()

        # Initialize a dictionary to count relevant words per speaker
        speaker_relevant_word_counts = {f"SPEAKER_{i}": 0 for i in range(self.number_people)}

        # Count relevant words for each speaker
        for i, sentences in enumerate(speaker_sentences):
            for sentence in sentences:
                # Tokenize the sentence and count relevant words
                words = sentence.lower().split()
                speaker_relevant_word_counts[f"SPEAKER_{i}"] += sum(word in relevant_words for word in words)

        return speaker_relevant_word_counts

    def analyze_speaker_activity(self, threshold=5):
        # Get relevant word counts per speaker
        word_counts = self.get_speaker_relevant_word_counts()

        # Determine activity level based on threshold
        activity_status = {speaker: "active" if count >= threshold else "not active" for speaker, count in
                           word_counts.items()}

        return activity_status

    def preprocess_text(self):
        cleaned_text = re.sub(r"SPEAKER_\d+\s*:", "", self.sst_text)
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(cleaned_text.lower())
        lemmatized_text = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
        return lemmatized_text

    def get_topics(self, num_words=1):
        """
        Perform topic modeling on the meeting text and print out the main topics.
        """
        cleaned_text = self.preprocess_text()
        word_count = len(cleaned_text.split())

        # Adjust the number of topics based on the word count
        if word_count < 500:
            num_topics = 1  # Fewer topics for shorter texts
        elif 500 <= word_count < 1000:
            num_topics = 3
        else:
            num_topics = 5  # More topics for longer texts

        # Load English tokenizer, tagger, parser, NER, and word vectors
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(cleaned_text.lower())

        texts = [[word for word in simple_preprocess(str(sent.text)) if word not in STOPWORDS] for sent in doc.sents if
                 len(sent.text.strip()) > 0]
        if not texts:
            print("No data available for LDA after preprocessing.")
            return

        dictionary = Dictionary(texts)
        dictionary.filter_extremes(no_below=2, no_above=0.5)
        corpus = [dictionary.doc2bow(text) for text in texts]

        if not corpus:
            print("No data available in the corpus for LDA.")
            return

        # Train the LDA model
        lda_model = LdaModel(corpus=corpus,
                             id2word=dictionary,
                             num_topics=num_topics,
                             random_state=100,
                             update_every=1,
                             chunksize=100,
                             passes=10,
                             alpha='auto',
                             per_word_topics=True)

        topics = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
        for topic_num, topic_terms in topics:
            print(f"Topic {topic_num + 1}: ", ", ".join([word for word, _ in topic_terms]))


def count_speakers(sst_text: str) -> int:
    # Use a regular expression to find all occurrences of speaker identifiers
    speaker_ids = re.findall(r"SPEAKER_(\d+)", sst_text)
    # Convert the list of speaker IDs to a set to remove duplicates, then count the elements
    unique_speakers = set(speaker_ids)
    return len(unique_speakers)
