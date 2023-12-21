from transformers import BartTokenizer, BartForConditionalGeneration
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


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

    def sentiments_analysis(self, speaker_sentences: [str]):
        analyzer = SentimentIntensityAnalyzer()
        sentiments = []
        for position, sentence in speaker_sentences:
            vs = analyzer.polarity_scores(sentence)
            sentiments[position] = str(vs)
        return sentiments

    def get_sentiments(self):
        speaker_sentences = self.get_speaker_sentences()
        return self.sentiments_analysis(speaker_sentences)
