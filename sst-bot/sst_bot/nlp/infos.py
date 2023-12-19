from transformers import BartTokenizer, BartForConditionalGeneration


class TextSummarizer:
    def __init__(self, sst_text: str, number_people: int, model_name: str = "philschmid/bart-large-cnn-samsum"):
        self.sst_text = sst_text
        self.number_people = number_people
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)

    def summarize_text(self, text: str) -> str:
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

    def get_summary(self) -> str:
        return self.summarize_text(self.sst_text)
