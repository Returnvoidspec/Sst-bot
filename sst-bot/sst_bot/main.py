from sst.Diarization import Diarization
from nlp.infos import TextSentiment, TextSummarizer


def analysis():
    """ bot partie return un path avec l'audio 
    bot_to_path()"""
    path = None
    diarization_mod = Diarization()
    sst_text = diarization_mod.Audio_to_text(path)

    summarization_mod = TextSummarizer(sst_text)
    summary = summarization_mod.get_summary()

    sentimentalize_mod = TextSentiment(sst_text)
    sentimentalize_mod.get_sentiments()


def main():
    analysis()
