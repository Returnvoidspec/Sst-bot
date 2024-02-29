from sst.Diarization import Diarization
from nlp.infos import TextSentiment, TextSummarizer, TextAnalysis


def analysis():
    path = "test_conv_short.wav"
    diarization_mod = Diarization()
    sst_text, speakers_number = diarization_mod.Audio_to_text(path)

    summarization_mod = TextSummarizer(sst_text, speakers_number)
    summary = summarization_mod.get_summary()

    sentimentalize_mod = TextSentiment(sst_text, speakers_number)
    speaker_sentiments, overall_sentiment = sentimentalize_mod.get_sentiments()
    print("Speaker Sentiments:", speaker_sentiments)
    print("Overall Sentiment:", overall_sentiment)

    print("\nResume of the meeting:" + summary)

    analysis_mod = TextAnalysis(sst_text, speakers_number)  # Assuming 2 speakers in the conversation
    activity_status = analysis_mod.analyze_speaker_activity()
    print("\nSpeaker Activity Status:", activity_status)

    print("\nExtracted Topics:")
    analysis_mod.get_topics()


def main():
    analysis()


if __name__ == '__main__':
    main()
