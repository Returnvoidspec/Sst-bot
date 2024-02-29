#!pip install git+https://github.com/openai/whisper.git.com/openai/whisper.git > /dev/null
#!pip install -q git+https://github.com/pyannote/pyannote-audio > /dev/null
# !pip install torchaudio
# !pip install IPython
# !pip install pydub webrtcvad
#!pip install ffmpeg-python


import whisper
import torch
from pyannote.audio import Pipeline


class Diarization:
    def __init__(self, model_size='large', language='English', auth_token = "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"):
        self.model_size = model_size
        self.language = language
        self.auth_token = auth_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_models()

    def _load_models(self):
        model_name = self.model_size
        if self.language == 'English' and self.model_size != 'large':
            model_name += '.en'
        self.whisper_model = whisper.load_model(model_name)

        self.diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                                             use_auth_token=self.auth_token)
        self.diarization_pipeline.to(self.device)

    def perform_diarization(self, path):
        """
        Perform diarization on an audio file.

        This method uses the loaded Whisper model to transcribe the audio and then applies the diarization pipeline to identify different speakers in the audio file. It returns both the transcription and diarization results.

        Parameters:
        - path (str): The file path to the audio file (.wav format) for which diarization is to be performed.

        Returns:
            - result: The transcription result from the Whisper model. It is a dictionary with keys such as 'segments', each containing information like 'start', 'end', and 'text' for each transcribed segment.
            - diarization_result: The result from the diarization pipeline, which includes information about the different speakers identified in the audio and their respective time segments.
        """
        result = self.whisper_model.transcribe(path)
        diarization_result = self.diarization_pipeline(path)
        return result, diarization_result

    def text_diarization_merge(self, result, diarization_result):
        """
        Merge the transcription and diarization results to associate text with speakers.

        This method takes the transcription result from the Whisper model and the diarization result, and merges them. It assigns each transcribed segment to the most likely speaker based on the time alignment. The method prints the speaker label along with the corresponding text segment.

        Parameters:
        - result (dict): The transcription result from the Whisper model. It is expected to contain 'segments', each with 'start', 'end', and 'text' keys.
        - diarization_result: The result from the diarization pipeline, containing information about identified speakers and their time segments.
        Returns:
        - The text with diarization as string
        """
        Text = ""
        for segment in result["segments"]:
            start, end = segment['start'], segment['end']
            text = segment["text"]
            best, save_speaker = None, None
            for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                distance = abs(start - turn.start) + abs(end - turn.end)
                if best is None or distance < best:
                    best, save_speaker = distance, speaker
            Text += f"{save_speaker} : {text}\n"
        return Text

    def Audio_to_text(self, path):
        result, diarization_result = self.perform_diarization(path)

        speaker_labels = set()
        for segment, _, speaker_label in diarization_result.itertracks(yield_label=True):
            speaker_labels.add(speaker_label)

        number_of_speakers = len(speaker_labels)

        return self.text_diarization_merge(result, diarization_result), number_of_speakers

