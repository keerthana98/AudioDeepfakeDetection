import librosa
import parselmouth
from parselmouth.praat import call

import numpy as np


class ProsodicFeatureExtractor:
    def __init__(self, y, sr, audio_arr, orig_sr):
        self.y = y
        self.sr = sr
        self.audio_arr = audio_arr
        self.orig_sr = orig_sr

    def extract(self, features_to_extract=None):
        feature_funcs = {
            'f0': self.extract_f0,
            'energy': self.extract_energy,
            'speaking_rate': self.extract_speaking_rate,
            'pauses': self.extract_pauses,
            'formants': self.extract_formants
        }

        if features_to_extract is None:
            features_to_extract = feature_funcs.keys()

        features = {}
        for feature in features_to_extract:
            if feature in feature_funcs:
                result = feature_funcs[feature]()
                if isinstance(result, tuple):
                    features.update(result)
                else:
                    features[feature] = result
                    
        return features

    def extract_f0(self):
        f0, voiced_flag, voiced_probs = librosa.pyin(self.y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        # Handle NaN values
        f0 = np.nan_to_num(f0)
        return f0

    def extract_energy(self):
        return librosa.feature.rms(y=self.y)[0]


    def extract_speaking_rate(self):
        try:
            snd = parselmouth.Sound(self.audio_arr, sampling_frequency=self.orig_sr)
            total_duration = snd.get_total_duration()
            intensity = snd.to_intensity()
            intensity_values = intensity.values.T
            threshold = 0.3 * max(intensity_values)
            syllable_count = len([1 for i in intensity_values if i > threshold])
            speaking_rate = syllable_count / total_duration
            return speaking_rate
        except Exception as e:
            print(f'Error extracting speaking rate: {e}')
            return None

    def extract_pauses(self):
        try:
            snd = parselmouth.Sound(self.audio_arr, sampling_frequency=self.orig_sr)
            silences = call(snd, "To TextGrid (silences)", 100, 0, -25, 0.1, 0.1, "silent", "sounding")
            pauses = [(call(silences, "Get start time of interval", 1, i), call(silences, "Get end time of interval", 1, i)) for i in range(1, call(silences, "Get number of intervals", 1) + 1) if call(silences, "Get label of interval", 1, i) == "silent"]
            return pauses
        except Exception as e:
            print(f'Error extracting pauses: {e}')
            return None
        
    def extract_formants(self):
        try:
            snd = parselmouth.Sound(self.audio_arr, sampling_frequency=self.orig_sr)
            formant = call(snd, "To Formant (burg)", 0.025, 5, 5500, 0.025, 50)
            formant_values = {}
            for i in range(1, 4):  # Extracting the first three formants
                formant_values[f'F{i}_mean'] = call(formant, "Get mean", i, 0, 0, "Hertz")
                formant_values[f'F{i}_stdev'] = call(formant, "Get standard deviation", i, 0, 0, "Hertz")
            return formant_values
        except Exception as e:
            print(f'Error extracting formants: {e}')
            return {}