import parselmouth
from parselmouth.praat import call
import numpy as np


class VoiceQualityFeatureExtractor:
    def __init__(self, audio_arr, orig_sr):
        self.audio_arr = audio_arr
        self.orig_sr = orig_sr

    def extract(self, features_to_extract=None):
        feature_funcs = {
            'jitter': self.extract_jitter,
            'shimmer': self.extract_shimmer,
            'hnr': self.extract_hnr
        }

        if features_to_extract is None:
            features_to_extract = feature_funcs.keys()

        features = {}
        for feature in features_to_extract:
            if feature in feature_funcs:
                feature_values = feature_funcs[feature]()
                if isinstance(feature_values, dict):
                    features.update(feature_values)
                else:
                    features[feature] = feature_values
        return features

    def extract_jitter(self):
        try:
            snd = parselmouth.Sound(self.audio_arr, sampling_frequency=self.orig_sr)
            point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
            jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            return {
                'jitter_local': jitter_local,
                'jitter_rap': jitter_rap,
                'jitter_ppq5': jitter_ppq5
            }
        except Exception as e:
            print(f'Error extracting jitter: {e}')
            return {
                'jitter_local': np.nan,
                'jitter_rap': np.nan,
                'jitter_ppq5': np.nan
            }

    def extract_shimmer(self):
        try:
            snd = parselmouth.Sound(self.audio_arr, sampling_frequency=self.orig_sr)
            point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
            shimmer_local = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq3 = call([snd, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq5 = call([snd, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_dda = call([snd, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            return {
                'shimmer_local': shimmer_local,
                'shimmer_apq3': shimmer_apq3,
                'shimmer_apq5': shimmer_apq5,
                'shimmer_dda': shimmer_dda
            }
        except Exception as e:
            print(f'Error extracting shimmer: {e}')
            return {
                'shimmer_local': np.nan,
                'shimmer_apq3': np.nan,
                'shimmer_apq5': np.nan,
                'shimmer_dda': np.nan
            }

    def extract_hnr(self):
        try:
            snd = parselmouth.Sound(self.audio_arr, sampling_frequency=self.orig_sr)
            harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            return {'hnr': hnr}
        except Exception as e:
            print(f'Error extracting HNR: {e}')
            return {'hnr': np.nan}