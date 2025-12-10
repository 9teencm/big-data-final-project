import joblib
import numpy as np

class Model:
    def __init__(self, model_path:str):
        self.model = joblib.load(model_path)
        print(f'已載入模型：{model_path}')
    
    def model_predict(self, pretest_score: float, video_duration: float, ai_parner: float):
        sample_input = np.array([[pretest_score, video_duration, ai_parner]])
        score_gap = self.model.predict(sample_input)
        return score_gap[0]