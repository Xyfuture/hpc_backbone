from models.deblur_gan.model.config import DeBlurConfig
from models.deblur_gan.model.predict import Predictor


def load_deblur_model(device='cpu',config=DeBlurConfig()):
    model = Predictor(device,config)
    return model