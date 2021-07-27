from sacred import Experiment
from Config import config_ingredient
import Evaluate
import os

ex = Experiment('Mixwaveunet Prediction', ingredients=[config_ingredient])

@ex.config
def cfg():
    model_path = os.path.join("checkpoints", "wet", "wet-1108000") # Load wet pretrained model by default

    input_path = {'hi-hat': 'E:\\phd\\evalTests\\input\\Hi-hat_44.1kHz.wav',
  'kick': 'E:\\phd\\evalTests\\input\\KickMain_44.1kHz.wav',
  'mix': None,
  'overhead_L': 'E:\\phd\\evalTests\\input\\ohL_44.1kHz.wav',
  'overhead_R': 'E:\\phd\\evalTests\\input\\ohR_44.1kHz.wav',
  'snare': 'E:\\phd\\evalTests\\input\\Snaretop545_44.1kHz.wav',
  'tom_1': 'E:\\phd\\evalTests\\input\\Tomhi_44.1kHz.wav',
  'tom_2': 'E:\\phd\\evalTests\\input\\Tomlow_44.1kHz.wav',
  'tom_3': None}

    output_path = 'audio_examples/outputs/wet_mix.wav'
    

@ex.automain
def main(cfg, model_path, input_path, output_path):
    model_config = cfg["model_config"]
    Evaluate.produce_outputs(model_config, model_path, input_path, output_path)