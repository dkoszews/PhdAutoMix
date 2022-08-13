from Config import cfg
import Evaluate
import os


def CreateEvalConfig():
    model_path = os.path.join("checkpoints", "stem_to_mix_model", "stem_to_mix_model-6400")

    input_path = {
                    "bass": "E:\\phd\\Experiments\\ListeningSamples\\Secretariat_Over_The_Top\\unet\src\\bass2.wav",
                    "bass_1": "E:\\phd\\Experiments\\ListeningSamples\\Secretariat_Over_The_Top\\unet\src\\bass1.wav",
                    "drums": "E:\\phd\\Experiments\\ListeningSamples\\Secretariat_Over_The_Top\\unet\src\\drums2.wav",
                    "drums_1": "E:\\phd\\Experiments\\ListeningSamples\\Secretariat_Over_The_Top\\unet\src\\drums1.wav",
                    "other": "E:\\phd\\Experiments\\ListeningSamples\\Secretariat_Over_The_Top\\unet\src\\other2.wav",
                    "other_1": "E:\\phd\\Experiments\\ListeningSamples\\Secretariat_Over_The_Top\\unet\src\\other1.wav",
                    "vocals": "E:\\phd\\Experiments\\ListeningSamples\\Secretariat_Over_The_Top\\unet\src\\vocals2.wav",
                    "vocals_1": "E:\\phd\\Experiments\\ListeningSamples\\Secretariat_Over_The_Top\\unet\src\\vocals1.wav"
    }

    output_path = 'audio_examples/outputs/secretariat2_unet.wav'
    return model_path, input_path, output_path


def main():
    model_path, input_path, output_path = CreateEvalConfig()
    model_config = cfg()
    model_config["input_names"] = ['bass', "bass_1", 'drums', 'drums_1', 'other', 'other_1', 'vocals', 'vocals_1']
    model_config["num_inputs"] = len(model_config["input_names"])
    model_config["num_outputs"] = 1 if model_config["mono_downmix"] else 2
    Evaluate.produce_outputs(model_config, model_path, input_path, output_path)


if __name__ == "__main__":
    main()
