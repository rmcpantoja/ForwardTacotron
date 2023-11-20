import torch
from models.forward_tacotron import ForwardTacotron
#from models.fast_pitch import FastPitch
from utils.text.symbols import phonemes
from typing import Optional

tts_model = ForwardTacotron.from_checkpoint('C:/Users/LENOVO_User/Documents/ForwardTacotron-NVDA/addon/synthDrivers/Forward/server/forward_step90k.pt')
tts_model.eval()

OPSET = 17
SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def custom_generate(text, synth_options):
    alpha = synth_options[0]
    pitch = synth_options[1]
    energy = synth_options[2]
    pitch_function = lambda x: x * pitch
    energy_function = lambda x: x * energy
    infer = tts_model.generate(
        text,
        alpha=alpha,
        pitch_function=pitch_function,
        energy_function=energy_function
    )
    mel = infer['mel_post']
    return mel

tts_model.forward = custom_generate
dummy_input_length = 50
rand = torch.randint(low=0, high=len(phonemes), size=(1, dummy_input_length), dtype=torch.long)
synth_inputs = torch.FloatTensor([1.0, 1.0, 1.0])
model_inputs = (rand, synth_inputs)
input_names = [
    "input",
    "synth_options"
]

torch.onnx.export(
    model = tts_model,
    args = model_inputs,
    f = 'forward_tacotron.onnx',
    opset_version=OPSET,
    input_names=input_names,
    output_names=['output'],
    dynamic_axes = {
        "input": {0: "batch_size", 1: "text"},
        "output": {0: "batch_size", 1: "time"}
    }
)

print("Modelo exportado correctamente a ONNX.")
