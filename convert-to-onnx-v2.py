import torch
from models.forward_tacotron import ForwardTacotron
from typing import Optional

tts_model = ForwardTacotron.from_checkpoint('C:/Users/LENOVO_User/Documents/ForwardTacotron-NVDA/addon/synthDrivers/Forward/server/forward_step90k.pt')
tts_model.eval()

OPSET = 15
SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def custom_generate(inputs):
    x, alpha, pitch, energy = inputs
    pitch_function = lambda x: x * pitch
    energy_function = lambda x: x * energy
    infer = tts_model.generate(x, alpha=alpha, pitch_function=pitch_function, energy_function=energy_function)
    mel = infer['mel_post']
    return mel

tts_model.forward = custom_generate
dummy_input_length = 50
x = torch.randint(low=0, high=20, size=(1, dummy_input_length), dtype=torch.long)
alpha: Optional[torch.Tensor] = 1.0
pitch: Optional[torch.Tensor] = 1.0
energy: Optional[torch.Tensor] = 1.0
model_inputs = [x, alpha, pitch, energy]
input_names = [
    "x",
    "alpha",
    "pitch",
    "energy"
]

torch.onnx.export(
    model = tts_model,
    args = model_inputs,
    f = 'forward_tacotron.onnx',
    opset_version=OPSET,
    input_names=input_names,
    output_names=['output'],
    dynamic_axes = {
        "x": {0: "batch_size", 1: "text"},
        "output": {0: "batch_size", 1: "time"}
    }
)

print("Modelo exportado correctamente a ONNX.")
