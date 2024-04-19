import os
import argparse
import torch
from models.forward_tacotron import ForwardTacotron
from utils.text.symbols import phonemes
import json
import convert_phonemes

"""
ONNX convertor for ⏩ ForwardTacotron
Lately, ONNX stuff for TTS models is popular, because these models provides a faster inference than the full PyTorch models. Faster inference is good to use these models, for example, in a screen reader. Also, ONNX models can be used in a multi-platform/system way such as IOS, Android Phone devices, etc.
The onnx compatibility has been fixed by Matthew C. (rmcpantoja).
"""

# ======================global vars======================
OPSET = 17
SEED = 1234
# ======================end global vars======================

# Declaring the convertor:
def run_convertor(model_path, save_path):
	if not os.path.exists(model_path):
		raise FileNotFoundError("Please give me an existing model!")
	checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
	config = checkpoint["config"]
	tts_model = ForwardTacotron.from_config(config)
	tts_model.load_state_dict(checkpoint['model'])
	tts_model.eval()
	# Configure seed:
	torch.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)
	# We create the custom generate() function to return the mel post only and acomodate synthesizer options into a list array:
	def custom_generate(text, synth_options):
		alpha = synth_options[0]
		pitch = synth_options[1]
		energy = synth_options[2]
		# Todo: try inferencing this pitch/energy with an ONNX model:
		pitch_function = lambda x: x * pitch
		energy_function = lambda x: x * energy
		infer = tts_model.generate(
			text,
			alpha=alpha,
			pitch_function=pitch_function,
			energy_function=energy_function,
			onnx=True
		)
		mel = infer['mel_post']
		return mel
	# We replace the forward function to the created one:
	tts_model.forward = custom_generate
	# We set the inputs and outputs for the ONNX model:
	dummy_input_length = 50
	rand = torch.randint(low=0, high=len(phonemes), size=(1, dummy_input_length), dtype=torch.long)
	synth_inputs = torch.FloatTensor(
		[1.0, 1.0, 1.0] # Alpha, pitch, energy
	)
	model_inputs = (rand, synth_inputs)
	input_names = [
		"input",
		"synth_options"
	]
	if save_path is None:
		save_path = model_path[:-3]+".onnx"
	# Finally, we export it:
	torch.onnx.export(
		model = tts_model,
		args = model_inputs,
		f = save_path,
		opset_version=OPSET,
		input_names=input_names,
		output_names=['output'],
		dynamic_axes = {
			"input": {0: "batch_size", 1: "text"},
			"output": {0: "batch_size", 1: "time"}
		}
	)
	# We create a new config:
	new_phonemes = convert_phonemes.get_phone_table()
	with open(f"{save_path}.json", "w", encoding="utf-8") as conf:
		json.dump(
			{
				"audio": {
					"sample_rate": config["dsp"]["sample_rate"],
				},
				"espeak": {
					"voice": config["preprocessing"]["language"],
				},
				"phoneme_id_map": new_phonemes
			},
			conf,
			ensure_ascii=False,
			indent=4
		)
	print("Checkpoint successfully converted to ONNX.")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Onnx conversor for ⏩ForwardTacotron")
	parser.add_argument(
		'--checkpoint_path',
		'-c',
		required=True,
		type=str,
		help='The full checkpoint (*.pt) file to convert.'
	)
	parser.add_argument(
		'--output_path',
		'-o',
		default=None,
		type=str,
		help='Output path to save the converted ONNX model.'
	)
	args = parser.parse_args()
	run_convertor(args.checkpoint_path, args.output_path)