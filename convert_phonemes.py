from utils.text.tokenizer import Tokenizer
import json
import os

t = Tokenizer()

def get_phone_table():
	global t
	symbol_to_id_compatible = {id: [symbol] for id, symbol in t.symbol_to_id.items()}
	# Debug:
	#print(symbol_to_id_compatible)
	return symbol_to_id_compatible

def to_json(phoneme_dict):
	with open("phoneme_ids.json", "w", encoding="utf-8") as ids:
		json.dump(
			{
				"phoneme_id_map": phoneme_dict
			}
		,
		ids,
		ensure_ascii=False,
		indent=4
	)

if __name__ == '__main__':
	phonemes = get_phone_table()
	# Save:
	if not os.path.exists("phoneme_ids.json"):
		to_json(phonemes)
		print("Phoneme map converted successfully.")