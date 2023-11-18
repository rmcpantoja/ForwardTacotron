from utils.text.cleaners import Cleaner

cleaner = Cleaner(cleaner_name='no_cleaners', use_phonemes=True, lang='en-us')
text = "hello there!"
print(text)
cleaned = cleaner(text)
print(cleaned)