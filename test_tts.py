import torch
from TTS.api import TTS
import datetime

'''
device = "cuda"

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

tts.tts_to_file(text="""
こんにちは、私の名前はニコラウスです。情報科学の博士号を持っています
""", speaker_wav="sample-niko.aac", language="ja", emotion="happy", file_path="output.wav")
'''

from translatepy.translators.google import GoogleTranslate

#gtranslate = GoogleTranslate()
#gtranslate.translate(text[i], translation_lang)

#test = gtranslate.text_to_speech("こんにちは、私の名前はニコラウスです。情報科学の博士号を持っています",source_language="ja")
#test.write_to_file("test.wav")

tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=True).to("cuda")
now = datetime.datetime.now()
tts.voice_conversion_to_file(source_wav="test.wav", target_wav="sample-niko.aac", file_path="output-test.wav")
print(datetime.datetime.now()-now)
