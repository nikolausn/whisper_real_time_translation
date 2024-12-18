from translatepy.translators.google import GoogleTranslate
import argparse
import io
import os
import speech_recognition as sr
from queue import Queue
from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform
from faster_whisper import WhisperModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="large-v3", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--device", default="auto", help="device to user for CTranslate2 inference",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--compute_type", default="auto", help="Type of quantization to use",
                        choices=["auto", "int8", "int8_floatt16", "float16", "int16", "float32"])
    parser.add_argument("--source_lang", default='en',
                        help="Source language speaker to focus on.", type=str)
    parser.add_argument("--translation_lang", default='English',
                        help="Which language should we translate into.", type=str)
    parser.add_argument("--threads", default=0,
                        help="number of threads used for CPU inference", type=int)
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=5,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=10,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)

    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    def record_callback(_, audio: sr.AudioData) -> None:
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.

        # record from microphone

        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # The last time a recording was retreived from the queue.
    phrase_time = None
    silent_time = None
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()

    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    source=None

    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    temp_file = NamedTemporaryFile().name
    transcription = []

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    total_phrase = timedelta(seconds=0)
    total_silent = timedelta(seconds=0)

    # for speaker en only tiny is enough
    model = "distil-medium.en"

    audio_model = WhisperModel(model, device = "cuda")

    last_transcription = None

    while True:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if data_queue.empty():
                phrase_complete=False
                if silent_time:
                    total_silent += (now - silent_time)
                if silent_time and total_silent > timedelta(seconds=3):
                    phrase_complete = True
                    print("idle",(now - silent_time), timedelta(seconds=record_timeout), total_silent)
                    total_silent=timedelta(seconds=0)
                silent_time = now

                # Read the transcription.
                text = ""

                #if not os.path.exists(temp_file):
                #    continue

                if phrase_complete:
                    if last_transcription:
                        #need_transcription = os.path.getsize(temp_file)
                        #print("idle",need_transcription,last_transcription)
                        #if need_transcription != last_transcription:
                        if wav_data != last_transcription:
                            segments, info = audio_model.transcribe(wav_data)
                            #last_transcription = os.path.getsize(temp_file)
                            last_transcription = wav_data
                            silent_time = datetime.utcnow()
                            total_silent = timedelta(seconds=0)
                            text = ""
                            for segment in segments:
                                text += segment.text
                            print("idle transcribe",text)
                            transcription.append(text)
                    if len(transcription) > 0:
                        result = " ".join(transcription)
                        print("result:",result)
                        gtranslate = GoogleTranslate()
                        #translate_text = str(gtranslate.translate(result, source_language="en", destination_language="ja"))
                        translate_text = str(gtranslate.translate(result, source_language="en", destination_language="es"))
                        test = gtranslate.text_to_speech(translate_text,source_language="es")
                        test.write_to_file(f"test-{datetime.now().strftime('%Y%d%m%H%M%S')}.wav")
                        #print("translate:",translate_text)

                    transcription = []
            else:
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time:
                    total_phrase += (now - phrase_time)
                    print((now - phrase_time), timedelta(seconds=phrase_timeout), total_phrase)
                #if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                if phrase_time and total_phrase > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                    total_phrase=timedelta(seconds=0)
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write wav data to the temporary file as bytes.
                """
                with open(temp_file, 'w+b') as f:
                    print(temp_file)
                    f.write(wav_data.read())
                """                    

                # Read the transcription.
                text = ""

                # If we detected a pause between recordings, add a new item to our transcripion.
                # Otherwise edit the existing one.
                if phrase_complete:
                    #transcription.append(text)
                    #need_transcription = os.path.getsize(temp_file)
                    #if need_transcription == last_transcription:
                    if wav_data == last_transcription:
                        continue
                    segments, info = audio_model.transcribe(wav_data)
                    #last_transcription = os.path.getsize(temp_file)
                    last_transcription = wav_data
                    silent_time = datetime.utcnow()
                    total_silent = timedelta(seconds=0)
                    for segment in segments:
                        text += segment.text
                    print(text)
                    transcription.append(text)
                    last_sample = bytes()

                    #gtranslate = GoogleTranslate()
                    #translate_text = str(gtranslate.translate(text, source_language="en", destination_language="ja"))

                    #test = gtranslate.text_to_speech(translate_text,source_language="ja")
                    #test.write_to_file(f"test-{datetime.now()}.wav")
                else:
                    #transcription[-1] = text
                    pass
                #last_four_elements = transcription[-10:]
                last_four_elements = transcription[-5:]
                result = ''.join(last_four_elements)

                # Clear the console to reprint the updated transcription.

                # Infinite loops are bad for processors, must sleep.
                sleep(0.25)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()