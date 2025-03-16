import os
import tempfile
import subprocess
import streamlit as st
from gtts import gTTS
import whisper
from deep_translator import GoogleTranslator
import soundfile as sf  # for writing audio files if needed


# ----------------------------
# Step 1: Extract Audio from Video using FFmpeg
# ----------------------------
def extract_audio(video_path, audio_path):
    # This FFmpeg command extracts audio from the video, converts it to WAV
    # with a sample rate of 16000 Hz and 1 audio channel.
    command = [
        "ffmpeg",
        "-y",  # overwrite output files without asking
        "-i",
        video_path,  # input video file
        "-vn",  # no video
        "-acodec",
        "pcm_s16le",  # use PCM 16-bit little endian codec
        "-ar",
        "16000",  # set audio sampling rate to 16 kHz
        "-ac",
        "1",  # set number of audio channels to 1
        audio_path,
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    st.write(f"Audio extracted and saved to {audio_path}")


# ----------------------------
# Step 2: Transcribe Audio with Whisper AI
# ----------------------------
def transcribe_audio_with_whisper(audio_path, model_name="base"):
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, word_timestamps=True)
    segments = result["segments"]  # list of dicts with 'start', 'end', and 'text'
    transcript_with_timestamps = [
        (seg["start"], seg["end"], seg["text"]) for seg in segments
    ]
    return transcript_with_timestamps


# ----------------------------
# Step 3: Translate Text with Timestamps to Urdu
# ----------------------------
def translate_text_with_timestamps(transcript_with_timestamps):
    translator = GoogleTranslator(source="en", target="ur")
    return [
        (start, end, translator.translate(text))
        for start, end, text in transcript_with_timestamps
    ]


# ----------------------------
# Step 4: Generate Urdu Voiceover Clips with Adjusted Timings
# ----------------------------
def generate_voice_clips(translated_segments, output_folder, speech_rate=12):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    clip_paths = []
    current_start_time = 0  # Track adjusted start time

    for i, (_, _, text) in enumerate(translated_segments):
        clip_path = os.path.join(output_folder, f"clip_{i}.mp3")
        tts = gTTS(text=text, lang="ur")
        tts.save(clip_path)

        # Estimate duration based on character count and speech rate
        estimated_duration = len(text) / speech_rate
        clip_paths.append(
            (current_start_time, current_start_time + estimated_duration, clip_path)
        )
        current_start_time += estimated_duration

    return clip_paths


# ----------------------------
# Step 5: Merge Adjusted Audio Segments with Video using FFmpeg
# ----------------------------
def merge_audio_segments_with_video(video_path, voice_clips, output_video_path):
    input_audio_clips = []
    filter_complex = ""

    for i, (start, _, clip_path) in enumerate(voice_clips):
        input_audio_clips.extend(["-i", clip_path])
        # Calculate delay in milliseconds (for stereo: provide delay for both channels)
        delay = int(start * 1000)
        filter_complex += f"[{i+1}:a]adelay={delay}|{delay}[a{i}];"

    filter_complex += (
        "".join([f"[a{i}]" for i in range(len(voice_clips))])
        + f"amix=inputs={len(voice_clips)}[outa]"
    )

    command = (
        ["ffmpeg", "-y", "-i", video_path]
        + input_audio_clips
        + [
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v:0",
            "-map",
            "[outa]",
            "-shortest",
            "-c:v",
            "copy",
            output_video_path,
        ]
    )
    subprocess.run(command, check=True)
    st.write(f"Final video saved to {output_video_path}")


# ----------------------------
# Streamlit App
# ----------------------------
def main():
    st.title("Video Translator with Urdu Voiceover")

    st.markdown("Upload a video file (mp4, mov, avi) to process:")
    video_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
    speech_rate = st.number_input("Speech Rate (characters per second)", value=12)

    if video_file is not None:
        # Save uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        ) as temp_video_file:
            temp_video_file.write(video_file.read())
            video_path = temp_video_file.name

        st.video(video_path)

        if st.button("Process Video"):
            # Step 1: Extract Audio
            with st.spinner("Extracting audio from video..."):
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".wav"
                ) as temp_audio_file:
                    audio_path = temp_audio_file.name
                extract_audio(video_path, audio_path)

            # Step 2: Transcribe Audio with Whisper
            with st.spinner("Transcribing audio with Whisper..."):
                transcript_segments = transcribe_audio_with_whisper(audio_path)
                st.write("Transcribed Segments:", transcript_segments)

            # Step 3: Translate Transcript to Urdu
            with st.spinner("Translating transcript to Urdu..."):
                translated_segments = translate_text_with_timestamps(
                    transcript_segments
                )
                st.write("Translated Segments:", translated_segments)

            # Step 4: Generate Urdu Voiceover Clips
            with st.spinner("Generating voiceover clips..."):
                output_folder = tempfile.mkdtemp()
                voice_clips = generate_voice_clips(
                    translated_segments, output_folder, speech_rate
                )
                st.write("Voice clips generated:", voice_clips)

            # Step 5: Merge Voiceover with Original Video
            with st.spinner("Merging voiceover with video..."):
                output_video_path = os.path.join(
                    tempfile.gettempdir(), "output_video.mp4"
                )
                merge_audio_segments_with_video(
                    video_path, voice_clips, output_video_path
                )

            st.video(output_video_path)
            st.success("Processing complete!")


if __name__ == "__main__":
    main()
