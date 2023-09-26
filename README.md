# Sound Sense

A simple audio transcription tool that captures audio input, saves it into WAV format, and transcribes it using [whisper.cpp](https://github.com/ggerganov/whisper.cpp).

This is a personal project and runs on an [Orange Pi 5B](http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-5B.html), running [Armbian 23.08](https://docs.armbian.com/), that's attached to a [Focusrite Scarlett 6i6](https://www.sweetwater.com/store/detail/Scarlet6i6G2--focusrite-scarlett-6i6-usb-audio-interface). You might need to make configuration changes if you want to use it in a different environment.

Include the path to the whisper.cpp base model as an argument:

    sound_sense -m /home/alexwoolford/whisper.cpp/models/ggml-base.en.bin

The executable transcribes audio to timestamped JSON files:

    {
        "file_name": "audio_20230824011133.wav",
        "transcriptions": [
            {
                "start": "2023-08-24T01:11:33.900+00:00",
                "end": "2023-08-24T01:11:34+00:00",
                "text": "Blessed be the fruit."
            },
            {
                "start": "2023-08-24T01:11:34+00:00",
                "end": "2023-08-24T01:11:34.100+00:00",
                "text": "May the Lord open."
            },
            {
                ...
            {
        ]
    }

It's possible to parse out the text from the metadata with the following jq command:

    jq '.transcriptions[].text' audio_*.json

<!-- 
- [X] TODO: only create WAV files when there's noise/speech
- [ ] TODO: try and use Orange Pi's GPU so the CPU isn't pegged all the time
- [ ] TODO: create iterate transcription test(s) to figure out what the best combinations of params is for the Orange Pi 5B
- [X] TODO: persist the transcriptions
- [X] TODO: capture real timestamps in the transcriptions
- [ ] TODO: chatGPT API call to periodically summarize transcripts
- [ ] TODO: speech patterns (words per minute)
- [ ] TODO: how much attention from the audience (is it a conversation, or a monologue)
- [ ] TODO: haptic feedback in real time
- [ ] TODO: sentiment analysis
- [ ] TODO: real-time feedback (slow down, speed up, pause for questions)
- [ ] TODO: speaker identification
- [ ] TODO: voice stress analysis
- [ ] TODO: context based search on another screen
- [X] TODO: trim text in JSON
- [ ] TODO: show example of output in README.md
- [X] TODO: rename transcript filetype to JSON
-->
