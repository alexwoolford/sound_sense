# Sound Sense

A simple audio transcription tool that captures audio input, saves it into WAV format, and transcribes it using [whisper.cpp](https://github.com/ggerganov/whisper.cpp).

This is a personal project and runs on an [Orange Pi 5B](http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-5B.html), running [Armbian 23.08](https://docs.armbian.com/), that's attached to a [Focusrite Scarlett 6i6](https://www.sweetwater.com/store/detail/Scarlet6i6G2--focusrite-scarlett-6i6-usb-audio-interface). You might need to make configuration changes if you want to use it in a different environment.

Make sure to set up the appropriate path to the Whisper model:


```rust
let model_path = "/home/alexwoolford/whisper.cpp/models/ggml-small.en.bin";
```
<!-- 
- [ ] TODO: only create WAV files when there's noise/speech
- [ ] TODO: try and use Orange Pi's GPU so the CPU isn't pegged all the time
- [ ] TODO: create iterate transcription test(s) to figure out what the best combinations of params is for the Orange Pi 5B
- [ ] TODO: persist the transcriptions
- [ ] TODO: capture real timestamps in the transcriptions
- [ ] TODO: chatGPT API call to periodically summarize transcripts
-->

