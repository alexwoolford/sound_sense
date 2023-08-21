use std::error::Error;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use get_if_addrs::get_if_addrs;
use hostname;
use std::sync::{Arc, Condvar, mpsc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use hound;

use cpal::{SampleRate, StreamConfig};
use log::{error, info};
use chrono::prelude::*;
use whisper_rs::{WhisperContext, FullParams, SamplingStrategy};

use crate::initialization::init_logger;

mod initialization;

use signal_hook::{iterator::Signals};
use signal_hook::consts::SIGINT;


static RUNNING: AtomicBool = AtomicBool::new(true);


fn get_filename_with_timestamp() -> String {
    let now = Utc::now();
    let timestamp = now.format("%Y%m%d%H%M%S").to_string();
    format!("audio_{}.wav", timestamp)
}

fn is_receiver_empty<T>(rx: &mpsc::Receiver<T>) -> bool {
    match rx.try_recv() {
        Ok(_) => false,
        Err(mpsc::TryRecvError::Empty) => true,
        Err(mpsc::TryRecvError::Disconnected) => true,
    }
}

fn i32_to_f32(sample: i32) -> f32 {
    const MAX_I32_AS_F32: f32 = std::i32::MAX as f32;
    sample as f32 / MAX_I32_AS_F32
}

fn main() -> Result<(), Box<dyn Error>> {
    init_logger()?;

    // Set up the signal handler
    let mut signals = Signals::new(&[SIGINT])?;
    std::thread::spawn(move || {
        for _ in signals.forever() {
            RUNNING.store(false, Ordering::Relaxed);
        }
    });

    info!("Number of logical cores is {}", num_cpus::get());

    // Get hostname
    let hostname = hostname::get()
        .expect("Failed to get hostname")
        .into_string()
        .expect("Failed to convert OsString into String");
    info!("Hostname: {}", hostname);

    match get_if_addrs() {
        Ok(if_addrs) => {
            for if_addr in if_addrs {
                info!("Interface \"{}\" IP: {}", if_addr.name, if_addr.ip());
            }
        }
        Err(e) => info!("Error getting network interface addresses: {}", e),
    }

    let host = cpal::default_host();
    let device = host.default_input_device().expect("no input device available");

    // stream to file
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 32, // Assuming I32 format for now
        sample_format: hound::SampleFormat::Int,
    };

    let (tx, rx) = mpsc::channel();

    let ready_for_transcription = Arc::new((Mutex::new(None::<String>), Condvar::new()));
    let ready_for_transcription_clone = Arc::clone(&ready_for_transcription);

    // The flag to indicate that we are done recording and the `wav_writer` should finish its work.
    let done_recording = Arc::new(AtomicBool::new(false));
    let done_recording_clone = done_recording.clone();

    let wav_writer = std::thread::spawn(move || {
        info!("wav_writer thread started...");

        let mut start_time = Instant::now();
        let mut current_filename = get_filename_with_timestamp();

        let writer_path = current_filename.clone();
        let mut writer = match hound::WavWriter::create(writer_path, spec) {
            Ok(w) => w,
            Err(e) => {
                error!("Failed to create WAV writer: {}", e);
                return;  // Return from the thread.
            }
        };

        // Change the while condition
        while !done_recording_clone.load(Ordering::Relaxed) || !is_receiver_empty(&rx) {

            // Change the try_recv to recv, so that we block until a message is available or the channel is closed.
            match rx.recv() {
                Ok(sample) => {
                    // Write the sample to the WAV file
                    writer.write_sample(sample).unwrap();

                    if start_time.elapsed() > Duration::from_secs(60) {
                        if let Err(e) = writer.finalize() {
                            error!("Failed to finalize WAV writer for {}: {}", current_filename, e);
                            return;  // Return from the thread.
                        }

                        // Notify about the finished file right after finalizing it.
                        let (lock, cvar) = &*ready_for_transcription_clone;
                        {
                            let mut ready_file = match lock.lock() {
                                Ok(guard) => guard,
                                Err(poisoned) => {
                                    error!("Mutex was poisoned. Recovering...");
                                    poisoned.into_inner()
                                }
                            };

                            *ready_file = Some(current_filename.clone());
                        }
                        cvar.notify_one();

                        start_time = Instant::now();
                        current_filename = get_filename_with_timestamp();
                        let writer_path = current_filename.clone();
                        writer = match hound::WavWriter::create(writer_path, spec) {
                            Ok(w) => w,
                            Err(e) => {
                                error!("Failed to create WAV writer for {}: {}", current_filename, e);
                                return;  // Return from the thread.
                            }
                        };
                    }

                }
                Err(_) => {
                    // Log the error when we cannot receive any more samples (this should happen when tx is dropped).
                    error!("Failed to receive sample. Exiting wav_writer loop.");
                    break;
                }
            }
        }

        writer.finalize().unwrap();
        info!("wav_writer thread finished...");

    });

    // 1. Instantiate the TranscriptionService after initializing the logger
    // let model_path = "/home/alexwoolford/whisper.cpp/models/ggml-medium.en.bin";
    // let model_path = "/home/alexwoolford/whisper.cpp/models/ggml-small.en.bin";
    let model_path = "/home/alexwoolford/whisper.cpp/models/ggml-base.en.bin";
    let transcription_service = Arc::new(TranscriptionService::new(model_path)?);


    let transcription = {
        let ready_for_transcription = Arc::clone(&ready_for_transcription);
        let transcription_service_clone = Arc::clone(&transcription_service);

        std::thread::spawn(move || {
            let (lock, cvar) = &*ready_for_transcription;

            loop {
                if !RUNNING.load(Ordering::Relaxed) {
                    break;
                }
                let mut ready_file = lock.lock().unwrap();
                while ready_file.is_none() && RUNNING.load(Ordering::Relaxed) {
                    ready_file = cvar.wait(ready_file).unwrap();
                }

                if let Some(filename) = ready_file.take() {
                    transcription_service_clone.transcribe_audio(&filename);
                }
            }

        })
    };

    let config = StreamConfig {
        channels: 6,
        sample_rate: SampleRate(16000),
        buffer_size: cpal::BufferSize::Default,
    };

    let channel_to_capture = 0; // This means first channel. Adjust if your microphone is on another channel.
    let total_channels = 6;

    // Inside your input stream callback, send samples to the writer:
    let stream = device.build_input_stream(
        &config,
        move |data: &[i32], _: &cpal::InputCallbackInfo| {

            for (idx, &sample) in data.iter().enumerate() {
                if idx % total_channels == channel_to_capture {
                    match tx.send(sample) {
                        Ok(_) => continue,
                        Err(e) => {
                            error!("Input stream callback: {}. Terminating.", e);
                            return;
                        }
                    }
                }
            }

        },
        |err| error!("An error occurred on stream: {}", err),
        None
    )?;

    stream.play()?;

    while RUNNING.load(Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_secs(1));
    }

    // Stop the audio input stream FIRST
    drop(stream);
    info!("Stream dropped...");

    // THEN indicate that recording is done.
    done_recording.store(true, Ordering::Relaxed);
    info!("Signaled that recording is done.");

    // Now you can join the wav_writer thread to ensure it finishes processing.
    if let Err(e) = wav_writer.join() {
        error!("Error in wav_writer thread: {:?}", e);
    }
    info!("wav_writer joined, exiting...");

    if let Err(e) = transcription.join() {
        error!("Error in transcriptions thread: {:?}", e);
    }
    info!("transcription joined, exiting...");

    Ok(())

}

// New struct to hold the WhisperContext
struct TranscriptionService {
    ctx: WhisperContext,
}

impl TranscriptionService {
    fn new(model_path: &str) -> Result<Self, Box<dyn Error>> {
        let ctx = WhisperContext::new(model_path).expect("Failed to load model");
        Ok(TranscriptionService { ctx })
    }

    fn transcribe_audio(&self, path: &str) {

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });

        params.set_n_threads(8);
        params.set_language(Some("en"));

        // Read the audio data from the provided path
        info!("Attempting to open file: {}", path);
        let mut reader = match hound::WavReader::open(path) {
            Ok(r) => r,
            Err(e) => {
                error!("Failed to open WAV file {}: {}", path, e);
                return;
            }
        };

        let audio_data: Vec<f32> = reader.samples::<i32>().map(|s| i32_to_f32(s.unwrap())).collect();

        let spec = reader.spec();
        let total_samples = reader.len() as f32;
        let audio_duration_seconds = total_samples / (spec.sample_rate as f32 * spec.channels as f32);

        let transcribe_start = Instant::now();

        // Transcribe the audio using the already loaded ctx from the struct
        let mut state = self.ctx.create_state().expect("failed to create state");
        state
            .full(params, &audio_data[..])
            .expect("failed to run model");

        let num_segments = state
            .full_n_segments()
            .expect("failed to get number of segments");

        info!("There were {} segments.", num_segments);

        let mut transcription_content = String::new();

        for i in 0..num_segments {
            info!("Processing segment {}", i);
            let segment = state
                .full_get_segment_text(i)
                .expect("failed to get segment");
            let start_timestamp = state
                .full_get_segment_t0(i)
                .expect("failed to get segment start timestamp");
            let end_timestamp = state
                .full_get_segment_t1(i)
                .expect("failed to get segment end timestamp");
            let segment_info = format!("[{} - {}]: {}\n", start_timestamp, end_timestamp, segment);
            info!("{}", &segment_info);

            // Append this segment's transcription to the full content
            transcription_content.push_str(&segment_info);
        }

        // Save the transcription to a file
        if let Err(e) = self.save_transcription_to_file(path, &transcription_content) {
            error!("Failed to save transcription to a file: {}", e);
        }

        let transcribe_duration = transcribe_start.elapsed();
        let transcribe_duration_seconds = transcribe_duration.as_secs_f32();
        info!("{}, {:.3} seconds of audio, transcribed in {:.3} seconds", path, audio_duration_seconds, transcribe_duration_seconds);

    }

    fn save_transcription_to_file(&self, path: &str, content: &str) -> Result<(), std::io::Error> {
        use std::fs::File;
        use std::io::prelude::*;

        let txt_path = path.replace(".wav", ".txt");
        let mut file = File::create(txt_path)?;
        file.write_all(content.as_bytes())?;
        Ok(())
    }

}
