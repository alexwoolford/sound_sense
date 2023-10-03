use std::error::Error;
use std::sync::{Arc, Condvar, mpsc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use chrono::prelude::*;
use clap::{arg, command, value_parser};
use std::path::PathBuf;
use cpal::{SampleRate, StreamConfig};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use log::{error, info};
use serde::Serialize;
use signal_hook::consts::SIGINT;
use signal_hook::iterator::Signals;

use crate::initialization::init_logger;
use crate::transcription::initialize_wav_writer;
use crate::transcription::transcription_service::TranscriptionService;


mod initialization;
mod transcription;

const DEFAULT_N_THREADS: i32 = 4;
const DEFAULT_CHANNELS: u16 = 6;
const EXCLUSION_TERMS: [&str; 4] = ["[silence]", "(Silence)", "[BLANK_AUDIO]", "[ Silence ]"];

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

#[derive(Serialize)]
pub struct TranscriptionSegment {
    start: String,
    end: String,
    text: String,
}

#[derive(Serialize)]
struct FullTranscription {
    file_name: String,
    transcriptions: Vec<TranscriptionSegment>,
}

fn main() -> Result<(), Box<dyn Error>> {
    init_logger()?;

    let matches = command!()
        .arg(
            arg!(
                -m --MODEL_PATH <MODEL_PATH> "The path to the model file"
            )
            .required(true)
            .value_parser(value_parser!(PathBuf)),
        )
        .get_matches();

    // Set up the signal handler
    let mut signals = Signals::new([SIGINT])?;
    std::thread::spawn(move || {
        for _ in signals.forever() {
            RUNNING.store(false, Ordering::Relaxed);
        }
    });

    info!("Number of logical cores is {}", num_cpus::get());

    let host = cpal::default_host();
    let device = match host.default_input_device() {
        Some(device) => device,
        None => {
            error!("No input device available. Exiting.");
            std::process::exit(1);
        }
    };

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

        let mut start_time: Instant;
        let mut current_filename: String;
        let mut writer: hound::WavWriter<std::io::BufWriter<std::fs::File>>;

        match initialize_wav_writer(spec) {
            Ok((start_time_val, current_filename_val, writer_val)) => {
                start_time = start_time_val;
                current_filename = current_filename_val;
                writer = writer_val;
            },
            Err(e) => {
                error!("Failed to initialize WAV writer: {}", e);
                return;
            }
        }

        while !done_recording_clone.load(Ordering::Relaxed) || !is_receiver_empty(&rx) {
            match rx.recv() {
                Ok(sample) => {
                    // Write the sample to the WAV file
                    if let Err(e) = writer.write_sample(sample) {
                        error!("Failed to write sample to WAV file for {}: {}", current_filename, e);
                        continue;
                    }

                    if start_time.elapsed() > Duration::from_secs(60) {
                        if let Err(e) = writer.finalize() {
                            error!("Failed to finalize WAV writer for {}: {}", current_filename, e);
                            return;  // Return from the thread.
                        }

                        info!("Wrote WAV file: {}", current_filename);

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
                            info!("Ready file: {:?}", ready_file);
                        }

                        cvar.notify_one();

                        start_time = Instant::now();
                        current_filename = get_filename_with_timestamp();
                        writer = match hound::WavWriter::create(current_filename.clone(), spec) {
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

        if let Err(e) = writer.finalize() {
            error!("Failed to finalize WAV writer for {}: {}", current_filename, e);
        }
        info!("wav_writer thread finished...");
    });

    let model_path = matches.get_one::<PathBuf>("MODEL_PATH")
        .ok_or("MODEL_PATH argument is required")?;

    let model_path_str = model_path.to_str()
        .ok_or("The provided MODEL_PATH contains invalid Unicode")?;

    let transcription_service = Arc::new(TranscriptionService::new(model_path_str)?);

    let transcription = {
        let ready_for_transcription = Arc::clone(&ready_for_transcription);
        let transcription_service_clone = Arc::clone(&transcription_service);
        info!("Transcription thread started...");
        std::thread::spawn(move || {
            let (lock, cvar) = &*ready_for_transcription;

            loop {
                if !RUNNING.load(Ordering::Relaxed) {
                    break;
                }

                let mut ready_file = match lock.lock() {
                    Ok(guard) => guard,
                    Err(poisoned) => {
                        error!("Mutex was poisoned. Recovering...");
                        poisoned.into_inner()
                    }
                };

                while ready_file.is_none() && RUNNING.load(Ordering::Relaxed) {
                    ready_file = match cvar.wait(ready_file) {
                        Ok(guard) => guard,
                        Err(poisoned) => {
                            error!("Mutex was poisoned after waiting. Recovering...");
                            poisoned.into_inner()
                        }
                    };
                }

                if let Some(filename) = ready_file.take() {
                    if let Err(e) = transcription_service_clone.transcribe_audio(&filename) {
                        error!("Failed to transcribe audio for {}: {}", filename, e);
                    }
                    info!("Transcription finished for {}", filename);
                }
            }
        })
    };

    let config = StreamConfig {
        channels: DEFAULT_CHANNELS,
        sample_rate: SampleRate(16000),
        buffer_size: cpal::BufferSize::Default,
    };

    let channel_to_capture = 0; // This means first channel. Adjust if your microphone is on another channel.
    let total_channels = DEFAULT_CHANNELS;

    // Inside your input stream callback, send samples to the writer:
    let stream = device.build_input_stream(
        &config,
        move |data: &[i32], _: &cpal::InputCallbackInfo| {
            for (idx, &sample) in data.iter().enumerate() {
                if idx % (total_channels as usize) == channel_to_capture {
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
        None,
    )?;

    stream.play()?;

    while RUNNING.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_secs(1));
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
