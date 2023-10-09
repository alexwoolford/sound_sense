use std::error::Error;
use std::sync::{Arc, Condvar, mpsc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use clap::{arg, command, value_parser};
use std::path::PathBuf;
use cpal::{SampleRate, StreamConfig};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use log::{error, info};
use serde::Serialize;
use signal_hook::consts::SIGINT;
use signal_hook::iterator::Signals;

use crate::initialization::init_logger;
use crate::transcription::{initialize_wav_writer};
use crate::transcription::transcription_service::TranscriptionService;


mod initialization;
mod transcription;


static RUNNING: AtomicBool = AtomicBool::new(true);

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

fn handle_signals(mut signals: Signals) {
    std::thread::spawn(move || {
        for _ in signals.forever() {
            RUNNING.store(false, Ordering::Relaxed);
        }
    });
}

fn initialize_audio_device(host: &cpal::Host) -> Result<cpal::Device, Box<dyn Error>> {
    match host.default_input_device() {
        Some(device) => Ok(device),
        None => Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "No input device available.",
        ))),
    }
}


fn configure_stream_to_file(config: &initialization::Config) -> hound::WavSpec {
    hound::WavSpec {
        channels: config.channels,
        sample_rate: config.sample_rate,
        bits_per_sample: config.bits_per_sample,
        sample_format: hound::SampleFormat::Int,
    }
}


fn start_transcription_thread(
    ready_for_transcription: Arc<(Mutex<Option<String>>, Condvar)>,
    transcription_service: Arc<TranscriptionService>,
) -> std::thread::JoinHandle<()> {
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
                if let Err(e) = transcription_service.transcribe_audio(&filename) {
                    error!("Failed to transcribe audio for {}: {}", filename, e);
                }
                info!("Transcription finished for {}", filename);
            }
        }
    })
}


fn handle_audio_input_stream(
    device: &cpal::Device,
    stream_config: &StreamConfig,
    channel_to_capture: usize,
    total_channels: u16,
    tx: mpsc::Sender<i32>,
) -> Result<cpal::Stream, Box<dyn Error>> {
    device.build_input_stream(
        &stream_config,
        move |data: &[i32], _: &cpal::InputCallbackInfo| {
            for (idx, &sample) in data.iter().enumerate() {
                if idx % (total_channels as usize) == channel_to_capture {
                    if tx.send(sample).is_err() {
                        error!("Failed to send sample. Terminating input stream callback.");
                        return;
                    }
                }
            }
        },
        |err| error!("An error occurred on stream: {}", err),
        None,
    )
        .map_err(|e| Box::new(e) as Box<dyn Error>)
}


fn run_loop() {
    while RUNNING.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_secs(1));
    }
}


fn cleanup_and_exit(
    stream: cpal::Stream,
    done_recording: Arc<AtomicBool>,
    wav_writer_thread_handle: std::thread::JoinHandle<()>,
    transcription: std::thread::JoinHandle<()>,
) -> Result<(), Box<dyn Error>> {
    // Stop the audio input stream FIRST
    drop(stream);
    info!("Stream dropped...");

    // THEN indicate that recording is done.
    done_recording.store(true, Ordering::Relaxed);
    info!("Signaled that recording is done.");

    // Now you can join the wav_writer thread to ensure it finishes processing.
    if let Err(e) = wav_writer_thread_handle.join() {
        error!("Error in wav_writer thread: {:?}", e);
    }
    info!("wav_writer joined, exiting...");

    if let Err(e) = transcription.join() {
        error!("Error in transcriptions thread: {:?}", e);
    }
    info!("transcription joined, exiting...");

    Ok(())
}



fn main() -> Result<(), Box<dyn Error>> {
    let config = initialization::Config::new();
    init_logger()?;

    handle_signals(Signals::new([SIGINT])?);

    info!("Number of logical cores is {}", num_cpus::get());

    let host = cpal::default_host();
    let device = initialize_audio_device(&host)?;

    let spec = configure_stream_to_file(&config);

    let (tx, rx) = mpsc::channel();

    let ready_for_transcription = Arc::new((Mutex::new(None::<String>), Condvar::new()));
    let ready_for_transcription_clone = Arc::clone(&ready_for_transcription);

    // The flag to indicate that we are done recording and the `wav_writer` should finish its work.
    let done_recording = Arc::new(AtomicBool::new(false));

    let wav_writer_thread_handle = spawn_wav_writer_thread(spec, done_recording.clone(), ready_for_transcription_clone, rx);

    let transcription_service = TranscriptionService::new(parse_command_line_args()?, &config)?;

    let transcription = start_transcription_thread(
        Arc::clone(&ready_for_transcription),
        Arc::new(transcription_service),
    );

    let stream_config = StreamConfig {
        channels: config.default_channels,
        sample_rate: SampleRate(config.sample_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    let stream = handle_audio_input_stream(
        &device,
        &stream_config,
        config.channel_to_capture,
        stream_config.channels,
        tx
    )?;

    stream.play()?;

    run_loop();

    cleanup_and_exit(stream, done_recording, wav_writer_thread_handle, transcription)

}

fn parse_command_line_args() -> Result<PathBuf, Box<dyn Error>> {
    let matches = command!()
        .arg(
            arg!(
                -m --MODEL_PATH <MODEL_PATH> "The path to the model file"
            )
                .required(true)
                .value_parser(value_parser!(PathBuf)),
        )
        .get_matches();

    let model_path = matches.get_one::<PathBuf>("MODEL_PATH")
        .ok_or("MODEL_PATH argument is required")?
        .to_path_buf();

    Ok(model_path)
}

pub trait WavWriter {
    fn write_sample(&mut self, sample: i32) -> Result<(), hound::Error>;
    fn finalize(self) -> Result<(), hound::Error>;
}

impl WavWriter for hound::WavWriter<std::io::BufWriter<std::fs::File>> {
    fn write_sample(&mut self, sample: i32) -> Result<(), hound::Error> {
        self.write_sample(sample)
    }

    fn finalize(self) -> Result<(), hound::Error> {
        self.finalize()
    }
}

fn write_wav(
    done_recording: Arc<AtomicBool>,
    ready_for_transcription: Arc<(Mutex<Option<String>>, Condvar)>,
    rx: mpsc::Receiver<i32>,
    mut writer: hound::WavWriter<std::io::BufWriter<std::fs::File>>,
    mut current_filename: String,
    spec: hound::WavSpec,
    mut start_time: Instant,
) {
    // Main logic for writing to the WAV file and handling transcription readiness.
    while !done_recording.load(Ordering::Relaxed) || !is_receiver_empty(&rx) {
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
                        return;
                    }

                    info!("Wrote WAV file: {}", current_filename);

                    // Notify about the finished file right after finalizing it.
                    let (lock, cvar) = &*ready_for_transcription;
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

                    writer = match initialize_wav_writer(spec) {
                        Ok((_, current_filename_val, writer_val)) => {
                            current_filename = current_filename_val;
                            *writer_val
                        },
                        Err(e) => {
                            error!("Failed to initialize WAV writer: {}", e);
                            return;
                        },
                    };

                }
            }
            Err(_) => {
                error!("Failed to receive sample. Exiting wav_writer loop.");
                break;
            }
        }
    }

    // Finalize the WAV writer upon exit of the loop.
    if let Err(e) = writer.finalize() {
        error!("Failed to finalize WAV writer for {}: {}", current_filename, e);
    }
    info!("wav_writer logic finished...");
}


fn spawn_wav_writer_thread(
    spec: hound::WavSpec,
    done_recording: Arc<AtomicBool>,
    ready_for_transcription: Arc<(Mutex<Option<String>>, Condvar)>,
    rx: mpsc::Receiver<i32>
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        info!("wav_writer thread started...");

        let start_time: Instant;
        let current_filename: String;
        let writer: hound::WavWriter<std::io::BufWriter<std::fs::File>>;

        match initialize_wav_writer(spec) {
            Ok((start_time_val, current_filename_val, writer_val)) => {
                start_time = start_time_val;
                current_filename = current_filename_val;
                writer = *writer_val;
            },
            Err(e) => {
                error!("Failed to initialize WAV writer: {}", e);
                return;
            }
        }

        write_wav(done_recording, ready_for_transcription, rx, writer, current_filename, spec, start_time);

    })
}

