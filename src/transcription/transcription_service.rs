use std::collections::HashSet;
use std::error::Error;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext};
use log::{error, info};
use std::time::{Instant};
use chrono::{TimeZone, Utc};
use crate::{FullTranscription, get_filename_with_timestamp};
use crate::transcription::{handle_segment_transcription, read_and_parse_audio, save_transcription_to_file};
use crate::DEFAULT_N_THREADS;
use crate::EXCLUSION_TERMS;


// New struct to hold the WhisperContext
pub(crate) struct TranscriptionService {
    ctx: WhisperContext,
}

impl TranscriptionService {
    pub(crate) fn new(model_path: &str) -> Result<Self, Box<dyn Error>> {
        let ctx = WhisperContext::new(model_path)?;
        info!("Start of transcribe_audio method");
        Ok(TranscriptionService { ctx })
    }

    pub(crate) fn transcribe_audio(&self, path: &str) -> Result<(), Box<dyn Error>> {
        info!("transcribe_audio: method entry");
        info!("Transcribing audio file: {}", path);
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        params.set_n_threads(DEFAULT_N_THREADS);
        params.set_language(Some("en"));

        let (audio_data, spec, total_samples) = read_and_parse_audio(path)?;

        let audio_duration_seconds = total_samples / (spec.sample_rate as f32 * spec.channels as f32);
        info!("Audio duration: {}", audio_duration_seconds);

        let transcribe_start = Instant::now();

        let mut state = self.ctx.create_state()?;
        info!("State created");

        state.full(params, &audio_data[..])?;
        info!("State: {:?}", state);

        let num_segments = match state.full_n_segments() {
            Ok(segments) => segments,
            Err(e) => {
                error!("Failed to get the number of segments: {}", e);
                return Err(Box::new(e));
            }
        };

        info!("There were {} segments.", num_segments);

        let current_filename = get_filename_with_timestamp();
        let base_time = match Utc.datetime_from_str(&current_filename, "audio_%Y%m%d%H%M%S.wav") {
            Ok(time) => time,
            Err(e) => {
                error!("Failed to parse timestamp from filename {}: {}", current_filename, e);
                return Err(Box::new(e));  // Return the error wrapped in a Box (or use another appropriate error type)
            }
        };

        let exclusion_set: HashSet<_> = EXCLUSION_TERMS.iter().map(|&s| s.to_string()).collect();
        let mut transcription_segments = Vec::new();

        for i in 0..num_segments {
            match handle_segment_transcription(&state, i, &base_time, &exclusion_set) {
                Ok(Some(segment_info)) => {
                    transcription_segments.push(segment_info);
                }
                Ok(None) => {
                    // Log or handle cases where no segment is returned (e.g., excluded segments, etc.)
                }
                Err(e) => {
                    error!("Failed to handle transcription for segment {}: {}", i, e);
                }
            }
        }

        let full_transcription = FullTranscription {
            file_name: current_filename,
            transcriptions: transcription_segments,
        };

        // Serialize full_transcription to JSON and save it
        match serde_json::to_string_pretty(&full_transcription) {
            Ok(json_data) => {
                if let Err(e) = save_transcription_to_file(path, &json_data) {
                    error!("Failed to save transcription to a file: {}", e);
                }
            },
            Err(e) => {
                error!("Failed to serialize full transcription to JSON: {}", e);
            }
        }

        let transcribe_duration = transcribe_start.elapsed();
        let transcribe_duration_seconds = transcribe_duration.as_secs_f32();
        info!("{}, {:.3} seconds of audio, transcribed in {:.3} seconds", path, audio_duration_seconds, transcribe_duration_seconds);

        Ok(())
    }

}
