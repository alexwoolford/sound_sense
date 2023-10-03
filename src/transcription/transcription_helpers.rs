use std::collections::HashSet;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use log::{debug, error, info};
use std::time::{Instant};
use chrono::Utc;
use whisper_rs::WhisperState;
use crate::TranscriptionSegment;


pub fn i32_to_f32(sample: i32) -> f32 {
    const MAX_I32_AS_F32: f32 = i32::MAX as f32;
    sample as f32 / MAX_I32_AS_F32
}

pub fn get_filename_with_timestamp() -> String {
    let now = Utc::now();
    let timestamp = now.format("%Y%m%d%H%M%S").to_string();
    format!("audio_{}.wav", timestamp)
}

pub fn initialize_wav_writer(spec: hound::WavSpec) -> Result<(Instant, String, hound::WavWriter<std::io::BufWriter<std::fs::File>>), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let current_filename = get_filename_with_timestamp();

    let writer_path = current_filename.clone();
    let writer = hound::WavWriter::create(writer_path, spec)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    Ok((start_time, current_filename, writer))
}

pub fn read_and_parse_audio(path: &str) -> Result<(Vec<f32>, hound::WavSpec, f32), Box<dyn Error>> {
    // Read the audio data from the provided path
    let mut reader = hound::WavReader::open(path)?;
    let audio_data: Vec<f32> = reader.samples::<i32>().filter_map(Result::ok).map(i32_to_f32).collect();
    info!("Audio data length: {}", audio_data.len());

    // Extract audio specifications
    let spec = reader.spec();

    // Calculate the total number of samples in the audio
    let total_samples = reader.len() as f32;
    info!("Total samples: {}", total_samples);

    Ok((audio_data, spec, total_samples))
}

pub fn handle_segment_transcription(
    state: &WhisperState,
    i: i32,
    base_time: &chrono::DateTime<chrono::Utc>,
    exclusion_set: &HashSet<String>,
) -> Result<Option<TranscriptionSegment>, Box<dyn std::error::Error>> {
    debug!("Transcribing segment {}...", i + 1);
    match state.full_get_segment_text(i) {
        Ok(segment) => {
            let trimmed_text = segment.trim().to_string();

            // Check if the trimmed text is not in the exclusion set
            if !exclusion_set.contains(&trimmed_text) {
                let start_timestamp = match state.full_get_segment_t0(i) {
                    Ok(timestamp) => timestamp,
                    Err(e) => {
                        error!("Failed to get segment start timestamp for segment {}: {}", i, e);
                        return Err(Box::new(e));
                    }
                };

                let end_timestamp = match state.full_get_segment_t1(i) {
                    Ok(timestamp) => timestamp,
                    Err(e) => {
                        error!("Failed to get segment end timestamp for segment {}: {}", i, e);
                        return Err(Box::new(e));
                    }
                };

                let start_time = *base_time + chrono::Duration::milliseconds(start_timestamp);
                let end_time = *base_time + chrono::Duration::milliseconds(end_timestamp);

                let segment_info = TranscriptionSegment {
                    start: start_time.to_rfc3339(),
                    end: end_time.to_rfc3339(),
                    text: trimmed_text,
                };
                debug!(
                    "pushing segment to transcription segments: start: {}, end: {}",
                    segment_info.start, segment_info.end
                );
                Ok(Some(segment_info))
            } else {
                Ok(None)
            }
        }
        Err(_) => {
            error!("Error getting transcription segment text.");
            Ok(None)
        }
    }
}

pub fn save_transcription_to_file(path: &str, content: &str) -> Result<(), std::io::Error> {
    info!("Saving transcription to: {}", path);

    let json_path = path.replace(".wav", ".json");
    let mut file = File::create(json_path)?;
    file.write_all(content.as_bytes())?;
    info!("Successfully saved transcription to file");
    Ok(())
}