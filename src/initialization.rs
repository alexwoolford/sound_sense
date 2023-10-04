use std::error::Error;
use std::io::Write;

use colored::*;

#[derive(Clone)]
pub struct Config {
    pub model_path: String,
    pub default_n_threads: i32,
    pub default_channels: u16,
    pub channels: u16,
    pub sample_rate: u32,
    pub bits_per_sample: u16,
    pub channel_to_capture: usize,
    pub exclusion_terms: Vec<String>,
}

impl Config {
    pub fn new() -> Self {
        Self {
            model_path: "".to_string(),
            default_n_threads: 4,
            default_channels: 6,

            // Hound wav encoding configs
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 32,
            channel_to_capture: 0, // 0 means the first channel

            exclusion_terms: vec![
                "[silence]".to_string(),
                "(Silence)".to_string(),
                "[BLANK_AUDIO]".to_string(),
                "[ Silence ]".to_string(),
            ],
        }
    }
}


// Initializes the logger for the application with the provided configuration.
pub fn init_logger() -> Result<(), Box<dyn Error>> {

    colored::control::set_override(true);

    let mut builder = env_logger::Builder::new();

    builder.filter_level(log::LevelFilter::Info);

    builder.format(|buf, record| {
        let level = record.level();
        let colored_level = match level {
            log::Level::Error => level.to_string().red(),
            log::Level::Warn => level.to_string().yellow(),
            log::Level::Info => level.to_string().green(),
            log::Level::Debug => level.to_string().blue(),
            log::Level::Trace => level.to_string().purple(),
        };

        let emoji = match level {
            log::Level::Error => "❌",
            log::Level::Warn => "⚠️",
            log::Level::Info => "✔️",
            log::Level::Debug => "🔍",
            log::Level::Trace => "🔬",
        };

        writeln!(
            buf,
            "{} {} [{}] {}",
            emoji,
            record.target().cyan(),
            colored_level,
            record.args()
        )
    });

    builder.init();

    Ok(())
}
