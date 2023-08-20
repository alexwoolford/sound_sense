use std::error::Error;
use std::io::Write;

use colored::*;

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
