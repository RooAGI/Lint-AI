#[cfg(feature = "agy")]
pub mod agy;
#[cfg(feature = "claude-code")]
pub mod claude_code;
#[cfg(feature = "codex")]
pub mod codex;
#[cfg(any(feature = "gemini-cli", feature = "agy"))]
pub mod gemini_cli;

const MAX_HOOK_INPUT_BYTES: u64 = 8 * 1024 * 1024;

pub(crate) fn read_bounded_json<T: serde::de::DeserializeOwned>() -> anyhow::Result<T> {
    use std::io::Read;
    let mut bytes = Vec::new();
    std::io::stdin()
        .lock()
        .take(MAX_HOOK_INPUT_BYTES + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_HOOK_INPUT_BYTES {
        anyhow::bail!("hook input exceeds {MAX_HOOK_INPUT_BYTES} bytes");
    }
    Ok(serde_json::from_slice(&bytes)?)
}

pub(crate) fn read_bounded_stdin() -> anyhow::Result<String> {
    use std::io::Read;
    let mut bytes = Vec::new();
    std::io::stdin()
        .lock()
        .take(MAX_HOOK_INPUT_BYTES + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_HOOK_INPUT_BYTES {
        anyhow::bail!("hook input exceeds {MAX_HOOK_INPUT_BYTES} bytes");
    }
    Ok(String::from_utf8(bytes)?)
}
pub mod mcp_health;
pub mod mcp_index;
pub mod mcp_tools;
pub mod mcp_transport;
pub mod recall;
pub mod session_recording;
