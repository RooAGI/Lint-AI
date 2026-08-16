use std::fmt;

#[derive(Default)]
pub struct Report {
    issues: Vec<String>,
}

impl Report {
    /// Create an empty report.
    pub fn new() -> Self {
        Self { issues: vec![] }
    }

    /// Add a new issue line.
    pub fn add(&mut self, msg: String) {
        self.issues.push(msg);
    }

    /// Print issues to stdout.
    pub fn print(&self) {
        if self.issues.is_empty() {
            println!("No issues found");
        } else {
            println!("Issues:");
            for i in &self.issues {
                println!("- {}", i);
            }
        }
    }
}

impl fmt::Display for Report {
    /// Render issues as a single string.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.issues.is_empty() {
            f.write_str("No issues found")
        } else {
            let mut out = String::from("Issues:");
            for i in &self.issues {
                out.push_str("\n- ");
                out.push_str(i);
            }
            f.write_str(&out)
        }
    }
}
