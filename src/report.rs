pub struct Report {
    issues: Vec<String>,
}

impl Report {
    pub fn new() -> Self {
        Self { issues: vec![] }
    }

    pub fn add(&mut self, msg: String) {
        self.issues.push(msg);
    }

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

    pub fn to_string(&self) -> String {
        if self.issues.is_empty() {
            "No issues found".to_string()
        } else {
            let mut out = String::from("Issues:");
            for i in &self.issues {
                out.push_str("\n- ");
                out.push_str(i);
            }
            out
        }
    }
}
