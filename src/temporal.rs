use chrono::{DateTime, Datelike, Duration, NaiveDate, Utc, Weekday};
use regex::Regex;
use std::collections::HashSet;
use std::sync::OnceLock;
use std::time::SystemTime;
use temps::chrono::{parse_to_datetime, Language};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TemporalTarget {
    pub target_date: NaiveDate,
    pub window_days: i64,
}

pub fn augment_query_with_temporal_context(query: &str, question_date: Option<&str>) -> String {
    let Some(base_date) = parse_date(question_date) else {
        return query.to_string();
    };

    let lower = query.to_lowercase();
    let mut tokens = Vec::new();
    let mut seen = HashSet::new();

    for marker in temporal_markers(&lower) {
        push_unique(&mut tokens, &mut seen, marker);
    }

    if lower.contains("today") {
        push_date_tokens(&mut tokens, "today", base_date);
    }
    if lower.contains("yesterday") {
        push_date_tokens(&mut tokens, "yesterday", base_date - Duration::days(1));
    }
    if lower.contains("tomorrow") {
        push_date_tokens(&mut tokens, "tomorrow", base_date + Duration::days(1));
    }

    if lower.contains("last week") {
        push_date_tokens(&mut tokens, "last week", base_date - Duration::weeks(1));
    }
    if lower.contains("this week") {
        push_date_tokens(&mut tokens, "this week", base_date);
    }
    if lower.contains("next week") {
        push_date_tokens(&mut tokens, "next week", base_date + Duration::weeks(1));
    }

    if lower.contains("last month") {
        push_date_tokens(&mut tokens, "last month", shift_months(base_date, -1));
    }
    if lower.contains("this month") {
        push_date_tokens(&mut tokens, "this month", base_date);
    }
    if lower.contains("next month") {
        push_date_tokens(&mut tokens, "next month", shift_months(base_date, 1));
    }

    if lower.contains("last year") {
        push_date_tokens(&mut tokens, "last year", shift_years(base_date, -1));
    }
    if lower.contains("this year") {
        push_date_tokens(&mut tokens, "this year", base_date);
    }
    if lower.contains("next year") {
        push_date_tokens(&mut tokens, "next year", shift_years(base_date, 1));
    }

    for (label, date) in period_patterns(&lower, base_date) {
        push_date_tokens(&mut tokens, &label, date);
    }

    for (label, date) in weekend_patterns(&lower, base_date) {
        push_date_tokens(&mut tokens, &label, date);
    }

    for (n, unit, phrase) in ago_patterns(&lower) {
        let resolved = match unit.as_str() {
            "day" => base_date - Duration::days(n as i64),
            "week" => base_date - Duration::weeks(n as i64),
            "month" => shift_months(base_date, -(n as i32)),
            "year" => shift_years(base_date, -(n as i32)),
            _ => base_date,
        };
        push_date_tokens(&mut tokens, &phrase, resolved);
    }

    for (n, unit, phrase) in future_patterns(&lower) {
        let resolved = match unit.as_str() {
            "day" => base_date + Duration::days(n as i64),
            "week" => base_date + Duration::weeks(n as i64),
            "month" => shift_months(base_date, n as i32),
            "year" => shift_years(base_date, n as i32),
            _ => base_date,
        };
        push_date_tokens(&mut tokens, &phrase, resolved);
    }

    for (prefix, weekday) in weekday_patterns(&lower) {
        let resolved = resolve_weekday(base_date, prefix.as_str(), weekday);
        push_date_tokens(
            &mut tokens,
            &format!("{} {}", prefix, weekday_name(weekday)),
            resolved,
        );
    }

    if tokens.is_empty() {
        return query.to_string();
    }

    format!("{} {}", query, tokens.join(" "))
}

pub fn parse_temporal_date(input: Option<&str>) -> Option<NaiveDate> {
    parse_date(input)
}

pub fn resolve_temporal_target(query: &str, anchor_date: Option<&str>) -> Option<TemporalTarget> {
    let base_date = parse_date(anchor_date)
        .unwrap_or_else(|| DateTime::<Utc>::from(SystemTime::now()).date_naive());
    if let Some(target) = resolve_temporal_target_with_temps(query, base_date) {
        return Some(target);
    }
    let lower = query.to_lowercase();

    if lower.contains("today") {
        return Some(TemporalTarget {
            target_date: base_date,
            window_days: 2,
        });
    }
    if lower.contains("yesterday") {
        return Some(TemporalTarget {
            target_date: base_date - Duration::days(1),
            window_days: 2,
        });
    }
    if lower.contains("tomorrow") {
        return Some(TemporalTarget {
            target_date: base_date + Duration::days(1),
            window_days: 2,
        });
    }

    if lower.contains("last week") {
        return Some(TemporalTarget {
            target_date: base_date - Duration::weeks(1),
            window_days: 7,
        });
    }
    if lower.contains("this week") {
        return Some(TemporalTarget {
            target_date: base_date,
            window_days: 7,
        });
    }
    if lower.contains("next week") {
        return Some(TemporalTarget {
            target_date: base_date + Duration::weeks(1),
            window_days: 7,
        });
    }

    if lower.contains("last month") {
        return Some(TemporalTarget {
            target_date: shift_months(base_date, -1),
            window_days: 14,
        });
    }
    if lower.contains("this month") {
        return Some(TemporalTarget {
            target_date: base_date,
            window_days: 14,
        });
    }
    if lower.contains("next month") {
        return Some(TemporalTarget {
            target_date: shift_months(base_date, 1),
            window_days: 14,
        });
    }

    if lower.contains("last year") {
        return Some(TemporalTarget {
            target_date: shift_years(base_date, -1),
            window_days: 30,
        });
    }
    if lower.contains("this year") {
        return Some(TemporalTarget {
            target_date: base_date,
            window_days: 30,
        });
    }
    if lower.contains("next year") {
        return Some(TemporalTarget {
            target_date: shift_years(base_date, 1),
            window_days: 30,
        });
    }

    if let Some((label, date)) = period_patterns(&lower, base_date).into_iter().next() {
        return Some(TemporalTarget {
            target_date: date,
            window_days: if label.contains("year") {
                30
            } else if label.contains("month") {
                14
            } else {
                7
            },
        });
    }

    if let Some((label, date)) = weekend_patterns(&lower, base_date).into_iter().next() {
        return Some(TemporalTarget {
            target_date: date,
            window_days: if label.contains("weekend") { 2 } else { 7 },
        });
    }

    if let Some((n, unit, _phrase)) = ago_patterns(&lower).into_iter().next() {
        let target_date = match unit.as_str() {
            "day" => base_date - Duration::days(n as i64),
            "week" => base_date - Duration::weeks(n as i64),
            "month" => shift_months(base_date, -(n as i32)),
            "year" => shift_years(base_date, -(n as i32)),
            _ => base_date,
        };
        return Some(TemporalTarget {
            target_date,
            window_days: match unit.as_str() {
                "day" => 2,
                "week" => 7,
                "month" => 14,
                "year" => 30,
                _ => 7,
            },
        });
    }

    if let Some((n, unit, _phrase)) = future_patterns(&lower).into_iter().next() {
        let target_date = match unit.as_str() {
            "day" => base_date + Duration::days(n as i64),
            "week" => base_date + Duration::weeks(n as i64),
            "month" => shift_months(base_date, n as i32),
            "year" => shift_years(base_date, n as i32),
            _ => base_date,
        };
        return Some(TemporalTarget {
            target_date,
            window_days: match unit.as_str() {
                "day" => 2,
                "week" => 7,
                "month" => 14,
                "year" => 30,
                _ => 7,
            },
        });
    }

    if let Some((prefix, weekday)) = weekday_patterns(&lower).into_iter().next() {
        return Some(TemporalTarget {
            target_date: resolve_weekday(base_date, prefix.as_str(), weekday),
            window_days: 2,
        });
    }

    None
}

fn resolve_temporal_target_with_temps(query: &str, base_date: NaiveDate) -> Option<TemporalTarget> {
    let now_date = DateTime::<Utc>::from(SystemTime::now()).date_naive();
    let tokens = query_word_tokens(query);
    let mut best: Option<(usize, usize, NaiveDate, String)> = None;

    for start in 0..tokens.len() {
        for end in (start + 1)..=tokens.len().min(start + 8) {
            let phrase = tokens[start..end].join(" ");
            let Ok(parsed) = parse_to_datetime(&phrase, Language::English) else {
                continue;
            };
            let parsed_date = parsed.date_naive();
            let span_len = end - start;
            let candidate = (span_len, start, parsed_date, phrase);
            let replace = best
                .as_ref()
                .map(|current| {
                    candidate.0 > current.0 || (candidate.0 == current.0 && candidate.1 < current.1)
                })
                .unwrap_or(true);
            if replace {
                best = Some(candidate);
            }
        }
    }

    let Some((_, _, parsed_date, phrase)) = best else {
        return None;
    };

    let delta_days = parsed_date.signed_duration_since(now_date).num_days();
    let target_date = base_date + Duration::days(delta_days);
    Some(TemporalTarget {
        target_date,
        window_days: window_days_for_phrase(&phrase),
    })
}

pub fn extract_temporal_terms(
    timestamp: Option<&str>,
    content: &str,
    headings: &[String],
) -> Vec<String> {
    let mut terms = Vec::new();
    let mut seen = HashSet::new();

    if let Some(date) = parse_date(timestamp) {
        push_date_tokens(&mut terms, "date", date);
        push_unique(
            &mut terms,
            &mut seen,
            format!("weekday {}", weekday_name(date.weekday())),
        );
        push_unique(
            &mut terms,
            &mut seen,
            format!("month {}", date.format("%B").to_string().to_lowercase()),
        );
        push_unique(&mut terms, &mut seen, format!("year {}", date.year()));
    }

    for text in headings
        .iter()
        .map(String::as_str)
        .chain(std::iter::once(content))
    {
        let lower = text.to_lowercase();
        for date in iso_date_mentions(&lower) {
            push_date_tokens(&mut terms, "date", date);
        }
        for (month, day) in month_day_mentions(&lower) {
            push_unique(&mut terms, &mut seen, format!("month {}", month));
            push_unique(&mut terms, &mut seen, format!("day {}", day));
            push_unique(&mut terms, &mut seen, format!("{} {}", month, day));
        }
        for weekday in standalone_weekday_mentions(&lower) {
            push_unique(
                &mut terms,
                &mut seen,
                format!("weekday {}", weekday_name(weekday)),
            );
        }
        for marker in temporal_markers(&lower) {
            push_unique(&mut terms, &mut seen, marker);
        }
    }

    terms.sort();
    terms.dedup();
    terms
}

fn temporal_markers(lower: &str) -> Vec<String> {
    let mut out = Vec::new();
    static MARKER_RE: OnceLock<Regex> = OnceLock::new();
    let re = MARKER_RE.get_or_init(|| {
        Regex::new(r"\b(two|three|four|five|six|seven|eight|nine|ten|1|2|3|4|5|6|7|8|9|10)\s+(day|week|month|year)s?\s+ago\b")
            .expect("valid temporal regex")
    });
    if re.is_match(lower) {
        out.push("relative temporal".to_string());
    }
    out
}

fn iso_date_mentions(lower: &str) -> Vec<NaiveDate> {
    static ISO_DATE_RE: OnceLock<Regex> = OnceLock::new();
    let re = ISO_DATE_RE.get_or_init(|| {
        Regex::new(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b").expect("valid iso date regex")
    });
    let mut out = Vec::new();
    for cap in re.captures_iter(lower) {
        let year = cap.get(1).and_then(|m| m.as_str().parse::<i32>().ok());
        let month = cap.get(2).and_then(|m| m.as_str().parse::<u32>().ok());
        let day = cap.get(3).and_then(|m| m.as_str().parse::<u32>().ok());
        if let (Some(year), Some(month), Some(day)) = (year, month, day) {
            if let Some(date) = NaiveDate::from_ymd_opt(year, month, day) {
                out.push(date);
            }
        }
    }
    out
}

fn month_day_mentions(lower: &str) -> Vec<(String, String)> {
    static MONTH_DAY_RE: OnceLock<Regex> = OnceLock::new();
    let re = MONTH_DAY_RE.get_or_init(|| {
        Regex::new(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})\b",
        )
        .expect("valid month-day regex")
    });
    let mut out = Vec::new();
    for cap in re.captures_iter(lower) {
        let month = cap.get(1).map(|m| m.as_str()).unwrap_or("").to_string();
        let day = cap.get(2).map(|m| m.as_str()).unwrap_or("").to_string();
        if !month.is_empty() && !day.is_empty() {
            out.push((month, day));
        }
    }
    out
}

fn standalone_weekday_mentions(lower: &str) -> Vec<Weekday> {
    static WEEKDAY_RE: OnceLock<Regex> = OnceLock::new();
    let re = WEEKDAY_RE.get_or_init(|| {
        Regex::new(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b")
            .expect("valid weekday mention regex")
    });
    let mut out = Vec::new();
    for cap in re.captures_iter(lower) {
        if let Some(weekday) = parse_weekday(cap.get(1).map(|m| m.as_str()).unwrap_or("")) {
            out.push(weekday);
        }
    }
    out
}

fn period_patterns(lower: &str, base: NaiveDate) -> Vec<(String, NaiveDate)> {
    let mut out = Vec::new();
    for (prefix, unit, delta) in [
        ("last", "week", -1i32),
        ("this", "week", 0i32),
        ("next", "week", 1i32),
        ("last", "month", -1i32),
        ("this", "month", 0i32),
        ("next", "month", 1i32),
        ("last", "year", -1i32),
        ("this", "year", 0i32),
        ("next", "year", 1i32),
    ] {
        let phrase = format!("{} {}", prefix, unit);
        if lower.contains(&phrase) {
            let resolved = match unit {
                "week" => base + Duration::weeks(delta as i64),
                "month" => shift_months(base, delta),
                "year" => shift_years(base, delta),
                _ => base,
            };
            out.push((phrase, resolved));
        }
    }
    if lower.contains("earlier this week") {
        out.push(("earlier this week".to_string(), current_week_start(base)));
    }
    if lower.contains("later this week") {
        out.push((
            "later this week".to_string(),
            current_week_start(base) + Duration::days(4),
        ));
    }
    if lower.contains("earlier this month") {
        out.push(("earlier this month".to_string(), base));
    }
    if lower.contains("later this month") {
        out.push(("later this month".to_string(), base));
    }
    if lower.contains("earlier this year") {
        out.push(("earlier this year".to_string(), base));
    }
    if lower.contains("later this year") {
        out.push(("later this year".to_string(), base));
    }
    out
}

fn weekend_patterns(lower: &str, base: NaiveDate) -> Vec<(String, NaiveDate)> {
    let mut out = Vec::new();
    for (prefix, delta) in [("last", -1), ("this", 0), ("next", 1)] {
        let phrase = format!("{} weekend", prefix);
        if lower.contains(&phrase) {
            let (sat, sun) = resolve_weekend(base, delta);
            out.push((phrase.clone(), sat));
            out.push((phrase, sun));
        }
    }
    out
}

fn ago_patterns(lower: &str) -> Vec<(u32, String, String)> {
    static AGO_RE: OnceLock<Regex> = OnceLock::new();
    let re = AGO_RE.get_or_init(|| {
        Regex::new(
            r"\b(one|two|three|four|five|six|seven|eight|nine|ten|1|2|3|4|5|6|7|8|9|10)\s+(day|week|month|year)s?\s+ago\b",
        )
            .expect("valid ago regex")
    });
    let mut out = Vec::new();
    for cap in re.captures_iter(lower) {
        let n_token = cap.get(1).map(|m| m.as_str()).unwrap_or("1");
        let n = word_to_num(n_token);
        let unit = cap.get(2).map(|m| m.as_str()).unwrap_or("day").to_string();
        let phrase = format!("{} {} ago", n_token, unit);
        out.push((n, unit, phrase));
    }
    out
}

fn future_patterns(lower: &str) -> Vec<(u32, String, String)> {
    static FUTURE_RE: OnceLock<Regex> = OnceLock::new();
    let re = FUTURE_RE.get_or_init(|| {
        Regex::new(
            r"\b(?:in|from now)\s+(one|two|three|four|five|six|seven|eight|nine|ten|1|2|3|4|5|6|7|8|9|10)\s+(day|week|month|year)s?\b|\b(one|two|three|four|five|six|seven|eight|nine|ten|1|2|3|4|5|6|7|8|9|10)\s+(day|week|month|year)s?\s+from now\b",
        )
        .expect("valid future regex")
    });
    let mut out = Vec::new();
    for cap in re.captures_iter(lower) {
        let n_token = cap
            .get(1)
            .or_else(|| cap.get(3))
            .map(|m| m.as_str())
            .unwrap_or("1");
        let unit = cap
            .get(2)
            .or_else(|| cap.get(4))
            .map(|m| m.as_str())
            .unwrap_or("day")
            .to_string();
        let phrase = format!("in {} {}", n_token, unit);
        out.push((word_to_num(n_token), unit, phrase));
    }
    out
}

fn weekday_patterns(lower: &str) -> Vec<(String, Weekday)> {
    static WEEKDAY_RE: OnceLock<Regex> = OnceLock::new();
    let re = WEEKDAY_RE.get_or_init(|| {
        Regex::new(
            r"\b(last|this|next)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        )
        .expect("valid weekday regex")
    });
    let mut out = Vec::new();
    for cap in re.captures_iter(lower) {
        let prefix = cap.get(1).map(|m| m.as_str()).unwrap_or("last").to_string();
        if let Some(weekday) = parse_weekday(cap.get(2).map(|m| m.as_str()).unwrap_or("")) {
            out.push((prefix, weekday));
        }
    }
    out
}

fn parse_date(input: Option<&str>) -> Option<NaiveDate> {
    let s = input?.trim();
    if s.is_empty() {
        return None;
    }
    NaiveDate::parse_from_str(s, "%Y-%m-%d")
        .ok()
        .or_else(|| NaiveDate::parse_from_str(s, "%Y-%m-%dT%H:%M:%S").ok())
        .or_else(|| NaiveDate::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f").ok())
}

fn query_word_tokens(query: &str) -> Vec<String> {
    static TOKEN_RE: OnceLock<Regex> = OnceLock::new();
    let re =
        TOKEN_RE.get_or_init(|| Regex::new(r"[A-Za-z0-9']+").expect("valid query token regex"));
    re.find_iter(query)
        .map(|m| m.as_str().to_lowercase())
        .collect()
}

fn window_days_for_phrase(phrase: &str) -> i64 {
    let lower = phrase.to_lowercase();
    if lower.contains("year") {
        30
    } else if lower.contains("month") {
        14
    } else if lower.contains("week") || lower.contains("weekend") {
        7
    } else if lower.contains("day")
        || lower.contains("today")
        || lower.contains("yesterday")
        || lower.contains("tomorrow")
        || weekday_patterns(&lower).into_iter().next().is_some()
    {
        2
    } else {
        7
    }
}

fn push_date_tokens(tokens: &mut Vec<String>, label: &str, date: NaiveDate) {
    let month = date.format("%B").to_string().to_lowercase();
    let weekday = date.format("%A").to_string().to_lowercase();
    let day = date.day().to_string();
    tokens.push(format!("{} {}", label, weekday));
    tokens.push(format!("{} {}", label, month));
    tokens.push(format!("{} {}", label, day));
    tokens.push(format!("{} {}", month, weekday));
}

fn push_unique(tokens: &mut Vec<String>, seen: &mut HashSet<String>, token: String) {
    if seen.insert(token.clone()) {
        tokens.push(token);
    }
}

fn parse_weekday(input: &str) -> Option<Weekday> {
    match input {
        "monday" => Some(Weekday::Mon),
        "tuesday" => Some(Weekday::Tue),
        "wednesday" => Some(Weekday::Wed),
        "thursday" => Some(Weekday::Thu),
        "friday" => Some(Weekday::Fri),
        "saturday" => Some(Weekday::Sat),
        "sunday" => Some(Weekday::Sun),
        _ => None,
    }
}

fn weekday_name(day: Weekday) -> &'static str {
    match day {
        Weekday::Mon => "monday",
        Weekday::Tue => "tuesday",
        Weekday::Wed => "wednesday",
        Weekday::Thu => "thursday",
        Weekday::Fri => "friday",
        Weekday::Sat => "saturday",
        Weekday::Sun => "sunday",
    }
}

fn resolve_weekday(base: NaiveDate, prefix: &str, target: Weekday) -> NaiveDate {
    let current_week_start = current_week_start(base);
    let target_offset = target.num_days_from_monday() as i64;
    match prefix {
        "this" => current_week_start + Duration::days(target_offset),
        "next" => current_week_start + Duration::days(7 + target_offset),
        _ => {
            let current = base.weekday().num_days_from_monday() as i64;
            let target = target.num_days_from_monday() as i64;
            let mut delta = (current - target).rem_euclid(7);
            if delta == 0 {
                delta = 7;
            }
            base - Duration::days(delta)
        }
    }
}

fn current_week_start(base: NaiveDate) -> NaiveDate {
    base - Duration::days(base.weekday().num_days_from_monday() as i64)
}

fn resolve_weekend(base: NaiveDate, offset_weeks: i64) -> (NaiveDate, NaiveDate) {
    let week_start = current_week_start(base) + Duration::weeks(offset_weeks);
    (
        week_start + Duration::days(5),
        week_start + Duration::days(6),
    )
}

fn shift_months(base: NaiveDate, months: i32) -> NaiveDate {
    let mut year = base.year();
    let mut month = base.month() as i32 + months;
    while month <= 0 {
        year -= 1;
        month += 12;
    }
    while month > 12 {
        year += 1;
        month -= 12;
    }
    let month_u32 = month as u32;
    let last_day = last_day_of_month(year, month_u32);
    let day = base.day().min(last_day);
    NaiveDate::from_ymd_opt(year, month_u32, day).unwrap_or(base)
}

fn shift_years(base: NaiveDate, years: i32) -> NaiveDate {
    let year = base.year() + years;
    let last_day = last_day_of_month(year, base.month());
    let day = base.day().min(last_day);
    NaiveDate::from_ymd_opt(year, base.month(), day).unwrap_or(base)
}

fn last_day_of_month(year: i32, month: u32) -> u32 {
    let next_month = if month == 12 { 1 } else { month + 1 };
    let next_year = if month == 12 { year + 1 } else { year };
    let first_next = NaiveDate::from_ymd_opt(next_year, next_month, 1).unwrap();
    (first_next - Duration::days(1)).day()
}

fn word_to_num(input: &str) -> u32 {
    match input {
        "one" | "1" => 1,
        "two" | "2" => 2,
        "three" | "3" => 3,
        "four" | "4" => 4,
        "five" | "5" => 5,
        "six" | "6" => 6,
        "seven" | "7" => 7,
        "eight" | "8" => 8,
        "nine" | "9" => 9,
        "ten" | "10" => 10,
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn augments_last_weekday() {
        let out = augment_query_with_temporal_context(
            "Who did I meet with during the lunch last Tuesday?",
            Some("2024-05-10"),
        );
        assert!(out.contains("tuesday") || out.contains("friday"));
    }

    #[test]
    fn augments_relative_ago_phrase() {
        let out =
            augment_query_with_temporal_context("What did I do two weeks ago?", Some("2024-05-10"));
        assert!(out.contains("relative temporal"));
    }

    #[test]
    fn augments_future_phrase() {
        let out =
            augment_query_with_temporal_context("What will I do in 3 weeks?", Some("2024-05-10"));
        assert!(out.contains("in 3 week"));
    }

    #[test]
    fn augments_weekday_prefixes() {
        let out =
            augment_query_with_temporal_context("What happened next Monday?", Some("2024-05-10"));
        assert!(out.contains("next monday"));
    }

    #[test]
    fn augments_weekend_phrase() {
        let out =
            augment_query_with_temporal_context("What did I do last weekend?", Some("2024-05-10"));
        assert!(out.contains("last weekend"));
    }

    #[test]
    fn resolves_relative_temporal_target() {
        let anchor = "2024-05-10";
        let anchor_date = NaiveDate::parse_from_str(anchor, "%Y-%m-%d").unwrap();
        let target = resolve_temporal_target("What happened two weeks ago?", Some(anchor))
            .expect("expected temporal target");
        let delta = target.target_date.signed_duration_since(anchor_date).num_days();
        assert!(
            (-16..=-12).contains(&delta),
            "expected ~14 days before anchor, got delta={delta}"
        );
        assert!(target.window_days > 0);
    }
}
