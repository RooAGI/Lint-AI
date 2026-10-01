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

pub const DEFAULT_RECENCY_HALF_LIFE_DAYS: f32 = 30.0;
pub const DEFAULT_RECENCY_MAX_BOOST: f32 = 0.25;

pub fn recency_boost(
    timestamp: Option<&str>,
    now: DateTime<Utc>,
    half_life_days: f32,
    max_boost: f32,
) -> Option<f32> {
    if half_life_days <= 0.0 || !half_life_days.is_finite() || max_boost <= 0.0 {
        return None;
    }
    let timestamp_date = parse_date(timestamp)?;
    let age_days = now
        .date_naive()
        .signed_duration_since(timestamp_date)
        .num_days()
        .max(0) as f32;
    Some((max_boost * 2.0f32.powf(-age_days / half_life_days)).min(max_boost))
}

pub fn augment_query_with_temporal_context(query: &str, question_date: Option<&str>) -> String {
    let Some(base_date) = parse_date(question_date) else {
        return query.to_string();
    };

    let lower = query.to_lowercase();
    let mut tokens = Vec::new();
    let mut seen = HashSet::new();

    // Chinese pre-layer: each hit contributes its label plus resolved
    // date tokens, mirroring the English markers below.
    for hit in chinese_temporal_hits(query, base_date) {
        push_date_tokens(&mut tokens, &hit.label, hit.date);
    }

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
    // Chinese pre-layer: relative words, explicit 年月日 dates, numeric
    // offsets (三天前), and weekday mentions resolve before the English
    // patterns below.
    if let Some(hit) = chinese_temporal_hits(query, base_date).into_iter().next() {
        return Some(TemporalTarget {
            target_date: hit.date,
            window_days: hit.window_days,
        });
    }

    // Korean pre-layer: native relative-date expressions and explicit
    // `2026년 9월 29일`-style dates, checked before the English patterns
    // (the scripts are disjoint, so order is just convention).
    if let Some(target) = resolve_korean_temporal_target(query, base_date) {
        return Some(target);
    }
    let lower = query.to_lowercase();

    // Spanish relative dates and weekday/month names resolve before the
    // English path (no substring overlap between the two languages).
    if let Some(target) = resolve_spanish_temporal_target(&lower, base_date) {
        return Some(target);
    }

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

    resolve_temporal_target_with_temps(query, base_date)
}

/// Korean temporal pre-layer for [`resolve_temporal_target`].
///
/// Table-driven over native relative-date expressions (`오늘`, `어제`,
/// `지난주`, …) plus a regex for explicit `2026년 9월 29일`-style dates.
/// Shapes mirror the English arms above (same window sizes). Returns
/// `None` when the query carries no Korean temporal expression, letting
/// the English path run unchanged.
fn resolve_korean_temporal_target(query: &str, base_date: NaiveDate) -> Option<TemporalTarget> {
    let day = |offset: i64| TemporalTarget {
        target_date: base_date + Duration::days(offset),
        window_days: 2,
    };
    let week = |offset: i64| TemporalTarget {
        target_date: base_date + Duration::weeks(offset),
        window_days: 7,
    };
    let month = |offset: i32| TemporalTarget {
        target_date: shift_months(base_date, offset),
        window_days: 14,
    };
    let year = |offset: i32| TemporalTarget {
        target_date: shift_years(base_date, offset),
        window_days: 30,
    };

    // Days.
    if query.contains("오늘") {
        return Some(day(0));
    }
    if query.contains("어제") {
        return Some(day(-1));
    }
    if query.contains("내일") {
        return Some(day(1));
    }
    if query.contains("그저께") || query.contains("그제") {
        return Some(day(-2));
    }
    if query.contains("모레") {
        return Some(day(2));
    }
    // Weeks.
    if query.contains("지난주") {
        return Some(week(-1));
    }
    if query.contains("이번주") {
        return Some(week(0));
    }
    if query.contains("다음주") {
        return Some(week(1));
    }
    // Months.
    if query.contains("지난달") {
        return Some(month(-1));
    }
    if query.contains("이번달") {
        return Some(month(0));
    }
    if query.contains("다음달") {
        return Some(month(1));
    }
    // Years.
    if query.contains("작년") {
        return Some(year(-1));
    }
    if query.contains("올해") {
        return Some(year(0));
    }
    if query.contains("내년") {
        return Some(year(1));
    }
    // Explicit `2026년 9월 29일` (also `2026년9월29일` — no spaces).
    if let Some(date) = parse_korean_explicit_date(query) {
        return Some(TemporalTarget {
            target_date: date,
            window_days: 2,
        });
    }
    None
}

/// Parse `YYYY년 M월 D일` (spaces optional) into a [`NaiveDate`].
fn parse_korean_explicit_date(query: &str) -> Option<NaiveDate> {
    static KO_DATE_RE: OnceLock<Regex> = OnceLock::new();
    let re = KO_DATE_RE.get_or_init(|| {
        Regex::new(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일").expect("valid Korean date regex")
    });
    let caps = re.captures(query)?;
    let y: i32 = caps.get(1)?.as_str().parse().ok()?;
    let m: u32 = caps.get(2)?.as_str().parse().ok()?;
    let d: u32 = caps.get(3)?.as_str().parse().ok()?;
    NaiveDate::from_ymd_opt(y, m, d)
}

/// Spanish temporal expressions, resolved before the English path.
/// Table- and regex-driven; returns the same `TemporalTarget` shape the
/// English branches produce. All matching is on the lowercased query.
fn resolve_spanish_temporal_target(lower: &str, base_date: NaiveDate) -> Option<TemporalTarget> {
    // Normalize Spanish number words ("dos" -> "2") so "hace dos semanas"
    // works. Spanish-only normalization is safe here: this layer only
    // probes for Spanish temporal patterns, and English text contains no
    // Spanish number words. (The auto-detecting wrapper would default
    // short queries like "hace dos semanas" to English and skip them.)
    let normalized = text2num::replace_numbers_in_text(lower, &text2num::Language::spanish(), 0.0);
    let lower = normalized.as_str();
    // Whole-word match: "hoy" must not fire inside "hoyuelos".
    let has_word = |word: &str| lower.split(|c: char| !c.is_alphabetic()).any(|w| w == word);

    // Relative days. "anteayer"/"antes de ayer" first: they contain "ayer".
    if has_word("anteayer") || lower.contains("antes de ayer") {
        return Some(TemporalTarget {
            target_date: base_date - Duration::days(2),
            window_days: 2,
        });
    }
    if has_word("ayer") {
        return Some(TemporalTarget {
            target_date: base_date - Duration::days(1),
            window_days: 2,
        });
    }
    if lower.contains("pasado mañana") || lower.contains("pasado manana") {
        return Some(TemporalTarget {
            target_date: base_date + Duration::days(2),
            window_days: 2,
        });
    }
    if standalone_manana(lower) {
        return Some(TemporalTarget {
            target_date: base_date + Duration::days(1),
            window_days: 2,
        });
    }
    if has_word("hoy") {
        return Some(TemporalTarget {
            target_date: base_date,
            window_days: 2,
        });
    }

    // Relative weeks / months / years.
    for (phrase, target_date, window_days) in [
        ("la semana pasada", base_date - Duration::weeks(1), 7),
        ("esta semana", base_date, 7),
        ("la próxima semana", base_date + Duration::weeks(1), 7),
        ("la proxima semana", base_date + Duration::weeks(1), 7),
        ("el mes pasado", shift_months(base_date, -1), 14),
        ("este mes", base_date, 14),
        ("el próximo mes", shift_months(base_date, 1), 14),
        ("el proximo mes", shift_months(base_date, 1), 14),
        ("el año pasado", shift_years(base_date, -1), 30),
        ("el ano pasado", shift_years(base_date, -1), 30),
        ("este año", base_date, 30),
        ("este ano", base_date, 30),
        ("el próximo año", shift_years(base_date, 1), 30),
        ("el proximo año", shift_years(base_date, 1), 30),
    ] {
        if lower.contains(phrase) {
            return Some(TemporalTarget {
                target_date,
                window_days,
            });
        }
    }

    // "hace 3 días" / "hace dos semanas" is digits-only here (number words
    // are normalized to digits upstream by `normalize_number_words`).
    static HACE_RE: OnceLock<Regex> = OnceLock::new();
    let hace_re = HACE_RE.get_or_init(|| {
        Regex::new(r"\bhace\s+(\d+)\s+(días?|dias?|semanas?|meses?|años?|anos?)\b")
            .expect("valid hace regex")
    });
    if let Some(cap) = hace_re.captures(lower) {
        let n: i64 = cap
            .get(1)
            .and_then(|m| m.as_str().parse().ok())
            .unwrap_or(0);
        let unit = cap.get(2).map(|m| m.as_str()).unwrap_or("");
        // Units arrive accented ("días", "años"); match both forms.
        let (target_date, window_days) = if unit.starts_with("día") || unit.starts_with("dia") {
            (base_date - Duration::days(n), 2)
        } else if unit.starts_with("semana") {
            (base_date - Duration::weeks(n), 7)
        } else if unit.starts_with("mes") {
            (shift_months(base_date, -(n as i32)), 14)
        } else {
            (shift_years(base_date, -(n as i32)), 30)
        };
        return Some(TemporalTarget {
            target_date,
            window_days,
        });
    }

    // "en 3 días" (future). The full phrase is required — bare "en" is
    // far too common to trigger on.
    static EN_RE: OnceLock<Regex> = OnceLock::new();
    let en_re = EN_RE.get_or_init(|| {
        Regex::new(r"\ben\s+(\d+)\s+(días?|dias?|semanas?|meses?|años?|anos?)\b")
            .expect("valid en-future regex")
    });
    if let Some(cap) = en_re.captures(lower) {
        let n: i64 = cap
            .get(1)
            .and_then(|m| m.as_str().parse().ok())
            .unwrap_or(0);
        let unit = cap.get(2).map(|m| m.as_str()).unwrap_or("");
        let (target_date, window_days) = if unit.starts_with("día") || unit.starts_with("dia") {
            (base_date + Duration::days(n), 2)
        } else if unit.starts_with("semana") {
            (base_date + Duration::weeks(n), 7)
        } else if unit.starts_with("mes") {
            (shift_months(base_date, n as i32), 14)
        } else {
            (shift_years(base_date, n as i32), 30)
        };
        return Some(TemporalTarget {
            target_date,
            window_days,
        });
    }

    // Weekday mentions: "el lunes", "este martes", "el próximo miércoles",
    // "el viernes pasado". Prefix maps onto `resolve_weekday`'s
    // this/next/last contract.
    const ES_WEEKDAYS: &[(&str, Weekday)] = &[
        ("lunes", Weekday::Mon),
        ("martes", Weekday::Tue),
        ("miércoles", Weekday::Wed),
        ("miercoles", Weekday::Wed),
        ("jueves", Weekday::Thu),
        ("viernes", Weekday::Fri),
        ("sábado", Weekday::Sat),
        ("sabado", Weekday::Sat),
        ("domingo", Weekday::Sun),
    ];
    for (name, weekday) in ES_WEEKDAYS {
        if has_word(name) {
            let prefix = if lower.contains(&format!("próximo {name}"))
                || lower.contains(&format!("proximo {name}"))
                || lower.contains(&format!("{name} próximo"))
                || lower.contains(&format!("{name} proximo"))
            {
                "next"
            } else if lower.contains(&format!("{name} pasado"))
                || lower.contains(&format!("pasado {name}"))
            {
                "last"
            } else {
                "this"
            };
            return Some(TemporalTarget {
                target_date: resolve_weekday(base_date, prefix, *weekday),
                window_days: 2,
            });
        }
    }

    // "29 de septiembre de 2026".
    static ES_DATE_RE: OnceLock<Regex> = OnceLock::new();
    let es_date_re = ES_DATE_RE.get_or_init(|| {
        Regex::new(
            r"\b(\d{1,2})\s+de\s+(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)\s+de\s+(\d{4})\b",
        )
        .expect("valid Spanish date regex")
    });
    if let Some(cap) = es_date_re.captures(lower) {
        let day: u32 = cap
            .get(1)
            .and_then(|m| m.as_str().parse().ok())
            .unwrap_or(0);
        let month: u32 = match cap.get(2).map(|m| m.as_str()).unwrap_or("") {
            "enero" => 1,
            "febrero" => 2,
            "marzo" => 3,
            "abril" => 4,
            "mayo" => 5,
            "junio" => 6,
            "julio" => 7,
            "agosto" => 8,
            "septiembre" | "setiembre" => 9,
            "octubre" => 10,
            "noviembre" => 11,
            _ => 12,
        };
        let year: i32 = cap
            .get(3)
            .and_then(|m| m.as_str().parse().ok())
            .unwrap_or(0);
        if let Some(date) = NaiveDate::from_ymd_opt(year, month, day) {
            return Some(TemporalTarget {
                target_date: date,
                window_days: 2,
            });
        }
    }

    // "29/09/2026" and "29-09-2026" (Spanish day-first convention; ISO
    // yyyy-mm-dd cannot match: its first group is 4 digits).
    static DMY_RE: OnceLock<Regex> = OnceLock::new();
    let dmy_re = DMY_RE.get_or_init(|| {
        Regex::new(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b").expect("valid dmy regex")
    });
    if let Some(cap) = dmy_re.captures(lower) {
        let day: u32 = cap
            .get(1)
            .and_then(|m| m.as_str().parse().ok())
            .unwrap_or(0);
        let month: u32 = cap
            .get(2)
            .and_then(|m| m.as_str().parse().ok())
            .unwrap_or(0);
        let year: i32 = cap
            .get(3)
            .and_then(|m| m.as_str().parse().ok())
            .unwrap_or(0);
        if let Some(date) = NaiveDate::from_ymd_opt(year, month, day) {
            return Some(TemporalTarget {
                target_date: date,
                window_days: 2,
            });
        }
    }

    None
}

/// True when "mañana"/"manana" is used as "tomorrow" rather than "morning".
/// "por la mañana", "de la mañana", "esta/esa/la mañana" mean "in the
/// morning" — only a standalone occurrence counts as tomorrow.
fn standalone_manana(lower: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"\b(mañana|manana)\b").expect("valid mañana regex"));
    for m in re.find_iter(lower) {
        let before = lower[..m.start()].trim_end();
        let is_morning = ["por la", "de la", "esta", "esa", "la"]
            .iter()
            .any(|p| before.ends_with(p));
        if !is_morning {
            return true;
        }
    }
    false
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

    let (_, _, parsed_date, phrase) = best?;

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
        // Chinese pre-layer: explicit dates, weekday mentions, numeric
        // offsets, and relative words. The anchor here is "today" (index
        // time), matching how English relative markers are treated.
        let index_today = DateTime::<Utc>::from(SystemTime::now()).date_naive();
        for hit in chinese_temporal_hits(text, index_today) {
            push_unique(&mut terms, &mut seen, format!("temporal {}", hit.label));
            push_date_tokens(&mut terms, &hit.label, hit.date);
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
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december|enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)\s+(\d{1,2})\b",
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
        Regex::new(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|lunes|martes|miércoles|miercoles|jueves|viernes|sábado|sabado|domingo)\b")
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
            r"\b(a couple of|couple of|a|an|one|two|three|four|five|six|seven|eight|nine|ten|1|2|3|4|5|6|7|8|9|10)\s+(day|week|month|year)s?\s+ago\b",
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
        .or_else(|| {
            DateTime::parse_from_rfc3339(s)
                .ok()
                .map(|date| date.date_naive())
        })
        .or_else(|| NaiveDate::parse_from_str(s, "%Y-%m-%dT%H:%M:%S").ok())
        .or_else(|| NaiveDate::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f").ok())
        .or_else(|| {
            s.parse::<i64>()
                .ok()
                .and_then(|seconds| DateTime::<Utc>::from_timestamp(seconds, 0))
                .map(|date| date.date_naive())
        })
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
        "monday" | "lunes" => Some(Weekday::Mon),
        "tuesday" | "martes" => Some(Weekday::Tue),
        "wednesday" | "miércoles" | "miercoles" => Some(Weekday::Wed),
        "thursday" | "jueves" => Some(Weekday::Thu),
        "friday" | "viernes" => Some(Weekday::Fri),
        "saturday" | "sábado" | "sabado" => Some(Weekday::Sat),
        "sunday" | "domingo" => Some(Weekday::Sun),
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
    // Bound the magnitude first: input numerals are unbounded (\d+), and
    // without this `base.month() as i32 + months` can overflow i32 while
    // the normalization loops below spin ~2^31/12 iterations on
    // adversarial input. 3.1M months (~258k years) still resolves inside
    // chrono's range; truly out-of-range results fall back to `base`.
    const MAX_MONTHS: i32 = 3_100_000; // ~258k years
    let months = months.clamp(-MAX_MONTHS, MAX_MONTHS);
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
    let last_day = match last_day_of_month(year, month_u32) {
        Some(d) => d,
        None => return base,
    };
    let day = base.day().min(last_day);
    NaiveDate::from_ymd_opt(year, month_u32, day).unwrap_or(base)
}

fn shift_years(base: NaiveDate, years: i32) -> NaiveDate {
    // saturating_add: input numerals are unbounded, so `base.year() + years`
    // could overflow i32 (panic in debug). Out-of-range years resolve to
    // `base` below instead of panicking.
    let year = base.year().saturating_add(years);
    let last_day = match last_day_of_month(year, base.month()) {
        Some(d) => d,
        None => return base,
    };
    let day = base.day().min(last_day);
    NaiveDate::from_ymd_opt(year, base.month(), day).unwrap_or(base)
}

/// Add `days` (possibly huge — input numerals are unbounded) to `base`
/// without panicking. The magnitude is clamped to just inside chrono's
/// representable range (~260k years); the result stays directionally
/// correct (far past / far future) instead of killing the process.
fn shift_days(base: NaiveDate, days: i64) -> NaiveDate {
    const MAX_DAYS: i64 = 95_000_000; // ~260k years, just inside chrono's ±262143-year range
    let days = days.clamp(-MAX_DAYS, MAX_DAYS);
    base.checked_add_signed(Duration::days(days)).unwrap_or(base)
}

fn last_day_of_month(year: i32, month: u32) -> Option<u32> {
    let next_month = if month == 12 { 1 } else { month + 1 };
    let next_year = if month == 12 { year + 1 } else { year };
    // Fallible: `from_ymd_opt` returns None outside chrono's year range
    // (e.g. "百万年前" -> year -997974). Callers fall back to `base`;
    // this must never panic on input text.
    let first_next = NaiveDate::from_ymd_opt(next_year, next_month, 1)?;
    Some((first_next - Duration::days(1)).day())
}

fn word_to_num(input: &str) -> u32 {
    match input {
        "a" | "an" | "one" | "1" => 1,
        "a couple of" | "couple of" => 2,
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

// ---------------------------------------------------------------------------
// Chinese temporal expressions.
//
// Pre-layer checked before the English patterns in `resolve_temporal_target`,
// `augment_query_with_temporal_context`, and `extract_temporal_terms`.
// Queries pass through `crate::lang::normalize_chinese_numbers` first, so
// 三天前 and 二〇二六年九月二十九日 reach the regexes as digit forms.
// ---------------------------------------------------------------------------

/// One Chinese temporal expression resolved against the anchor date.
struct ChineseTemporalHit {
    label: String,
    date: NaiveDate,
    window_days: i64,
}

/// Relative day/week/month/year words, checked longest-first.
/// Each push records (label, resolved date, window_days).
fn chinese_relative_hits(query: &str, base: NaiveDate) -> Vec<ChineseTemporalHit> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut push = |label: &str, date: NaiveDate, window_days: i64| {
        if seen.insert(label.to_string()) {
            out.push(ChineseTemporalHit {
                label: label.to_string(),
                date,
                window_days,
            });
        }
    };

    // Day scale. Longest-match-first: "大前天" contains "前天" (and
    // "大后天" contains "后天") as a substring, so the doubled form must
    // be checked — and stripped — before the base form to avoid
    // mistagging it (e.g. memory "上上周去杭州" tagged as "上周").
    let q_day = query.replace("大前天", "").replace("大后天", "");
    if query.contains("大前天") {
        push("大前天", base - Duration::days(3), 2);
    }
    if query.contains("大后天") {
        push("大后天", base + Duration::days(3), 2);
    }
    if q_day.contains("前天") {
        push("前天", base - Duration::days(2), 2);
    }
    if q_day.contains("昨天") {
        push("昨天", base - Duration::days(1), 2);
    }
    if q_day.contains("今天") {
        push("今天", base, 2);
    }
    if q_day.contains("明天") {
        push("明天", base + Duration::days(1), 2);
    }
    if q_day.contains("后天") {
        push("后天", base + Duration::days(2), 2);
    }
    // Week scale. Longest-match-first: "上上周" contains "上周".
    let q_week = query.replace("上上周", "").replace("下下周", "");
    if query.contains("上上周") {
        push("上上周", base - Duration::weeks(2), 7);
    }
    if query.contains("下下周") {
        push("下下周", base + Duration::weeks(2), 7);
    }
    if q_week.contains("上周") {
        push("上周", base - Duration::weeks(1), 7);
    }
    if q_week.contains("本周") || q_week.contains("这周") {
        push("本周", base, 7);
    }
    if q_week.contains("下周") {
        push("下周", base + Duration::weeks(1), 7);
    }
    // Month scale. Longest-match-first: "上上个月" contains "上个月".
    let q_month = query.replace("上上个月", "").replace("下下个月", "");
    if query.contains("上上个月") {
        push("上上月", shift_months(base, -2), 14);
    }
    if query.contains("下下个月") {
        push("下下月", shift_months(base, 2), 14);
    }
    if q_month.contains("上个月") || q_month.contains("上月") {
        push("上月", shift_months(base, -1), 14);
    }
    if q_month.contains("这个月") || q_month.contains("本月") {
        push("本月", base, 14);
    }
    if q_month.contains("下个月") || q_month.contains("下月") {
        push("下月", shift_months(base, 1), 14);
    }
    // Year scale.
    if query.contains("去年") {
        push("去年", shift_years(base, -1), 30);
    }
    if query.contains("今年") {
        push("今年", base, 30);
    }
    if query.contains("明年") {
        push("明年", shift_years(base, 1), 30);
    }
    out
}

/// Explicit dates: 2026年9月29日 / 2026年9月 / 9月29日 (anchor year).
/// `normalized` must already have Chinese numerals converted to digits.
fn chinese_explicit_date_hits(normalized: &str, base: NaiveDate) -> Vec<ChineseTemporalHit> {
    static YMD_RE: OnceLock<Regex> = OnceLock::new();
    static YM_RE: OnceLock<Regex> = OnceLock::new();
    static MD_RE: OnceLock<Regex> = OnceLock::new();
    let ymd_re =
        YMD_RE.get_or_init(|| Regex::new(r"(\d{4})年(\d{1,2})月(\d{1,2})[日号]?").unwrap());
    let ym_re = YM_RE.get_or_init(|| Regex::new(r"(\d{4})年(\d{1,2})月").unwrap());
    let md_re = MD_RE.get_or_init(|| Regex::new(r"(\d{1,2})月(\d{1,2})[日号]").unwrap());

    let mut out = Vec::new();
    let mut ymd_spans: Vec<(usize, usize)> = Vec::new();
    for cap in ymd_re.captures_iter(normalized) {
        let (Some(y), Some(m), Some(d)) = (
            cap.get(1).and_then(|x| x.as_str().parse::<i32>().ok()),
            cap.get(2).and_then(|x| x.as_str().parse::<u32>().ok()),
            cap.get(3).and_then(|x| x.as_str().parse::<u32>().ok()),
        ) else {
            continue;
        };
        if let Some(date) = NaiveDate::from_ymd_opt(y, m, d) {
            let m0 = cap.get(0).unwrap();
            ymd_spans.push((m0.start(), m0.end()));
            out.push(ChineseTemporalHit {
                label: m0.as_str().to_string(),
                date,
                window_days: 2,
            });
        }
    }
    for cap in ym_re.captures_iter(normalized) {
        let m0 = cap.get(0).unwrap();
        // Skip the 年月 prefix of an already-matched 年月日.
        if ymd_spans
            .iter()
            .any(|(s, e)| *s <= m0.start() && m0.end() <= *e)
        {
            continue;
        }
        let (Some(y), Some(m)) = (
            cap.get(1).and_then(|x| x.as_str().parse::<i32>().ok()),
            cap.get(2).and_then(|x| x.as_str().parse::<u32>().ok()),
        ) else {
            continue;
        };
        if let Some(date) = NaiveDate::from_ymd_opt(y, m, 1) {
            out.push(ChineseTemporalHit {
                label: m0.as_str().to_string(),
                date,
                window_days: 14,
            });
        }
    }
    for cap in md_re.captures_iter(normalized) {
        let m0 = cap.get(0).unwrap();
        // Skip a 月日 inside an already-matched 年月日.
        if ymd_spans
            .iter()
            .any(|(s, e)| *s <= m0.start() && m0.end() <= *e)
        {
            continue;
        }
        let (Some(m), Some(d)) = (
            cap.get(1).and_then(|x| x.as_str().parse::<u32>().ok()),
            cap.get(2).and_then(|x| x.as_str().parse::<u32>().ok()),
        ) else {
            continue;
        };
        if let Some(date) = NaiveDate::from_ymd_opt(base.year(), m, d) {
            out.push(ChineseTemporalHit {
                label: m0.as_str().to_string(),
                date,
                window_days: 2,
            });
        }
    }
    out
}

/// Numeric relative offsets: 三天前 / 两周后 / 3个月前 / 5年后.
/// `normalized` must already have Chinese numerals converted to digits.
fn chinese_offset_hits(normalized: &str, base: NaiveDate) -> Vec<ChineseTemporalHit> {
    static OFFSET_RE: OnceLock<Regex> = OnceLock::new();
    let re = OFFSET_RE
        .get_or_init(|| Regex::new(r"(\d+)(天|日|个星期|星期|周|个月|月|年)(前|后)").unwrap());
    let mut out = Vec::new();
    for cap in re.captures_iter(normalized) {
        let n: i64 = cap
            .get(1)
            .and_then(|x| x.as_str().parse().ok())
            .unwrap_or(0);
        if n == 0 {
            continue;
        }
        // Input numerals are unbounded (\d+): clamp once, up front, to a
        // magnitude date arithmetic can represent. This bounds every
        // downstream multiplication and cast (days, weeks, months, years).
        // Unrepresentable offsets resolve to `base` in the shift helpers.
        let n = n.min(95_000_000);
        let unit = cap.get(2).map(|x| x.as_str()).unwrap_or("");
        let future = cap.get(3).map(|x| x.as_str() == "后").unwrap_or(false);
        let sign = if future { 1 } else { -1 };
        let (date, window_days) = match unit {
            "天" | "日" => (shift_days(base, sign * n), 2),
            "星期" | "个星期" | "周" => (shift_days(base, sign * n * 7), 7),
            "个月" | "月" => (shift_months(base, (sign * n) as i32), 14),
            "年" => (shift_years(base, (sign * n) as i32), 30),
            _ => continue,
        };
        out.push(ChineseTemporalHit {
            label: cap.get(0).unwrap().as_str().to_string(),
            date,
            window_days,
        });
    }
    out
}

/// Chinese weekday mentions: 上/这/下 + 星期|周|礼拜 + 一..日.
/// Bare 星期三 resolves to the coming one (matching "this" semantics).
fn chinese_weekday_hits(query: &str, base: NaiveDate) -> Vec<ChineseTemporalHit> {
    static WD_RE: OnceLock<Regex> = OnceLock::new();
    let re =
        WD_RE.get_or_init(|| Regex::new(r"(上|这|下)?(星期|周|礼拜)([一二三四五六日天])").unwrap());
    let mut out = Vec::new();
    for cap in re.captures_iter(query) {
        let day_char = cap.get(3).map(|x| x.as_str()).unwrap_or("");
        let weekday = match day_char {
            "一" => Weekday::Mon,
            "二" => Weekday::Tue,
            "三" => Weekday::Wed,
            "四" => Weekday::Thu,
            "五" => Weekday::Fri,
            "六" => Weekday::Sat,
            "日" | "天" => Weekday::Sun,
            _ => continue,
        };
        let prefix = match cap.get(1).map(|x| x.as_str()) {
            Some("上") => "last",
            Some("下") => "next",
            _ => "this",
        };
        out.push(ChineseTemporalHit {
            label: cap.get(0).unwrap().as_str().to_string(),
            date: resolve_weekday(base, prefix, weekday),
            window_days: 2,
        });
    }
    out
}

/// All Chinese temporal hits for `query` against `base`, in priority order:
/// explicit dates, weekday mentions, numeric offsets, relative words.
fn chinese_temporal_hits(query: &str, base: NaiveDate) -> Vec<ChineseTemporalHit> {
    let normalized = crate::lang::normalize_chinese_numbers(query);
    let mut out = Vec::new();
    out.extend(chinese_explicit_date_hits(&normalized, base));
    out.extend(chinese_weekday_hits(query, base));
    out.extend(chinese_offset_hits(&normalized, base));
    out.extend(chinese_relative_hits(query, base));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn anchor() -> &'static str {
        // A fixed Tuesday: 2026-09-29.
        "2026-09-29"
    }

    #[test]
    fn chinese_relative_day_words_resolve() {
        let base = NaiveDate::from_ymd_opt(2026, 9, 29).unwrap();
        let t = |q: &str| {
            resolve_temporal_target(q, Some(anchor()))
                .unwrap()
                .target_date
        };
        assert_eq!(t("我昨天见了王老师"), base - Duration::days(1));
        assert_eq!(t("我明天要去北京"), base + Duration::days(1));
        assert_eq!(t("今天天气很好"), base);
        assert_eq!(t("前天买的菜"), base - Duration::days(2));
        assert_eq!(t("后天出发"), base + Duration::days(2));
    }

    #[test]
    fn chinese_relative_week_month_year_resolve() {
        let t = |q: &str| resolve_temporal_target(q, Some(anchor())).unwrap();
        let hit = t("上周我们开会了");
        assert_eq!(
            hit.target_date,
            NaiveDate::from_ymd_opt(2026, 9, 22).unwrap()
        );
        assert_eq!(hit.window_days, 7);
        let hit = t("下周要交报告");
        assert_eq!(
            hit.target_date,
            NaiveDate::from_ymd_opt(2026, 10, 6).unwrap()
        );
        let hit = t("上个月去了上海");
        assert_eq!(
            hit.target_date,
            NaiveDate::from_ymd_opt(2026, 8, 29).unwrap()
        );
        assert_eq!(hit.window_days, 14);
        let hit = t("去年毕业的");
        assert_eq!(
            hit.target_date,
            NaiveDate::from_ymd_opt(2025, 9, 29).unwrap()
        );
        assert_eq!(hit.window_days, 30);
        let hit = t("今年的目标");
        assert_eq!(
            hit.target_date,
            NaiveDate::from_ymd_opt(2026, 9, 29).unwrap()
        );
    }

    #[test]
    fn chinese_doubled_temporal_forms_longest_match_first() {
        // Regression: "上上周" contains "上周" as a substring and was
        // mistagged as last week. Doubled forms must resolve to their own
        // offset, never the base form's.
        let t = |q: &str| resolve_temporal_target(q, Some(anchor())).unwrap();
        let hit = t("上上周去杭州出差");
        assert_eq!(
            hit.target_date,
            NaiveDate::from_ymd_opt(2026, 9, 15).unwrap()
        );
        assert_eq!(hit.window_days, 7);
        let hit = t("下下周要去北京");
        assert_eq!(
            hit.target_date,
            NaiveDate::from_ymd_opt(2026, 10, 13).unwrap()
        );
        let hit = t("大前天买的菜");
        assert_eq!(
            hit.target_date,
            NaiveDate::from_ymd_opt(2026, 9, 26).unwrap()
        );
        let hit = t("大后天出发");
        assert_eq!(
            hit.target_date,
            NaiveDate::from_ymd_opt(2026, 10, 2).unwrap()
        );
        let hit = t("上上个月去了南京");
        assert_eq!(
            hit.target_date,
            NaiveDate::from_ymd_opt(2026, 7, 29).unwrap()
        );
        assert_eq!(hit.window_days, 14);
        let hit = t("下下个月交房");
        assert_eq!(
            hit.target_date,
            NaiveDate::from_ymd_opt(2026, 11, 29).unwrap()
        );
        // Base forms still resolve when the doubled form is absent.
        let hit = t("上周我们开会了");
        assert_eq!(
            hit.target_date,
            NaiveDate::from_ymd_opt(2026, 9, 22).unwrap()
        );
        let hit = t("前天买的菜");
        assert_eq!(
            hit.target_date,
            NaiveDate::from_ymd_opt(2026, 9, 27).unwrap()
        );
    }

    #[test]
    fn chinese_explicit_dates_parse() {
        let t = |q: &str| {
            resolve_temporal_target(q, Some(anchor()))
                .unwrap()
                .target_date
        };
        assert_eq!(
            t("会议在2026年9月29日举行"),
            NaiveDate::from_ymd_opt(2026, 9, 29).unwrap()
        );
        // Chinese numerals are normalized before matching.
        assert_eq!(
            t("会议在二〇二六年九月二十九日举行"),
            NaiveDate::from_ymd_opt(2026, 9, 29).unwrap()
        );
        // Month-day without year uses the anchor year.
        assert_eq!(
            t("10月1日放假"),
            NaiveDate::from_ymd_opt(2026, 10, 1).unwrap()
        );
    }

    #[test]
    fn chinese_numeric_offsets_resolve() {
        let base = NaiveDate::from_ymd_opt(2026, 9, 29).unwrap();
        let t = |q: &str| {
            resolve_temporal_target(q, Some(anchor()))
                .unwrap()
                .target_date
        };
        assert_eq!(t("三天前买的书"), base - Duration::days(3));
        assert_eq!(t("两周后考试"), base + Duration::weeks(2));
        assert_eq!(t("三个月前入职"), shift_months(base, -3));
    }

    #[test]
    fn chinese_weekday_mentions_resolve() {
        // Anchor 2026-09-29 is a Tuesday.
        let t = |q: &str| {
            resolve_temporal_target(q, Some(anchor()))
                .unwrap()
                .target_date
        };
        assert_eq!(
            t("上星期一开会"),
            NaiveDate::from_ymd_opt(2026, 9, 28).unwrap()
        );
        assert_eq!(
            t("下周三交报告"),
            NaiveDate::from_ymd_opt(2026, 10, 7).unwrap()
        );
    }

    #[test]
    fn chinese_augment_adds_date_tokens() {
        let out = augment_query_with_temporal_context("我昨天见了谁", Some(anchor()));
        assert!(out.contains("昨天"), "expected Chinese label, got {out:?}");
        assert!(
            out.contains("monday"),
            "expected resolved weekday, got {out:?}"
        );
    }

    #[test]
    fn chinese_extract_temporal_terms_finds_dates() {
        let terms = extract_temporal_terms(None, "会议在2026年9月29日举行", &[]);
        assert!(
            terms.iter().any(|t| t.contains("2026年9月29日")),
            "expected the explicit date, got {terms:?}"
        );
    }

    #[test]
    fn korean_relative_dates() {
        let base = NaiveDate::from_ymd_opt(2026, 9, 29).unwrap();
        let t = resolve_temporal_target("오늘 뭐 했어?", None).unwrap();
        // base_date falls back to today when anchor is None; just check
        // the shape resolves (window 2) and the date is a real date.
        assert_eq!(t.window_days, 2);

        let t = resolve_korean_temporal_target("어제 만난 사람", base).unwrap();
        assert_eq!(t.target_date, NaiveDate::from_ymd_opt(2026, 9, 28).unwrap());
        assert_eq!(t.window_days, 2);

        let t = resolve_korean_temporal_target("내일 일정", base).unwrap();
        assert_eq!(t.target_date, NaiveDate::from_ymd_opt(2026, 9, 30).unwrap());

        let t = resolve_korean_temporal_target("지난주 회의", base).unwrap();
        assert_eq!(t.target_date, NaiveDate::from_ymd_opt(2026, 9, 22).unwrap());
        assert_eq!(t.window_days, 7);

        let t = resolve_korean_temporal_target("다음달 여행", base).unwrap();
        assert_eq!(t.target_date, NaiveDate::from_ymd_opt(2026, 10, 29).unwrap());
        assert_eq!(t.window_days, 14);

        let t = resolve_korean_temporal_target("작년 여름", base).unwrap();
        assert_eq!(t.target_date, NaiveDate::from_ymd_opt(2025, 9, 29).unwrap());
        assert_eq!(t.window_days, 30);
    }

    #[test]
    fn korean_explicit_date() {
        let base = NaiveDate::from_ymd_opt(2026, 9, 29).unwrap();
        let t = resolve_korean_temporal_target("2026년 9월 29일에 뭐 했어?", base).unwrap();
        assert_eq!(t.target_date, base);
        let t = resolve_korean_temporal_target("2024년5월1일 회의", base).unwrap();
        assert_eq!(t.target_date, NaiveDate::from_ymd_opt(2024, 5, 1).unwrap());
        // Non-temporal Korean query: no target.
        assert!(resolve_korean_temporal_target("학교에 갔다", base).is_none());
    }

    #[test]
    fn spanish_relative_days() {
        let base = NaiveDate::from_ymd_opt(2026, 9, 29).unwrap(); // a Tuesday
        let t = |q: &str| resolve_spanish_temporal_target(&q.to_lowercase(), base).unwrap();
        assert_eq!(t("¿Qué hice ayer?").target_date.to_string(), "2026-09-28");
        assert_eq!(t("¿Qué hago hoy?").target_date.to_string(), "2026-09-29");
        assert_eq!(t("Nos vemos mañana").target_date.to_string(), "2026-09-30");
        assert_eq!(t("Llegó anteayer").target_date.to_string(), "2026-09-27");
        assert_eq!(
            t("Sale pasado mañana").target_date.to_string(),
            "2026-10-01"
        );
    }

    #[test]
    fn spanish_manana_morning_is_not_tomorrow() {
        // "por la mañana" = in the morning, not tomorrow.
        let base = NaiveDate::from_ymd_opt(2026, 9, 29).unwrap();
        assert!(
            resolve_spanish_temporal_target(
                &"¿Qué hiciste ayer por la mañana?".to_lowercase(),
                base
            )
            .unwrap()
            .target_date
            .to_string()
                == "2026-09-28"
        );
        // No day word at all: "mañana" alone in a morning phrase is not
        // tomorrow either (falls through to None here; English path may
        // still match "morning"-less queries).
        assert!(
            resolve_spanish_temporal_target(&"desayunamos por la mañana".to_lowercase(), base)
                .is_none()
        );
    }

    #[test]
    fn spanish_weeks_months_years() {
        let base = NaiveDate::from_ymd_opt(2026, 9, 29).unwrap();
        let t = |q: &str| resolve_spanish_temporal_target(&q.to_lowercase(), base).unwrap();
        assert_eq!(t("la semana pasada").target_date.to_string(), "2026-09-22");
        assert_eq!(t("esta semana").target_date.to_string(), "2026-09-29");
        assert_eq!(t("la próxima semana").target_date.to_string(), "2026-10-06");
        assert_eq!(t("el mes pasado").target_date.to_string(), "2026-08-29");
        assert_eq!(t("el año pasado").target_date.to_string(), "2025-09-29");
    }

    #[test]
    fn spanish_hace_and_en() {
        let base = NaiveDate::from_ymd_opt(2026, 9, 29).unwrap();
        let t = |q: &str| resolve_spanish_temporal_target(&q.to_lowercase(), base).unwrap();
        assert_eq!(t("hace 3 días").target_date.to_string(), "2026-09-26");
        // Number words are normalized upstream ("dos" -> "2").
        assert_eq!(t("hace dos semanas").target_date.to_string(), "2026-09-15");
        assert_eq!(t("hace 2 semanas").target_date.to_string(), "2026-09-15");
        assert_eq!(t("en 5 días").target_date.to_string(), "2026-10-04");
    }

    #[test]
    fn spanish_absolute_dates() {
        let base = NaiveDate::from_ymd_opt(2026, 9, 29).unwrap();
        let t = |q: &str| resolve_spanish_temporal_target(&q.to_lowercase(), base).unwrap();
        assert_eq!(
            t("el 29 de septiembre de 2026").target_date.to_string(),
            "2026-09-29"
        );
        assert_eq!(t("29/09/2026").target_date.to_string(), "2026-09-29");
        assert_eq!(t("29-09-2026").target_date.to_string(), "2026-09-29");
        // ISO yyyy-mm-dd must NOT match the day-first pattern.
        assert!(resolve_spanish_temporal_target(&"2026-09-29".to_lowercase(), base).is_none());
    }

    #[test]
    fn spanish_weekdays() {
        // 2026-09-29 is a Tuesday. "el lunes" -> most recent Monday.
        let base = NaiveDate::from_ymd_opt(2026, 9, 29).unwrap();
        let t = |q: &str| resolve_spanish_temporal_target(&q.to_lowercase(), base).unwrap();
        assert_eq!(t("el lunes").target_date.to_string(), "2026-09-28");
        assert_eq!(
            t("el próximo viernes").target_date.to_string(),
            "2026-10-09"
        );
    }

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
        let delta = target
            .target_date
            .signed_duration_since(anchor_date)
            .num_days();
        assert!(
            (-16..=-12).contains(&delta),
            "expected ~14 days before anchor, got delta={delta}"
        );
        assert!(target.window_days > 0);
    }

    #[test]
    fn resolves_last_weekday_against_explicit_anchor() {
        let target = resolve_temporal_target("Who did I meet last Tuesday?", Some("2024-05-10"))
            .expect("expected temporal target");
        assert_eq!(
            target.target_date,
            NaiveDate::from_ymd_opt(2024, 5, 7).unwrap()
        );
    }

    #[test]
    fn resolves_article_and_couple_ago_phrases() {
        let one_week = resolve_temporal_target("What happened a week ago?", Some("2024-05-10"))
            .expect("expected one-week target");
        assert_eq!(
            one_week.target_date,
            NaiveDate::from_ymd_opt(2024, 5, 3).unwrap()
        );

        let couple_days =
            resolve_temporal_target("What did I cook a couple of days ago?", Some("2024-05-10"))
                .expect("expected two-day target");
        assert_eq!(
            couple_days.target_date,
            NaiveDate::from_ymd_opt(2024, 5, 8).unwrap()
        );
    }

    #[test]
    fn reveals_bug_rfc3339_timestamp_with_timezone_is_not_parsed() {
        assert_eq!(
            parse_temporal_date(Some("2024-05-10T14:30:00+00:00")),
            Some(NaiveDate::from_ymd_opt(2024, 5, 10).unwrap())
        );
    }

    #[test]
    fn reveals_bug_recency_decay_does_not_prioritize_recent_memory() {
        let now = DateTime::parse_from_rfc3339("2024-07-01T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let boost = recency_boost(
            Some("2024-06-01T12:00:00Z"),
            now,
            DEFAULT_RECENCY_HALF_LIFE_DAYS,
            DEFAULT_RECENCY_MAX_BOOST,
        )
        .expect("timestamp should receive a recency boost");
        assert!((boost - 0.125).abs() < 0.001, "boost={boost}");
    }

    #[test]
    fn extreme_temporal_offsets_never_panic() {
        // Regression: "百万年前" (a million years ago, from MIRACL corpus
        // docs 6769570#0 / 2736869#0) panicked `last_day_of_month` via
        // `NaiveDate::from_ymd_opt(...).unwrap()` on year -997974, killing
        // the whole indexing run. Out-of-range offsets now resolve to
        // `base` instead of panicking.
        let base = NaiveDate::from_ymd_opt(2026, 9, 30).unwrap();
        // Out-of-chrono-range magnitudes resolve to base, never panic.
        assert_eq!(shift_years(base, -1_000_000), base);
        assert_eq!(shift_years(base, 1_000_000), base);
        assert_eq!(shift_years(base, i32::MAX), base);
        assert_eq!(shift_years(base, i32::MIN), base);
        assert!(shift_days(base, i64::MAX) > base);
        assert!(shift_days(base, i64::MIN) < base);
        assert_eq!(last_day_of_month(-997_974, 9), None);
        assert_eq!(last_day_of_month(2026, 9), Some(30));
        assert_eq!(last_day_of_month(2026, 2), Some(28));
        // Clamped magnitudes stay representable: no panic, and the
        // direction is preserved (far past / far future).
        assert!(shift_months(base, -1_000_000_000) < base);
        assert!(shift_months(base, 1_000_000_000) > base);
        assert!(shift_months(base, i32::MIN) < base);
    }

    #[test]
    fn chinese_geological_time_does_not_panic() {
        // The exact trigger from the MIRACL zh corpus:
        // "290.1–283.5百万年前" (Artinskian stage, doc 6769570#0).
        // Must be recognized as a temporal hit without panicking.
        let base = NaiveDate::from_ymd_opt(2026, 9, 30).unwrap();
        let hits = chinese_temporal_hits("亚丁斯克期290.1–283.5百万年前", base);
        assert!(
            !hits.is_empty(),
            "expected the offset to be recognized as a temporal hit"
        );
        // Absurd magnitudes from raw digit strings must not panic either.
        let hits = chinese_temporal_hits("99999999999999999999天前发生了大事", base);
        let _ = hits;
        let hits = chinese_temporal_hits("99999999999999999999年前发生了大事", base);
        let _ = hits;
    }
}

#[cfg(test)]
mod unix_timestamp_tests {
    use super::parse_temporal_date;

    #[test]
    fn parses_unix_epoch_seconds_for_timeline_dates() {
        let date = parse_temporal_date(Some("1788710927")).expect("epoch timestamp should parse");
        assert_eq!(date.to_string(), "2026-09-06");
    }
}
