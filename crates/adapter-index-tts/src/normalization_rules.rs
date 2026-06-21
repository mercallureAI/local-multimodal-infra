use super::*;

pub(crate) fn lightweight_tn_placeholder_pass(text: &str, use_chinese_rules: bool) -> String {
    let text = normalize_fullwidth_ascii(text);
    let (text, emails) = save_email_placeholders(&text);
    let text = if use_chinese_rules {
        normalize_chinese_tn(&text)
    } else {
        normalize_english_tn(&text)
    };
    restore_email_placeholders(&text, &emails)
}

pub(crate) fn normalize_fullwidth_ascii(text: &str) -> String {
    text.chars()
        .map(|ch| match ch {
            '０'..='９' => char::from_u32('0' as u32 + (ch as u32 - '０' as u32)).unwrap_or(ch),
            'Ａ'..='Ｚ' => char::from_u32('A' as u32 + (ch as u32 - 'Ａ' as u32)).unwrap_or(ch),
            'ａ'..='ｚ' => char::from_u32('a' as u32 + (ch as u32 - 'ａ' as u32)).unwrap_or(ch),
            '＠' => '@',
            '．' => '.',
            '－' => '-',
            '／' => '/',
            '：' => ':',
            '％' => '%',
            '＋' => '+',
            '＄' => '$',
            '￥' => '￥',
            '　' => ' ',
            _ => ch,
        })
        .collect()
}

pub(crate) fn save_email_placeholders(text: &str) -> (String, Vec<String>) {
    let mut out = String::with_capacity(text.len());
    let mut emails = Vec::new();
    let mut pos = 0;
    while pos < text.len() {
        if let Some((email, end)) = parse_simple_email_at(text, pos) {
            out.push_str(&format!(
                "<email_{}>",
                official_placeholder_suffix(emails.len())
            ));
            emails.push(email);
            pos = end;
            continue;
        }
        let ch = text[pos..].chars().next().expect("valid char boundary");
        out.push(ch);
        pos += ch.len_utf8();
    }
    (out, emails)
}

pub(crate) fn restore_email_placeholders(text: &str, emails: &[String]) -> String {
    let mut out = text.to_string();
    for (index, email) in emails.iter().enumerate() {
        out = out.replace(
            &format!("<email_{}>", official_placeholder_suffix(index)),
            email,
        );
    }
    out
}

pub(crate) fn parse_simple_email_at(text: &str, pos: usize) -> Option<(String, usize)> {
    if pos > 0 && text[..pos].chars().next_back().is_some_and(is_email_char) {
        return None;
    }
    let bytes = text.as_bytes();
    let mut i = pos;
    while i < text.len() && bytes[i].is_ascii_alphanumeric() {
        i += 1;
    }
    if i == pos || bytes.get(i).copied() != Some(b'@') {
        return None;
    }
    i += 1;
    let domain_start = i;
    while i < text.len() && bytes[i].is_ascii_alphanumeric() {
        i += 1;
    }
    if i == domain_start || bytes.get(i).copied() != Some(b'.') {
        return None;
    }
    i += 1;
    let tld_start = i;
    while i < text.len() && bytes[i].is_ascii_alphabetic() {
        i += 1;
    }
    if i == tld_start || text[i..].chars().next().is_some_and(is_email_char) {
        return None;
    }
    Some((text[pos..i].to_string(), i))
}

pub(crate) fn is_email_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '@' | '.')
}

pub(crate) fn normalize_chinese_tn(text: &str) -> String {
    let chars = text.chars().collect::<Vec<_>>();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        if let Some((s, end)) = parse_chinese_date_ymd(&chars, i) {
            out.push_str(&s);
            i = end;
        } else if let Some((s, end)) = parse_slash_date(&chars, i, true) {
            out.push_str(&s);
            i = end;
        } else if let Some((s, end)) = parse_time(&chars, i, true) {
            out.push_str(&s);
            i = end;
        } else if let Some((s, end)) = parse_money(&chars, i, true) {
            out.push_str(&s);
            i = end;
        } else if let Some((s, end)) = parse_percentage(&chars, i, true) {
            out.push_str(&s);
            i = end;
        } else if let Some((s, end)) = parse_number_with_unit(&chars, i, true) {
            out.push_str(&s);
            i = end;
        } else if chars[i].is_ascii_digit() {
            let (digits, end) = collect_number(&chars, i);
            out.push_str(&chinese_number_like(&digits));
            i = end;
        } else if chars[i] == '+' && i > 0 && chars.get(i - 1).is_some_and(|c| c.is_ascii_digit()) {
            out.push_str("加");
            i += 1;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

pub(crate) fn normalize_english_tn(text: &str) -> String {
    let chars = text.chars().collect::<Vec<_>>();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        if let Some((s, end)) = parse_slash_date(&chars, i, false) {
            out.push_str(&s);
            i = end;
        } else if let Some((s, end)) = parse_time(&chars, i, false) {
            out.push_str(&s);
            i = end;
        } else if let Some((s, end)) = parse_money(&chars, i, false) {
            out.push_str(&s);
            i = end;
        } else if let Some((s, end)) = parse_percentage(&chars, i, false) {
            out.push_str(&s);
            i = end;
        } else if chars[i].is_ascii_digit() {
            if (i > 0 && chars[i - 1].is_ascii_alphabetic())
                || chars.get(i + 1).is_some_and(|ch| ch.is_ascii_alphabetic())
            {
                out.push(chars[i]);
                i += 1;
                continue;
            }
            let (digits, end) = collect_number(&chars, i);
            out.push_str(&english_number_like(&digits));
            i = end;
        } else if chars[i] == '+' {
            out.push_str(" plus ");
            i += 1;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

pub(crate) fn apply_char_rep_map(text: &str, map: &[(&str, &str)]) -> String {
    let mut out = text.to_string();
    for (from, to) in map {
        out = out.replace(from, to);
    }
    out
}

pub(crate) fn collapse_spaces(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut previous_space = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !previous_space {
                out.push(' ');
                previous_space = true;
            }
        } else {
            out.push(ch);
            previous_space = false;
        }
    }
    out.trim().to_string()
}

pub(crate) fn parse_chinese_date_ymd(chars: &[char], start: usize) -> Option<(String, usize)> {
    let (year, mut i) = collect_exact_digits(chars, start, 4)?;
    if chars.get(i).copied() != Some('年') {
        return None;
    }
    i += 1;
    let (month, month_end) = collect_one_or_two_digits(chars, i)?;
    if chars.get(month_end).copied() != Some('月') {
        return None;
    }
    i = month_end + 1;
    let (day, day_end) = collect_one_or_two_digits(chars, i)?;
    if chars.get(day_end).copied() != Some('日') && chars.get(day_end).copied() != Some('号') {
        return None;
    }
    Some((
        format!(
            "{}年{}月{}日",
            chinese_digits(&year),
            chinese_cardinal(month.parse::<u32>().ok()?),
            chinese_cardinal(day.parse::<u32>().ok()?)
        ),
        day_end + 1,
    ))
}

pub(crate) fn parse_slash_date(
    chars: &[char],
    start: usize,
    chinese: bool,
) -> Option<(String, usize)> {
    let (year, mut i) = collect_exact_digits(chars, start, 4)?;
    if chars.get(i).copied() != Some('/') {
        return None;
    }
    i += 1;
    let (month, month_end) = collect_one_or_two_digits(chars, i)?;
    if chars.get(month_end).copied() != Some('/') {
        return None;
    }
    i = month_end + 1;
    let (day, day_end) = collect_one_or_two_digits(chars, i)?;
    if chinese {
        Some((
            format!(
                "{}年{}月{}日",
                chinese_digits(&year),
                chinese_cardinal(month.parse::<u32>().ok()?),
                chinese_cardinal(day.parse::<u32>().ok()?)
            ),
            day_end,
        ))
    } else {
        Some((
            format!(
                "{} slash {} slash {}",
                english_digits(&year),
                english_number_like(&month),
                english_number_like(&day)
            ),
            day_end,
        ))
    }
}

pub(crate) fn parse_time(chars: &[char], start: usize, chinese: bool) -> Option<(String, usize)> {
    let (hour, mut i) = collect_one_or_two_digits(chars, start)?;
    if chars.get(i).copied() != Some(':') {
        return None;
    }
    i += 1;
    let (minute, end) = collect_exact_digits(chars, i, 2)?;
    let hour_value = hour.parse::<u32>().ok()?;
    let minute_value = minute.parse::<u32>().ok()?;
    if hour_value > 29 || minute_value > 59 {
        return None;
    }
    let (suffix, suffix_end) = parse_ampm(chars, end);
    if chinese {
        let prefix = match suffix.as_deref() {
            Some("AM") => "上午",
            Some("PM") => "下午",
            _ => "",
        };
        let time = if minute_value == 0 {
            format!("{prefix}{}点", chinese_cardinal(hour_value))
        } else {
            format!(
                "{prefix}{}点{}分",
                chinese_cardinal(hour_value),
                chinese_cardinal(minute_value)
            )
        };
        Some((time, suffix_end))
    } else {
        let suffix = suffix.map(|s| format!(" {s}")).unwrap_or_default();
        let time = if minute_value == 0 {
            format!("{} o'clock{suffix}", english_small_number(hour_value))
        } else {
            format!(
                "{} {}{suffix}",
                english_small_number(hour_value),
                english_two_digit_minute(minute_value)
            )
        };
        Some((time, suffix_end))
    }
}

pub(crate) fn parse_ampm(chars: &[char], start: usize) -> (Option<String>, usize) {
    let mut i = start;
    while chars.get(i).is_some_and(|ch| ch.is_ascii_whitespace()) {
        i += 1;
    }
    let a = chars.get(i).copied().map(|ch| ch.to_ascii_uppercase());
    let m = chars.get(i + 1).copied().map(|ch| ch.to_ascii_uppercase());
    if matches!((a, m), (Some('A'), Some('M')) | (Some('P'), Some('M'))) {
        (Some(format!("{}{}", a.unwrap(), m.unwrap())), i + 2)
    } else {
        (None, start)
    }
}

pub(crate) fn parse_money(chars: &[char], start: usize, chinese: bool) -> Option<(String, usize)> {
    let currency = match chars.get(start).copied()? {
        '$' => {
            if chinese {
                "美元"
            } else {
                " dollars"
            }
        }
        '¥' | '￥' => {
            if chinese {
                "元"
            } else {
                " yuan"
            }
        }
        '€' => {
            if chinese {
                "欧元"
            } else {
                " euros"
            }
        }
        '£' => {
            if chinese {
                "英镑"
            } else {
                " pounds"
            }
        }
        _ => return None,
    };
    let (number, end) = collect_number(chars, start + 1);
    if number.is_empty() {
        return Some((currency.trim().to_string(), start + 1));
    }
    if chinese {
        Some((format!("{}{}", chinese_number_like(&number), currency), end))
    } else {
        Some((format!("{}{}", english_number_like(&number), currency), end))
    }
}

pub(crate) fn parse_percentage(
    chars: &[char],
    start: usize,
    chinese: bool,
) -> Option<(String, usize)> {
    let (number, end) = collect_number(chars, start);
    if number.is_empty() || chars.get(end).copied() != Some('%') {
        return None;
    }
    if chinese {
        Some((format!("百分之{}", chinese_number_like(&number)), end + 1))
    } else {
        Some((format!("{} percent", english_number_like(&number)), end + 1))
    }
}

pub(crate) fn parse_number_with_unit(
    chars: &[char],
    start: usize,
    chinese: bool,
) -> Option<(String, usize)> {
    let (number, mut end) = collect_number(chars, start);
    if number.is_empty() {
        return None;
    }
    let rest = chars[end..].iter().collect::<String>();
    let units = [
        ("km/h", "千米每小时", " kilometers per hour"),
        ("KM/H", "千米每小时", " kilometers per hour"),
        ("m/s", "米每秒", " meters per second"),
        ("M/S", "米每秒", " meters per second"),
        ("km", "千米", " kilometers"),
        ("KM", "千米", " kilometers"),
        ("kg", "千克", " kilograms"),
        ("KG", "千克", " kilograms"),
        ("GB", "GB", " gigabytes"),
        ("MB", "MB", " megabytes"),
        ("℃", "摄氏度", " degrees Celsius"),
        ("g", "克", " grams"),
        ("G", "G", " g"),
        ("m", "米", " meters"),
        ("M", "M", " m"),
    ];
    for (raw, zh, en) in units {
        if rest.starts_with(raw) {
            end += raw.chars().count();
            return if chinese {
                Some((format!("{}{}", chinese_number_like(&number), zh), end))
            } else {
                Some((format!("{}{}", english_number_like(&number), en), end))
            };
        }
    }
    None
}

pub(crate) fn collect_number(chars: &[char], start: usize) -> (String, usize) {
    let mut out = String::new();
    let mut i = start;
    while chars.get(i).is_some_and(|ch| ch.is_ascii_digit()) {
        out.push(chars[i]);
        i += 1;
    }
    if chars.get(i).copied() == Some('.') && chars.get(i + 1).is_some_and(|ch| ch.is_ascii_digit())
    {
        out.push('.');
        i += 1;
        while chars.get(i).is_some_and(|ch| ch.is_ascii_digit()) {
            out.push(chars[i]);
            i += 1;
        }
    }
    (out, i)
}

pub(crate) fn collect_exact_digits(
    chars: &[char],
    start: usize,
    len: usize,
) -> Option<(String, usize)> {
    if start + len > chars.len()
        || !chars[start..start + len]
            .iter()
            .all(|ch| ch.is_ascii_digit())
    {
        return None;
    }
    Some((chars[start..start + len].iter().collect(), start + len))
}

pub(crate) fn collect_one_or_two_digits(chars: &[char], start: usize) -> Option<(String, usize)> {
    if !chars.get(start).is_some_and(|ch| ch.is_ascii_digit()) {
        return None;
    }
    let mut end = start + 1;
    if chars.get(end).is_some_and(|ch| ch.is_ascii_digit()) {
        end += 1;
    }
    Some((chars[start..end].iter().collect(), end))
}

pub(crate) fn chinese_number_like(number: &str) -> String {
    if let Some((left, right)) = number.split_once('.') {
        return format!(
            "{}点{}",
            chinese_cardinal(left.parse::<u32>().unwrap_or(0)),
            chinese_digits(right)
        );
    }
    if number.len() > 4 || number.starts_with('0') && number.len() > 1 {
        chinese_digits(number)
    } else {
        chinese_cardinal(number.parse::<u32>().unwrap_or(0))
    }
}

pub(crate) fn chinese_digits(digits: &str) -> String {
    digits.chars().map(chinese_digit).collect()
}

pub(crate) fn chinese_digit(ch: char) -> char {
    match ch {
        '0' => '零',
        '1' => '一',
        '2' => '二',
        '3' => '三',
        '4' => '四',
        '5' => '五',
        '6' => '六',
        '7' => '七',
        '8' => '八',
        '9' => '九',
        _ => ch,
    }
}

pub(crate) fn chinese_cardinal(n: u32) -> String {
    match n {
        0..=10 => [
            "零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
        ][n as usize]
            .to_string(),
        11..=19 => format!(
            "十{}",
            if n % 10 == 0 {
                "".to_string()
            } else {
                chinese_cardinal(n % 10)
            }
        ),
        20..=99 => {
            let tens = n / 10;
            let ones = n % 10;
            if ones == 0 {
                format!("{}十", chinese_cardinal(tens))
            } else {
                format!("{}十{}", chinese_cardinal(tens), chinese_cardinal(ones))
            }
        }
        100..=9999 => chinese_digits(&n.to_string()),
        _ => chinese_digits(&n.to_string()),
    }
}

pub(crate) fn english_number_like(number: &str) -> String {
    if let Some((left, right)) = number.split_once('.') {
        return format!(
            "{} point {}",
            english_small_number(left.parse::<u32>().unwrap_or(0)),
            english_digits(right)
        );
    }
    if number.len() > 2 || number.starts_with('0') && number.len() > 1 {
        english_digits(number)
    } else {
        english_small_number(number.parse::<u32>().unwrap_or(0))
    }
}

pub(crate) fn english_digits(digits: &str) -> String {
    digits
        .chars()
        .map(|ch| english_digit(ch).to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

pub(crate) fn english_digit(ch: char) -> &'static str {
    match ch {
        '0' => "zero",
        '1' => "one",
        '2' => "two",
        '3' => "three",
        '4' => "four",
        '5' => "five",
        '6' => "six",
        '7' => "seven",
        '8' => "eight",
        '9' => "nine",
        _ => "",
    }
}

pub(crate) fn english_small_number(n: u32) -> String {
    const SMALL: [&str; 20] = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ];
    const TENS: [&str; 10] = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];
    match n {
        0..=19 => SMALL[n as usize].to_string(),
        20..=99 => {
            if n % 10 == 0 {
                TENS[(n / 10) as usize].to_string()
            } else {
                format!("{} {}", TENS[(n / 10) as usize], SMALL[(n % 10) as usize])
            }
        }
        _ => english_digits(&n.to_string()),
    }
}

pub(crate) fn english_two_digit_minute(n: u32) -> String {
    if n < 10 {
        format!("oh {}", english_small_number(n))
    } else {
        english_small_number(n)
    }
}

pub(crate) fn is_plausible_time_pair(hour: &str, minute: &str) -> bool {
    if hour.is_empty() || hour.len() > 2 || minute.len() != 2 {
        return false;
    }
    let Ok(hour) = hour.parse::<u32>() else {
        return false;
    };
    let Ok(minute) = minute.parse::<u32>() else {
        return false;
    };
    hour <= 29 && minute <= 59
}
