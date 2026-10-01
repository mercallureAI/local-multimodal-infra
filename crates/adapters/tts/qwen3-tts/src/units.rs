//! Units Qwen3-TTS misreads, written out in Chinese text before it is
//! tokenized.

/// Units after a number, as Chinese text reads them: the ones Qwen3-TTS
/// misreads ("20kHz" as noise, "35ms" as 毫米, "5L" as 毫升, "65W" as 晚,
/// "256MB" as MG, "64bit" as B体, "1Gbps" as noise). Those it reads well
/// (km, cm, g, ℃, %, V) are left; so are "A" ("3A" games) and "B" ("7B"
/// models). Binary data units read as decimal ones. Longer units first
/// ("MB/s" before "MB").
const UNITS: [(&str, &str); 53] = [
    ("bytes", "字节"),
    ("Bytes", "字节"),
    ("byte", "字节"),
    ("Byte", "字节"),
    ("bits", "比特"),
    ("Bits", "比特"),
    ("KB/s", "千字节每秒"),
    ("kB/s", "千字节每秒"),
    ("MB/s", "兆字节每秒"),
    ("GB/s", "G字节每秒"),
    ("Kb/s", "千比特每秒"),
    ("kb/s", "千比特每秒"),
    ("Mb/s", "兆比特每秒"),
    ("Gb/s", "G比特每秒"),
    ("Kbps", "千比特每秒"),
    ("kbps", "千比特每秒"),
    ("Mbps", "兆比特每秒"),
    ("Gbps", "G比特每秒"),
    ("GHz", "吉赫兹"),
    ("MHz", "兆赫兹"),
    ("kHz", "千赫兹"),
    ("mAh", "毫安时"),
    ("kWh", "千瓦时"),
    ("fps", "帧每秒"),
    ("bit", "比特"),
    ("Bit", "比特"),
    ("KiB", "千字节"),
    ("KIB", "千字节"),
    ("MiB", "兆字节"),
    ("MIB", "兆字节"),
    ("GiB", "G字节"),
    ("GIB", "G字节"),
    ("TiB", "T字节"),
    ("TIB", "T字节"),
    ("Hz", "赫兹"),
    ("mA", "毫安"),
    ("ms", "毫秒"),
    ("kW", "千瓦"),
    ("Wh", "瓦时"),
    ("kg", "公斤"),
    ("nm", "纳米"),
    ("KB", "千字节"),
    ("kB", "千字节"),
    ("MB", "兆字节"),
    ("GB", "G字节"),
    ("TB", "T字节"),
    ("PB", "P字节"),
    ("Kb", "千比特"),
    ("kb", "千比特"),
    ("Mb", "兆比特"),
    ("Gb", "G比特"),
    ("W", "瓦"),
    ("L", "升"),
];

/// `text` with the [`UNITS`] that follow a number in Chinese text written
/// out (English reads them as they are). Whether it is Chinese is told by
/// what comes before the number, so text still being written reads the same
/// once more of it comes.
pub(crate) fn read_units(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 8);
    let mut rest = text;
    'scan: while let Some(c) = rest.chars().next() {
        let number = out.strip_suffix(' ').unwrap_or(&out);
        let before = number
            .trim_end_matches(|c: char| c.is_ascii_digit() || c == '.')
            .trim_end();
        let after_number = number.ends_with(|c: char| c.is_ascii_digit())
            && before.ends_with(|c: char| {
                ('\u{4e00}'..='\u{9fff}').contains(&c)
                    || ('\u{3000}'..='\u{303f}').contains(&c)
                    || ('\u{ff00}'..='\u{ffef}').contains(&c)
            });
        if after_number {
            for (unit, reading) in UNITS {
                let Some(tail) = rest.strip_prefix(unit) else {
                    continue;
                };
                // A word that only starts like one ("20Hzx") is not it.
                if tail.starts_with(|c: char| c.is_ascii_alphanumeric()) {
                    continue;
                }
                if out.ends_with(' ') {
                    out.pop();
                }
                out.push_str(reading);
                rest = tail;
                continue 'scan;
            }
        }
        out.push(c);
        rest = &rest[c.len_utf8()..];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn units_after_a_number_are_read_out_in_chinese_text() {
        assert_eq!(
            read_units("频率范围大约是 20Hz 到 20kHz，主频 1.8GHz、带宽 5 MHz。"),
            "频率范围大约是 20赫兹 到 20千赫兹，主频 1.8吉赫兹、带宽 5兆赫兹。"
        );
        assert_eq!(read_units("刷新率是60Hz、144Hz"), "刷新率是60赫兹、144赫兹");
        assert_eq!(
            read_units("延迟 35ms，重 20kg，电池 5000mAh，支持 65W 快充，一桶 5L，144fps，5nm，1.5kW，8kWh。"),
            "延迟 35毫秒，重 20公斤，电池 5000毫安时，支持 65瓦 快充，一桶 5升，144帧每秒，5纳米，1.5千瓦，8千瓦时。"
        );
        assert_eq!(
            read_units("内存 32GB，硬盘 2TB，缓存 256MB，文件 500KB，镜像 4GiB，分区 512MIB。"),
            "内存 32G字节，硬盘 2T字节，缓存 256兆字节，文件 500千字节，镜像 4G字节，分区 512兆字节。"
        );
        assert_eq!(
            read_units("带宽 1Gbps，下载 12MB/s，读写 3.5GB/s，字段 64bit，头部 20Byte，每字符 8bits，7B 模型。"),
            "带宽 1G比特每秒，下载 12兆字节每秒，读写 3.5G字节每秒，字段 64比特，头部 20字节，每字符 8比特，7B 模型。"
        );
        // Units it reads well, and "3A", are left.
        assert_eq!(
            read_units("长 15km，气温 28℃，3A 大作"),
            "长 15km，气温 28℃，3A 大作"
        );
        // Not after a number, not a unit, or English text: as it is.
        assert_eq!(read_units("Hz 是频率单位"), "Hz 是频率单位");
        assert_eq!(read_units("型号 5Hzx 的"), "型号 5Hzx 的");
        assert_eq!(read_units("It runs at 60Hz."), "It runs at 60Hz.");
        assert_eq!(read_units("20Hz 是下限"), "20Hz 是下限");
        // English before Chinese stays English as more text comes.
        assert_eq!(
            read_units("The 2.4GHz band, 也就是"),
            "The 2.4GHz band, 也就是"
        );
    }
}
