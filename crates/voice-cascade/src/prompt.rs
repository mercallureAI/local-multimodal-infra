//! What the chat model is told. The pipeline is described as it runs; when
//! to speak is the model's call through the `silence` tool.

use crate::protocol::SessionConfig;
use serde_json::{json, Value};

const SYSTEM_GROUP: &str = r#"你是语音助手"{name}"，在一个多人语音频道里。你收到的是频道里最新一句话的转写（可能有识别错误）。频道里的人大多在互相聊天，只有叫到"{name}"（或同音字{aliases}）的话，或者紧接着和你对话的话，才是对你说的。

根据这句话选择一种处理方式：
- 不是对你说的：调用 silence。
"#;

const SYSTEM_PRIVATE: &str = r#"你是语音助手"{name}"，正在和{speaker}一对一语音聊天。你收到的是对方最新一句话的转写（可能有识别错误）。

根据这句话选择一种处理方式：
- 没有需要回应的内容（只有"嗯""啊"之类的语气词、咳嗽、背景声音），或者是在和别人说话：调用 silence。
"#;

const SYSTEM_TASKS: &str = r#"- 对你说的，且需要实时信息（时间、日期、天气、新闻、价格、比分等）、联网搜索、查资料、或者执行操作（设置提醒、发消息、控制设备、记录内容等）：调用 backend_task。后台会去完成，结果出来后交给你，由你念出来。
- 对你说的，闲聊或者凭常识就能回答的：直接回答，不调用工具。

你自己不知道现在的时间、日期、天气和任何最新消息，这些一律用 backend_task。

你说的话会被合成语音直接播放出来，所以回答用自然、简短的中文口语，一般一到三句话；不用列表、标题、表情符号（emoji）和网址。

示例（左边是听到的话，右边是你的输出）：
"老张你那边信号不好，听不清" => <tool_call>
{"name": "silence", "arguments": {}}
</tool_call>
"{name}，今天几号？" => <tool_call>
{"name": "backend_task", "arguments": {"task": "查询今天的日期"}}
</tool_call>
"{name}，你觉得猫可爱还是狗可爱？" => 我觉得都挺可爱的，不过猫更安静一点，我选猫。
"{name}，给大家唱首歌吧。" => 好呀，我来唱两句：小星星，亮晶晶，满天都是小星星。
"{name}，三十七乘以四等于多少？" => 三十七乘以四等于一百四十八。
"{name}，25加17是多少？" => 25加17等于42。"#;

const PERSONA: &str = "\n\n补充设定：\n";

/// Backend words for the model, not said by the speaker.
pub const NOTE: &str =
    "（后台消息，不是对方说的话：{text}\n需要的话，用口语简短地告诉对方；不需要就调用 silence。）";
pub const OPENING: &str = "（现在由你先开口，不是对方说的话：{text}\n用口语简短地说。）";

pub fn system(config: &SessionConfig) -> String {
    let name = config.name.as_str();
    let mut prompt = if config.group {
        let aliases: String = config
            .aliases
            .iter()
            .filter(|alias| !alias.is_empty() && alias.as_str() != name)
            .map(|alias| format!("、{alias}"))
            .collect();
        SYSTEM_GROUP.replace("{aliases}", &aliases)
    } else {
        let speaker = config
            .speaker
            .as_deref()
            .filter(|s| !s.is_empty())
            .unwrap_or("对方");
        SYSTEM_PRIVATE.replace("{speaker}", speaker)
    };
    prompt.push_str(SYSTEM_TASKS);
    let mut prompt = prompt.replace("{name}", name);
    if let Some(extra) = config
        .instructions
        .as_deref()
        .filter(|s| !s.trim().is_empty())
    {
        prompt.push_str(PERSONA);
        prompt.push_str(extra.trim());
    }
    prompt
}

pub fn silence_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "silence",
            "description": "没有需要回应的内容，保持沉默。",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    })
}

pub fn backend_task_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "backend_task",
            "description": "交给后台执行：联网搜索、查询实时信息、执行操作。结果出来后由你念出来。",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "要完成的任务，一句完整的话，包含所有必要细节。"
                    }
                },
                "required": ["task"]
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_prompt_names_the_bot_and_its_aliases() {
        let config = SessionConfig {
            name: "小乐".into(),
            aliases: vec!["小月".into(), "小乐".into()],
            group: true,
            instructions: Some("说话带点东北口音。".into()),
            ..SessionConfig::default()
        };
        let prompt = system(&config);
        assert!(prompt.contains(r#"叫到"小乐"（或同音字、小月）"#));
        assert!(prompt.contains(r#""小乐，今天几号？""#));
        assert!(prompt.ends_with("补充设定：\n说话带点东北口音。"));
    }

    #[test]
    fn private_prompt_names_the_speaker() {
        let config = SessionConfig {
            name: "小乐".into(),
            speaker: Some("张三".into()),
            ..SessionConfig::default()
        };
        assert!(system(&config).contains("正在和张三一对一语音聊天"));
    }
}
