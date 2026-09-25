//! The conversation as the chat model reads it. Turns are generated while
//! the speaker goes on talking, so a reply lands after the message it
//! answers (its anchor), not simply at the end, and a task's result after
//! the call that asked for it.

use local_core::ChatMessage;
use std::collections::VecDeque;

/// Beyond these the oldest messages go (the prompt must stay well inside the
/// chat model's context; Chinese text is about a token per character).
const MAX_MESSAGES: usize = 32;
const MAX_CHARS: usize = 4000;

#[derive(Debug)]
struct Entry {
    id: u64,
    /// The message this one answers (a reply inserted after its anchor).
    answers: Option<u64>,
    message: ChatMessage,
}

#[derive(Debug, Default)]
pub(crate) struct History {
    entries: VecDeque<Entry>,
    next_id: u64,
}

impl History {
    /// Appends `message`; returns its id.
    pub(crate) fn push(&mut self, message: ChatMessage) -> u64 {
        let at = self.entries.len();
        self.insert(at, None, message)
    }

    /// The id of the last message (what a reply started now answers).
    pub(crate) fn last_id(&self) -> Option<u64> {
        self.entries.back().map(|entry| entry.id)
    }

    /// Inserts a reply right after `anchor`, the last message when it was
    /// asked for (at the end when that one is gone). `answers` is the
    /// utterance it answers, if any: when that one is gone (replaced by a
    /// longer utterance, or trimmed) the reply is dropped and false returned.
    pub(crate) fn insert_after(
        &mut self,
        anchor: Option<u64>,
        answers: Option<u64>,
        message: ChatMessage,
    ) -> bool {
        if answers.is_some_and(|id| self.position(id).is_none()) {
            return false;
        }
        let at = anchor
            .and_then(|anchor| self.position(anchor))
            .map_or(self.entries.len(), |index| index + 1);
        self.insert(at, answers, message);
        true
    }

    /// Inserts a tool result after the assistant message that made the call
    /// (and the results already given to it); at the end when that message
    /// is gone. Returns its id.
    pub(crate) fn insert_tool_result(&mut self, call_id: &str, message: ChatMessage) -> u64 {
        let call = self.entries.iter().position(|entry| {
            entry.message.role == "assistant"
                && entry
                    .message
                    .tool_calls
                    .iter()
                    .any(|call| call.id == call_id)
        });
        let Some(mut at) = call.map(|index| index + 1) else {
            return self.push(message);
        };
        while self
            .entries
            .get(at)
            .is_some_and(|entry| entry.message.role == "tool")
        {
            at += 1;
        }
        self.insert(at, None, message)
    }

    /// The id the next message gets (ids grow in the order messages come).
    pub(crate) fn next_id(&self) -> u64 {
        self.next_id
    }

    /// Removes the utterance `id` (the speaker went on: a longer one replaces
    /// it) and the plain replies to it (a `silence` call is one). Tasks
    /// handed off for it, their results and backend notes stay: they
    /// happened.
    pub(crate) fn remove_utterance(&mut self, id: u64) {
        self.entries.retain(|entry| {
            let plain = entry
                .message
                .tool_calls
                .iter()
                .all(|call| call.name == "silence");
            entry.id != id && !(entry.answers == Some(id) && plain)
        });
    }

    pub(crate) fn messages(&self) -> Vec<ChatMessage> {
        self.entries
            .iter()
            .map(|entry| entry.message.clone())
            .collect()
    }

    fn position(&self, id: u64) -> Option<usize> {
        self.entries.iter().position(|entry| entry.id == id)
    }

    fn insert(&mut self, at: usize, answers: Option<u64>, message: ChatMessage) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.entries.insert(
            at,
            Entry {
                id,
                answers,
                message,
            },
        );
        self.trim();
        id
    }

    fn trim(&mut self) {
        let chars = |m: &ChatMessage| {
            m.content.as_deref().map_or(0, |c| c.chars().count())
                + m.tool_calls
                    .iter()
                    .map(|c| c.arguments.chars().count())
                    .sum::<usize>()
        };
        let mut total: usize = self.entries.iter().map(|e| chars(&e.message)).sum();
        while self.entries.len() > MAX_MESSAGES || (total > MAX_CHARS && self.entries.len() > 1) {
            if let Some(removed) = self.entries.pop_front() {
                total -= chars(&removed.message);
            }
            // A tool result may not open the history without its call.
            while self
                .entries
                .front()
                .is_some_and(|entry| entry.message.role == "tool")
            {
                if let Some(removed) = self.entries.pop_front() {
                    total -= chars(&removed.message);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use local_core::ChatToolCall;

    fn message(role: &str, text: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: Some(text.into()),
            ..ChatMessage::default()
        }
    }

    fn call(id: &str) -> ChatMessage {
        ChatMessage {
            role: "assistant".into(),
            content: Some(String::new()),
            tool_calls: vec![ChatToolCall {
                id: id.into(),
                name: "backend_task".into(),
                arguments: "{}".into(),
            }],
            ..ChatMessage::default()
        }
    }

    fn contents(history: &History) -> Vec<String> {
        history
            .messages()
            .into_iter()
            .map(|m| format!("{}:{}", m.role, m.content.unwrap_or_default()))
            .collect()
    }

    #[test]
    fn a_reply_lands_after_the_message_it_answers() {
        let mut history = History::default();
        let a = history.push(message("user", "a"));
        let anchor = history.last_id();
        history.push(message("user", "b")); // said while the bot answered a
        assert!(history.insert_after(anchor, Some(a), message("assistant", "to a")));
        assert_eq!(contents(&history), ["user:a", "assistant:to a", "user:b"]);
    }

    #[test]
    fn a_reply_to_a_replaced_utterance_is_dropped() {
        let mut history = History::default();
        let first = history.push(message("user", "Jarvis,"));
        let anchor = history.last_id();
        assert!(history.insert_after(anchor, Some(first), message("assistant", "Yes?")));
        history.remove_utterance(first); // "Jarvis," became a longer request
        history.push(message("user", "Jarvis, what time is it?"));
        assert!(!history.insert_after(anchor, Some(first), message("assistant", "late")));
        assert_eq!(contents(&history), ["user:Jarvis, what time is it?"]);
    }

    #[test]
    fn a_replaced_utterance_keeps_its_tasks_and_later_notes() {
        let mut history = History::default();
        let first = history.push(message("user", "turn on the light"));
        let anchor = history.last_id();
        history.push(message("user", "(note)"));
        assert!(history.insert_after(anchor, Some(first), call("c1")));
        let mut result = message("tool", "done");
        result.tool_call_id = Some("c1".into());
        history.insert_tool_result("c1", result);
        history.remove_utterance(first);
        assert_eq!(
            contents(&history),
            ["assistant:", "tool:done", "user:(note)"]
        );
    }

    #[test]
    fn a_relay_survives_the_chatter_it_followed() {
        let mut history = History::default();
        let chatter = history.push(message("user", "B: well"));
        let anchor = history.last_id();
        // B goes on while the bot tells a task's result.
        history.remove_utterance(chatter);
        history.push(message("user", "B: well, anyway"));
        assert!(history.insert_after(anchor, None, message("assistant", "it's done")));
        assert_eq!(
            contents(&history),
            ["user:B: well, anyway", "assistant:it's done"]
        );
    }

    #[test]
    fn a_silence_call_goes_with_its_utterance() {
        let mut history = History::default();
        let first = history.push(message("user", "Jarvis,"));
        let mut silence = call("s1");
        silence.tool_calls[0].name = "silence".into();
        assert!(history.insert_after(Some(first), Some(first), silence));
        history.remove_utterance(first);
        assert!(history.messages().is_empty());
    }

    #[test]
    fn a_tool_result_follows_its_call() {
        let mut history = History::default();
        history.push(message("user", "time?"));
        history.push(call("c1"));
        history.push(message("user", "and the weather?"));
        let mut result = message("tool", "three o'clock");
        result.tool_call_id = Some("c1".into());
        history.insert_tool_result("c1", result);
        assert_eq!(
            contents(&history),
            [
                "user:time?",
                "assistant:",
                "tool:three o'clock",
                "user:and the weather?"
            ]
        );
        history.insert_tool_result("gone", message("tool", "late"));
        assert_eq!(contents(&history).last().unwrap(), "tool:late");
    }

    #[test]
    fn long_histories_lose_their_oldest_messages() {
        let mut history = History::default();
        let mut result = message("tool", "result");
        result.tool_call_id = Some("c".into());
        history.push(message("user", &"x".repeat(MAX_CHARS - 10)));
        history.push(result);
        history.push(message("user", &"y".repeat(20)));
        // The first message went, and the tool result it left at the front.
        assert_eq!(contents(&history), [format!("user:{}", "y".repeat(20))]);
        for i in 0..MAX_MESSAGES + 5 {
            history.push(message("user", &i.to_string()));
        }
        assert_eq!(history.messages().len(), MAX_MESSAGES);
    }
}
