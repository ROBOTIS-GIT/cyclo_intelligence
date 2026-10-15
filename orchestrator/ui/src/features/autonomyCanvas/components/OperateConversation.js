import { useEffect, useRef, useState } from "react";
import { FiSend } from "react-icons/fi";
import { autonomyRequest } from "../hooks/autonomyApi";
import "./OperateConversation.css";

export default function OperateConversation() {
  const [chat, setChat] = useState({ messages: [], pending: false });
  const [text, setText] = useState("");
  const [error, setError] = useState("");
  const [sending, setSending] = useState(false);
  const end = useRef(null);
  const alive = useRef(true);
  const requestEpoch = useRef(0);
  useEffect(() => {
    alive.current = true;
    let timer;
    async function poll() {
      const epoch = requestEpoch.current;
      try {
        const next = await autonomyRequest("/autonomy/conversation");
        if (alive.current && epoch === requestEpoch.current) { setChat(next); setError(""); }
      } catch (e) { if (alive.current) setError(e.message); }
      if (alive.current) timer = setTimeout(poll, 1000);
    }
    void poll();
    return () => { alive.current = false; clearTimeout(timer); };
  }, []);
  const latestMessageId = chat.messages.at(-1)?.id;
  useEffect(() => { end.current?.scrollIntoView?.({ block: "nearest", behavior: "smooth" }); }, [latestMessageId, chat.pending]);
  async function submit(event) {
    event.preventDefault();
    if (!text.trim() || sending || chat.pending) return;
    const message = text.trim();
    ++requestEpoch.current; setSending(true); setError("");
    try {
      const next = await autonomyRequest("/autonomy/conversation", "POST", {
        text: message, request_id: window.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`,
      });
      if (alive.current) { setChat(next); setText(""); }
    } catch (e) { if (alive.current) setError(e.message); }
    finally { ++requestEpoch.current; if (alive.current) setSending(false); }
  }
  return <div className="operate-conversation">
    <div className="operate-chat-messages" role="log" aria-label="Conversation" aria-live="polite">
      {!chat.messages.length && <p className="operate-chat-empty">Tell the robot to reverse patrol, pause, resume, or follow a task instruction.</p>}
      {chat.messages.map(message => <div key={message.id} className={`operate-chat-message is-${message.role}`}>
        <span>{message.role === "user" ? "You" : "Robot"}</span><p>{message.content}</p>
      </div>)}
      {(sending || chat.pending) && <p role="status">Thinking…</p>}
      <div ref={end}/>
    </div>
    {(error || chat.error) && <p className="operate-chat-error" role="alert">{error || chat.error}</p>}
    <form onSubmit={submit} className="operate-chat-compose">
      <textarea aria-label="Message to robot" value={text} maxLength={1000} rows={3}
        placeholder="Give the robot an instruction…" onChange={e => setText(e.target.value)}
        onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); void submit(e); } }}/>
      <button type="submit" title="Send message" aria-label="Send message" disabled={!text.trim() || sending || chat.pending}><FiSend/></button>
    </form>
  </div>;
}
