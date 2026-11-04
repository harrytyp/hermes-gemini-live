"""The Gemini Live wire, and its translation to the browser's frame vocabulary.

Field names and casing here are what the endpoint accepted during the spike; the
two facts most likely to be re-guessed wrongly are recorded inline.
"""

from __future__ import annotations

WS_URI = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)

#: Mic bytes go up at 16 kHz; the model speaks back at 24 kHz. One constant each so
#: neither direction silently inherits the other's rate.
IN_MIME = "audio/pcm;rate=16000"
OUT_MIME = "audio/pcm;rate=24000"


def wire_model(model: str) -> str:
    """The bare id with the API's ``models/`` prefix, exactly once."""
    return model if model.startswith("models/") else f"models/{model}"


def build_setup(*, model: str, voice: str, instructions: str,
                thinking_level: str | None = None,
                activity: dict | None = None,
                tools: list[dict] | None = None,
                resume_handle: str | None = None) -> dict:
    setup = {
        "model": wire_model(model),
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
        },
        "systemInstruction": {"parts": [{"text": instructions}]},
        "inputAudioTranscription": {},
        "outputAudioTranscription": {},
        # The handle is what lets a call outlive one socket: Gemini hands one out with
        # every ``sessionResumptionUpdate`` and continues the same conversation when the
        # next setup carries it. Which makes the difference between a 15-minute call and
        # one that keeps going.
        "sessionResumption": {"handle": resume_handle} if resume_handle else {},
        # Google owns the trim: it cuts at 80% of the context window toward half of
        # that, and never cuts the system instruction. Without it an audio-only session
        # hard-stops near 15 minutes.
        "contextWindowCompression": {"slidingWindow": {}},
    }
    if activity:
        setup["realtimeInputConfig"] = activity
    if tools:
        setup["tools"] = [{"functionDeclarations": tools}]
    if thinking_level:
        setup["generationConfig"]["thinkingConfig"] = {"thinkingLevel": thinking_level}
    return {"setup": setup}


def audio_chunk(b64: str) -> dict:
    return {"realtimeInput": {"audio": {"mimeType": IN_MIME, "data": b64}}}


def text_input(text: str) -> dict:
    """A text turn, for keystrokes and for tests that need no microphone.

    ``turnComplete`` must NOT ride ``realtimeInput`` — the endpoint answers
    ``Unknown name "turnComplete" at 'realtime_input'`` and closes 1007.
    """
    return {"realtimeInput": {"text": text}}


def function_response(call_id: str, result: str, name: str | None = None) -> dict:
    entry = {"id": call_id, "response": {"result": result}}
    if name:
        entry["name"] = name
    return {"toolResponse": {"functionResponses": [entry]}}


def _parts(value) -> list:
    return value if isinstance(value, list) else []


def browser_frames(raw: dict) -> list[dict]:
    """One Live frame in, zero or more browser frames out. Never raises.

    Unknown frames are dropped rather than forwarded: the renderer's contract is small
    and closed, and a new server field must not become an unhandled message type.
    """
    out: list[dict] = []
    if "setupComplete" in raw:
        out.append({"type": "ready"})

    if error := raw.get("error"):
        out.append({"type": "error",
                    "detail": str(error.get("message") or error)[:400],
                    "code": error.get("code")})

    content = raw.get("serverContent") or {}
    if content.get("interrupted"):
        # Server VAD already stopped generation; there is no upstream cancel to send.
        # The renderer's only job is to stop playing what it queued.
        out.append({"type": "speech_started"})
    for part in _parts((content.get("modelTurn") or {}).get("parts")):
        inline = part.get("inlineData") or {}
        if inline.get("data"):
            out.append({"type": "audio", "data": inline["data"],
                        "mime": inline.get("mimeType") or OUT_MIME})
        if text := (part.get("text") or "").strip():
            out.append({"type": "out_text", "text": text})
    if content.get("turnComplete"):
        out.append({"type": "turn_complete"})

    for field, kind in (("inputTranscription", "in_text"), ("outputTranscription", "out_text")):
        transcript = raw.get(field) or {}
        if text := (transcript.get("text") or ""):
            out.append({"type": kind, "text": text})

    if tool_call := raw.get("toolCall"):
        out.append({"type": "tool_call",
                    "calls": [{"id": c.get("id"), "name": c.get("name"),
                               "args": c.get("args") or {}}
                              for c in _parts(tool_call.get("functionCalls"))]})
    if cancelled := raw.get("toolCallCancellation"):
        out.append({"type": "tool_cancelled", "ids": list(cancelled.get("ids") or [])})

    if usage := raw.get("usageMetadata"):
        out.append({"type": "usage",
                    "total_tokens": usage.get("totalTokenCount"),
                    "input_tokens": usage.get("promptTokenCount"),
                    "output_tokens": usage.get("candidatesTokenCount")})

    if resumption := raw.get("sessionResumptionUpdate"):
        out.append({"type": "resumption", "handle": resumption.get("newHandle"),
                    "resumable": resumption.get("resumable")})

    if isinstance(raw.get("goAway"), dict):
        # Terminal: the socket dies ABORTED right after this. Surfaced as its own frame so
        # the user is told the call is ending, rather than watching it stop with no reason.
        out.append({"type": "go_away", "timeLeft": str(raw["goAway"].get("timeLeft") or "")})
    return out
