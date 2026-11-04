"""The spoken transcript: delta merging, the context a task travels with, and rows."""

from hermes_gemini_live import transcript


def test_same_speaker_deltas_merge_into_one_utterance():
    log = transcript.Transcript()
    log.add(transcript.USER, "stell bitte ")
    log.add(transcript.USER, "einen Termin ein")
    log.turn_complete()
    rows = log.pending_rows()
    assert rows == [{"role": "user", "content": "stell bitte einen Termin ein",
                     "at": rows[0]["at"]}]


def test_a_speaker_change_starts_a_new_fragment():
    log = transcript.Transcript()
    log.add(transcript.USER, "hallo")
    log.add(transcript.ASSISTANT, "hi")
    log.turn_complete()
    assert [row["role"] for row in log.pending_rows()] == ["user", "assistant"]


def test_interruption_settles_what_the_model_managed_to_say():
    log = transcript.Transcript()
    log.add(transcript.ASSISTANT, "Ich habe drei")
    log.interrupted()
    rows = log.pending_rows()
    assert rows and rows[0]["role"] == "assistant"
    assert rows[0]["content"] == "Ich habe drei"


def test_rows_are_handed_out_once():
    log = transcript.Transcript()
    log.add(transcript.USER, "eins")
    log.turn_complete()
    assert len(log.pending_rows()) == 1
    assert log.pending_rows() == []


def test_unsaid_text_is_never_written():
    log = transcript.Transcript()
    log.add(transcript.USER, "   ")
    log.turn_complete()
    assert log.pending_rows() == []


def test_context_is_labelled_and_capped():
    log = transcript.Transcript()
    log.add(transcript.USER, "mach bitte die Rechnung")
    log.add(transcript.ASSISTANT, "Ich schaue nach")
    window = log.context(max_chars=6000)
    assert "User: mach bitte die Rechnung" in window
    assert "Hermes: Ich schaue nach" in window


def test_the_context_window_never_exceeds_its_cap():
    log = transcript.Transcript()
    for index in range(40):
        log.add(transcript.USER if index % 2 == 0 else transcript.ASSISTANT,
                f"satz nummer {index} " + "x" * 300)
        log.turn_complete()
    window = log.context(max_chars=6000)
    assert len(window) <= 6000
    assert "satz nummer 39" in window


def test_context_block_names_itself_so_the_worker_does_not_mistake_it_for_the_task():
    block = transcript.labelled_block("User: mach das")
    assert "spoken exchange" in block
    assert "User: mach das" in block
    assert transcript.labelled_block("") == ""


def test_the_ring_drops_the_oldest_fragments():
    log = transcript.Transcript(max_fragments=10, keep=5)
    for index in range(12):
        log.add(transcript.USER, f"satz {index}")
        log.turn_complete()
    # The ring trims only when it overflows, so it stays within the cap and the oldest
    # sentences are the ones that went.
    assert len(log) <= 10
    assert "satz 11" in log.context()
    assert "satz 0" not in log.context()
