// hermes-gemini-live — voice control that lives IN the composer, beside core's own voice
// affordances. Uncompiled ESM, one self-contained file: the loader resolves only
// @hermes/plugin-sdk and react*, and JSX is unavailable in this format, so every element
// goes through createElement.
//
// Audio leaves this window as 16 kHz s16le frames over a sibling WebSocket; the Gemini
// key is never sent here and never read here. The relay owns it.
import { icons, resolveSiblingWsUrl } from '@hermes/plugin-sdk'
import { createElement as h, useEffect, useState } from 'react'

const RELAY_PATH = '/api/plugins/hermes-gemini-live/relay'
const SEND_RATE = 16000
const SEND_CHUNK = 1600
const DEFAULT_TUNING = { echoCancellation: true, noiseSuppression: true, autoGainControl: true, halfDuplex: false }

const WORKLET_SOURCE = `
class ResampleToPcm extends AudioWorkletProcessor {
  constructor () {
    super()
    this.queue = new Float32Array(${SEND_CHUNK * 4})
    this.length = 0
    this.carry = 0
  }
  push (frame) {
    if (this.length + frame.length > this.queue.length) {
      const grown = new Float32Array((this.queue.length + frame.length) * 2)
      grown.set(this.queue.subarray(0, this.length))
      this.queue = grown
    }
    this.queue.set(frame, this.length)
    this.length += frame.length
  }
  drain () {
    const step = sampleRate / ${SEND_RATE}
    const out = new Int16Array(Math.max(0, Math.floor(this.length / step)))
    for (let i = 0; i < out.length; i++) {
      const at = i * step + this.carry
      const low = Math.floor(at)
      const high = Math.min(low + 1, this.length - 1)
      const value = this.queue[low] * (1 - (at - low)) + this.queue[high] * (at - low)
      out[i] = Math.max(-1, Math.min(1, value)) * 32767
    }
    this.carry = (this.length / step) % 1
    this.queue.copyWithin(0, Math.floor(out.length * step))
    this.length -= Math.floor(out.length * step)
    return out
  }
  process (inputs) {
    const channel = inputs[0] && inputs[0][0]
    if (channel && channel.length) {
      this.push(Float32Array.prototype.slice.call(channel))
      if (this.length >= Math.ceil(${SEND_CHUNK} * (sampleRate / ${SEND_RATE}))) {
        const pcm = this.drain()
        if (pcm.length) this.port.postMessage({ pcm, peak: peakOf(pcm) }, [pcm.buffer])
      }
    }
    return true
  }
}
function peakOf (pcm) {
  let peak = 0
  for (let i = 0; i < pcm.length; i += 8) peak = Math.max(peak, Math.abs(pcm[i]))
  return peak / 32767
}
registerProcessor('pcm-out', ResampleToPcm)
`

function toBase64 (pcm) {
  const bytes = new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength)
  let binary = ''
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000))
  }
  return btoa(binary)
}

function fromBase64 (text) {
  const binary = atob(text)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return bytes
}

/**
 * Capture one mic stream into 16 kHz s16le chunks. Constraints come from the backend,
 * because echo cancellation is the difference between a call and a model that keeps
 * interrupting itself through a loudspeaker.
 */
async function startCapture (onChunk, tuning) {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      autoGainControl: tuning.autoGainControl,
      echoCancellation: tuning.echoCancellation,
      noiseSuppression: tuning.noiseSuppression,
      // An empty id means "whatever the system picked"; an exact id is what makes the
      // headset in the drawer actually get used.
      ...(tuning.inputDevice ? { deviceId: { exact: tuning.inputDevice } } : {})
    }
  })
  const context = new AudioContext()
  await context.resume()
  const source = context.createMediaStreamSource(stream)
  const sink = context.createGain()
  sink.gain.value = 0
  source.connect(sink)
  sink.connect(context.destination)

  const stop = () => {
    stream.getTracks().forEach((track) => track.stop())
    void context.close().catch(() => {})
  }

  try {
    const url = URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: 'text/javascript' }))
    await context.audioWorklet.addModule(url)
    URL.revokeObjectURL(url)
    const node = new AudioWorkletNode(context, 'pcm-out')
    node.port.onmessage = (event) => onChunk(event.data.pcm, event.data.peak)
    source.connect(node)
    return stop
  } catch {
    // Chromium without a worklet URL path: same math on the main thread, same contract.
    const node = context.createScriptProcessor(2048, 1, 1)
    let carry = 0
    node.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0)
      const step = context.sampleRate / SEND_RATE
      const count = Math.floor(input.length / step)
      if (!count) return
      const pcm = new Int16Array(count)
      for (let i = 0; i < count; i++) {
        const at = i * step + carry
        pcm[i] = Math.max(-1, Math.min(1, input[Math.min(Math.floor(at), input.length - 1)])) * 32767
      }
      carry = (input.length / step) % 1
      let peak = 0
      for (let i = 0; i < pcm.length; i += 8) peak = Math.max(peak, Math.abs(pcm[i]))
      onChunk(pcm, peak / 32767)
    }
    source.connect(node)
    node.connect(sink)
    return stop
  }
}

/** Playback for the model's 24 kHz speech, flushed the moment the user talks over it. */
function createPlayer () {
  const context = new AudioContext()
  const gain = context.createGain()
  const analyser = context.createAnalyser()
  analyser.fftSize = 512
  analyser.smoothingTimeConstant = 0.65
  const bins = new Uint8Array(analyser.frequencyBinCount)
  gain.connect(analyser)
  analyser.connect(context.destination)
  let next = 0
  const live = new Set()

  return {
    /** Route playback to one output device. Chromium-only; a refusal is reported, not fatal. */
    async setSink (id) {
      if (!id) return ''
      if (typeof context.setSinkId !== 'function') return 'Dieser Host kann kein Ausgabegerät wählen.'
      try {
        await context.setSinkId(id)
        return ''
      } catch {
        return 'Das Ausgabegerät ist nicht verfügbar.'
      }
    },
    /** True while scheduled audio is still queued — a far better half-duplex signal than
     *  "the last frame arrived N ms ago", which goes false the moment the model pauses. */
    speaking () {
      return live.size > 0 || next - context.currentTime > 0.05
    },
    play (bytes) {
      const samples = new Int16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 1)
      const buffer = context.createBuffer(1, samples.length, 24000)
      const channel = buffer.getChannelData(0)
      for (let i = 0; i < samples.length; i++) channel[i] = samples[i] / 32767
      // Backpressure guard (dandacompany): if the queue is more than 15 s ahead of the
      // clock the stream has run away, and playing it out would put the voice minutes
      // behind the conversation. Drop the backlog and resynchronise instead.
      if (next - context.currentTime > 15) {
        live.forEach((queued) => {
          try {
            queued.stop()
          } catch {
            /* already finished */
          }
        })
        live.clear()
        next = context.currentTime
      }
      const node = context.createBufferSource()
      node.buffer = buffer
      node.connect(gain)
      next = Math.max(next, context.currentTime)
      node.start(next)
      next += buffer.duration
      live.add(node)
      node.onended = () => live.delete(node)
    },
    level () {
      analyser.getByteFrequencyData(bins)
      let peak = 0
      for (let i = 0; i < bins.length; i++) peak = Math.max(peak, bins[i])
      return Math.sqrt(peak / 255)
    },
    flush () {
      live.forEach((node) => {
        try {
          node.stop()
        } catch {
          /* already finished */
        }
      })
      live.clear()
      next = 0
    },
    close () {
      this.flush()
      void context.close().catch(() => {})
    }
  }
}

/**
 * Chrome copied from core's own voice surfaces
 * (`apps/desktop/src/app/chat/composer/voice-activity.tsx`): the same five-bar meter with
 * weights [0.5, 0.78, 1, 0.78, 0.5], the same mono `m:ss` clock, the same primary-tinted
 * disc. The row itself keeps core's geometry but no fill of its own — it sits on the
 * composer, so a second box inside it reads as a panel rather than as the app's voice lane.
 */
const PILL = 'flex h-8 items-center gap-2 px-1 text-xs text-muted-foreground'
const DISC = 'flex size-5 shrink-0 items-center justify-center rounded-full bg-primary/15 text-primary'
const BAR_WEIGHTS = [0.5, 0.78, 1, 0.78, 0.5]

function formatElapsed (seconds) {
  const safe = Math.max(0, Math.floor(seconds))
  const minutes = Math.floor(safe / 60)
  return `${minutes}:${(safe % 60).toString().padStart(2, '0')}`
}

function VoiceLevelBars ({ level, active }) {
  const normalized = Math.max(0, Math.min(level, 1))
  return h('div', { 'aria-hidden': 'true', className: 'flex h-4 items-center gap-0.5' },
    ...BAR_WEIGHTS.map((weight, index) => {
      const height = active ? 0.25 + Math.min(0.68, normalized * weight) : 0.25
      return h('span', {
        key: index,
        className: ['w-0.5 rounded-full bg-current transition-[height,opacity] duration-100 ease-out',
          active ? 'opacity-80' : 'animate-pulse opacity-45'].join(' '),
        style: { height: `${height * 100}%` }
      })
    }))
}

/**
 * The call is owned at module scope, not by the component. A `composer.actions` row is
 * remounted whenever the chat route changes — opening a session, switching pane — and a
 * call that dies with its row dies while the user is still talking, so the row only
 * subscribes and renders. Nothing here tears the socket down except Stop, a dropped
 * relay, or a frame that says the call failed.
 */
const call = {
  state: 'idle', detail: '', level: 0, elapsed: 0, speaking: false, pending: [], waiting: [],
  status: null,
  muted: false,
  socket: null, player: null, stopCapture: null,
  lastAudio: 0, lastInText: 0, startedAt: 0, taskAt: 0, micPeak: 0,
  tuning: DEFAULT_TUNING, raf: 0, listeners: new Set()
}

function publish () {
  for (const listener of call.listeners) listener()
}

function useCall () {
  const [, rerender] = useState(0)
  useEffect(() => {
    const listener = () => rerender((n) => n + 1)
    call.listeners.add(listener)
    return () => call.listeners.delete(listener)
  }, [])
  return call
}

const message = (error) => (error && error.message ? error.message : String(error))

// One loop drives meter, speaking and the clock for the life of the call — not of the row.
function beginMeter () {
  if (call.raf) return
  const tick = () => {
    const now = Date.now()
    const played = call.player ? call.player.level() : 0
    call.speaking = now - call.lastAudio < 400 || played > 0.02
    call.level = call.speaking ? played : call.micPeak
    call.elapsed = (now - call.startedAt) / 1000
    publish()
    call.raf = window.requestAnimationFrame(tick)
  }
  call.raf = window.requestAnimationFrame(tick)
}

/** Stop everything the call holds. `note` replaces the visible reason; omit it to keep. */
function tearDown (note) {
  if (call.raf) {
    window.cancelAnimationFrame(call.raf)
    call.raf = 0
  }
  const ws = call.socket
  call.socket = null
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'close' }))
    ws.close()
  }
  if (call.stopCapture) {
    call.stopCapture()
    call.stopCapture = null
  }
  call.player?.flush()
  call.state = 'idle'
  call.level = 0
  call.speaking = false
  call.pending = []
  call.waiting = []
  call.lastAudio = 0
  call.lastInText = 0
  call.micPeak = 0
  call.taskAt = 0
  call.muted = false
  if (note !== undefined) call.detail = note
  publish()
}

function toggleMute () {
  call.muted = !call.muted
  publish()
}

async function loadStatus (rest) {
  try {
    call.status = await rest('/status')
  } catch (error) {
    // Keep the backend's own words: "404 Plugin not found" means this home has the
    // plugin disabled, which is a different fix from a missing key.
    call.status = { ok: false, detail: 'backend: ' + message(error) }
  }
  const devices = (call.status && call.status.devices) || {}
  call.tuning = {
    ...DEFAULT_TUNING,
    ...((call.status && call.status.audio) || {}),
    inputDevice: devices.input || '',
    outputDevice: devices.output || ''
  }
  publish()
}

function appendChunk (bytes, peak) {
  // Mute drops the frame at the last step before the socket, so the Live session, its
  // context and any run in flight all stay alive — the model just stops hearing the room.
  // There is no wire-level equivalent: pushToTalk, serverVad and activityEnd all close 1007.
  if (call.muted) return
  call.micPeak = peak
  const tuning = call.tuning
  // Half-duplex: when a loudspeaker beats the canceller, the model's own reply arrives
  // at the mic and Live reads it as a listener interrupting. Muting the uplink while it
  // speaks costs talk-over interruption and buys back a call that actually finishes.
  // Half-duplex asks "is audio actually still queued?", not "did a frame arrive lately":
  // a model that pauses mid-sentence would otherwise un-mute the mic into its own tail.
  const speakingNow = call.player && typeof call.player.speaking === 'function'
    ? call.player.speaking()
    : Date.now() - call.lastAudio < 600
  if (tuning.halfDuplex && speakingNow) return
  const ws = call.socket
  if (!ws || ws.readyState !== WebSocket.OPEN) return
  ws.send(JSON.stringify({ type: 'audio', data: toBase64(bytes) }))
}

function receive (event) {
  let frame
  try {
    frame = JSON.parse(event.data)
  } catch {
    return
  }
  if (frame.type === 'audio') {
    call.lastAudio = Date.now()
    call.player?.play(fromBase64(frame.data))
  } else if (frame.type === 'speech_started') {
    call.player?.flush()
    call.lastAudio = 0
  } else if (frame.type === 'in_text' && frame.text) {
    // The user's own words are the only honest start of a "thinking" window: an
    // extended-thinking model is silent while it works, and silence alone reads dead.
    call.lastInText = Date.now()
  } else if (frame.type === 'call_session') {
    // The relay minted (or reused) the session this call writes into. Kept for the panel
    // so "where did that call go" has an answer without opening the sidebar.
    call.sessionId = frame.id
    publish()
  } else if (frame.type === 'task_started') {
    if (!call.taskAt) call.taskAt = Date.now()
    if (!call.pending.includes(frame.id)) call.pending = [...call.pending, frame.id]
    publish()
  } else if (frame.type === 'task_done') {
    call.pending = call.pending.filter((id) => id !== frame.id)
    call.waiting = call.waiting.filter((id) => id !== frame.id)
    if (frame.state && frame.state !== 'completed') {
      call.detail = 'Hermes could not finish that task (' + frame.state + ')'
    }
    publish()
  } else if (frame.type === 'task_needs_input') {
    // A run parked on an approval is not "still working" — it is waiting for a human, and
    // the difference is the one thing the user cannot hear while the model is quiet.
    if (!call.waiting.includes(frame.id)) call.waiting = [...call.waiting, frame.id]
    call.detail = frame.question || 'Hermes is asking you something'
    publish()
  } else if (frame.type === 'task_update') {
    if (frame.action === 'stop') {
      call.pending = call.pending.filter((id) => id !== frame.id)
      call.waiting = call.waiting.filter((id) => id !== frame.id)
    } else {
      call.waiting = call.waiting.filter((id) => id !== frame.id)
    }
    publish()
  } else if (frame.type === 'task_failed') {
    call.detail = frame.detail || 'Hermes refused that'
    publish()
  } else if (frame.type === 'lane') {
    call.detail = 'Hermes work is unavailable: ' + (frame.detail || frame.status)
    publish()
  } else if (frame.type === 'go_away') {
    call.detail = 'Gemini is ending this session' +
      (frame.timeLeft ? ' (' + frame.timeLeft + ')' : '') + ' — press Start to go on'
    publish()
  } else if (frame.type === 'error') {
    tearDown(String(frame.detail || 'the voice call failed'))
  }
}

/**
 * The user's own answer to a run parked on an approval. This press is the only thing that
 * can approve one: the voice model has no approve verb, because it also hears the room and
 * reads run output, and neither may unlock a command.
 */
function answerApproval (choice) {
  const id = call.waiting[0]
  const ws = call.socket
  if (!id || !ws || ws.readyState !== WebSocket.OPEN) return
  ws.send(JSON.stringify({ type: 'approval', id, choice }))
}

/**
 * The relay socket URL. A Desktop in native-auth mode mints plugin sockets through its own
 * bridge, and the upgrade then carries a reserved grant header — the only credential a
 * renderer can present on an /api/plugins path. resolveSiblingWsUrl alone swaps the path on a
 * /api/ws ticket, which this host answers with 403 ("the relay socket failed"). Token-mode and
 * dashboard hosts keep the sibling mint, which carries ?token=.
 */
async function relayUrl () {
  const bridge = globalThis.hermesDesktop
  if (bridge && typeof bridge.getPluginWsUrl === 'function') {
    try {
      const connection = typeof bridge.getConnection === 'function' ? await bridge.getConnection() : null
      if (!connection || connection.authMode === 'native' || !connection.authMode) {
        const minted = await bridge.getPluginWsUrl({
          pluginId: 'hermes-gemini-live',
          path: RELAY_PATH.replace('/api/plugins/hermes-gemini-live', '')
        })
        if (typeof minted === 'string' && minted) return minted
      }
    } catch {
      // fall through: the sibling mint is still right for a token-mode host
    }
  }
  return resolveSiblingWsUrl({}, RELAY_PATH)
}

async function start (rest) {
  if (call.state !== 'idle') return
  call.state = 'connecting'
  call.detail = ''
  call.pending = []
  call.waiting = []
  call.level = 0
  call.speaking = false
  call.lastAudio = 0
  call.lastInText = 0
  call.taskAt = 0
  call.startedAt = Date.now()
  beginMeter()
  publish()
  void loadStatus(rest)
  call.player = call.player || createPlayer()
  // The speaker choice rides the AudioContext, not the microphone: routing it wrong is
  // silent (the model talks into a device nobody is listening to), so it is reported.
  if (call.tuning.outputDevice && typeof call.player.setSink === 'function') {
    void call.player.setSink(call.tuning.outputDevice).then(note => {
      if (note) { call.detail = note; publish() }
    })
  }
  let url
  try {
    url = await relayUrl()
  } catch (error) {
    tearDown(message(error))
    return
  }
  const ws = new WebSocket(url)
  call.socket = ws
  ws.onopen = async () => {
    call.state = 'live'
    publish()
    try {
      call.stopCapture = await startCapture(appendChunk, call.tuning)
    } catch (error) {
      call.detail = 'microphone unavailable: ' + message(error)
      publish()
    }
  }
  ws.onmessage = receive
  ws.onclose = () => {
    // Only the live socket matters: Stop replaces it before its own close arrives. Keep
    // the frame that caused it (goAway, an error) when the server told us one.
    if (call.socket !== ws) return
    tearDown(call.detail || 'the voice call dropped')
  }
  ws.onerror = () => {
    call.detail = 'the relay socket failed'
    publish()
  }
}

function TalkControl ({ rest }) {
  useCall()
  const status = call.status
  const blocked = !!(status && !status.ok)

  useEffect(() => {
    if (!status) void loadStatus(rest)
  }, [rest, status])

  if (call.state === 'idle') {
    return h('button', {
      type: 'button',
      title: call.detail || (blocked && status ? status.detail : 'Start a Gemini Live call'),
      'aria-label': 'Start Gemini Live call',
      disabled: blocked,
      onClick: () => void start(rest),
      className: 'inline-flex size-7 shrink-0 items-center justify-center rounded-full ' +
        'text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:opacity-50'
    }, h(icons.Mic, { className: icons.iconSize.sm }))
  }

  // A Hermes run takes tens of seconds and an extended-thinking model goes silent while
  // it works, so neither gap may look like a frozen app. Each gets its own label, clock
  // and disc, all built from chrome the app already uses: a spinning Loader2 to mean
  // "not idle", and the meter's own pulse (core's `animate-pulse opacity-45`) to mean
  // "alive, nothing coming out".
  const now = Date.now()
  const pending = call.pending
  const working = pending.length > 0
  // A task parked on an approval is still "working" by the count, but saying so hides the
  // one thing the user can act on — so the ask wins the label.
  const waiting = call.waiting.length > 0
  const thinking = call.state === 'live' && !working && !call.speaking &&
    call.lastInText > 0 && now - call.lastInText < 20000
  const busy = working || thinking || call.state === 'connecting'
  const counted = working ? call.taskAt : (thinking ? call.lastInText : call.startedAt)
  const shownSeconds = working || thinking ? (now - counted) / 1000 : call.elapsed
  const label = call.state === 'connecting' ? 'Connecting'
    : waiting ? (call.waiting.length > 1 ? `Hermes needs you · ${call.waiting.length} tasks`
      : 'Hermes needs you')
      : working ? (pending.length > 1 ? `Hermes is working · ${pending.length} tasks` : 'Hermes is working')
        : thinking ? 'Thinking'
          : call.speaking ? 'Speaking'
            : call.muted ? 'Muted'
              : call.detail || 'Listening'

  return h('div', { 'aria-live': 'polite', role: 'status', className: PILL },
    h('div', { className: DISC },
      busy
        ? h(icons.Loader2, { className: ['animate-spin', icons.iconSize.xs].join(' ') })
        : h(call.muted ? icons.MicOff : (call.speaking ? icons.Volume2 : icons.Mic),
          { className: icons.iconSize.xs })),
    h('div', { className: 'flex min-w-0 flex-1 items-center gap-2' },
      h('span', { className: 'truncate font-medium text-foreground/85' }, label),
      h('span', { 'aria-hidden': 'true', className: 'font-mono text-[0.6875rem] text-muted-foreground/85' },
        formatElapsed(shownSeconds))),
    h(VoiceLevelBars, { active: !thinking, level: call.level }),
    waiting && h('button', {
      type: 'button',
      'aria-label': 'Approve what Hermes is asking, once',
      title: call.detail || 'Approve once',
      onClick: () => answerApproval('approve'),
      className: 'inline-flex h-6 shrink-0 items-center rounded-full px-2 text-[0.6875rem] ' +
        'bg-muted text-foreground transition-colors hover:bg-muted/70'
    }, 'Approve'),
    waiting && h('button', {
      type: 'button',
      'aria-label': 'Deny what Hermes is asking',
      title: call.detail || 'Deny',
      onClick: () => answerApproval('deny'),
      className: 'inline-flex h-6 shrink-0 items-center rounded-full px-2 text-[0.6875rem] ' +
        'text-muted-foreground transition-colors hover:bg-muted hover:text-foreground'
    }, 'Deny'),
    h('button', {
      type: 'button',
      'aria-label': call.muted ? 'Unmute the microphone' : 'Mute the microphone',
      'aria-pressed': call.muted,
      title: call.muted
        ? 'The model still hears nothing; tasks it started keep running'
        : 'Stop hearing the room without hanging up',
      onClick: toggleMute,
      className: 'inline-flex size-6 shrink-0 items-center justify-center rounded-full ' +
        (call.muted ? 'bg-muted text-foreground' : 'text-muted-foreground') +
        ' transition-colors hover:bg-muted hover:text-foreground'
    }, h(call.muted ? icons.MicOff : icons.Mic, { className: icons.iconSize.xs })),
    h('button', {
      type: 'button',
      'aria-label': 'Stop Gemini Live call',
      onClick: () => tearDown(),
      className: 'inline-flex h-6 shrink-0 items-center gap-1 rounded-full px-2 text-[0.6875rem] ' +
        'text-muted-foreground transition-colors hover:bg-muted hover:text-foreground'
    }, h(icons.VolumeX, { className: icons.iconSize.xs }), 'Stop'))
}

/* ─────────────────────────────── settings page ───────────────────────────────
 * The form the plugin contributes to Settings ▸ Plugins (`registerSettingsPage`): the app
 * renders it as its own rail entry with a gear in the plugin's card, so the preferences sit
 * where every other plugin's do — outside the chat composer, so opening them can never
 * tear down a running call.
 *
 * One control per field, chosen by the schema the backend reports: a dropdown for every
 * enumerated choice (model, voice, protocol depth, delegation target, long-call policy),
 * a number for the two millisecond knobs, a checkbox for audio flags, a textarea for the
 * extra instruction. Nothing here is free text that the backend would have to guess at.
 */
const CONTROL_CLASS = 'h-8 w-full rounded-md border border-(--ui-stroke-secondary) ' +
  'bg-transparent px-2 text-sm text-foreground focus-visible:outline-none ' +
  'focus-visible:ring-1 focus-visible:ring-(--ui-accent)'

function FieldRow ({ field, value, onChange }) {
  let control
  if (field.kind === 'enum') {
    control = h('select', {
      className: CONTROL_CLASS,
      value: value == null ? field.default : value,
      onChange: event => onChange(event.target.value)
    }, (field.choices || []).map(choice => h('option', { key: choice, value: choice }, choice)))
  } else if (field.kind === 'bool') {
    control = h('label', { className: 'inline-flex items-center gap-2 text-sm' },
      h('input', {
        type: 'checkbox',
        checked: value == null ? Boolean(field.default) : Boolean(value),
        onChange: event => onChange(event.target.checked)
      }),
      h('span', { className: 'text-muted-foreground' }, value ? 'an' : 'aus'))
  } else if (field.kind === 'int') {
    control = h('input', {
      type: 'number',
      className: CONTROL_CLASS,
      min: field.min,
      max: field.max,
      value: value == null ? field.default : value,
      onChange: event => onChange(Number(event.target.value))
    })
  } else {
    control = h('textarea', {
      className: CONTROL_CLASS + ' h-auto py-2',
      rows: 3,
      maxLength: field.max_len || 2000,
      value: value == null ? field.default : value,
      onChange: event => onChange(event.target.value)
    })
  }
  return h('div', { className: 'grid gap-1 border-b border-(--ui-stroke-tertiary) pb-3' },
    h('div', { className: 'text-sm font-medium text-foreground' }, field.label),
    field.help ? h('div', { className: 'text-xs text-muted-foreground' }, field.help) : null,
    control)
}


/** Microphones and speakers the browser can see. Labels are empty until microphone
 *  permission has been granted once, so the fallback is the kind plus its index. */
function useAudioDevices (active) {
  const [devices, setDevices] = useState({ inputs: [], outputs: [] })
  useEffect(() => {
    if (!active || !navigator.mediaDevices || typeof navigator.mediaDevices.enumerateDevices !== 'function') {
      return
    }
    let live = true
    const read = () => navigator.mediaDevices.enumerateDevices().then(list => {
      if (!live) return
      const pick = kind => list
        .filter(device => device.kind === kind)
        .map((device, index) => ({
          id: device.deviceId,
          label: device.label || (kind === 'audioinput'
            ? `Mikrofon ${index + 1}`
            : `Ausgabe ${index + 1}`)
        }))
      setDevices({ inputs: pick('audioinput'), outputs: pick('audiooutput') })
    }).catch(() => undefined)
    void read()
    navigator.mediaDevices.addEventListener?.('devicechange', read)
    return () => {
      live = false
      navigator.mediaDevices.removeEventListener?.('devicechange', read)
    }
  }, [active])
  return devices
}

function DeviceRow ({ label, value, options, onChange }) {
  return h('div', { className: 'grid gap-1 border-b border-(--ui-stroke-tertiary) pb-3' },
    h('div', { className: 'text-sm font-medium text-foreground' }, label),
    h('select', {
      className: CONTROL_CLASS,
      value: value || '',
      onChange: event => onChange(event.target.value)
    },
    h('option', { value: '' }, 'Systemstandard'),
    options.map(option => h('option', { key: option.id, value: option.id }, option.label))))
}

function money (value) {
  return '$' + (Math.round(value * 1000) / 1000).toFixed(3)
}

function UsageBlock ({ usage }) {
  if (!usage) return null
  const day = usage.last_24h || {}
  const all = usage.all_time || {}
  const line = (label, bucket) => h('div', { className: 'flex items-baseline justify-between gap-3 text-sm' },
    h('span', { className: 'text-muted-foreground' }, label),
    h('span', { className: 'font-mono text-[0.75rem] text-foreground' },
      `${bucket.calls || 0} Anrufe · ${Math.round((bucket.duration_ms || 0) / 1000)} s · ` +
      `${bucket.input || 0}+${bucket.output || 0} tok · ${money(bucket.usd || 0)}` +
      (bucket.unpriced ? ` (${bucket.unpriced} ohne Preis)` : '')))
  return h('div', { className: 'grid gap-2 rounded-lg border border-(--ui-stroke-tertiary) p-3' },
    h('div', { className: 'text-sm font-medium text-foreground' }, 'Verbrauch'),
    line('Letzte 24 Stunden', day),
    line('Gesamt', all),
    h('div', { className: 'text-[0.6875rem] text-(--ui-text-quaternary)' },
      `Schätzung aus veröffentlichten Preisen (${usage.price_date}); keine Rechnung. ` +
      'Delegierte Hermes-Arbeit ist nicht enthalten.'))
}

function SettingsPage ({ rest }) {
  const [fields, setFields] = useState([])
  const [values, setValues] = useState({})
  const [draft, setDraft] = useState({})
  const [status, setStatus] = useState('loading')
  const [note, setNote] = useState('')
  const [usage, setUsage] = useState(null)
  const devices = useAudioDevices(status !== 'error')

  useEffect(() => {
    let live = true
    rest('/settings').then(payload => {
      if (!live) return
      setFields(payload.fields || [])
      setValues(payload.values || {})
      setDraft(payload.values || {})
      setStatus('ready')
    }).catch(error => {
      if (!live) return
      setStatus('error')
      setNote(message(error))
    })
    rest('/usage').then(payload => { if (live) setUsage(payload) }).catch(() => undefined)
    return () => { live = false }
  }, [rest])

  const dirty = fields.some(field => draft[field.name] !== values[field.name])

  async function save () {
    setStatus('saving')
    setNote('')
    try {
      const payload = await rest('/settings', { method: 'PUT', body: draft })
      setValues(payload.values || {})
      setDraft(payload.values || {})
      setFields(payload.fields || fields)
      setStatus('ready')
      setNote('Gespeichert. Gilt ab dem nächsten Anruf.')
    } catch (error) {
      setStatus('ready')
      setNote(message(error))
    }
  }

  async function reloadModules () {
    setStatus('saving')
    setNote('')
    try {
      const payload = await rest('/reload', { method: 'POST' })
      setStatus('ready')
      setNote('Module neu geladen (' + (payload.reloaded || []).length + '). Gilt ab dem nächsten Anruf.')
    } catch (error) {
      setStatus('ready')
      setNote(message(error))
    }
  }

  if (status === 'loading') {
    return h('div', { className: 'p-6 text-sm text-muted-foreground' }, 'Einstellungen werden geladen …')
  }

  return h('div', { className: 'mx-auto flex w-full max-w-2xl flex-col gap-4 p-6' },
    h('div', { className: 'grid gap-1' },
      h('h1', { className: 'text-lg font-semibold text-foreground' }, 'Gemini Live'),
      h('p', { className: 'text-xs text-muted-foreground' },
        'Stimme, Modell, Protokolltiefe und das Verhalten der delegierten Aufgaben. ' +
        'Änderungen gelten für den nächsten Anruf.')),
    h('div', { className: 'grid gap-4' },
      h(DeviceRow, {
        label: 'Mikrofon',
        value: draft.input_device,
        options: devices.inputs,
        onChange: next => setDraft(previous => ({ ...previous, input_device: next }))
      }),
      h(DeviceRow, {
        label: 'Lautsprecher',
        value: draft.output_device,
        options: devices.outputs,
        onChange: next => setDraft(previous => ({ ...previous, output_device: next }))
      })),
    h('div', { className: 'grid gap-4' },
      // The two device ids have their own pickers above; a textarea would be a second,
      // worse way to set the same thing.
      fields.filter(field => field.name !== 'input_device' && field.name !== 'output_device')
        .map(field => h(FieldRow, {
        key: field.name,
        field,
        value: draft[field.name],
        onChange: next => setDraft(previous => ({ ...previous, [field.name]: next }))
      }))),
    h('div', { className: 'flex items-center gap-3' },
      h('button', {
        type: 'button',
        disabled: !dirty || status === 'saving',
        onClick: () => void save(),
        className: 'inline-flex h-8 items-center rounded-md bg-(--ui-accent) px-3 text-sm ' +
          'font-medium text-(--ui-accent-foreground) disabled:opacity-40'
      }, status === 'saving' ? 'Speichert …' : 'Speichern'),
      dirty
        ? h('button', {
          type: 'button',
          onClick: () => setDraft({ ...values }),
          className: 'text-xs text-muted-foreground underline-offset-2 hover:underline'
        }, 'Verwerfen')
        : null,
      h('button', {
        type: 'button',
        title: 'Code-Änderungen dieses Plugins laden, ohne Hermes neu zu starten',
        disabled: status === 'saving',
        onClick: () => void reloadModules(),
        className: 'text-xs text-muted-foreground underline-offset-2 hover:underline disabled:opacity-40'
      }, 'Module neu laden'),
      note ? h('span', { className: 'text-xs text-muted-foreground' }, note) : null),
    h(UsageBlock, { usage }),
    h('p', { className: 'text-[0.6875rem] text-(--ui-text-quaternary)' },
      'Ablage: ' + 'plugin-data/hermes-gemini-live/settings.json (überschreibt GEMINI_LIVE_*-Variablen).'))
}

export default {
  id: 'hermes-gemini-live',
  name: 'Gemini Live',
  defaultEnabled: true,
  register (ctx) {
    ctx.register({
      id: 'talk',
      area: 'composer.actions',
      order: 40,
      render: () => h(TalkControl, { rest: ctx.rest })
    })
    // Settings ▸ Plugins is this app's home for plugin preferences (WoW-AddOns style):
    // one rail entry per plugin, own sub-pages beneath it. Optional chaining because a
    // host older than the settings-page API must still get the call button.
    ctx.registerSettingsPage?.({
      id: 'settings',
      title: 'Gemini Live',
      icon: 'mic',
      order: 40,
      render: () => h(SettingsPage, { rest: ctx.rest })
    })
  }
}
