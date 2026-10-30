"""
voice/read_along.py – Real-time browser-native Web Speech API (speechSynthesis) read-along component.

Converts chatbot responses into structured HTML with synchronized word-level highlighting
and browser-native audio controls (Play, Pause, Resume, Stop, Voice selector, Speed control).
"""

from __future__ import annotations

import html
import json
import logging
import re
from typing import Tuple

logger = logging.getLogger(__name__)


def extract_spoken_content(text: str) -> str:
    """
    Extract only the main medical answer from response text,
    stripping repeated medical disclaimers and source citation sections.
    """
    if not text:
        return ""

    cleaned = text

    # Strip medical disclaimer blocks
    disclaimer_markers = [
        "---\n\n⚠️ DISCLAIMER",
        "---\n\nDISCLAIMER",
        "⚠️ DISCLAIMER",
        "⚠️ **DISCLAIMER",
        "DISCLAIMER:",
        "⚠️ Disclaimer",
    ]
    for marker in disclaimer_markers:
        if marker in cleaned:
            cleaned = cleaned.split(marker)[0]

    # Strip source citations
    source_markers = [
        "📖 Sources",
        "### Sources",
        "## Sources",
        "**Sources:**",
        "Sources:",
        "[Source:",
    ]
    for marker in source_markers:
        if marker in cleaned:
            cleaned = cleaned.split(marker)[0]

    # Strip multi-line code blocks
    cleaned = re.sub(r"```[\s\S]*?```", "", cleaned)

    # Convert Setext headings so divider rows are converted to standard markdown headings
    cleaned = re.sub(r"(?m)^([^\n#\-*•`~=\s][^\n]*)\n\s*={2,}\s*$", r"## \1", cleaned)
    cleaned = re.sub(r"(?m)^([^\n#\-*•`~=\s][^\n]*)\n\s*-{2,}\s*$", r"### \1", cleaned)

    # Strip standalone divider rows (e.g. '====', '----', '____', '****', '~~~~')
    cleaned = re.sub(r"(?m)^\s*[-=_*~]{2,}\s*$", "", cleaned)
    cleaned = re.sub(r"[-=_*~]{2,}", " ", cleaned)

    # Strip decorative Unicode emoji ranges
    cleaned = re.sub(r"[\U00010000-\U0010ffff]", "", cleaned)
    cleaned = re.sub(r"[\u2600-\u27bf\u2300-\u23ff\u2b50\u200d\u200c]", "", cleaned)

    return cleaned.strip()


def prepare_read_along_data(text: str) -> Tuple[str, str, int]:
    """
    Parse markdown response into plain speech text and synchronized HTML word spans.

    Parameters
    ----------
    text:
        Raw markdown response string.

    Returns
    -------
    Tuple of:
      - plain_speech_text: clean text string for SpeechSynthesisUtterance.
      - html_body: HTML preserving headings, paragraphs, and list items with word spans.
      - word_count: total number of words.
    """
    spoken_text = extract_spoken_content(text)
    if not spoken_text:
        fallback = "No spoken answer available."
        return fallback, f"<p class='read-along-p'>{fallback}</p>", 4

    lines = [line.strip() for line in spoken_text.split("\n")]

    plain_words = []
    html_blocks = []
    in_list = False
    char_offset = 0
    word_index = 0

    i = 0
    num_lines = len(lines)
    while i < num_lines:
        line = lines[i]
        i += 1

        if not line:
            if in_list:
                html_blocks.append("</ul>")
                in_list = False
            continue

        # Skip divider rows (====, ----, ____, ****, ~~~~)
        if re.match(r"^[-=_*~]{2,}$", line):
            continue

        # Check if next line is a Setext heading underline
        is_setext_h1 = False
        is_setext_h2 = False
        if i < num_lines and lines[i]:
            next_line = lines[i]
            if re.match(r"^={2,}$", next_line):
                is_setext_h1 = True
                i += 1  # Consume the underline row
            elif re.match(r"^-{2,}$", next_line) and not line.startswith(("-", "*", "•")):
                is_setext_h2 = True
                i += 1  # Consume the underline row

        # Detect heading or list
        is_h3 = line.startswith("###") or is_setext_h2
        is_h2 = ((line.startswith("##") and not line.startswith("###")) or is_setext_h1)
        is_h1 = line.startswith("#") and not line.startswith("##") and not line.startswith("###")
        is_bullet = line.startswith(("-", "*", "•")) or bool(re.match(r"^\d+\.", line))

        # Clean line markup
        content = re.sub(r"^#{1,6}\s*", "", line)
        content = re.sub(r"^\s*[-*•]\s+", "", content)
        content = re.sub(r"^\s*\d+\.\s+", "", content)

        # Strip bold/italic/backtick markup for tokenization
        content = re.sub(r"\*\*([^*]+)\*\*", r"\1", content)
        content = re.sub(r"\*([^*]+)\*", r"\1", content)
        content = re.sub(r"__([^_]+)__", r"\1", content)
        content = re.sub(r"_([^_]+)_", r"\1", content)
        content = re.sub(r"`([^`]+)`", r"\1", content)
        content = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", content)
        content = content.strip()

        if not content:
            continue

        raw_words = content.split()
        # Filter out standalone symbol tokens (e.g. "====", "----", "***")
        words = [w for w in raw_words if any(c.isalnum() for c in w)]
        if not words:
            continue

        # Tag selection
        if is_h1 or is_h2:
            tag = "h4"
            if in_list:
                html_blocks.append("</ul>")
                in_list = False
        elif is_h3:
            tag = "h5"
            if in_list:
                html_blocks.append("</ul>")
                in_list = False
        elif is_bullet:
            tag = "li"
            if not in_list:
                html_blocks.append("<ul class='read-along-list'>")
                in_list = True
        else:
            tag = "p"
            if in_list:
                html_blocks.append("</ul>")
                in_list = False

        spans = []
        for w in words:
            start = char_offset
            end = start + len(w)
            w_esc = html.escape(w)
            spans.append(
                f'<span class="speech-word" id="sw_{word_index}" data-start="{start}" data-end="{end}">{w_esc}</span>'
            )
            plain_words.append(w)
            char_offset = end + 1
            word_index += 1

        inner = " ".join(spans)
        html_blocks.append(f"<{tag} class='read-along-{tag}'>{inner}</{tag}>")

    if in_list:
        html_blocks.append("</ul>")

    plain_speech = " ".join(plain_words)
    return plain_speech, "\n".join(html_blocks), word_index


def generate_read_along_component_html(text: str, component_id: str = "ra_0") -> str:
    """
    Generate self-contained HTML/CSS/JS for the browser-native read-along player.

    Parameters
    ----------
    text:
        Assistant markdown response text.
    component_id:
        Unique identifier for the component instance.

    Returns
    -------
    Complete HTML string suitable for st.components.v1.html().
    """
    plain_speech, html_body, word_count = prepare_read_along_data(text)
    plain_speech_json = json.dumps(plain_speech)
    component_id_json = json.dumps(component_id)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
  :root {{
    --bg-card: #f8fafc;
    --border-card: #e2e8f0;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --btn-primary: #0284c7;
    --btn-primary-hover: #0369a1;
    --btn-danger: #e11d48;
    --btn-danger-hover: #be123c;
    --highlight-bg: #fde047;
    --highlight-text: #0f172a;
    --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }}

  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg-card: #0f172a;
      --border-card: #334155;
      --text-primary: #f1f5f9;
      --text-secondary: #94a3b8;
      --highlight-bg: #facc15;
      --highlight-text: #000000;
    }}
  }}

  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}

  body {{
    font-family: var(--font-family);
    color: var(--text-primary);
    background: transparent;
    padding: 6px 4px;
    line-height: 1.6;
    font-size: 14px;
  }}

  .read-along-container {{
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: 10px;
    padding: 12px 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }}

  /* Controls Toolbar */
  .toolbar {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border-card);
    margin-bottom: 12px;
  }}

  .btn {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    border: none;
    cursor: pointer;
    transition: all 0.15s ease-in-out;
    outline: none;
  }}

  .btn-primary {{
    background: var(--btn-primary);
    color: #ffffff;
  }}
  .btn-primary:hover:not(:disabled) {{
    background: var(--btn-primary-hover);
  }}

  .btn-danger {{
    background: var(--btn-danger);
    color: #ffffff;
  }}
  .btn-danger:hover:not(:disabled) {{
    background: var(--btn-danger-hover);
  }}

  .btn:disabled {{
    opacity: 0.45;
    cursor: not-allowed;
  }}

  .select-wrapper {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 12px;
    color: var(--text-secondary);
  }}

  select {{
    padding: 5px 8px;
    font-size: 12px;
    border-radius: 6px;
    border: 1px solid var(--border-card);
    background: var(--bg-card);
    color: var(--text-primary);
    outline: none;
    cursor: pointer;
    max-width: 170px;
  }}

  .status-badge {{
    margin-left: auto;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 12px;
    background: var(--border-card);
    color: var(--text-secondary);
  }}

  .status-badge.active {{
    background: #dbeafe;
    color: #1d4ed8;
  }}

  .status-badge.paused {{
    background: #fef3c7;
    color: #b45309;
  }}

  /* Content Viewport */
  .text-viewport {{
    max-height: 180px;
    overflow-y: auto;
    padding-right: 6px;
    scroll-behavior: smooth;
  }}

  .text-viewport::-webkit-scrollbar {{
    width: 6px;
  }}
  .text-viewport::-webkit-scrollbar-thumb {{
    background: var(--border-card);
    border-radius: 3px;
  }}

  .read-along-h4 {{
    font-size: 15px;
    font-weight: 700;
    margin: 8px 0 4px 0;
    color: var(--text-primary);
  }}

  .read-along-h5 {{
    font-size: 14px;
    font-weight: 600;
    margin: 6px 0 3px 0;
    color: var(--text-primary);
  }}

  .read-along-p {{
    margin-bottom: 8px;
  }}

  .read-along-list {{
    margin-left: 20px;
    margin-bottom: 8px;
  }}

  .read-along-list li {{
    margin-bottom: 4px;
  }}

  /* Real-Time Word Highlighting */
  .speech-word {{
    display: inline;
    border-radius: 3px;
    padding: 1px 2px;
    transition: background-color 0.08s ease, color 0.08s ease, box-shadow 0.08s ease;
  }}

  .speech-word.speaking-active {{
    background-color: var(--highlight-bg) !important;
    color: var(--highlight-text) !important;
    font-weight: 700;
    box-shadow: 0 0 0 2px var(--highlight-bg), 0 2px 4px rgba(0,0,0,0.18);
  }}

  .unsupported-banner {{
    background: #fff1f2;
    color: #be123c;
    border: 1px solid #fecdd3;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    display: none;
    margin-bottom: 8px;
  }}
</style>
</head>
<body>

<div class="read-along-container">
  <div id="unsupported-msg" class="unsupported-banner">
    ⚠️ Web Speech API (speechSynthesis) is not supported in this browser. Please use Chrome, Edge, or Safari.
  </div>

  <div class="toolbar">
    <button id="btn-play" class="btn btn-primary" title="Play speech">▶ Listen</button>
    <button id="btn-pause" class="btn btn-primary" style="display:none;" title="Pause speech">⏸ Pause</button>
    <button id="btn-resume" class="btn btn-primary" style="display:none;" title="Resume speech">▶ Resume</button>
    <button id="btn-stop" class="btn btn-danger" disabled title="Stop playback">⏹ Stop</button>

    <div class="select-wrapper">
      <span>🗣 Voice:</span>
      <select id="voice-select" title="Choose English voice">
        <option value="">Default English Voice</option>
      </select>
    </div>

    <div class="select-wrapper">
      <span>⚡ Speed:</span>
      <select id="rate-select">
        <option value="0.85">0.85x</option>
        <option value="1.0" selected>1.0x</option>
        <option value="1.2">1.2x</option>
        <option value="1.4">1.4x</option>
      </select>
    </div>

    <span id="status-badge" class="status-badge">Ready</span>
  </div>

  <div class="text-viewport" id="viewport">
    {html_body}
  </div>
</div>

<script>
(function() {{
  const SPEECH_TEXT = {plain_speech_json};
  const COMPONENT_ID = {component_id_json};

  const btnPlay = document.getElementById('btn-play');
  const btnPause = document.getElementById('btn-pause');
  const btnResume = document.getElementById('btn-resume');
  const btnStop = document.getElementById('btn-stop');
  const voiceSelect = document.getElementById('voice-select');
  const rateSelect = document.getElementById('rate-select');
  const statusBadge = document.getElementById('status-badge');
  const viewport = document.getElementById('viewport');
  const unsupportedMsg = document.getElementById('unsupported-msg');

  // Check browser support
  const hasSpeech = 'speechSynthesis' in window && 'SpeechSynthesisUtterance' in window;
  if (!hasSpeech) {{
    unsupportedMsg.style.display = 'block';
    btnPlay.disabled = true;
    statusBadge.textContent = 'Unsupported';
    return;
  }}

  const synth = window.speechSynthesis;
  let activeUtterance = null;
  let isPaused = false;
  let activeWordSpan = null;
  let keepAliveTimer = null;

  // Cross-component coordination via BroadcastChannel
  let syncChannel = null;
  try {{
    syncChannel = new BroadcastChannel('mediassist_read_along_sync');
    syncChannel.onmessage = (event) => {{
      if (event.data && event.data.type === 'START_PLAYING' && event.data.senderId !== COMPONENT_ID) {{
        // Another message started playing -- stop and reset this one
        stopPlayback(false);
      }}
    }};
  }} catch (e) {{
    // BroadcastChannel unsupported or restricted in sandbox
  }}

  // Build index of word spans
  const wordSpans = Array.from(document.querySelectorAll('.speech-word'));
  const wordMap = wordSpans.map(el => ({{
    el: el,
    start: parseInt(el.getAttribute('data-start'), 10),
    end: parseInt(el.getAttribute('data-end'), 10)
  }}));

  // Populate English voices
  function populateVoices() {{
    const voices = synth.getVoices();
    voiceSelect.innerHTML = '';
    const englishVoices = voices.filter(v => v.lang && v.lang.toLowerCase().startsWith('en'));
    
    if (englishVoices.length === 0) {{
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'System Default Voice';
      voiceSelect.appendChild(opt);
      return;
    }}

    englishVoices.forEach((voice, i) => {{
      const opt = document.createElement('option');
      opt.value = voice.name;
      opt.textContent = voice.name + ' (' + voice.lang + ')';
      // Prefer natural/neural voices by default
      if (voice.name.includes('Natural') || voice.name.includes('Online') || voice.name.includes('Google') || voice.default) {{
        opt.selected = true;
      }}
      voiceSelect.appendChild(opt);
    }});
  }}

  populateVoices();
  if (synth.onvoiceschanged !== undefined) {{
    synth.onvoiceschanged = populateVoices;
  }}

  // Highlight word matching charIndex
  function highlightWord(charIndex) {{
    if (!wordMap.length) return;

    let target = null;
    for (let i = 0; i < wordMap.length; i++) {{
      const item = wordMap[i];
      const nextStart = (i < wordMap.length - 1) ? wordMap[i + 1].start : Infinity;
      if (charIndex >= item.start && charIndex < nextStart) {{
        target = item.el;
        break;
      }}
    }}

    if (!target && wordMap.length > 0) {{
      target = wordMap[0].el;
    }}

    if (target) {{
      if (activeWordSpan && activeWordSpan !== target) {{
        activeWordSpan.classList.remove('speaking-active');
      }}
      target.classList.add('speaking-active');
      activeWordSpan = target;

      // Scroll into view inside the viewport
      target.scrollIntoView({{ behavior: 'smooth', block: 'nearest', inline: 'nearest' }});
    }}
  }}

  function clearHighlight() {{
    if (activeWordSpan) {{
      activeWordSpan.classList.remove('speaking-active');
      activeWordSpan = null;
    }}
  }}

  function setStatus(text, stateClass) {{
    statusBadge.textContent = text;
    statusBadge.className = 'status-badge ' + (stateClass || '');
  }}

  function stopPlayback(cancelSynth = true) {{
    if (keepAliveTimer) {{
      clearInterval(keepAliveTimer);
      keepAliveTimer = null;
    }}

    if (cancelSynth) {{
      synth.cancel();
    }}

    activeUtterance = null;
    isPaused = false;
    clearHighlight();

    btnPlay.style.display = 'inline-flex';
    btnPlay.disabled = false;
    btnPause.style.display = 'none';
    btnResume.style.display = 'none';
    btnStop.disabled = true;

    setStatus('Ready', '');
  }}

  function startPlayback() {{
    // Stop any existing speech across the browser
    synth.cancel();

    // Notify other components to stop
    if (syncChannel) {{
      try {{
        syncChannel.postMessage({{ type: 'START_PLAYING', senderId: COMPONENT_ID }});
      }} catch (e) {{}}
    }}

    clearHighlight();

    const utterance = new SpeechSynthesisUtterance(SPEECH_TEXT);
    activeUtterance = utterance;

    // Apply voice
    const voices = synth.getVoices();
    const selectedVoiceName = voiceSelect.value;
    if (selectedVoiceName) {{
      const matched = voices.find(v => v.name === selectedVoiceName);
      if (matched) utterance.voice = matched;
    }}

    // Apply rate
    utterance.rate = parseFloat(rateSelect.value) || 1.0;
    utterance.pitch = 1.0;

    // Word boundary event for real-time word highlighting
    utterance.onboundary = (event) => {{
      if (event.name === 'word') {{
        highlightWord(event.charIndex);
      }}
    }};

    utterance.onstart = () => {{
      btnPlay.style.display = 'none';
      btnPause.style.display = 'inline-flex';
      btnResume.style.display = 'none';
      btnStop.disabled = false;
      setStatus('Speaking…', 'active');
    }};

    utterance.onpause = () => {{
      btnPause.style.display = 'none';
      btnResume.style.display = 'inline-flex';
      setStatus('Paused', 'paused');
    }};

    utterance.onresume = () => {{
      btnResume.style.display = 'none';
      btnPause.style.display = 'inline-flex';
      setStatus('Speaking…', 'active');
    }};

    utterance.onend = () => {{
      stopPlayback(false);
      setStatus('Completed', '');
      setTimeout(() => setStatus('Ready', ''), 2000);
    }};

    utterance.onerror = (e) => {{
      console.warn('SpeechSynthesis error:', e);
      stopPlayback(false);
    }};

    // Chrome long-speech keepalive fix
    keepAliveTimer = setInterval(() => {{
      if (synth.speaking && !synth.paused) {{
        synth.pause();
        synth.resume();
      }}
    }}, 12000);

    synth.speak(utterance);
  }}

  function pausePlayback() {{
    if (synth.speaking && !synth.paused) {{
      synth.pause();
      isPaused = true;
      btnPause.style.display = 'none';
      btnResume.style.display = 'inline-flex';
      setStatus('Paused', 'paused');
    }}
  }}

  function resumePlayback() {{
    if (synth.paused) {{
      synth.resume();
      isPaused = false;
      btnResume.style.display = 'none';
      btnPause.style.display = 'inline-flex';
      setStatus('Speaking…', 'active');
    }} else if (!synth.speaking) {{
      startPlayback();
    }}
  }}

  // Event Listeners
  btnPlay.addEventListener('click', startPlayback);
  btnPause.addEventListener('click', pausePlayback);
  btnResume.addEventListener('click', resumePlayback);
  btnStop.addEventListener('click', () => stopPlayback(true));

  // Change rate while playing
  rateSelect.addEventListener('change', () => {{
    if (synth.speaking) {{
      // Restart with new rate seamlessly
      startPlayback();
    }}
  }});

  // Change voice while playing
  voiceSelect.addEventListener('change', () => {{
    if (synth.speaking) {{
      startPlayback();
    }}
  }});

  // Cleanup on unload
  window.addEventListener('beforeunload', () => {{
    synth.cancel();
    if (keepAliveTimer) clearInterval(keepAliveTimer);
  }});
}})();
</script>
</body>
</html>"""

