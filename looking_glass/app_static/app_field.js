/* Field display: the lens read-out drawn as a moving field of text.
 *
 * One column per input position. A column is that position's next-token
 * distribution at the chosen depth, drawn as ROWS lines of text:
 *
 *   rows a word fills   its probability (ROWS rows = 100%), most likely on top
 *   grey fragments      probability the top tokens do not cover
 *   wave height, flicker  entropy of the distribution (uncertain = choppy)
 *   coloured block      a tracked token
 *   outlined block      the model's answer, in the selected column
 *
 * The depth control blends between neighbouring layers so the field settles
 * as it is turned towards the last layer.  Motion adds no information that
 * the still picture lacks; it can be switched off and follows the system's
 * reduced-motion setting.
 *
 * The scrub-through-depth control is inspired by the "main knob" in
 * Chuanqi Sun's Spaceship IDE (https://github.com/chuanqisun/spaceship).
 */
(function (global) {
  'use strict';

  const ROWS = 16;
  const FONT_SIZE = 12;
  const LINE = 16;
  const MAX_COLUMN_CHARS = 12;
  const MIN_COLUMN_CHARS = 7;
  const GUTTER_CHARS = 6;
  const HEADER_LINES = 3;
  const WAVE_MARGIN = 8;
  const INK = '26, 26, 25';
  const FALLBACK_ALPHABET = '·:+~-=';

  /* ---------- pure functions (covered by tests/app_field_contract.cjs) ---------- */

  /* Deterministic pseudo-random number in [0, 1) from three integers. */
  function hash(a, b, c) {
    let h = (a * 374761393 + b * 668265263 + c * 2147483647) | 0;
    h = (h ^ (h >>> 13)) * 1274126177 | 0;
    h = h ^ (h >>> 16);
    return (h >>> 0) / 4294967296;
  }

  /* Entropy as a share of the largest possible entropy for this vocabulary, in [0, 1]. */
  function normalizedEntropy(entropy, vocabulary) {
    const most = Math.log(Math.max(2, vocabulary));
    return Math.min(1, Math.max(0, entropy / most));
  }

  /* Blend two cells' top-token lists. fraction 0 is all `lower`, 1 is all `upper`.
   * A token missing from one list counts as probability 0 there (it is below that list's cut-off). */
  function blendTokens(lower, upper, fraction) {
    const merged = new Map();
    for (const entry of lower) merged.set(entry.token_id, {token_id: entry.token_id, text: entry.text, probability: entry.probability * (1 - fraction)});
    for (const entry of upper) {
      const known = merged.get(entry.token_id);
      if (known) known.probability += entry.probability * fraction;
      else merged.set(entry.token_id, {token_id: entry.token_id, text: entry.text, probability: entry.probability * fraction});
    }
    return Array.from(merged.values()).filter((entry) => entry.probability > 0)
      .sort((a, b) => b.probability - a.probability || a.token_id - b.token_id);
  }

  /* Give each token a whole number of rows in proportion to its probability.
   * Returns an array of length `rows`: tokens from the top, then null for uncovered mass.
   * The most likely token always gets at least one row so a column is never blank. */
  function allocateRows(tokens, rows) {
    const covered = Math.min(1, tokens.reduce((sum, entry) => sum + entry.probability, 0));
    const budget = Math.max(tokens.length ? 1 : 0, Math.round(covered * rows));
    const shares = tokens.map((entry, index) => {
      const exact = entry.probability * rows;
      return {index, count: Math.floor(exact), remainder: exact - Math.floor(exact)};
    });
    let used = shares.reduce((sum, share) => sum + share.count, 0);
    const byRemainder = shares.slice().sort((a, b) => b.remainder - a.remainder || a.index - b.index);
    for (const share of byRemainder) {
      if (used >= budget) break;
      share.count += 1; used += 1;
    }
    if (tokens.length && shares[0].count === 0) shares[0].count = 1;
    const result = [];
    for (const share of shares) {
      for (let n = 0; n < share.count && result.length < rows; n++) result.push(tokens[share.index]);
    }
    while (result.length < rows) result.push(null);
    return result;
  }

  /* Repeat a label to fill a fixed number of characters. */
  function tile(label, width) {
    const unit = label.length ? label + ' ' : ' ';
    let text = '';
    while (text.length < width) text += unit;
    return text.slice(0, width);
  }

  /* Widest column, in characters, that lets every column fit; never narrower than the minimum. */
  function columnChars(available, gutterChars, columns, charWidth) {
    if (columns < 1 || charWidth <= 0) return MAX_COLUMN_CHARS;
    const each = Math.floor((available / charWidth - gutterChars) / columns) - 2;
    return Math.min(MAX_COLUMN_CHARS, Math.max(MIN_COLUMN_CHARS, each));
  }

  /* First layer (by position in `layers`) at which the answer is the top token, or -1. */
  function resolvedAt(layers, position) {
    for (let index = 0; index < layers.length; index++) {
      const cell = layers[index].cells.find((entry) => entry.position === position);
      if (cell && cell.answer_rank === 1) return index;
    }
    return -1;
  }

  /* ---------- the display ---------- */

  function create(parts, hooks) {
    const lens = global.MLGLens;
    const canvas = parts.canvas, context = canvas.getContext('2d');
    const reduced = global.matchMedia ? global.matchMedia('(prefers-reduced-motion: reduce)') : {matches: true};
    const state = {
      data: null, pins: [], selected: null,
      depth: 0, active: false, motion: !reduced.matches, playing: false, playStart: 0, playFrom: 0,
      frame: 0, lastDraw: 0, charWidth: 7.2, chars: MAX_COLUMN_CHARS, shownLayer: -1, hits: [], settle: new Map(), size: {width: 0, height: 0},
    };

    function layerCount() { return state.data ? state.data.layers.length : 0; }
    function columnWidth() { return (state.chars + 2) * state.charWidth; }
    function gutter() { return GUTTER_CHARS * state.charWidth; }
    function fieldTop() { return HEADER_LINES * LINE + WAVE_MARGIN; }

    function measure() {
      context.font = '400 ' + FONT_SIZE + 'px \'JetBrains Mono\', monospace';
      state.charWidth = context.measureText('MMMMMMMMMM').width / 10;
      const columns = state.data ? state.data.tokens.length : 0, available = parts.scroll.clientWidth || 0;
      state.chars = columnChars(available - 8, GUTTER_CHARS, columns, state.charWidth);
      const width = Math.max(available, Math.ceil(gutter() + columns * columnWidth() + 8));
      const height = fieldTop() + ROWS * LINE + WAVE_MARGIN + 6;
      const ratio = Math.min(global.devicePixelRatio || 1, 8192 / Math.max(1, width));
      canvas.style.width = width + 'px'; canvas.style.height = height + 'px';
      canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      state.size = {width, height};
    }

    function cellFor(layerPosition, position) {
      const layer = state.data.layers[layerPosition];
      return layer.cells.find((entry) => entry.position === position);
    }

    /* The blended distribution for one position at the current depth. */
    function column(position) {
      const low = Math.floor(state.depth), high = Math.min(layerCount() - 1, low + 1), fraction = state.depth - low;
      const lower = cellFor(low, position), upper = cellFor(high, position);
      const tokens = blendTokens(lower.top, upper.top, fraction);
      const entropy = lower.entropy * (1 - fraction) + upper.entropy * fraction;
      return {tokens, uncertainty: normalizedEntropy(entropy, state.data.vocabulary_size)};
    }

    function alphabetOf(tokens) {
      let letters = '';
      for (const entry of tokens) letters += entry.text.replace(/\s/g, '');
      return letters.length >= 4 ? letters : FALLBACK_ALPHABET;
    }

    function draw(now) {
      const data = state.data;
      if (!data) return;
      const width = state.size.width, height = state.size.height, charWidth = state.charWidth, columnW = columnWidth();
      const moving = state.motion && state.active;
      context.clearRect(0, 0, width, height);
      context.textBaseline = 'top';
      state.hits = [];

      // Gutter: rows are cumulative probability from the top.
      context.font = '400 10.5px \'JetBrains Mono\', monospace';
      context.fillStyle = 'rgba(' + INK + ', 0.55)';
      const marks = [[0, '0%'], [ROWS / 2, '50%'], [ROWS, '100%']];
      for (const mark of marks) context.fillText(mark[1], 0, fieldTop() + mark[0] * LINE - (mark[0] === ROWS ? 12 : 0));

      data.tokens.forEach((token, index) => {
        const left = gutter() + index * columnW, position = token.position;
        const selected = state.selected && state.selected.position === position;
        if (selected) {
          context.fillStyle = 'rgba(' + INK + ', 0.05)';
          context.fillRect(left - charWidth / 2, 0, columnW, height);
        }
        // Header: position, input token, and a rule for the surface.
        context.font = '400 10px \'JetBrains Mono\', monospace';
        context.fillStyle = 'rgba(' + INK + ', 0.55)';
        context.fillText(String(position), left, 2);
        context.font = (selected ? '600 ' : '400 ') + FONT_SIZE + 'px \'JetBrains Mono\', monospace';
        context.fillStyle = 'rgba(' + INK + ', 1)';
        context.fillText(lens.tokenLabel(token.text).slice(0, state.chars), left, LINE);
        context.fillStyle = 'rgba(' + INK + ', ' + (selected ? 1 : 0.25) + ')';
        context.fillRect(left, 2 * LINE + 6, state.chars * charWidth, selected ? 2 : 1);

        const mixed = column(position), rows = allocateRows(mixed.tokens, ROWS), alphabet = alphabetOf(mixed.tokens);
        const answer = data.answers.find((entry) => entry.position === position);
        const wave = moving ? (0.5 + 5 * mixed.uncertainty) * Math.sin(now * 0.0016 + index * 0.55) : 0;
        const top = fieldTop() + wave;
        const hit = {left, right: left + state.chars * charWidth, top, position, rows: [], tokens: mixed.tokens};

        for (let row = 0; row < ROWS; row++) {
          const y = top + row * LINE, entry = rows[row];
          hit.rows.push(entry);
          if (!entry) {
            // Uncovered probability: fragments of the candidate words, flickering faster when uncertain.
            const period = 90 + 900 * (1 - mixed.uncertainty);
            let text = '';
            for (let k = 0; k < state.chars; k++) {
              const phase = hash(index, row * 31 + k, 7);
              const tick = moving ? Math.floor(now / period + phase * 4) : 0;
              text += hash(index, row * 31 + k, tick + 11) < 0.78 ? alphabet[Math.floor(hash(index * 131 + k, row, tick) * alphabet.length)] : ' ';
            }
            context.font = '400 ' + FONT_SIZE + 'px \'JetBrains Mono\', monospace';
            context.fillStyle = 'rgba(' + INK + ', ' + (0.14 + 0.16 * mixed.uncertainty).toFixed(3) + ')';
            context.fillText(text, left, y);
            continue;
          }
          let text = tile(lens.tokenLabel(entry.text), state.chars);
          // A row that just changed word scrambles briefly, then settles.
          const key = index + ':' + row, previous = state.settle.get(key);
          if (!previous || previous.id !== entry.token_id) {
            state.settle.set(key, {id: entry.token_id, start: now, length: previous && moving ? 120 + 220 * hash(index, row, entry.token_id) : 0});
          } else if (moving && now - previous.start < previous.length) {
            const left01 = 1 - (now - previous.start) / previous.length;
            let scrambled = '';
            for (let k = 0; k < text.length; k++) {
              scrambled += hash(index + k, row, Math.floor(now / 40)) < left01 ? alphabet[Math.floor(hash(k, index, Math.floor(now / 40) + row) * alphabet.length)] : text[k];
            }
            text = scrambled;
          }
          const pin = state.pins.find((candidate) => candidate.token_id === entry.token_id);
          const weight = entry.probability >= 0.25 ? '600 ' : '400 ';
          context.font = weight + FONT_SIZE + 'px \'JetBrains Mono\', monospace';
          if (pin) {
            const fill = lens.SERIES[pin.slot];
            context.fillStyle = fill;
            context.fillRect(left - 2, y - 1, state.chars * charWidth + 4, LINE);
            context.fillStyle = lens.inkOn(fill);
          } else {
            context.fillStyle = 'rgba(' + INK + ', ' + (0.38 + 0.62 * Math.min(1, entry.probability)).toFixed(3) + ')';
          }
          context.fillText(text, left, y + 1);
        }
        // In the selected column, outline the block of rows held by the model's answer.
        if (answer && selected) {
          const first = rows.findIndex((entry) => entry && entry.token_id === answer.token_id);
          if (first >= 0) {
            let count = 0;
            while (first + count < ROWS && rows[first + count] && rows[first + count].token_id === answer.token_id) count++;
            context.strokeStyle = 'rgba(' + INK + ', 1)';
            context.lineWidth = 1.5;
            context.strokeRect(left - 3, top + first * LINE - 1.5, state.chars * charWidth + 6, count * LINE + 1);
          }
        }
        state.hits.push(hit);
      });
    }

    /* ---------- animation ---------- */

    function needsFrames() {
      return state.active && state.data && !global.document.hidden && (state.motion || state.playing);
    }
    function tick(now) {
      state.frame = 0;
      if (!state.active || !state.data) return;
      if (state.playing) {
        const last = layerCount() - 1, elapsed = now - state.playStart;
        const depth = state.motion ? state.playFrom + elapsed / 550 : state.playFrom + Math.floor(elapsed / 550);
        if (depth >= last) { setDepth(last); stopPlaying(); } else setDepth(depth);
      }
      if (now - state.lastDraw >= 33) { state.lastDraw = now; draw(now); }
      if (needsFrames()) state.frame = global.requestAnimationFrame(tick);
    }
    function schedule() {
      if (!state.active || !state.data) return;
      if (needsFrames()) { if (!state.frame) state.frame = global.requestAnimationFrame(tick); }
      else draw(global.performance ? global.performance.now() : 0);
    }
    function halt() { if (state.frame) global.cancelAnimationFrame(state.frame); state.frame = 0; }

    /* ---------- depth control ---------- */

    function describeDepth() {
      const data = state.data, layers = data.layers, low = Math.floor(state.depth), fraction = state.depth - low;
      let text = fraction < 0.005 ? 'Layer ' + layers[low].layer_index
        : 'Layers ' + layers[low].layer_index + '–' + layers[Math.min(layers.length - 1, low + 1)].layer_index;
      if (state.selected) {
        const answer = data.answers.find((entry) => entry.position === state.selected.position);
        const at = resolvedAt(layers, state.selected.position);
        if (answer && at >= 0) text += ' · at position ' + state.selected.position + ' the answer ' + lens.tokenLabel(answer.text) + ' first tops the read-out at layer ' + layers[at].layer_index;
      }
      parts.status.textContent = text;
      parts.depth.setAttribute('aria-valuetext', text);
    }
    function setDepth(value) {
      const last = Math.max(0, layerCount() - 1);
      state.depth = Math.min(last, Math.max(0, value));
      parts.depth.value = String(state.depth);
      if (!state.data) return;
      describeDepth();
      // The read-out and chart follow the nearest whole layer.
      const nearest = Math.round(state.depth);
      if (nearest !== state.shownLayer) { state.shownLayer = nearest; hooks.layer(state.data.layers[nearest].layer_index); }
    }
    function stopPlaying() {
      state.playing = false;
      parts.play.setAttribute('aria-pressed', 'false');
      parts.play.textContent = '▶';
      parts.play.setAttribute('aria-label', 'Play through the layers');
    }
    function startPlaying() {
      if (layerCount() < 2) return;
      const last = layerCount() - 1;
      state.playFrom = state.depth >= last ? 0 : state.depth;
      state.playStart = global.performance.now();
      state.playing = true;
      parts.play.setAttribute('aria-pressed', 'true');
      parts.play.textContent = '❙❙';
      parts.play.setAttribute('aria-label', 'Pause');
      schedule();
    }

    parts.depth.addEventListener('input', () => { stopPlaying(); setDepth(Number(parts.depth.value)); schedule(); });
    parts.depth.addEventListener('keydown', (event) => {
      const step = {ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1}[event.key];
      if (!step || event.shiftKey) return;        // Shift + arrow keeps the fine step
      event.preventDefault(); stopPlaying();
      setDepth(step > 0 ? Math.floor(state.depth + 1 + 1e-9) : Math.ceil(state.depth - 1 - 1e-9)); schedule();
    });
    parts.play.addEventListener('click', () => { if (state.playing) { stopPlaying(); schedule(); } else startPlaying(); });
    parts.motion.addEventListener('click', () => {
      state.motion = !state.motion;
      parts.motion.setAttribute('aria-pressed', String(state.motion));
      halt(); schedule();
    });

    /* ---------- pointer ---------- */

    function locate(event) {
      const box = canvas.getBoundingClientRect(), x = event.clientX - box.left, y = event.clientY - box.top;
      const hit = state.hits.find((candidate) => x >= candidate.left - 3 && x <= candidate.right + 3);
      if (!hit) return null;
      const row = Math.floor((y - hit.top) / LINE);
      return {hit, entry: row >= 0 && row < ROWS ? hit.rows[row] : null, inField: row >= 0 && row < ROWS, x: event.clientX, y: event.clientY};
    }
    canvas.addEventListener('pointermove', (event) => {
      const found = locate(event);
      if (!found || !found.inField) { canvas.style.cursor = found ? 'pointer' : ''; hooks.hideTip(); return; }
      canvas.style.cursor = 'pointer';
      const uncovered = Math.max(0, 1 - found.hit.tokens.reduce((sum, entry) => sum + entry.probability, 0));
      if (found.entry) {
        const rank = found.hit.tokens.findIndex((entry) => entry.token_id === found.entry.token_id) + 1;
        hooks.showTip('Position ' + found.hit.position, [[found.entry.text, '#' + rank + ' · ' + lens.percent(found.entry.probability)]],
          'Click to track this token', {left: found.x, width: 0, top: found.y - 6, bottom: found.y + 12, height: 0});
      } else {
        hooks.showTip('Position ' + found.hit.position, [], lens.percent(uncovered) + ' of the probability is outside the top ' + found.hit.tokens.length + ' tokens shown',
          {left: found.x, width: 0, top: found.y - 6, bottom: found.y + 12, height: 0});
      }
    });
    canvas.addEventListener('pointerleave', () => hooks.hideTip());
    canvas.addEventListener('click', (event) => {
      const found = locate(event);
      if (!found) return;
      hooks.select(found.hit.position);
      if (found.entry) hooks.track(found.entry.token_id, found.entry.text);
    });

    /* ---------- wiring ---------- */

    function setData(data, pins, selected) {
      const fresh = !state.data || !data || state.data.capture_id !== data.capture_id || state.data.forward !== data.forward
        || state.data.position_offset !== data.position_offset || state.data.layers.length !== data.layers.length;
      state.data = data; state.pins = pins || []; state.selected = selected || null;
      if (!data) { halt(); return; }
      const last = data.layers.length - 1;
      parts.depth.max = String(Math.max(1, last)); parts.depth.disabled = last < 1; parts.play.disabled = last < 1;
      if (fresh) { state.settle.clear(); stopPlaying(); state.depth = last; }
      state.shownLayer = -1;
      followSelection();
      measure(); schedule(); reveal();
    }
    /* Outside playback, the depth control sits on the selected cell's layer. */
    function followSelection() {
      let index = -1;
      if (state.selected) index = state.data.layers.findIndex((layer) => layer.layer_index === state.selected.layer);
      if (index >= 0 && !state.playing && Math.round(state.depth) !== index) {
        state.shownLayer = index;       // the selection already names this layer; nothing to report back
        setDepth(index);
        return;
      }
      if (index >= 0 && Math.round(state.depth) === index) state.shownLayer = index;
      setDepth(state.depth);            // clamps to the layer range and refreshes the status text
    }
    function reveal() {
      if (!state.selected || !state.data) return;
      const index = state.data.tokens.findIndex((token) => token.position === state.selected.position);
      if (index < 0) return;
      const left = gutter() + index * columnWidth(), right = left + columnWidth(), view = parts.scroll;
      if (left < view.scrollLeft + gutter()) view.scrollLeft = Math.max(0, left - gutter());
      else if (right > view.scrollLeft + view.clientWidth) view.scrollLeft = right - view.clientWidth;
    }
    function setSelected(selected) {
      state.selected = selected || null;
      if (!state.data) return;
      followSelection(); schedule(); reveal();
    }
    function setActive(active) {
      state.active = !!active;
      if (!state.active) { halt(); stopPlaying(); return; }
      if (state.data) { measure(); schedule(); }
    }
    function resize() { if (state.active && state.data) { measure(); schedule(); } }

    parts.motion.setAttribute('aria-pressed', String(state.motion));
    if (global.document.fonts && global.document.fonts.load) {
      global.document.fonts.load(FONT_SIZE + 'px \'JetBrains Mono\'').then(resize, () => {});
    }
    if (global.addEventListener) {
      global.addEventListener('resize', resize);
      global.document.addEventListener('visibilitychange', () => { if (global.document.hidden) halt(); else schedule(); });
    }
    return {setData, setSelected, setActive, depth: () => state.depth};
  }

  global.MLGField = {create, hash, normalizedEntropy, blendTokens, allocateRows, tile, columnChars, resolvedAt, ROWS};
})(typeof window !== 'undefined' ? window : globalThis);
