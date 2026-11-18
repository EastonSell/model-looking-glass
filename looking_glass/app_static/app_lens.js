/* Lens view: a layer-by-position grid of what each captured layer would say next.
 *
 * Every visual element maps to a number returned by POST /api/lens:
 *   cell text    top-ranked token at (layer, position)
 *   cell colour  that token's probability, or the rank of the model's answer / a tracked token
 *   bold text    the cell already agrees with the model's final answer at that position
 *   chart        rank of the answer and of each tracked token through the layers
 * Nothing here is decorative, and no tensor ever reaches the browser.
 */
(function (global) {
  'use strict';

  /* One-hue sequential ramp, light to dark (magnitude). */
  const RAMP = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
    '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b'];
  /* Categorical slots for tracked tokens, in fixed order. A token keeps its slot until untracked. */
  const SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'];
  const ANSWER_COLOR = '#52514e';
  const MAX_PINNED = 8;
  const TOP_K = 10;
  const WINDOW = 128;
  const SVG = 'http://www.w3.org/2000/svg';

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function svg(tag, attributes) {
    const node = document.createElementNS(SVG, tag);
    for (const name of Object.keys(attributes || {})) node.setAttribute(name, attributes[name]);
    return node;
  }

  /* Make whitespace in a token visible without changing its meaning. */
  function tokenParts(text) {
    if (text === '') return {lead: false, body: '(empty)', trail: false};
    const shown = text.replace(/\n/g, '⏎').replace(/\t/g, '⇥');
    if (shown.trim() === '') return {lead: false, body: '␣'.repeat(Math.min(shown.length, 4)), trail: false};
    const lead = shown.startsWith(' '), trail = shown.endsWith(' ');
    return {lead, body: shown.trim(), trail};
  }
  function tokenNode(text, className) {
    const parts = tokenParts(text), node = element('span', className || 'lens-token');
    if (parts.lead) node.append(element('span', 'lens-space', '·'));
    node.append(document.createTextNode(parts.body));
    if (parts.trail) node.append(element('span', 'lens-space', '·'));
    return node;
  }
  function tokenLabel(text) {
    const parts = tokenParts(text);
    return (parts.lead ? '·' : '') + parts.body + (parts.trail ? '·' : '');
  }

  function percent(probability) {
    if (probability >= 0.0995) return (probability * 100).toFixed(0) + '%';
    if (probability >= 0.00095) return (probability * 100).toFixed(1) + '%';
    return '<0.1%';
  }
  function rankText(rank) {
    return '#' + rank.toLocaleString('en-US');
  }

  /* Magnitude in [0, 1] for the sequential ramp. Rank uses a log scale: #1 is darkest. */
  function rankMagnitude(rank, vocabulary) {
    const span = Math.log10(Math.max(10, vocabulary));
    return Math.max(0, 1 - Math.log10(Math.max(1, rank)) / span);
  }
  function rampColor(magnitude) {
    const index = Math.round(Math.min(1, Math.max(0, magnitude)) * (RAMP.length - 1));
    return RAMP[index];
  }
  function luminance(hex) {
    const channel = (offset) => {
      const value = parseInt(hex.slice(offset, offset + 2), 16) / 255;
      return value <= 0.03928 ? value / 12.92 : Math.pow((value + 0.055) / 1.055, 2.4);
    };
    return 0.2126 * channel(1) + 0.7152 * channel(3) + 0.0722 * channel(5);
  }
  /* Ink on a filled cell: whichever of black / white has more contrast (always >= 4.5:1 on this ramp). */
  function inkOn(hex) {
    const fill = luminance(hex);
    return (fill + 0.05) / 0.05 >= 1.05 / (fill + 0.05) ? '#000000' : '#ffffff';
  }

  function create(root, bridge) {
    const $ = (id) => root.querySelector('#' + id);
    const view = {
      run: null, busy: false, active: false,
      forward: 0, offset: 0, request: 0, data: null,
      pins: [],            // [{token_id, text, slot}]
      colorBy: 'probability', colorPin: null,
      selected: null,      // {layer, position}
      cells: new Map(),    // "layer:position" -> button
    };

    function status(message, error) {
      $('lens-status').textContent = message || '';
      $('lens-status').classList.toggle('error', !!error);
      $('lens-status').hidden = !message;
    }

    /* ---------- data ---------- */

    async function load(resolveText) {
      const run = view.run;
      if (!run || !run.lens_available) return;
      const request = ++view.request;
      const body = {capture_id: run.capture_id, forward: view.forward, top_k: TOP_K,
        position_offset: view.offset, position_limit: WINDOW,
        pinned_token_ids: view.pins.map((pin) => pin.token_id)};
      if (resolveText) body.resolve_text = resolveText;
      root.setAttribute('aria-busy', 'true');
      try {
        const data = await bridge.api('lens', body);
        if (request !== view.request) return;
        view.data = data;
        if (resolveText) addResolved(resolveText, data.resolved || []);
        else render();
      } catch (error) {
        if (request !== view.request) return;
        view.data = null;
        $('lens-body').hidden = true;
        status(error.message || 'The lens read-out failed.', true);
      } finally {
        if (request === view.request) root.removeAttribute('aria-busy');
      }
    }

    /* ---------- tracked tokens ---------- */

    function freeSlot() {
      for (let slot = 0; slot < SERIES.length; slot++) {
        if (!view.pins.some((pin) => pin.slot === slot)) return slot;
      }
      return -1;
    }
    function track(tokenId, text) {
      if (view.pins.some((pin) => pin.token_id === tokenId)) return true;
      if (view.pins.length >= MAX_PINNED) {
        status('Up to ' + MAX_PINNED + ' tokens can be tracked at once. Remove one first.', true);
        return false;
      }
      view.pins.push({token_id: tokenId, text, slot: freeSlot()});
      view.colorBy = 'pinned'; view.colorPin = tokenId; $('lens-color').value = 'pinned';
      return true;
    }
    function untrack(tokenId) {
      view.pins = view.pins.filter((pin) => pin.token_id !== tokenId);
      if (view.colorPin === tokenId) view.colorPin = view.pins.length ? view.pins[0].token_id : null;
      if (!view.pins.length && view.colorBy === 'pinned') { view.colorBy = 'probability'; $('lens-color').value = 'probability'; }
      load();
    }
    function addResolved(typed, candidates) {
      const single = candidates.filter((candidate) => candidate.single_token);
      if (!single.length) {
        render();
        const pieces = candidates.length ? candidates[0].token_count : 0;
        status(pieces ? '“' + typed + '” is ' + pieces + ' tokens in this vocabulary. Track one token at a time, for example by clicking a word in the read-out.'
          : '“' + typed + '” has no token in this vocabulary.', true);
        return;
      }
      let added = false;
      for (const candidate of single) added = track(candidate.token_id, candidate.text) || added;
      $('lens-pin-text').value = '';
      if (added) load(); else render();
    }
    function renderPins() {
      const list = $('lens-pin-list');
      list.replaceChildren();
      for (const pin of view.pins) {
        const item = element('li', 'lens-pin');
        item.append(element('span', 'lens-swatch'));
        item.firstChild.style.background = SERIES[pin.slot];
        const focus = element('button', 'lens-pin-focus');
        focus.type = 'button';
        focus.append(tokenNode(pin.text));
        focus.title = 'Colour the grid by the rank of ' + tokenLabel(pin.text) + ' (token ' + pin.token_id + ')';
        focus.setAttribute('aria-pressed', String(view.colorBy === 'pinned' && view.colorPin === pin.token_id));
        focus.addEventListener('click', () => {
          view.colorBy = 'pinned'; view.colorPin = pin.token_id; $('lens-color').value = 'pinned'; render();
        });
        const remove = element('button', 'lens-pin-remove', '×');
        remove.type = 'button';
        remove.setAttribute('aria-label', 'Stop tracking ' + tokenLabel(pin.text));
        remove.title = 'Stop tracking';
        remove.addEventListener('click', () => untrack(pin.token_id));
        item.append(focus, remove);
        list.append(item);
      }
      $('lens-color').querySelector('option[value=pinned]').disabled = !view.pins.length;
    }

    /* ---------- grid ---------- */

    function cellValue(cell) {
      const vocabulary = view.data.vocabulary_size;
      if (view.colorBy === 'answer') return rankMagnitude(cell.answer_rank, vocabulary);
      if (view.colorBy === 'pinned') {
        const pinned = cell.pinned.find((entry) => entry.token_id === view.colorPin);
        return pinned ? rankMagnitude(pinned.rank, vocabulary) : 0;
      }
      return cell.top[0].probability;
    }
    function layerByIndex(layerIndex) {
      return view.data.layers.find((layer) => layer.layer_index === layerIndex);
    }
    function cellAt(layerIndex, position) {
      const layer = layerByIndex(layerIndex);
      return layer ? layer.cells.find((cell) => cell.position === position) : null;
    }

    function renderGrid() {
      const data = view.data, head = $('lens-grid-head'), body = $('lens-grid-body');
      head.replaceChildren(); body.replaceChildren(); view.cells.clear();

      const tokens = element('tr');
      tokens.append(element('th', 'lens-corner', 'input'));
      for (const token of data.tokens) {
        const th = element('th', 'lens-column');
        th.scope = 'col';
        th.append(element('span', 'lens-position', String(token.position)), tokenNode(token.text));
        th.title = 'Position ' + token.position + ' · token ' + token.token_id + ' · ' + JSON.stringify(token.text);
        tokens.append(th);
      }
      head.append(tokens);

      const answers = element('tr', 'lens-answers');
      const answerHead = element('th', 'lens-row', data.reaches_model_output ? 'model says' : 'top layer says');
      answerHead.scope = 'row';
      answerHead.title = data.reaches_model_output
        ? 'The model’s actual next-token prediction after each position'
        : 'The top captured layer is not the model’s last layer, so this is that layer’s read-out, not the model’s output';
      answers.append(answerHead);
      for (const answer of data.answers) {
        const td = element('td', 'lens-answer');
        td.append(tokenNode(answer.text));
        td.title = 'After position ' + answer.position + ': ' + JSON.stringify(answer.text) + ' (token ' + answer.token_id + ')';
        answers.append(td);
      }
      body.append(answers);

      const layers = data.layers.slice().reverse();   // last layer on top, like the output above it
      for (const layer of layers) {
        const row = element('tr');
        const th = element('th', 'lens-row', 'L' + layer.layer_index);
        th.scope = 'row'; th.title = layer.module_path;
        row.append(th);
        for (const cell of layer.cells) {
          const td = element('td'), button = element('button', 'lens-cell');
          button.type = 'button';
          const top = cell.top[0], fill = rampColor(cellValue(cell));
          button.style.background = fill; button.style.color = inkOn(fill);
          button.append(tokenNode(top.text));
          if (cell.answer_rank === 1) button.classList.add('agrees');
          button.dataset.layer = String(layer.layer_index);
          button.dataset.position = String(cell.position);
          button.setAttribute('aria-label', 'Layer ' + layer.layer_index + ', position ' + cell.position + ': '
            + tokenLabel(top.text) + ', ' + percent(top.probability) + '. Model answer rank ' + cell.answer_rank + '.');
          button.addEventListener('click', () => select(layer.layer_index, cell.position, true));
          button.addEventListener('pointerenter', () => showCellTip(button, layer, cell));
          button.addEventListener('pointerleave', hideTip);
          button.addEventListener('focus', () => showCellTip(button, layer, cell));
          button.addEventListener('blur', hideTip);
          view.cells.set(layer.layer_index + ':' + cell.position, button);
          td.append(button); row.append(td);
        }
        body.append(row);
      }
      markSelected();
    }

    function markSelected() {
      for (const button of view.cells.values()) button.removeAttribute('aria-current');
      if (!view.selected) return;
      const button = view.cells.get(view.selected.layer + ':' + view.selected.position);
      if (button) button.setAttribute('aria-current', 'true');
      for (const th of $('lens-grid-head').querySelectorAll('th.lens-column')) th.classList.remove('selected');
      const column = view.selected.position - view.data.position_offset;
      const header = $('lens-grid-head').querySelectorAll('th.lens-column')[column];
      if (header) header.classList.add('selected');
    }

    function moveFocus(event) {
      const button = event.target.closest ? event.target.closest('.lens-cell') : null;
      if (!button) return;
      const steps = {ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [1, 0], ArrowDown: [-1, 0]}[event.key];
      if (!steps) return;
      const order = view.data.layers.map((layer) => layer.layer_index);
      const row = order.indexOf(Number(button.dataset.layer)) + steps[0];
      const position = Number(button.dataset.position) + steps[1];
      if (row < 0 || row >= order.length) return;
      const next = view.cells.get(order[row] + ':' + position);
      if (next) { event.preventDefault(); next.focus(); }
    }

    function renderLegend() {
      const legend = $('lens-legend');
      legend.replaceChildren();
      let low, high, title;
      if (view.colorBy === 'probability') { low = '0%'; high = '100%'; title = 'Probability of the cell’s top token'; }
      else {
        const what = view.colorBy === 'answer' ? 'the model’s answer at that position'
          : tokenLabel((view.pins.find((pin) => pin.token_id === view.colorPin) || {text: ''}).text);
        low = rankText(view.data.vocabulary_size) + ' (last)'; high = '#1'; title = 'Rank of ' + what + ', log scale';
      }
      const strip = element('span', 'lens-ramp');
      strip.setAttribute('aria-hidden', 'true');
      for (const color of RAMP) { const step = element('span'); step.style.background = color; strip.append(step); }
      legend.append(element('span', 'lens-legend-title', title), element('span', 'lens-legend-end', low), strip,
        element('span', 'lens-legend-end', high),
        element('span', 'lens-legend-note', 'Bold = already agrees with the row above the grid'));
    }

    /* ---------- tooltip ---------- */

    function showCellTip(anchor, layer, cell) {
      const tip = $('lens-tooltip');
      tip.replaceChildren();
      tip.append(element('strong', '', 'Layer ' + layer.layer_index + ' · position ' + cell.position));
      const list = element('ol');
      for (const entry of cell.top.slice(0, 3)) {
        const item = element('li');
        item.append(tokenNode(entry.text), element('span', 'lens-tip-value', percent(entry.probability)));
        list.append(item);
      }
      tip.append(list);
      const answer = view.data.answers.find((entry) => entry.position === cell.position);
      if (answer) {
        const line = element('p');
        line.append(document.createTextNode('Answer '), tokenNode(answer.text),
          document.createTextNode(' ' + rankText(cell.answer_rank) + ' · ' + percent(cell.answer_probability)));
        tip.append(line);
      }
      placeTip(anchor.getBoundingClientRect());
    }
    function placeTip(rect) {
      const tip = $('lens-tooltip'), frame = root.getBoundingClientRect();
      tip.hidden = false;
      const width = tip.offsetWidth, height = tip.offsetHeight;
      let left = rect.left + rect.width / 2 - width / 2 - frame.left + root.scrollLeft;
      left = Math.max(4, Math.min(left, root.clientWidth - width - 4));
      let top = rect.top - height - 8 - frame.top + root.scrollTop;
      if (top < root.scrollTop + 4) top = rect.bottom + 8 - frame.top + root.scrollTop;
      tip.style.left = left + 'px'; tip.style.top = top + 'px';
    }
    function hideTip() { $('lens-tooltip').hidden = true; }

    /* ---------- read-out and chart for the selected cell ---------- */

    function select(layerIndex, position, byUser) {
      view.selected = {layer: layerIndex, position};
      markSelected(); renderReadout(); renderChart();
      if (byUser) hideTip();
    }

    function renderReadout() {
      const cell = cellAt(view.selected.layer, view.selected.position);
      const token = view.data.tokens.find((entry) => entry.position === view.selected.position);
      const list = $('lens-readout-list');
      list.replaceChildren();
      if (!cell) return;
      $('lens-readout-heading').textContent = 'Read-out at layer ' + view.selected.layer;
      const context = $('lens-readout-context');
      context.replaceChildren(document.createTextNode('After position ' + cell.position + ' '), tokenNode(token ? token.text : ''),
        document.createTextNode(' · top ' + cell.top.length + ' of ' + view.data.vocabulary_size.toLocaleString('en-US')
          + ' tokens · entropy ' + cell.entropy.toFixed(2) + ' nats'));
      const scale = Math.max(cell.top[0].probability, 1e-9);
      cell.top.forEach((entry, index) => {
        const item = element('li');
        const pick = element('button', 'lens-readout-row');
        pick.type = 'button';
        const tracked = view.pins.some((pin) => pin.token_id === entry.token_id);
        pick.title = tracked ? 'Already tracked' : 'Track ' + tokenLabel(entry.text) + ' (token ' + entry.token_id + ') through the layers';
        pick.setAttribute('aria-pressed', String(tracked));
        const bar = element('span', 'lens-bar'), fill = element('span');
        fill.style.width = Math.max(1, entry.probability / scale * 100) + '%';
        bar.append(fill);
        pick.append(element('span', 'lens-readout-rank', String(index + 1)), tokenNode(entry.text), bar,
          element('span', 'lens-readout-value', percent(entry.probability)));
        pick.addEventListener('click', () => { if (track(entry.token_id, entry.text)) load(); });
        item.append(pick); list.append(item);
      });
      const answer = view.data.answers.find((entry) => entry.position === cell.position);
      const line = $('lens-readout-answer');
      line.replaceChildren();
      if (answer) {
        line.append(document.createTextNode(view.data.reaches_model_output ? 'Model’s answer ' : 'Top layer’s answer '),
          tokenNode(answer.text), document.createTextNode(' is ' + rankText(cell.answer_rank) + ' here, at ' + percent(cell.answer_probability) + '.'));
      }
    }

    function chartSeries() {
      const position = view.selected.position, layers = view.data.layers;
      const answer = view.data.answers.find((entry) => entry.position === position);
      const series = [];
      if (answer) {
        series.push({name: (view.data.reaches_model_output ? 'model’s answer ' : 'top layer’s answer ') + tokenLabel(answer.text),
          short: tokenLabel(answer.text), color: ANSWER_COLOR, points: layers.map((layer) => {
            const cell = layer.cells.find((entry) => entry.position === position);
            return {layer: layer.layer_index, rank: cell.answer_rank, probability: cell.answer_probability};
          })});
      }
      for (const pin of view.pins) {
        if (answer && pin.token_id === answer.token_id) { series[0].color = SERIES[pin.slot]; continue; }
        series.push({name: tokenLabel(pin.text), short: tokenLabel(pin.text), color: SERIES[pin.slot], points: layers.map((layer) => {
          const cell = layer.cells.find((entry) => entry.position === position);
          const pinned = cell.pinned.find((entry) => entry.token_id === pin.token_id);
          return {layer: layer.layer_index, rank: pinned.rank, probability: pinned.probability};
        })});
      }
      return series;
    }

    function renderChart() {
      const host = $('lens-chart'), series = chartSeries(), layers = view.data.layers.map((layer) => layer.layer_index);
      const token = view.data.tokens.find((entry) => entry.position === view.selected.position);
      $('lens-trajectory-context').replaceChildren(document.createTextNode('After position ' + view.selected.position + ' '),
        tokenNode(token ? token.text : ''), document.createTextNode(' · rank #1 means this layer would already say it'));
      host.replaceChildren();
      const legend = $('lens-chart-legend');
      legend.replaceChildren();
      for (const entry of series) {
        const key = element('span', 'lens-key'), mark = element('span', 'lens-key-line');
        mark.style.background = entry.color;
        key.append(mark, document.createTextNode(entry.name));
        legend.append(key);
      }
      legend.hidden = series.length < 2;

      const width = Math.max(320, host.clientWidth || 480), height = 220;
      const margin = {top: 12, right: series.length <= 4 ? 96 : 16, bottom: 30, left: 52};
      const plotWidth = width - margin.left - margin.right, plotHeight = height - margin.top - margin.bottom;
      const decades = Math.ceil(Math.log10(Math.max(10, view.data.vocabulary_size)));
      const x = (index) => margin.left + (layers.length === 1 ? plotWidth / 2 : index / (layers.length - 1) * plotWidth);
      const y = (rank) => margin.top + Math.log10(Math.max(1, rank)) / decades * plotHeight;
      const chart = svg('svg', {viewBox: '0 0 ' + width + ' ' + height, width, height, role: 'img',
        'aria-label': 'Line chart of token rank by layer; the same numbers are in the table below'});

      for (let decade = 0; decade <= decades; decade++) {
        const rank = Math.pow(10, decade), line = y(rank);
        chart.append(svg('line', {x1: margin.left, x2: margin.left + plotWidth, y1: line, y2: line, class: 'lens-gridline'}));
        const label = svg('text', {x: margin.left - 8, y: line + 4, 'text-anchor': 'end', class: 'lens-axis'});
        label.textContent = rank >= 1000 ? '#' + rank / 1000 + 'K' : '#' + rank;
        chart.append(label);
      }
      const every = Math.max(1, Math.ceil(layers.length / 12));
      layers.forEach((layer, index) => {
        if (index % every && index !== layers.length - 1) return;
        const label = svg('text', {x: x(index), y: height - 10, 'text-anchor': 'middle', class: 'lens-axis'});
        label.textContent = 'L' + layer;
        chart.append(label);
      });
      const marker = svg('line', {x1: 0, x2: 0, y1: margin.top, y2: margin.top + plotHeight, class: 'lens-crosshair', visibility: 'hidden'});
      chart.append(marker);

      const ends = [];
      for (const entry of series) {
        const path = entry.points.map((point, index) => (index ? 'L' : 'M') + x(index).toFixed(1) + ' ' + y(point.rank).toFixed(1)).join(' ');
        chart.append(svg('path', {d: path, class: 'lens-line', stroke: entry.color}));
        const last = entry.points[entry.points.length - 1];
        chart.append(svg('circle', {cx: x(entry.points.length - 1), cy: y(last.rank), r: 4, fill: entry.color, class: 'lens-dot'}));
        ends.push({entry, at: y(last.rank)});
      }
      // Direct end labels only while they stay readable: few series, no collisions.
      ends.sort((a, b) => a.at - b.at);
      const crowded = ends.some((end, index) => index > 0 && end.at - ends[index - 1].at < 13);
      if (series.length <= 4 && !crowded) {
        for (const end of ends) {
          const label = svg('text', {x: margin.left + plotWidth + 10, y: end.at + 4, class: 'lens-end'});
          label.textContent = end.entry.short.length > 12 ? end.entry.short.slice(0, 11) + '\u2026' : end.entry.short;
          chart.append(label);
        }
      } else legend.hidden = false;

      // Hover layer: snap a crosshair to the nearest layer and list every series there.
      const capture = svg('rect', {x: margin.left, y: margin.top, width: plotWidth, height: plotHeight, fill: 'transparent'});
      const describe = (index) => {
        const tip = $('lens-tooltip');
        tip.replaceChildren(element('strong', '', 'Layer ' + layers[index]));
        const list = element('ul');
        for (const entry of series) {
          const item = element('li'), mark = element('span', 'lens-key-line');
          mark.style.background = entry.color;
          item.append(mark, document.createTextNode(entry.name),
            element('span', 'lens-tip-value', rankText(entry.points[index].rank) + ' · ' + percent(entry.points[index].probability)));
          list.append(item);
        }
        tip.append(list);
      };
      capture.addEventListener('pointermove', (event) => {
        const box = chart.getBoundingClientRect(), scale = width / box.width;
        const offset = (event.clientX - box.left) * scale - margin.left;
        const index = layers.length === 1 ? 0 : Math.max(0, Math.min(layers.length - 1, Math.round(offset / plotWidth * (layers.length - 1))));
        marker.setAttribute('x1', x(index)); marker.setAttribute('x2', x(index)); marker.setAttribute('visibility', 'visible');
        describe(index);
        placeTip({left: box.left + x(index) / scale, width: 0, top: box.top + margin.top / scale, bottom: box.top + margin.top / scale, height: 0});
      });
      capture.addEventListener('pointerleave', () => { marker.setAttribute('visibility', 'hidden'); hideTip(); });
      chart.append(capture);
      host.append(chart);

      const head = $('lens-chart-table-head'), body = $('lens-chart-table-body');
      head.replaceChildren(); body.replaceChildren();
      const first = element('tr');
      first.append(element('th', '', 'Layer'));
      for (const entry of series) first.append(element('th', '', entry.name + ' rank'), element('th', '', entry.name + ' probability'));
      head.append(first);
      layers.slice().reverse().forEach((layer) => {
        const index = layers.indexOf(layer), row = element('tr');
        row.append(element('td', '', 'L' + layer));
        for (const entry of series) row.append(element('td', '', rankText(entry.points[index].rank)), element('td', '', percent(entry.points[index].probability)));
        body.append(row);
      });
    }

    /* ---------- whole view ---------- */

    function render() {
      const data = view.data, run = view.run;
      renderPins();
      if (!run || !run.lens_available || !data) return;
      $('lens-body').hidden = false;
      const total = data.forward_count;
      $('lens-forward-status').textContent = 'Step ' + (data.forward + 1) + ' of ' + total;
      $('lens-forward-status').title = 'Each generation step is one full forward pass over the prompt plus the tokens generated so far';
      $('lens-forward-prev').disabled = view.busy || data.forward <= 0;
      $('lens-forward-next').disabled = view.busy || data.forward >= total - 1;
      const shown = data.tokens.length, paged = data.position_total > shown;
      $('lens-window').hidden = !paged;
      if (paged) {
        $('lens-window-status').textContent = 'Positions ' + data.position_offset + '–' + (data.position_offset + shown - 1) + ' of ' + data.position_total;
        $('lens-window-prev').disabled = data.position_offset <= 0;
        $('lens-window-next').disabled = data.position_offset + shown >= data.position_total;
      }
      const notes = [];
      if (run.warning) notes.push('Untrained demo weights: the read-out is real arithmetic on random numbers, so the words mean nothing.');
      if (!data.reaches_model_output) notes.push('The model’s last layer was not captured, so the top row is not the model’s real output.');
      if (data.consistency && !data.consistency.matches) notes.push('Check failed: the last layer’s read-out does not match the recorded next token. Treat this read-out as unreliable.');
      if (data.skipped_events) notes.push(data.skipped_events + ' block capture(s) could not be placed on the grid and were left out.');
      status(notes.join(' '), !!(data.consistency && !data.consistency.matches));
      $('lens-scope').textContent = data.scope + (data.consistency && data.consistency.matches
        ? ' Check passed: the last layer’s read-out equals the token the model actually produced.' : '');
      const last = data.tokens[data.tokens.length - 1].position, top = data.layers[data.layers.length - 1].layer_index;
      if (!view.selected || !cellAt(view.selected.layer, view.selected.position)) view.selected = {layer: top, position: last};
      renderGrid(); renderLegend(); renderReadout(); renderChart();
    }

    function setRun(run) {
      const changed = !run || !view.run || run.capture_id !== view.run.capture_id;
      view.run = run || null;
      if (changed) {
        view.request++; view.data = null; view.forward = 0; view.offset = 0; view.selected = null; view.pins = [];
        view.colorBy = 'probability'; view.colorPin = null; $('lens-color').value = 'probability';
        $('lens-body').hidden = true; hideTip(); renderPins();
      }
      if (!run) { status(''); return; }
      if (!run.lens_available) { $('lens-body').hidden = true; status(run.lens_reason || 'The lens read-out is unavailable for this capture.'); return; }
      if (changed && view.active) load();
    }
    function setActive(active) {
      view.active = !!active;
      if (view.active && view.run && view.run.lens_available && !view.data) load();
      if (view.active && view.data) renderChart();   // its width depends on being visible
    }
    function setBusy(busy) {
      view.busy = !!busy;
      for (const id of ['lens-pin-add', 'lens-pin-text', 'lens-color', 'lens-open-vector']) $(id).disabled = view.busy;
      if (view.data) {
        $('lens-forward-prev').disabled = view.busy || view.data.forward <= 0;
        $('lens-forward-next').disabled = view.busy || view.data.forward >= view.data.forward_count - 1;
      }
    }

    $('lens-forward-prev').addEventListener('click', () => { view.forward = Math.max(0, view.forward - 1); view.offset = 0; view.selected = null; load(); });
    $('lens-forward-next').addEventListener('click', () => { view.forward += 1; view.offset = 0; view.selected = null; load(); });
    $('lens-window-prev').addEventListener('click', () => { view.offset = Math.max(0, view.offset - WINDOW); view.selected = null; load(); });
    $('lens-window-next').addEventListener('click', () => { view.offset += WINDOW; view.selected = null; load(); });
    $('lens-color').addEventListener('change', () => {
      view.colorBy = $('lens-color').value;
      if (view.colorBy === 'pinned' && view.colorPin === null && view.pins.length) view.colorPin = view.pins[0].token_id;
      if (view.data) { renderPins(); renderGrid(); renderLegend(); }
    });
    function trackTyped() {
      const typed = $('lens-pin-text').value;
      if (typed && view.run && view.run.lens_available && !view.busy) load(typed);
    }
    $('lens-pin-add').addEventListener('click', trackTyped);
    $('lens-pin-text').addEventListener('keydown', (event) => { if (event.key === 'Enter') { event.preventDefault(); trackTyped(); } });
    $('lens-grid').addEventListener('keydown', moveFocus);
    $('lens-open-vector').addEventListener('click', async () => {
      if (!view.selected || !view.data) return;
      const layer = layerByIndex(view.selected.layer);
      try { await bridge.selectVector(layer.event_index, view.selected.position); }
      catch (error) { status(error.message || 'Could not open that vector.', true); }
    });
    if (global.addEventListener) global.addEventListener('resize', () => { if (view.active && view.data && view.selected) renderChart(); });

    return {setRun, setActive, setBusy};
  }

  global.MLGLens = {create, tokenParts, percent, rankMagnitude, rampColor, inkOn};
})(typeof window !== 'undefined' ? window : globalThis);
