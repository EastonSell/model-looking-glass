'use strict';
(() => {
  const $ = (id) => document.getElementById(id);
  const token = document.querySelector('meta[name="mlg-token"]').content;
  const state = {model: null, run: null, busy: false, events: [], eventOffset: 0, eventTotal: 0, event: null, vector: 0, valueOffset: 0, valueTotal: 0, eventRequest: 0, valueRequest: 0, imageRequest: 0, image: null};
  const pageSize = 100, valuePageSize = 256;
  const display = (value) => typeof value === 'string' ? value : JSON.stringify(value, null, 2);
  const number = (value) => typeof value === 'number' ? (Object.is(value, -0) ? '-0' : String(value)) : String(value ?? '—');
  const bytes = (value) => { const n = Number(value); if (!Number.isFinite(n)) return '—'; return n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KiB` : `${(n / 1048576).toFixed(2)} MiB`; };
  function notice(message, error = false) { $('notice').textContent = message; $('notice').hidden = !message; $('notice').classList.toggle('error', error); }
  async function api(path, body) {
    const options = {headers: {'X-MLG-Token': token}, credentials: 'same-origin', cache: 'no-store'};
    if (body !== undefined) { options.method = 'POST'; options.headers['Content-Type'] = 'application/json'; options.body = JSON.stringify(body); }
    const response = await fetch(`/api/${path}`, options);
    let data;
    try { data = await response.json(); } catch (_) { throw new Error(`The local app returned an invalid response (${response.status}).`); }
    if (!response.ok || data.error) throw new Error(data.error || `Request failed (${response.status}).`);
    return data;
  }
  function syncButtons() {
    ['inspect-button', 'load-button', 'model-path', 'open-button', 'open-path', 'vector-index', 'vector-view', 'capture-mode', 'layers', 'max-input', 'max-output', 'max-events', 'trace-mib', 'prompt', 'image-file'].forEach((id) => { $(id).disabled = state.busy; });
    $('run-button').disabled = state.busy || !state.model || state.model.state === 'quarantined';
    $('load-button').disabled = state.busy || state.model?.state === 'quarantined';
    $('save-button').disabled = state.busy || !state.run;
    $('clear-button').disabled = state.busy || !state.run;
    $('events-prev').disabled = state.busy || state.eventOffset === 0;
    $('events-next').disabled = state.busy || state.eventOffset + pageSize >= state.eventTotal;
    $('values-prev').disabled = state.busy || state.valueOffset === 0;
    $('values-next').disabled = state.busy || state.valueOffset + valuePageSize >= state.valueTotal;
    $('view-vector').disabled = state.busy || !state.event;
    document.querySelectorAll('#event-body button').forEach((button) => { button.disabled = state.busy; });
    document.querySelectorAll('#target-options input').forEach((input) => { input.disabled = state.busy; });
  }
  async function action(message, callback) {
    if (state.busy) return;
    state.busy = true; syncButtons(); notice(message); $('run-status').textContent = message;
    try { await callback(); notice(''); $('run-status').textContent = 'Ready.'; }
    catch (error) {
      try { const current = await api('status'); setModel(current.model); applyLimits(current.limits); } catch (_) { /* Preserve the original failure if status is unavailable. */ }
      const quarantined = state.model?.state === 'quarantined';
      notice((error.message || String(error)) + (quarantined ? ' This model owner is quarantined. Stop and restart the app before loading or running a model again.' : ''), true);
      $('run-status').textContent = quarantined ? 'Quarantined — restart the app to recover.' : 'Stopped. See the message above.';
    }
    finally { state.busy = false; syncButtons(); }
  }
  function applyLimits(limits) {
    if (Number.isSafeInteger(limits?.max_trace_bytes) && limits.max_trace_bytes >= 1048576) { const max = Math.floor(limits.max_trace_bytes / 1048576); $('trace-mib').max = String(max); if (Number($('trace-mib').value) > max) $('trace-mib').value = String(max); }
  }
  function intField(id) {
    const field = $(id), value = Number(field.value);
    if (!Number.isSafeInteger(value) || value < Number(field.min) || value > Number(field.max)) throw new Error(`${field.labels[0].textContent} must be a whole number from ${field.min} to ${field.max}.`);
    return value;
  }
  function settings() {
    const custom = $('capture-mode').value === 'custom';
    let layers = null;
    if (custom && $('layers').value.trim()) {
      const terms = $('layers').value.split(',').map((s) => s.trim());
      if (terms.some((s) => !/^\d+$/.test(s))) throw new Error('Layer indices must be comma-separated, nonnegative whole numbers.');
      layers = terms.map(Number);
      if (layers.some((n) => !Number.isSafeInteger(n)) || new Set(layers).size !== layers.length) throw new Error('Layer indices must be distinct, safe integers.');
    }
    const targets = custom ? Array.from(document.querySelectorAll('#target-options input:checked'), (el) => el.value) : 'auto';
    if (custom && !targets.length) throw new Error('Select at least one observation point.');
    return {targets, layers, max_new_tokens: intField('max-output'), max_input_tokens: intField('max-input'), max_events: intField('max-events'), max_trace_bytes: intField('trace-mib') * 1048576};
  }
  function setModel(info) {
    state.model = info; $('model-badge').textContent = info ? (info.state === 'quarantined' ? 'Quarantined · restart app' : `Loaded: ${info.name || info.family || 'model'}`) : 'No model loaded';
    $('model-info').textContent = info ? display(info) : 'No model loaded.';
    $('target-options').replaceChildren();
    const targets = info?.targets || ['attn', 'mlp', 'block'];
    for (const target of targets) { const name = typeof target === 'string' ? target : target.name; if (!name) continue; const label = document.createElement('label'), input = document.createElement('input'); input.type = 'checkbox'; input.value = name; input.checked = true; label.append(input, document.createTextNode(name)); $('target-options').append(label); }
    const family = String(info?.family || '').toLowerCase();
    const vision = family.includes('vit'); $('image-input-group').hidden = !vision; $('prompt').hidden = vision; document.querySelector('label[for="prompt"]').hidden = vision;
    $('prompt-help').textContent = vision ? 'Classify one image. The app captures supported ViT module outputs; patch positions are not text tokens.' : family.includes('distilbert') ? 'Include [MASK] in your prompt. DistilBERT predicts masked tokens; it is not a chat or autoregressive generation model.' : family.includes('t5') ? 'Use a T5 task prompt, such as “translate English to German: …”. Generation uses explicit uncached forward steps.' : 'GPT-2 continues your prompt with greedy decoding. Each uncached forward step has its own capture coordinates.';
    syncButtons();
  }
  function clearView() {
    state.run = null; state.event = null; state.events = []; state.eventOffset = 0; state.eventTotal = 0; state.valueOffset = 0; state.valueTotal = 0; state.eventRequest++; state.valueRequest++;
    $('result-panel').hidden = true; $('trace-panel').hidden = true; $('vector-panel').hidden = true; $('event-body').replaceChildren(); $('value-body').replaceChildren(); $('bin-body').replaceChildren(); $('bin-plot').replaceChildren(); $('answer').textContent = ''; $('capture-model').textContent = ''; $('capture-info').textContent = ''; $('event-metadata').textContent = ''; $('run-details').textContent = ''; $('save-status').textContent = 'Optional. Saving includes potentially sensitive prompts, model output, metadata, and full tensor values.'; syncButtons();
  }
  function showRun(run) {
    state.run = run; $('result-panel').hidden = false; $('trace-panel').hidden = false; $('answer').textContent = display(run.answer ?? 'No textual output.'); $('task-badge').textContent = run.task || 'Observed run';
    const captureModel = run.model; $('capture-model').textContent = captureModel ? `Capture model: ${captureModel.name || captureModel.family || captureModel.architecture || 'recorded model'} — independent of the model currently loaded above.` : 'Model identity was not recorded in this capture.';
    $('capture-info').textContent = display({model: captureModel || null, settings: run.settings || null});
    $('run-metrics').replaceChildren();
    for (const [value, label] of [[run.event_count ?? '—', 'captured observations'], [bytes(run.total_bytes), 'retained tensor payload'], [Array.isArray(run.steps) ? run.steps.length : '—', 'explicit forward steps']]) { const box = document.createElement('div'), strong = document.createElement('strong'), span = document.createElement('span'); box.className = 'metric'; strong.textContent = String(value); span.textContent = label; box.append(strong, span); $('run-metrics').append(box); }
    $('coverage').textContent = typeof run.coverage === 'string' ? run.coverage : 'Full retained vectors at selected supported observation points. See the coverage record for exact scope.';
    $('run-details').textContent = display({coverage: run.coverage, steps: run.steps, stop_reason: run.stop_reason, estimates: run.estimates, tokens: run.tokens, image_metadata: run.image_metadata}); syncButtons();
  }
  function cell(row, value) { const td = document.createElement('td'); td.textContent = String(value ?? '—'); row.append(td); return td; }
  function eventName(event) { return event.metadata?.module_path || event.module_path || event.metadata?.site?.module_path || `Observation ${event.index}`; }
  function renderEvents() {
    const filter = $('event-filter').value.toLowerCase(); $('event-body').replaceChildren(); let shown = 0;
    for (const event of state.events) {
      const meta = event.metadata || {}; const forward = meta.app_forward_index ?? meta.forward_index ?? event.forward_index ?? '—';
      if (filter && !`${event.index} ${eventName(event)} ${forward} ${meta.layer_index ?? ''}`.toLowerCase().includes(filter)) continue;
      shown++; const row = document.createElement('tr'); row.classList.toggle('selected', event.index === state.event?.index); cell(row, event.index); const nameCell = document.createElement('td'), button = document.createElement('button'); button.type = 'button'; button.textContent = eventName(event); button.disabled = state.busy; button.addEventListener('click', () => action('Reading selected observation…', () => chooseEvent(event))); nameCell.append(button); row.append(nameCell); cell(row, forward); cell(row, (event.shape || meta.saved_shape || meta.shape || []).join(' × ')); cell(row, bytes(event.tensor_bytes)); $('event-body').append(row);
    }
    if (!shown) { const row = document.createElement('tr'); const td = cell(row, 'No observations match this page filter.'); td.colSpan = 5; $('event-body').append(row); }
    $('event-total').textContent = `${state.eventTotal} observations`;
    $('event-page').textContent = state.eventTotal ? `${state.eventOffset + 1}–${Math.min(state.eventOffset + state.events.length, state.eventTotal)} of ${state.eventTotal}${filter ? ` · ${shown} shown` : ''}` : 'No observations'; syncButtons();
  }
  async function loadEvents() {
    const request = ++state.eventRequest; const result = await api(`events?offset=${state.eventOffset}&limit=${pageSize}`);
    if (request !== state.eventRequest) return;
    state.events = result.events || []; state.eventTotal = Number(result.total || 0); renderEvents();
    if (!state.event && state.events.length) await chooseEvent(state.events[0]);
  }
  async function chooseEvent(event) {
    state.event = event; state.vector = 0; state.valueOffset = 0; $('vector-index').value = '0';
    const shape = event.shape || event.metadata?.saved_shape || event.metadata?.shape || []; const count = event.vector_count ?? shape.slice(0, -1).reduce((a, b) => a * b, 1);
    $('vector-index').max = String(Math.max(0, count - 1)); $('vector-panel').hidden = false; $('vector-heading').textContent = eventName(event); $('dtype-badge').textContent = event.dtype || event.metadata?.dtype || ''; $('vector-location').textContent = `${shape.join(' × ')} · ${count} vectors. Vector index flattens the preceding axes in row-major order; feature coordinates span the final axis. Consult metadata for source batch/token positions.`; $('event-metadata').textContent = display(event.metadata || event); renderEvents();
    await loadVector();
  }
  async function loadVector() {
    if (!state.event) return;
    const vector = Number($('vector-index').value);
    if (!Number.isSafeInteger(vector) || vector < 0 || vector > Number($('vector-index').max)) throw new Error('Choose a vector index in the displayed range.');
    if (vector !== state.vector) state.valueOffset = 0; state.vector = vector;
    const view = $('vector-view').value, request = ++state.valueRequest;
    const result = await api(`tensor?event=${state.event.index}&vector=${vector}&offset=${state.valueOffset}&limit=${valuePageSize}&view=${view}`);
    if (request !== state.valueRequest) return;
    $('reduction-note').hidden = view !== 'reduced'; $('raw-values').hidden = view !== 'raw'; $('reduced-values').hidden = view !== 'reduced';
    state.valueTotal = Number(result.vector_width ?? result.total ?? 0);
    if (view === 'raw') {
      $('value-body').replaceChildren(); const values = result.values || [], bits = result.raw_bits || [];
      values.forEach((value, i) => { const row = document.createElement('tr'); cell(row, Number(result.offset ?? state.valueOffset) + i); cell(row, number(value)); cell(row, bits[i] === undefined ? '—' : String(bits[i])); $('value-body').append(row); });
      $('value-page').textContent = `${state.valueOffset}–${Math.max(state.valueOffset, state.valueOffset + values.length - 1)} · ${state.valueTotal} features (zero-based)`;
    } else renderBins(result.bins || []);
    syncButtons();
  }
  function renderBins(bins) {
    $('bin-body').replaceChildren(); $('bin-plot').replaceChildren(); const ns = 'http://www.w3.org/2000/svg', svg = document.createElementNS(ns, 'svg'); svg.setAttribute('viewBox', '0 0 800 150'); svg.setAttribute('preserveAspectRatio', 'none');
    const max = Math.max(...bins.map((bin) => Math.abs(Number(bin.mean) || 0)), 1e-30); const width = 800 / Math.max(bins.length, 1);
    const baseline = document.createElementNS(ns, 'line'); for (const [key, value] of Object.entries({x1: '0', x2: '800', y1: '75', y2: '75', stroke: 'var(--line)'})) baseline.setAttribute(key, value); svg.append(baseline);
    bins.forEach((bin, index) => {
      const row = document.createElement('tr'); cell(row, `${bin.start}–${bin.end} (end exclusive)`); cell(row, number(bin.mean)); cell(row, number(bin.min)); cell(row, number(bin.max)); $('bin-body').append(row);
      const mean = Number(bin.mean); if (!Number.isFinite(mean)) return; const height = Math.abs(mean) / max * 68, rect = document.createElementNS(ns, 'rect');
      for (const [key, value] of Object.entries({x: String(index * width), y: String(mean >= 0 ? 75 - height : 75), width: String(Math.max(.5, width - 1)), height: String(Math.max(.5, height)), fill: mean >= 0 ? 'var(--positive)' : 'var(--negative)'})) rect.setAttribute(key, value);
      const title = document.createElementNS(ns, 'title'); title.textContent = `Features ${bin.start}–${bin.end}: mean ${number(bin.mean)}`; rect.append(title); svg.append(rect);
    }); $('bin-plot').append(svg);
  }
  $('capture-mode').addEventListener('change', () => { $('custom-capture').hidden = $('capture-mode').value !== 'custom'; });
  $('inspect-button').addEventListener('click', () => action('Inspecting local checkpoint metadata…', async () => { const path = $('model-path').value.trim(); if (!path) throw new Error('Enter a local checkpoint folder.'); const info = await api('inspect', {path}); $('model-info').textContent = `Inspection only — not loaded.\n${display(info)}`; }));
  $('load-button').addEventListener('click', () => action('Loading local model into RAM. This may take a while…', async () => { const path = $('model-path').value.trim(); if (!path) throw new Error('Enter a local checkpoint folder.'); clearView(); setModel(null); const info = await api('load', {path}); setModel(info); }));
  $('open-button').addEventListener('click', () => action('Opening saved capture into RAM…', async () => { const path = $('open-path').value.trim(); if (!path) throw new Error('Enter a saved capture folder.'); clearView(); showRun(await api('open', {path})); await loadEvents(); }));
  $('run-button').addEventListener('click', () => action('Running model and capturing supported outputs…', async () => { const options = settings(), prompt = $('prompt').value; const vision = String(state.model?.family || '').toLowerCase().includes('vit'); if (vision && !state.image) throw new Error('Choose an image first.'); if (!vision && !prompt.trim()) throw new Error('Enter a prompt first.'); clearView(); const body = {prompt, settings: options}; if (vision) body.image_base64 = state.image; showRun(await api('run', body)); await loadEvents(); }));
  $('image-file').addEventListener('change', async () => { const request = ++state.imageRequest; state.image = null; const file = $('image-file').files[0]; if (!file) return; if (file.size > 4 * 1048576) { $('image-file').value = ''; notice('Choose an image no larger than 4 MiB.', true); return; } try { const result = await new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.onerror = () => reject(new Error('Could not read this image.')); reader.readAsDataURL(file); }); if (request !== state.imageRequest) return; state.image = String(result).split(',')[1]; $('image-description').textContent = `${file.name} · ${bytes(file.size)} · held in this browser until Clear captured run or page close.`; } catch (error) { notice(error.message, true); } });
  $('save-button').addEventListener('click', () => action('Saving the complete capture…', async () => { const name = $('save-name').value.trim(); if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(name)) throw new Error('Use a save name of 1–64 letters, numbers, underscores, or hyphens, starting with a letter or number.'); const result = await api('save', {name}); $('save-status').textContent = `Saved explicitly: ${result.path || display(result)}. This export contains full capture data; keep it private when inputs are private.`; }));
  $('clear-button').addEventListener('click', () => action('Clearing captured results…', async () => { await api('clear', {}); clearView(); $('prompt').value = ''; state.imageRequest++; state.image = null; $('image-file').value = ''; $('image-description').textContent = 'Select a local PNG, JPEG, or WebP. It is sent only to this local app.'; }));
  $('event-filter').addEventListener('input', renderEvents);
  $('events-prev').addEventListener('click', () => action('Loading previous observations…', async () => { state.eventOffset = Math.max(0, state.eventOffset - pageSize); await loadEvents(); }));
  $('events-next').addEventListener('click', () => action('Loading next observations…', async () => { state.eventOffset += pageSize; await loadEvents(); }));
  $('view-vector').addEventListener('click', () => action('Reading retained vector…', loadVector));
  $('vector-view').addEventListener('change', () => action('Changing display view…', async () => { state.valueOffset = 0; await loadVector(); }));
  $('values-prev').addEventListener('click', () => action('Reading previous coordinates…', async () => { state.valueOffset = Math.max(0, state.valueOffset - valuePageSize); await loadVector(); }));
  $('values-next').addEventListener('click', () => action('Reading next coordinates…', async () => { state.valueOffset += valuePageSize; await loadVector(); }));
  action('Connecting to your local session…', async () => { const result = await api('status'); setModel(result.model); applyLimits(result.limits); for (const item of result.models || []) { const option = document.createElement('option'); option.value = typeof item === 'string' ? item : item.path; option.label = typeof item === 'string' ? item : item.name || item.path; $('model-choices').append(option); } if (result.run) { showRun(result.run); await loadEvents(); } });
})();
