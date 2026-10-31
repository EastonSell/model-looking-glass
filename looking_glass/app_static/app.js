'use strict';
(() => {
  const $ = (id) => document.getElementById(id);
  const token = document.querySelector('meta[name="mlg-token"]').content;
  const state = {coordinate: null, experiment: null, witnessRequest: 0, witnessOffset: 0, witnessTotal: 0, vectorListRequest: 0, vectorListOffset: 0, vectorListTotal: 0, vectorOptions: [], browse: null, saved: [], searchTimer: null, eventQuery: '', filterError: false, eventTotalAll: 0, devices: [], cuda: null, model: null, run: null, busy: false, events: [], eventOffset: 0, eventTotal: 0, event: null, vector: 0, valueOffset: 0, valueTotal: 0, eventRequest: 0, valueRequest: 0, imageRequest: 0, image: null};
  const pageSize = 100, valuePageSize = 256;
  const display = (value) => typeof value === 'string' ? value : JSON.stringify(value, null, 2);
  const number = (value) => typeof value === 'number' ? (Object.is(value, -0) ? '-0' : String(value)) : String(value ?? '—');
  const bytes = (value) => { const n = Number(value); if (!Number.isFinite(n)) return '—'; return n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KiB` : `${(n / 1048576).toFixed(2)} MiB`; };
  function revealOptions(id) { $(id).open = true; }
  function notice(message, error = false) { $('notice').textContent = message; $('notice').hidden = !message; $('notice').classList.toggle('error', error); }
  async function api(path, body) {
    if (body === undefined && /^(events|vectors|tensor)\?/.test(path) && state.run?.capture_id) path += `&capture_id=${encodeURIComponent(state.run.capture_id)}`;
    const options = {headers: {'X-MLG-Token': token}, credentials: 'same-origin', cache: 'no-store'};
    if (body !== undefined) { options.method = 'POST'; options.headers['Content-Type'] = 'application/json'; options.body = JSON.stringify(body); }
    const response = await fetch(`/api/${path}`, options);
    let data;
    try { data = await response.json(); } catch (_) { throw new Error(`The local app returned an invalid response (${response.status}).`); }
    if (!response.ok || data?.error) throw new Error(data?.error || `Request failed (${response.status}).`);
    return data;
  }
  function syncButtons() {
    ['intervention-operation', 'intervention-feature', 'intervention-value', 'witness-phase', 'experiment-save-name', 'vector-picker', 'browse-button', 'browse-go', 'browse-parent', 'browse-select', 'browse-path', 'saved-runs', 'refresh-saved', 'inspect-button', 'load-button', 'model-path', 'execution-device', 'vram-mib', 'refresh-devices', 'open-button', 'open-path', 'vector-index', 'vector-view', 'capture-mode', 'layers', 'max-input', 'max-output', 'max-events', 'trace-mib', 'prompt', 'image-file'].forEach((id) => { $(id).disabled = state.busy; });
    $('clear-event-filters').disabled = state.busy || !state.run;
    $('run-button').textContent = state.run ? 'Replace capture →' : 'Run & capture →';
    $('replace-warning').hidden = !state.run;
    $('run-button').disabled = state.busy || !state.model || state.model.state === 'quarantined';
    $('demo-button').disabled = state.busy || state.model?.state === 'quarantined';
    $('load-button').disabled = state.busy || state.model?.state === 'quarantined';
    $('unload-button').disabled = state.busy || !state.model || state.model.state === 'quarantined';
    $('browse-parent').disabled = state.busy || !state.browse?.parent;
    document.querySelectorAll('#folder-list button').forEach((button) => { button.disabled = state.busy; });
    $('vram-mib').disabled = state.busy || $('execution-device').value === 'cpu';
    $('save-button').disabled = state.busy || !state.run;
    $('clear-button').disabled = state.busy || !state.run;
    $('events-prev').disabled = state.busy || state.eventOffset === 0;
    $('events-next').disabled = state.busy || state.eventOffset + pageSize >= state.eventTotal;
    $('values-prev').disabled = state.busy || state.valueOffset === 0;
    $('values-next').disabled = state.busy || state.valueOffset + valuePageSize >= state.valueTotal;
    $('vector-prev').disabled = state.busy || !state.event || Number($('vector-index').value) <= 0;
    $('vector-next').disabled = state.busy || !state.event || Number($('vector-index').value) >= Number($('vector-index').max);
    $('vectors-page-prev').disabled = state.busy || !state.event || state.vectorListOffset === 0;
    $('vectors-page-next').disabled = state.busy || !state.event || state.vectorListOffset + 100 >= state.vectorListTotal;
    $('intervene-button').disabled = state.busy || !interventionReady();
    $('intervention-status').textContent = interventionReady() ? 'Ready for the selected, successfully read source vector.' : interventionReason();
    $('save-experiment').disabled = state.busy || !state.experiment;
    $('witness-prev').disabled = state.busy || !state.experiment || state.witnessOffset === 0;
    $('witness-next').disabled = state.busy || !state.experiment || state.witnessOffset + valuePageSize >= state.witnessTotal;
    $('view-vector').disabled = state.busy || !state.event;
    document.querySelectorAll('#event-body button').forEach((button) => { button.disabled = state.busy; });
    document.querySelectorAll('#target-options input').forEach((input) => { input.disabled = state.busy; });
  }
  async function action(message, callback) {
    if (state.busy) return;
    state.busy = true; syncButtons(); notice(message); $('run-status').textContent = message;
    try { await callback(); if (state.model?.state === 'quarantined') { notice('Model owner quarantined. Stop and restart the app before loading or running a model again. Saved captures can still be inspected.', true); $('run-status').textContent = 'Quarantined — restart the app to recover.'; } else { notice(''); $('run-status').textContent = 'Ready.'; } }
    catch (error) {
      try { const current = await api('status'); if (state.run) { const same = current.run?.capture_id && current.run.capture_id === state.run.capture_id; state.run.intervention_available = !!(same && current.run.intervention_available); state.run.intervention_reason = same ? current.run.intervention_reason : 'The displayed capture is no longer bound to the current server revision; reopen or make a fresh capture.'; } setModel(current.model); applyLimits(current.limits); applyDevices(current); } catch (_) { /* Preserve the original failure if status is unavailable. */ }
      const quarantined = state.model?.state === 'quarantined';
      notice((error.message || String(error)) + (quarantined ? ' This model owner is quarantined. Stop and restart the app before loading or running a model again.' : ''), true);
      $('run-status').textContent = quarantined ? 'Quarantined — restart the app to recover.' : 'Stopped. See the message above.';
    }
    finally { state.busy = false; syncButtons(); }
  }
  function applyLimits(limits) {
    if (Number.isSafeInteger(limits?.max_vram_bytes) && limits.max_vram_bytes >= 1048576) { const max = Math.floor(limits.max_vram_bytes / 1048576); $('vram-mib').max = String(max); if (Number($('vram-mib').value) > max) $('vram-mib').value = String(max); }
    if (Number.isSafeInteger(limits?.max_trace_bytes) && limits.max_trace_bytes >= 1048576) { const max = Math.floor(limits.max_trace_bytes / 1048576); $('trace-mib').max = String(max); if (Number($('trace-mib').value) > max) $('trace-mib').value = String(max); }
  }
  function applyDevices(status) {
    $('model-discovery-warning').textContent = status.model_discovery_warning ? `Model folder discovery unavailable: ${status.model_discovery_warning}` : '';
    $('model-discovery-warning').hidden = !status.model_discovery_warning;
    state.devices = Array.isArray(status.devices) ? status.devices : [];
    state.cuda = status.cuda || null;
    const select = $('execution-device'), selected = select.value || 'auto';
    select.replaceChildren();
    function option(id, label, disabled = false) { const el = document.createElement('option'); el.value = id; el.textContent = label; el.disabled = disabled; select.append(el); }
    option('auto', 'Auto — prefer CUDA for supported families'); option('cpu', 'CPU');
    for (const device of state.devices) {
      if (typeof device.id !== 'string' || !/^cuda:\d+$/.test(device.id)) continue;
      option(device.id, `${device.id} · ${device.name || 'CUDA GPU'}${device.available === false ? ' · unavailable' : ''}`, device.available === false);
    }
    if (selected !== 'auto' && selected !== 'cpu' && !state.devices.some((device) => device.id === selected && device.available !== false)) {
      if (!state.devices.some((device) => device.id === selected)) option(selected, `${selected} · unavailable`, true);
    }
    select.value = selected;
    const available = state.devices.filter((device) => /^cuda:\d+$/.test(device.id || '') && device.available !== false);
    const lines = available.map((device) => `${device.id}: ${device.name || 'CUDA GPU'} · ${bytes(device.free_bytes)} free / ${bytes(device.total_bytes)} total at last check`);
    if (!lines.length) lines.push(`CUDA unavailable${state.cuda?.reason ? ': ' + state.cuda.reason : ' or not reported'}. Auto will use CPU; an explicit CUDA choice will not fall back.`);
    else lines.push('Auto prefers available CUDA for GPT-2/T5/DistilBERT/ViT and supported standard Llama, with its reason recorded. Actual GPU validation remains outstanding. An explicit CUDA failure will not silently switch to CPU.');
    $('device-status').textContent = lines.join('\n'); syncButtons();
  }
  function loadSettings() {
    const device = $('execution-device').value;
    if (device !== 'auto' && device !== 'cpu' && !state.devices.some((item) => item.id === device && item.available !== false)) { revealOptions('model-options'); throw new Error('The selected CUDA device is unavailable. Choose an available device; no CPU fallback was performed.'); }
    return device === 'cpu' ? {device} : {device, max_vram_bytes: intField('vram-mib') * 1048576};
  }
  function intField(id) {
    const field = $(id), value = Number(field.value);
    if (!Number.isSafeInteger(value) || value < Number(field.min) || value > Number(field.max)) { revealOptions(id === 'vram-mib' ? 'model-options' : id === 'intervention-feature' ? 'intervention-options' : 'capture-options'); throw new Error(`${field.labels[0].textContent} must be a whole number from ${field.min} to ${field.max}.`); }
    return value;
  }
  function settings() {
    const custom = $('capture-mode').value === 'custom';
    let layers = null;
    if (custom && $('layers').value.trim()) {
      const terms = $('layers').value.split(',').map((s) => s.trim());
      if (terms.some((s) => !/^\d+$/.test(s))) { revealOptions('capture-options'); throw new Error('Layer indices must be comma-separated, nonnegative whole numbers.'); }
      layers = terms.map(Number);
      if (layers.some((n) => !Number.isSafeInteger(n)) || new Set(layers).size !== layers.length) { revealOptions('capture-options'); throw new Error('Layer indices must be distinct, safe integers.'); }
    }
    const targets = custom ? Array.from(document.querySelectorAll('#target-options input:checked'), (el) => el.value) : 'auto';
    if (custom && !targets.length) { revealOptions('capture-options'); throw new Error('Select at least one observation point.'); }
    return {targets, layers, max_new_tokens: intField('max-output'), max_input_tokens: intField('max-input'), max_events: intField('max-events'), max_trace_bytes: intField('trace-mib') * 1048576};
  }
  function invalidateInspection() {
    $('inspection-panel').hidden = true; $('inspection-checks').replaceChildren(); $('inspection-details').textContent = ''; $('inspection-status').textContent = ''; $('inspection-path').textContent = ''; $('inspection-summary').textContent = ''; $('inspection-memory').textContent = '';
  }
  function showInspection(info, path) {
    revealOptions('model-options');
    $('inspection-panel').hidden = false; $('inspection-path').textContent = `Inspected folder: ${info.path || path}`;
    const checks = info.inspection || {}; $('inspection-checks').replaceChildren();
    for (const [key,label] of [['metadata_checked','Configuration'],['weight_headers_checked','Safetensors headers'],['input_assets_loadable','Local tokenizer / image processor']]) {
      const row=document.createElement('li');row.textContent=`${label}: ${checks[key] === true ? 'checked' : checks[key] === false ? 'not verified' : 'unknown — server did not report this check'}`;$('inspection-checks').append(row);
    }
    $('inspection-status').textContent = checks.note || 'Inspection only. See individual check results; missing results are not passes.';
    const input = info.input_summary || {}; const inputLabel = `${input.class || 'class not reported'}${Number.isSafeInteger(input.vocabulary_entries) ? ` · ${input.vocabulary_entries} tokenizer vocabulary entries` : ''}${input.image_size ? ` · image size ${display(input.image_size)}` : ''}`;
    const task = info.task || input.kind || 'not reported', devices = Array.isArray(info.supported_devices) ? info.supported_devices.join(', ') : 'not reported';
    $('inspection-summary').textContent = `Family: ${info.family || 'not reported'} · task/input: ${task} · supported devices: ${devices}. Input assets: ${inputLabel}. Weights loaded: ${checks.weights_loaded === false ? 'no' : checks.weights_loaded === true ? 'reported yes — inspect exact record' : 'not reported'}. Device inference tested: ${checks.inference_tested === false ? 'no' : checks.inference_tested === true ? 'reported yes — inspect exact record' : 'not reported'}.`;
    $('inspection-memory').textContent = `Estimated float32 weight storage: ${Number.isSafeInteger(info.estimated_float32_weight_bytes) ? bytes(info.estimated_float32_weight_bytes) : 'not reported'}. This is not peak RAM or VRAM; loading, inference and captures need additional memory.`;
    $('inspection-details').textContent = display(info);
  }
  const isDemo = (info) => info?.provenance?.kind === 'synthetic_untrained_demo';
  function setModel(info) {
    const prior = state.model, selectedTargets = new Set(Array.from(document.querySelectorAll('#target-options input:checked'), (input) => input.value));
    const preserveTargets = prior && info && prior.path === info.path && prior.family === info.family;
    state.model = info; $('demo-model-warning').hidden = !isDemo(info); $('model-badge').textContent = info ? (info.state === 'quarantined' ? 'Quarantined · restart app' : `${isDemo(info) ? 'Untrained CPU demo' : 'Loaded'}: ${info.name || info.family || 'model'}${info.device ? ' · ' + info.device : ''}`) : 'No model loaded';
    $('model-info').textContent = info ? display(info) : 'No model loaded.';
    $('target-options').replaceChildren();
    const targets = info?.targets || ['attn', 'mlp', 'block'];
    for (const target of targets) { const name = typeof target === 'string' ? target : target.name; if (!name) continue; const label = document.createElement('label'), input = document.createElement('input'); input.type = 'checkbox'; input.value = name; input.checked = !preserveTargets || selectedTargets.has(name); label.append(input, document.createTextNode(name)); $('target-options').append(label); }
    const family = String(info?.family || '').toLowerCase();
    const vision = family.includes('vit'); $('image-input-group').hidden = !vision; $('prompt').hidden = vision; document.querySelector('label[for="prompt"]').hidden = vision;
    $('prompt-help').textContent = vision ? 'Classify one image. The app captures supported ViT module outputs; patch positions are not text tokens.' : family.includes('distilbert') ? 'Include [MASK] in your prompt. DistilBERT predicts masked tokens; it is not a chat or autoregressive generation model.' : family === 'llama' ? 'Standard Llama completes your raw/plain prompt with greedy, uncached forward steps; no automatic chat template is applied. This is completion, not a chat conversation. Only default or static Llama 3 RoPE is supported; dynamic, linear, YaRN, LongRoPE and partial rotary configurations are unsupported.' : family.includes('t5') ? 'Use a T5 task prompt, such as “translate English to German: …”. Generation uses explicit uncached forward steps.' : 'GPT-2 continues your prompt with greedy decoding. Each uncached forward step has its own capture coordinates.';
    syncButtons();
  }
  function clearView() {
    clearExperiment(); resetVectorPicker(); clearTimeout(state.searchTimer); state.filterError = false; state.eventQuery = ''; $('source-options-summary').textContent = 'Exact source filters'; $('event-filter').value = ''; $('source-token').value = ''; $('source-forward').value = ''; $('source-stack').value = ''; state.eventTotalAll = 0; state.run = null; state.event = null; state.events = []; state.eventOffset = 0; state.eventTotal = 0; state.valueOffset = 0; state.valueTotal = 0; state.eventRequest++; state.valueRequest++;
    $('imported-control-warning').hidden = true; $('imported-comparison').hidden = true; clearDistribution('imported'); $('demo-capture-warning').hidden = true; $('result-panel').hidden = true; $('trace-panel').hidden = true; $('vector-panel').hidden = true; $('event-body').replaceChildren(); $('value-body').replaceChildren(); $('bin-body').replaceChildren(); $('bin-plot').replaceChildren(); $('answer').textContent = ''; $('capture-model').textContent = ''; $('capture-info').textContent = ''; $('event-metadata').textContent = ''; $('run-details').textContent = ''; $('save-status').textContent = 'Optional. Saving includes potentially sensitive prompts, model output, metadata, and full tensor values.'; syncButtons();
  }
  function showRun(run) {
    state.run = run; $('imported-comparison').hidden = run.run_kind !== 'intervention_comparison'; if (run.run_kind === 'intervention_comparison') { renderDistribution('imported', run.experiment || {}); controlWarning('imported-control-warning', run.experiment || {}, run.model); } else clearDistribution('imported'); $('result-panel').hidden = false; $('trace-panel').hidden = false; $('answer').textContent = display(run.answer ?? 'No textual output.'); $('task-badge').textContent = run.task || 'Observed run';
    const captureModel = run.model; $('demo-capture-warning').hidden = !isDemo(captureModel); $('capture-model').textContent = captureModel ? `Capture model: ${captureModel.name || captureModel.family || captureModel.architecture || 'recorded model'}${captureModel.device ? ' on ' + captureModel.device : ''} — independent of the model currently loaded above.` : 'Model identity was not recorded in this capture.';
    $('capture-info').textContent = display({model: captureModel || null, settings: run.settings || null});
    $('run-metrics').replaceChildren();
    for (const [value, label] of [[run.event_count ?? '—', 'captured observations'], [bytes(run.total_bytes), 'CPU RAM tensor payload'], [Array.isArray(run.steps) ? run.steps.length : '—', 'explicit forward steps']]) { const box = document.createElement('div'), strong = document.createElement('strong'), span = document.createElement('span'); box.className = 'metric'; strong.textContent = String(value); span.textContent = label; box.append(strong, span); $('run-metrics').append(box); }
    $('coverage').textContent = typeof run.coverage === 'string' ? run.coverage : 'Full retained vectors at selected supported observation points. See the coverage record for exact scope.';
    $('run-details').textContent = display({run_kind: run.run_kind, experiment: run.experiment, coverage: run.coverage, steps: run.steps, stop_reason: run.stop_reason, estimates: run.estimates, tokens: run.tokens, image_metadata: run.image_metadata}); syncButtons();
  }
  function cell(row, value) { const td = document.createElement('td'); td.textContent = String(value ?? '—'); row.append(td); return td; }
  function eventName(event) { return event.metadata?.module_path || event.module_path || event.metadata?.site?.module_path || `Observation ${event.index}`; }
  function sourceSummary(event) {
    const summary = event.source_summary;
    if (!summary) return 'Source unknown — not recorded by this catalog.';
    if (summary.status === 'multiple') return `${summary.vector_count ?? event.vector_count ?? '?'} vectors · multiple source positions; select a vector to inspect.`;
    if (summary.status === 'known' && summary.coordinate?.status === 'known') { const match = Number.isSafeInteger(event.matched_vector_index) && (summary.vector_count ?? event.vector_count ?? 1) > 1 ? `Match vector ${event.matched_vector_index} of ${summary.vector_count ?? event.vector_count} (index zero-based) · ` : ''; return match + coordinateLabel(summary.coordinate, event.matched_vector_index ?? 0); }
    return `Source unknown: ${summary.reason || 'No validated source mapping.'}`;
  }
  function sourceFilterQuery() {
    let query = '';
    for (const [id, key] of [['source-token','token_position'], ['source-forward','forward']]) {
      const raw = $(id).value.trim(); if (!raw) continue;
      if (!/^\d+$/.test(raw) || !Number.isSafeInteger(Number(raw))) throw new Error('Source position and forward filters must be nonnegative safe whole numbers, or blank.');
      query += `&${key}=${Number(raw)}`;
    }
    const stack = $('source-stack').value;
    if (stack && !['input','encoder','decoder','image'].includes(stack)) throw new Error('Choose a supported source stack or All stacks.');
    if (stack) query += `&stack=${encodeURIComponent(stack)}`;
    return query;
  }
  function renderEvents() {
    const filter = state.eventQuery || $('source-token').value || $('source-forward').value || $('source-stack').value; $('event-body').replaceChildren(); let shown = 0;
    for (const event of state.events) {
      const meta = event.metadata || {}; const forward = meta.app_forward_index ?? meta.forward_index ?? event.forward_index ?? '—';
      shown++; const row = document.createElement('tr'); row.classList.toggle('selected', event.index === state.event?.index); cell(row, event.index); const nameCell = document.createElement('td'), button = document.createElement('button'); button.type = 'button'; button.textContent = eventName(event); button.disabled = state.busy; button.addEventListener('click', () => action('Reading selected observation…', () => chooseEvent(event))); nameCell.append(button); row.append(nameCell); cell(row, forward); cell(row, sourceSummary(event)); cell(row, (event.shape || meta.saved_shape || meta.shape || []).join(' × ')); cell(row, bytes(event.tensor_bytes)); $('event-body').append(row);
    }
    if (!shown) { const row = document.createElement('tr'); const td = cell(row, 'No observations match this search.'); td.colSpan = 6; $('event-body').append(row); }
    $('event-total').textContent = `${state.eventTotal} matching / ${state.eventTotalAll} total observations`;
    $('event-page').textContent = state.eventTotal ? `${state.eventOffset + 1}–${Math.min(state.eventOffset + state.events.length, state.eventTotal)} of ${state.eventTotal}${filter ? ' · filtered across all captures' : ''}` : 'No observations'; syncButtons();
  }
  async function loadEvents() {
    const request = ++state.eventRequest, filters = sourceFilterQuery(); const result = await api(`events?offset=${state.eventOffset}&limit=${pageSize}&q=${encodeURIComponent(state.eventQuery)}${filters}`);
    if (request !== state.eventRequest) return;
    state.events = result.events || []; state.eventTotal = Number(result.total || 0); state.eventTotalAll = Number(result.total_all ?? result.total ?? 0); renderEvents();
    if (!state.event && state.events.length) await chooseEvent(state.events[0]);
    if (request === state.eventRequest && state.filterError) { state.filterError = false; notice(''); }
  }
  async function chooseEvent(event) {
    resetVectorPicker(); state.valueRequest++; state.event = null; $('vector-panel').hidden = true;
    const shape = event.shape || event.metadata?.saved_shape || event.metadata?.shape || []; const count = event.vector_count ?? shape.slice(0, -1).reduce((a, b) => a * b, 1);
    const matched = event.matched_vector_index ?? 0;
    if (!Number.isSafeInteger(matched) || matched < 0 || matched >= count) throw new Error('The event catalog returned an invalid matched vector index; refresh the capture instead of guessing a source.');
    state.event = event; state.vector = matched; state.valueOffset = 0; $('vector-index').value = String(matched);
    $('vector-index').max = String(Math.max(0, count - 1)); $('vector-panel').hidden = false; $('vector-heading').textContent = eventName(event); $('dtype-badge').textContent = event.dtype || event.metadata?.dtype || ''; $('vector-location').textContent = `${shape.join(' × ')} · ${count} vectors. Vector index flattens the preceding axes in row-major order; feature coordinates span the final axis. Consult metadata for source batch/token positions.`; $('event-metadata').textContent = display(event.metadata || event); renderEvents();
    await loadVector();
  }
  async function loadVector() {
    if (!state.event) return;
    const request = ++state.valueRequest; state.coordinate = null; syncButtons();
    // Never display the preceding tensor beneath a newly selected coordinate.
    $('value-body').replaceChildren(); $('bin-body').replaceChildren(); $('bin-plot').replaceChildren();
    $('raw-values').hidden = true; $('reduced-values').hidden = true; $('reduction-note').hidden = true;
    $('vector-coordinate').textContent = 'Coordinate unavailable until this vector request succeeds.';
    $('value-page').textContent = 'Values unavailable until this request succeeds.'; state.valueTotal = 0;
    const vector = Number($('vector-index').value);
    if (!Number.isSafeInteger(vector) || vector < 0 || vector > Number($('vector-index').max)) { revealOptions('vector-options'); throw new Error('Choose a vector index in the displayed range.'); }
    if (vector !== state.vector) state.valueOffset = 0; state.vector = vector;
    const view = $('vector-view').value;
    const result = await api(`tensor?event=${state.event.index}&vector=${vector}&offset=${state.valueOffset}&limit=${valuePageSize}&view=${view}`);
    if (request !== state.valueRequest) return;
    state.coordinate = result.coordinate || null; $('intervention-feature').max = String(Math.max(0, Number(result.vector_width ?? result.total ?? 0) - 1));
    $('vector-coordinate').textContent = coordinateLabel(result.coordinate, vector);
    $('reduction-note').hidden = view !== 'reduced'; $('raw-values').hidden = view !== 'raw'; $('reduced-values').hidden = view !== 'reduced';
    state.valueTotal = Number(result.vector_width ?? result.total ?? 0);
    if (view === 'raw') {
      $('value-body').replaceChildren(); const values = result.values || [], bits = result.raw_bits || [];
      values.forEach((value, i) => { const row = document.createElement('tr'); cell(row, Number(result.offset ?? state.valueOffset) + i); cell(row, number(value)); cell(row, bits[i] === undefined ? '—' : String(bits[i])); $('value-body').append(row); });
      $('value-page').textContent = `${state.valueOffset}–${Math.max(state.valueOffset, state.valueOffset + values.length - 1)} · ${state.valueTotal} features (zero-based)`;
    } else renderBins(result.bins || []);
    const page = Math.floor(vector / 100) * 100;
    if (!state.vectorOptions.length || page !== state.vectorListOffset) { state.vectorListOffset = page; await loadVectorList(); }
    else $('vector-picker').value = String(vector);
    syncButtons();
  }
  function resetVectorPicker() {
    state.coordinate = null; state.vectorListRequest++; state.vectorListOffset = 0; state.vectorListTotal = 0; state.vectorOptions = [];
    $('vector-picker').replaceChildren(); $('vectors-page').textContent = ''; $('vector-list-status').textContent = '';
    $('vector-coordinate').textContent = 'Coordinate unavailable until a vector is read.';
  }
  function coordinateLabel(coordinate, index) {
    const c = coordinate;
    if (!c || c.status !== 'known') return `Vector ${index} · source unknown: ${c?.reason || 'No validated source mapping was recorded.'}`;
    const caveat = c.reason ? ` · ${c.reason}` : '';
    const location = `Vector ${index} · forward ${c.app_forward_index ?? '?'} · ${c.stack || 'unknown stack'} · batch ${c.batch_position ?? '?'}`;
    if (c.image_coordinate) return `${location} · processed-image coordinate ${JSON.stringify(c.image_coordinate)} (not original-image pixels or words)${caveat}`;
    const piece = c.token_piece == null ? 'tokenizer piece unavailable' : `piece ${JSON.stringify(c.token_piece)}`;
    return `${location} · token position ${c.token_position ?? '?'} · ID ${c.token_id ?? '?'} · ${piece}${c.token_label_truncated ? ' · label truncated' : ''}${caveat}`;
  }
  async function loadVectorList() {
    if (!state.event) return;
    const request = ++state.vectorListRequest, eventIndex = state.event.index;
    $('vector-list-status').textContent = 'Reading source positions…';
    try {
      const result = await api(`vectors?event=${eventIndex}&offset=${state.vectorListOffset}&limit=100`);
      if (request !== state.vectorListRequest || state.event?.index !== eventIndex) return;
      state.vectorOptions = result.vectors || []; state.vectorListTotal = Number(result.total || 0); $('vector-picker').replaceChildren();
      for (const item of state.vectorOptions) { const option = document.createElement('option'); option.value = String(item.index); option.textContent = coordinateLabel(item.coordinate, item.index); $('vector-picker').append(option); }
      $('vector-picker').value = state.vectorOptions.some((item) => item.index === state.vector) ? String(state.vector) : '';
      $('vectors-page').textContent = state.vectorListTotal ? `${state.vectorListOffset + 1}–${Math.min(state.vectorListOffset + state.vectorOptions.length, state.vectorListTotal)} of ${state.vectorListTotal} vectors` : 'No source positions';
      $('vector-list-status').textContent = 'Picker labels describe saved source coordinates. Selecting one reads that exact flat vector index.';
    } catch (error) {
      if (request !== state.vectorListRequest || state.event?.index !== eventIndex) return;
      state.vectorOptions = []; $('vector-picker').replaceChildren(); $('vectors-page').textContent = '';
      $('vector-list-status').textContent = `Source-position list unavailable: ${error.message || String(error)}. Use the numeric vector index or previous/next vector controls; retained values remain available.`;
    }
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
  async function browse(path, root) {
    revealOptions('model-options');
    const result = await api('browse-models', root ? {path, root} : {path}); state.browse = result;
    $('folder-browser').hidden = false; $('browse-path').value = result.path;
    $('browse-status').textContent = `Root: ${result.root}${result.truncated ? ' · Folder list truncated; enter a known subfolder path to navigate further.' : ''}`;
    $('folder-list').replaceChildren();
    for (const folder of result.folders || []) {
      const row = document.createElement('div'), enter = document.createElement('button'); row.className = 'folder-row';
      enter.type = 'button'; enter.className = 'quiet'; enter.textContent = folder.name + (folder.is_checkpoint ? ' · checkpoint candidate' : ' /');
      enter.addEventListener('click', () => action('Browsing local folders…', () => browse(folder.path, result.root))); row.append(enter);
      if (folder.is_checkpoint) { const use = document.createElement('button'); use.type = 'button'; use.className = 'secondary'; use.textContent = 'Use checkpoint'; use.addEventListener('click', () => { if (!state.busy) { $('model-path').value = folder.path; invalidateInspection(); $('browse-status').textContent = 'Checkpoint path selected. Inspect or Load explicitly when ready.'; } }); row.append(use); }
      $('folder-list').append(row);
    }
    if (!(result.folders || []).length) $('folder-list').textContent = 'No visible subfolders.';
    syncButtons();
  }
  async function refreshSaved() {
    const result = await api('saved'), selected = $('saved-runs').value; state.saved = result.runs || []; $('saved-runs').replaceChildren();
    const placeholder = document.createElement('option'); placeholder.value = ''; placeholder.textContent = 'Choose a saved capture…'; $('saved-runs').append(placeholder);
    for (const run of state.saved) { const option = document.createElement('option'); option.value = run.path; option.disabled = run.status !== 'complete_unverified'; option.textContent = `${run.name} · ${run.status === 'complete_unverified' ? 'not yet verified' : 'incomplete'} · ${bytes(run.tensor_file_bytes)}`; $('saved-runs').append(option); }
    $('saved-runs').value = state.saved.some((run) => run.path === selected && run.status === 'complete_unverified') ? selected : '';
    $('saved-status').textContent = `${state.saved.length} listed capture folders${result.truncated ? ' · list truncated' : ''}. Listing does not verify tensor integrity; Open validates the selected capture. Incomplete exports cannot be opened from this list.`;
  }
  function interventionReady() {
    return !!(state.run?.intervention_available && state.run.capture_id && state.run.run_kind !== 'intervention_comparison' && ['gpt2', 'llama'].includes(state.model?.family) && state.model.state !== 'quarantined' && state.event && Number($('vector-index').value) === state.vector && state.coordinate?.status === 'known');
  }
  function interventionReason() {
    if (!state.model || state.model.state === 'quarantined') return 'Load a ready live GPT-2 or supported standard Llama model to replay. Unload or quarantine disables new edits; saved comparisons remain inspectable.';
    if (!['gpt2', 'llama'].includes(state.model.family)) return 'Fixed-input interventions support live GPT-2 or supported standard Llama only. Other model captures remain inspectable but cannot be edited here.';
    if (!state.run?.intervention_available || state.run.run_kind === 'intervention_comparison') return state.run?.intervention_reason || 'This capture is read-only for intervention; make a fresh supported live GPT-2 or standard Llama capture.';
    if (state.coordinate?.status !== 'known' || Number($('vector-index').value) !== state.vector) return 'Read a vector with validated source coordinates first. Unknown source mappings cannot be edited.';
    return 'A current live GPT-2 or supported standard Llama capture identifier is required.';
  }
  function clearWitness() {
    state.witnessRequest++; state.witnessTotal = 0; $('witness-body').replaceChildren(); $('witness-coordinate').textContent = ''; $('witness-page').textContent = ''; $('witness-status').textContent = '';
  }
  function clearExperiment() {
    $('experiment-control-warning').hidden = true; $('experiment-control-warning').textContent = ''; clearDistribution('experiment'); $('demo-experiment-warning').hidden = true; state.experiment = null; state.witnessOffset = 0; clearWitness(); $('experiment-panel').hidden = true; $('experiment-token-body').replaceChildren(); $('experiment-details').textContent = ''; $('experiment-outcome').textContent = ''; $('experiment-controls').textContent = ''; $('experiment-metrics').textContent = ''; $('experiment-startup-status').textContent = ''; $('experiment-save-status').textContent = 'Optional export includes private fixed input, metadata and full witness tensors. Original capture is separate.';
  }
  function clearDistribution(prefix) {
    $(`${prefix}-distribution-body`).replaceChildren(); $(`${prefix}-distribution-note`).textContent = ''; $(`${prefix}-distribution-table`).hidden = true; $(`${prefix}-token-body`).replaceChildren();
  }
  function renderDistribution(prefix, experiment) {
    clearDistribution(prefix);
    const d = experiment.next_token_distribution;
    const finite = (value) => typeof value === 'number' && Number.isFinite(value);
    const usable = d && d.normalization === 'full vocabulary' && d.computation_dtype === 'float64' && Number.isSafeInteger(d.vocabulary_size) && d.vocabulary_size > 0 && Number.isSafeInteger(d.readout_position) && d.readout_position >= 0 && (experiment.readout_position === undefined || experiment.readout_position === d.readout_position) && Array.isArray(d.rows) && d.rows.length > 0 && d.rows.length <= 10 && d.rows.every((row) => row !== null && typeof row === 'object' && !Array.isArray(row)) && new Set(d.rows.map((row) => row.token_id)).size === d.rows.length && d.rows.every((row) => Number.isSafeInteger(row.token_id) && row.token_id >= 0 && row.token_id < d.vocabulary_size && typeof row.text === 'string' && ['baseline_logit','intervention_logit','baseline_log_probability','intervention_log_probability','baseline_probability','intervention_probability','probability_delta'].every((key) => finite(row[key])) && row.baseline_probability >= 0 && row.baseline_probability <= 1 && row.intervention_probability >= 0 && row.intervention_probability <= 1 && Math.abs(row.probability_delta) <= 1 && Math.abs(row.probability_delta - (row.intervention_probability - row.baseline_probability)) <= 8 * Number.EPSILON && ['baseline_rank','intervention_rank'].every((key) => Number.isSafeInteger(row[key]) && row[key] >= 1 && row[key] <= d.vocabulary_size));
    $(`${prefix}-legacy`).hidden = !!usable;
    if (!usable) {
      $(`${prefix}-distribution-note`).textContent = 'Probability unavailable: this record lacks compatible full-vocabulary normalized metadata. Legacy ranked logits are shown below; no probabilities are inferred or normalized from a top-token list.';
      const before = experiment.baseline_top_next_tokens || [], after = experiment.intervention_top_next_tokens || [];
      for (let i=0; i < Math.max(before.length, after.length); i++) { const row=document.createElement('tr'); cell(row,i+1); cell(row,before[i] ? `${JSON.stringify(before[i].text)} · ${before[i].token_id}` : '—'); cell(row,number(before[i]?.logit)); cell(row,after[i] ? `${JSON.stringify(after[i].text)} · ${after[i].token_id}` : '—'); cell(row,number(after[i]?.logit)); $(`${prefix}-token-body`).append(row); }
      return;
    }
    $(`${prefix}-distribution-note`).textContent = `Full-vocabulary normalized probabilities (${d.vocabulary_size} tokens). Only the top-5 union is shown; these rows need not sum to 100%.`;
    const percent = (value, signed = false) => `${signed && value > 0 ? '+' : ''}${Number((value * 100).toPrecision(6))}`;
    for (const item of d.rows) { const row=document.createElement('tr'); cell(row,`${item.text_unavailable ? 'text unavailable' : JSON.stringify(item.text)} · ${item.token_id}${item.text_truncated ? ' · text truncated' : ''}`); cell(row,percent(item.baseline_probability)); cell(row,percent(item.intervention_probability)); cell(row,percent(item.probability_delta,true)); $(`${prefix}-distribution-body`).append(row); }
    $(`${prefix}-distribution-table`).hidden = false;
  }
  function controlWarning(id, experiment, model) {
    const controls = experiment.controls || {}, entries = Object.entries(controls);
    const required = ['null_full_logits_bitwise_equal', 'restoration_full_logits_bitwise_equal', 'source_vector_bitwise_equal', 'other_token_positions_at_selected_site_bitwise_equal'];
    if (['add', 'replace'].includes(experiment.operation)) required.push('other_features_at_selected_vector_bitwise_equal');
    if (model?.family === 'llama') required.push('static_rotary_state_unchanged', 'fixed_input_ids_mask_unchanged');
    const failed = entries.some(([, value]) => value === false), incomplete = !['zero', 'add', 'replace'].includes(experiment.operation) || required.some((key) => controls[key] !== true) || entries.some(([, value]) => value !== true);
    $(id).hidden = !incomplete;
    $(id).textContent = failed ? 'A recorded control check FAILED. Do not treat this comparison as validated; inspect the controls and exact metadata.' : incomplete ? 'Control validation is incomplete or unavailable. Inspect the exact metadata before relying on this comparison.' : '';
    return incomplete;
  }
  function showExperiment(summary) {
    $('demo-experiment-warning').hidden = !isDemo(summary.model); state.experiment = summary; const e = summary.experiment || {}, c = e.comparison || {}; $('experiment-panel').hidden = false;
    $('experiment-outcome').textContent = c.selected_vector_changed === false ? 'No-op: the selected vector did not change. Do not interpret this as evidence about the effect of changing it.' : c.selected_vector_changed === true ? `Selected vector changed. Baseline next-token ID ${c.baseline_next_token_id ?? '?'} → edited ID ${c.intervention_next_token_id ?? '?'}. This is one fixed-input comparison, not a generated continuation or explanation.` : 'Comparison returned without a verified selected-vector change flag; inspect the complete metadata.';
    if (controlWarning('experiment-control-warning', e, summary.model)) revealOptions('comparison-checks');
    const controls = Object.entries(e.controls || {}); $('experiment-controls').textContent = controls.length ? 'Recorded controls: ' + controls.map(([key,value]) => `${key}: ${value === true ? 'passed' : value === false ? 'FAILED' : display(value)}`).join(' · ') : 'No control results recorded.';
    $('experiment-metrics').textContent = `Full-logits max |Δ|: ${number(c.full_logits_max_abs_delta)} · full-logits L2 Δ: ${number(c.full_logits_l2_delta)} · last-position max |Δ|: ${number(c.last_position_max_abs_delta)}`;
    $('experiment-details').textContent = display(summary); renderDistribution('experiment', e);
    syncButtons();
  }
  async function loadWitness() {
    if (!state.experiment) return;
    clearWitness(); const request=state.witnessRequest, phase=$('witness-phase').value; $('witness-status').textContent='Reading exact witness values…';
    try { const result=await api(`experiment-tensor?event=${phase}&vector=0&offset=${state.witnessOffset}&limit=${valuePageSize}&experiment_id=${encodeURIComponent(state.experiment.experiment_id || "")}`);
      if(request!==state.witnessRequest || !state.experiment) return;
      state.witnessTotal=Number(result.vector_width ?? result.total ?? 0); $('witness-coordinate').textContent=coordinateLabel(result.coordinate,0);
      (result.values || []).forEach((value,i)=>{const row=document.createElement('tr');cell(row,Number(result.offset ?? state.witnessOffset)+i);cell(row,number(value));cell(row,result.raw_bits?.[i] ?? '—');$('witness-body').append(row);});
      $('witness-page').textContent=`${state.witnessOffset}–${Math.max(state.witnessOffset,state.witnessOffset+(result.values || []).length-1)} of ${state.witnessTotal} features (zero-based)`; $('witness-status').textContent='Exact retained witness; independent of original capture values.';
    } catch(error) {if(request===state.witnessRequest) { revealOptions('witness-options'); $('witness-status').textContent=`Witness unavailable: ${error.message || String(error)}. The comparison summary remains available.`; }}
    syncButtons();
  }
  async function restoreExperimentOptional() {
    try { const summary=await api('experiment'); if(summary?.run_kind==='intervention_comparison') {showExperiment(summary); await loadWitness();} }
    catch(error) {$('experiment-startup-status').textContent=`Comparison could not be restored: ${error.message || String(error)}. Original capture access is unaffected.`;}
  }
  async function refreshSavedOptional() {
    try { await refreshSaved(); }
    catch (error) {
      revealOptions('saved-options'); state.saved = []; $('saved-runs').replaceChildren();
      $('saved-status').textContent = `Saved-capture catalog unavailable: ${error.message || String(error)}. This does not discard a RAM capture or undo a completed save. Refresh the catalog to retry, or enter a known export path.`;
    }
  }
  $('browse-button').addEventListener('click', () => action('Browsing local folders…', () => browse($('model-path').value.trim())));
  $('browse-go').addEventListener('click', () => action('Browsing local folders…', () => browse($('browse-path').value.trim(), state.browse?.root)));
  $('browse-parent').addEventListener('click', () => action('Browsing parent folder…', async () => { if (state.browse?.parent) await browse(state.browse.parent, state.browse.root); }));
  $('browse-select').addEventListener('click', () => { if (!state.busy && state.browse) { $('model-path').value = state.browse.path; invalidateInspection(); $('browse-status').textContent = 'Folder path selected. Inspect or Load explicitly when ready.'; } });
  $('refresh-saved').addEventListener('click', () => action('Listing saved captures…', refreshSaved));
  $('saved-runs').addEventListener('change', () => { const run = state.saved.find((item) => item.path === $('saved-runs').value && item.status === 'complete_unverified'); if (run) $('open-path').value = run.path; });
  $('unload-button').addEventListener('click', () => action('Releasing the model owner; keeping captured vectors…', async () => { await api('unload', {}); const current = await api('status'); setModel(current.model); applyLimits(current.limits); applyDevices(current); }));
  $('execution-device').addEventListener('change', syncButtons);
  $('refresh-devices').addEventListener('click', () => action('Checking actual device availability…', async () => { const current = await api('status'); setModel(current.model); applyLimits(current.limits); applyDevices(current); }));
  $('capture-mode').addEventListener('change', () => { $('custom-capture').hidden = $('capture-mode').value !== 'custom'; });
  $('model-path').addEventListener('input', invalidateInspection);
  $('inspect-button').addEventListener('click', () => action('Checking local configuration, weight headers and input assets…', async () => {
    const path = $('model-path').value.trim(); if (!path) throw new Error('Enter a local checkpoint folder.'); invalidateInspection();
    try { showInspection(await api('inspect', {path}), path); }
    catch(error) { revealOptions('model-options'); $('inspection-panel').hidden=false; $('inspection-path').textContent=`Requested folder: ${path}`; $('inspection-status').textContent=`Inspection failed: ${error.message || String(error)}. Check the complete local snapshot: config.json, matching safetensors and matching tokenizer or processor assets. No download was started.`; throw error; }
  }));
  $('demo-button').addEventListener('click', () => action('Building the tiny untrained CPU demo in RAM…', async () => {
    const result = await api('demo', {}); clearView(); setModel(result.model);
    $('capture-mode').value = 'auto'; $('custom-capture').hidden = true; $('layers').value = '';
    for (const [id, maximum] of [['max-output', 8], ['max-input', 120]]) { const current = Number($(id).value); $(id).value = String(Number.isSafeInteger(current) && current >= Number($(id).min) ? Math.min(current, maximum) : maximum); }
    if (!$('prompt').value.trim()) $('prompt').value = 'Hello, tiny model!';
  }));
  $('load-button').addEventListener('click', () => action('Loading local model on the selected device. This may take a while…', async () => { const path = $('model-path').value.trim(); if (!path) throw new Error('Enter a local checkpoint folder.'); const execution = loadSettings(); const info = await api('load', {path, ...execution}); clearView(); setModel(info); }));
  $('open-button').addEventListener('click', () => action('Opening saved capture into RAM…', async () => { const path = $('open-path').value.trim(); if (!path) throw new Error('Enter a saved capture folder.'); const opened = await api('open', {path}); clearView(); showRun(opened); await loadEvents(); $('saved-status').textContent = 'Selected capture opened and validated. Other listed exports remain unverified.'; }));
  $('run-button').addEventListener('click', () => action('Running model and capturing supported outputs…', async () => { const options = settings(), prompt = $('prompt').value; const vision = String(state.model?.family || '').toLowerCase().includes('vit'); if (vision && !state.image) throw new Error('Choose an image first.'); if (!vision && !prompt.trim()) throw new Error('Enter a prompt first.'); clearView(); const body = {prompt, settings: options}; if (vision) body.image_base64 = state.image; showRun(await api('run', body)); await loadEvents(); }));
  $('image-file').addEventListener('change', async () => { const request = ++state.imageRequest; state.image = null; const file = $('image-file').files[0]; if (!file) return; if (file.size > 4 * 1048576) { $('image-file').value = ''; notice('Choose an image no larger than 4 MiB.', true); return; } try { const result = await new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.onerror = () => reject(new Error('Could not read this image.')); reader.readAsDataURL(file); }); if (request !== state.imageRequest) return; state.image = String(result).split(',')[1]; $('image-description').textContent = `${file.name} · ${bytes(file.size)} · held in this browser until Clear captured run or page close.`; } catch (error) { notice(error.message, true); } });
  $('save-button').addEventListener('click', () => action('Saving the complete capture…', async () => { const name = $('save-name').value.trim(); if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(name)) throw new Error('Use a save name of 1–64 letters, numbers, underscores, or hyphens, starting with a letter or number.'); $('save-status').textContent = 'No new export confirmed for this save attempt.'; const result = await api('save', {name, capture_id: state.run.capture_id}); $('save-status').textContent = `Saved explicitly: ${result.path || display(result)}. This export contains full capture data; keep it private when inputs are private.`; await refreshSavedOptional(); }));
  $('clear-button').addEventListener('click', () => action('Clearing captured results…', async () => { await api('clear', {}); clearView(); $('prompt').value = ''; state.imageRequest++; state.image = null; $('image-file').value = ''; $('image-description').textContent = 'Select a local PNG, JPEG, or WebP. It is sent only to this local app.'; }));
  function scheduleEventSearch() {
    clearTimeout(state.searchTimer); state.eventRequest++; state.valueRequest++;
    $('source-options-summary').textContent = ['Exact source filters', $('source-token').value ? 'position ' + $('source-token').value : '', $('source-forward').value ? 'forward ' + $('source-forward').value : '', $('source-stack').value].filter(Boolean).join(' · ');
    resetVectorPicker(); state.eventQuery = $('event-filter').value.trim(); state.eventOffset = 0; state.event = null; state.events = []; state.eventTotal = 0; $('event-body').replaceChildren(); $('event-page').textContent = 'Updating filters…'; $('event-total').textContent = 'Filtering…'; $('vector-panel').hidden = true; syncButtons();
    const revision = state.eventRequest; const search = async () => { if (!state.run || revision !== state.eventRequest) return; if (state.busy) { state.searchTimer = setTimeout(search, 250); return; } try { await loadEvents(); } catch (error) { if (revision + 1 === state.eventRequest) { state.filterError = true; revealOptions('source-options'); notice(error.message || String(error), true); $('event-page').textContent = 'Filter results unavailable. Adjust or clear filters to retry.'; } } }; state.searchTimer = setTimeout(search, 250);
  }
  $('event-filter').addEventListener('input', scheduleEventSearch);
  $('source-token').addEventListener('input', scheduleEventSearch);
  $('source-forward').addEventListener('input', scheduleEventSearch);
  $('source-stack').addEventListener('change', scheduleEventSearch);
  $('clear-event-filters').addEventListener('click', () => { $('event-filter').value = ''; $('source-token').value = ''; $('source-forward').value = ''; $('source-stack').value = ''; scheduleEventSearch(); });
  $('events-prev').addEventListener('click', () => action('Loading previous observations…', async () => { state.eventOffset = Math.max(0, state.eventOffset - pageSize); await loadEvents(); }));
  $('events-next').addEventListener('click', () => action('Loading next observations…', async () => { state.eventOffset += pageSize; await loadEvents(); }));
  $('vector-picker').addEventListener('change', () => action('Reading selected source position…', async () => { const selected = state.vectorOptions.find((item) => String(item.index) === $('vector-picker').value); if (selected) { $('vector-index').value = String(selected.index); await loadVector(); } }));
  $('vector-prev').addEventListener('click', () => action('Reading previous vector…', async () => { $('vector-index').value = String(Math.max(0, Number($('vector-index').value) - 1)); await loadVector(); }));
  $('vector-next').addEventListener('click', () => action('Reading next vector…', async () => { $('vector-index').value = String(Math.min(Number($('vector-index').max), Number($('vector-index').value) + 1)); await loadVector(); }));
  $('vectors-page-prev').addEventListener('click', () => action('Reading source-position picker…', async () => { state.vectorListOffset = Math.max(0, state.vectorListOffset - 100); await loadVectorList(); }));
  $('vectors-page-next').addEventListener('click', () => action('Reading source-position picker…', async () => { state.vectorListOffset += 100; await loadVectorList(); }));
  $('vector-index').addEventListener('input', syncButtons);
  $('intervention-operation').addEventListener('change',()=>{$('intervention-feature-fields').hidden=$('intervention-operation').value==='zero';});
  $('intervene-button').addEventListener('click',()=>action('Replaying fixed input and checking controls…',async()=>{
    if(!interventionReady()) throw new Error(interventionReason());
    const operation=$('intervention-operation').value;
    if(!['zero','add','replace'].includes(operation)) throw new Error('Choose Zero, Add or Set.');
    const body={capture_id:state.run.capture_id,event:state.event.index,vector:state.vector,operation};
    if(operation!=='zero'){body.feature_index=intField('intervention-feature');const input=$('intervention-value').value.trim();if(!input || !Number.isFinite(Number(input))){revealOptions('intervention-options');throw new Error('Enter an explicit finite numeric value for the feature edit.');}body.value=Number(input);}
    clearExperiment();const summary=await api('intervene',body);showExperiment(summary);$('witness-phase').value='0';await loadWitness();
  }));
  $('witness-phase').addEventListener('change',()=>{state.witnessOffset=0;return loadWitness();});
  $('witness-prev').addEventListener('click',()=>{if(state.busy)return;state.witnessOffset=Math.max(0,state.witnessOffset-valuePageSize);return loadWitness();});
  $('witness-next').addEventListener('click',()=>{if(state.busy)return;state.witnessOffset+=valuePageSize;return loadWitness();});
  $('save-experiment').addEventListener('click',()=>action('Saving only the separate comparison…',async()=>{const name=$('experiment-save-name').value.trim();if(!/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(name))throw new Error('Use a comparison name of 1–64 letters, numbers, underscores or hyphens, starting with a letter or number.');$('experiment-save-status').textContent='No new comparison export confirmed for this save attempt.';const result=await api('save-experiment',{name,experiment_id:state.experiment.experiment_id});$('experiment-save-status').textContent=`Saved comparison explicitly: ${result.path || display(result)}. Original capture export is separate.`;await refreshSavedOptional();}));
  $('view-vector').addEventListener('click', () => action('Reading retained vector…', loadVector));
  $('vector-view').addEventListener('change', () => action('Changing display view…', async () => { state.valueOffset = 0; await loadVector(); }));
  $('values-prev').addEventListener('click', () => action('Reading previous coordinates…', async () => { state.valueOffset = Math.max(0, state.valueOffset - valuePageSize); await loadVector(); }));
  $('values-next').addEventListener('click', () => action('Reading next coordinates…', async () => { state.valueOffset += valuePageSize; await loadVector(); }));
  action('Connecting to your local session…', async () => { const result = await api('status'); setModel(result.model); applyLimits(result.limits); applyDevices(result); for (const item of result.models || []) { const option = document.createElement('option'); option.value = typeof item === 'string' ? item : item.path; option.label = typeof item === 'string' ? item : item.name || item.path; $('model-choices').append(option); } if (result.run) { showRun(result.run); await loadEvents(); } await refreshSavedOptional(); await restoreExperimentOptional(); });
})();
