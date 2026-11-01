'use strict';
(() => {
  const $ = (id) => document.getElementById(id);
  const token = document.querySelector('meta[name="mlg-token"]').content;
  const state = {browseRequest: 0, browseFresh: false, settingsOpener: 'session-summary', mapButtons: [], outputGeneration: 0, outputStep: 0, tokenOffset: 0, tokenTotal: 0, mapAutoCapture: null, mapRequest: 0, mapOffset: 0, mapTotal: 0, mapForward: 0, mapCapture: null, reference: null, referenceFresh: false, referenceRequest: 0, referenceOffset: 0, referenceTotal: 0, referenceMode: null, coordinate: null, experiment: null, witnessRequest: 0, witnessOffset: 0, witnessTotal: 0, vectorListRequest: 0, vectorListOffset: 0, vectorListTotal: 0, vectorOptions: [], browse: null, saved: [], searchTimer: null, eventQuery: '', filterError: false, eventTotalAll: 0, devices: [], cuda: null, model: null, run: null, busy: false, events: [], eventOffset: 0, eventTotal: 0, event: null, vector: 0, valueOffset: 0, valueTotal: 0, eventRequest: 0, valueRequest: 0, imageRequest: 0, image: null};
  const pageSize = 100, valuePageSize = 256;
  const display = (value) => typeof value === 'string' ? value : JSON.stringify(value, null, 2);
  const number = (value) => typeof value === 'number' ? (Object.is(value, -0) ? '-0' : String(value)) : String(value ?? '—');
  const bytes = (value) => { const n = Number(value); if (!Number.isFinite(n)) return '—'; return n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KiB` : `${(n / 1048576).toFixed(2)} MiB`; };
  let sceneApi=null, sceneFailed=false;
  const sceneSites=new Map();
  function setDock(side,open){
    const dock=$(`${side}-dock`),body=$(`${side}-dock-body`),button=$(`${side}-dock-toggle`);
    body.hidden=!open;dock.setAttribute('data-collapsed',String(!open));button.setAttribute('aria-expanded',String(open));
    button.textContent=side==='selection'?(open?'‹':'›'):(open?'›':'‹');
    const label=side==='selection'?'selected capture':'prompt and response';button.setAttribute('aria-label',`${open?'Collapse':'Expand'} ${label}`);button.setAttribute('title',`${open?'Collapse':'Expand'} ${label} (${side==='selection'?'[':']'})`);
    if(sceneApi)sceneApi.resize();
  }
  function sceneUnavailable(reason){
    sceneFailed=true;$('scene-render-status').textContent=reason||'3D rendering unavailable. Use the captured-site list.';$('scene-view').value='list';$('layer-map-options').open=true;syncSceneVisibility();
  }
  function syncSceneVisibility(){
    const scene=$('scene-view').value==='scene'&&!sceneFailed&&!!sceneApi;
    $('scene-canvas').hidden=!scene||!state.mapCapture;$('layer-map-body').hidden=scene&&!!state.mapCapture;
    if(sceneApi){sceneApi.setActive(scene&&!!state.mapCapture);if(scene&&state.mapCapture)sceneApi.resize();}
  }
  function replaceSceneLayers(layers){
    sceneSites.clear();for(const entry of state.mapButtons)sceneSites.set(entry.sceneKey,entry);
    if(sceneApi&&!sceneFailed){try{sceneApi.setLayers(layers);}catch(_){sceneUnavailable('Captured-site scene unavailable. Use the exact list.');}}
    syncSceneVisibility();
  }
  function initializeScene(){
    const library=typeof window!=='undefined'?window.MLGScene:null;
    if(!library||typeof library.create!=='function'){sceneUnavailable('3D assets unavailable. Use the captured-site list.');return;}
    try{sceneApi=library.create($('scene-canvas'),{onUnavailable:sceneUnavailable,onSelect:key=>{const entry=sceneSites.get(key);if(sceneFailed||!entry||state.busy||!state.mapButtons.includes(entry))return;return entry.select();}});syncSceneVisibility();}
    catch(_){sceneUnavailable('3D rendering unavailable. Use the captured-site list.');}
  }

  function revealOptions(id) { $(id).open = true; if (['vector-options','source-position-options','feature-values-options','intervention-options','reference-options'].includes(id)) setDock('selection',true); if (id === 'map-tools-options') $('layer-map-options').open = true; if (id === 'source-options') $('event-catalog-options').open = true; if (id === 'vector-options') $('source-position-options').open = true; if (['model-options','saved-options','capture-options','privacy-options','download-options'].includes(id)) $('session-options').open = true; }
  function notice(message, error = false) { $('notice').textContent = message; $('notice').hidden = !message; $('notice').classList.toggle('error', error); }
  async function api(path, body) {
    if (body === undefined && /^(events|vectors|tensor|layer-map)\?/.test(path) && state.run?.capture_id) path += `&capture_id=${encodeURIComponent(state.run.capture_id)}`;
    const options = {headers: {'X-MLG-Token': token}, credentials: 'same-origin', cache: 'no-store'};
    if (body !== undefined) { options.method = 'POST'; options.headers['Content-Type'] = 'application/json'; options.body = JSON.stringify(body); }
    const response = await fetch(`/api/${path}`, options);
    let data;
    try { data = await response.json(); } catch (_) { throw new Error(`The local app returned an invalid response (${response.status}).`); }
    if (!response.ok || data?.error) throw new Error(data?.error || `Request failed (${response.status}).`);
    return data;
  }
  function syncButtons() {
    syncDownloads();
    $('dock-exact').disabled = state.busy || !state.event; $('dock-edit').disabled = state.busy || !state.event; $('dock-reference').disabled = state.busy || !state.event;
    $('open-settings').disabled = state.busy;
    // Highlight canonical sites, not just their first event: singleton token captures share a site.
    const coordinate = state.coordinate, metadata = state.event?.metadata;
    const selector = metadata?.output_path;
    const validSelector = Array.isArray(selector) && selector.every(Number.isSafeInteger);
    for (const row of new Set(state.mapButtons.map(item => item.row))) row.setAttribute('data-selected', 'false');
    let selectedSheet = false, selectedSceneKey=null;
    for (const item of state.mapButtons) {
      const selected = item.capture === state.run?.capture_id && item.capture === state.mapCapture && item.forward === state.mapForward && coordinate?.status === 'known' && coordinate.app_forward_index === item.forward && coordinate.stack === item.stack && coordinate.layer_index === item.layer && coordinate.module_path === item.path && metadata?.module_kind === item.kind && validSelector && JSON.stringify(selector) === item.selector;
      item.button.setAttribute('aria-pressed', String(selected)); if (selected) { item.row.setAttribute('data-selected', 'true'); selectedSheet = true; selectedSceneKey=item.sceneKey; }
    }
    if(sceneApi&&!sceneFailed)sceneApi.selectSite(selectedSceneKey);
    $('selection-empty').hidden=!!state.event;
    $('scene-selection-status').textContent = !state.mapCapture || !state.event ? '' : selectedSheet ? `Selected · position ${coordinate.token_position ?? '?'}` : coordinate?.status !== 'known' ? 'Source unknown · no highlight.' : 'Selection outside this layer page or forward.';
    const stepCount = Array.isArray(state.run?.steps) ? state.run.steps.length : 0;
    $('output-step').disabled = state.busy || !stepCount; $('input-token-stack').disabled = state.busy || !stepCount;
    $('output-prev').disabled = state.busy || state.outputStep === 0; $('output-next').disabled = state.busy || state.outputStep + 1 >= stepCount;
    $('input-tokens-prev').disabled = state.busy || state.tokenOffset === 0; $('input-tokens-next').disabled = state.busy || state.tokenOffset + 48 >= state.tokenTotal;
    document.querySelectorAll('#input-token-strip button').forEach(button => { button.disabled = state.busy || button.dataset?.unavailable === 'true'; });
    ['intervention-operation', 'intervention-feature', 'intervention-value', 'witness-phase', 'experiment-save-name', 'vector-picker', 'browse-button', 'browse-go', 'browse-parent', 'browse-select', 'browse-path', 'saved-runs', 'refresh-saved', 'inspect-button', 'load-button', 'model-path', 'execution-device', 'vram-mib', 'refresh-devices', 'open-button', 'open-path', 'vector-index', 'vector-view', 'capture-mode', 'layers', 'max-input', 'max-output', 'max-events', 'trace-mib', 'prompt', 'image-file'].forEach((id) => { $(id).disabled = state.busy; });
    $('layer-map-load').disabled = state.busy || !state.run; $('layer-map-forward').disabled = state.busy || !state.run;
    $('layer-map-prev').disabled = state.busy || !state.run || state.mapOffset === 0; $('layer-map-next').disabled = state.busy || !state.run || state.mapOffset + 12 >= state.mapTotal;
    document.querySelectorAll('#layer-map-body button').forEach((button) => { button.disabled = state.busy; });
    $('clear-event-filters').disabled = state.busy || !state.run;
    $('reference-selection-status').textContent = referenceSelectionReady() ? 'The selected live vector is eligible to keep. Comparison also requires the same model owner and module site.' : state.run?.reference_reason || 'Select and read a known vector in a supported live GPT-2 or standard Llama capture. Imported or unknown-source captures are read-only.';
    $('pin-reference').disabled = state.busy || !referenceSelectionReady();
    $('compare-reference').disabled = state.busy || !referenceSelectionReady() || !state.referenceFresh || !state.reference?.ready_for_compare;
    $('inspect-reference').disabled = state.busy || !state.reference;
    $('clear-reference').disabled = state.busy || !state.reference;
    $('refresh-reference').disabled = state.busy;
    $('reference-prev').disabled = state.busy || state.referenceOffset === 0 || !state.referenceMode;
    $('reference-next').disabled = state.busy || !state.referenceMode || state.referenceOffset + valuePageSize >= state.referenceTotal;
    $('run-button').textContent = state.run ? 'Run again' : 'Run';
    $('replace-warning').hidden = !state.run || !state.model || state.model.state === 'quarantined';
    $('run-button').disabled = state.busy || !state.model || state.model.state === 'quarantined';
    $('demo-button').disabled = state.busy || state.model?.state === 'quarantined';
    $('load-button').disabled = state.busy || state.model?.state === 'quarantined';
    $('unload-button').disabled = state.busy || !state.model || state.model.state === 'quarantined';
    $('browse-parent').disabled = state.busy || !state.browseFresh || !state.browse?.parent; $('browse-select').disabled = state.busy || !state.browseFresh || !state.browse?.path;
    document.querySelectorAll('#folder-list button').forEach((button) => { button.disabled = state.busy || !state.browseFresh; });
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
    let completed = false; state.busy = true; syncButtons(); notice(message); $('run-status').textContent = message;
    try { await callback(); completed = true; if (state.model?.state === 'quarantined') { notice('Model owner quarantined. Stop and restart the app before loading or running a model again. Saved captures can still be inspected.', true); $('run-status').textContent = 'Quarantined — restart the app to recover.'; } else { notice(''); $('run-status').textContent = 'Ready.'; } }
    catch (error) {
      try { const current = await api('status'); if (error.reconcileRun && (!state.run?.capture_id || current.run?.capture_id !== state.run.capture_id)) clearView(); if (error.reconcileRun && state.experiment && Object.hasOwn(current, 'experiment') && current.experiment?.experiment_id !== state.experiment.experiment_id) clearExperiment(); if (state.run) { const same = current.run?.capture_id && current.run.capture_id === state.run.capture_id; state.run.intervention_available = !!(same && current.run.intervention_available); state.run.intervention_reason = same ? current.run.intervention_reason : 'The displayed capture is no longer bound to the current server revision; reopen or make a fresh capture.'; state.run.reference_available = !!(same && current.run.reference_available); state.run.reference_reason = same ? current.run.reference_reason : 'The displayed capture revision is no longer current.'; } setModel(current.model); applyLimits(current.limits); applyDevices(current); } catch (_) { if (error.reconcileRun) { clearView(); error.message += ' The Run outcome could not be verified; previous capture values are hidden. Refresh or reopen a saved capture before continuing.'; } /* Preserve the original failure if status is unavailable. */ }
      const quarantined = state.model?.state === 'quarantined';
      notice((error.message || String(error)) + (quarantined ? ' This model owner is quarantined. Stop and restart the app before loading or running a model again.' : ''), true);
      $('run-status').textContent = quarantined ? 'Quarantined — restart the app to recover.' : 'Stopped. See the message above.';
    }
    finally { state.busy = false; syncButtons(); if (completed && state.run?.capture_id && $('layer-map-options').open && state.mapAutoCapture !== state.run.capture_id) { state.mapAutoCapture = state.run.capture_id; await loadLayerMap(); } }
  }
  function applyLimits(limits) {
    if (Number.isSafeInteger(limits?.max_vram_bytes) && limits.max_vram_bytes >= 1048576) { const max = Math.floor(limits.max_vram_bytes / 1048576); $('vram-mib').max = String(max); if (Number($('vram-mib').value) > max) $('vram-mib').value = String(max); }
    if (Number.isSafeInteger(limits?.max_trace_bytes) && limits.max_trace_bytes >= 1048576) { const max = Math.floor(limits.max_trace_bytes / 1048576); $('trace-mib').max = String(max); if (Number($('trace-mib').value) > max) $('trace-mib').value = String(max); }
  }
  function applyDevices(status) {
    if (Object.hasOwn(status, 'reference')) setReference(status.reference);
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
    $('scene-empty').hidden = false;
    state.outputGeneration++; state.outputStep = 0; state.tokenOffset = 0; state.tokenTotal = 0;
    $('output-readout').replaceChildren(); $('input-token-strip').replaceChildren(); $('output-note').textContent = ''; $('input-token-note').textContent = ''; $('input-token-page').textContent = ''; $('output-step').value = '0';
    clearLayerMap();
    clearExperiment(); resetVectorPicker(); clearTimeout(state.searchTimer); state.filterError = false; state.eventQuery = ''; $('source-options-summary').textContent = 'Exact source filters'; $('event-filter').value = ''; $('source-token').value = ''; $('source-forward').value = ''; $('source-stack').value = ''; state.eventTotalAll = 0; state.run = null; state.event = null; state.events = []; state.eventOffset = 0; state.eventTotal = 0; state.valueOffset = 0; state.valueTotal = 0; state.eventRequest++; state.valueRequest++;
    $('imported-control-warning').hidden = true; $('imported-comparison').hidden = true; clearDistribution('imported'); $('demo-capture-warning').hidden = true; $('result-panel').hidden = true; $('trace-panel').hidden = true; $('vector-panel').hidden = true; $('event-body').replaceChildren(); $('value-body').replaceChildren(); $('bin-body').replaceChildren(); $('bin-plot').replaceChildren(); $('answer').textContent = ''; $('capture-model').textContent = ''; $('capture-info').textContent = ''; $('event-metadata').textContent = ''; $('run-details').textContent = ''; $('save-status').textContent = 'Optional. Saving includes potentially sensitive prompts, model output, metadata, and full tensor values.'; syncButtons();
  }
  function showRun(run) {
    $('scene-empty').hidden = true;
    state.outputStep = 0; state.tokenOffset = 0;
    state.run = run; $('imported-comparison').hidden = run.run_kind !== 'intervention_comparison'; if (run.run_kind === 'intervention_comparison') { renderDistribution('imported', run.experiment || {}); if(controlWarning('imported-control-warning', run.experiment || {}, run.model))setDock('query',true); } else clearDistribution('imported'); $('result-panel').hidden = false; $('trace-panel').hidden = false; $('answer').textContent = display(run.answer ?? 'No textual output.'); $('task-badge').textContent = run.task || 'Observed run';
    const captureModel = run.model; $('demo-capture-warning').hidden = !isDemo(captureModel); if(isDemo(captureModel))setDock('query',true); $('capture-model').textContent = captureModel ? `Capture model: ${captureModel.name || captureModel.family || captureModel.architecture || 'recorded model'}${captureModel.device ? ' on ' + captureModel.device : ''} — independent of the model currently loaded above.` : 'Model identity was not recorded in this capture.';
    $('capture-info').textContent = display({model: captureModel || null, settings: run.settings || null});
    $('run-metrics').replaceChildren();
    for (const [value, label] of [[run.event_count ?? '—', 'captured observations'], [bytes(run.total_bytes), 'CPU RAM tensor payload'], [Array.isArray(run.steps) ? run.steps.length : '—', 'explicit forward steps']]) { const box = document.createElement('div'), strong = document.createElement('strong'), span = document.createElement('span'); box.className = 'metric'; strong.textContent = String(value); span.textContent = label; box.append(strong, span); $('run-metrics').append(box); }
    $('coverage').textContent = typeof run.coverage === 'string' ? run.coverage : 'Full retained vectors at selected supported observation points. See the coverage record for exact scope.';
    $('run-details').textContent = display({run_kind: run.run_kind, experiment: run.experiment, coverage: run.coverage, steps: run.steps, stop_reason: run.stop_reason, estimates: run.estimates, tokens: run.tokens, token_labels: run.token_labels, token_labels_note: run.token_labels_note, image_metadata: run.image_metadata}); renderForward(); syncButtons();
  }
  const safeId = value => Number.isSafeInteger(value) && value >= 0;
  const shortText = value => typeof value === 'string' ? value.slice(0, 512) : null;
  function renderTokenEvidence(c) {
    const output=$('token-evidence-output'), candidates=$('token-evidence-candidates'), disclosure=$('token-evidence-candidates-options');
    output.replaceChildren();candidates.replaceChildren();disclosure.hidden=true;
    const unavailable=reason=>{$('token-evidence-status').textContent=`Unavailable here: ${reason}`;};
    if(state.run?.run_kind==='intervention_comparison'){unavailable('comparison witnesses are not ordinary forward readouts. See the comparison record.');return;}
    if(!c||c.status!=='known'){unavailable('select a verified source position.');return;}
    if(c.batch_position!==0||!safeId(c.app_forward_index)||!safeId(c.token_position)||!safeId(c.token_id)){unavailable('recorded source coordinates are incomplete.');return;}
    const family=state.run?.model?.family, steps=state.run?.steps;
    const supported=(['gpt2','llama','distilbert'].includes(family)&&c.stack==='input')||(family==='t5'&&c.stack==='decoder');
    if(!supported||c.image_coordinate){unavailable(family==='vit'||c.stack==='image'?'image patches have no token prediction. See the whole-model class output.':family==='t5'&&c.stack==='encoder'?'encoder positions have no next-token readout. Choose a captured decoder position.':'this stream has no position-matched token readout. See the whole-model response.');return;}
    if(!Array.isArray(steps)||steps.length>4096||steps.some(s=>!s||typeof s!=='object'||Array.isArray(s)||!safeId(s.app_forward_index))){unavailable('forward records are missing or malformed.');return;}
    const matches=steps.filter(s=>s.app_forward_index===c.app_forward_index);
    if(matches.length!==1){unavailable('the saved forward is missing or ambiguous.');return;}
    const step=matches[0], metadata=state.event?.metadata;
    if(typeof step.run_id!=='string'||!step.run_id||step.run_id.length>256||metadata?.run_id!==step.run_id||steps.filter(s=>s.run_id===step.run_id).length!==1||(metadata.app_forward_index!==undefined&&metadata.app_forward_index!==c.app_forward_index)){unavailable('capture and readout run identifiers do not match.');return;}
    const ids=family==='t5'?step.decoder_token_ids:step.input_token_ids;
    if(!Array.isArray(ids)||!ids.length||ids.length>65536||ids.some(id=>!safeId(id))||c.token_position>=ids.length||ids[c.token_position]!==c.token_id){unavailable('the source token does not match the saved input.');return;}
    let selected, rows=[];
    const validText=row=>row.text===undefined||typeof row.text==='string';
    if(family==='distilbert'){
      const predictions=step.mask_predictions;
      if(!Array.isArray(predictions)||predictions.length>ids.length||predictions.some(row=>!row||typeof row!=='object'||Array.isArray(row)||!safeId(row.position)||row.position>=ids.length||!safeId(row.token_id)||!validText(row))||new Set(predictions.map(row=>row.position)).size!==predictions.length){unavailable('mask predictions are malformed or ambiguous.');return;}
      selected=predictions.find(row=>row.position===c.token_position);
      if(!selected){unavailable('no mask prediction was recorded here. Choose a captured mask position.');return;}
    }else{
      if(c.token_position!==ids.length-1){unavailable('no prediction was saved here. Choose the last input position if captured.');return;}
      if(['gpt2','llama'].includes(family)){
        const mask=step.input_attention_mask;
        if(!Array.isArray(mask)||mask.length!==ids.length||mask.some(value=>value!==0&&value!==1)||mask[c.token_position]!==1){unavailable('the recorded attention mask is missing, invalid or masked here.');return;}
      }
      if(!safeId(step.selected_token_id)){unavailable('the selected output token ID is missing.');return;}
      if(step.top_next_tokens!==undefined){
        if(!Array.isArray(step.top_next_tokens)||step.top_next_tokens.length>10||step.top_next_tokens.some(row=>!row||typeof row!=='object'||Array.isArray(row)||!safeId(row.token_id)||!Number.isFinite(row.logit)||!validText(row))||new Set(step.top_next_tokens.map(row=>row.token_id)).size!==step.top_next_tokens.length){unavailable('candidate logits are malformed or ambiguous.');return;}
        rows=step.top_next_tokens;
      }
      selected=rows.find(row=>row.token_id===step.selected_token_id)||{token_id:step.selected_token_id};
      if(Number.isFinite(selected.logit)&&rows.some(row=>row.logit>selected.logit)){unavailable('the saved selected token conflicts with candidate logits.');return;}
    }
    const textFor=row=>typeof row.text!=='string'?'Text unavailable':row.text===''?'(empty text)':row.text.trim()===''?JSON.stringify(row.text.slice(0,512)):row.text.slice(0,512);
    const title=document.createElement('strong'), id=document.createElement('span');title.textContent=textFor(selected);id.textContent=`Token ID ${selected.token_id}${typeof selected.text==='string'&&selected.text.length>512?' · label truncated':''}`;output.append(title,id);
    $('token-evidence-status').textContent=`Recorded ${family==='distilbert'?'mask prediction':'next token'} · forward ${c.app_forward_index} · position ${c.token_position}.`;
    for(const row of rows){const item=document.createElement('div');item.className='readout-row';const text=document.createElement('span'), score=document.createElement('span');text.textContent=`ID ${row.token_id} · ${textFor(row)}${typeof row.text==='string'&&row.text.length>512?' · truncated':''}`;score.textContent=`Logit ${number(row.logit)}`;item.append(text,score);candidates.append(item);}
    disclosure.hidden=!rows.length;
  }

  function renderReadableSource(c) {
    renderTokenEvidence(c);
    const box = $('source-readable'); box.replaceChildren();
    const title = document.createElement('strong'), meta = document.createElement('span');
    if (!c || c.status !== 'known') { title.textContent = 'Source not yet known'; meta.textContent = c?.reason || 'Select and read a captured source position. No token meaning is inferred from vector values.'; }
    else if (c.image_coordinate) {
      const image = c.image_coordinate;
      title.textContent = image.token_kind === 'cls' ? 'Image · CLS' : `Image patch · row ${image.patch_row ?? '?'}, column ${image.patch_column ?? '?'}`;
      meta.textContent = `Processed-image position ${c.token_position ?? '?'} · forward ${c.app_forward_index ?? '?'} · not a text token or original-image coordinates`;
    } else {
      const decoded = shortText(c.token_text);
      title.textContent = decoded === null ? 'Decoded token text unavailable' : decoded === '' ? '(empty decoded token)' : decoded.trim() === '' ? JSON.stringify(decoded) + ' (whitespace)' : decoded;
      meta.textContent = `${c.stack || 'unknown stack'} · position ${c.token_position ?? '?'} · token ID ${c.token_id ?? '?'} · forward ${c.app_forward_index ?? '?'}${c.token_label_truncated ? ' · display label truncated' : ''}`;
    }
    box.append(title, meta);
  }
  function renderForward() {
    const generation = ++state.outputGeneration, capture = state.run?.capture_id;
    const steps = Array.isArray(state.run?.steps) ? state.run.steps : [];
    state.outputStep = Math.max(0, Math.min(state.outputStep, Math.max(0, steps.length - 1)));
    $('output-step').value = String(state.outputStep); $('output-step').max = String(Math.max(0, steps.length - 1));
    const step = steps[state.outputStep], family = state.run?.model?.family;
    const record = step && typeof step === 'object' && !Array.isArray(step) ? step : {};
    const forward = record.app_forward_index;
    const knownForward = safeId(forward) && steps.filter(item => item?.app_forward_index === forward).length === 1;
    const out = $('output-readout'); out.replaceChildren();
    $('output-note').textContent = knownForward ? `Forward ${forward} · whole-model logits. A selected output is not an input position until recorded later.` : 'Forward provenance unavailable. Recorded outputs are display-only; no source navigation is inferred.';
    function outputRow(label, text, detail) { const row = document.createElement('div'); row.className = 'readout-row'; const badge = document.createElement('span'); badge.className = 'readout-label'; badge.textContent = label; const word = document.createElement('strong'); word.textContent = text; const score = document.createElement('span'); score.className = 'readout-score'; score.textContent = detail; row.append(badge, word, score); out.append(row); }
    if (family === 'vit') {
      const rows = Array.isArray(record.classes) ? record.classes.slice(0, 10) : [];
      for (const item of rows) if (item && safeId(item.class_id) && typeof item.label === 'string' && Number.isFinite(item.probability) && item.probability >= 0 && item.probability <= 1) outputRow(`Class ${item.class_id}`, shortText(item.label), `${Number((item.probability * 100).toPrecision(5))}% · recorded model probability`);
      $('output-note').textContent = knownForward ? `Forward ${forward} · whole-model class probabilities, not confidence or text tokens.` : 'Image readout provenance unavailable.';
    } else if (family === 'distilbert') {
      const rows = Array.isArray(record.mask_predictions) ? record.mask_predictions.slice(0, 48) : [];
      for (const item of rows) if (item && safeId(item.position) && safeId(item.token_id) && typeof item.text === 'string') outputRow(`mask position ${item.position}`, shortText(item.text), `Predicted token ID ${item.token_id}`);
      if (knownForward) $('output-note').textContent = `Forward ${forward} · simultaneous mask predictions, not replacement captured inputs. No scores recorded.`;
    } else {
      const rows = Array.isArray(record.top_next_tokens) ? record.top_next_tokens.slice(0, 10).filter(item => item && safeId(item.token_id) && typeof item.text === 'string' && Number.isFinite(item.logit)) : [];
      if (safeId(record.selected_token_id)) { const selected = rows.find(item => item.token_id === record.selected_token_id); outputRow('Selected next output', selected ? shortText(selected.text) || '(empty text)' : 'Token text unavailable', `Token ID ${record.selected_token_id}`); }
      for (const item of rows) outputRow(`ID ${item.token_id}`, shortText(item.text) || '(empty text)', `Logit ${number(item.logit)}`);
    }
    if (!out.children.length) outputRow('Recorded readout', 'Output details unavailable', 'See the preserved metadata; no scores are inferred.');
    const stackSelect = $('input-token-stack'), previous = stackSelect.value; stackSelect.replaceChildren();
    const stacks = family === 't5' ? ['encoder', 'decoder'] : ['gpt2','llama','distilbert'].includes(family) ? ['input'] : [];
    for (const stack of stacks) { const option = document.createElement('option'); option.value = stack; option.textContent = stack === 'decoder' ? 'Decoder inputs' : stack === 'encoder' ? 'Encoder inputs' : 'Input tokens'; stackSelect.append(option); }
    stackSelect.value = stacks.includes(previous) ? previous : stacks[0] || '';
    const stack = stackSelect.value, list = stack === 'decoder' ? record.decoder_token_ids : record.input_token_ids;
    const ids = stacks.length && Array.isArray(list) ? list : [];
    state.tokenTotal = ids.length; if (state.tokenOffset >= ids.length) state.tokenOffset = 0;
    const strip = $('input-token-strip'); strip.replaceChildren();
    $('input-token-note').textContent = family === 'vit' ? 'Processed image patches are not words. Select a captured image site in the layer explorer.' : stacks.length ? `${stack} inputs actually recorded for this forward. Click a position to filter existing captures; no automatic token alignment. Individually decoded tokens may not concatenate into the response.` : 'Input token provenance unavailable for this capture. Use exact source coordinates instead.';
    for (let position = state.tokenOffset; position < Math.min(ids.length, state.tokenOffset + 48); position++) {
      const id = ids[position], valid = safeId(id), label = valid ? state.run?.token_labels?.[String(id)] : null;
      const text = shortText(label?.text), button = document.createElement('button'); button.type = 'button'; button.className = 'token-chip';
      const word = document.createElement('strong'); word.textContent = text === null ? 'Text unavailable' : text === '' ? '∅' : text.trim() === '' ? JSON.stringify(text) : text;
      const info = document.createElement('small'); info.textContent = `pos ${position} · ID ${valid ? id : '?'}${label?.truncated ? ' · truncated' : ''}`; button.append(word, info);
      const enabled = !!capture && knownForward && valid; button.disabled = !enabled || state.busy; button.setAttribute('data-unavailable', String(!enabled));
      button.addEventListener('click', () => {
        if (state.busy || generation !== state.outputGeneration || capture !== state.run?.capture_id || !enabled) return;
        $('event-filter').value = ''; $('source-token').value = String(position); $('source-forward').value = String(forward); $('source-stack').value = stack;
        scheduleEventSearch();
      }); strip.append(button);
    }
    $('input-token-page').textContent = ids.length ? `${state.tokenOffset + 1}–${Math.min(state.tokenOffset + 48, ids.length)} of ${ids.length} recorded input positions` : 'No recorded text input positions.';
    syncButtons();
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
  function clearLayerMap() {
    state.mapButtons = []; state.mapAutoCapture = null; state.mapRequest++; state.mapOffset = 0; state.mapTotal = 0; state.mapForward = 0; state.mapCapture = null;
    replaceSceneLayers([]);
    $('layer-map-forward').value = '0'; $('layer-map-body').replaceChildren(); $('layer-map-page').textContent = '';
    $('layer-map-status').textContent = 'Open or refresh to list captured sites. No tensor values are read to build the map.';
  }
  async function loadLayerMap(offset = 0) {
    if (state.busy || !state.run?.capture_id) return;
    const raw = $('layer-map-forward').value.trim();
    if (!/^\d+$/.test(raw) || !Number.isSafeInteger(Number(raw))) { revealOptions('map-tools-options'); $('layer-map-status').textContent = 'Choose a nonnegative safe whole forward number.'; return; }
    const forward = Number(raw), capture = state.run.capture_id, request = ++state.mapRequest;
    state.mapButtons = []; state.mapTotal = 0; state.mapCapture = null; replaceSceneLayers([]); $('layer-map-body').replaceChildren(); $('layer-map-page').textContent = ''; $('layer-map-status').textContent = 'Reading captured-site metadata…'; syncButtons();
    try {
      const result = await api(`layer-map?forward=${forward}&offset=${offset}&limit=12`);
      if (request !== state.mapRequest || capture !== state.run?.capture_id) return;
      if (result.capture_id !== capture || result.selected_forward !== forward) throw new Error('Layer map revision or forward does not match this capture; refresh instead of guessing.');
      state.mapCapture = capture; state.mapForward = forward; state.mapOffset = offset; state.mapTotal = result.total_layers;
      const groups = new Map(), sceneLayers=[];
      for (const layer of result.layers || []) {
        const sceneLayer={key:JSON.stringify([capture,request,forward,layer.stack,layer.layer_index,sceneLayers.length]),label:`${layer.stack} · ${layer.layer_index}`,sites:[]};sceneLayers.push(sceneLayer);
        let group = groups.get(layer.stack);
        if (!group) { group = document.createElement('section'); group.className = 'layer-map-stack'; const heading = document.createElement('h4'); heading.textContent = `${layer.stack} stack`; group.append(heading); groups.set(layer.stack, group); $('layer-map-body').append(group); }
        const row = document.createElement('div'); row.className = 'layer-map-row'; const label = document.createElement('strong'); label.textContent = `Layer ${layer.layer_index}`; row.append(label); const nodes = document.createElement('div'); nodes.className = 'layer-map-nodes'; row.append(nodes); group.append(row);
        for (const node of layer.nodes || []) {
          const button = document.createElement('button'); button.type = 'button'; button.className = 'layer-map-node';
          const kind = document.createElement('span'); kind.className = 'sheet-module-kind'; kind.textContent = node.module_kind; const path = document.createElement('small'); path.className = 'sheet-module-path'; path.textContent = node.module_path; button.append(kind, path); button.setAttribute('title', `${layer.stack} layer ${layer.layer_index} · ${node.module_path} · first captured vector`); button.setAttribute('aria-label', `${layer.stack} layer ${layer.layer_index} · ${node.module_kind} · ${node.module_path} · output ${JSON.stringify(node.output_path)} · first captured vector`); button.setAttribute('aria-pressed', 'false'); const entry = {sceneKey:JSON.stringify([capture,request,forward,layer.stack,layer.layer_index,state.mapButtons.length]),button, row, capture, forward, stack: layer.stack, layer: layer.layer_index, path: node.module_path, kind: node.module_kind, selector: JSON.stringify(node.output_path), eventIndex: node.representative_event.index}; state.mapButtons.push(entry);
          const counts = document.createElement('small'); counts.className = 'sheet-module-counts'; counts.textContent = `${node.event_count} captured events · ${node.vector_count} vectors · ${bytes(node.tensor_bytes)} · output ${JSON.stringify(node.output_path)}`; button.append(counts);
          entry.select = () => {
            if (state.busy || !state.mapButtons.includes(entry) || capture !== state.run?.capture_id || state.mapCapture !== capture || forward !== state.mapForward) return;
            return action('Reading the first captured vector at this module site…', async () => {
              const selectionRequest = state.mapRequest, selected = node.representative_event, vector = selected.matched_vector_index ?? 0;
              setDock('selection',true); await chooseEvent(selected);
              if (selectionRequest !== state.mapRequest || !state.mapButtons.includes(entry) || capture !== state.run?.capture_id || state.event?.index !== selected.index || state.vector !== vector || Number($('vector-index').value) !== vector) return;
              $('vector-heading').focus({preventScroll: true}); $('vector-panel').scrollIntoView({block: 'start'});
            });
          };
          button.addEventListener('click',entry.select);sceneLayer.sites.push({key:entry.sceneKey,label:node.module_kind});
          nodes.append(button);
        }
      }
      replaceSceneLayers(sceneLayers);
      $('layer-map-status').textContent = `Map forward ${forward}: ${result.known_events} mapped + ${result.unmapped_events} unmapped of ${result.selected_forward_events} events. ${result.other_forward_events} events on other forwards; ${result.missing_forward_events} lack a forward; ${result.total_all_events} total. Unmapped observations remain in the exact table. First captured vectors only, not automatic token alignment.`;
      $('layer-map-page').textContent = state.mapTotal ? `${offset + 1}–${Math.min(offset + 12, state.mapTotal)} of ${state.mapTotal} captured layers` : 'No mapped layers for this forward.';
    } catch (error) {
      if (request !== state.mapRequest || capture !== state.run?.capture_id) return;
      revealOptions('map-tools-options'); state.mapCapture = null; state.mapTotal = 0; state.mapButtons=[]; replaceSceneLayers([]); $('layer-map-body').replaceChildren(); $('layer-map-status').textContent = `Layer map unavailable: ${error.message || error}. The exact event table remains available.`;
    } finally { if (request === state.mapRequest) syncButtons(); }
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
    invalidateReferenceView(); const request = ++state.valueRequest; state.coordinate = null; renderReadableSource(null); syncButtons();
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
    $('vector-coordinate').textContent = coordinateLabel(result.coordinate, vector); renderReadableSource(result.coordinate);
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
    invalidateReferenceView(); state.coordinate = null; state.vectorListRequest++; state.vectorListOffset = 0; state.vectorListTotal = 0; state.vectorOptions = [];
    $('vector-picker').replaceChildren(); $('vectors-page').textContent = ''; $('vector-list-status').textContent = '';
    $('vector-coordinate').textContent = 'Coordinate unavailable until a vector is read.'; renderReadableSource(null);
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
      revealOptions('vector-options');
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
  function invalidateBrowse(resetRoot = false) {
    state.browseRequest++; state.browseFresh = false; if (resetRoot) state.browse = null;
    $('folder-list').replaceChildren(); $('browse-current').textContent = ''; $('browse-discovery-details').textContent = '';
    $('browse-status').textContent = 'Folder path changed. Browse or Go to check this location before selecting a path.'; syncButtons();
  }
  function candidateLabel(folder) {
    const c = folder?.candidate;
    if (!c || typeof c !== 'object') return folder?.is_checkpoint ? 'Unverified checkpoint candidate; Inspect checks local assets.' : 'Classification unavailable. Navigate or Inspect explicitly.';
    const status = ['candidate','unverified','unsupported','incomplete','unreadable','none'].includes(c.status) ? c.status : 'unverified';
    return `${status}: ${typeof c.label === 'string' ? c.label : 'Metadata check'}${typeof c.family === 'string' ? ' · ' + c.family : ''}${typeof c.architecture === 'string' ? ' · ' + c.architecture : ''}${typeof c.reason === 'string' ? ' — ' + c.reason : ''} · Metadata only; Inspect verifies local assets, not inference.`;
  }
  async function browse(path, root) {
    revealOptions('model-options'); const request = ++state.browseRequest; state.browseFresh = false;
    $('folder-browser').hidden = false; $('folder-list').replaceChildren(); $('browse-current').textContent = ''; $('browse-discovery-details').textContent = ''; $('browse-status').textContent = 'Reading this location only…'; syncButtons();
    try {
      const result = await api('browse-models', root ? {path, root} : {path});
      if (request !== state.browseRequest) return;
      state.browse = result; state.browseFresh = true; $('browse-path').value = typeof result.path === 'string' ? result.path : '';
      const current = () => !state.busy && state.browseFresh && request === state.browseRequest && state.browse === result;
      if (result.mode === 'locations') {
        $('browse-status').textContent = result.notice || 'Choose a local location. No folders scanned until selected.';
        $('browse-current').textContent = 'Location suggestions only—not discovered models. Choose where to browse.';
        for (const location of (Array.isArray(result.locations) ? result.locations : []).slice(0, 64)) {
          if (typeof location?.path !== 'string' || typeof location?.name !== 'string') continue;
          const button = document.createElement('button'); button.type = 'button'; button.className = 'quiet'; button.textContent = `${location.name} · ${location.path}`;
          button.addEventListener('click', () => { if (current()) return action('Checking selected local folder…', () => browse(location.path)); });
          const row = document.createElement('div'); row.className = 'folder-row'; row.append(button); $('folder-list').append(row);
        }
        if (!$('folder-list').children.length) $('folder-list').textContent = 'No normal locations are available. Enter a local folder path explicitly.';
      } else {
        const discovery = result.discovery && typeof result.discovery === 'object' ? result.discovery : null;
        $('browse-status').textContent = `Root: ${result.root || 'not recorded'}${result.notice ? ' · ' + result.notice : ''}${result.truncated ? ' · Folder list truncated; navigate to a known subfolder explicitly.' : ''}${Number.isSafeInteger(discovery?.skipped_entries) ? ' · ' + discovery.skipped_entries + ' skipped entries' : ''}${discovery?.budget_exhausted ? ' · Metadata budget exhausted; some folders remain unverified.' : ''}`;
        $('browse-discovery-details').textContent = discovery ? display(discovery) : 'No discovery diagnostics supplied by this server.';
        $('browse-current').textContent = `Current folder: ${result.path || 'unavailable'} — ${candidateLabel(result.current)}`;
        for (const folder of (Array.isArray(result.folders) ? result.folders : []).slice(0, 512)) {
          if (typeof folder?.path !== 'string' || typeof folder?.name !== 'string') continue;
          const row = document.createElement('div'), enter = document.createElement('button'), description = document.createElement('small'); row.className = 'folder-row';
          enter.type = 'button'; enter.className = 'quiet'; enter.textContent = folder.name + (folder.is_checkpoint ? ' · checkpoint candidate' : ' /');
          description.textContent = candidateLabel(folder); const main = document.createElement('div'); main.className = 'folder-entry'; main.append(enter, description); row.append(main);
          enter.addEventListener('click', () => { if (current()) return action('Checking selected local folder…', () => browse(folder.path, result.root)); });
          if (folder.is_checkpoint || ['candidate','unverified','incomplete'].includes(folder.candidate?.status)) { const use = document.createElement('button'); use.type = 'button'; use.className = 'secondary'; use.textContent = 'Use path'; use.addEventListener('click', () => { if (current()) { $('model-path').value = folder.path; invalidateInspection(); $('browse-status').textContent = 'Path selected only. Metadata is not a compatibility guarantee. Inspect or Load explicitly when ready.'; } }); row.append(use); }
          $('folder-list').append(row);
        }
        if (!(result.folders || []).length) $('folder-list').textContent = 'No visible immediate subfolders. The current folder can still be selected above.';
      }
    } catch (error) {
      if (request !== state.browseRequest) return;
      $('browse-status').textContent = `Folder discovery unavailable: ${error.message || error}. Model and capture are unchanged by browsing.`;
      throw error;
    } finally { if (request === state.browseRequest) syncButtons(); }
  }
  async function refreshSaved() {
    const result = await api('saved'), selected = $('saved-runs').value; state.saved = result.runs || []; $('saved-runs').replaceChildren();
    const placeholder = document.createElement('option'); placeholder.value = ''; placeholder.textContent = 'Choose a saved capture…'; $('saved-runs').append(placeholder);
    for (const run of state.saved) { const option = document.createElement('option'); option.value = run.path; option.disabled = run.status !== 'complete_unverified'; option.textContent = `${run.name} · ${run.status === 'complete_unverified' ? 'not yet verified' : 'incomplete'} · ${bytes(run.tensor_file_bytes)}`; $('saved-runs').append(option); }
    $('saved-runs').value = state.saved.some((run) => run.path === selected && run.status === 'complete_unverified') ? selected : '';
    $('saved-status').textContent = `${state.saved.length} listed capture folders${result.truncated ? ' · list truncated' : ''}. Listing does not verify tensor integrity; Open validates the selected capture. Incomplete exports cannot be opened from this list.`;
  }
  function referenceSelectionReady() {
    return !!(state.run?.reference_available && state.run.capture_id && state.run.run_kind !== 'intervention_comparison' && ['gpt2','llama'].includes(state.model?.family) && state.model.state !== 'quarantined' && state.event && state.coordinate?.status === 'known' && Number($('vector-index').value) === state.vector);
  }
  function invalidateReferenceView() {
    state.referenceRequest++; state.referenceMode = null; state.referenceOffset = 0; state.referenceTotal = 0;
    $('reference-values-panel').hidden = true; $('reference-value-body').replaceChildren(); $('reference-bit-body').replaceChildren(); $('reference-pair-label').textContent = ''; $('reference-metrics').textContent = ''; $('reference-details').textContent = ''; $('reference-page').textContent = ''; $('reference-read-status').textContent = '';
  }
  function setReference(reference) {
    if ((reference?.reference_id || null) !== (state.reference?.reference_id || null)) invalidateReferenceView();
    state.reference = reference; state.referenceFresh = true;
    $('reference-summary').textContent = reference ? `Reference vector · pinned${reference.coordinate?.token_position != null ? ' at source position ' + reference.coordinate.token_position : ''}` : 'Reference vector · empty';
    $('reference-identity').textContent = reference ? `${reference.model?.name || reference.model?.family || 'Recorded model'} · ${coordinateLabel(reference.coordinate, 'pinned')} · ${bytes(reference.total_bytes)} retained` : 'No reference is pinned. Select a known vector in a live capture, then keep it here.';
    $('reference-status').textContent = reference ? (reference.ready_for_compare ? 'Reference retained. Comparison still requires a compatible selected live vector; the server checks model and module identity.' : reference.reason || 'Reference retained for inspection; current model/capture is not compatible for comparison.') : '';
    $('reference-source-details').textContent = reference ? display(reference) : ''; syncButtons();
  }
  async function refreshReferenceOptional() {
    try { setReference(await api('reference')); }
    catch(error) { state.referenceFresh = false; revealOptions('reference-options'); $('reference-status').textContent = `Reference status unavailable: ${error.message || String(error)}. The last known pin is retained here; refresh before comparing.`; syncButtons(); }
  }
  async function readReference(mode, offset = 0) {
    if (!state.reference) return;
    if (mode === 'compare' && (!referenceSelectionReady() || !state.referenceFresh || !state.reference.ready_for_compare)) { revealOptions('reference-options'); $('reference-status').textContent = state.run?.reference_reason || state.reference.reason || 'Choose a compatible known vector in a current live capture before comparing.'; return; }
    const referenceId = state.reference.reference_id, captureId = state.run?.capture_id, event = state.event?.index, vector = state.vector;
    invalidateReferenceView(); const request = state.referenceRequest; state.referenceMode = mode; state.referenceOffset = offset;
    $('reference-values-panel').hidden = false; $('reference-read-status').textContent = 'Reading retained feature coordinates…';
    try {
      const path = mode === 'compare' ? `compare-vector?reference_id=${encodeURIComponent(referenceId)}&capture_id=${encodeURIComponent(captureId)}&event=${event}&vector=${vector}&offset=${offset}&limit=${valuePageSize}` : `reference-tensor?reference_id=${encodeURIComponent(referenceId)}&offset=${offset}&limit=${valuePageSize}`;
      const result = await api(path);
      if (request !== state.referenceRequest || state.reference?.reference_id !== referenceId || (mode === 'compare' && (state.run?.capture_id !== captureId || state.event?.index !== event || state.vector !== vector))) return;
      if (result.reference_id !== referenceId || (mode === 'compare' && result.capture_id !== captureId)) throw new Error('Reference response revisions do not match the selected reference and capture.');
      const left = mode === 'compare' ? result.reference_values || [] : result.values || [], right = result.current_values || [], leftBits = mode === 'compare' ? result.reference_raw_bits || [] : result.raw_bits || [], rightBits = result.current_raw_bits || [];
      state.referenceTotal = Number(result.total || 0);
      left.forEach((value,i) => { const row=document.createElement('tr'); cell(row,Number(result.offset ?? offset)+i); cell(row,number(value)); cell(row,mode==='compare'?number(right[i]):'—'); cell(row,mode==='compare'?number(result.deltas?.[i]):'—'); $('reference-value-body').append(row); const bits=document.createElement('tr'); cell(bits,Number(result.offset ?? offset)+i); cell(bits,leftBits[i] ?? '—'); cell(bits,mode==='compare'?(rightBits[i] ?? '—'):'—'); $('reference-bit-body').append(bits); });
      $('reference-pair-label').textContent = mode === 'compare' ? `Reference: ${coordinateLabel(result.reference?.coordinate,'pinned')}
Current: ${coordinateLabel(result.current?.coordinate,vector)}
Same feature coordinates—not token alignment. Inputs and source positions may differ.` : coordinateLabel(result.coordinate,'pinned');
      if (mode === 'compare') { const m=result.metrics || {}; $('reference-metrics').textContent=`Full-vector max |Δ| ${number(m.max_abs_delta)} · L1 ${number(m.l1)} · L2 ${number(m.l2)} · cosine ${m.cosine == null ? 'unavailable: ' + (m.cosine_reason || 'zero or undefined norm') : number(m.cosine)}. Raw-bit changes ${number(m.raw_bit_changed_count)}; numeric nonzero deltas ${number(m.numeric_nonzero_count)} (signed zero can differ only in bits). These are numerical diagnostics, not semantic similarity or explanations.`; }
      $('reference-details').textContent=display(result); $('reference-page').textContent=`${offset}–${Math.max(offset,offset+left.length-1)} of ${state.referenceTotal} features (zero-based)`; $('reference-read-status').textContent=mode==='compare'?'Reference/current values retained; deltas computed in float64 by the server.':'Exact reference values; no current capture is required.';
    } catch(error) { if(request===state.referenceRequest) { $('reference-value-body').replaceChildren(); $('reference-bit-body').replaceChildren(); $('reference-pair-label').textContent=''; $('reference-metrics').textContent=''; $('reference-details').textContent=''; state.referenceTotal=0; revealOptions('reference-options'); $('reference-read-status').textContent=`Reference read unavailable: ${error.message || String(error)}. No previous values are shown under this request.`; } }
    syncButtons();
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
  $('browse-parent').addEventListener('click', () => action('Browsing parent folder…', async () => { if (state.browseFresh && state.browse?.parent) await browse(state.browse.parent, state.browse.root); }));
  $('browse-select').addEventListener('click', () => { if (!state.busy && state.browseFresh && typeof state.browse?.path === 'string' && state.browse.path) { $('model-path').value = state.browse.path; invalidateInspection(); $('browse-status').textContent = 'Folder path selected. Inspect or Load explicitly when ready.'; } });
  $('refresh-saved').addEventListener('click', () => action('Listing saved captures…', refreshSaved));
  $('saved-runs').addEventListener('change', () => { const run = state.saved.find((item) => item.path === $('saved-runs').value && item.status === 'complete_unverified'); if (run) $('open-path').value = run.path; });
  $('unload-button').addEventListener('click', () => action('Releasing the model owner; keeping captured vectors…', async () => { await api('unload', {}); const current = await api('status'); setModel(current.model); applyLimits(current.limits); applyDevices(current); }));
  $('execution-device').addEventListener('change', syncButtons);
  $('refresh-devices').addEventListener('click', () => action('Checking actual device availability…', async () => { const current = await api('status'); setModel(current.model); applyLimits(current.limits); applyDevices(current); }));
  $('capture-mode').addEventListener('change', () => { $('custom-capture').hidden = $('capture-mode').value !== 'custom'; });
  $('model-path').addEventListener('input', () => { invalidateInspection(); invalidateBrowse(true); });
  $('browse-path').addEventListener('input', () => invalidateBrowse());
  $('inspect-button').addEventListener('click', () => action('Checking local configuration, weight headers and input assets…', async () => {
    const path = $('model-path').value.trim(); if (!path) throw new Error('Enter a local checkpoint folder.'); invalidateInspection();
    try { showInspection(await api('inspect', {path}), path); }
    catch(error) { revealOptions('model-options'); $('inspection-panel').hidden=false; $('inspection-path').textContent=`Requested folder: ${path}`; $('inspection-status').textContent=`Inspection failed: ${error.message || String(error)}. Check the complete local snapshot: config.json, matching safetensors and matching tokenizer or processor assets. No download was started.`; throw error; }
  }));
  $('demo-button').addEventListener('click', () => action('Building the tiny untrained CPU demo in RAM…', async () => {
    const result = await api('demo', {}); clearView(); setModel(result.model);
    $('capture-mode').value = 'auto'; $('custom-capture').hidden = true; $('layers').value = '';
    for (const [id, maximum] of [['max-output', 8], ['max-input', 120]]) { const current = Number($(id).value); $(id).value = String(Number.isSafeInteger(current) && current >= Number($(id).min) ? Math.min(current, maximum) : maximum); }
    if (!$('prompt').value.trim()) $('prompt').value = 'Hello, tiny model!'; await refreshReferenceOptional();
  }));
  $('load-button').addEventListener('click', () => action('Loading local model on the selected device. This may take a while…', async () => { const path = $('model-path').value.trim(); if (!path) throw new Error('Enter a local checkpoint folder.'); const execution = loadSettings(); const info = await api('load', {path, ...execution}); clearView(); setModel(info); await refreshReferenceOptional(); }));
  $('open-button').addEventListener('click', () => action('Opening saved capture into RAM…', async () => { const path = $('open-path').value.trim(); if (!path) throw new Error('Enter a saved capture folder.'); const opened = await api('open', {path}); clearView(); showRun(opened); await loadEvents(); $('saved-status').textContent = 'Selected capture opened and validated. Other listed exports remain unverified.'; await refreshReferenceOptional(); }));
  $('scene-view').value = 'scene'; $('layer-map-body').setAttribute('data-view', 'scene');
  $('scene-view').addEventListener('change', () => { const view = $('scene-view').value; if (!['scene','list'].includes(view)||sceneFailed) $('scene-view').value = 'list'; $('layer-map-body').setAttribute('data-view',$('scene-view').value);syncSceneVisibility(); });
  $('open-settings').addEventListener('click', () => { state.settingsOpener = 'open-settings'; $('session-options').open = true; $('model-path').focus(); });
  function restoreSettingsFocus() { const opener = state.settingsOpener === 'open-settings' && !$('scene-empty').hidden ? 'open-settings' : 'session-summary'; $(opener).focus(); }
  $('session-summary').addEventListener('click', () => { state.settingsOpener = 'session-summary'; });
  $('session-options').addEventListener('keydown', event => { if (event.key === 'Escape' && $('session-options').open) { event.preventDefault(); $('session-options').open = false; restoreSettingsFocus(); } });
  $('session-options').addEventListener('toggle', () => { if (!$('session-options').open) restoreSettingsFocus(); });
  $('dock-exact').addEventListener('click', () => { if (!state.event || state.busy) return; revealOptions('feature-values-options'); $('feature-values-options').scrollIntoView({block:'nearest'}); });
  $('dock-edit').addEventListener('click', () => { if (!state.event || state.busy) return; revealOptions('intervention-options'); $('intervention-operation').focus(); });
  $('dock-reference').addEventListener('click', () => { if (!state.event || state.busy) return; revealOptions('reference-options'); $('reference-options').scrollIntoView({block:'nearest'}); });
  $('output-step').addEventListener('change', () => { const raw = $('output-step').value.trim(), index = Number(raw); if (!/^\d+$/.test(raw) || !safeId(index) || index >= (state.run?.steps?.length || 0)) { state.outputGeneration++; $('output-readout').replaceChildren(); $('input-token-strip').replaceChildren(); $('input-token-page').textContent = ''; $('output-note').textContent = 'Choose a recorded forward record in range.'; return; } state.outputStep = index; state.tokenOffset = 0; renderForward(); });
  $('output-prev').addEventListener('click', () => { if (state.busy || state.outputStep === 0) return; state.outputStep--; state.tokenOffset = 0; renderForward(); });
  $('output-next').addEventListener('click', () => { if (state.busy || state.outputStep + 1 >= (state.run?.steps?.length || 0)) return; state.outputStep++; state.tokenOffset = 0; renderForward(); });
  $('input-token-stack').addEventListener('change', () => { state.tokenOffset = 0; renderForward(); });
  $('input-tokens-prev').addEventListener('click', () => { if (state.busy) return; state.tokenOffset = Math.max(0, state.tokenOffset - 48); renderForward(); });
  $('input-tokens-next').addEventListener('click', () => { if (state.busy || state.tokenOffset + 48 >= state.tokenTotal) return; state.tokenOffset += 48; renderForward(); });
  $('layer-map-options').addEventListener('toggle', () => { if ($('layer-map-options').open && !state.mapCapture) return loadLayerMap(); });
  $('layer-map-forward').addEventListener('input', () => { state.mapRequest++; state.mapCapture = null; state.mapTotal = 0; state.mapOffset = 0; state.mapButtons=[]; replaceSceneLayers([]); $('layer-map-body').replaceChildren(); $('layer-map-page').textContent = ''; $('layer-map-status').textContent = 'Forward changed. Load / refresh to list its captured sites.'; syncButtons(); });
  $('layer-map-load').addEventListener('click', () => loadLayerMap());
  $('layer-map-prev').addEventListener('click', () => loadLayerMap(Math.max(0, state.mapOffset - 12)));
  $('layer-map-next').addEventListener('click', () => loadLayerMap(state.mapOffset + 12));
  $('run-button').addEventListener('click', () => action('Running model and capturing supported outputs…', async () => { const options = settings(), prompt = $('prompt').value; const vision = String(state.model?.family || '').toLowerCase().includes('vit'); if (vision && !state.image) {setDock('query',true);throw new Error('Choose an image first.');} if (!vision && !prompt.trim()) {setDock('query',true);throw new Error('Enter a prompt first.');} // Cancel old read generations without discarding a capture the server may preserve on preflight rejection.
    clearTimeout(state.searchTimer); state.eventRequest++; state.valueRequest++; state.vectorListRequest++; state.witnessRequest++; state.referenceRequest++; state.mapRequest++;
    const body = {prompt, settings: options}; if (vision) body.image_base64 = state.image;
    let result;
    try { result = await api('run', body); } catch (error) { error.reconcileRun = true; throw error; }
    clearView(); showRun(result); await loadEvents(); await refreshReferenceOptional(); }));
  $('image-file').addEventListener('change', async () => { const request = ++state.imageRequest; state.image = null; const file = $('image-file').files[0]; if (!file) return; if (file.size > 4 * 1048576) { $('image-file').value = ''; notice('Choose an image no larger than 4 MiB.', true); return; } try { const result = await new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.onerror = () => reject(new Error('Could not read this image.')); reader.readAsDataURL(file); }); if (request !== state.imageRequest) return; state.image = String(result).split(',')[1]; $('image-description').textContent = `${file.name} · ${bytes(file.size)} · held in this browser until Clear captured run or page close.`; } catch (error) { notice(error.message, true); } });
  $('save-button').addEventListener('click', () => action('Saving the complete capture…', async () => { const name = $('save-name').value.trim(); if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(name)) throw new Error('Use a save name of 1–64 letters, numbers, underscores, or hyphens, starting with a letter or number.'); $('save-status').textContent = 'No new export confirmed for this save attempt.'; const result = await api('save', {name, capture_id: state.run.capture_id}); $('save-status').textContent = `Saved explicitly: ${result.path || display(result)}. This export contains full capture data; keep it private when inputs are private.`; await refreshSavedOptional(); }));
  $('clear-button').addEventListener('click', () => action('Clearing captured results…', async () => { await api('clear', {}); clearView(); $('prompt').value = ''; state.imageRequest++; state.image = null; $('image-file').value = ''; $('image-description').textContent = 'Select a local PNG, JPEG, or WebP. It is sent only to this local app.'; await refreshReferenceOptional(); }));
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
  $('pin-reference').addEventListener('click',()=>action('Keeping the selected vector as a RAM reference…',async()=>{if(!referenceSelectionReady())throw new Error(state.run?.reference_reason || 'Select a known vector in a supported live capture.');const summary=await api('pin-reference',{capture_id:state.run.capture_id,event:state.event.index,vector:state.vector,expected_reference_id:state.reference?.reference_id ?? null});setReference(summary);revealOptions('reference-options');}));
  $('clear-reference').addEventListener('click',()=>action('Clearing only the pinned reference…',async()=>{if(!state.reference)return;const result=await api('clear-reference',{reference_id:state.reference.reference_id});if(Object.hasOwn(result,'reference'))setReference(result.reference);else await refreshReferenceOptional();}));
  $('refresh-reference').addEventListener('click',()=>action('Refreshing reference status…',refreshReferenceOptional));
  $('inspect-reference').addEventListener('click',()=>{if(state.busy)return;return readReference('inspect');});
  $('compare-reference').addEventListener('click',()=>{if(state.busy)return;return readReference('compare');});
  $('reference-prev').addEventListener('click',()=>{if(state.busy||!state.referenceMode)return;return readReference(state.referenceMode,Math.max(0,state.referenceOffset-valuePageSize));});
  $('reference-next').addEventListener('click',()=>{if(state.busy||!state.referenceMode)return;return readReference(state.referenceMode,state.referenceOffset+valuePageSize);});
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
  // Downloads have independent state: they never replace the live owner or capture.
  const downloads = {initialized:false,catalogRequest:0,catalogFresh:false,cards:[],request:0,timer:null,job:null,root:null,fresh:false,pending:false};
  const downloadActive = () => ['running','cancelling'].includes(downloads.job?.state);
  function syncDownloads() {
    const blocked = state.busy || downloads.pending;
    for (const card of downloads.cards) card.button.disabled = blocked || !downloads.catalogFresh || !downloads.fresh || downloadActive();
    $('cancel-download').disabled = blocked || !downloads.fresh || downloads.job?.state !== 'running';
    $('use-download').disabled = blocked || !downloads.fresh || downloads.job?.state !== 'complete' || typeof downloads.job.path !== 'string' || !downloads.job.path;
    $('refresh-download').disabled = downloads.pending;
    $('browse-downloads').disabled = blocked || !downloads.catalogFresh || !downloads.root;
  }
  function stopDownloadPoll() { if(downloads.timer !== null)clearTimeout(downloads.timer); downloads.timer=null; }
  function showDownloadLocation() {
    const complete=downloads.job?.state==='complete' && typeof downloads.job.path==='string';
    $('download-location').textContent=complete?`Saved folder: ${downloads.job.path}`:downloads.root?`Download folder: ${downloads.root}`:'';
  }
  function acceptDownload(job) {
    if(!job || !['idle','running','cancelling','complete','cancelled','failed'].includes(job.state) || (job.state !== 'idle' && (typeof job.job_id !== 'string' || typeof job.model_id !== 'string')))throw new Error('Invalid download status');
    downloads.job=job;downloads.fresh=true;
    const labels={idle:'No download running.',running:'Downloading',cancelling:'Cancelling',complete:'Complete — use the folder, then Inspect or Load.',cancelled:'Download cancelled.',failed:'Download failed.'};
    $('download-status').textContent=`${labels[job.state]}${job.state!=='idle'?` · ${job.model_id}`:''}${downloadActive()?` · ${bytes(job.received_bytes)} / ${bytes(job.total_bytes)}${job.file?` · ${job.file}`:''}`:''}${job.error?` ${job.error}`:''}`;
    const progress=$('download-progress');progress.hidden=!downloadActive();progress.max=Number.isFinite(job.total_bytes)&&job.total_bytes>0?job.total_bytes:1;progress.value=Number.isFinite(job.received_bytes)?Math.min(progress.max,Math.max(0,job.received_bytes)):0;
    showDownloadLocation();
    if(downloadActive())downloads.timer=setTimeout(()=>refreshDownload(),1200);
  }
  async function downloadRequest(path='download',body) {
    stopDownloadPoll();const request=++downloads.request;downloads.pending=true;syncDownloads();
    try {const job=await api(path,body);if(request!==downloads.request)return;acceptDownload(job);}
    catch(error){if(request!==downloads.request)return;downloads.fresh=false;$('download-progress').hidden=true;revealOptions('download-options');$('download-status').textContent=`Download outcome unavailable: ${error.message||error}. Refresh download status before starting, cancelling or using files.`;}
    finally {if(request===downloads.request){downloads.pending=false;syncDownloads();}}
  }
  function refreshDownload(){return downloadRequest();}
  function populateCatalogDevices(){
    const field=$('catalog-device'), selected=field.value||'auto';field.replaceChildren();
    for(const device of [{id:'auto',name:'Auto'},{id:'cpu',name:'CPU'},...state.devices.filter(d=>/^cuda:\d+$/.test(d.id||'')&&d.available!==false)]){const option=document.createElement('option');option.value=device.id;option.textContent=device.name||device.id;field.append(option);}
    if(!['auto','cpu',...state.devices.filter(d=>d.available!==false).map(d=>d.id)].includes(selected)){const option=document.createElement('option');option.value=selected;option.textContent=`${selected} · unavailable`;field.append(option);}
    field.value=selected;
  }
  async function refreshCatalog(){
    const request=++downloads.catalogRequest;downloads.catalogFresh=false;downloads.cards=[];$('catalog-models').replaceChildren();$('catalog-details').textContent='';syncDownloads();
    const raw=$('catalog-context').value.trim(),context=Number(raw),device=$('catalog-device').value||'auto';
    if(!/^\d+$/.test(raw)||!Number.isSafeInteger(context)||context<1||context>4096){revealOptions('download-options');$('catalog-status').textContent='Choose 1–4096 context tokens for the estimate.';return;}
    $('catalog-status').textContent='Reading the curated model list…';
    try {
      const result=await api(`model-catalog?device=${encodeURIComponent(device)}&context_tokens=${context}&show_all=${$('catalog-all').checked?'1':'0'}`);if(request!==downloads.catalogRequest)return;
      if(!Array.isArray(result.models))throw new Error('Invalid model catalog');
      downloads.catalogFresh=true;downloads.root=typeof result.download_root==='string'?result.download_root:null;showDownloadLocation();$('catalog-details').textContent=display(result);
      $('catalog-status').textContent=`${result.models.length} model${result.models.length===1?'':'s'} · ${result.resolved_device||device}. ${Number.isSafeInteger(result.hidden_too_large)&&result.hidden_too_large>0?`${result.hidden_too_large} over-limit models hidden. `:''}Estimates only, not a fit guarantee.`;
      for(const model of result.models.slice(0,64)){
        if(!model||typeof model.id!=='string'||typeof model.revision!=='string')continue;
        const card=document.createElement('div');card.className='catalog-card';const name=document.createElement('strong');name.textContent=typeof model.title==='string'?model.title:model.id;
        const summary=document.createElement('p');summary.className='hint';const fit=model.fit||{},fitLabel={cpu:'CPU RAM not measured',likely_fits:'May fit GPU memory',too_large:'Over server/device or context limit',unknown:'Memory fit unknown'}[fit.status]||'Memory fit unknown';
        summary.textContent=`${bytes(model.download_bytes)} download · ${fitLabel}. ${model.validation==='prior_cpu_verified'?'CPU checked with this app; GPU unverified.':'Metadata only; execution unverified.'}`;
        const detail=document.createElement('details'),heading=document.createElement('summary');heading.textContent='Size, license & assumptions';const record=document.createElement('pre');record.className='detail-json';record.textContent=display(model);detail.append(heading);
        if(typeof model.license_url==='string'&&/^https:\/\/huggingface\.co\/[A-Za-z0-9._/-]+$/.test(model.license_url)){const link=document.createElement('a');link.textContent=`Model card / license: ${model.license||'see card'}`;link.setAttribute('href',model.license_url);link.setAttribute('target','_blank');link.setAttribute('rel','noopener noreferrer');detail.append(link);}detail.append(record);
        const button=document.createElement('button');button.type='button';button.textContent='Download';button.addEventListener('click',()=>{if(state.busy||downloads.pending||!downloads.fresh||downloadActive()||!downloads.catalogFresh||request!==downloads.catalogRequest)return;return downloadRequest('download',{model_id:model.id,revision:model.revision});});
        downloads.cards.push({button});card.append(name,summary,detail,button);$('catalog-models').append(card);
      }
    }catch(error){if(request!==downloads.catalogRequest)return;downloads.catalogFresh=false;$('catalog-models').replaceChildren();revealOptions('download-options');$('catalog-status').textContent=`Model list unavailable: ${error.message||error}. Your model and capture are unchanged.`;}
    syncDownloads();
  }
  $('download-options').addEventListener('toggle',async()=>{if(!$('download-options').open)return;populateCatalogDevices();if(downloads.initialized)return;downloads.initialized=true;await refreshCatalog();if(!downloads.pending)await refreshDownload();});
  $('refresh-catalog').addEventListener('click',refreshCatalog);
  $('catalog-context').addEventListener('input',()=>{downloads.catalogRequest++;downloads.catalogFresh=false;downloads.cards=[];$('catalog-models').replaceChildren();$('catalog-details').textContent='';$('catalog-status').textContent='Estimate changed. Refresh models to use it.';syncDownloads();});
  for(const id of ['catalog-device','catalog-context','catalog-all'])$(id).addEventListener('change',refreshCatalog);
  $('browse-downloads').addEventListener('click',()=>{
    if(state.busy||downloads.pending||!downloads.catalogFresh||!downloads.root)return;
    const root=downloads.root;
    return action('Browsing downloaded models…',async()=>{
      await browse(root,root);
      if(state.browseFresh && state.browse?.path===root){
        $('browse-status').textContent+=' Open a download folder, then its model subfolder. Nothing is loaded automatically.';
        $('browse-path').focus({preventScroll:true});$('folder-browser').scrollIntoView({block:'nearest',behavior:'instant'});
      }
    });
  });
  $('refresh-download').addEventListener('click',()=>{if(!downloads.pending)return refreshDownload();});
  $('cancel-download').addEventListener('click',()=>{if(!state.busy&&!downloads.pending&&downloads.fresh&&downloads.job?.state==='running')return downloadRequest('cancel-download',{job_id:downloads.job.job_id});});
  $('use-download').addEventListener('click',()=>{if(state.busy||downloads.pending||!downloads.fresh||downloads.job?.state!=='complete'||typeof downloads.job.path!=='string'||!downloads.job.path)return;$('model-path').value=downloads.job.path;invalidateInspection();invalidateBrowse(true);$('download-status').textContent='Folder selected. Inspect or Load when ready; your current model and capture are unchanged.';});

  for(const side of ['selection','query']){
    $(`${side}-dock-toggle`).addEventListener('click',()=>setDock(side,$(`${side}-dock-body`).hidden));
    $(`${side}-dock`).addEventListener('keydown',event=>{if(event.key==='Escape'){event.preventDefault();setDock(side,false);$(`${side}-dock-toggle`).focus();}});
  }
  document.addEventListener('keydown',event=>{const tag=(event.target?.tagName||event.target?.tag||'').toLowerCase();if(event.ctrlKey||event.metaKey||event.altKey||event.target?.isContentEditable||['input','textarea','select'].includes(tag))return;const side=event.key==='['?'selection':event.key===']'?'query':null;if(side){event.preventDefault();setDock(side,$(`${side}-dock-body`).hidden);}});
  if(typeof window!=='undefined'&&window.addEventListener){window.addEventListener('resize',()=>{if(sceneApi)sceneApi.resize();});window.addEventListener('pagehide',()=>{if(sceneApi)sceneApi.dispose();});}
  initializeScene();
  action('Connecting to your local session…', async () => { const result = await api('status'); setModel(result.model); applyLimits(result.limits); applyDevices(result); for (const item of result.models || []) { const option = document.createElement('option'); option.value = typeof item === 'string' ? item : item.path; option.label = typeof item === 'string' ? item : item.name || item.path; $('model-choices').append(option); } if (result.run) { showRun(result.run); await loadEvents(); } await refreshSavedOptional(); await restoreExperimentOptional(); await refreshReferenceOptional(); });
})();
