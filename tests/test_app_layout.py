"""Static HTML contracts, not browser rendering or assistive-technology certification."""
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}


class Element:
    def __init__(self, tag, attrs=(), parent=None):
        self.tag, self.attrs, self.parent = tag, dict(attrs), parent
        self.children = []
        self.text = ""

    def ancestors(self):
        node = self.parent
        while node is not None:
            yield node
            node = node.parent


class Page(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.root = Element("root")
        self.stack = [self.root]
        self.elements = []
        self.feed(source)
        self.close()
        assert self.stack == [self.root], "unclosed HTML elements"

    def handle_starttag(self, tag, attrs):
        element = Element(tag, attrs, self.stack[-1])
        element.parent.children.append(element)
        self.elements.append(element)
        if tag not in VOID:
            self.stack.append(element)

    def handle_endtag(self, tag):
        assert self.stack[-1].tag == tag, f"mismatched closing {tag} after {self.stack[-1].tag}"
        self.stack.pop()

    def handle_data(self, data):
        for element in self.stack:
            element.text += data

    @property
    def ids(self):
        return {element.attrs["id"]: element for element in self.elements if "id" in element.attrs}


def page():
    return Page((ROOT / "looking_glass/app_static/index.html").read_text())


def test_unique_ids_and_resolvable_accessible_labels():
    doc = page()
    ids = doc.ids
    assert len(ids) == sum("id" in node.attrs for node in doc.elements)
    labelled = {node.attrs["for"] for node in doc.elements if node.tag == "label" and "for" in node.attrs}
    for target in labelled:
        assert target in ids, target
        assert ids[target].tag in {"input", "select", "textarea"}, target
    for node in doc.elements:
        for attribute in ("aria-labelledby", "aria-describedby"):
            for target in node.attrs.get(attribute, "").split():
                assert target in ids, (node.tag, attribute, target)
        if node.tag in {"input", "select", "textarea"}:
            assert node.attrs.get("id") in labelled or node.attrs.get("aria-label"), node.attrs
        if node.tag == "button":
            assert node.attrs.get("type") == "button", node.attrs
            assert node.text.strip() or node.attrs.get("aria-label"), node.attrs


def test_disclosures_have_native_named_noninteractive_summaries():
    doc = page()
    details = [node for node in doc.elements if node.tag == "details"]
    assert details
    for node in details:
        assert "open" not in node.attrs or node.attrs.get("id") == "layer-map-options", "only the main captured-layer explorer starts open"
        summaries = [child for child in node.children if child.tag == "summary"]
        assert len(summaries) == 1 and node.children[0] is summaries[0], node.attrs
        summary = summaries[0]
        assert summary.text.strip(), node.attrs
        assert "tabindex" not in summary.attrs and "role" not in summary.attrs
        for child in doc.elements:
            if summary in child.ancestors():
                assert child.tag not in {"a", "button", "input", "select", "textarea", "details"}, node.attrs


def test_core_actions_and_critical_notices_stay_outside_disclosures():
    ids = page().ids
    core = ("model-badge",
            "prompt", "run-button", "vector-picker",
            "source-readable", "notice", "run-status", "replace-warning",
            "demo-model-warning", "demo-capture-warning", "demo-experiment-warning",
            "experiment-outcome")
    for identifier in core:
        assert not any(node.tag == "details" for node in ids[identifier].ancestors()), identifier


def test_advanced_workflows_are_named_closed_disclosures_not_removed():
    ids = page().ids
    groups = {
        "model-options": ("execution-device", "vram-mib", "browse-button", "inspect-button", "model-info"),
        "saved-options": ("saved-runs", "open-path", "open-button"),
        "capture-options": ("capture-mode", "layers", "trace-mib", "max-events"),
        "source-options": ("source-token", "source-forward", "source-stack", "clear-event-filters"),
        "vector-options": ("vector-index", "vector-view", "view-vector"),
        "intervention-options": ("intervention-operation", "intervention-feature", "intervene-button"),
        "comparison-checks": ("experiment-controls", "experiment-metrics"),
        "witness-options": ("witness-phase", "witness-body", "witness-prev", "witness-next"),
        "reference-options": ("pin-reference", "inspect-reference", "compare-reference", "clear-reference"),
    }
    for identifier, controls in groups.items():
        assert identifier in ids, f"missing progressive disclosure: {identifier}"
        group = ids[identifier]
        assert group.tag == "details" and "open" not in group.attrs, identifier
        assert "hidden" not in group.attrs and "aria-hidden" not in group.attrs, identifier
        for control in controls:
            assert group in ids[control].ancestors(), (identifier, control)


def test_failed_control_notice_is_not_hidden_inside_details():
    ids = page().ids
    for identifier in ("experiment-control-warning", "imported-control-warning"):
        assert identifier in ids
        warning = ids[identifier]
        assert warning.attrs.get("role") in {"alert", "status"}
        assert not any(node.tag == "details" for node in warning.ancestors())


def test_reference_disclosure_remains_reachable_without_a_capture():
    group = page().ids["reference-options"]
    assert not any("hidden" in node.attrs or node.attrs.get("id") in
                   {"result-panel", "trace-panel", "vector-panel", "experiment-panel"}
                   for node in group.ancestors())



def test_layer_map_is_optional_grounded_navigation():
    ids = page().ids
    group = ids["layer-map-options"]
    assert group.tag == "details" and "open" in group.attrs
    assert ids["trace-panel"] in list(group.ancestors())
    for name in ("layer-map-forward", "layer-map-load", "layer-map-prev", "layer-map-next", "layer-map-body", "layer-map-status"):
        assert group in list(ids[name].ancestors())
    assert "not a neuron" in group.text
    assert "first captured vector" in group.text
    assert ids["layer-map-status"].attrs.get("role") == "status"


def test_layer_map_cards_wrap_long_paths_without_primary_button_fill():
    import re
    css = (ROOT / "looking_glass/app_static/app.css").read_text()
    rule = re.search(r"\.layer-map-node\s*\{([^}]+)\}", css).group(1)
    for declaration in ("white-space: normal", "min-width: 0", "overflow-wrap: anywhere", "flex: 1 1 12rem", "max-width: 100%", "background: var(--accent-soft)", "color: var(--text)", "border-color: var(--line)"):
        assert declaration.replace(" ", "") in rule.replace(" ", ""), declaration


def test_map_focus_target_is_programmatic_not_an_extra_tab_stop():
    assert page().ids["vector-heading"].attrs.get("tabindex") == "-1"


def test_readable_explorer_and_outputs_are_primary_not_raw_numbers():
    ids = page().ids
    for identifier in ("output-readout", "input-token-strip", "source-readable", "explorer-grid"):
        assert identifier in ids
    assert ids["scene-workspace"] in list(ids["vector-panel"].ancestors())
    assert ids["explorer-grid"] in list(ids["layer-map-options"].ancestors())
    assert ids["feature-values-options"] in list(ids["raw-values"].ancestors())
    assert ids["event-catalog-options"] in list(ids["event-body"].ancestors())
    for name in ("source-readable",):
        assert not any(n.tag == "details" for n in ids[name].ancestors())


def test_workspace_has_local_dark_presentation_and_truthful_readout():
    doc = page()
    css = (ROOT / "looking_glass/app_static/app.css").read_text()
    assert "color-scheme: dark" in css
    assert "complete model output—not from decoding the selected intermediate activation" in doc.root.text
    assert "Hidden feature numbers are not token IDs" in doc.root.text
    assert "/app_scene.js" in (ROOT / "looking_glass/app_static/index.html").read_text()
    assert "prefers-reduced-motion: reduce" in css and "perspective:" not in css
    for node in doc.elements:
        if node.tag in {"script", "link"}:
            target = node.attrs.get("src", node.attrs.get("href", ""))
            assert target.startswith("/") and not target.startswith("//"), target


def test_scene_composition_and_accessible_controls_preserve_truth():
    ids = page().ids
    assert ids["header-controls"] in list(ids["run-button"].ancestors())
    assert ids["header-controls"] in list(ids["model-badge"].ancestors())
    assert ids["input-card"] in list(ids["input-token-strip"].ancestors())
    assert ids["result-panel"] in list(ids["output-readout"].ancestors())
    assert ids["scene-workspace"] in list(ids["vector-panel"].ancestors())
    assert not any(n.tag == "details" for n in ids["demo-model-warning"].ancestors())
    assert ids["scene-view"].tag == "select"
    assert "not neurons" in ids["layer-map-options"].text
    assert "Choose a model" in ids["scene-empty"].text
    for name in ("dock-exact", "dock-edit", "dock-reference"):
        assert ids[name].tag == "button"
        assert ids["vector-panel"] in list(ids[name].ancestors())


def test_output_card_keeps_predictions_primary_and_details_reachable():
    ids = page().ids
    detail = ids["capture-details-options"]
    assert detail.tag == "details" and "open" not in detail.attrs
    for name in ("capture-model", "capture-info", "run-metrics", "coverage", "save-button", "save-status"):
        assert detail in ids[name].ancestors()
    for name in ("answer", "demo-capture-warning", "imported-control-warning"):
        assert not any(n.tag == "details" for n in ids[name].ancestors())


def test_scene_has_flat_reduced_motion_and_keyboard_front_plane():
    css = (ROOT / "looking_glass/app_static/app.css").read_text()
    assert "prefers-reduced-motion: reduce" in css
    assert ".layer-map-node[aria-pressed=true]" in css
    assert "focus-visible" in css
    assert "rotateY(" not in css and "perspective:" not in css
    assert "transition:all" not in css and "transition: all" not in css
    ids = page().ids
    assert ids["scene-view"].tag == "select"
    assert ids["session-summary"].tag == "summary"
    assert "role" not in ids["session-options"].attrs  # Non-modal; no synthetic focus trap.


def test_discovery_is_explicit_and_wireframe_is_not_fake_model_data():
    ids = page().ids
    assert ids["browse-current"].attrs.get("role") == "status"
    assert ids["browse-discovery-details"].tag == "pre"
    assert "immediate subfolders" in ids["folder-browser"].text
    assert "Inspect" in ids["folder-browser"].text
    css = (ROOT / "looking_glass/app_static/app.css").read_text()
    assert "#04050a" in css and "rgba(18,20,28,.78)" in css
    assert "radial-gradient" not in css
    assert "rotateY(" not in css  # Real scene geometry lives in app_scene.js, not CSS illusions.


def test_minimal_default_keeps_explanations_and_scores_collapsed():
    ids = page().ids
    groups = {
        'prompt-guide': ['prompt-help'],
        'input-tokens-options': ['input-token-strip', 'input-token-note'],
        'readout-options': ['output-readout', 'output-note'],
        'map-tools-options': ['layer-map-forward', 'layer-map-status'],
        'source-position-options': ['vector-coordinate'],
    }
    for group, contents in groups.items():
        assert ids[group].tag == 'details' and 'open' not in ids[group].attrs
        for item in contents:
            assert ids[group] in ids[item].ancestors(), (group, item)
    for item in ['prompt', 'answer', 'layer-map-body', 'vector-picker', 'source-readable']:
        assert not any(n.tag == 'details' and 'open' not in n.attrs for n in ids[item].ancestors()), item


def test_default_paragraphs_are_short_and_not_a_wall_of_instructions():
    doc = page()
    for node in doc.elements:
        if node.tag != 'p':
            continue
        if any('hidden' in n.attrs or (n.tag == 'details' and 'open' not in n.attrs) for n in [node, *node.ancestors()]):
            continue
        # Optional comparison panels start hidden. Default guidance stays one short sentence.
        assert len(node.text.split()) <= 35, node.text


def test_model_downloads_remain_an_explicit_closed_settings_workflow():
    ids = page().ids
    group = ids['download-options']
    assert group.tag == 'details' and 'open' not in group.attrs
    assert ids['session-options'] in group.ancestors()
    for identifier in ('catalog-device', 'catalog-context', 'catalog-all', 'catalog-models', 'download-status', 'cancel-download', 'use-download'):
        assert group in ids[identifier].ancestors()
    assert 'does not load' in group.text
    assert 'server limits, not your Load GPU allowance' in group.text
    assert 'a lower Load allowance may still reject it' in group.text
    assert group in ids['download-location'].ancestors()


def test_download_reuse_is_a_native_explicit_settings_action():
    ids=page().ids
    assert 'browse-downloads' in ids
    button=ids['browse-downloads']
    assert button.tag == 'button' and button.attrs.get('type') == 'button'
    assert 'disabled' in button.attrs
    assert ids['download-options'] in button.ancestors()
    assert 'Browse downloads' in button.text


def test_selected_token_evidence_is_qualified_and_candidates_are_collapsed():
    ids = page().ids
    evidence = ids['token-evidence']
    assert ids['vector-panel'] in evidence.ancestors()
    assert 'Whole-model output, not decoding this layer' in evidence.text
    assert ids['token-evidence-candidates-options'].tag == 'details'
    assert 'open' not in ids['token-evidence-candidates-options'].attrs
    assert ids['token-evidence-candidates-options'] in ids['token-evidence-candidates'].ancestors()
    assert ids['token-evidence-status'].attrs.get('role') == 'status'


def test_real_scene_and_two_docks_replace_perspective_dashboard():
    doc=page(); ids=doc.ids
    assert ids['selection-dock'] in ids['vector-panel'].ancestors()
    assert ids['query-dock'] in ids['prompt'].ancestors()
    assert ids['query-dock'] in ids['answer'].ancestors()
    assert ids['scene-canvas'].attrs.get('aria-hidden') == 'true'
    assert ids['scene-render-status'].attrs.get('role') == 'status'
    for side in ('selection','query'):
        assert ids[f'{side}-dock-toggle'].attrs['aria-controls'] == f'{side}-dock-body'
        assert ids[f'{side}-dock-toggle'].attrs['aria-expanded'] == 'true'
    sources=[e.attrs['src'] for e in doc.elements if e.tag=='script']
    assert sources[-4:] == ['/vendor/three-0.128.0.min.js','/vendor/OrbitControls-0.128.0.js','/app_scene.js','/app.js']
    css=(ROOT/'looking_glass/app_static/app.css').read_text()
    assert '#04050a' in css
    assert 'rotateY(' not in css and 'perspective:' not in css
    assert 'text-transform:uppercase' not in css and 'text-transform: uppercase' not in css
    assert not any(e.attrs.get('class') in ('brand','brand-mark') for e in doc.elements)


def test_alerts_and_scene_docks_keep_accessible_neutral_chrome():
    ids=page().ids
    for name in ('experiment-panel','experiment-control-warning'):
        assert ids['selection-dock-body'] not in ids[name].ancestors()
        assert ids['query-dock-body'] not in ids[name].ancestors()
    css=(ROOT/'looking_glass/app_static/app.css').read_text()
    assert 'height:calc(94.7dvh - 8px)' in css
    assert 'text-align:center' not in css and 'text-align:right' not in css
    assert '--error-text:#e8eaf0' in css and '--warning-text:#e8eaf0' in css
    assert '::-webkit-scrollbar{width:4px;height:4px}' in css
