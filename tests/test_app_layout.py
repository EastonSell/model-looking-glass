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
        assert "open" not in node.attrs, "advanced details start collapsed"
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
    core = ("model-path", "load-button", "demo-button", "model-badge",
            "prompt", "run-button", "event-filter", "event-body", "vector-picker",
            "vector-coordinate", "raw-values", "notice", "run-status", "replace-warning",
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

