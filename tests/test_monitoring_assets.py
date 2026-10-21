import ast
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ALERTS = ROOT / "docs" / "prometheus-alerts.yml"
DASHBOARD = ROOT / "docs" / "grafana-dashboard.json"
MONITORING = ROOT / "docs" / "monitoring.md"


def defined_metrics() -> set[str]:
    tree = ast.parse((ROOT / "modelship" / "metrics.py").read_text())
    return {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"Counter", "Gauge", "Histogram"}
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }


def alert_rules() -> list[dict]:
    groups = yaml.safe_load(ALERTS.read_text())["groups"]
    return [rule for group in groups for rule in group["rules"]]


def test_alerts_and_dashboard_reference_only_defined_metrics():
    text = ALERTS.read_text() + DASHBOARD.read_text()
    names = re.findall(r"ray_(modelship_[a-z0-9_]+)", text)
    referenced = {re.sub(r"_(bucket|sum|count)$", "", name) for name in names}

    assert referenced - defined_metrics() == set()


def test_every_defined_metric_is_documented():
    docs = MONITORING.read_text()

    assert [name for name in sorted(defined_metrics()) if f"ray_{name}" not in docs] == []


def test_alert_rules_have_the_prometheus_shape():
    rules = alert_rules()

    assert rules
    for rule in rules:
        assert rule["alert"]
        assert rule["expr"]
        assert rule["labels"]["severity"] in {"critical", "warning"}
        assert rule["annotations"]["summary"]
    names = [rule["alert"] for rule in rules]
    assert len(names) == len(set(names))


def test_every_alert_is_documented():
    docs = MONITORING.read_text()

    assert [rule["alert"] for rule in alert_rules() if f"`{rule['alert']}`" not in docs] == []


def test_dashboard_panel_ids_are_unique():
    ids = [panel["id"] for panel in json.loads(DASHBOARD.read_text())["panels"]]

    assert len(ids) == len(set(ids))
