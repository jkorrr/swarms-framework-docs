"""Real Swarms generation plus message-contract and application rendering tests."""

from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading

import pytest

import catalogs as app

HERE = Path(__file__).parent


@pytest.fixture
def source():
    return app.parse_json((HERE / "source.json").read_text(encoding="utf-8"))


@pytest.fixture
def fixtures():
    return app.parse_json((HERE / "fixtures.json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def runtime(tmp_path, monkeypatch):
    app.prepare_runtime(tmp_path / "runtime")
    attempts = []

    def forbidden(*args, **kwargs):
        attempts.append(True)
        raise AssertionError("Unexpected socket connection")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    yield
    assert not attempts


def generate(source, fixtures):
    return app.generate_catalogs(source, list(fixtures), mode="scripted",
                                 agent_factory=lambda locale: app.make_agent(locale,
                                     app.ScriptedModel(json.dumps(fixtures[locale], ensure_ascii=False))))


def test_real_agents_and_workflow_keep_locale_mapping_in_reverse_completion(source, fixtures, monkeypatch):
    from swarms.structs.conversation import Conversation
    from swarms.telemetry.otel import swarm_telemetry

    assert not swarm_telemetry().ready
    both_running = threading.Barrier(2)
    french_recorded = threading.Event()
    completion_roles = []
    original_add = Conversation.add

    def observe_add(self, role, content, *args, **kwargs):
        result = original_add(self, role, content, *args, **kwargs)
        if (self.name.startswith("concurrent_workflow_name_message-localization_")
                and role.startswith("catalog-")):
            completion_roles.append(role)
            if role == "catalog-fr":
                french_recorded.set()
        return result

    # Observe real conversation writes; do not replace the workflow or Agent.
    monkeypatch.setattr(Conversation, "add", observe_add)
    backends = {}

    class OrderedModel(app.ScriptedModel):
        def __init__(self, locale):
            super().__init__(json.dumps(fixtures[locale], ensure_ascii=False))
            self.locale = locale

        def run(self, **kwargs):
            both_running.wait(timeout=5)
            if self.locale == "es":
                assert french_recorded.wait(timeout=5)
            return super().run(**kwargs)

    def agent_factory(locale):
        backends[locale] = OrderedModel(locale)
        return app.make_agent(locale, backends[locale])

    bundle = app.generate_catalogs(source, ["es", "fr"], agent_factory=agent_factory, mode="scripted")
    assert completion_roles == ["catalog-fr", "catalog-es"]
    assert [row["locale"] for row in bundle["results"]] == ["es", "fr"]
    assert all(row["status"] == "accepted" for row in bundle["results"])
    assert bundle["catalogs"]["es"]["welcome"] == "¡Hola, {user}!"
    assert bundle["catalogs"]["fr"]["welcome"] == "Bonjour, {user} !"
    assert bundle["framework_version"] == "15.0.3"
    for backend in backends.values():
        assert len(backend.calls) == 1
        assert "order_ready" in str(backend.calls[0])
        assert "{order_id}" in str(backend.calls[0])


def test_render_unicode_reordered_fields_braces_and_source_fallback(source, fixtures):
    store = app.CatalogStore(generate(source, fixtures))
    assert store.render("es", "order_ready", {"user": "Zoë", "order_id": "A-7"}) == {
        "text": "Para Zoë: el pedido A-7 está listo.", "requested_locale": "es",
        "used_locale": "es", "fallback": False}
    assert store.render("fr", "literal_braces", {})["text"] == "Utilisez { et } pour les accolades littérales."
    fallback = store.render("fr-CA", "welcome", {"user": "Zoë"})
    assert fallback == {"text": "Welcome, Zoë!", "requested_locale": "fr-CA",
                        "used_locale": "en", "fallback": True}
    assert not store.render("en", "welcome", {"user": "A"})["fallback"]
    assert store.render("es", "welcome", {"user": "{still a value}"})["text"] == "¡Hola, {still a value}!"


@pytest.mark.parametrize("fault", ["missing_key", "extra_key", "wrong_locale", "missing_placeholder", "malformed_brace"])
def test_invalid_locale_is_excluded_without_losing_other_locale(source, fixtures, fault):
    if fault == "missing_key":
        del fixtures["fr"]["messages"]["cart_count"]
    elif fault == "extra_key":
        fixtures["fr"]["messages"]["extra"] = "Unexpected"
    elif fault == "wrong_locale":
        fixtures["fr"]["locale"] = "es"
    elif fault == "missing_placeholder":
        fixtures["fr"]["messages"]["welcome"] = "Bonjour !"
    else:
        fixtures["fr"]["messages"]["welcome"] = "Bonjour {user"
    bundle = generate(source, fixtures)
    assert [r["status"] for r in bundle["results"]] == ["accepted", "invalid_catalog"]
    assert set(bundle["catalogs"]) == {"es"}
    result = app.CatalogStore(bundle).render("fr", "welcome", {"user": "Zoë"})
    assert result["fallback"] and result["text"] == "Welcome, Zoë!"


def test_swallowed_backend_failure_is_not_treated_as_translation(source, fixtures):
    bundle = app.generate_catalogs(source, ["es", "fr"], agent_factory=lambda locale:
        app.make_agent(locale, app.ScriptedModel(json.dumps(fixtures[locale]), fail=locale == "fr")))
    assert [row["status"] for row in bundle["results"]] == ["accepted", "execution_error"]
    assert "fr" not in bundle["catalogs"]


@pytest.mark.parametrize("output", ["[]", "not JSON", '{"locale":"fr","locale":"fr","messages":{}}'])
def test_malformed_agent_json_is_not_accepted(source, fixtures, output):
    bundle = app.generate_catalogs(source, ["fr"], agent_factory=lambda locale:
        app.make_agent(locale, app.ScriptedModel(output)))
    assert bundle["results"][0]["status"] == "invalid_catalog"
    assert bundle["catalogs"] == {}


@pytest.mark.parametrize("template", ["{", "}", "{}", "{0}", "{user.name}", "{user[0]}",
                                      "{user!r}", "{count:03}", "{count:{width}}", "  ", 42])
def test_unsupported_template_syntax(template):
    with pytest.raises(ValueError):
        app.placeholders(template)


def test_repeated_placeholder_counts_are_preserved():
    source = {"schema_version": 1, "source_locale": "en", "messages": {"repeat": "{name}, hello {name}"}}
    app.validate_source(source)
    candidate = {"locale": "fr", "messages": {"repeat": "Bonjour {name}"}}
    with pytest.raises(ValueError, match="counts differ"):
        app.validate_translation(source, "fr", candidate)
    candidate["messages"]["repeat"] = "{name}, bonjour {name}"
    assert app.validate_translation(source, "fr", candidate) == candidate["messages"]


@pytest.mark.parametrize("key,values", [("unknown", {}), ("welcome", {}),
                                       ("welcome", {"user": 42}),
                                       ("welcome", {"user": "A", "unused": "B"})])
def test_render_rejects_missing_extra_nonstrings_and_unknown_keys(source, key, values):
    store = app.CatalogStore({"schema_version": 1, "source": source, "catalogs": {}})
    with pytest.raises(ValueError):
        store.render("en", key, values)


def test_store_revalidates_loaded_catalogs_and_copies_them(source, fixtures):
    bundle = {"schema_version": 1, "source": source, "catalogs": {"es": fixtures["es"]["messages"]}}
    store = app.CatalogStore(bundle)
    bundle["catalogs"]["es"]["welcome"] = "Changed {wrong}"
    assert store.render("es", "welcome", {"user": "A"})["text"] == "¡Hola, A!"
    with pytest.raises(ValueError):
        app.CatalogStore(bundle)


@pytest.mark.parametrize("locales", [[], ["es", "es"], ["en"], ["en_US"], [42]])
def test_invalid_locale_configuration_stops_before_agent_creation(source, locales):
    with pytest.raises(ValueError):
        app.generate_catalogs(source, locales, agent_factory=lambda _: pytest.fail("Unexpected Agent"))


def test_factory_role_must_match_locale(source):
    with pytest.raises(ValueError, match="role"):
        app.generate_catalogs(source, ["es"], agent_factory=lambda _: app.make_agent("fr", app.ScriptedModel("unused")))


def test_cli_generate_render_utf8_fallback_and_partial_failure(tmp_path, fixtures):
    args = [sys.executable, str(HERE / "catalogs.py")]
    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    output = tmp_path / "traductions.json"
    generate_args = ["generate", "--source", str(HERE / "source.json"), "--locales", "es", "fr",
                     "--fixtures", str(HERE / "fixtures.json"), "--output", str(output),
                     "--workspace", str(tmp_path / "cli-runtime")]
    run = subprocess.run(args + generate_args, env=env, capture_output=True, timeout=60)
    assert run.returncode == 0, run.stderr.decode("utf-8", errors="replace")
    raw = output.read_bytes()
    assert json.loads(raw)["mode"] == "scripted"
    for locale, expected, fallback in [("es", "¡Hola, Zoë!", False), ("de", "Welcome, Zoë!", True)]:
        rendered = subprocess.run(args + ["render", str(output), "--locale", locale,
                                          "--key", "welcome", "--value", "user=Zoë"],
                                  env=env, capture_output=True, timeout=20)
        assert rendered.returncode == 0
        result = json.loads(rendered.stdout.decode("utf-8"))
        assert result["text"] == expected and result["fallback"] is fallback
    again = subprocess.run(args + generate_args, env=env, capture_output=True, timeout=20)
    assert again.returncode == 2 and output.read_bytes() == raw
    fixtures["fr"]["messages"]["welcome"] = "Bonjour"
    invalid = tmp_path / "invalid-fixtures.json"
    invalid.write_text(json.dumps(fixtures), encoding="utf-8")
    generate_args[generate_args.index(str(HERE / "fixtures.json"))] = str(invalid)
    generate_args[generate_args.index(str(output))] = str(tmp_path / "partial.json")
    partial = subprocess.run(args + generate_args, env=env, capture_output=True, timeout=60)
    assert partial.returncode == 1
    assert set(json.loads((tmp_path / "partial.json").read_text(encoding="utf-8"))["catalogs"]) == {"es"}
