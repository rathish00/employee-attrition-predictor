"""
Smoke-tests app/app.py by stubbing out `streamlit` well enough to execute
its real code paths — login gate, all three dashboard pages, and both
batch-upload outcomes — without needing a real streamlit install or a
browser.

Honest limitation: this stub's cache_resource decorator just calls the
function directly with no real caching, so it CANNOT reproduce
Streamlit's actual "no UI elements inside a cached function" rule (the
bug that shipped and broke the first deploy). It catches NameErrors,
AttributeErrors, wrong call signatures, and logic bugs in the data flow —
not that specific class of caching violation. Treat this as a floor, not
a ceiling, on correctness.
"""
import io
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class _RerunSignal(Exception):
    """Mirrors streamlit's real behavior: st.rerun() halts the script."""


class SessionState(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

    def __setattr__(self, key, value):
        self[key] = value


class _Ctx:
    """Context manager that also forwards attribute access to the parent
    stub, so both `with st.sidebar:` + bare `st.radio(...)` AND
    `st.sidebar.radio(...)` work, matching how real streamlit containers
    behave."""

    def __init__(self, parent=None):
        self._parent = parent

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        if self._parent is not None and hasattr(self._parent, name):
            return getattr(self._parent, name)
        return lambda *a, **kw: None


class _SecretsStub(dict):
    def get(self, key, default=None):
        return dict.get(self, key, default)


class StreamlitStub(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = SessionState()
        self.secrets = _SecretsStub()  # empty by default -> auth falls back to demo creds
        self.sidebar = _Ctx(parent=self)
        self._uploaded_file = None
        self._button_results: dict[str, bool] = {}
        self._form_submit_results: dict[str, bool] = {}
        self._text_input_values: dict[str, str] = {}
        self._radio_results: dict[str, str] = {}

    # ---- no-op renderers ----
    def set_page_config(self, **kw): pass
    def markdown(self, *a, **kw): pass
    def title(self, *a, **kw): pass
    def caption(self, *a, **kw): pass
    def subheader(self, *a, **kw): pass
    def divider(self): pass
    def dataframe(self, *a, **kw): pass
    def download_button(self, *a, **kw): pass
    def header(self, *a, **kw): pass
    def bar_chart(self, *a, **kw): pass
    def metric(self, *a, **kw): pass
    def progress(self, *a, **kw): pass
    def image(self, *a, **kw): pass

    def success(self, *a, **kw): print("[st.success]", *a)
    def error(self, *a, **kw): print("[st.error]", *a)
    def info(self, *a, **kw): print("[st.info]", *a)
    def warning(self, *a, **kw): print("[st.warning]", *a)

    def stop(self):
        raise SystemExit("st.stop() called")

    def rerun(self):
        raise _RerunSignal()

    # ---- layout ----
    def columns(self, n, **kw):
        return [_Ctx(parent=self) for _ in (range(n) if isinstance(n, int) else n)]

    def container(self, **kw):
        return _Ctx(parent=self)

    def expander(self, *a, **kw):
        return _Ctx(parent=self)

    def form(self, *a, **kw):
        return _Ctx(parent=self)

    def cache_resource(self, *a, **kw):
        def deco(fn):
            return fn
        return deco if not (a and callable(a[0])) else a[0]

    # ---- widgets (return sensible fixed values so the script runs fully) ----
    def slider(self, label, min_val=None, max_val=None, value=None, **kw):
        return value if value is not None else (min_val or 0)

    def select_slider(self, label, options, value=None, **kw):
        return value if value is not None else options[0]

    def selectbox(self, label, options, **kw):
        return options[0] if options else None

    def radio(self, label, options, **kw):
        if label in self._radio_results:
            return self._radio_results[label]
        return options[0] if options else None

    def number_input(self, label, min_val=None, max_val=None, value=None, **kw):
        return value if value is not None else 0

    def text_input(self, label, *a, **kw):
        return self._text_input_values.get(label, "")

    def file_uploader(self, *a, **kw):
        return self._uploaded_file

    def button(self, label, *a, **kw):
        return self._button_results.get(label, True)

    def form_submit_button(self, label="Submit", *a, **kw):
        return self._form_submit_results.get(label, False)


def run_scenario(name, compiled_code, *, authenticated=True, uploaded_file=None,
                  button_results=None, form_submit_results=None,
                  text_input_values=None, radio_results=None) -> None:
    stub = StreamlitStub()
    stub.session_state["authenticated"] = authenticated
    stub._uploaded_file = uploaded_file
    stub._button_results = button_results or {}
    stub._form_submit_results = form_submit_results or {}
    stub._text_input_values = text_input_values or {}
    stub._radio_results = radio_results or {}
    sys.modules["streamlit"] = stub

    spec_path = PROJECT_ROOT / "app" / "app.py"
    namespace = {"__name__": "app_under_test", "__file__": str(spec_path)}
    try:
        exec(compiled_code, namespace)
        namespace["main"]()
        print(f"SCENARIO PASSED: {name}")
    except _RerunSignal:
        print(f"SCENARIO PASSED (stopped via st.rerun): {name}")
    except SystemExit as e:
        print(f"SCENARIO STOPPED via st.stop(): {name} -> {e}")
        raise
    except Exception as e:
        print(f"SCENARIO FAILED: {name} -> {type(e).__name__}: {e}")
        raise


def main() -> None:
    spec_path = PROJECT_ROOT / "app" / "app.py"
    compiled = compile(spec_path.read_text(), str(spec_path), "exec")

    run_scenario("login screen renders (not authenticated)", compiled, authenticated=False)

    run_scenario(
        "login with correct credentials",
        compiled,
        authenticated=False,
        form_submit_results={"Sign In": True},
        text_input_values={"Username": "admin", "Password": "attrition2026"},
    )

    run_scenario(
        "login with wrong password",
        compiled,
        authenticated=False,
        form_submit_results={"Sign In": True},
        text_input_values={"Username": "admin", "Password": "wrong"},
    )

    run_scenario(
        "single-employee prediction",
        compiled,
        authenticated=True,
        radio_results={"Navigate": "Single Employee"},
        button_results={"🚪 Logout": False, "🔍 Predict Risk": True},
    )

    import pandas as pd
    raw = pd.read_csv(PROJECT_ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv").head(10)
    buf = io.BytesIO(raw.to_csv(index=False).encode())
    buf.name = "batch.csv"
    run_scenario(
        "batch CSV upload with valid schema",
        compiled,
        authenticated=True,
        radio_results={"Navigate": "Batch Scoring"},
        uploaded_file=buf,
        button_results={"🚪 Logout": False},
    )

    bad = raw.drop(columns=["OverTime"])
    buf2 = io.BytesIO(bad.to_csv(index=False).encode())
    buf2.name = "bad_batch.csv"
    run_scenario(
        "batch CSV upload with missing column (expect handled error)",
        compiled,
        authenticated=True,
        radio_results={"Navigate": "Batch Scoring"},
        uploaded_file=buf2,
        button_results={"🚪 Logout": False},
    )

    run_scenario(
        "model insights page",
        compiled,
        authenticated=True,
        radio_results={"Navigate": "Model Insights"},
        button_results={"🚪 Logout": False},
    )

    run_scenario(
        "logout",
        compiled,
        authenticated=True,
        radio_results={"Navigate": "Single Employee"},
        button_results={"🚪 Logout": True, "🔍 Predict Risk": False},
    )

    print("\nALL SMOKE TEST SCENARIOS PASSED.")


if __name__ == "__main__":
    main()
