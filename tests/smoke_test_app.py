"""
Smoke-tests app/app.py by stubbing out `streamlit` well enough to execute
every code path once: model loading, the single-employee prediction flow,
and the batch-scoring flow — without needing a real streamlit install or
a browser. This won't catch CSS/rendering issues, but it does catch
NameErrors, AttributeErrors, wrong call signatures, and logic bugs in the
actual data flow between the UI and the model.
"""
import io
import sys
import types
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class StreamlitStub(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.sidebar = _Ctx()
        self._widget_queue = []
        self._uploaded_file = None

    # no-op renderers
    def set_page_config(self, **kw): pass
    def markdown(self, *a, **kw): pass
    def title(self, *a, **kw): pass
    def caption(self, *a, **kw): pass
    def subheader(self, *a, **kw): pass
    def divider(self): pass
    def success(self, *a, **kw): print("[st.success]", *a)
    def error(self, *a, **kw): print("[st.error]", *a)
    def info(self, *a, **kw): print("[st.info]", *a)
    def warning(self, *a, **kw): print("[st.warning]", *a)
    def dataframe(self, *a, **kw): pass
    def download_button(self, *a, **kw): pass
    def header(self, *a, **kw): pass
    def toast(self, *a, **kw): print("[st.toast]", *a)

    def stop(self):
        raise SystemExit("st.stop() called")

    def columns(self, n, **kw):
        return [_Ctx() for _ in (range(n) if isinstance(n, int) else n)]

    def cache_resource(self, *a, **kw):
        def deco(fn):
            return fn
        return deco if not (a and callable(a[0])) else a[0]

    # widgets: return sensible fixed defaults so the script executes fully
    def slider(self, label, min_val=None, max_val=None, value=None, **kw):
        return value if value is not None else (min_val or 0)

    def select_slider(self, label, options, value=None, **kw):
        return value if value is not None else options[0]

    def selectbox(self, label, options, **kw):
        return options[0] if options else None

    def number_input(self, label, min_val=None, max_val=None, value=None, **kw):
        return value if value is not None else 0

    def file_uploader(self, *a, **kw):
        return self._uploaded_file  # None by default; set by test scenarios

    def button(self, *a, **kw):
        return True  # force the prediction branch to execute


spec_path = PROJECT_ROOT / "app" / "app.py"
code = spec_path.read_text()
compiled = compile(code, str(spec_path), "exec")


def run_scenario(name: str, uploaded_file=None) -> None:
    stub = StreamlitStub()
    stub._uploaded_file = uploaded_file
    sys.modules["streamlit"] = stub
    namespace = {"__name__": "app_under_test", "__file__": str(spec_path)}
    try:
        exec(compiled, namespace)
        namespace["main"]()
        print(f"SCENARIO PASSED: {name}")
    except SystemExit as e:
        print(f"SCENARIO STOPPED EARLY via st.stop(): {name} -> {e}")
        raise
    except Exception as e:
        print(f"SCENARIO FAILED: {name} -> {type(e).__name__}: {e}")
        raise


import pandas as pd  # noqa: E402

# Scenario 1: no CSV uploaded, single-employee predict button pressed
run_scenario("single-employee prediction, no batch upload")

# Scenario 2: batch CSV uploaded with correct schema
raw = pd.read_csv(PROJECT_ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv").head(10)
buf = io.BytesIO(raw.to_csv(index=False).encode())
buf.name = "batch.csv"
run_scenario("batch CSV upload with valid schema", uploaded_file=buf)

# Scenario 3: batch CSV uploaded with a missing required column (should error gracefully, not crash)
bad = raw.drop(columns=["OverTime"])
buf2 = io.BytesIO(bad.to_csv(index=False).encode())
buf2.name = "bad_batch.csv"
run_scenario("batch CSV upload with missing column (expect handled error)", uploaded_file=buf2)

print("\nALL SMOKE TEST SCENARIOS PASSED.")
