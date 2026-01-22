from __future__ import annotations

import base64
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
import streamlit as st
from typing import List, Sequence



# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Family Workout Generator",
    page_icon="💪",
    layout="wide",
)


# ---------------------------
# Constants / schema
# ---------------------------
SECTIONS = ("main", "accessory", "finisher")
DEFAULT_DATA_FILE = "workout_data.json"
DEFAULT_HISTORY_FILE = "workout_history.json"


# ---------------------------
# Helpers: validation
# ---------------------------
def filter_items(items: Sequence[str], query: str) -> List[str]:
    terms = [t for t in query.casefold().split() if t.strip()]
    if not terms:
        return list(items)
    out = []
    for x in items:
        x_cf = x.casefold()
        if all(t in x_cf for t in terms):
            out.append(x)
    return out

def validate_schema(data: dict) -> None:
    if not isinstance(data, dict):
        raise ValueError("Top-level data must be a JSON object.")

    for key in ("prescriptions", "lengths", "days"):
        if key not in data:
            raise ValueError(f"Missing required key: {key}")

    if not isinstance(data["prescriptions"], dict):
        raise ValueError("'prescriptions' must be an object.")
    for req in SECTIONS:
        if req not in data["prescriptions"] or not isinstance(data["prescriptions"][req], str):
            raise ValueError(f"'prescriptions.{req}' must exist and be a string.")

    if not isinstance(data["lengths"], dict) or not data["lengths"]:
        raise ValueError("'lengths' must be a non-empty object.")
    for length_name, cfg in data["lengths"].items():
        if not isinstance(cfg, dict):
            raise ValueError(f"'lengths.{length_name}' must be an object.")
        for req in ("mains", "accessories", "finisher"):
            if req not in cfg:
                raise ValueError(f"'lengths.{length_name}' missing '{req}'.")
        if not isinstance(cfg["mains"], int) or cfg["mains"] < 1:
            raise ValueError(f"'lengths.{length_name}.mains' must be int >= 1.")
        if not isinstance(cfg["accessories"], int) or cfg["accessories"] < 0:
            raise ValueError(f"'lengths.{length_name}.accessories' must be int >= 0.")
        if not isinstance(cfg["finisher"], bool):
            raise ValueError(f"'lengths.{length_name}.finisher' must be boolean.")

    if not isinstance(data["days"], dict) or not data["days"]:
        raise ValueError("'days' must be a non-empty object.")
    for day_name, sec in data["days"].items():
        if not isinstance(sec, dict):
            raise ValueError(f"'days.{day_name}' must be an object.")
        for req in SECTIONS:
            if req not in sec or not isinstance(sec[req], list) or not sec[req]:
                raise ValueError(f"'days.{day_name}.{req}' must be a non-empty list.")
            if not all(isinstance(x, str) and x.strip() for x in sec[req]):
                raise ValueError(f"'days.{day_name}.{req}' must contain non-empty strings only.")


def validate_invariants(data: dict) -> None:
    # Strict no-repeat feasibility: mains >= 2 * max_mains
    max_mains = max(int(cfg["mains"]) for cfg in data["lengths"].values())
    needed = 2 * max_mains
    for day_name, sec in data["days"].items():
        if len(sec["main"]) < needed:
            raise ValueError(
                f"Invariant failed: day '{day_name}' has {len(sec['main'])} main lifts; "
                f"needs >= {needed} to guarantee no-repeat with max_mains={max_mains}."
            )


# ---------------------------
# Helpers: patch apply (in-memory)
# ---------------------------
def _split_path(ptr: str) -> List[str]:
    if ptr == "":
        return []
    if not ptr.startswith("/"):
        raise ValueError(f"Invalid JSON pointer: {ptr}")
    parts = ptr.lstrip("/").split("/")
    return [p.replace("~1", "/").replace("~0", "~") for p in parts]


def _get_parent(doc: Any, path: str) -> Tuple[Any, str]:
    parts = _split_path(path)
    if not parts:
        raise ValueError("Path points to document root; not supported in this patcher.")
    key = parts[-1]
    cur = doc
    for p in parts[:-1]:
        if isinstance(cur, list):
            cur = cur[int(p)]
        elif isinstance(cur, dict):
            cur = cur[p]
        else:
            raise TypeError(f"Cannot traverse non-container at '{p}'.")
    return cur, key


def apply_patch(doc: Any, ops: List[dict]) -> Any:
    for op in ops:
        kind = op["op"]
        path = op["path"]

        if kind in ("add", "replace"):
            value = op["value"]
            parent, key = _get_parent(doc, path)
            if isinstance(parent, list):
                idx = int(key)
                if kind == "add":
                    parent.insert(idx, value)
                else:
                    parent[idx] = value
            elif isinstance(parent, dict):
                parent[key] = value
            else:
                raise TypeError(f"Parent at {path} is not a container.")
        elif kind == "remove":
            parent, key = _get_parent(doc, path)
            if isinstance(parent, list):
                del parent[int(key)]
            elif isinstance(parent, dict):
                del parent[key]
            else:
                raise TypeError(f"Parent at {path} is not a container.")
        else:
            raise ValueError(f"Unsupported op: {kind}")
    return doc


# ---------------------------
# Backend: GitHub storage (recommended for Streamlit Cloud persistence)
# ---------------------------
@dataclass(frozen=True)
class GitHubConfig:
    token: str
    repo: str         # "owner/repo"
    branch: str       # e.g. "main"
    data_path: str    # e.g. "workout_data.json"
    history_path: str # e.g. "workout_history.json"


def get_github_config() -> Optional[GitHubConfig]:
    """
    Reads GitHub config from Streamlit secrets if available, otherwise env vars.
    IMPORTANT: st.secrets raises StreamlitSecretNotFoundError when no secrets.toml exists,
    so we must guard access with try/except.
    """
    secrets = None
    try:
        secrets = st.secrets  # may raise if no secrets.toml
    except Exception:
        secrets = None

    def _get(key: str) -> Optional[str]:
        # secrets.toml
        if secrets is not None:
            try:
                if "github" in secrets and key in secrets["github"]:
                    return str(secrets["github"][key])
            except Exception:
                pass
        # environment variables
        return os.getenv(key)

    token = _get("GITHUB_TOKEN")
    repo = _get("GITHUB_REPO")
    branch = _get("GITHUB_BRANCH") or "main"
    data_path = _get("GITHUB_DATA_PATH") or DEFAULT_DATA_FILE
    history_path = _get("GITHUB_HISTORY_PATH") or DEFAULT_HISTORY_FILE

    if token and repo:
        return GitHubConfig(
            token=token,
            repo=repo,
            branch=branch,
            data_path=data_path,
            history_path=history_path,
        )
    return None


def _gh_headers(cfg: GitHubConfig) -> Dict[str, str]:
    return {
        "Authorization": f"token {cfg.token}",
        "Accept": "application/vnd.github+json",
    }


def github_get_file(cfg: GitHubConfig, path: str) -> Tuple[dict, str]:
    """
    Returns: (json_content, sha)
    Uses GitHub "contents" API. If file doesn't exist, raises.
    """
    url = f"https://api.github.com/repos/{cfg.repo}/contents/{path}"
    params = {"ref": cfg.branch}
    r = requests.get(url, headers=_gh_headers(cfg), params=params, timeout=20)
    r.raise_for_status()
    payload = r.json()
    content_b64 = payload["content"]
    sha = payload["sha"]
    raw = base64.b64decode(content_b64).decode("utf-8")
    return json.loads(raw), sha


def github_put_file(cfg: GitHubConfig, path: str, data: dict, sha: Optional[str], message: str) -> None:
    url = f"https://api.github.com/repos/{cfg.repo}/contents/{path}"
    raw = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    body = {
        "message": message,
        "content": base64.b64encode(raw.encode("utf-8")).decode("utf-8"),
        "branch": cfg.branch,
    }
    if sha:
        body["sha"] = sha

    r = requests.put(url, headers=_gh_headers(cfg), json=body, timeout=20)
    r.raise_for_status()


# ---------------------------
# Local storage (dev-only / fallback)
# ---------------------------
def load_local_json(path: str, default: dict) -> dict:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_local_json(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


# ---------------------------
# App state: load/save data + history
# ---------------------------
def default_history(days: Sequence[str]) -> dict:
    return {"last_main": {d: [] for d in days}}


def load_data_and_history() -> Tuple[dict, dict, str]:
    """
    Returns (data, history, backend_label)
    """
    cfg = get_github_config()
    if cfg:
        data, data_sha = github_get_file(cfg, cfg.data_path)
        history, hist_sha = None, None
        try:
            history, hist_sha = github_get_file(cfg, cfg.history_path)
        except Exception:
            # history might not exist yet
            history = default_history(list(data["days"].keys()))
            hist_sha = None

        st.session_state["_gh_data_sha"] = data_sha
        st.session_state["_gh_hist_sha"] = hist_sha
        return data, history, f"GitHub: {cfg.repo}@{cfg.branch}"

    # local fallback (note: not persistent on Community Cloud across restarts) :contentReference[oaicite:1]{index=1}
    data = load_local_json(DEFAULT_DATA_FILE, default={})
    if not data:
        raise RuntimeError(f"Missing {DEFAULT_DATA_FILE}. Add it to your repo.")
    history = load_local_json(DEFAULT_HISTORY_FILE, default=default_history(list(data["days"].keys())))
    return data, history, "Local file (dev)"


def save_data_and_history(data: dict, history: dict, message: str) -> None:
    cfg = get_github_config()
    if cfg:
        # optimistic concurrency: include sha; if conflict occurs, reload and retry once
        data_sha = st.session_state.get("_gh_data_sha")
        hist_sha = st.session_state.get("_gh_hist_sha")

        # Save data
        github_put_file(cfg, cfg.data_path, data, data_sha, message=f"{message} (data)")
        # Refresh sha after write
        new_data, new_data_sha = github_get_file(cfg, cfg.data_path)
        st.session_state["_gh_data_sha"] = new_data_sha

        # Save history
        github_put_file(cfg, cfg.history_path, history, hist_sha, message=f"{message} (history)")
        try:
            _, new_hist_sha = github_get_file(cfg, cfg.history_path)
            st.session_state["_gh_hist_sha"] = new_hist_sha
        except Exception:
            st.session_state["_gh_hist_sha"] = None
        return

    # local fallback
    save_local_json(DEFAULT_DATA_FILE, data)
    save_local_json(DEFAULT_HISTORY_FILE, history)


# ---------------------------
# Workout generation logic
# ---------------------------
def pick_unique(rng: random.Random, pool: Sequence[str], n: int) -> Tuple[str, ...]:
    n = min(n, len(pool))
    return tuple(rng.sample(list(pool), k=n))


def pick_mains_no_repeat(rng: random.Random, pool: Sequence[str], n: int, last: Sequence[str]) -> Tuple[str, ...]:
    forbidden = {x.strip().casefold() for x in last}
    candidates = [x for x in pool if x.strip().casefold() not in forbidden]
    if len(candidates) < n:
        raise RuntimeError("No-repeat constraint unsatisfiable. Add more main lifts to the day.")
    return tuple(rng.sample(candidates, k=n))


def generate_workout(data: dict, history: dict, day: str, length: str) -> Tuple[str, dict]:
    cfg = data["lengths"][length]
    sec = data["days"][day]

    last_main_all = history.get("last_main", {})
    last_main = last_main_all.get(day, [])

    rng = random.Random()
    mains = pick_mains_no_repeat(rng, sec["main"], cfg["mains"], last_main)
    accessories = pick_unique(rng, sec["accessory"], cfg["accessories"])
    finisher = rng.choice(sec["finisher"]) if cfg["finisher"] else None

    # update history
    history.setdefault("last_main", {})
    history["last_main"][day] = list(mains)

    # format output
    p = data["prescriptions"]
    lines = []
    lines.append(f"Day type: {day.capitalize()}")
    lines.append(f"Length: {length.capitalize()}")
    lines.append("")
    lines.append("Main Lift(s):")
    for ex in mains:
        lines.append(f"  • {ex}: {p['main']}")
    lines.append("")
    lines.append("Accessories:")
    for ex in accessories:
        lines.append(f"  • {ex}: {p['accessory']}")
    if finisher:
        lines.append("")
        lines.append("Finisher:")
        lines.append(f"  • {finisher}")
        lines.append(f"  • {p['finisher']}")
    return "\n".join(lines), history


# ---------------------------
# Auth (optional)
# ---------------------------
# def require_password() -> None:
    # If APP_PASSWORD is set, require it; otherwise open access.
    pw = None
    if "APP_PASSWORD" in st.secrets:
        pw = str(st.secrets["APP_PASSWORD"])
    else:
        pw = os.getenv("APP_PASSWORD")

    if not pw:
        return

    if st.session_state.get("auth_ok"):
        return

    st.title("🔒 Family Workout Generator")
    entered = st.text_input("Enter password", type="password")
    if st.button("Unlock", type="primary"):
        if entered == pw:
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")


# ---------------------------
# UI: main
# ---------------------------
# require_password()

# Load once per session; provide refresh button
if "data" not in st.session_state or "history" not in st.session_state:
    data, history, backend_label = load_data_and_history()
    validate_schema(data)
    validate_invariants(data)
    st.session_state["data"] = data
    st.session_state["history"] = history
    st.session_state["backend_label"] = backend_label
    st.session_state["last_edit"] = None  # undo


data = st.session_state["data"]
history = st.session_state["history"]

st.title("💪 Family Workout Generator")
st.caption(f"Storage: {st.session_state.get('backend_label','unknown')}")

colA, colB = st.columns([1, 1], gap="large")

with colA:
    st.subheader("Generate a workout")
    day = st.selectbox("Day type", options=list(data["days"].keys()))
    length = st.selectbox("Session length", options=list(data["lengths"].keys()))

    if st.button("Generate", type="primary"):
        try:
            workout_text, new_history = generate_workout(data, history, day, length)
            st.session_state["history"] = new_history
            save_data_and_history(st.session_state["data"], st.session_state["history"], message="Generate workout (update history)")
            st.session_state["last_workout"] = workout_text
            st.success("Workout generated.")
        except Exception as e:
            st.error(str(e))

    if st.button("Refresh data from storage"):
        try:
            data2, history2, backend_label = load_data_and_history()
            validate_schema(data2)
            validate_invariants(data2)
            st.session_state["data"] = data2
            st.session_state["history"] = history2
            st.session_state["backend_label"] = backend_label
            st.success("Reloaded.")
            st.rerun()
        except Exception as e:
            st.error(str(e))

with colB:
    st.subheader("Workout output")
    workout_text = st.session_state.get("last_workout", "Generate a workout to see it here.")
    st.code(workout_text, language="text")


st.divider()
tab_edit, tab_admin = st.tabs(["Edit exercises", "Admin"])

with tab_edit:
    st.subheader("Edit exercises (shared)")

    edit_col1, edit_col2 = st.columns([1, 1], gap="large")

    with edit_col1:
        st.markdown("#### Remove")
        r_day = st.selectbox("Remove from day", options=list(data["days"].keys()), key="r_day")
        r_section = st.selectbox("Section", options=list(SECTIONS), key="r_section")

        full_list = list(data["days"][r_day][r_section])
        query = st.text_input("Search", value="", key="r_search")
        filtered = filter_items(full_list, query)

        if not filtered:
            st.info("No matches.")
            chosen = None
        else:
            # Streamlit selectbox supports typing search inside the widget, too.
            chosen = st.selectbox("Choose exercise to remove", options=filtered, key="r_choice")

        if st.button("Remove selected", type="secondary", disabled=(chosen is None)):
            try:
                idx = find_index_case_insensitive(full_list, chosen)
                if idx < 0:
                    raise ValueError("Exercise not found (data changed). Refresh and try again.")

                ops = [{"op": "remove", "path": f"/days/{r_day}/{r_section}/{idx}"}]
                # Prepare undo (re-add at original index)
                undo = {
                    "type": "remove",
                    "day": r_day,
                    "section": r_section,
                    "name": full_list[idx],
                    "index": idx,
                }

                updated = json.loads(json.dumps(data))  # deep copy
                apply_patch(updated, ops)
                validate_schema(updated)
                validate_invariants(updated)

                st.session_state["data"] = updated
                st.session_state["last_edit"] = undo
                save_data_and_history(st.session_state["data"], st.session_state["history"], message="Edit exercises")
                st.success("Removed.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    with edit_col2:
        st.markdown("#### Add")
        a_day = st.selectbox("Add to day", options=list(data["days"].keys()), key="a_day")
        a_section = st.selectbox("Section", options=list(SECTIONS), key="a_section")
        new_name = st.text_input("Exercise name", key="a_name").strip()

        if st.button("Add", type="primary", disabled=(not new_name)):
            try:
                current = list(data["days"][a_day][a_section])
                if any(x.strip().casefold() == new_name.casefold() for x in current):
                    raise ValueError("That exercise already exists in this list.")

                idx = len(current)
                ops = [{"op": "add", "path": f"/days/{a_day}/{a_section}/{idx}", "value": new_name}]
                undo = {"type": "add", "day": a_day, "section": a_section, "name": new_name}

                updated = json.loads(json.dumps(data))  # deep copy
                apply_patch(updated, ops)
                validate_schema(updated)
                validate_invariants(updated)

                st.session_state["data"] = updated
                st.session_state["last_edit"] = undo
                save_data_and_history(st.session_state["data"], st.session_state["history"], message="Edit exercises")
                st.success("Added.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    st.markdown("---")
    st.markdown("#### Undo last edit")
    last_edit = st.session_state.get("last_edit")
    if not last_edit:
        st.caption("No edit to undo.")
    else:
        st.caption(f"Last edit: {last_edit['type']} ({last_edit['day']}/{last_edit['section']}) — {last_edit['name']}")
        if st.button("Undo", type="secondary"):
            try:
                updated = json.loads(json.dumps(data))  # deep copy

                if last_edit["type"] == "add":
                    day = last_edit["day"]
                    section = last_edit["section"]
                    name = last_edit["name"]
                    arr = list(updated["days"][day][section])
                    idx = find_index_case_insensitive(arr, name)
                    if idx < 0:
                        raise ValueError("Cannot undo: added exercise not found.")
                    ops = [{"op": "remove", "path": f"/days/{day}/{section}/{idx}"}]

                elif last_edit["type"] == "remove":
                    day = last_edit["day"]
                    section = last_edit["section"]
                    name = last_edit["name"]
                    idx0 = int(last_edit.get("index", 0))
                    arr = list(updated["days"][day][section])
                    insert_at = max(0, min(idx0, len(arr)))
                    ops = [{"op": "add", "path": f"/days/{day}/{section}/{insert_at}", "value": name}]
                else:
                    raise ValueError("Unknown undo type.")

                apply_patch(updated, ops)
                validate_schema(updated)
                validate_invariants(updated)

                st.session_state["data"] = updated
                st.session_state["last_edit"] = None
                save_data_and_history(st.session_state["data"], st.session_state["history"], message="Undo edit")
                st.success("Undo complete.")
                st.rerun()
            except Exception as e:
                st.error(str(e))


with tab_admin:
    st.subheader("Admin / Notes")

    st.markdown(
        """
- If you're on Streamlit Community Cloud, **writing to local files is not persistent across restarts**.  
  Use the GitHub backend (below) or another external datastore to keep edits permanent. :contentReference[oaicite:2]{index=2}
- Use Streamlit **Secrets Management** for passwords and GitHub tokens. :contentReference[oaicite:3]{index=3}
"""
    )

    st.markdown("### Current config (detected)")
    cfg = get_github_config()
    if cfg:
        st.success("Using GitHub storage backend.")
        st.json(
            {
                "repo": cfg.repo,
                "branch": cfg.branch,
                "data_path": cfg.data_path,
                "history_path": cfg.history_path,
            }
        )
    else:
        st.warning("Using local file backend (good for local dev; not reliable for Cloud persistence).")
        st.code(
            f"Expected files: {DEFAULT_DATA_FILE}, {DEFAULT_HISTORY_FILE}\n"
            "To enable GitHub storage, set secrets: github.GITHUB_TOKEN and github.GITHUB_REPO",
            language="text",
        )
