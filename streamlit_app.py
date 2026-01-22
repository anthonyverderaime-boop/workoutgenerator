from __future__ import annotations

import base64
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
import streamlit as st


# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Tony's Fitness Dojo: Workout Generator",
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
# Backend: GitHub storage
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
    Safe secrets access: Streamlit throws if no secrets.toml exists.
    """
    secrets = None
    try:
        secrets = st.secrets
    except Exception:
        secrets = None

    def _get(key: str) -> Optional[str]:
        if secrets is not None:
            try:
                if "github" in secrets and key in secrets["github"]:
                    return str(secrets["github"][key])
            except Exception:
                pass
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
    body: Dict[str, Any] = {
        "message": message,
        "content": base64.b64encode(raw.encode("utf-8")).decode("utf-8"),
        "branch": cfg.branch,
    }
    if sha:
        body["sha"] = sha

    r = requests.put(url, headers=_gh_headers(cfg), json=body, timeout=20)
    r.raise_for_status()


# ---------------------------
# Local storage (dev-only fallback)
# ---------------------------
def load_local_json(path: str, default: dict) -> dict:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
        if not txt:
            return default
        return json.loads(txt)


def save_local_json(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


# ---------------------------
# History: per-user schema + migration
# ---------------------------
def default_history(days: Sequence[str]) -> dict:
    # New schema: per-user
    return {"last_main_by_user": {}}


def ensure_history_schema(history: dict, days: Sequence[str]) -> dict:
    """
    Ensures history has:
      history["last_main_by_user"][user_id][day] -> list[str]
    Migrates old shared schema ("last_main") if present.
    """
    if not isinstance(history, dict):
        history = {}

    if "last_main_by_user" not in history or not isinstance(history["last_main_by_user"], dict):
        history["last_main_by_user"] = {}

    # Migration from old shared schema
    old = history.get("last_main")
    if isinstance(old, dict):
        # preserve old behavior under a special key
        history["last_main_by_user"].setdefault("_shared", {})
        for d in days:
            history["last_main_by_user"]["_shared"][d] = list(old.get(d, []))
        history.pop("last_main", None)

    # Ensure each user record is shaped correctly
    for user_id, per_day in list(history["last_main_by_user"].items()):
        if not isinstance(per_day, dict):
            history["last_main_by_user"][user_id] = {d: [] for d in days}
            continue
        for d in days:
            v = per_day.get(d, [])
            history["last_main_by_user"][user_id][d] = list(v) if isinstance(v, list) else []

    return history


# ---------------------------
# Load/save data & history (GitHub or local)
# ---------------------------
def load_data_and_history() -> Tuple[dict, dict, str]:
    cfg = get_github_config()
    if cfg:
        data, data_sha = github_get_file(cfg, cfg.data_path)

        try:
            history, hist_sha = github_get_file(cfg, cfg.history_path)
        except Exception:
            history = default_history(list(data["days"].keys()))
            hist_sha = None

        st.session_state["_gh_data_sha"] = data_sha
        st.session_state["_gh_hist_sha"] = hist_sha

        history = ensure_history_schema(history, list(data["days"].keys()))
        return data, history, f"GitHub: {cfg.repo}@{cfg.branch}"

    data = load_local_json(DEFAULT_DATA_FILE, default={})
    if not data:
        raise RuntimeError(f"Missing {DEFAULT_DATA_FILE}. Add it next to streamlit_app.py.")
    history = load_local_json(DEFAULT_HISTORY_FILE, default=default_history(list(data["days"].keys())))
    history = ensure_history_schema(history, list(data["days"].keys()))
    return data, history, "Local file (dev)"


def save_data_only(data: dict, message: str) -> None:
    cfg = get_github_config()
    if cfg:
        data_sha = st.session_state.get("_gh_data_sha")
        github_put_file(cfg, cfg.data_path, data, data_sha, message=message)
        _, new_sha = github_get_file(cfg, cfg.data_path)
        st.session_state["_gh_data_sha"] = new_sha
        return
    save_local_json(DEFAULT_DATA_FILE, data)


def save_history_only(history: dict, message: str) -> None:
    cfg = get_github_config()
    if cfg:
        hist_sha = st.session_state.get("_gh_hist_sha")
        try:
            github_put_file(cfg, cfg.history_path, history, hist_sha, message=message)
            _, new_sha = github_get_file(cfg, cfg.history_path)
            st.session_state["_gh_hist_sha"] = new_sha
            return
        except requests.HTTPError as e:
            # If two people generate at the same time, we can merge and retry once
            status = getattr(e.response, "status_code", None)
            if status in (409, 422):
                latest, latest_sha = github_get_file(cfg, cfg.history_path)
                days = list(st.session_state["data"]["days"].keys())
                latest = ensure_history_schema(latest, days)

                # Merge per-user maps (ours overwrites same-user entries; preserves others)
                merged = latest
                merged.setdefault("last_main_by_user", {})
                merged["last_main_by_user"].update(history.get("last_main_by_user", {}))

                github_put_file(cfg, cfg.history_path, merged, latest_sha, message=message + " (merge)")
                _, new_sha = github_get_file(cfg, cfg.history_path)
                st.session_state["_gh_hist_sha"] = new_sha
                st.session_state["history"] = merged
                return
            raise
    save_local_json(DEFAULT_HISTORY_FILE, history)


# ---------------------------
# User identity: per-person history key
# ---------------------------
def _get_user_email() -> Optional[str]:
    """
    Prefer st.user (new), fallback to st.experimental_user (older).
    On Community Cloud, email is commonly available when the viewer is logged in. :contentReference[oaicite:1]{index=1}
    """
    # New API
    try:
        u = getattr(st, "user", None)
        if u is not None:
            email = None
            try:
                email = u.get("email")  # dict-like
            except Exception:
                email = getattr(u, "email", None)
            if email:
                return str(email).strip().lower()
    except Exception:
        pass

    # Older API (may be deprecated over time) :contentReference[oaicite:2]{index=2}
    try:
        eu = getattr(st, "experimental_user", None)
        if eu is not None:
            email = None
            try:
                email = eu.get("email")
            except Exception:
                email = getattr(eu, "email", None)
            if email:
                return str(email).strip().lower()
    except Exception:
        pass

    return None


def _normalize_name_key(name: str) -> Optional[str]:
    n = name.strip()
    if not n:
        return None
    return "name:" + n.casefold()


def get_or_choose_user_id(history: dict) -> Tuple[str, str]:
    """
    Returns (user_id_key, display_name).
    If email is available, use it. Otherwise ask user to pick/enter a name.
    """
    email = _get_user_email()
    if email:
        return f"email:{email}", email

    st.sidebar.markdown("### Who are you?")
    known = sorted(
        k for k in history.get("last_main_by_user", {}).keys()
        if isinstance(k, str) and (k.startswith("name:"))
    )
    labels = ["(new)"] + [k.replace("name:", "") for k in known]

    choice = st.sidebar.selectbox("Select your name", options=labels, index=0)
    if choice == "(new)":
        entered = st.sidebar.text_input("Type your name").strip()
        key = _normalize_name_key(entered)
        if key:
            return key, entered
        # temporary anonymous key (session only)
        anon = st.session_state.setdefault("_anon_user_key", f"name:guest-{random.randint(1000,9999)}")
        return anon, anon.replace("name:", "")
    else:
        key = _normalize_name_key(choice)
        return key, choice


# ---------------------------
# Search helper
# ---------------------------
def filter_items(items: Sequence[str], query: str) -> List[str]:
    terms = [t for t in query.casefold().split() if t.strip()]
    if not terms:
        return list(items)
    out: List[str] = []
    for x in items:
        x_cf = x.casefold()
        if all(t in x_cf for t in terms):
            out.append(x)
    return out


def find_index_case_insensitive(items: Sequence[str], target: str) -> int:
    t = target.strip().casefold()
    for i, x in enumerate(items):
        if x.strip().casefold() == t:
            return i
    return -1


# ---------------------------
# Workout generation (per-user history)
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


def generate_workout(data: dict, history: dict, user_id: str, day: str, length: str) -> Tuple[str, dict]:
    cfg = data["lengths"][length]
    sec = data["days"][day]

    history = ensure_history_schema(history, list(data["days"].keys()))
    by_user = history.setdefault("last_main_by_user", {})
    by_user.setdefault(user_id, {d: [] for d in data["days"].keys()})
    last_main = by_user[user_id].get(day, [])

    rng = random.Random()
    mains = pick_mains_no_repeat(rng, sec["main"], cfg["mains"], last_main)
    accessories = pick_unique(rng, sec["accessory"], cfg["accessories"])
    finisher = rng.choice(sec["finisher"]) if cfg["finisher"] else None

    # Update THIS USER only
    by_user[user_id][day] = list(mains)

    # Format output
    p = data["prescriptions"]
    lines = []
    lines.append(f"User: {user_id}")
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


# =========================
# App start
# =========================

# Load once per session
if "data" not in st.session_state or "history" not in st.session_state:
    data, history, backend_label = load_data_and_history()
    validate_schema(data)
    validate_invariants(data)
    st.session_state["data"] = data
    st.session_state["history"] = history
    st.session_state["backend_label"] = backend_label
    st.session_state["last_edit"] = None  # undo for edits
    st.session_state["last_workout"] = None

data = st.session_state["data"]
history = st.session_state["history"]

# Determine per-user key
user_id, user_label = get_or_choose_user_id(history)

st.title("💪 Tony's Fitness Dojo: Workout Generator  💪")
st.caption(f"Storage: {st.session_state.get('backend_label','unknown')} • User: {user_label}")

colA, colB = st.columns([1, 1], gap="large")

with colA:
    st.subheader("Generate a workout")
    day = st.selectbox("Day type", options=list(data["days"].keys()))
    length = st.selectbox("Session length", options=list(data["lengths"].keys()))

    if st.button("Generate", type="primary"):
        try:
            workout_text, new_history = generate_workout(data, history, user_id, day, length)
            st.session_state["history"] = new_history
            save_history_only(st.session_state["history"], message=f"Update history for {user_id}")
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
    workout_text = st.session_state.get("last_workout") or "Generate a workout to see it here."
    st.code(workout_text, language="text")


st.divider()
tab_edit, tab_admin = st.tabs(["Edit exercises", "Admin"])

with tab_edit:
    st.subheader("Edit exercises (shared lists)")

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
            chosen = st.selectbox("Choose exercise to remove", options=filtered, key="r_choice")

        if st.button("Remove selected", type="secondary", disabled=(chosen is None)):
            try:
                idx = find_index_case_insensitive(full_list, chosen)
                if idx < 0:
                    raise ValueError("Exercise not found (data changed). Refresh and try again.")

                ops = [{"op": "remove", "path": f"/days/{r_day}/{r_section}/{idx}"}]
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
                save_data_only(st.session_state["data"], message=f"Edit exercises by {user_id}")
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
                save_data_only(st.session_state["data"], message=f"Edit exercises by {user_id}")
                st.success("Added.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    st.markdown("---")
    st.markdown("#### Undo last edit (lists)")
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
                save_data_only(st.session_state["data"], message=f"Undo edit by {user_id}")
                st.success("Undo complete.")
                st.rerun()
            except Exception as e:
                st.error(str(e))


with tab_admin:
    st.subheader("Admin / Notes")
    st.markdown(
        """
- “No-repeat mains” is now **per user**:
  - Preferred ID: logged-in email via `st.user.email` (or fallback `st.experimental_user.email`). :contentReference[oaicite:3]{index=3}
  - Local dev fallback: select/enter a name in the sidebar.
- Exercise edits are shared (everyone sees the same lists).
"""
    )
    st.markdown("### Current backend")
    st.code(st.session_state.get("backend_label", "unknown"), language="text")
    st.markdown("### Known users in history file")
    users = sorted((st.session_state["history"].get("last_main_by_user", {}) or {}).keys())
    st.write(users if users else "(none yet)")