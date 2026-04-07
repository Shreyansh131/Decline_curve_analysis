import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
from scipy.optimize import curve_fit

st.set_page_config(page_title="DCA Dashboard", layout="wide")
st.title("📊 Decline Curve Analysis (DCA) Dashboard")

# ─────────────────────────────────────────
# Session state
# ─────────────────────────────────────────
for k, v in {
    "combined_df": pd.DataFrame(),
    "downloaded_wells": [],
    "dl_count": 0,
    "start_slider": 0,
    "f_start_slider": 0,
    "w_comments": [],
    "w_plateaus": [],
    "f_comments": [],
    "f_plateaus": [],
    "_comm_id_ctr": 0,
    "_plat_id_ctr": 0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Phase display config ──────────────────────────────────────────────────
PHASE_NAMES = [
    "Development Phase 1",
    "Development Phase 2 (Infill → P + I Drilling)",
    "Development Phase 3",
    "Development Phase 4",
    "Development Phase 5",
]
PHASE_FILL_COLORS   = ["rgba(0,150,255,0.08)",  "rgba(0,230,118,0.08)",
                       "rgba(255,180,0,0.08)",   "rgba(180,0,255,0.08)",  "rgba(255,80,80,0.08)"]
PHASE_FONT_COLORS   = ["#a0c8e6", "#a0e6b8", "#e6d4a0", "#d4a0e6", "#e6a0a0"]
PHASE_BORDER_COLORS = ["rgba(0,150,255,0.55)", "rgba(0,230,118,0.55)",
                       "rgba(255,180,0,0.55)",  "rgba(180,0,255,0.55)",  "rgba(255,80,80,0.55)"]

def _chart_config(filename: str) -> dict:
    """Plotly config with PNG download button."""
    return {
        "toImageButtonOptions": {
            "format":   "png",
            "filename": filename,
            "scale":    2,
            "width":    1600,
            "height":   800,
        },
        "displayModeBar": True,
        "displaylogo":    False,
        "modeBarButtonsToRemove": ["select2d", "lasso2d"],
    }

# ─────────────────────────────────────────
# Helpers — DCA math
# ─────────────────────────────────────────
@st.cache_data
def load_file(file):
    return pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)


@st.cache_data
def clean_data(df, date_col, rate_col):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[rate_col] = pd.to_numeric(df[rate_col], errors="coerce")
    return df.dropna(subset=[date_col, rate_col])


def rmse(a, b):
    return np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2))


def exponential(qi, Di, t):   return qi * np.exp(-Di * t)


def hyperbolic(qi, Di, b, t): return qi / ((1 + b * Di * t) ** (1 / b))


def harmonic(qi, Di, t):      return qi / (1 + Di * t)


def fit_dca(actual, t_pos, mask, model="Hyperbolic"):
    q_seg = actual[mask];
    t_seg = t_pos[mask]
    qi_guess = float(q_seg[0]) if len(q_seg) > 0 else 1.0
    try:
        if model == "Exponential":
            popt, _ = curve_fit(lambda t, qi, Di: exponential(qi, Di, t), t_seg, q_seg,
                                p0=[qi_guess, 0.3], bounds=([0, 1e-6], [qi_guess * 5, 5.0]), maxfev=5000)
            return popt[0], popt[1], 0.5
        elif model == "Hyperbolic":
            popt, _ = curve_fit(lambda t, qi, Di, b: hyperbolic(qi, Di, b, t), t_seg, q_seg,
                                p0=[qi_guess, 0.3, 0.5], bounds=([0, 1e-6, 0.01], [qi_guess * 5, 5.0, 1.0]),
                                maxfev=5000)
            return popt[0], popt[1], popt[2]
        elif model == "Harmonic":
            popt, _ = curve_fit(lambda t, qi, Di: harmonic(qi, Di, t), t_seg, q_seg,
                                p0=[qi_guess, 0.3], bounds=([0, 1e-6], [qi_guess * 5, 5.0]), maxfev=5000)
            return popt[0], popt[1], 0.5
    except Exception:
        return qi_guess, 0.3, 0.5

def auto_detect_phase_start(actual_vals, after_idx: int, window: int = 6) -> int:
    """Detect trough before next upswing after `after_idx`."""
    s = pd.Series(actual_vals)
    if len(s) < after_idx + 12:
        return min(after_idx + max(6, (len(s) - after_idx) // 2), len(s) - 2)
    smoothed = s.rolling(window=window, center=True, min_periods=1).mean()
    trend    = smoothed.diff(periods=12).fillna(0)
    is_inc   = pd.Series(False, index=s.index)
    is_inc.iloc[after_idx:] = trend.iloc[after_idx:] > 0
    streaks  = is_inc.ne(is_inc.shift()).cumsum()
    streak_lengths = is_inc.groupby(streaks).sum()
    if streak_lengths.max() < 4:
        return min(after_idx + max(6, (len(s) - after_idx) // 2), len(s) - 2)
    sid   = streak_lengths.idxmax()
    sidxs = is_inc[streaks == sid].index
    ss    = sidxs[0]
    s_s   = max(after_idx, ss - 18)
    s_e   = min(len(s), ss + 6)
    return int(smoothed.iloc[s_s:s_e].idxmin())


def make_dl_name(count):
    if count == 0: return "DCA.xlsx"
    if count == 1: return "DCA_updated.xlsx"
    return f"DCA_updated_{count}.xlsx"


def to_excel_bytes(df):
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df.to_excel(w, index=False)
    return out.getvalue()


# ─────────────────────────────────────────
# ★ Perforation / Squeeze History Engine
# ─────────────────────────────────────────
def _apply_squeeze(active: list, sq_s: float, sq_e: float) -> tuple:
    """
    Apply a squeeze over depth range [sq_s, sq_e] to the list of active intervals.
    Returns (new_active, removed_list) where removed_list holds the original
    interval dicts that were fully or partially closed.
    """
    new_active, removed = [], []
    for iv in active:
        iv_s, iv_e = iv["start"], iv["end"]
        if sq_e <= iv_s or sq_s >= iv_e:
            new_active.append(iv)
            continue
        if sq_s <= iv_s and sq_e >= iv_e:
            removed.append(iv)
            continue
        if sq_s <= iv_s < sq_e < iv_e:
            trimmed = dict(iv, start=sq_e, length=round(iv_e - sq_e, 3))
            new_active.append(trimmed)
            removed.append(iv)
            continue
        if iv_s < sq_s < iv_e <= sq_e:
            trimmed = dict(iv, end=sq_s, length=round(sq_s - iv_s, 3))
            new_active.append(trimmed)
            removed.append(iv)
            continue
        if iv_s < sq_s and sq_e < iv_e:
            left = dict(iv, end=sq_s, length=round(sq_s - iv_s, 3))
            right = dict(iv, start=sq_e, length=round(iv_e - sq_e, 3))
            new_active.extend([left, right])
            removed.append(iv)
            continue
        new_active.append(iv)
    return new_active, removed


def _intervals_overlap(s1, e1, s2, e2) -> bool:
    return not (e1 <= s2 or s1 >= e2)


def _classify_perforation(new_s: float, new_e: float, all_historical_intervals: list) -> str:
    if not all_historical_intervals:
        return "Initial"
    for iv in all_historical_intervals:
        iv_s, iv_e = iv.get("start"), iv.get("end")
        if iv_s is None or iv_e is None:
            continue
        if _intervals_overlap(new_s, new_e, iv_s, iv_e):
            return "Reperforation"
    return "Additional"


def analyze_perf_squeeze_history(
        well_events: pd.DataFrame,
        study_col: str,
        start_col,
        end_col,
        interval_col,
        date_col: str,
) -> tuple:
    active: list = []
    all_ever: list = []
    timeline: list = []
    prev_total = 0.0

    first_perf_year = None

    for _, row in well_events.sort_values(date_col).iterrows():
        study_raw = str(row.get(study_col, "")).strip()
        study_lc = study_raw.lower()

        is_perf = "perf" in study_lc
        is_squeeze = "squeeze" in study_lc or "sqz" in study_lc

        if not (is_perf or is_squeeze):
            continue

        date = row[date_col]
        if hasattr(date, 'year'):
            current_year = date.year
        else:
            current_year = pd.to_datetime(date).year

        def _safe_float(col):
            if col and col in row.index:
                v = row[col]
                return float(v) if pd.notna(v) else None
            return None

        s = _safe_float(start_col)
        e = _safe_float(end_col)
        L_given = _safe_float(interval_col)
        L = L_given if L_given is not None else (round(e - s, 3) if s is not None and e is not None else None)

        added_ivs: list = []
        removed_ivs: list = []
        perf_type = None

        if is_perf:
            if s is not None and e is not None:
                if first_perf_year is None:
                    first_perf_year = current_year

                if current_year == first_perf_year:
                    perf_type = "Initial"
                else:
                    perf_type = _classify_perforation(s, e, all_ever)

                new_iv = {
                    "start": s, "end": e, "length": L,
                    "added_date": date, "status": "OPEN",
                    "closed_date": None, "perf_type": perf_type,
                }
                active.append(new_iv)
                all_ever.append(new_iv)
                added_ivs = [new_iv]
        elif is_squeeze:
            if s is not None and e is not None:
                active, removed_ivs = _apply_squeeze(active, s, e)
                for riv in removed_ivs:
                    riv["status"] = "SQUEEZED"
                    riv["closed_date"] = date

        total_open = sum((iv["length"] or 0) for iv in active)
        delta = round(total_open - prev_total, 3)
        prev_total = total_open

        timeline.append({
            "date": date,
            "event": "PERFORATION" if is_perf else "SQUEEZE",
            "int_start": s,
            "int_end": e,
            "length": L,
            "study_raw": study_raw,
            "delta": delta,
            "snapshot": [dict(iv) for iv in active],
            "total_open": total_open,
            "added_ivs": added_ivs,
            "removed_ivs": removed_ivs,
            "perf_type": perf_type,
        })

    return timeline, active, all_ever


def _fmt_intervals(snapshot: list) -> str:
    if not snapshot:
        return "—  *(none)*"
    parts = []
    for iv in sorted(snapshot, key=lambda x: x.get("start") or 0):
        s, e, L = iv.get("start"), iv.get("end"), iv.get("length")
        if s is not None and e is not None:
            tag = f"[{s:.1f}–{e:.1f}"
            if L is not None:
                tag += f" · {L:.1f}m"
            tag += "]"
            parts.append(tag)
    return "  ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# ★ Date-grouped Timeline Renderer
# ─────────────────────────────────────────────────────────────────────────────
_PERF_TYPE_COLOR = {
    "Initial": {"bg": "rgba(0,100,255,0.12)", "border": "#4da6ff", "text": "#a8d4ff", "icon": "🔵"},
    "Additional": {"bg": "rgba(0,230,118,0.10)", "border": "#00e676", "text": "#b9ffd7", "icon": "🟢"},
    "Reperforation": {"bg": "rgba(255,235,59,0.10)", "border": "#ffeb3b", "text": "#fff9c4", "icon": "🟡"},
}
_SQZ_COLOR = {"bg": "rgba(255,152,0,0.10)", "border": "#ff9800", "text": "#ffe0b2", "icon": "🔴"}


def _render_grouped_timeline(timeline: list):
    if not timeline:
        st.info("ℹ️ No events to display.")
        return

    from collections import OrderedDict
    date_groups: "OrderedDict[str, list]" = OrderedDict()
    for ev in timeline:
        date_key = (
            ev["date"].strftime("%Y-%m-%d")
            if hasattr(ev["date"], "strftime") else str(ev["date"])
        )
        date_groups.setdefault(date_key, []).append(ev)

    for date_key, events in date_groups.items():
        perf_events = [e for e in events if e["event"] == "PERFORATION"]
        sqz_events = [e for e in events if e["event"] == "SQUEEZE"]

        has_perf = bool(perf_events)
        has_sqz = bool(sqz_events)
        if has_perf and has_sqz:
            hdr_bg = "rgba(160,80,200,0.13)"
            hdr_border = "#ce93d8"
            hdr_text = "#e1bee7"
            hdr_icon = "🔵🔴"
            hdr_label = "Perforation + Squeeze"
        elif has_perf:
            first_pt = perf_events[0].get("perf_type") or "Additional"
            c = _PERF_TYPE_COLOR.get(first_pt, _PERF_TYPE_COLOR["Additional"])
            hdr_bg = c["bg"];
            hdr_border = c["border"]
            hdr_text = c["text"];
            hdr_icon = c["icon"]
            types_present = list(dict.fromkeys(
                e.get("perf_type") or "Additional" for e in perf_events
            ))
            hdr_label = " + ".join(types_present) + " Perforation"
        else:
            c = _SQZ_COLOR
            hdr_bg = c["bg"];
            hdr_border = c["border"]
            hdr_text = c["text"];
            hdr_icon = c["icon"]
            hdr_label = "Cement Squeeze"

        try:
            display_date = pd.Timestamp(date_key).strftime("%d %b %Y")
        except Exception:
            display_date = date_key

        st.markdown(
            f"""
            <div style="
                background:{hdr_bg};
                border-left:4px solid {hdr_border};
                border-radius:7px;
                padding:9px 16px;
                margin:14px 0 6px 0;
            ">
                <span style="color:{hdr_border}; font-weight:700; font-size:0.95rem;">
                    {hdr_icon} {hdr_label}
                </span>
                <span style="color:#888; font-size:0.82rem; margin-left:10px;">
                    [ {display_date} ]
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if perf_events:
            type_groups: "OrderedDict[str, list]" = OrderedDict()
            for ev in perf_events:
                pt = ev.get("perf_type") or "Additional"
                type_groups.setdefault(pt, []).append(ev)

            for ptype, p_evs in type_groups.items():
                c = _PERF_TYPE_COLOR.get(ptype, _PERF_TYPE_COLOR["Additional"])

                if len(type_groups) > 1:
                    st.markdown(
                        f"<div style='margin:4px 0 2px 14px; color:{c['border']};"
                        f" font-size:0.80rem; font-weight:600;'>"
                        f"{c['icon']} {ptype} Perforation</div>",
                        unsafe_allow_html=True,
                    )

                rows = []
                for ev in p_evs:
                    depth_str = (
                        f"{ev['int_start']:.1f} – {ev['int_end']:.1f}"
                        if ev["int_start"] is not None and ev["int_end"] is not None else "—"
                    )
                    length_str = f"{ev['length']:.2f} m" if ev["length"] is not None else "—"
                    delta_str = (f"+{ev['delta']:.1f} m" if ev["delta"] >= 0
                                 else f"{ev['delta']:.1f} m")
                    open_str = f"{ev['total_open']:.1f} m  ({len(ev['snapshot'])} intervals)"
                    ev_date = ev["date"]
                    rows.append({
                        "Type": ptype,
                        "Date": (ev_date.strftime("%d %b %Y")
                                 if hasattr(ev_date, "strftime") else str(ev_date)),
                        "Depth Range (m)": depth_str,
                        "Length": length_str,
                        "Δ Open": delta_str,
                        "Total Open": open_str,
                    })

                phase_df = pd.DataFrame(rows)
                _bg = c["bg"]
                _text = c["text"]

                def _style_perf(row, _bg=_bg, _text=_text):
                    s = f"background-color:{_bg}; color:{_text}; font-size:0.82rem;"
                    return [s] * len(row)

                st.dataframe(
                    phase_df.style.apply(_style_perf, axis=1),
                    use_container_width=True,
                    hide_index=True,
                )

        if sqz_events:
            if has_perf:
                st.markdown(
                    f"<div style='margin:6px 0 2px 14px; color:{_SQZ_COLOR['border']};"
                    f" font-size:0.80rem; font-weight:600;'>"
                    f"{_SQZ_COLOR['icon']} Cement Squeeze</div>",
                    unsafe_allow_html=True,
                )

            rows = []
            for ev in sqz_events:
                depth_str = (
                    f"{ev['int_start']:.1f} – {ev['int_end']:.1f}"
                    if ev["int_start"] is not None and ev["int_end"] is not None else "—"
                )
                length_str = f"{ev['length']:.2f} m" if ev["length"] is not None else "—"
                delta_str = f"{ev['delta']:.1f} m"
                open_str = f"{ev['total_open']:.1f} m  ({len(ev['snapshot'])} intervals)"
                ev_date = ev["date"]
                rows.append({
                    "Date": (ev_date.strftime("%d %b %Y")
                             if hasattr(ev_date, "strftime") else str(ev_date)),
                    "Depth Range (m)": depth_str,
                    "Length": length_str,
                    "Δ Open": delta_str,
                    "Total Open": open_str,
                })

            sqz_df = pd.DataFrame(rows)

            def _style_sqz(row):
                s = (f"background-color:{_SQZ_COLOR['bg']};"
                     f" color:{_SQZ_COLOR['text']}; font-size:0.82rem;")
                return [s] * len(row)

            st.dataframe(
                sqz_df.style.apply(_style_sqz, axis=1),
                use_container_width=True,
                hide_index=True,
            )

    st.caption(
        "🔵 Blue = Initial Perforation  |  "
        "🟢 Green = Additional Perforation  |  "
        "🟡 Yellow = Reperforation  |  "
        "🔴 Red = Squeeze / Cement job"
    )

# ─────────────────────────────────────────────────────────────────────────────
# ★ Continuous Interval Timeline (Gantt-style depth vs time)
# ─────────────────────────────────────────────────────────────────────────────
_TYPE_GANTT_COLORS = {
    "Initial": "rgba(77,166,255,0.80)",
    "Additional": "rgba(0,230,118,0.80)",
    "Reperforation": "rgba(255,235,59,0.85)",
    "Squeezed": "rgba(255,152,0,0.70)",
}


def _render_continuous_interval_timeline(timeline: list, well_name: str, prod_dates=None):
    if not timeline:
        st.info("No timeline data available.")
        return

    last_date = max(ev["date"] for ev in timeline)
    if prod_dates is not None and len(prod_dates) > 0:
        pd_max = pd.Timestamp(prod_dates.max())
        if pd_max > pd.Timestamp(last_date):
            last_date = pd_max

    class _Slot:
        _id_ctr = 0

        def __init__(self, s, e, L, open_date, perf_type):
            _Slot._id_ctr += 1
            self.id = _Slot._id_ctr
            self.start = s
            self.end = e
            self.length = L
            self.open_date = open_date
            self.close_date = None
            self.perf_type = perf_type

        def overlaps(self, sq_s, sq_e):
            return not (self.end <= sq_s or self.start >= sq_e)

    active_slots: list = []
    closed_slots: list = []

    for ev in timeline:
        date = ev["date"]

        if ev["event"] == "PERFORATION":
            if ev["int_start"] is not None and ev["int_end"] is not None:
                L = ev["length"] or round(ev["int_end"] - ev["int_start"], 3)
                pt = ev.get("perf_type") or "Additional"
                slot = _Slot(ev["int_start"], ev["int_end"], L, date, pt)
                active_slots.append(slot)

        elif ev["event"] == "SQUEEZE":
            if ev["int_start"] is not None and ev["int_end"] is not None:
                sq_s, sq_e = ev["int_start"], ev["int_end"]
                new_active = []
                for slot in active_slots:
                    if not slot.overlaps(sq_s, sq_e):
                        new_active.append(slot)
                        continue
                    slot.close_date = date
                    closed_slots.append(slot)

                    iv_s, iv_e = slot.start, slot.end
                    if iv_s < sq_s:
                        left = _Slot(iv_s, sq_s, round(sq_s - iv_s, 3), slot.open_date, slot.perf_type)
                        left.open_date = slot.open_date
                        new_active.append(left)
                    if sq_e < iv_e:
                        right = _Slot(sq_e, iv_e, round(iv_e - sq_e, 3), slot.open_date, slot.perf_type)
                        right.open_date = slot.open_date
                        new_active.append(right)

                active_slots = new_active

    all_segments = closed_slots + active_slots

    if not all_segments:
        st.info("No interval segments to plot.")
        return

    fig = go.Figure()
    legend_shown = set()

    for slot in sorted(all_segments, key=lambda x: x.start):
        t_start = slot.open_date
        t_end = slot.close_date if slot.close_date is not None else last_date
        depth_lo = slot.start
        depth_hi = slot.end
        ptype = slot.perf_type

        color = _TYPE_GANTT_COLORS.get(ptype, "rgba(150,150,150,0.7)")
        seg_name = f"{ptype} Perforation"
        show_leg = seg_name not in legend_shown
        if show_leg:
            legend_shown.add(seg_name)

        is_open = slot.close_date is None
        hover = (
                f"<b>{ptype} Perforation</b><br>"
                f"Depth: {depth_lo:.1f} – {depth_hi:.1f} m<br>"
                f"Length: {slot.length:.2f} m<br>"
                f"Opened: {t_start.strftime('%d %b %Y') if hasattr(t_start, 'strftime') else t_start}<br>"
                + (f"Squeezed: {t_end.strftime('%d %b %Y') if hasattr(t_end, 'strftime') else t_end}"
                   if not is_open
                   else f"Still open as of {pd.Timestamp(t_end).strftime('%d %b %Y') if hasattr(t_end, 'strftime') else t_end}")
        )

        fig.add_trace(go.Scatter(
            x=[t_start, t_end, t_end, t_start, t_start],
            y=[depth_lo, depth_lo, depth_hi, depth_hi, depth_lo],
            fill="toself",
            fillcolor=color,
            line=dict(color=color.replace("0.80", "1.0").replace("0.85", "1.0").replace("0.70", "1.0"), width=1),
            mode="lines",
            name=seg_name,
            showlegend=show_leg,
            hoverinfo="text",
            hovertext=hover,
            legendgroup=seg_name,
        ))

    for ev in timeline:
        if ev["event"] == "SQUEEZE" and ev["int_start"] is not None:
            ev_date = ev["date"]
            fig.add_shape(
                type="line",
                x0=ev_date, x1=ev_date,
                y0=ev["int_start"], y1=ev["int_end"],
                line=dict(color="#ff9800", width=3, dash="dot"),
                layer="above",
            )

    fig.update_layout(
        xaxis=dict(title=dict(text="Date", font=dict(size=14)), showgrid=False),
        yaxis=dict(
            title=dict(text="Depth (m)", font=dict(size=14)),
            autorange="reversed",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
        ),
        height=500,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=90, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.0,
            xanchor="left", x=0,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        title=dict(
            text="Perforation Interval Continuity (Depth vs Time)",
            font=dict(size=13, color="white"),
            x=0, xanchor="left",
            yanchor="top", y=1.0,
            pad=dict(b=36),
        ),
    )

    st.plotly_chart(fig, use_container_width=True, config=_chart_config(f"perf_continuity_{well_name}"))
    st.caption(
        "🔵 Blue = Initial Perforation  |  🟢 Green = Additional Perforation  |  🟡 Yellow = Reperforation  "
        "|  🟠 Orange dashed line = Squeeze event"
    )

def render_perf_history_section(
        timeline: list,
        final_active: list,
        all_ever: list,
        well_name: str,
        prod_dates,
):
    if not timeline:
        st.info("ℹ️ No events to display.")
        return

    n_perf = sum(1 for e in timeline if e["event"] == "PERFORATION")
    n_sqz = sum(1 for e in timeline if e["event"] == "SQUEEZE")
    tot_open = final_active and sum((iv.get("length") or 0) for iv in final_active) or 0.0

    st.markdown(
        f"""
        <div style="
            display:flex; gap:18px; flex-wrap:wrap;
            background:rgba(255,255,255,0.04);
            border:1px solid rgba(255,255,255,0.1);
            border-radius:10px; padding:12px 18px; margin-bottom:10px;
        ">
          <div style="text-align:center">
            <div style="font-size:1.5rem;font-weight:700;color:#00e676">{n_perf}</div>
            <div style="font-size:0.72rem;color:#aaa">PERFORATION<br>EVENTS</div>
          </div>
          <div style="text-align:center">
            <div style="font-size:1.5rem;font-weight:700;color:#ff9800">{n_sqz}</div>
            <div style="font-size:0.72rem;color:#aaa">SQUEEZE<br>EVENTS</div>
          </div>
          <div style="text-align:center">
            <div style="font-size:1.5rem;font-weight:700;color:#fff">{len(final_active)}</div>
            <div style="font-size:0.72rem;color:#aaa">OPEN<br>INTERVALS</div>
          </div>
          <div style="text-align:center">
            <div style="font-size:1.5rem;font-weight:700;color:#29b6f6">{tot_open:.1f} m</div>
            <div style="font-size:0.72rem;color:#aaa">TOTAL OPEN<br>PERFORATION</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_grouped_timeline(timeline)
    st.markdown("##### 📈 Perforation Interval Continuity")
    _render_continuous_interval_timeline(timeline, well_name, prod_dates)

def _build_chart_event_map(timeline: list) -> list:
    result = []
    for ev in timeline:
        if ev["event"] == "PERFORATION":
            result.append({
                "date": ev["date"],
                "event_type": ev.get("perf_type") or "Additional",
                "int_start": ev["int_start"],
                "int_end": ev["int_end"],
                "length": ev["length"],
            })
        else:
            result.append({
                "date": ev["date"],
                "event_type": "Squeeze",
                "int_start": ev["int_start"],
                "int_end": ev["int_end"],
                "length": ev["length"],
            })
    return result

# ─────────────────────────────────────────
# Comment & Plateau UI helpers
# ─────────────────────────────────────────
def _apply_comments_to_fig(fig, comments: list):
    for c in comments:
        fig.add_annotation(x=c["date"], y=c["rate"], text=f"<b>{c['text']}</b>",
            font=dict(size=13, color=c.get("color", "white")), bgcolor="rgba(30,30,30,0.85)",
            bordercolor=c.get("color", "white"), borderwidth=1, borderpad=4,
            xanchor="left", yanchor="bottom", ax=c.get("ax", 50), ay=c.get("ay", -45),
            arrowhead=2, arrowwidth=1.5, arrowcolor=c.get("color", "white"), showarrow=True)

def _apply_plateaus_to_fig(fig, plateaus: list):
    for p in plateaus:
        fig.add_shape(type="line", x0=p["start"], x1=p["end"], y0=p["rate"], y1=p["rate"],
            line=dict(color=p["color"], width=p.get("width", 3), dash="solid"), layer="above")
        fig.add_annotation(x=p["end"], y=p["rate"], text=f"<b>Plateau {p['rate']:,.0f}</b>",
            font=dict(size=11, color=p["color"]), bgcolor="rgba(0,0,0,0.6)",
            bordercolor=p["color"], borderwidth=1, showarrow=False, xanchor="left", yanchor="middle")

def _comment_editor_ui(comments_key, date_labels, actual_vals):
    with st.expander("💬 Add / Manage Annotations on Chart", expanded=False):
        st.markdown("**Add new annotation:**")
        cc1, cc2, cc3, cc4 = st.columns([2, 2, 3, 2])
        c_date = cc1.selectbox("Date", date_labels, key=f"{comments_key}_c_date")
        c_rate = cc2.number_input("Y (Rate) position", value=float(np.mean(actual_vals)), key=f"{comments_key}_c_rate")
        c_text = cc3.text_input("Comment text", key=f"{comments_key}_c_text")
        c_color = cc4.color_picker("Color", "#ffffff", key=f"{comments_key}_c_color")
        ca1, ca2 = st.columns(2)
        c_ax = ca1.slider("Arrow X offset", -250, 250, 60, key=f"{comments_key}_c_ax")
        c_ay = ca2.slider("Arrow Y offset", -250, 250, -45, key=f"{comments_key}_c_ay")
        if st.button("➕ Add Annotation", key=f"{comments_key}_add"):
            st.session_state["_comm_id_ctr"] += 1
            st.session_state[comments_key].append({"id": st.session_state["_comm_id_ctr"], "date": c_date, "rate": c_rate, "text": c_text, "color": c_color, "ax": c_ax, "ay": c_ay})
            st.rerun()
        if st.session_state[comments_key]:
            st.markdown("**Existing annotations:**")
            for cm in list(st.session_state[comments_key]):
                ec1, ec2, ec3 = st.columns([5, 1, 1])
                ec1.markdown(f"📌 **{cm['date']}** — {cm['text']}")
                new_ax = ec2.number_input("ax", value=cm["ax"], key=f"{comments_key}_eax_{cm['id']}", step=5, label_visibility="collapsed")
                new_ay = ec2.number_input("ay", value=cm["ay"], key=f"{comments_key}_eay_{cm['id']}", step=5, label_visibility="collapsed")
                cm["ax"] = new_ax; cm["ay"] = new_ay
                if ec3.button("🗑️", key=f"{comments_key}_del_{cm['id']}"):
                    st.session_state[comments_key] = [x for x in st.session_state[comments_key] if x["id"] != cm["id"]]
                    st.rerun()

def _plateau_editor_ui(plateaus_key, date_labels, actual_vals):
    with st.expander("📏 Add / Manage Plateau (Constant Rate) Lines", expanded=False):
        st.markdown("**Add new plateau line:**")
        pc1, pc2, pc3 = st.columns(3)
        p_start = pc1.selectbox("Start date", date_labels, key=f"{plateaus_key}_p_start")
        p_end = pc2.selectbox("End date", date_labels, index=min(len(date_labels)-1, 12), key=f"{plateaus_key}_p_end")
        p_rate = pc3.number_input("Rate (Y value)", value=float(np.max(actual_vals)*0.8), key=f"{plateaus_key}_p_rate")
        pp1, pp2, pp3 = st.columns(3)
        p_color = pp1.color_picker("Line color", "#ffff00", key=f"{plateaus_key}_p_color")
        p_width = pp2.slider("Line width", 1, 8, 3, key=f"{plateaus_key}_p_width")
        if pp3.button("➕ Add Plateau", key=f"{plateaus_key}_add"):
            st.session_state["_plat_id_ctr"] += 1
            st.session_state[plateaus_key].append({"id": st.session_state["_plat_id_ctr"], "start": p_start, "end": p_end, "rate": p_rate, "color": p_color, "width": p_width})
            st.rerun()
        if st.session_state[plateaus_key]:
            st.markdown("**Existing plateaus:**")
            for pl in list(st.session_state[plateaus_key]):
                ep1, ep2 = st.columns([5, 1])
                ep1.markdown(f"📏 **{pl['start']}** → **{pl['end']}** rate={pl['rate']:,.0f}")
                if ep2.button("🗑️", key=f"{plateaus_key}_del_{pl['id']}"):
                    st.session_state[plateaus_key] = [x for x in st.session_state[plateaus_key] if x["id"] != pl["id"]]
                    st.rerun()


# ─────────────────────────────────────────
# File upload
# ─────────────────────────────────────────
file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
if file is None:
    st.stop()

df = load_file(file)
st.subheader("🔍 Data Preview")
st.dataframe(df.head())

# ─────────────────────────────────────────
# Perforation / Squeeze data upload
# ─────────────────────────────────────────
st.markdown("---")
perf_file = st.file_uploader(
    "📂 Upload Perforation / Squeeze Data  (optional — separate sheet)",
    type=["csv", "xlsx"],
    key="perf_upload",
)
perf_df_raw = None
if perf_file is not None:
    perf_df_raw = load_file(perf_file)
    with st.expander("🔍 Perforation Data Preview", expanded=False):
        st.dataframe(perf_df_raw.head(10))
st.markdown("---")

# ─────────────────────────────────────────
# Sidebar — Column Selection
# ─────────────────────────────────────────
sb = st.sidebar
sb.header("⚙️ Column Selection")
well_col = sb.selectbox("Well ID Column", df.columns)
date_col = sb.selectbox("Date Column", df.columns)
rate_col = sb.selectbox("Rate Column", df.columns)

sb.markdown("---")
sb.header("🔧 Filter Columns (Optional)")

_none = "— None —"
_col_opts = [_none] + list(df.columns)

status_col_raw = sb.selectbox("Well Status Column  (Live / Dead)", _col_opts,
                              help="Column whose values distinguish live wells from dead ones.")
status_col = None if status_col_raw == _none else status_col_raw

type_col_raw = sb.selectbox("Well Type Column  (Production / Injection)", _col_opts,
                            help="Column whose values distinguish producers from injectors.")
type_col = None if type_col_raw == _none else type_col_raw

perf_col_raw = sb.selectbox("🔩 Perforation Column (optional)", _col_opts,
                            help="Numeric column whose values will be overlaid on the chart as a secondary axis.")
perf_col = None if perf_col_raw == _none else perf_col_raw

# ── Perforation sheet column mapping ─────────────────────────
if perf_df_raw is not None:
    sb.markdown("---")
    sb.header("🔩 Perf Sheet Columns")
    _p_cols = list(perf_df_raw.columns)
    _p_none_opt = [_none] + _p_cols
    p_well_col = sb.selectbox("Well ID Column", _p_cols, key="p_well_col")
    p_date_col_p = sb.selectbox("Date Column", _p_cols, key="p_date_col")
    p_study_col = sb.selectbox("Study/Type Column", _p_cols, key="p_study_col")
    p_start_col_r = sb.selectbox("Start Depth Column", _p_none_opt, key="p_start_col")
    p_end_col_r = sb.selectbox("End Depth Column", _p_none_opt, key="p_end_col")
    p_interval_r = sb.selectbox("Interval Column", _p_none_opt, key="p_interval_col")
    p_start_col = None if p_start_col_r == _none else p_start_col_r
    p_end_col = None if p_end_col_r == _none else p_end_col_r
    p_interval_col_p = None if p_interval_r == _none else p_interval_r
else:
    p_well_col = p_date_col_p = p_study_col = None
    p_start_col = p_end_col = p_interval_col_p = None

sb.markdown("---")
sb.header("🗺️ Field Selection")
field_col_options = [_none] + list(df.columns)
field_col_raw = sb.selectbox("Field Column (optional)", field_col_options,
                             help="Select the column that identifies which field a well belongs to.")
field_col = None if field_col_raw == _none else field_col_raw

view_mode = "Well View"
if field_col:
    view_mode = sb.radio("🔭 View Mode", ["Well View", "Field View"], horizontal=True)

df = clean_data(df, date_col, rate_col)

# ══════════════════════════════════════════════════════════════
#  FIELD VIEW
# ══════════════════════════════════════════════════════════════
if view_mode == "Field View" and field_col:

    sb.markdown("---")
    sb.header("🛢️ Field Picker")
    available_fields = sorted(df[field_col].dropna().unique().tolist())
    selected_field = sb.selectbox("Select Field", available_fields)

    field_base = df[df[field_col] == selected_field]

    if status_col:
        status_vals = sorted(field_base[status_col].dropna().unique().tolist())
        status_filter = sb.selectbox("🟢 Well Status", ["All"] + status_vals, key="f_status_filter")
        if status_filter != "All":
            field_base = field_base[field_base[status_col] == status_filter]

    if type_col:
        type_vals = sorted(field_base[type_col].dropna().unique().tolist())
        type_filter = sb.selectbox("🔵 Well Type", ["All"] + type_vals, key="f_type_filter")
        if type_filter != "All":
            field_base = field_base[field_base[type_col] == type_filter]

    _agg_cols = {rate_col: "sum"}
    if perf_col:
        field_base_num = field_base.copy()
        field_base_num[perf_col] = pd.to_numeric(field_base_num[perf_col], errors="coerce")
        _agg_cols[perf_col] = "sum"
    else:
        field_base_num = field_base

    field_df = (
        field_base_num
        .groupby(date_col, as_index=False).agg(_agg_cols)
        .sort_values(date_col).reset_index(drop=True)
    )

    if field_df.empty:
        st.warning("No data available for the selected field / status / type combination.")
        st.stop()

    f_actual = field_df[rate_col].values
    f_dates = field_df[date_col]
    f_n = len(field_df)
    f_peak_idx = int(np.argmax(f_actual))
    wells_in_field = sorted(field_base[well_col].dropna().unique().tolist())

    if ("last_field" not in st.session_state or st.session_state.last_field != selected_field):
        st.session_state.f_start_slider = 0
        st.session_state.last_field = selected_field

    sb.header("📍 DCA Start Point")
    f_start_mode = sb.radio("Mode:", ["Slider + Peak Finder", "Exact Date Picker", "Manual Input"],
                            horizontal=False, key="f_start_mode")

    if f_start_mode == "Slider + Peak Finder":
        if sb.button("📈 Jump to Peak Rate", key="f_peak_btn"):
            st.session_state.f_start_slider = f_peak_idx
        f_start_idx = sb.slider("Drag to select start index", min_value=0, max_value=f_n - 2, key="f_start_slider")
        lo, hi = max(0, f_start_idx - 2), min(f_n, f_start_idx + 3)
        nearby = field_df[[date_col, rate_col]].iloc[lo:hi].copy()
        nearby[date_col] = nearby[date_col].dt.strftime("%Y-%m-%d")
        nearby.index = range(lo, hi);
        nearby.index.name = "idx"
        sb.caption("📋 Points around current selection:");
        sb.dataframe(nearby, use_container_width=True)
    elif f_start_mode == "Exact Date Picker":
        f_date_labels = f_dates.dt.strftime("%Y-%m-%d").tolist()
        f_picked_label = sb.selectbox("Pick exact date", f_date_labels,
                                      index=st.session_state.f_start_slider, key="f_date_pick")
        f_start_idx = f_date_labels.index(f_picked_label)
        st.session_state.f_start_slider = f_start_idx
    else:
        f_manual = sb.text_input("Date (YYYY-MM-DD)", value=str(f_dates.iloc[0].date()), key="f_manual_date")
        try:
            f_ts = pd.Timestamp(f_manual)
            f_start_idx = int((f_dates - f_ts).abs().argmin())
            st.session_state.f_start_slider = f_start_idx
        except Exception:
            sb.error("Invalid date — use YYYY-MM-DD");
            f_start_idx = 0

    sb.info(f"**Date:** {f_dates.iloc[f_start_idx].date()}  \n"
            f"**Field rate:** {f_actual[f_start_idx]:.1f}  \n"
            f"**Peak at:** {f_dates.iloc[f_peak_idx].date()} ({f_actual[f_peak_idx]:.1f})")

    sb.header("📉 DCA Parameters")
    sb.header("📌 Show Models")
    f_show_exp = sb.checkbox("Exponential", True, key="f_exp")
    f_show_hyp = sb.checkbox("Hyperbolic", True, key="f_hyp")
    f_show_har = sb.checkbox("Harmonic", True, key="f_har")
    f_qi = sb.number_input("Initial Rate (qi)", value=round(float(f_actual[f_start_idx]), 2), key=f"f_qi_{f_start_idx}")
    f_Di = sb.slider("Decline Rate (Di, /year)", 0.001, 2.0, 0.3, key="f_Di")
    f_b = sb.slider("Decline Exponent (b)", 0.01, 1.0, 0.5, key="f_b")

    # ── Multi-Phase Development Tool (Field View) ──
    sb.markdown("---")
    sb.header("🚧 Multi-Phase Development")
    f_date_labels_all = f_dates.dt.strftime("%Y-%m-%d").tolist()
    f_n_phases = int(sb.number_input("Number of Development Phases", 1, 5, 1, key="f_n_phases"))
    f_phase_boundaries = []
    if f_n_phases > 1:
        for _pi in range(f_n_phases - 1):
            _after = f_peak_idx if _pi == 0 else (f_date_labels_all.index(str(f_phase_boundaries[-1].date())) if f_phase_boundaries else f_peak_idx)
            _auto_pi = auto_detect_phase_start(f_actual, _after)
            _pp = sb.selectbox(f"Phase {_pi + 2} Start Date", f_date_labels_all, index=_auto_pi, key=f"f_pb_{_pi}")
            f_phase_boundaries.append(pd.Timestamp(_pp))

    f_t_days = (f_dates - f_dates.iloc[f_start_idx]).dt.days.values
    f_t_pos = np.where(f_t_days >= 0, f_t_days / 365.25, 0)
    f_mask = f_t_days >= 0

    f_exp_c = exponential(f_qi, f_Di, f_t_pos)
    f_hyp_c = hyperbolic(f_qi, f_Di, f_b, f_t_pos)
    f_har_c = harmonic(f_qi, f_Di, f_t_pos)

    _filter_tags = []
    if status_col and "f_status_filter" in st.session_state and st.session_state.f_status_filter != "All":
        _filter_tags.append(f"Status: {st.session_state.f_status_filter}")
    if type_col and "f_type_filter" in st.session_state and st.session_state.f_type_filter != "All":
        _filter_tags.append(f"Type: {st.session_state.f_type_filter}")
    _filter_str = f"  •  🔍 {' | '.join(_filter_tags)}" if _filter_tags else ""

    st.subheader(f"🗺️ Field: {selected_field}  —  {len(wells_in_field)} well(s){_filter_str}")
    with st.expander(f"Wells in this field ({len(wells_in_field)})", expanded=False):
        st.write(", ".join(str(w) for w in wells_in_field))

    st.subheader("📈 Field Production + Decline Curves")
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=f_dates, y=f_actual, mode="lines+markers", name="Field Actual",
                               line=dict(color="#00c8ff", width=3), marker=dict(size=5, opacity=0.6),
                               fill="tozeroy", fillcolor="rgba(0,200,255,0.07)",
                               hovertemplate="<b>Actual</b> %{x|%Y-%m-%d}: %{y:,.1f}<extra></extra>"))
    f_d2 = f_dates[f_mask]
    if f_show_exp: fig_f.add_trace(go.Scatter(x=f_d2, y=f_exp_c[f_mask], mode="lines", name="Exponential",
                                              line=dict(color="orange", width=2, dash="dash")))
    if f_show_hyp: fig_f.add_trace(go.Scatter(x=f_d2, y=f_hyp_c[f_mask], mode="lines", name="Hyperbolic",
                                              line=dict(color="limegreen", width=2, dash="dash")))
    if f_show_har: fig_f.add_trace(go.Scatter(x=f_d2, y=f_har_c[f_mask], mode="lines", name="Harmonic",
                                              line=dict(color="red", width=2, dash="dash")))

    # Peak Marker
    fig_f.add_trace(go.Scatter(
        x=[f_dates.iloc[f_peak_idx]], y=[f_actual[f_peak_idx]],
        mode="markers", name="Peak",
        marker=dict(size=13, color="white", symbol="triangle-up", line=dict(color="gray", width=1)),
        showlegend=True,
        hovertemplate=f"<b>Peak</b><br>Date: {f_dates.iloc[f_peak_idx].strftime('%b %Y')}<br>Rate: {f_actual[f_peak_idx]:,.0f}<extra></extra>"
    ))

    f_y_min, f_y_max = float(np.nanmin(f_actual)), float(np.nanmax(f_actual))

    # DCA Start vertical line
    fig_f.add_trace(go.Scatter(
        x=[f_dates.iloc[f_start_idx], f_dates.iloc[f_start_idx]], y=[f_y_min, f_y_max],
        mode="lines", name="DCA Start", line=dict(color="yellow", width=2, dash="dot")
    ))

    # DCA Start star marker
    fig_f.add_trace(go.Scatter(
        x=[f_dates.iloc[f_start_idx]], y=[f_qi],
        mode="markers", name="Start ★",
        marker=dict(size=14, color="yellow", symbol="star"),
        showlegend=True,
        hovertemplate=f"<b>DCA Start</b><br>Date: {f_dates.iloc[f_start_idx].strftime('%b %Y')}<br>qi = {f_qi:,.0f}<extra></extra>"
    ))

    # Arrow annotations
    _f_peak_date = f_dates.iloc[f_peak_idx]
    _f_peak_val = f_actual[f_peak_idx]

    fig_f.add_annotation(
        x=_f_peak_date, y=_f_peak_val,
        text=f"<b>Peak<br>{_f_peak_val:,.0f}</b>",
        font=dict(size=15, color="white"),
        bgcolor="rgba(40,40,40,0.80)",
        bordercolor="white", borderwidth=1,
        xanchor="left", yanchor="bottom",
        ax=50, ay=-55,
        arrowhead=2, arrowwidth=1.5, arrowcolor="white", showarrow=True,
    )

    _f_start_date = f_dates.iloc[f_start_idx]
    fig_f.add_annotation(
        x=_f_start_date, y=f_qi,
        text=f"<b>DCA Start<br>{_f_start_date.strftime('%b %Y')}<br>qi = {f_qi:,.0f}</b>",
        font=dict(size=15, color="yellow"),
        bgcolor="rgba(40,40,40,0.80)",
        bordercolor="yellow", borderwidth=1,
        xanchor="left", yanchor="middle",
        ax=65, ay=0,
        arrowhead=2, arrowwidth=1.5, arrowcolor="yellow", showarrow=True,
    )

    # ── Multi-Phase Chart Annotations ──
    _f_phase_dates = [f_dates.iloc[0]] + f_phase_boundaries + [f_dates.iloc[-1]]
    for _pi in range(f_n_phases):
        _px0 = _f_phase_dates[_pi]
        _px1 = _f_phase_dates[_pi+1]
        _pname = PHASE_NAMES[_pi] if _pi < len(PHASE_NAMES) else f"Phase {_pi+1}"
        _pfc = PHASE_FILL_COLORS[_pi % len(PHASE_FILL_COLORS)]
        _pbc = PHASE_BORDER_COLORS[_pi % len(PHASE_BORDER_COLORS)]
        _pfont = PHASE_FONT_COLORS[_pi % len(PHASE_FONT_COLORS)]

        fig_f.add_vrect(x0=_px0, x1=_px1, fillcolor=_pfc, layer="below", line_width=0,
                        annotation_text=f"<b>{_pname}</b>", annotation_position="top left",
                        annotation_font_size=14, annotation_font_color=_pfont)
        if _pi > 0:
            _bdate = _f_phase_dates[_pi]
            _near_idx = (f_dates - _bdate).abs().argmin()
            _b_rate = f_actual[_near_idx]
            fig_f.add_vline(x=_bdate, line_width=2, line_dash="dash", line_color=_pbc)
            fig_f.add_annotation(x=_bdate, y=_b_rate, text=f"<b>{_pname}<br>Rate: {_b_rate:,.0f}</b>",
                                 font=dict(size=13, color=_pfont), bgcolor="rgba(20,20,20,0.85)", bordercolor=_pfont,
                                 borderwidth=1, xanchor="left", yanchor="middle", ax=50, ay=30,
                                 arrowhead=2, arrowwidth=1.5, arrowcolor=_pfont, showarrow=True)
            # Highlighted bottom box
            fig_f.add_annotation(x=_bdate, y=f_y_min + (_pi - 1) * (f_y_max - f_y_min) * 0.05,
                                 text=f"<b>▲ {_pname} starts here  |  Rate: {_b_rate:,.0f}</b>",
                                 font=dict(size=12, color=_pfont), bgcolor=_pbc.replace("0.55", "0.30"),
                                 bordercolor=_pfont, borderwidth=2, xanchor="left", yanchor="top", showarrow=False, yref="y")

    _apply_plateaus_to_fig(fig_f, st.session_state["f_plateaus"])
    _apply_comments_to_fig(fig_f, st.session_state["f_comments"])

    fig_f.update_layout(
        title=dict(text=f"🗺️ Field: <b>{selected_field}</b>", font=dict(size=22, color="white"), x=0, xanchor="left"),
        xaxis=dict(title=dict(text="Date", font=dict(size=18)), tickfont=dict(size=16)),
        yaxis=dict(title=dict(text=f"Total Field Rate ({rate_col})", font=dict(size=18)), tickfont=dict(size=16)),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=15),
        ),
        margin=dict(l=60, r=30, t=80, b=60),
        height=640, template="plotly_dark", hovermode="x unified",
    )
    st.plotly_chart(fig_f, use_container_width=True, config=_chart_config(f"field_{selected_field}"))

    _comment_editor_ui("f_comments", f_date_labels_all, f_actual)
    _plateau_editor_ui("f_plateaus", f_date_labels_all, f_actual)

    st.subheader("📊 Model Performance (RMSE) — Field")
    f_results = []
    f_seg = f_actual[f_mask]
    if f_show_exp: f_results.append(("Exponential", rmse(f_seg, f_exp_c[f_mask])))
    if f_show_hyp: f_results.append(("Hyperbolic", rmse(f_seg, f_hyp_c[f_mask])))
    if f_show_har: f_results.append(("Harmonic", rmse(f_seg, f_har_c[f_mask])))
    f_best_model = None
    if f_results:
        f_mdf = pd.DataFrame(f_results, columns=["Model", "RMSE"])
        st.dataframe(f_mdf)
        f_best_model = f_mdf.loc[f_mdf.RMSE.idxmin(), "Model"]
        st.success(f"🏆 Best Fit: {f_best_model}")

    st.subheader("📊 Field Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Peak Rate", f"{f_actual.max():,.1f}")
    c2.metric("Avg Rate", f"{f_actual.mean():,.1f}")
    c3.metric("Total Volume", f"{f_actual.sum():,.1f}")
    c4.metric("Data Points", f_n)

    st.subheader("⬇️ Download Field DCA Data")
    f_curves_map = {}
    if f_show_exp: f_curves_map["Exponential"] = f_exp_c
    if f_show_hyp: f_curves_map["Hyperbolic"] = f_hyp_c
    if f_show_har: f_curves_map["Harmonic"] = f_har_c
    if not f_curves_map:
        st.warning("Select at least one model above.")
        st.stop()
    f_curve_options = list(f_curves_map.keys())
    f_recommended_idx = f_curve_options.index(f_best_model) if f_best_model and f_best_model in f_curve_options else 0
    if f_best_model and f_best_model in f_curve_options:
        st.info(f"💡 **Recommended:** {f_best_model} (lowest RMSE — pre-selected below)")
    f_chosen_curve = st.radio("**Step 1 — Select curve to download:**", f_curve_options,
                              index=f_recommended_idx, horizontal=True, key="f_dl_curve")
    f_out = field_df[[date_col, rate_col]].copy()
    f_out.insert(0, "Field", selected_field)
    f_out["DCA"] = np.where(f_mask, np.round(f_curves_map[f_chosen_curve], 4), np.nan)
    f_out["Type"] = np.where(f_mask, f_chosen_curve, "")
    with st.expander("Preview data to be downloaded"):
        st.dataframe(f_out.head(20))
    st.download_button(label=f"⬇️ Download {selected_field} — {f_chosen_curve} DCA",
                       data=to_excel_bytes(f_out), file_name=f"{selected_field}_{f_chosen_curve}_DCA.xlsx")
    st.stop()

# ══════════════════════════════════════════════════════════════
#  WELL VIEW
# ══════════════════════════════════════════════════════════════
sb.markdown("---")
sb.header("🛢️ Well Selection")

well_pool = df.copy()
if status_col:
    status_vals_w = sorted(well_pool[status_col].dropna().unique().tolist())
    status_filter_w = sb.selectbox("🟢 Well Status", ["All"] + status_vals_w, key="w_status_filter")
    if status_filter_w != "All":
        well_pool = well_pool[well_pool[status_col] == status_filter_w]

if type_col:
    type_vals_w = sorted(well_pool[type_col].dropna().unique().tolist())
    type_filter_w = sb.selectbox("🔵 Well Type", ["All"] + type_vals_w, key="w_type_filter")
    if type_filter_w != "All":
        well_pool = well_pool[well_pool[type_col] == type_filter_w]

available_wells = well_pool[well_col].dropna().unique()
if len(available_wells) == 0:
    st.warning("No wells match the selected Status / Type filters.")
    st.stop()

selected_well = sb.selectbox("Select Well", available_wells)
filtered = well_pool[well_pool[well_col] == selected_well].sort_values(date_col).reset_index(drop=True)
n = len(filtered)
actual = filtered[rate_col].values
dates = filtered[date_col]

if "last_well" not in st.session_state or st.session_state.last_well != selected_well:
    st.session_state.start_slider = 0
    st.session_state.last_well = selected_well

# ─────────────────────────────────────────
# Sidebar — DCA Start Point
# ─────────────────────────────────────────
sb.header("📍 DCA Start Point")
start_mode = sb.radio("Mode:", ["Slider + Peak Finder", "Exact Date Picker", "Manual Input"], horizontal=False)
peak_idx = int(np.argmax(actual))

if start_mode == "Slider + Peak Finder":
    if sb.button("📈 Jump to Peak Rate"):
        st.session_state.start_slider = peak_idx
    start_idx = sb.slider("Drag to select start index", min_value=0, max_value=n - 2, key="start_slider")
    lo, hi = max(0, start_idx - 2), min(n, start_idx + 3)
    nearby = filtered[[date_col, rate_col]].iloc[lo:hi].copy()
    nearby[date_col] = nearby[date_col].dt.strftime("%Y-%m-%d")
    nearby.index = range(lo, hi);
    nearby.index.name = "idx"
    sb.caption("📋 Points around current selection:");
    sb.dataframe(nearby, use_container_width=True)
elif start_mode == "Exact Date Picker":
    date_labels = dates.dt.strftime("%Y-%m-%d").tolist()
    picked_label = sb.selectbox("Pick exact date", date_labels, index=st.session_state.start_slider)
    start_idx = date_labels.index(picked_label)
    st.session_state.start_slider = start_idx
else:
    manual_date_str = sb.text_input("Date (YYYY-MM-DD)", value=str(dates.iloc[0].date()))
    try:
        manual_ts = pd.Timestamp(manual_date_str)
        start_idx = int((dates - manual_ts).abs().argmin())
        st.session_state.start_slider = start_idx
    except Exception:
        sb.error("Invalid date — use YYYY-MM-DD");
        start_idx = 0

sb.info(f"**Date:** {dates.iloc[start_idx].date()}  \n"
        f"**Actual rate:** {actual[start_idx]:.1f}  \n"
        f"**Peak is at:** {dates.iloc[peak_idx].date()} ({actual[peak_idx]:.1f})")

# ─────────────────────────────────────────
# Sidebar — DCA Parameters
# ─────────────────────────────────────────
sb.header("📉 DCA Parameters")
sb.header("📌 Show Models")
show_exp = sb.checkbox("Exponential", True)
show_hyp = sb.checkbox("Hyperbolic", True)
show_har = sb.checkbox("Harmonic", True)

qi = sb.number_input("Initial Rate (qi)", value=round(float(actual[start_idx]), 2), key=f"qi_{start_idx}")
Di = sb.slider("Decline Rate (Di, /year)", 0.001, 2.0, 0.3)
b = sb.slider("Decline Exponent (b)", 0.01, 1.0, 0.5)

date_labels_all = dates.dt.strftime("%Y-%m-%d").tolist()

# ─────────────────────────────────────────
# Compute curves
# ─────────────────────────────────────────
t_days = (dates - dates.iloc[start_idx]).dt.days.values
t_pos = np.where(t_days >= 0, t_days / 365.25, 0)
mask = t_days >= 0

exp_c = exponential(qi, Di, t_pos)
hyp_c = hyperbolic(qi, Di, b, t_pos)
har_c = harmonic(qi, Di, t_pos)

# ─────────────────────────────────────────
# ★ Build perf event map for chart annotations
# ─────────────────────────────────────────
_chart_event_map = []
_pre_timeline = []
if perf_df_raw is not None and p_well_col and p_date_col_p and p_study_col:
    _pre_well_ev = perf_df_raw[
        perf_df_raw[p_well_col].astype(str).str.strip() == str(selected_well).strip()
        ].copy()
    _pre_well_ev[p_date_col_p] = pd.to_datetime(_pre_well_ev[p_date_col_p], errors="coerce")
    _pre_well_ev = _pre_well_ev.dropna(subset=[p_date_col_p]).sort_values(p_date_col_p).reset_index(drop=True)
    _mask_ps = _pre_well_ev[p_study_col].astype(str).str.lower().str.contains("perf|squeeze|sqz", na=False)
    _pre_well_ev = _pre_well_ev[_mask_ps].reset_index(drop=True)

    if not _pre_well_ev.empty and p_start_col is not None and p_end_col is not None:
        _pre_timeline, _, _ = analyze_perf_squeeze_history(
            _pre_well_ev,
            study_col=p_study_col,
            start_col=p_start_col,
            end_col=p_end_col,
            interval_col=p_interval_col_p,
            date_col=p_date_col_p,
        )
        _chart_event_map = _build_chart_event_map(_pre_timeline)

# ─────────────────────────────────────────
# Main production plot
# ─────────────────────────────────────────
_w_tags = []
if status_col and "w_status_filter" in st.session_state and st.session_state.w_status_filter != "All":
    _w_tags.append(f"Status: {st.session_state.w_status_filter}")
if type_col and "w_type_filter" in st.session_state and st.session_state.w_type_filter != "All":
    _w_tags.append(f"Type: {st.session_state.w_type_filter}")
_w_filter_str = f"  •  🔍 {' | '.join(_w_tags)}" if _w_tags else ""

st.subheader(f"📈 Well: {selected_well}{_w_filter_str}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=dates, y=actual, mode="lines+markers", name="Actual",
                         line=dict(color="#1f77b4", width=3), marker=dict(size=5, opacity=0.6),
                         hovertemplate="<b>Actual</b> %{x|%Y-%m-%d}: %{y:.1f}<extra></extra>"))

d2 = dates[mask]
if show_exp: fig.add_trace(
    go.Scatter(x=d2, y=exp_c[mask], mode="lines", name="Exponential", line=dict(color="orange", width=2, dash="dash")))
if show_hyp: fig.add_trace(go.Scatter(x=d2, y=hyp_c[mask], mode="lines", name="Hyperbolic",
                                      line=dict(color="limegreen", width=2, dash="dash")))
if show_har: fig.add_trace(
    go.Scatter(x=d2, y=har_c[mask], mode="lines", name="Harmonic", line=dict(color="red", width=2, dash="dash")))

fig.add_trace(go.Scatter(
    x=[dates.iloc[peak_idx]], y=[actual[peak_idx]],
    mode="markers", name="Peak",
    marker=dict(size=14, color="white", symbol="triangle-up", line=dict(color="gray", width=1)),
    showlegend=True,
    hovertemplate=f"<b>Peak</b><br>Date: {dates.iloc[peak_idx].strftime('%b %Y')}<br>Rate: {actual[peak_idx]:,.0f}<extra></extra>",
))

y_min, y_max = float(np.nanmin(actual)), float(np.nanmax(actual))
fig.add_trace(go.Scatter(
    x=[dates.iloc[start_idx], dates.iloc[start_idx]], y=[y_min, y_max],
    mode="lines", name="DCA Start",
    line=dict(color="yellow", width=2, dash="dot"),
))
fig.add_trace(go.Scatter(
    x=[dates.iloc[start_idx]], y=[qi],
    mode="markers", name="Start ★",
    marker=dict(size=16, color="yellow", symbol="star"),
    showlegend=True,
    hovertemplate=f"<b>DCA Start</b><br>Date: {dates.iloc[start_idx].strftime('%b %Y')}<br>qi = {qi:,.0f}<extra></extra>",
))

if perf_col and perf_col in filtered.columns:
    perf_vals = pd.to_numeric(filtered[perf_col], errors="coerce").values
    fig.add_trace(go.Scatter(x=dates, y=perf_vals, name="Perforation", mode="none",
                             hovertemplate="<b>Perforation</b>: %{y:,.2f}<extra></extra>", showlegend=True))

# ── Smart Annotations for Perf ──
_TYPE_VLINE_COLORS = {
    "Initial": "#4da6ff",
    "Additional": "#00e676",
    "Reperforation": "#ffeb3b",
    "Squeeze": "#ff9800",
}
_TYPE_SHORT_LABELS = {
    "Initial": "Initial Perforation",
    "Additional": "Addl.",
    "Reperforation": "Reperf.",
    "Squeeze": "Squeeze",
}


def _add_event_vline(fig, ms_timestamp, snapped_date, raw_date,
                     color, label, depth_hover, etype,
                     ann_dates_list):
    PROXIMITY_DAYS = 180
    proximity_count = sum(
        1 for (prev_d, _) in ann_dates_list
        if abs((snapped_date - prev_d).days) <= PROXIMITY_DAYS
    )
    _y_label = 0.95 - (proximity_count % 4) * 0.09
    ann_dates_list.append((snapped_date, _y_label))

    fig.add_vline(
        x=ms_timestamp,
        line=dict(color=color, width=1.8, dash="dot"),
        annotation=dict(
            text=f"<b>{label}</b>",
            font=dict(size=13, color=color),
            bgcolor="rgba(15,15,15,0.82)",
            bordercolor=color,
            borderwidth=1,
            borderpad=4,
            yref="paper",
            y=_y_label,
            xanchor="center",
            yanchor="middle",
            showarrow=False,
        ),
    )
    fig.add_trace(go.Scatter(
        x=[snapped_date], y=[None], mode="none",
        name=label, showlegend=False,
        hovertemplate=(
                f"<b>{etype}</b>"
                f"<br>Event Date: {raw_date.strftime('%Y-%m-%d') if hasattr(raw_date, 'strftime') else raw_date}"
                + (f"<br>{depth_hover}" if depth_hover else "")
                + "<extra></extra>"
        ),
    ))


_ann_dates_list = []

if _chart_event_map:
    for ev_info in _chart_event_map:
        _raw_date = ev_info["date"]
        _etype = ev_info["event_type"]
        _ev_color = _TYPE_VLINE_COLORS.get(_etype, "#ffffff")
        _ev_label = _TYPE_SHORT_LABELS.get(_etype, _etype)

        future_dates = dates[dates >= _raw_date]
        snapped_date = future_dates.iloc[0] if not future_dates.empty else dates.iloc[-1]
        ms_timestamp = int(snapped_date.timestamp() * 1000)

        _depth_parts = []
        if ev_info.get("int_start") is not None:
            _depth_parts.append(f"Start: {ev_info['int_start']:.1f} m")
        if ev_info.get("int_end") is not None:
            _depth_parts.append(f"End: {ev_info['int_end']:.1f} m")
        if ev_info.get("length") is not None:
            _depth_parts.append(f"Length: {ev_info['length']:.1f} m")
        _depth_hover = "<br>".join(_depth_parts)

        _add_event_vline(fig, ms_timestamp, snapped_date, _raw_date,
                         _ev_color, _ev_label, _depth_hover, _etype,
                         _ann_dates_list)

elif perf_df_raw is not None and p_well_col and p_date_col_p and p_study_col:
    _well_perf_chart = perf_df_raw[
        perf_df_raw[p_well_col].astype(str).str.strip() == str(selected_well).strip()
        ].copy()
    _well_perf_chart[p_date_col_p] = pd.to_datetime(_well_perf_chart[p_date_col_p], errors="coerce")
    _well_perf_chart = _well_perf_chart.dropna(subset=[p_date_col_p])
    mask_chart = _well_perf_chart[p_study_col].astype(str).str.lower().str.contains("perf|squeeze|sqz", na=False)
    _well_perf_chart = _well_perf_chart[mask_chart]

    for _, ev in _well_perf_chart.iterrows():
        _study_lc = str(ev[p_study_col]).strip().lower()
        _is_perf = "perf" in _study_lc
        _ev_color = "#00e676" if _is_perf else "#ff9800"
        _ev_label = "PERF" if _is_perf else "SQZ"
        _raw_ev_date = ev[p_date_col_p]

        future_dates = dates[dates >= _raw_ev_date]
        snapped_date = future_dates.iloc[0] if not future_dates.empty else dates.iloc[-1]
        ms_timestamp = int(snapped_date.timestamp() * 1000)

        _depth_parts = []
        if p_start_col and p_start_col in ev.index:
            _depth_parts.append(f"Start: {ev[p_start_col]}")
        if p_end_col and p_end_col in ev.index:
            _depth_parts.append(f"End: {ev[p_end_col]}")
        if p_interval_col_p and p_interval_col_p in ev.index:
            _depth_parts.append(f"Interval: {ev[p_interval_col_p]}")
        _depth_hover = "<br>".join(_depth_parts) if _depth_parts else ""
        _etype = "Perforation" if _is_perf else "Squeeze"

        _add_event_vline(fig, ms_timestamp, snapped_date, _raw_ev_date,
                         _ev_color, _ev_label, _depth_hover, _etype,
                         _ann_dates_list)

# ── Arrow annotations for Peak and DCA Start ★ ──
_peak_date = dates.iloc[peak_idx]
_peak_val = actual[peak_idx]

fig.add_annotation(
    x=_peak_date, y=_peak_val,
    text=f"<b>Peak<br>{_peak_val:,.0f}</b>",
    font=dict(size=15, color="white"),
    bgcolor="rgba(40,40,40,0.80)",
    bordercolor="white", borderwidth=1,
    xanchor="left", yanchor="bottom",
    ax=50, ay=-55,
    arrowhead=2, arrowwidth=1.5, arrowcolor="white", showarrow=True,
)

_start_date = dates.iloc[start_idx]
fig.add_annotation(
    x=_start_date, y=qi,
    text=f"<b>DCA Start<br>{_start_date.strftime('%b %Y')}<br>qi = {qi:,.0f}</b>",
    font=dict(size=15, color="yellow"),
    bgcolor="rgba(40,40,40,0.80)",
    bordercolor="yellow", borderwidth=1,
    xanchor="left", yanchor="middle",
    ax=65, ay=0,
    arrowhead=2, arrowwidth=1.5, arrowcolor="yellow", showarrow=True,
)

_apply_plateaus_to_fig(fig, st.session_state["w_plateaus"])
_apply_comments_to_fig(fig, st.session_state["w_comments"])

fig.update_layout(
    title=dict(text=f"🛢️ Well: <b>{selected_well}</b>", font=dict(size=22, color="white"), x=0, xanchor="left"),
    xaxis=dict(title=dict(text="Date", font=dict(size=18)), tickfont=dict(size=16)),
    yaxis=dict(title=dict(text="Production Rate", font=dict(size=18)), tickfont=dict(size=16)),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="right", x=1, font=dict(size=15),
    ),
    margin=dict(l=60, r=30, t=80, b=60),
    height=640, template="plotly_dark", hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True, config=_chart_config(f"well_{selected_well}"))

_comment_editor_ui("w_comments", date_labels_all, actual)
_plateau_editor_ui("w_plateaus", date_labels_all, actual)

# ─────────────────────────────────────────
# RMSE
# ─────────────────────────────────────────
st.subheader("📊 Model Performance (RMSE)")
results = []
seg = actual[mask]
if show_exp: results.append(("Exponential", rmse(seg, exp_c[mask])))
if show_hyp: results.append(("Hyperbolic", rmse(seg, hyp_c[mask])))
if show_har: results.append(("Harmonic", rmse(seg, har_c[mask])))

best_model = None
if results:
    mdf = pd.DataFrame(results, columns=["Model", "RMSE"])
    st.dataframe(mdf)
    best_model = mdf.loc[mdf.RMSE.idxmin(), "Model"]
    st.success(f"🏆 Best Fit: {best_model}")

# ─────────────────────────────────────────
# Well Statistics
# ─────────────────────────────────────────
st.subheader("📊 Well Statistics")
wc1, wc2, wc3, wc4 = st.columns(4)
wc1.metric("Peak Rate", f"{actual.max():,.1f}")
wc2.metric("Avg Rate", f"{actual.mean():,.1f}")
wc3.metric("Total Volume", f"{actual.sum():,.1f}")
wc4.metric("Data Points", n)

# ═══════════════════════════════════════════════════════════════════
#  ★ PERFORATION / SQUEEZE INTERVAL HISTORY ANALYSIS  (Well View)
# ═══════════════════════════════════════════════════════════════════
if perf_df_raw is not None and p_well_col and p_date_col_p and p_study_col:

    well_ev_raw = perf_df_raw[
        perf_df_raw[p_well_col].astype(str).str.strip() == str(selected_well).strip()
        ].copy()
    well_ev_raw[p_date_col_p] = pd.to_datetime(well_ev_raw[p_date_col_p], errors="coerce")
    well_ev_raw = well_ev_raw.dropna(subset=[p_date_col_p]).sort_values(p_date_col_p).reset_index(drop=True)

    mask_perf_sqz = well_ev_raw[p_study_col].astype(str).str.lower().str.contains("perf|squeeze|sqz", na=False)
    well_ev_raw = well_ev_raw[mask_perf_sqz].reset_index(drop=True)

    if well_ev_raw.empty:
        st.info("ℹ️ No perforation / squeeze events found for this well in the uploaded sheet.")
    else:
        _study_series = well_ev_raw[p_study_col].astype(str).str.strip().str.lower()
        n_perf = int((_study_series.str.contains("perf")).sum())
        n_sqz = int((_study_series.str.contains("squeeze|sqz")).sum())

        _has_depth_for_calc = p_start_col is not None and p_end_col is not None
        _total_perf_interval = 0.0
        _total_squeeze_matched = 0.0
        _total_squeeze_raw = 0.0

        if _has_depth_for_calc:
            _active_calc: list = []
            for _, _row in well_ev_raw.sort_values(p_date_col_p).iterrows():
                _sc_lc = str(_row.get(p_study_col, "")).strip().lower()
                _s_v = _row.get(p_start_col)
                _e_v = _row.get(p_end_col)
                _s = float(_s_v) if _s_v is not None and pd.notna(_s_v) else None
                _e = float(_e_v) if _e_v is not None and pd.notna(_e_v) else None
                _L_raw_v = _row.get(p_interval_col_p) if p_interval_col_p else None
                _L = (float(_L_raw_v) if _L_raw_v is not None and pd.notna(_L_raw_v)
                      else (round(_e - _s, 3) if _s is not None and _e is not None else 0.0))

                if "perf" in _sc_lc and _s is not None and _e is not None:
                    _active_calc.append({"start": _s, "end": _e, "length": _L})
                    _total_perf_interval += _L
                elif ("squeeze" in _sc_lc or "sqz" in _sc_lc) and _s is not None and _e is not None:
                    _before_open = sum(iv["length"] for iv in _active_calc)
                    _active_calc, _ = _apply_squeeze(_active_calc, _s, _e)
                    _after_open = sum(iv["length"] for iv in _active_calc)
                    _total_squeeze_matched += (_before_open - _after_open)
                    _total_squeeze_raw += _L
        else:
            if p_interval_col_p:
                _perf_rows = well_ev_raw[_study_series.str.contains("perf")]
                _sqz_rows = well_ev_raw[_study_series.str.contains("squeeze|sqz")]
                _total_perf_interval = pd.to_numeric(_perf_rows[p_interval_col_p], errors="coerce").sum()
                _total_squeeze_raw = pd.to_numeric(_sqz_rows[p_interval_col_p], errors="coerce").sum()
                _total_squeeze_matched = _total_squeeze_raw

        st.subheader("🔩 Perforation and Squeeze Time Intervals")

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric(
            "🟢 Perforation Events", f"{n_perf}",
            delta=f"{_total_perf_interval:,.2f} m  total perforated" if _total_perf_interval > 0 else None,
            delta_color="normal",
        )
        mc2.metric(
            "🔴 Squeeze Events", f"{n_sqz}",
            delta=f"{_total_squeeze_raw:,.2f} m  squeeze interval" if _total_squeeze_raw > 0 else None,
            delta_color="inverse",
        )
        mc3.metric("📏 Total Perforated (m)",
                   f"{_total_perf_interval:,.2f}" if _total_perf_interval > 0 else "—")
        mc4.metric(
            "📐 Effectively Squeezed (m)",
            f"{_total_squeeze_matched:,.2f}" if _total_squeeze_matched > 0 else "—",
            help=(
                "Actual reduction in open perforation length after depth-matching squeezes to existing "
                "perforations. May differ from the raw squeeze interval when squeeze range only partially "
                "overlaps with perforated intervals."
            ) if _has_depth_for_calc
            else "Raw squeeze interval (enable Start/End Depth for depth-matched value).",
        )

        _disp_col_map = {p_date_col_p: "Date", p_study_col: "Study / Type"}
        if p_start_col and p_start_col in well_ev_raw.columns:
            _disp_col_map[p_start_col] = "Start Depth"
        if p_end_col and p_end_col in well_ev_raw.columns:
            _disp_col_map[p_end_col] = "End Depth"
        if p_interval_col_p and p_interval_col_p in well_ev_raw.columns:
            _disp_col_map[p_interval_col_p] = "Interval"

        disp_df = well_ev_raw[[c for c in _disp_col_map if c in well_ev_raw.columns]].rename(
            columns=_disp_col_map).copy()
        disp_df["Date"] = disp_df["Date"].dt.strftime("%Y-%m-%d")


        def _colour_events(row):
            study_v = str(row.get("Study / Type", "")).strip().lower()
            if "perf" in study_v:
                return ["background-color: rgba(0,200,100,0.20); color: #b3ffcc"] * len(row)
            elif "squeeze" in study_v or "sqz" in study_v:
                return ["background-color: rgba(255,140,0,0.20); color: #ffe0b3"] * len(row)
            else:
                return [""] * len(row)


        styled_events = disp_df.style.apply(_colour_events, axis=1).format(
            {c: "{:,.2f}" for c in ["Start Depth", "End Depth", "Interval"] if c in disp_df.columns}
        )
        st.dataframe(styled_events, use_container_width=True, hide_index=True)
        st.caption("🟢 Green = Perforation  |  🟠 Orange = Squeeze")

    # ── ★ DEEP INTERVAL ANALYSIS ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔬 Final Perforation (After Cementation)")

    _has_depth = p_start_col is not None and p_end_col is not None
    if not _has_depth:
        st.info("ℹ️ Assign **Start Depth** and **End Depth** columns in the sidebar to enable interval tracking.")
    elif well_ev_raw.empty:
        st.info("ℹ️ No events for this well to analyse.")
    else:
        timeline_data, final_active, all_ever = analyze_perf_squeeze_history(
            well_ev_raw,
            study_col=p_study_col,
            start_col=p_start_col,
            end_col=p_end_col,
            interval_col=p_interval_col_p,
            date_col=p_date_col_p,
        )
        render_perf_history_section(
            timeline=timeline_data,
            final_active=final_active,
            all_ever=all_ever,
            well_name=str(selected_well),
            prod_dates=dates,
        )

        if final_active:
            st.markdown("##### 🟢 Currently Open Intervals")
            open_cards_html = ""
            for iv in sorted(final_active, key=lambda x: x.get("start") or 0):
                s, e, L = iv.get("start"), iv.get("end"), iv.get("length")
                ad = iv.get("added_date")
                ad_str = ad.strftime("%d %b %Y") if hasattr(ad, "strftime") else str(ad)
                pt = iv.get("perf_type", "")
                pt_color = _PERF_TYPE_COLOR.get(pt, {}).get("border", "#00e676")
                open_cards_html += f"""
                <div style="
                    display:inline-block; margin:4px;
                    background:rgba(0,230,118,0.14);
                    border:1px solid {pt_color};
                    border-radius:8px; padding:6px 12px;
                    font-size:0.82rem; color:#b9ffd7;
                ">
                    <b>{s:.1f} – {e:.1f} m</b>
                    {'  ·  ' + str(round(L, 2)) + ' m' if L else ''}
                    <span style="color:{pt_color}; font-size:0.75rem; margin-left:6px">
                        {pt} · opened {ad_str}
                    </span>
                </div>"""
            st.markdown(open_cards_html, unsafe_allow_html=True)
        else:
            st.warning("⚠️ All intervals are currently squeezed — no open perforations remain.")

# ─────────────────────────────────────────
# Download section
# ─────────────────────────────────────────
base_cols = filtered[[well_col, date_col, rate_col]].copy()


def build_stacked(name, curve):
    row = base_cols.copy()
    row["DCA"] = np.where(mask, np.round(curve, 4), np.nan)
    row["Type"] = np.where(mask, name, "")
    return row


curves_map = {}
if show_exp: curves_map["Exponential"] = exp_c
if show_hyp: curves_map["Hyperbolic"] = hyp_c
if show_har: curves_map["Harmonic"] = har_c

st.subheader("⬇️ Download DCA Data")
if not curves_map:
    st.warning("Select at least one model above.")
    st.stop()

curve_options = list(curves_map.keys())
recommended_idx = (curve_options.index(best_model)
                   if best_model and best_model in curve_options else 0)
if best_model and best_model in curve_options:
    st.info(f"💡 **Recommended:** {best_model} (lowest RMSE — pre-selected below)")

chosen_curve = st.radio("**Step 1 — Select curve to download:**", curve_options,
                        index=recommended_idx, horizontal=True)
chosen_df = build_stacked(chosen_curve, curves_map[chosen_curve])

st.markdown("**Step 2 — Download mode:**")
already_in_combined = selected_well in st.session_state.downloaded_wells

if already_in_combined:
    st.warning(
        f"⚠️ **{selected_well}** is already in your combined file. You can only download it individually this time.")
    dl_mode = "Individual"
elif st.session_state.combined_df.empty:
    dl_mode = "Individual"
    st.info("💡 This will be your first download. Future wells can be combined with it.")
else:
    already_list = ", ".join(st.session_state.downloaded_wells)
    dl_mode = st.radio(f"Combined includes: **{already_list}**",
                       ["Individual", "Add to combined file"], horizontal=True)

with st.expander("Preview data to be downloaded"):
    st.dataframe(chosen_df.head(20))

fname = make_dl_name(st.session_state.dl_count)
if dl_mode == "Individual":
    dl_data = chosen_df
    btn_text = f"⬇️ Download {selected_well} — {chosen_curve}  ({fname})"
else:
    dl_data = pd.concat([st.session_state.combined_df, chosen_df], ignore_index=True)
    btn_text = (f"⬇️ Download Combined "
                f"({', '.join(st.session_state.downloaded_wells + [selected_well])})  ({fname})")

if st.download_button(btn_text, data=to_excel_bytes(dl_data), file_name=fname):
    if dl_mode != "Individual":
        st.session_state.combined_df = dl_data.copy()
    if selected_well not in st.session_state.downloaded_wells:
        st.session_state.combined_df = pd.concat(
            [st.session_state.combined_df, chosen_df], ignore_index=True
        ) if not st.session_state.combined_df.empty else chosen_df.copy()
        st.session_state.downloaded_wells.append(selected_well)
    st.session_state.dl_count += 1

if st.session_state.downloaded_wells:
    st.caption(
        f"📁 Combined history: **{', '.join(st.session_state.downloaded_wells)}** |  "
        f"Next filename: **{make_dl_name(st.session_state.dl_count)}**"
    )

if st.button("🗑️ Reset all download history"):
    st.session_state.combined_df = pd.DataFrame()
    st.session_state.downloaded_wells = []
    st.session_state.dl_count = 0
    st.rerun()
