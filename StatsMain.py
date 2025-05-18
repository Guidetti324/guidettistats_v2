###############################################################################
#  PSYC‑250 – Statistical Tables Explorer
#  ---------------------------------------------------------------------------
#  Seven complete tabs:
#      1) t‑Distribution           4) Chi‑Square
#      2) z‑Distribution           5) Mann‑Whitney U
#      3) F‑Distribution           6) Wilcoxon Signed‑Rank T
#      7) Binomial
#
#  NEW FEATURES ADDED FOR (t, z, Mann‑Whitney U, Wilcoxon T, Binomial):
#   1) A "Cumulative Table Note" explaining how to interpret the table for
#      one‑ vs. two‑tailed tests.
#   2) A "P‑Value Calculation Explanation" section next to the table,
#      showing how the table lookup leads to p, depending on one‑ vs. two‑tailed.
#   3) Automatic plot shading based on sign of the test statistic for one‑tailed
#      (negative → left tail, positive → right tail). For two‑tailed, shade both tails.
#
#  F‑Distribution and Chi‑Square remain as before, since they are always one‑tailed.
#
#  To run:   streamlit run app.py
###############################################################################

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.switch_backend("Agg")  # headless backend

###############################################################################
#                                COMMON SETUP
###############################################################################

def show_cumulative_note():
    """
    Shows the standardized cumulative probability note (for z, t, Mann-Whitney, 
    Wilcoxon, and Binomial).
    """
    st.info(
        "Note: The values in this table represent cumulative probabilities "
        "(i.e., the area under the curve to the left of a given value). For "
        "one‑tailed tests, use the area directly. For two‑tailed tests, you must "
        "double the area in the tail beyond your observed value (i.e., "
        "p=2×(1−P(Z≤|z|))). The same logic applies for t‑distributions. The table "
        "itself does not change—only how you interpret it does."
    )


def place_label(ax, placed, x, y, txt, *, color="blue"):
    """
    Place text on the plot, shifting it slightly if it would collide
    with previously placed labels.
    """
    dx = dy = 0.0
    for (xx, yy) in placed:
        if abs(x - xx) < 0.15 and abs(y - yy) < 0.05:
            dx += 0.06
            dy += 0.04
    ax.text(x + dx, y + dy, txt, color=color,
            ha="left", va="bottom", fontsize=8, clip_on=True)
    placed.append((x + dx, y + dy))


def style_cell(html: str, cid: str, color: str = "red", px: int = 2) -> str:
    """
    Give the <td id="cid"> a border of the specified color & thickness.
    If color="blue" and px=3, it indicates the final intersection cell.
    """
    return html.replace(
        f'id="{cid}"',
        f'id="{cid}" style="border:{px}px solid {color};"',
        1
    )


def wrap_table(css: str, table: str) -> str:
    return f"<style>{css}</style><table>{table}</table>"


def container(html: str, *, height: int = 460) -> str:
    """
    Scrollable container for large HTML tables so the page doesn't grow indefinitely.
    """
    return f'<div style="overflow:auto; max-height:{height}px;">{html}</div>'


CSS_BASE = (
    "table{border-collapse:collapse}"
    "th,td{border:1px solid #000;height:30px;text-align:center;"
    "font-family:sans-serif;font-size:0.9rem}"
    "th{background:#fafafa}"
)

###############################################################################
#                       TAB 1: t‑Distribution
###############################################################################

def plot_t(t_calc, df, alpha, tail):
    """
    Plot the t‑distribution. For one‑tailed, decide left vs. right tail
    based on sign of t_calc. For two‑tailed, shade both tails.
    """
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    xs = np.linspace(-4, 4, 400)
    ys = stats.t.pdf(xs, df)

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25,
                    label="Fail to Reject H0")

    placed_labels = []

    if tail.startswith("one"):
        # For one-tailed: decide which side based on sign of t_calc
        crit_pos = stats.t.ppf(1 - alpha, df)
        crit_neg = -crit_pos

        if t_calc >= 0:
            # shade right side
            ax.fill_between(xs[xs >= crit_pos], ys[xs >= crit_pos],
                            color="red", alpha=0.3, label="Reject H0")
            ax.axvline(crit_pos, color="green", ls="--")
            place_label(ax, placed_labels, crit_pos, stats.t.pdf(crit_pos, df)+0.02,
                        f"tcrit={crit_pos:.2f}", color="green")
        else:
            # shade left side
            ax.fill_between(xs[xs <= crit_neg], ys[xs <= crit_neg],
                            color="red", alpha=0.3, label="Reject H0")
            ax.axvline(crit_neg, color="green", ls="--")
            place_label(ax, placed_labels, crit_neg, stats.t.pdf(crit_neg, df)+0.02,
                        f"tcrit={crit_neg:.2f}", color="green")
    else:
        # two-tailed: shade both tails
        crit = stats.t.ppf(1 - alpha/2, df)
        ax.fill_between(xs[xs >= crit], ys[xs >= crit], color="red", alpha=0.3)
        ax.fill_between(xs[xs <= -crit], ys[xs <= -crit], color="red", alpha=0.3,
                        label="Reject H0")
        ax.axvline(crit, color="green", ls="--")
        ax.axvline(-crit, color="green", ls="--")
        place_label(ax, placed_labels, crit, stats.t.pdf(crit, df)+0.02,
                    f"+tcrit={crit:.2f}", color="green")
        place_label(ax, placed_labels, -crit, stats.t.pdf(-crit, df)+0.02,
                    f"–tcrit={crit:.2f}", color="green")

    ax.axvline(t_calc, color="blue", ls="--")
    place_label(ax, placed_labels, t_calc, stats.t.pdf(t_calc, df)+0.02,
                f"tcalc={t_calc:.2f}", color="blue")

    ax.set_xlabel("t")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title("t‑Distribution")
    fig.tight_layout()
    return fig


def build_t_html(df: int, alpha: float, tail: str) -> str:
    """
    Single-step highlight for t-table row & column + intersection.
    """
    rows = list(range(max(1, df-5), df+6))
    heads = [
        ("one", 0.10), ("one", 0.05), ("one", 0.01), ("one", 0.001),
        ("two", 0.10), ("two", 0.05), ("two", 0.01), ("two", 0.001)
    ]
    mode = "one" if tail.startswith("one") else "two"
    col_idx = next(i for i,(m,a) in enumerate(heads, start=1)
                   if m==mode and np.isclose(a, alpha))

    head_html = "".join(f"<th>{m}_{a}</th>" for m,a in heads)
    body_html = ""
    for r in rows:
        row_cells = f'<td id="t_{r}_0">{r}</td>'
        for i,(m,a) in enumerate(heads, start=1):
            val = stats.t.ppf(1 - a if m=="one" else 1 - a/2, r)
            row_cells += f'<td id="t_{r}_{i}">{val:.2f}</td>'
        body_html += f"<tr>{row_cells}</tr>"

    table_code = f"<tr><th>df</th>{head_html}</tr>{body_html}"
    html = wrap_table(CSS_BASE, table_code)

    # highlight entire row
    for i in range(len(heads)+1):
        html = style_cell(html, f"t_{df}_{i}")
    # highlight entire column
    for rr in rows:
        html = style_cell(html, f"t_{rr}_{col_idx}")
    # intersection
    html = style_cell(html, f"t_{df}_{col_idx}", color="blue", px=3)
    return html


def t_table(df: int, alpha: float, tail: str):
    code = build_t_html(df, alpha, tail)
    st.markdown(container(code), unsafe_allow_html=True)


def t_apa(t_val: float, df: int, alpha: float, tail: str):
    """
    Show the dynamic explanation and final APA lines for the t-distribution.
    """
    # Calculate p and critical value
    if tail.startswith("one"):
        p_calc = stats.t.sf(abs(t_val), df)  # single-sided
        crit = stats.t.ppf(1 - alpha, df)
        reject = (abs(t_val) > abs(crit)) if t_val < 0 else (t_val > crit)
    else:
        p_calc = stats.t.sf(abs(t_val), df) * 2
        crit = stats.t.ppf(1 - alpha/2, df)
        reject = (abs(t_val) > crit)

    decision = "rejected" if reject else "failed to reject"
    if reject:
        reason_stats = "because t(calc) exceeded t(crit)"
        reason_p = "because p < α"
    else:
        reason_stats = "because t(calc) did not exceed t(crit)"
        reason_p = "because p ≥ α"

    # "table" cdf value
    cdf_val = stats.t.cdf(t_val, df)

    # Explanation text
    if tail.startswith("one"):
        if t_val >= 0:
            expl = (
                f"Lookup: P(T ≤ {t_val:.2f}) = {cdf_val:.4f}.\n\n"
                f"For a **one‑tailed** test with a positive statistic, "
                f"p = 1 − {cdf_val:.4f} = {(1-cdf_val):.4f}."
            )
        else:
            expl = (
                f"Lookup: P(T ≤ {t_val:.2f}) = {cdf_val:.4f}.\n\n"
                f"For a **one‑tailed** test with a negative statistic, "
                f"p = {cdf_val:.4f} (left tail)."
            )
    else:
        expl = (
            f"Lookup: P(T ≤ {t_val:.2f}) = {cdf_val:.4f}.\n\n"
            f"For a **two‑tailed** test, p = 2 × min({cdf_val:.4f}, "
            f"{1-cdf_val:.4f}) = {2*min(cdf_val,1-cdf_val):.4f}."
        )

    st.write(expl)

    st.markdown(
        "**APA interpretation**  \n"
        f"Calculated statistic: *t*({df})={t_val:.2f}, *p*={p_calc:.3f}.  \n"
        f"Critical statistic: t(crit)={crit:.2f}, *p*={alpha:.3f}.  \n"
        f"Comparison of statistics → H0 **{decision}** ({reason_stats}).  \n"
        f"Comparison of *p*-values → H0 **{decision}** ({reason_p}).  \n"
        f"**APA 7 report:** *t*({df})={t_val:.2f}, *p*={p_calc:.3f} "
        f"({tail}). The null hypothesis was **{decision}** at α={alpha:.2f}."
    )


def tab_t():
    st.subheader("Tab 1 • t‑Distribution")

    c1, c2 = st.columns(2)
    with c1:
        t_val = st.number_input("t statistic", value=2.10, key="t_val")
        df = st.number_input("df", min_value=1, value=10, step=1, key="t_df")
    with c2:
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="t_alpha")
        tail = st.radio("Tail", ["one‑tailed", "two‑tailed"], key="t_tail")

    if st.button("Update Plot", key="t_plot"):
        st.pyplot(plot_t(t_val, df, alpha, tail))

    st.write("**t‑table** (single highlight)")
    # Table & note in left column, explanation in right column
    ctable, cexp = st.columns([2,1])
    with ctable:
        t_table(df, alpha, tail)
        show_cumulative_note()
    with cexp:
        st.subheader("P‑value Calculation Explanation")
        t_apa(t_val, df, alpha, tail)


###############################################################################
#                           TAB 2: z‑Distribution
###############################################################################

def plot_z(z_calc, alpha, tail):
    """
    For one‑tailed, if z_calc>=0, shade the right tail; if negative, shade left tail.
    For two‑tailed, shade both tails.
    """
    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    xs = np.linspace(-4,4,400)
    ys = stats.norm.pdf(xs)

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25,
                    label="Fail to Reject H0")

    placed = []

    if tail.startswith("one"):
        crit_pos = stats.norm.ppf(1 - alpha)
        crit_neg = -crit_pos
        if z_calc >= 0:
            ax.fill_between(xs[xs>=crit_pos], ys[xs>=crit_pos],
                            color="red", alpha=0.3, label="Reject H0")
            ax.axvline(crit_pos, color="green", ls="--")
            place_label(ax, placed, crit_pos, stats.norm.pdf(crit_pos)+0.02,
                        f"z₍crit₎={crit_pos:.2f}", color="green")
        else:
            ax.fill_between(xs[xs<=crit_neg], ys[xs<=crit_neg],
                            color="red", alpha=0.3, label="Reject H0")
            ax.axvline(crit_neg, color="green", ls="--")
            place_label(ax, placed, crit_neg, stats.norm.pdf(crit_neg)+0.02,
                        f"z₍crit₎={crit_neg:.2f}", color="green")
    else:
        crit = stats.norm.ppf(1 - alpha/2)
        ax.fill_between(xs[xs>=crit], ys[xs>=crit], color="red", alpha=0.3)
        ax.fill_between(xs[xs<=-crit], ys[xs<=-crit], color="red", alpha=0.3,
                        label="Reject H0")
        ax.axvline(crit, color="green", ls="--")
        ax.axvline(-crit, color="green", ls="--")
        place_label(ax, placed, crit, stats.norm.pdf(crit)+0.02,
                    f"+z₍crit₎={crit:.2f}", color="green")
        place_label(ax, placed, -crit, stats.norm.pdf(-crit)+0.02,
                    f"–z₍crit₎={crit:.2f}", color="green")

    ax.axvline(z_calc, color="blue", ls="--")
    place_label(ax, placed, z_calc, stats.norm.pdf(z_calc)+0.02,
                f"z₍calc₎={z_calc:.2f}", color="blue")

    ax.set_xlabel("z")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title("z‑Distribution")
    fig.tight_layout()
    return fig


def build_z_html(z_val: float, alpha: float, tail: str) -> str:
    """
    Single-step highlight for z-table.
    """
    z_val = np.clip(z_val, -3.49, 3.49)
    row = np.floor(z_val*10)/10
    col = round(z_val-row, 2)

    Rows = np.round(np.arange(-3.4, 3.5, 0.1), 1)
    Cols = np.round(np.arange(0, 0.1, 0.01), 2)

    # find nearest col
    col = min(Cols, key=lambda c: abs(c - col))

    idx_arr = np.where(Rows == row)[0]
    if len(idx_arr) > 0:
        irow = idx_arr[0]
    else:
        irow = len(Rows)//2
    show_rows = Rows[max(0, irow-10): irow+11]

    head = "".join(f"<th>{c:.2f}</th>" for c in Cols)
    body = ""
    for rr in show_rows:
        row_html = f'<td id="z_{rr:.1f}_0">{rr:.1f}</td>'
        for cc in Cols:
            cdf_val = stats.norm.cdf(rr + cc)
            row_html += f'<td id="z_{rr:.1f}_{cc:.2f}">{cdf_val:.4f}</td>'
        body += f"<tr>{row_html}</tr>"

    table_code = f"<tr><th>z.x</th>{head}</tr>{body}"
    html = wrap_table(CSS_BASE, table_code)

    # highlight row
    for cc in Cols:
        html = style_cell(html, f"z_{row:.1f}_{cc:.2f}")
    html = style_cell(html, f"z_{row:.1f}_0")

    # highlight column
    for rr in show_rows:
        html = style_cell(html, f"z_{rr:.1f}_{col:.2f}")

    # intersection in blue
    html = style_cell(html, f"z_{row:.1f}_{col:.2f}", color="blue", px=3)
    return html


def z_table(z_val: float, alpha: float, tail: str):
    code = build_z_html(z_val, alpha, tail)
    st.markdown(container(code), unsafe_allow_html=True)


def z_apa(z_val: float, alpha: float, tail: str):
    """
    Show the dynamic explanation and final APA lines for the z-distribution.
    """
    # Actual p
    if tail.startswith("one"):
        p_calc = stats.norm.sf(abs(z_val))
        crit_pos = stats.norm.ppf(1 - alpha)
        crit_neg = -crit_pos
        crit = crit_pos if z_val>0 else crit_neg
        reject = (abs(z_val) > abs(crit))
    else:
        p_calc = stats.norm.sf(abs(z_val))*2
        crit = stats.norm.ppf(1 - alpha/2)
        reject = (abs(z_val) > crit)

    decision = "rejected" if reject else "failed to reject"
    if reject:
        reason_stats = "because z₍calc₎ exceeded z₍crit₎"
        reason_p = "because p < α"
    else:
        reason_stats = "because z₍calc₎ did not exceed z₍crit₎"
        reason_p = "because p ≥ α"

    # table cdf
    table_val = stats.norm.cdf(z_val)
    # explanation
    if tail.startswith("one"):
        if z_val >= 0:
            expl = (
                f"Lookup: P(Z ≤ {z_val:.2f}) = {table_val:.4f}\n\n"
                f"For a **one‑tailed** test with positive z, p = 1 − {table_val:.4f}"
            )
        else:
            expl = (
                f"Lookup: P(Z ≤ {z_val:.2f}) = {table_val:.4f}\n\n"
                f"For a **one‑tailed** test with negative z, p = {table_val:.4f}"
            )
    else:
        expl = (
            f"Lookup: P(Z ≤ {z_val:.2f}) = {table_val:.4f}\n\n"
            f"For a **two‑tailed** test, p = 2 × min({table_val:.4f}, "
            f"{1 - table_val:.4f})"
        )

    st.write(expl)

    st.markdown(
        "**APA interpretation**  \n"
        f"Calculated statistic: *z*={z_val:.2f}, *p*={p_calc:.3f}.  \n"
        f"Critical statistic: z₍crit₎={crit:.2f}, *p*={alpha:.3f}.  \n"
        f"Statistic comparison → H₀ **{decision}** ({reason_stats}).  \n"
        f"*p* comparison → H₀ **{decision}** ({reason_p}).  \n"
        f"**APA 7 report:** *z*={z_val:.2f}, *p*={p_calc:.3f} ({tail}). "
        f"The null hypothesis was **{decision}** at α={alpha:.2f}."
    )


def tab_z():
    st.subheader("Tab 2 • z‑Distribution")

    c1, c2 = st.columns(2)
    with c1:
        z_val = st.number_input("z statistic", value=1.64, key="z_val")
    with c2:
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="z_alpha")
        tail = st.radio("Tail", ["one‑tailed", "two‑tailed"], key="z_tail")

    if st.button("Update Plot", key="z_plot"):
        st.pyplot(plot_z(z_val, alpha, tail))

    st.write("**z‑table** (single highlight)")
    ctable, cexp = st.columns([2,1])
    with ctable:
        z_table(z_val, alpha, tail)
        show_cumulative_note()
    with cexp:
        st.subheader("P‑value Calculation Explanation")
        z_apa(z_val, alpha, tail)


###############################################################################
#                        TAB 3: F‑Distribution (unchanged)
###############################################################################

def plot_f(f_calc, df1, df2, alpha):
    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    xs = np.linspace(0, stats.f.ppf(0.995, df1, df2)*1.1, 400)
    ys = stats.f.pdf(xs, df1, df2)
    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25, label="Fail to Reject H0")

    crit = stats.f.ppf(1 - alpha, df1, df2)
    ax.fill_between(xs[xs>=crit], ys[xs>=crit], color="red", alpha=0.3,
                    label="Reject H0")
    ax.axvline(crit, color="green", ls="--")
    ax.axvline(f_calc, color="blue", ls="--")

    place_label(ax, [], crit, stats.f.pdf(crit, df1, df2)+0.02,
                f"F₍crit₎={crit:.2f}", color="green")
    place_label(ax, [], f_calc, stats.f.pdf(f_calc, df1, df2)+0.02,
                f"F₍calc₎={f_calc:.2f}", color="blue")

    ax.set_xlabel("F")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title(f"F‑Distribution (df1={df1}, df2={df2})")
    fig.tight_layout()
    return fig


def build_f_table(df1: int, df2: int, alpha: float) -> str:
    rows = list(range(max(1,df1-5), df1+6))
    cols = list(range(max(1,df2-5), df2+6))
    col_idx = cols.index(df2)+1

    head = "".join(f"<th>{c}</th>" for c in cols)
    body = ""
    for r in rows:
        row_html = f'<td id="f_{r}_0">{r}</td>'
        for i,c in enumerate(cols, start=1):
            val = stats.f.ppf(1 - alpha, r, c)
            row_html += f'<td id="f_{r}_{i}">{val:.2f}</td>'
        body += f"<tr>{row_html}</tr>"

    code = f"<tr><th>df1\\df2</th>{head}</tr>{body}"
    html = wrap_table(CSS_BASE, code)

    # highlight row
    for i in range(len(cols)+1):
        html = style_cell(html, f"f_{df1}_{i}")
    # highlight col
    for rr in rows:
        html = style_cell(html, f"f_{rr}_{col_idx}")
    # intersection
    html = style_cell(html, f"f_{df1}_{col_idx}", color="blue", px=3)
    return html


def f_table(df1: int, df2: int, alpha: float):
    code = build_f_table(df1, df2, alpha)
    st.markdown(container(code), unsafe_allow_html=True)


def f_apa(f_val: float, df1: int, df2: int, alpha: float):
    p_calc = stats.f.sf(f_val, df1, df2)
    crit = stats.f.ppf(1 - alpha, df1, df2)
    reject = (f_val>crit)
    decision = "rejected" if reject else "failed to reject"
    reason_stats = ("because F₍calc₎ exceeded F₍crit₎"
                    if reject else "because F₍calc₎ did not exceed F₍crit₎")
    reason_p = ("because p < α" if reject else "because p ≥ α")

    st.markdown(
        "**APA interpretation**  \n"
        f"Calculated statistic: *F*({df1},{df2})={f_val:.2f}, *p*={p_calc:.3f}.  \n"
        f"Critical statistic: F₍crit₎={crit:.2f}, *p*={alpha:.3f}.  \n"
        f"Statistic comparison → H0 **{decision}** ({reason_stats}).  \n"
        f"*p* comparison → H0 **{decision}** ({reason_p}).  \n"
        f"**APA 7 report:** *F*({df1},{df2})={f_val:.2f}, *p*={p_calc:.3f}. "
        f"The null hypothesis was **{decision}** at α={alpha:.2f}."
    )


def tab_f():
    st.subheader("Tab 3 • F‑Distribution")

    c1, c2 = st.columns(2)
    with c1:
        f_val = st.number_input("F statistic", value=4.32, key="f_val")
        df1 = st.number_input("df1 (numerator)", min_value=1, value=5, step=1, key="f_df1")
    with c2:
        df2 = st.number_input("df2 (denominator)", min_value=1, value=20, step=1, key="f_df2")
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="f_alpha")

    if st.button("Update Plot", key="f_plot"):
        st.pyplot(plot_f(f_val, df1, df2, alpha))

    st.write("**F‑table** (always one‑tailed, no new cumulative note or p expl.)")
    f_table(df1, df2, alpha)
    f_apa(f_val, df1, df2, alpha)


###############################################################################
#                          TAB 4: Chi-Square (unchanged)
###############################################################################

def plot_chi(chi_calc, df, alpha):
    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    xs = np.linspace(0, stats.chi2.ppf(0.995, df)*1.1, 400)
    ys = stats.chi2.pdf(xs, df)
    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25, label="Fail to Reject H0")

    crit = stats.chi2.ppf(1 - alpha, df)
    ax.fill_between(xs[xs>=crit], ys[xs>=crit], color="red", alpha=0.3,
                    label="Reject H0")
    ax.axvline(crit, color="green", ls="--")
    ax.axvline(chi_calc, color="blue", ls="--")
    place_label(ax, [], crit, stats.chi2.pdf(crit, df)+0.02,
                f"χ²₍crit₎={crit:.2f}", color="green")
    place_label(ax, [], chi_calc, stats.chi2.pdf(chi_calc, df)+0.02,
                f"χ²₍calc₎={chi_calc:.2f}", color="blue")

    ax.set_xlabel("χ²")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title(f"χ²‑Distribution (df={df})")
    fig.tight_layout()
    return fig

def build_chi_table(df: int, alpha: float) -> str:
    rows = list(range(max(1,df-5), df+6))
    alphas = [0.10,0.05,0.01,0.001]
    col_idx = alphas.index(alpha)+1

    head = "".join(f"<th>{a}</th>" for a in alphas)
    body = ""
    for r in rows:
        row_html = f'<td id="chi_{r}_0">{r}</td>'
        for i,a in enumerate(alphas, start=1):
            val = stats.chi2.ppf(1 - a, r)
            row_html += f'<td id="chi_{r}_{i}">{val:.2f}</td>'
        body += f"<tr>{row_html}</tr>"

    table_code = f"<tr><th>df\\α</th>{head}</tr>{body}"
    html = wrap_table(CSS_BASE, table_code)

    # highlight row, col, intersection
    for i in range(len(alphas)+1):
        html = style_cell(html, f"chi_{df}_{i}")
    for rr in rows:
        html = style_cell(html, f"chi_{rr}_{col_idx}")
    html = style_cell(html, f"chi_{df}_{col_idx}", color="blue", px=3)
    return html

def chi_table(df: int, alpha: float):
    code = build_chi_table(df, alpha)
    st.markdown(container(code), unsafe_allow_html=True)

def chi_apa(chi_val: float, df: int, alpha: float):
    p_calc = stats.chi2.sf(chi_val, df)
    crit = stats.chi2.ppf(1 - alpha, df)
    reject = (chi_val>crit)
    decision = "rejected" if reject else "failed to reject"
    reason_stats = "because χ²₍calc₎ exceeded χ²₍crit₎" if reject else "because χ²₍calc₎ did not exceed χ²₍crit₎"
    reason_p = "because p < α" if reject else "because p ≥ α"

    st.markdown(
        "**APA interpretation**  \n"
        f"Calculated statistic: χ²({df})={chi_val:.2f}, *p*={p_calc:.3f}.  \n"
        f"Critical statistic: χ²₍crit₎={crit:.2f}, *p*={alpha:.3f}.  \n"
        f"Statistic comparison → H0 **{decision}** ({reason_stats}).  \n"
        f"*p* comparison → H0 **{decision}** ({reason_p}).  \n"
        f"**APA 7 report:** χ²({df})={chi_val:.2f}, *p*={p_calc:.3f}. "
        f"The null hypothesis was **{decision}** at α={alpha:.2f}."
    )


def tab_chi():
    st.subheader("Tab 4 • Chi‑Square")

    c1, c2 = st.columns(2)
    with c1:
        chi_val = st.number_input("χ² statistic", value=7.88, key="chi_val")
        df = st.number_input("df", min_value=1, value=3, step=1, key="chi_df")
    with c2:
        alpha = st.selectbox("α", [0.10,0.05,0.01,0.001],
                             index=1, key="chi_alpha")

    if st.button("Update Plot", key="chi_plot"):
        st.pyplot(plot_chi(chi_val, df, alpha))

    st.write("**χ²‑table**")
    chi_table(df, alpha)
    chi_apa(chi_val, df, alpha)


###############################################################################
#                   TAB 5: Mann‑Whitney U (updated)
###############################################################################

def plot_u(u_calc, n1, n2, alpha, tail):
    """
    Normal approx. We'll interpret sign around the midpoint (mu) to decide
    shading for one-tailed. For two-tailed, shade both tails.
    """
    μ = n1*n2/2
    σ = np.sqrt(n1*n2*(n1+n2+1)/12)

    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    xs = np.linspace(μ-4*σ, μ+4*σ, 400)
    ys = stats.norm.pdf(xs, μ, σ)

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25,
                    label="Fail to Reject H0")

    from math import floor
    zcrit = stats.norm.ppf(alpha if tail.startswith("one") else alpha/2)
    crit_val = floor(μ + zcrit*σ)
    hi_val = n1*n2 - crit_val

    if tail.startswith("one"):
        if u_calc >= μ:
            ax.fill_between(xs[xs>=crit_val], ys[xs>=crit_val], color="red", alpha=0.3,
                            label="Reject H0")
            ax.axvline(crit_val, color="green", ls="--")
            place_label(ax, [], crit_val, stats.norm.pdf(crit_val, μ, σ)+0.02,
                        f"Ucrit={crit_val}", color="green")
        else:
            ax.fill_between(xs[xs<=crit_val], ys[xs<=crit_val], color="red", alpha=0.3,
                            label="Reject H0")
            ax.axvline(crit_val, color="green", ls="--")
            place_label(ax, [], crit_val, stats.norm.pdf(crit_val, μ, σ)+0.02,
                        f"Ucrit={crit_val}", color="green")
    else:
        ax.fill_between(xs[xs<=crit_val], ys[xs<=crit_val], color="red", alpha=0.3)
        ax.fill_between(xs[xs>=hi_val], ys[xs>=hi_val], color="red", alpha=0.3,
                        label="Reject H0")
        ax.axvline(crit_val, color="green", ls="--")
        ax.axvline(hi_val, color="green", ls="--")
        place_label(ax, [], crit_val, stats.norm.pdf(crit_val, μ, σ)+0.02,
                    f"Ucrit={crit_val}", color="green")
        place_label(ax, [], hi_val, stats.norm.pdf(hi_val, μ, σ)+0.02,
                    f"Ucrit={hi_val}", color="green")

    ax.axvline(u_calc, color="blue", ls="--")
    place_label(ax, [], u_calc, stats.norm.pdf(u_calc, μ, σ)+0.02,
                f"Ucalc={u_calc}", color="blue")

    ax.set_xlabel("U")
    ax.set_ylabel("Approx. density")
    ax.legend()
    ax.set_title("Mann‑Whitney U")
    fig.tight_layout()
    return fig


def u_crit(n1:int, n2:int, alpha:float, tail:str)->int:
    μ = n1*n2/2
    σ = np.sqrt(n1*n2*(n1+n2+1)/12)
    from math import floor
    z = stats.norm.ppf(alpha if tail.startswith("one") else alpha/2)
    return int(floor(μ + z*σ))


def build_u_table(n1:int, n2:int, alpha:float, tail:str)->str:
    rows = list(range(max(2,n1-5), n1+6))
    cols = list(range(max(2,n2-5), n2+6))
    col_idx = cols.index(n2)+1

    head = "".join(f"<th>{c}</th>" for c in cols)
    body = ""
    for r in rows:
        row_html = f'<td id="u_{r}_0">{r}</td>'
        for i,c in enumerate(cols, start=1):
            val = u_crit(r,c,alpha,tail)
            row_html += f'<td id="u_{r}_{i}">{val}</td>'
        body += f"<tr>{row_html}</tr>"

    code = f"<tr><th>n1\\n2</th>{head}</tr>{body}"
    html = wrap_table(CSS_BASE, code)

    # highlight row, col, intersection
    for i in range(len(cols)+1):
        html = style_cell(html, f"u_{n1}_{i}")
    for rr in rows:
        html = style_cell(html, f"u_{rr}_{col_idx}")
    html = style_cell(html, f"u_{n1}_{col_idx}", color="blue", px=3)
    return html


def u_table(n1:int, n2:int, alpha:float, tail:str):
    code = build_u_table(n1,n2,alpha,tail)
    st.markdown(container(code), unsafe_allow_html=True)


def u_apa(u_val: int, n1: int, n2: int, alpha: float, tail: str):
    """
    Show dynamic explanation and final APA lines for Mann‑Whitney U.
    """
    μ = n1*n2/2
    σ = np.sqrt(n1*n2*(n1+n2+1)/12)

    # sign-based approach for one-tailed
    if tail.startswith("one"):
        if u_val >= μ:
            p_calc = 1 - stats.norm.cdf((u_val-μ)/σ)
        else:
            p_calc = stats.norm.cdf((u_val-μ)/σ)
    else:
        raw = stats.norm.cdf((u_val-μ)/σ)
        p_calc = 2*min(raw, 1-raw)

    cval = u_crit(n1,n2,alpha,tail)
    # rejection region depends on sign
    if tail.startswith("one"):
        reject = (u_val <= cval) if u_val < μ else (u_val >= cval)
    else:
        hi = n1*n2 - cval
        reject = (u_val <= cval) or (u_val >= hi)

    decision = "rejected" if reject else "failed to reject"
    if reject:
        reason_stats = "because U₍calc₎ was in the rejection region"
        reason_p = "because p < α"
    else:
        reason_stats = "because U₍calc₎ was not in the rejection region"
        reason_p = "because p ≥ α"

    # table-based explanation
    normal_cdf_val = stats.norm.cdf((u_val-μ)/σ)
    if tail.startswith("one"):
        if u_val>=μ:
            expl = (
                f"Lookup: P(U ≤ {u_val}) ~ {normal_cdf_val:.4f}, "
                f"but U > mean ⇒ p = 1 − {normal_cdf_val:.4f}"
            )
        else:
            expl = (
                f"Lookup: P(U ≤ {u_val}) ~ {normal_cdf_val:.4f}. "
                f"U < mean ⇒ p = {normal_cdf_val:.4f}"
            )
    else:
        expl = (
            f"Lookup: P(U ≤ {u_val}) ~ {normal_cdf_val:.4f}. "
            f"For two‑tailed ⇒ p = 2 × min({normal_cdf_val:.4f}, {1 - normal_cdf_val:.4f})."
        )

    st.write(expl)

    st.markdown(
        "**APA interpretation**  \n"
        f"Calculated statistic: *U*={u_val}, *p*={p_calc:.3f}.  \n"
        f"Critical statistic: U(crit)={cval}, *p*={alpha:.3f}.  \n"
        f"Statistic comparison → H0 **{decision}** ({reason_stats}).  \n"
        f"*p* comparison → H0 **{decision}** ({reason_p}).  \n"
        f"**APA 7 report:** *U*={u_val}, *p*={p_calc:.3f} "
        f"({tail}). The null hypothesis was **{decision}** at α={alpha:.2f}."
    )


def tab_u():
    st.subheader("Tab 5 • Mann‑Whitney U")

    c1, c2 = st.columns(2)
    with c1:
        u_val = st.number_input("U statistic", value=23, key="u_val")
        n1 = st.number_input("n₁", min_value=2, value=10, step=1, key="u_n1")
    with c2:
        n2 = st.number_input("n₂", min_value=2, value=12, step=1, key="u_n2")
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="u_alpha")
        tail = st.radio("Tail", ["one‑tailed", "two‑tailed"], key="u_tail")

    if st.button("Update Plot", key="u_plot"):
        st.pyplot(plot_u(u_val, n1, n2, alpha, tail))

    st.write("**U‑table** (single highlight)")
    ctable, cexp = st.columns([2,1])
    with ctable:
        u_table(n1, n2, alpha, tail)
        show_cumulative_note()
    with cexp:
        st.subheader("P‑value Calculation Explanation")
        u_apa(u_val, n1, n2, alpha, tail)


###############################################################################
#                    TAB 6: Wilcoxon Signed‑Rank T (unchanged note/p)
###############################################################################

def w_crit(n: int, alpha: float, tail: str)->int:
    μ = n*(n+1)/4
    σ = np.sqrt(n*(n+1)*(2*n+1)/24)
    from math import floor
    z = stats.norm.ppf(alpha if tail.startswith("one") else alpha/2)
    return int(floor(μ + z*σ))


def plot_w(t_calc, n, alpha, tail):
    μ = n*(n+1)/4
    σ = np.sqrt(n*(n+1)*(2*n+1)/24)

    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    xs = np.linspace(μ-4*σ, μ+4*σ, 400)
    ys = stats.norm.pdf(xs, μ, σ)

    ax.plot(xs, ys, "k")
    ax.fill_between(xs, ys, color="lightgrey", alpha=0.25, label="Fail to Reject H0")

    crit_val = w_crit(n, alpha, tail)
    hi_val = n*(n+1)//2 - crit_val

    if tail.startswith("one"):
        if t_calc>=μ:
            ax.fill_between(xs[xs>=crit_val], ys[xs>=crit_val], color="red", alpha=0.3,
                            label="Reject H0")
            ax.axvline(crit_val, color="green", ls="--")
            place_label(ax, [], crit_val, stats.norm.pdf(crit_val, μ, σ)+0.02,
                        f"Tcrit={crit_val}", color="green")
        else:
            ax.fill_between(xs[xs<=crit_val], ys[xs<=crit_val], color="red", alpha=0.3,
                            label="Reject H0")
            ax.axvline(crit_val, color="green", ls="--")
            place_label(ax, [], crit_val, stats.norm.pdf(crit_val, μ, σ)+0.02,
                        f"Tcrit={crit_val}", color="green")
    else:
        ax.fill_between(xs[xs<=crit_val], ys[xs<=crit_val], color="red", alpha=0.3)
        ax.fill_between(xs[xs>=hi_val], ys[xs>=hi_val], color="red", alpha=0.3,
                        label="Reject H0")
        ax.axvline(crit_val, color="green", ls="--")
        ax.axvline(hi_val, color="green", ls="--")
        place_label(ax, [], crit_val, stats.norm.pdf(crit_val, μ, σ)+0.02,
                    f"Tcrit={crit_val}", color="green")
        place_label(ax, [], hi_val, stats.norm.pdf(hi_val, μ, σ)+0.02,
                    f"Tcrit={hi_val}", color="green")

    ax.axvline(t_calc, color="blue", ls="--")
    place_label(ax, [], t_calc, stats.norm.pdf(t_calc, μ, σ)+0.02,
                f"Tcalc={t_calc}", color="blue")

    ax.set_xlabel("T")
    ax.set_ylabel("Approx. density")
    ax.legend()
    ax.set_title("Wilcoxon Signed‑Rank T")
    fig.tight_layout()
    return fig


def build_w_html(n:int, alpha:float, tail:str)->str:
    rows = list(range(max(5,n-5), n+6))
    alphas = [0.10,0.05,0.01,0.001]
    col_idx = alphas.index(alpha)+1

    head = "".join(f"<th>{a}</th>" for a in alphas)
    body = ""
    for r in rows:
        row_html = f'<td id="w_{r}_0">{r}</td>'
        for i,a in enumerate(alphas, start=1):
            val = w_crit(r,a,tail)
            row_html += f'<td id="w_{r}_{i}">{val}</td>'
        body += f"<tr>{row_html}</tr>"

    code = f"<tr><th>N\\α</th>{head}</tr>{body}"
    html = wrap_table(CSS_BASE, code)

    # highlight row
    for i in range(len(alphas)+1):
        html = style_cell(html, f"w_{n}_{i}")
    # highlight col
    for rr in rows:
        html = style_cell(html, f"w_{rr}_{col_idx}")
    # intersection
    html = style_cell(html, f"w_{n}_{col_idx}", color="blue", px=3)
    return html


def w_table(n:int, alpha:float, tail:str):
    code = build_w_html(n, alpha, tail)
    st.markdown(container(code), unsafe_allow_html=True)


def w_apa(t_val: int, n: int, alpha: float, tail: str):
    μ = n*(n+1)/4
    σ = np.sqrt(n*(n+1)*(2*n+1)/24)
    if tail.startswith("one"):
        if t_val>=μ:
            p_calc = 1 - stats.norm.cdf((t_val-μ)/σ)
        else:
            p_calc = stats.norm.cdf((t_val-μ)/σ)
    else:
        raw = stats.norm.cdf((t_val-μ)/σ)
        p_calc = 2*min(raw, 1-raw)

    crit = w_crit(n, alpha, tail)
    hi = n*(n+1)//2 - crit
    if tail.startswith("one"):
        reject = (t_val>=crit if t_val>μ else t_val<=crit)
    else:
        reject = (t_val<=crit) or (t_val>=hi)

    decision = "rejected" if reject else "failed to reject"
    if reject:
        reason_stats = "because T₍calc₎ fell into the rejection region"
        reason_p = "because p < α"
    else:
        reason_stats = "because T₍calc₎ did not fall into the rejection region"
        reason_p = "because p ≥ α"

    normal_cdf_val = stats.norm.cdf((t_val-μ)/σ)
    if tail.startswith("one"):
        if t_val>=μ:
            expl = (
                f"Lookup: P(T ≤ {t_val}) ~ {normal_cdf_val:.4f}, T>mean ⇒ p=1−{normal_cdf_val:.4f}."
            )
        else:
            expl = (
                f"Lookup: P(T ≤ {t_val}) ~ {normal_cdf_val:.4f}, T<mean ⇒ p={normal_cdf_val:.4f}."
            )
    else:
        expl = (
            f"Lookup: P(T ≤ {t_val}) ~ {normal_cdf_val:.4f}. "
            f"Two‑tailed ⇒ p=2×min({normal_cdf_val:.4f},{1-normal_cdf_val:.4f})."
        )
    st.write(expl)

    st.markdown(
        "**APA interpretation**  \n"
        f"Calculated statistic: *T*={t_val}, *p*={p_calc:.3f}.  \n"
        f"Critical statistic: T₍crit₎={crit}, *p*={alpha:.3f}.  \n"
        f"Statistic comparison → H0 **{decision}** ({reason_stats}).  \n"
        f"*p* comparison → H0 **{decision}** ({reason_p}).  \n"
        f"**APA 7 report:** *T*={t_val}, *p*={p_calc:.3f} ({tail}). "
        f"The null hypothesis was **{decision}** at α={alpha:.2f}."
    )


def tab_w():
    st.subheader("Tab 6 • Wilcoxon Signed‑Rank T")

    c1, c2 = st.columns(2)
    with c1:
        t_val = st.number_input("T statistic", value=15, key="w_val")
        n = st.number_input("N (non-zero diffs)", min_value=5, value=12,
                            step=1, key="w_n")
    with c2:
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="w_alpha")
        tail = st.radio("Tail", ["one‑tailed", "two‑tailed"], key="w_tail")

    if st.button("Update Plot", key="w_plot"):
        st.pyplot(plot_w(t_val, n, alpha, tail))

    st.write("**T‑table** (single highlight)")
    ctable, cexp = st.columns([2,1])
    with ctable:
        w_table(n, alpha, tail)
        show_cumulative_note()
    with cexp:
        st.subheader("P‑value Calculation Explanation")
        w_apa(t_val, n, alpha, tail)


###############################################################################
#                           TAB 7: Binomial
###############################################################################

def critical_binom(n: int, p: float, alpha: float):
    """
    Two‑tailed (k_lo, k_hi).
    """
    cum = 0.0
    k_lo = 0
    for k in range(n+1):
        cum += stats.binom.pmf(k, n, p)
        if cum >= alpha/2:
            k_lo = k
            break
    cum = 0.0
    k_hi = n
    for k in range(n, -1, -1):
        cum += stats.binom.pmf(k, n, p)
        if cum >= alpha/2:
            k_hi = k
            break
    return k_lo, k_hi


def plot_binom(k, n, p):
    xs = np.arange(n+1)
    ys = stats.binom.pmf(xs, n, p)
    fig, ax = plt.subplots(figsize=(12,4), dpi=100)
    ax.bar(xs, ys, color="lightgrey")
    ax.bar(k, stats.binom.pmf(k,n,p), color="blue", label=f"k={k}")

    ax.set_xlabel("k")
    ax.set_ylabel("P(X=k)")
    ax.legend()
    ax.set_title(f"Binomial (n={n}, p={p})")
    fig.tight_layout()
    return fig


def build_binom_html(k: int, n: int, p: float) -> str:
    """
    Single-step highlight for binomial table near k±5
    """
    k_vals = list(range(max(0,k-5), min(n,k+5)+1))
    head = "<th>P(X=k)</th><th>P(X≤k)</th><th>P(X≥k)</th>"
    body = ""
    for kv in k_vals:
        pmf_val = stats.binom.pmf(kv,n,p)
        cdf_val = stats.binom.cdf(kv,n,p)
        if kv>0:
            ccdf_val = 1 - stats.binom.cdf(kv-1,n,p)
        else:
            ccdf_val =1.0

        row_html = (
            f'<td id="b_{kv}_0">{kv}</td>'
            f'<td id="b_{kv}_1">{pmf_val:.4f}</td>'
            f'<td id="b_{kv}_2">{cdf_val:.4f}</td>'
            f'<td id="b_{kv}_3">{ccdf_val:.4f}</td>'
        )
        body += f"<tr>{row_html}</tr>"

    table_code = f"<tr><th>k</th>{head}</tr>{body}"
    html = wrap_table(CSS_BASE, table_code)

    # highlight entire row for k
    for i in range(4):
        html = style_cell(html, f"b_{k}_{i}")
    # highlight pmf cell in blue
    html = style_cell(html, f"b_{k}_1", color="blue", px=3)
    return html


def binom_table(k: int, n: int, p: float):
    code = build_binom_html(k,n,p)
    st.markdown(container(code), unsafe_allow_html=True)


def binom_apa(k: int, n: int, p: float, alpha: float, tail: str):
    """
    Show dynamic explanation and final APA lines for the binomial test.
    """
    cdf_val = stats.binom.cdf(k, n, p)
    pmf_val = stats.binom.pmf(k, n, p)
    # two-tailed approach or one-tailed approach
    midpoint = n * p

    if tail.startswith("one"):
        if k >= midpoint:
            # right side
            # p = 1 - P(X ≤ k-1)
            if k>0:
                p_calc = 1.0 - stats.binom.cdf(k-1, n, p)
            else:
                p_calc = 1.0
        else:
            # left side
            p_calc = stats.binom.cdf(k, n, p)
    else:
        # two-tailed
        p_2tail = 2.0 * min(cdf_val, 1.0 - cdf_val + pmf_val)
        p_2tail = min(p_2tail, 1.0)
        p_calc = p_2tail

    k_lo, k_hi = critical_binom(n, p, alpha)
    reject = (k <= k_lo) or (k >= k_hi)
    decision = "rejected" if reject else "failed to reject"
    if reject:
        reason_stats = "because k was within the rejection region"
        reason_p = "because p < α"
    else:
        reason_stats = "because k was not in the rejection region"
        reason_p = "because p ≥ α"

    # Explanation
    st.write(f"Lookup: P(X ≤ {k}) = {cdf_val:.4f}.")

    if tail.startswith("one"):
        if k >= midpoint:
            st.write(
                f"For a **one‑tailed** test (right side), p = 1 − P(X ≤ {k-1}) "
                f"= 1 − {stats.binom.cdf(k-1,n,p):.4f}."
            )
        else:
            st.write(
                f"For a **one‑tailed** test (left side), p = P(X ≤ {k}) "
                f"= {cdf_val:.4f}."
            )
    else:
        st.write(
            f"For a **two‑tailed** test, p = 2 × min({cdf_val:.4f}, "
            f"(1−{cdf_val:.4f})+{pmf_val:.4f}) = {p_calc:.4f} (capped at 1)."
        )

    st.markdown(
        "**APA interpretation**  \n"
        f"Calculated statistic: k={k}, *p*={p_calc:.3f}.  \n"
        f"Critical region: k ≤ {k_lo} or k ≥ {k_hi}, *p*={alpha:.3f}.  \n"
        f"Statistic comparison → H₀ **{decision}** ({reason_stats}).  \n"
        f"*p* comparison → H₀ **{decision}** ({reason_p}).  \n"
        f"**APA 7 report:** Exact binomial test, *p*={p_calc:.3f} "
        f"({tail}). The null hypothesis was **{decision}** at α={alpha:.2f}."
    )


def tab_binom():
    st.subheader("Tab 7 • Binomial")

    c1, c2 = st.columns(2)
    with c1:
        n = st.number_input("n (trials)", min_value=1, value=20, step=1, key="b_n")
        p = st.number_input("π (null proportion)", value=0.50,
                            step=0.01, min_value=0.01, max_value=0.99, key="b_p")
    with c2:
        k = st.number_input("k (successes)", min_value=0, value=10, step=1, key="b_k")
        alpha = st.number_input("α", value=0.05, step=0.01,
                                min_value=0.0001, max_value=0.5, key="b_alpha")
        tail = st.radio("Tail", ["one‑tailed", "two‑tailed"], key="b_tail")

    if st.button("Update Plot", key="b_plot"):
        st.pyplot(plot_binom(k,n,p))

    st.write("**Binomial table** (k±5, single highlight)")
    ctable, cexp = st.columns([2,1])
    with ctable:
        binom_table(k,n,p)
        show_cumulative_note()
    with cexp:
        st.subheader("P‑value Calculation Explanation")
        binom_apa(k,n,p,alpha,tail)


###############################################################################
#                                  MAIN
###############################################################################

def main():
    st.set_page_config("PSYC250 – Statistical Tables Explorer", layout="wide")
    st.title("Oli's – Statistical Table Explorer")

    tabs = st.tabs([
        "t‑Dist", "z‑Dist", "F‑Dist", "Chi‑Square",
        "Mann–Whitney U", "Wilcoxon T", "Binomial"
    ])

    with tabs[0]:
        tab_t()            # (new features)
    with tabs[1]:
        tab_z()            # (new features)
    with tabs[2]:
        tab_f()            # (no new note/p explanation needed)
    with tabs[3]:
        tab_chi()          # (no new note/p explanation needed)
    with tabs[4]:
        tab_u()            # (new features)
    with tabs[5]:
        tab_w()            # (new features)
    with tabs[6]:
        tab_binom()        # (new features)


if __name__ == "__main__":
    main()
