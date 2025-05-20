import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from io import StringIO # For reading CSV string
import itertools
# from scipy.special import comb # Not using for exact Mann-Whitney in this version

# Helper function to create APA style p-value string
def apa_p_value(p_val):
    if not isinstance(p_val, (int, float)) or np.isnan(p_val):
        return "p N/A"
    try:
        p_val_float = float(p_val)
        if p_val_float < 0.001:
            return "p < .001"
        else:
            return f"p = {p_val_float:.3f}"
    except (ValueError, TypeError):
        return "p N/A (format err)"

# Helper function to format critical values or p-values for display
def format_value_for_display(value, decimals=3, default_str="N/A"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default_str
    try:
        return f"{float(value):.{decimals}f}"
    except (ValueError, TypeError):
        return str(value) if value is not None else default_str

def get_dynamic_df_window(all_df_options, selected_df_val, window_size=5):
    """
    Creates a window of df values around the selected_df_val.
    all_df_options: List of all possible df values (can include 'z (∞)' or be numeric).
    selected_df_val: The df value selected by the user (can be 'z (∞)' or numeric).
    window_size: Number of rows to show above and below the selected_df_val.
    Returns a list of df values for the table rows.
    """
    try:
        # Convert 'z (∞)' to a comparable large number for finding index
        temp_selected_df = float('inf') if selected_df_val == 'z (∞)' else float(selected_df_val)
        
        closest_idx = -1
        min_diff = float('inf')

        comparable_options = []
        for option in all_df_options:
            if option == 'z (∞)':
                comparable_options.append(float('inf'))
            elif isinstance(option, (int, float)) or (isinstance(option, str) and option.replace('.', '', 1).isdigit()):
                comparable_options.append(float(option))
            else: 
                comparable_options.append(float('-inf')) 

        for i, option_val_numeric in enumerate(comparable_options):
            diff = abs(option_val_numeric - temp_selected_df)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
            elif diff == min_diff and option_val_numeric == temp_selected_df:
                closest_idx = i
        
        if closest_idx == -1: 
            if selected_df_val in all_df_options:
                try:
                    closest_idx = all_df_options.index(selected_df_val)
                except ValueError: 
                    return all_df_options[:window_size*2+1]
            else: 
                return all_df_options[:window_size*2+1]

        start_idx = max(0, closest_idx - window_size)
        end_idx = min(len(all_df_options), closest_idx + window_size + 1)
        
        windowed_df_options = all_df_options[start_idx:end_idx]
        
        return windowed_df_options
    except Exception: 
        return all_df_options[:min(len(all_df_options), window_size*2+1)]


# --- Tab 1: t-distribution ---
# (Code remains the same as the last fully correct version)
def tab_t_distribution():
    st.header("t-Distribution Explorer")
    col1, col2 = st.columns([2, 1.5]) 

    with col1:
        st.subheader("Inputs")
        alpha_t_input = st.number_input("Alpha (α)", 0.0001, 0.5, 0.05, 0.0001, format="%.4f", key="alpha_t_input")
        
        all_df_values_t = list(range(1, 31)) + [35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 1000, 'z (∞)']
        df_t_selected_display = st.selectbox("Degrees of Freedom (df)", options=all_df_values_t, index=all_df_values_t.index(10), key="df_t_selectbox") 

        if df_t_selected_display == 'z (∞)':
            df_t_calc = np.inf 
        else:
            df_t_calc = int(df_t_selected_display)

        tail_t = st.radio("Tail Selection", ("Two-tailed", "One-tailed (right)", "One-tailed (left)"), key="tail_t_radio")
        test_stat_t = st.number_input("Calculated t-statistic", value=0.0, format="%.3f", key="test_stat_t_input")

        st.subheader("Distribution Plot")
        fig_t, ax_t = plt.subplots(figsize=(8,5)) 
        
        if np.isinf(df_t_calc): 
            dist_label_plot = 'Standard Normal (z)'
            crit_func_ppf_plot = stats.norm.ppf
            crit_func_pdf_plot = stats.norm.pdf
            std_dev_plot = 1.0
        else: 
            dist_label_plot = f't-distribution (df={df_t_calc})'
            crit_func_ppf_plot = lambda q_val: stats.t.ppf(q_val, df_t_calc)
            crit_func_pdf_plot = lambda x_val: stats.t.pdf(x_val, df_t_calc)
            std_dev_plot = stats.t.std(df_t_calc) if df_t_calc > 0 and not np.isinf(df_t_calc) else 1.0
        
        plot_min_t = min(crit_func_ppf_plot(0.0001), test_stat_t - 2*std_dev_plot, -4.0)
        plot_max_t = max(crit_func_ppf_plot(0.9999), test_stat_t + 2*std_dev_plot, 4.0)
        if abs(test_stat_t) > 4 and abs(test_stat_t) > plot_max_t * 0.8 : 
            plot_min_t = min(plot_min_t, test_stat_t -1)
            plot_max_t = max(plot_max_t, test_stat_t +1)
        
        x_t_plot = np.linspace(plot_min_t, plot_max_t, 500) 
        y_t_plot = crit_func_pdf_plot(x_t_plot)
        ax_t.plot(x_t_plot, y_t_plot, 'b-', lw=2, label=dist_label_plot)
        
        crit_val_t_upper_plot, crit_val_t_lower_plot = None, None
        if tail_t == "Two-tailed":
            crit_val_t_upper_plot = crit_func_ppf_plot(1 - alpha_t_input / 2)
            crit_val_t_lower_plot = crit_func_ppf_plot(alpha_t_input / 2)
            if crit_val_t_upper_plot is not None and not np.isnan(crit_val_t_upper_plot):
                 x_fill_upper = np.linspace(crit_val_t_upper_plot, plot_max_t, 100)
                 ax_t.fill_between(x_fill_upper, crit_func_pdf_plot(x_fill_upper), color='red', alpha=0.5, label=f'α/2 = {alpha_t_input/2:.4f}')
                 ax_t.axvline(crit_val_t_upper_plot, color='red', linestyle='--', lw=1)
            if crit_val_t_lower_plot is not None and not np.isnan(crit_val_t_lower_plot):
                 x_fill_lower = np.linspace(plot_min_t, crit_val_t_lower_plot, 100)
                 ax_t.fill_between(x_fill_lower, crit_func_pdf_plot(x_fill_lower), color='red', alpha=0.5)
                 ax_t.axvline(crit_val_t_lower_plot, color='red', linestyle='--', lw=1)
        elif tail_t == "One-tailed (right)":
            crit_val_t_upper_plot = crit_func_ppf_plot(1 - alpha_t_input)
            if crit_val_t_upper_plot is not None and not np.isnan(crit_val_t_upper_plot):
                x_fill_upper = np.linspace(crit_val_t_upper_plot, plot_max_t, 100)
                ax_t.fill_between(x_fill_upper, crit_func_pdf_plot(x_fill_upper), color='red', alpha=0.5, label=f'α = {alpha_t_input:.4f}')
                ax_t.axvline(crit_val_t_upper_plot, color='red', linestyle='--', lw=1)
        else: # One-tailed (left)
            crit_val_t_lower_plot = crit_func_ppf_plot(alpha_t_input)
            if crit_val_t_lower_plot is not None and not np.isnan(crit_val_t_lower_plot):
                x_fill_lower = np.linspace(plot_min_t, crit_val_t_lower_plot, 100)
                ax_t.fill_between(x_fill_lower, crit_func_pdf_plot(x_fill_lower), color='red', alpha=0.5, label=f'α = {alpha_t_input:.4f}')
                ax_t.axvline(crit_val_t_lower_plot, color='red', linestyle='--', lw=1)

        ax_t.axvline(test_stat_t, color='green', linestyle='-', lw=2, label=f'Test Stat = {test_stat_t:.3f}')
        ax_t.set_title(f'{dist_label_plot} with Critical Region(s)')
        ax_t.set_xlabel('t-value' if not np.isinf(df_t_calc) else 'z-value')
        ax_t.set_ylabel('Probability Density')
        ax_t.legend()
        ax_t.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_t)

        st.subheader("Critical t-Values (Upper Tail)")
        
        table_df_window = get_dynamic_df_window(all_df_values_t, df_t_selected_display, window_size=5)
        table_alpha_cols = [0.10, 0.05, 0.025, 0.01, 0.005] 

        table_rows = []
        for df_iter_display in table_df_window:
            df_iter_calc = np.inf if df_iter_display == 'z (∞)' else int(df_iter_display)
            row_data = {'df': str(df_iter_display)}
            for alpha_col in table_alpha_cols:
                if np.isinf(df_iter_calc):
                    cv = stats.norm.ppf(1 - alpha_col)
                else:
                    cv = stats.t.ppf(1 - alpha_col, df_iter_calc)
                row_data[f"α = {alpha_col:.3f}"] = format_value_for_display(cv)
            table_rows.append(row_data)
        
        df_t_table = pd.DataFrame(table_rows).set_index('df')

        def style_t_table(df_to_style):
            style = pd.DataFrame('', index=df_to_style.index, columns=df_to_style.columns)
            selected_df_str = str(df_t_selected_display) 

            if selected_df_str in df_to_style.index: 
                style.loc[selected_df_str, :] = 'background-color: lightblue;'

            target_alpha_for_col_highlight = alpha_t_input 
            if tail_t == "Two-tailed":
                target_alpha_for_col_highlight = alpha_t_input / 2.0
            
            closest_alpha_col_val = min(table_alpha_cols, key=lambda x: abs(x - target_alpha_for_col_highlight))
            highlight_col_name = f"α = {closest_alpha_col_val:.3f}"

            if highlight_col_name in df_to_style.columns:
                for r_idx in df_to_style.index:
                     current_r_style = style.loc[r_idx, highlight_col_name]
                     style.loc[r_idx, highlight_col_name] = (current_r_style + ';' if current_r_style and not current_r_style.endswith(';') else current_r_style) + 'background-color: lightgreen;'
                
                if selected_df_str in df_to_style.index: 
                    current_c_style = style.loc[selected_df_str, highlight_col_name]
                    style.loc[selected_df_str, highlight_col_name] = (current_c_style + ';' if current_c_style and not current_c_style.endswith(';') else '') + 'font-weight: bold; border: 2px solid red; background-color: yellow;'
            return style

        st.markdown(df_t_table.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                       {'selector': 'td', 'props': [('text-align', 'center')]}])
                                     .apply(style_t_table, axis=None).to_html(), unsafe_allow_html=True)
        st.caption(f"Table shows upper-tail critical values. Highlighted row for df='{df_t_selected_display}', column for α closest to your test, and specific cell in red.")
        st.markdown("""
        **Table Interpretation Note:**
        * The table displays upper-tail critical values (t<sub>α</sub>).
        * For **One-tailed (right) tests**, use the α column matching your chosen significance level.
        * For **One-tailed (left) tests**, use the α column matching your chosen significance level and take the *negative* of the table value.
        * For **Two-tailed tests**, if your total significance level is α<sub>total</sub>, look up the column for α = α<sub>total</sub>/2. The critical values are ± the table value.
        """)

    with col2: # Summary section
        st.subheader("P-value Calculation Explanation")
        if np.isinf(df_t_calc):
            p_val_func_sf = stats.norm.sf
            p_val_func_cdf = stats.norm.cdf
            dist_name_p_summary = "Z"
        else:
            p_val_func_sf = lambda val: stats.t.sf(val, df_t_calc)
            p_val_func_cdf = lambda val: stats.t.cdf(val, df_t_calc)
            dist_name_p_summary = "T"

        st.markdown(f"""
        The p-value is the probability of observing a test statistic as extreme as, or more extreme than, the calculated statistic ({test_stat_t:.3f}), assuming the null hypothesis is true.
        * For a **two-tailed test**, it's `2 * P({dist_name_p_summary} ≥ |{test_stat_t:.3f}|)`.
        * For a **one-tailed (right) test**, it's `P({dist_name_p_summary} ≥ {test_stat_t:.3f})`.
        * For a **one-tailed (left) test**, it's `P({dist_name_p_summary} ≤ {test_stat_t:.3f})`.
        """)

        st.subheader("Summary")
        p_val_t_one_right_summary = p_val_func_sf(test_stat_t)
        p_val_t_one_left_summary = p_val_func_cdf(test_stat_t)
        p_val_t_two_summary = 2 * p_val_func_sf(abs(test_stat_t))
        p_val_t_two_summary = min(p_val_t_two_summary, 1.0) 

        crit_val_display_summary = "N/A"
        p_val_for_crit_val_display_summary = alpha_t_input 

        if tail_t == "Two-tailed":
            crit_val_display_summary = f"±{format_value_for_display(crit_val_t_upper_plot)}" if crit_val_t_upper_plot is not None else "N/A"
            p_val_calc_summary = p_val_t_two_summary
            decision_crit_summary = abs(test_stat_t) > crit_val_t_upper_plot if crit_val_t_upper_plot is not None and not np.isnan(crit_val_t_upper_plot) else False
            comparison_crit_str_summary = f"|{test_stat_t:.3f}| ({abs(test_stat_t):.3f}) > {format_value_for_display(crit_val_t_upper_plot)}" if decision_crit_summary else f"|{test_stat_t:.3f}| ({abs(test_stat_t):.3f}) ≤ {format_value_for_display(crit_val_t_upper_plot)}"
        elif tail_t == "One-tailed (right)":
            crit_val_display_summary = format_value_for_display(crit_val_t_upper_plot) if crit_val_t_upper_plot is not None else "N/A"
            p_val_calc_summary = p_val_t_one_right_summary
            decision_crit_summary = test_stat_t > crit_val_t_upper_plot if crit_val_t_upper_plot is not None and not np.isnan(crit_val_t_upper_plot) else False
            comparison_crit_str_summary = f"{test_stat_t:.3f} > {format_value_for_display(crit_val_t_upper_plot)}" if decision_crit_summary else f"{test_stat_t:.3f} ≤ {format_value_for_display(crit_val_t_upper_plot)}"
        else: # One-tailed (left)
            crit_val_display_summary = format_value_for_display(crit_val_t_lower_plot) if crit_val_t_lower_plot is not None else "N/A"
            p_val_calc_summary = p_val_t_one_left_summary
            decision_crit_summary = test_stat_t < crit_val_t_lower_plot if crit_val_t_lower_plot is not None and not np.isnan(crit_val_t_lower_plot) else False
            comparison_crit_str_summary = f"{test_stat_t:.3f} < {format_value_for_display(crit_val_t_lower_plot)}" if decision_crit_summary else f"{test_stat_t:.3f} ≥ {format_value_for_display(crit_val_t_lower_plot)}"

        decision_p_alpha_summary = p_val_calc_summary < alpha_t_input
        
        df_report_str_summary = "∞" if np.isinf(df_t_calc) else str(df_t_calc)
        stat_symbol_summary = "z" if np.isinf(df_t_calc) else "t"

        st.markdown(f"""
        1.  **Critical Value ({tail_t})**: {crit_val_display_summary}
            * *Associated p-value (α or α/2 per tail)*: {p_val_for_crit_val_display_summary:.4f}
        2.  **Calculated Test Statistic**: {test_stat_t:.3f}
            * *Calculated p-value*: {format_value_for_display(p_val_calc_summary, decimals=4)} ({apa_p_value(p_val_calc_summary)})
        3.  **Decision (Critical Value Method)**: The null hypothesis is **{'rejected' if decision_crit_summary else 'not rejected'}**.
            * *Reason*: Because {stat_symbol_summary}(calc) {comparison_crit_str_summary} relative to {stat_symbol_summary}(crit).
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_summary else 'not rejected'}**.
            * *Reason*: Because {apa_p_value(p_val_calc_summary)} is {'less than' if decision_p_alpha_summary else 'not less than'} α ({alpha_t_input:.4f}).
        5.  **APA 7 Style Report**:
            *{stat_symbol_summary}*({df_report_str_summary}) = {test_stat_t:.2f}, {apa_p_value(p_val_calc_summary)}. The null hypothesis was {'rejected' if decision_p_alpha_summary else 'not rejected'} at the α = {alpha_t_input:.2f} level.
        """)

# --- Tab 2: z-distribution ---
def tab_z_distribution():
    st.header("z-Distribution (Standard Normal) Explorer")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_z_hyp = st.number_input("Alpha (α) for Hypothesis Test", 0.0001, 0.5, 0.05, 0.0001, format="%.4f", key="alpha_z_hyp_key")
        tail_z_hyp = st.radio("Tail Selection for Hypothesis Test", ("Two-tailed", "One-tailed (right)", "One-tailed (left)"), key="tail_z_hyp_key")
        test_stat_z_hyp = st.number_input("Your Calculated z-statistic (also for Table Lookup)", value=0.0, format="%.3f", key="test_stat_z_hyp_key", min_value=-3.99, max_value=3.99, step=0.01)
        
        z_lookup_val = test_stat_z_hyp 

        st.subheader("Distribution Plot")
        fig_z, ax_z = plt.subplots(figsize=(8,5))
        
        plot_min_z = min(stats.norm.ppf(0.00001), test_stat_z_hyp - 2, -4.0) 
        plot_max_z = max(stats.norm.ppf(0.99999), test_stat_z_hyp + 2, 4.0) 
        if abs(test_stat_z_hyp) > 3.5 : 
            plot_min_z = min(plot_min_z, test_stat_z_hyp - 0.5)
            plot_max_z = max(plot_max_z, test_stat_z_hyp + 0.5)

        x_z_plot = np.linspace(plot_min_z, plot_max_z, 500)
        y_z_plot = stats.norm.pdf(x_z_plot)
        ax_z.plot(x_z_plot, y_z_plot, 'b-', lw=2, label='Standard Normal Distribution (z)')

        x_fill_lookup = np.linspace(plot_min_z, z_lookup_val, 100)
        ax_z.fill_between(x_fill_lookup, stats.norm.pdf(x_fill_lookup), color='skyblue', alpha=0.5, label=f'P(Z < {z_lookup_val:.3f})')
        
        crit_val_z_upper_plot, crit_val_z_lower_plot = None, None 
        if tail_z_hyp == "Two-tailed":
            crit_val_z_upper_plot = stats.norm.ppf(1 - alpha_z_hyp / 2)
            crit_val_z_lower_plot = stats.norm.ppf(alpha_z_hyp / 2)
            if crit_val_z_upper_plot is not None and not np.isnan(crit_val_z_upper_plot):
                x_fill_upper = np.linspace(crit_val_z_upper_plot, plot_max_z, 100)
                ax_z.fill_between(x_fill_upper, stats.norm.pdf(x_fill_upper), color='red', alpha=0.3, label=f'Crit. Region α/2')
                ax_z.axvline(crit_val_z_upper_plot, color='red', linestyle='--', lw=1)
            if crit_val_z_lower_plot is not None and not np.isnan(crit_val_z_lower_plot):
                x_fill_lower = np.linspace(plot_min_z, crit_val_z_lower_plot, 100)
                ax_z.fill_between(x_fill_lower, stats.norm.pdf(x_fill_lower), color='red', alpha=0.3)
                ax_z.axvline(crit_val_z_lower_plot, color='red', linestyle='--', lw=1)
        elif tail_z_hyp == "One-tailed (right)":
            crit_val_z_upper_plot = stats.norm.ppf(1 - alpha_z_hyp)
            if crit_val_z_upper_plot is not None and not np.isnan(crit_val_z_upper_plot):
                x_fill_upper = np.linspace(crit_val_z_upper_plot, plot_max_z, 100)
                ax_z.fill_between(x_fill_upper, stats.norm.pdf(x_fill_upper), color='red', alpha=0.3, label=f'Crit. Region α')
                ax_z.axvline(crit_val_z_upper_plot, color='red', linestyle='--', lw=1)
        else: 
            crit_val_z_lower_plot = stats.norm.ppf(alpha_z_hyp)
            if crit_val_z_lower_plot is not None and not np.isnan(crit_val_z_lower_plot):
                x_fill_lower = np.linspace(plot_min_z, crit_val_z_lower_plot, 100)
                ax_z.fill_between(x_fill_lower, stats.norm.pdf(x_fill_lower), color='red', alpha=0.3, label=f'Crit. Region α')
                ax_z.axvline(crit_val_z_lower_plot, color='red', linestyle='--', lw=1)

        ax_z.axvline(test_stat_z_hyp, color='green', linestyle='-', lw=2, label=f'Your z-stat = {test_stat_z_hyp:.3f}')
        ax_z.set_title('Standard Normal Distribution')
        ax_z.set_xlabel('z-value')
        ax_z.set_ylabel('Probability Density')
        ax_z.legend(fontsize='small')
        ax_z.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_z)

        st.subheader("Standard Normal Table: Cumulative P(Z < z)")
        st.markdown("This table shows the area to the left of a given z-score. Your calculated z-statistic is used for highlighting.")
        
        all_z_row_labels = [f"{val:.1f}" for val in np.round(np.arange(-3.4, 3.5, 0.1), 1)]
        z_col_labels_str = [f"{val:.2f}" for val in np.round(np.arange(0.00, 0.10, 0.01), 2)]

        z_target_for_table_row_numeric = round(test_stat_z_hyp, 1) 
        try:
            closest_row_idx = min(range(len(all_z_row_labels)), key=lambda i: abs(float(all_z_row_labels[i]) - z_target_for_table_row_numeric))
        except ValueError: 
            closest_row_idx = len(all_z_row_labels) // 2

        window_size = 5
        start_idx = max(0, closest_row_idx - window_size)
        end_idx = min(len(all_z_row_labels), closest_row_idx + window_size + 1)
        z_table_display_rows_str = all_z_row_labels[start_idx:end_idx]


        table_data_z_lookup = []
        for z_r_str_idx in z_table_display_rows_str:
            z_r_val = float(z_r_str_idx)
            row = { 'z': z_r_str_idx } 
            for z_c_str_idx in z_col_labels_str:
                z_c_val = float(z_c_str_idx)
                current_z_val = round(z_r_val + z_c_val, 2)
                prob = stats.norm.cdf(current_z_val)
                row[z_c_str_idx] = format_value_for_display(prob, decimals=4)
            table_data_z_lookup.append(row)
        
        df_z_lookup_table = pd.DataFrame(table_data_z_lookup).set_index('z')

        def style_z_lookup_table(df_to_style):
            data = df_to_style 
            style_df = pd.DataFrame('', index=data.index, columns=data.columns)
            
            try:
                z_target = test_stat_z_hyp 
                
                # Determine closest row label string
                z_target_base_numeric = round(z_target,1) 
                actual_row_labels_float = [float(label) for label in data.index]
                closest_row_float_val = min(actual_row_labels_float, key=lambda x_val: abs(x_val - z_target_base_numeric))
                highlight_row_label = f"{closest_row_float_val:.1f}"

                # Determine closest column label string
                z_target_second_decimal_part = round(abs(z_target - closest_row_float_val), 2) 
                
                actual_col_labels_float = [float(col_str) for col_str in data.columns]
                closest_col_float_val = min(actual_col_labels_float, key=lambda x_val: abs(x_val - z_target_second_decimal_part))
                highlight_col_label = f"{closest_col_float_val:.2f}"


                if highlight_row_label in style_df.index:
                    for col_name_iter in style_df.columns: 
                        style_df.loc[highlight_row_label, col_name_iter] = 'background-color: lightblue;'
                
                if highlight_col_label in style_df.columns:
                    for r_idx_iter in style_df.index: 
                        current_style = style_df.loc[r_idx_iter, highlight_col_label]
                        style_df.loc[r_idx_iter, highlight_col_label] = (current_style + ';' if current_style and not current_style.endswith(';') else current_style) + 'background-color: lightgreen;'
                
                if highlight_row_label in style_df.index and highlight_col_label in style_df.columns:
                    current_cell_style = style_df.loc[highlight_row_label, highlight_col_label]
                    style_df.loc[highlight_row_label, highlight_col_label] = (current_cell_style + ';' if current_cell_style and not current_cell_style.endswith(';') else '') + 'font-weight: bold; border: 2px solid red; background-color: yellow;'
            
            except Exception: 
                pass 
            return style_df
        
        st.markdown(df_z_lookup_table.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                               {'selector': 'td', 'props': [('text-align', 'center')]}])
                                     .apply(style_z_lookup_table, axis=None).to_html(), unsafe_allow_html=True)
        st.caption(f"Table shows P(Z < z). Highlighted cell, row, and column are closest to your calculated z-statistic of {test_stat_z_hyp:.3f}.")


    with col2: # Summary for Z-distribution hypothesis test
        st.subheader("Hypothesis Test Summary")
        st.markdown(f"""
        Based on your inputs for hypothesis testing (α = {alpha_z_hyp:.4f}, {tail_z_hyp}):
        """)
        p_val_z_one_right_summary = stats.norm.sf(test_stat_z_hyp)
        p_val_z_one_left_summary = stats.norm.cdf(test_stat_z_hyp)
        p_val_z_two_summary = 2 * stats.norm.sf(abs(test_stat_z_hyp))
        p_val_z_two_summary = min(p_val_z_two_summary, 1.0)

        crit_val_display_z = "N/A"
        p_val_for_crit_val_display_z = alpha_z_hyp

        if tail_z_hyp == "Two-tailed":
            crit_val_display_z = f"±{format_value_for_display(crit_val_z_upper_plot)}" if crit_val_z_upper_plot is not None else "N/A"
            p_val_calc_z_summary = p_val_z_two_summary
            decision_crit_z_summary = abs(test_stat_z_hyp) > crit_val_z_upper_plot if crit_val_z_upper_plot is not None and not np.isnan(crit_val_z_upper_plot) else False
            comparison_crit_str_z = f"|{test_stat_z_hyp:.3f}| ({abs(test_stat_z_hyp):.3f}) > {format_value_for_display(crit_val_z_upper_plot)}" if decision_crit_z_summary else f"|{test_stat_z_hyp:.3f}| ({abs(test_stat_z_hyp):.3f}) ≤ {format_value_for_display(crit_val_z_upper_plot)}"
        elif tail_z_hyp == "One-tailed (right)":
            crit_val_display_z = format_value_for_display(crit_val_z_upper_plot) if crit_val_z_upper_plot is not None else "N/A"
            p_val_calc_z_summary = p_val_z_one_right_summary
            decision_crit_z_summary = test_stat_z_hyp > crit_val_z_upper_plot if crit_val_z_upper_plot is not None and not np.isnan(crit_val_z_upper_plot) else False
            comparison_crit_str_z = f"{test_stat_z_hyp:.3f} > {format_value_for_display(crit_val_z_upper_plot)}" if decision_crit_z_summary else f"{test_stat_z_hyp:.3f} ≤ {format_value_for_display(crit_val_z_upper_plot)}"
        else: # One-tailed (left)
            crit_val_display_z = format_value_for_display(crit_val_z_lower_plot) if crit_val_z_lower_plot is not None else "N/A"
            p_val_calc_z_summary = p_val_z_one_left_summary
            decision_crit_z_summary = test_stat_z_hyp < crit_val_z_lower_plot if crit_val_z_lower_plot is not None and not np.isnan(crit_val_z_lower_plot) else False
            comparison_crit_str_z = f"{test_stat_z_hyp:.3f} < {format_value_for_display(crit_val_z_lower_plot)}" if decision_crit_z_summary else f"{test_stat_z_hyp:.3f} ≥ {format_value_for_display(crit_val_z_lower_plot)}"

        decision_p_alpha_z_summary = p_val_calc_z_summary < alpha_z_hyp
        
        st.markdown(f"""
        1.  **Critical Value ({tail_z_hyp})**: {crit_val_display_z}
            * *Associated p-value (α or α/2 per tail)*: {p_val_for_crit_val_display_z:.4f}
        2.  **Your Calculated Test Statistic**: {test_stat_z_hyp:.3f}
            * *Calculated p-value*: {format_value_for_display(p_val_calc_z_summary, decimals=4)} ({apa_p_value(p_val_calc_z_summary)})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_z_summary else 'not rejected'}**.
            * *Reason*: z(calc) {comparison_crit_str_z} relative to z(crit).
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_z_summary else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_calc_z_summary)} is {'less than' if decision_p_alpha_z_summary else 'not less than'} α ({alpha_z_hyp:.4f}).
        5.  **APA 7 Style Report**:
            *z* = {test_stat_z_hyp:.2f}, {apa_p_value(p_val_calc_z_summary)}. The null hypothesis was {'rejected' if decision_p_alpha_z_summary else 'not rejected'} at α = {alpha_z_hyp:.2f}.
        """)


# --- Tab 3: F-distribution ---
def tab_f_distribution():
    st.header("F-Distribution Explorer")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_f_input = st.number_input("Alpha (α) for Table/Plot", 0.0001, 0.5, 0.05, 0.0001, format="%.4f", key="alpha_f_input_tab3")
        
        all_df1_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 24, 30, 40, 50, 60, 80, 100, 120, 1000]
        all_df2_options = list(range(1,21)) + [22, 24, 26, 28, 30, 35, 40, 45, 50, 60, 80, 100, 120, 1000] 

        df1_f_selected = st.selectbox("Numerator df (df₁) for Plot & Summary", options=all_df1_options, index=all_df1_options.index(3), key="df1_f_selectbox_tab3") 
        df2_f_selected = st.selectbox("Denominator df (df₂) for Plot & Summary", options=all_df2_options, index=all_df2_options.index(20), key="df2_f_selectbox_tab3") 
        
        tail_f = st.radio("Tail Selection (for plot & summary)", ("One-tailed (right)", "Two-tailed (for variance test)"), key="tail_f_radio_tab3")
        test_stat_f = st.number_input("Calculated F-statistic", value=1.0, format="%.3f", min_value=0.001, key="test_stat_f_input_tab3")

        st.subheader("Distribution Plot")
        fig_f, ax_f = plt.subplots(figsize=(8,5))
        
        plot_min_f = 0.001
        plot_max_f = 5.0 
        try:
            plot_max_f = max(stats.f.ppf(0.999, df1_f_selected, df2_f_selected), test_stat_f * 1.5, 5.0)
            if test_stat_f > stats.f.ppf(0.999, df1_f_selected, df2_f_selected) * 1.2 : 
                plot_max_f = test_stat_f * 1.2
        except Exception: pass

        x_f_plot = np.linspace(plot_min_f, plot_max_f, 500)
        y_f_plot = stats.f.pdf(x_f_plot, df1_f_selected, df2_f_selected)
        ax_f.plot(x_f_plot, y_f_plot, 'b-', lw=2, label=f'F-dist (df₁={df1_f_selected}, df₂={df2_f_selected})')

        crit_val_f_upper_plot, crit_val_f_lower_plot = None, None
        alpha_for_plot = alpha_f_input 
        if tail_f == "One-tailed (right)":
            crit_val_f_upper_plot = stats.f.ppf(1 - alpha_for_plot, df1_f_selected, df2_f_selected)
            if crit_val_f_upper_plot is not None and not np.isnan(crit_val_f_upper_plot):
                x_fill_upper = np.linspace(crit_val_f_upper_plot, plot_max_f, 100)
                ax_f.fill_between(x_fill_upper, stats.f.pdf(x_fill_upper, df1_f_selected, df2_f_selected), color='red', alpha=0.5, label=f'α = {alpha_for_plot:.4f}')
                ax_f.axvline(crit_val_f_upper_plot, color='red', linestyle='--', lw=1)
        else: 
            crit_val_f_upper_plot = stats.f.ppf(1 - alpha_for_plot / 2, df1_f_selected, df2_f_selected)
            crit_val_f_lower_plot = stats.f.ppf(alpha_for_plot / 2, df1_f_selected, df2_f_selected)
            if crit_val_f_upper_plot is not None and not np.isnan(crit_val_f_upper_plot):
                x_fill_upper = np.linspace(crit_val_f_upper_plot, plot_max_f, 100)
                ax_f.fill_between(x_fill_upper, stats.f.pdf(x_fill_upper, df1_f_selected, df2_f_selected), color='red', alpha=0.5, label=f'α/2 = {alpha_for_plot/2:.4f}')
                ax_f.axvline(crit_val_f_upper_plot, color='red', linestyle='--', lw=1)
            if crit_val_f_lower_plot is not None and not np.isnan(crit_val_f_lower_plot):
                x_fill_lower = np.linspace(plot_min_f, crit_val_f_lower_plot, 100)
                ax_f.fill_between(x_fill_lower, stats.f.pdf(x_fill_lower, df1_f_selected, df2_f_selected), color='red', alpha=0.5)
                ax_f.axvline(crit_val_f_lower_plot, color='red', linestyle='--', lw=1)

        ax_f.axvline(test_stat_f, color='green', linestyle='-', lw=2, label=f'Test Stat = {test_stat_f:.3f}')
        ax_f.set_title(f'F-Distribution (df₁={df1_f_selected}, df₂={df2_f_selected}) with Critical Region(s)')
        ax_f.set_xlabel('F-value')
        ax_f.set_ylabel('Probability Density')
        ax_f.legend()
        ax_f.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_f)

        st.subheader(f"Critical F-Values for α = {alpha_f_input:.3f} (Upper Tail)")
        table_df1_display_cols = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 60, 120] 
        table_df2_window = get_dynamic_df_window(all_df2_options, df2_f_selected, window_size=5)


        f_table_data = []
        for df2_val_iter in table_df2_window: 
            df2_val_calc = int(df2_val_iter) 
            row = {'df₂': str(df2_val_iter)} 
            for df1_val_iter in table_df1_display_cols: 
                cv = stats.f.ppf(1 - alpha_f_input, df1_val_iter, df2_val_calc)
                row[f"df₁={df1_val_iter}"] = format_value_for_display(cv)
            f_table_data.append(row)
        
        df_f_table = pd.DataFrame(f_table_data).set_index('df₂')

        def style_f_table(df_to_style):
            style = pd.DataFrame('', index=df_to_style.index, columns=df_to_style.columns)
            selected_df2_str = str(df2_f_selected) 
            
            if selected_df2_str in df_to_style.index:
                style.loc[selected_df2_str, :] = 'background-color: lightblue;'
            
            closest_df1_col_val = min(table_df1_display_cols, key=lambda x: abs(x - df1_f_selected))
            highlight_col_name_f = f"df₁={closest_df1_col_val}"

            if highlight_col_name_f in df_to_style.columns:
                for r_idx in df_to_style.index:
                    current_r_style = style.loc[r_idx, highlight_col_name_f]
                    style.loc[r_idx, highlight_col_name_f] = (current_r_style + ';' if current_r_style and not current_r_style.endswith(';') else current_r_style) + 'background-color: lightgreen;'
                
                if selected_df2_str in df_to_style.index : 
                    current_c_style = style.loc[selected_df2_str, highlight_col_name_f]
                    style.loc[selected_df2_str, highlight_col_name_f] = (current_c_style + ';' if current_c_style and not current_c_style.endswith(';') else '') + 'font-weight: bold; border: 2px solid red; background-color: yellow;'
            return style

        st.markdown(df_f_table.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                       {'selector': 'td', 'props': [('text-align', 'center')]}])
                                     .apply(style_f_table, axis=None).to_html(), unsafe_allow_html=True)
        st.caption(f"Table shows F-critical values for user-selected α={alpha_f_input:.3f} (upper tail). Highlighted for df₁ closest to {df1_f_selected} and df₂ closest to {df2_f_selected}.")
        st.markdown("""
        **Table Interpretation Note:**
        * This table shows upper-tail critical values F<sub>α, df₁, df₂</sub> for the selected α.
        * For **ANOVA (typically one-tailed right)**, use this table directly with your chosen α.
        * For **Two-tailed variance tests** (H₀: σ₁²=σ₂² vs H₁: σ₁²≠σ₂²), you need two critical values:
            * Upper: F<sub>α/2, df₁, df₂</sub> (Look up using α/2 with this table).
            * Lower: F<sub>1-α/2, df₁, df₂</sub> = 1 / F<sub>α/2, df₂, df₁</sub> (requires swapping df and taking reciprocal of value from table looked up with α/2).
        """)

    with col2: # Summary for F-distribution
        st.subheader("P-value Calculation Explanation")
        st.markdown(f"""
        The p-value is the probability of an F-statistic as extreme as, or more extreme than, {test_stat_f:.3f}.
        * **One-tailed (right)**: `P(F ≥ {test_stat_f:.3f})` (i.e., `stats.f.sf(test_stat_f, df1_f_selected, df2_f_selected)`)
        * **Two-tailed (for variance test)**: `2 * min(P(F ≤ F_calc), P(F ≥ F_calc))` (i.e., `2 * min(stats.f.cdf(test_stat_f, df1_f_selected, df2_f_selected), stats.f.sf(test_stat_f, df1_f_selected, df2_f_selected))`)
        """)

        st.subheader("Summary")
        p_val_f_one_right_summary = stats.f.sf(test_stat_f, df1_f_selected, df2_f_selected)
        cdf_f_summary = stats.f.cdf(test_stat_f, df1_f_selected, df2_f_selected)
        sf_f_summary = stats.f.sf(test_stat_f, df1_f_selected, df2_f_selected)
        p_val_f_two_summary = 2 * min(cdf_f_summary, sf_f_summary)
        p_val_f_two_summary = min(p_val_f_two_summary, 1.0)

        crit_val_f_display_summary = "N/A"
        p_val_for_crit_val_f_display_summary = alpha_f_input 

        if tail_f == "One-tailed (right)":
            crit_val_f_display_summary = format_value_for_display(crit_val_f_upper_plot) if crit_val_f_upper_plot is not None else "N/A"
            p_val_calc_f_summary = p_val_f_one_right_summary
            decision_crit_f_summary = test_stat_f > crit_val_f_upper_plot if crit_val_f_upper_plot is not None and not np.isnan(crit_val_f_upper_plot) else False
            comparison_crit_str_f = f"{test_stat_f:.3f} > {format_value_for_display(crit_val_f_upper_plot)}" if decision_crit_f_summary else f"{test_stat_f:.3f} ≤ {format_value_for_display(crit_val_f_upper_plot)}"
        else: # Two-tailed
            crit_val_f_display_summary = f"Lower: {format_value_for_display(crit_val_f_lower_plot)}, Upper: {format_value_for_display(crit_val_f_upper_plot)}" \
                                         if crit_val_f_lower_plot is not None and crit_val_f_upper_plot is not None else "N/A"
            p_val_calc_f_summary = p_val_f_two_summary
            decision_crit_f_summary = (test_stat_f > crit_val_f_upper_plot if crit_val_f_upper_plot is not None and not np.isnan(crit_val_f_upper_plot) else False) or \
                                     (test_stat_f < crit_val_f_lower_plot if crit_val_f_lower_plot is not None and not np.isnan(crit_val_f_lower_plot) else False)
            comparison_crit_str_f = f"{test_stat_f:.3f} > {format_value_for_display(crit_val_f_upper_plot)} or {test_stat_f:.3f} < {format_value_for_display(crit_val_f_lower_plot)}" if decision_crit_f_summary else f"{format_value_for_display(crit_val_f_lower_plot)} ≤ {test_stat_f:.3f} ≤ {format_value_for_display(crit_val_f_upper_plot)}"
        
        decision_p_alpha_f_summary = p_val_calc_f_summary < alpha_f_input 
        
        st.markdown(f"""
        1.  **Critical Value(s) ({tail_f}) for α={alpha_f_input:.4f}**: {crit_val_f_display_summary}
            * *Associated p-value (α or α/2 per tail)*: {p_val_for_crit_val_f_display_summary:.4f}
        2.  **Calculated Test Statistic**: {test_stat_f:.3f}
            * *Calculated p-value*: {format_value_for_display(p_val_calc_f_summary, decimals=4)} ({apa_p_value(p_val_calc_f_summary)})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_f_summary else 'not rejected'}**.
            * *Reason*: F(calc) {comparison_crit_str_f} relative to F(crit).
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_f_summary else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_calc_f_summary)} is {'less than' if decision_p_alpha_f_summary else 'not less than'} α ({alpha_f_input:.4f}).
        5.  **APA 7 Style Report**:
            *F*({df1_f_selected}, {df2_f_selected}) = {test_stat_f:.2f}, {apa_p_value(p_val_calc_f_summary)}. The null hypothesis was {'rejected' if decision_p_alpha_f_summary else 'not rejected'} at α = {alpha_f_input:.2f}.
        """)

# --- Tab 4: Chi-square distribution ---
def tab_chi_square_distribution():
    st.header("Chi-square (χ²) Distribution Explorer")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_chi2_input = st.number_input("Alpha (α)", 0.0001, 0.5, 0.05, 0.0001, format="%.4f", key="alpha_chi2_input_tab4")
        all_df_chi2_options = list(range(1, 31)) + [35, 40, 45, 50, 60, 70, 80, 90, 100]
        df_chi2_selected = st.selectbox("Degrees of Freedom (df)", options=all_df_chi2_options, index=all_df_chi2_options.index(5), key="df_chi2_selectbox_tab4") 
        
        tail_chi2 = st.radio("Tail Selection", ("One-tailed (right)", "Two-tailed (e.g. for variance)"), key="tail_chi2_radio_tab4")
        test_stat_chi2 = st.number_input("Calculated χ²-statistic", value=float(df_chi2_selected), format="%.3f", min_value=0.001, key="test_stat_chi2_input_tab4")

        st.subheader("Distribution Plot")
        fig_chi2, ax_chi2 = plt.subplots(figsize=(8,5))
        
        plot_min_chi2 = 0.001
        plot_max_chi2 = 10.0 
        try:
            plot_max_chi2 = max(stats.chi2.ppf(0.999, df_chi2_selected), test_stat_chi2 * 1.5, 10.0)
            if test_stat_chi2 > stats.chi2.ppf(0.999, df_chi2_selected) * 1.2:
                plot_max_chi2 = test_stat_chi2 * 1.2
        except Exception: pass

        x_chi2_plot = np.linspace(plot_min_chi2, plot_max_chi2, 500) 
        y_chi2_plot = stats.chi2.pdf(x_chi2_plot, df_chi2_selected)
        ax_chi2.plot(x_chi2_plot, y_chi2_plot, 'b-', lw=2, label=f'χ²-distribution (df={df_chi2_selected})')

        crit_val_chi2_upper_plot, crit_val_chi2_lower_plot = None, None
        if tail_chi2 == "One-tailed (right)":
            crit_val_chi2_upper_plot = stats.chi2.ppf(1 - alpha_chi2_input, df_chi2_selected)
            if crit_val_chi2_upper_plot is not None and not np.isnan(crit_val_chi2_upper_plot):
                x_fill_upper = np.linspace(crit_val_chi2_upper_plot, plot_max_chi2, 100)
                ax_chi2.fill_between(x_fill_upper, stats.chi2.pdf(x_fill_upper, df_chi2_selected), color='red', alpha=0.5, label=f'α = {alpha_chi2_input:.4f}')
                ax_chi2.axvline(crit_val_chi2_upper_plot, color='red', linestyle='--', lw=1)
        else: 
            crit_val_chi2_upper_plot = stats.chi2.ppf(1 - alpha_chi2_input / 2, df_chi2_selected)
            crit_val_chi2_lower_plot = stats.chi2.ppf(alpha_chi2_input / 2, df_chi2_selected)
            if crit_val_chi2_upper_plot is not None and not np.isnan(crit_val_chi2_upper_plot):
                x_fill_upper_chi2 = np.linspace(crit_val_chi2_upper_plot, plot_max_chi2, 100)
                ax_chi2.fill_between(x_fill_upper_chi2, stats.chi2.pdf(x_fill_upper_chi2, df_chi2_selected), color='red', alpha=0.5, label=f'α/2 = {alpha_chi2_input/2:.4f}')
                ax_chi2.axvline(crit_val_chi2_upper_plot, color='red', linestyle='--', lw=1)
            if crit_val_chi2_lower_plot is not None and not np.isnan(crit_val_chi2_lower_plot):
                x_fill_lower_chi2 = np.linspace(plot_min_chi2, crit_val_chi2_lower_plot, 100)
                ax_chi2.fill_between(x_fill_lower_chi2, stats.chi2.pdf(x_fill_lower_chi2, df_chi2_selected), color='red', alpha=0.5)
                ax_chi2.axvline(crit_val_chi2_lower_plot, color='red', linestyle='--', lw=1)

        ax_chi2.axvline(test_stat_chi2, color='green', linestyle='-', lw=2, label=f'Test Stat = {test_stat_chi2:.3f}')
        ax_chi2.set_title(f'χ²-Distribution (df={df_chi2_selected}) with Critical Region(s)')
        ax_chi2.set_xlabel('χ²-value')
        ax_chi2.set_ylabel('Probability Density')
        ax_chi2.legend(); ax_chi2.grid(True); st.pyplot(fig_chi2)

        st.subheader("Critical χ²-Values (Upper Tail)")
        table_df_window_chi2 = get_dynamic_df_window(all_df_chi2_options, df_chi2_selected, window_size=5)
        table_alpha_cols_chi2 = [0.10, 0.05, 0.025, 0.01, 0.005] 
        
        chi2_table_rows = []
        for df_iter_val in table_df_window_chi2:
            df_iter_calc = int(df_iter_val)
            row_data = {'df': str(df_iter_val)}
            for alpha_col in table_alpha_cols_chi2:
                cv = stats.chi2.ppf(1 - alpha_col, df_iter_calc) 
                row_data[f"α = {alpha_col:.3f}"] = format_value_for_display(cv)
            chi2_table_rows.append(row_data)
        
        df_chi2_table = pd.DataFrame(chi2_table_rows).set_index('df')

        def style_chi2_table(df_to_style):
            style = pd.DataFrame('', index=df_to_style.index, columns=df_to_style.columns)
            selected_df_str = str(df_chi2_selected)

            if selected_df_str in df_to_style.index:
                style.loc[selected_df_str, :] = 'background-color: lightblue;'
            
            target_alpha_for_col_highlight = alpha_chi2_input
            if tail_chi2 == "Two-tailed (e.g. for variance)":
                 target_alpha_for_col_highlight = alpha_chi2_input / 2.0 

            closest_alpha_col_val = min(table_alpha_cols_chi2, key=lambda x: abs(x - target_alpha_for_col_highlight))
            highlight_col_name = f"α = {closest_alpha_col_val:.3f}"

            if highlight_col_name in df_to_style.columns:
                for r_idx in df_to_style.index:
                     current_r_style = style.loc[r_idx, highlight_col_name]
                     style.loc[r_idx, highlight_col_name] = (current_r_style if current_r_style else '') + ' background-color: lightgreen;'
                if selected_df_str in df_to_style.index:
                    current_c_style = style.loc[selected_df_str, highlight_col_name]
                    style.loc[selected_df_str, highlight_col_name] = (current_c_style if current_c_style else '') + ' font-weight: bold; border: 2px solid red; background-color: yellow;'
            return style
        
        st.markdown(df_chi2_table.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                       {'selector': 'td', 'props': [('text-align', 'center')]}])
                                     .apply(style_chi2_table, axis=None).to_html(), unsafe_allow_html=True)
        st.caption(f"Table shows upper-tail critical χ²-values. Highlighted for df={df_chi2_selected} and α closest to your test.")
        st.markdown("""
        **Table Interpretation Note:**
        * The table displays upper-tail critical values (χ²<sub>α</sub>).
        * For **One-tailed (right) tests** (e.g., goodness-of-fit, independence), use the α column matching your chosen significance level.
        * For **Two-tailed tests on variance**, if your total significance level is α<sub>total</sub>:
            * Upper critical value: Look up column for α = α<sub>total</sub>/2.
            * Lower critical value: Use `stats.chi2.ppf(α_total/2, df)` (not directly in this table's main columns).
        """)


    with col2: # Summary for Chi-square
        st.subheader("P-value Calculation Explanation")
        st.markdown(f"""
        The p-value is the probability of a χ²-statistic as extreme as, or more extreme than, {test_stat_chi2:.3f}.
        * **One-tailed (right)**: `P(χ² ≥ {test_stat_chi2:.3f})` (i.e., `stats.chi2.sf(test_stat_chi2, df_chi2_selected)`)
        * **Two-tailed**: `2 * min(P(χ² ≤ {test_stat_chi2:.3f}), P(χ² ≥ {test_stat_chi2:.3f}))` (i.e., `2 * min(stats.chi2.cdf(test_stat_chi2, df_chi2_selected), stats.chi2.sf(test_stat_chi2, df_chi2_selected))`)
        """)

        st.subheader("Summary")
        p_val_chi2_one_right_summary = stats.chi2.sf(test_stat_chi2, df_chi2_selected)
        cdf_chi2_summary = stats.chi2.cdf(test_stat_chi2, df_chi2_selected)
        sf_chi2_summary = stats.chi2.sf(test_stat_chi2, df_chi2_selected)
        p_val_chi2_two_summary = 2 * min(cdf_chi2_summary, sf_chi2_summary)
        p_val_chi2_two_summary = min(p_val_chi2_two_summary, 1.0)

        crit_val_chi2_display_summary = "N/A"
        p_val_for_crit_val_chi2_display_summary = alpha_chi2_input

        if tail_chi2 == "One-tailed (right)":
            crit_val_chi2_display_summary = format_value_for_display(crit_val_chi2_upper_plot) if crit_val_chi2_upper_plot is not None else "N/A"
            p_val_calc_chi2_summary = p_val_chi2_one_right_summary
            decision_crit_chi2_summary = test_stat_chi2 > crit_val_chi2_upper_plot if crit_val_chi2_upper_plot is not None and not np.isnan(crit_val_chi2_upper_plot) else False
            comparison_crit_str_chi2 = f"{test_stat_chi2:.3f} > {format_value_for_display(crit_val_chi2_upper_plot)}" if decision_crit_chi2_summary else f"{test_stat_chi2:.3f} ≤ {format_value_for_display(crit_val_chi2_upper_plot)}"
        else: # Two-tailed
            crit_val_chi2_display_summary = f"Lower: {format_value_for_display(crit_val_chi2_lower_plot)}, Upper: {format_value_for_display(crit_val_chi2_upper_plot)}" \
                                            if crit_val_chi2_lower_plot is not None and crit_val_chi2_upper_plot is not None else "N/A"
            p_val_calc_chi2_summary = p_val_chi2_two_summary
            decision_crit_chi2_summary = (test_stat_chi2 > crit_val_chi2_upper_plot if crit_val_chi2_upper_plot is not None and not np.isnan(crit_val_chi2_upper_plot) else False) or \
                                         (test_stat_chi2 < crit_val_chi2_lower_plot if crit_val_chi2_lower_plot is not None and not np.isnan(crit_val_chi2_lower_plot) else False)
            comparison_crit_str_chi2 = f"{test_stat_chi2:.3f} > {format_value_for_display(crit_val_chi2_upper_plot)} or {test_stat_chi2:.3f} < {format_value_for_display(crit_val_chi2_lower_plot)}" if decision_crit_chi2_summary else f"{format_value_for_display(crit_val_chi2_lower_plot)} ≤ {test_stat_chi2:.3f} ≤ {format_value_for_display(crit_val_chi2_upper_plot)}"

        decision_p_alpha_chi2_summary = p_val_calc_chi2_summary < alpha_chi2_input
        
        st.markdown(f"""
        1.  **Critical Value(s) ({tail_chi2})**: {crit_val_chi2_display_summary}
            * *Associated p-value (α or α/2 per tail)*: {p_val_for_crit_val_chi2_display_summary:.4f}
        2.  **Calculated Test Statistic**: {test_stat_chi2:.3f}
            * *Calculated p-value*: {format_value_for_display(p_val_calc_chi2_summary, decimals=4)} ({apa_p_value(p_val_calc_chi2_summary)})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_chi2_summary else 'not rejected'}**.
            * *Reason*: χ²(calc) {comparison_crit_str_chi2} relative to χ²(crit).
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_chi2_summary else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_calc_chi2_summary)} is {'less than' if decision_p_alpha_chi2_summary else 'not less than'} α ({alpha_chi2_input:.4f}).
        5.  **APA 7 Style Report**:
            χ²({df_chi2_selected}) = {test_stat_chi2:.2f}, {apa_p_value(p_val_calc_chi2_summary)}. The null hypothesis was {'rejected' if decision_p_alpha_chi2_summary else 'not rejected'} at α = {alpha_chi2_input:.2f}.
        """)


# --- Tab 5: Mann-Whitney U Test ---
def tab_mann_whitney_u():
    st.header("Mann-Whitney U Test (Normal Approximation)")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_mw = st.number_input("Alpha (α)", 0.0001, 0.5, 0.05, 0.0001, format="%.4f", key="alpha_mw_input")
        n1_mw = st.number_input("Sample Size Group 1 (n1)", 1, 1000, 10, 1, key="n1_mw_input") 
        n2_mw = st.number_input("Sample Size Group 2 (n2)", 1, 1000, 12, 1, key="n2_mw_input") 
        tail_mw = st.radio("Tail Selection", ("Two-tailed", "One-tailed (right)", "One-tailed (left)"), key="tail_mw_radio")
        u_stat_mw = st.number_input("Calculated U-statistic", value=float(n1_mw*n2_mw/2), format="%.1f", min_value=0.0, max_value=float(n1_mw*n2_mw), key="u_stat_mw_input")
        
        st.info("This tab uses the Normal Approximation for the Mann-Whitney U test. For small samples (e.g., n1 or n2 < 10), this is an approximation; exact methods/tables are preferred for definitive conclusions.")

        mu_u = (n1_mw * n2_mw) / 2
        sigma_u_sq = (n1_mw * n2_mw * (n1_mw + n2_mw + 1)) / 12
        sigma_u = np.sqrt(sigma_u_sq) if sigma_u_sq > 0 else 0
        
        z_calc_mw = 0.0
        if sigma_u > 0:
            if u_stat_mw < mu_u: z_calc_mw = (u_stat_mw + 0.5 - mu_u) / sigma_u
            elif u_stat_mw > mu_u: z_calc_mw = (u_stat_mw - 0.5 - mu_u) / sigma_u
        elif n1_mw > 0 and n2_mw > 0 : 
            st.warning("Standard deviation (σ_U) for normal approximation is zero. Check sample sizes. z_calc set to 0.")


        st.markdown(f"**Normal Approx. Parameters:** μ<sub>U</sub> = {mu_u:.2f}, σ<sub>U</sub> = {sigma_u:.2f}")
        st.markdown(f"**z-statistic (from U, for approx.):** {z_calc_mw:.3f}")

        st.subheader("Distribution Plot (Normal Approximation)")
        fig_mw, ax_mw = plt.subplots(figsize=(8,5))
        plot_min_z_mw = min(stats.norm.ppf(0.0001), z_calc_mw - 2, -4.0)
        plot_max_z_mw = max(stats.norm.ppf(0.9999), z_calc_mw + 2, 4.0)
        if abs(z_calc_mw) > 4 and abs(z_calc_mw) > plot_max_z_mw * 0.8:
            plot_min_z_mw = min(plot_min_z_mw, z_calc_mw -1); plot_max_z_mw = max(plot_max_z_mw, z_calc_mw +1)
        
        x_norm_mw = np.linspace(plot_min_z_mw, plot_max_z_mw, 500)
        y_norm_mw = stats.norm.pdf(x_norm_mw)
        ax_mw.plot(x_norm_mw, y_norm_mw, 'b-', lw=2, label='Standard Normal Distribution (Approx.)')
        
        crit_z_upper_mw_plot, crit_z_lower_mw_plot = None, None
        if tail_mw == "Two-tailed": crit_z_upper_mw_plot = stats.norm.ppf(1 - alpha_mw / 2); crit_z_lower_mw_plot = stats.norm.ppf(alpha_mw / 2)
        elif tail_mw == "One-tailed (right)": crit_z_upper_mw_plot = stats.norm.ppf(1 - alpha_mw)
        else: crit_z_lower_mw_plot = stats.norm.ppf(alpha_mw)

        if crit_z_upper_mw_plot is not None and not np.isnan(crit_z_upper_mw_plot):
            x_fill_upper = np.linspace(crit_z_upper_mw_plot, plot_max_z_mw, 100)
            ax_mw.fill_between(x_fill_upper, stats.norm.pdf(x_fill_upper), color='red', alpha=0.5, label=f'Approx. Crit. Region')
            ax_mw.axvline(crit_z_upper_mw_plot, color='red', linestyle='--', lw=1)
        if crit_z_lower_mw_plot is not None and not np.isnan(crit_z_lower_mw_plot):
            x_fill_lower = np.linspace(plot_min_z_mw, crit_z_lower_mw_plot, 100)
            ax_mw.fill_between(x_fill_lower, stats.norm.pdf(x_fill_lower), color='red', alpha=0.5, label=f'Approx. Crit. Region' if crit_z_upper_mw_plot is None else None) 
            ax_mw.axvline(crit_z_lower_mw_plot, color='red', linestyle='--', lw=1)

        ax_mw.axvline(z_calc_mw, color='green', linestyle='-', lw=2, label=f'Approx. z_calc = {z_calc_mw:.3f}')
        ax_mw.set_title('Normal Approximation for Mann-Whitney U'); ax_mw.legend(); ax_mw.grid(True); st.pyplot(fig_mw)

        st.subheader("Critical z-Values (Upper Tail) for Normal Approximation")
        table_alpha_cols_z_crit = [0.10, 0.05, 0.025, 0.01, 0.005]
        z_crit_table_rows = [{'Distribution': 'z (Standard Normal)'}]
        for alpha_c in table_alpha_cols_z_crit:
            z_crit_table_rows[0][f"α = {alpha_c:.3f}"] = format_value_for_display(stats.norm.ppf(1-alpha_c))
        df_z_crit_table_mw = pd.DataFrame(z_crit_table_rows).set_index('Distribution')
        
        def style_z_crit_table_mw(df_to_style):
            style = pd.DataFrame('', index=df_to_style.index, columns=df_to_style.columns)
            target_alpha_for_col = alpha_mw
            if tail_mw == "Two-tailed": target_alpha_for_col = alpha_mw / 2.0
            closest_alpha_col = min(table_alpha_cols_z_crit, key=lambda x: abs(x - target_alpha_for_col))
            highlight_col = f"α = {closest_alpha_col:.3f}"
            if highlight_col in df_to_style.columns:
                style.loc[:, highlight_col] = 'background-color: lightgreen; font-weight: bold; border: 2px solid red;'
            return style
        st.markdown(df_z_crit_table_mw.style.apply(style_z_crit_table_mw, axis=None).to_html(), unsafe_allow_html=True)
        st.caption("Compare your calculated z-statistic to these critical z-values.")


    with col2: 
        st.subheader("P-value Calculation Explanation")
        p_val_calc_mw = float('nan')
        
        if tail_mw == "Two-tailed": p_val_calc_mw = 2 * stats.norm.sf(abs(z_calc_mw))
        elif tail_mw == "One-tailed (right)": p_val_calc_mw = stats.norm.sf(z_calc_mw)
        else:  p_val_calc_mw = stats.norm.cdf(z_calc_mw)
        p_val_calc_mw = min(p_val_calc_mw, 1.0) if not np.isnan(p_val_calc_mw) else float('nan')
        
        st.markdown(f"""
        The U statistic ({u_stat_mw:.1f}) is used to calculate an approximate z-statistic ({z_calc_mw:.3f}).
        The p-value is then derived from the standard normal distribution.
        * **Two-tailed**: `2 * P(Z ≥ |{z_calc_mw:.3f}|)`
        * **One-tailed (right)**: `P(Z ≥ {z_calc_mw:.3f})` 
        * **One-tailed (left)**: `P(Z ≤ {z_calc_mw:.3f})` 
        """)

        st.subheader("Summary")
        crit_val_z_display_mw = "N/A"
        if tail_mw == "Two-tailed": crit_val_z_display_mw = f"±{format_value_for_display(crit_z_upper_mw_plot)}"
        elif tail_mw == "One-tailed (right)": crit_val_z_display_mw = format_value_for_display(crit_z_upper_mw_plot)
        else: crit_val_z_display_mw = format_value_for_display(crit_z_lower_mw_plot)
        
        decision_crit_mw = False
        if crit_z_upper_mw_plot is not None: 
            if tail_mw == "Two-tailed": decision_crit_mw = abs(z_calc_mw) > crit_z_upper_mw_plot if not np.isnan(crit_z_upper_mw_plot) else False
            elif tail_mw == "One-tailed (right)": decision_crit_mw = z_calc_mw > crit_z_upper_mw_plot if not np.isnan(crit_z_upper_mw_plot) else False
        if crit_z_lower_mw_plot is not None and tail_mw == "One-tailed (left)":
             decision_crit_mw = z_calc_mw < crit_z_lower_mw_plot if not np.isnan(crit_z_lower_mw_plot) else False
        
        comparison_crit_str_mw = f"Approx. z_calc ({z_calc_mw:.3f}) vs z_crit ({crit_val_z_display_mw})"
        decision_p_alpha_mw = p_val_calc_mw < alpha_mw if not np.isnan(p_val_calc_mw) else False
        
        st.markdown(f"""
        1.  **Critical z-value ({tail_mw})**: {crit_val_z_display_mw}
            * *Associated p-value (α or α/2 per tail)*: {alpha_mw:.4f}
        2.  **Calculated U-statistic**: {u_stat_mw:.1f} (Approx. z-statistic: {z_calc_mw:.3f})
            * *Calculated p-value (Normal Approx.)*: {format_value_for_display(p_val_calc_mw, decimals=4)} ({apa_p_value(p_val_calc_mw)})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_mw else 'not rejected'}**.
            * *Reason*: {comparison_crit_str_mw}.
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_mw else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_calc_mw)} is {'less than' if decision_p_alpha_mw else 'not less than'} α ({alpha_mw:.4f}).
        5.  **APA 7 Style Report (based on Normal Approximation)**:
            A Mann-Whitney U test (using normal approximation) indicated that the outcome for group 1 (n<sub>1</sub>={n1_mw}) was {'' if decision_p_alpha_mw else 'not '}statistically significantly different from group 2 (n<sub>2</sub>={n2_mw}), *U* = {u_stat_mw:.1f}, *z* = {z_calc_mw:.2f}, {apa_p_value(p_val_calc_mw)}. The null hypothesis was {'rejected' if decision_p_alpha_mw else 'not rejected'} at α = {alpha_mw:.2f}.
        """)

# --- Wilcoxon Signed-Rank T Test ---
def tab_wilcoxon_t():
    st.header("Wilcoxon Signed-Rank T Test (Normal Approximation)")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_w = st.number_input("Alpha (α)", 0.0001, 0.5, 0.05, 0.0001, format="%.4f", key="alpha_w_input")
        n_w = st.number_input("Sample Size (n, non-zero differences)", 1, 1000, 15, 1, key="n_w_input") 
        tail_w = st.radio("Tail Selection", ("Two-tailed", "One-tailed (right)", "One-tailed (left)"), key="tail_w_radio")
        t_stat_w_input = st.number_input("Calculated T-statistic (sum of ranks)", value=float(n_w*(n_w+1)/4 / 2 if n_w >0 else 0), format="%.1f", min_value=0.0, max_value=float(n_w*(n_w+1)/2 if n_w > 0 else 0), key="t_stat_w_input")
        
        st.info("This tab uses the Normal Approximation for the Wilcoxon Signed-Rank T test. For small n (e.g., < 20), this is an approximation; exact methods/tables are preferred for definitive conclusions.")

        mu_t_w = n_w * (n_w + 1) / 4
        sigma_t_w_sq = n_w * (n_w + 1) * (2 * n_w + 1) / 24
        sigma_t_w = np.sqrt(sigma_t_w_sq) if sigma_t_w_sq > 0 else 0
        
        z_calc_w = 0.0
        if sigma_t_w > 0:
            if t_stat_w_input < mu_t_w: 
                z_calc_w = (t_stat_w_input + 0.5 - mu_t_w) / sigma_t_w
            elif t_stat_w_input > mu_t_w: 
                z_calc_w = (t_stat_w_input - 0.5 - mu_t_w) / sigma_t_w
        elif n_w > 0: 
            st.warning("Standard deviation (σ_T) for normal approximation is zero. Check sample size n. z_calc set to 0.")

        st.markdown(f"**Normal Approx. Parameters:** μ<sub>T</sub> = {mu_t_w:.2f}, σ<sub>T</sub> = {sigma_t_w:.2f}")
        st.markdown(f"**z-statistic (from T, for approx.):** {z_calc_w:.3f}")
        
        st.subheader("Distribution Plot (Normal Approximation)")
        fig_w, ax_w = plt.subplots(figsize=(8,5)); 
        plot_min_z_w = min(stats.norm.ppf(0.0001), z_calc_w - 2, -4.0)
        plot_max_z_w = max(stats.norm.ppf(0.9999), z_calc_w + 2, 4.0)
        if abs(z_calc_w) > 4 and abs(z_calc_w) > plot_max_z_w * 0.8:
             plot_min_z_w = min(plot_min_z_w, z_calc_w -1)
             plot_max_z_w = max(plot_max_z_w, z_calc_w +1)

        x_norm_w = np.linspace(plot_min_z_w, plot_max_z_w, 500)
        y_norm_w = stats.norm.pdf(x_norm_w)
        ax_w.plot(x_norm_w, y_norm_w, 'b-', lw=2, label='Standard Normal Distribution (Approx.)')
        
        crit_z_upper_w_plot, crit_z_lower_w_plot = None, None
        if tail_w == "Two-tailed": crit_z_upper_w_plot = stats.norm.ppf(1 - alpha_w / 2); crit_z_lower_w_plot = stats.norm.ppf(alpha_w / 2)
        elif tail_w == "One-tailed (right)": crit_z_upper_w_plot = stats.norm.ppf(1 - alpha_w)
        else: crit_z_lower_w_plot = stats.norm.ppf(alpha_w)

        if crit_z_upper_w_plot is not None and not np.isnan(crit_z_upper_w_plot):
            x_fill_upper = np.linspace(crit_z_upper_w_plot, plot_max_z_w, 100)
            ax_w.fill_between(x_fill_upper, stats.norm.pdf(x_fill_upper), color='red', alpha=0.5, label=f'Approx. Crit. Region')
            ax_w.axvline(crit_z_upper_w_plot, color='red', linestyle='--', lw=1)
        if crit_z_lower_w_plot is not None and not np.isnan(crit_z_lower_w_plot):
            x_fill_lower = np.linspace(plot_min_z_w, crit_z_lower_w_plot, 100)
            ax_w.fill_between(x_fill_lower, stats.norm.pdf(x_fill_lower), color='red', alpha=0.5, label=f'Approx. Crit. Region' if crit_z_upper_w_plot is None else None)
            ax_w.axvline(crit_z_lower_w_plot, color='red', linestyle='--', lw=1)

        ax_w.axvline(z_calc_w, color='green', linestyle='-', lw=2, label=f'Approx. z_calc = {z_calc_w:.3f}')
        ax_w.set_title('Normal Approx. for Wilcoxon T: Critical z Region(s)'); ax_w.legend(); ax_w.grid(True); st.pyplot(fig_w)

        st.subheader("Critical z-Values (Upper Tail) for Normal Approximation")
        table_alpha_cols_z_crit_w = [0.10, 0.05, 0.025, 0.01, 0.005]
        z_crit_table_rows_w = [{'Distribution': 'z (Standard Normal)'}]
        for alpha_c in table_alpha_cols_z_crit_w:
            z_crit_table_rows_w[0][f"α = {alpha_c:.3f}"] = format_value_for_display(stats.norm.ppf(1-alpha_c))
        df_z_crit_table_w = pd.DataFrame(z_crit_table_rows_w).set_index('Distribution')
        
        def style_z_crit_table_w(df_to_style): 
            style = pd.DataFrame('', index=df_to_style.index, columns=df_to_style.columns)
            target_alpha_for_col = alpha_w
            if tail_w == "Two-tailed": target_alpha_for_col = alpha_w / 2.0
            closest_alpha_col = min(table_alpha_cols_z_crit_w, key=lambda x: abs(x - target_alpha_for_col))
            highlight_col = f"α = {closest_alpha_col:.3f}"
            if highlight_col in df_to_style.columns:
                style.loc[:, highlight_col] = 'background-color: lightgreen; font-weight: bold; border: 2px solid red;'
            return style
        st.markdown(df_z_crit_table_w.style.apply(style_z_crit_table_w, axis=None).to_html(), unsafe_allow_html=True)
        st.caption("Compare your calculated z-statistic to these critical z-values.")


    with col2: # Summary for Wilcoxon T
        st.subheader("P-value Calculation Explanation")
        p_val_calc_w = float('nan')
        
        if tail_w == "Two-tailed": p_val_calc_w = 2 * stats.norm.sf(abs(z_calc_w))
        elif tail_w == "One-tailed (right)": p_val_calc_w = stats.norm.sf(z_calc_w)
        else: p_val_calc_w = stats.norm.cdf(z_calc_w)
        p_val_calc_w = min(p_val_calc_w, 1.0) if not np.isnan(p_val_calc_w) else float('nan')

        st.markdown(f"""
        The T statistic ({t_stat_w_input:.1f}) is converted to an approximate z-statistic ({z_calc_w:.3f}) using μ<sub>T</sub>={mu_t_w:.2f}, σ<sub>T</sub>={sigma_t_w:.2f} (with continuity correction). P-value from normal distribution.
        * **Two-tailed**: `2 * P(Z ≥ |{z_calc_w:.3f}|)` 
        * **One-tailed (right)**: `P(Z ≥ {z_calc_w:.3f})` 
        * **One-tailed (left)**: `P(Z ≤ {z_calc_w:.3f})` 
        """)

        st.subheader("Summary")
        crit_val_display_w = "N/A"
        if tail_w == "Two-tailed": crit_val_display_w = f"±{format_value_for_display(crit_z_upper_w_plot)}"
        elif tail_w == "One-tailed (right)": crit_val_display_w = format_value_for_display(crit_z_upper_w_plot)
        else: crit_val_display_w = format_value_for_display(crit_z_lower_w_plot)

        decision_crit_w = False
        if crit_z_upper_w_plot is not None:
             if tail_w == "Two-tailed": decision_crit_w = abs(z_calc_w) > crit_z_upper_w_plot if not np.isnan(crit_z_upper_w_plot) else False
             elif tail_w == "One-tailed (right)": decision_crit_w = z_calc_w > crit_z_upper_w_plot if not np.isnan(crit_z_upper_w_plot) else False
        if crit_z_lower_w_plot is not None and tail_w == "One-tailed (left)":
             decision_crit_w = z_calc_w < crit_z_lower_w_plot if not np.isnan(crit_z_lower_w_plot) else False
        
        comparison_crit_str_w = f"Approx. z_calc ({z_calc_w:.3f}) vs z_crit ({crit_val_display_w})"
        decision_p_alpha_w = p_val_calc_w < alpha_w if not np.isnan(p_val_calc_w) else False
        
        st.markdown(f"""
        1.  **Critical z-value ({tail_w})**: {crit_val_display_w}
            * *Associated p-value (α or α/2 per tail)*: {alpha_w:.4f}
        2.  **Calculated T-statistic**: {t_stat_w_input:.1f} (Approx. z-statistic: {z_calc_w:.3f})
            * *Calculated p-value (Normal Approx.)*: {format_value_for_display(p_val_calc_w, decimals=4)} ({apa_p_value(p_val_calc_w)})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_w else 'not rejected'}**.
            * *Reason*: {comparison_crit_str_w}.
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_w else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_calc_w)} is {'less than' if decision_p_alpha_w else 'not less than'} α ({alpha_w:.4f}).
        5.  **APA 7 Style Report (based on Normal Approximation)**:
            A Wilcoxon signed-rank test (using normal approximation) indicated that the median difference was {'' if decision_p_alpha_w else 'not '}statistically significant, *T* = {t_stat_w_input:.1f}, *z* = {z_calc_w:.2f}, {apa_p_value(p_val_calc_w)}. The null hypothesis was {'rejected' if decision_p_alpha_w else 'not rejected'} at α = {alpha_w:.2f} (n={n_w}).
        """)

# --- Binomial Test ---
def tab_binomial_test():
    st.header("Binomial Test Explorer")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_b = st.number_input("Alpha (α)", 0.0001, 0.5, 0.05, 0.0001, format="%.4f", key="alpha_b_input")
        n_b = st.number_input("Number of Trials (n)", 1, 1000, 20, 1, key="n_b_input")
        p_null_b = st.number_input("Null Hypothesis Probability (p₀)", 0.00, 1.00, 0.5, 0.01, format="%.2f", key="p_null_b_input")
        k_success_b = st.number_input("Number of Successes (k)", 0, n_b, int(n_b * p_null_b), 1, key="k_success_b_input")
        
        tail_options_b = {
            f"Two-tailed (p ≠ {p_null_b})": "two-sided",
            f"One-tailed (right, p > {p_null_b})": "greater",
            f"One-tailed (left, p < {p_null_b})": "less"
        }
        tail_b_display = st.radio("Tail Selection (Alternative Hypothesis)", 
                                  list(tail_options_b.keys()), 
                                  key="tail_b_display_radio")
        tail_b_scipy = tail_options_b[tail_b_display]


        st.subheader("Binomial Distribution Plot")
        fig_b, ax_b = plt.subplots(figsize=(8,5))
        x_b = np.arange(0, n_b + 1)
        y_b_pmf = stats.binom.pmf(x_b, n_b, p_null_b)
        ax_b.bar(x_b, y_b_pmf, label=f'Binomial PMF (n={n_b}, p₀={p_null_b})', alpha=0.7, color='skyblue')
        
        ax_b.scatter([k_success_b], [stats.binom.pmf(k_success_b, n_b, p_null_b)], color='green', s=100, zorder=5, label=f'Observed k = {k_success_b}')
        
        if tail_b_scipy == "greater": 
            crit_region_indices = x_b[x_b >= k_success_b]
            if len(crit_region_indices) > 0 :
                 ax_b.bar(crit_region_indices, y_b_pmf[crit_region_indices], color='salmon', alpha=0.6, label=f'P(X ≥ {k_success_b})')
        elif tail_b_scipy == "less": 
            crit_region_indices = x_b[x_b <= k_success_b]
            if len(crit_region_indices) > 0:
                ax_b.bar(crit_region_indices, y_b_pmf[crit_region_indices], color='salmon', alpha=0.6, label=f'P(X ≤ {k_success_b})')

        ax_b.set_title(f'Binomial Distribution (n={n_b}, p₀={p_null_b})')
        ax_b.set_xlabel('Number of Successes (k)')
        ax_b.set_ylabel('Probability Mass P(X=k)')
        ax_b.legend(); ax_b.grid(True); st.pyplot(fig_b)

        st.subheader("Probability Table Snippet")
        all_k_values_binomial = list(range(n_b + 1))
        table_k_window_binomial = get_dynamic_df_window(all_k_values_binomial, k_success_b, window_size=5)
        
        if len(table_k_window_binomial) > 0:
            table_data_b = {
                "k": table_k_window_binomial,
                "P(X=k)": [format_value_for_display(stats.binom.pmf(k_val, n_b, p_null_b), decimals=4) for k_val in table_k_window_binomial],
                "P(X≤k) (CDF)": [format_value_for_display(stats.binom.cdf(k_val, n_b, p_null_b), decimals=4) for k_val in table_k_window_binomial],
                "P(X≥k)": [format_value_for_display(stats.binom.sf(k_val -1, n_b, p_null_b), decimals=4) for k_val in table_k_window_binomial] 
            }
            df_table_b = pd.DataFrame(table_data_b)
            
            def highlight_k_row_b(row):
                if int(row["k"]) == k_success_b:
                    return ['background-color: yellow'] * len(row)
                return [''] * len(row)
            st.markdown(df_table_b.style.apply(highlight_k_row_b, axis=1).to_html(), unsafe_allow_html=True)
            st.caption(f"Table shows probabilities around k={k_success_b}. Highlighted row is your observed k.")
        else:
            st.info("Not enough range to display table snippet (e.g., n is very small).")
        st.markdown("""**Table Interpretation Note:** This table helps understand probabilities around the observed k. Critical regions for binomial tests are based on these cumulative probabilities compared to α.""")

    with col2: # Summary for Binomial Test
        st.subheader("P-value Calculation Explanation")
        st.markdown(f"""
        The p-value for a binomial test is the probability of observing k={k_success_b} successes, or results more extreme, given n={n_b} trials and null probability p₀={p_null_b}.
        * **Two-tailed (p ≠ {p_null_b})**: Sum of P(X=i) for all i where P(X=i) ≤ P(X={k_success_b}).
        * **One-tailed (right, p > {p_null_b})**: `P(X ≥ {k_success_b}) = stats.binom.sf({k_success_b}-1, n_b, p_null_b)`
        * **One-tailed (left, p < {p_null_b})**: `P(X ≤ {k_success_b}) = stats.binom.cdf({k_success_b}, n_b, p_null_b)`
        """)

        st.subheader("Summary")
        p_val_b_one_left = stats.binom.cdf(k_success_b, n_b, p_null_b)
        p_val_b_one_right = stats.binom.sf(k_success_b - 1, n_b, p_null_b) 

        if tail_b_scipy == "two-sided":
            p_observed_k = stats.binom.pmf(k_success_b, n_b, p_null_b)
            p_val_calc_b = 0
            for i in range(n_b + 1):
                if stats.binom.pmf(i, n_b, p_null_b) <= p_observed_k + 1e-9: # Tolerance
                    p_val_calc_b += stats.binom.pmf(i, n_b, p_null_b)
            p_val_calc_b = min(p_val_calc_b, 1.0) 
        elif tail_b_scipy == "greater":
            p_val_calc_b = p_val_b_one_right
        else: 
            p_val_calc_b = p_val_b_one_left
        
        crit_val_b_display = "Exact critical k values are complex for binomial; decision based on p-value."
        decision_p_alpha_b = p_val_calc_b < alpha_b
        decision_crit_b = decision_p_alpha_b 
        
        st.markdown(f"""
        1.  **Critical Region**: {crit_val_b_display}
            * *Significance level (α)*: {alpha_b:.4f}
        2.  **Observed Number of Successes (k)**: {k_success_b}
            * *Calculated p-value*: {format_value_for_display(p_val_calc_b, decimals=4)} ({apa_p_value(p_val_calc_b)})
        3.  **Decision (Critical Region Method - based on p-value)**: H₀ is **{'rejected' if decision_crit_b else 'not rejected'}**.
            * *Reason*: For discrete tests, the p-value method is generally preferred for decision making.
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_b else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_calc_b)} is {'less than' if decision_p_alpha_b else 'not less than'} α ({alpha_b:.4f}).
        5.  **APA 7 Style Report**:
            A binomial test was performed to assess whether the proportion of successes (k={k_success_b}, n={n_b}) was different from the null hypothesis proportion of p₀={p_null_b}. The result was {'' if decision_p_alpha_b else 'not '}statistically significant, {apa_p_value(p_val_calc_b)}. The null hypothesis was {'rejected' if decision_p_alpha_b else 'not rejected'} at α = {alpha_b:.2f}.
        """)


# --- Tab 8: Tukey HSD ---
def tab_tukey_hsd():
    st.header("Tukey HSD (Honestly Significant Difference) Explorer")
    col1, col2 = st.columns([2, 1.5])
    
    with col1:
        st.subheader("Inputs")
        alpha_tukey = st.number_input("Alpha (α)", 0.0001, 0.5, 0.05, 0.0001, format="%.4f", key="alpha_tukey_hsd_input")
        all_k_tukey_options = list(range(2, 21)) + [25,30,35,40,45,50]
        all_df_error_options = list(range(1,31)) + [35,40,45,50,60,70,80,90,100,120,1000,2000]

        k_tukey_selected = st.selectbox("Number of Groups (k)", options=all_k_tukey_options, index=all_k_tukey_options.index(3), key="k_tukey_selectbox")
        df_error_tukey_selected = st.selectbox("Degrees of Freedom for Error (df_error)", options=all_df_error_options, index=all_df_error_options.index(20), key="df_error_tukey_selectbox")
        
        test_stat_tukey_q = st.number_input("Calculated q-statistic (for a pair)", value=1.0, format="%.3f", min_value=0.0, key="test_stat_tukey_q_input")

        st.subheader("Studentized Range q Distribution (Conceptual Plot)")
        st.markdown("The Tukey HSD test uses the studentized range q distribution. The plot below is illustrative.")
        
        q_crit_tukey_plot = float('nan') # Initialize to NaN
        source_q_crit_plot = "Not calculated"
        
        try:
            from statsmodels.stats.libqsturng import qsturng
            q_crit_tukey_plot = qsturng(1 - alpha_tukey, k_tukey_selected, df_error_tukey_selected)
            source_q_crit_plot = "statsmodels"
            if np.isnan(q_crit_tukey_plot): # If statsmodels returns NaN
                source_q_crit_plot = "statsmodels (returned NaN)"
        except ImportError:
            st.error("`statsmodels` library not found. Tukey HSD calculations cannot be performed. Please install `statsmodels`.")
            source_q_crit_plot = "statsmodels not available"
        except Exception as e: 
            st.warning(f"Error calculating critical q-value with statsmodels: {e}. Value may be N/A.")
            source_q_crit_plot = "statsmodels (calculation error)"
        
        fig_tukey, ax_tukey = plt.subplots(figsize=(8,5))
        if not np.isnan(q_crit_tukey_plot):
            plot_max_q = max(q_crit_tukey_plot * 1.5, test_stat_tukey_q * 1.5, 5.0)
            if test_stat_tukey_q > q_crit_tukey_plot * 1.2: plot_max_q = test_stat_tukey_q * 1.2
            
            x_placeholder = np.linspace(0.01, plot_max_q, 200)
            try: 
                shape_param = float(k_tukey_selected) 
                scale_param = max(q_crit_tukey_plot, test_stat_tukey_q, 1.0) / (shape_param * 2 if shape_param > 0 else 2) 
                if shape_param <=0 : shape_param = 2.0 
                if scale_param <=0 : scale_param = 0.5 
                y_placeholder = stats.gamma.pdf(x_placeholder, a=shape_param, scale=scale_param)
                ax_tukey.plot(x_placeholder, y_placeholder, 'b-', lw=2, label=f'Conceptual q-like shape')
                ax_tukey.axvline(q_crit_tukey_plot, color='red', linestyle='--', lw=1.5, label=f'Critical q = {q_crit_tukey_plot:.3f}')
                x_fill_crit = np.linspace(q_crit_tukey_plot, plot_max_q, 100)
                y_fill_crit = stats.gamma.pdf(x_fill_crit, a=shape_param, scale=scale_param)
                ax_tukey.fill_between(x_fill_crit, y_fill_crit, color='red', alpha=0.5)
            except Exception:
                 ax_tukey.text(0.5, 0.6, "Plotting error for conceptual shape.", ha='center')
            ax_tukey.axvline(test_stat_tukey_q, color='green', linestyle='-', lw=2, label=f'Test q = {test_stat_tukey_q:.3f}')
        else:
            ax_tukey.text(0.5, 0.5, "Critical q not available for plotting (statsmodels issue).", ha='center')
        ax_tukey.set_title(f'Conceptual q-Distribution (α={alpha_tukey:.3f})'); ax_tukey.legend(); ax_tukey.grid(True); st.pyplot(fig_tukey)
        st.info(f"Critical q source for plot: {source_q_crit_plot}")

        st.subheader(f"Critical q-Values for α = {alpha_tukey:.3f}")
        table_k_cols_display = [2,3,4,5,6,7,8,9,10,12,15,20] 
        table_df_error_window = get_dynamic_df_window(all_df_error_options, df_error_tukey_selected, window_size=5)

        tukey_table_data = []
        for df_err_iter in table_df_error_window:
            df_err_calc = int(df_err_iter) 
            row = {'df_error': str(df_err_iter)}
            for k_val_iter in table_k_cols_display:
                q_c_val = float('nan')
                try:
                    from statsmodels.stats.libqsturng import qsturng
                    q_c_val = qsturng(1 - alpha_tukey, k_val_iter, df_err_calc)
                except Exception: # Catch import or calculation error
                    pass # q_c_val remains NaN
                row[f"k={k_val_iter}"] = format_value_for_display(q_c_val)
            tukey_table_data.append(row)
        df_tukey_table = pd.DataFrame(tukey_table_data).set_index('df_error')

        def style_tukey_table(df_to_style):
            style = pd.DataFrame('', index=df_to_style.index, columns=df_to_style.columns)
            selected_df_err_str = str(df_error_tukey_selected) 
            
            if selected_df_err_str in df_to_style.index: 
                style.loc[selected_df_err_str, :] = 'background-color: lightblue;'
            
            closest_k_col_val = min(table_k_cols_display, key=lambda x: abs(x - k_tukey_selected))
            highlight_col_name = f"k={closest_k_col_val}"

            if highlight_col_name in df_to_style.columns:
                for r_idx in df_to_style.index:
                     current_r_style = style.loc[r_idx, highlight_col_name]
                     style.loc[r_idx, highlight_col_name] = (current_r_style + ';' if current_r_style and not current_r_style.endswith(';') else current_r_style) + 'background-color: lightgreen;'
                if selected_df_err_str in df_to_style.index:
                    current_c_style = style.loc[selected_df_err_str, highlight_col_name]
                    style.loc[selected_df_err_str, highlight_col_name] = (current_c_style + ';' if current_c_style and not current_c_style.endswith(';') else '') + 'font-weight: bold; border: 2px solid red; background-color: yellow;'
            return style

        st.markdown(df_tukey_table.style.apply(style_tukey_table, axis=None).to_html(), unsafe_allow_html=True)
        st.caption(f"Table shows q-critical values for α={alpha_tukey:.3f}. Highlighted for df_error closest to {df_error_tukey_selected} and k closest to {k_tukey_selected}.")


    with col2: # Summary for Tukey HSD
        st.subheader("P-value Calculation Explanation")
        p_val_calc_tukey_num = float('nan') # CORRECTED INITIALIZATION
        p_val_source = "Not calculated"
        psturng_func = None 

        try:
            from statsmodels.stats.libqsturng import psturng as psturng_imported
            psturng_func = psturng_imported 
        except ImportError:
            p_val_source = "statsmodels.stats.libqsturng.psturng module/function not found."
        
        if psturng_func: 
            try:
                if test_stat_tukey_q is not None and k_tukey_selected > 0 and df_error_tukey_selected > 0:
                    p_val_calc_tukey_num = 1 - psturng_func(test_stat_tukey_q, float(k_tukey_selected), float(df_error_tukey_selected))
                    p_val_calc_tukey_num = max(0, min(p_val_calc_tukey_num, 1.0)) 
                    p_val_source = "statsmodels.stats.libqsturng.psturng"
            except Exception as e_psturng: 
                p_val_source = f"Error during p-value calculation: {e_psturng}"
                p_val_calc_tukey_num = float('nan') 
        
        st.markdown(f"""
        The p-value for a specific calculated q-statistic ({test_stat_tukey_q:.3f}) from Tukey's HSD is P(Q ≥ {test_stat_tukey_q:.3f}).
        * Source for p-value: {p_val_source}
        """)

        st.subheader("Summary")
        q_crit_tukey_summary = q_crit_tukey_plot 
        
        decision_crit_tukey = False
        if isinstance(q_crit_tukey_summary, (int, float)) and not np.isnan(q_crit_tukey_summary):
            decision_crit_tukey = test_stat_tukey_q > q_crit_tukey_summary
            comparison_crit_str_tukey = f"q(calc) ({test_stat_tukey_q:.3f}) > q(crit) ({q_crit_tukey_summary:.3f})" if decision_crit_tukey else f"q(calc) ({test_stat_tukey_q:.3f}) ≤ q(crit) ({q_crit_tukey_summary:.3f})"
        else:
            comparison_crit_str_tukey = "Critical q not available or non-numeric for comparison."

        decision_p_alpha_tukey = False
        if isinstance(p_val_calc_tukey_num, (int, float)) and not np.isnan(p_val_calc_tukey_num):
            decision_p_alpha_tukey = p_val_calc_tukey_num < alpha_tukey
            reason_p_alpha_tukey_display = f"Because {apa_p_value(p_val_calc_tukey_num)} is {'less than' if decision_p_alpha_tukey else 'not less than'} α ({alpha_tukey:.4f})."
        else: 
            reason_p_alpha_tukey_display = "p-value for q_calc not computed or resulted in error."
            if decision_crit_tukey: 
                 decision_p_alpha_tukey = True 
                 reason_p_alpha_tukey_display += " Decision aligned with critical value method."
            
        st.markdown(f"""
        1.  **Critical q-value**: {format_value_for_display(q_crit_tukey_summary)} (Source: {source_q_crit_plot})
            * *Associated significance level (α)*: {alpha_tukey:.4f}
        2.  **Calculated q-statistic (for one pair)**: {test_stat_tukey_q:.3f}
            * *Calculated p-value*: {format_value_for_display(p_val_calc_tukey_num, decimals=4)} ({apa_p_value(p_val_calc_tukey_num)})
        3.  **Decision (Critical Value Method)**: For this pair, H₀ (no difference) is **{'rejected' if decision_crit_tukey else 'not rejected'}**.
            * *Reason*: {comparison_crit_str_tukey}.
        4.  **Decision (p-value Method)**: H₀ (no difference) is **{'rejected' if decision_p_alpha_tukey else 'not rejected'}**.
            * *Reason*: {reason_p_alpha_tukey_display}
        5.  **APA 7 Style Report (for this specific comparison)**:
            A Tukey HSD comparison for one pair yielded *q*({k_tukey_selected}, {df_error_tukey_selected}) = {test_stat_tukey_q:.2f}. {('The associated p-value was ' + apa_p_value(p_val_calc_tukey_num) + '. ') if p_val_calc_tukey_num is not None and not np.isnan(p_val_calc_tukey_num) else ''}The null hypothesis of no difference for this pair was {'rejected' if decision_p_alpha_tukey else 'not rejected'} at α = {alpha_tukey:.2f}.
        """)


# --- Kruskal-Wallis H Test ---
def tab_kruskal_wallis():
    st.header("Kruskal-Wallis H Test (Chi-square Approximation)")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_kw = st.number_input("Alpha (α)", 0.0001, 0.5, 0.05, 0.0001, format="%.4f", key="alpha_kw_input")
        k_groups_kw = st.number_input("Number of Groups (k)", 2, 50, 3, 1, key="k_groups_kw_input") 
        df_kw = k_groups_kw - 1
        st.markdown(f"Degrees of Freedom (df) = k - 1 = {df_kw}")
        test_stat_h_kw = st.number_input("Calculated H-statistic", value=float(df_kw if df_kw > 0 else 0.5), format="%.3f", min_value=0.0, key="test_stat_h_kw_input")
        st.caption("Note: Chi-square approximation is best if each group size ≥ 5.")

        st.subheader("Chi-square Distribution Plot (Approximation for H)")
        fig_kw, ax_kw = plt.subplots(figsize=(8,5))
        crit_val_chi2_kw_plot = None 
        
        if df_kw > 0:
            crit_val_chi2_kw_plot = stats.chi2.ppf(1 - alpha_kw, df_kw)
            plot_min_chi2_kw = 0.001
            plot_max_chi2_kw = max(stats.chi2.ppf(0.999, df_kw), test_stat_h_kw * 1.5, 10.0)
            if test_stat_h_kw > stats.chi2.ppf(0.999, df_kw) * 1.2:
                plot_max_chi2_kw = test_stat_h_kw * 1.2

            x_chi2_kw_plot = np.linspace(plot_min_chi2_kw, plot_max_chi2_kw, 500)
            y_chi2_kw_plot = stats.chi2.pdf(x_chi2_kw_plot, df_kw)
            ax_kw.plot(x_chi2_kw_plot, y_chi2_kw_plot, 'b-', lw=2, label=f'χ²-distribution (df={df_kw})')

            if isinstance(crit_val_chi2_kw_plot, (int, float)) and not np.isnan(crit_val_chi2_kw_plot):
                x_fill_upper_kw = np.linspace(crit_val_chi2_kw_plot, plot_max_chi2_kw, 100)
                ax_kw.fill_between(x_fill_upper_kw, stats.chi2.pdf(x_fill_upper_kw, df_kw), color='red', alpha=0.5, label=f'α = {alpha_kw:.4f}')
                ax_kw.axvline(crit_val_chi2_kw_plot, color='red', linestyle='--', lw=1, label=f'χ²_crit = {crit_val_chi2_kw_plot:.3f}')
            
            ax_kw.axvline(test_stat_h_kw, color='green', linestyle='-', lw=2, label=f'H_calc = {test_stat_h_kw:.3f}')
            ax_kw.set_title(f'χ²-Approximation for Kruskal-Wallis H (df={df_kw})')
        else:
            ax_kw.text(0.5, 0.5, "df must be > 0 (k > 1 for meaningful test)", ha='center', va='center')
            ax_kw.set_title('Plot Unavailable (df=0)')
            
        ax_kw.legend()
        ax_kw.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_kw)
        
        st.subheader("Critical χ²-Values (Upper Tail)")
        all_df_chi2_options_kw = list(range(1, 31)) + [35, 40, 45, 50, 60, 70, 80, 90, 100]
        table_df_window_chi2_kw = get_dynamic_df_window(all_df_chi2_options_kw, df_kw, window_size=5)
        table_alpha_cols_chi2 = [0.10, 0.05, 0.025, 0.01, 0.005]

        chi2_table_rows = []
        for df_iter_val in table_df_window_chi2_kw: # df_iter_val is already a number from window
            df_iter_calc = int(df_iter_val)
            row_data = {'df': str(df_iter_val)}
            for alpha_c in table_alpha_cols_chi2:
                cv = stats.chi2.ppf(1 - alpha_c, df_iter_calc)
                row_data[f"α = {alpha_c:.3f}"] = format_value_for_display(cv)
            chi2_table_rows.append(row_data)
        df_chi2_table_kw = pd.DataFrame(chi2_table_rows).set_index('df')

        def style_chi2_table_kw(df_to_style):
            style = pd.DataFrame('', index=df_to_style.index, columns=df_to_style.columns)
            selected_df_str = str(df_kw) 

            if selected_df_str in df_to_style.index:
                style.loc[selected_df_str, :] = 'background-color: lightblue;'
            
            closest_alpha_col_val = min(table_alpha_cols_chi2, key=lambda x: abs(x - alpha_kw))
            highlight_col_name = f"α = {closest_alpha_col_val:.3f}"

            if highlight_col_name in df_to_style.columns:
                for r_idx in df_to_style.index:
                    current_r_style = style.loc[r_idx, highlight_col_name]
                    style.loc[r_idx, highlight_col_name] = (current_r_style if current_r_style else '') + ' background-color: lightgreen;'
                if selected_df_str in df_to_style.index:
                    current_c_style = style.loc[selected_df_str, highlight_col_name]
                    style.loc[selected_df_str, highlight_col_name] = (current_c_style if current_c_style else '') + ' font-weight: bold; border: 2px solid red; background-color: yellow;'
            return style
        
        if df_kw > 0:
            st.markdown(df_chi2_table_kw.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                                {'selector': 'td', 'props': [('text-align', 'center')]}])
                                         .apply(style_chi2_table_kw, axis=None).to_html(), unsafe_allow_html=True)
            st.caption(f"Table shows upper-tail critical χ²-values. Highlighted for df={df_kw} and α closest to your test.")
        else:
            st.warning("df must be > 0 to generate table (k > 1).")


        st.markdown("""
        **Cumulative Table Note:** Kruskal-Wallis H is approx. χ² distributed (df = k-1). Test is right-tailed: large H suggests group differences.
        """)

    with col2:
        st.subheader("P-value Calculation Explanation")
        st.markdown(f"""
        The p-value is P(χ² ≥ H_calc) assuming H₀ (all group medians are equal) is true.
        * `P(χ² ≥ {test_stat_h_kw:.3f}) = stats.chi2.sf({test_stat_h_kw:.3f}, df={df_kw})` (if df > 0)
        """)
        st.subheader("Summary")
        p_val_calc_kw_num = float('nan') 
        decision_crit_kw = False
        comparison_crit_str_kw = "Test not valid (df must be > 0)"
        decision_p_alpha_kw = False
        apa_H_stat = f"*H*({df_kw if df_kw > 0 else 'N/A'}) = {format_value_for_display(test_stat_h_kw, decimals=2)}"
        
        summary_crit_val_chi2_kw_display_str = "N/A (df=0)"
        if df_kw > 0:
            p_val_calc_kw_num = stats.chi2.sf(test_stat_h_kw, df_kw) 
            
            if isinstance(crit_val_chi2_kw_plot, (int, float)) and not np.isnan(crit_val_chi2_kw_plot):
                summary_crit_val_chi2_kw_display_str = format_value_for_display(crit_val_chi2_kw_plot)
                decision_crit_kw = test_stat_h_kw > crit_val_chi2_kw_plot
                comparison_crit_str_kw = f"H({format_value_for_display(test_stat_h_kw)}) > χ²_crit({format_value_for_display(crit_val_chi2_kw_plot)})" if decision_crit_kw else f"H({format_value_for_display(test_stat_h_kw)}) ≤ χ²_crit({format_value_for_display(crit_val_chi2_kw_plot)})"
            else:
                summary_crit_val_chi2_kw_display_str = "N/A (calc error)"
                comparison_crit_str_kw = "Comparison not possible (critical value is N/A or NaN)"
            
            if isinstance(p_val_calc_kw_num, (int, float)) and not np.isnan(p_val_calc_kw_num):
                decision_p_alpha_kw = p_val_calc_kw_num < alpha_kw
        
        p_val_calc_kw_num_display_str = format_value_for_display(p_val_calc_kw_num, decimals=4)
        apa_p_val_calc_kw_str = apa_p_value(p_val_calc_kw_num)


        st.markdown(f"""
        1.  **Critical χ²-value (df={df_kw})**: {summary_crit_val_chi2_kw_display_str}
            * *Associated p-value (α)*: {alpha_kw:.4f}
        2.  **Calculated H-statistic**: {format_value_for_display(test_stat_h_kw)}
            * *Calculated p-value (from χ² approx.)*: {p_val_calc_kw_num_display_str} ({apa_p_val_calc_kw_str})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_kw else 'not rejected'}**.
            * *Reason*: {comparison_crit_str_kw}.
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_kw else 'not rejected'}**.
            * *Reason*: {apa_p_val_calc_kw_str} is {'less than' if decision_p_alpha_kw else 'not less than'} α ({alpha_kw:.4f}).
        5.  **APA 7 Style Report**:
            A Kruskal-Wallis H test showed that there was {'' if decision_p_alpha_kw else 'not '}a statistically significant difference in medians between the k={k_groups_kw} groups, {apa_H_stat}, {apa_p_val_calc_kw_str}. The null hypothesis was {'rejected' if decision_p_alpha_kw else 'not rejected'} at α = {alpha_kw:.2f}.
        """)


def tab_friedman_test():
    st.header("Friedman Test (Chi-square Approximation)")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_fr = st.number_input("Alpha (α)", 0.0001, 0.5, 0.05, 0.0001, format="%.4f", key="alpha_fr_input")
        k_conditions_fr = st.number_input("Number of Conditions/Treatments (k)", 2, 50, 3, 1, key="k_conditions_fr_input") 
        n_blocks_fr = st.number_input("Number of Blocks/Subjects (n)", 2, 200, 10, 1, key="n_blocks_fr_input") 
        
        df_fr = k_conditions_fr - 1
        st.markdown(f"Degrees of Freedom (df) = k - 1 = {df_fr}")
        test_stat_q_fr = st.number_input("Calculated Friedman Q-statistic (or χ²_r)", value=float(df_fr if df_fr > 0 else 0.5), format="%.3f", min_value=0.0, key="test_stat_q_fr_input")

        if n_blocks_fr <= 10 or k_conditions_fr <= 3 : 
            st.warning("Small n or k. Friedman’s χ² approximation may be less reliable. Exact methods preferred if available.")

        st.subheader("Chi-square Distribution Plot (Approximation for Q)")
        fig_fr, ax_fr = plt.subplots(figsize=(8,5))
        crit_val_chi2_fr_plot = None 
        
        if df_fr > 0:
            crit_val_chi2_fr_plot = stats.chi2.ppf(1 - alpha_fr, df_fr)
            plot_min_chi2_fr = 0.001
            plot_max_chi2_fr = max(stats.chi2.ppf(0.999, df_fr), test_stat_q_fr * 1.5, 10.0)
            if test_stat_q_fr > stats.chi2.ppf(0.999, df_fr) * 1.2:
                plot_max_chi2_fr = test_stat_q_fr * 1.2

            x_chi2_fr_plot = np.linspace(plot_min_chi2_fr, plot_max_chi2_fr, 500)
            y_chi2_fr_plot = stats.chi2.pdf(x_chi2_fr_plot, df_fr)
            ax_fr.plot(x_chi2_fr_plot, y_chi2_fr_plot, 'b-', lw=2, label=f'χ²-distribution (df={df_fr})')
            
            if isinstance(crit_val_chi2_fr_plot, (int,float)) and not np.isnan(crit_val_chi2_fr_plot):
                x_fill_upper_fr = np.linspace(crit_val_chi2_fr_plot, plot_max_chi2_fr, 100)
                ax_fr.fill_between(x_fill_upper_fr, stats.chi2.pdf(x_fill_upper_fr, df_fr), color='red', alpha=0.5, label=f'α = {alpha_fr:.4f}')
                ax_fr.axvline(crit_val_chi2_fr_plot, color='red', linestyle='--', lw=1, label=f'χ²_crit = {crit_val_chi2_fr_plot:.3f}')
            
            ax_fr.axvline(test_stat_q_fr, color='green', linestyle='-', lw=2, label=f'Q_calc = {test_stat_q_fr:.3f}')
            ax_fr.set_title(f'χ²-Approximation for Friedman Q (df={df_fr})')
        else:
            ax_fr.text(0.5, 0.5, "df must be > 0 (k > 1 for meaningful test)", ha='center', va='center')
            ax_fr.set_title('Plot Unavailable (df=0)')
        ax_fr.legend()
        ax_fr.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_fr)

        st.subheader("Critical χ²-Values (Upper Tail)") # Same table as Kruskal-Wallis
        all_df_chi2_options_fr = list(range(1, 31)) + [35, 40, 45, 50, 60, 70, 80, 90, 100]
        table_df_window_chi2_fr = get_dynamic_df_window(all_df_chi2_options_fr, df_fr, window_size=5)
        table_alpha_cols_chi2_fr = [0.10, 0.05, 0.025, 0.01, 0.005]

        chi2_table_rows_fr = []
        for df_iter_val in table_df_window_chi2_fr: # df_iter_val is already a number
            df_iter_calc = int(df_iter_val)
            row_data = {'df': str(df_iter_val)}
            for alpha_c in table_alpha_cols_chi2_fr:
                cv = stats.chi2.ppf(1 - alpha_c, df_iter_calc)
                row_data[f"α = {alpha_c:.3f}"] = format_value_for_display(cv)
            chi2_table_rows_fr.append(row_data)
        df_chi2_table_fr = pd.DataFrame(chi2_table_rows_fr).set_index('df')

        def style_chi2_table_fr(df_to_style): # Similar styling to KW
            style = pd.DataFrame('', index=df_to_style.index, columns=df_to_style.columns)
            selected_df_str = str(df_fr) 
            
            if selected_df_str in df_to_style.index:
                style.loc[selected_df_str, :] = 'background-color: lightblue;'
            
            closest_alpha_col_val = min(table_alpha_cols_chi2_fr, key=lambda x: abs(x - alpha_fr))
            highlight_col_name = f"α = {closest_alpha_col_val:.3f}"

            if highlight_col_name in df_to_style.columns:
                for r_idx in df_to_style.index:
                    current_r_style = style.loc[r_idx, highlight_col_name]
                    style.loc[r_idx, highlight_col_name] = (current_r_style if current_r_style else '') + ' background-color: lightgreen;'
                if selected_df_str in df_to_style.index:
                    current_c_style = style.loc[selected_df_str, highlight_col_name]
                    style.loc[selected_df_str, highlight_col_name] = (current_c_style if current_c_style else '') + ' font-weight: bold; border: 2px solid red; background-color: yellow;'
            return style
        
        if df_fr > 0:
            st.markdown(df_chi2_table_fr.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]},
                                                                {'selector': 'td', 'props': [('text-align', 'center')]}])
                                         .apply(style_chi2_table_fr, axis=None).to_html(), unsafe_allow_html=True)
            st.caption(f"Table shows upper-tail critical χ²-values. Highlighted for df={df_fr} and α closest to your test.")
        else:
            st.warning("df must be > 0 to generate table (k > 1).")
        st.markdown("""
        **Cumulative Table Note:** Friedman Q statistic is approx. χ² distributed (df = k-1). Test is right-tailed.
        """)

    with col2:
        st.subheader("P-value Calculation Explanation")
        st.markdown(f"""
        P-value is P(χ² ≥ Q_calc) assuming H₀ (all condition medians are equal across blocks) is true.
        * `P(χ² ≥ {test_stat_q_fr:.3f}) = stats.chi2.sf({test_stat_q_fr:.3f}, df={df_fr})` (if df > 0)
        """)

        st.subheader("Summary")
        p_val_calc_fr_num = float('nan') 
        decision_crit_fr = False
        comparison_crit_str_fr = "Test not valid (df must be > 0)"
        decision_p_alpha_fr = False
        apa_Q_stat = f"χ²<sub>r</sub>({df_fr if df_fr > 0 else 'N/A'}) = {format_value_for_display(test_stat_q_fr, decimals=2)}"
        
        summary_crit_val_chi2_fr_display_str = "N/A (df=0)"
        if df_fr > 0:
            p_val_calc_fr_num = stats.chi2.sf(test_stat_q_fr, df_fr) 
            if isinstance(crit_val_chi2_fr_plot, (int,float)) and not np.isnan(crit_val_chi2_fr_plot):
                summary_crit_val_chi2_fr_display_str = f"{crit_val_chi2_fr_plot:.3f}"
                decision_crit_fr = test_stat_q_fr > crit_val_chi2_fr_plot
                comparison_crit_str_fr = f"Q({test_stat_q_fr:.3f}) > χ²_crit({crit_val_chi2_fr_plot:.3f})" if decision_crit_fr else f"Q({test_stat_q_fr:.3f}) ≤ χ²_crit({crit_val_chi2_fr_plot:.3f})"
            else: 
                summary_crit_val_chi2_fr_display_str = "N/A (calc error)"
                comparison_crit_str_fr = "Comparison not possible (critical value is N/A or NaN)"

            if isinstance(p_val_calc_fr_num, (int, float)) and not np.isnan(p_val_calc_fr_num):
                decision_p_alpha_fr = p_val_calc_fr_num < alpha_fr
        
        p_val_calc_fr_num_display_str = format_value_for_display(p_val_calc_fr_num, decimals=4)
        apa_p_val_calc_fr_str = apa_p_value(p_val_calc_fr_num)

        st.markdown(f"""
        1.  **Critical χ²-value (df={df_fr})**: {summary_crit_val_chi2_fr_display_str}
            * *Associated p-value (α)*: {alpha_fr:.4f}
        2.  **Calculated Q-statistic (χ²_r)**: {format_value_for_display(test_stat_q_fr)}
            * *Calculated p-value (from χ² approx.)*: {p_val_calc_fr_num_display_str} ({apa_p_val_calc_fr_str})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_fr else 'not rejected'}**.
            * *Reason*: {comparison_crit_str_fr}.
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_fr else 'not rejected'}**.
            * *Reason*: {apa_p_val_calc_fr_str} is {'less than' if decision_p_alpha_fr else 'not less than'} α ({alpha_fr:.4f}).
        5.  **APA 7 Style Report**:
            A Friedman test indicated that there was {'' if decision_p_alpha_fr else 'not '}a statistically significant difference in medians across the k={k_conditions_fr} conditions for n={n_blocks_fr} blocks, {apa_Q_stat}, {apa_p_val_calc_fr_str}. The null hypothesis was {'rejected' if decision_p_alpha_fr else 'not rejected'} at α = {alpha_fr:.2f}.
        """, unsafe_allow_html=True)


# --- Main app ---
def main():
    st.set_page_config(page_title="Statistical Table Explorer", layout="wide")
    st.title("🔢 Statistical Table Explorer")
    st.markdown("""
    This application provides an interactive way to explore various statistical distributions and tests. 
    Select a tab to begin. On each tab, you can adjust parameters like alpha, degrees of freedom, 
    and input a calculated test statistic to see how it compares to critical values and to understand p-value calculations.
    Ensure you have `statsmodels` installed for full functionality on the Tukey HSD tab (`pip install statsmodels`).
    """)

    tab_names = [
        "t-Distribution", "z-Distribution", "F-Distribution", "Chi-square (χ²)",
        "Mann-Whitney U", "Wilcoxon Signed-Rank T", "Binomial Test",
        "Tukey HSD", "Kruskal-Wallis H", "Friedman Test"
    ]
    
    tabs = st.tabs(tab_names)

    with tabs[0]:
        tab_t_distribution()
    with tabs[1]:
        tab_z_distribution()
    with tabs[2]:
        tab_f_distribution()
    with tabs[3]:
        tab_chi_square_distribution()
    with tabs[4]:
        tab_mann_whitney_u()
    with tabs[5]:
        tab_wilcoxon_t()
    with tabs[6]:
        tab_binomial_test()
    with tabs[7]:
        tab_tukey_hsd()
    with tabs[8]:
        tab_kruskal_wallis()
    with tabs[9]:
        tab_friedman_test()

if __name__ == "__main__":
    main()
