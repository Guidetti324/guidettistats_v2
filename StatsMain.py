import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from io import StringIO # For reading CSV string

# Helper function to create APA style p-value string
def apa_p_value(p_val):
    # Ensure p_val is a float or int before checking isnan, otherwise np.isnan can fail
    if not isinstance(p_val, (int, float)) or np.isnan(p_val):
        return "p N/A"
    
    try:
        p_val_float = float(p_val) # Ensure it's a float for comparison
        if p_val_float < 0.001:
            return "p < .001"
        else:
            return f"p = {p_val_float:.3f}"
    except (ValueError, TypeError):
        return "p N/A (format err)"


# Embedded CSV data for Tukey HSD fallback
TUKEY_CSV_DATA = """df,k,alpha_0.01,alpha_0.05,alpha_0.10
1,2,90.030,17.970,8.990
1,3,135.000,26.980,13.480
1,4,164.300,32.820,16.360
1,5,185.700,37.080,18.480
1,6,202.200,40.410,20.150
2,2,14.000,6.085,3.927
2,3,19.020,8.331,5.040
2,4,22.290,9.798,5.757
2,5,24.720,10.880,6.286
2,6,26.630,11.740,6.701
3,2,8.260,4.501,3.182
3,3,10.620,5.910,3.953
3,4,12.170,6.825,4.498
3,5,13.330,7.515,4.903
3,6,14.240,8.037,5.221
5,2,5.700,3.639,2.768
5,3,6.980,4.602,3.401
5,4,7.800,5.218,3.813
5,5,8.420,5.673,4.102
5,6,8.910,6.033,4.328
5,10,10.850,7.540,5.350
10,2,4.470,3.151,2.409
10,3,5.270,3.877,2.913
10,4,5.830,4.327,3.240
10,5,6.260,4.654,3.481
10,6,6.620,4.909,3.671
10,10,7.940,6.076,4.500
20,2,3.960,2.950,2.280
20,3,4.640,3.578,2.722
20,4,5.060,3.983,3.000
20,5,5.390,4.295,3.207
20,6,5.660,4.544,3.372
20,10,6.770,5.556,4.080
20,20,8.000,6.800,5.000
60,2,3.520,2.756,2.116
60,3,4.100,3.314,2.523
60,4,4.480,3.631,2.762
60,5,4.770,3.859,2.933
60,6,4.990,4.039,3.066
60,10,5.830,4.823,3.620
60,20,6.970,5.890,4.400
120,2,3.360,2.617,2.000
120,3,3.980,3.356,2.500
120,4,4.360,3.685,2.750
120,5,4.650,3.919,2.920
120,6,4.850,4.103,3.050
120,10,5.500,4.686,3.450
120,20,6.800,5.300,3.980
"""

# Function to get Tukey q critical value from CSV
def get_tukey_q_from_csv(df_error, k, alpha):
    try:
        df_tukey = pd.read_csv(StringIO(TUKEY_CSV_DATA))
    except Exception as e:
        st.error(f"Error reading embedded Tukey CSV data: {e}")
        return None

    alpha_col_map = {0.01: 'alpha_0.01', 0.05: 'alpha_0.05', 0.10: 'alpha_0.10'}
    alpha_lookup_key = alpha
    if alpha not in alpha_col_map:
        st.warning(f"Alpha value {alpha:.4f} not directly available in CSV (0.01, 0.05, 0.10). Using alpha=0.05 for CSV lookup if exact alpha column not present.")
        alpha_lookup_key = 0.05 
    
    target_col = alpha_col_map.get(alpha_lookup_key, 'alpha_0.05')

    df_filtered_k = df_tukey[df_tukey['k'] == k]
    k_to_use = k
    if df_filtered_k.empty:
        available_k = sorted(df_tukey['k'].unique())
        lower_k_values = [val for val in available_k if val < k]
        if not lower_k_values: 
            k_to_use = min(available_k)
            st.warning(f"k value {k} is smaller than any k in CSV. Using smallest available k={k_to_use}.")
        else:
            k_to_use = max(lower_k_values)
            st.warning(f"Exact k={k} not found in CSV. Using nearest lower k={k_to_use}.")
        df_filtered_k = df_tukey[df_tukey['k'] == k_to_use]
        if df_filtered_k.empty:
            st.error(f"Could not find data for k={k_to_use} in CSV after attempting fallback.")
            return None

    df_filtered_k_sorted = df_filtered_k.sort_values('df')
    exact_match = df_filtered_k_sorted[df_filtered_k_sorted['df'] == df_error]
    if not exact_match.empty:
        return exact_match.iloc[0][target_col]

    lower_dfs = df_filtered_k_sorted[df_filtered_k_sorted['df'] < df_error]
    if not lower_dfs.empty:
        chosen_row = lower_dfs.iloc[-1]
        st.warning(f"Exact df={df_error} not found for k={k_to_use} in CSV. Using nearest lower df={chosen_row['df']}.")
        return chosen_row[target_col]

    higher_dfs = df_filtered_k_sorted[df_filtered_k_sorted['df'] > df_error]
    if not higher_dfs.empty:
        chosen_row = higher_dfs.iloc[0]
        st.warning(f"Exact df={df_error} not found for k={k_to_use} in CSV, no lower df available. Using nearest higher df={chosen_row['df']}.")
        return chosen_row[target_col]
        
    st.error(f"Could not find a suitable value in CSV for df={df_error}, k={k_to_use}, alpha={alpha_lookup_key:.4f}.")
    return None


# --- Tab 1: t-distribution ---
def tab_t_distribution():
    st.header("t-Distribution Explorer")
    col1, col2 = st.columns([2, 1.5]) 

    with col1:
        st.subheader("Inputs")
        alpha_t = st.number_input("Alpha (α)", 0.0001, 0.5, 0.05, 0.0001, format="%.4f", key="alpha_t")
        
        df_options = list(range(1, 31)) + [40, 60, 80, 100, 1000, 'z (∞)']
        df_t_display = st.selectbox("Degrees of Freedom (df) for Table Lookup & Plot", options=df_options, index=9, key="df_t_display_selectbox") 

        if df_t_display == 'z (∞)':
            df_t_calc = np.inf
        else:
            df_t_calc = int(df_t_display)

        tail_t = st.radio("Tail Selection", ("Two-tailed", "One-tailed (right)", "One-tailed (left)"), key="tail_t")
        test_stat_t = st.number_input("Calculated t-statistic", value=0.0, format="%.3f", key="test_stat_t")

        st.subheader("Distribution Plot")
        fig_t, ax_t = plt.subplots(figsize=(8,5)) 
        
        if np.isinf(df_t_calc): 
            dist_label = 'Standard Normal (z)'
            plot_min_t = min(stats.norm.ppf(0.0001), test_stat_t - 2, -4.0)
            plot_max_t = max(stats.norm.ppf(0.9999), test_stat_t + 2, 4.0)
            x_t = np.linspace(plot_min_t, plot_max_t, 500)
            y_t = stats.norm.pdf(x_t)
            crit_func_ppf = stats.norm.ppf
            crit_func_pdf = stats.norm.pdf
        else: 
            dist_label = f't-distribution (df={df_t_calc})'
            std_dev_t = stats.t.std(df_t_calc) if df_t_calc > 0 and not np.isinf(df_t_calc) else 1 
            plot_min_t = min(stats.t.ppf(0.0001, df_t_calc), test_stat_t - 2*std_dev_t, -4.0)
            plot_max_t = max(stats.t.ppf(0.9999, df_t_calc), test_stat_t + 2*std_dev_t, 4.0)
            x_t = np.linspace(plot_min_t, plot_max_t, 500)
            y_t = stats.t.pdf(x_t, df_t_calc)
            crit_func_ppf = lambda q: stats.t.ppf(q, df_t_calc)
            crit_func_pdf = lambda x_val: stats.t.pdf(x_val, df_t_calc)


        if abs(test_stat_t) > 4 and abs(test_stat_t) > plot_max_t * 0.8 : 
            plot_min_t = min(plot_min_t, test_stat_t -1)
            plot_max_t = max(plot_max_t, test_stat_t +1)
            x_t = np.linspace(plot_min_t, plot_max_t, 500) 
            y_t = crit_func_pdf(x_t)


        ax_t.plot(x_t, y_t, 'b-', lw=2, label=dist_label)
        crit_val_t_upper = None
        crit_val_t_lower = None

        if tail_t == "Two-tailed":
            crit_val_t_upper = crit_func_ppf(1 - alpha_t / 2)
            crit_val_t_lower = crit_func_ppf(alpha_t / 2)
            if crit_val_t_upper is not None and not np.isnan(crit_val_t_upper):
                 x_fill_upper = np.linspace(crit_val_t_upper, plot_max_t, 100)
                 ax_t.fill_between(x_fill_upper, crit_func_pdf(x_fill_upper), color='red', alpha=0.5, label=f'α/2 = {alpha_t/2:.4f}')
                 ax_t.axvline(crit_val_t_upper, color='red', linestyle='--', lw=1)
            if crit_val_t_lower is not None and not np.isnan(crit_val_t_lower):
                 x_fill_lower = np.linspace(plot_min_t, crit_val_t_lower, 100)
                 ax_t.fill_between(x_fill_lower, crit_func_pdf(x_fill_lower), color='red', alpha=0.5)
                 ax_t.axvline(crit_val_t_lower, color='red', linestyle='--', lw=1)
        elif tail_t == "One-tailed (right)":
            crit_val_t_upper = crit_func_ppf(1 - alpha_t)
            if crit_val_t_upper is not None and not np.isnan(crit_val_t_upper):
                x_fill_upper = np.linspace(crit_val_t_upper, plot_max_t, 100)
                ax_t.fill_between(x_fill_upper, crit_func_pdf(x_fill_upper), color='red', alpha=0.5, label=f'α = {alpha_t:.4f}')
                ax_t.axvline(crit_val_t_upper, color='red', linestyle='--', lw=1)
        else: 
            crit_val_t_lower = crit_func_ppf(alpha_t)
            if crit_val_t_lower is not None and not np.isnan(crit_val_t_lower):
                x_fill_lower = np.linspace(plot_min_t, crit_val_t_lower, 100)
                ax_t.fill_between(x_fill_lower, crit_func_pdf(x_fill_lower), color='red', alpha=0.5, label=f'α = {alpha_t:.4f}')
                ax_t.axvline(crit_val_t_lower, color='red', linestyle='--', lw=1)

        ax_t.axvline(test_stat_t, color='green', linestyle='-', lw=2, label=f'Test Stat = {test_stat_t:.3f}')
        ax_t.set_title(f'{dist_label} with Critical Region(s)')
        ax_t.set_xlabel('t-value' if not np.isinf(df_t_calc) else 'z-value')
        ax_t.set_ylabel('Probability Density')
        ax_t.legend()
        ax_t.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_t)

        st.subheader("t-Value Table")
        one_tail_alphas_cols = [0.25, 0.10, 0.05, 0.025, 0.01, 0.005, 0.001, 0.0005]
                
        df_display_table_rows = []
        for df_val_iter_display in df_options:
            row_dict = {'df': str(df_val_iter_display)}
            current_df_iter_calc = np.inf if df_val_iter_display == 'z (∞)' else int(df_val_iter_display)
            for one_alpha_col_val in one_tail_alphas_cols: 
                if np.isinf(current_df_iter_calc):
                    crit_val_for_col = stats.norm.ppf(1-one_alpha_col_val)
                else:
                    crit_val_for_col = stats.t.ppf(1-one_alpha_col_val, current_df_iter_calc)
                
                formatted_cv = f"{crit_val_for_col:.3f}" if not np.isnan(crit_val_for_col) else "N/A"
                row_dict[f"α₁={one_alpha_col_val:.4f}"] = formatted_cv
                # For the two-tail column, the value is the same (it's t_alpha/2), but the header indicates 2*alpha
                row_dict[f"α₂={one_alpha_col_val*2:.4f}"] = formatted_cv 
            df_display_table_rows.append(row_dict)

        df_t_table_display = pd.DataFrame(df_display_table_rows).set_index('df')

        # Styling function
        def highlight_t_table_flat(dataframe_to_style): # Renamed arg for clarity
            # When Styler.apply is used with axis=None, the function receives the DataFrame.
            data = dataframe_to_style 
            attr_df = pd.DataFrame('', index=data.index, columns=data.columns)
            selected_df_str = str(df_t_display) # df_t_display is from the selectbox

            # Highlight selected DF row
            if selected_df_str in data.index:
                attr_df.loc[selected_df_str, :] = 'background-color: lightblue;'
            
            target_alpha_for_style = alpha_t # User's input alpha
            highlight_col_name = None # The column name in df_t_table_display to highlight
            
            if tail_t == "Two-tailed":
                # User's alpha_t is the two-tail alpha. Find the column header α₂=alpha_t
                # The actual critical value comes from one-tailed alpha_t/2
                
                # Find the two-tail column header closest to user's alpha_t
                closest_two_tail_header_alpha_val = min([a*2 for a in one_tail_alphas_cols], key=lambda x:abs(x-target_alpha_for_style))
                highlight_col_name = f'α₂={closest_two_tail_header_alpha_val:.4f}'

            else: # One-tailed
                # User's alpha_t is the one-tail alpha. Find the column header α₁=alpha_t
                closest_one_tail_header_alpha_val = min(one_tail_alphas_cols, key=lambda x:abs(x-target_alpha_for_style))
                highlight_col_name = f'α₁={closest_one_tail_header_alpha_val:.4f}'

            # Apply column and cell highlighting
            if highlight_col_name and highlight_col_name in data.columns:
                # Apply style to the entire column first
                for r_idx in data.index: # Iterate over all rows for this column
                    current_style = attr_df.loc[r_idx, highlight_col_name]
                    attr_df.loc[r_idx, highlight_col_name] = (current_style if current_style else '') + ' background-color: lightgreen;'

                # Then, apply specific style to the cell
                if selected_df_str in data.index:
                    current_cell_style = attr_df.loc[selected_df_str, highlight_col_name]
                    attr_df.loc[selected_df_str, highlight_col_name] = (current_cell_style if current_cell_style else '') + ' font-weight: bold; border: 2px solid red;'
            
            return attr_df

        st.markdown(df_t_table_display.style.set_table_styles([{'selector': 'th', 'props': [('font-size', '10pt'), ('text-align', 'center')]},
                                                       {'selector': 'td', 'props': [('text-align', 'center')]}])
                                     .apply(highlight_t_table_flat, axis=None).to_html(), unsafe_allow_html=True)
        st.caption(f"Highlighted row for df={df_t_display}, column for selected α={alpha_t:.4f} ({tail_t}), and specific critical value in red. α₁: One-Tail, α₂: Two-Tails.")


        st.markdown("""
        **Cumulative Table Note:**
        * The table shows upper critical values (t<sub>α</sub>). For left-tailed tests, use the negative of these values.
        * For **one-tailed tests**, find your df and the column `α₁` matching your chosen α.
        * For **two-tailed tests**, find your df and the column `α₂` matching your total chosen α. The value shown is t<sub>α/2</sub>.
        """)

    with col2:
        st.subheader("P-value Calculation Explanation")
        if np.isinf(df_t_calc):
            p_val_func_sf = stats.norm.sf
            p_val_func_cdf = stats.norm.cdf
            dist_name_p = "Z"
        else:
            p_val_func_sf = lambda val: stats.t.sf(val, df_t_calc)
            p_val_func_cdf = lambda val: stats.t.cdf(val, df_t_calc)
            dist_name_p = "T"

        st.markdown(f"""
        The p-value is the probability of observing a test statistic as extreme as, or more extreme than, the calculated statistic ({test_stat_t:.3f}), assuming the null hypothesis is true.
        * For a **two-tailed test**, it's `2 * P({dist_name_p} ≥ |{test_stat_t:.3f}|)`.
        * For a **one-tailed (right) test**, it's `P({dist_name_p} ≥ {test_stat_t:.3f})`.
        * For a **one-tailed (left) test**, it's `P({dist_name_p} ≤ {test_stat_t:.3f})`.
        """)

        st.subheader("Summary")
        p_val_t_one_right = p_val_func_sf(test_stat_t)
        p_val_t_one_left = p_val_func_cdf(test_stat_t)
        p_val_t_two = 2 * p_val_func_sf(abs(test_stat_t))
        p_val_t_two = min(p_val_t_two, 1.0) 


        crit_val_display = "N/A"
        p_val_for_crit_val_display = alpha_t 

        if tail_t == "Two-tailed":
            crit_val_display = f"±{crit_val_t_upper:.3f}" if crit_val_t_upper is not None and not np.isnan(crit_val_t_upper) else "N/A"
            p_val_calc = p_val_t_two
            decision_crit = abs(test_stat_t) > crit_val_t_upper if crit_val_t_upper is not None and not np.isnan(crit_val_t_upper) else False
            comparison_crit_str = f"|{test_stat_t:.3f}| ({abs(test_stat_t):.3f}) > {crit_val_t_upper:.3f}" if decision_crit else f"|{test_stat_t:.3f}| ({abs(test_stat_t):.3f}) ≤ {crit_val_t_upper:.3f}"
        elif tail_t == "One-tailed (right)":
            crit_val_display = f"{crit_val_t_upper:.3f}" if crit_val_t_upper is not None and not np.isnan(crit_val_t_upper) else "N/A"
            p_val_calc = p_val_t_one_right
            decision_crit = test_stat_t > crit_val_t_upper if crit_val_t_upper is not None and not np.isnan(crit_val_t_upper) else False
            comparison_crit_str = f"{test_stat_t:.3f} > {crit_val_t_upper:.3f}" if decision_crit else f"{test_stat_t:.3f} ≤ {crit_val_t_upper:.3f}"
        else: 
            crit_val_display = f"{crit_val_t_lower:.3f}" if crit_val_t_lower is not None and not np.isnan(crit_val_t_lower) else "N/A"
            p_val_calc = p_val_t_one_left
            decision_crit = test_stat_t < crit_val_t_lower if crit_val_t_lower is not None and not np.isnan(crit_val_t_lower) else False
            comparison_crit_str = f"{test_stat_t:.3f} < {crit_val_t_lower:.3f}" if decision_crit else f"{test_stat_t:.3f} ≥ {crit_val_t_lower:.3f}"

        decision_p_alpha = p_val_calc < alpha_t
        
        df_report_str = "∞" if np.isinf(df_t_calc) else str(df_t_calc)
        stat_symbol = "z" if np.isinf(df_t_calc) else "t"

        st.markdown(f"""
        1.  **Critical Value ({tail_t})**: {crit_val_display}
            * *Associated p-value (α or α/2 per tail)*: {p_val_for_crit_val_display:.4f}
        2.  **Calculated Test Statistic**: {test_stat_t:.3f}
            * *Calculated p-value*: {p_val_calc:.4f} ({apa_p_value(p_val_calc)})
        3.  **Decision (Critical Value Method)**: The null hypothesis is **{'rejected' if decision_crit else 'not rejected'}**.
            * *Reason*: Because {stat_symbol}(calc) {comparison_crit_str} relative to {stat_symbol}(crit).
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha else 'not rejected'}**.
            * *Reason*: Because {apa_p_value(p_val_calc)} is {'less than' if decision_p_alpha else 'not less than'} α ({alpha_t:.4f}).
        5.  **APA 7 Style Report**:
            *{stat_symbol}*({df_report_str}) = {test_stat_t:.2f}, {apa_p_value(p_val_calc)}. The null hypothesis was {'rejected' if decision_p_alpha else 'not rejected'} at the α = {alpha_t:.2f} level.
        """)

# --- Tab 2: z-distribution (No changes from previous version, assumed to be working) ---
def tab_z_distribution():
    st.header("z-Distribution (Normal) Explorer")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_z = st.number_input("Alpha (α)", 0.0001, 0.5, 0.05, 0.0001, format="%.4f", key="alpha_z")
        tail_z = st.radio("Tail Selection", ("Two-tailed", "One-tailed (right)", "One-tailed (left)"), key="tail_z")
        test_stat_z = st.number_input("Calculated z-statistic", value=0.0, format="%.3f", key="test_stat_z")

        st.subheader("Distribution Plot")
        fig_z, ax_z = plt.subplots(figsize=(8,5))
        
        plot_min_z = min(stats.norm.ppf(0.0001), test_stat_z - 2, -4.0)
        plot_max_z = max(stats.norm.ppf(0.9999), test_stat_z + 2, 4.0)
        if abs(test_stat_z) > 4:
            plot_min_z = min(plot_min_z, test_stat_z -1)
            plot_max_z = max(plot_max_z, test_stat_z +1)

        x_z = np.linspace(plot_min_z, plot_max_z, 500)
        y_z = stats.norm.pdf(x_z)
        ax_z.plot(x_z, y_z, 'b-', lw=2, label='Standard Normal Distribution (z)')

        crit_val_z_upper = None
        crit_val_z_lower = None

        if tail_z == "Two-tailed":
            crit_val_z_upper = stats.norm.ppf(1 - alpha_z / 2)
            crit_val_z_lower = stats.norm.ppf(alpha_z / 2)
            if crit_val_z_upper is not None and not np.isnan(crit_val_z_upper):
                x_fill_upper = np.linspace(crit_val_z_upper, plot_max_z, 100)
                ax_z.fill_between(x_fill_upper, stats.norm.pdf(x_fill_upper), color='red', alpha=0.5, label=f'α/2 = {alpha_z/2:.4f}')
                ax_z.axvline(crit_val_z_upper, color='red', linestyle='--', lw=1)
            if crit_val_z_lower is not None and not np.isnan(crit_val_z_lower):
                x_fill_lower = np.linspace(plot_min_z, crit_val_z_lower, 100)
                ax_z.fill_between(x_fill_lower, stats.norm.pdf(x_fill_lower), color='red', alpha=0.5)
                ax_z.axvline(crit_val_z_lower, color='red', linestyle='--', lw=1)
        elif tail_z == "One-tailed (right)":
            crit_val_z_upper = stats.norm.ppf(1 - alpha_z)
            if crit_val_z_upper is not None and not np.isnan(crit_val_z_upper):
                x_fill_upper = np.linspace(crit_val_z_upper, plot_max_z, 100)
                ax_z.fill_between(x_fill_upper, stats.norm.pdf(x_fill_upper), color='red', alpha=0.5, label=f'α = {alpha_z:.4f}')
                ax_z.axvline(crit_val_z_upper, color='red', linestyle='--', lw=1)
        else: 
            crit_val_z_lower = stats.norm.ppf(alpha_z)
            if crit_val_z_lower is not None and not np.isnan(crit_val_z_lower):
                x_fill_lower = np.linspace(plot_min_z, crit_val_z_lower, 100)
                ax_z.fill_between(x_fill_lower, stats.norm.pdf(x_fill_lower), color='red', alpha=0.5, label=f'α = {alpha_z:.4f}')
                ax_z.axvline(crit_val_z_lower, color='red', linestyle='--', lw=1)

        ax_z.axvline(test_stat_z, color='green', linestyle='-', lw=2, label=f'Test Stat = {test_stat_z:.3f}')
        ax_z.set_title('Standard Normal Distribution with Critical Region(s)')
        ax_z.set_xlabel('z-value')
        ax_z.set_ylabel('Probability Density')
        ax_z.legend()
        ax_z.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_z)

        st.subheader("Critical Value Table Snippet")
        alphas_table_z_list = [0.10, 0.05, 0.025, 0.01, 0.005, alpha_z, alpha_z/2 if tail_z == "Two-tailed" else alpha_z]
        alphas_table_z_list = sorted(list(set(a for a in alphas_table_z_list if 0.00005 < a < 0.50005)))

        table_data_z_list = []
        for a_val_one_tail in alphas_table_z_list:
            a_val_two_tail = a_val_one_tail * 2
            cv_upper = stats.norm.ppf(1 - a_val_one_tail)
            cv_lower = stats.norm.ppf(a_val_one_tail)
            table_data_z_list.append({
                "α (One-Tail)": f"{a_val_one_tail:.4f}",
                "α (Two-Tail)": f"{a_val_two_tail:.4f}" if a_val_two_tail <= 0.51 else "-",
                "z_crit (Lower)": f"{cv_lower:.3f}",
                "z_crit (Upper)": f"{cv_upper:.3f}"
            })
        df_table_z = pd.DataFrame(table_data_z_list)
        
        def highlight_alpha_row_z(row):
            highlight = False
            if tail_z == "Two-tailed":
                if abs(float(row["α (One-Tail)"]) - (alpha_z / 2)) < 1e-5 :
                    highlight = True
            else: 
                if abs(float(row["α (One-Tail)"]) - alpha_z) < 1e-5:
                    highlight = True
            return ['background-color: yellow'] * len(row) if highlight else [''] * len(row)

        st.markdown(df_table_z.style.apply(highlight_alpha_row_z, axis=1).to_html(), unsafe_allow_html=True)
        st.caption(f"Table shows critical z-values. Highlighted row corresponds to your selected α={alpha_z:.4f} ({tail_z}).")
        st.markdown("""
        **Cumulative Table Note:** (Same as t-distribution)
        * For **one-tailed tests**, find your α in 'α (One-Tail)'.
        * For **two-tailed tests**, find α/2 in 'α (One-Tail)'.
        """)

    with col2:
        st.subheader("P-value Calculation Explanation")
        st.markdown(f"""
        The p-value is the probability of observing a z-statistic as extreme as, or more extreme than, {test_stat_z:.3f}, assuming H₀ is true.
        * **Two-tailed**: `2 * P(Z ≥ |{test_stat_z:.3f}|)` (i.e., `2 * stats.norm.sf(abs(test_stat_z))`)
        * **One-tailed (right)**: `P(Z ≥ {test_stat_z:.3f})` (i.e., `stats.norm.sf(test_stat_z)`)
        * **One-tailed (left)**: `P(Z ≤ {test_stat_z:.3f})` (i.e., `stats.norm.cdf(test_stat_z)`)
        """)

        st.subheader("Summary")
        p_val_z_one_right = stats.norm.sf(test_stat_z)
        p_val_z_one_left = stats.norm.cdf(test_stat_z)
        p_val_z_two = 2 * stats.norm.sf(abs(test_stat_z))
        p_val_z_two = min(p_val_z_two, 1.0)

        crit_val_z_display = "N/A"
        p_val_for_crit_val_z_display = alpha_z

        if tail_z == "Two-tailed":
            crit_val_z_display = f"±{crit_val_z_upper:.3f}" if crit_val_z_upper is not None and not np.isnan(crit_val_z_upper) else "N/A"
            p_val_calc_z = p_val_z_two
            decision_crit_z = abs(test_stat_z) > crit_val_z_upper if crit_val_z_upper is not None and not np.isnan(crit_val_z_upper) else False
            comparison_crit_str_z = f"|{test_stat_z:.3f}| ({abs(test_stat_z):.3f}) > {crit_val_z_upper:.3f}" if decision_crit_z else f"|{test_stat_z:.3f}| ({abs(test_stat_z):.3f}) ≤ {crit_val_z_upper:.3f}"
        elif tail_z == "One-tailed (right)":
            crit_val_z_display = f"{crit_val_z_upper:.3f}" if crit_val_z_upper is not None and not np.isnan(crit_val_z_upper) else "N/A"
            p_val_calc_z = p_val_z_one_right
            decision_crit_z = test_stat_z > crit_val_z_upper if crit_val_z_upper is not None and not np.isnan(crit_val_z_upper) else False
            comparison_crit_str_z = f"{test_stat_z:.3f} > {crit_val_z_upper:.3f}" if decision_crit_z else f"{test_stat_z:.3f} ≤ {crit_val_z_upper:.3f}"
        else: 
            crit_val_z_display = f"{crit_val_z_lower:.3f}" if crit_val_z_lower is not None and not np.isnan(crit_val_z_lower) else "N/A"
            p_val_calc_z = p_val_z_one_left
            decision_crit_z = test_stat_z < crit_val_z_lower if crit_val_z_lower is not None and not np.isnan(crit_val_z_lower) else False
            comparison_crit_str_z = f"{test_stat_z:.3f} < {crit_val_z_lower:.3f}" if decision_crit_z else f"{test_stat_z:.3f} ≥ {crit_val_z_lower:.3f}"

        decision_p_alpha_z = p_val_calc_z < alpha_z
        
        st.markdown(f"""
        1.  **Critical Value ({tail_z})**: {crit_val_z_display}
            * *Associated p-value (α or α/2 per tail)*: {p_val_for_crit_val_z_display:.4f}
        2.  **Calculated Test Statistic**: {test_stat_z:.3f}
            * *Calculated p-value*: {p_val_calc_z:.4f} ({apa_p_value(p_val_calc_z)})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_z else 'not rejected'}**.
            * *Reason*: z(calc) {comparison_crit_str_z} relative to z(crit).
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_z else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_calc_z)} is {'less than' if decision_p_alpha_z else 'not less than'} α ({alpha_z:.4f}).
        5.  **APA 7 Style Report**:
            *z* = {test_stat_z:.2f}, {apa_p_value(p_val_calc_z)}. The null hypothesis was {'rejected' if decision_p_alpha_z else 'not rejected'} at α = {alpha_z:.2f}.
        """)

# --- Tab 3: F-distribution (No changes from previous version, assumed to be working) ---
def tab_f_distribution():
    st.header("F-Distribution Explorer")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_f = st.number_input("Alpha (α)", 0.0001, 0.5, 0.05, 0.0001, format="%.4f", key="alpha_f")
        dfn_f = st.number_input("Numerator Degrees of Freedom (df1)", 1, 1000, 3, 1, key="dfn_f")
        dfd_f = st.number_input("Denominator Degrees of Freedom (df2)", 1, 1000, 20, 1, key="dfd_f")
        tail_f = st.radio("Tail Selection", ("One-tailed (right)", "Two-tailed (for variance test)"), key="tail_f")
        test_stat_f = st.number_input("Calculated F-statistic", value=1.0, format="%.3f", min_value=0.001, key="test_stat_f")

        st.subheader("Distribution Plot")
        fig_f, ax_f = plt.subplots(figsize=(8,5))
        
        plot_min_f = 0.001
        plot_max_f = max(stats.f.ppf(0.999, dfn_f, dfd_f), test_stat_f * 1.5, 5.0)
        if test_stat_f > stats.f.ppf(0.999, dfn_f, dfd_f) * 1.2 : 
             plot_max_f = test_stat_f * 1.2

        x_f = np.linspace(plot_min_f, plot_max_f, 500)
        y_f = stats.f.pdf(x_f, dfn_f, dfd_f)
        ax_f.plot(x_f, y_f, 'b-', lw=2, label=f'F-dist (df1={dfn_f}, df2={dfd_f})')

        crit_val_f_upper = None
        crit_val_f_lower = None

        if tail_f == "One-tailed (right)":
            crit_val_f_upper = stats.f.ppf(1 - alpha_f, dfn_f, dfd_f)
            if crit_val_f_upper is not None and not np.isnan(crit_val_f_upper):
                x_fill_upper = np.linspace(crit_val_f_upper, plot_max_f, 100)
                ax_f.fill_between(x_fill_upper, stats.f.pdf(x_fill_upper, dfn_f, dfd_f), color='red', alpha=0.5, label=f'α = {alpha_f:.4f}')
                ax_f.axvline(crit_val_f_upper, color='red', linestyle='--', lw=1)
        else: 
            crit_val_f_upper = stats.f.ppf(1 - alpha_f / 2, dfn_f, dfd_f)
            crit_val_f_lower = stats.f.ppf(alpha_f / 2, dfn_f, dfd_f)
            if crit_val_f_upper is not None and not np.isnan(crit_val_f_upper):
                x_fill_upper = np.linspace(crit_val_f_upper, plot_max_f, 100)
                ax_f.fill_between(x_fill_upper, stats.f.pdf(x_fill_upper, dfn_f, dfd_f), color='red', alpha=0.5, label=f'α/2 = {alpha_f/2:.4f}')
                ax_f.axvline(crit_val_f_upper, color='red', linestyle='--', lw=1)
            if crit_val_f_lower is not None and not np.isnan(crit_val_f_lower):
                x_fill_lower = np.linspace(plot_min_f, crit_val_f_lower, 100)
                ax_f.fill_between(x_fill_lower, stats.f.pdf(x_fill_lower, dfn_f, dfd_f), color='red', alpha=0.5)
                ax_f.axvline(crit_val_f_lower, color='red', linestyle='--', lw=1)


        ax_f.axvline(test_stat_f, color='green', linestyle='-', lw=2, label=f'Test Stat = {test_stat_f:.3f}')
        ax_f.set_title(f'F-Distribution (df1={dfn_f}, df2={dfd_f}) with Critical Region(s)')
        ax_f.set_xlabel('F-value')
        ax_f.set_ylabel('Probability Density')
        ax_f.legend()
        ax_f.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_f)

        st.subheader("Critical Value Table Snippet")
        alphas_table_f_list = [0.10, 0.05, 0.025, 0.01, 0.005, alpha_f, alpha_f/2 if tail_f == "Two-tailed (for variance test)" else alpha_f]
        alphas_table_f_list = sorted(list(set(a for a in alphas_table_f_list if 0.00005 < a < 0.50005)))
        
        table_data_f_list = []
        for a_val_one_tail in alphas_table_f_list: 
            cv_upper = stats.f.ppf(1 - a_val_one_tail, dfn_f, dfd_f)
            cv_lower = stats.f.ppf(a_val_one_tail, dfn_f, dfd_f)
            table_data_f_list.append({
                "α (Upper Tail Area)": f"{a_val_one_tail:.4f}",
                "F_crit (Lower)": f"{cv_lower:.3f}",
                "F_crit (Upper)": f"{cv_upper:.3f}"
            })
        df_table_f = pd.DataFrame(table_data_f_list)
        
        def highlight_alpha_row_f(row):
            highlight = False
            current_alpha_in_table = float(row["α (Upper Tail Area)"])
            if tail_f == "Two-tailed (for variance test)":
                if abs(current_alpha_in_table - (alpha_f / 2)) < 1e-5 :
                    highlight = True
            else: 
                if abs(current_alpha_in_table - alpha_f) < 1e-5:
                    highlight = True
            return ['background-color: yellow'] * len(row) if highlight else [''] * len(row)

        st.markdown(df_table_f.style.apply(highlight_alpha_row_f, axis=1).to_html(), unsafe_allow_html=True)
        st.caption(f"Table shows critical F-values for df1={dfn_f}, df2={dfd_f}. Highlighted for your selected α={alpha_f:.4f} ({tail_f}).")
        st.markdown("""
        **Cumulative Table Note:**
        * F-distribution tables typically provide upper-tail critical values. 'α (Upper Tail Area)' is the area to the right of 'F_crit (Upper)'.
        * For **ANOVA (One-tailed right)**, find your α in 'α (Upper Tail Area)' and use 'F_crit (Upper)'.
        * For **Two-tailed variance tests**, find α/2 in 'α (Upper Tail Area)'. Use 'F_crit (Lower)' and 'F_crit (Upper)'.
        """)

    with col2:
        st.subheader("P-value Calculation Explanation")
        st.markdown(f"""
        The p-value is the probability of an F-statistic as extreme as, or more extreme than, {test_stat_f:.3f}.
        * **One-tailed (right)**: `P(F ≥ {test_stat_f:.3f})` (i.e., `stats.f.sf(test_stat_f, dfn_f, dfd_f)`)
        * **Two-tailed (for variance test)**: `2 * min(P(F ≤ F_calc), P(F ≥ F_calc))` (i.e., `2 * min(stats.f.cdf(test_stat_f, dfn_f, dfd_f), stats.f.sf(test_stat_f, dfn_f, dfd_f))`)
        """)

        st.subheader("Summary")
        p_val_f_one_right = stats.f.sf(test_stat_f, dfn_f, dfd_f)
        cdf_f = stats.f.cdf(test_stat_f, dfn_f, dfd_f)
        sf_f = stats.f.sf(test_stat_f, dfn_f, dfd_f)
        p_val_f_two = 2 * min(cdf_f, sf_f)
        p_val_f_two = min(p_val_f_two, 1.0)


        crit_val_f_display = "N/A"
        p_val_for_crit_val_f_display = alpha_f

        if tail_f == "One-tailed (right)":
            crit_val_f_display = f"{crit_val_f_upper:.3f}" if crit_val_f_upper is not None and not np.isnan(crit_val_f_upper) else "N/A"
            p_val_calc_f = p_val_f_one_right
            decision_crit_f = test_stat_f > crit_val_f_upper if crit_val_f_upper is not None and not np.isnan(crit_val_f_upper) else False
            comparison_crit_str_f = f"{test_stat_f:.3f} > {crit_val_f_upper:.3f}" if decision_crit_f else f"{test_stat_f:.3f} ≤ {crit_val_f_upper:.3f}"
        else: 
            crit_val_f_display = f"Lower: {crit_val_f_lower:.3f}, Upper: {crit_val_f_upper:.3f}" if crit_val_f_lower is not None and not np.isnan(crit_val_f_lower) and crit_val_f_upper is not None and not np.isnan(crit_val_f_upper) else "N/A"
            p_val_calc_f = p_val_f_two
            decision_crit_f = (test_stat_f > crit_val_f_upper if crit_val_f_upper is not None and not np.isnan(crit_val_f_upper) else False) or \
                              (test_stat_f < crit_val_f_lower if crit_val_f_lower is not None and not np.isnan(crit_val_f_lower) else False)
            comparison_crit_str_f = f"{test_stat_f:.3f} > {crit_val_f_upper:.3f} or {test_stat_f:.3f} < {crit_val_f_lower:.3f}" if decision_crit_f else f"{crit_val_f_lower:.3f} ≤ {test_stat_f:.3f} ≤ {crit_val_f_upper:.3f}"


        decision_p_alpha_f = p_val_calc_f < alpha_f
        
        st.markdown(f"""
        1.  **Critical Value(s) ({tail_f})**: {crit_val_f_display}
            * *Associated p-value (α or α/2 per tail)*: {p_val_for_crit_val_f_display:.4f}
        2.  **Calculated Test Statistic**: {test_stat_f:.3f}
            * *Calculated p-value*: {p_val_calc_f:.4f} ({apa_p_value(p_val_calc_f)})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_f else 'not rejected'}**.
            * *Reason*: F(calc) {comparison_crit_str_f} relative to F(crit).
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_f else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_calc_f)} is {'less than' if decision_p_alpha_f else 'not less than'} α ({alpha_f:.4f}).
        5.  **APA 7 Style Report**:
            *F*({dfn_f}, {dfd_f}) = {test_stat_f:.2f}, {apa_p_value(p_val_calc_f)}. The null hypothesis was {'rejected' if decision_p_alpha_f else 'not rejected'} at α = {alpha_f:.2f}.
        """)

# --- Tab 4: Chi-square distribution (No changes from previous version, assumed to be working) ---
def tab_chi_square_distribution():
    st.header("Chi-square (χ²) Distribution Explorer")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_chi2 = st.number_input("Alpha (α)", 0.0001, 0.5, 0.05, 0.0001, format="%.4f", key="alpha_chi2")
        df_chi2 = st.number_input("Degrees of Freedom (df)", 1, 1000, 5, 1, key="df_chi2")
        tail_chi2 = st.radio("Tail Selection", ("One-tailed (right)", "Two-tailed (e.g. for variance)"), key="tail_chi2")
        test_stat_chi2 = st.number_input("Calculated χ²-statistic", value=float(df_chi2), format="%.3f", min_value=0.001, key="test_stat_chi2")

        st.subheader("Distribution Plot")
        fig_chi2, ax_chi2 = plt.subplots(figsize=(8,5))
        
        plot_min_chi2 = 0.001
        plot_max_chi2 = max(stats.chi2.ppf(0.999, df_chi2), test_stat_chi2 * 1.5, 10.0)
        if test_stat_chi2 > stats.chi2.ppf(0.999, df_chi2) * 1.2:
            plot_max_chi2 = test_stat_chi2 * 1.2

        x_chi2 = np.linspace(plot_min_chi2, plot_max_chi2, 500) 
        y_chi2 = stats.chi2.pdf(x_chi2, df_chi2)
        ax_chi2.plot(x_chi2, y_chi2, 'b-', lw=2, label=f'χ²-distribution (df={df_chi2})')

        crit_val_chi2_upper = None
        crit_val_chi2_lower = None

        if tail_chi2 == "One-tailed (right)":
            crit_val_chi2_upper = stats.chi2.ppf(1 - alpha_chi2, df_chi2)
            if crit_val_chi2_upper is not None and not np.isnan(crit_val_chi2_upper):
                x_fill_upper = np.linspace(crit_val_chi2_upper, plot_max_chi2, 100)
                ax_chi2.fill_between(x_fill_upper, stats.chi2.pdf(x_fill_upper, df_chi2), color='red', alpha=0.5, label=f'α = {alpha_chi2:.4f}')
                ax_chi2.axvline(crit_val_chi2_upper, color='red', linestyle='--', lw=1)
        else: 
            crit_val_chi2_upper = stats.chi2.ppf(1 - alpha_chi2 / 2, df_chi2)
            crit_val_chi2_lower = stats.chi2.ppf(alpha_chi2 / 2, df_chi2)
            if crit_val_chi2_upper is not None and not np.isnan(crit_val_chi2_upper):
                x_fill_upper_chi2 = np.linspace(crit_val_chi2_upper, plot_max_chi2, 100)
                ax_chi2.fill_between(x_fill_upper_chi2, stats.chi2.pdf(x_fill_upper_chi2, df_chi2), color='red', alpha=0.5, label=f'α/2 = {alpha_chi2/2:.4f}')
                ax_chi2.axvline(crit_val_chi2_upper, color='red', linestyle='--', lw=1)
            if crit_val_chi2_lower is not None and not np.isnan(crit_val_chi2_lower):
                x_fill_lower_chi2 = np.linspace(plot_min_chi2, crit_val_chi2_lower, 100)
                ax_chi2.fill_between(x_fill_lower_chi2, stats.chi2.pdf(x_fill_lower_chi2, df_chi2), color='red', alpha=0.5)
                ax_chi2.axvline(crit_val_chi2_lower, color='red', linestyle='--', lw=1)


        ax_chi2.axvline(test_stat_chi2, color='green', linestyle='-', lw=2, label=f'Test Stat = {test_stat_chi2:.3f}')
        ax_chi2.set_title(f'χ²-Distribution (df={df_chi2}) with Critical Region(s)')
        ax_chi2.set_xlabel('χ²-value')
        ax_chi2.set_ylabel('Probability Density')
        ax_chi2.legend()
        ax_chi2.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_chi2)

        st.subheader("Critical Value Table Snippet")
        alphas_table_chi2_list = [0.10, 0.05, 0.025, 0.01, 0.005, alpha_chi2, alpha_chi2/2 if tail_chi2 == "Two-tailed (e.g. for variance)" else alpha_chi2]
        alphas_table_chi2_list = sorted(list(set(a for a in alphas_table_chi2_list if 0.00005 < a < 0.50005)))

        table_data_chi2_list = []
        for a_val_one_tail in alphas_table_chi2_list: 
            cv_upper = stats.chi2.ppf(1 - a_val_one_tail, df_chi2)
            cv_lower = stats.chi2.ppf(a_val_one_tail, df_chi2)
            table_data_chi2_list.append({
                "α (Tail Area)": f"{a_val_one_tail:.4f}",
                "χ²_crit (Lower)": f"{cv_lower:.3f}",
                "χ²_crit (Upper)": f"{cv_upper:.3f}"
            })
        df_table_chi2 = pd.DataFrame(table_data_chi2_list)
        
        def highlight_alpha_row_chi2(row):
            highlight = False
            current_alpha_in_table = float(row["α (Tail Area)"])
            if tail_chi2 == "Two-tailed (e.g. for variance)":
                if abs(current_alpha_in_table - (alpha_chi2 / 2)) < 1e-5 :
                    highlight = True
            else: 
                if abs(current_alpha_in_table - alpha_chi2) < 1e-5:
                    highlight = True
            return ['background-color: yellow'] * len(row) if highlight else [''] * len(row)

        st.markdown(df_table_chi2.style.apply(highlight_alpha_row_chi2, axis=1).to_html(), unsafe_allow_html=True)
        st.caption(f"Table shows critical χ²-values for df={df_chi2}. Highlighted for your selected α={alpha_chi2:.4f} ({tail_chi2}).")
        st.markdown("""
        **Cumulative Table Note:**
        * 'α (Tail Area)' is the area in one tail. For an upper critical value, it's the area to the right. For a lower critical value, it's the area to the left.
        * For **One-tailed (right) tests**, find your α in 'α (Tail Area)' and use 'χ²_crit (Upper)'.
        * For **Two-tailed tests**, find α/2 in 'α (Tail Area)'. Use 'χ²_crit (Lower)' and 'χ²_crit (Upper)'.
        """)

    with col2:
        st.subheader("P-value Calculation Explanation")
        st.markdown(f"""
        The p-value is the probability of a χ²-statistic as extreme as, or more extreme than, {test_stat_chi2:.3f}.
        * **One-tailed (right)**: `P(χ² ≥ {test_stat_chi2:.3f})` (i.e., `stats.chi2.sf(test_stat_chi2, df_chi2)`)
        * **Two-tailed**: `2 * min(P(χ² ≤ {test_stat_chi2:.3f}), P(χ² ≥ {test_stat_chi2:.3f}))` (i.e., `2 * min(stats.chi2.cdf(test_stat_chi2, df_chi2), stats.chi2.sf(test_stat_chi2, df_chi2))`)
        """)

        st.subheader("Summary")
        p_val_chi2_one_right = stats.chi2.sf(test_stat_chi2, df_chi2)
        cdf_chi2 = stats.chi2.cdf(test_stat_chi2, df_chi2)
        sf_chi2 = stats.chi2.sf(test_stat_chi2, df_chi2)
        p_val_chi2_two = 2 * min(cdf_chi2, sf_chi2)
        p_val_chi2_two = min(p_val_chi2_two, 1.0)

        crit_val_chi2_display = "N/A"
        p_val_for_crit_val_chi2_display = alpha_chi2

        if tail_chi2 == "One-tailed (right)":
            crit_val_chi2_display = f"{crit_val_chi2_upper:.3f}" if crit_val_chi2_upper is not None and not np.isnan(crit_val_chi2_upper) else "N/A"
            p_val_calc_chi2 = p_val_chi2_one_right
            decision_crit_chi2 = test_stat_chi2 > crit_val_chi2_upper if crit_val_chi2_upper is not None and not np.isnan(crit_val_chi2_upper) else False
            comparison_crit_str_chi2 = f"{test_stat_chi2:.3f} > {crit_val_chi2_upper:.3f}" if decision_crit_chi2 else f"{test_stat_chi2:.3f} ≤ {crit_val_chi2_upper:.3f}"
        else: 
            crit_val_chi2_display = f"Lower: {crit_val_chi2_lower:.3f}, Upper: {crit_val_chi2_upper:.3f}" if crit_val_chi2_lower is not None and not np.isnan(crit_val_chi2_lower) and crit_val_chi2_upper is not None and not np.isnan(crit_val_chi2_upper) else "N/A"
            p_val_calc_chi2 = p_val_chi2_two
            decision_crit_chi2 = (test_stat_chi2 > crit_val_chi2_upper if crit_val_chi2_upper is not None and not np.isnan(crit_val_chi2_upper) else False) or \
                                 (test_stat_chi2 < crit_val_chi2_lower if crit_val_chi2_lower is not None and not np.isnan(crit_val_chi2_lower) else False)
            comparison_crit_str_chi2 = f"{test_stat_chi2:.3f} > {crit_val_chi2_upper:.3f} or {test_stat_chi2:.3f} < {crit_val_chi2_lower:.3f}" if decision_crit_chi2 else f"{crit_val_chi2_lower:.3f} ≤ {test_stat_chi2:.3f} ≤ {crit_val_chi2_upper:.3f}"

        decision_p_alpha_chi2 = p_val_calc_chi2 < alpha_chi2
        
        st.markdown(f"""
        1.  **Critical Value(s) ({tail_chi2})**: {crit_val_chi2_display}
            * *Associated p-value (α or α/2 per tail)*: {p_val_for_crit_val_chi2_display:.4f}
        2.  **Calculated Test Statistic**: {test_stat_chi2:.3f}
            * *Calculated p-value*: {p_val_calc_chi2:.4f} ({apa_p_value(p_val_calc_chi2)})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_chi2 else 'not rejected'}**.
            * *Reason*: χ²(calc) {comparison_crit_str_chi2} relative to χ²(crit).
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_chi2 else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_calc_chi2)} is {'less than' if decision_p_alpha_chi2 else 'not less than'} α ({alpha_chi2:.4f}).
        5.  **APA 7 Style Report**:
            χ²({df_chi2}) = {test_stat_chi2:.2f}, {apa_p_value(p_val_calc_chi2)}. The null hypothesis was {'rejected' if decision_p_alpha_chi2 else 'not rejected'} at α = {alpha_chi2:.2f}.
        """)

# --- Tab 5: Mann-Whitney U Test (No changes from previous version, assumed to be working) ---
def tab_mann_whitney_u():
    st.header("Mann-Whitney U Test (Normal Approximation)")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_mw = st.number_input("Alpha (α)", 0.0001, 0.5, 0.05, 0.0001, format="%.4f", key="alpha_mw")
        n1_mw = st.number_input("Sample Size Group 1 (n1)", 1, 1000, 10, 1, key="n1_mw") 
        n2_mw = st.number_input("Sample Size Group 2 (n2)", 1, 1000, 12, 1, key="n2_mw") 
        tail_mw = st.radio("Tail Selection", ("Two-tailed", "One-tailed (right)", "One-tailed (left)"), key="tail_mw")
        u_stat_mw = st.number_input("Calculated U-statistic", value=float(n1_mw*n2_mw/2), format="%.1f", min_value=0.0, max_value=float(n1_mw*n2_mw), key="u_stat_mw")
        st.caption("Note: Normal approximation is best for n1, n2 > ~10. U is typically the smaller of U1 or U2.")

        mu_u = (n1_mw * n2_mw) / 2
        sigma_u_sq = (n1_mw * n2_mw * (n1_mw + n2_mw + 1)) / 12
        sigma_u = np.sqrt(sigma_u_sq) if sigma_u_sq > 0 else 0
        
        z_calc_mw = 0.0
        if sigma_u > 0:
            if u_stat_mw < mu_u:
                z_calc_mw = (u_stat_mw + 0.5 - mu_u) / sigma_u
            elif u_stat_mw > mu_u:
                z_calc_mw = (u_stat_mw - 0.5 - mu_u) / sigma_u
            else: 
                z_calc_mw = 0.0 
        else: 
            z_calc_mw = 0.0
            if n1_mw > 0 and n2_mw > 0: st.warning("Standard deviation (σ_U) is zero. Check sample sizes. z_calc set to 0.")


        st.markdown(f"**Normal Approximation Parameters:** μ<sub>U</sub> = {mu_u:.2f}, σ<sub>U</sub> = {sigma_u:.2f}")
        st.markdown(f"**Calculated z-statistic (from U, with continuity correction):** {z_calc_mw:.3f}")

        st.subheader("Standard Normal Distribution Plot (for z_calc)")
        fig_mw, ax_mw = plt.subplots(figsize=(8,5))
        
        plot_min_z_mw = min(stats.norm.ppf(0.0001), z_calc_mw - 2, -4.0)
        plot_max_z_mw = max(stats.norm.ppf(0.9999), z_calc_mw + 2, 4.0)
        if abs(z_calc_mw) > 4:
            plot_min_z_mw = min(plot_min_z_mw, z_calc_mw -1)
            plot_max_z_mw = max(plot_max_z_mw, z_calc_mw +1)

        x_norm_mw = np.linspace(plot_min_z_mw, plot_max_z_mw, 500)
        y_norm_mw = stats.norm.pdf(x_norm_mw)
        ax_mw.plot(x_norm_mw, y_norm_mw, 'b-', lw=2, label='Standard Normal Distribution')

        crit_z_upper_mw, crit_z_lower_mw = None, None
        if tail_mw == "Two-tailed":
            crit_z_upper_mw = stats.norm.ppf(1 - alpha_mw / 2)
            crit_z_lower_mw = stats.norm.ppf(alpha_mw / 2)
            if crit_z_upper_mw is not None and not np.isnan(crit_z_upper_mw):
                x_fill_upper_mw = np.linspace(crit_z_upper_mw, plot_max_z_mw, 100)
                ax_mw.fill_between(x_fill_upper_mw, stats.norm.pdf(x_fill_upper_mw), color='red', alpha=0.5, label=f'α/2 = {alpha_mw/2:.4f}')
                ax_mw.axvline(crit_z_upper_mw, color='red', linestyle='--', lw=1)
            if crit_z_lower_mw is not None and not np.isnan(crit_z_lower_mw):
                x_fill_lower_mw = np.linspace(plot_min_z_mw, crit_z_lower_mw, 100)
                ax_mw.fill_between(x_fill_lower_mw, stats.norm.pdf(x_fill_lower_mw), color='red', alpha=0.5)
                ax_mw.axvline(crit_z_lower_mw, color='red', linestyle='--', lw=1)
        elif tail_mw == "One-tailed (right)":
            crit_z_upper_mw = stats.norm.ppf(1 - alpha_mw)
            if crit_z_upper_mw is not None and not np.isnan(crit_z_upper_mw):
                x_fill_upper_mw = np.linspace(crit_z_upper_mw, plot_max_z_mw, 100)
                ax_mw.fill_between(x_fill_upper_mw, stats.norm.pdf(x_fill_upper_mw), color='red', alpha=0.5, label=f'α = {alpha_mw:.4f}')
                ax_mw.axvline(crit_z_upper_mw, color='red', linestyle='--', lw=1)
        else: 
            crit_z_lower_mw = stats.norm.ppf(alpha_mw)
            if crit_z_lower_mw is not None and not np.isnan(crit_z_lower_mw):
                x_fill_lower_mw = np.linspace(plot_min_z_mw, crit_z_lower_mw, 100)
                ax_mw.fill_between(x_fill_lower_mw, stats.norm.pdf(x_fill_lower_mw), color='red', alpha=0.5, label=f'α = {alpha_mw:.4f}')
                ax_mw.axvline(crit_z_lower_mw, color='red', linestyle='--', lw=1)

        ax_mw.axvline(z_calc_mw, color='green', linestyle='-', lw=2, label=f'z_calc = {z_calc_mw:.3f}')
        ax_mw.set_title('Normal Approx. for Mann-Whitney U: Critical z Region(s)')
        ax_mw.set_xlabel('z-value')
        ax_mw.set_ylabel('Probability Density')
        ax_mw.legend()
        ax_mw.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_mw)

        st.subheader("Critical z-Value Table Snippet (for U test's z_calc)")
        alphas_table_z_mw_list = [0.10, 0.05, 0.025, 0.01, 0.005, alpha_mw, alpha_mw/2 if tail_mw == "Two-tailed" else alpha_mw]
        alphas_table_z_mw_list = sorted(list(set(a for a in alphas_table_z_mw_list if 0.00005 < a < 0.50005)))
        
        table_data_z_mw_list = []
        for a_val_one_tail in alphas_table_z_mw_list:
            a_val_two_tail = a_val_one_tail * 2
            cv_upper = stats.norm.ppf(1 - a_val_one_tail)
            cv_lower = stats.norm.ppf(a_val_one_tail)
            table_data_z_mw_list.append({
                "α (One-Tail)": f"{a_val_one_tail:.4f}",
                "α (Two-Tail)": f"{a_val_two_tail:.4f}" if a_val_two_tail <= 0.51 else "-",
                "z_crit (Lower)": f"{cv_lower:.3f}",
                "z_crit (Upper)": f"{cv_upper:.3f}"
            })
        df_table_z_mw = pd.DataFrame(table_data_z_mw_list)
        
        def highlight_alpha_row_z_mw(row): 
            highlight = False
            if tail_mw == "Two-tailed":
                if abs(float(row["α (One-Tail)"]) - (alpha_mw / 2)) < 1e-5 :
                    highlight = True
            else: 
                if abs(float(row["α (One-Tail)"]) - alpha_mw) < 1e-5:
                    highlight = True
            return ['background-color: yellow'] * len(row) if highlight else [''] * len(row)
        st.markdown(df_table_z_mw.style.apply(highlight_alpha_row_z_mw, axis=1).to_html(), unsafe_allow_html=True)
        st.caption(f"Highlighted row for α={alpha_mw:.4f} ({tail_mw}). Compare calculated z from U to these critical z-values.")
        st.markdown("""
        **Cumulative Table Note:**
        * The Mann-Whitney U test (with normal approximation) converts U to a z-statistic. This table shows critical z-values.
        * Small sample exact U tables are different and not shown here.
        """)

    with col2:
        st.subheader("P-value Calculation Explanation")
        st.markdown(f"""
        The U statistic ({u_stat_mw:.1f}) is converted to a z-statistic ({z_calc_mw:.3f}) using μ<sub>U</sub>={mu_u:.2f}, σ<sub>U</sub>={sigma_u:.2f} (with continuity correction). The p-value is from the standard normal distribution based on this z_calc_mw.
        * **Two-tailed**: `2 * P(Z ≥ |{z_calc_mw:.3f}|)`
        * **One-tailed (right)**: `P(Z ≥ {z_calc_mw:.3f})` 
        * **One-tailed (left)**: `P(Z ≤ {z_calc_mw:.3f})` 
        The interpretation of "right" and "left" for U depends on definition of U and H1. This explorer assumes z_calc_mw directionality matches tail selection for p-value.
        """)

        st.subheader("Summary")
        p_val_mw_one_right = stats.norm.sf(z_calc_mw)
        p_val_mw_one_left = stats.norm.cdf(z_calc_mw)
        p_val_mw_two = 2 * stats.norm.sf(abs(z_calc_mw))
        p_val_mw_two = min(p_val_mw_two, 1.0)

        crit_val_z_display_mw = "N/A"
        p_val_for_crit_val_mw_display = alpha_mw

        if tail_mw == "Two-tailed":
            crit_val_z_display_mw = f"±{crit_z_upper_mw:.3f}" if crit_z_upper_mw is not None and not np.isnan(crit_z_upper_mw) else "N/A"
            p_val_calc_mw = p_val_mw_two
            decision_crit_mw = abs(z_calc_mw) > crit_z_upper_mw if crit_z_upper_mw is not None and not np.isnan(crit_z_upper_mw) else False
            comparison_crit_str_mw = f"|z_calc| ({abs(z_calc_mw):.3f}) > {crit_z_upper_mw:.3f}" if decision_crit_mw else f"|z_calc| ({abs(z_calc_mw):.3f}) ≤ {crit_z_upper_mw:.3f}"
        elif tail_mw == "One-tailed (right)":
            crit_val_z_display_mw = f"{crit_z_upper_mw:.3f}" if crit_z_upper_mw is not None and not np.isnan(crit_z_upper_mw) else "N/A"
            p_val_calc_mw = p_val_mw_one_right
            decision_crit_mw = z_calc_mw > crit_z_upper_mw if crit_z_upper_mw is not None and not np.isnan(crit_z_upper_mw) else False
            comparison_crit_str_mw = f"z_calc ({z_calc_mw:.3f}) > {crit_z_upper_mw:.3f}" if decision_crit_mw else f"z_calc ({z_calc_mw:.3f}) ≤ {crit_z_upper_mw:.3f}"
        else: 
            crit_val_z_display_mw = f"{crit_z_lower_mw:.3f}" if crit_z_lower_mw is not None and not np.isnan(crit_z_lower_mw) else "N/A"
            p_val_calc_mw = p_val_mw_one_left
            decision_crit_mw = z_calc_mw < crit_z_lower_mw if crit_z_lower_mw is not None and not np.isnan(crit_z_lower_mw) else False
            comparison_crit_str_mw = f"z_calc ({z_calc_mw:.3f}) < {crit_z_lower_mw:.3f}" if decision_crit_mw else f"z_calc ({z_calc_mw:.3f}) ≥ {crit_z_lower_mw:.3f}"

        decision_p_alpha_mw = p_val_calc_mw < alpha_mw
        
        st.markdown(f"""
        1.  **Critical z-value ({tail_mw}) for U test's z_calc**: {crit_val_z_display_mw}
            * *Associated p-value (α or α/2 per tail)*: {p_val_for_crit_val_mw_display:.4f}
        2.  **Calculated U-statistic**: {u_stat_mw:.1f}
            * *Converted z-statistic (z_calc)*: {z_calc_mw:.3f}
            * *Calculated p-value (from z_calc)*: {p_val_calc_mw:.4f} ({apa_p_value(p_val_calc_mw)})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_mw else 'not rejected'}**.
            * *Reason*: {comparison_crit_str_mw} relative to z(crit).
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_mw else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_calc_mw)} is {'less than' if decision_p_alpha_mw else 'not less than'} α ({alpha_mw:.4f}).
        5.  **APA 7 Style Report (based on z-approximation)**:
            A Mann-Whitney U test indicated that the outcome for group 1 (n<sub>1</sub>={n1_mw}) was {'' if decision_p_alpha_mw else 'not '}statistically significantly different from group 2 (n<sub>2</sub>={n2_mw}), *U* = {u_stat_mw:.1f}, *z* = {z_calc_mw:.2f}, {apa_p_value(p_val_calc_mw)}. The null hypothesis was {'rejected' if decision_p_alpha_mw else 'not rejected'} at α = {alpha_mw:.2f}.
        """)

# --- Tab 6: Wilcoxon Signed-Rank T Test (No changes from previous version, assumed to be working) ---
def tab_wilcoxon_t():
    st.header("Wilcoxon Signed-Rank T Test (Normal Approximation)")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_w = st.number_input("Alpha (α)", 0.0001, 0.5, 0.05, 0.0001, format="%.4f", key="alpha_w")
        n_w = st.number_input("Sample Size (n, non-zero differences)", 1, 1000, 15, 1, key="n_w") 
        tail_w = st.radio("Tail Selection", ("Two-tailed", "One-tailed (right)", "One-tailed (left)"), key="tail_w")
        t_stat_w = st.number_input("Calculated T-statistic (sum of ranks)", value=float(n_w*(n_w+1)/4 / 2 if n_w >0 else 0), format="%.1f", min_value=0.0, max_value=float(n_w*(n_w+1)/2 if n_w > 0 else 0), key="t_stat_w")
        st.caption("Note: Normal approximation best for n > ~15-20. T is usually the smaller of T+ or T- for two-tailed tests.")

        mu_t_w = n_w * (n_w + 1) / 4
        sigma_t_w_sq = n_w * (n_w + 1) * (2 * n_w + 1) / 24
        sigma_t_w = np.sqrt(sigma_t_w_sq) if sigma_t_w_sq > 0 else 0
        
        z_calc_w = 0.0
        if sigma_t_w > 0:
            if t_stat_w < mu_t_w: 
                z_calc_w = (t_stat_w + 0.5 - mu_t_w) / sigma_t_w
            elif t_stat_w > mu_t_w: 
                z_calc_w = (t_stat_w - 0.5 - mu_t_w) / sigma_t_w
            else: 
                z_calc_w = 0.0
        else:
            z_calc_w = 0.0
            if n_w > 0: st.warning("Standard deviation (σ_T) is zero. Check sample size n. z_calc set to 0.")


        st.markdown(f"**Normal Approximation Parameters:** μ<sub>T</sub> = {mu_t_w:.2f}, σ<sub>T</sub> = {sigma_t_w:.2f}")
        st.markdown(f"**Calculated z-statistic (from T, with continuity correction):** {z_calc_w:.3f}")

        st.subheader("Standard Normal Distribution Plot (for z_calc)")
        fig_w, ax_w = plt.subplots(figsize=(8,5))
        
        plot_min_z_w = min(stats.norm.ppf(0.0001), z_calc_w - 2, -4.0)
        plot_max_z_w = max(stats.norm.ppf(0.9999), z_calc_w + 2, 4.0)
        if abs(z_calc_w) > 4:
            plot_min_z_w = min(plot_min_z_w, z_calc_w -1)
            plot_max_z_w = max(plot_max_z_w, z_calc_w +1)
        
        x_norm_w = np.linspace(plot_min_z_w, plot_max_z_w, 500)
        y_norm_w = stats.norm.pdf(x_norm_w)
        ax_w.plot(x_norm_w, y_norm_w, 'b-', lw=2, label='Standard Normal Distribution')

        crit_z_upper_w, crit_z_lower_w = None, None
        if tail_w == "Two-tailed":
            crit_z_upper_w = stats.norm.ppf(1 - alpha_w / 2)
            crit_z_lower_w = stats.norm.ppf(alpha_w / 2)
            if crit_z_upper_w is not None and not np.isnan(crit_z_upper_w):
                x_fill_upper_w = np.linspace(crit_z_upper_w, plot_max_z_w, 100)
                ax_w.fill_between(x_fill_upper_w, stats.norm.pdf(x_fill_upper_w), color='red', alpha=0.5, label=f'α/2 = {alpha_w/2:.4f}')
                ax_w.axvline(crit_z_upper_w, color='red', linestyle='--', lw=1)
            if crit_z_lower_w is not None and not np.isnan(crit_z_lower_w):
                x_fill_lower_w = np.linspace(plot_min_z_w, crit_z_lower_w, 100)
                ax_w.fill_between(x_fill_lower_w, stats.norm.pdf(x_fill_lower_w), color='red', alpha=0.5)
                ax_w.axvline(crit_z_lower_w, color='red', linestyle='--', lw=1)
        elif tail_w == "One-tailed (right)": 
            crit_z_upper_w = stats.norm.ppf(1 - alpha_w)
            if crit_z_upper_w is not None and not np.isnan(crit_z_upper_w):
                x_fill_upper_w = np.linspace(crit_z_upper_w, plot_max_z_w, 100)
                ax_w.fill_between(x_fill_upper_w, stats.norm.pdf(x_fill_upper_w), color='red', alpha=0.5, label=f'α = {alpha_w:.4f}')
                ax_w.axvline(crit_z_upper_w, color='red', linestyle='--', lw=1)
        else: 
            crit_z_lower_w = stats.norm.ppf(alpha_w)
            if crit_z_lower_w is not None and not np.isnan(crit_z_lower_w):
                x_fill_lower_w = np.linspace(plot_min_z_w, crit_z_lower_w, 100)
                ax_w.fill_between(x_fill_lower_w, stats.norm.pdf(x_fill_lower_w), color='red', alpha=0.5, label=f'α = {alpha_w:.4f}')
                ax_w.axvline(crit_z_lower_w, color='red', linestyle='--', lw=1)

        ax_w.axvline(z_calc_w, color='green', linestyle='-', lw=2, label=f'z_calc = {z_calc_w:.3f}')
        ax_w.set_title('Normal Approx. for Wilcoxon T: Critical z Region(s)')
        ax_w.set_xlabel('z-value')
        ax_w.set_ylabel('Probability Density')
        ax_w.legend()
        ax_w.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_w)

        st.subheader("Critical z-Value Table Snippet (for T test's z_calc)")
        alphas_table_z_w_list = [0.10, 0.05, 0.025, 0.01, 0.005, alpha_w, alpha_w/2 if tail_w == "Two-tailed" else alpha_w]
        alphas_table_z_w_list = sorted(list(set(a for a in alphas_table_z_w_list if 0.00005 < a < 0.50005)))
        
        table_data_z_w_list = []
        for a_val_one_tail in alphas_table_z_w_list:
            a_val_two_tail = a_val_one_tail * 2
            cv_upper = stats.norm.ppf(1 - a_val_one_tail)
            cv_lower = stats.norm.ppf(a_val_one_tail)
            table_data_z_w_list.append({
                "α (One-Tail)": f"{a_val_one_tail:.4f}",
                "α (Two-Tail)": f"{a_val_two_tail:.4f}" if a_val_two_tail <= 0.51 else "-",
                "z_crit (Lower)": f"{cv_lower:.3f}",
                "z_crit (Upper)": f"{cv_upper:.3f}"
            })
        df_table_z_w = pd.DataFrame(table_data_z_w_list)
        
        def highlight_alpha_row_z_w(row): 
            highlight = False
            if tail_w == "Two-tailed":
                if abs(float(row["α (One-Tail)"]) - (alpha_w / 2)) < 1e-5 :
                    highlight = True
            else: 
                if abs(float(row["α (One-Tail)"]) - alpha_w) < 1e-5:
                    highlight = True
            return ['background-color: yellow'] * len(row) if highlight else [''] * len(row)
        st.markdown(df_table_z_w.style.apply(highlight_alpha_row_z_w, axis=1).to_html(), unsafe_allow_html=True)
        st.caption(f"Highlighted row for α={alpha_w:.4f} ({tail_w}). Compare calculated z from T to these critical z-values.")
        st.markdown("""
        **Cumulative Table Note:**
        * Wilcoxon T is converted to z. This table shows critical z-values. Small sample exact T tables are different.
        * For two-tailed, T is usually smaller of T+ or T-. For one-tailed, T is T+ or T- based on H1.
        """)

    with col2:
        st.subheader("P-value Calculation Explanation")
        st.markdown(f"""
        The T statistic ({t_stat_w:.1f}) is converted to z ({z_calc_w:.3f}) using μ<sub>T</sub>={mu_t_w:.2f}, σ<sub>T</sub>={sigma_t_w:.2f} (with continuity correction). P-value from normal distribution.
        * **Two-tailed**: `2 * P(Z ≥ |{z_calc_w:.3f}|)` 
        * **One-tailed (right)**: `P(Z ≥ {z_calc_w:.3f})` 
        * **One-tailed (left)**: `P(Z ≤ {z_calc_w:.3f})` 
        This explorer uses the calculated z_calc_w's sign to determine the p-value for one-tailed tests.
        """)

        st.subheader("Summary")
        
        p_val_w_one_right_tail_on_z = stats.norm.sf(z_calc_w) 
        p_val_w_one_left_tail_on_z = stats.norm.cdf(z_calc_w)  
        p_val_w_two_tail_on_z = 2 * stats.norm.sf(abs(z_calc_w)) 
        p_val_w_two_tail_on_z = min(p_val_w_two_tail_on_z, 1.0)


        crit_val_z_display_w = "N/A"
        p_val_for_crit_val_w_display = alpha_w

        if tail_w == "Two-tailed":
            crit_val_z_display_w = f"±{crit_z_upper_w:.3f}" if crit_z_upper_w is not None and not np.isnan(crit_z_upper_w) else "N/A"
            p_val_calc_w = p_val_w_two_tail_on_z
            decision_crit_w = abs(z_calc_w) > crit_z_upper_w if crit_z_upper_w is not None and not np.isnan(crit_z_upper_w) else False
            comparison_crit_str_w = f"|z_calc| ({abs(z_calc_w):.3f}) > {crit_z_upper_w:.3f}" if decision_crit_w else f"|z_calc| ({abs(z_calc_w):.3f}) ≤ {crit_z_upper_w:.3f}"
        elif tail_w == "One-tailed (right)": 
            crit_val_z_display_w = f"{crit_z_upper_w:.3f}" if crit_z_upper_w is not None and not np.isnan(crit_z_upper_w) else "N/A"
            p_val_calc_w = p_val_w_one_right_tail_on_z 
            decision_crit_w = z_calc_w > crit_z_upper_w if crit_z_upper_w is not None and not np.isnan(crit_z_upper_w) else False
            comparison_crit_str_w = f"z_calc ({z_calc_w:.3f}) > {crit_z_upper_w:.3f}" if decision_crit_w else f"z_calc ({z_calc_w:.3f}) ≤ {crit_z_upper_w:.3f}"
        else: 
            crit_val_z_display_w = f"{crit_z_lower_w:.3f}" if crit_z_lower_w is not None and not np.isnan(crit_z_lower_w) else "N/A"
            p_val_calc_w = p_val_w_one_left_tail_on_z 
            decision_crit_w = z_calc_w < crit_z_lower_w if crit_z_lower_w is not None and not np.isnan(crit_z_lower_w) else False
            comparison_crit_str_w = f"z_calc ({z_calc_w:.3f}) < {crit_z_lower_w:.3f}" if decision_crit_w else f"z_calc ({z_calc_w:.3f}) ≥ {crit_z_lower_w:.3f}"

        decision_p_alpha_w = p_val_calc_w < alpha_w
        
        st.markdown(f"""
        1.  **Critical z-value ({tail_w}) for T test's z_calc**: {crit_val_z_display_w}
            * *Associated p-value (α or α/2 per tail)*: {p_val_for_crit_val_w_display:.4f}
        2.  **Calculated T-statistic**: {t_stat_w:.1f}
            * *Converted z-statistic (z_calc)*: {z_calc_w:.3f}
            * *Calculated p-value (from z_calc)*: {p_val_calc_w:.4f} ({apa_p_value(p_val_calc_w)})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_w else 'not rejected'}**.
            * *Reason*: {comparison_crit_str_w} relative to z(crit).
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_w else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_calc_w)} is {'less than' if decision_p_alpha_w else 'not less than'} α ({alpha_w:.4f}).
        5.  **APA 7 Style Report (based on z-approximation)**:
            A Wilcoxon signed-rank test indicated that the median difference was {'' if decision_p_alpha_w else 'not '}statistically significant, *T* = {t_stat_w:.1f}, *z* = {z_calc_w:.2f}, {apa_p_value(p_val_calc_w)}. The null hypothesis was {'rejected' if decision_p_alpha_w else 'not rejected'} at α = {alpha_w:.2f} (n={n_w}).
        """)

# --- Tab 7: Binomial Test (No changes from previous version, assumed to be working) ---
def tab_binomial_test():
    st.header("Binomial Test Explorer")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_b = st.number_input("Alpha (α)", 0.0001, 0.5, 0.05, 0.0001, format="%.4f", key="alpha_b")
        n_b = st.number_input("Number of Trials (n)", 1, 1000, 20, 1, key="n_b")
        p_null_b = st.number_input("Null Hypothesis Probability (p₀)", 0.00, 1.00, 0.5, 0.01, format="%.2f", key="p_null_b")
        k_success_b = st.number_input("Number of Successes (k)", 0, n_b, int(n_b * p_null_b), 1, key="k_success_b")
        
        tail_options_b = {
            f"Two-tailed (p ≠ {p_null_b})": "two-sided",
            f"One-tailed (right, p > {p_null_b})": "greater",
            f"One-tailed (left, p < {p_null_b})": "less"
        }
        tail_b_display = st.radio("Tail Selection (Alternative Hypothesis)", 
                                  list(tail_options_b.keys()), 
                                  key="tail_b_display")
        tail_b_scipy = tail_options_b[tail_b_display]


        st.subheader("Binomial Distribution Plot")
        fig_b, ax_b = plt.subplots(figsize=(8,5))
        x_b = np.arange(0, n_b + 1)
        y_b_pmf = stats.binom.pmf(x_b, n_b, p_null_b)
        ax_b.bar(x_b, y_b_pmf, label=f'Binomial PMF (n={n_b}, p₀={p_null_b})', alpha=0.7, color='skyblue')
        
        ax_b.scatter([k_success_b], [stats.binom.pmf(k_success_b, n_b, p_null_b)], color='green', s=100, zorder=5, label=f'Observed k = {k_success_b}')
        
        if tail_b_scipy == "two-sided":
            pass 
        elif tail_b_scipy == "greater": 
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
        ax_b.legend()
        ax_b.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_b)

        st.subheader("Probability Table Snippet")
        k_start_table = max(0, k_success_b - 3)
        k_end_table = min(n_b, k_success_b + 3)
        k_range_table = np.arange(k_start_table, k_end_table + 1)
        
        if len(k_range_table) > 0:
            table_data_b = {
                "k": k_range_table,
                "P(X=k)": [f"{stats.binom.pmf(k_val, n_b, p_null_b):.4f}" for k_val in k_range_table],
                "P(X≤k) (CDF)": [f"{stats.binom.cdf(k_val, n_b, p_null_b):.4f}" for k_val in k_range_table],
                "P(X≥k) (1-CDF(k-1))": [f"{stats.binom.sf(k_val -1, n_b, p_null_b):.4f}" for k_val in k_range_table] 
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

        st.markdown("""
        **Cumulative Table Note:**
        * `P(X=k)`: Probability of exactly k successes.
        * `P(X≤k)` (CDF): Cumulative probability of k or fewer successes.
        * `P(X≥k)` (1-CDF(k-1) or SF(k-1)): Cumulative probability of k or more successes.
        These are used for p-values. For two-tailed, it's more complex (sum of equally or less likely outcomes).
        """)

    with col2:
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
                if stats.binom.pmf(i, n_b, p_null_b) <= p_observed_k + 1e-9: 
                    p_val_calc_b += stats.binom.pmf(i, n_b, p_null_b)
            p_val_calc_b = min(p_val_calc_b, 1.0) 
        elif tail_b_scipy == "greater":
            p_val_calc_b = p_val_b_one_right
        else: 
            p_val_calc_b = p_val_b_one_left
        
        crit_val_b_display = "Exact critical k values are complex for binomial."
        if tail_b_scipy == "greater":
            k_crit_b_approx = stats.binom.isf(alpha_b, n_b, p_null_b) 
            crit_val_b_display = f"Reject if k ≥ {int(k_crit_b_approx)+1 if stats.binom.sf(int(k_crit_b_approx),n_b,p_null_b) > alpha_b else int(k_crit_b_approx)} (approx.)"
        elif tail_b_scipy == "less":
            k_crit_b_approx = stats.binom.ppf(alpha_b, n_b, p_null_b) 
            crit_val_b_display = f"Reject if k ≤ {int(k_crit_b_approx)} (approx.)"
        else: 
            k_crit_low_b_approx = stats.binom.ppf(alpha_b/2, n_b, p_null_b)
            k_crit_high_b_approx = stats.binom.isf(alpha_b/2, n_b, p_null_b)
            crit_val_b_display = f"Reject if k ≤ {int(k_crit_low_b_approx)} or k ≥ {int(k_crit_high_b_approx)+1 if stats.binom.sf(int(k_crit_high_b_approx),n_b,p_null_b) > alpha_b/2 else int(k_crit_high_b_approx)} (approx.)"

        p_val_for_crit_val_b_display = alpha_b
        decision_p_alpha_b = p_val_calc_b < alpha_b

        decision_crit_b = decision_p_alpha_b 
        comparison_crit_str_b = f"Observed k={k_success_b} falls in rejection region" if decision_crit_b else f"Observed k={k_success_b} does not fall in rejection region"
        
        st.markdown(f"""
        1.  **Approximate Critical Region ({tail_b_display})**: {crit_val_b_display}
            * *Associated significance level (α)*: {p_val_for_crit_val_b_display:.4f}
        2.  **Observed Number of Successes (k)**: {k_success_b}
            * *Calculated p-value*: {p_val_calc_b:.4f} ({apa_p_value(p_val_calc_b)})
        3.  **Decision (Critical Region Method - based on p-value for discrete)**: H₀ is **{'rejected' if decision_crit_b else 'not rejected'}**.
            * *Reason*: {comparison_crit_str_b}. (For discrete tests, p-value method is more direct).
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_b else 'not rejected'}**.
            * *Reason*: {apa_p_value(p_val_calc_b)} is {'less than' if decision_p_alpha_b else 'not less than'} α ({alpha_b:.4f}).
        5.  **APA 7 Style Report**:
            A binomial test was performed to assess whether the proportion of successes (k={k_success_b}, n={n_b}) was different from the null hypothesis proportion of p₀={p_null_b}. The result was {'' if decision_p_alpha_b else 'not '}statistically significant, {apa_p_value(p_val_calc_b)}. The null hypothesis was {'rejected' if decision_p_alpha_b else 'not rejected'} at α = {alpha_b:.2f}.
        """)

# --- Tab 8: Tukey HSD (No changes from previous version, assumed to be working) ---
def tab_tukey_hsd():
    st.header("Tukey HSD (Honestly Significant Difference) Explorer")
    col1, col2 = st.columns([2, 1.5])
    
    with col1:
        st.subheader("Inputs")
        alpha_tukey = st.number_input("Alpha (α)", 0.0001, 0.5, 0.05, 0.0001, format="%.4f", key="alpha_tukey_hsd")
        k_tukey = st.number_input("Number of Groups (k)", 2, 50, 3, 1, key="k_tukey_hsd")
        df_error_tukey = st.number_input("Degrees of Freedom for Error (within-group df)", 1, 2000, 20, 1, key="df_error_tukey_hsd")
        test_stat_tukey_q = st.number_input("Calculated q-statistic (for a pair)", value=1.0, format="%.3f", min_value=0.0, key="test_stat_tukey_q")

        st.subheader("Studentized Range q Distribution (Conceptual Plot)")
        st.markdown("The Tukey HSD test uses the studentized range q distribution. The plot below is illustrative.")
        
        q_crit_tukey = None
        source_q_crit = "Not calculated"
        q_crit_tukey_str_for_message = "N/A" 

        try:
            from statsmodels.stats.libqsturng import qsturng
            q_crit_tukey = qsturng(1 - alpha_tukey, k_tukey, df_error_tukey)
            source_q_crit = "statsmodels.stats.libqsturng"
            if isinstance(q_crit_tukey, (int, float)) and not np.isnan(q_crit_tukey):
                q_crit_tukey_str_for_message = f"{q_crit_tukey:.3f}"
            elif q_crit_tukey is not None: 
                q_crit_tukey_str_for_message = str(q_crit_tukey)
            tukey_message = f"Critical q ({alpha_tukey:.3f}, k={k_tukey}, df={df_error_tukey}) = {q_crit_tukey_str_for_message} (from {source_q_crit})"
        except ImportError:
            initial_message = "Statsmodels `qsturng` not available. Using CSV fallback."
            st.warning(initial_message)
            q_crit_tukey = get_tukey_q_from_csv(df_error_tukey, k_tukey, alpha_tukey)
            source_q_crit = "CSV Fallback"
            if isinstance(q_crit_tukey, (int, float)) and not np.isnan(q_crit_tukey):
                q_crit_tukey_str_for_message = f"{q_crit_tukey:.3f}"
                tukey_message = f"{initial_message}\nCritical q from CSV = {q_crit_tukey_str_for_message}. This may be an approximation."
            elif q_crit_tukey is not None:
                q_crit_tukey_str_for_message = str(q_crit_tukey)
                tukey_message = f"{initial_message}\nCritical q from CSV = {q_crit_tukey_str_for_message} (non-numeric). This may be an approximation."
            else: 
                q_crit_tukey_str_for_message = "N/A"
                tukey_message = f"{initial_message}\nCSV fallback failed to find a value."
                st.error("Could not determine critical q value from CSV.")
        except Exception as e: 
            initial_message = f"Error using `statsmodels.stats.libqsturng`: {e}. Using CSV fallback."
            st.warning(initial_message)
            q_crit_tukey = get_tukey_q_from_csv(df_error_tukey, k_tukey, alpha_tukey)
            source_q_crit = "CSV Fallback (after error)"
            if isinstance(q_crit_tukey, (int, float)) and not np.isnan(q_crit_tukey):
                q_crit_tukey_str_for_message = f"{q_crit_tukey:.3f}"
                tukey_message = f"{initial_message}\nCritical q from CSV = {q_crit_tukey_str_for_message}. This may be an approximation."
            elif q_crit_tukey is not None:
                q_crit_tukey_str_for_message = str(q_crit_tukey)
                tukey_message = f"{initial_message}\nCritical q from CSV = {q_crit_tukey_str_for_message} (non-numeric). This may be an approximation."
            else: 
                q_crit_tukey_str_for_message = "N/A"
                tukey_message = f"{initial_message}\nCSV fallback failed to find a value."
                st.error("Could not determine critical q value from CSV.")
        
        fig_tukey, ax_tukey = plt.subplots(figsize=(8,5))
        if q_crit_tukey is not None and isinstance(q_crit_tukey, (int, float)) and not np.isnan(q_crit_tukey):
            plot_max_q = max(q_crit_tukey * 1.5, test_stat_tukey_q * 1.5, 5.0)
            if test_stat_tukey_q > q_crit_tukey * 1.2: plot_max_q = test_stat_tukey_q * 1.2
            
            x_placeholder = np.linspace(0.01, plot_max_q, 200)
            shape_param = float(k_tukey) 
            scale_param = max(q_crit_tukey, test_stat_tukey_q, 1.0) / (shape_param * 2 if shape_param > 0 else 2) 
            if shape_param <=0 : shape_param = 2.0 
            if scale_param <=0 : scale_param = 0.5 

            try:
                y_placeholder = stats.gamma.pdf(x_placeholder, a=shape_param, scale=scale_param)
                ax_tukey.plot(x_placeholder, y_placeholder, 'b-', lw=2, label=f'Conceptual q-like distribution shape')
                ax_tukey.axvline(q_crit_tukey, color='red', linestyle='--', lw=1.5, label=f'Critical q = {q_crit_tukey:.3f}')
                
                x_fill_crit = np.linspace(q_crit_tukey, plot_max_q, 100)
                y_fill_crit = stats.gamma.pdf(x_fill_crit, a=shape_param, scale=scale_param)
                ax_tukey.fill_between(x_fill_crit, y_fill_crit, color='red', alpha=0.5)
            except Exception as plot_e:
                 ax_tukey.text(0.5, 0.6, f"Plotting error: {plot_e}", ha='center', va='center', color='red')

            ax_tukey.axvline(test_stat_tukey_q, color='green', linestyle='-', lw=2, label=f'Test q = {test_stat_tukey_q:.3f}')
            ax_tukey.set_title(f'Conceptual q-Distribution (α={alpha_tukey:.3f})')
            ax_tukey.set_xlabel('q-value')
            ax_tukey.set_ylabel('Density (Illustrative)')
        else:
            ax_tukey.text(0.5, 0.5, "Critical q not available or non-numeric for plotting.", ha='center', va='center')
            ax_tukey.set_title('Plot Unavailable')

        ax_tukey.legend()
        ax_tukey.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_tukey)
        st.info(tukey_message)


        st.subheader("Critical Value Table Snippet")
        alphas_table_tukey_list = [0.10, 0.05, 0.01]
        if alpha_tukey not in alphas_table_tukey_list:
            alphas_table_tukey_list.append(alpha_tukey)
        alphas_table_tukey_list = sorted(list(set(a for a in alphas_table_tukey_list if 0.00005 < a < 0.50005)))

        table_data_tukey_list = []
        for a_val in alphas_table_tukey_list:
            q_c_table = None
            source_table = ""
            q_c_table_str = "N/A"
            try:
                from statsmodels.stats.libqsturng import qsturng
                q_c_table = qsturng(1 - a_val, k_tukey, df_error_tukey)
                source_table = "statsmodels"
                if isinstance(q_c_table, (int,float)) and not np.isnan(q_c_table): q_c_table_str = f"{q_c_table:.3f}"
                elif q_c_table is not None: q_c_table_str = str(q_c_table)

            except: 
                q_c_table = get_tukey_q_from_csv(df_error_tukey, k_tukey, a_val)
                source_table = "CSV Fallback" if q_c_table is not None else "Not Found"
                if isinstance(q_c_table, (int,float)) and not np.isnan(q_c_table): q_c_table_str = f"{q_c_table:.3f}"
                elif q_c_table is not None: q_c_table_str = str(q_c_table)

            
            table_data_tukey_list.append({
                "Alpha (α)": f"{a_val:.4f}",
                "Critical q": q_c_table_str,
                "Source": source_table
            })
        df_table_tukey = pd.DataFrame(table_data_tukey_list)

        def highlight_alpha_row_tukey(row):
            if abs(float(row["Alpha (α)"]) - alpha_tukey) < 1e-5:
                return ['background-color: yellow'] * len(row)
            return [''] * len(row)
        st.markdown(df_table_tukey.style.apply(highlight_alpha_row_tukey, axis=1).to_html(), unsafe_allow_html=True)
        st.caption(f"Table for k={k_tukey}, df_error={df_error_tukey}. Highlighted for your selected α.")

        st.markdown("""
        **Cumulative Table Note:** Tukey's HSD uses the studentized range (q) statistic. If your calculated q for a pair of means exceeds the critical q, that pair is significantly different.
        """)

    with col2:
        st.subheader("P-value Calculation Explanation")
        p_val_calc_tukey = None
        p_val_source = "Not calculated"
        try:
            from statsmodels.stats.libqsturng import psturng 
            if test_stat_tukey_q is not None and k_tukey > 0 and df_error_tukey > 0:
                 p_val_calc_tukey = 1 - psturng(test_stat_tukey_q, float(k_tukey), float(df_error_tukey))
                 p_val_calc_tukey = max(0, min(p_val_calc_tukey, 1.0)) 
                 p_val_source = "statsmodels.stats.libqsturng.psturng"
        except ImportError:
            p_val_source = "statsmodels not available for p-value calc."
        except Exception as e_psturng:
            p_val_source = f"Error during p-value calc with psturng: {e_psturng}"


        st.markdown(f"""
        The p-value for a specific calculated q-statistic ({test_stat_tukey_q:.3f}) from Tukey's HSD is the probability P(Q ≥ {test_stat_tukey_q:.3f}).
        * This can be calculated using `1 - psturng(q_calc, k, df_error)` from `statsmodels.stats.libqsturng`.
        * If `statsmodels` is not available or fails, this explorer cannot calculate the p-value for your specific q-statistic.
        * The "associated p-value" for the *critical q* value is, by definition, alpha (α).
        Current p-value calculation status: {p_val_source}
        """)

        st.subheader("Summary")
        p_val_for_crit_val_tukey_display = alpha_tukey
        
        q_crit_tukey_display_val = "N/A"
        if isinstance(q_crit_tukey, (int, float)) and not np.isnan(q_crit_tukey):
            q_crit_tukey_display_val = f"{q_crit_tukey:.3f}"
        elif q_crit_tukey is not None:
            q_crit_tukey_display_val = str(q_crit_tukey)


        apa_p_val_calc_tukey_display = "p N/A"
        p_val_calc_tukey_formatted_str = "N/A (requires statsmodels.stats.libqsturng.psturng)"

        if p_val_calc_tukey is not None and not np.isnan(p_val_calc_tukey):
            apa_p_val_calc_tukey_display = apa_p_value(p_val_calc_tukey)
            p_val_calc_tukey_formatted_str = f"{p_val_calc_tukey:.4f} ({apa_p_val_calc_tukey_display})"
        
        decision_crit_tukey = False
        comparison_crit_str_tukey = "Critical q not available or non-numeric"
        if q_crit_tukey is not None and isinstance(q_crit_tukey, (int, float)) and not np.isnan(q_crit_tukey):
            decision_crit_tukey = test_stat_tukey_q > q_crit_tukey
            comparison_crit_str_tukey = f"q(calc) ({test_stat_tukey_q:.3f}) > q(crit) ({q_crit_tukey:.3f})" if decision_crit_tukey else f"q(calc) ({test_stat_tukey_q:.3f}) ≤ q(crit) ({q_crit_tukey:.3f})"
        
        decision_p_alpha_tukey = False
        reason_p_alpha_tukey_display = "p-value for q_calc not computed or q_crit not available/numeric."
        if p_val_calc_tukey is not None and not np.isnan(p_val_calc_tukey):
            decision_p_alpha_tukey = p_val_calc_tukey < alpha_tukey
            reason_p_alpha_tukey_display = f"Because {apa_p_val_calc_tukey_display} is {'less than' if decision_p_alpha_tukey else 'not less than'} α ({alpha_tukey:.4f})."
        elif q_crit_tukey is not None and isinstance(q_crit_tukey, (int, float)) and not np.isnan(q_crit_tukey): 
             decision_p_alpha_tukey = decision_crit_tukey 
             reason_p_alpha_tukey_display = f"Decision based on critical value method as p-value for q_calc was not computed."


        st.markdown(f"""
        1.  **Critical q-value**: {q_crit_tukey_display_val} (Source: {source_q_crit})
            * *Associated significance level (α)*: {p_val_for_crit_val_tukey_display:.4f}
        2.  **Calculated q-statistic (for one pair)**: {test_stat_tukey_q:.3f}
            * *Calculated p-value*: {p_val_calc_tukey_formatted_str} 
        3.  **Decision (Critical Value Method)**: For this pair, H₀ (no difference) is **{'rejected' if decision_crit_tukey else 'not rejected'}**.
            * *Reason*: {comparison_crit_str_tukey}.
        4.  **Decision (p-value Method)**: H₀ (no difference) is **{'rejected' if decision_p_alpha_tukey else 'not rejected'}**.
            * *Reason*: {reason_p_alpha_tukey_display}
        5.  **APA 7 Style Report (for this specific comparison)**:
            A Tukey HSD comparison for one pair yielded *q*({k_tukey}, {df_error_tukey}) = {test_stat_tukey_q:.2f}. {('The associated p-value was ' + apa_p_val_calc_tukey_display + '. ') if p_val_calc_tukey is not None and not np.isnan(p_val_calc_tukey) else ''}The null hypothesis of no difference for this pair was {'rejected' if decision_p_alpha_tukey else 'not rejected'} at α = {alpha_tukey:.2f}.
        """)
        st.caption("Note: A full Tukey HSD analysis involves all pairwise comparisons. This tab focuses on interpreting a single calculated q-statistic.")

# --- Tab 9: Kruskal-Wallis Test ---
def tab_kruskal_wallis():
    st.header("Kruskal-Wallis H Test (Chi-square Approximation)")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_kw = st.number_input("Alpha (α)", 0.0001, 0.5, 0.05, 0.0001, format="%.4f", key="alpha_kw")
        k_groups_kw = st.number_input("Number of Groups (k)", 2, 50, 3, 1, key="k_groups_kw") 
        df_kw = k_groups_kw - 1
        st.markdown(f"Degrees of Freedom (df) = k - 1 = {df_kw}")
        test_stat_h_kw = st.number_input("Calculated H-statistic", value=float(df_kw if df_kw > 0 else 0.5), format="%.3f", min_value=0.0, key="test_stat_h_kw")
        st.caption("Note: Chi-square approximation is best if each group size ≥ 5.")

        st.subheader("Chi-square Distribution Plot (Approximation for H)")
        fig_kw, ax_kw = plt.subplots(figsize=(8,5))
        crit_val_chi2_kw = None 
        
        if df_kw > 0:
            plot_min_chi2_kw = 0.001
            plot_max_chi2_kw = max(stats.chi2.ppf(0.999, df_kw), test_stat_h_kw * 1.5, 10.0)
            if test_stat_h_kw > stats.chi2.ppf(0.999, df_kw) * 1.2:
                plot_max_chi2_kw = test_stat_h_kw * 1.2

            x_chi2_kw = np.linspace(plot_min_chi2_kw, plot_max_chi2_kw, 500)
            y_chi2_kw = stats.chi2.pdf(x_chi2_kw, df_kw)
            ax_kw.plot(x_chi2_kw, y_chi2_kw, 'b-', lw=2, label=f'χ²-distribution (df={df_kw})')

            crit_val_chi2_kw = stats.chi2.ppf(1 - alpha_kw, df_kw) 
            if isinstance(crit_val_chi2_kw, (int, float)) and not np.isnan(crit_val_chi2_kw):
                x_fill_upper_kw = np.linspace(crit_val_chi2_kw, plot_max_chi2_kw, 100)
                ax_kw.fill_between(x_fill_upper_kw, stats.chi2.pdf(x_fill_upper_kw, df_kw), color='red', alpha=0.5, label=f'α = {alpha_kw:.4f}')
                ax_kw.axvline(crit_val_chi2_kw, color='red', linestyle='--', lw=1, label=f'χ²_crit = {crit_val_chi2_kw:.3f}')
            
            ax_kw.axvline(test_stat_h_kw, color='green', linestyle='-', lw=2, label=f'H_calc = {test_stat_h_kw:.3f}')
            ax_kw.set_title(f'χ²-Approximation for Kruskal-Wallis H (df={df_kw})')
            ax_kw.set_xlabel('χ²-value / H-statistic')
            ax_kw.set_ylabel('Probability Density')
        else:
            ax_kw.text(0.5, 0.5, "df must be > 0 (k > 1 for meaningful test)", ha='center', va='center')
            ax_kw.set_title('Plot Unavailable (df=0)')
            
        ax_kw.legend()
        ax_kw.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_kw)

        st.subheader("Critical χ² Value Table Snippet (for H test)")
        if df_kw > 0:
            alphas_table_chi2_kw_list = [0.10, 0.05, 0.025, 0.01, 0.005, alpha_kw]
            alphas_table_chi2_kw_list = sorted(list(set(a for a in alphas_table_chi2_kw_list if 0.00005 < a < 0.50005)))
            
            table_data_chi2_kw_list = []
            for a_val_one_tail in alphas_table_chi2_kw_list:
                cv_upper = stats.chi2.ppf(1 - a_val_one_tail, df_kw)
                cv_upper_str = f"{cv_upper:.3f}" if isinstance(cv_upper, (int,float)) and not np.isnan(cv_upper) else "N/A"
                table_data_chi2_kw_list.append({
                    "α (Right Tail)": f"{a_val_one_tail:.4f}",
                    "Critical χ² (Upper)": cv_upper_str
                })
            df_table_chi2_kw = pd.DataFrame(table_data_chi2_kw_list)
            
            def highlight_alpha_row_chi2_kw(row):
                if abs(float(row["α (Right Tail)"]) - alpha_kw) < 1e-5:
                    return ['background-color: yellow'] * len(row)
                return [''] * len(row)
            st.markdown(df_table_chi2_kw.style.apply(highlight_alpha_row_chi2_kw, axis=1).to_html(), unsafe_allow_html=True)
            st.caption(f"Table shows critical χ²-values for df={df_kw}. Highlighted for your α.")
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
        p_val_for_crit_val_kw_display = alpha_kw
        p_val_calc_kw_num = float('nan') 
        decision_crit_kw = False
        comparison_crit_str_kw = "Test not valid (df must be > 0)"
        decision_p_alpha_kw = False
        apa_H_stat = f"*H*({df_kw if df_kw > 0 else 'N/A'}) = {test_stat_h_kw:.2f}"
        
        summary_crit_val_chi2_kw_display_str = "N/A (df=0)"
        if df_kw > 0:
            p_val_calc_kw_num = stats.chi2.sf(test_stat_h_kw, df_kw) 
            
            # crit_val_chi2_kw was calculated in plot section if df_kw > 0
            if isinstance(crit_val_chi2_kw, (int, float)) and not np.isnan(crit_val_chi2_kw):
                summary_crit_val_chi2_kw_display_str = f"{crit_val_chi2_kw:.3f}"
                decision_crit_kw = test_stat_h_kw > crit_val_chi2_kw
                comparison_crit_str_kw = f"H({test_stat_h_kw:.3f}) > χ²_crit({crit_val_chi2_kw:.3f})" if decision_crit_kw else f"H({test_stat_h_kw:.3f}) ≤ χ²_crit({crit_val_chi2_kw:.3f})"
            else:
                 summary_crit_val_chi2_kw_display_str = "N/A (calc error)"
                 comparison_crit_str_kw = "Comparison not possible (critical value is N/A or NaN)"
            
            if isinstance(p_val_calc_kw_num, (int, float)) and not np.isnan(p_val_calc_kw_num):
                decision_p_alpha_kw = p_val_calc_kw_num < alpha_kw
            # else decision_p_alpha_kw remains False
        else: 
             apa_H_stat = f"*H* = {test_stat_h_kw:.2f} (df={df_kw}, test invalid)"
        
        p_val_calc_kw_num_display_str = "N/A"
        if isinstance(p_val_calc_kw_num, (int, float)) and not np.isnan(p_val_calc_kw_num):
            try:
                p_val_calc_kw_num_display_str = f"{p_val_calc_kw_num:.4f}"
            except (ValueError, TypeError):
                p_val_calc_kw_num_display_str = "N/A (format err)"
        
        apa_p_val_calc_kw_str = apa_p_value(p_val_calc_kw_num)


        st.markdown(f"""
        1.  **Critical χ²-value (df={df_kw})**: {summary_crit_val_chi2_kw_display_str}
            * *Associated p-value (α)*: {p_val_for_crit_val_kw_display:.4f}
        2.  **Calculated H-statistic**: {test_stat_h_kw:.3f}
            * *Calculated p-value (from χ² approx.)*: {p_val_calc_kw_num_display_str} ({apa_p_val_calc_kw_str})
        3.  **Decision (Critical Value Method)**: H₀ is **{'rejected' if decision_crit_kw else 'not rejected'}**.
            * *Reason*: {comparison_crit_str_kw}.
        4.  **Decision (p-value Method)**: H₀ is **{'rejected' if decision_p_alpha_kw else 'not rejected'}**.
            * *Reason*: {apa_p_val_calc_kw_str} is {'less than' if decision_p_alpha_kw else 'not less than'} α ({alpha_kw:.4f}).
        5.  **APA 7 Style Report**:
            A Kruskal-Wallis H test showed that there was {'' if decision_p_alpha_kw else 'not '}a statistically significant difference in medians between the k={k_groups_kw} groups, {apa_H_stat}, {apa_p_val_calc_kw_str}. The null hypothesis was {'rejected' if decision_p_alpha_kw else 'not rejected'} at α = {alpha_kw:.2f}.
        """)

# --- Tab 10: Friedman Test ---
def tab_friedman_test():
    st.header("Friedman Test (Chi-square Approximation)")
    col1, col2 = st.columns([2, 1.5])

    with col1:
        st.subheader("Inputs")
        alpha_fr = st.number_input("Alpha (α)", 0.0001, 0.5, 0.05, 0.0001, format="%.4f", key="alpha_fr")
        k_conditions_fr = st.number_input("Number of Conditions/Treatments (k)", 2, 50, 3, 1, key="k_conditions_fr") 
        n_blocks_fr = st.number_input("Number of Blocks/Subjects (n)", 2, 200, 10, 1, key="n_blocks_fr") 
        
        df_fr = k_conditions_fr - 1
        st.markdown(f"Degrees of Freedom (df) = k - 1 = {df_fr}")
        test_stat_q_fr = st.number_input("Calculated Friedman Q-statistic (or χ²_r)", value=float(df_fr if df_fr > 0 else 0.5), format="%.3f", min_value=0.0, key="test_stat_q_fr")

        if n_blocks_fr <= 10 or k_conditions_fr <= 3 : 
            st.warning("Small n or k. Friedman’s χ² approximation may be less reliable. Exact methods preferred if available.")

        st.subheader("Chi-square Distribution Plot (Approximation for Q)")
        fig_fr, ax_fr = plt.subplots(figsize=(8,5))
        crit_val_chi2_fr = None 
        
        if df_fr > 0:
            plot_min_chi2_fr = 0.001
            plot_max_chi2_fr = max(stats.chi2.ppf(0.999, df_fr), test_stat_q_fr * 1.5, 10.0)
            if test_stat_q_fr > stats.chi2.ppf(0.999, df_fr) * 1.2:
                plot_max_chi2_fr = test_stat_q_fr * 1.2

            x_chi2_fr = np.linspace(plot_min_chi2_fr, plot_max_chi2_fr, 500)
            y_chi2_fr = stats.chi2.pdf(x_chi2_fr, df_fr)
            ax_fr.plot(x_chi2_fr, y_chi2_fr, 'b-', lw=2, label=f'χ²-distribution (df={df_fr})')

            crit_val_chi2_fr = stats.chi2.ppf(1 - alpha_fr, df_fr) 
            if isinstance(crit_val_chi2_fr, (int,float)) and not np.isnan(crit_val_chi2_fr):
                x_fill_upper_fr = np.linspace(crit_val_chi2_fr, plot_max_chi2_fr, 100)
                ax_fr.fill_between(x_fill_upper_fr, stats.chi2.pdf(x_fill_upper_fr, df_fr), color='red', alpha=0.5, label=f'α = {alpha_fr:.4f}')
                ax_fr.axvline(crit_val_chi2_fr, color='red', linestyle='--', lw=1, label=f'χ²_crit = {crit_val_chi2_fr:.3f}')
            
            ax_fr.axvline(test_stat_q_fr, color='green', linestyle='-', lw=2, label=f'Q_calc = {test_stat_q_fr:.3f}')
            ax_fr.set_title(f'χ²-Approximation for Friedman Q (df={df_fr})')
            ax_fr.set_xlabel('χ²-value / Q-statistic')
            ax_fr.set_ylabel('Probability Density')
        else:
            ax_fr.text(0.5, 0.5, "df must be > 0 (k > 1 for meaningful test)", ha='center', va='center')
            ax_fr.set_title('Plot Unavailable (df=0)')

        ax_fr.legend()
        ax_fr.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_fr)

        st.subheader("Critical χ² Value Table Snippet (for Q test)")
        if df_fr > 0:
            alphas_table_chi2_fr_list = [0.10, 0.05, 0.025, 0.01, 0.005, alpha_fr]
            alphas_table_chi2_fr_list = sorted(list(set(a for a in alphas_table_chi2_fr_list if 0.00005 < a < 0.50005)))
            
            table_data_chi2_fr_list = []
            for a_val_one_tail in alphas_table_chi2_fr_list:
                cv_upper = stats.chi2.ppf(1 - a_val_one_tail, df_fr)
                cv_upper_str = f"{cv_upper:.3f}" if isinstance(cv_upper, (int,float)) and not np.isnan(cv_upper) else "N/A"
                table_data_chi2_fr_list.append({
                    "α (Right Tail)": f"{a_val_one_tail:.4f}",
                    "Critical χ² (Upper)": cv_upper_str
                })
            df_table_chi2_fr = pd.DataFrame(table_data_chi2_fr_list)
            
            def highlight_alpha_row_chi2_fr(row):
                if abs(float(row["α (Right Tail)"]) - alpha_fr) < 1e-5:
                    return ['background-color: yellow'] * len(row)
                return [''] * len(row)
            st.markdown(df_table_chi2_fr.style.apply(highlight_alpha_row_chi2_fr, axis=1).to_html(), unsafe_allow_html=True)
            st.caption(f"Table shows critical χ²-values for df={df_fr}. Highlighted for your α.")
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
        p_val_for_crit_val_fr_display = alpha_fr
        p_val_calc_fr_num = float('nan') 
        decision_crit_fr = False
        comparison_crit_str_fr = "Test not valid (df must be > 0)"
        decision_p_alpha_fr = False
        apa_Q_stat = f"χ²<sub>r</sub>({df_fr if df_fr > 0 else 'N/A'}) = {test_stat_q_fr:.2f}"
        
        summary_crit_val_chi2_fr_display_str = "N/A (df=0)"
        if df_fr > 0:
            p_val_calc_fr_num = stats.chi2.sf(test_stat_q_fr, df_fr) 
            
            # crit_val_chi2_fr was calculated in plot section if df_fr > 0
            if isinstance(crit_val_chi2_fr, (int,float)) and not np.isnan(crit_val_chi2_fr):
                summary_crit_val_chi2_fr_display_str = f"{crit_val_chi2_fr:.3f}"
                decision_crit_fr = test_stat_q_fr > crit_val_chi2_fr
                comparison_crit_str_fr = f"Q({test_stat_q_fr:.3f}) > χ²_crit({crit_val_chi2_fr:.3f})" if decision_crit_fr else f"Q({test_stat_q_fr:.3f}) ≤ χ²_crit({crit_val_chi2_fr:.3f})"
            else: 
                summary_crit_val_chi2_fr_display_str = "N/A (calc error)"
                comparison_crit_str_fr = "Comparison not possible (critical value is N/A or NaN)"

            if isinstance(p_val_calc_fr_num, (int, float)) and not np.isnan(p_val_calc_fr_num):
                decision_p_alpha_fr = p_val_calc_fr_num < alpha_fr
            # else decision_p_alpha_fr remains False
        elif df_fr <=0 :
             apa_Q_stat = f"χ²<sub>r</sub> = {test_stat_q_fr:.2f} (df={df_fr}, test invalid)"

        p_val_calc_fr_num_display_str = "N/A" 
        if isinstance(p_val_calc_fr_num, (int, float)) and not np.isnan(p_val_calc_fr_num):
            try:
                p_val_calc_fr_num_display_str = f"{p_val_calc_fr_num:.4f}"
            except (ValueError, TypeError): 
                p_val_calc_fr_num_display_str = "N/A (format err)"
        
        apa_p_val_calc_fr_str = apa_p_value(p_val_calc_fr_num)

        st.markdown(f"""
        1.  **Critical χ²-value (df={df_fr})**: {summary_crit_val_chi2_fr_display_str}
            * *Associated p-value (α)*: {p_val_for_crit_val_fr_display:.4f}
        2.  **Calculated Q-statistic (χ²_r)**: {test_stat_q_fr:.3f}
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
