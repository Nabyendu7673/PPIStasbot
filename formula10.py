import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from scipy.stats import kruskal, f_oneway, ttest_ind, chi2_contingency, pearsonr, mannwhitneyu
import xgboost as xgb
import base64

# Set Streamlit page configuration
st.set_page_config(page_title="PPIcheck Statsbot", layout="wide")

# Set the main title of the application
st.title("PPIcheck Statsbot")

# Footer moved to top and updated
st.markdown('<p style="text-align: center;">Developed with ❤️ from MEDAI Labs. &copy; 2024. All rights reserved.</p>', unsafe_allow_html=True)

# --- Introduction Section ---
st.markdown(
    """
    <div style="background-color:#e9ecef; padding: 20px; border-radius: 8px; margin-bottom: 30px; box-shadow: 2px 2px 8px #adb5bd;">
        <h3 style="color: #343a40; margin-top: 0;">👋 Welcome to PPIcheck Statsbot</h3>
        <p style="color: #495057;">
        This app helps you generate synthetic patient data based on key factors related to Proton Pump Inhibitor (PPI) use and associated risks.
        </p>
        <p style="color: #495057;">
        You can:
        </p>
        <ul style="color: #495057;">
            <li>Configure dataset parameters from the sidebar</li>
            <li>Generate and view PPIcheck scores</li>
            <li>Run statistical tests and machine learning models on the generated data</li>
            <li>Perform manual statistical analysis on selected variables</li>
        </ul>
        <p style="color: #495057;">
        Perfect for exploring risk patterns, validating scoring systems, or simulating clinical scenarios.
        </p>
        <p style="color: #495057;">
        Get started by adjusting your inputs on the left and clicking the buttons below to begin your analysis!
        </p>
    </div>
    """, unsafe_allow_html=True
)


# --- Synthetic Dataset Generator with Noise ---
# Caches the output of this function to improve performance
@st.cache_data
def generate_synthetic_data(n=2000, noise_level=0.0):
    np.random.seed(42) # For reproducibility
    data = pd.DataFrame({
        'age': np.random.randint(18, 90, size=n),
        'bmi': np.round(np.random.normal(24, 5, size=n), 1),
        'nsaid_class': np.random.choice(['Propionic', 'Acetic', 'Oxicam', 'COX-2'], size=n),
        'nsaid_dose_pct': np.random.randint(0, 101, size=n),
        'antiplatelet': np.random.choice(['None', 'Aspirin', 'Clopidogrel', 'Ticagrelor', 'Dipyridamole', 'Ticlopidine', 'Abciximab'], size=n),
        'antiplatelet_dose': np.random.randint(0, 325, size=n),
        'anticoagulant': np.random.choice(['None', 'UFH_SC', 'UFH_IV', 'LMWH', 'Dalteparin', 'Fondaparinux', 'Argatroban', 'Bivalirudin'], size=n),
        'anticoagulant_dose': np.random.rand(n)*10,
        'indication_score': np.random.randint(0, 9, size=n),
        'ppi_dose': np.random.choice([0, 20, 40, 80], size=n),
        'ppi_route': np.random.choice(['None', 'Oral', 'IV'], size=n)
    })

    # Add noise to numerical columns if noise_level is greater than 0
    if noise_level > 0:
        for col in ['age', 'bmi', 'nsaid_dose_pct', 'antiplatelet_dose', 'anticoagulant_dose', 'indication_score']:
            noise = np.random.normal(0, noise_level * np.std(data[col]), size=n)
            data[col] = np.clip(data[col] + noise, 0, None) # Ensure values are non-negative
            if col != 'bmi': # Keep BMI as float, convert others to int
                data[col] = data[col].astype(int)

    return data

# --- Scoring Logic ---
# Caches the output of this function
@st.cache_data
def compute_scores(df):
    # Define required columns for scoring
    required_raw_cols = ['nsaid_class', 'nsaid_dose_pct', 'antiplatelet', 'antiplatelet_dose',
                     'anticoagulant', 'anticoagulant_dose', 'indication_score', 'ppi_dose', 'ppi_route']
    # Check if all required columns exist in the DataFrame
    if not all(col in df.columns for col in required_raw_cols):
        missing = [col for col in required_raw_cols if col not in df.columns]
        st.error(f"Uploaded data must contain the following columns for scoring: {', '.join(missing)}")
        return None # Return None to indicate failure

    for col in ['nsaid_dose_pct', 'antiplatelet_dose', 'anticoagulant_dose', 'indication_score', 'ppi_dose']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Function to score NSAID usage
    def score_nsaid(row):
        nsaid_class = str(row['nsaid_class']) if pd.notna(row['nsaid_class']) else ''
        base = {'Propionic': 3, 'Acetic': 5, 'Oxicam': 4, 'COX-2': 1}.get(nsaid_class, 0)
        nsaid_dose_pct = row['nsaid_dose_pct']
        if pd.isna(nsaid_dose_pct): return base
        if nsaid_dose_pct <= 25: return base
        elif nsaid_dose_pct <= 50: return base + 1
        elif nsaid_dose_pct <= 75: return base + 2
        else: return base + 3

    # Function to score antiplatelet usage
    def score_antiplatelet(row):
        ap = str(row['antiplatelet']) if pd.notna(row['antiplatelet']) else ''
        d = row['antiplatelet_dose']
        if ap == 'None' or pd.isna(d): return 0
        if ap == 'Aspirin':
            return 2 if d <= 75 else 3 if d <= 150 else 4
        if ap in ['Clopidogrel', 'Ticagrelor', 'Prasugrel']:
            return 2 if d < 75 else 3 if d < 300 else 4
        if ap == 'Dipyridamole': return 1
        if ap == 'Ticlopidine': return 2
        if ap in ['Abciximab', 'Eptifibatide', 'Tirofiban']: return 3
        return 0

    # Function to score anticoagulant usage
    def score_anticoagulant(row):
        ac = str(row['anticoagulant']) if pd.notna(row['anticoagulant']) else ''
        d = row['anticoagulant_dose']
        if ac == 'None' or pd.isna(d): return 0
        if ac == 'UFH_SC': return 2
        if ac == 'UFH_IV': return 3
        if ac == 'LMWH': return 3 if d >= 1 else 2
        if ac == 'Dalteparin': return 3 if d >= 200 else 2
        if ac == 'Fondaparinux': return 2 if d >= 5 else 1
        if ac == 'Argatroban': return 3 if d >= 5 else 2
        if ac == 'Bivalirudin': return 3 if d >= 1.5 else 2
        return 0

    # Function to determine PPI protection score
    def ppi_protection(row):
        ppi_route = str(row['ppi_route']) if pd.notna(row['ppi_route']) else ''
        ppi_dose = row['ppi_dose']
        if ppi_route == 'Oral' and (pd.notna(ppi_dose) and ppi_dose >= 20): return -1
        if ppi_route == 'IV' and (pd.notna(ppi_dose) and ppi_dose >= 40): return -2
        return 0

    # Apply scoring functions to create new columns
    df['nsaid_score'] = df.apply(score_nsaid, axis=1)
    df['antiplatelet_score'] = df.apply(score_antiplatelet, axis=1)
    df['anticoagulant_score'] = df.apply(score_anticoagulant, axis=1)

    # Calculate medication_score by summing component scores, filling NaN with 0
    df['medication_score'] = pd.to_numeric(df['nsaid_score'], errors='coerce').fillna(0) + \
                             pd.to_numeric(df['antiplatelet_score'], errors='coerce').fillna(0) + \
                             pd.to_numeric(df['anticoagulant_score'], errors='coerce').fillna(0)

    df['ppi_score'] = df.apply(ppi_protection, axis=1)

    # Calculate high_risk_flag and triple_flag, then total_score
    if 'medication_score' in df.columns and 'indication_score' in df.columns and 'ppi_score' in df.columns:
        indication_score_numeric = pd.to_numeric(df['indication_score'], errors='coerce').fillna(0)
        medication_score_numeric = pd.to_numeric(df['medication_score'], errors='coerce').fillna(0)
        ppi_score_numeric = pd.to_numeric(df['ppi_score'], errors='coerce').fillna(0)

        df['high_risk_flag'] = ((medication_score_numeric >= 6) | (indication_score_numeric >= 6)).astype(int)
        nsaid_score_numeric = pd.to_numeric(df['nsaid_score'], errors='coerce').fillna(0)
        antiplatelet_score_numeric = pd.to_numeric(df['antiplatelet_score'], errors='coerce').fillna(0)
        anticoagulant_score_numeric = pd.to_numeric(df['anticoagulant_score'], errors='coerce').fillna(0)

        df['triple_flag'] = ((nsaid_score_numeric > 0) & (antiplatelet_score_numeric > 0) & (anticoagulant_score_numeric > 0)).astype(int) * 2

        df['total_score'] = medication_score_numeric + indication_score_numeric + ppi_score_numeric + df['high_risk_flag'] + df['triple_flag']
    else:
        st.error("Failed to compute flags or total score due to missing component scores.")
        return None

    return df

# Caches the output of this function
@st.cache_data
def assign_labels(df, threshold=8):
    if 'total_score' not in df.columns:
        st.error("Cannot assign labels. 'total_score' column is missing. Please ensure data is scored first.")
        return None
    df['label'] = (df['total_score'] >= threshold).astype(int)
    return df

# Function to create a download link for a DataFrame as CSV
def download_dataframe_as_csv(df, filename="download.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

# Function to create a download link for text content
def download_text_file(text, filename="download.txt"):
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/text;base64,{b64}" download="{filename}">Download Text File</a>'
    return href


# --- UI for data generation or upload ---
st.sidebar.subheader("Data Source")
data_source = st.sidebar.radio("Choose data source:", ("Generate Synthetic Data", "Upload CSV File"))

df = None # Initialize df outside the if blocks

if data_source == "Generate Synthetic Data":
    st.sidebar.subheader("Synthetic Data Settings")
    n_samples = st.sidebar.slider("Number of Samples", 100, 5000, 2000, 100)
    noise_level = st.sidebar.slider("Noise Level (0 = clean)", 0.0, 5.0, 0.1, 0.1)

    if st.sidebar.button("Generate Dataset"):
        df = generate_synthetic_data(n=n_samples, noise_level=noise_level)
        df = compute_scores(df)
        if df is not None: # Only assign labels if scoring was successful
            df = assign_labels(df)
            st.session_state.df = df
            st.success(f"Generated {n_samples} synthetic samples with noise level {noise_level}")
            st.dataframe(df.head())
            # Add download button for the generated data
            st.markdown(download_dataframe_as_csv(st.session_state.df, "synthetic_data.csv"), unsafe_allow_html=True)
        else:
             st.error("Failed to generate and score data due to missing columns.")


elif data_source == "Upload CSV File":
    st.sidebar.subheader("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_df = df # Store the original uploaded dataframe
            st.session_state.df = df.copy() # Work with a copy in session_state.df
            st.success("CSV file uploaded successfully!")
            st.dataframe(df.head())

            # Optionally, compute scores and labels for the uploaded data
            st.sidebar.markdown("---") # Add a separator
            st.sidebar.subheader("Process Uploaded Data")
            # Use st.cache_data for the processed uploaded data if the original uploaded_file hasn-t changed
            @st.cache_data
            def process_uploaded_data(uploaded_df):
                 scored_df = compute_scores(uploaded_df.copy()) # Use a copy of the original upload
                 if scored_df is not None:
                     labeled_df = assign_labels(scored_df.copy()) # Use a copy
                     return labeled_df # Return the processed dataframe
                 return None # Return None if scoring failed

            if st.sidebar.button("Score and Label Uploaded Data"):
                 if 'uploaded_df' in st.session_state and st.session_state.uploaded_df is not None:
                     processed_df = process_uploaded_data(st.session_state.uploaded_df) # Call the cached function
                     if processed_df is not None:
                         st.session_state.df = processed_df
                         st.success("Data scored and labeled successfully!")
                         st.dataframe(st.session_state.df.head())
                         st.markdown(download_dataframe_as_csv(st.session_state.df, "scored_uploaded_data.csv"), unsafe_allow_html=True)
                     else:
                         # Error message is already displayed by compute_scores or assign_labels
                         pass
                 else:
                     st.warning("Please upload a CSV file first.")


        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.session_state.df = None # Clear session state on error
            st.session_state.uploaded_df = None # Clear uploaded_df on error


# --- Check if data is available for analysis ---
# Use st.session_state.df for analysis readiness checks
if 'df' not in st.session_state or st.session_state.df is None or st.session_state.df.empty:
    st.warning("Please generate or upload data first to run analyses.")
    data_available = False
else:
    data_available = True
    df = st.session_state.df # Use the dataframe from session state

    # Check if 'label' and required feature columns exist for ML
    required_ml_cols = ['age', 'bmi', 'nsaid_score', 'antiplatelet_score', 'anticoagulant_score',
                        'indication_score', 'ppi_score', 'high_risk_flag', 'triple_flag', 'label']
    ml_ready_classification = all(col in df.columns for col in required_ml_cols) and df['label'].nunique() > 1

    # Check if required columns exist for statistical tests
    required_stat_cols = ['indication_score', 'ppi_route', 'nsaid_score', 'ppi_dose', 'bmi', 'label',
                          'high_risk_flag', 'age', 'antiplatelet', 'anticoagulant', 'medication_score', 'total_score']
    stat_ready = all(col in df.columns for col in required_stat_cols)

    if data_available and not stat_ready:
        missing_stat_cols = [col for col in required_stat_cols if col not in df.columns]
        st.warning(f"Missing columns for statistical tests: {', '.join(missing_stat_cols)}. Please ensure your data is scored and labeled.")

    # Check if required columns exist for Classification ML
    if data_available and not ml_ready_classification:
         missing_ml_cols_clf = [col for col in required_ml_cols if col not in df.columns]
         warnings = []
         if missing_ml_cols_clf:
              warnings.append(f"Missing columns for Classification ML: {', '.join(missing_ml_cols_clf)}.")
         if ml_ready_classification and df['label'].nunique() < 2:
              warnings.append("Classification ML requires the 'label' column to have at least two unique values.")

         if warnings:
             st.warning("ML Assessment readiness issues: " + " ".join(warnings))


# --- Automated Statistical Tests and Table Display ---
st.sidebar.markdown("---") # Add a separator
st.sidebar.subheader("Automated Statistical Analysis")
if st.sidebar.button("Run All Predefined Statistical Tests", disabled=not (data_available and stat_ready)):
    if not (data_available and stat_ready):
         pass # Warning is shown above
    else:
        st.markdown(
            """
            <h2 style="color: #007bff; text-align: center; margin-bottom: 20px;">Automated Statistical Test Results and Interpretations</h2>
            """, unsafe_allow_html=True
        )

        test_results = []

        # Helper to determine conclusion based on p-value
        def get_conclusion(p_value):
            if pd.isna(p_value):
                return "Insufficient data"
            return "Statistically Significant (p < 0.05)" if p_value < 0.05 else "Not Statistically Significant (p >= 0.05)"

        # 1. One-Way ANOVA: Indication Score across PPI Routes
        if 'ppi_route' in df.columns and 'indication_score' in df.columns:
            # Ensure groups have enough data after dropping NaNs
            anova_groups = [g['indication_score'].dropna().values for name, g in df.groupby("ppi_route") if len(g['indication_score'].dropna()) > 1]
            if len(anova_groups) > 1 and all(len(g) > 0 for g in anova_groups):
                try:
                    f_statistic, p_value_anova = f_oneway(*anova_groups)
                    test_results.append({
                        "Test": "One-Way ANOVA",
                        "Variables": "Indication Score by PPI Route",
                        "Statistic Type": "F-statistic",
                        "Statistic": round(f_statistic, 3) if not pd.isna(f_statistic) else np.nan,
                        "P-value": round(p_value_anova, 3) if not pd.isna(p_value_anova) else np.nan,
                        "Conclusion": get_conclusion(p_value_anova)
                    })
                except ValueError:
                     test_results.append({
                        "Test": "One-Way ANOVA",
                        "Variables": "Indication Score by PPI Route",
                        "Statistic Type": "F-statistic",
                        "Statistic": np.nan,
                        "P-value": np.nan,
                        "Conclusion": "Insufficient data (Issue with groups or data after dropping NaNs)"
                    })
            else:
                 test_results.append({
                    "Test": "One-Way ANOVA",
                    "Variables": "Indication Score by PPI Route",
                    "Statistic Type": "F-statistic",
                    "Statistic": np.nan,
                    "P-value": np.nan,
                    "Conclusion": "Insufficient data or variability in PPI Route (need at least two groups with >1 data point)"
                })
        else:
             test_results.append({
                "Test": "One-Way ANOVA",
                "Variables": "Indication Score by PPI Route",
                "Statistic Type": "F-statistic",
                "Statistic": np.nan,
                "P-value": np.nan,
                "Conclusion": "Missing 'ppi_route' or 'indication_score' column"
            })


        # 2. Kruskal-Wallis Test: NSAID Score across PPI Doses
        if 'ppi_dose' in df.columns and 'nsaid_score' in df.columns:
            kruskal_groups = [g['nsaid_score'].dropna().values for name, g in df.groupby("ppi_dose") if len(g['nsaid_score'].dropna()) > 0]
            if len(kruskal_groups) > 1 and all(len(g) > 0 for g in kruskal_groups):
                try:
                    h_statistic, p_value_kruskal = kruskal(*kruskal_groups)
                    test_results.append({
                        "Test": "Kruskal-Wallis Test",
                        "Variables": "NSAID Score by PPI Dose",
                        "Statistic Type": "H-statistic",
                        "Statistic": round(h_statistic, 3) if not pd.isna(h_statistic) else np.nan,
                        "P-value": round(p_value_kruskal, 3) if not pd.isna(p_value_kruskal) else np.nan,
                        "Conclusion": get_conclusion(p_value_kruskal)
                    })
                except ValueError:
                     test_results.append({
                        "Test": "Kruskal-Wallis Test",
                        "Variables": "NSAID Score by PPI Dose",
                        "Statistic Type": "H-statistic",
                        "Statistic": np.nan,
                        "P-value": np.nan,
                        "Conclusion": "Insufficient data (Issue with groups or data after dropping NaNs)"
                    })
            else:
                 test_results.append({
                    "Test": "Kruskal-Wallis Test",
                    "Variables": "NSAID Score by PPI Dose",
                    "Statistic Type": "H-statistic",
                    "Statistic": np.nan,
                    "P-value": np.nan,
                    "Conclusion": "Insufficient data or variability in PPI Dose (need at least two groups with >0 data point)"
                })
        else:
             test_results.append({
                "Test": "Kruskal-Wallis Test",
                "Variables": "NSAID Score by PPI Dose",
                "Statistic Type": "H-statistic",
                "Statistic": np.nan,
                "P-value": np.nan,
                "Conclusion": "Missing 'ppi_dose' or 'nsaid_score' column"
            })


        # 3. Independent t-test: BMI between Label 0 and Label 1
        if 'label' in df.columns and 'bmi' in df.columns and df['label'].nunique() > 1:
            group0 = df[df['label'] == 0]['bmi'].dropna()
            group1 = df[df['label'] == 1]['bmi'].dropna()
            if len(group0) > 1 and len(group1) > 1: # Need at least 2 data points in each group for t-test
                try:
                    t_statistic, p_value_ttest = ttest_ind(group0, group1)
                    test_results.append({
                        "Test": "Independent t-Test",
                        "Variables": "BMI by Label (0 vs 1)",
                        "Statistic Type": "t-statistic",
                        "Statistic": round(t_statistic, 3) if not pd.isna(t_statistic) else np.nan,
                        "P-value": round(p_value_ttest, 3) if not pd.isna(p_value_ttest) else np.nan,
                        "Conclusion": get_conclusion(p_value_ttest)
                    })
                except ValueError:
                     test_results.append({
                        "Test": "Independent t-Test",
                        "Variables": "BMI by Label (0 vs 1)",
                        "Statistic Type": "t-statistic",
                        "Statistic": np.nan,
                        "P-value": np.nan,
                        "Conclusion": "Insufficient data (Issue with groups or data after dropping NaNs)"
                    })
            else:
                test_results.append({
                    "Test": "Independent t-Test",
                    "Variables": "BMI by Label (0 vs 1)",
                    "Statistic Type": "t-statistic",
                    "Statistic": np.nan,
                    "P-value": np.nan,
                    "Conclusion": "Insufficient data (Both labels not present, BMI missing, or not enough data in groups after dropping NaNs)"
                })
        else:
            test_results.append({
                "Test": "Independent t-Test",
                "Variables": "BMI by Label (0 vs 1)",
                "Statistic Type": "t-statistic",
                "Statistic": np.nan,
                "P-value": np.nan,
                "Conclusion": "Missing 'label' or 'bmi' column, or 'label' has only one unique value"
            })

        # 4. Chi-Square Test: High Risk Flag vs Label
        st.markdown("### Detailed Chi-Square Test: High Risk Flag vs Label")
        if 'high_risk_flag' in df.columns and 'label' in df.columns:
            # Drop rows where either high_risk_flag or label is NaN
            df_chi2_hr_label = df[['high_risk_flag', 'label']].dropna()
            if not df_chi2_hr_label.empty:
                contingency_hr_label = pd.crosstab(df_chi2_hr_label['high_risk_flag'], df_chi2_hr_label['label'])

                st.markdown(r"""
                    **Hypotheses:**
                    * Null Hypothesis ($H_0$): There is no association between High Risk Flag and the final Label.
                    * Alternate Hypothesis ($H_1$): There is an association between High Risk Flag and the final Label.
                    """)

                st.markdown("**Observed Contingency Table:**")
                st.dataframe(contingency_hr_label)

                if contingency_hr_label.shape[0] > 1 and contingency_hr_label.shape[1] > 1 and contingency_hr_label.values.sum() >= 5:
                    try:
                        stat_chi2_hr_label, p_value_chi2_hr_label, dof_chi2_hr_label, expected_chi2_hr_label = chi2_contingency(contingency_hr_label, correction=False)

                        st.markdown("**Expected Contingency Table:**")
                        st.dataframe(pd.DataFrame(expected_chi2_hr_label, index=contingency_hr_label.index, columns=contingency_hr_label.columns))

                        st.markdown(r"""
                            **Chi-Square Statistic Calculation:**
                            The Chi-Square statistic is calculated as the sum of the squared differences between the observed ($O$) and expected ($E$) frequencies, divided by the expected frequencies:
                            $$\chi^2 = \sum \frac{(O - E)^2}{E}$$
                            Degrees of Freedom (DOF) = (Number of Rows - 1) * (Number of Columns - 1) = (%i - 1) * (%i - 1) = %i
                            """ % (contingency_hr_label.shape[0], contingency_hr_label.shape[1], dof_chi2_hr_label))

                        # Check for Yates' correction applicability and apply if needed
                        yates_applied = False
                        if contingency_hr_label.shape == (2, 2) and (expected_chi2_hr_label < 5).any():
                             st.markdown(r"""
                                **Yates' Correction for Continuity:**
                                Yates' correction is applied to the Chi-Square test for 2x2 contingency tables when the expected frequencies in any cell are less than 5. It reduces the difference between observed and expected frequencies by 0.5 before squaring, to better approximate the continuous Chi-Square distribution. The formula becomes:
                                $$\chi^2 = \sum \frac{(|O - E| - 0.5)^2}{E}$$
                                We will now calculate the Chi-Square statistic with Yates' correction.
                                """)
                             stat_chi2_hr_label_yates, p_value_chi2_hr_label_yates, _, _ = chi2_contingency(contingency_hr_label, correction=True)
                             stat_to_report = round(stat_chi2_hr_label_yates, 3)
                             p_value_to_report = round(p_value_chi2_hr_label_yates, 3)
                             yates_applied = True
                             st.write(f"Chi-squared statistic (with Yates' correction): {stat_to_report}")
                             st.write(f"P-value (with Yates' correction): {p_value_to_report}")

                        else:
                             stat_to_report = round(stat_chi2_hr_label, 3)
                             p_value_to_report = round(p_value_chi2_hr_label, 3)
                             st.write(f"Chi-squared statistic: {stat_to_report}")
                             st.write(f"P-value: {p_value_to_report}")


                        test_results.append({
                            "Test": "Chi-Square Test",
                            "Variables": "High Risk Flag vs Label",
                            "Statistic Type": f"Chi2-statistic ({'with Yates' if yates_applied else 'without Yates'})",
                            "Statistic": stat_to_report,
                            "P-value": p_value_to_report,
                            "Conclusion": get_conclusion(p_value_to_report)
                        })

                    except ValueError:
                         test_results.append({
                            "Test": "Chi-Square Test",
                            "Variables": "High Risk Flag vs Label",
                            "Statistic Type": "Chi2-statistic",
                            "Statistic": np.nan,
                            "P-value": np.nan,
                            "Conclusion": "Insufficient data (Issue with groups or data after dropping NaNs)"
                        })
                else:
                     test_results.append({
                        "Test": "Chi-Square Test",
                        "Variables": "High Risk Flag vs Label",
                        "Statistic Type": "Chi2-statistic",
                        "Statistic": np.nan,
                        "P-value": np.nan,
                        "Conclusion": "Insufficient data or variability (need >1 row/col and sum >= 5 after dropping NaNs)"
                    })
            else:
                test_results.append({
                    "Test": "Chi-Square Test",
                    "Variables": "High Risk Flag vs Label",
                    "Statistic Type": "Chi2-statistic",
                    "Statistic": np.nan,
                    "P-value": np.nan,
                    "Conclusion": "Insufficient data (no non-null pairs after dropping NaNs)"
                })
        else:
             test_results.append({
                "Test": "Chi-Square Test",
                "Variables": "High Risk Flag vs Label",
                "Statistic Type": "Chi2-statistic",
                "Statistic": np.nan,
                "P-value": np.nan,
                "Conclusion": "Missing 'high_risk_flag' or 'label' column"
            })


        # 5. Pearson Correlation: Age vs BMI
        if 'age' in df.columns and 'bmi' in df.columns and df['age'].nunique() > 1 and df['bmi'].nunique() > 1:
            # Drop rows with NaN in either 'age' or 'bmi' for correlation
            df_corr = df[['age', 'bmi']].dropna()
            if len(df_corr) > 1:
                corr_age_bmi, p_value_age_bmi = pearsonr(df_corr['age'], df_corr['bmi'])
                test_results.append({
                    "Test": "Pearson Correlation",
                    "Variables": "Age vs BMI",
                    "Statistic Type": "Correlation Coefficient",
                    "Statistic": round(corr_age_bmi, 3) if not pd.isna(corr_age_bmi) else np.nan,
                    "P-value": round(p_value_age_bmi, 3) if not pd.isna(p_value_age_bmi) else np.nan,
                    "Conclusion": get_conclusion(p_value_age_bmi) + (f" (Correlation: {corr_age_bmi:.3f})" if not pd.isna(corr_age_bmi) else "")
                })
            else:
                 test_results.append({
                    "Test": "Pearson Correlation",
                    "Variables": "Age vs BMI",
                    "Statistic Type": "Correlation Coefficient",
                    "Statistic": np.nan,
                    "P-value": np.nan,
                    "Conclusion": "Insufficient data (not enough non-null pairs)"
                })
        else:
            test_results.append({
                "Test": "Pearson Correlation",
                "Variables": "Age vs BMI",
                "Statistic Type": "Correlation Coefficient",
                "Statistic": np.nan,
                "P-value": np.nan,
                "Conclusion": "Insufficient data (Age or BMI is constant or missing)"
            })

        # 6. Mann-Whitney U Test: Medication Score by Label (0 vs 1)
        if 'medication_score' in df.columns and 'label' in df.columns and df['label'].nunique() > 1:
             group0 = df[df['label'] == 0]['medication_score'].dropna()
             group1 = df[df['label'] == 1]['medication_score'].dropna()
             if len(group0) > 0 and len(group1) > 0: # Need at least 1 data point in each group for Mann-Whitney U
                try: # Mann-Whitney requires at least one observation in each group
                    u_statistic, p_value_mannwhitneyu = mannwhitneyu(group0, group1)
                    test_results.append({
                        "Test": "Mann-Whitney U Test",
                        "Variables": "Medication Score by Label (0 vs 1)",
                        "Statistic Type": "U-statistic",
                        "Statistic": round(u_statistic, 3) if not pd.isna(u_statistic) else np.nan,
                        "P-value": round(p_value_mannwhitneyu, 3) if not pd.isna(p_value_mannwhitneyu) else np.nan,
                        "Conclusion": get_conclusion(p_value_mannwhitneyu)
                    })
                except ValueError:
                     test_results.append({
                        "Test": "Mann-Whitney U Test",
                        "Variables": "Medication Score by Label (0 vs 1)",
                        "Statistic Type": "U-statistic",
                        "Statistic": np.nan,
                        "P-value": np.nan,
                        "Conclusion": "Insufficient data (Issue with groups or data after dropping NaNs)"
                    })
             else:
                 test_results.append({
                    "Test": "Mann-Whitney U Test",
                    "Variables": "Medication Score by Label (0 vs 1)",
                    "Statistic Type": "U-statistic",
                    "Statistic": np.nan,
                    "P-value": np.nan,
                    "Conclusion": "Insufficient data (Both labels not present, Medication Score missing, or not enough data in groups after dropping NaNs)"
                })
        else:
             test_results.append({
                "Test": "Mann-Whitney U Test",
                "Variables": "Medication Score by Label (0 vs 1)",
                "Statistic Type": "U-statistic",
                "Statistic": np.nan,
                "P-value": np.nan,
                "Conclusion": "Missing 'medication_score' or 'label' column, or 'label' has only one unique value"
            })


        # 7. Chi-Square Test: Antiplatelet vs Anticoagulant
        st.markdown("### Detailed Chi-Square Test: Antiplatelet vs Anticoagulant")
        if 'antiplatelet' in df.columns and 'anticoagulant' in df.columns:
            # Drop rows where either antiplatelet or anticoagulant is NaN
            df_chi2_ap_ac = df[['antiplatelet', 'anticoagulant']].dropna()
            if not df_chi2_ap_ac.empty:
                contingency_ap_ac = pd.crosstab(df_chi2_ap_ac['antiplatelet'], df_chi2_ap_ac['anticoagulant'])

                st.markdown(r"""
                    **Hypotheses:**
                    * Null Hypothesis ($H_0$): There is no association between Antiplatelet and Anticoagulant medication use.
                    * Alternate Hypothesis ($H_1$): There is an association between Antiplatelet and Anticoagulant medication use.
                    """)

                st.markdown("**Observed Contingency Table:**")
                st.dataframe(contingency_ap_ac)

                # Check if the contingency table has more than one row/column and sufficient data
                if contingency_ap_ac.shape[0] > 1 and contingency_ap_ac.shape[1] > 1 and contingency_ap_ac.values.sum() >= 5: # Adding a check for total observations as chi-square assumption
                    try:
                        stat_chi2_ap_ac, p_value_chi2_ap_ac, dof_chi2_ap_ac, expected_chi2_ap_ac = chi2_contingency(contingency_ap_ac, correction=False)

                        st.markdown("**Expected Contingency Table:**")
                        st.dataframe(pd.DataFrame(expected_chi2_ap_ac, index=contingency_ap_ac.index, columns=contingency_ap_ac.columns))

                        st.markdown(r"""
                            **Chi-Square Statistic Calculation:**
                            The Chi-Square statistic is calculated as the sum of the squared differences between the observed ($O$) and expected ($E$) frequencies, divided by the expected frequencies:
                            $$\chi^2 = \sum \frac{(O - E)^2}{E}$$
                            Degrees of Freedom (DOF) = (Number of Rows - 1) * (Number of Columns - 1) = (%i - 1) * (%i - 1) = %i
                            """ % (contingency_ap_ac.shape[0], contingency_ap_ac.shape[1], dof_chi2_ap_ac))

                        # Check for Yates' correction applicability and apply if needed
                        yates_applied = False
                        if contingency_ap_ac.shape == (2, 2) and (expected_chi2_ap_ac < 5).any():
                             st.markdown(r"""
                                **Yates' Correction for Continuity:**
                                Yates' correction is applied to the Chi-Square test for 2x2 contingency tables when the expected frequencies in any cell are less than 5. It reduces the difference between observed and expected frequencies by 0.5 before squaring, to better approximate the continuous Chi-Square distribution. The formula becomes:
                                $$\chi^2 = \sum \frac{(|O - E| - 0.5)^2}{E}$$
                                We will now calculate the Chi-Square statistic with Yates' correction.
                                """)
                             stat_chi2_ap_ac_yates, p_value_chi2_ap_ac_yates, _, _ = chi2_contingency(contingency_ap_ac, correction=True)
                             stat_to_report = round(stat_chi2_ap_ac_yates, 3)
                             p_value_to_report = round(p_value_chi2_ap_ac_yates, 3)
                             yates_applied = True
                             st.write(f"Chi-squared statistic (with Yates' correction): {stat_to_report}")
                             st.write(f"P-value (with Yates' correction): {p_value_to_report}")

                        else:
                             stat_to_report = round(stat_chi2_ap_ac, 3)
                             p_value_to_report = round(p_value_chi2_ap_ac, 3)
                             st.write(f"Chi-squared statistic: {stat_to_report}")
                             st.write(f"P-value: {p_value_to_report}")

                        test_results.append({
                            "Test": "Chi-Square Test",
                            "Variables": "Antiplatelet vs Anticoagulant",
                            "Statistic Type": f"Chi2-statistic ({'with Yates' if yates_applied else 'without Yates'})",
                            "Statistic": stat_to_report,
                            "P-value": p_value_to_report,
                            "Conclusion": get_conclusion(p_value_to_report)
                        })

                    except ValueError:
                         test_results.append({
                            "Test": "Chi-Square Test",
                            "Variables": "Antiplatelet vs Anticoagulant",
                            "Statistic Type": "Chi2-statistic",
                            "Statistic": np.nan,
                            "P-value": np.nan,
                            "Conclusion": "Insufficient data (Issue with contingency table after dropping NaNs)"
                        })
                else:
                     test_results.append({
                        "Test": "Chi-Square Test",
                        "Variables": "Antiplatelet vs Anticoagulant",
                        "Statistic Type": "Chi2-statistic",
                        "Statistic": np.nan,
                        "P-value": np.nan,
                        "Conclusion": "Insufficient data or variability (need >1 row/col and sum >= 5 after dropping NaNs)"
                    })
            else:
                test_results.append({
                    "Test": "Chi-Square Test",
                    "Variables": "Antiplatelet vs Anticoagulant",
                    "Statistic Type": "Chi2-statistic",
                    "Statistic": np.nan,
                    "P-value": np.nan,
                    "Conclusion": "Insufficient data (no non-null pairs after dropping NaNs)"
                })
        else:
             test_results.append({
                "Test": "Chi-Square Test",
                "Variables": "Antiplatelet vs Anticoagulant",
                "Statistic Type": "Chi2-statistic",
                "Statistic": np.nan,
                "P-value": np.nan,
                "Conclusion": "Missing 'antiplatelet' or 'anticoagulant' column"
            })

        # --- Added Statistical Tests ---

        # 8. Correlation Matrix for Numerical Variables (Textual Summary)
        st.markdown("### Correlation Matrix (Numerical Variables)")
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numerical_cols) > 1:
            try:
                df_corr_matrix = df[numerical_cols].dropna()
                if len(df_corr_matrix) > 1:
                    corr_matrix = df_corr_matrix.corr()
                    st.write("Pairwise Pearson Correlation Coefficients:")
                    st.dataframe(corr_matrix) # Display as a DataFrame

                    # Generate summary for correlation matrix
                    correlation_summary_points = []
                    
                    # Store correlations to sort by absolute value later
                    correlations_to_summarize = []

                    for i in range(len(corr_matrix.columns)):
                        for j in range(i + 1, len(corr_matrix.columns)):
                            col1 = corr_matrix.columns[i]
                            col2 = corr_matrix.columns[j]
                            corr_val = corr_matrix.iloc[i, j]

                            # Only highlight correlations between 0.7 and 0.9 (exclusive of 0.9)
                            if 0.7 <= abs(corr_val) < 0.9:
                                correlations_to_summarize.append((abs(corr_val), col1, col2, corr_val))

                    # Sort by absolute correlation value in descending order
                    correlations_to_summarize.sort(key=lambda x: x[0], reverse=True)

                    for abs_corr, col1, col2, corr_val in correlations_to_summarize:
                        relationship_type = "positive" if corr_val > 0 else "negative"
                        correlation_summary_points.append(f"A notable {relationship_type} linear relationship ({corr_val:.3f}) exists between **{col1}** and **{col2}**.")
                        # Limit to 3 points
                        if len(correlation_summary_points) >= 3:
                            break # Exit loop once 3 points are collected

                    correlation_summary = ""
                    if correlation_summary_points:
                        correlation_summary += "Key correlations (between 0.7 and 0.9, exclusive of 0.9) observed:\n\n"
                        for point in correlation_summary_points:
                            correlation_summary += f"- {point}\n"
                        if len(correlations_to_summarize) > len(correlation_summary_points):
                            correlation_summary += f"- ... and {len(correlations_to_summarize) - len(correlation_summary_points)} more such correlations.\n"
                    else:
                        correlation_summary += "No correlations were found in the range of 0.7 to 0.9 (exclusive of 0.9) between the numerical variables."

                    st.markdown(correlation_summary) # Display summary directly below the table

                    test_results.append({
                        "Test": "Correlation Matrix",
                        "Variables": ", ".join(numerical_cols),
                        "Statistic Type": "Correlation Coefficients",
                        "Statistic": "See table",
                        "P-value": np.nan, # P-values for individual correlations could be added if needed, using np.nan
                        "Conclusion": correlation_summary # This will be added to the overall interpretation
                    })
                else:
                    st.info("Not enough data points with non-null numerical values to generate a correlation matrix.")
                    test_results.append({
                        "Test": "Correlation Matrix",
                        "Variables": ", ".join(numerical_cols),
                        "Statistic Type": "Correlation Coefficients",
                        "Statistic": np.nan,
                        "P-value": np.nan,
                        "Conclusion": "Insufficient data points with non-null numerical values."
                    })
            except Exception as e:
                st.warning(f"Could not generate correlation matrix: {e}")
                test_results.append({
                    "Test": "Correlation Matrix",
                    "Variables": ", ".join(numerical_cols),
                    "Statistic Type": "Correlation Coefficients",
                    "Statistic": np.nan,
                    "P-value": np.nan,
                    "Conclusion": f"Could not generate correlation matrix: {e}"
                })
        else:
            st.info("Not enough numerical columns to generate a correlation matrix.")
            test_results.append({
                "Test": "Correlation Matrix",
                "Variables": "N/A",
                "Statistic Type": "Correlation Coefficients",
                "Statistic": np.nan,
                "P-value": np.nan,
                "Conclusion": "Insufficient numerical columns."
            })


        # Display results in a table
        st.markdown("### Summary of Statistical Test Results")
        results_df = pd.DataFrame(test_results)

        # Convert 'P-value' column to numeric, coercing errors to NaN
        results_df['P-value'] = pd.to_numeric(results_df['P-value'], errors='coerce')

        # Apply styling to the DataFrame for better readability
        st.dataframe(results_df.style.background_gradient(cmap='Blues', subset=['P-value']).set_properties(**{'font-size': '10pt'}))


        # Download button for Statistical Results Table
        st.markdown(download_dataframe_as_csv(results_df, "statistical_results.csv"), unsafe_allow_html=True)


        # Consolidated Interpretation Box
        st.markdown("### Interpretation of Automated Statistical Test Results")
        interpretation_html = """
            <div style="background-color:#e9ecef; padding: 15px; border-left: 5px solid #007bff; border-radius: 5px; margin-top: 10px;">
            <p style="color: #495057;">
            Below are detailed interpretations for each automated statistical test presented in the table above, based on their p-values and relevance to the PPIcheck scoring system:
            </p>
            <ul>
            """

        for index, row in results_df.iterrows():
            test_name = row['Test']
            variables = row['Variables']
            # Corrected: Changed 'P_value' to 'P-value' to match the DataFrame column name
            p_value = row['P-value'] 
            conclusion_text = row['Conclusion']
            statistic_value = row['Statistic']

            interpretation_html += f"""<li><p style="color: #495057;"><b>{test_name} ({variables}):</b> """

            if conclusion_text.startswith("Insufficient data") or conclusion_text.startswith("Missing"):
                interpretation_html += f"""<span style="color: #6c757d;">{conclusion_text}.</span> <i>This test could not be performed due to data limitations.</i></p></li>"""
            elif test_name == "Correlation Matrix":
                 interpretation_html += f"""<span style="color: #495057;">{conclusion_text}</span></p></li>"""
            elif p_value is not None and not pd.isna(p_value): # Check if p_value is valid for inferential tests
                if p_value < 0.05:
                    interpretation_html += f"""
                        <span style="color: #28a745;">Statistically Significant (p = {p_value:.3f}).</span>
                        There is strong evidence from this dataset to suggest a real association or difference between {variables}.
                        """
                    if test_name == "One-Way ANOVA":
                        interpretation_html += "This suggests the mean Indication Score differs significantly across different PPI Routes in your data."
                    elif test_name == "Kruskal-Wallis Test":
                         interpretation_html += "This indicates the distribution of NSAID Scores differs significantly across different PPI Doses in your data."
                    elif test_name == "Independent t-Test":
                         interpretation_html += "This suggests the mean BMI is significantly different between the Low Risk (Label 0) and High Risk (Label 1) groups in your data."
                    elif test_name == "Chi-Square Test" and "High Risk Flag vs Label" in variables:
                         interpretation_html += "This confirms a significant association between the High Risk Flag and the final risk Label in your data."
                    elif test_name == "Pearson Correlation":
                         interpretation_html += f"There is a significant linear relationship between Age and BMI (Correlation: {statistic_value:.3f}) in your data."
                    elif test_name == "Mann-Whitney U Test":
                         interpretation_html += "This indicates the distribution of Medication Scores differs significantly between the Low Risk (Label 0) and High Risk (Label 1) groups in your data."
                    elif test_name == "Chi-Square Test" and "Antiplatelet vs Anticoagulant" in variables:
                         interpretation_html += "This suggests a significant association between Antiplatelet and Anticoagulant medication use in your data."

                    interpretation_html += "</p></li>"

                else: # p_value >= 0.05
                    interpretation_html += f"""
                        <span style="color: #dc3545;">Not Statistically Significant (p = {p_value:.3f}).</span>
                        There is insufficient evidence from this dataset to conclude a real association or difference between {variables}. Any observed patterns could be due to random variation in your data.
                        </p></li>"""
            else: # Handle cases where p_value might be None or NaN but not caught by "Insufficient data"
                 interpretation_html += f"""<span style="color: #6c757d;">Could not determine significance.</span> <i>An issue occurred while processing the p-value for this test.</i></p></li>"""


        interpretation_html += """
            </ul>
            <p style="color: #495057;">
            These interpretations are based on the dataset currently loaded in the app. Ensure your data meets the requirements for each test.
            </p>
            </div>
            """
        st.markdown(interpretation_html, unsafe_allow_html=True)


# --- Manual Statistical Analysis (Individual Tests) ---
st.sidebar.markdown("---") # Add a separator
st.sidebar.subheader("Manual Statistical Analysis (Individual Tests)")

# Only show manual options if data is available
if data_available:
    analysis_type = st.sidebar.selectbox("Select Individual Analysis Type", ["Descriptive Statistics", "Correlation Analysis", "Group Comparison (T-test/ANOVA)", "Categorical Association (Chi-squared/Kruskal)"])

    if analysis_type == "Descriptive Statistics":
        st.subheader("Descriptive Statistics")
        st.write(st.session_state.df.describe())

    elif analysis_type == "Correlation Analysis":
        st.subheader("Correlation Analysis")
        numerical_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
        if len(numerical_cols) >= 2:
            col1 = st.selectbox("Select first column", numerical_cols)
            col2 = st.selectbox("Select second column", numerical_cols)
            if st.button("Calculate Pearson Correlation"):
                try:
                    # Drop rows with NaN in selected columns for correlation
                    df_corr_manual = st.session_state.df[[col1, col2]].dropna()
                    if len(df_corr_manual) > 1:
                        corr, p_value_manual_corr = pearsonr(df_corr_manual[col1], df_corr_manual[col2])
                        st.write(f"Pearson correlation between {col1} and {col2}: {corr:.3f}")
                        st.write(f"P-value: {p_value_manual_corr:.3f}")

                        # Dynamic Explanation based on results
                        st.markdown(f"""
                            ---
                            ### Interpretation of Pearson Correlation ({col1} vs {col2})

                            The Pearson correlation coefficient ($r$) for {col1} and {col2} is **{corr:.3f}**.

                            This value indicates:

                            * **Strength:** {'A strong' if abs(corr) >= 0.7 else 'A moderate' if abs(corr) >= 0.4 else 'A weak' if abs(corr) >= 0.1 else 'A negligible'} linear relationship.
                            * **Direction:** {'Positive' if corr > 0.1 else 'Negative' if corr < -0.1 else 'No clear'} linear relationship.
                                * A positive correlation ({corr:.3f} > 0) suggests that as {col1} increases, {col2} tends to increase.
                                * A negative correlation ({corr:.3f} < 0) suggests that as {col1} increases, {col2} tends to decrease.
                                * A correlation near zero ({corr:.3f}) suggests no clear linear trend between the variables.

                            The p-value for this correlation is **{p_value_manual_corr:.3f}**.

                            Based on a common significance level of 0.05:

                            * {'**Statistically Significant (p < 0.05)**: We reject the null hypothesis. There is sufficient evidence to conclude there is a statistically significant linear relationship between ' + col1 + ' and ' + col2 + ' in the population from which this data was sampled.' if p_value_manual_corr < 0.05 else '**Not Statistically Significant (p >= 0.05)**: We fail to reject the null hypothesis. There is not enough evidence from this sample to conclude there is a statistically significant linear relationship between ' + col1 + ' and ' + col2 + ' in the population. The observed correlation could be due to random chance.'}

                            Remember that correlation does not imply causation. This analysis only describes the linear association between the two variables in your dataset.
                            """)

                    else:
                         st.warning("Insufficient data points with non-null values in selected columns for correlation analysis.")
                except Exception as e:
                    st.error(f"Error calculating correlation: {e}")
        else:
            st.warning("Need at least two numerical columns for correlation analysis.")


    elif analysis_type == "Group Comparison (T-test/ANOVA)":
        st.subheader("Group Comparison (T-test/ANOVA)")
        numerical_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = st.session_state.df.select_dtypes(include='object').columns.tolist()

        if numerical_cols and categorical_cols:
            numerical_var = st.selectbox("Select numerical variable", numerical_cols)
            categorical_var = st.selectbox("Select grouping variable", categorical_cols)

            groups = st.session_state.df[categorical_var].unique()

            if len(groups) >= 2:
                 # Filter out groups with insufficient data after dropping NaNs
                 group_data = [st.session_state.df[st.session_state.df[categorical_var] == group][numerical_var].dropna() for group in groups]
                 valid_group_data = [data for data in group_data if len(data) > 1] # Need at least 2 samples for t-test/ANOVA

                 if len(valid_group_data) == 2:
                    st.write(f"Performing independent samples t-test for {numerical_var} by {categorical_var}")
                    if st.button("Run T-test"):
                        try:
                            t_stat, p_value = ttest_ind(valid_group_data[0], valid_group_data[1])
                            st.write(f"T-statistic: {t_stat:.3f}, P-value: {p_value:.3f}")
                        except ValueError:
                             st.error("Error performing t-test. Ensure groups have variability.")
                        except Exception as e:
                            st.error(f"An unexpected error occurred during t-test: {e}")

                 elif len(valid_group_data) > 2:
                    st.write(f"Performing one-way ANOVA for {numerical_var} by {categorical_var}")
                    if st.button("Run ANOVA"):
                        try:
                            f_stat, p_value = f_oneway(*valid_group_data)
                            st.write(f"F-statistic: {f_stat:.3f}, P-value: {p_value:.3f}")
                        except ValueError:
                             st.error("Error performing ANOVA. Ensure groups have variability.")
                        except Exception as e:
                            st.error(f"An unexpected error occurred during ANOVA: {e}")
                 else:
                     st.warning("Not enough data points with non-null values in groups (need at least two groups with >1 data point) for T-test/ANOVA.")

            else:
                st.warning("Grouping variable must have at least two unique values for comparison.")
        else:
            st.warning("Need at least one numerical and one categorical column for group comparison.")

    elif analysis_type == "Categorical Association (Chi-squared/Kruskal)":
        st.subheader("Categorical Association (Chi-squared/Kruskal)")
        categorical_cols = st.session_state.df.select_dtypes(include='object').columns.tolist()
        numerical_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()

        analysis_method = st.radio("Select method", ["Chi-squared (Categorical vs Categorical)", "Kruskal-Wallis (Numerical vs Categorical)"])

        if analysis_method == "Chi-squared (Categorical vs Categorical)":
            if len(categorical_cols) >= 2:
                cat_col1 = st.selectbox("Select first categorical column", categorical_cols)
                cat_col2 = st.selectbox("Select second categorical column", categorical_cols)
                if st.button("Calculate Chi-squared"):
                    try:
                        # Drop rows with NaN in selected columns for chi-squared
                        df_chi2_manual = st.session_state.df[[cat_col1, cat_col2]].dropna()
                        if not df_chi2_manual.empty:
                            contingency_table = pd.crosstab(df_chi2_manual[cat_col1], df_chi2_manual[cat_col2])

                            st.markdown(r"""
                                **Hypotheses:**
                                * Null Hypothesis ($H_0$): There is no association between {cat_col1} and {cat_col2}.
                                * Alternate Hypothesis ($H_1$): There is an association between {cat_col1} and {cat_col2}.
                                """)

                            st.markdown("**Observed Contingency Table:**")
                            st.dataframe(contingency_table)

                            # Check if the contingency table has more than one row/column and sufficient data
                            if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1 and contingency_table.values.sum() >= 5:
                                # Calculate expected values without correction first to check for small values
                                _, _, _, expected_manual_check = chi2_contingency(contingency_table, correction=False)
                                yates_applied_manual = False
                                if contingency_table.shape == (2, 2) and (expected_manual_check < 5).any():
                                     st.info("Applying Yates' Correction for Continuity as expected frequencies are less than 5 in a 2x2 table.")
                                     chi2_stat_manual, p_value_manual, dof_manual, ex_manual = chi2_contingency(contingency_table, correction=True)
                                     yates_applied_manual = True
                                else:
                                     chi2_stat_manual, p_value_manual, dof_manual, ex_manual = chi2_contingency(contingency_table, correction=False)

                                st.markdown("**Expected Contingency Table:**")
                                st.dataframe(pd.DataFrame(ex_manual, index=contingency_table.index, columns=contingency_table.columns))

                                st.markdown(r"""
                                    **Chi-Square Statistic Calculation:**
                                    The Chi-Square statistic is calculated as the sum of the squared differences between the observed ($O$) and expected ($E$) frequencies, divided by the expected frequencies:
                                    $$\chi^2 = \sum \frac{(O - E)^2}{E}$$
                                    Degrees of Freedom (DOF) = (Number of Rows - 1) * (Number of Columns - 1) = (%i - 1) * (%i - 1) = %i
                                    """ % (contingency_table.shape[0], contingency_table.shape[1], dof_manual))

                                if yates_applied_manual:
                                     st.markdown("""
                                        **Yates' Correction for Continuity:**
                                        Yates' correction was applied because it is a 2x2 table and at least one expected frequency was less than 5. It adjusts the calculation to improve the approximation to the Chi-Square distribution.
                                        """)

                                st.write(f"Chi-squared statistic ({'with Yates' if yates_applied_manual else 'without Yates'}): {chi2_stat_manual:.3f}")
                                st.write(f"P-value: {p_value_manual:.3f}")

                            else:
                                st.warning("Insufficient data or variability in selected columns (need >1 row/col and sum >= 5 after dropping NaNs) for Chi-squared test.")
                        else:
                             st.warning("Insufficient data points with non-null values in selected columns for Chi-squared test.")
                    except ValueError as e:
                         st.error(f"Error calculating Chi-squared: {e}. This might happen if the contingency table has zero cells.")
                    except Exception as e:
                         st.error(f"An unexpected error occurred during Chi-squared calculation: {e}")
            else:
                st.warning("Need at least two categorical columns for Chi-squared test.")

        elif analysis_method == "Kruskal-Wallis (Numerical vs Categorical)":
             if numerical_cols and categorical_cols:
                numerical_var = st.selectbox("Select numerical variable for Kruskal-Wallis", numerical_cols)
                categorical_var = st.selectbox("Select grouping variable for Kruskal-Wallis", categorical_cols)

                groups = st.session_state.df[categorical_var].unique()
                if len(groups) >= 2:
                    # Filter out groups with insufficient data after dropping NaNs
                    group_data = [st.session_state.df[st.session_state.df[categorical_var] == group][numerical_var].dropna() for group in groups]
                    valid_group_data = [data for data in group_data if len(data) > 0] # Need at least 1 sample for Kruskal-Wallis

                    if len(valid_group_data) >= 2:
                        if st.button("Run Kruskal-Wallis"):
                            try:
                                h_stat, p_value = kruskal(*valid_group_data)
                                st.write(f"Kruskal-Wallis H-statistic: {h_stat:.3f}, P-value: {p_value:.3f}")
                            except ValueError as e:
                                 st.error(f"Error calculating Kruskal-Wallis: {e}. This might happen if all groups have the same values or insufficient data points.")
                            except Exception as e:
                                 st.error(f"An unexpected error occurred during Kruskal-Wallis calculation: {e}")
                    else:
                         st.warning("Need at least two groups with data points (after dropping NaNs) for Kruskal-Wallis test.")
                else:
                    st.warning("Grouping variable must have at least two unique values for Kruskal-Wallis test.")
             else:
                st.warning("Need at least one numerical and one categorical column for Kruskal-Wallis test.")
else:
    st.info("Upload or generate data to enable manual statistical analysis options.")


# --- Machine Learning Model Training and Evaluation ---
st.sidebar.markdown("---") # Add a separator
st.sidebar.subheader("Machine Learning Assessment")
if st.sidebar.button("Run ML Assessment", disabled=not (data_available and ml_ready_classification)):
    if not (data_available and ml_ready_classification):
        pass # Warning is shown above
    else:
        # Using the specified features for ML assessment
        # Decide whether to run classification or regression based on data readiness
        if ml_ready_classification:
            st.markdown(
                """
                <h2 style="color: #007bff; text-align: center; margin-top: 30px; margin-bottom: 20px;">Machine Learning Model Performance (Classification)</h2>
                """, unsafe_allow_html=True
            )
            required_ml_clf_features = ['age', 'bmi', 'nsaid_score', 'antiplatelet_score', 'anticoagulant_score',
                                         'indication_score', 'ppi_score', 'high_risk_flag', 'triple_flag']

            # Drop rows with NaN in features or label before splitting
            df_ml_clf = df[required_ml_clf_features + ['label']].dropna() # Ensure label is also included for dropna
            if len(df_ml_clf) < 2 or df_ml_clf['label'].nunique() < 2:
                 st.warning("Insufficient data points with non-null values or only one class present after handling missing values for Classification ML assessment.")
            else:
                X_ml_clf = df_ml_clf[required_ml_clf_features]
                y_ml_clf = df_ml_clf['label']
                X_train, X_test, y_train, y_test = train_test_split(X_ml_clf, y_ml_clf, test_size=0.2, random_state=42, stratify=y_ml_clf) # Added stratify

                models = {
                    'Logistic Regression': LogisticRegression(max_iter=1000),
                    'Random Forest': RandomForestClassifier(),
                    'XGBoost': xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False),
                    'Gradient Boosting': GradientBoostingClassifier()
                }

                ml_results_text = "Machine Learning Model Performance Metrics (Classification):\n\n"


                for name, model in models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]

                    st.markdown(f"### {name}")

                    # Box for Model Performance Metrics and Interpretation
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    auc = roc_auc_score(y_test, y_prob)

                    ml_metrics_html = f"""
                        <div style="background-color:#e9ecef; padding: 15px; border-left: 5px solid #007bff; border-radius: 5px; margin-top: 10px; margin-bottom: 20px;">
                            <h4 style="color: #343a40; margin-top: 0;'>Model Performance Metrics:</h4>
                            <p style="color: #495057;"><b>Accuracy:</b> {accuracy:.2f}</p>
                            <p style="color: #495057;"><b>Precision:</b> {precision:.2f}</p>
                            <p style="color: #495057;"><b>F1 Score:</b> {f1:.2f}</p>
                            <p style="color: #495057;"><b>AUC:</b> {auc:.2f}</p>
                            <h4 style="color: #343a40; margin-top: 15px;'>Interpretation:</h4>
                            <p style="color: #495057;">
                            These metrics show how well the model performs on the test data. High scores (closer to 1.0 for Accuracy, Precision, F1, and AUC) indicate better performance. The AUC represents the model's ability to distinguish between the positive and negative classes.
                            </p>
                            <h4 style="color: #343a40; margin-top: 15px;'>Confusion Matrix:</h4>
                    """
                    # Display confusion matrix as a DataFrame with explicit labels
                    cm = confusion_matrix(y_test, y_pred)
                    cm_df = pd.DataFrame(cm, 
                                         index=['Actual Negative (Label 0)', 'Actual Positive (Label 1)'],
                                         columns=['Predicted Negative (Label 0)', 'Predicted Positive (Label 1)'])
                    
                    # Add True Positive, True Negative, False Positive, False Negative labels
                    cm_labeled_df = pd.DataFrame({
                        'Metric': ['True Negative (TN)', 'False Positive (FP)', 'False Negative (FN)', 'True Positive (TP)'],
                        'Count': [cm[0,0], cm[0,1], cm[1,0], cm[1,1]]
                    })

                    ml_metrics_html += cm_df.to_html(classes='table table-striped') + "<br>" # Convert original CM to HTML table
                    ml_metrics_html += "<p style='color: #495057;'><b>Detailed Confusion Matrix Breakdown:</b></p>"
                    ml_metrics_html += cm_labeled_df.to_html(classes='table table-striped') # Convert labeled CM to HTML table
                    ml_metrics_html += """
                        </div>
                    """
                    st.markdown(ml_metrics_html, unsafe_allow_html=True)


                    ml_results_text += f"{name}:\n"
                    ml_results_text += f"  Accuracy: {accuracy:.2f}\n"
                    ml_results_text += f"  Precision: {precision:.2f}\n"
                    ml_results_text += f"  F1 Score: {f1:.2f}\n"
                    ml_results_text += f"  AUC: {auc:.2f}\n"
                    ml_results_text += f"  Confusion Matrix:\n"
                    ml_results_text += f"    True Negative (TN): {cm[0,0]}\n"
                    ml_results_text += f"    False Positive (FP): {cm[0,1]}\n"
                    ml_results_text += f"    False Negative (FN): {cm[1,0]}\n"
                    ml_results_text += f"    True Positive (TP): {cm[1,1]}\n\n"


                # Download button for ML Results
                st.markdown(download_text_file(ml_results_text, "ml_performance_metrics_classification.txt"), unsafe_allow_html=True)

        if not ml_ready_classification: # Check only classification readiness now
             st.info("Data not ready for ML assessment. Check warnings above.")


# Add professional CSS styling
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        max-width: 1200px;
        padding: 2rem 3rem;
        margin: 0 auto;
    }

    /* Header styling */
    h1 {
        color: #2c3e50;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        text-align: center;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem 1.5rem;
    }

    /* Button styling */
    .stButton>button {
        background-color: #28a745; /* Changed button color to green */
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.3); /* Added subtle border */
    }

    /* Button styling on hover */
    .stButton>button:hover {
        background-color: rgba(255, 255, 255, 0.8); /* White with slight transparency */
        color: #333; /* Dark text for contrast */
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        backdrop-filter: blur(5px); /* Glass effect */
        -webkit-backdrop-filter: blur(5px); /* Safari support */
    }

    /* Table header styling */
    .stDataFrame > div > div > div > div > div > div > table th {
        background-color: #FFA500; /* Orange color */
        color: #333; /* Darker text for contrast */
    }


    /* Table styling */
    .stTable {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Chart container styling */
    .stPlotlyChart {
        border-radius: 8px;
        box_shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }

        h1 {
            font_size: 2rem;
        }
    }

    /* Header Footer styling (for the moved footer) */
    .header-footer {
        width: 100%;
        background-color: #f8f9fa;
        color: #6c757d;
        text-align: center;
        padding: 10px 0;
        font-size: 0.9rem;
        border-bottom: 1px solid #ced4da; /* Added a subtle border bottom */
        margin-bottom: 20px; /* Space between header-footer and next section */
    }
</style>
""", unsafe_allow_html=True
)
