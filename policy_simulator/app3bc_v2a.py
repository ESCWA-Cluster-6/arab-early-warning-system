# app3bc_v2.py - User-friendly version with clear explanations
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV

# Optional MI (can be toggled in UI)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

import shap
import optuna
import warnings, random

# ──────────────────────────────
# PAGE CONFIG & STYLE
# ──────────────────────────────
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 11})

st.set_page_config(page_title="Conflict Risk Early Warning System", layout="wide")
PURPLE = "#FBF8FF"
st.markdown(f"""
    <style>
        .stApp, [data-testid="stAppViewContainer"] {{ background: {PURPLE}; }}
        [data-testid="stSidebar"] > div:first-child {{ background: {PURPLE}; }}
        [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
        .metric-label {{ font-size: 14px !important; }}
        .metric-value {{ font-size: 24px !important; font-weight: bold !important; }}
    </style>
""", unsafe_allow_html=True)
plt.rcParams["figure.facecolor"] = "none"
plt.rcParams["axes.facecolor"]   = "none"
plt.rcParams["savefig.facecolor"] = "none"

st.title("🌍 Conflict Risk Early Warning System")
st.markdown("""
**Welcome to the Conflict Risk Early Warning System** - A data-driven tool to help policymakers understand and prevent conflicts before they occur.

This system analyzes **45 development indicators** across multiple dimensions (economic, social, environmental, and governance) to:
- 🎯 **Predict** which countries are at risk of conflict in the next year
- 🔍 **Identify** the most important factors driving conflict risk
- 📊 **Simulate** how policy interventions could reduce (or increase) conflict probability
""")

with st.expander("ℹ️ How to Use This Tool", expanded=False):
    st.markdown("""
    **For Policymakers and Development Practitioners:**
    
    1. **📊 Data Overview Tab**: Understand data quality and relationships between indicators
    2. **🎯 Model Performance Tab**: See how accurately we can predict conflicts
    3. **🔍 Risk Factors Tab**: Identify which factors matter most for conflict prevention
    4. **🎛️ Impact Calculator Tab**: Test how changing multiple factors affects conflict risk
    5. **🏛️ Country Analysis Tab**: Examine specific policy interventions for individual countries
    
    **Key Terms Explained:**
    - **Conflict Risk**: The probability (0-100%) that a country will experience conflict next year
    - **Risk Factors**: Development indicators that influence conflict probability
    - **Policy Intervention**: Simulated changes to see potential impact on conflict risk
    - **pp (percentage points)**: The absolute change in probability (e.g., from 20% to 25% = +5pp)
    """)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ARAB_COUNTRIES = [
    'Algeria','Bahrain','Comoros','Djibouti','Egypt','Iraq','Jordan',
    'Kuwait','Lebanon','Libya','Mauritania','Morocco','Oman','Palestine',
    'Qatar','Saudi Arabia','Somalia','Sudan','Syrian Arab Republic',
    'Tunisia','United Arab Emirates','Yemen'
]

# ──────────────────────────────
# Helper: readable labels
# ──────────────────────────────
def make_readable_name(feature_name: str) -> str:
    readable_names = {
    '01-fatper_nm': 'Fatalities per 100,000 Population',
    '02-political_stability_nm': 'Political Stability',
    '03-conflict proximity_nm': 'Proximity to Other Conflicts',
    '04-state_authority_nm': 'Government Authority & Control',
    '05-disp_rate_nm': 'Population Displacement Rate',
    '06-voice_accountability_nm': 'Voice & Accountability',
    '7-ren_water_nm': 'Renewable Water Resources per Person',
    '8-wather_withd_nm': 'Water Withdrawal Rate',
    '9-pop_dis_nm': 'Internally Displaced Population',
    '10-dis_dicp_nm': 'Disaster-Related Displacement',
    '11-adap_strat_nm': 'Climate Adaptation Strategies',
    '12-cli_fin_nm': 'Climate Finance Received',
    '13-remit_nm': 'Remittances (% of GDP)',
    '14-oda_nm': 'Official Development Assistance (% of GNI)',
    '15-food_ins_nm': 'Food Insecurity Level',
    '16-undernour_nm': 'Undernourishment Rate',
    '17-gini_nm': 'Income Inequality (Gini Index)',
    '18-topbottom_ratio_nm': 'Top-to-Bottom Income Ratio',
    '19-govt_debt_nm': 'Government Debt (% of GDP)',
    '20-int_use_nm': 'Internet Usage Rate',
    '21-unemp_nm': 'Unemployment Rate',
    '22-youth_nm': 'Youth Population Share',
    '23-mr5_nm': 'Under-5 Mortality Rate',
    '24-mmr_nm': 'Maternal Mortality Ratio',
    '25-exp_sch_nm': 'Expected Years of Schooling',
    '26-mean_sch_nm': 'Mean Years of Schooling',
    '27-soc_pro_nm': 'Social Protection Coverage',
    '28-water_serv_nm': 'Access to Improved Water Sources',
    '29-san_serv_nm': 'Access to Sanitation Services',
    '30-uhc_nm': 'Universal Health Coverage Index',
    '31-gpi_nm': 'Global Peace Index Score',
    '32-lfp_fem_nm': 'Female Labor Force Participation',
    '33-control_corruption_nm': 'Control of Corruption',
    '34-rule_law_nm': 'Rule of Law',
    '35-government_eff_nm': 'Government Effectiveness',
    '36-osi_nm': 'Open Society Index',
    '37-emp_to_pop_nm': 'Employment-to-Population Ratio',
    '38-territory_control_nm': 'Territorial Control',
    '39-H_index_nm': 'Health Index Score',
    '40-health_exp_nm': 'Health Expenditure (% of GDP)',
    '41-mean_schooling_nm': 'Mean Schooling Years (Alt Source)',
    '42-refugees_per_100k_nm': 'Refugees per 100,000 Population',
    '43-out_of_school_nm': 'Out-of-School Children',
    '44-precipitation_nm': 'Average Annual Precipitation',
    '45-rd_gdp_nm': 'R&D Expenditure (% of GDP)',
    '46-tax_gdp_nm': 'Tax Revenue (% of GDP)',
    '47-tech_dependence_nm': 'Technological Dependence Index',
    '48-water_stress_nm': 'Water Stress Level',
    '49-debt_gni_nm': 'Debt-to-GNI Ratio',
    '50-humanitarian_aid_nm': 'Humanitarian Aid Received',
    '51-agri_land_nm': 'Agricultural Land Area (% of Total Land)',
    '52-gdp_ppp_nm': 'GDP per Capita (PPP, USD)',
    '53-agri_gdp_nm': 'Agriculture Share of GDP (%)',
    '54-vdem_political_pol_nm': 'Political Polarization',
    '55-vdem_discussion_nm': 'Public Discussion Quality',
    '56- vdem_participation_nm': 'Political Participation'
}

    return readable_names.get(feature_name, feature_name.replace('_nm','').replace('_',' ').title())

# ──────────────────────────────
# LOAD RAW
# ──────────────────────────────
@st.cache_data
def load_raw() -> pd.DataFrame:
    df_ind = pd.read_excel("data/data_risk_v2.xlsx", sheet_name = 'data_indicators')
    df_con = pd.read_csv("data/Conflict_Status_by_Year.csv")

    df_con = df_con.melt(id_vars=['Unnamed: 0'], var_name='years', value_name='conflict')
    df_con.rename(columns={'Unnamed: 0': 'country_name'}, inplace=True)
    df_con['years'] = df_con['years'].astype(int)

    df_ind['years'] = df_ind['years'].astype(int)
    df = df_ind.merge(df_con, on=['country_name','years'], how='left')
    df['conflict'] = df['conflict'].fillna(0).astype(int)

    df = df[df['years'] >= 2006].copy()

    # Remove the two specified columns if they exist
    cols_to_remove = [
        '04-state_authority_nm', 
        '18-topbottom_ratio_nm',
        '35-government_eff_nm'
        ]
    for col in cols_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df

raw_df = load_raw()
meta_cols = ['country_code','country_name','years','conflict']
candidate_cols = [c for c in raw_df.columns if c not in meta_cols]

# ──────────────────────────────
# TIME-SAFE PANEL PREP
# ──────────────────────────────
def forward_fill_panel(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df2 = df.sort_values(['country_name','years']).copy()
    df2[cols] = df2.groupby('country_name')[cols].ffill()
    return df2

def build_target_and_split(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df2 = df.sort_values(['country_name','years']).copy()
    df2['conflict_next_year'] = df2.groupby('country_name')['conflict'].shift(-1)
    df2 = df2.dropna(subset=['conflict_next_year']).copy()
    df2['conflict_next_year'] = df2['conflict_next_year'].astype(int)

    df2['max_year'] = df2.groupby('country_name')['years'].transform('max')
    train = df2[df2['years'] < df2['max_year']].copy()
    test  = df2[df2['years'] == df2['max_year']].copy()

    train['max_train_year'] = train.groupby('country_name')['years'].transform('max')
    val = train[train['years'] == train['max_train_year']].copy()
    trn = train[train['years'] < train['max_train_year']].copy()
    return trn, val, test

def select_features_on_train(train_df: pd.DataFrame, cols: List[str], missing_thresh: float = 0.25) -> List[str]:
    miss = train_df[cols].isna().mean()
    keep = miss[miss <= missing_thresh].index.tolist()
    return keep

def fit_preprocessor(train_df: pd.DataFrame, cols: List[str]) -> Dict:
    means = train_df[cols].mean(skipna=True).values
    stds  = train_df[cols].std(skipna=True, ddof=0).replace(0, 1.0).values

    X_tr_scaled = (train_df[cols].values - means) / stds
    knn = KNNImputer(n_neighbors=5)
    knn.fit(X_tr_scaled)

    X_tr_imp_scaled = knn.transform(X_tr_scaled)
    X_tr_imp = X_tr_imp_scaled * stds + means

    medians = np.nanmedian(X_tr_imp, axis=0)
    X_tr_imp = np.where(np.isnan(X_tr_imp), medians, X_tr_imp)

    scaler = StandardScaler()
    scaler.fit(X_tr_imp)

    return {"means": means, "stds": stds, "knn": knn, "medians": medians, "scaler": scaler, "cols": cols}

def transform_with_preproc(df: pd.DataFrame, pre: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = pre["cols"]
    means, stds, knn, medians, scaler = pre["means"], pre["stds"], pre["knn"], pre["medians"], pre["scaler"]

    X = df[cols].values
    X_scaled = (X - means) / stds
    X_imp_scaled = knn.transform(X_scaled)
    X_imp = X_imp_scaled * stds + means
    X_imp = np.where(np.isnan(X_imp), medians, X_imp)

    X_model = scaler.transform(X_imp)

    X_imp_df = pd.DataFrame(X_imp, columns=cols, index=df.index)
    X_model_df = pd.DataFrame(X_model, columns=cols, index=df.index)
    return X_model_df, X_imp_df

@st.cache_data
def build_time_safe_datasets(missing_thresh: float = 0.25):
    ff_df = forward_fill_panel(raw_df, candidate_cols)

    tmp = ff_df.copy()
    tmp['conflict_next_year'] = tmp.groupby('country_name')['conflict'].shift(-1)
    tmp = tmp.dropna(subset=['conflict_next_year'])
    tmp['conflict_next_year'] = tmp['conflict_next_year'].astype(int)
    tmp['max_year'] = tmp.groupby('country_name')['years'].transform('max')

    train_all = tmp[tmp['years'] < tmp['max_year']].copy()
    test_all  = tmp[tmp['years'] == tmp['max_year']].copy()

    keep_cols = select_features_on_train(train_all, [c for c in tmp.columns if c not in meta_cols+['conflict_next_year','max_year']], missing_thresh)

    trn, val, tst = build_target_and_split(ff_df[meta_cols + keep_cols], keep_cols)

    pre = fit_preprocessor(trn, keep_cols)

    X_tr_m, X_tr_imp = transform_with_preproc(trn, pre)
    X_va_m, X_va_imp = transform_with_preproc(val, pre)
    X_te_m, X_te_imp = transform_with_preproc(tst, pre)

    y_tr = trn['conflict_next_year'].values
    y_va = val['conflict_next_year'].values
    y_te = tst['conflict_next_year'].values

    arab_mask_all = ff_df['country_name'].isin(ARAB_COUNTRIES)
    arab_df = ff_df.loc[arab_mask_all, ['country_code','country_name','years'] + keep_cols].copy()

    return {
        "keep_cols": keep_cols,
        "pre": pre,
        "splits": {
            "trn_df": trn, "val_df": val, "tst_df": tst,
            "X_tr": X_tr_m, "y_tr": y_tr,
            "X_va": X_va_m, "y_va": y_va,
            "X_te": X_te_m, "y_te": y_te,
            "X_tr_imp": X_tr_imp, "X_va_imp": X_va_imp, "X_te_imp": X_te_imp
        },
        "arab_df": arab_df
    }

data_bundle = build_time_safe_datasets(missing_thresh=0.40)
feature_cols = data_bundle["keep_cols"]
pre = data_bundle["pre"]
spl = data_bundle["splits"]
arab_df = data_bundle["arab_df"]

def get_latest_good_year(arab_df, feature_cols, max_missing_pct=0.4, min_countries=5):
    """
    Returns the most recent year with average missingness across feature_cols
    <= max_missing_pct (fraction, not percent) and at least min_countries present.

    Parameters
    ----------
    arab_df : pd.DataFrame
        DataFrame containing 'years', 'country_name', and feature columns.
    feature_cols : list
        List of feature column names to assess.
    max_missing_pct : float
        Maximum allowed average missingness fraction (0.4 = 40%).
    min_countries : int
        Minimum number of countries that must have data in that year.
    """
    # Calculate mean missingness per year across all countries/indicators
    year_quality = (
        arab_df.groupby('years')[feature_cols]
        .apply(lambda df: df.isna().mean().mean())
        .sort_index(ascending=False)
    )
    # Iterate from latest to oldest year
    for year, missing_frac in year_quality.items():
        country_count = arab_df.loc[arab_df['years'] == year, 'country_name'].nunique()
        if missing_frac <= max_missing_pct and country_count >= min_countries:
            return int(year)
    # Fallback: return latest year available
    return int(arab_df['years'].max())

# ──────────────────────────────
# TRAIN PREDICTIVE MODEL
# ──────────────────────────────
@st.cache_resource
def train_predictive(X_tr, y_tr, X_va, y_va):
    def objective(trial):
        penalty = trial.suggest_categorical('penalty', ['l1','l2'])
        C = trial.suggest_float('C', 1e-4, 1e4, log=True)
        spw = trial.suggest_float('scale_pos_weight', 1.0, 100.0)
        solver = 'liblinear' if penalty=='l1' else 'lbfgs'

        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', LogisticRegression(
                penalty=penalty, C=C, solver=solver,
                class_weight={0:1.0, 1:spw}, max_iter=2000, random_state=SEED
            ))
        ])
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_va)
        return f1_score(y_va, preds, zero_division=0)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=25, show_progress_bar=False)
    best = study.best_trial.params
    solver = 'liblinear' if best['penalty']=='l1' else 'lbfgs'

    base = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', LogisticRegression(
            penalty=best['penalty'], C=best['C'], solver=solver,
            class_weight={0:1.0, 1:best['scale_pos_weight']},
            max_iter=5000, random_state=SEED
        ))
    ])
    base.fit(X_tr, y_tr)
    cal = CalibratedClassifierCV(estimator=base, method='isotonic', cv='prefit')
    cal.fit(X_va, y_va)
    return cal, best

pred_model, pred_params = train_predictive(spl["X_tr"], spl["y_tr"], spl["X_va"], spl["y_va"])

# ──────────────────────────────
# SIDEBAR – Policy Settings
# ──────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Analysis Settings")
    st.markdown("**Adjust these settings to control the analysis complexity:**")
    
    with st.expander("⚙️ Advanced Settings", expanded=True):
        st.markdown("**Model Complexity Control**")
        st.markdown("*Lower values = simpler model (fewer factors)*")
        logC = st.slider("Simplicity Level", -4.0, -1.0, -3.0, 0.5, 
                        help="Controls how many factors the model considers. Lower = simpler",
                        key="logC_slider")
        
        st.markdown("**Feature Selection**")
        st.markdown("*Higher values = focus on most important factors*")
        l1r  = st.slider("Focus Level", 0.70, 1.00, 0.90, 0.05,
                        help="How much to focus on only the most important factors",
                        key="l1r_slider")
        
        st.markdown("---")
        st.markdown("**Advanced Analysis Options**")
        use_mi = st.checkbox("Enable uncertainty analysis (slower but more robust)", 
                           value=False, 
                           help="Provides confidence intervals but takes longer to compute",
                           key="use_mi")
        m_runs = st.slider("Analysis iterations", 3, 7, 5, 1, 
                         disabled=not use_mi,
                         help="More iterations = more reliable uncertainty estimates",
                         key="mi_draws_slider")
        
        if use_mi:
            st.info("⏱️ Uncertainty analysis enabled - calculations will take longer but provide confidence intervals")

C_user = 10 ** logC

# ──────────────────────────────
# POLICY MODEL
# ──────────────────────────────
@st.cache_resource
def train_policy(X_tr, y_tr, X_va, y_va, C_val, l1_ratio):
    clf = LogisticRegression(
        penalty='elasticnet', l1_ratio=l1_ratio, C=C_val,
        solver='saga', max_iter=5000, random_state=SEED,
        class_weight={0:1.0,1:5.0}
    ).fit(X_tr, y_tr)
    cal = CalibratedClassifierCV(estimator=clf, method='isotonic', cv='prefit')
    cal.fit(X_va, y_va)

    bg = X_tr[: min(500, len(X_tr))]
    expl = shap.LinearExplainer(clf, masker=bg)
    shap_vals = expl.shap_values(X_va)
    mean_shap = np.abs(shap_vals).mean(axis=0)
    nnz = int(np.sum(np.abs(clf.coef_[0]) > 1e-6))
    f1_tr = f1_score(y_tr, clf.predict(X_tr), zero_division=0)
    return {"clf": clf, "cal": cal, "mean_shap": mean_shap, "nnz": nnz, "f1_tr": f1_tr}

policy_info = train_policy(spl["X_tr"], spl["y_tr"], spl["X_va"], spl["y_va"], C_user, l1r)
pol_clf = policy_info["clf"]
pol_cal = policy_info["cal"]
mean_shap = policy_info["mean_shap"]

# ──────────────────────────────
# Optional: MULTIPLE IMPUTATION
# ──────────────────────────────
@st.cache_resource
def fit_policy_with_mi(trn_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: List[str],
                       C_val: float, l1_ratio: float, m: int = 5, seed: int = 42):
    rng = np.random.RandomState(seed)
    models, coefs = [], []
    shap_means = []

    for i in range(m):
        imi = IterativeImputer(random_state=int(rng.randint(0, 1e9)), sample_posterior=True, max_iter=15)
        X_tr_imp = imi.fit_transform(trn_df[feature_cols].values)
        X_val_imp = imi.transform(val_df[feature_cols].values)

        scaler = StandardScaler()
        X_tr_m = scaler.fit_transform(X_tr_imp)
        X_val_m = scaler.transform(X_val_imp)

        clf = LogisticRegression(
            penalty='elasticnet', solver='saga', l1_ratio=l1_ratio,
            C=C_val, max_iter=5000, random_state=int(rng.randint(0, 1e9)),
            class_weight={0:1.0,1:5.0}
        ).fit(X_tr_m, trn_df['conflict_next_year'])

        cal = CalibratedClassifierCV(estimator=clf, method='isotonic', cv='prefit')
        cal.fit(X_val_m, val_df['conflict_next_year'])

        models.append({"scaler": scaler, "clf": clf, "cal": cal, "imputer": imi})
        coefs.append(clf.coef_[0])

        expl = shap.LinearExplainer(clf, masker=X_tr_m[: min(500, len(X_tr_m))])
        shap_vals = expl.shap_values(X_val_m)
        shap_means.append(np.abs(shap_vals).mean(axis=0))

    coefs = np.vstack(coefs)
    shap_means = np.vstack(shap_means)
    coef_mean = coefs.mean(axis=0)
    coef_lo, coef_hi = np.percentile(coefs, [2.5, 97.5], axis=0)
    sign_consistency = np.mean(np.sign(coefs) == np.sign(coef_mean), axis=0)

    shap_mean = shap_means.mean(axis=0)
    shap_lo, shap_hi = np.percentile(shap_means, [2.5, 97.5], axis=0)

    return {
        "models": models,
        "coef_mean": coef_mean, "coef_lo": coef_lo, "coef_hi": coef_hi,
        "sign_consistency": sign_consistency,
        "shap_mean": shap_mean, "shap_lo": shap_lo, "shap_hi": shap_hi
    }

mi_bundle = None
if use_mi:
    mi_bundle = fit_policy_with_mi(
        data_bundle["splits"]["trn_df"]["conflict_next_year"].to_frame().join(data_bundle["splits"]["trn_df"][feature_cols]),
        data_bundle["splits"]["val_df"]["conflict_next_year"].to_frame().join(data_bundle["splits"]["val_df"][feature_cols]),
        feature_cols, C_user, l1r, m=m_runs, seed=SEED
    )

# ──────────────────────────────
# EVALUATE PREDICTIVE MODEL
# ──────────────────────────────
test_probs = pred_model.predict_proba(spl["X_te"])[:,1]
y_pred = (test_probs >= 0.5).astype(int)
prec = precision_score(spl["y_te"], y_pred, zero_division=0)
rec  = recall_score(spl["y_te"], y_pred, zero_division=0)
f1   = f1_score(spl["y_te"], y_pred, zero_division=0)
cm   = confusion_matrix(spl["y_te"], y_pred)

# ──────────────────────────────
# TABS
# ──────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Overview", "🎯 Model Performance", "🔍 Key Risk Factors",
    "🎛️ Explore Conflict's Impact on Development", "🏛️ Policy Simulator"
])

with tab1:
    st.header("📊 Understanding the Data")
    st.markdown("""
    This analysis focuses on **Arab countries** from 2006 onwards, using **45 development indicators** 
    to understand conflict dynamics. The data combines economic, social, environmental, and governance metrics.
    """)
    
    arab_only = arab_df.copy()
    miss_by_row = arab_only[feature_cols].isna().mean(axis=1)
    miss_by_cty = miss_by_row.groupby(arab_only['country_name']).mean().sort_values()
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📈 Data Completeness by Country")
        st.markdown("*Lower percentages indicate better data quality*")
        quality_df = (miss_by_cty*100).round(1).rename("% Missing Data").to_frame()
        quality_df['Data Quality'] = quality_df['% Missing Data'].apply(
            lambda x: '🟢 Excellent' if x < 10 else '🟡 Good' if x < 25 else '🟠 Fair' if x < 40 else '🔴 Limited'
        )
        st.dataframe(quality_df, use_container_width=True)
    
    with col2:
        st.subheader("🔗 How Indicators Relate to Each Other")
        st.markdown("""
        The heatmap below shows correlations between indicators:
        - **Red areas**: Indicators that tend to increase together
        - **Blue areas**: When one increases, the other decreases
        - **White areas**: Little to no relationship
        """)

    corr_matrix = pd.DataFrame(spl["X_tr"], columns=feature_cols).corr()
    corr_matrix.index = [make_readable_name(c) for c in corr_matrix.index]
    corr_matrix.columns = [make_readable_name(c) for c in corr_matrix.columns]
    
    # Select top correlations for display
    top_features = corr_matrix.abs().mean().nlargest(20).index
    corr_subset = corr_matrix.loc[top_features, top_features]
    
    mask = np.triu(np.ones_like(corr_subset, dtype=bool))
    fig_corr, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_subset, mask=mask, cmap="RdBu_r", center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8, "label": "Correlation Strength"}, 
                ax=ax, vmin=-1, vmax=1, annot=False)
    ax.set_title("Relationships Between Top 20 Risk Indicators", fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
    st.pyplot(fig_corr)

with tab2:
    st.header("🎯 How Well Can We Predict Conflicts?")
    st.markdown("""
    Our model analyzes historical patterns to predict whether a country will experience conflict in the next year.
    Here's how accurate our predictions are:
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: 
        st.metric("Precision", f"{prec:.1%}", 
                 help="When we predict conflict, how often are we right?")
    with col2: 
        st.metric("Recall", f"{rec:.1%}",
                 help="Of all actual conflicts, how many do we catch?")
    with col3: 
        st.metric("Overall Accuracy (F1)", f"{f1:.1%}",
                 help="Combined measure of precision and recall")
    with col4: 
        st.metric("Key Factors Used", policy_info["nnz"],
                 help="Number of indicators actively used for predictions")

    st.subheader("📊 Prediction Breakdown")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm,
                    xticklabels=['Predicted: No Conflict', 'Predicted: Conflict'],
                    yticklabels=['Actual: No Conflict', 'Actual: Conflict'],
                    annot_kws={"size": 14})
        ax_cm.set_title("Prediction Accuracy Matrix", fontsize=14, pad=15)
        ax_cm.set_xlabel("Model Predictions", fontsize=12)
        ax_cm.set_ylabel("Actual Outcomes", fontsize=12)
        st.pyplot(fig_cm)
    
    with col2:
        st.markdown("### Understanding the Results")
        st.info("""
        **How to read the matrix:**
        - **Top-left (True Negatives)**: Countries correctly identified as peaceful
        - **Bottom-right (True Positives)**: Countries correctly identified as at-risk
        - **Top-right (False Positives)**: False alarms - predicted conflict but stayed peaceful
        - **Bottom-left (False Negatives)**: Missed warnings - conflicts we didn't predict
        
        **Model Performance:**
        - The model correctly identifies approximately **{:.0f}%** of future conflicts
        - When the model predicts conflict, it's correct about **{:.0f}%** of the time
        """.format(rec*100, prec*100))

with tab3:
    st.header("🔍 Most Important Conflict Risk Factors")
    st.markdown("""
    These are the development indicators that have the strongest relationship with future conflict risk.
    Understanding these factors helps policymakers know where they may want to focus their efforts.
    """)
    
    if mi_bundle is not None:
        st.success("✅ Advanced uncertainty analysis enabled - showing confidence intervals")
        shap_mean = mi_bundle["shap_mean"]; shap_lo = mi_bundle["shap_lo"]; shap_hi = mi_bundle["shap_hi"]
        sign_consistency = mi_bundle["sign_consistency"]
        sorted_idx = np.argsort(shap_mean)[-15:]
        
        fig_bar, ax_bar = plt.subplots(figsize=(10, 9))
        y = range(len(sorted_idx))
        bars = ax_bar.barh(y, shap_mean[sorted_idx],
                    xerr=[shap_mean[sorted_idx]-shap_lo[sorted_idx],
                          shap_hi[sorted_idx]-shap_mean[sorted_idx]],
                    capsize=5, alpha=0.9, color='steelblue')
        ax_bar.set_yticks(y)
        ax_bar.set_yticklabels([make_readable_name(feature_cols[i]) for i in sorted_idx])
        ax_bar.set_xlabel("Importance Score (with 95% confidence intervals)", fontsize=12)
        ax_bar.set_title("Top 15 Factors Linked with Conflict Risk", fontsize=14, fontweight='bold')
        ax_bar.grid(axis='x', alpha=0.3)
        
        # Add importance levels
        for i, (idx, val) in enumerate(zip(sorted_idx, shap_mean[sorted_idx])):
            if val > np.percentile(shap_mean, 90):
                ax_bar.text(val + 0.001, i, ' Critical', va='center', fontsize=9, color='darkred', fontweight='bold')
            elif val > np.percentile(shap_mean, 75):
                ax_bar.text(val + 0.001, i, ' High', va='center', fontsize=9, color='darkorange')
        
        plt.tight_layout(); st.pyplot(fig_bar)

        st.subheader("📊 Factor Reliability Analysis")
        st.markdown("How consistently do these factors predict conflict across different scenarios?")
        
        reliability_df = pd.DataFrame({
            "Risk Factor": [make_readable_name(feature_cols[i]) for i in range(len(feature_cols))],
            "Direction Consistency": sign_consistency,
            "Reliability Rating": pd.cut(sign_consistency, 
                                        bins=[0, 0.7, 0.85, 0.95, 1.0],
                                        labels=['⚠️ Variable', '🟡 Moderate', '🟢 Reliable', '✅ Highly Reliable'])
        }).sort_values("Direction Consistency", ascending=False)
        
        st.dataframe(reliability_df.head(20).reset_index(drop=True), use_container_width=True)
        
    else:
        sorted_idx = np.argsort(mean_shap)[-15:]
        fig_bar, ax_bar = plt.subplots(figsize=(10, 9))
        bars = ax_bar.barh(range(15), mean_shap[sorted_idx], alpha=0.9, color='steelblue')
        ax_bar.set_yticks(range(15))
        ax_bar.set_yticklabels([make_readable_name(feature_cols[i]) for i in sorted_idx])
        ax_bar.set_xlabel("Importance Score", fontsize=12)
        ax_bar.set_title("Top 15 Factors Linked with Conflict Risk", fontsize=14, fontweight='bold')
        ax_bar.grid(axis='x', alpha=0.3)
        
        # Add importance levels
        for i, val in enumerate(mean_shap[sorted_idx]):
            if val > np.percentile(mean_shap, 90):
                ax_bar.text(val + 0.001, i, ' Critical', va='center', fontsize=9, color='darkred', fontweight='bold')
            elif val > np.percentile(mean_shap, 75):
                ax_bar.text(val + 0.001, i, ' High', va='center', fontsize=9, color='darkorange')
        
        plt.tight_layout(); st.pyplot(fig_bar)
        
        with st.expander("💡 Want more detailed analysis?"):
            st.info("Enable 'uncertainty analysis' in the sidebar for confidence intervals and reliability ratings")

with tab4:
    st.header("🎛️ Conflict Impact on Development Simulator")
    st.markdown("""
    **Explore how future conflict risk could affect development indicators** across the Arab region.
    This tool simulates the potential deterioration of economic, social, environmental, and governance 
    indicators based on projected conflict risk levels, helping policymakers understand the cascading 
    effects of conflict on development outcomes.
    """)

    # 1) Conflict risk scenario
    st.subheader("1️⃣ Set Conflict Risk Scenario")
    st.markdown("""
    Assume a future conflict risk level (e.g., from an external early warning system or intelligence assessment).
    This tool will show the expected impact on various development indicators.
    """)
    
    rel_change_odds = st.slider(
        "Projected change in conflict likelihood compared to baseline:", 
        -90, 400, 50, 10, 
        format="%d%%",
        help="How much more (or less) likely is conflict compared to current baseline? Positive values = increased risk",
        key="odds_change_tab4"
    )
    
    delta_logit = np.log1p(rel_change_odds / 100.0)
    
    if rel_change_odds > 0:
        st.warning(f"⚠️ **Scenario:** Conflict risk increases by {rel_change_odds}% above baseline")
        st.markdown("*This simulation shows how development indicators would likely deteriorate under increased conflict risk*")
    elif rel_change_odds < 0:
        st.success(f"✅ **Scenario:** Conflict risk decreases by {abs(rel_change_odds)}% below baseline")
        st.markdown("*This simulation shows how development indicators could improve with reduced conflict risk*")
    else:
        st.info("No change from baseline - adjust the slider above")

    # 2) Scope selection
    st.subheader("2️⃣ Select Geographic Scope")
    col1, col2 = st.columns(2)
    with col1:
        target_year = get_latest_good_year(arab_df, feature_cols, max_missing_pct=0.4)
        st.info(f"Using most recent baseline data: **{target_year}**")

    with col2:
        year_mask = (arab_df['years'] == target_year)
        scope_df = arab_df.loc[year_mask, ['country_name'] + feature_cols].copy()
        options = ["🌍 Arab Region (Average)"] + ["🏳️ " + c for c in sorted(scope_df['country_name'].unique().tolist())]
        scope_choice = st.selectbox("Geographic Focus", options, 
                                   help="Analyze impacts on the entire region or a specific country",
                                   key="scope_choice_tab4")
        scope_choice = scope_choice.replace("🌍 ", "").replace("🏳️ ", "")  # Clean up for processing

    # 3) Calculate impacts on indicators
    X_year_model, X_year_imp = transform_with_preproc(arab_df.loc[year_mask, ['country_code','country_name','years'] + feature_cols], pre)
    X_year_model.index = arab_df.loc[year_mask, :].index
    feat_df = pd.DataFrame(X_year_model, columns=feature_cols)

    Xv = pd.DataFrame(spl["X_va"], columns=feature_cols)
    Xq01 = Xv.quantile(0.01).values
    Xq99 = Xv.quantile(0.99).values

    bg = spl["X_tr"][: min(500, len(spl["X_tr"]))]
    expl = shap.LinearExplainer(pol_clf, masker=bg)
    beta = pol_clf.coef_[0].copy()

    if scope_choice == "Arab Region (Average)":
        X_mat = feat_df.values
        shap_vals = expl.shap_values(X_mat)
        w = np.abs(shap_vals).mean(axis=0)
        x_base = X_mat.mean(axis=0)

        denom = np.sum(w * (beta ** 2))
        if denom <= 0:
            w = np.ones_like(beta); denom = np.sum(beta ** 2) if np.sum(beta ** 2) > 0 else 1e-9
        
        # Note: we're calculating what changes in indicators would be associated with the conflict risk change
        dx = (delta_logit * w * beta) / denom

        room_pos = np.maximum(0.25, Xq99 - x_base)
        room_neg = np.maximum(0.25, x_base - Xq01)
        dx = np.where(dx > 0, np.minimum(dx, room_pos), np.maximum(dx, -room_neg))

        X_mod_all = feat_df.values + dx
        p_base_all = pol_cal.predict_proba(feat_df.values)[:, 1]
        p_mod_all  = pol_cal.predict_proba(X_mod_all)[:, 1]
        delta_pp   = (p_mod_all - p_base_all) * 100
        p_base_avg = p_base_all.mean() * 100
        p_mod_avg  = p_mod_all.mean() * 100

        scope_label = "Arab Region (Average)"
        x_for_table = x_base

    else:
        row_idx = scope_df.index[scope_df['country_name'] == scope_choice][0]
        x_row = feat_df.loc[row_idx].values.reshape(1, -1)
        shap_vals = expl.shap_values(x_row)
        w = np.abs(shap_vals).ravel()
        x_base = x_row.ravel()

        denom = np.sum(w * (beta ** 2))
        if denom <= 0:
            w = np.ones_like(beta); denom = np.sum(beta ** 2) if np.sum(beta ** 2) > 0 else 1e-9
        
        # Note: sign flipped for impact ON indicators
        dx = -(delta_logit * w * beta) / denom

        room_pos = np.maximum(0.25, Xq99 - x_base)
        room_neg = np.maximum(0.25, x_base - Xq01)
        dx = np.where(dx > 0, np.minimum(dx, room_pos), np.maximum(dx, -room_neg))

        x_mod = (x_base + dx).reshape(1, -1)
        p_base = pol_cal.predict_proba(x_row)[:, 1][0] * 100
        p_mod  = pol_cal.predict_proba(x_mod)[:, 1][0] * 100
        delta_pp = np.array([p_mod - p_base])
        p_base_avg, p_mod_avg = p_base, p_mod

        scope_label = scope_choice
        x_for_table = x_base

    # 4) Display results
    st.subheader("3️⃣ Expected Development Impacts")
    
    st.markdown(f"**How {rel_change_odds}% {'increased' if rel_change_odds > 0 else 'decreased'} conflict risk would affect development in {scope_label}:**")
    
    col1, col2, col3 = st.columns(3)
    with col1: 
        # Count severely affected indicators
        severe_impact = np.sum(np.abs(dx) > 1.0)
        st.metric("Severely Affected Indicators", f"{severe_impact}",
                 help="Number of indicators expected to change by >1 standard deviation")
    with col2: 
        # Count moderately affected indicators  
        moderate_impact = np.sum((np.abs(dx) > 0.5) & (np.abs(dx) <= 1.0))
        st.metric("Moderately Affected Indicators", f"{moderate_impact}", 
                 help="Number of indicators expected to change by 0.5-1 standard deviations")
    with col3:
        # Average absolute impact
        avg_impact = np.mean(np.abs(dx))
        st.metric("Average Impact Magnitude", f"{avg_impact:.2f} SD",
                 help="Mean absolute change across all indicators (in standard deviations)")

    # Visualization
    st.subheader("📊 Most Affected Development Indicators")
    topk = 15
    idx_sorted = np.argsort(np.abs(dx))[-topk:]
    vals = dx[idx_sorted]
    ylabels = [make_readable_name(feature_cols[i]) for i in idx_sorted]

    fig_sim, ax_sim = plt.subplots(figsize=(10, 8))
    
    # Color based on whether it's deterioration or improvement
    colors = []
    for i in idx_sorted:
        val = dx[i]
        feat_name = feature_cols[i]
        # For most indicators, negative change = deterioration (e.g., lower GDP, education, health)
        # For some indicators, positive change = deterioration (e.g., mortality, inequality, unemployment)
        deterioration_positive = any(x in feat_name.lower() for x in ['mortality', 'gini', 'unemp', 'food_ins', 'undernour', 'debt', 'disp', 'conflict'])
        
        if deterioration_positive:
            colors.append('#e74c3c' if val > 0 else '#2ecc71')  # Red if increasing (bad), green if decreasing (good)
        else:
            colors.append('#2ecc71' if val > 0 else '#e74c3c')  # Green if increasing (good), red if decreasing (bad)
    
    bars = ax_sim.barh(range(topk), vals, color=colors, alpha=0.85)
    ax_sim.set_yticks(range(topk))
    ax_sim.set_yticklabels(ylabels)
    ax_sim.set_xlabel("Expected Change (in standard deviations)", fontsize=12)
    
    if rel_change_odds > 0:
        ax_sim.set_title(f"Development Indicators Most Affected by {rel_change_odds}% Increased Conflict Risk", 
                         fontsize=14, fontweight='bold')
    else:
        ax_sim.set_title(f"Development Improvements from {abs(rel_change_odds)}% Reduced Conflict Risk", 
                         fontsize=14, fontweight='bold')
    
    ax_sim.axvline(0, color='gray', linestyle='--', alpha=0.6)
    ax_sim.grid(axis='x', alpha=0.3)
    
    # Add impact severity annotations
    for i, val in enumerate(vals):
        if abs(val) > 1.5:
            ax_sim.text(val/2, i, '⚠️ Severe', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        elif abs(val) > 0.75:
            ax_sim.text(val/2, i, 'High', ha='center', va='center', fontsize=9, color='white')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e74c3c', alpha=0.85, label='Deterioration'),
                      Patch(facecolor='#2ecc71', alpha=0.85, label='Improvement')]
    ax_sim.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    st.pyplot(fig_sim)

    # Interpretation guide
    st.info("""
    **📖 How to interpret these results:**
    - **Red bars** indicate deterioration in development outcomes
    - **Green bars** indicate improvement in development outcomes  
    - **Larger bars** mean more severe impacts on that indicator
    - The analysis shows how conflict risk cascades through different development dimensions
    """)

    # Detailed table
    with st.expander("📋 View Detailed Impact Assessment"):
        scaler = pre["scaler"]
        raw_delta = dx * scaler.scale_
        raw_base  = x_for_table * scaler.scale_ + scaler.mean_
        raw_after = raw_base + raw_delta

        # Determine impact direction for each indicator
        impact_directions = []
        for i, feat in enumerate(feature_cols):
            val = dx[i]
            deterioration_positive = any(x in feat.lower() for x in ['mortality', 'gini', 'unemp', 'food_ins', 'undernour', 'debt', 'disp', 'conflict'])
            
            if abs(val) < 0.1:
                impact_directions.append("—")
            elif deterioration_positive:
                impact_directions.append("📉 Worsens" if val > 0 else "📈 Improves")
            else:
                impact_directions.append("📈 Improves" if val > 0 else "📉 Worsens")

        out = pd.DataFrame({
            "Development Indicator": [make_readable_name(feature_cols[i]) for i in range(len(feature_cols))],
            "Impact Magnitude": np.abs(dx),
            "Current Value": raw_base,
            "Expected Change": raw_delta,
            "Projected Value": raw_after,
            "Impact Direction": impact_directions,
            "Severity": pd.cut(np.abs(dx), 
                              bins=[0, 0.25, 0.75, 1.5, float('inf')],
                              labels=['🟢 Minor', '🟡 Moderate', '🟠 High', '🔴 Severe'])
        })
        
        out_sorted = out.sort_values("Impact Magnitude", ascending=False)
        st.dataframe(out_sorted.head(20).reset_index(drop=True).round(2), use_container_width=True)
        
        st.markdown("""
        **Understanding the impacts:**
        - **Severity levels** indicate how much each indicator deviates from normal variation
        - **Impact direction** shows whether the indicator improves or deteriorates
        - These projections assume the conflict risk scenario persists for one year
        - Actual impacts may vary based on conflict intensity and duration
        """)

with tab5:
    st.header("🏛️ Country-Specific Policy Analysis")
    st.markdown("""
    **Test individual policy interventions** to see their impact on conflict risk across different countries.
    This tool helps policymakers understand which interventions work best in which contexts.
    """)
    
    # Policy selection
    st.subheader("1️⃣ Choose a Policy Intervention")
    readable_options = {make_readable_name(feat): feat for feat in feature_cols}
    selected_readable = st.selectbox(
        "Select which development indicator to improve:", 
        list(readable_options.keys()),
        help="Choose a specific area for policy intervention",
        key="policy_area_tab6"
    )
    feat_choice = readable_options[selected_readable]
    
    delta_choice = st.slider(
        "How much can you realistically improve this indicator?", 
        -2.0, 2.0, 1.0, 0.1,
        format="%.1f standard deviations",
        help="Positive = improvement, Negative = deterioration. 1 SD ≈ significant policy effort",
        key="delta_sd_tab6"
    )
    
    if delta_choice > 0:
        st.info(f"📈 Simulating **improvement** in {selected_readable}")
    elif delta_choice < 0:
        st.warning(f"📉 Simulating **deterioration** in {selected_readable}")
    else:
        st.info("No change selected")

    # Year selection
    st.subheader("2️⃣ Analysis Period")
    target_year = get_latest_good_year(arab_df, feature_cols, max_missing_pct=0.4)
    st.info(f"Using most recent feasible baseline year: **{target_year}**")


    year_mask = (arab_df['years'] == target_year)
    meta = arab_df.loc[year_mask, ['country_code','country_name','years']].copy()
    
    if len(meta) == 0:
        st.error("No data available for the selected year")
    else:
        # Calculate impacts
        X_year_model, X_year_imp = transform_with_preproc(arab_df.loc[year_mask], pre)

        feat_min = X_year_model[feat_choice].min()
        feat_max = X_year_model[feat_choice].max()

        X_mod = X_year_model.copy()
        X_mod[feat_choice] = np.clip(X_mod[feat_choice] + delta_choice, feat_min, feat_max)

        p_base = pol_cal.predict_proba(X_year_model)[:,1]
        p_mod  = pol_cal.predict_proba(X_mod)[:,1]
        effects = (p_mod - p_base) * 100

        pre_knn_missing_frac = arab_df.loc[year_mask, feature_cols].isna().mean(axis=1).values
        df_effects = pd.DataFrame({
            "country_name": meta["country_name"].values,
            "Effect (pp)": effects,
            "% imputed": (pre_knn_missing_frac*100)
        })

        # Add regional average
        avg_effect = df_effects["Effect (pp)"].mean()
        region_row = pd.DataFrame({
            "country_name": ["🌍 ARAB REGION AVERAGE"],
            "Effect (pp)": [avg_effect],
            "% imputed": [np.nan],
            "__is_region__": [1],
        })
        df_effects["__is_region__"] = 0
        df_effects = pd.concat([df_effects, region_row], ignore_index=True)
        df_effects = df_effects.sort_values(["__is_region__", "Effect (pp)"], ascending=[True, True]).drop(columns="__is_region__")

        # Results
        st.subheader("3️⃣ Expected Impact by Country")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            countries_benefiting = sum(df_effects["Effect (pp)"] < -0.5)
            st.metric("Countries Benefiting", f"{countries_benefiting}",
                     help="Countries with >0.5pp risk reduction")
        with col2:
            avg_impact = df_effects["Effect (pp)"][:-1].mean()  # Exclude region average
            st.metric("Average Impact", f"{avg_impact:+.2f}pp",
                     help="Mean change in conflict probability")
        with col3:
            max_benefit = df_effects["Effect (pp)"][:-1].min()
            st.metric("Maximum Benefit", f"{max_benefit:.2f}pp",
                     help="Largest risk reduction achieved")

        # Visualization
        action_word = "Improving" if delta_choice > 0 else "Reducing"
        st.markdown(f"### Impact of {action_word} **{selected_readable}** in {target_year}")
        
        fig_h = max(8, 0.4 * len(df_effects) + 2)
        fig_cty, ax_cty = plt.subplots(figsize=(11, fig_h))
        
        # Color based on effect
        colors = []
        for i, row in df_effects.iterrows():
            if row["country_name"] == "🌍 ARAB REGION AVERAGE":
                colors.append('#1b5e20' if row["Effect (pp)"] < 0 else '#b71c1c')
            else:
                if row["Effect (pp)"] < -1:
                    colors.append('#2ecc71')  # Strong benefit
                elif row["Effect (pp)"] < 0:
                    colors.append('#27ae60')  # Moderate benefit
                elif row["Effect (pp)"] > 1:
                    colors.append('#e74c3c')  # Risk increase
                else:
                    colors.append('#95a5a6')  # Minimal effect
        
        bars = ax_cty.barh(df_effects["country_name"], df_effects["Effect (pp)"], color=colors, alpha=0.85)
        ax_cty.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax_cty.set_xlabel("Change in Conflict Risk (percentage points)", fontsize=12)
        ax_cty.set_title(f"Country-by-Country Impact Analysis", fontsize=14, fontweight='bold')
        ax_cty.grid(axis='x', alpha=0.3)
        
        # Add value labels for significant changes
        for bar, val in zip(bars, df_effects["Effect (pp)"]):
            if abs(val) > 1:
                ax_cty.text(val + (0.1 if val > 0 else -0.1), bar.get_y() + bar.get_height()/2,
                          f'{val:+.1f}pp', va='center', ha='left' if val > 0 else 'right',
                          fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig_cty)

        # Detailed results table
        with st.expander("📊 View Detailed Country Results"):
            st.markdown("**Understanding the results:**")
            st.markdown("""
            - **Negative values (green)** = Conflict risk decreases ✅
            - **Positive values (red)** = Conflict risk increases ⚠️
            - **Data quality** affects reliability - countries with less missing data have more reliable estimates
            """)
            
            display_df = df_effects.copy()
            display_df['Impact Assessment'] = display_df['Effect (pp)'].apply(
                lambda x: '✅ Strong Benefit' if x < -1 else 
                         '🟢 Moderate Benefit' if x < -0.5 else
                         '➖ Minimal Effect' if abs(x) < 0.5 else
                         '⚠️ Risk Increase' if x > 0.5 else '🔴 Strong Risk Increase'
            )
            display_df['Data Quality'] = display_df['% imputed'].apply(
                lambda x: '🟢 Excellent' if x < 10 else 
                         '🟡 Good' if x < 25 else 
                         '🟠 Fair' if x < 40 else 
                         '🔴 Limited' if x < 60 else 
                         '—' if pd.isna(x) else '⚫ Poor'
            )
            display_df = display_df[['country_name', 'Effect (pp)', 'Impact Assessment', 'Data Quality']]
            display_df.columns = ['Country', 'Risk Change (pp)', 'Assessment', 'Data Quality']
            st.dataframe(display_df.reset_index(drop=True).round(2), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("### 📝 Technical Notes")
with st.expander("Understanding the Methodology"):
    st.markdown(f"""
    **Model Configuration:**
    - **Prediction Window:** Next year conflict risk
    - **Training Period:** 2006-present
    - **Model Complexity:** {'Simple' if C_user > 0.001 else 'Complex'} (using {policy_info['nnz']} key factors)
    - **Uncertainty Analysis:** {'✅ Enabled' if use_mi else '❌ Disabled'} {f'(running {m_runs} iterations)' if use_mi else ''}
    
    **Data Processing:**
    - Missing data is handled using advanced statistical imputation
    - All indicators are standardized for fair comparison
    - Time-aware validation ensures realistic performance estimates
    
    **Interpretation Guide:**
    - **Risk probabilities** range from 0% (no risk) to 100% (certain conflict)
    - **Percentage points (pp)** measure absolute changes in probability
    - **Standard deviations** measure the size of change relative to typical variation
    
    **Limitations:**
    - Predictions are based on historical patterns and may not capture unprecedented events
    - Data quality varies by country and indicator
    - The model assumes that relationships between factors remain relatively stable
    """)

st.caption("🔒 This tool uses rigorous time-aware validation and calibrated probabilities to ensure reliable risk assessments.")