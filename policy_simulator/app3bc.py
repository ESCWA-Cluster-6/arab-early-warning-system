# app3bb.py  (time-safe, calibrated; sklearn>=1.4 fix for CalibratedClassifierCV)
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

# ─────────────────────────────
# PAGE CONFIG & STYLE
# ─────────────────────────────
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 11})

st.set_page_config(page_title="Conflict Risk Policy Simulator", layout="wide")
PURPLE = "#FBF8FF"
st.markdown(f"""
    <style>
        .stApp, [data-testid="stAppViewContainer"] {{ background: {PURPLE}; }}
        [data-testid="stSidebar"] > div:first-child {{ background: {PURPLE}; }}
        [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
    </style>
""", unsafe_allow_html=True)
plt.rcParams["figure.facecolor"] = "none"
plt.rcParams["axes.facecolor"]   = "none"
plt.rcParams["savefig.facecolor"] = "none"

st.title("🌍 Conflict Risk Policy Simulator (Time-Safe & Calibrated)")
st.caption("Predictive model for accuracy; sparse **policy model** for simulations. Preprocessing is time-safe; imputation and scaling are fit on train only.")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ARAB_COUNTRIES = [
    'Algeria','Bahrain','Comoros','Djibouti','Egypt','Iraq','Jordan',
    'Kuwait','Lebanon','Libya','Mauritania','Morocco','Oman','Palestine',
    'Qatar','Saudi Arabia','Somalia','Sudan','Syrian Arab Republic',
    'Tunisia','United Arab Emirates','Yemen'
]

# ─────────────────────────────
# Helper: readable labels
# ─────────────────────────────
def make_readable_name(feature_name: str) -> str:
    readable_names = {
        '01-fatper_nm': 'Fatalities per 100,000 Population',
        '02-fat_lagged_nm': 'Lagged Fatalities Rate',
        '03-political_stability_nm': 'Political Stability Index',
        '04-conflict proximity_nm': 'Conflict Proximity Index',
        '05-state_authority_nm': 'State Authority Strength',
        '06-disp_rate_nm': 'Displacement Rate',
        '07-voice_accountability_nm': 'Voice and Accountability',
        '08-mil_expd_nm': 'Military Expenditure (% of GDP)',
        '09-armed_forces_nm': 'Armed Forces Personnel (% of Population)',
        '10-agri_value_nm': 'Agricultural Value Added (% of GDP)',
        '11-ren_water_nm': 'Renewable Water Resources (per capita)',
        '12-wather_withd_nm': 'Freshwater Withdrawals (% of Resources)',
        '13-agri_land_nm': 'Agricultural Land (% of Land Area)',
        '14-pop_dis_nm': 'Population Displacement',
        '15-dis_dicp_nm': 'Disaster Displacement',
        '16-adap_strat_nm': 'Adaptation Strategies Index',
        '17-cli_fin_nm': 'Climate Finance Received',
        '18-remit_nm': 'Remittance Inflows (% of GDP)',
        '19-oda_nm': 'Official Development Assistance Received (% of GNI)',
        '20-cer_imp_nm': 'Cereal Import Dependency Ratio',
        '21-food_ins_nm': 'Food Insecurity Prevalence',
        '22-undernour_nm': 'Undernourishment Prevalence',
        '23-food_imp_nm': 'Food Import Dependency',
        '24-gini_nm': 'Gini Coefficient',
        '25-topbottom_ratio_nm': 'Income Quintile Ratio (Top 20% / Bottom 20%)',
        '26-gdppc_nm': 'GDP per Capita',
        '27-gdpg_nm': 'GDP Growth Rate',
        '28-govt_debt_nm': 'Government Debt (% of GDP)',
        '29-int_use_nm': 'Individuals Using the Internet (% of Population)',
        '30-unemp_nm': 'Unemployment Rate',
        '31-youth_nm': 'Youth Population Share',
        '32-mr5_nm': 'Under-5 Mortality Rate',
        '33-mmr_nm': 'Maternal Mortality Ratio',
        '34-exp_sch_nm': 'Expected Years of Schooling',
        '35-mean_sch_nm': 'Average Years of Education',
        '36-soc_pro_nm': 'Social Protection Coverage',
        '37-water_serv_nm': 'Access to Basic Water Services',
        '38-san_serv_nm': 'Access to Sanitation Services',
        '39-uhc_nm': 'Universal Health Coverage Index',
        '40-gpi_nm': 'Global Peace Index',
        '41-lfp_fem_nm': 'Female Labor Force Participation Rate',
        '42-control_corruption_nm': 'Control of Corruption',
        '43-rule_law_nm': 'Rule of Law',
        '44-government_eff_nm': 'Government Effectiveness',
        '45-osi_nm': 'Open Society Index'
    }
    return readable_names.get(feature_name, feature_name.replace('_nm','').replace('_',' ').title())

# ─────────────────────────────
# LOAD RAW
# ─────────────────────────────
@st.cache_data
def load_raw() -> pd.DataFrame:
    df_ind = pd.read_excel("data/data_risk_indicators.xlsx")
    df_con = pd.read_csv("data/Conflict_Status_by_Year.csv")

    df_con = df_con.melt(id_vars=['Unnamed: 0'], var_name='years', value_name='conflict')
    df_con.rename(columns={'Unnamed: 0': 'country_name'}, inplace=True)
    df_con['years'] = df_con['years'].astype(int)

    df_ind['years'] = df_ind['years'].astype(int)
    df = df_ind.merge(df_con, on=['country_name','years'], how='left')
    df['conflict'] = df['conflict'].fillna(0).astype(int)

    df = df[df['years'] >= 2006].copy()
    return df

raw_df = load_raw()
meta_cols = ['country_code','country_name','years','conflict']
candidate_cols = [c for c in raw_df.columns if c not in meta_cols]

# ─────────────────────────────
# TIME-SAFE PANEL PREP
# ─────────────────────────────
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

data_bundle = build_time_safe_datasets(missing_thresh=0.25)
feature_cols = data_bundle["keep_cols"]
pre = data_bundle["pre"]
spl = data_bundle["splits"]
arab_df = data_bundle["arab_df"]

# ─────────────────────────────
# TRAIN PREDICTIVE MODEL (Optuna on TRN; calibrate on VAL)
# ─────────────────────────────
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
    # sklearn>=1.4: use estimator=
    cal = CalibratedClassifierCV(estimator=base, method='isotonic', cv='prefit')
    cal.fit(X_va, y_va)
    return cal, best

pred_model, pred_params = train_predictive(spl["X_tr"], spl["y_tr"], spl["X_va"], spl["y_va"])

# ─────────────────────────────
# SIDEBAR — Policy regularization & MI toggle
# ─────────────────────────────
with st.sidebar:
    st.subheader("Policy Model Settings")
    logC = st.slider("log10(C)", -4.0, -1.0, -3.0, 0.5, key="logC_slider")
    l1r  = st.slider("L1 ratio", 0.70, 1.00, 0.90, 0.05, key="l1r_slider")
    use_mi = st.checkbox("Enable Multiple Imputation for interpretability (slow)", value=False, key="use_mi")
    m_runs = st.slider("MI draws (m)", 3, 7, 5, 1, disabled=not use_mi, key="mi_draws_slider")
C_user = 10 ** logC

# ─────────────────────────────
# POLICY MODEL (train on TRN; calibrate on VAL)
# ─────────────────────────────
@st.cache_resource
def train_policy(X_tr, y_tr, X_va, y_va, C_val, l1_ratio):
    clf = LogisticRegression(
        penalty='elasticnet', l1_ratio=l1_ratio, C=C_val,
        solver='saga', max_iter=5000, random_state=SEED,
        class_weight={0:1.0,1:5.0}
    ).fit(X_tr, y_tr)
    # sklearn>=1.4: use estimator=
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

# ─────────────────────────────
# Optional: MULTIPLE IMPUTATION bundle (for Tab 3)
# ─────────────────────────────
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

        # sklearn>=1.4: use estimator=
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

# ─────────────────────────────
# EVALUATE PREDICTIVE MODEL on TEST (time-aware)
# ─────────────────────────────
test_probs = pred_model.predict_proba(spl["X_te"])[:,1]
y_pred = (test_probs >= 0.5).astype(int)
prec = precision_score(spl["y_te"], y_pred, zero_division=0)
rec  = recall_score(spl["y_te"], y_pred, zero_division=0)
f1   = f1_score(spl["y_te"], y_pred, zero_division=0)
cm   = confusion_matrix(spl["y_te"], y_pred)

# ─────────────────────────────
# TABS
# ─────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Overview", "🎯 Model Accuracy", "🔍 Key Risk Factors",
    "🎛️ Development Impact Calculator", "🏛️ Country-Specific Intervention"
])

with tab1:
    st.header("📊 Understanding the Data: Arab Countries Focus")
    st.caption("Features retained (train-only threshold, forward-fill only). Correlations shown after model standardization.")
    arab_only = arab_df.copy()
    miss_by_row = arab_only[feature_cols].isna().mean(axis=1)
    miss_by_cty = miss_by_row.groupby(arab_only['country_name']).mean().sort_values()
    st.write("**Average % of indicators imputed (post-ffill, pre-KNN) by country**")
    st.dataframe((miss_by_cty*100).round(1).rename("% imputed (avg)").to_frame())

    corr_matrix = pd.DataFrame(spl["X_tr"], columns=feature_cols).corr()
    corr_matrix.index = [make_readable_name(c) for c in corr_matrix.index]
    corr_matrix.columns = [make_readable_name(c) for c in corr_matrix.columns]
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    fig_corr, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm", center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Indicator Correlations (standardized modeling space)", fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
    st.pyplot(fig_corr)

with tab2:
    st.header("🎯 Predictive Model — Time-Aware Test Performance")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Precision", f"{prec:.1%}")
    with c2: st.metric("Recall", f"{rec:.1%}")
    with c3: st.metric("F1-Score", f"{f1:.1%}")
    with c4: st.metric("Policy Non-zeros", policy_info["nnz"])

    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm,
                xticklabels=['No Conflict Pred', 'Conflict Pred'],
                yticklabels=['Actual No Conflict', 'Actual Conflict'])
    ax_cm.set_title("Predictive Model: Confusion Matrix (Test)", fontsize=14, pad=15)
    st.pyplot(fig_cm)

with tab3:
    st.header("🔍 Policy-Relevant Factors")
    if mi_bundle is not None:
        shap_mean = mi_bundle["shap_mean"]; shap_lo = mi_bundle["shap_lo"]; shap_hi = mi_bundle["shap_hi"]
        sign_consistency = mi_bundle["sign_consistency"]
        sorted_idx = np.argsort(shap_mean)[-12:]
        fig_bar, ax_bar = plt.subplots(figsize=(10, 8))
        y = range(len(sorted_idx))
        ax_bar.barh(y, shap_mean[sorted_idx],
                    xerr=[shap_mean[sorted_idx]-shap_lo[sorted_idx],
                          shap_hi[sorted_idx]-shap_mean[sorted_idx]],
                    capsize=5, alpha=0.9)
        ax_bar.set_yticks(y)
        ax_bar.set_yticklabels([make_readable_name(feature_cols[i]) for i in sorted_idx])
        ax_bar.set_xlabel("Importance (mean |SHAP|) — MI ensemble with 95% bands")
        ax_bar.set_title("Top Policy-Relevant Predictors (Multiple Imputation)")
        ax_bar.grid(axis='x', alpha=0.3); plt.tight_layout(); st.pyplot(fig_bar)

        out = pd.DataFrame({
            "Feature": [make_readable_name(feature_cols[i]) for i in range(len(feature_cols))],
            "Sign consistency": sign_consistency
        }).sort_values("Sign consistency", ascending=False)
        st.dataframe(out.reset_index(drop=True))
    else:
        sorted_idx = np.argsort(mean_shap)[-12:]
        fig_bar, ax_bar = plt.subplots(figsize=(10, 8))
        ax_bar.barh(range(12), mean_shap[sorted_idx], alpha=0.9)
        ax_bar.set_yticks(range(12))
        ax_bar.set_yticklabels([make_readable_name(feature_cols[i]) for i in sorted_idx])
        ax_bar.set_xlabel("Importance (mean |SHAP|)")
        ax_bar.set_title("Top Policy-Relevant Predictors (single imputation)")
        ax_bar.grid(axis='x', alpha=0.3); plt.tight_layout(); st.pyplot(fig_bar)

with tab4:
    st.header("🎛️ Development Impact Calculator — Country / Region Specific")

    # 1) Desired odds change -> delta in log-odds
    rel_change_odds = st.slider("Target change in conflict risk (odds)", -90, 400, 50, 10, format="%d%%", key="odds_change_tab4")
    delta_logit = np.log1p(rel_change_odds / 100.0)
    if rel_change_odds < 0:
        st.success(f"📉 Targeting {abs(rel_change_odds)}% lower odds")
    elif rel_change_odds > 0:
        st.warning(f"📈 Simulating {rel_change_odds}% higher odds")
    else:
        st.info("No change selected.")

    # 2) Choose year and scope (country or region avg)
    year_min, year_max = int(arab_df['years'].min()), int(arab_df['years'].max())
    target_year = st.slider("📅 Year for Analysis", year_min, year_max, max(year_min, year_max - 1), key="year_calc_tab4")
    year_mask = (arab_df['years'] == target_year)
    scope_df = arab_df.loc[year_mask, ['country_name'] + feature_cols].copy()
    options = ["Arab Region (Average)"] + sorted(scope_df['country_name'].unique().tolist())
    scope_choice = st.selectbox("🌍 Scope", options, key="scope_choice_tab4")

    # 3) Get standardized modeling matrix for the selected year, plus helper stats
    X_year_model, X_year_imp = transform_with_preproc(arab_df.loc[year_mask, ['country_code','country_name','years'] + feature_cols], pre)
    X_year_model.index = arab_df.loc[year_mask, :].index  # align indices
    feat_df = pd.DataFrame(X_year_model, columns=feature_cols)

    # Bounds based on validation distribution (keeps us in-distribution)
    Xv = pd.DataFrame(spl["X_va"], columns=feature_cols)
    Xq01 = Xv.quantile(0.01).values
    Xq99 = Xv.quantile(0.99).values

    # 4) Local SHAP weights at the chosen baseline
    bg = spl["X_tr"][: min(500, len(spl["X_tr"]))]
    expl = shap.LinearExplainer(pol_clf, masker=bg)

    beta = pol_clf.coef_[0].copy()

    if scope_choice == "Arab Region (Average)":
        # Use region-average baseline vector and average |SHAP| as weights
        X_mat = feat_df.values
        shap_vals = expl.shap_values(X_mat)         # shape: [n, F]
        w = np.abs(shap_vals).mean(axis=0)          # mean |SHAP| per feature
        x_base = X_mat.mean(axis=0)                 # region-average point (in model space)

        # Allocate dx to achieve desired delta on the logit: minimize weighted move subject to beta·dx = delta
        denom = np.sum(w * (beta ** 2))
        if denom <= 0:  # fallback if degenerate
            w = np.ones_like(beta); denom = np.sum(beta ** 2) if np.sum(beta ** 2) > 0 else 1e-9
        dx = (delta_logit * w * beta) / denom

        # Clip moves to remain in-distribution (and avoid huge jumps). Also require at least ±0.25 SD slack.
        room_pos = np.maximum(0.25, Xq99 - x_base)
        room_neg = np.maximum(0.25, x_base - Xq01)
        dx = np.where(dx > 0, np.minimum(dx, room_pos), np.maximum(dx, -room_neg))

        # Apply to all countries this year to compute average risk change
        X_mod_all = feat_df.values + dx
        p_base_all = pol_cal.predict_proba(feat_df.values)[:, 1]
        p_mod_all  = pol_cal.predict_proba(X_mod_all)[:, 1]
        delta_pp   = (p_mod_all - p_base_all) * 100
        p_base_avg = p_base_all.mean() * 100
        p_mod_avg  = p_mod_all.mean() * 100

        scope_label = "Arab Region (Average)"
        x_for_table = x_base  # for raw value conversion below

    else:
        # Single country vector & local |SHAP| weights
        row_idx = scope_df.index[scope_df['country_name'] == scope_choice][0]
        x_row = feat_df.loc[row_idx].values.reshape(1, -1)
        shap_vals = expl.shap_values(x_row)         # shape: [1, F]
        w = np.abs(shap_vals).ravel()
        x_base = x_row.ravel()

        denom = np.sum(w * (beta ** 2))
        if denom <= 0:
            w = np.ones_like(beta); denom = np.sum(beta ** 2) if np.sum(beta ** 2) > 0 else 1e-9
        dx = (delta_logit * w * beta) / denom

        room_pos = np.maximum(0.25, Xq99 - x_base)
        room_neg = np.maximum(0.25, x_base - Xq01)
        dx = np.where(dx > 0, np.minimum(dx, room_pos), np.maximum(dx, -room_neg))

        x_mod = (x_base + dx).reshape(1, -1)
        p_base = pol_cal.predict_proba(x_row)[:, 1][0] * 100
        p_mod  = pol_cal.predict_proba(x_mod)[:, 1][0] * 100
        delta_pp = np.array([p_mod - p_base])  # for consistent handling
        p_base_avg, p_mod_avg = p_base, p_mod

        scope_label = scope_choice
        x_for_table = x_base

    # 5) Display results
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Baseline risk", f"{p_base_avg:.1f}%")
    with c2: st.metric("Post-change risk", f"{p_mod_avg:.1f}%", delta=f"{(p_mod_avg - p_base_avg):+.2f} pp")
    with c3: st.metric("Target odds change", f"{rel_change_odds:+d}%")

    # Sort features by size of SD move
    topk = 12
    idx_sorted = np.argsort(np.abs(dx))[-topk:]
    vals = dx[idx_sorted]
    ylabels = [make_readable_name(feature_cols[i]) for i in idx_sorted]

    fig_sim, ax_sim = plt.subplots(figsize=(10, 7))
    colors = ['#2e7d32' if v > 0 else '#c62828' for v in vals]
    ax_sim.barh(range(topk), vals, color=colors, alpha=0.92)
    ax_sim.set_yticks(range(topk)); ax_sim.set_yticklabels(ylabels)
    ax_sim.set_xlabel("Required Indicator Change (standard deviations)")
    ax_sim.set_title(f"{scope_label}: Feature changes associated with {rel_change_odds:+d}% odds change")
    ax_sim.axvline(0, color='gray', linestyle='--', alpha=0.6); ax_sim.grid(axis='x', alpha=0.3)
    plt.tight_layout(); st.pyplot(fig_sim)

    # 6) Convert SD deltas back to RAW units for readability
    scaler = pre["scaler"]
    raw_delta = dx * scaler.scale_
    raw_base  = x_for_table * scaler.scale_ + scaler.mean_
    raw_after = raw_base + raw_delta

    out = pd.DataFrame({
        "Indicator": [make_readable_name(feature_cols[i]) for i in range(len(feature_cols))],
        "Δ (SD)": dx,
        "Baseline (raw)": raw_base,
        "Change (raw)": raw_delta,
        "Post-change (raw)": raw_after
    })
    out_top = out.iloc[idx_sorted].sort_values("Δ (SD)", key=lambda s: np.abs(s), ascending=False)
    st.caption("Top moved indicators (converted back to raw units):")
    st.dataframe(out_top.reset_index(drop=True).round(3))

with tab5:
    st.header("🏛️ Country-Specific Policy Analysis (Calibrated • Time-Safe)")
    readable_options = {make_readable_name(feat): feat for feat in feature_cols}
    selected_readable = st.selectbox("📊 Select Policy Area to Change", list(readable_options.keys()), key="policy_area_tab6")
    feat_choice = readable_options[selected_readable]
    delta_choice = st.slider("📈 How much to change it? (SD)", -2.0, 2.0, 1.0, 0.1, format="%.1f SD", key="delta_sd_tab6")

    year_min, year_max = int(arab_df['years'].min()), int(arab_df['years'].max())
    target_year = st.slider("📅 Year for Analysis", year_min, year_max, max(year_min, year_max-1), key="year_policy_tab6")

    year_mask = (arab_df['years'] == target_year)
    meta = arab_df.loc[year_mask, ['country_code','country_name','years']].copy()
    if len(meta) == 0:
        st.info("No Arab countries have data in the selected year.")
    else:
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
            "% imputed (pre-KNN)": (pre_knn_missing_frac*100)
        })

        # Add Arab Region (Avg) bar and pin it to the bottom
        avg_effect = df_effects["Effect (pp)"].mean()
        region_row = pd.DataFrame({
            "country_name": ["Arab Region (Avg)"],
            "Effect (pp)": [avg_effect],
            "% imputed (pre-KNN)": [np.nan],
            "__is_region__": [1],
        })
        df_effects["__is_region__"] = 0
        df_effects = pd.concat([df_effects, region_row], ignore_index=True)
        df_effects = df_effects.sort_values(["__is_region__", "Effect (pp)"], ascending=[True, True]).drop(columns="__is_region__")

        st.markdown(f"### 🎯 Impact of {'Increasing' if delta_choice>0 else 'Decreasing'} {selected_readable} in {target_year}")
        fig_h = max(6, 0.35 * len(df_effects) + 2)
        fig_cty, ax_cty = plt.subplots(figsize=(10, fig_h))
        colors = ['#2ecc71' if x < 0 else '#e74c3c' for x in df_effects["Effect (pp)"]]
        # Dark green/red for the region-average bar (last row)
        colors[-1] = '#1b5e20' if df_effects["Effect (pp)"].iloc[-1] < 0 else '#b71c1c'
        ax_cty.barh(df_effects["country_name"], df_effects["Effect (pp)"], color=colors, alpha=0.90)
        ax_cty.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax_cty.set_xlabel("Change in Conflict Risk (percentage points)")
        ax_cty.grid(axis='x', alpha=0.3)
        plt.tight_layout(); st.pyplot(fig_cty)

        st.caption("Data quality per country (lower % is better):")
        st.dataframe(df_effects.reset_index(drop=True).round(2))

st.markdown("---")
st.caption(
    f"Time-safe prep: forward-fill only; KNN impute in standardized space; scaler & imputer fit on TRAIN only; "
    f"validation = last train year per country; isotonic calibration on validation. "
    f"Policy knobs: C={C_user:.4f}. "
    f"{'MI enabled' if use_mi else 'Single imputation (fast)'} with m={m_runs if use_mi else 0}."
)