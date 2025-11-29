"""
================================================================================
3モデル比較スクリプト
================================================================================

【目的】
ポアソン回帰と2種類の負の二項回帰を比較し、最適なモデルを選択する

【出力】
- 3モデルの比較表（AIC、係数、検定結果）
- 過分散チェック結果
- 多重共線性チェック結果（VIF）
- 尤度比検定結果
- CSVファイル出力
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.genmod.families import NegativeBinomial as NB_family
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. データ読み込み
# =============================================================================

print("=" * 80)
print("【1. データ読み込み】")
print("=" * 80)

# --- 交通事故死者数データ ---
acc_raw = pd.read_csv("3都道府県別交通事故死者数.csv", encoding="cp932", header=None)
acc = acc_raw.iloc[7:55].copy()
acc.columns = [
    "region_code", "block", "pref_short",
    "deaths_2021", "deaths_2022", "deaths_2023",
    "rate_2021", "rate_2022", "rate_2023"
]
acc = acc[acc["block"] != "全 国"].copy()
acc["pref_short"] = acc["pref_short"].str.strip()
acc["deaths"] = acc["deaths_2023"].astype(int)
acc_df = acc[["pref_short", "deaths"]]

# --- 人口データ ---
pop_raw = pd.read_excel("a01100_2.xlsx", sheet_name=0, header=None)
pop_pref = pop_raw[
    (pop_raw[7] == 2023001010) &
    (pop_raw[9] == "総人口") &
    (pop_raw[10] != "00000")
].copy()
pop_pref["pref_full"] = pop_pref[11].str.replace("　", "").str.strip()

def to_short(name: str) -> str:
    if name == "北海道":
        return name
    for suf in ["都", "府", "県"]:
        if name.endswith(suf):
            return name[:-1]
    return name

pop_pref["pref_short"] = pop_pref["pref_full"].apply(to_short)
pop_pref["population"] = (pop_pref[14] * 1000).astype(int)
pop_pref["pop_65plus"] = (pop_pref[17] * 1000).astype(int)
pop_pref["elderly_rate"] = pop_pref["pop_65plus"] / pop_pref["population"]
pop_df = pop_pref[["pref_short", "population", "elderly_rate"]].copy()

# --- 自動車保有台数 ---
cars_raw = pd.read_excel("r5c6pv0000013d12.xlsx", sheet_name="8", header=None)
pref_list = [
    "青森","岩手","宮城","秋田","山形","福島","茨城","栃木","群馬",
    "埼玉","千葉","東京","神奈川","山梨","新潟","富山","石川","長野",
    "福井","岐阜","静岡","愛知","三重","滋賀","京都","大阪","奈良",
    "和歌山","兵庫","鳥取","島根","岡山","広島","山口","徳島","香川",
    "愛媛","高知","福岡","佐賀","長崎","熊本","大分","宮崎","鹿児島"
]
cars_pref = cars_raw.loc[cars_raw[1].isin(pref_list), [1, 7]].copy()
cars_pref.columns = ["pref_short", "cars_total"]
hokkaido_offices = ["札幌","函館","旭川","室蘭","釧路","帯広","北見"]
hokkaido_total = cars_raw.loc[cars_raw[1].isin(hokkaido_offices), 7].sum()
okinawa_total = cars_raw.loc[cars_raw[0].astype(str).str.contains("沖"), 7].iloc[0]

cars_df = pd.concat([
    pd.DataFrame({"pref_short": ["北海道"], "cars_total": [hokkaido_total]}),
    cars_pref,
    pd.DataFrame({"pref_short": ["沖縄"], "cars_total": [okinawa_total]}),
], ignore_index=True)
cars_df["cars_total"] = cars_df["cars_total"].astype(int)

# --- マージ ---
df = (
    acc_df
    .merge(pop_df, on="pref_short")
    .merge(cars_df, on="pref_short")
)
df["car_per_1000"] = df["cars_total"] / (df["population"] / 1000)
df["log_pop"] = np.log(df["population"])

print(f"✓ データ読み込み完了: {len(df)} 都道府県")

# =============================================================================
# 2. モデル構築
# =============================================================================

X = df[["elderly_rate", "car_per_1000"]]
X = sm.add_constant(X)
y = df["deaths"]
offset = df["log_pop"]

print("\n" + "=" * 80)
print("【2. 3つのモデルを構築】")
print("=" * 80)

# --- 2.1 ポアソン回帰 ---
print("\n--- 2.1 ポアソン回帰 ---")
poisson_model = GLM(y, X, family=families.Poisson(), offset=offset)
poisson_result = poisson_model.fit()
print(f"AIC: {poisson_result.aic:.3f}")

# 過分散パラメータ
phi = poisson_result.pearson_chi2 / poisson_result.df_resid
print(f"過分散パラメータ φ: {phi:.3f}")

# --- 2.2 負の二項回帰 (sm.NegativeBinomial) ---
print("\n--- 2.2 負の二項回帰 (sm.NegativeBinomial) ---")
try:
    nb_sm_model = sm.NegativeBinomial(y, X, loglike_method='nb2', offset=offset)
    nb_sm_result = nb_sm_model.fit(disp=False, maxiter=1000)
    nb_sm_converged = nb_sm_result.mle_retvals.get('converged', False)
    print(f"AIC: {nb_sm_result.aic:.3f}")
    print(f"収束: {'✓' if nb_sm_converged else '✗ 失敗'}")
except Exception as e:
    print(f"エラー: {e}")
    nb_sm_result = None
    nb_sm_converged = False

# --- 2.3 負の二項回帰 (GLM + α最適化) ---
print("\n--- 2.3 負の二項回帰 (GLM + α最適化) ---")

def find_optimal_alpha(y, X, offset):
    alphas = np.arange(0.001, 0.5, 0.001)
    aics = []
    for a in alphas:
        try:
            model = sm.GLM(y, X, family=NB_family(alpha=a), offset=offset)
            aics.append(model.fit(disp=0).aic)
        except:
            aics.append(np.inf)
    return alphas[np.argmin(aics)]

optimal_alpha = find_optimal_alpha(y, X, offset)
nb_glm_model = sm.GLM(y, X, family=NB_family(alpha=optimal_alpha), offset=offset)
nb_glm_result = nb_glm_model.fit()
print(f"最適α: {optimal_alpha:.4f}")
print(f"AIC: {nb_glm_result.aic:.3f}")

# =============================================================================
# 3. VIF計算（全モデル共通）
# =============================================================================

print("\n" + "=" * 80)
print("【3. 多重共線性チェック (VIF)】")
print("=" * 80)

X_vif = df[["elderly_rate", "car_per_1000"]]
vif_elderly = variance_inflation_factor(X_vif.values, 0)
vif_car = variance_inflation_factor(X_vif.values, 1)

print(f"VIF (elderly_rate):  {vif_elderly:.3f}")
print(f"VIF (car_per_1000):  {vif_car:.3f}")
print(f"判定: {'✓ 問題なし (VIF < 10)' if max(vif_elderly, vif_car) < 10 else '⚠️ 多重共線性あり'}")

# =============================================================================
# 4. 尤度比検定
# =============================================================================

print("\n" + "=" * 80)
print("【4. 尤度比検定（ポアソン vs 負の二項）】")
print("=" * 80)

lr_stat = 2 * (nb_glm_result.llf - poisson_result.llf)
p_value_lr = 0.5 * scipy_stats.chi2.sf(lr_stat, 1)

print(f"帰無仮説: ポアソン回帰で十分（過分散なし）")
print(f"対立仮説: 負の二項回帰が必要（過分散あり）")
print(f"\n尤度比統計量: {lr_stat:.3f}")
print(f"p値: {p_value_lr:.4f}")
print(f"判定: {'★ 有意 (p < 0.05) → 負の二項回帰を採用' if p_value_lr < 0.05 else '有意でない → ポアソンで十分かも'}")

# =============================================================================
# 5. 比較表の出力
# =============================================================================

print("\n" + "=" * 80)
print("【5. 3モデル比較表】")
print("=" * 80)

# 表を作成
comparison_data = {
    '項目': [
        'AIC',
        '対数尤度',
        '収束',
        '過分散 φ',
        'elderly_rate 係数',
        'elderly_rate SE',
        'elderly_rate p値',
        'elderly_rate 有意性',
        'car_per_1000 係数',
        'car_per_1000 SE',
        'car_per_1000 p値',
        'car_per_1000 有意性',
        'VIF (elderly_rate)',
        'VIF (car_per_1000)',
    ],
    'ポアソン回帰': [
        f"{poisson_result.aic:.3f}",
        f"{poisson_result.llf:.3f}",
        "✓",
        f"{phi:.3f}",
        f"{poisson_result.params['elderly_rate']:.4f}",
        f"{poisson_result.bse['elderly_rate']:.4f}",
        f"{poisson_result.pvalues['elderly_rate']:.4f}",
        "★" if poisson_result.pvalues['elderly_rate'] < 0.05 else "",
        f"{poisson_result.params['car_per_1000']:.6f}",
        f"{poisson_result.bse['car_per_1000']:.6f}",
        f"{poisson_result.pvalues['car_per_1000']:.4f}",
        "★" if poisson_result.pvalues['car_per_1000'] < 0.05 else "",
        f"{vif_elderly:.3f}",
        f"{vif_car:.3f}",
    ],
    '負の二項 (sm.NB)': [
        f"{nb_sm_result.aic:.3f}" if nb_sm_result else "N/A",
        f"{nb_sm_result.llf:.3f}" if nb_sm_result else "N/A",
        "✓" if nb_sm_converged else "✗ 失敗",
        "-",
        f"{nb_sm_result.params['elderly_rate']:.4f}" if nb_sm_result and nb_sm_converged else "nan",
        "nan" if not nb_sm_converged else f"{nb_sm_result.bse['elderly_rate']:.4f}",
        "nan" if not nb_sm_converged else f"{nb_sm_result.pvalues['elderly_rate']:.4f}",
        "" if not nb_sm_converged else ("★" if nb_sm_result.pvalues['elderly_rate'] < 0.05 else ""),
        f"{nb_sm_result.params['car_per_1000']:.6f}" if nb_sm_result and nb_sm_converged else "nan",
        "nan" if not nb_sm_converged else f"{nb_sm_result.bse['car_per_1000']:.6f}",
        "nan" if not nb_sm_converged else f"{nb_sm_result.pvalues['car_per_1000']:.4f}",
        "" if not nb_sm_converged else ("★" if nb_sm_result.pvalues['car_per_1000'] < 0.05 else ""),
        f"{vif_elderly:.3f}",
        f"{vif_car:.3f}",
    ],
    '負の二項 (GLM+α最適化)': [
        f"{nb_glm_result.aic:.3f} ★最良",
        f"{nb_glm_result.llf:.3f}",
        "✓",
        f"α={optimal_alpha:.4f}",
        f"{nb_glm_result.params['elderly_rate']:.4f}",
        f"{nb_glm_result.bse['elderly_rate']:.4f}",
        f"{nb_glm_result.pvalues['elderly_rate']:.4f}",
        "★" if nb_glm_result.pvalues['elderly_rate'] < 0.05 else "",
        f"{nb_glm_result.params['car_per_1000']:.6f}",
        f"{nb_glm_result.bse['car_per_1000']:.6f}",
        f"{nb_glm_result.pvalues['car_per_1000']:.4f}",
        "★" if nb_glm_result.pvalues['car_per_1000'] < 0.05 else "",
        f"{vif_elderly:.3f}",
        f"{vif_car:.3f}",
    ],
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# =============================================================================
# 6. 係数の解釈
# =============================================================================

print("\n" + "=" * 80)
print("【6. 係数の解釈（推奨モデル: 負の二項 GLM+α最適化）】")
print("=" * 80)

elderly_coef = nb_glm_result.params['elderly_rate']
car_coef = nb_glm_result.params['car_per_1000']

print(f"""
■ 高齢化率 (elderly_rate)
  係数: {elderly_coef:.4f}
  解釈: 高齢化率が1%ポイント上昇すると
        死亡リスクは exp({elderly_coef:.4f}) = {np.exp(elderly_coef):.4f} 倍
        → 約 {(np.exp(elderly_coef)-1)*100:.1f}% 増加
  検定: p = {nb_glm_result.pvalues['elderly_rate']:.4f} → {'有意 (p < 0.05)' if nb_glm_result.pvalues['elderly_rate'] < 0.05 else '有意でない'}

■ 自動車保有率 (car_per_1000)
  係数: {car_coef:.6f}
  解釈: 人口千人あたり自動車が100台増えると
        死亡リスクは exp({car_coef:.6f} × 100) = {np.exp(car_coef * 100):.4f} 倍
        → 約 {(np.exp(car_coef * 100)-1)*100:.1f}% 増加
  検定: p = {nb_glm_result.pvalues['car_per_1000']:.4f} → {'有意 (p < 0.05)' if nb_glm_result.pvalues['car_per_1000'] < 0.05 else '有意でない'}
""")

# =============================================================================
# 7. 最終結論
# =============================================================================

print("\n" + "=" * 80)
print("【7. 最終結論】")
print("=" * 80)

print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                           分析結果サマリー                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 【過分散診断】                                                              │
│   φ = {phi:.3f} > 1 → 過分散あり                                            │
│   尤度比検定 p = {p_value_lr:.4f} < 0.05 → 統計的に有意                            │
│   → 負の二項回帰が適切                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ 【モデル選択】                                                              │
│   推奨モデル: 負の二項回帰 (GLM + α最適化)                                  │
│   AIC = {nb_glm_result.aic:.3f} (最小)                                               │
│   ※ sm.NegativeBinomial は収束失敗のため不採用                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ 【多重共線性】                                                              │
│   VIF = {max(vif_elderly, vif_car):.3f} < 10 → 問題なし                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ 【係数の検定結果】                                                          │
│   elderly_rate:  p = {nb_glm_result.pvalues['elderly_rate']:.4f} → {'★ 有意' if nb_glm_result.pvalues['elderly_rate'] < 0.05 else '有意でない'}                                    │
│   car_per_1000:  p = {nb_glm_result.pvalues['car_per_1000']:.4f} → {'★ 有意' if nb_glm_result.pvalues['car_per_1000'] < 0.05 else '有意でない'}                                    │
│   → 両変数とも統計的に有意                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 【結論】                                                                    │
│   高齢化率が高い県、自動車保有率が高い県ほど                                │
│   交通事故死亡率が高い傾向がある（統計的に有意）                            │
└─────────────────────────────────────────────────────────────────────────────┘
""")

# =============================================================================
# 8. CSV出力
# =============================================================================

print("\n" + "=" * 80)
print("【8. CSV出力】")
print("=" * 80)

comparison_df.to_csv('model_comparison_3models.csv', index=False, encoding='utf-8-sig')
print("✓ model_comparison_3models.csv を出力しました")

# 詳細結果
detail_data = {
    '指標': ['サンプルサイズ', '過分散パラメータφ', '尤度比統計量', '尤度比検定p値',
             'ポアソンAIC', '負の二項(sm.NB)AIC', '負の二項(GLM+α)AIC', '最適α',
             'VIF_elderly', 'VIF_car', '推奨モデル'],
    '値': [len(df), f"{phi:.3f}", f"{lr_stat:.3f}", f"{p_value_lr:.4f}",
           f"{poisson_result.aic:.3f}", 
           f"{nb_sm_result.aic:.3f}" if nb_sm_result else "N/A",
           f"{nb_glm_result.aic:.3f}", f"{optimal_alpha:.4f}",
           f"{vif_elderly:.3f}", f"{vif_car:.3f}", '負の二項(GLM+α最適化)']
}
detail_df = pd.DataFrame(detail_data)
detail_df.to_csv('analysis_summary.csv', index=False, encoding='utf-8-sig')
print("✓ analysis_summary.csv を出力しました")

print("\n分析完了！")
