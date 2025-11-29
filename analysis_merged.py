"""
================================================================================
都道府県別交通事故死亡率 × 高齢化・クルマ依存度（統計モデリング課題）
================================================================================

【完成版】ポアソン回帰 vs 負の二項回帰の包括的比較

このスクリプトでは以下を実行します：
1. データ読み込み・前処理
2. 探索的データ分析（相関係数、散布図）
3. ポアソン回帰 + 過分散診断
4. 負の二項回帰（2つの方法で実装）
   - sm.NegativeBinomial（alpha推定）
   - sm.GLM + alpha最適化（より安定）
5. 準ポアソン回帰
6. モデル比較（AIC、尤度比検定）
7. 多重共線性チェック（VIF）
8. 残差診断プロット
9. 結果のCSV出力

【重要な発見】
- sm.NegativeBinomial は収束しないことがある → 収束確認が必須
- sm.GLM + NegativeBinomial(alpha=固定) は不適切な結果を出す
- 正しく実装すれば、過分散がある場合は負の二項回帰のAICが低くなる
================================================================================
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.genmod.families import NegativeBinomial as NB_family
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats as scipy_stats
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 0. 設定
# =============================================================================

# 日本語フォント設定（環境に応じて変更）
try:
    plt.rcParams['font.family'] = ['Hiragino Sans', 'Yu Gothic', 'Meirio', 'MS Gothic']
except:
    pass

# =============================================================================
# 1. データ読み込み
# =============================================================================

print("=" * 80)
print("【1. データ読み込み】")
print("=" * 80)

# --- 1.1 交通事故死者数データ ---
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

# --- 1.2 人口＆高齢化率（第11表）---
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

# --- 1.3 自動車保有台数 ---
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

# --- 1.4 マージ ---
df = (
    acc_df
    .merge(pop_df, on="pref_short")
    .merge(cars_df, on="pref_short")
)
df["car_per_1000"] = df["cars_total"] / (df["population"] / 1000)
df["log_pop"] = np.log(df["population"])
df["deaths_per_100k"] = df["deaths"] / df["population"] * 100000

print(f"✓ データ読み込み完了")
print(f"  - 都道府県数: {len(df)}")
print(f"  - 欠損値: {df.isnull().sum().sum()} 件")

# =============================================================================
# 2. 探索的データ分析（EDA）
# =============================================================================

print("\n" + "=" * 80)
print("【2. 探索的データ分析】")
print("=" * 80)

print("\n--- 2.1 基本統計量 ---")
stats_cols = ['deaths', 'population', 'elderly_rate', 'car_per_1000', 'deaths_per_100k']
print(df[stats_cols].describe().round(3))

print("\n--- 2.2 相関係数行列 ---")
corr_cols = ['deaths_per_100k', 'elderly_rate', 'car_per_1000']
corr_matrix = df[corr_cols].corr()
print(corr_matrix.round(3))

# 説明変数の準備
X = df[["elderly_rate", "car_per_1000"]]
X = sm.add_constant(X)
y = df["deaths"]
offset = df["log_pop"]

# =============================================================================
# 3. ポアソン回帰
# =============================================================================

print("\n" + "=" * 80)
print("【3. ポアソン回帰 (Poisson Regression)】")
print("=" * 80)

poisson_model = GLM(y, X, family=families.Poisson(), offset=offset)
poisson_result = poisson_model.fit()

print(poisson_result.summary())

# 過分散診断
pearson_chi2 = poisson_result.pearson_chi2
df_resid = poisson_result.df_resid
phi = pearson_chi2 / df_resid

print("\n" + "-" * 40)
print("過分散診断")
print("-" * 40)
print(f"ピアソンカイ二乗 (Pearson Chi2): {pearson_chi2:.3f}")
print(f"残差自由度 (Df Residuals): {df_resid}")
print(f"過分散パラメータ (φ^): {phi:.3f}")

if phi > 1.2:
    print(f"\n⚠️ φ = {phi:.3f} > 1.2 なので過分散の懸念あり")
    print("   → 負の二項回帰または準ポアソン回帰を検討")
else:
    print("\n✓ 過分散なし（ポアソン回帰で問題なし）")

# =============================================================================
# 4. 負の二項回帰 - 方法A: sm.NegativeBinomial（友人の方法）
# =============================================================================

print("\n" + "=" * 80)
print("【4. 負の二項回帰 - 方法A: sm.NegativeBinomial】")
print("=" * 80)

print("""
この方法は alpha を最尤推定します。
ただし、収束しないことがあるため、収束状況の確認が必須です。
""")

nb_result_A = None
nb_converged_A = False

try:
    nb_model_A = sm.NegativeBinomial(
        y,
        X,
        loglike_method='nb2',  # 負の二項モデルのタイプ
        offset=offset
    )
    nb_result_A = nb_model_A.fit(disp=False, maxiter=1000)
    
    # 収束確認
    nb_converged_A = nb_result_A.mle_retvals.get('converged', False)
    
    print(nb_result_A.summary())
    
    print("\n" + "-" * 40)
    print("収束状況の確認")
    print("-" * 40)
    print(f"収束: {'✓ 成功' if nb_converged_A else '⚠️ 失敗'}")
    print(f"推定alpha: {nb_result_A.params[-1]:.4f}")
    print(f"対数尤度: {nb_result_A.llf:.3f}")
    print(f"AIC: {nb_result_A.aic:.3f}")
    
    if not nb_converged_A:
        print("\n⚠️ 警告: モデルが収束していません。AICは参考値として扱ってください。")
        
except Exception as e:
    print(f"エラー: {e}")

# =============================================================================
# 5. 負の二項回帰 - 方法B: GLM + alpha最適化（より安定）
# =============================================================================

print("\n" + "=" * 80)
print("【5. 負の二項回帰 - 方法B: GLM + alpha最適化】")
print("=" * 80)

print("""
この方法はグリッドサーチでalphaを最適化します。
sm.NegativeBinomial より安定して収束することが多いです。
""")

def find_optimal_alpha(y, X, offset, alpha_range=(0.001, 2.0), step=0.005):
    """負の二項回帰の最適なalphaをグリッドサーチで見つける"""
    alphas = np.arange(alpha_range[0], alpha_range[1], step)
    aics = []
    
    for a in alphas:
        try:
            model = sm.GLM(y, X, family=NB_family(alpha=a), offset=offset)
            result = model.fit(disp=0)
            aics.append(result.aic)
        except:
            aics.append(np.inf)
    
    best_idx = np.argmin(aics)
    return alphas[best_idx], aics[best_idx]

print("alphaの最適化中...")
optimal_alpha, _ = find_optimal_alpha(y, X, offset)
print(f"最適alpha: {optimal_alpha:.4f}")

# 最適alphaで負の二項回帰を実行
nb_model_B = sm.GLM(y, X, family=NB_family(alpha=optimal_alpha), offset=offset)
nb_result_B = nb_model_B.fit()

print(nb_result_B.summary())

print("\n" + "-" * 40)
print("モデル情報")
print("-" * 40)
print(f"最適alpha: {optimal_alpha:.4f}")
print(f"対数尤度: {nb_result_B.llf:.3f}")
print(f"AIC: {nb_result_B.aic:.3f}")

# =============================================================================
# 6. 準ポアソン回帰（Quasi-Poisson）
# =============================================================================

print("\n" + "=" * 80)
print("【6. 準ポアソン回帰 (Quasi-Poisson)】")
print("=" * 80)

print("""
準ポアソン回帰は、過分散を考慮して標準誤差を補正します。
分布の仮定を変えずに、分散を scale × μ とします。
AICは定義されませんが、係数の信頼区間が適切になります。
""")

quasi_poisson_result = poisson_model.fit(scale='X2')
print(quasi_poisson_result.summary())

# =============================================================================
# 7. モデル比較
# =============================================================================

print("\n" + "=" * 80)
print("【7. モデル比較】")
print("=" * 80)

print("\n--- 7.1 AIC比較 ---")

# 結果を格納
results_dict = {
    'ポアソン回帰': {
        'AIC': poisson_result.aic,
        'LLF': poisson_result.llf,
        'converged': True
    },
    '負の二項 (方法A: sm.NB)': {
        'AIC': nb_result_A.aic if nb_result_A else None,
        'LLF': nb_result_A.llf if nb_result_A else None,
        'converged': nb_converged_A
    },
    '負の二項 (方法B: GLM+α最適化)': {
        'AIC': nb_result_B.aic,
        'LLF': nb_result_B.llf,
        'converged': True
    }
}

print(f"""
┌───────────────────────────────────────┬────────────┬────────────┬──────────┐
│ モデル                                │ AIC        │ 対数尤度   │ 収束     │
├───────────────────────────────────────┼────────────┼────────────┼──────────┤
│ ポアソン回帰                          │ {poisson_result.aic:>10.3f} │ {poisson_result.llf:>10.3f} │ ✓        │
""")

if nb_result_A:
    conv_mark = '✓' if nb_converged_A else '⚠️'
    print(f"│ 負の二項 (方法A: sm.NB)               │ {nb_result_A.aic:>10.3f} │ {nb_result_A.llf:>10.3f} │ {conv_mark}        │")

print(f"│ 負の二項 (方法B: GLM+α最適化)         │ {nb_result_B.aic:>10.3f} │ {nb_result_B.llf:>10.3f} │ ✓        │")
print(f"│ 準ポアソン回帰                        │   (定義なし) │ {quasi_poisson_result.llf:>10.3f} │ ✓        │")
print("└───────────────────────────────────────┴────────────┴────────────┴──────────┘")
print(f"\n過分散パラメータ φ = {phi:.3f}")

# --- 7.2 尤度比検定 ---
print("\n--- 7.2 尤度比検定（過分散の統計的検定）---")

# 方法Bの結果を使用（より安定）
if nb_result_B.llf > poisson_result.llf:
    lr_stat = 2 * (nb_result_B.llf - poisson_result.llf)
    # 負の二項のalphaは0以上なので、境界上の検定
    p_value = 0.5 * scipy_stats.chi2.sf(lr_stat, 1)
    
    print(f"帰無仮説: ポアソン回帰で十分（過分散なし）")
    print(f"対立仮説: 負の二項回帰が必要（過分散あり）")
    print(f"\n尤度比統計量: {lr_stat:.3f}")
    print(f"p値（近似）: {p_value:.6f}")
    
    if p_value < 0.05:
        print("\n結論: p < 0.05 なので過分散は統計的に有意")
        print("      → 負の二項回帰を採用すべき")
    else:
        print("\n結論: p ≥ 0.05 なので過分散は統計的に有意でない")
        print("      → ポアソン回帰で十分かもしれない")
else:
    print("負の二項回帰の対数尤度がポアソンより低いため、検定をスキップ")

# --- 7.3 推奨モデルの決定 ---
print("\n--- 7.3 推奨モデル ---")

# AICが最小のモデルを選択
aic_comparison = {
    'ポアソン回帰': poisson_result.aic,
    '負の二項 (GLM+α最適化)': nb_result_B.aic
}

if nb_result_A and nb_converged_A:
    aic_comparison['負の二項 (sm.NB)'] = nb_result_A.aic

best_model_name = min(aic_comparison, key=aic_comparison.get)
best_aic = aic_comparison[best_model_name]

print(f"🏆 推奨モデル: {best_model_name}")
print(f"   AIC: {best_aic:.3f}")

if best_model_name == 'ポアソン回帰':
    recommended_result = poisson_result
elif best_model_name == '負の二項 (GLM+α最適化)':
    recommended_result = nb_result_B
else:
    recommended_result = nb_result_A

# =============================================================================
# 8. 多重共線性チェック（VIF）
# =============================================================================

print("\n" + "=" * 80)
print("【8. 多重共線性チェック（VIF）】")
print("=" * 80)

X_vif = df[["elderly_rate", "car_per_1000"]]
vif_data = pd.DataFrame()
vif_data["変数"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]

print(vif_data.to_string(index=False))
print(f"\nVIF最大値: {vif_data['VIF'].max():.2f}")

if vif_data['VIF'].max() < 5:
    print("✓ VIF < 5: 多重共線性の問題なし")
elif vif_data['VIF'].max() < 10:
    print("⚠️ 5 < VIF < 10: 軽度の多重共線性あり（許容範囲）")
else:
    print("❌ VIF > 10: 深刻な多重共線性あり（要対策）")

# =============================================================================
# 9. 係数の解釈
# =============================================================================

print("\n" + "=" * 80)
print(f"【9. 係数の解釈（{best_model_name}）】")
print("=" * 80)

elderly_coef = recommended_result.params['elderly_rate']
car_coef = recommended_result.params['car_per_1000']
elderly_se = recommended_result.bse['elderly_rate']
car_se = recommended_result.bse['car_per_1000']

print(f"""
■ 高齢化率 (elderly_rate)
  - 係数: {elderly_coef:.4f} (SE: {elderly_se:.4f})
  - 解釈: 高齢化率が1%ポイント上昇すると
          死亡リスクは exp({elderly_coef:.4f}) = {np.exp(elderly_coef):.4f} 倍
          つまり {(np.exp(elderly_coef)-1)*100:.2f}% {'増加' if elderly_coef > 0 else '減少'}

■ 自動車保有率 (car_per_1000)
  - 係数: {car_coef:.6f} (SE: {car_se:.6f})
  - 解釈: 人口千人あたり自動車台数が1台増えると
          死亡リスクは exp({car_coef:.6f}) = {np.exp(car_coef):.6f} 倍
          100台増加で {(np.exp(car_coef*100)-1)*100:.2f}% {'増加' if car_coef > 0 else '減少'}
""")

# =============================================================================
# 10. 残差診断プロット
# =============================================================================

print("\n" + "=" * 80)
print("【10. 残差診断プロット】")
print("=" * 80)

try:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 10.1 ピアソン残差 vs 予測値
    residuals = poisson_result.resid_pearson
    fitted = poisson_result.fittedvalues
    
    axes[0, 0].scatter(fitted, residuals, alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[0, 0].set_xlabel('Fitted values')
    axes[0, 0].set_ylabel('Pearson residuals')
    axes[0, 0].set_title('Residuals vs Fitted (Poisson)')
    
    # 10.2 Q-Qプロット
    scipy_stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot (Poisson)')
    
    # 10.3 散布図：高齢化率 vs 死亡率
    axes[1, 0].scatter(df['elderly_rate'], df['deaths_per_100k'], alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[1, 0].set_xlabel('Elderly rate')
    axes[1, 0].set_ylabel('Deaths per 100k')
    axes[1, 0].set_title(f'Elderly Rate vs Death Rate (r={corr_matrix.loc["deaths_per_100k", "elderly_rate"]:.3f})')
    
    # 10.4 散布図：自動車保有率 vs 死亡率
    axes[1, 1].scatter(df['car_per_1000'], df['deaths_per_100k'], alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[1, 1].set_xlabel('Cars per 1000 people')
    axes[1, 1].set_ylabel('Deaths per 100k')
    axes[1, 1].set_title(f'Car Ownership vs Death Rate (r={corr_matrix.loc["deaths_per_100k", "car_per_1000"]:.3f})')
    
    plt.tight_layout()
    plt.savefig('diagnostic_plots.png', dpi=150, bbox_inches='tight')
    print("✓ 残差診断プロットを diagnostic_plots.png に保存しました")
    plt.close()
    
except Exception as e:
    print(f"プロット作成エラー: {e}")

# =============================================================================
# 11. 結果のCSV出力
# =============================================================================

print("\n" + "=" * 80)
print("【11. 結果出力】")
print("=" * 80)

# --- 11.1 モデル比較結果 ---
summary_data = {
    '指標': [
        'サンプルサイズ',
        '過分散パラメータ φ',
        'ポアソン回帰 AIC',
        'ポアソン回帰 対数尤度',
        '負の二項 (sm.NB) AIC',
        '負の二項 (sm.NB) 収束',
        '負の二項 (GLM+α最適化) AIC',
        '負の二項 最適alpha',
        '尤度比検定 統計量',
        '尤度比検定 p値',
        '推奨モデル',
        'elderly_rate 係数',
        'elderly_rate 標準誤差',
        'car_per_1000 係数',
        'car_per_1000 標準誤差',
        'VIF (elderly_rate)',
        'VIF (car_per_1000)',
        '相関: deaths_per_100k vs elderly_rate',
        '相関: deaths_per_100k vs car_per_1000',
        '相関: elderly_rate vs car_per_1000'
    ],
    '値': [
        len(df),
        f"{phi:.3f}",
        f"{poisson_result.aic:.3f}",
        f"{poisson_result.llf:.3f}",
        f"{nb_result_A.aic:.3f}" if nb_result_A else "N/A",
        "Yes" if nb_converged_A else "No",
        f"{nb_result_B.aic:.3f}",
        f"{optimal_alpha:.4f}",
        f"{lr_stat:.3f}" if 'lr_stat' in dir() else "N/A",
        f"{p_value:.6f}" if 'p_value' in dir() else "N/A",
        best_model_name,
        f"{elderly_coef:.6f}",
        f"{elderly_se:.6f}",
        f"{car_coef:.6f}",
        f"{car_se:.6f}",
        f"{vif_data.loc[vif_data['変数']=='elderly_rate', 'VIF'].values[0]:.3f}",
        f"{vif_data.loc[vif_data['変数']=='car_per_1000', 'VIF'].values[0]:.3f}",
        f"{corr_matrix.loc['deaths_per_100k', 'elderly_rate']:.3f}",
        f"{corr_matrix.loc['deaths_per_100k', 'car_per_1000']:.3f}",
        f"{corr_matrix.loc['elderly_rate', 'car_per_1000']:.3f}"
    ]
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('model_comparison_results.csv', index=False, encoding='utf-8-sig')
print("✓ モデル比較結果を model_comparison_results.csv に保存しました")

# --- 11.2 都道府県別分析結果 ---
df_output = df[['pref_short', 'deaths', 'population', 'elderly_rate', 
                'car_per_1000', 'deaths_per_100k']].copy()
df_output['predicted_deaths'] = recommended_result.fittedvalues.round(1)
df_output['residual'] = (df_output['deaths'] - df_output['predicted_deaths']).round(1)
df_output['pearson_residual'] = recommended_result.resid_pearson.round(3)
df_output.to_csv('prefecture_analysis.csv', index=False, encoding='utf-8-sig')
print("✓ 都道府県別分析結果を prefecture_analysis.csv に保存しました")

# =============================================================================
# 12. 最終サマリー
# =============================================================================

print("\n" + "=" * 80)
print("【12. 最終サマリー】")
print("=" * 80)

print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                           分析結果サマリー                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ データ                                                                      │
│   - サンプルサイズ: {len(df)} 都道府県                                           │
│   - 目的変数: 交通事故死亡者数 (2023年)                                     │
│   - 説明変数: 高齢化率, 自動車保有率 (人口千人あたり)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ 過分散診断                                                                  │
│   - 過分散パラメータ φ = {phi:.3f}                                              │
│   - 判定: {'過分散あり → 負の二項回帰を検討' if phi > 1.2 else '過分散なし → ポアソンでOK'}                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ モデル比較                                                                  │
│   - ポアソン回帰 AIC: {poisson_result.aic:.3f}                                      │
│   - 負の二項回帰 AIC (GLM+α最適化): {nb_result_B.aic:.3f}                          │
│   - 推奨モデル: {best_model_name}                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ 多重共線性                                                                  │
│   - VIF最大値: {vif_data['VIF'].max():.2f} → {'問題なし' if vif_data['VIF'].max() < 5 else '要注意'}                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ 出力ファイル                                                                │
│   - model_comparison_results.csv: モデル比較の詳細結果                      │
│   - prefecture_analysis.csv: 都道府県別の予測値・残差                       │
│   - diagnostic_plots.png: 残差診断プロット                                  │
└─────────────────────────────────────────────────────────────────────────────┘
""")

print("分析完了！")
