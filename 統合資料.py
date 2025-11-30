import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor  # VIF計算用

# Matplotlibの日本語対応設定 (環境に合わせて調整してください)
plt.rcParams['font.family'] = 'Meiryo'
plt.rcParams['axes.unicode_minus'] = False

# 図保存用ディレクトリ
os.makedirs("figures", exist_ok=True)

# ============================================================================
# 1. データ読み込み：交通事故死者数・人口＆高齢化率・自動車保有台数
# ============================================================================

# -------------------------
# 1-1. 交通事故死者数データ
# -------------------------
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

# -------------------------
# 1-2. 人口＆高齢化率（第11表）
# -------------------------
pop_raw = pd.read_excel("a01100_2.xlsx", sheet_name=0, header=None)
pop_pref = pop_raw[
    (pop_raw[7] == 2023001010) &    # 2023年1月1日現在
    (pop_raw[9] == "総人口") &      # 総人口
    (pop_raw[10] != "00000")        # 全国行を除外
].copy()
pop_pref["pref_full"] = pop_pref[11].str.replace("　", "").str.strip()

def to_short(name: str) -> str:
    """「北海道」「東京都」「高知県」→「北海道」「東京」「高知」などに変換"""
    if name == "北海道":
        return name
    for suf in ["都", "府", "県"]:
        if name.endswith(suf):
            return name[:-1]
    return name

pop_pref["pref_short"] = pop_pref["pref_full"].apply(to_short)
pop_pref["population"]  = (pop_pref[14] * 1000).astype(int)   # 単位を人に
pop_pref["pop_65plus"]  = (pop_pref[17] * 1000).astype(int)
pop_pref["elderly_rate"] = pop_pref["pop_65plus"] / pop_pref["population"]
pop_df = pop_pref[["pref_short", "population", "elderly_rate"]].copy()

# -------------------------
# 1-3. 自動車保有台数（r5c6pv...）
# -------------------------
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

# 北海道は複数の陸運支局の合計
hokkaido_offices = ["札幌","函館","旭川","室蘭","釧路","帯広","北見"]
hokkaido_total = cars_raw.loc[cars_raw[1].isin(hokkaido_offices), 7].sum()

# 沖縄は「沖」が含まれる行を抽出
okinawa_total = cars_raw.loc[cars_raw[0].astype(str).str.contains("沖"), 7].iloc[0]

cars_df = pd.concat([
    pd.DataFrame({"pref_short": ["北海道"], "cars_total": [hokkaido_total]}),
    cars_pref,
    pd.DataFrame({"pref_short": ["沖縄"],  "cars_total": [okinawa_total]}),
], ignore_index=True)

cars_df["cars_total"] = cars_df["cars_total"].astype(int)

# ============================================================================
# 2. 3つのデータをマージし、派生変数を作成
# ============================================================================

df = (
    acc_df
    .merge(pop_df,  on="pref_short")
    .merge(cars_df, on="pref_short")
)

# クルマ依存度（人口千人あたり自動車台数）
df["car_per_1000"] = df["cars_total"] / (df["population"] / 1000)

# 10万人あたり交通事故死亡率
df["deaths_per_100k"] = (df["deaths"] / df["population"]) * 100000

# ポアソン回帰用のoffset（人口の対数）
df["log_pop"] = np.log(df["population"])

print("### データフレーム先頭 5行 ###")
print(df.head())
print("\n### 統計量 (describe) ###")
print(df.describe())

# ============================================================================
# 3. 相関係数の確認（死亡率・高齢化率・クルマ依存度）
# ============================================================================

print("\n### 主要変数の相関係数 (corr()) ###")
corr_mat = df[["deaths_per_100k", "elderly_rate", "car_per_1000"]].corr()
print(corr_mat)

# ============================================================================
# 4. 多重共線性の確認：相関係数 & VIF
# ============================================================================

# elderly_rate と car_per_1000 の単純相関
correlation = df["elderly_rate"].corr(df["car_per_1000"])
print("\n================================================================================")
print(f"高齢化率 (elderly_rate) と クルマ依存度 (car_per_1000) の相関係数: {correlation:.4f}")
print("================================================================================")

# VIFの計算
X = df[["elderly_rate", "car_per_1000"]]
X = sm.add_constant(X)  # const, elderly_rate, car_per_1000

vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

print("\n### VIF (Variance Inflation Factor) 結果 ###")
print(vif_data)

max_vif = vif_data.loc[vif_data["Variable"] != "const", "VIF"].max()

print("\n--------------------------------------------------------------------------------")
if max_vif > 10.0:
    print(f"⚠️ VIFの最大値 ({max_vif:.2f}) が 10 を超えています。多重共線性の可能性に注意してください。")
else:
    print(f"✅ VIFの最大値 ({max_vif:.2f}) は 10 以下です。深刻な多重共線性の問題はなさそうです。")
print("--------------------------------------------------------------------------------")

# ============================================================================
# 5. ポアソン回帰 (GLM) の推定
# ============================================================================

model = sm.GLM(
    df["deaths"],
    X,  # 上でVIFに使ったものと同じ説明変数
    family=sm.families.Poisson(),
    offset=df["log_pop"]
)
result = model.fit()

print("\n### GLM (Poisson Regression) 結果 ###")
print(result.summary())

# ============================================================================
# 6. 散布図3種類：視覚的な関係の確認
# ============================================================================
# ============================================================================
# 6. 散布図：各図を1枚ずつ保存
# ============================================================================

print("\n### 散布図（3種類）を個別に描画＆保存中... ###")

# 6-1. 高齢化率 vs 10万人あたり死亡率
plt.figure(figsize=(6, 5))
plt.scatter(df["elderly_rate"] * 100, df["deaths_per_100k"], alpha=0.7)
plt.title('高齢化率 vs 10万人あたり死亡率')
plt.xlabel('高齢化率 (%)')
plt.ylabel('10万人あたり死亡率')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("figures/scatter_elderly_vs_deaths.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# 6-2. クルマ依存度 vs 10万人あたり死亡率
plt.figure(figsize=(6, 5))
plt.scatter(df["car_per_1000"], df["deaths_per_100k"], alpha=0.7)
plt.title('自動車台数 vs 10万人あたり死亡率')
plt.xlabel('人口千人あたり自動車台数')
plt.ylabel('10万人あたり死亡率')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("figures/scatter_car_vs_deaths.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# 6-3. 高齢化率 vs クルマ依存度
plt.figure(figsize=(6, 5))
plt.scatter(df["elderly_rate"] * 100, df["car_per_1000"], alpha=0.7)
plt.title('高齢化率 vs 自動車台数')
plt.xlabel('高齢化率 (%)')
plt.ylabel('人口千人あたり自動車台数')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("figures/scatter_elderly_vs_car.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ============================================================================
# 7. 残差診断 (1)：ピアソン残差の Q-Q プロット
# ============================================================================

pearson_residuals = result.resid_pearson

print("\n### ピアソン残差の Q-Q プロットを描画＆保存中... ###")

fig = sm.qqplot(
    pearson_residuals,
    line='45',   # 理論線
    fit=True
)
fig.suptitle('ピアソン残差の Q-Qプロット (Poisson Regression)', fontsize=16)

plt.savefig("figures/qqplot_pearson_residuals.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print("\n================================================================================")
print("【Q-Qプロットの見方】")
print("・点が45度線の近くに並んでいれば、残差の分布が理論分布（ここでは正規分布）に近い。")
print("・両端で大きく外れていれば、外れ値や裾の重さの違いを示唆。")
print("・S字状のパターンは、分布形が理論分布と系統的に異なるサイン。")
print("================================================================================")

# ============================================================================
# 8. 残差診断 (2)：ピアソン残差 vs 予測値
# ============================================================================

fitted_values = result.fittedvalues

print("\n### ピアソン残差 vs 予測値の散布図を描画＆保存中... ###")

plt.figure(figsize=(6, 5))
plt.scatter(fitted_values, pearson_residuals, alpha=0.7)
plt.axhline(y=0, linestyle='--')  # 残差0の基準線

plt.title('ピアソン残差 vs 予測値 (Poisson Regression)', fontsize=14)
plt.xlabel('予測値 (Fitted Values: $\hat{\mu}$)', fontsize=12)
plt.ylabel('ピアソン残差 (Pearson Residuals)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("figures/pearson_residuals_vs_fitted.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print("\n================================================================================")
print("【残差 vs 予測値プロットの見方】")
print("・理想：残差が0の周りにランダムに散らばる → モデルの仮定と整合的。")
print("・扇形（ファン形状）：予測値が大きいほど残差のばらつきが増える → 過分散などの可能性。")
print("・曲線パターン：リンク関数が不適切 or 重要な説明変数の欠落などを示唆。")
print("・大きく離れた点：モデルがうまく説明できていない都道府県（外れ値候補）。")
print("================================================================================")
