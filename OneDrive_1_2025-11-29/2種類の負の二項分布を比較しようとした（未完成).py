import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.genmod.families import links

# =========================
# 1. 交通事故死者数データ (省略)
# =========================
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

# =========================
# 2. 人口＆高齢化率（第11表） (省略)
# =========================
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

# =========================
# 3. 自動車保有台数（r5c6pv...） (省略)
# =========================
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

# =========================
# 4. 3つのデータをマージして説明変数を作成
# =========================
df = (
    acc_df
    .merge(pop_df, on="pref_short")
    .merge(cars_df, on="pref_short")
)
df["car_per_1000"] = df["cars_total"] / (df["population"] / 1000)

df["log_pop"] = np.log(df["population"])
X = df[["elderly_rate", "car_per_1000"]]
X = sm.add_constant(X)

# =========================
# 5. モデル1: ポアソン回帰 (Poisson Regression)
# =========================
poisson_model = GLM(
    df["deaths"],
    X,
    family=families.Poisson(),
    offset=df["log_pop"]
)
poisson_result = poisson_model.fit()
print("### モデル 1: ポアソン回帰 (Poisson Regression) ###")
print(poisson_result.summary())

# ポアソンモデルの適合度チェック
pearson_chi2 = poisson_result.pearson_chi2
df_residuals = poisson_result.df_resid
overdispersion_param = pearson_chi2 / df_residuals

print("\n================================================================================")
print("             ポアソンモデルの適合度チェック")
print("================================================================================")
print(f"過分散パラメータ (phi^ = Pearson Chi2 / Df Residuals): {overdispersion_param:.3f}")

if overdispersion_param > 1.2:
    print("\n⚠️ 1.2 を大きく超えるため、過分散が懸念されます。負の二項回帰を検討すべきです。")
    
print("================================================================================")

# --- 負の二項回帰の比較をここから開始 ---

## 6. モデル2: 負の二項回帰 (NB2 Type 2)
# 負の二項回帰の一般的な実装。分散が平均の二乗に比例する (Var(Y) = mu + alpha*mu^2)
nb2_model = sm.NegativeBinomial(
    df["deaths"],
    X,
    loglikelihood_method='nb-2',
    offset=df["log_pop"]
)
nb2_result = nb2_model.fit(disp=False)

print("\n\n### モデル 2: 負の二項回帰 (NB2 Type 2) ###")
print(nb2_result.summary())


## 7. モデル3: 準ポアソン回帰 (Quasi-Poisson) - NB1型の分散構造近似
# Quasi-Poissonは、分散が平均に比例する構造 (Var(Y) = phi * mu) を持つため、
# 負の二項分布NB1型 (Var(Y) = (1+alpha) * mu) の線形分散構造を最もよく近似します。
# ただし、Quasi-PoissonはAICの計算ができないため、ここでは情報量規準の比較から除外します。
# 代わりに、GLMの枠組みで、より厳密な**NB1型**を推定します。

# statsmodelsにはNB1を明示的に指定する引数がないため、
# ここでは、NB2の結果と比較し、過分散パラメータ alpha の値を重視します。

# ********** NB1型モデルの推定 (Statsmodelsでの実装の難しさ) **********
# statsmodelsのNegativeBinomialクラスは、NB2型をデフォルトとしており、
# NB1型 (Var(Y) = (1+alpha) * mu) を直接サポートしていません。
# したがって、ここではNB2の結果のみを比較し、NB1型は概念的な比較に留めます。
# NB1型とNB2型の分散構造:
# NB1: Var(Y) = phi * mu
# NB2: Var(Y) = mu + alpha * mu^2
# *******************************************************************


# =========================
# 8. AICの比較 (Poisson vs NB2)
# =========================
print("\n================================================================================")
print("             モデル比較 (AIC: 小さい方が良いモデル)")
print("================================================================================")

# AICの取得
aic_poisson = poisson_result.aic
aic_nb2 = nb2_result.aic

print(f"ポアソン回帰 (Poisson) の AIC:            {aic_poisson:.3f}")
print(f"負の二項回帰 (NB2 Type 2) の AIC:        {aic_nb2:.3f}")

if aic_nb2 < aic_poisson:
    print(f"\n🏆 **負の二項回帰 (NB2 Type 2)** の AIC が小さく、データに対する適合度が高いと評価されます。")
    print("これは、過分散が存在し、分散が平均の二乗に比例して増加するという仮定が適切であることを示唆します。")
else:
    print(f"\n🏆 **ポアソン回帰 (Poisson)** の AIC が小さく、よりシンプルなモデルが推奨されます。")

print("\n--------------------------------------------------------------------------------")
print("             負の二項分布の2つの型の違い")
print("--------------------------------------------------------------------------------")
print("【NB1 Type 1 (線形分散型)】: 分散は平均に比例 (Var(Y) = phi * mu)")
print("【NB2 Type 2 (二次分散型)】: 分散は平均の二乗に比例 (Var(Y) = mu + alpha * mu^2)")
print("statsmodelsでは通常NB2型が使用されます。NB1型はQuasi-Poissonによって近似できますが、AIC比較はできません。")
print("================================================================================")