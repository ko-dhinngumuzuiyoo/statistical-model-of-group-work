import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
 
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
# 5. ポアソン回帰の推定
# =========================
# モデル1: Poisson (従来のモデル)
poisson_model = GLM(
    df["deaths"],
    X,
    family=families.Poisson(),
    offset=df["log_pop"]
)
poisson_result = poisson_model.fit()
print("### モデル 1: ポアソン回帰 (Poisson Regression) ###")
print(poisson_result.summary())
 
# ピアソン残差から過分散パラメータを計算
pearson_chi2 = poisson_result.pearson_chi2
df_residuals = poisson_result.df_resid
overdispersion_param = pearson_chi2 / df_residuals
 
print("\n================================================================================")
print("             ポアソンモデルの適合度チェック")
print("================================================================================")
print(f"過分散パラメータ (phi^ = Pearson Chi2 / Df Residuals): {overdispersion_param:.3f}")
 
if overdispersion_param > 1.2: # 1.2を大きく超えると過分散の懸念あり
    print("\n⚠️ 1.2 を大きく超えるため、過分散が懸念されます。")
   
print("================================================================================")
 
 
# =========================
# 6. 負の二項回帰の推定 (追記)
# =========================
# モデル2: Negative Binomial (負の二項回帰)
# statsmodelsでは、負の二項モデルは NegativeBinomial クラスを使用し、GLMではない
# 負の二項モデルは、過分散を捕捉するための追加パラメータ 'alpha' を持つ
nb_model = sm.NegativeBinomial(
    df["deaths"],
    X,
    loglikelihood_method='nb-2', # 負の二項モデルの一般的に使用されるタイプ
    offset=df["log_pop"]
)
nb_result = nb_model.fit(disp=False) # disp=Falseで冗長な出力を抑制
 
print("\n\n### モデル 2: 負の二項回帰 (Negative Binomial Regression) ###")
print(nb_result.summary())
 
 
# =========================
# 7. AICの比較 (追記)
# =========================
print("\n================================================================================")
print("             モデル比較 (AIC: 小さい方が良いモデル)")
print("================================================================================")
 
# AICの取得
aic_poisson = poisson_result.aic
aic_nb = nb_result.aic
 
print(f"ポアソン回帰 (Poisson) の AIC:            {aic_poisson:.3f}")
print(f"負の二項回帰 (Negative Binomial) の AIC: {aic_nb:.3f}")
 
if aic_nb < aic_poisson:
    print(f"\n🏆 **負の二項回帰 (Negative Binomial)** の AIC が小さく、データに対する適合度が高いと評価されます。")
    print("これは、過分散が存在し、それをモデルが考慮できている可能性を示唆します。")
else:
    print(f"\n🏆 **ポアソン回帰 (Poisson)** の AIC が小さく、よりシンプルなモデルが推奨されます。")
 
print("================================================================================")
 