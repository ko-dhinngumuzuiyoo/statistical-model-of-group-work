import pandas as pd
import numpy as np
import statsmodels.api as sm

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
# 4. 3つのデータをマージして説明変数を作成 (省略)
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

model = sm.GLM(
    df["deaths"],
    X,
    family=sm.families.Poisson(),
    offset=df["log_pop"]
)

result = model.fit()
print(result.summary())

# =========================
# 5. 過分散パラメータの計算 (追記部分)
# =========================

# ピアソンカイ二乗統計量を取得
# result.pearson_chi2は、モデル全体のピアソン残差の二乗和（Pearson chi2）に相当
pearson_chi2 = result.pearson_chi2

# 残差自由度 (Degrees of Freedom Residuals) を取得
df_residuals = result.df_resid

# 過分散パラメータ (φ) を計算
# φ^ = Pearson Chi2 / Df Residuals
overdispersion_param = pearson_chi2 / df_residuals

print("\n================================================================================")
print("             過分散パラメータの計算")
print("================================================================================")
print(f"ピアソンカイ二乗 (Pearson Chi2): {pearson_chi2:.3f}")
print(f"残差自由度 (Df Residuals):      {df_residuals:.0f}")
print(f"過分散パラメータ (phi^):        {overdispersion_param:.3f}")
print("================================================================================")

# 過分散の判定
if overdispersion_param > 1.0:
    print(f"\n💡 過分散パラメータが 1.0 より大きいため、**過分散**の可能性があります。")
    print("分散が平均よりも大きいことを示唆します。")
    print("この場合、標準誤差が過小評価されている可能性があるため、")
    print("準ポアソンモデル（Quasi-Poisson）や負の二項モデル（Negative Binomial）の検討が推奨されます。")
else:
    print("\n💡 過分散パラメータは 1.0 に近いため、ポアソンモデルの仮定は概ね妥当と考えられます。")