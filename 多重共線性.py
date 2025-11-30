import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor # VIF計算用

# =========================
# 1. 交通事故死者数データ
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
# 2. 人口＆高齢化率（第11表）
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
pop_pref["population"]  = (pop_pref[14] * 1000).astype(int)
pop_pref["pop_65plus"]  = (pop_pref[17] * 1000).astype(int)
pop_pref["elderly_rate"] = pop_pref["pop_65plus"] / pop_pref["population"]
pop_df = pop_pref[["pref_short", "population", "elderly_rate"]].copy()

# =========================
# 3. 自動車保有台数（r5c6pv...）
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
    pd.DataFrame({"pref_short": ["沖縄"],  "cars_total": [okinawa_total]}),
], ignore_index=True)

cars_df["cars_total"] = cars_df["cars_total"].astype(int)

# =========================
# 4. 3つのデータをマージして説明変数を作成
# =========================
df = (
    acc_df
    .merge(pop_df,  on="pref_short")
    .merge(cars_df, on="pref_short")
)

# クルマ依存度（人口千人あたりの自動車台数）
df["car_per_1000"] = df["cars_total"] / (df["population"] / 1000)

# ====================================================================
# 5. 追加要素: 相関係数とVIFの計算
# ====================================================================

# 1. elderly_rate と car_per_1000 の相関係数を表示
correlation = df["elderly_rate"].corr(df["car_per_1000"])

print("================================================================================")
print(f"高齢化率 (elderly_rate) と クルマ依存度 (car_per_1000) の相関係数: {correlation:.4f}")
print("================================================================================")


# 2. VIF (分散拡大係数) の計算
X = df[["elderly_rate", "car_per_1000"]]
X = sm.add_constant(X)  # 切片（const）もVIF計算に含める

# VIFを格納するデータフレームを作成
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [
    variance_inflation_factor(X.values, i) 
    for i in range(X.shape[1])
]

print("\n### VIF (Variance Inflation Factor) 結果 ###")
print(vif_data)

# VIF > 10 のチェックとコメント
max_vif = vif_data['VIF'].drop(vif_data[vif_data['Variable'] == 'const'].index).max()

print("\n--------------------------------------------------------------------------------")
if max_vif > 10.0:
    print(f"⚠️ **VIFの最大値 ({max_vif:.2f}) が 10 を超えています。**\n   多重共線性が存在するため、回帰結果の解釈には注意が必要です。")
else:
    print(f"✅ **VIFの最大値 ({max_vif:.2f}) は 10 以下です。**\n   多重共線性の深刻な問題はないと考えられます。")
print("--------------------------------------------------------------------------------")

# =========================
# 6. GLM (Poisson Regression) の推定
# =========================
df["log_pop"] = np.log(df["population"])
# GLMの推定に使用するXは、VIF計算で作成したものと同じ
# X = df[["elderly_rate", "car_per_1000"]]
# X = sm.add_constant(X) 

model = sm.GLM(
    df["deaths"],
    X,
    family=sm.families.Poisson(),
    offset=df["log_pop"]
)

result = model.fit()

print("\n### GLM (Poisson Regression) 結果 ###")
print(result.summary())

print("\n", df)