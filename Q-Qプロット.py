import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Matplotlibの日本語対応設定 (実行環境に合わせて調整してください)
plt.rcParams['font.family'] = 'Meiryo'
plt.rcParams['axes.unicode_minus'] = False

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
df["log_pop"] = np.log(df["population"])
X = df[["elderly_rate", "car_per_1000"]]
X = sm.add_constant(X)

# =========================
# 5. GLM (Poisson Regression) の推定
# =========================
model = sm.GLM(
    df["deaths"],
    X,
    family=sm.families.Poisson(),
    offset=df["log_pop"]
)
result = model.fit()

print("### GLM (Poisson Regression) 結果 ###")
print(result.summary())

print("\n", df)

# ====================================================================
# 6. 追加要素: Q-Qプロットの生成
# ====================================================================

# ピアソン残差を取得
pearson_residuals = result.resid_pearson

print("\n### Q-Qプロットを生成中... ###")
# Q-Qプロットを作成。sm.qqplot()は自動でグラフを生成
fig = sm.qqplot(
    pearson_residuals,
    line='45', # 45度線（理論的な正規分布）を描画
    fit=True
)

fig.suptitle('ピアソン残差の Q-Qプロット (Poisson Regression)', fontsize=16)
plt.show()

# ====================================================================
# 7. Q-Qプロットの解釈に関するガイダンス
# ====================================================================
print("\n================================================================================")
print("                       Q-Qプロットの解釈")
print("================================================================================")
print("Q-Qプロットは、残差が**正規分布**に従っているかをチェックします。")
print("理想的な状態では、プロットされた点（残差）が中央の45度線上に乗るはずです。")
print("1. **中央部のズレ**: データが線から大きく離れている場合、残差の分布が正規分布から逸脱していることを意味します。ポアソン回帰では厳密な正規性は期待されませんが、大きな逸脱は問題を示唆します。")
print("2. **両端のズレ**: 上端または下端の点が線から大きく離れている場合、それは**外れ値**（Outliers）である可能性が高く、モデルが特定の都道府県のデータをうまく説明できていないことを示します。")
print("3. **S字型**: 残差の裾野（両端）が理論的な正規分布の裾野と異なる広がりを持っていることを示唆します。")
print("================================================================================")