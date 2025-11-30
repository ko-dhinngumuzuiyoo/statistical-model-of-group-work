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


# ====================================================================
# 6. 追加要素: ピアソン残差 vs 予測値の散布図の生成
# ====================================================================

# ピアソン残差を取得
# Pearson Residuals (rp) = (観測値 - 期待値) / sqrt(分散の推定値)
pearson_residuals = result.resid_pearson

# 予測値 (fitted values, 期待値 $\hat{\mu}$) を取得
fitted_values = result.fittedvalues

print("\n### ピアソン残差 vs 予測値の散布図を生成中... ###")

# 散布図の作成
plt.figure(figsize=(10, 6))
plt.scatter(fitted_values, pearson_residuals, alpha=0.7)

# 基準線 (残差 = 0) を描画
plt.axhline(y=0, color='r', linestyle='--')

# 軸ラベルとタイトル
plt.title('ピアソン残差 vs 予測値 (Poisson Regression)', fontsize=14)
plt.xlabel('予測値 (Fitted Values: $\hat{\mu}$)', fontsize=12)
plt.ylabel('ピアソン残差 (Pearson Residuals)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)

plt.show()

# ====================================================================
# 7. 残差プロットの解釈に関するガイダンス
# ====================================================================
print("\n================================================================================")
print("                    残差プロットの解釈")
print("================================================================================")
print("残差プロットの理想的な状態は、残差が0の線の周りに**特定のパターンなくランダムに散らばっている**ことです。")
print("1. **ファン形状 (扇状)**: 残差の広がりが予測値の増加とともに大きくなる場合、**分散が一定ではない**ことを示唆し、ポアソンモデルの仮定（分散＝平均）が破られている可能性（過分散）があります。")
print("2. **曲線パターン**: 残差がU字型や逆U字型などの曲線パターンを示す場合、モデルに重要な説明変数が含まれていないか、リンク関数（Log関数）が適切でない可能性があります。")
print("3. **外れ値**: ゼロの線から大きく離れた点（残差の絶対値が大きい点）は、モデルがその都道府県の死亡者数をうまく説明できていない**外れ値**である可能性があります。")
print("================================================================================")