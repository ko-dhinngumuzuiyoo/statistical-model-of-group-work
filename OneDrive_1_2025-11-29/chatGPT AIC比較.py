import pandas as pd
import numpy as np
import statsmodels.api as sm

# =========================
# 1. 交通事故死者数データ
# =========================
acc_raw = pd.read_csv("3都道府県別交通事故死者数.csv", encoding="cp932", header=None)

# 7行目〜54行目くらいがデータ本体
acc = acc_raw.iloc[7:55].copy()
acc.columns = [
    "region_code", "block", "pref_short",
    "deaths_2021", "deaths_2022", "deaths_2023",
    "rate_2021", "rate_2022", "rate_2023"
]

# 「全国」行を削除
acc = acc[acc["block"] != "全 国"].copy()

# 目的変数：2023年の死亡者数
acc["pref_short"] = acc["pref_short"].str.strip()
acc["deaths"] = acc["deaths_2023"].astype(int)
acc_df = acc[["pref_short", "deaths"]]

# =========================
# 2. 人口＆高齢化率（第11表）
# =========================
pop_raw = pd.read_excel("a01100_2.xlsx", sheet_name=0, header=None)

# 2023年10月1日時点・総人口・都道府県別だけフィルタ
pop_pref = pop_raw[
    (pop_raw[7] == 2023001010) &   # 時間軸コード
    (pop_raw[9] == "総人口") &    # 人口区分
    (pop_raw[10] != "00000")      # 全国（00000）を除外
].copy()

# 都道府県名整形
pop_pref["pref_full"] = pop_pref[11].str.replace("　", "").str.strip()

def to_short(name: str) -> str:
    if name == "北海道":
        return name
    for suf in ["都", "府", "県"]:
        if name.endswith(suf):
            return name[:-1]
    return name

pop_pref["pref_short"] = pop_pref["pref_full"].apply(to_short)

# 人口と65歳以上人口（単位：千人）→ 人数に直す
pop_pref["population"]  = (pop_pref[14] * 1000).astype(int)  # 総数
pop_pref["pop_65plus"]  = (pop_pref[17] * 1000).astype(int)  # 65歳以上
pop_pref["elderly_rate"] = pop_pref["pop_65plus"] / pop_pref["population"]

pop_df = pop_pref[["pref_short", "population", "elderly_rate"]].copy()

# =========================
# 3. 自動車保有台数（r5c6pv...）
# =========================
cars_raw = pd.read_excel("r5c6pv0000013d12.xlsx", sheet_name="8", header=None)

# 北海道以外の都府県は「1列目に県名」が入っている
pref_list = [
    "青森","岩手","宮城","秋田","山形","福島","茨城","栃木","群馬",
    "埼玉","千葉","東京","神奈川","山梨","新潟","富山","石川","長野",
    "福井","岐阜","静岡","愛知","三重","滋賀","京都","大阪","奈良",
    "和歌山","兵庫","鳥取","島根","岡山","広島","山口","徳島","香川",
    "愛媛","高知","福岡","佐賀","長崎","熊本","大分","宮崎","鹿児島"
]

cars_pref = cars_raw.loc[cars_raw[1].isin(pref_list), [1, 7]].copy()
cars_pref.columns = ["pref_short", "cars_total"]

# 北海道は札幌・函館など7支局の合計を取る
hokkaido_offices = ["札幌","函館","旭川","室蘭","釧路","帯広","北見"]
hokkaido_total = cars_raw.loc[cars_raw[1].isin(hokkaido_offices), 7].sum()

# 沖縄は0列目に名前が入っている
okinawa_total = cars_raw.loc[cars_raw[0].astype(str).str.contains("沖"), 7].iloc[0]

cars_df = pd.concat([
    pd.DataFrame({"pref_short": ["北海道"], "cars_total": [hokkaido_total]}),
    cars_pref,
    pd.DataFrame({"pref_short": ["沖縄"],   "cars_total": [okinawa_total]}),
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

# 中身チェック
print(df.head())
print(df.describe())

df["log_pop"] = np.log(df["population"])

X = df[["elderly_rate", "car_per_1000"]]
X = sm.add_constant(X)  # 切片を入れる

model = sm.GLM(
    df["deaths"],
    X,
    family=sm.families.Poisson(),
    offset=df["log_pop"]  # 人口の対数をオフセットに
)

result = model.fit()
print(result.summary())

print(df)
#print(df.isnull().sum())


import pandas as pd
import numpy as np
import statsmodels.api as sm

#（ここまでは元のコードそのまま）-------------------------------------
# ...（データ読み込み〜GLM Poisson モデルまで同じ）
# ----------------------------------------------------------------------

# ======== ここから追加：過分散チェック（ピアソン残差） ========

# ピアソン残差
pearson_resid = result.resid_pearson

# サンプル数 n、推定パラメータ数 p（切片＋説明変数）
n = df.shape[0]
p = X.shape[1]

# 過分散パラメータ φ（phi）
phi = np.sum(pearson_resid**2) / (n - p)

print("\n==============================")
print("【過分散パラメータ φ（ピアソン）】")
print("==============================")
print(f"phi = {phi:.3f}")

# 解釈の目安
print("\n【解釈】")
if phi < 1.2:
    print("≈ 1 前後 → ポアソンでほぼ問題なし")
elif phi < 1.5:
    print("1.2〜1.5 → やや過分散の可能性")
elif phi < 2:
    print("1.5〜2 → 過分散の可能性が高い")
else:
    print("2 以上 → 強い過分散、負の二項回帰を検討すべき")

# ======= 最後に df を確認 =======
print("\n", df)

pearson_resid = result.resid_pearson

# サンプル数 n、説明変数の数 p（切片を含む）
n = df.shape[0]
p = X.shape[1]

# 過分散パラメータ φ（phi）
phi = np.sum(pearson_resid**2) / (n - p)

print("\n==============================")
print("【過分散パラメータ φ（ピアソン）】")
print("==============================")
print(f"phi = {phi:.3f}")

print("\n【解釈】")
if phi < 1.2:
    print("≈ 1 前後 → ポアソンでほぼ問題なし")
elif 1.2 <= phi < 1.5:
    print("1.2〜1.5 → やや過分散の可能性")
elif 1.5 <= phi < 2:
    print("1.5〜2 → 過分散の可能性が高い")
else:
    print("2 以上 → 強い過分散、負の二項回帰を検討すべき")

# =============================================================
# ここから追加：負の二項回帰（Negative Binomial）を推定
# =============================================================

# Negative Binomial も Poisson と同じ X, offset を使用
nb_model = sm.GLM(
    df["deaths"],
    X,
    family=sm.families.NegativeBinomial(),  # ← NB に変更
    offset=df["log_pop"]
)

nb_result = nb_model.fit()
print("\n==============================")
print("【Negative Binomial 回帰 結果】")
print("==============================")
print(nb_result.summary())

# =============================================================
# AIC 比較（Poisson vs Negative Binomial）
# =============================================================
print("\n==============================")
print("【AIC 比較】")
print("==============================")
print(f"AIC (Poisson)        : {result.aic:.2f}")
print(f"AIC (Neg. Binomial)  : {nb_result.aic:.2f}")

if nb_result.aic + 1e-6 < result.aic:
    print("\n→ **Negative Binomial の方が AIC が小さく、より良いモデルと判断されます。**")
else:
    print("\n→ **Poisson モデルで十分（AIC は NB より良い or ほぼ同じ）。**")

# =============================================================
# df の確認
# =============================================================
print("\n", df)