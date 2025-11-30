import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt # 追加: 可視化のため

# Matplotlibの日本語対応設定 (Windows/Mac/Linux環境に合わせて調整してください)
plt.rcParams['font.family'] = 'Meiryo' # 例: Windows環境のフォント
plt.rcParams['axes.unicode_minus'] = False # マイナス記号を正しく表示

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

# =========================
# 5. 新しい変数：10万人あたり死亡率の計算 (追加)
# =========================
df["deaths_per_100k"] = (df["deaths"] / df["population"]) * 100000

# 中身チェック
print("### データフレーム先頭 5行と統計量 ###")
print(df.head())
print(df.describe())

# =========================
# 6. 相関係数の確認 (追加)
# =========================
print("\n### 主要変数の相関係数 (corr()) ###")
correlation_matrix = df[["deaths_per_100k", "elderly_rate", "car_per_1000"]].corr()
print(correlation_matrix)

# =========================
# 7. 散布図の作成 (追加)
# =========================
print("\n### 散布図の作成中... ###")

# 図の全体サイズを設定
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('交通事故死亡率と関連要因の分析', fontsize=16)

# --- 散布図 1: 高齢化率 vs 死亡率 ---
axes[0].scatter(df["elderly_rate"] * 100, df["deaths_per_100k"], alpha=0.7)
axes[0].set_title('① 高齢化率 vs 10万人あたり死亡率')
axes[0].set_xlabel('高齢化率 (%)')
axes[0].set_ylabel('10万人あたり死亡率')
axes[0].grid(True, linestyle='--', alpha=0.6)

# --- 散布図 2: 自動車台数 vs 死亡率 ---
axes[1].scatter(df["car_per_1000"], df["deaths_per_100k"], alpha=0.7, color='green')
axes[1].set_title('② 自動車台数 vs 10万人あたり死亡率')
axes[1].set_xlabel('人口千人あたり自動車台数')
axes[1].set_ylabel('10万人あたり死亡率')
axes[1].grid(True, linestyle='--', alpha=0.6)

# --- 散布図 3: 高齢化率 vs 自動車台数 ---
axes[2].scatter(df["elderly_rate"] * 100, df["car_per_1000"], alpha=0.7, color='red')
axes[2].set_title('③ 高齢化率 vs 自動車台数')
axes[2].set_xlabel('高齢化率 (%)')
axes[2].set_ylabel('人口千人あたり自動車台数')
axes[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # タイトルと図の重なりを調整
plt.show()

# =========================
# 8. ポアソン回帰の推定 (元のコード)
# =========================
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
print("\n### GLM (Poisson Regression) 結果 ###")
print(result.summary())