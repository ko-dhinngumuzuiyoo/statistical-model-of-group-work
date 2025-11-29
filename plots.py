import matplotlib.pyplot as plt

# 人口10万人あたり死亡事故率
df["death_rate_per_100k"] = df["deaths"] / df["population"] * 100000

plt.figure()
plt.scatter(df["elderly_rate"] * 100, df["death_rate_per_100k"])
plt.xlabel("高齢化率（％）")
plt.ylabel("人口10万人あたり死亡事故率")
plt.title("高齢化率と人口10万人あたり交通事故死亡率（都道府県別）")
plt.grid(True)

plt.tight_layout()
plt.savefig("fig_elderly_vs_deathrate.png", dpi=300)
# plt.show()  # 対話環境なら表示

plt.figure()
plt.scatter(df["car_per_1000"], df["death_rate_per_100k"])
plt.xlabel("人口1000人あたり自動車台数")
plt.ylabel("人口10万人あたり死亡事故率")
plt.title("クルマ依存度と人口10万人あたり交通事故死亡率（都道府県別）")
plt.grid(True)

plt.tight_layout()
plt.savefig("fig_car_vs_deathrate.png", dpi=300)
# plt.show()

# 予測値（μ_i）
mu_hat = result.predict()  # λ_i の推定値（件数）

# 人口10万人あたりの「予測死亡事故率」
df["fitted_rate_per_100k"] = mu_hat / df["population"] * 100000

plt.figure()
plt.scatter(df["death_rate_per_100k"], df["fitted_rate_per_100k"])
max_val = max(df["death_rate_per_100k"].max(), df["fitted_rate_per_100k"].max())

# 理想線（y = x）
plt.plot([0, max_val], [0, max_val])

plt.xlabel("実測の人口10万人あたり死亡事故率")
plt.ylabel("予測された人口10万人あたり死亡事故率")
plt.title("実測値 vs 予測値（ポアソン回帰）")
plt.grid(True)

plt.tight_layout()
plt.savefig("fig_observed_vs_fitted.png", dpi=300)
# plt.show()
