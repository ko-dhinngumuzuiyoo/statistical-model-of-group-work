import os
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 図保存用ディレクトリ
os.makedirs("figures", exist_ok=True)

# 負の二項回帰のピアソン残差と予測値
pearson_nb = nb_glm_result.resid_pearson
fitted_nb  = nb_glm_result.fittedvalues

# 1. ピアソン残差 vs 予測値（負の二項回帰）
plt.figure(figsize=(6, 5))
plt.scatter(fitted_nb, pearson_nb, alpha=0.7)
plt.axhline(y=0, linestyle="--")
plt.title("ピアソン残差 vs 予測値（負の二項回帰）")
plt.xlabel("予測値 (Fitted Values: $\hat{\\mu}$)")
plt.ylabel("ピアソン残差 (Pearson Residuals)")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("figures/nb_pearson_residuals_vs_fitted.png", dpi=300, bbox_inches="tight")
plt.close()

# 2. ピアソン残差の Q-Q プロット（負の二項回帰）
fig = sm.qqplot(pearson_nb, line="45", fit=True)
fig.suptitle("ピアソン残差の Q-Qプロット（負の二項回帰）", fontsize=14)
plt.tight_layout()
plt.savefig("figures/nb_qqplot_pearson_residuals.png", dpi=300, bbox_inches="tight")
plt.close()
