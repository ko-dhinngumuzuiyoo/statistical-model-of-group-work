# Statistical Model of Group Work

交通事故死者数の統計モデル分析プロジェクト

## 概要

都道府県別の交通事故死者数を、人口、高齢化率、自動車保有台数などの変数で説明する統計モデルを構築します。

## データソース

- `3都道府県別交通事故死者数.csv`: 都道府県別交通事故死者数データ
- `a01100_2.xlsx`: 人口・高齢化率データ（第11表）
- `r5c6pv0000013d12.xlsx`: 自動車保有台数データ

## セットアップ

### 1. 仮想環境の作成とアクティベート

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. 必要なパッケージのインストール

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

## 実行方法

```powershell
python analysis.py
```

または、セットアップと実行を一度に行う場合：

```powershell
.\setup_and_run.ps1
```

## 必要なパッケージ

- pandas
- numpy
- statsmodels
- openpyxl

詳細は `requirements.txt` を参照してください。

## 分析内容

- 目的変数: 2023年の都道府県別交通事故死者数
- 説明変数: 高齢化率、人口千人あたりの自動車保有台数
- モデル: ポアソン回帰モデル（人口の対数をオフセットとして使用）

