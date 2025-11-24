# UTF-8エンコーディング設定
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "仮想環境を作成中..." -ForegroundColor Green
python -m venv venv

if ($LASTEXITCODE -ne 0) {
    Write-Host "エラー: 仮想環境の作成に失敗しました" -ForegroundColor Red
    exit 1
}

Write-Host "仮想環境をアクティベート中..." -ForegroundColor Green
& .\venv\Scripts\Activate.ps1

Write-Host "必要なパッケージをインストール中..." -ForegroundColor Green
python -m pip install --upgrade pip
python -m pip install pandas numpy statsmodels openpyxl

Write-Host ""
Write-Host "スクリプトを実行中..." -ForegroundColor Green
python analysis.py

Write-Host ""
Write-Host "完了しました。Enterキーを押して終了してください。"
Read-Host

