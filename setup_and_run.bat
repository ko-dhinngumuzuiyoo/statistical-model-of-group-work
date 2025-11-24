@echo off
chcp 65001 >nul
echo 仮想環境を作成中...
python -m venv venv
if errorlevel 1 (
    echo エラー: 仮想環境の作成に失敗しました
    pause
    exit /b 1
)

echo 仮想環境をアクティベート中...
call venv\Scripts\activate.bat

echo 必要なパッケージをインストール中...
pip install --upgrade pip
pip install pandas numpy statsmodels openpyxl

echo.
echo スクリプトを実行中...
python analysis.py

pause

