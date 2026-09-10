# QHHT 沪铜预测

沪铜(CU) 期货 v5 概率预测 + 微信推送。

## GitHub Actions 云端定时（推荐）

1. 仓库 Settings → Secrets → Actions，添加：
   - `TUSHARE_TOKEN`
   - `PUSHPLUS_TOKEN`（两个微信号用英文逗号连接）
2. Actions → QHHT Daily Forecast → Run workflow 手动测试
3. 每天北京时间 08:30 自动运行（周一至周五）

## 本地运行

```cmd
pip install -r requirements.txt
copy .env.example .env
python run_forecast.py
python daily_forecast.py --refresh
```
