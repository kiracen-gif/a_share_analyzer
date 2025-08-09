# A股一键分析 · 在线版（V1）

零代码的股票分析工作流：输入 A 股代码 → 自动拉取数据 → 计算指标 → 打分 → 给出操作建议，并可融合行业/政策/估值打分。

## 本地运行
1. 安装 Python 3.10+
2. `pip install -r requirements.txt`
3. 设置 Tushare Token：
   - 方式1：环境变量 `export TUSHARE_TOKEN=你的token`
   - 方式2：在运行时在侧边栏输入
4. 运行：`streamlit run app.py`

## 部署到 Streamlit Cloud
1. 将本目录推到 GitHub 仓库
2. 在 Streamlit Cloud 创建应用，入口 `app.py`
3. 在 **Secrets** 添加：
   ```
   TUSHARE_TOKEN="你的token"
   ```
4. 保存部署，打开在线链接即可使用。

## 说明
- 技术指标：MA20/50/200、ROC20、ATR14、52周高低
- 技术面打分：-100~+100（偏趋势风格）
- 行业/政策/公司相对/估值：V1 支持 **手动打分**（未来可接入自动化来源）
- 综合评分：技术×60% + 其它标准化×40%
- 风险控制：止损=收盘-2×ATR14；仓位=资金×风险预算/(入场-止损)

> 仅用于研究参考，不构成投资建议。