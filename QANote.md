# Q&A Notes

## Turbulence 指标在两个 Portfolio 文件中的使用情况

**问题**：`fundamental_portfolio_drl.py` 与 `fundamental_portfolio.py` 是否实际应用了 Turbulence（市场动荡指数）？

**结论**：两份脚本目前都 **没有实质性地使用** Turbulence 指标。

| 文件 | 相关代码 | 是否启用 Turbulence | 说明 |
|------|---------|-------------------|------|
| `fundamental_portfolio_drl.py` | ```python
FeatureEngineer(use_turbulence=False)
``` | 否 | `FeatureEngineer` 支持生成 `turbulence` 列，但显式传入 `False`，且后续 `StockPortfolioEnv` 未设置 `turbulence_threshold`，相当于功能关闭。 |
| `fundamental_portfolio.py` | *无* “turbulence” 字样 | 否 | 这是纯 PyPortfolioOpt 的权重优化脚本，仅用价格和预测收益率做均值-方差 / 最小方差 / 等权重优化，未涉及风险指标。 |

**代码出处**：
- `fundamental_portfolio_drl.py` 第 490 行附近：
  ```python
  fe = FeatureEngineer(use_technical_indicator=True,
                       use_turbulence=False,
                       ...)
  ```
- `fundamental_portfolio.py` 全文搜索无 `turbulence` 关键字。

**项目内真正的 Turbulence 计算** 在旧目录 `old_repo_ensemble_strategy/preprocessing/preprocessors.py` 中实现，使用 **马氏距离**：
\[(p_t-\mu)^T \Sigma^{-1}(p_t-\mu)\]
但新的 DRL / 组合优化脚本未复用此流程。

**若需启用 Turbulence 风控**：
1. 在 `FeatureEngineer` 初始化时改为 `use_turbulence=True` 并确保数据含 `turbulence` 列；
2. 在 `StockPortfolioEnv` 创建参数中设置 `turbulence_threshold`，例如 `env_kwargs['turbulence_threshold']=30`；
3. 或在 PyPortfolioOpt 流程中把 Turbulence 加入约束 / 风险惩罚项，需额外编写逻辑。


## 投资绩效常用比率

| 指标 | 公式 | 含义 | 解读 |
|------|------|------|------|
| **索提诺比率 (Sortino Ratio)** | \( S = \frac{R_{ann} - R_f}{\sigma_{d}} \) 其中：<br>• \(R_{ann}\)：年化收益率<br>• \(R_f\)：无风险收益率 (年化)<br>• \(\sigma_{d}\)：**下行波动率** = 只对负收益取平方后求标准差，再年化 | 衡量单位下行风险可获得的超额回报。只关注 downside risk，避免夏普比率对“好波动”的惩罚。 | 数值越高越好；>2 视为非常优异，<0 说明收益落后于无风险利率且承担了下行风险。 |
| **卡尔马比率 (Calmar Ratio)** | \( C = \frac{R_{ann}}{|\text{MaxDrawdown}|} \) | 用最大回撤度量风险，表示每承受 1 单位最大回撤可获得的年化回报。 | 常用在 CTA / 对冲基金；>3 佳，≈1 一般。回撤为 0 时不定义。 |
| **信息比率 (Information Ratio)** | \( IR = \frac{R_{p}-R_{b}}{\sigma_{\text{TE}}} \) 其中：<br>• \(R_{p}\)：组合年化收益<br>• \(R_{b}\)：基准年化收益<br>• \(\sigma_{\text{TE}}\)：跟踪误差 (Tracking Error) = \(\text{std}(R_{p}-R_{b})\) 年化 | 衡量主动管理相对基准的超额收益 / 单位主动风险。 | >0.5 良好，>1 优秀；负值表示跑输基准。 |

> **年化处理**  
> 若数据为季度频率：年化因子 4；若日频：年化因子 252。

**Python 示例**
```python
import numpy as np

def sortino(returns, rf=0.0, periods=252):
    downside = returns[returns < 0].std() * np.sqrt(periods)
    annual_ret = returns.mean() * periods
    return np.nan if downside == 0 else (annual_ret - rf) / downside

def calmar(cum_returns):
    max_dd = (cum_returns.cummax() - cum_returns).max()
    annual_ret = (1 + cum_returns.iloc[-1]) ** (252/len(cum_returns)) - 1
    return np.nan if max_dd == 0 else annual_ret / abs(max_dd)

def information_ratio(returns_p, returns_b, periods=252):
    active = returns_p - returns_b
    tracking_err = active.std() * np.sqrt(periods)
    active_ret = active.mean() * periods
    return np.nan if tracking_err == 0 else active_ret / tracking_err
```

