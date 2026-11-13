import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, lasso_path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

# ==================== 1. 全局页面配置与工业级 UI 样式 ====================
st.set_page_config(
    page_title="盾构推力预测轻量化智能体",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

st.markdown("""
<style>
    .hero-banner {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 55%, #415a77 100%);
        color: #ffffff; padding: 24px 30px; border-radius: 12px; margin-bottom: 20px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    }
    .hero-banner h1 { color: #ffffff; font-size: 26px; font-weight: 700; margin: 0 0 8px 0; }
    .hero-banner p { color: #e0e1dd; font-size: 14.5px; margin: 0; line-height: 1.5; }
    .diag-box {
        background-color: #f8f9fa; border-left: 5px solid #2b5876;
        padding: 14px 18px; border-radius: 8px; margin-bottom: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .diag-header { font-weight: 700; color: #1b263b; font-size: 15px; margin-bottom: 5px; }
    .diag-content { color: #33415c; font-size: 13.5px; line-height: 1.6; }
    .badge-order {
        display: inline-block; background: #e0fbfc; color: #005f73;
        border: 1px solid #94d2bd; padding: 4px 12px; border-radius: 16px;
        font-weight: 600; font-size: 13.5px; margin-right: 6px; margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ==================== 2. 随机工程数据合成器 ====================
class RandomTBMDataGenerator:
    """支持自定义样本量、地质突变比例与噪声水平的随机盾构工程数据生成器"""

    @staticmethod
    def generate(n_rings: int = 240, mutation_ratio: float = 0.25, noise_level: float = 0.04, seed: int = 2026) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rings = np.arange(1, n_rings + 1)

        n_m = int(n_rings * mutation_ratio)
        n_u = n_rings - n_m

        # 常规地层段与地质突变段参数随机生成
        E_normal = rng.uniform(5.0, 14.0, n_u)
        E_mutate = rng.uniform(22.0, 68.0, n_m) if n_m > 0 else np.array([])
        E = np.concatenate([E_normal, E_mutate])

        D = np.full(n_rings, rng.choice([6.28, 6.34, 6.40]))
        H = np.clip(np.linspace(rng.uniform(8.5, 10.5), rng.uniform(12.0, 16.0), n_rings) + rng.normal(0, 0.3, n_rings), 6.0, 25.0)
        v = rng.uniform(16.0, 52.0, n_rings)
        omega = np.concatenate([rng.uniform(0.6, 1.1, n_u), rng.uniform(1.0, 1.45, n_m)]) if n_m > 0 else rng.uniform(0.6, 1.1, n_rings)
        pe = rng.uniform(135.0, 310.0, n_rings)
        mu = np.concatenate([rng.uniform(0.28, 0.37, n_u), rng.uniform(0.20, 0.27, n_m)]) if n_m > 0 else rng.uniform(0.28, 0.37, n_rings)
        K0 = np.concatenate([rng.uniform(0.48, 0.58, n_u), rng.uniform(0.36, 0.46, n_m)]) if n_m > 0 else rng.uniform(0.48, 0.58, n_rings)
        phi = np.concatenate([rng.uniform(20.0, 28.0, n_u), rng.uniform(30.0, 39.0, n_m)]) if n_m > 0 else rng.uniform(20.0, 28.0, n_rings)
        gamma = np.concatenate([rng.uniform(18.2, 19.6, n_u), rng.uniform(20.0, 21.8, n_m)]) if n_m > 0 else rng.uniform(18.2, 19.6, n_rings)
        c = np.concatenate([rng.uniform(18.0, 32.0, n_u), rng.uniform(32.0, 52.0, n_m)]) if n_m > 0 else rng.uniform(18.0, 32.0, n_rings)
        W = rng.uniform(170.0, 340.0, n_rings)

        # 物理单位转换后合成基准推力
        E_kPa = E * 1000.0
        v_m = v / 1000.0
        phi_rad = np.deg2rad(phi)

        # 随机采样一组符合物理意义的真实底层系数
        true_theta = np.array([
            rng.uniform(0.65, 1.15),     # Π1: 主动支护
            rng.uniform(1.10, 2.00),     # Π2: 地层自稳
            rng.uniform(-1.60, -0.80),   # Π3: 地应力约束
            rng.uniform(-5.20, -2.80),   # Π4: 黏聚力剪切
            rng.uniform(0.015, 0.055),   # Π5: 内摩擦剪切
            rng.uniform(0.005, 0.035),   # Π6: 土体压缩
            rng.uniform(1.80, 2.90)      # Π7: 岩土本构复合
        ])

        F_mech = (
            true_theta[0] * pe * (D ** 2)
            + true_theta[1] * W * (D ** 2)
            + true_theta[2] * (1.0 + K0) * gamma * H * (D ** 2)
            + true_theta[3] * (c * v_m / omega) * D
            + true_theta[4] * np.tan(phi_rad) * (v_m / omega) * E_kPa * D
            + true_theta[5] * (mu / ((1.0 + mu) * (1.0 - 2.0 * mu))) * (v_m / omega) * E_kPa * D
            + true_theta[6] * (2.0 * c * (1.0 + mu) / np.tan(phi_rad)) * (D ** 2)
        )
        F_base = np.abs(F_mech) + 8000.0 * (np.abs(F_mech) < 6000.0)

        # 在突变段叠加未建模外部阻力（如盾壳摩擦/刀具磨损）与非线性扰动以检验六维偏差归因
        unmodeled_resistance = np.concatenate([np.zeros(n_u), rng.uniform(900.0, 2400.0, n_m)]) if n_m > 0 else np.zeros(n_rings)
        nonlinear_term = 0.04 * (pe / E) ** 2 * (D ** 2)
        random_noise = rng.normal(0, noise_level * np.mean(F_base), n_rings)

        F = np.clip(F_base + unmodeled_resistance + nonlinear_term + random_noise, 6000.0, 32000.0)

        df = pd.DataFrame({
            'Ring': rings, 'E': np.round(E, 3), 'D': D, 'omega': np.round(omega, 3),
            'H': np.round(H, 2), 'v': np.round(v, 2), 'mu': np.round(mu, 3),
            'K0': np.round(K0, 3), 'phi': np.round(phi, 2), 'pe': np.round(pe, 2),
            'gamma': np.round(gamma, 2), 'c': np.round(c, 2), 'W': np.round(W, 2),
            'F': np.round(F, 1)
        })

        # 注入少量缺失值与物理越界异常值以验证数据清洗功能
        if n_rings >= 40:
            na_indices = rng.choice(n_rings - 5, size=3, replace=False)
            df.loc[na_indices[0], 'pe'] = np.nan
            df.loc[na_indices[1], 'v'] = np.nan
            df.loc[na_indices[2], 'c'] = np.nan
            df.loc[int(n_rings * 0.2), 'mu'] = 0.62   # 泊松比越界 (>=0.5)
            df.loc[int(n_rings * 0.5), 'E'] = -2.5    # 弹性模量越界 (<=0)

        return df


# ==================== 3. 力学逻辑引导建模与六维归因核心智能体 ====================
class TBMIntelligentAgent:
    REQUIRED_COLS = ['E', 'D', 'omega', 'H', 'v', 'mu', 'K0', 'phi', 'pe', 'gamma', 'c', 'W', 'F']
    PI_NAMES = ['Π1', 'Π2', 'Π3', 'Π4', 'Π5', 'Π6', 'Π7']
    PI_INFO = [
        ("Π1", "开挖面稳定控制", "pe / E", "开挖面土舱压力主动支护效应"),
        ("Π2", "开挖面稳定控制", "W / E", "周围地层三维应力自稳承载能力"),
        ("Π3", "初始地应力与埋深", "(1+K0)γH / E", "垂直与水平环境应力场约束效应"),
        ("Π4", "刀盘-土体动态交互", "(c/E)·(v/ωD)", "黏聚力主导的掘进土体剪切效应"),
        ("Π5", "刀盘-土体动态交互", "tan(φ)·(v/ωD)", "内摩擦角主导的掘进土体剪切效应"),
        ("Π6", "刀盘-土体动态交互", "[μ/((1+μ)(1-2μ))]·(v/ωD)", "刀盘推进挤压引起的土体压缩效应"),
        ("Π7", "岩土本构复合参数", "2c(1+μ) / (E·tanφ)", "剪压复合破坏下的土体材料本构特征")
    ]

    def __init__(self, raw_df: pd.DataFrame, test_size: float = 0.25, cv_folds: int = 5):
        self.raw_df = raw_df.copy()
        self.test_size = test_size
        self.cv_folds = cv_folds

    def execute(self):
        # Step 1: 特征完整性检查
        missing = [c for c in self.REQUIRED_COLS if c not in self.raw_df.columns]
        if missing:
            raise ValueError(f"数据表缺失以下核心物理特征列：{missing}")

        df = self.raw_df.copy()
        if 'Ring' not in df.columns:
            df['Ring'] = np.arange(1, len(df) + 1)

        # Step 2: 缺失值统计与线性插值修复
        na_counts = df[self.REQUIRED_COLS].isna().sum().to_dict()
        total_na = int(sum(na_counts.values()))
        df[self.REQUIRED_COLS] = df[self.REQUIRED_COLS].interpolate(method='linear').bfill().ffill()

        # Step 3: 物理边界约束异常值拦截
        valid_mask = (
            (df['E'] > 0) & (df['D'] > 0) & (df['omega'] > 0) &
            (df['mu'] > 0) & (df['mu'] < 0.5) &
            (df['phi'] > 0) & (df['phi'] < 90) & (df['F'] > 0)
        )
        outlier_rings = df.loc[~valid_mask, 'Ring'].tolist()
        df_clean = df[valid_mask].reset_index(drop=True)

        # Step 4: 国际标准单位制 (SI) 强制统一
        df_clean['E_kPa'] = df_clean['E'] * 1000.0          # MPa -> kPa
        df_clean['v_m_min'] = df_clean['v'] / 1000.0        # mm/min -> m/min
        df_clean['phi_rad'] = np.deg2rad(df_clean['phi'])   # deg -> rad

        self.clean_report = {
            "raw_rows": len(self.raw_df),
            "clean_rows": len(df_clean),
            "total_na_filled": total_na,
            "na_detail": {k: v for k, v in na_counts.items() if v > 0},
            "outlier_count": len(outlier_rings),
            "outlier_rings": outlier_rings
        }
        self.df_clean = df_clean

        # Step 5: 构造基础无量纲 π 群与 7 个主导力学因子 Π1~Π7
        E, D, w = df_clean['E_kPa'], df_clean['D'], df_clean['omega']
        v, mu, phi = df_clean['v_m_min'], df_clean['mu'], df_clean['phi_rad']

        self.basic_pi = pd.DataFrame({
            'π1(H/D)': df_clean['H'] / D,
            'π2(v/ωD)': v / (w * D),
            'π3(μ)': mu,
            'π4(K0)': df_clean['K0'],
            'π5(φ)': phi,
            'π6(pe/E)': df_clean['pe'] / E,
            'π7(γD/E)': df_clean['gamma'] * D / E,
            'π8(c/E)': df_clean['c'] / E,
            'π9(W/E)': df_clean['W'] / E,
            'πF(F/ED²)': df_clean['F'] / (E * (D ** 2))
        })

        X_pi = pd.DataFrame(index=df_clean.index)
        X_pi['Π1'] = df_clean['pe'] / E
        X_pi['Π2'] = df_clean['W'] / E
        X_pi['Π3'] = (1.0 + df_clean['K0']) * df_clean['gamma'] * df_clean['H'] / E
        X_pi['Π4'] = (df_clean['c'] / E) * (v / (w * D))
        X_pi['Π5'] = np.tan(phi) * (v / (w * D))
        X_pi['Π6'] = (mu / ((1.0 + mu) * (1.0 - 2.0 * mu))) * (v / (w * D))
        X_pi['Π7'] = (2.0 * df_clean['c'] * (1.0 + mu)) / (E * np.tan(phi))
        y_pi = df_clean['F'] / (E * (D ** 2))

        self.X_pi, self.y_pi = X_pi, y_pi

        # Step 6: 训练/测试集划分与加性线性 LassoCV 辨识
        idx_train, idx_test = train_test_split(df_clean.index, test_size=self.test_size, random_state=42)
        self.idx_train, self.idx_test = idx_train, idx_test
        X_train, X_test = X_pi.loc[idx_train], X_pi.loc[idx_test]
        y_train, y_test = y_pi.loc[idx_train], y_pi.loc[idx_test]

        col_std = X_train.std(axis=0).replace(0, 1.0).values
        X_train_norm = X_train.values / col_std
        X_test_norm = X_test.values / col_std

        alphas = np.logspace(-7, -1, 160)
        model = LassoCV(alphas=alphas, fit_intercept=False, cv=self.cv_folds, max_iter=50000)
        model.fit(X_train_norm, y_train.values)

        self.best_lambda = model.alpha_
        self.theta_norm = model.coef_
        self.theta_phys = model.coef_ / col_std
        self.col_std = col_std

        # Step 7: Lasso 路径追踪
        alphas_path, coef_path_norm, _ = lasso_path(X_train_norm, y_train.values, alphas=alphas)
        self.alphas_path = alphas_path
        self.coef_path_norm = coef_path_norm
        pred_path = X_train_norm @ coef_path_norm
        self.mse_path = np.mean((pred_path - y_train.values[:, None]) ** 2, axis=0)
        self.n_feat_path = np.sum(np.abs(coef_path_norm) > 1e-7, axis=0)

        entry_step = [np.argmax(np.abs(coef_path_norm[i, :]) > 1e-7) for i in range(7)]
        self.entry_order = [self.PI_NAMES[i] for i in np.argsort(entry_step)]

        # Step 8: 还原推力预测值并计算评价指标 (R2, MAPE, RMSE)
        scale_tr = df_clean.loc[idx_train, 'E_kPa'].values * (df_clean.loc[idx_train, 'D'].values ** 2)
        scale_te = df_clean.loc[idx_test, 'E_kPa'].values * (df_clean.loc[idx_test, 'D'].values ** 2)

        self.F_train_true = df_clean.loc[idx_train, 'F'].values
        self.F_train_pred = (X_train.values @ self.theta_phys) * scale_tr
        self.F_test_true = df_clean.loc[idx_test, 'F'].values
        self.F_test_pred = (X_test.values @ self.theta_phys) * scale_te

        self.metrics_train = self._eval_metrics(self.F_train_true, self.F_train_pred)
        self.metrics_test = self._eval_metrics(self.F_test_true, self.F_test_pred)

        # Step 9: 六维偏差定量归因分析
        self.diag = self._analyze_deviations()

    @staticmethod
    def _eval_metrics(y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return {"R2": r2, "MAPE": mape, "RMSE": rmse}

    def get_plain_formula(self) -> str:
        t = self.theta_phys
        return (
            f"F = {t[0]:.4f}*pe*D^2 + ({t[1]:.4f})*W*D^2 + ({t[2]:.4f})*(1+K0)*γ*H*D^2 "
            f"+ ({t[3]:.4f})*(c*v/ω)*D + ({t[4]:.5f})*tan(φ)*(v/ω)*E*D "
            f"+ ({t[5]:.5f})*[μ/((1+μ)(1-2μ))]*(v/ω)*E*D + ({t[6]:.4f})*[2c(1+μ)/tan(φ)]*D^2"
        )

    def _analyze_deviations(self):
        err = self.F_test_pred - self.F_test_true
        rel_err = err / self.F_test_true
        abs_rel_err = np.abs(rel_err)

        # 1. 数据质量指标
        anomaly_rate = (self.clean_report['total_na_filled'] + self.clean_report['outlier_count']) / max(1, self.clean_report['raw_rows']) * 100.0
        v_cv = self.df_clean['v'].std() / self.df_clean['v'].mean() * 100.0

        # 2. 工况外推程度 (3D PCA 物理特征空间包络检测)
        pca3 = PCA(n_components=3)
        Z_tr = pca3.fit_transform(self.X_pi.loc[self.idx_train].values / self.col_std)
        Z_te = pca3.transform(self.X_pi.loc[self.idx_test].values / self.col_std)
        center = Z_tr.mean(axis=0, keepdims=True)
        r95 = np.percentile(cdist(Z_tr, center), 95)
        dist_te = cdist(Z_te, center).ravel()
        extrap_mask = dist_te > r95
        extrap_rate = np.mean(extrap_mask) * 100.0

        # 3. 样本规模指标
        n_train = len(self.idx_train)
        active_factors = int(np.sum(np.abs(self.theta_norm) > 1e-6))

        # 4. 线性假设检验 (残差与二阶非线性项相关性)
        quad_r, _ = pearsonr(self.F_test_pred ** 2, -err)

        # 5. 无量纲因子机制覆盖 (偏大 vs 偏小占比与未建模阻力偏移)
        over_mask = rel_err > 0.06     # 预测显著偏大 (> +6%)
        under_mask = rel_err < -0.06   # 预测显著偏小 (< -6%, 即实测推力更大)
        over_rate = np.mean(over_mask) * 100.0
        under_rate = np.mean(under_mask) * 100.0

        # 6. 地质突变检测 (相邻环弹性模量梯度)
        E_grad_all = np.abs(self.df_clean['E'].diff() / self.df_clean['E']).fillna(0)
        E_grad_te = E_grad_all.loc[self.idx_test].values
        jump_r, _ = pearsonr(E_grad_te, abs_rel_err)

        # 逐环归因标签生成
        reasons = []
        for i in range(len(rel_err)):
            r_tags = []
            if abs_rel_err[i] <= 0.05:
                reasons.append("✅ 误差在正常范围内 (≤5%)")
                continue
            if extrap_mask[i]:
                r_tags.append("工况外推越界")
            if E_grad_te[i] > 0.20:
                r_tags.append("地层模量突变")
            if rel_err[i] < -0.06:
                r_tags.append("预测偏小(未覆盖盾壳摩阻/结泥饼)")
            elif rel_err[i] > 0.06:
                r_tags.append("预测偏大(线性假设截断/参数阶跃)")
            reasons.append(" | ".join(r_tags) if r_tags else "操作参数高频波动")

        detail_df = self.df_clean.loc[self.idx_test, ['Ring', 'E', 'H', 'v', 'omega', 'pe', 'F']].copy()
        detail_df['预测推力 F_pred (kN)'] = np.round(self.F_test_pred, 1)
        detail_df['绝对偏差 (kN)'] = np.round(err, 1)
        detail_df['相对偏差 (%)'] = np.round(rel_err * 100.0, 2)
        detail_df['偏差主导原因诊断'] = reasons
        detail_df = detail_df.sort_values('Ring').reset_index(drop=True)

        return {
            "anomaly_rate": anomaly_rate,
            "v_cv": v_cv,
            "extrap_rate": extrap_rate,
            "n_train": n_train,
            "active_factors": active_factors,
            "quad_r": abs(quad_r),
            "over_rate": over_rate,
            "under_rate": under_rate,
            "jump_r": abs(jump_r),
            "Z_tr": Z_tr,
            "Z_te": Z_te,
            "extrap_mask": extrap_mask,
            "detail_df": detail_df
        }


# ==================== 4. 侧边栏与主界面交互逻辑 ====================
st.markdown("""
<div class="hero-banner">
    <h1>🛡️ 力学逻辑引导的盾构推力预测轻量化智能体</h1>
    <p>自动化数据治理 · 无量纲力学参量构造 · 加性线性正则化参数辨识 · Lasso路径追踪 · 六维偏差定量归因与辅助决策</p>
</div>
""", unsafe_allow_html=True)

if "rand_seed" not in st.session_state:
    st.session_state.rand_seed = 2026

with st.sidebar:
    st.header("🎛️ 智能体数据与模型配置")
    source_mode = st.radio(
        "1. 选择工程数据接入方式：",
        ["🎲 随机生成一组盾构工程数据", "📂 上传本地真实工程数据表"]
    )

    if source_mode == "🎲 随机生成一组盾构工程数据":
        st.caption("可自定义随机生成的样本规模、地质突变比例与噪声水平：")
        n_rings_input = st.slider("生成掘进总环数 (样本规模)", 80, 600, 240, 20)
        mutation_input = st.slider("地质突变区段占比", 0.0, 0.50, 0.25, 0.05)
        noise_input = st.slider("传感器测量噪声水平", 0.01, 0.10, 0.04, 0.01)

        if st.button("🔄 点击重新随机生成一组新数据", use_container_width=True):
            st.session_state.rand_seed = int(np.random.randint(1000, 99999))

        raw_df = RandomTBMDataGenerator.generate(
            n_rings=n_rings_input,
            mutation_ratio=mutation_input,
            noise_level=noise_input,
            seed=st.session_state.rand_seed
        )
        st.success(f"已生成随机数据集 (随机种子: {st.session_state.rand_seed})")
        csv_bytes = raw_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "⬇️ 下载当前随机生成的数据表 (.csv)",
            data=csv_bytes,
            file_name=f"random_tbm_data_seed{st.session_state.rand_seed}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        uploaded = st.file_uploader("上传包含 13 个核心字段的 CSV 或 XLSX", type=["csv", "xlsx"])
        template_df = RandomTBMDataGenerator.generate(n_rings=20, seed=100)
        st.download_button(
            "📄 下载标准表头数据模板 (.csv)",
            data=template_df.to_csv(index=False).encode('utf-8-sig'),
            file_name="tbm_standard_template.csv",
            mime="text/csv",
            use_container_width=True
        )
        if uploaded is not None:
            raw_df = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
        else:
            st.info("👈 请上传您的 Excel/CSV 数据文件，或切换至【随机生成一组盾构工程数据】。")
            st.stop()

    st.divider()
    st.subheader("2. 正则化学习参数设置")
    test_ratio = st.slider("测试集划分比例", 0.15, 0.40, 0.25, 0.05)
    cv_folds = st.selectbox("Lasso 交叉验证折数 (CV)", [3, 5, 10], index=1)

# 运行智能体核心流水线
try:
    agent = TBMIntelligentAgent(raw_df, test_size=test_ratio, cv_folds=cv_folds)
    agent.execute()
except Exception as e:
    st.error(f"❌ 数据检查未通过：{e}")
    st.stop()

# 顶部核心指标概览
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("有效掘进环样本", f"{agent.clean_report['clean_rows']} 环", f"修复缺失{agent.clean_report['total_na_filled']}处 / 剔除异常{agent.clean_report['outlier_count']}环")
m2.metric("测试集决定系数 R²", f"{agent.metrics_test['R2']:.4f}", f"训练集 R²: {agent.metrics_train['R2']:.4f}")
m3.metric("测试集平均误差 MAPE", f"{agent.metrics_test['MAPE']:.2f}%", f"训练集 MAPE: {agent.metrics_train['MAPE']:.2f}%")
m4.metric("测试集均方根误差 RMSE", f"{agent.metrics_test['RMSE']:.1f} kN", f"训练集 RMSE: {agent.metrics_train['RMSE']:.1f} kN")
m5.metric("最优正则化超参数 λ*", f"{agent.best_lambda:.2e}", f"激活 {agent.diag['active_factors']}/7 个无量纲因子")

# 五大功能模块标签页
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1️⃣ 数据预处理与单位统一",
    "2️⃣ 无量纲力学参量构造",
    "3️⃣ 参数识别与 Lasso 路径分析",
    "4️⃣ 实测对比与六维偏差归因",
    "5️⃣ 施工辅助决策推力预演台"
])

# -------------------- Tab 1: 数据预处理 --------------------
with tab1:
    st.subheader("模块 1：特征完整性检查、缺失/异常值修复与国际标准单位统一")
    c_left, c_right = st.columns([1.15, 1.85])
    with c_left:
        st.markdown("##### （1）数据质量自动化体检报告")
        st.success("✅ **特征完整性校验通过**：已集齐全部 12 项输入特征与目标推力 $F$。")
        st.info(
            f"📊 **预处理执行明细**：\n"
            f"- 原始输入总环数：**{agent.clean_report['raw_rows']} 环**\n"
            f"- 缺失值线性插值修复：**{agent.clean_report['total_na_filled']} 处**（明细：{agent.clean_report['na_detail'] or '无'}）\n"
            f"- 物理越界异常环剔除：**{agent.clean_report['outlier_count']} 环**（剔除环号：{agent.clean_report['outlier_rings'] or '无'}，拦截规则：泊松比 $\\mu \\in (0, 0.5)$、模量 $E > 0$、转速 $\\omega > 0$）\n"
            f"- 最终合格建模样本：**{agent.clean_report['clean_rows']} 环**"
        )
        st.markdown("##### （2）物理单位强制统一转换标准")
        unit_table = pd.DataFrame({
            "变量符号": ["E", "pe, c, W", "D, H", "v", "omega", "phi", "gamma", "F"],
            "原始单位": ["MPa", "kPa", "m", "mm/min", "rev/min", "度 (°)", "kN/m³", "kN"],
            "统一后标准单位": ["kPa (×1000)", "kPa", "m", "m/min (÷1000)", "rev/min", "弧度 (rad)", "kN/m³", "kN"],
            "基本量纲 [M,L,T]": ["M¹L⁻¹T⁻²", "M¹L⁻¹T⁻²", "L¹", "L¹T⁻¹", "T⁻¹", "无量纲", "M¹L⁻²T⁻²", "M¹L¹T⁻²"]
        })
        st.dataframe(unit_table, use_container_width=True, hide_index=True)

    with c_right:
        st.markdown("##### （3）清洗后工程特征统计分布表")
        st.dataframe(
            agent.df_clean[agent.REQUIRED_COLS].describe().T[['count', 'mean', 'std', 'min', '50%', 'max']].style.format("{:.2f}"),
            use_container_width=True
        )
        st.markdown("##### （4）清洗并统一单位后的数据集前 10 环预览")
        st.dataframe(agent.df_clean[['Ring'] + agent.REQUIRED_COLS].head(10).style.format("{:.2f}"), use_container_width=True, hide_index=True)

# -------------------- Tab 2: 无量纲参量构造 --------------------
with tab2:
    st.subheader("模块 2：基于量纲分析与主导力学机理的无量纲参量自动构造")
    st.markdown("以岩土弹性模量 $E$、刀盘直径 $D$、刀盘转速 $\\omega$ 为基本重复变量，将 12 个有单位工程特征重组为 **7 个物理意义明确的无量纲自变量 $\\Pi_1 \\sim \\Pi_7$** 与 **无量纲推力 $\\Pi_F = F / (ED^2)$**：")

    info_df = pd.DataFrame(agent.PI_INFO, columns=["因子符号", "所属力学控制机制", "无量纲数学定义", "工程物理含义"])
    st.dataframe(info_df, use_container_width=True, hide_index=True)

    col_p1, col_p2 = st.columns([1.4, 1.0])
    with col_p1:
        st.markdown("##### 自动构造的 7 个无量纲力学因子 ($\\Pi_1 \\sim \\Pi_7$) 与无量纲推力 ($\\Pi_F$) 数值表")
        st.dataframe(agent.X_pi.assign(Π_F=agent.y_pi).head(12).style.format("{:.5e}"), use_container_width=True)
    with col_p2:
        st.markdown("##### 无量纲力学参量相关性矩阵热力图")
        fig_corr, ax_corr = plt.subplots(figsize=(6, 4.5))
        corr_mat = agent.X_pi.assign(Π_F=agent.y_pi).corr().values
        im_c = ax_corr.imshow(corr_mat, cmap='coolwarm', vmin=-1, vmax=1)
        labels_c = agent.PI_NAMES + ['Π_F']
        ax_corr.set_xticks(range(8)); ax_corr.set_xticklabels(labels_c)
        ax_corr.set_yticks(range(8)); ax_corr.set_yticklabels(labels_c)
        for i in range(8):
            for j in range(8):
                ax_corr.text(j, i, f"{corr_mat[i, j]:.2f}", ha='center', va='center', fontsize=8, color='black')
        plt.colorbar(im_c, ax=ax_corr, fraction=0.046, pad=0.04)
        plt.tight_layout()
        st.pyplot(fig_corr)

# -------------------- Tab 3: 参数识别与 Lasso 路径 --------------------
with tab3:
    st.subheader("模块 3：加性线性力学框架参数识别、超参数优化与 Lasso 路径分析")

    t = agent.theta_phys
    st.markdown("##### （1）自动识别输出的显式盾构推力预测表达式")
    st.latex(
        rf"F = {t[0]:.3f}\,p_e D^2 + ({t[1]:.3f})\,W D^2 + ({t[2]:.3f})\,(1+K_0)\gamma H D^2 "
        rf"+ ({t[3]:.3f})\,\frac{{cv}}{{\omega}}D + ({t[4]:.4f})\,\tan\varphi\frac{{v}}{{\omega}}ED "
        rf"+ ({t[5]:.4f})\,\frac{{\mu}}{{(1+\mu)(1-2\mu)}}\frac{{v}}{{\omega}}ED "
        rf"+ ({t[6]:.3f})\,\frac{{2c(1+\mu)}}{{\tan\varphi}}D^2"
    )
    st.code(agent.get_plain_formula(), language="text")

    st.markdown("##### （2）Lasso 路径特征激活顺序（反映当前工况下的主导力学机制优先级）")
    badges_html = "".join([f'<span class="badge-order">第 {idx+1} 顺位：{name}</span>' for idx, name in enumerate(agent.entry_order)])
    st.markdown(badges_html, unsafe_allow_html=True)

    fig_lasso, (ax_l1, ax_l2) = plt.subplots(1, 2, figsize=(16, 5.0))
    palette = ['#0077b6', '#e63946', '#2a9d8f', '#f4a261', '#8338ec', '#fb8500', '#3a86ff']
    for i in range(7):
        ax_l1.plot(agent.alphas_path, agent.coef_path_norm[i, :], color=palette[i], linewidth=2.0, label=agent.PI_NAMES[i])
    ax_l1.axvline(agent.best_lambda, color='#d90429', linestyle='--', linewidth=1.8, label=f'最优 λ* = {agent.best_lambda:.2e}')
    ax_l1.set_xscale('log'); ax_l1.invert_xaxis()
    ax_l1.set_xlabel("正则化超参数 λ (由强惩罚到弱惩罚)")
    ax_l1.set_ylabel("标准化无量纲因子权重系数")
    ax_l1.set_title("(a) Lasso 正则化路径：无量纲因子权重演化轨迹")
    ax_l1.legend(ncol=2, fontsize=9)
    ax_l1.grid(True, linestyle='--', alpha=0.4)

    ax_l2_twin = ax_l2.twinx()
    ax_l2.plot(agent.alphas_path, agent.mse_path, 'b-', linewidth=2.0, label='均方误差 (MSE)')
    ax_l2.axvline(agent.best_lambda, color='#d90429', linestyle='--', linewidth=1.8, label=f'最优 λ* = {agent.best_lambda:.2e}')
    ax_l2_twin.step(agent.alphas_path, agent.n_feat_path, where='post', color='#2b9348', linewidth=2.0, label='激活特征数')
    ax_l2.set_xscale('log'); ax_l2.invert_xaxis()
    ax_l2.set_xlabel("正则化超参数 λ (由强惩罚到弱惩罚)")
    ax_l2.set_ylabel("无量纲均方误差 MSE", color='b')
    ax_l2_twin.set_ylabel("激活的无量纲力学因子数量", color='#2b9348')
    ax_l2_twin.set_yticks(range(0, 8))
    ax_l2.set_title("(b) 正则化超参数寻优曲线与入选特征阶跃图")
    ax_l2.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig_lasso)

# -------------------- Tab 4: 实测对比与六维偏差归因 --------------------
with tab4:
    st.subheader("模块 4：实测推力对比分析与六大因素偏差归因诊断")

    # 1. 实测与预测推力对比图 + 3D PCA 特征空间图
    fig_comp = plt.figure(figsize=(18, 5.2))
    gs_c = gridspec.GridSpec(1, 3, width_ratios=[1.6, 1.0, 1.1], wspace=0.28)

    # 子图1：逐环推力对比与误差柱状图
    ax_c1 = fig_comp.add_subplot(gs_c[0])
    order_te = np.argsort(agent.df_clean.loc[agent.idx_test, 'Ring'].values)
    rings_te = agent.df_clean.loc[agent.idx_test, 'Ring'].values[order_te]
    f_true_te = agent.F_test_true[order_te]
    f_pred_te = agent.F_test_pred[order_te]
    abs_err_te = np.abs(f_true_te - f_pred_te)

    ax_c1_err = ax_c1.twinx()
    ax_c1_err.bar(rings_te, abs_err_te, width=max(1.5, len(rings_te)*0.03), color='#90e0ef', alpha=0.45, label='绝对偏差 |ΔF|')
    ax_c1_err.set_ylabel("绝对误差 Error (kN)", color='#0077b6')
    ax_c1_err.set_ylim(0, max(abs_err_te) * 2.6)

    ax_c1.plot(rings_te, f_true_te, 'o-', color='#d62828', markersize=4, linewidth=1.7, label='实测推力 True')
    ax_c1.plot(rings_te, f_pred_te, 's-', color='#003049', markersize=4, linewidth=1.7, label='预测推力 Pred')
    ax_c1.set_xlabel("掘进环号 (Ring Number)")
    ax_c1.set_ylabel("盾构推力 Thrust (kN)")
    ax_c1.set_title("(a) 测试集实测推力与预测推力逐环对比")
    ax_c1.legend(loc='upper left', fontsize=8.5)
    ax_c1_err.legend(loc='upper right', fontsize=8.5)
    ax_c1.grid(True, linestyle='--', alpha=0.4)

    # 子图2：1:1 对角散点图
    ax_c2 = fig_comp.add_subplot(gs_c[1])
    low_f, high_f = min(f_true_te.min(), f_pred_te.min()) * 0.92, max(f_true_te.max(), f_pred_te.max()) * 1.08
    ax_c2.plot([low_f, high_f], [low_f, high_f], 'k--', linewidth=1.4, label='1:1 理想线')
    ax_c2.fill_between([low_f, high_f], [low_f*0.92, high_f*0.92], [low_f*1.08, high_f*1.08], color='gray', alpha=0.15, label='±8% 误差带')
    ax_c2.scatter(f_true_te, f_pred_te, c='#2a9d8f', edgecolors='k', s=45, alpha=0.85, label='测试集样本')
    ax_c2.set_xlim(low_f, high_f); ax_c2.set_ylim(low_f, high_f)
    ax_c2.set_xlabel("实测推力 True Thrust (kN)")
    ax_c2.set_ylabel("预测推力 Predicted Thrust (kN)")
    ax_c2.set_title(f"(b) 实测 vs 预测散点分布 (R²={agent.metrics_test['R2']:.3f})")
    ax_c2.legend(loc='upper left', fontsize=8.5)
    ax_c2.grid(True, linestyle='--', alpha=0.4)

    # 子图3：3D PCA 物理特征空间工况外推检测
    ax_c3 = fig_comp.add_subplot(gs_c[2], projection='3d')
    Z_tr, Z_te, ext_mask = agent.diag['Z_tr'], agent.diag['Z_te'], agent.diag['extrap_mask']
    ax_c3.scatter(Z_tr[:, 0], Z_tr[:, 1], Z_tr[:, 2], c='#457b9d', s=18, alpha=0.45, label='训练集簇')
    ax_c3.scatter(Z_te[~ext_mask, 0], Z_te[~ext_mask, 1], Z_te[~ext_mask, 2], c='#2a9d8f', s=32, label='正常测试环')
    if np.any(ext_mask):
        ax_c3.scatter(Z_te[ext_mask, 0], Z_te[ext_mask, 1], Z_te[ext_mask, 2], c='#e63946', marker='^', s=55, label='外推越界环')
    ax_c3.set_xlabel("PC1"); ax_c3.set_ylabel("PC2"); ax_c3.set_zlabel("PC3")
    ax_c3.set_title("(c) 3D Π因子特征空间与工况外推检测")
    ax_c3.legend(loc='upper left', fontsize=8)
    st.pyplot(fig_comp)

    # 2. 六维偏差原因深度剖析卡片
    st.markdown("##### （2）结合六大核心因素的推力预测偏差偏大/偏小原因诊断")
    d = agent.diag
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown(f"""
        <div class="diag-box">
            <div class="diag-header">① 数据质量因素 — 缺失/异常修复占比 {d['anomaly_rate']:.1f}% | 掘进速度变异度 {d['v_cv']:.1f}%</div>
            <div class="diag-content">
                <b>偏大/偏小机理</b>：地质勘察参数为钻孔离散采样，而盾构推进速度、土舱压力为连续高频采集，按环号对齐时存在地层参数“阶跃失真”，导致传感器瞬态波动环出现随机偏大或偏小。
            </div>
        </div>
        <div class="diag-box">
            <div class="diag-header">② 工况外推程度 — 3D 物理特征空间外推越界率 {d['extrap_rate']:.1f}%</div>
            <div class="diag-content">
                <b>偏大/偏小机理</b>：在 $\\Pi_1 \\sim \\Pi_7$ 构成的 3D PCA 物理特征空间中，有 <b>{d['extrap_rate']:.1f}%</b> 的测试环超出了训练集 95% 置信包络半径（图中红色三角）。处于外推盲区的样本缺乏同分布训练点约束，易造成系统性偏离。
            </div>
        </div>
        <div class="diag-box">
            <div class="diag-header">③ 样本规模因素 — 训练集样本量 {d['n_train']} 环 | 稀疏激活因子 {d['active_factors']}/7 个</div>
            <div class="diag-content">
                <b>偏大/偏小机理</b>：受限于有限掘进环样本规模，L1 正则化（$\\lambda^* = {agent.best_lambda:.2e}$）通过压缩次要因子权重防止过拟合，但正则化收缩效应会削平极端高推力峰值（导致峰值处<b>预测偏小</b>）并抬高极低推力谷值（导致谷值处<b>预测偏大</b>）。
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_d2:
        st.markdown(f"""
        <div class="diag-box">
            <div class="diag-header">④ 线性假设因素 — 二阶非线性残差相关度 r = {d['quad_r']:.2f}</div>
            <div class="diag-content">
                <b>偏大/偏小机理</b>：加性线性框架 $\\Pi_F^* = \\sum \\theta_i^* \\Pi_i$ 保证了强可解释性与跨地质迁移能力，但忽略了各无量纲因子间的高阶非线性耦合。在软硬岩土交替段，纯线性叠加会产生截断偏差。
            </div>
        </div>
        <div class="diag-box">
            <div class="diag-header">⑤ 无量纲因子机制覆盖 — 预测偏小占比 {d['under_rate']:.1f}% vs 预测偏大占比 {d['over_rate']:.1f}%</div>
            <div class="diag-content">
                <b>偏大/偏小机理</b>：现有 7 个无量纲因子主要覆盖<b>刀盘正面支护、地应力与切削压缩</b>。当出现连续多环<b>预测明显偏小（实测推力偏大）</b>时，是因为盾壳外周摩阻力剧增、刀盘结泥饼、同步注浆压力异常或刀具磨损等外部阻力机制未被现有 7 因子覆盖。
            </div>
        </div>
        <div class="diag-box">
            <div class="diag-header">⑥ 地质突变因素 — 模量突变梯度与误差相关系数 r = {d['jump_r']:.2f}</div>
            <div class="diag-content">
                <b>偏大/偏小机理</b>：测试集相邻环弹性模量相对变化率 $|\\Delta E / E|$ 与预测误差呈正相关（$r={d['jump_r']:.2f}$）。当盾构突入高刚度地层突变带时，围岩弹性模量 $E$ 骤增，引起实测推力剧烈跃升与瞬态冲击偏差。
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("##### （3）测试集逐环实测对比与偏差原因自动定位明细表")
    st.dataframe(d['detail_df'], use_container_width=True, hide_index=True)

# -------------------- Tab 5: 施工辅助决策预演台 --------------------
with tab5:
    st.subheader("模块 5：施工辅助决策交互舱 — 实时调控掘进参数预演推力响应")
    st.markdown("拖动下方滑块模拟不同地质条件与盾构操控参数，智能体将实时解算总推力预测值及 7 个无量纲力学机制的分项贡献：")

    s1, s2, s3, s4 = st.columns(4)
    med = agent.df_clean.median()
    sim_E = s1.slider("地层弹性模量 E (MPa)", float(agent.df_clean['E'].min()), float(agent.df_clean['E'].max()), float(med['E']))
    sim_H = s2.slider("隧道埋深 H (m)", float(agent.df_clean['H'].min()), float(agent.df_clean['H'].max()), float(med['H']))
    sim_pe = s3.slider("土舱支护压力 pe (kPa)", float(agent.df_clean['pe'].min()), float(agent.df_clean['pe'].max()), float(med['pe']))
    sim_v = s4.slider("推进速度 v (mm/min)", float(agent.df_clean['v'].min()), float(agent.df_clean['v'].max()), float(med['v']))

    s5, s6, s7, s8 = st.columns(4)
    sim_w = s5.slider("刀盘转速 ω (rev/min)", float(agent.df_clean['omega'].min()), float(agent.df_clean['omega'].max()), float(med['omega']))
    sim_c = s6.slider("土体黏聚力 c (kPa)", float(agent.df_clean['c'].min()), float(agent.df_clean['c'].max()), float(med['c']))
    sim_phi = s7.slider("内摩擦角 φ (°)", float(agent.df_clean['phi'].min()), float(agent.df_clean['phi'].max()), float(med['phi']))
    sim_W = s8.slider("地基承载力 W (kPa)", float(agent.df_clean['W'].min()), float(agent.df_clean['W'].max()), float(med['W']))

    E_k, D_m, v_m, phi_r, mu_v, K0_v, gam_v = sim_E * 1000.0, med['D'], sim_v / 1000.0, np.deg2rad(sim_phi), med['mu'], med['K0'], med['gamma']
    sim_pi_vec = np.array([
        sim_pe / E_k,
        sim_W / E_k,
        (1.0 + K0_v) * gam_v * sim_H / E_k,
        (sim_c / E_k) * (v_m / (sim_w * D_m)),
        np.tan(phi_r) * (v_m / (sim_w * D_m)),
        (mu_v / ((1.0 + mu_v) * (1.0 - 2.0 * mu_v))) * (v_m / (sim_w * D_m)),
        (2.0 * sim_c * (1.0 + mu_v)) / (E_k * np.tan(phi_r))
    ])
    contrib_kN = sim_pi_vec * agent.theta_phys * E_k * (D_m ** 2)
    pred_total_kN = float(np.sum(contrib_kN))

    st.success(f"🚀 **实时预演总推力预测结果**：**{pred_total_kN:,.1f} kN**")
    contrib_df = pd.DataFrame({
        "无量纲因子": agent.PI_NAMES,
        "力学机制名称": [item[3] for item in agent.PI_INFO],
        "当前无量纲数值": sim_pi_vec,
        "分项推力贡献 (kN)": np.round(contrib_kN, 1)
    })
    st.dataframe(contrib_df, use_container_width=True, hide_index=True)