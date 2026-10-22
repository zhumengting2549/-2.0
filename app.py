import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
import numpy as np

# 绘图字体配置，图表文字英文，防止乱码
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="农业种植决策智能体", layout="wide")
st.title("🌾 农业种植决策智能体")

# 初始化会话存储
if "res_data" not in st.session_state:
    st.session_state["res_data"] = None

# 作物基础参数库
crop_info = {
    "水稻": {
        "variety": ["常规稻", "杂交稻"],
        "常规稻": {
            "seed_range": [80.0, 140.0],
            "fer_range": [100.0, 160.0],
            "pesticide_range": [20.0, 40.0],
            "labor_range": [300.0, 450.0],
            "yield_range": [450.0, 600.0],
            "sale_price": 2.8,
            "opt_rain": 950,
            "opt_temp": 3400
        },
        "杂交稻": {
            "seed_range": [60.0, 110.0],
            "fer_range": [120.0, 180.0],
            "pesticide_range": [20.0, 40.0],
            "labor_range": [320.0, 480.0],
            "yield_range": [500.0, 680.0],
            "sale_price": 3.0,
            "opt_rain": 1000,
            "opt_temp": 3500
        }
    },
    "玉米": {
        "variety": ["普通玉米", "甜糯玉米"],
        "普通玉米": {
            "seed_range": [40.0, 70.0],
            "fer_range": [80.0, 130.0],
            "pesticide_range": [15.0, 30.0],
            "labor_range": [240.0, 360.0],
            "yield_range": [400.0, 550.0],
            "sale_price": 2.2,
            "opt_rain": 800,
            "opt_temp": 3200
        },
        "甜糯玉米": {
            "seed_range": [70.0, 100.0],
            "fer_range": [90.0, 140.0],
            "pesticide_range": [18.0, 35.0],
            "labor_range": [280.0, 400.0],
            "yield_range": [320.0, 450.0],
            "sale_price": 3.6,
            "opt_rain": 850,
            "opt_temp": 3300
        }
    }
}

# 全国主产省份城市气候基础库（多年平均：降水mm，≥10℃活动积温℃）
city_climate = {
    "四川省": {
        "成都市": {"rain": 950, "temp_sum": 3400},
        "遂宁市": {"rain": 920, "temp_sum": 3350},
        "绵阳市": {"rain": 880, "temp_sum": 3200},
        "南充市": {"rain": 980, "temp_sum": 3450},
        "达州市": {"rain": 1100, "temp_sum": 3500},
        "泸州市": {"rain": 1150, "temp_sum": 3800},
        "宜宾市": {"rain": 1120, "temp_sum": 3750},
        "德阳市": {"rain": 890, "temp_sum": 3300},
        "乐山市": {"rain": 1200, "temp_sum": 3600},
        "眉山市": {"rain": 980, "temp_sum": 3500},
        "内江市": {"rain": 1050, "temp_sum": 3650},
        "广安市": {"rain": 1080, "temp_sum": 3550}
    },
    "江苏省": {
        "南京市": {"rain": 1050, "temp_sum": 4000},
        "苏州市": {"rain": 1100, "temp_sum": 4100},
        "徐州市": {"rain": 860, "temp_sum": 3800},
        "连云港市": {"rain": 900, "temp_sum": 3750},
        "淮安市": {"rain": 980, "temp_sum": 3900},
        "盐城市": {"rain": 1020, "temp_sum": 3950},
        "扬州市": {"rain": 1030, "temp_sum": 4000},
        "泰州市": {"rain": 1040, "temp_sum": 4050},
        "镇江市": {"rain": 1080, "temp_sum": 4050},
        "南通市": {"rain": 1060, "temp_sum": 4100}
    },
    "山东省": {
        "济南市": {"rain": 680, "temp_sum": 4200},
        "青岛市": {"rain": 720, "temp_sum": 3900},
        "潍坊市": {"rain": 650, "temp_sum": 4100},
        "德州市": {"rain": 580, "temp_sum": 4300},
        "聊城市": {"rain": 590, "temp_sum": 4350},
        "滨州市": {"rain": 570, "temp_sum": 4250},
        "东营市": {"rain": 560, "temp_sum": 4200},
        "济宁市": {"rain": 700, "temp_sum": 4250},
        "泰安市": {"rain": 750, "temp_sum": 4100},
        "菏泽市": {"rain": 690, "temp_sum": 4350}
    },
    "黑龙江省": {
        "哈尔滨市": {"rain": 550, "temp_sum": 2600},
        "齐齐哈尔市": {"rain": 480, "temp_sum": 2500},
        "绥化市": {"rain": 520, "temp_sum": 2550},
        "大庆市": {"rain": 440, "temp_sum": 2650},
        "佳木斯市": {"rain": 580, "temp_sum": 2450},
        "牡丹江市": {"rain": 540, "temp_sum": 2700},
        "黑河市": {"rain": 510, "temp_sum": 2200},
        "鸡西市": {"rain": 530, "temp_sum": 2500},
        "双鸭山市": {"rain": 560, "temp_sum": 2400},
        "七台河市": {"rain": 520, "temp_sum": 2480},
        "鹤岗市": {"rain": 570, "temp_sum": 2350},
        "伊春市": {"rain": 620, "temp_sum": 2300}
    },
    "河南省": {
        "郑州市": {"rain": 640, "temp_sum": 4300},
        "商丘市": {"rain": 700, "temp_sum": 4400},
        "周口市": {"rain": 720, "temp_sum": 4450},
        "驻马店市": {"rain": 800, "temp_sum": 4400},
        "南阳市": {"rain": 850, "temp_sum": 4500},
        "信阳市": {"rain": 1100, "temp_sum": 4450},
        "开封市": {"rain": 620, "temp_sum": 4350},
        "新乡市": {"rain": 610, "temp_sum": 4300},
        "濮阳市": {"rain": 580, "temp_sum": 4300},
        "许昌市": {"rain": 660, "temp_sum": 4350}
    },
    "湖北省": {
        "武汉市": {"rain": 1200, "temp_sum": 4000},
        "襄阳市": {"rain": 850, "temp_sum": 3900},
        "荆州市": {"rain": 1100, "temp_sum": 4100},
        "黄冈市": {"rain": 1250, "temp_sum": 4150},
        "孝感市": {"rain": 1150, "temp_sum": 4100},
        "宜昌市": {"rain": 1180, "temp_sum": 4000},
        "咸宁市": {"rain": 1400, "temp_sum": 4200},
        "随州市": {"rain": 950, "temp_sum": 4000}
    },
    "湖南省": {
        "长沙市": {"rain": 1350, "temp_sum": 4200},
        "衡阳市": {"rain": 1300, "temp_sum": 4300},
        "岳阳市": {"rain": 1300, "temp_sum": 4150},
        "常德市": {"rain": 1200, "temp_sum": 4100},
        "益阳市": {"rain": 1400, "temp_sum": 4200},
        "邵阳市": {"rain": 1350, "temp_sum": 4250},
        "永州市": {"rain": 1400, "temp_sum": 4400},
        "株洲市": {"rain": 1320, "temp_sum": 4250}
    },
    "河北省": {
        "石家庄市": {"rain": 570, "temp_sum": 4400},
        "保定市": {"rain": 550, "temp_sum": 4300},
        "邯郸市": {"rain": 590, "temp_sum": 4450},
        "邢台市": {"rain": 560, "temp_sum": 4400},
        "沧州市": {"rain": 600, "temp_sum": 4350},
        "衡水市": {"rain": 530, "temp_sum": 4350},
        "唐山市": {"rain": 650, "temp_sum": 4000},
        "廊坊市": {"rain": 550, "temp_sum": 4250}
    },
    "安徽省": {
        "合肥市": {"rain": 1000, "temp_sum": 4000},
        "阜阳市": {"rain": 900, "temp_sum": 4100},
        "宿州市": {"rain": 850, "temp_sum": 4050},
        "亳州市": {"rain": 840, "temp_sum": 4100},
        "六安市": {"rain": 1150, "temp_sum": 4000},
        "滁州市": {"rain": 980, "temp_sum": 4000},
        "安庆市": {"rain": 1300, "temp_sum": 4150},
        "蚌埠市": {"rain": 920, "temp_sum": 4050}
    },
    "吉林省": {
        "长春市": {"rain": 560, "temp_sum": 2700},
        "松原市": {"rain": 460, "temp_sum": 2750},
        "四平市": {"rain": 550, "temp_sum": 2800},
        "白城市": {"rain": 400, "temp_sum": 2700},
        "吉林市": {"rain": 680, "temp_sum": 2600},
        "通化市": {"rain": 800, "temp_sum": 2750},
        "辽源市": {"rain": 600, "temp_sum": 2750}
    },
    "辽宁省": {
        "沈阳市": {"rain": 700, "temp_sum": 3200},
        "锦州市": {"rain": 570, "temp_sum": 3300},
        "辽阳市": {"rain": 720, "temp_sum": 3250},
        "铁岭市": {"rain": 700, "temp_sum": 3100},
        "朝阳市": {"rain": 480, "temp_sum": 3350},
        "盘锦市": {"rain": 650, "temp_sum": 3300},
        "营口市": {"rain": 750, "temp_sum": 3300}
    },
    "广西壮族自治区": {
        "南宁市": {"rain": 1300, "temp_sum": 4800},
        "桂林市": {"rain": 1900, "temp_sum": 4500},
        "柳州市": {"rain": 1500, "temp_sum": 4700},
        "贵港市": {"rain": 1450, "temp_sum": 4850},
        "玉林市": {"rain": 1600, "temp_sum": 4900},
        "来宾市": {"rain": 1400, "temp_sum": 4800}
    },
    "广东省": {
        "广州市": {"rain": 1700, "temp_sum": 4850},
        "湛江市": {"rain": 1600, "temp_sum": 5200},
        "茂名市": {"rain": 1750, "temp_sum": 5100},
        "惠州市": {"rain": 1800, "temp_sum": 4900},
        "梅州市": {"rain": 1500, "temp_sum": 4750},
        "肇庆市": {"rain": 1650, "temp_sum": 4900}
    }
}
province_list = list(city_climate.keys())

# 细化风险参考面板
risk_table = pd.DataFrame([
    {"指标类型": "干旱风险", "等级": "无干旱", "参考区间": "0.00 - 0.20"},
    {"指标类型": "干旱风险", "等级": "轻度干旱", "参考区间": "0.21 - 0.40"},
    {"指标类型": "干旱风险", "等级": "中度干旱", "参考区间": "0.41 - 0.60"},
    {"指标类型": "干旱风险", "等级": "强干旱", "参考区间": "0.61 - 0.80"},
    {"指标类型": "干旱风险", "等级": "严重干旱", "参考区间": "0.81 - 1.00"},

    {"指标类型": "虫害风险", "等级": "无虫害", "参考区间": "0.00 - 0.20"},
    {"指标类型": "虫害风险", "等级": "轻度虫害", "参考区间": "0.21 - 0.40"},
    {"指标类型": "虫害风险", "等级": "中度虫害", "参考区间": "0.41 - 0.60"},
    {"指标类型": "虫害风险", "等级": "强虫害", "参考区间": "0.61 - 0.80"},
    {"指标类型": "虫害风险", "等级": "严重虫害", "参考区间": "0.81 - 1.00"},

    {"指标类型": "年降水量(mm)", "等级": "偏少", "参考区间": "600 - 800"},
    {"指标类型": "年降水量(mm)", "等级": "适宜", "参考区间": "801 - 1100"},
    {"指标类型": "年降水量(mm)", "等级": "偏多", "参考区间": "1101 - 1400"},

    {"指标类型": "年积温(℃)", "等级": "不足", "参考区间": "2000 - 2800"},
    {"指标类型": "年积温(℃)", "等级": "适宜", "参考区间": "2801 - 3800"},
    {"指标类型": "年积温(℃)", "等级": "充足", "参考区间": "3801 - 4800"},
])

with st.expander("📋 参考指标区间（辅助输入）"):
    st.table(risk_table)

# 输入区域
col1, col2 = st.columns(2)
with col1:
    crop = st.selectbox("🌱 选择作物", ["水稻", "玉米"])
    variety_list = crop_info[crop]["variety"]
    variety = st.selectbox("🌿 选择品种", variety_list)
with col2:
    drought_risk = st.slider("☀️ 干旱风险", 0.0, 1.0, 0.2, 0.01)
    pest_risk = st.slider("🐛 虫害风险", 0.0, 1.0, 0.2, 0.01)

col3, col4 = st.columns(2)
with col3:
    selected_province = st.selectbox("🗺️ 选择省份", province_list)
    city_list = list(city_climate[selected_province].keys())
    selected_city = st.selectbox("🏙️ 选择城市", city_list)
    default_rain = city_climate[selected_province][selected_city]["rain"]
    default_temp = city_climate[selected_province][selected_city]["temp_sum"]
with col4:
    rain = st.number_input("💧 年降水量 mm", value=default_rain)
    temp_sum = st.number_input("🌡️ 年积温 ℃", value=default_temp)

st.info(f"📍 当前地区：{selected_province} - {selected_city}（自动载入当地多年平均气候基准）")

base = crop_info[crop][variety]
opt_rain = base["opt_rain"]
opt_temp = base["opt_temp"]

st.info(f"""💡 该品种参考成本区间：
种子 {base['seed_range'][0]}-{base['seed_range'][1]} 元/亩
肥料 {base['fer_range'][0]}-{base['fer_range'][1]} 元/亩
农药 {base['pesticide_range'][0]}-{base['pesticide_range'][1]} 元/亩
人工 {base['labor_range'][0]}-{base['labor_range'][1]} 元/亩
亩产 {base['yield_range'][0]}-{base['yield_range'][1]} kg
售价 {base['sale_price']} 元/kg
""")

st.subheader("✏️ 填写基础投入（每亩）")
c1,c2,c3,c4 = st.columns(4)
with c1:
    seed_cost = st.number_input("种子成本", value=base["seed_range"][0])
with c2:
    fer_cost = st.number_input("肥料成本", value=base["fer_range"][0])
with c3:
    pesticide_cost = st.number_input("农药成本", value=base["pesticide_range"][0])
with c4:
    labor_cost = st.number_input("人工成本", value=base["labor_range"][0])

base_yield = st.number_input("📊 无灾害基础亩产 kg", value=base["yield_range"][0])

# ========== 预测按钮 ==========
run_predict = st.button("🚀 开始预测", type="primary")

if run_predict:
    valid_flag = True
    if seed_cost < 0 or fer_cost <0 or pesticide_cost <0 or labor_cost <0 or base_yield <0:
        st.warning("⚠️ 成本、产量不能填写负数，请重新输入！")
        valid_flag = False

    if valid_flag:
        # 【新版减产权重：拉开差距！虫害权重最高，干旱次之，降水、积温更小】
        pest_loss = pest_risk * 0.40
        drought_loss = drought_risk * 0.30
        # 降水偏离适宜值带来损失（高斯型偏离惩罚）
        rain_dev = abs(rain - opt_rain) / opt_rain
        rain_loss = np.clip(rain_dev * 0.15, 0, 0.15)
        # 积温偏离适宜值带来损失
        temp_dev = abs(temp_sum - opt_temp) / opt_temp
        temp_loss = np.clip(temp_dev * 0.10, 0, 0.10)

        total_loss = pest_loss + drought_loss + rain_loss + temp_loss
        actual_yield = base_yield * (1 - total_loss)
        actual_yield = max(actual_yield, 0)

        total_cost = seed_cost + fer_cost + pesticide_cost + labor_cost
        sale_income = actual_yield * base["sale_price"]
        profit = sale_income - total_cost

        res_data = {
            "province": selected_province,
            "city": selected_city,
            "crop": crop,
            "variety": variety,
            "drought_risk": drought_risk,
            "pest_risk": pest_risk,
            "rain": rain,
            "temp_sum": temp_sum,
            "base_yield": base_yield,
            "actual_yield": actual_yield,
            "total_cost": total_cost,
            "income": sale_income,
            "profit": profit
        }
        # 存入会话缓存
        st.session_state["res_data"] = res_data

        st.subheader("📈 测算结果表")
        result_df = pd.DataFrame({
            "项目": ["种子成本", "肥料成本", "农药成本", "人工成本", "基础亩产", "灾害后实际亩产", "销售收入", "总成本", "净利润"],
            "数值": [seed_cost, fer_cost, pesticide_cost, labor_cost, base_yield, actual_yield, sale_income, total_cost, profit]
        })
        st.dataframe(result_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(6,3))
        cost_names = ["Seed", "Fertilizer", "Pesticide", "Labor"]
        cost_values = [seed_cost, fer_cost, pesticide_cost, labor_cost]
        ax.bar(cost_names, cost_values)
        ax.set_xlabel("Cost Type")
        ax.set_ylabel("Value (Yuan/Mu)")
        ax.set_title("Cost Composition")
        st.pyplot(fig)

        def get_drought_tip(r):
            if r <= 0.20:
                return "当前无干旱风险，土壤墒情良好，维持常规灌溉管理即可。"
            elif r <= 0.40:
                return "轻度干旱，留意田间墒情监测，必要时少量补水。"
            elif r <= 0.60:
                return "中度干旱，需要定期灌溉保墒，采取秸秆覆盖减少水分蒸发。"
            elif r <= 0.80:
                return "强干旱风险，建议启用滴灌设施，优先保障作物关键生育期供水。"
            else:
                return "严重干旱，供水压力大，评估是否调整种植方案，做好抗旱减灾准备。"

        def get_pest_tip(r):
            if r <= 0.20:
                return "当前无虫害风险，定期巡查田间即可。"
            elif r <= 0.40:
                return "轻度虫害风险，放置诱虫板监测虫口密度，提前预防。"
            elif r <= 0.60:
                return "中度虫害风险，优先使用生物农药，避免虫害扩散。"
            elif r <= 0.80:
                return "强虫害风险，及时开展统防统治，缩短巡查周期。"
            else:
                return "严重虫害，虫害爆发风险极高，尽快开展应急防治，评估减产损失。"

        st.subheader("📌 农事调控建议")
        st.write("☀️ 干旱建议：", get_drought_tip(drought_risk))
        st.write("🐛 虫害建议：", get_pest_tip(pest_risk))

        # ========== SHAP条形图（现在四个变量全部参与模型，四根柱子都出来、长短不一样） ==========
        st.subheader("🔍 SHAP影响分析")
        st.write("说明：下图展示四个环境因素对种植收益影响程度的大小排序。")
        X = np.array([[drought_risk, pest_risk, rain, temp_sum]])

        def model_predict(x):
            dr, pr, r, t = x.T
            pl = pr * 0.40
            dl = dr * 0.30
            r_dev = np.clip(abs(r - opt_rain)/opt_rain * 0.15, 0, 0.15)
            t_dev = np.clip(abs(t - opt_temp)/opt_temp * 0.10, 0, 0.10)
            loss = pl + dl + r_dev + t_dev
            yld = base_yield * (1 - loss)
            yld = np.clip(yld, 0, None)
            inc = yld * base["sale_price"]
            profit = inc - total_cost
            return profit

        explainer = shap.Explainer(model_predict, np.array([[0,0,600,2200],[1,1,1800,5000]]))
        shap_values = explainer(X)

        fig2, ax2 = plt.subplots(figsize=(6,3))
        shap.summary_plot(shap_values,
                          feature_names=["Drought Risk", "Pest Risk", "Rainfall", "Accumulated Temperature"],
                          plot_type="bar",
                          show=False)
        ax2.set_title("SHAP Feature Importance")
        st.pyplot(fig2)

        st.markdown("""
💡 图表解读：
柱子越长，代表这个因素对种植利润的影响越大。
模型权重：虫害风险 > 干旱风险 > 降水量 > 活动积温。
👉 农户优先防控虫害、干旱减灾；其次关注降水是否偏离适宜区间，积温影响相对最小。
""")

        st.subheader("⚖️ 同环境下水稻、玉米收益对比")
        rice_v = crop_info["水稻"]["variety"][0]
        corn_v = crop_info["玉米"]["variety"][0]
        rice_base = crop_info["水稻"][rice_v]
        corn_base = crop_info["玉米"][corn_v]

        # 对比作物也用同一套损失公式
        def calc_profit(crop_base):
            pl = pest_risk * 0.40
            dl = drought_risk * 0.30
            r_dev = np.clip(abs(rain - crop_base["opt_rain"])/crop_base["opt_rain"] * 0.15, 0, 0.15)
            t_dev = np.clip(abs(temp_sum - crop_base["opt_temp"])/crop_base["opt_temp"] * 0.10, 0, 0.10)
            loss = pl + dl + r_dev + t_dev
            yld = base_yield * (1 - loss)
            yld = max(yld,0)
            income = yld * crop_base["sale_price"]
            return income - total_cost

        rice_profit = calc_profit(rice_base)
        corn_profit = calc_profit(corn_base)

        compare_df = pd.DataFrame({
            "作物": ["水稻", "玉米"],
            "每亩收益(元)": [rice_profit, corn_profit]
        })
        st.dataframe(compare_df, use_container_width=True)
        fig3, ax3 = plt.subplots(figsize=(5,3))
        ax3.bar(["Rice","Corn"], [rice_profit, corn_profit])
        ax3.set_xlabel("Crop Type")
        ax3.set_ylabel("Profit (Yuan/Mu)")
        ax3.set_title("Profit Comparison")
        st.pyplot(fig3)

st.markdown("---")
st.caption("💬 农业种植决策智能体｜AI应用创新大赛作品")
