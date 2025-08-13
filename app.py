import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from collections import Counter
import os



# matplotlib.rcParams['font.family'] = 'Tajwal'  # أو 'Cairo' أو 'Amiri' حسب المتاح
# matplotlib.rcParams['axes.unicode_minus'] = False  # 
st.set_page_config(
    layout="wide",  # هذا يخلي العرض يملى الشاشة كاملة
    page_title="لوحة تحكم المحولات",
    page_icon="⚡"
)
st.markdown(
    """
    <style>
    /* تغيير اتجاه النص والمحتوى للصفحة */
    html, body, .main {
        direction: rtl;
        text-align: right;
    }
    /* تعديل اتجاه القوائم والنصوص داخل sidebar لو عندك */
    .css-1d391kg {  /* هذا اسم كلاس sidebar الافتراضي، ممكن يختلف */
        direction: rtl;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    * {
        font-family: 'IBM Plex Sans Arabic', sans-serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# # 1. قراءة الملفات
Transformer_data_2018 = pd.read_excel('data/info/Transformer_data_2018.xlsx')
Transformer_data_2020 = pd.read_excel('data/info/Transformer_data_2020.xlsx')
Transformer_data_2022 = pd.read_excel('data/info/Transformer_data_2022.xlsx')
transformer_data_2023 = pd.read_excel('data/info/Transformer_data_2023.xlsx')

low_nutrient_trend = pd.read_excel('data/low_nutrient_trend.xlsx')
preventive_maintenance_work = pd.read_excel('data/preventive_maintenance_work.xlsx')
subscriber_malfunctions = pd.read_excel('data/subscriber_malfunctions.xlsx')
development_projects = pd.read_excel('data/development_projects.xlsx')

transformer_loads_summer_2016 = pd.read_excel('data/loads/transformer_loads_summer_2016.xlsx')
transformer_loads_summer_2017 = pd.read_excel('data/loads/transformer_loads_summer_2017.xlsx')
transformer_loads_summer_2018 = pd.read_excel('data/loads/transformer_loads_summer_2018.xlsx')
transformer_loads_summer_2019 = pd.read_excel('data/loads/transformer_loads_summer_2019.xlsx')
transformer_loads_summer_2022 = pd.read_excel('data/loads/transformer_loads_summer_2022.xls')
transformer_loads_summer_2023 = pd.read_csv('data/loads/transformer_loads_summer_2023.csv')

transformer_loads_winter_2017 = pd.read_excel('data/loads/transformer_loads_winter_2017.xlsx')
transformer_loads_winter_2018 = pd.read_excel('data/loads/transformer_loads_winter_2018.xlsx')
transformer_loads_winter_2019 = pd.read_excel('data/loads/transformer_loads_winter_2019.xlsx')
transformer_loads_winter_2021 = pd.read_excel('data/loads/transformer_loads_winter_2021.xlsx')
transformer_loads_winter_2023 = pd.read_csv('data/loads/transformer_loads_winter_2023.csv')

# تنظيف وتجهيز
for df in [transformer_loads_summer_2023, transformer_loads_winter_2023]:
    df.columns = df.columns.str.strip()
    df['تاريخ القياس'] = pd.to_datetime(df['تاريخ القياس'], dayfirst=True, errors='coerce')
    df['V_avg'] = df[['الجهد بين الفازات RS', 'الجهد بين الفازات RT', 'الجهد بين الفازات ST']].mean(axis=1)
    df['I_max'] = df[['قياس التيار R', 'قياس التيار S', 'قياس التيار T']].max(axis=1)
    df['V_avg'] = pd.to_numeric(df['V_avg'], errors='coerce')
    df['I_max'] = pd.to_numeric(df['I_max'], errors='coerce')
    df['Load_kVA'] = (np.sqrt(3) * df['V_avg'] * df['I_max']) / 1000
    df['load_ratio'] = df['Load_kVA'] / df['قدرة_السكينة']

# دمج البيانات
all_data = pd.concat([transformer_loads_summer_2023.assign(season='Summer'), transformer_loads_winter_2023.assign(season='Winter')])
all_data = all_data.sort_values('تاريخ القياس')

# حساب أعلى نسبة تحميل
agg_df = all_data.groupby('اسم_المحول')['load_ratio'].max().reset_index()

# تصنيف الحالة
agg_df['load_status'] = pd.cut(
    agg_df['load_ratio'],
    bins=[0, 0.8, 1.0, np.inf],
    labels=['طبيعي', 'عالي', 'حمل زائد']
)

# تهيئة جدول الصيانة إذا ما كان موجود في الـ session_state
CSV_FILE = "maintenance_table.csv"

if "maintenance_table" not in st.session_state:
    try:
        if os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:
            st.session_state.maintenance_table = pd.read_csv(CSV_FILE).to_dict(orient='records')
        else:
            st.session_state.maintenance_table = []
    except pd.errors.EmptyDataError:
        st.session_state.maintenance_table = []





def generate_recommendations(loads_df, transformer_info, selected_transformer_id=None, selected_transformer=None):
    recs = []
    final_summary = []

    # فلترة على المحول المحدد
    if selected_transformer_id is not None:
        loads_df = loads_df[loads_df['Transformer_id'] == selected_transformer_id]
    elif selected_transformer is not None:
        loads_df = loads_df[loads_df['اسم_المحول'] == selected_transformer]

    if loads_df.empty:
        return recs

    # إضافة الموسم
    loads_df['الموسم'] = loads_df['تاريخ القياس'].dt.month.apply(
        lambda m: "شتوي" if m in [12, 1, 2, 3] else ("صيفي" if m in [6, 7, 8] else "انتقالي")
    )

    # حساب المتوسط والانحراف ونسبة عدم التوازن لكل صف
    mean_current = loads_df[['قياس التيار R', 'قياس التيار S', 'قياس التيار T']].mean(axis=1)
    std_current = loads_df[['قياس التيار R', 'قياس التيار S', 'قياس التيار T']].std(axis=1)
    loads_df['Imbalance'] = (std_current / mean_current).fillna(0).replace(np.inf, 0)

    # تحليل السكاكين - الاتجاهات الحالية والماضي
    all_directions = loads_df['اتجاه السكينة'].unique()
    # السنوات القديمة لتحليل الاتجاهات السابقة
    years = loads_df['تاريخ القياس'].dt.year.unique()
    min_year = years.min()
    max_year = years.max()

    # بناء سجل للاتجاهات لكل سنة (لتحديد التغييرات)
    directions_per_year = {}
    for y in years:
        directions_per_year[y] = set(loads_df[loads_df['تاريخ القياس'].dt.year == y]['اتجاه السكينة'].unique())

    # تحديد الاتجاهات الأساسية على أغلب السنوات (أكثر من نصف السنوات)
    all_dirs_across_years = []
    for y in years:
        all_dirs_across_years.extend(list(directions_per_year[y]))
    dir_counts = Counter(all_dirs_across_years)
    base_directions = {d for d, c in dir_counts.items() if c >= (len(years) / 2)}

    # نحلل التغيرات في الاتجاهات
    unusual_dirs = set()
    for d in all_directions:
        # ظهر الاتجاه كم مرة في السنوات؟
        count_in_years = sum([1 for y in years if d in directions_per_year[y]])
        if d not in base_directions and count_in_years == 1:
            unusual_dirs.add(d)
            # توصية خطأ اتجاه
            recs.append({
                "title": f"خطأ محتمل في اتجاه السكينة: {d}",
                "message": f"اتجاه السكينة '{d}' ظهر مرة واحدة فقط ويختلف عن الاتجاهات الأساسية {sorted(base_directions)}. يُرجى التحقق وتصحيح اتجاه السكينة.",
                "severity": "warning",
                "action": f"مراجعة ملف البيانات وتصحيح اتجاه السكينة '{d}' إذا كان خطأً.",
                "due_in_days": 1,
                "status": "جديد"
            })
            final_summary.append(f"هناك احتمال وجود خطأ في اتجاه السكينة '{d}' بسبب ظهوره لمرة واحدة فقط.")

    # 2. هل في اتجاه جديد ظهر حديثًا (مرتين أو أكثر متتاليتين حديثًا)؟
    recent_years = sorted(years)[-2:]  # آخر سنتين
    recent_dirs = set()
    for y in recent_years:
        recent_dirs |= directions_per_year[y]
    new_dirs = recent_dirs - base_directions
    for d in new_dirs:
        # عدد مرات الظهور في السنتين الأخيرتين
        count_recent = sum([1 for y in recent_years if d in directions_per_year[y]])
        if count_recent >= 2:
            recs.append({
                "title": f"إضافة سكينة جديدة: {d}",
                "message": f"اتجاه السكينة '{d}' تم إضافته حديثًا منذ {recent_years[0]} ويظهر في البيانات الحديثة.",
                "severity": "info",
                "action": f"تحديث الوثائق والتأكد من تركيب السكينة '{d}' بشكل صحيح.",
                "due_in_days": 30,
                "status": "مراقبة"
            })
            final_summary.append(f"تمت إضافة سكينة جديدة باتجاه '{d}' منذ سنة {recent_years[0]}.")

    # 3. هل استُبدلت السكاكين بلوحة سكادا حديثًا؟
    has_scada_recent = any("سكادا" in str(d) for d in recent_dirs)
    has_scada_before = any("سكادا" in str(d) for y in years[:-2] for d in directions_per_year[y])
    if has_scada_recent and not has_scada_before:
        recs.append({
            "title": "استبدال السكاكين بلوحة سكادا",
            "message": "تم استبدال سكاكين المحول بلوحة سكادا في البيانات الحديثة.",
            "severity": "info",
            "action": "تحديث النظام وتوثيق التغيير لضمان دقة القياسات.",
            "due_in_days": 3,
            "status": "مراقبة"
        })
        final_summary.append("تم استبدال سكاكين المحول بلوحة سكادا في البيانات الحديثة.")

    # تحليل التوازن لكل اتجاه (كما في السابق)
    for direction in all_directions:
        dir_data = loads_df[loads_df['اتجاه السكينة'] == direction].copy()
        latest_year = dir_data['تاريخ القياس'].max().year
        recent_data = dir_data[dir_data['تاريخ القياس'].dt.year == latest_year]

        avg_imbalance = recent_data['Imbalance'].mean()
        historical_avg_imbalance = dir_data['Imbalance'].mean()

        if avg_imbalance > 0.2:
            severity = "error" if avg_imbalance > 0.4 else "warning"
            action = f"إرسال طاقم فني للكشف على السكينة {direction} وتوازن الأحمال"
            due_days = 3 if severity == "error" else 7
            status = "جديد"

            message = f"السكينة {direction} غير متزنة ({avg_imbalance:.2f})."
            message += " المشكلة مزمنة." if historical_avg_imbalance > 0.2 else " المشكلة جديدة."

            recs.append({
                "title": f"عدم توازن — {direction}",
                "message": message,
                "severity": severity,
                "examples": recent_data[['تاريخ القياس', 'Imbalance']].head(3).to_dict(orient='records'),
                "action": action,
                "due_in_days": due_days,
                "chart_type": "imbalance",
                "status": status
            })

            final_summary.append(f"السكينة {direction}: مشكلة عدم توازن ({'مزمن' if historical_avg_imbalance > 0.2 else 'جديد'}).")

        else:
            recs.append({
                "title": f"السكينة {direction} متزنة",
                "message": "لا توجد مشكلة حالية لكن يُفضل المراقبة الدورية.",
                "severity": "success",
                "action": f"متابعة الأحمال على السكينة {direction} كل شهر",
                "due_in_days": 30,
                "status": "مراقبة"
            })

    # تحليل موسمي للمحول
    capacity = None
    if 'KVA' in transformer_info:
        if isinstance(transformer_info, pd.DataFrame):
            capacity = transformer_info['KVA'].iloc[0]
        elif isinstance(transformer_info, dict):
            capacity = transformer_info.get('KVA', None)

    seasonal_loads = loads_df.groupby('الموسم')['Load_kVA'].mean().to_dict()

    if capacity is not None:
        if seasonal_loads.get('شتوي', 0) > capacity:
            recs.append({
                "title": "تجاوز السعة شتاءً",
                "message": f"الأحمال الشتوية ({seasonal_loads['شتوي']:.1f} KVA) تتجاوز السعة ({capacity} KVA).",
                "severity": "error",
                "action": "مراجعة الاشتراكات الشتوية أو إعادة توزيع الأحمال",
                "due_in_days": 10,
                "status": "جديد"
            })
            final_summary.append("الأحمال الشتوية تتجاوز سعة المحول.")

        elif seasonal_loads.get('شتوي', 0) <= capacity:
            final_summary.append("الأحمال الشتوية ضمن السعة الطبيعية.")

        if seasonal_loads.get('صيفي', 0) < capacity * 0.5:
            final_summary.append("الأحمال الصيفية أقل بكثير من السعة.")
        else:
            final_summary.append("الأحمال الصيفية ضمن السعة الطبيعية.")

    # التنبؤ بالحمل المستقبلي
    df_predict = loads_df.dropna(subset=['Load_kVA']).copy()
    df_predict['Days'] = (df_predict['تاريخ القياس'] - df_predict['تاريخ القياس'].min()).dt.days

    future_warnings = []
    if not df_predict.empty:
        X = df_predict[['Days']]
        y = df_predict['Load_kVA']
        model = LinearRegression()
        model.fit(X, y)

        future_days = np.array([[df_predict['Days'].max() + i] for i in [30, 90, 180]])
        future_predictions = model.predict(future_days)

        if capacity is not None:
            for horizon, pred in zip(["شهر", "3 أشهر", "6 أشهر"], future_predictions):
                if pred > capacity:
                    recs.append({
                        "title": f"توقع تجاوز السعة بعد {horizon}",
                        "message": f"الحمل المتوقع ({pred:.1f} KVA) سيتجاوز السعة ({capacity} KVA).",
                        "severity": "warning",
                        "action": "إجراءات وقائية لإعادة توزيع الأحمال قبل الموعد المتوقع",
                        "due_in_days": 30 if horizon == "شهر" else (90 if horizon == "3 أشهر" else 180),
                        "status": "مراقبة"
                    })
                    future_warnings.append(f"من المتوقع تجاوز السعة خلال {horizon}.")

                elif pred > capacity * 0.9:
                    recs.append({
                        "title": f"اقتراب من السعة بعد {horizon}",
                        "message": f"الحمل المتوقع ({pred:.1f} KVA) سيقترب من 90% من السعة.",
                        "severity": "info",
                        "action": "مراجعة الزيادات المتوقعة في الأحمال",
                        "due_in_days": 30 if horizon == "شهر" else (90 if horizon == "3 أشهر" else 180),
                        "status": "مراقبة"
                    })
                    future_warnings.append(f"من المتوقع الاقتراب من السعة خلال {horizon}.")

            if not future_warnings:
                final_summary.append("من المتوقع أن يظل الحمل ضمن سعة المحول على الأقل حتى 6 أشهر قادمة.")
            else:
                final_summary.extend(future_warnings)
        
        # تجميع التوصية الختامية بشكل إنساني وسلس
        imbalance_dirs = [s for s in final_summary if "السكينة" in s and "عدم توازن" in s]
        direction_issues = []
        for rec in recs:
            if rec['title'].startswith("خطأ محتمل في اتجاه السكينة") or rec['title'].startswith("إضافة سكينة جديدة") or rec['title'].startswith("استبدال السكاكين"):
                direction_issues.append(rec['message'])

        final_message_parts = []

        if direction_issues:
            # final_message_parts.append("هناك ملاحظات على اتجاهات السكاكين تشمل:\n-" + "\n- ".join(direction_issues))
            final_message_parts.append("هناك ملاحظات على اتجاهات السكاكين تشمل:\n-  " + "\n- ".join(direction_issues))

        if imbalance_dirs:
            final_message_parts.append("بالإضافة إلى وجود مشاكل في توازن الأحمال على بعض السكاكين.")

        other_summaries = [s for s in final_summary if "السكينة" not in s or "عدم توازن" not in s]
        if other_summaries:
            final_message_parts.append("أما بخصوص الأحمال الموسمية والتوقعات المستقبلية:\n- " + "\n- ".join(other_summaries))

        final_text = "\n\n".join(final_message_parts) if final_message_parts else "لا توجد توصيات خاصة حالياً."

        recs.append({
            "title": "توصية ختامية",
            "message": final_text,
            "severity": "info",
            "action": "مراجعة وتنفيذ التوصيات المذكورة أعلاه",
            "due_in_days": max([r['due_in_days'] for r in recs]) if recs else 30,
            "status": "جديد"
        })

    return recs



def display_recommendations(recs):
    severity_styles = {
        "error":   {"icon": "🚨", "bg": "#ffcccc", "border": "#ff4d4d"},
        "warning": {"icon": "⚠️", "bg": "#fff3cd", "border": "#ffcc00"},
        "success": {"icon": "✅", "bg": "#d4edda", "border": "#28a745"},
        "info":    {"icon": "ℹ️", "bg": "#cce5ff", "border": "#007bff"},
    }

    if "maintenance_table" not in st.session_state:
        st.session_state.maintenance_table = []

    for idx, r in enumerate(recs):
        style = severity_styles.get(r['severity'], severity_styles["info"])
        icon = style["icon"]

        # بطاقة التوصية
        st.markdown(
            f"""
            <div style="
                background-color:{style['bg']};
                border-left: 6px solid {style['border']};
                padding: 12px;
                margin-bottom: 10px;
                border-radius: 8px;
                color: #212529;
                font-family: Arial, sans-serif;">
                <h4 style="margin:0; font-size:18px; font-weight:bold; color:#212529;">
                    {icon} {r['title']}
                </h4>
                <p style="margin:5px 0; font-size:15px; color:#212529;">
                    {r['message']}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if 'examples' in r and r['examples']:
            with st.expander("📊 عرض أمثلة القياسات"):
                st.dataframe(pd.DataFrame(r['examples']))

        # هل التوصية موجودة؟
        # in_maintenance = any(item['title'] == r['title'] for item in st.session_state.maintenance_table)
        in_maintenance = any(
            item['title'] == r['title'] and
            item.get('Transformer_id') == st.session_state.get('selected_transformer_id')
            for item in st.session_state.maintenance_table
        )

        st.markdown(
            f"""
            <div style="
                background-color:#f8f9fa;
                border: 1px solid #ddd;
                padding: 10px;
                margin-bottom: 15px;
                border-radius: 6px;
                color:#212529;">
                <b>🛠 الإجراء المقترح:</b> {r['action']}<br>
                <b>⏳ المهلة:</b> {r['due_in_days']} يوم<br>
                <b>📌 حالة التنفيذ:</b> {r['status']}
            </div>
            """,
            unsafe_allow_html=True
        )

        if idx == len(recs) - 1:
            pass  # ما تعرضش أزرار
        else:
            if not in_maintenance:
                if st.button("➕ إضافة لجدول الصيانة", key=f"add_{idx}"):
                    rec_with_transformer = r.copy()
                    rec_with_transformer['Transformer_name'] = st.session_state.get('selected_transformer', 'غير معروف')
                    rec_with_transformer['Transformer_id'] = st.session_state.get('selected_transformer_id', None)
                    st.session_state.maintenance_table.append(rec_with_transformer)
                    save_maintenance_table()
                    st.rerun()

                st.markdown(
                    f"""
                    <style>
                    div.stButton > button[key="{f'add_{idx}'}"] {{
                        background-color: #28a745 !important;  /* أخضر */
                        color: white !important;
                        font-weight: bold;
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True
                )
            else:
                if st.button("❌ إزالة من جدول الصيانة", key=f"remove_{idx}"):
                    st.session_state.maintenance_table = [
                        item for item in st.session_state.maintenance_table if item['title'] != r['title']
                    ]
                    save_maintenance_table()
                    st.rerun()

                st.markdown(
                    f"""
                    <style>
                    div.stButton > button[key="{f'remove_{idx}'}"] {{
                        background-color: #dc3545 !important;  /* أحمر */
                        color: white !important;
                        font-weight: bold;
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown("---")


def display_maintenance_tab():
    st.header("جدولة الصيانة")
    if st.session_state.maintenance_table:
        total_days = 0
        type_counter = {}
        
        for idx, r in enumerate(st.session_state.maintenance_table):
            # حساب الإحصائيات
            total_days += r.get('due_in_days', 0)
            issue_type = r['title'].split("—")[0].strip() if "—" in r['title'] else r['title']
            type_counter[issue_type] = type_counter.get(issue_type, 0) + 1

            # عرض البطاقة
            st.markdown(
                f"""
                <div style="
                    background-color:#f8f9fa;
                    border-left: 6px solid #007bff;
                    padding: 12px;
                    margin-bottom: 10px;
                    border-radius: 8px;
                    color: #212529;
                    font-family: Arial, sans-serif;">
                    <h4 style="margin:0; font-size:18px; font-weight:bold; color:#212529;">
                        🛠 {r['title']}
                    </h4>
                    <p style="margin:5px 0; font-size:15px; color:#212529;">
                        {r['message']}
                    </p>
                    <b>🔌 المحول:</b> {r.get('Transformer_name', 'غير محدد')}<br>
                    <b>الإجراء:</b> {r['action']}<br>
                    <b>المهلة:</b> {r['due_in_days']} يوم<br>
                    <b>الحالة:</b> {r['status']}
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button("❌ إزالة", key=f"remove_tab_{idx}"):
                st.session_state.maintenance_table.pop(idx)
                save_maintenance_table()
                st.rerun()

        # فاصل وإحصائيات
        st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)
        st.subheader("📊 ملخص الصيانة")
        st.write(f"**إجمالي التوصيات:** {len(st.session_state.maintenance_table)} توصية")
        st.write(f"**إجمالي الأيام المطلوبة:** {total_days} يوم")
        for t, count in type_counter.items():
            st.write(f"**{t}:** {count} توصية")
    else:
        st.info("لا توجد توصيات مضافة لجدول الصيانة بعد.")

def save_maintenance_table():
    df = pd.DataFrame(st.session_state.maintenance_table)
    df.to_csv(CSV_FILE, index=False)



# عنوان جانبي
st.sidebar.title("📁 نظام متابعة المحولات ")

# 🧠 أضف هذا CSS في أعلى الكود بعد import streamlit
st.markdown("""
<style>
/* تكبير الخط لعناصر الراديو */
.css-1c7y2kd, .css-16idsys {
    font-size: 18px !important;
}

/* ترك مسافة بين العناصر */
.css-1c7y2kd > div {
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# 🎯 قائمة الراديو داخل الشريط الجانبي
page = st.sidebar.radio(
    "", 
    [
        "📑 دليل الأستخدام",
        "⚙ تحليل عام ",
        "📊 لوحة تحليل الأحمال", 
        "🔧 الصيانة والأعطال",
        "🏗 المشاريع التطويرية",
        "📈 تحليلات متقدمة",
    ]
)


st.sidebar.markdown("---")
st.sidebar.caption("© 2025 نظام تحليل المحولات الكهربائية")

if page ==  "📑 دليل الأستخدام":
    st.title("📖 دليل المستخدم - نظام تحليل المحولات")
    st.markdown("######")
    st.header("🎥 فيديو توضيحي للنظام")
    video_url = "https://youtu.be/76rf-H1Q13s"
    st.video(video_url)
    
    # تبويبات الدليل
    tab1, tab2, tab3, tab4 = st.tabs(["نظرة عامة", "البدء السريع", "التوصيات", "الأسئلة الشائعة"])
    
    with tab1:  # نظرة عامة
        st.header("⚡ نظرة عامة على النظام")
        st.markdown("""
        ### 🌟 الميزات الرئيسية
            - تحليل شامل لجميع محولات الشبكة (مدينة دير البلح)
            - مراقبة الأحمال حسب البيانات المتوفرة
            - توليد توصيات ذكية للصيانة ومتابعة المحولات
        
        ### 📊 البيانات المدعومة
            - بيانات الأحمال التاريخية (2016-2023)
            - مواصفات المحولات الفنية
            - سجلات الصيانة الوقائية
        """)
        
        
    with tab2:  # البدء السريع
        st.header("🚀 البدء السريع")
        
        # col1, col2 = st.columns(2)
        # with col1:
        st.markdown("""
        ### 1. تصفح البيانات
            - اختر المحول من القائمة الجانبية
            - استعرض البيانات حسب السنة والموسم
            - استخدم الفلاتر لتركيز التحليل
        
        ### 2. قراءة النتائج
            - اطلع على التوصيات الذكية
            - استعرض الرسوم البيانية التفاعلية
            - تصدير التقارير عند الحاجة
        """)
            
        # with col2:
        st.markdown("""
        ### 3. إدارة الصيانة
            - أضف توصيات الصيانة إلى الجدول
            - تتبع حالة التنفيذ
            - سجل الأعطال والإصلاحات

        ### 4. التحليل المتقدم
            - استخدم أدوات التنبؤ المستقبلي
            - حلل اتجاهات الأحمال
            - قارن بين المحولات
        """)
    
    with tab3:  # التوصيات
        st.header("🛠️ فهم التوصيات")
        
        # شرح أنواع التوصيات
        st.subheader("📋 أنواع التوصيات")
        # col1, col2 = st.columns(2)
        
        # with col1:
        st.markdown("""
        ### 🚨 توصيات حرجة (أحمر)
            - تتطلب تدخلاً فورياً
            - مهلة قصيرة (1-3 أيام)
            - أمثلة:
            - تجاوز السعة
            - خلل فني خطير
            - خطر على السلامة
        """)
            # عرض أمثلة حية كما تظهر في النظام
        # st.subheader("🎯 أمثلة حية من النظام")
        
        # مثال 1: توصية حرجة (error)
        st.markdown("""
        <div style="
            background-color:#ffcccc;
            border-left: 6px solid #ff4d4d;
            padding: 12px;
            margin-bottom: 10px;
            border-radius: 8px;
            margin-right: 35px;
            color: #212529;">
            <h4 style="margin:0; font-size:18px; font-weight:bold; color:#212529;">
                🚨 تجاوز السعة الشتوية
            </h4>
            <p style="margin:5px 0; font-size:15px; color:#212529;">
                المحول يتجاوز 110% من سعته الاسمية خلال فصل الشتاء.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="
            background-color:#f8f9fa;
            border: 1px solid #ddd;
            padding: 10px;
            margin-bottom: 15px;
            border-radius: 6px;
            margin-right: 35px;
            color:#212529;">
            <b>🛠 الإجراء المقترح:</b> مراجعة الاشتراكات الشتوية أو إعادة توزيع الأحمال<br>
            <b>⏳ المهلة:</b> 10 أيام<br>
            <b>📌 حالة التنفيذ:</b> جديد
        </div>
        """, unsafe_allow_html=True)

        # with col2:
        st.markdown("""
        ### ⚠️ توصيات تحذيرية (أصفر)
            - تحتاج لاهتمام خلال أسبوع
            - مشاكل متوسطة الخطورة
            - أمثلة:
            - عدم توازن في الأحمال
            - ارتفاع درجات الحرارة
            - تدهور في العزل
        """)
        # مثال 2: توصية تحذيرية (warning)
        st.markdown("""
        <div style="
            background-color:#fff3cd;
            border-left: 6px solid #ffcc00;
            padding: 12px;
            margin-bottom: 10px;
            margin-right: 35px;
            border-radius: 8px;
            color: #212529;">
            <h4 style="margin:0; font-size:18px; font-weight:bold; color:#212529;">
                ⚠️ عدم توازن - السكينة الشمالية
            </h4>
            <p style="margin:5px 0; font-size:15px; color:#212529;">
                السكينة الشمالية غير متزنة (0.35). المشكلة مزمنة.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="
            background-color:#f8f9fa;
            border: 1px solid #ddd;
            padding: 10px;
            margin-bottom: 15px;
            margin-right: 35px;
            border-radius: 6px;
            color:#212529;">
            <b>🛠 الإجراء المقترح:</b> إرسال طاقم فني للكشف على السكينة الشمالية وتوازن الأحمال<br>
            <b>⏳ المهلة:</b> 7 أيام<br>
            <b>📌 حالة التنفيذ:</b> جديد
        </div>
        """, unsafe_allow_html=True)

        
        # col3, col4 = st.columns(2)
        
        # with col3:
        st.markdown("""
        ### ℹ️ توصيات معلوماتية (أزرق)
            - للمعرفة والمتابعة
            - مهلة طويلة (شهر أو أكثر)
            - أمثلة:
            - تغييرات في النظام
            - ملاحظات للتحسين
            - تحديثات فنية
        """)

        # مثال 3: توصية معلوماتية (info)
        st.markdown("""
        <div style="
            background-color:#cce5ff;
            border-left: 6px solid #007bff;
            padding: 12px;
            margin-bottom: 10px;
            margin-right: 35px;
            border-radius: 8px;
            color: #212529;">
            <h4 style="margin:0; font-size:18px; font-weight:bold; color:#212529;">
                ℹ️ إضافة سكينة جديدة
            </h4>
            <p style="margin:5px 0; font-size:15px; color:#212529;">
                اتجاه السكينة 'الغربية' تم إضافته حديثاً ويظهر في البيانات الحديثة.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="
            background-color:#f8f9fa;
            border: 1px solid #ddd;
            padding: 10px;
            margin-bottom: 15px;
            margin-right: 35px;
            border-radius: 6px;
            color:#212529;">
            <b>🛠 الإجراء المقترح:</b> تحديث الوثائق والتأكد من تركيب السكينة الغربية<br>
            <b>⏳ المهلة:</b> 30 يوم<br>
            <b>📌 حالة التنفيذ:</b> مراقبة
        </div>
        """, unsafe_allow_html=True)
            
        # with col4:
        st.markdown("""
        ### ✅ توصيات إيجابية (أخضر)
            - تأكيد على أداء جيد
            - لا تحتاج إجراء عاجل
            - أمثلة:
            - أداء مستقر
            - تحسن في القياسات
            - صيانة ناجحة
        """)
        st.markdown("""
        <div style="
            background-color:#d4edda;
            border-left: 6px solid #28a745;
            padding: 12px;
            margin-bottom: 10px;
            border-radius: 8px;
            margin-right: 35px;
            color: #212529;">
            <h4 style="margin:0; font-size:18px; font-weight:bold; color:#212529;">
                ✅ السكينة متزنة
            </h4>
            <p style="margin:5px 0; font-size:15px; color:#212529;">
                السكينة الشمالية متزنة بشكل جيد.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="
            background-color:#f8f9fa;
            border: 1px solid #ddd;
            padding: 10px;
            margin-right: 25px;
            margin-bottom: 35px;
            border-radius: 6px;
            color:#212529;">
            <b>🛠 الإجراء المقترح:</b> متابعة الأداء حسب الجدول الزمني<br>
            <b>⏳ المهلة:</b> غير محددة<br>
            <b>📌 حالة التنفيذ:</b> جيد
        </div>
        """, unsafe_allow_html=True)

        
        # كيفية التعامل مع التوصيات
        st.subheader("🔧 آلية التعامل مع التوصيات")
        st.markdown("""
        ###### 
            - مراجعة الأولويات: تصفح التوصيات حسب لونها ودرجة أهميتها
            - التفاصيل الكاملة: انقر على كل توصية لرؤية التفاصيل والأمثلة
            - إضافة للصيانة: اضغط على زر "إضافة لجدول الصيانة" للتوصيات المهمة
            - المتابعة: تتبع حالة التنفيذ من خلال جدول الصيانة
            - التحديث: قم بتحديث حالة التوصية بعد معالجتها
        """)
        #
    
    with tab4:  # الأسئلة الشائعة
        st.header("❓ الأسئلة الشائعة")
        
        with st.expander("كيف أتصفح بيانات محول معين؟"):
            st.markdown("""
            1. انتقل إلى صفحة **تحليل عام**
            2. اختر المحول من القائمة المنسدلة
            3. استعرض البيانات في التبويبات المختلفة
            4. استخدم الفلاتر لتخصيص العرض
            """)
            
        with st.expander("ما هي أنواع التحاليل المتاحة؟"):
            st.markdown("""
            - **تحليل الأحمال**: نسب التحميل والتوازن بين الفازات
            - **تحليل السعة**: مقارنة الحمل مع السعة الاسمية
            - **تحليل الصيانة**: متابعة الأعطال والتوصيات
            - **تحليل الاتجاهات**: التنبؤ بالأحمال المستقبلية
            """)
            
        with st.expander("كيف أضيف مهمة صيانة جديدة؟"):
            st.markdown("""
            1. انتقل إلى صفحة **التحليل الفردي للمحول في قسم لوحة تحليل الأحمال **
            2. اختر التوصية المطلوبة
            3. اضغط على زر 'إضافة لجدول الصيانة'
            4. حدد موعد التنفيذ والمسؤول
            """)
    
    # قسم الدعم الفني
    st.markdown("---")
    st.header("📞 الدعم الفني")
    st.markdown("""
    📋 **لتعبئة نموذج التقييم البعدي:** [ اضغط هنا](https://forms.gle/LAzFGYyjjyUxjuKS6)
    """)

    st.markdown("""
        ###### للمساعدة الفنية يرجى التواصل عبر:
                - البريد الإلكتروني: ana.naji.1996@gmail.com
            -  الهاتف: 00970-595192140        
        """)   
    
    # إضافة تذييل الصفحة
    st.markdown("---")
    st.caption("© 2025 نظام تحليل محولات مدينة دير البلح الذكي - جميع الحقوق محفوظة")

    
# صفحات مختلفة حسب الاختيار
elif page ==  "⚙ تحليل عام ":
    st.title( "⚙ تحليل عام ")
    # توحيد أسماء الأعمدة
    rename_dict = {
        'إسم المحول': 'اسم_المحول',
        'KVA': 'KVA',
        'الرقم المتسلسل': 'الرقم المتسلسل',
        'سنة التصنيع': 'سنة التصنيع',
        'الخط المغذي': 'الخط المغذى',
        'نوع المحول': 'نوع التركيب',
        'الاتجاه': 'الاتجاه',
        'نوع القاطع ': 'نوع القاطع',
        'جسم المحول': 'جسم المحول',
        'مستوى الزيت': 'مستوى الزيت',
        'السيلكاجيل': 'السليكا جل',
        'التأريض': 'التأريض',
        'مانع صواعق': 'مانع الصواعق',
        'حالة القاطع': 'حالة القاطع',
        'ملكية المحول': 'الملكية',
        'Z%': 'Z%',
        'خزان احتياطي': 'خزان احتياطي',
        'طبيعة احمال المحول': 'طبيعة الأحمال',
    }

    Transformer_data_2018.rename(columns=rename_dict, inplace=True)
    Transformer_data_2020.rename(columns=rename_dict, inplace=True)
    Transformer_data_2022.rename(columns=rename_dict, inplace=True)

    # إضافة عمود السنة
    Transformer_data_2018['سنة_البيانات'] = 2018
    Transformer_data_2020['سنة_البيانات'] = 2020
    Transformer_data_2022['سنة_البيانات'] = 2022
    transformer_data_2023['سنة_البيانات'] = 2023

    # تحديد الأعمدة المشتركة مع الأسبقية لأحدث البيانات
    common_cols = [
        'Transformer_id', 'اسم_المحول', 'KVA', 'الشركة المصنعة', 
        'سنة التصنيع', 'حالة القاطع', 'الاتجاه', 'سنة_البيانات','طبيعة الأحمال', 'خزان احتياطي', 'Z%', 'الملكية',
        'مانع الصواعق','التأريض', 'السيلكا جل', 'مستوى الزيت', 'جسم المحول', 'نوع القاطع', 'نوع التركيب', 'الخط المغذى',
    ]

    missing_cols = [col for col in common_cols if col not in Transformer_data_2018.columns]
    print("الأعمدة المفقودة:", missing_cols)

    # دمج البيانات مع الحفاظ على الهيكل الموحد
    df_all = pd.concat([
        Transformer_data_2018[common_cols],
        Transformer_data_2020[common_cols],
        Transformer_data_2022[common_cols],
        transformer_data_2023[common_cols]
    ], ignore_index=True)

        # دالة مساعدة لرسم مخططات دائرية مع hover يعرض أسماء المحولات
    def plot_pie_with_hover(df, column_name, title, hole_size=0.4):
        if column_name not in df.columns or 'اسم_المحول' not in df.columns:
            st.warning(f"⚠️ البيانات غير مكتملة لرسم مخطط '{title}'")
            return

        df_counts = (
            df.groupby(column_name, dropna=False)
            .agg(
                count=(column_name, 'size'),
                المحولات=('اسم_المحول', lambda x: '<br>'.join(x.astype(str)))
            )
            .reset_index()
        )

        fig = px.pie(
            df_counts,
            names=column_name,
            values='count',
            title='',
            hole=hole_size,
            hover_data={'المحولات': True}
        )

        fig.update_traces(
            hovertemplate="<b>%{label}</b><br>عدد: %{value}<br>المحولات:<br>%{customdata[0]}"
        )

        st.plotly_chart(fig, use_container_width=True)

    # دالة مساعدة لرسم مخططات عمودية مع hover يعرض أسماء المحولات
    def plot_bar_with_hover(df, column_name, title):
        if column_name not in df.columns or 'اسم_المحول' not in df.columns:
            st.warning(f"⚠️ البيانات غير مكتملة لرسم مخطط '{title}'")
            return

        df_counts = (
            df.groupby(column_name, dropna=False)
            .agg(
                count=(column_name, 'size'),
                المحولات=('اسم_المحول', lambda x: '<br>'.join(x.astype(str)))
            )
            .reset_index()
        )

        fig = px.bar(
            df_counts,
            x=column_name,
            y='count',
            title='',
            hover_data={'المحولات': True}
        )

        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>عدد: %{y}<br>المحولات:<br>%{customdata[0]}"
        )

        st.plotly_chart(fig, use_container_width=True)


    # مخططات وأرقام تحليل عام
    tabs = st.tabs(["📈 نظرة عامة", "🔍 تحليل فردي", "🗂️ بيانات خام"])
    # ✅ تبويب نظرة عامة
    with tabs[0]:
        # الحصول على قائمة سنوات البيانات الموجودة
        st.header("📌 مؤشرات أداء عامة للمحولات")
        years = sorted(df_all['سنة_البيانات'].dropna().unique(), reverse=True)

        # إضافة اختيار في sidebar لاختيار السنة (أو "كل السنوات")
        selected_year = st.selectbox("اختر سنة البيانات للعرض", options=["كل السنوات"] + years, index=0)

        # تطبيق الفلترة بناءً على الاختيار
        if selected_year != "كل السنوات":
            df_filtered = df_all[df_all['سنة_البيانات'] == selected_year]
        else:
            df_filtered = df_all.copy()

        # عرض مؤشرات عامة باستخدام df_filtered
        col1, col2, col3, col4 = st.columns(4)
        total_transformers = df_filtered['Transformer_id'].nunique()
        avg_capacity = df_filtered['KVA'].mean()
        oldest_year = df_filtered['سنة التصنيع'].min()
        newest_year = df_filtered['سنة التصنيع'].max()

        with col1:
            st.metric("عدد المحولات", total_transformers)
        with col2:
            st.metric("متوسط السعة", f"{avg_capacity:.2f} KVA")
        with col3:
            st.metric("نطاق سنوات التصنيع", f"{oldest_year}-{newest_year}")
        with col4:
            st.metric("عدد الشركات المصنعة", df_filtered['الشركة المصنعة'].nunique() if 'الشركة المصنعة' in df_filtered.columns else "غير متوفر")

        st.markdown("---")

        col1, col2 = st.columns(2)

        # ✅ مخطط سعة المحولات (KVA)
        if 'KVA' in df_filtered.columns:
            with col1:
                st.markdown("##### 📊 سعة المحولات (KVA)")
                plot_pie_with_hover(df_filtered, 'KVA', 'سعة المحولات (KVA)', hole_size=0.4)

        # ✅ مخطط حالة القاطع
        if 'حالة القاطع' in df_filtered.columns:
            with col2:
                st.markdown("##### 📊 حالة القاطع")
                plot_pie_with_hover(df_filtered, 'حالة القاطع', 'حالة القاطع', hole_size=0.4)

        st.markdown("---")

        col1, col2 = st.columns(2)

        # ✅ مخطط الشركة المصنعة
        if 'الشركة المصنعة' in df_filtered.columns:
            with col1:
                st.markdown("##### 📊 الشركة المصنعة")
                plot_pie_with_hover(df_filtered, 'الشركة المصنعة', 'الشركة المصنعة', hole_size=0.4)

        # ✅ مخطط توزيع البيانات حسب السنوات (Bar chart)
        if 'سنة_البيانات' in df_filtered.columns:
            with col2:
                st.markdown("##### 📅 توزيع البيانات حسب السنوات")
                if 'اسم_المحول' not in df_filtered.columns:
                    st.error("⚠️ عمود 'اسم_المحول' غير موجود في البيانات!")
                else:
                    year_counts = (
                        df_filtered.groupby('سنة_البيانات', dropna=False)
                        .agg(
                            count=('سنة_البيانات', 'size'),
                            المحولات=('اسم_المحول', lambda x: '<br>'.join(x.astype(str)))
                        )
                        .reset_index()
                    )
                    fig_years = px.bar(
                        year_counts,
                        x='سنة_البيانات',
                        y='count',
                        title="",
                        hover_data={'المحولات': True}
                    )
                    fig_years.update_traces(
                        hovertemplate="<b>%{x}</b><br>عدد: %{y}<br>المحولات:<br>%{customdata[0]}"
                    )
                    st.plotly_chart(fig_years, use_container_width=True)

        st.markdown("---")

        col1, col2 = st.columns(2)

        # ✅ مخطط العلاقة بين عمر المحول وسعته
        if all(col in df_filtered.columns for col in ['KVA', 'سنة التصنيع']):
            with col1:
                st.markdown("##### 📊 العلاقة بين عمر المحول وسعته")
                df_filtered['العمر'] = 2023 - df_filtered['سنة التصنيع']
                fig_age_cap = px.scatter(
                    df_filtered,
                    x='العمر',
                    y='KVA',
                    trendline="lowess",
                    title=''
                )
                st.plotly_chart(fig_age_cap, use_container_width=True)

        # ✅ مخطط اتجاه التغذية
        if 'الاتجاه' in df_filtered.columns:
            with col2:
                st.markdown("##### 📊 اتجاه التغذية")
                plot_pie_with_hover(df_filtered, 'الاتجاه', 'اتجاه التغذية', hole_size=0.4)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### توزيع المحولات حسب الخط المغذي ")
            # مخطط توزيع الخطوط المغذية (treemap)
            if 'الخط المغذى' in df_filtered.columns:
                feeder_counts = df_filtered['الخط المغذى'].value_counts().reset_index()
                fig_feeder = px.treemap(
                    feeder_counts,
                    path=['الخط المغذى'],
                    values='count',
                    title=''
                )
                st.plotly_chart(fig_feeder, use_container_width=True)

        # st.markdown("---")
        with col2:
            # مخطط نوع القواطع (bar with hover)
            if 'نوع القاطع' in df_filtered.columns:
                st.markdown("##### أنواع القواطع المستخدمة")
                plot_bar_with_hover(df_filtered, 'نوع القاطع', 'أنواع القواطع المستخدمة')

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            # مخطط وجود التأريض (pie with hover)
            if 'التأريض' in df_filtered.columns:
                st.markdown("##### وجود نظام التأريض")
                plot_pie_with_hover(df_filtered, 'التأريض', 'وجود نظام التأريض')
        with col2:
            # تحليل Z% (box plot بدون hover أسماء)
            if 'Z%' in df_filtered.columns:
                st.markdown("##### تحليل نسبة Z%")
                fig_z = px.box(
                    df_filtered,
                    y='Z%',
                    points="all",
                    title=''
                )
                st.plotly_chart(fig_z, use_container_width=True)
        st.markdown("---")

        col1, col2 = st.columns(2)

        # جسم المحول
        if 'جسم المحول' in df_all.columns:
            with col1:
                st.markdown("##### 📦 جسم المحول")
                plot_pie_with_hover(df_filtered, 'جسم المحول', 'جسم المحول', hole_size=0.3)

        # مستوى الزيت (bar with hover)
        if 'مستوى الزيت' in df_all.columns:
            with col2:
                st.markdown("##### 🛢️ مستوى الزيت")
                plot_bar_with_hover(df_filtered, 'مستوى الزيت', 'مستوى الزيت')

        st.markdown("---")

        col1, col2 = st.columns(2)
        # السيلكا جل
        if 'السيلكا جل' in df_all.columns:
            with col1:
                st.markdown("##### 💠 السيلكا جل")
                plot_pie_with_hover(df_filtered, 'السيلكا جل', 'السيلكا جل', hole_size=0.4)
        # مانع الصواعق (bar with hover)
        if 'مانع الصواعق' in df_all.columns:
            with col2:
                st.markdown("##### ⚡ مانع الصواعق")
                plot_bar_with_hover(df_filtered, 'مانع الصواعق', 'مانع الصواعق')
        st.markdown("---")
        col4, col5 = st.columns(2)
        # الملكية (pie with hover)
        if 'الملكية' in df_all.columns:
            with col4:
                st.markdown("##### 🏷️ ملكية المحول")
                plot_pie_with_hover(df_filtered, 'الملكية', 'الملكية', hole_size=0.2)

        # خزان احتياطي (pie with hover)
        if 'خزان احتياطي' in df_all.columns:
            with col5:
                st.markdown("##### 🛢️ خزان احتياطي")
                plot_pie_with_hover(df_filtered, 'خزان احتياطي', 'خزان احتياطي', hole_size=0.3)
        st.markdown("---")
        col6, _ = st.columns(2)

        # طبيعة الأحمال (pie with hover)
        if 'طبيعة الاحمال' in df_all.columns:
            with col6:
                st.markdown("#####🔌 طبيعة الأحمال")
                plot_pie_with_hover(df_filtered, 'طبيعة الاحمال', 'طبيعة الأحمال', hole_size=0.3)

    with tabs[1]:
        st.header("📊 التحليل التاريخي للمحول")

        # 1. اختيار المحول
        selected_transformer = st.selectbox(
            "اختر اسم المحول", 
            sorted(df_all['اسم_المحول'].dropna().unique()),
            key="transformer_select"
        )

        # 2. تصفية بيانات المحول المحدد
        filtered = df_all[df_all['اسم_المحول'] == selected_transformer]

        # 3. معالجة البيانات مع حساب تغير جوهري
        def process_transformer_data(filtered_df):
            filtered_df = filtered_df.sort_values('سنة_البيانات').reset_index(drop=True)

            # حساب عمر المحول إذا متوفر سنة التصنيع
            if 'سنة التصنيع' in filtered_df.columns:
                filtered_df['عمر_المحول'] = filtered_df['سنة_البيانات'] - filtered_df['سنة التصنيع']

            # الأعمدة التي نراقب تغيرها
            cols_to_check = ['KVA', 'حالة القاطع', 'الشركة المصنعة', 'عمر_المحول']

            changes = []
            for i in range(len(filtered_df)):
                if i == 0:
                    changes.append(False)  # لا تغيير في أول صف
                    continue

                prev_row = filtered_df.loc[i-1, cols_to_check]
                curr_row = filtered_df.loc[i, cols_to_check]

                diff_count = sum(prev_row != curr_row)
                threshold = len(cols_to_check) / 2  # نصف الأعمدة

                changes.append(diff_count >= threshold)

            filtered_df['تغير_جوهري'] = changes
            return filtered_df

        processed_data = process_transformer_data(filtered)

        # 4. عرض البيانات الأساسية
        st.subheader(f"مؤشرات الاداء لتغير التاريخي لمحول : {selected_transformer}")

        # 5. التحليل المرئي: مؤشرات الأداء
        if not processed_data.empty:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("أعلى سعة سجلت", f"{processed_data['KVA'].max()} KVA")
            with col2:
                st.metric("عدد التعديلات الجوهرية", processed_data['تغير_جوهري'].sum())
            with col3:
                last_year = processed_data['سنة_البيانات'].max()
                st.metric("آخر تحديث", last_year)

            st.subheader(f"ملاحظات عامة : {selected_transformer}")
            if processed_data['تغير_جوهري'].any():
                st.warning("⚠️ تم حدوث تغييرات جوهرية في بيانات المحول خلال فترة التشغيل")
        st.markdown("---")


        # 6. عرض المخططات جنب بعض
        if not processed_data.empty:
            col_line, col_pie = st.columns(2)

            with col_line:
                st.markdown("##### 📊 تطور سعة المحول")
                fig = px.line(
                    processed_data, 
                    x='سنة_البيانات', 
                    y='KVA',
                    title='',
                    markers=True,
                    labels={'KVA': 'السعة (KVA)', 'سنة_البيانات': 'السنة'}
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_pie:
                if 'حالة القاطع' in processed_data.columns:

                    st.markdown("##### 📊 توزيع حالات المحول التاريخية ")
                    status_counts = processed_data['حالة القاطع'].value_counts().reset_index()
                    status_counts.columns = ['حالة القاطع', 'count']
                    fig2 = px.pie(
                        status_counts,
                        names='حالة القاطع',
                        values='count',
                        title='توزيع حالات المحول التاريخية'
                    )
                    st.plotly_chart(fig2, use_container_width=True)
        st.markdown("---")

        # 7. دمج مع بيانات الأحمال (إذا متوفرة)
        if 'transformer_loads' in locals():
            try:
                merged_data = pd.merge(
                    transformer_loads,
                    processed_data,
                    left_on='Transformer_id',
                    right_on='Transformer_id',
                    how='left'
                )
                
                st.subheader("البيانات المدمجة مع الأحمال")
                st.dataframe(
                    merged_data[
                        ['تاريخ القياس', 'KVA', 'حالة القاطع', 'Load_kVA']
                    ].head(),
                    hide_index=True
                )
            except Exception as e:
                st.error(f"حدث خطأ في دمج البيانات: {str(e)}")

        st.subheader(f"البيانات التاريخية للمحول: {selected_transformer}")

        num_rows = len(processed_data)
        height = min(300, 40 + num_rows * 35)

        st.dataframe(
            processed_data[
                ['سنة_البيانات', 'KVA', 'حالة القاطع', 'الشركة المصنعة', 'عمر_المحول', 'تغير_جوهري']
            ].sort_values('سنة_البيانات', ascending=False),
            height=height,
            use_container_width=True,
        )

        # جعل الجدول من اليمين لليسار
        st.markdown(
            """
            <style>
            div[data-testid="stDataFrame"] > div > div > div {
                direction: rtl;
                text-align: right;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    
    # ✅ تبويب بيانات خام
    with tabs[2]:

        # إضافة عمود السنة لكل DataFrame قبل الدمج (تأكد من تنفيذها قبل هذا الجزء)
        Transformer_data_2018['year'] = 2018
        Transformer_data_2020['year'] = 2020
        Transformer_data_2022['year'] = 2022
        transformer_data_2023['year'] = 2023

        # دمج بيانات المحولات
        all_transformer_data = pd.concat([
            Transformer_data_2018,
            Transformer_data_2020,
            Transformer_data_2022,
            transformer_data_2023
        ], ignore_index=True)

        st.header("🗂️ البيانات الكاملة للمحولات  (خام)")

        
        # أعمدة الفلترة
        col1, col2 = st.columns(2)

        with col1:
            transformer_options = ['الكل'] + sorted(all_transformer_data['اسم_المحول'].dropna().unique())
            selected_transformer = st.selectbox("اختر اسم المحول:", transformer_options, index=0, key="select_transformer")

        with col2:
            if selected_transformer == "الكل":
                available_years = sorted(all_transformer_data['year'].unique(), reverse=True)
            else:
                # فلترة السنوات حسب المحول المحدد فقط
                filtered_by_transformer = all_transformer_data[all_transformer_data['اسم_المحول'] == selected_transformer]
                available_years = sorted(filtered_by_transformer['year'].unique(), reverse=True)
            
            year_filter = st.selectbox("اختر السنة:", ["الكل"] + available_years, index=0, key="select_year")
        st.markdown("---")
        # فلترة البيانات بناءً على الاختيارات
        filtered_data = all_transformer_data.copy()

        if selected_transformer != "الكل":
            filtered_data = filtered_data[filtered_data['اسم_المحول'] == selected_transformer]

        if year_filter != "الكل":
            filtered_data = filtered_data[filtered_data['year'] == year_filter]

        # حساب ارتفاع الجدول حسب عدد الصفوف، مع الحد الأدنى والحد الأقصى للارتفاع
        rows_count = filtered_data.shape[0]
        row_height = 40  # تقريبا ارتفاع صف واحد بالبيكسل

        # احسب ارتفاع الجدول (لكن لا يزيد عن 1000 بكسل ولا يقل عن 300)
        table_height = min(max(rows_count * row_height, 200), 1000)

        

        # عرض البيانات أو رسالة تحذير
        if not filtered_data.empty:
            st.subheader("بيانات المحولات المفلترة")
            st.dataframe(
                filtered_data,
                height=table_height,
                use_container_width=True
            )
        else:
            st.warning("⚠️ لا توجد بيانات متاحة حسب معايير الفلترة المحددة")

        # دالة عرض بيانات محول
        def show_transformer_history(name, key_prefix=""):
            filtered = df_all[df_all['اسم_المحول'] == name]
            filtered_unique = filtered.drop_duplicates(subset=[col for col in filtered.columns if col != 'العام'])
            filtered_unique = filtered_unique.sort_values('سنة_البيانات')

            st.subheader(f"📅 تطور المحول: {name}")
            st.dataframe(filtered_unique)

            # التحذيرات ⚠️
            warnings = []
            if 'مستوى الزيت' in filtered_unique.columns:
                oil_vals = pd.to_numeric(filtered_unique['مستوى الزيت'], errors='coerce')  # تحويل لأرقام مع تجاهل النصوص
                low_oil = filtered_unique[oil_vals < 30]
                if not low_oil.empty:
                    warnings.append("⚠️ مستوى الزيت منخفض في بعض السنوات!")

            if 'الطقة الحالية' in filtered_unique.columns:
                current_vals = pd.to_numeric(filtered_unique['الطقة الحالية'], errors='coerce')
                delta = current_vals.diff().abs()
                if (delta > 2).any():
                    warnings.append("⚠️ تغيّر مفاجئ في الطقة الحالية!")

            if warnings:
                st.warning("\n".join(warnings))

            # رسم بياني 📈
            numeric_cols = []
            for col in ['مستوى الزيت', 'الطقة الحالية', 'Z%']:
                if col in filtered_unique.columns:
                    # نتحقق إذا العمود فيه أرقام قابلة للتحويل
                    if pd.to_numeric(filtered_unique[col], errors='coerce').notna().any():
                        numeric_cols.append(col)

            if numeric_cols:
                selected_metric = st.selectbox(
                    "📈 اختر خاصية لعرضها بيانيًا:",
                    numeric_cols,
                    key=f"{key_prefix}_metric"
                )

                if selected_metric:
                    metric_vals = pd.to_numeric(filtered_unique[selected_metric], errors='coerce')
                    fig, ax = plt.subplots()
                    ax.plot(filtered_unique['سنة_البيانات'], metric_vals, marker='o')
                    ax.set_title(f"{selected_metric} عبر السنوات")
                    ax.set_xlabel("سنة_البيانات")
                    ax.set_ylabel(selected_metric)
                    ax.grid(True)
                    st.pyplot(fig)


elif page == "📊 لوحة تحليل الأحمال":
    st.title("📊 لوحة تحليل الأحمال")
    # ضع هنا كودك الخاص بهذه الصفحة مثل الكروت أو الرسوم
    # إنشاء التاب بار الأفقي
    tabs = st.tabs(["📈 نظرة عامة", "🔍 تحليل فردي", "🗂️ جدول الصيانة", "🗂️ بيانات خام"])

    # ✅ تبويب نظرة عامة
    with tabs[0]:
        # إحصائيات عامة
        st.header("🧾 ملخص بيانات الأحمال")
        # حساب القيم
        num_transformers = agg_df['اسم_المحول'].nunique()
        max_load = all_data['Load_kVA'].max()
        avg_load_ratio = all_data['load_ratio'].mean()
        overloaded_count = agg_df[agg_df['load_status'] == 'حمل زائد'].shape[0]

        # عرضهم في أعمدة (كروت)
        # CSS مخصص لتصميم الكروت
        st.markdown("""
        <style>
        .card-container {
            display: flex;
            gap: 1rem;
            margin-bottom: 2rem;
        }
        .card {
            flex: 1;
            padding: 1rem;
            border-radius: 15px;
            background-color: #f9f9f9;
            box-shadow: 1px 1px 6px rgba(0,0,0,0.1);
            text-align: center;
        }
        .card h3 {
            margin-bottom: 0.5rem;
            font-size: 18px;
            color: #333;
        }
        .card p {
            font-size: 24px;
            font-weight: bold;
            color: #2c7be5;
            margin: 0;
        }
        </style>
        """, unsafe_allow_html=True)

        # HTML لعرض الكروت
        st.markdown(f"""
        <div class="card-container">
            <div class="card">
                <h3>🔢 عدد المحولات</h3>
                <p>{num_transformers}</p>
            </div>
            <div class="card">
                <h3>⚡ أعلى حمل مسجل (ك.ف.أ)</h3>
                <p>{max_load:.2f}</p>
            </div>
            <div class="card">
                <h3>📊 متوسط نسبة الحمل</h3>
                <p>{avg_load_ratio*100:.1f}%</p>
            </div>
            <div class="card">
                <h3>🚨 محولات تجاوزت السعة</h3>
                <p>{overloaded_count}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        # --------------------------------------------
        
        status_group = agg_df.groupby('load_status')['اسم_المحول'].agg(list).reset_index()
        status_group['count'] = status_group['اسم_المحول'].apply(len)
        status_group['tooltip'] = status_group['اسم_المحول'].apply(lambda names: '<br>'.join(names))
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.pie(
                status_group,
                values='count',
                names='load_status',
                title='',
                hover_data={'tooltip': True},
                hole=0.4
            )
            fig1.update_traces(hovertemplate='%{label}: %{value} محول<br>%{customdata[0]}')
            fig1.update_layout(showlegend=True)
            st.markdown("##### 📊 توزيع حالة التحميل")
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.line(
                agg_df,
                x='اسم_المحول',
                y='load_ratio',
                color='load_status',
                title='',
                hover_data={'load_ratio': ':.2f', 'load_status': True},
                markers=True,
                labels={
                    'اسم_المحول': 'اسم المحول',
                    'load_ratio': 'نسبة التحميل القصوى',
                    'load_status': 'حالة التحميل'
                }
            )
            fig2.add_shape(
                type='line', x0=-0.5, x1=len(agg_df)-0.5, y0=0.8, y1=0.8,
                line=dict(color='orange', dash='dot'), name='80%'
            )

            fig2.update_layout(
                xaxis=dict(
                    tickangle=-45,
                    title='اسم المحول',
                    automargin=True,
                    tickfont=dict(size=7),
                    type='category',
                    fixedrange=False  # يسمح بالسحب والتحريك
                ),
                yaxis=dict(
                    title='نسبة التحميل',
                    title_standoff=20,  # مسافة بين العنوان والمحور
                    title_font=dict(size=12),
                    side='right',       # يحرك العنوان والمحور لليمين
                    tickangle=0,        # لتدوير عناوين الأرقام تحت المحور Y إذا احتجت
                    automargin=True
                ),
                margin=dict(l=40, r=80, t=60, b=120),  # زد الهامش الأيمن لتناسب العنوان
                width=2000,
            )
            st.markdown("##### 📈 النسبة القصوى للتحميل لكل محول")
            st.plotly_chart(fig2, use_container_width=True)
        
                
        # ------------------------------------------
        # قسم تحليل جميع المحولات
        # ------------------------------------------
        
        
        # دمج بيانات الأحمال
        all_loads = pd.concat([transformer_loads_summer_2023, transformer_loads_winter_2023], ignore_index=True)
        
        # معالجة البيانات
        def preprocess_data(df):
            # تحويل أعمدة التيار إلى أرقام، مع التعامل مع القيم غير القابلة للتحويل
            for col in ['قياس التيار R', 'قياس التيار S', 'قياس التيار T']:
                df[col] = pd.to_numeric(df[col], errors='coerce')  # القيم غير الرقمية تصبح NaN

            # حساب المتوسط بعد التحويل
            currents = df[['قياس التيار R', 'قياس التيار S', 'قياس التيار T']].mean(axis=1)
            
            # باقي المعالجة (مثلاً إضافة العمود الجديد)
            df['متوسط التيار'] = currents

            return df
        
        all_loads = preprocess_data(all_loads)
        
        # جدول التوصيات
        recommendations = []
        
        # تحليل كل محول
        for transformer_id, group in all_loads.groupby('Transformer_id'):
            # الحصول على بيانات المحول الأساسية
            transformer_info = transformer_data_2023[transformer_data_2023['Transformer_id'] == transformer_id].iloc[0]
            capacity = transformer_info['KVA']
            transformer_name = transformer_info['اسم_المحول']
            manufacturing_year = transformer_info['سنة التصنيع']
            transformer_age = 2023 - manufacturing_year
            
            # تحليل الأحمال
            group = group.sort_values('تاريخ القياس')
            group['Days'] = (group['تاريخ القياس'] - group['تاريخ القياس'].min()).dt.days
            group['Max_Load_KVA'] = group[['قياس التيار R', 'قياس التيار S', 'قياس التيار T']].max(axis=1) * 400 / 1000
            group['Imbalance'] = group[['قياس التيار R', 'قياس التيار S', 'قياس التيار T']].std(axis=1)
            
            # تحليل الاتجاه الزمني بنافذة متحركة
            window_size = min(30, len(group))
            group['Rolling_Avg'] = group['Max_Load_KVA'].rolling(window=window_size).mean()
            
            # حساب المؤشرات
            max_load = group['Max_Load_KVA'].max()
            avg_load = group['Max_Load_KVA'].mean()
            load_ratio = max_load / capacity
            over_80_pct = (group['Max_Load_KVA'] > capacity * 0.8).mean() * 100
            imbalance_avg = group['Imbalance'].mean()
            
            # تحليل الاتجاه
            if len(group) >= window_size:
                trend_diff = group['Rolling_Avg'].iloc[-1] - group['Rolling_Avg'].iloc[0]
                trend_dir = "تزايد" if trend_diff > 0 else "تناقص"
                group = group.dropna(subset=['Days', 'Max_Load_KVA'])
                if not group.empty:
                    X = group[['Days']]
                    y = group['Max_Load_KVA']

                    model = LinearRegression()
                    model.fit(X, y)

                    future_days = np.array([[group['Days'].max() + 180]])
                    future_load = model.predict(future_days)[0]
                else:
                    future_load = None  # أو ضع قيمة افتراضية
                    print(f"⛔ لا توجد بيانات كافية لتدريب النموذج للمحول {transformer_id}")
            else:
                trend_diff = 0
                trend_dir = "غير محدد"
                future_load = max_load
            
            # حساب درجة الخطورة
            risk_score = 0
            
            # 1. تجاوز السعة
            if max_load > capacity * 1.1:
                risk_score += 3
                capacity_status = "تجاوز خطير"
            elif max_load > capacity:
                risk_score += 2
                capacity_status = "تجاوز"
            else:
                capacity_status = "ضمن السعة"
            
            # 2. نسبة التحميل العالية
            if over_80_pct > 70:
                risk_score += 2
            elif over_80_pct > 50:
                risk_score += 1
            
            # 3. اتجاه التحميل
            if trend_dir == "تزايد" and abs(trend_diff) > capacity * 0.1:
                risk_score += 1
            
            # 4. عدم التوازن
            if imbalance_avg > 0.2:
                risk_score += 1
            
            # 5. العمر
            if transformer_age > 20:
                risk_score += 1
            
            # توليد التوصية
            if risk_score >= 5:
                recommendation = "🔴 خطر عالي: يتطلب تدخل فوري (استبدال/توسعة)"
                action = "نقترح إيقاف التشغيل فوراً واتخاذ إجراءات عاجلة"
            elif risk_score >= 3:
                recommendation = "🟠 خطر متوسط: يحتاج مراقبة مكثفة"
                action = "زيادة وتيرة الصيانة وتخفيف الأحمال خلال الذروة"
            elif risk_score >= 1:
                recommendation = "🟡 خطر منخفض: مراقبة روتينية"
                action = "المتابعة حسب الجدول الزمني المعتاد"
            else:
                recommendation = "🟢 وضع طبيعي"
                action = "لا إجراءات ضرورية حالياً"
            
            # إضافة تفاصيل إضافية
            details = []
            if max_load > capacity:
                details.append(f"تجاوز السعة بنسبة {(max_load/capacity-1)*100:.1f}%")
            if over_80_pct > 50:
                details.append(f"{over_80_pct:.1f}% من القراءات فوق 80% من السعة")
            if imbalance_avg > 0.2:
                details.append(f"عدم توازن عالٍ ({imbalance_avg:.2f})")
            if transformer_age > 20:
                details.append(f"عمر المحول {transformer_age} سنة")
            
            details_str = "، ".join(details) if details else "لا توجد مشاكل رئيسية"
            
            # حفظ التوصية
            recommendations.append({
                'ID المحول': transformer_id,
                'اسم المحول': transformer_name,
                'السعة (KVA)': capacity,
                'أعلى حمل (KVA)': f"{max_load:.1f}",
                'الحالة': capacity_status,
                'درجة الخطورة': risk_score,
                'التوصية': recommendation,
                'الإجراء المقترح': action,
                'التفاصيل': details_str,
                'الحمل المتوقع بعد 6 أشهر': f"{future_load:.1f}",
                'اتجاه الحمل': trend_dir
            })
        
        st.markdown("---")

        # عرض النتائج
        if recommendations:
            rec_df = pd.DataFrame(recommendations)
            
            # تصنيف حسب درجة الخطورة
            rec_df = rec_df.sort_values('درجة الخطورة', ascending=False)
            
            # تصور بياني
            st.markdown("##### 📈 تصور بيانات الأحمال")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.bar(rec_df, x='اسم المحول', y='أعلى حمل (KVA)', 
                            color='درجة الخطورة',
                            title='',
                            hover_data=['السعة (KVA)'])
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = px.scatter(rec_df, x='السعة (KVA)', y='أعلى حمل (KVA)',
                                color='التوصية',
                                size='درجة الخطورة',
                                title='',
                                hover_name='اسم المحول')
                st.plotly_chart(fig2, use_container_width=True)
            
            st.markdown("---")
            st.markdown("##### 📊 تحليل شامل لجميع المحولات")

            # عرض الجدول
            st.dataframe(rec_df, height=600)
            
            
        else:
            st.warning("لا توجد بيانات كافية لتحليل المحولات.")
    

    # ✅ تبويب تحليل فردي
    with tabs[1]:
        # ------------------------------------------
        # قسم اختيار المحول
        # ------------------------------------------
        st.header("🔍 تحليل مفصل للمحول")

        # دمج جميع بيانات المحولات من السنوات المختلفة
        all_transformer_data = pd.concat([
            Transformer_data_2018,
            Transformer_data_2020,
            Transformer_data_2022,
            transformer_data_2023
        ]).drop_duplicates(subset=['Transformer_id'], keep='last')

        # دمج جميع بيانات الأحمال
        all_loads = pd.concat([
            transformer_loads_summer_2016,
            transformer_loads_summer_2017,
            transformer_loads_summer_2018,
            transformer_loads_summer_2019,
            transformer_loads_summer_2022,
            transformer_loads_summer_2023,
            transformer_loads_winter_2017,
            transformer_loads_winter_2018,
            transformer_loads_winter_2019,
            transformer_loads_winter_2021,
            transformer_loads_winter_2023
        ])

        # قسم الفلترة المتداخلة
        col1, col2, col3 = st.columns(3)

        with col1:
            # # فلتر المحول (الأساسي)
            transformer_options = sorted(all_transformer_data['اسم_المحول'].dropna().unique())
            selected_transformer = st.selectbox("اختر اسم المحول:", transformer_options)
            
            # تطبيق فلتر المحول
            transformer_data = all_transformer_data[all_transformer_data['اسم_المحول'] == selected_transformer]
            selected_transformer_id = transformer_data['Transformer_id'].iloc[0]

            # تخزين القيم في session_state
            st.session_state['selected_transformer'] = selected_transformer
            st.session_state['selected_transformer_id'] = selected_transformer_id

            transformer_loads = all_loads[all_loads['Transformer_id'] == selected_transformer_id]

        with col2:
            # فلتر الموسم (يعتمد على المحول المختار)
            # نستخرج المواسم المتاحة لهذا المحول فقط
            transformer_loads['تاريخ القياس'] = pd.to_datetime(transformer_loads['تاريخ القياس'], errors='coerce')
            
            # دالة لتحديد الموسم من الشهر
            def get_season(month):
                if month in [12, 1, 2, 3, 4, 5]:
                    return 'شتوي'
                elif month in [6, 7, 8, 9, 10, 11]:
                    return 'صيفي'
                # return 'موسم آخر'
            
            transformer_loads['الموسم'] = transformer_loads['تاريخ القياس'].dt.month.apply(get_season)
            available_seasons = ['الكل'] + sorted(transformer_loads['الموسم'].dropna().unique())
            season_filter = st.selectbox("اختر الموسم:", available_seasons)

            # تطبيق فلتر الموسم
            if season_filter != 'الكل':
                transformer_loads = transformer_loads[transformer_loads['الموسم'] == season_filter]

        with col3:
            # فلتر اتجاه السكينة (يعتمد على المحول والموسم المختار)
            if 'اتجاه السكينة' in transformer_loads.columns:
                # نستخرج الاتجاهات المتاحة بناءً على الفلاتر السابقة
                available_directions = ['الكل'] + sorted(transformer_loads['اتجاه السكينة'].dropna().astype(str).unique())
                direction_filter = st.selectbox("اختر اتجاه السكينة:", available_directions)
                
                # تطبيق فلتر الاتجاه
                if direction_filter != 'الكل':
                    transformer_loads = transformer_loads[transformer_loads['اتجاه السكينة'] == direction_filter]
            else:
                direction_filter = "الكل"

        st.markdown("---")

        # ------------------------------------------
        # قسم المعلومات الأساسية
        # ------------------------------------------
        
        st.subheader(f" 📋 معلومات أساسية عن محول  {selected_transformer}")


        # تصفية حسب اتجاه السكينة إذا لم يكن "الكل"
        # if direction_filter != "الكل" and 'اتجاه السكينة' in transformer_loads.columns:
        #     transformer_loads = transformer_loads[transformer_loads["اتجاه السكينة"] == direction_filter]
        # عرض المعلومات الأساسية في أعمدة
        col1, col2, col3 = st.columns(3)
        # def calculate_remaining_life(year_of_manufacture, design_life_years=30):
        #     current_year = datetime.datetime.now().year
        #     age = current_year - year_of_manufacture
        #     remaining_life = design_life_years - age
        #     return max(remaining_life, 0)
        # دالة للحصول على العمر الافتراضي حسب الشركة المصنعة
        def get_expected_life(manufacturer):
            lifetimes = {
                'ABB': 35,
                'ARDAN': 30,
                'ASTOR': 27,
                'best': 27,
                'ELKIMA': 27,
                'French': 35,
                'imefy': 27,
                'MEKSAN': 27,
                'SEM': 27,
                'TRAFO': 30,
                'TURKEY': 27,
                'VONROLL': 32,
                'ZENNARO': 27,
                'لايوجد': 30,
                None: 30,
                'nan': 30
            }
            # التعامل مع القيم النصية التي تمثل NaN
            if pd.isna(manufacturer):
                return 30
            return lifetimes.get(str(manufacturer).strip(), 30)  # القيمة الافتراضية 30 إذا غير موجودة

        # حساب العمر الافتراضي المتبقي
        def calculate_remaining_life(manufacturer, manufacture_year):
            expected_life = get_expected_life(manufacturer)
            try:
                manufacture_year = int(manufacture_year)
                current_year = datetime.now().year
                age = current_year - manufacture_year
                remaining_life = expected_life - age
                if remaining_life < 0:
                    remaining_life = 0
                return remaining_life
            except Exception as e:
                return None

        # st.write(transformer_data.columns.tolist())
        with col1:
            st.metric("السعة الاسمية (KVA)", transformer_data['KVA'].iloc[0])
            if 'سنة التصنيع' in transformer_data.columns:
                st.metric("سنة التصنيع", transformer_data['سنة التصنيع'].iloc[0])

            if 'سنة التصنيع' in transformer_data.columns:
                # st.metric("سنة التصنيع", transformer_data['سنة التصنيع'].iloc[0])
                year_of_manufacture_raw = transformer_data['سنة التصنيع'].iloc[0]
                # st.write(f"سنة التصنيع الخام: {year_of_manufacture_raw} (نوع: {type(year_of_manufacture_raw)})")
                try:
                    year_of_manufacture = int(year_of_manufacture_raw)
                    remaining_life = calculate_remaining_life(year_of_manufacture)
                    # st.write(f"العمر المتبقي محسوب: {remaining_life}")
                except Exception as e:
                    # st.write(f"خطأ في تحويل سنة التصنيع إلى عدد صحيح: {e}")
                    remaining_life = None
            else:
                remaining_life = None

        with col2:
            if 'الشركة المصنعة' in transformer_data.columns:
                st.metric("الشركة المصنعة", transformer_data['الشركة المصنعة'].iloc[0])
            if 'نسبة التحميل حسب اعلى فاز' in transformer_data.columns:
                st.metric("نسبة التحميل الحالية", 
                        f"{transformer_data['نسبة التحميل حسب اعلى فاز'].iloc[0]}%")

        with col3:
            if 'حالة القاطع' in transformer_data.columns:
                st.metric("حالة المحول", transformer_data['حالة القاطع'].iloc[0])
                # حساب العمر الافتراضي المتبقي
                manufacturer = transformer_data['الشركة المصنعة'].iloc[0] if 'الشركة المصنعة' in transformer_data.columns else None
                manufacture_year = transformer_data['سنة التصنيع'].iloc[0] if 'سنة التصنيع' in transformer_data.columns else None

                remaining_life = calculate_remaining_life(manufacturer, manufacture_year)

                if remaining_life is not None:
                    st.metric("العمر الافتراضي المتبقي (سنة)", remaining_life)
                else:
                    st.write("العمر الافتراضي المتبقي غير متوفر")
        
        st.markdown("---")

        # ------------------------------------------
        # قسم بطاقات الأداء الرئيسية
        # ------------------------------------------
        # حساب القيم لتحليل المحول
        
        st.subheader(f" 📋  مؤشرات الأداء حول محول  {selected_transformer}")
        if not transformer_loads.empty:
            # حساب Max_Load_KVA
            معامل_تحويل = 1.732 / 1000
            جهد_الخط = 400
            cols = ['قياس التيار R', 'قياس التيار S', 'قياس التيار T']
            transformer_loads[cols] = transformer_loads[cols].apply(pd.to_numeric, errors='coerce')
            transformer_loads['Load_kVA'] = transformer_loads[cols].max(axis=1) * جهد_الخط * معامل_تحويل
            transformer_loads['load_ratio'] = transformer_loads['Load_kVA'] / transformer_data['KVA'].iloc[0]
            
            max_load = transformer_loads['Load_kVA'].max()
            min_load = transformer_loads['Load_kVA'].min()
            avg_load = transformer_loads['Load_kVA'].mean()
            over_80_count = (transformer_loads["load_ratio"] > 0.7).sum()
            over_100_count = (transformer_loads["load_ratio"] > 0.95).sum()
            capacity = transformer_data['KVA'].iloc[0]
            most_used_direction = transformer_loads['اتجاه السكينة'].mode()[0] if 'اتجاه السكينة' in transformer_loads.columns else "غير متوفر"

            # تعريف الستايل والكروت
            st.markdown("""
            <style>
            .card-container {
                display: flex;
                gap: 1rem;
                margin-bottom: 2rem;
                flex-wrap: nowrap;
            }
            .card-container.row-4 {
                flex-wrap: nowrap;
            }
            .card-container.row-2 {
                flex-wrap: nowrap;
            }
            .card {
                flex: 1 1 100%;
                padding: 1rem;
                border-radius: 15px;
                background-color: #f9f9f9;
                box-shadow: 1px 1px 6px rgba(0,0,0,0.1);
                text-align: center;
                transition: transform 0.2s ease;
            }
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 2px 4px 10px rgba(0,0,0,0.15);
            }
            .card h3 {
                margin-bottom: 0.5rem;
                font-size: 18px;
                color: #333;
            }
            .card p {
                font-size: 24px;
                font-weight: bold;
                margin: 0;
            }
            .row-4 .card {
                flex: 1 1 calc(25% - 1rem);
            }
            .row-2 .card {
                flex: 1 1 calc(33.33% - 1rem);
            }
            .card.blue p { color: #2563eb; }
            .card.green p { color: #16a34a; }
            .card.orange p { color: #f97316; }
            .card.red p { color: #dc2626; }
            </style>
            """, unsafe_allow_html=True)

            # عرض بطاقات الأداء
            st.markdown(f"""
            <div class="card-container row-4">
                <div class="card blue">
                    <h3>⚡️ أعلى حمل (ك.ف.أ)</h3>
                    <p>{max_load:.2f}</p>
                </div>
                <div class="card green">
                    <h3>🔻 أدنى حمل</h3>
                    <p>{min_load:.2f}</p>
                </div>
                <div class="card blue">
                    <h3>📊 متوسط الحمل</h3>
                    <p>{avg_load:.2f}</p>
                </div>
                <div class="card orange">
                    <h3>🚨 مرات تجاوز 80%</h3>
                    <p>{over_80_count}</p>
                </div>
            </div>

            <div class="card-container row-2">
                <div class="card red">
                    <h3>⚠️ مرات تجاوز 100%</h3>
                    <p>{over_100_count}</p>
                </div>
                <div class="card blue">
                    <h3>📦 سعة المحول</h3>
                    <p>{capacity} ك.ف.أ</p>
                </div>
                <div class="card green">
                    <h3>🔁 اتجاه السكينة الأكثر استخدامًا</h3>
                    <p>{most_used_direction}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        
        # ------------------------------------------
        # قسم التحليل البصري
        # ------------------------------------------
        
        st.subheader(f" 📋 تحليل بصري لأحمال المحول  {selected_transformer}")
        st.markdown("######")

        if not transformer_loads.empty:
            # صف للرسم الشريطي والدائري
            col1, col2 = st.columns(2)

            # تحويل العمود لتاريخ إذا لم يكن كذلك
            transformer_loads["تاريخ القياس"] = pd.to_datetime(transformer_loads["تاريخ القياس"], dayfirst=True)

            # تعريف دالة لحساب الموسم بناء على الشهر
            def get_season(date_str):
                date = pd.to_datetime(date_str, dayfirst=True)
                month = date.month
                if month in [12, 1, 2, 3, 4, 5]:
                    return "شتوي"
                elif month in [6, 7, 8, 9, 10, 11]:
                    return "صيفي"
                else:
                    return "موسم آخر"

            transformer_loads["الموسم"] = transformer_loads["تاريخ القياس"].apply(get_season)

            with col1:
                # تحضير البيانات
                st.markdown("##### 📉 توزيع التجاوزات الموسمية")
                seasonal_data = transformer_loads.assign(
                    تجاوز_80 = lambda x: x["load_ratio"] > 0.7,
                    الموسم = lambda x: x["تاريخ القياس"].dt.month.map({
                        12: "شتوي", 1: "شتوي", 2: "شتوي", 3: "شتوي", 4: "شتوي", 5: "شتوي",
                        6: "صيفي", 7: "صيفي", 8: "صيفي", 9: "صيفي", 10: "صيفي", 11: "صيفي",
                    }).fillna("انتقالي")
                )
                rtl = "\u200F"
                fig = px.sunburst(
                    seasonal_data,
                    path=['الموسم', 'تجاوز_80'],
                    color='الموسم',
                    color_discrete_map={'شتوي':'#636EFA', 'صيفي':'#EF553B', 'انتقالي':'#00CC96'},
                    title=''
                )
                fig.update_traces(
                    textinfo="label+percent parent",
                    hovertemplate="<b>%{label}</b><br>النسبة: %{percentParent:.1%}<br>العدد: %{value}"
                )
                fig.update_layout(
                    font=dict(family="IBM Plex Sans Arabic", size=14)
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("##### 📉 توزيع اتجاهات السكينة")
                if 'اتجاه السكينة' in transformer_loads.columns:
                    dir_counts = transformer_loads["اتجاه السكينة"].value_counts().reset_index()
                    dir_counts.columns = ['اتجاه السكينة', 'التكرار']
                    fig_pie = px.pie(dir_counts, names='اتجاه السكينة', values='التكرار',
                                    color_discrete_sequence=px.colors.qualitative.Pastel,
                                    hole=0)
                    fig_pie.update_traces(textinfo='percent+label', hovertemplate='اتجاه السكينة: %{label}<br>التكرار: %{value}<extra></extra>')
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.warning("لا توجد بيانات عن اتجاه السكينة")
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                # تحضير البيانات
                st.markdown("##### 📉 توزيع التجاوزات حسب الشهر ونصف السنة")
                heatmap_data = transformer_loads.assign(
                    شهر = lambda x: x["تاريخ القياس"].dt.month,
                    نصف_سنة = lambda x: x["تاريخ القياس"].dt.month.map(
                        lambda m: "الأول" if m <= 6 else "الثاني"
                    ),
                    تجاوز = lambda x: x["load_ratio"] > 0.8
                ).pivot_table(
                    index='نصف_سنة',   # الصفوف
                    columns='شهر',     # الأعمدة
                    values='تجاوز',
                    aggfunc='sum',
                    fill_value=0
                )

                # إنشاء المخطط
                fig = px.imshow(
                    heatmap_data,
                    labels=dict(x="الشهر", y="نصف السنة", color="عدد التجاوزات"),
                    color_continuous_scale='OrRd',
                    title=''
                )
                fig.update_xaxes(side="top")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("##### 📉 تراكم التجاوزات بمرور الوقت")
                cumulative_data = transformer_loads.assign(
                    تاريخ = lambda x: x["تاريخ القياس"].dt.date,
                    تجاوز = lambda x: x["load_ratio"] > 0.8
                ).groupby(['تاريخ', 'الموسم'])['تجاوز'].sum().reset_index()
                
                fig = px.area(
                    cumulative_data,
                    x='تاريخ',
                    y='تجاوز',
                    color='الموسم',
                    title='',
                    labels={'تجاوز': 'عدد التجاوزات', 'تاريخ': 'التاريخ'},
                    color_discrete_map={'شتوي':'#636EFA', 'صيفي':'#EF553B', 'انتقالي':'#00CC96'}
                )
                fig.update_traces(mode='lines+markers')
                st.plotly_chart(fig, use_container_width=True)
           
            st.markdown("---")
            # الرسم الخطي للأحمال بمرور الوقت
            st.markdown("##### 📈 تطور الأحمال بمرور الوقت")
            load_chart_data = transformer_loads.set_index("تاريخ القياس").sort_index()
            fig = px.line(
                load_chart_data,
                x=load_chart_data.index,
                y="Load_kVA",
                labels={
                    "x": "تاريخ القياس",
                    "Load_kVA": "الحمل (kVA)"
                },
                title=""
            )

            st.plotly_chart(fig, use_container_width=True)

            analysis_data = transformer_loads.assign(
                تجاوز_80 = lambda x: np.where(x["load_ratio"] > 0.8, "تجاوز", "طبيعي"),
                الشهر = lambda x: x["تاريخ القياس"].dt.month_name(),
                اليوم = lambda x: x["تاريخ القياس"].dt.day_name(),
                الساعة = lambda x: x["تاريخ القياس"].dt.hour
            )
            st.markdown("---")
            # الصف الأول: المخطط الموسمي
            col1, col2, = st.columns(2)
            with col1:
                st.markdown("##### 📈المخطط الموسمي للأحمال")
                fig1 = px.pie(
                    analysis_data,
                    names='تجاوز_80',
                    facet_col='الموسم',
                    title=''
                )
                st.plotly_chart(fig1, use_container_width=True)

            # الصف الثاني: توزيع يوم/ساعة
            with col2:
                st.markdown("##### 📈توزيع الأحمال يوم/ ساعة ")
                fig2 = px.density_heatmap(
                    analysis_data,
                    x='الساعة',
                    y='اليوم',
                    z='load_ratio',
                    histfunc="avg",
                    title=''
                )
                st.plotly_chart(fig2, use_container_width=True)
            st.markdown("---")

            # ------------------------------------------
            # قسم التحليل المتقدم
            # ------------------------------------------

            if not transformer_loads.empty:
                # تحليل موسمي
                transformer_loads['الموسم'] = transformer_loads['تاريخ القياس'].dt.month.apply(
                    lambda m: "شتوي" if m in [12, 1, 2, 3] else ("صيفي" if m in [6, 7, 8] else "انتقالي")
                )
                if season_filter != "الكل":
                    transformer_loads = transformer_loads[transformer_loads["الموسم"] == season_filter]
                    transformer_loads = transformer_loads[transformer_loads["اتجاه السكينة"] == direction_filter]
                #  تحليل الاتجاه والتنبؤ
                st.subheader(f" 📋 تحليل الاتجاه والتنبؤ لمحول {selected_transformer}")
                # st.subheader("تحليل الاتجاه والتنبؤ")
                
                transformer_loads['تاريخ القياس'] = pd.to_datetime(transformer_loads['تاريخ القياس'])
                transformer_loads['Days'] = (transformer_loads['تاريخ القياس'] - transformer_loads['تاريخ القياس'].min()).dt.days
                transformer_loads = transformer_loads.dropna(subset=['Load_kVA', 'تاريخ القياس'])
                X = transformer_loads[['Days']]
                y = transformer_loads['Load_kVA']
                model = LinearRegression()
                if not X.empty and not y.empty:
                    model.fit(X, y)
                    
                    # فقط بعد التأكد من التدريب، نفذ التنبؤ
                    future_days = np.array([[transformer_loads['Days'].max() + i] for i in [30, 90, 180]])
                    future_predictions = model.predict(future_days)

                    # تابع عرض النتائج
                    ...
                else:
                    st.warning("لا توجد بيانات كافية لإجراء التنبؤ.")
                    future_predictions = []  # أو قيم افتراضية
                
                # عرض النتائج
                col1, col2 = st.columns(2)
                
                # تأكد من أن y لا يحتوي على NaN
                y = transformer_loads['Load_kVA'].dropna()

                # إذا بقي فارغًا بعد التنظيف
                if y.empty:
                    st.warning("لا توجد بيانات كافية للتنبؤ بالحمل.")
                else:
                    # نفذ التنبؤات كالمعتاد
                    model.fit(X, y)
                    future_days = np.array([[transformer_loads['Days'].max() + i] for i in [30, 90, 180]])
                    future_predictions = model.predict(future_days)

                    delta1 = future_predictions[0] - y.iloc[-1]
                    delta6 = future_predictions[2] - y.iloc[-1]

                    with col1:
                        st.metric("الاتجاه الحالي", 
                                "تصاعدي" if model.coef_[0] > 0 else "تنازلي",
                                delta=f"{model.coef_[0]:.2f} KVA/يوم")

                        st.metric("الحمل المتوقع بعد شهر", 
                                f"{future_predictions[0]:.1f} KVA",
                                delta=f"{delta1:.1f} KVA")

                    with col2:
                        st.metric("معدل النمو اليومي", 
                                f"{model.coef_[0]:.2f} KVA/يوم")

                        st.metric("الحمل المتوقع بعد 6 أشهر", 
                                f"{future_predictions[2]:.1f} KVA",
                                delta=f"{delta6:.1f} KVA")
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(f" 📋 تحليل متقدم لأحمال ")
                    # رسم بياني للأحمال مع الوقت
                    fig = px.line(transformer_loads, x='تاريخ القياس', y='Load_kVA',
                                color='الموسم',
                                title=f"",
                                hover_data=['قياس التيار R', 'قياس التيار S', 'قياس التيار T'])
                    
                    fig.add_hline(y=transformer_data['KVA'].iloc[0], line_dash="dash", 
                                line_color="red", annotation_text="السعة القصوى")
                    
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    # تحليل التوازن بين الفازات
                    st.subheader(f" 📋 تحليل التوازن بين الفازات")
                    transformer_loads['Imbalance'] = transformer_loads[['قياس التيار R', 'قياس التيار S', 'قياس التيار T']].std(axis=1)

                    imbalance_data = transformer_loads[['تاريخ القياس', 'Imbalance']].copy()
                    imbalance_data['Imbalance_Status'] = np.where(
                        imbalance_data['Imbalance'] > 0.2, "غير متوازن", "متوازن"
                    )
                    fig_imbalance = px.scatter(imbalance_data, x='تاريخ القياس', y='Imbalance',
                                            color='Imbalance_Status',
                                            title="",
                                            labels={'Imbalance': 'معدل عدم التوازن'})
                    
                    fig_imbalance.add_hline(y=0.2, line_dash="dash", 
                                        line_color="red", annotation_text="حد التحذير")
                    
                    st.plotly_chart(fig_imbalance, use_container_width=True)
                st.markdown("---")
                # توصية مخصصة للمحول
                st.subheader(f" 🛠 توصيات مخصصة لمحول {selected_transformer}")
                

                recs = generate_recommendations(transformer_loads, transformer_data, selected_transformer=selected_transformer)
                
                display_recommendations(recs)


        else:
            st.warning("لا توجد بيانات أحمال متاحة لهذا المحول.")

    with tabs[2]:
        display_maintenance_tab()



    # ✅ تبويب بيانات خام

    with tabs[3]:
        st.header("🗂️ البيانات الكاملة لأحمال المحولات (خام)")
        
        # دمج جميع بيانات الأحمال في DataFrame واحد
        summer_loads = pd.concat([
            transformer_loads_summer_2016.assign(season='صيفي', year=2016),
            transformer_loads_summer_2017.assign(season='صيفي', year=2017),
            transformer_loads_summer_2018.assign(season='صيفي', year=2018),
            transformer_loads_summer_2019.assign(season='صيفي', year=2019),
            transformer_loads_summer_2022.assign(season='صيفي', year=2022),
            transformer_loads_summer_2023.assign(season='صيفي', year=2023)
        ])
        
        winter_loads = pd.concat([
            transformer_loads_winter_2017.assign(season='شتوي', year=2017),
            transformer_loads_winter_2018.assign(season='شتوي', year=2018),
            transformer_loads_winter_2019.assign(season='شتوي', year=2019),
            transformer_loads_winter_2021.assign(season='شتوي', year=2021),
            transformer_loads_winter_2023.assign(season='شتوي', year=2023)
        ])
        
        all_loads_combined = pd.concat([summer_loads, winter_loads])
        
        # إنشاء أعمدة الفلترة
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # فلترة حسب المحول - الإختيار الافتراضي "الكل"
            transformer_options = ['الكل'] + sorted(all_transformer_data['اسم_المحول'].dropna().unique())
            selected_transformer = st.selectbox(
                "اختر اسم المحول:",
                transformer_options,
                index=0,
                key="select_transformer"
            )
            
            if selected_transformer == "الكل":
                filtered_data = all_loads_combined
                selected_transformer_id = None
            else:
                transformer_data = all_transformer_data[all_transformer_data['اسم_المحول'] == selected_transformer]
                selected_transformer_id = transformer_data['Transformer_id'].iloc[0]
                filtered_data = all_loads_combined[all_loads_combined['Transformer_id'] == selected_transformer_id]
        
        with col2:
            # فلترة حسب السنة - تتغير بناءً على المحول المحدد
            if selected_transformer == "الكل":
                available_years = sorted(all_loads_combined['year'].unique(), reverse=True)
            else:
                available_years = sorted(filtered_data['year'].unique(), reverse=True)
                
            year_filter = st.selectbox(
                "اختر السنة:",
                ["الكل"] + available_years,
                index=0,
                key="select_year"
            )
            
            if year_filter != "الكل":
                filtered_data = filtered_data[filtered_data['year'] == year_filter]
        
        with col3:
            # فلترة حسب الموسم - تتغير بناءً على المحول والسنة المحددين
            if selected_transformer == "الكل" and year_filter == "الكل":
                season_options = ["الكل", "صيفي", "شتوي"]
            else:
                available_seasons = filtered_data['season'].unique()
                season_options = ["الكل"] + list(available_seasons)
                
            season_filter = st.selectbox(
                "اختر الموسم:",
                season_options,
                index=0,
                key="select_season"
            )
            
            if season_filter != "الكل":
                filtered_data = filtered_data[filtered_data['season'] == season_filter]
        
        with col4:
            # فلترة حسب اتجاه السكينة - تتغير بناءً على الفلاتر السابقة
            if 'اتجاه السكينة' in filtered_data.columns:
                if len(filtered_data) > 0:
                    available_directions = ['الكل'] + sorted(filtered_data['اتجاه السكينة'].dropna().astype(str).unique())
                else:
                    available_directions = ['الكل']
                    
                direction_filter = st.selectbox(
                    "اختر اتجاه السكينة:",
                    available_directions,
                    index=0,
                    key="select_direction"
                )
                
                if direction_filter != "الكل":
                    filtered_data = filtered_data[filtered_data['اتجاه السكينة'] == direction_filter]
            else:
                direction_filter = "الكل"
                st.selectbox("اختر اتجاه السكينة:", ["الكل"], disabled=True)
        
        # عرض البيانات مع تبويب حسب الموسم
        if not filtered_data.empty:
            tab1, tab2 = st.tabs(["إحصائيات مختصرة", "عرض جدولي"])
            
            with tab1:
                st.subheader("📊 ملخص إحصائي")
                
                # حساب السعة إذا كان محول معين محدد
                capacity = None
                if selected_transformer != "الكل":
                    capacity = transformer_data['KVA'].iloc[0]
                    st.metric("سعة المحول", f"{capacity} kVA")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("عدد القراءات", filtered_data.shape[0])
                    st.metric("أعلى حمل", f"{filtered_data['Load_kVA'].max():.2f} kVA")
                
                with col2:
                    st.metric("متوسط الحمل", f"{filtered_data['Load_kVA'].mean():.2f} kVA")
                    st.metric("أدنى حمل", f"{filtered_data['Load_kVA'].min():.2f} kVA")
                
                with col3:
                    if capacity:
                        overload_percentage = (filtered_data['Load_kVA'] > capacity).mean() * 100
                        st.metric("نسبة تجاوز السعة", f"{overload_percentage:.1f}%")
                        high_load_percentage = (filtered_data['Load_kVA'] > capacity * 0.8).mean() * 100
                        st.metric("نسبة الأحمال العالية (>80%)", f"{high_load_percentage:.1f}%")
                    else:
                        st.metric("عدد المحولات", filtered_data['Transformer_id'].nunique())
                
                # رسم توزيع الأحمال
                st.plotly_chart(
                    px.histogram(
                        filtered_data,
                        x='Load_kVA',
                        nbins=20,
                        title='توزيع قيم الأحمال',
                        labels={'Load_kVA': 'الحمل (kVA)'},
                        color='season' if season_filter == "الكل" else None
                    ),
                    use_container_width=True
                )
            with tab2:
                st.dataframe(
                    filtered_data,
                    height=600,
                    column_config={
                        "تاريخ القياس": st.column_config.DatetimeColumn("التاريخ", format="DD/MM/YYYY HH:mm"),
                        "Load_kVA": st.column_config.NumberColumn("الحمل (kVA)", format="%.2f"),
                        "season": "الموسم",
                        "year": "السنة"
                    },
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.warning("⚠️ لا توجد بيانات متاحة للعرض حسب معايير الفلترة المحددة")
            
            # عرض خيارات متاحة للمستخدم لتعديل الفلترة
            if selected_transformer != "الكل":
                available_data = all_loads_combined[all_loads_combined['Transformer_id'] == selected_transformer_id]
                
                st.info("البيانات المتوفرة لهذا المحول:")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("السنوات المتاحة:", ", ".join(map(str, sorted(available_data['year'].unique()))))
                
                with col2:
                    st.write("المواسم المتاحة:", ", ".join(available_data['season'].unique()))
                
                if 'اتجاه السكينة' in available_data.columns:
                    st.write("اتجاهات السكينة المتاحة:", ", ".join(available_data['اتجاه السكينة'].dropna().astype(str).unique()))
                st.info(f"السنوات المتوفرة لهذا المحول: {', '.join(map(str, sorted(available_years)))}")
elif page == "🔧 الصيانة والأعطال":
    st.title("🔧 الصيانة والأعطال")
    # جدول أو بيانات مفصلة
    tabs = st.tabs(["📈 نظرة عامة", "🔍 تحليل فردي", "🗂️ بيانات خام"])
    # ✅ تبويب نظرة عامة
    with tabs[0]:
        st.write("هنا محتوى نظرة عامة الصيانة والاعطال...")
    # ✅ تبويب تحليل فردي
    with tabs[1]:
        st.write("هنا محتوى تحليل فردي للصيانة والاعطال...")
    # ✅ تبويب بيانات خام
    with tabs[2]:
        st.write("هنا محتوى بيانات خام للصيانة والاعطال...")
elif page == "🏗 المشاريع التطويرية":
    st.title("🏗 المشاريع التطويرية")
    # جدول أو بيانات مفصلة
    tabs = st.tabs(["📈 نظرة عامة", "🔍 تحليل فردي", "🗂️ بيانات خام"])
    # ✅ تبويب نظرة عامة
    with tabs[0]:
        st.write("هنا محتوى نظرة عامة على المشاريع التطويرية ...")
    
    # ✅ تبويب تحليل فردي
    with tabs[1]:
        st.write("هنا محتوى تحليل فردي للمشاريع التطويرية ...")
    
    # ✅ تبويب بيانات خام
    with tabs[2]:
        st.write("هنا محتوى بيانات الخام للمشاريع التطويرية ...")
elif page == "📈 تحليلات متقدمة":
    st.title("📈 تحليلات متقدمة")
    # جدول أو بيانات مفصلة
    
    tabs = st.tabs(["📈 نظرة عامة", "🔍 تحليل فردي", "🗂️ بيانات خام"])
    # ✅ تبويب نظرة عامة
    with tabs[0]:
        st.write("هنا محتوى نظرة عامة التحليلات المتقدمة  ...")
    
    # ✅ تبويب تحليل فردي
    with tabs[1]:
        st.write("هنا محتوى  تحليل فردي التحليلات المتقدمة  ...")
    
    # ✅ تبويب بيانات خام
    with tabs[2]:
        #!/usr/bin/env python3
        # -*- coding: utf-8 -*-
        """
        نظام تحليل المحولات الكهربائية المتقدم
        =======================================

        ملف موحد يحتوي على جميع مكونات النظام

        المؤلف: AI Assistant
        التاريخ: 2024
        الإصدار: 2.0
        """
        
        