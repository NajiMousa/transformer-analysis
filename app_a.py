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

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import json
import io

class TransformerRecommendationSystem:
    def __init__(self):
        self.imbalance_threshold = 0.2  # حد عدم التوازن
        self.high_imbalance_threshold = 0.4  # حد عدم التوازن العالي
        self.capacity_warning_threshold = 0.85  # تحذير عند 85% من السعة
        self.capacity_critical_threshold = 0.95  # خطر عند 95% من السعة
        
    def calculate_load_kva(self, row):
        """حساب الحمل بالـ KVA"""
        try:
            # تحويل القيم إلى أرقام
            ir = float(str(row['قياس التيار R']).replace(',', '.')) if pd.notna(row['قياس التيار R']) else 0
            is_val = float(str(row['قياس التيار S']).replace(',', '.')) if pd.notna(row['قياس التيار S']) else 0
            it = float(str(row['قياس التيار T']).replace(',', '.')) if pd.notna(row['قياس التيار T']) else 0
            
            # متوسط الجهد (تقريبي)
            voltage = 400  # جهد خط افتراضي
            avg_current = (ir + is_val + it) / 3
            load_kva = (voltage * avg_current * np.sqrt(3)) / 1000
            return load_kva
        except:
            return 0

    def calculate_imbalance(self, row):
        """حساب نسبة عدم التوازن"""
        try:
            ir = float(str(row['قياس التيار R']).replace(',', '.')) if pd.notna(row['قياس التيار R']) else 0
            is_val = float(str(row['قياس التيار S']).replace(',', '.')) if pd.notna(row['قياس التيار S']) else 0
            it = float(str(row['قياس التيار T']).replace(',', '.')) if pd.notna(row['قياس التيار T']) else 0
            
            currents = [ir, is_val, it]
            if sum(currents) == 0:
                return 0
            
            mean_current = np.mean(currents)
            if mean_current == 0:
                return 0
                
            std_current = np.std(currents)
            imbalance = std_current / mean_current
            return imbalance
        except:
            return 0

    def add_season(self, df):
        """إضافة الموسم للبيانات"""
        df['تاريخ القياس'] = pd.to_datetime(df['تاريخ القياس'], errors='coerce')
        df['الموسم'] = df['تاريخ القياس'].dt.month.apply(
            lambda m: "شتوي" if m in [12, 1, 2, 3] else ("صيفي" if m in [6, 7, 8] else "انتقالي")
        )
        return df

    def prepare_data(self, loads_df):
        """تحضير البيانات للتحليل"""
        # تنظيف وتحضير البيانات
        loads_df = self.add_season(loads_df)
        loads_df['Load_kVA'] = loads_df.apply(self.calculate_load_kva, axis=1)
        loads_df['Imbalance'] = loads_df.apply(self.calculate_imbalance, axis=1)
        
        # إضافة السنة
        loads_df['السنة'] = loads_df['تاريخ القياس'].dt.year
        
        return loads_df

    def analyze_feeder_historical(self, feeder_data, feeder_name):
        """تحليل تاريخي لسكينة معينة"""
        analysis = {
            'feeder_name': feeder_name,
            'current_status': 'جيدة',
            'historical_pattern': 'مستقرة',
            'seasonal_behavior': {},
            'trend': 'مستقر',
            'recommendations': []
        }
        
        if feeder_data.empty:
            return analysis
        
        # التحليل الموسمي
        seasonal_imbalance = feeder_data.groupby('الموسم')['Imbalance'].mean()
        
        # أحدث سنة
        latest_year = feeder_data['السنة'].max()
        current_data = feeder_data[feeder_data['السنة'] == latest_year]
        current_imbalance = current_data['Imbalance'].mean()
        
        # البيانات التاريخية (السنوات السابقة)
        historical_data = feeder_data[feeder_data['السنة'] < latest_year]
        historical_imbalance = historical_data['Imbalance'].mean() if not historical_data.empty else 0
        
        # تحديد الحالة الحالية
        if current_imbalance > self.high_imbalance_threshold:
            analysis['current_status'] = 'سيئة'
        elif current_imbalance > self.imbalance_threshold:
            analysis['current_status'] = 'تحتاج مراقبة'
        
        # تحديد النمط التاريخي
        if not historical_data.empty:
            if historical_imbalance > self.imbalance_threshold and current_imbalance > self.imbalance_threshold:
                analysis['historical_pattern'] = 'مزمنة'
            elif historical_imbalance <= self.imbalance_threshold and current_imbalance > self.imbalance_threshold:
                analysis['historical_pattern'] = 'مشكلة جديدة'
            elif historical_imbalance > self.imbalance_threshold and current_imbalance <= self.imbalance_threshold:
                analysis['historical_pattern'] = 'محسنة'
        
        # السلوك الموسمي
        for season, imbalance in seasonal_imbalance.items():
            if imbalance > self.imbalance_threshold:
                analysis['seasonal_behavior'][season] = 'غير متزنة'
            else:
                analysis['seasonal_behavior'][season] = 'متزنة'
        
        return analysis

    def generate_feeder_recommendations(self, feeder_analysis):
        """توليد توصيات لسكينة معينة"""
        recommendations = []
        feeder_name = feeder_analysis['feeder_name']
        
        # توصيات حسب الحالة الحالية والتاريخ
        if feeder_analysis['current_status'] == 'سيئة':
            if feeder_analysis['historical_pattern'] == 'مزمنة':
                recommendations.append({
                    "title": f"مشكلة مزمنة - السكينة {feeder_name}",
                    "message": f"السكينة {feeder_name} تعاني من عدم توازن مزمن يتطلب تدخل فوري",
                    "severity": "error",
                    "action": f"إرسال فريق فني متخصص للكشف الشامل على السكينة {feeder_name} وإعادة توزيع الأحمال",
                    "due_in_days": 3,
                    "status": "جديد",
                    "priority": "عالية جداً"
                })
            else:
                recommendations.append({
                    "title": f"مشكلة جديدة - السكينة {feeder_name}",
                    "message": f"السكينة {feeder_name} تظهر عدم توازن جديد يتطلب تحقيق سريع",
                    "severity": "error",
                    "action": f"فحص الاشتراكات الجديدة أو التغييرات الحديثة على السكينة {feeder_name}",
                    "due_in_days": 5,
                    "status": "جديد",
                    "priority": "عالية"
                })
                
        elif feeder_analysis['current_status'] == 'تحتاج مراقبة':
            if feeder_analysis['historical_pattern'] == 'مزمنة':
                recommendations.append({
                    "title": f"مراقبة دورية - السكينة {feeder_name}",
                    "message": f"السكينة {feeder_name} تحسنت لكن تحتاج مراقبة مستمرة بسبب التاريخ المزمن",
                    "severity": "warning",
                    "action": f"مراقبة أسبوعية للأحمال على السكينة {feeder_name} مع تقرير شهري",
                    "due_in_days": 7,
                    "status": "مراقبة",
                    "priority": "متوسطة"
                })
            else:
                recommendations.append({
                    "title": f"مراقبة احترازية - السكينة {feeder_name}",
                    "message": f"السكينة {feeder_name} تظهر بداية عدم توازن يتطلب المراقبة",
                    "severity": "warning",
                    "action": f"مراقبة نصف شهرية ومراجعة الأحمال الجديدة على السكينة {feeder_name}",
                    "due_in_days": 14,
                    "status": "مراقبة",
                    "priority": "متوسطة"
                })
                
        else:  # حالة جيدة
            if feeder_analysis['historical_pattern'] == 'محسنة':
                recommendations.append({
                    "title": f"تحسن ملحوظ - السكينة {feeder_name}",
                    "message": f"السكينة {feeder_name} أظهرت تحسناً كبيراً مقارنة بالسنوات السابقة",
                    "severity": "success",
                    "action": f"مواصلة المراقبة الشهرية للحفاظ على الأداء الجيد للسكينة {feeder_name}",
                    "due_in_days": 30,
                    "status": "مراقبة",
                    "priority": "منخفضة"
                })
            elif feeder_analysis['historical_pattern'] == 'مزمنة':
                recommendations.append({
                    "title": f"مراقبة وقائية - السكينة {feeder_name}",
                    "message": f"السكينة {feeder_name} جيدة حالياً لكن لها تاريخ في عدم التوازن",
                    "severity": "info",
                    "action": f"مراقبة دورية كل شهرين للسكينة {feeder_name} للتأكد من استمرار الوضع الجيد",
                    "due_in_days": 60,
                    "status": "مراقبة",
                    "priority": "منخفضة"
                })
            else:
                recommendations.append({
                    "title": f"أداء مستقر - السكينة {feeder_name}",
                    "message": f"السكينة {feeder_name} تعمل بكفاءة عالية ولا تحتاج تدخل",
                    "severity": "success",
                    "action": f"مراقبة ربع سنوية روتينية للسكينة {feeder_name}",
                    "due_in_days": 90,
                    "status": "مراقبة",
                    "priority": "منخفضة"
                })
        
        # توصيات موسمية
        seasonal_issues = []
        for season, status in feeder_analysis['seasonal_behavior'].items():
            if status == 'غير متزنة':
                seasonal_issues.append(season)
        
        if seasonal_issues:
            recommendations.append({
                "title": f"مشكلة موسمية - السكينة {feeder_name}",
                "message": f"السكينة {feeder_name} تظهر عدم توازن في فصول: {', '.join(seasonal_issues)}",
                "severity": "warning",
                "action": f"تحضير خطة موسمية لإعادة توزيع الأحمال على السكينة {feeder_name} قبل بداية الفصول المتأثرة",
                "due_in_days": 30,
                "status": "مراقبة",
                "priority": "متوسطة"
            })
        
        return recommendations

    def analyze_transformer_capacity(self, loads_df, transformer_capacity):
        """تحليل سعة المحول"""
        capacity_analysis = {
            'seasonal_loads': {},
            'capacity_utilization': {},
            'predictions': {},
            'recommendations': []
        }
        
        # التحليل الموسمي للأحمال
        seasonal_loads = loads_df.groupby('الموسم')['Load_kVA'].agg(['mean', 'max']).to_dict()
        
        for season in ['شتوي', 'صيفي']:
            if season in seasonal_loads['mean']:
                avg_load = seasonal_loads['mean'][season]
                max_load = seasonal_loads['max'][season]
                
                capacity_analysis['seasonal_loads'][season] = {
                    'average': avg_load,
                    'maximum': max_load,
                    'utilization': (avg_load / transformer_capacity) * 100,
                    'max_utilization': (max_load / transformer_capacity) * 100
                }
                
                # توصيات السعة
                if max_load > transformer_capacity:
                    capacity_analysis['recommendations'].append({
                        "title": f"تجاوز السعة - فصل {season}",
                        "message": f"المحول يتجاوز السعة المقررة ({transformer_capacity} KVA) في فصل {season} بحمولة قصوى {max_load:.1f} KVA",
                        "severity": "error",
                        "action": "مراجعة عاجلة للاشتراكات وإعادة توزيع الأحمال أو ترقية المحول",
                        "due_in_days": 7,
                        "status": "جديد",
                        "priority": "عالية جداً"
                    })
                elif avg_load > transformer_capacity * self.capacity_critical_threshold:
                    capacity_analysis['recommendations'].append({
                        "title": f"اقتراب من السعة القصوى - فصل {season}",
                        "message": f"المحول يقترب من السعة القصوى في فصل {season} بمتوسط حمولة {avg_load:.1f} KVA",
                        "severity": "warning",
                        "action": "مراقبة يومية وإعداد خطة طوارئ لتوزيع الأحمال",
                        "due_in_days": 14,
                        "status": "مراقبة",
                        "priority": "عالية"
                    })
                elif avg_load < transformer_capacity * 0.3:
                    capacity_analysis['recommendations'].append({
                        "title": f"استغلال منخفض للسعة - فصل {season}",
                        "message": f"المحول مستغل بأقل من 30% من السعة في فصل {season}",
                        "severity": "info",
                        "action": "دراسة إمكانية توصيل أحمال إضافية أو تقليل حجم المحول",
                        "due_in_days": 60,
                        "status": "دراسة",
                        "priority": "منخفضة"
                    })
        
        return capacity_analysis

    def generate_predictive_analysis(self, loads_df, transformer_capacity):
        """تحليل تنبؤي للأحمال"""
        predictions = {}
        
        # إعداد البيانات للتنبؤ
        df_predict = loads_df.dropna(subset=['Load_kVA']).copy()
        if df_predict.empty:
            return predictions
            
        df_predict = df_predict.sort_values('تاريخ القياس')
        df_predict['Days'] = (df_predict['تاريخ القياس'] - df_predict['تاريخ القياس'].min()).dt.days
        
        X = df_predict[['Days']].values
        y = df_predict['Load_kVA'].values
        
        try:
            model = LinearRegression()
            model.fit(X, y)
            
            # التنبؤ للفترات القادمة
            last_day = df_predict['Days'].max()
            future_periods = [30, 90, 180, 365]  # شهر، 3 أشهر، 6 أشهر، سنة
            period_names = ["شهر", "3 أشهر", "6 أشهر", "سنة"]
            
            for period, name in zip(future_periods, period_names):
                future_load = model.predict([[last_day + period]])[0]
                predictions[name] = {
                    'predicted_load': future_load,
                    'utilization': (future_load / transformer_capacity) * 100,
                    'risk_level': self.assess_risk_level(future_load, transformer_capacity)
                }
        except:
            pass
            
        return predictions

    def assess_risk_level(self, predicted_load, capacity):
        """تقييم مستوى المخاطر"""
        utilization = predicted_load / capacity
        
        if utilization > 1.0:
            return "عالي جداً"
        elif utilization > 0.95:
            return "عالي"
        elif utilization > 0.85:
            return "متوسط"
        elif utilization > 0.7:
            return "منخفض"
        else:
            return "آمن"

    def generate_comprehensive_recommendations(self, loads_df, transformer_info, selected_transformer_id=None):
        """توليد توصيات شاملة للمحول"""
        
        all_recommendations = []
        executive_summary = []
        
        # تحضير البيانات
        loads_df = self.prepare_data(loads_df)
        
        # فلترة المحول المحدد
        if selected_transformer_id:
            loads_df = loads_df[loads_df['Transformer_id'] == selected_transformer_id]
        
        if loads_df.empty:
            return all_recommendations, executive_summary
        
        # جلب معلومات المحول
        transformer_capacity = None
        if isinstance(transformer_info, pd.DataFrame):
            transformer_capacity = transformer_info['قدرة المحول KVA'].iloc[0] if 'قدرة المحول KVA' in transformer_info.columns else 500
        elif isinstance(transformer_info, dict):
            transformer_capacity = transformer_info.get('قدرة المحول KVA', 500)
        else:
            transformer_capacity = 500  # قيمة افتراضية
        
        # 1. تحليل السكاكين
        st.subheader("🔌 تحليل السكاكين")
        feeders = loads_df['اتجاه السكينة'].unique()
        feeder_analyses = {}
        
        for feeder in feeders:
            feeder_data = loads_df[loads_df['اتجاه السكينة'] == feeder].copy()
            feeder_analysis = self.analyze_feeder_historical(feeder_data, feeder)
            feeder_analyses[feeder] = feeder_analysis
            
            # توليد توصيات السكينة
            feeder_recs = self.generate_feeder_recommendations(feeder_analysis)
            all_recommendations.extend(feeder_recs)
            
            # إضافة للملخص التنفيذي
            if feeder_analysis['current_status'] != 'جيدة':
                executive_summary.append(f"السكينة {feeder}: {feeder_analysis['current_status']} ({feeder_analysis['historical_pattern']})")
        
        # 2. تحليل سعة المحول
        st.subheader("⚡ تحليل السعة")
        capacity_analysis = self.analyze_transformer_capacity(loads_df, transformer_capacity)
        all_recommendations.extend(capacity_analysis['recommendations'])
        
        # إضافة تحليل السعة للملخص
        for season, data in capacity_analysis['seasonal_loads'].items():
            if data['max_utilization'] > 100:
                executive_summary.append(f"تجاوز السعة في فصل {season} ({data['max_utilization']:.1f}%)")
            elif data['utilization'] < 30:
                executive_summary.append(f"استغلال منخفض في فصل {season} ({data['utilization']:.1f}%)")
        
        # 3. التحليل التنبؤي
        st.subheader("🔮 التحليل التنبؤي")
        predictions = self.generate_predictive_analysis(loads_df, transformer_capacity)
        
        for period, pred_data in predictions.items():
            risk = pred_data['risk_level']
            if risk in ["عالي جداً", "عالي"]:
                all_recommendations.append({
                    "title": f"تحذير تنبؤي - خلال {period}",
                    "message": f"التنبؤ يشير لتجاوز محتمل للسعة خلال {period} (مخاطر {risk})",
                    "severity": "error" if risk == "عالي جداً" else "warning",
                    "action": "إعداد خطة وقائية لزيادة السعة أو توزيع الأحمال",
                    "due_in_days": 30 if period == "شهر" else 60,
                    "status": "تخطيط",
                    "priority": "عالية"
                })
                executive_summary.append(f"توقع مشاكل خلال {period}")
        
        # 4. توصيات متقدمة
        advanced_recs = self.generate_advanced_recommendations(loads_df, feeder_analyses, capacity_analysis)
        all_recommendations.extend(advanced_recs)
        
        # 5. الملخص التنفيذي النهائي
        if executive_summary:
            final_summary = self.create_executive_summary(executive_summary, all_recommendations)
            all_recommendations.append(final_summary)
        
        return all_recommendations, feeder_analyses, capacity_analysis, predictions

    def generate_advanced_recommendations(self, loads_df, feeder_analyses, capacity_analysis):
        """توصيات متقدمة بناءً على التحليل الشامل"""
        advanced_recs = []
        
        # تحليل العلاقة بين مشاكل السكاكين ومشاكل السعة
        problematic_feeders = [f for f, analysis in feeder_analyses.items() 
                             if analysis['current_status'] in ['سيئة', 'تحتاج مراقبة']]
        
        seasonal_capacity_issues = [season for season, data in capacity_analysis['seasonal_loads'].items()
                                  if data.get('max_utilization', 0) > 95]
        
        if problematic_feeders and seasonal_capacity_issues:
            advanced_recs.append({
                "title": "ترابط مشاكل السكاكين والسعة",
                "message": f"مشاكل السعة في {', '.join(seasonal_capacity_issues)} قد ترتبط بعدم توازن السكاكين: {', '.join(problematic_feeders)}",
                "severity": "warning",
                "action": "دراسة شاملة للعلاقة بين مشاكل السكاكين وتجاوز السعة وإعداد خطة متكاملة",
                "due_in_days": 21,
                "status": "دراسة",
                "priority": "عالية"
            })
        
        # توصية الصيانة الدورية
        maintenance_priority = "عالية" if problematic_feeders else "متوسطة"
        maintenance_days = 30 if problematic_feeders else 90
        
        advanced_recs.append({
            "title": "برنامج الصيانة الدورية",
            "message": f"وضع برنامج صيانة دورية للمحول بأولوية {maintenance_priority}",
            "severity": "info",
            "action": "إعداد جدول صيانة شامل يشمل فحص السكاكين والحمولات والمعدات الواقية",
            "due_in_days": maintenance_days,
            "status": "تخطيط",
            "priority": maintenance_priority
        })
        
        return advanced_recs

    def create_executive_summary(self, summary_points, all_recommendations):
        """إنشاء الملخص التنفيذي النهائي"""
        
        # حساب الإحصائيات
        error_count = len([r for r in all_recommendations if r.get('severity') == 'error'])
        warning_count = len([r for r in all_recommendations if r.get('severity') == 'warning'])
        
        # تحديد الأولوية الإجمالية
        if error_count > 0:
            overall_priority = "عالية جداً"
            overall_status = "يتطلب تدخل فوري"
        elif warning_count > 2:
            overall_priority = "عالية"
            overall_status = "يتطلب مراقبة مكثفة"
        elif warning_count > 0:
            overall_priority = "متوسطة"
            overall_status = "يتطلب مراقبة"
        else:
            overall_priority = "منخفضة"
            overall_status = "وضع مستقر"
        
        # حساب المهلة الإجمالية (أقل مهلة من التوصيات الحرجة)
        critical_recs = [r for r in all_recommendations if r.get('severity') in ['error', 'warning']]
        overall_due_days = min([r['due_in_days'] for r in critical_recs], default=90)
        
        summary_message = f"الوضع العام للمحول: {overall_status}. "
        summary_message += f"المشاكل الرئيسية: {' — '.join(summary_points[:3])}. " if summary_points else "لا توجد مشاكل حرجة. "
        summary_message += f"إجمالي التوصيات: {len(all_recommendations)} ({error_count} حرجة، {warning_count} تحذيرية)"
        
        return {
            "title": "📋 الملخص التنفيذي",
            "message": summary_message,
            "severity": "error" if error_count > 0 else ("warning" if warning_count > 0 else "info"),
            "action": f"تنفيذ جميع التوصيات حسب الأولوية المحددة، بدءاً من التوصيات الحرجة",
            "due_in_days": overall_due_days,
            "status": "جديد",
            "priority": overall_priority,
            "statistics": {
                "total_recommendations": len(all_recommendations),
                "critical": error_count,
                "warnings": warning_count,
                "overall_status": overall_status
            }
        }

    def create_data_visualization(self, loads_df, feeder_name=None):
        """إنشاء الرسوم البيانية للبيانات"""
        
        if feeder_name:
            data = loads_df[loads_df['اتجاه السكينة'] == feeder_name].copy()
            title_suffix = f"- السكينة {feeder_name}"
        else:
            data = loads_df.copy()
            title_suffix = "- جميع السكاكين"
        
        visualizations = {}
        
        # 1. رسم بياني لعدم التوازن عبر الزمن
        if not data.empty:
            fig_imbalance = px.line(
                data, 
                x='تاريخ القياس', 
                y='Imbalance', 
                color='اتجاه السكينة',
                title=f'تطور عدم التوازن عبر الزمن {title_suffix}',
                labels={'Imbalance': 'نسبة عدم التوازن', 'تاريخ القياس': 'التاريخ'}
            )
            fig_imbalance.add_hline(y=0.2, line_dash="dash", line_color="orange", 
                                  annotation_text="حد التحذير (0.2)")
            fig_imbalance.add_hline(y=0.4, line_dash="dash", line_color="red", 
                                  annotation_text="حد الخطر (0.4)")
            visualizations['imbalance_trend'] = fig_imbalance
            
            # 2. رسم بياني للأحمال
            fig_load = px.line(
                data,
                x='تاريخ القياس',
                y='Load_kVA',
                color='اتجاه السكينة',
                title=f'تطور الأحمال عبر الزمن {title_suffix}',
                labels={'Load_kVA': 'الحمل (KVA)', 'تاريخ القياس': 'التاريخ'}
            )
            visualizations['load_trend'] = fig_load
            
            # 3. رسم بياني للتيارات الثلاثة
            if feeder_name:
                # تحضير البيانات للرسم
                currents_data = []
                for _, row in data.iterrows():
                    try:
                        ir = float(str(row['قياس التيار R']).replace(',', '.'))
                        is_val = float(str(row['قياس التيار S']).replace(',', '.'))
                        it = float(str(row['قياس التيار T']).replace(',', '.'))
                        
                        currents_data.extend([
                            {'تاريخ القياس': row['تاريخ القياس'], 'الفاز': 'R', 'التيار': ir},
                            {'تاريخ القياس': row['تاريخ القياس'], 'الفاز': 'S', 'التيار': is_val},
                            {'تاريخ القياس': row['تاريخ القياس'], 'الفاز': 'T', 'التيار': it}
                        ])
                    except:
                        continue
                
                if currents_data:
                    currents_df = pd.DataFrame(currents_data)
                    fig_phases = px.line(
                        currents_df,
                        x='تاريخ القياس',
                        y='التيار',
                        color='الفاز',
                        title=f'تطور التيارات الثلاثة - السكينة {feeder_name}',
                        labels={'التيار': 'التيار (أمبير)', 'تاريخ القياس': 'التاريخ'}
                    )
                    visualizations['phases_trend'] = fig_phases
        
        return visualizations

    def save_recommendations_state(self, recommendations, file_path="recommendations_state.json"):
        """حفظ حالة التوصيات"""
        try:
            # إعداد البيانات للحفظ
            state_data = {
                "timestamp": datetime.now().isoformat(),
                "recommendations": []
            }
            
            for rec in recommendations:
                state_data["recommendations"].append({
                    "id": hash(rec['title'] + rec['message']),  # معرف فريد
                    "title": rec['title'],
                    "message": rec['message'],
                    "severity": rec['severity'],
                    "action": rec['action'],
                    "due_in_days": rec['due_in_days'],
                    "status": rec.get('status', 'جديد'),
                    "priority": rec.get('priority', 'متوسطة'),
                    "created_date": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat()
                })
            
            # حفظ في ملف JSON
            print("📂 جاري قراءة:", file_path)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            st.error(f"خطأ في حفظ حالة التوصيات: {str(e)}")
            return False

    def load_recommendations_state(self, file_path="recommendations_state.json"):
        """تحميل حالة التوصيات المحفوظة"""
        try:
            if Path(file_path).exists():
                print("📂 جاري قراءة:", file_path)

                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        except Exception as e:
            st.error(f"خطأ في تحميل حالة التوصيات: {str(e)}")
            return None

    def update_recommendation_status(self, rec_id, new_status, file_path="recommendations_state.json"):
        """تحديث حالة توصية معينة"""
        try:
            state_data = self.load_recommendations_state(file_path)
            if state_data:
                for rec in state_data["recommendations"]:
                    if rec["id"] == rec_id:
                        rec["status"] = new_status
                        rec["last_updated"] = datetime.now().isoformat()
                        break
                
                # حفظ التحديث
                print("📂 جاري قراءة:", file_path)

                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(state_data, f, ensure_ascii=False, indent=2)
                return True
            return False
        except Exception as e:
            st.error(f"خطأ في تحديث حالة التوصية: {str(e)}")
            return False


def display_enhanced_recommendations(recommendations, feeder_analyses=None, capacity_analysis=None, predictions=None):
    """عرض التوصيات المطورة مع إمكانيات تفاعلية"""
    
    severity_styles = {
        "error":   {"icon": "🚨", "bg": "#ffebee", "border": "#f44336", "color": "#c62828"},
        "warning": {"icon": "⚠️", "bg": "#fff8e1", "border": "#ff9800", "color": "#f57c00"},
        "success": {"icon": "✅", "bg": "#e8f5e9", "border": "#4caf50", "color": "#2e7d32"},
        "info":    {"icon": "ℹ️", "bg": "#e3f2fd", "border": "#2196f3", "color": "#1565c0"},
    }
    
    # إحصائيات سريعة
    error_count = len([r for r in recommendations if r.get('severity') == 'error'])
    warning_count = len([r for r in recommendations if r.get('severity') == 'warning'])
    success_count = len([r for r in recommendations if r.get('severity') == 'success'])
    info_count = len([r for r in recommendations if r.get('severity') == 'info'])
    
    # عرض الإحصائيات
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🚨 حرجة", error_count)
    with col2:
        st.metric("⚠️ تحذيرية", warning_count)
    with col3:
        st.metric("✅ جيدة", success_count)
    with col4:
        st.metric("ℹ️ معلوماتية", info_count)
    
    # فلترة التوصيات
    st.subheader("🔍 فلترة التوصيات")
    severity_filter = st.multiselect(
        "اختر نوع التوصيات:",
        ["error", "warning", "success", "info"],
        default=["error", "warning"],
        format_func=lambda x: {"error": "🚨 حرجة", "warning": "⚠️ تحذيرية", 
                              "success": "✅ جيدة", "info": "ℹ️ معلوماتية"}[x]
    )
    
    # عرض التوصيات المفلترة
    filtered_recommendations = [r for r in recommendations if r.get('severity') in severity_filter]
    
    for i, rec in enumerate(filtered_recommendations):
        style = severity_styles.get(rec.get('severity', 'info'), severity_styles["info"])
        
        with st.container():
            # العنوان والرسالة
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, {style['bg']} 0%, white 100%);
                    border-left: 6px solid {style['border']};
                    padding: 20px;
                    margin: 15px 0;
                    border-radius: 12px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    color: {style['color']};
                    font-family: 'Segoe UI', Arial, sans-serif;">
                    
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <span style="font-size: 24px; margin-right: 10px;">{style['icon']}</span>
                        <h3 style="margin: 0; color: {style['color']}; font-size: 20px; font-weight: bold;">
                            {rec['title']}
                        </h3>
                    </div>
                    
                    <p style="margin: 10px 0; font-size: 16px; line-height: 1.6; color: #333;">
                        {rec['message']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # تفاصيل إضافية قابلة للطي
            with st.expander("📋 تفاصيل التوصية", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**🛠 الإجراء المطلوب:**")
                    st.write(rec['action'])
                    
                    st.markdown(f"**⏰ المهلة الزمنية:**")
                    st.write(f"{rec['due_in_days']} يوم")
                
                with col2:
                    st.markdown(f"**📊 الأولوية:**")
                    priority = rec.get('priority', 'متوسطة')
                    priority_color = {"عالية جداً": "🔴", "عالية": "🟠", "متوسطة": "🟡", "منخفضة": "🟢"}
                    st.write(f"{priority_color.get(priority, '🟡')} {priority}")
                    
                    # إدارة الحالة
                    st.markdown(f"**📌 الحالة:**")
                    current_status = rec.get('status', 'جديد')
                    new_status = st.selectbox(
                        "تحديث الحالة:",
                        ["جديد", "قيد التنفيذ", "مراقبة", "تم الحل", "مؤجل"],
                        index=["جديد", "قيد التنفيذ", "مراقبة", "تم الحل", "مؤجل"].index(current_status),
                        key=f"status_{i}"
                    )
                    
                    if new_status != current_status:
                        st.success(f"تم تحديث الحالة إلى: {new_status}")
                
                # عرض البيانات المساندة إذا توفرت
                if 'examples' in rec and rec['examples']:
                    st.markdown("**📊 أمثلة من البيانات:**")
                    st.dataframe(pd.DataFrame(rec['examples']))
            
            # أزرار الإجراءات
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button(f"📊 عرض البيانات", key=f"show_data_{i}"):
                    st.session_state[f"show_viz_{i}"] = True
            
            with col2:
                if st.button(f"📝 إضافة ملاحظة", key=f"add_note_{i}"):
                    st.session_state[f"show_note_{i}"] = True
            
            # عرض الرسوم البيانية
            if st.session_state.get(f"show_viz_{i}", False):
                st.markdown("### 📊 الرسوم البيانية")
                # هنا يمكن إضافة المخططات المناسبة
                st.info("سيتم عرض المخططات البيانية هنا")
                
                if st.button(f"إخفاء البيانات", key=f"hide_viz_{i}"):
                    st.session_state[f"show_viz_{i}"] = False
            
            # إضافة الملاحظات
            if st.session_state.get(f"show_note_{i}", False):
                note = st.text_area(f"أضف ملاحظة للتوصية:", key=f"note_text_{i}")
                if st.button(f"حفظ الملاحظة", key=f"save_note_{i}"):
                    st.success("تم حفظ الملاحظة بنجاح")
                    st.session_state[f"show_note_{i}"] = False
    
    # ملخص الإجراءات المطلوبة
    if filtered_recommendations:
        st.markdown("---")
        st.subheader("📅 جدول الإجراءات")
        
        # إنشاء جدول بالإجراءات المطلوبة
        actions_data = []
        for rec in filtered_recommendations:
            actions_data.append({
                "التوصية": rec['title'][:50] + "..." if len(rec['title']) > 50 else rec['title'],
                "الأولوية": rec.get('priority', 'متوسطة'),
                "المهلة (أيام)": rec['due_in_days'],
                "الحالة": rec.get('status', 'جديد'),
                "النوع": rec.get('severity', 'info')
            })
        
        if actions_data:
            df_actions = pd.DataFrame(actions_data)
            st.dataframe(
                df_actions,
                use_container_width=True,
                column_config={
                    "النوع": st.column_config.SelectboxColumn(
                        "النوع",
                        options=["error", "warning", "success", "info"],
                        format_func=lambda x: {"error": "🚨 حرجة", "warning": "⚠️ تحذيرية", 
                                              "success": "✅ جيدة", "info": "ℹ️ معلوماتية"}.get(x, x)
                    )
                    }
            )
    
    # أزرار الإجراءات العامة
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 حفظ التوصيات", type="primary"):
            # حفظ التوصيات
            system = TransformerRecommendationSystem()
            if system.save_recommendations_state(recommendations):
                st.success("تم حفظ التوصيات بنجاح!")
            else:
                st.error("فشل في حفظ التوصيات")
    
    with col2:
        if st.button("📊 تقرير شامل"):
            st.info("سيتم إنشاء التقرير الشامل...")
    
    with col3:
        if st.button("📧 إرسال للفريق"):
            st.info("سيتم إرسال التوصيات للفريق المختص...")


def generate_enhanced_recommendations(loads_df, transformer_info, selected_transformer_id=None):
    """الدالة الرئيسية لتوليد التوصيات المطورة"""
    
    # إنشاء نظام التوصيات
    system = TransformerRecommendationSystem()
    
    # توليد التوصيات الشاملة
    recommendations, feeder_analyses, capacity_analysis, predictions = system.generate_comprehensive_recommendations(
        loads_df, transformer_info, selected_transformer_id
    )
    
    # عرض التحليلات التفصيلية
    if feeder_analyses:
        st.subheader("🔌 تحليل السكاكين التفصيلي")
        for feeder_name, analysis in feeder_analyses.items():
            with st.expander(f"السكينة {feeder_name} - {analysis['current_status']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**الحالة الحالية:** {analysis['current_status']}")
                    st.write(f"**النمط التاريخي:** {analysis['historical_pattern']}")
                
                with col2:
                    st.write("**السلوك الموسمي:**")
                    for season, status in analysis['seasonal_behavior'].items():
                        status_icon = "✅" if status == "متزنة" else "⚠️"
                        st.write(f"{status_icon} {season}: {status}")
    
    if capacity_analysis['seasonal_loads']:
        st.subheader("⚡ تحليل السعة التفصيلي")
        for season, data in capacity_analysis['seasonal_loads'].items():
            with st.expander(f"فصل {season}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("متوسط الحمولة", f"{data['average']:.1f} KVA")
                    st.metric("أقصى حمولة", f"{data['maximum']:.1f} KVA")
                with col2:
                    st.metric("نسبة الاستغلال المتوسطة", f"{data['utilization']:.1f}%")
                    st.metric("أقصى نسبة استغلال", f"{data['max_utilization']:.1f}%")
    
    if predictions:
        st.subheader("🔮 التنبؤات")
        pred_data = []
        for period, pred in predictions.items():
            pred_data.append({
                "الفترة": period,
                "الحمولة المتوقعة": f"{pred['predicted_load']:.1f} KVA",
                "نسبة الاستغلال": f"{pred['utilization']:.1f}%",
                "مستوى المخاطر": pred['risk_level']
            })
        
        if pred_data:
            st.dataframe(pd.DataFrame(pred_data), use_container_width=True)
    
    # عرض التوصيات
    st.subheader("📋 التوصيات الشاملة")
    display_enhanced_recommendations(recommendations, feeder_analyses, capacity_analysis, predictions)
    
    return recommendations, feeder_analyses, capacity_analysis, predictions


def create_sample_data():
    """إنشاء بيانات تجريبية للعرض"""
    np.random.seed(42)
    
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='M')
    feeders = ['شمالي', 'جنوبي', 'شرقي', 'غربي']
    
    data = []
    for date in dates:
        for feeder in feeders:
            # محاكاة عدم التوازن موسمي
            season = "شتوي" if date.month in [12, 1, 2, 3] else "صيفي"
            
            base_current = 30 if season == "شتوي" else 20
            
            # إضافة عدم التوازن لبعض السكاكين
            if feeder == "شمالي" and season == "شتوي":
                ir = base_current + np.random.normal(10, 5)
                is_val = base_current + np.random.normal(-5, 3)
                it = base_current + np.random.normal(0, 2)
            elif feeder == "غربي":
                ir = base_current + np.random.normal(8, 4)
                is_val = base_current + np.random.normal(8, 4)
                it = base_current + np.random.normal(-10, 3)
            else:
                ir = base_current + np.random.normal(0, 3)
                is_val = base_current + np.random.normal(0, 3)
                it = base_current + np.random.normal(0, 3)
            
            # التأكد من القيم الموجبة
            ir = max(5, ir)
            is_val = max(5, is_val)
            it = max(5, it)
            
            data.append({
                'رقم التكليف': len(data) + 1,
                'Transformer_id': 1,
                'تاريخ القياس': date,
                'نظام القياس': season,
                'اسم_المحول': 'محول تجريبي 1',
                'اتجاه السكينة': feeder,
                'قدرة_السكينة': 400,
                'قياس التيار R': int(ir),
                'قياس التيار S': int(is_val),
                'قياس التيار T': int(it),
                'قياس التيار N': int(abs(ir - is_val - it)),
                'الجهد بين الفازات RS': 400,
                'الجهد بين الفازات RT': 400,
                'الجهد بين الفازات ST': 400,
                'الجهد بين الفازات RN': 230,
                'الجهد بين الفازات SN': 230,
                'الجهد بين الفازات TN': 230
            })
    
    return pd.DataFrame(data)


def load_and_process_data(uploaded_files):
    """تحميل ومعالجة البيانات من الملفات المرفوعة"""
    all_data = []
    
    for file in uploaded_files:
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file, encoding="cp1256")
            else:
                df = pd.read_excel(file)
            
            # إضافة معلومات المصدر
            df['مصدر_الملف'] = file.name
            all_data.append(df)
            
        except Exception as e:
            st.error(f"خطأ في قراءة الملف {file.name}: {str(e)}")
            continue
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    else:
        return None


def main():
    """الدالة الرئيسية لتشغيل النظام"""
    # إعداد الصفحة
    st.set_page_config(
        page_title="نظام تحليل المحولات الكهربائية",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # العنوان الرئيسي
    st.title("⚡ نظام تحليل المحولات الكهربائية المتقدم")
    st.markdown("---")

    # الشريط الجانبي
    st.sidebar.header("⚙️ إعدادات التحليل")

    # تحميل البيانات
    uploaded_files = st.sidebar.file_uploader(
        "📁 تحميل ملفات البيانات",
        type=['csv', 'xlsx'],
        accept_multiple_files=True,
        help="اختر ملفات البيانات (CSV أو Excel)"
    )

    # معلومات المحول
    st.sidebar.subheader("📊 معلومات المحول")
    transformer_capacity = st.sidebar.number_input(
        "قدرة المحول (KVA)",
        min_value=100,
        max_value=5000,
        value=500,
        step=50
    )

    transformer_id = st.sidebar.number_input(
        "رقم المحول (اختياري)",
        min_value=1,
        max_value=1000,
        value=1,
        help="اتركه فارغاً لتحليل جميع المحولات"
    )

    # إعدادات التحليل
    st.sidebar.subheader("🔧 إعدادات متقدمة")
    analysis_mode = st.sidebar.selectbox(
        "نوع التحليل",
        ["شامل", "موسمي فقط", "السكاكين فقط", "السعة فقط"],
        help="اختر نوع التحليل المطلوب"
    )

    imbalance_threshold = st.sidebar.slider(
        "حد عدم التوازن",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="الحد الأدنى لاعتبار السكينة غير متزنة"
    )

    # المحتوى الرئيسي
    if uploaded_files:
        # تحميل البيانات الحقيقية
        with st.spinner("⏳ جاري تحميل ومعالجة البيانات..."):
            loads_df = load_and_process_data(uploaded_files)
            
        if loads_df is not None:
            st.success(f"✅ تم تحميل {len(loads_df)} سجل من {len(uploaded_files)} ملف")
            
            # عرض معلومات أساسية عن البيانات
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("إجمالي السجلات", len(loads_df))
            with col2:
                unique_transformers = loads_df['Transformer_id'].nunique() if 'Transformer_id' in loads_df.columns else 1
                st.metric("عدد المحولات", unique_transformers)
            with col3:
                unique_feeders = loads_df['اتجاه السكينة'].nunique() if 'اتجاه السكينة' in loads_df.columns else 0
                st.metric("عدد السكاكين", unique_feeders)
            with col4:
                date_range = "غير محدد"
                if 'تاريخ القياس' in loads_df.columns:
                    loads_df['تاريخ القياس'] = pd.to_datetime(loads_df['تاريخ القياس'], errors='coerce')
                    if not loads_df['تاريخ القياس'].isna().all():
                        min_date = loads_df['تاريخ القياس'].min()
                        max_date = loads_df['تاريخ القياس'].max()
                        date_range = f"{min_date.strftime('%Y-%m')} - {max_date.strftime('%Y-%m')}"
                st.metric("الفترة الزمنية", date_range)
            
            # عرض عينة من البيانات
            with st.expander("👁️ معاينة البيانات", expanded=False):
                st.dataframe(loads_df.head(10), use_container_width=True)
        
        else:
            st.error("❌ فشل في تحميل البيانات. يرجى التحقق من صيغة الملفات.")
            loads_df = None

    else:
        # استخدام البيانات التجريبية
        st.info("📝 لم يتم تحميل ملفات. سيتم استخدام بيانات تجريبية للعرض.")
        
        if st.button("🎲 إنشاء بيانات تجريبية"):
            with st.spinner("⏳ جاري إنشاء البيانات التجريبية..."):
                loads_df = create_sample_data()
            st.success("✅ تم إنشاء البيانات التجريبية بنجاح!")
            
            # عرض معلومات البيانات التجريبية
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("إجمالي السجلات", len(loads_df))
            with col2:
                st.metric("عدد المحولات", 1)
            with col3:
                st.metric("عدد السكاكين", 4)
            with col4:
                st.metric("الفترة الزمنية", "2022-2023")
                
            with st.expander("👁️ معاينة البيانات التجريبية", expanded=False):
                st.dataframe(loads_df.head(10), use_container_width=True)
        else:
            loads_df = None

    # تشغيل التحليل
    if loads_df is not None and not loads_df.empty:
        
        st.markdown("---")
        st.header("🔍 تحليل المحول")
        
        # معلومات المحول
        transformer_info = {'قدرة المحول KVA': transformer_capacity}
        
        # أزرار التحليل
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 بدء التحليل الشامل", type="primary", use_container_width=True):
                st.session_state['run_analysis'] = True
                st.session_state['analysis_type'] = 'شامل'
        
        with col2:
            if st.button("📊 تحليل سريع", use_container_width=True):
                st.session_state['run_analysis'] = True
                st.session_state['analysis_type'] = 'سريع'
        
        with col3:
            if st.button("🔄 إعادة تعيين", use_container_width=True):
                st.session_state['run_analysis'] = False
                st.rerun()
        
        # تشغيل التحليل
        if st.session_state.get('run_analysis', False):
            
            st.markdown("---")
            
            # شريط التقدم
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # الخطوة 1: تحضير البيانات
                status_text.text("🔄 جاري تحضير البيانات...")
                progress_bar.progress(20)
                
                # إنشاء نظام التوصيات
                system = TransformerRecommendationSystem()
                system.imbalance_threshold = imbalance_threshold
                
                # الخطوة 2: تحليل البيانات
                status_text.text("🔍 جاري تحليل البيانات...")
                progress_bar.progress(40)
                
                # تحضير البيانات
                processed_df = system.prepare_data(loads_df.copy())
                
                # الخطوة 3: توليد التوصيات
                status_text.text("📋 جاري توليد التوصيات...")
                progress_bar.progress(60)
                
                recommendations, feeder_analyses, capacity_analysis, predictions = system.generate_comprehensive_recommendations(
                    processed_df, 
                    transformer_info, 
                    transformer_id if transformer_id else None
                )
                
                # الخطوة 4: إنشاء الرسوم البيانية
                status_text.text("📊 جاري إنشاء الرسوم البيانية...")
                progress_bar.progress(80)
                
                visualizations = system.create_data_visualization(processed_df)
                
                # الخطوة 5: عرض النتائج
                status_text.text("✅ تم إكمال التحليل!")
                progress_bar.progress(100)
                
                # إخفاء شريط التقدم
                progress_bar.empty()
                status_text.empty()
                
                # عرض النتائج
                st.success("🎉 تم إكمال التحليل بنجاح!")
                
                # الإحصائيات السريعة
                st.subheader("📊 الإحصائيات السريعة")
                
                error_count = len([r for r in recommendations if r.get('severity') == 'error'])
                warning_count = len([r for r in recommendations if r.get('severity') == 'warning'])
                success_count = len([r for r in recommendations if r.get('severity') == 'success'])
                info_count = len([r for r in recommendations if r.get('severity') == 'info'])
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric(
                        "🚨 مشاكل حرجة",
                        error_count,
                        delta=f"-{error_count}" if error_count > 0 else None,
                        delta_color="inverse"
                    )
                
                with col2:
                    st.metric(
                        "⚠️ تحذيرات", 
                        warning_count,
                        delta=f"-{warning_count}" if warning_count > 0 else None,
                        delta_color="inverse"
                    )
                
                with col3:
                    st.metric("✅ حالات جيدة", success_count)
                
                with col4:
                    st.metric("ℹ️ معلومات", info_count)
                
                with col5:
                    st.metric("📋 إجمالي التوصيات", len(recommendations))
                
                # التبويبات للعرض
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📋 التوصيات",
                    "📊 التحليل التفصيلي", 
                    "📈 الرسوم البيانية",
                    "📄 التقرير الكامل"
                ])
                
                with tab1:
                    st.header("📋 التوصيات الشاملة")
                    display_enhanced_recommendations(recommendations, feeder_analyses, capacity_analysis, predictions)
                
                with tab2:
                    st.header("📊 التحليل التفصيلي")
                    
                    # تحليل السكاكين
                    if feeder_analyses:
                        st.subheader("🔌 تحليل السكاكين")
                        
                        for feeder_name, analysis in feeder_analyses.items():
                            # تحديد لون الحالة
                            status_color = {
                                'جيدة': '🟢',
                                'تحتاج مراقبة': '🟡', 
                                'سيئة': '🔴'
                            }.get(analysis['current_status'], '⚪')
                            
                            with st.expander(f"{status_color} السكينة {feeder_name} - {analysis['current_status']}", expanded=analysis['current_status'] != 'جيدة'):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**📊 الحالة الحالية**")
                                    st.info(f"الحالة: {analysis['current_status']}")
                                    st.info(f"النمط التاريخي: {analysis['historical_pattern']}")
                                    st.info(f"الاتجاه: {analysis['trend']}")
                                
                                with col2:
                                    st.markdown("**🌡️ السلوك الموسمي**")
                                    for season, status in analysis['seasonal_behavior'].items():
                                        status_icon = "✅" if status == "متزنة" else "⚠️"
                                        st.write(f"{status_icon} **{season}:** {status}")
                    
                    # تحليل السعة
                    if capacity_analysis and capacity_analysis['seasonal_loads']:
                        st.subheader("⚡ تحليل السعة")
                        
                        for season, data in capacity_analysis['seasonal_loads'].items():
                            with st.expander(f"🌡️ فصل {season}", expanded=data['max_utilization'] > 90):
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric(
                                        "متوسط الحمولة", 
                                        f"{data['average']:.1f} KVA",
                                        delta=f"{data['utilization']:.1f}%" if data['utilization'] < 100 else None,
                                        delta_color="normal" if data['utilization'] < 85 else "inverse"
                                    )
                                
                                with col2:
                                    st.metric(
                                        "أقصى حمولة", 
                                        f"{data['maximum']:.1f} KVA",
                                        delta=f"{data['max_utilization']:.1f}%" if data['max_utilization'] < 100 else f"+{data['max_utilization']-100:.1f}%",
                                        delta_color="normal" if data['max_utilization'] < 100 else "inverse"
                                    )
                                
                                with col3:
                                    # تقييم الوضع
                                    if data['max_utilization'] > 100:
                                        st.error("🚨 تجاوز السعة")
                                    elif data['max_utilization'] > 85:
                                        st.warning("⚠️ اقتراب من السعة")
                                    elif data['utilization'] < 30:
                                        st.info("💡 استغلال منخفض")
                                    else:
                                        st.success("✅ ضمن الحدود الطبيعية")
                    
                    # التحليل التنبؤي
                    if predictions:
                        st.subheader("🔮 التحليل التنبؤي")
                        
                        pred_df_data = []
                        for period, pred in predictions.items():
                            risk_color = {
                                "آمن": "🟢",
                                "منخفض": "🟡", 
                                "متوسط": "🟠",
                                "عالي": "🔴",
                                "عالي جداً": "🚨"
                            }.get(pred['risk_level'], "⚪")
                            
                            pred_df_data.append({
                                "الفترة": period,
                                "الحمولة المتوقعة (KVA)": f"{pred['predicted_load']:.1f}",
                                "نسبة الاستغلال (%)": f"{pred['utilization']:.1f}",
                                "مستوى المخاطر": f"{risk_color} {pred['risk_level']}"
                            })
                        
                        if pred_df_data:
                            pred_df = pd.DataFrame(pred_df_data)
                            st.dataframe(pred_df, use_container_width=True)
                            
                            # تحذيرات التنبؤات
                            high_risk_predictions = [p for p in predictions.values() if p['risk_level'] in ['عالي', 'عالي جداً']]
                            if high_risk_predictions:
                                st.error(f"⚠️ تم اكتشاف {len(high_risk_predictions)} توقع عالي المخاطر!")
                
                with tab3:
                    st.header("📈 الرسوم البيانية")
                    
                    # اختيار السكينة للعرض
                    selected_feeder = st.selectbox(
                        "اختر السكينة للعرض:",
                        ['الكل'] + list(processed_df['اتجاه السكينة'].unique()) if 'اتجاه السكينة' in processed_df.columns else ['الكل'],
                        key="viz_feeder_select"
                    )
                    
                    # إنشاء الرسوم البيانية
                    if selected_feeder != 'الكل':
                        viz_data = processed_df[processed_df['اتجاه السكينة'] == selected_feeder]
                        viz_title_suffix = f"- السكينة {selected_feeder}"
                    else:
                        viz_data = processed_df
                        viz_title_suffix = "- جميع السكاكين"
                    
                    if not viz_data.empty:
                        
                        # رسم بياني لعدم التوازن
                        st.subheader(f"📊 تطور عدم التوازن عبر الزمن {viz_title_suffix}")
                        
                        fig_imbalance = px.line(
                            viz_data,
                            x='تاريخ القياس',
                            y='Imbalance',
                            color='اتجاه السكينة' if selected_feeder == 'الكل' else None,
                            title=f'تطور عدم التوازن عبر الزمن {viz_title_suffix}',
                            labels={'Imbalance': 'نسبة عدم التوازن', 'تاريخ القياس': 'التاريخ'}
                        )
                        
                        # إضافة خطوط المرجع
                        fig_imbalance.add_hline(
                            y=imbalance_threshold, 
                            line_dash="dash", 
                            line_color="orange",
                            annotation_text=f"حد التحذير ({imbalance_threshold})"
                        )
                        fig_imbalance.add_hline(
                            y=system.high_imbalance_threshold, 
                            line_dash="dash", 
                            line_color="red",
                            annotation_text=f"حد الخطر ({system.high_imbalance_threshold})"
                        )
                        
                        fig_imbalance.update_layout(height=500)
                        st.plotly_chart(fig_imbalance, use_container_width=True)
                        
                        # رسم بياني للأحمال
                        st.subheader(f"⚡ تطور الأحمال عبر الزمن {viz_title_suffix}")
                        
                        fig_load = px.line(
                            viz_data,
                            x='تاريخ القياس',
                            y='Load_kVA',
                            color='اتجاه السكينة' if selected_feeder == 'الكل' else None,
                            title=f'تطور الأحمال عبر الزمن {viz_title_suffix}',
                            labels={'Load_kVA': 'الحمل (KVA)', 'تاريخ القياس': 'التاريخ'}
                        )
                        
                        # إضافة خط السعة
                        fig_load.add_hline(
                            y=transformer_capacity,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"سعة المحول ({transformer_capacity} KVA)"
                        )
                        
                        fig_load.update_layout(height=500)
                        st.plotly_chart(fig_load, use_container_width=True)
                        
                        # رسم بياني للتيارات الثلاثة (للسكينة المحددة فقط)
                        if selected_feeder != 'الكل':
                            st.subheader(f"🔌 تطور التيارات الثلاثة - السكينة {selected_feeder}")
                            
                            # تحضير بيانات التيارات
                            currents_data = []
                            for _, row in viz_data.iterrows():
                                try:
                                    ir = float(str(row['قياس التيار R']).replace(',', '.'))
                                    is_val = float(str(row['قياس التيار S']).replace(',', '.'))
                                    it = float(str(row['قياس التيار T']).replace(',', '.'))
                                    
                                    currents_data.extend([
                                        {'تاريخ القياس': row['تاريخ القياس'], 'الفاز': 'R', 'التيار': ir},
                                        {'تاريخ القياس': row['تاريخ القياس'], 'الفاز': 'S', 'التيار': is_val},
                                        {'تاريخ القياس': row['تاريخ القياس'], 'الفاز': 'T', 'التيار': it}
                                    ])
                                except:
                                    continue
                            
                            if currents_data:
                                currents_df = pd.DataFrame(currents_data)
                                fig_phases = px.line(
                                    currents_df,
                                    x='تاريخ القياس',
                                    y='التيار',
                                    color='الفاز',
                                    title=f'تطور التيارات الثلاثة - السكينة {selected_feeder}',
                                    labels={'التيار': 'التيار (أمبير)', 'تاريخ القياس': 'التاريخ'}
                                )
                                fig_phases.update_layout(height=500)
                                st.plotly_chart(fig_phases, use_container_width=True)
                        
                        # رسوم بيانية إضافية
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # توزيع عدم التوازن
                            st.subheader("📊 توزيع عدم التوازن")
                            fig_hist = px.histogram(
                                viz_data,
                                x='Imbalance',
                                color='اتجاه السكينة' if selected_feeder == 'الكل' else None,
                                title='توزيع قيم عدم التوازن',
                                labels={'Imbalance': 'نسبة عدم التوازن', 'count': 'التكرار'}
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)
                        
                        with col2:
                            # متوسط الأحمال الموسمية
                            st.subheader("🌡️ الأحمال الموسمية")
                            if 'الموسم' in viz_data.columns:
                                seasonal_avg = viz_data.groupby(['الموسم', 'اتجاه السكينة'])['Load_kVA'].mean().reset_index() if selected_feeder == 'الكل' else viz_data.groupby('الموسم')['Load_kVA'].mean().reset_index()
                                
                                fig_seasonal = px.bar(
                                    seasonal_avg,
                                    x='الموسم',
                                    y='Load_kVA',
                                    color='اتجاه السكينة' if selected_feeder == 'الكل' and 'اتجاه السكينة' in seasonal_avg.columns else None,
                                    title='متوسط الأحمال الموسمية',
                                    labels={'Load_kVA': 'متوسط الحمل (KVA)', 'الموسم': 'الموسم'}
                                )
                                st.plotly_chart(fig_seasonal, use_container_width=True)
                
                with tab4:
                    st.header("📄 التقرير الكامل")
                    
                    # معلومات التقرير
                    st.markdown(f"""
                    ### 📋 معلومات التقرير
                    
                    - **تاريخ التقرير:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    - **المحول المحلل:** {transformer_id if transformer_id else 'جميع المحولات'}
                    - **قدرة المحول:** {transformer_capacity} KVA
                    - **نوع التحليل:** {st.session_state.get('analysis_type', 'شامل')}
                    - **عدد السجلات:** {len(processed_df)}
                    - **الفترة الزمنية:** {date_range if 'date_range' in locals() else 'غير محدد'}
                    """)
                    
                    # الملخص التنفيذي
                    executive_summary = next((r for r in recommendations if 'الملخص التنفيذي' in r['title']), None)
                    if executive_summary:
                        st.markdown("### 📊 الملخص التنفيذي")
                        st.markdown(f"**الوضع العام:** {executive_summary.get('message', '')}")
                        
                        if 'statistics' in executive_summary:
                            stats = executive_summary['statistics']
                            st.markdown(f"""
                            **الإحصائيات:**
                            - إجمالي التوصيات: {stats.get('total_recommendations', 0)}
                            - التوصيات الحرجة: {stats.get('critical', 0)}
                            - التحذيرات: {stats.get('warnings', 0)}
                            - الحالة العامة: {stats.get('overall_status', 'غير محدد')}
                            """)
                    
                    # تصدير التقرير
                    st.markdown("---")
                    st.subheader("💾 تصدير التقرير")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("📊 تصدير Excel", use_container_width=True):
                            # إنشاء ملف Excel
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl',encoding="utf-8") as writer:
                                # ورقة التوصيات
                                recs_data = []
                                for rec in recommendations:
                                    recs_data.append({
                                        'العنوان': rec['title'],
                                        'الرسالة': rec['message'],
                                        'النوع': rec['severity'],
                                        'الإجراء': rec['action'],
                                        'المهلة_بالأيام': rec['due_in_days'],
                                        'الحالة': rec.get('status', 'جديد'),
                                        'الأولوية': rec.get('priority', 'متوسطة')
                                    })
                                pd.DataFrame(recs_data).to_excel(writer, sheet_name='التوصيات', index=False)
                                
                                # ورقة البيانات المعالجة
                                processed_df.to_excel(writer, sheet_name='البيانات_المعالجة', index=False)
                                
                                # ورقة تحليل السكاكين
                                if feeder_analyses:
                                    feeder_data = []
                                    for feeder, analysis in feeder_analyses.items():
                                        feeder_data.append({
                                            'السكينة': feeder,
                                            'الحالة_الحالية': analysis['current_status'],
                                            'النمط_التاريخي': analysis['historical_pattern'],
                                            'الاتجاه': analysis['trend']
                                        })
                                    pd.DataFrame(feeder_data).to_excel(writer, sheet_name='تحليل_السكاكين', index=False)
                            
                            st.download_button(
                                label="تحميل تقرير Excel",
                                data=output.getvalue(),
                                file_name=f"transformer_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                    
                    with col2:
                        if st.button("📄 تصدير PDF", use_container_width=True):
                            st.info("📄 سيتم توفير تصدير PDF قريباً")
                    
                    with col3:
                        if st.button("💾 حفظ الحالة", use_container_width=True):
                            if system.save_recommendations_state(recommendations):
                                st.success("تم حفظ حالة التوصيات!")
                            else:
                                st.error("فشل في حفظ حالة التوصيات")
            
            except Exception as e:
                st.error(f"❌ حدث خطأ أثناء التحليل: {str(e)}")
                st.exception(e)

    else:
        st.info("📂 يرجى تحميل ملفات البيانات أو إنشاء بيانات تجريبية لبدء التحليل.")
        
        # معلومات عن النظام
        with st.expander("ℹ️ معلومات عن النظام", expanded=True):
            st.markdown("""
            ## 🔍 نظام تحليل المحولات الكهربائية المتقدم
            
            هذا النظام يوفر تحليلاً شاملاً لأداء المحولات الكهربائية ويقدم توصيات مفصلة للصيانة والتشغيل.
            
            ### ✨ الميزات الرئيسية:
            
            - **📊 التحليل الشامل:** تحليل عدم التوازن، السعة، والأداء الموسمي
            - **🔮 التنبؤ:** توقع المشاكل المستقبلية باستخدام تحليل الاتجاهات
            - **📋 التوصيات الذكية:** توصيات مفصلة مع خطط عمل ومواعيد تنفيذ
            - **📈 الرسوم البيانية التفاعلية:** عرض بصري للبيانات والتحليلات
            - **💾 إدارة الحالة:** تتبع وحفظ حالة التوصيات للمتابعة
            
            ### 📁 صيغ الملفات المدعومة:
            - ملفات CSV
            - ملفات Excel (.xlsx, .xls)
            
            ### 🎯 أنواع التحليل:
            1. **شامل:** تحليل كامل لجميع جوانب المحول
            2. **موسمي:** التركيز على الأداء الموسمي
            3. **السكاكين:** تحليل مفصل لتوازن السكاكين
            4. **السعة:** تحليل استغلال السعة والتحميل الزائد
            """)

    # الشريط الجانبي - معلومات إضافية
    st.sidebar.markdown("---")
    st.sidebar.subheader("📞 الدعم والمساعدة")
    st.sidebar.markdown("""
    - 📧 **البريد الإلكتروني:** support@transformers.com
    - 📱 **الهاتف:** +970-XXX-XXXX
    - 🌐 **الموقع:** www.transformers-analysis.com
    """)

    st.sidebar.markdown("---")
    st.sidebar.caption("© 2024 نظام تحليل المحولات الكهربائية")
main()