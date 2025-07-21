#!/usr/bin/env python3
"""
SysWatch Pro Analytics Engine - 엔터프라이즈급 데이터 분석 및 리포팅
빅데이터 처리, 고급 통계 분석, 자동 보고서 생성, 예측 모델링

Copyright (C) 2025 SysWatch Technologies Ltd.
Analytics Division - Enterprise Technology
"""

import os
import sys
import time
import json
import sqlite3
import threading
import subprocess
import smtplib
import zipfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import logging

# 데이터 분석 라이브러리
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# 통계 및 머신러닝
try:
    from scipy import stats, signal
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.decomposition import PCA
    HAS_ADVANCED_ANALYTICS = True
except ImportError:
    HAS_ADVANCED_ANALYTICS = False

# 고급 시각화
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# 웹 리포팅
try:
    from jinja2 import Template, Environment, FileSystemLoader
    HAS_TEMPLATING = True
except ImportError:
    HAS_TEMPLATING = False

# 엑셀 리포팅
try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
    from openpyxl.chart import LineChart, BarChart, PieChart, Reference
    HAS_EXCEL = True
except ImportError:
    HAS_EXCEL = False

from syswatch_quantum import QUANTUM_THEME, quantum_monitor

@dataclass
class AnalyticsReport:
    """분석 보고서 구조"""
    id: str
    title: str
    report_type: str  # daily, weekly, monthly, custom, incident
    generation_time: float
    period_start: float
    period_end: float
    metrics_summary: Dict[str, Any]
    performance_analysis: Dict[str, Any]
    security_analysis: Dict[str, Any]
    recommendations: List[str]
    charts: List[str] = field(default_factory=list)
    raw_data_size: int = 0
    processing_time: float = 0.0

@dataclass
class PerformanceTrend:
    """성능 트렌드 분석"""
    component: str
    trend_direction: str  # increasing, decreasing, stable, volatile
    trend_strength: float  # 0.0 - 1.0
    correlation_factors: List[str]
    forecast_7d: float
    forecast_30d: float
    confidence_interval: Tuple[float, float]
    anomaly_count: int

@dataclass
class SystemInsight:
    """시스템 인사이트"""
    insight_type: str  # bottleneck, optimization, pattern, anomaly
    severity: str  # low, medium, high, critical
    confidence: float
    title: str
    description: str
    impact_analysis: str
    recommended_actions: List[str]
    evidence: Dict[str, Any]
    estimated_improvement: str

class QuantumAnalyticsEngine:
    """양자 분석 엔진"""
    
    def __init__(self):
        self.analytics_db_path = "analytics.db"
        self.reports_directory = Path("reports")
        self.charts_directory = Path("charts")
        self.templates_directory = Path("templates")
        
        # 디렉토리 생성
        for directory in [self.reports_directory, self.charts_directory, self.templates_directory]:
            directory.mkdir(exist_ok=True)
        
        # 데이터베이스 초기화
        self.init_analytics_database()
        
        # 보고서 스케줄러
        self.scheduler_running = False
        self.scheduled_reports = []
        
        # 성능 분석 캐시
        self.performance_cache = {}
        self.cache_ttl = 300  # 5분
        
        # 분석 설정
        self.analytics_config = {
            'enable_advanced_analytics': HAS_ADVANCED_ANALYTICS,
            'enable_plotly_charts': HAS_PLOTLY,
            'enable_excel_reports': HAS_EXCEL,
            'data_retention_days': 365,
            'report_formats': ['html', 'pdf', 'excel'],
            'auto_email_reports': False,
            'email_recipients': [],
            'smtp_server': '',
            'smtp_port': 587,
            'smtp_username': '',
            'smtp_password': '',
            'chart_style': 'quantum',
            'include_raw_data': True,
            'compress_reports': True
        }
        
        # 로깅 설정
        self.setup_analytics_logging()
        
        # 차트 스타일 설정
        self.setup_chart_styles()
    
    def init_analytics_database(self):
        """분석 데이터베이스 초기화"""
        conn = sqlite3.connect(self.analytics_db_path)
        cursor = conn.cursor()
        
        # 보고서 메타데이터
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reports (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                report_type TEXT NOT NULL,
                generation_time REAL NOT NULL,
                period_start REAL NOT NULL,
                period_end REAL NOT NULL,
                file_path TEXT,
                file_size INTEGER,
                processing_time REAL,
                status TEXT DEFAULT 'completed'
            )
        ''')
        
        # 성능 트렌드 분석
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_trends (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                component TEXT NOT NULL,
                trend_direction TEXT NOT NULL,
                trend_strength REAL NOT NULL,
                forecast_7d REAL,
                forecast_30d REAL,
                confidence_lower REAL,
                confidence_upper REAL,
                anomaly_count INTEGER DEFAULT 0
            )
        ''')
        
        # 시스템 인사이트
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                insight_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                confidence REAL NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                impact_analysis TEXT,
                recommended_actions TEXT,
                evidence TEXT,
                estimated_improvement TEXT,
                is_addressed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # 보고서 스케줄
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS report_schedule (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_type TEXT NOT NULL,
                frequency TEXT NOT NULL,
                last_generated REAL,
                next_generation REAL,
                recipients TEXT,
                is_active BOOLEAN DEFAULT TRUE
            )
        ''')
        
        # 성능 벤치마크
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_benchmarks (
                component TEXT PRIMARY KEY,
                baseline_value REAL NOT NULL,
                target_value REAL,
                current_value REAL,
                last_updated REAL NOT NULL,
                benchmark_type TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def setup_analytics_logging(self):
        """분석 로깅 설정"""
        self.analytics_logger = logging.getLogger('QuantumAnalytics')
        self.analytics_logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler('quantum_analytics.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.analytics_logger.addHandler(handler)
    
    def setup_chart_styles(self):
        """차트 스타일 설정"""
        # Matplotlib 스타일
        plt.style.use('dark_background')
        
        # 양자 테마 색상 팔레트
        self.quantum_colors = [
            QUANTUM_THEME['quantum_cyan'],
            QUANTUM_THEME['quantum_purple'],
            QUANTUM_THEME['quantum_green'],
            QUANTUM_THEME['quantum_yellow'],
            QUANTUM_THEME['quantum_orange'],
            QUANTUM_THEME['quantum_red'],
            QUANTUM_THEME['quantum_blue'],
            QUANTUM_THEME['quantum_pink']
        ]
        
        # Seaborn 설정
        if 'sns' in globals():
            sns.set_palette(self.quantum_colors)
            sns.set_style("dark")
    
    def generate_comprehensive_report(self, 
                                    report_type: str = "daily",
                                    period_days: int = 1,
                                    formats: List[str] = None) -> AnalyticsReport:
        """종합 보고서 생성"""
        if formats is None:
            formats = ['html']
        
        start_time = time.time()
        
        # 기간 설정
        end_time = time.time()
        start_time_period = end_time - (period_days * 24 * 60 * 60)
        
        # 보고서 ID 생성
        report_id = f"{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # 데이터 수집
            raw_data = self.collect_analytics_data(start_time_period, end_time)
            
            # 성능 분석
            performance_analysis = self.analyze_performance_data(raw_data)
            
            # 보안 분석
            security_analysis = self.analyze_security_data(raw_data)
            
            # 메트릭스 요약
            metrics_summary = self.generate_metrics_summary(raw_data)
            
            # 트렌드 분석
            trends = self.analyze_trends(raw_data)
            
            # 인사이트 생성
            insights = self.generate_insights(raw_data, performance_analysis, security_analysis)
            
            # 권장사항 생성
            recommendations = self.generate_recommendations(insights, trends)
            
            # 차트 생성
            charts = self.generate_charts(raw_data, performance_analysis, report_id)
            
            # 보고서 객체 생성
            report = AnalyticsReport(
                id=report_id,
                title=f"SysWatch Pro {report_type.title()} Report",
                report_type=report_type,
                generation_time=time.time(),
                period_start=start_time_period,
                period_end=end_time,
                metrics_summary=metrics_summary,
                performance_analysis=performance_analysis,
                security_analysis=security_analysis,
                recommendations=recommendations,
                charts=charts,
                raw_data_size=len(str(raw_data)),
                processing_time=time.time() - start_time
            )
            
            # 보고서 파일 생성
            report_files = []
            for format_type in formats:
                file_path = self.export_report(report, raw_data, format_type)
                if file_path:
                    report_files.append(file_path)
            
            # 데이터베이스에 저장
            self.save_report_metadata(report, report_files)
            
            self.analytics_logger.info(f"Report {report_id} generated successfully")
            
            return report
        
        except Exception as e:
            self.analytics_logger.error(f"Report generation failed: {e}")
            raise
    
    def collect_analytics_data(self, start_time: float, end_time: float) -> Dict[str, Any]:
        """분석용 데이터 수집"""
        try:
            # 메인 데이터베이스에서 메트릭스 수집
            main_db_path = "syswatch_quantum.db"
            
            data = {
                'metrics': [],
                'alerts': [],
                'predictions': [],
                'security_events': [],
                'period_start': start_time,
                'period_end': end_time
            }
            
            if os.path.exists(main_db_path):
                conn = sqlite3.connect(main_db_path)
                
                # 메트릭스 데이터
                df_metrics = pd.read_sql_query('''
                    SELECT * FROM metrics 
                    WHERE timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                ''', conn, params=(start_time, end_time))
                
                if not df_metrics.empty:
                    data['metrics'] = df_metrics.to_dict('records')
                
                # 알림 데이터
                df_alerts = pd.read_sql_query('''
                    SELECT * FROM alerts 
                    WHERE timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                ''', conn, params=(start_time, end_time))
                
                if not df_alerts.empty:
                    data['alerts'] = df_alerts.to_dict('records')
                
                conn.close()
            
            # 보안 데이터베이스에서 보안 이벤트 수집
            security_db_path = "security.db"
            if os.path.exists(security_db_path):
                conn = sqlite3.connect(security_db_path)
                
                df_security = pd.read_sql_query('''
                    SELECT * FROM threats 
                    WHERE timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                ''', conn, params=(start_time, end_time))
                
                if not df_security.empty:
                    data['security_events'] = df_security.to_dict('records')
                
                conn.close()
            
            # 현재 시스템 상태 추가
            if quantum_monitor:
                current_metrics = quantum_monitor.get_current_metrics()
                data['current_state'] = {
                    'cpu_cores': current_metrics.cpu_cores,
                    'memory_percent': current_metrics.memory_percent,
                    'disk_read': current_metrics.disk_read,
                    'disk_write': current_metrics.disk_write,
                    'network_sent': current_metrics.network_sent,
                    'network_recv': current_metrics.network_recv,
                    'process_count': current_metrics.process_count,
                    'uptime': current_metrics.uptime
                }
            
            return data
        
        except Exception as e:
            self.analytics_logger.error(f"Data collection failed: {e}")
            return {'metrics': [], 'alerts': [], 'predictions': [], 'security_events': []}
    
    def analyze_performance_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """성능 데이터 분석"""
        analysis = {
            'cpu_analysis': {},
            'memory_analysis': {},
            'disk_analysis': {},
            'network_analysis': {},
            'overall_health': {},
            'bottlenecks': [],
            'optimization_opportunities': []
        }
        
        try:
            metrics = raw_data.get('metrics', [])
            if not metrics:
                return analysis
            
            df = pd.DataFrame(metrics)
            
            # CPU 분석
            if 'cpu_avg' in df.columns:
                analysis['cpu_analysis'] = {
                    'average': float(df['cpu_avg'].mean()),
                    'peak': float(df['cpu_avg'].max()),
                    'min': float(df['cpu_avg'].min()),
                    'std_dev': float(df['cpu_avg'].std()),
                    'percentile_95': float(df['cpu_avg'].quantile(0.95)),
                    'time_above_80': len(df[df['cpu_avg'] > 80]) / len(df) * 100,
                    'trend': self.calculate_trend(df['cpu_avg'].values)
                }
                
                # CPU 병목 감지
                if analysis['cpu_analysis']['average'] > 70:
                    analysis['bottlenecks'].append({
                        'component': 'CPU',
                        'severity': 'high' if analysis['cpu_analysis']['average'] > 85 else 'medium',
                        'description': f"High CPU usage average: {analysis['cpu_analysis']['average']:.1f}%",
                        'impact': 'System responsiveness degraded'
                    })
            
            # 메모리 분석
            if 'memory_percent' in df.columns:
                analysis['memory_analysis'] = {
                    'average': float(df['memory_percent'].mean()),
                    'peak': float(df['memory_percent'].max()),
                    'min': float(df['memory_percent'].min()),
                    'std_dev': float(df['memory_percent'].std()),
                    'percentile_95': float(df['memory_percent'].quantile(0.95)),
                    'time_above_80': len(df[df['memory_percent'] > 80]) / len(df) * 100,
                    'trend': self.calculate_trend(df['memory_percent'].values)
                }
                
                # 메모리 병목 감지
                if analysis['memory_analysis']['average'] > 80:
                    analysis['bottlenecks'].append({
                        'component': 'Memory',
                        'severity': 'high' if analysis['memory_analysis']['average'] > 90 else 'medium',
                        'description': f"High memory usage average: {analysis['memory_analysis']['average']:.1f}%",
                        'impact': 'Potential swapping and performance degradation'
                    })
            
            # 디스크 I/O 분석
            if 'disk_read' in df.columns and 'disk_write' in df.columns:
                df['disk_total_io'] = df['disk_read'] + df['disk_write']
                
                analysis['disk_analysis'] = {
                    'avg_read': float(df['disk_read'].mean()),
                    'avg_write': float(df['disk_write'].mean()),
                    'peak_read': float(df['disk_read'].max()),
                    'peak_write': float(df['disk_write'].max()),
                    'total_io_avg': float(df['disk_total_io'].mean()),
                    'io_trend': self.calculate_trend(df['disk_total_io'].values)
                }
            
            # 네트워크 분석
            if 'network_sent' in df.columns and 'network_recv' in df.columns:
                df['network_total'] = df['network_sent'] + df['network_recv']
                
                analysis['network_analysis'] = {
                    'avg_sent': float(df['network_sent'].mean()),
                    'avg_recv': float(df['network_recv'].mean()),
                    'peak_sent': float(df['network_sent'].max()),
                    'peak_recv': float(df['network_recv'].max()),
                    'total_traffic_avg': float(df['network_total'].mean()),
                    'traffic_trend': self.calculate_trend(df['network_total'].values)
                }
            
            # 전체 시스템 건강도
            health_scores = []
            if analysis['cpu_analysis']:
                cpu_health = max(0, 100 - analysis['cpu_analysis']['average'])
                health_scores.append(cpu_health)
            
            if analysis['memory_analysis']:
                memory_health = max(0, 100 - analysis['memory_analysis']['average'])
                health_scores.append(memory_health)
            
            if health_scores:
                analysis['overall_health'] = {
                    'score': float(np.mean(health_scores)),
                    'grade': self.calculate_health_grade(np.mean(health_scores)),
                    'components_analyzed': len(health_scores)
                }
            
            # 최적화 기회 식별
            if analysis['cpu_analysis'].get('time_above_80', 0) > 20:
                analysis['optimization_opportunities'].append({
                    'category': 'CPU',
                    'opportunity': 'Process optimization',
                    'description': 'CPU usage frequently exceeds 80%',
                    'potential_improvement': '10-20% performance gain',
                    'effort': 'Medium'
                })
            
            if analysis['memory_analysis'].get('time_above_80', 0) > 15:
                analysis['optimization_opportunities'].append({
                    'category': 'Memory',
                    'opportunity': 'Memory management',
                    'description': 'Memory usage frequently exceeds 80%',
                    'potential_improvement': '15-25% performance gain',
                    'effort': 'Low to Medium'
                })
        
        except Exception as e:
            self.analytics_logger.error(f"Performance analysis failed: {e}")
        
        return analysis
    
    def analyze_security_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """보안 데이터 분석"""
        analysis = {
            'threat_summary': {},
            'alert_patterns': {},
            'security_score': 100.0,
            'risk_assessment': {},
            'incident_timeline': [],
            'security_recommendations': []
        }
        
        try:
            alerts = raw_data.get('alerts', [])
            security_events = raw_data.get('security_events', [])
            
            # 알림 분석
            if alerts:
                df_alerts = pd.DataFrame(alerts)
                
                # 심각도별 분포
                severity_counts = df_alerts['severity'].value_counts().to_dict()
                analysis['threat_summary'] = {
                    'total_alerts': len(alerts),
                    'severity_distribution': severity_counts,
                    'critical_count': severity_counts.get('critical', 0),
                    'warning_count': severity_counts.get('warning', 0),
                    'info_count': severity_counts.get('info', 0)
                }
                
                # 보안 점수 계산
                score_deduction = 0
                score_deduction += severity_counts.get('critical', 0) * 20
                score_deduction += severity_counts.get('warning', 0) * 5
                score_deduction += severity_counts.get('info', 0) * 1
                
                analysis['security_score'] = max(0, 100 - score_deduction)
            
            # 보안 이벤트 분석
            if security_events:
                df_security = pd.DataFrame(security_events)
                
                # 위협 유형별 분류
                threat_types = df_security['threat_type'].value_counts().to_dict()
                
                analysis['risk_assessment'] = {
                    'total_threats': len(security_events),
                    'threat_types': threat_types,
                    'malware_incidents': threat_types.get('malware', 0),
                    'intrusion_attempts': threat_types.get('intrusion', 0),
                    'suspicious_activities': threat_types.get('suspicious_activity', 0),
                    'vulnerabilities': threat_types.get('vulnerability', 0)
                }
                
                # 시간대별 인시던트 패턴
                if 'timestamp' in df_security.columns:
                    df_security['hour'] = pd.to_datetime(df_security['timestamp'], unit='s').dt.hour
                    hourly_incidents = df_security['hour'].value_counts().sort_index()
                    
                    analysis['alert_patterns'] = {
                        'peak_hour': int(hourly_incidents.idxmax()),
                        'peak_hour_count': int(hourly_incidents.max()),
                        'quiet_hour': int(hourly_incidents.idxmin()),
                        'hourly_distribution': hourly_incidents.to_dict()
                    }
            
            # 보안 권장사항 생성
            if analysis['threat_summary'].get('critical_count', 0) > 0:
                analysis['security_recommendations'].append({
                    'priority': 'URGENT',
                    'category': 'Threat Response',
                    'recommendation': 'Address critical security alerts immediately',
                    'impact': 'High'
                })
            
            if analysis['security_score'] < 80:
                analysis['security_recommendations'].append({
                    'priority': 'HIGH',
                    'category': 'Security Posture',
                    'recommendation': 'Comprehensive security audit recommended',
                    'impact': 'Medium to High'
                })
            
            if analysis['risk_assessment'].get('malware_incidents', 0) > 0:
                analysis['security_recommendations'].append({
                    'priority': 'HIGH',
                    'category': 'Malware Protection',
                    'recommendation': 'Update antivirus and run full system scan',
                    'impact': 'High'
                })
        
        except Exception as e:
            self.analytics_logger.error(f"Security analysis failed: {e}")
        
        return analysis
    
    def generate_metrics_summary(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """메트릭스 요약 생성"""
        summary = {
            'period_info': {},
            'data_points': 0,
            'coverage': {},
            'key_statistics': {},
            'system_overview': {}
        }
        
        try:
            metrics = raw_data.get('metrics', [])
            start_time = raw_data.get('period_start')
            end_time = raw_data.get('period_end')
            
            if start_time and end_time:
                period_hours = (end_time - start_time) / 3600
                summary['period_info'] = {
                    'start': datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
                    'end': datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S'),
                    'duration_hours': round(period_hours, 2),
                    'duration_days': round(period_hours / 24, 2)
                }
            
            if metrics:
                df = pd.DataFrame(metrics)
                summary['data_points'] = len(df)
                
                # 데이터 커버리지
                expected_points = int((end_time - start_time) / 60)  # 1분 간격 가정
                coverage_percent = (len(df) / expected_points) * 100 if expected_points > 0 else 0
                
                summary['coverage'] = {
                    'actual_points': len(df),
                    'expected_points': expected_points,
                    'coverage_percent': min(100, coverage_percent)
                }
                
                # 주요 통계
                if 'cpu_avg' in df.columns:
                    summary['key_statistics']['cpu'] = {
                        'avg': float(df['cpu_avg'].mean()),
                        'max': float(df['cpu_avg'].max()),
                        'min': float(df['cpu_avg'].min())
                    }
                
                if 'memory_percent' in df.columns:
                    summary['key_statistics']['memory'] = {
                        'avg': float(df['memory_percent'].mean()),
                        'max': float(df['memory_percent'].max()),
                        'min': float(df['memory_percent'].min())
                    }
                
                if 'process_count' in df.columns:
                    summary['key_statistics']['processes'] = {
                        'avg': float(df['process_count'].mean()),
                        'max': int(df['process_count'].max()),
                        'min': int(df['process_count'].min())
                    }
            
            # 현재 시스템 개요
            current_state = raw_data.get('current_state', {})
            if current_state:
                summary['system_overview'] = {
                    'cpu_cores': len(current_state.get('cpu_cores', [])),
                    'current_cpu': float(np.mean(current_state.get('cpu_cores', [0]))),
                    'current_memory': current_state.get('memory_percent', 0),
                    'uptime_hours': round(current_state.get('uptime', 0) / 3600, 1),
                    'active_processes': current_state.get('process_count', 0)
                }
        
        except Exception as e:
            self.analytics_logger.error(f"Metrics summary generation failed: {e}")
        
        return summary
    
    def analyze_trends(self, raw_data: Dict[str, Any]) -> List[PerformanceTrend]:
        """트렌드 분석"""
        trends = []
        
        try:
            metrics = raw_data.get('metrics', [])
            if not metrics:
                return trends
            
            df = pd.DataFrame(metrics)
            
            # 각 주요 메트릭에 대한 트렌드 분석
            trend_components = ['cpu_avg', 'memory_percent', 'disk_read', 'disk_write']
            
            for component in trend_components:
                if component in df.columns:
                    trend = self.calculate_detailed_trend(df[component].values, component)
                    if trend:
                        trends.append(trend)
        
        except Exception as e:
            self.analytics_logger.error(f"Trend analysis failed: {e}")
        
        return trends
    
    def calculate_detailed_trend(self, data: np.array, component: str) -> Optional[PerformanceTrend]:
        """상세 트렌드 계산"""
        try:
            if len(data) < 10:  # 최소 데이터 포인트 필요
                return None
            
            # 트렌드 방향 및 강도 계산
            x = np.arange(len(data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
            
            # 트렌드 방향 결정
            if abs(slope) < 0.01:
                trend_direction = "stable"
            elif slope > 0:
                trend_direction = "increasing"
            else:
                trend_direction = "decreasing"
            
            # 변동성 확인
            volatility = np.std(data) / np.mean(data) if np.mean(data) > 0 else 0
            if volatility > 0.3:
                trend_direction = "volatile"
            
            # 트렌드 강도 (R-squared 값 사용)
            trend_strength = r_value ** 2
            
            # 예측 (간단한 선형 외삽)
            forecast_7d = data[-1] + slope * (7 * 24 * 60)  # 7일 후 (분 단위로 가정)
            forecast_30d = data[-1] + slope * (30 * 24 * 60)  # 30일 후
            
            # 신뢰 구간 계산
            confidence_interval = (
                float(forecast_7d - 1.96 * std_err),
                float(forecast_7d + 1.96 * std_err)
            )
            
            # 이상치 개수
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            anomaly_count = len(data[(data < lower_bound) | (data > upper_bound)])
            
            return PerformanceTrend(
                component=component,
                trend_direction=trend_direction,
                trend_strength=float(trend_strength),
                correlation_factors=[],  # 추후 구현
                forecast_7d=float(forecast_7d),
                forecast_30d=float(forecast_30d),
                confidence_interval=confidence_interval,
                anomaly_count=anomaly_count
            )
        
        except Exception as e:
            self.analytics_logger.error(f"Detailed trend calculation failed for {component}: {e}")
            return None
    
    def calculate_trend(self, data: np.array) -> str:
        """간단한 트렌드 계산"""
        try:
            if len(data) < 5:
                return "insufficient_data"
            
            # 선형 회귀로 기울기 계산
            x = np.arange(len(data))
            slope, _, r_value, _, _ = stats.linregress(x, data)
            
            # R-squared가 낮으면 불안정
            if r_value ** 2 < 0.1:
                return "volatile"
            
            if slope > 0.1:
                return "increasing"
            elif slope < -0.1:
                return "decreasing"
            else:
                return "stable"
        
        except Exception:
            return "unknown"
    
    def calculate_health_grade(self, score: float) -> str:
        """건강도 등급 계산"""
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        elif score >= 50:
            return "D"
        else:
            return "F"
    
    def generate_insights(self, raw_data: Dict[str, Any], 
                         performance_analysis: Dict[str, Any],
                         security_analysis: Dict[str, Any]) -> List[SystemInsight]:
        """시스템 인사이트 생성"""
        insights = []
        
        try:
            # 성능 기반 인사이트
            cpu_analysis = performance_analysis.get('cpu_analysis', {})
            if cpu_analysis.get('average', 0) > 80:
                insights.append(SystemInsight(
                    insight_type="bottleneck",
                    severity="high" if cpu_analysis['average'] > 90 else "medium",
                    confidence=0.9,
                    title="CPU Performance Bottleneck Detected",
                    description=f"CPU usage averaging {cpu_analysis['average']:.1f}% indicates performance constraints",
                    impact_analysis="System responsiveness may be degraded, affecting user experience",
                    recommended_actions=[
                        "Identify high CPU usage processes",
                        "Consider upgrading CPU or optimizing workload",
                        "Implement process scheduling optimization"
                    ],
                    evidence=cpu_analysis,
                    estimated_improvement="10-25% performance improvement possible"
                ))
            
            # 메모리 기반 인사이트
            memory_analysis = performance_analysis.get('memory_analysis', {})
            if memory_analysis.get('average', 0) > 85:
                insights.append(SystemInsight(
                    insight_type="bottleneck",
                    severity="high",
                    confidence=0.85,
                    title="Memory Pressure Detected",
                    description=f"Memory usage averaging {memory_analysis['average']:.1f}% indicates memory pressure",
                    impact_analysis="Potential swapping and significant performance degradation",
                    recommended_actions=[
                        "Add more RAM",
                        "Optimize memory-intensive applications",
                        "Implement memory cleanup procedures"
                    ],
                    evidence=memory_analysis,
                    estimated_improvement="20-40% performance improvement with memory upgrade"
                ))
            
            # 보안 기반 인사이트
            security_score = security_analysis.get('security_score', 100)
            if security_score < 70:
                insights.append(SystemInsight(
                    insight_type="security",
                    severity="critical" if security_score < 50 else "high",
                    confidence=0.95,
                    title="Security Posture Requires Attention",
                    description=f"Security score of {security_score:.1f} indicates multiple security concerns",
                    impact_analysis="System may be vulnerable to attacks and data breaches",
                    recommended_actions=[
                        "Address critical security alerts immediately",
                        "Perform comprehensive security audit",
                        "Update security policies and procedures"
                    ],
                    evidence=security_analysis,
                    estimated_improvement="Significant risk reduction with proper security measures"
                ))
            
            # 최적화 기회 인사이트
            optimization_opportunities = performance_analysis.get('optimization_opportunities', [])
            if optimization_opportunities:
                for opp in optimization_opportunities:
                    insights.append(SystemInsight(
                        insight_type="optimization",
                        severity="medium",
                        confidence=0.7,
                        title=f"Optimization Opportunity: {opp['category']}",
                        description=opp['description'],
                        impact_analysis=f"Potential improvement: {opp['potential_improvement']}",
                        recommended_actions=[f"Implement {opp['opportunity'].lower()}"],
                        evidence=opp,
                        estimated_improvement=opp['potential_improvement']
                    ))
        
        except Exception as e:
            self.analytics_logger.error(f"Insight generation failed: {e}")
        
        return insights
    
    def generate_recommendations(self, insights: List[SystemInsight], 
                               trends: List[PerformanceTrend]) -> List[str]:
        """권장사항 생성"""
        recommendations = []
        
        try:
            # 인사이트 기반 권장사항
            high_severity_insights = [i for i in insights if i.severity in ['critical', 'high']]
            
            if high_severity_insights:
                recommendations.append("🚨 URGENT: Address high-severity system issues immediately")
                
                for insight in high_severity_insights[:3]:  # 상위 3개만
                    recommendations.extend(insight.recommended_actions[:2])  # 각각 최대 2개 액션
            
            # 트렌드 기반 권장사항
            increasing_trends = [t for t in trends if t.trend_direction == "increasing"]
            
            for trend in increasing_trends:
                if trend.component == "cpu_avg" and trend.forecast_7d > 85:
                    recommendations.append("📈 CPU usage trending upward - consider capacity planning")
                elif trend.component == "memory_percent" and trend.forecast_7d > 90:
                    recommendations.append("📈 Memory usage trending upward - plan memory upgrade")
            
            # 일반적인 권장사항
            if not recommendations:
                recommendations.extend([
                    "✅ System performance appears stable",
                    "🔍 Continue regular monitoring and maintenance",
                    "📊 Consider setting up automated alerts for proactive management"
                ])
            
            # 최대 10개 권장사항으로 제한
            recommendations = recommendations[:10]
        
        except Exception as e:
            self.analytics_logger.error(f"Recommendation generation failed: {e}")
            recommendations = ["❌ Unable to generate recommendations due to analysis error"]
        
        return recommendations
    
    def generate_charts(self, raw_data: Dict[str, Any], 
                       performance_analysis: Dict[str, Any],
                       report_id: str) -> List[str]:
        """차트 생성"""
        chart_files = []
        
        try:
            metrics = raw_data.get('metrics', [])
            if not metrics:
                return chart_files
            
            df = pd.DataFrame(metrics)
            
            # 타임스탬프를 datetime으로 변환
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # 1. 종합 성능 차트
            chart_file = self.create_performance_overview_chart(df, report_id)
            if chart_file:
                chart_files.append(chart_file)
            
            # 2. CPU 상세 차트
            if 'cpu_avg' in df.columns:
                chart_file = self.create_cpu_detailed_chart(df, report_id)
                if chart_file:
                    chart_files.append(chart_file)
            
            # 3. 메모리 분석 차트
            if 'memory_percent' in df.columns:
                chart_file = self.create_memory_analysis_chart(df, report_id)
                if chart_file:
                    chart_files.append(chart_file)
            
            # 4. 디스크 I/O 차트
            if 'disk_read' in df.columns and 'disk_write' in df.columns:
                chart_file = self.create_disk_io_chart(df, report_id)
                if chart_file:
                    chart_files.append(chart_file)
            
            # 5. 성능 히트맵
            chart_file = self.create_performance_heatmap(df, report_id)
            if chart_file:
                chart_files.append(chart_file)
        
        except Exception as e:
            self.analytics_logger.error(f"Chart generation failed: {e}")
        
        return chart_files
    
    def create_performance_overview_chart(self, df: pd.DataFrame, report_id: str) -> Optional[str]:
        """종합 성능 개요 차트"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('System Performance Overview', fontsize=20, color=QUANTUM_THEME['quantum_cyan'])
            fig.patch.set_facecolor(QUANTUM_THEME['void_black'])
            
            # CPU 차트
            if 'cpu_avg' in df.columns and 'datetime' in df.columns:
                axes[0, 0].plot(df['datetime'], df['cpu_avg'], 
                               color=QUANTUM_THEME['quantum_red'], linewidth=2, alpha=0.8)
                axes[0, 0].fill_between(df['datetime'], df['cpu_avg'], 
                                       color=QUANTUM_THEME['quantum_red'], alpha=0.3)
                axes[0, 0].set_title('CPU Usage (%)', color=QUANTUM_THEME['text_primary'])
                axes[0, 0].axhline(y=80, color=QUANTUM_THEME['quantum_orange'], linestyle='--', alpha=0.7)
                axes[0, 0].set_ylim(0, 100)
            
            # 메모리 차트
            if 'memory_percent' in df.columns and 'datetime' in df.columns:
                axes[0, 1].plot(df['datetime'], df['memory_percent'], 
                               color=QUANTUM_THEME['quantum_yellow'], linewidth=2, alpha=0.8)
                axes[0, 1].fill_between(df['datetime'], df['memory_percent'], 
                                       color=QUANTUM_THEME['quantum_yellow'], alpha=0.3)
                axes[0, 1].set_title('Memory Usage (%)', color=QUANTUM_THEME['text_primary'])
                axes[0, 1].axhline(y=80, color=QUANTUM_THEME['quantum_orange'], linestyle='--', alpha=0.7)
                axes[0, 1].set_ylim(0, 100)
            
            # 디스크 I/O 차트
            if 'disk_read' in df.columns and 'disk_write' in df.columns and 'datetime' in df.columns:
                axes[1, 0].plot(df['datetime'], df['disk_read'], 
                               color=QUANTUM_THEME['quantum_green'], linewidth=2, label='Read', alpha=0.8)
                axes[1, 0].plot(df['datetime'], df['disk_write'], 
                               color=QUANTUM_THEME['quantum_blue'], linewidth=2, label='Write', alpha=0.8)
                axes[1, 0].set_title('Disk I/O (MB/s)', color=QUANTUM_THEME['text_primary'])
                axes[1, 0].legend()
            
            # 네트워크 차트
            if 'network_sent' in df.columns and 'network_recv' in df.columns and 'datetime' in df.columns:
                axes[1, 1].plot(df['datetime'], df['network_sent'], 
                               color=QUANTUM_THEME['quantum_purple'], linewidth=2, label='Sent', alpha=0.8)
                axes[1, 1].plot(df['network_recv'], 
                               color=QUANTUM_THEME['quantum_cyan'], linewidth=2, label='Received', alpha=0.8)
                axes[1, 1].set_title('Network Traffic (MB/s)', color=QUANTUM_THEME['text_primary'])
                axes[1, 1].legend()
            
            # 스타일 적용
            for ax in axes.flat:
                ax.set_facecolor(QUANTUM_THEME['dark_matter'])
                ax.tick_params(colors=QUANTUM_THEME['text_secondary'])
                ax.grid(True, alpha=0.3, color=QUANTUM_THEME['cosmic_dust'])
                
                # x축 레이블 회전
                for label in ax.get_xticklabels():
                    label.set_rotation(45)
                    label.set_fontsize(8)
            
            plt.tight_layout()
            
            # 파일 저장
            chart_file = self.charts_directory / f"{report_id}_performance_overview.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight', 
                       facecolor=QUANTUM_THEME['void_black'], edgecolor='none')
            plt.close()
            
            return str(chart_file)
        
        except Exception as e:
            self.analytics_logger.error(f"Performance overview chart creation failed: {e}")
            return None
    
    def create_cpu_detailed_chart(self, df: pd.DataFrame, report_id: str) -> Optional[str]:
        """CPU 상세 차트"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
            fig.suptitle('CPU Performance Analysis', fontsize=20, color=QUANTUM_THEME['quantum_cyan'])
            fig.patch.set_facecolor(QUANTUM_THEME['void_black'])
            
            # CPU 사용률 시계열
            if 'cpu_avg' in df.columns and 'datetime' in df.columns:
                ax1.plot(df['datetime'], df['cpu_avg'], 
                        color=QUANTUM_THEME['quantum_red'], linewidth=2, alpha=0.8)
                ax1.fill_between(df['datetime'], df['cpu_avg'], 
                               color=QUANTUM_THEME['quantum_red'], alpha=0.3)
                
                # 임계값 선
                ax1.axhline(y=80, color=QUANTUM_THEME['quantum_orange'], linestyle='--', alpha=0.7, label='Warning (80%)')
                ax1.axhline(y=90, color=QUANTUM_THEME['quantum_red'], linestyle='--', alpha=0.7, label='Critical (90%)')
                
                ax1.set_title('CPU Usage Over Time', color=QUANTUM_THEME['text_primary'])
                ax1.set_ylabel('CPU Usage (%)')
                ax1.set_ylim(0, 100)
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # CPU 사용률 히스토그램
            if 'cpu_avg' in df.columns:
                ax2.hist(df['cpu_avg'], bins=30, color=QUANTUM_THEME['quantum_red'], 
                        alpha=0.7, edgecolor=QUANTUM_THEME['quantum_cyan'])
                ax2.set_title('CPU Usage Distribution', color=QUANTUM_THEME['text_primary'])
                ax2.set_xlabel('CPU Usage (%)')
                ax2.set_ylabel('Frequency')
                ax2.grid(True, alpha=0.3)
                
                # 평균선 추가
                mean_cpu = df['cpu_avg'].mean()
                ax2.axvline(x=mean_cpu, color=QUANTUM_THEME['quantum_yellow'], 
                           linestyle='-', linewidth=3, label=f'Average: {mean_cpu:.1f}%')
                ax2.legend()
            
            # 스타일 적용
            for ax in [ax1, ax2]:
                ax.set_facecolor(QUANTUM_THEME['dark_matter'])
                ax.tick_params(colors=QUANTUM_THEME['text_secondary'])
            
            plt.tight_layout()
            
            # 파일 저장
            chart_file = self.charts_directory / f"{report_id}_cpu_detailed.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight', 
                       facecolor=QUANTUM_THEME['void_black'], edgecolor='none')
            plt.close()
            
            return str(chart_file)
        
        except Exception as e:
            self.analytics_logger.error(f"CPU detailed chart creation failed: {e}")
            return None
    
    def create_memory_analysis_chart(self, df: pd.DataFrame, report_id: str) -> Optional[str]:
        """메모리 분석 차트"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle('Memory Usage Analysis', fontsize=20, color=QUANTUM_THEME['quantum_cyan'])
            fig.patch.set_facecolor(QUANTUM_THEME['void_black'])
            
            # 메모리 사용률 시계열
            if 'memory_percent' in df.columns and 'datetime' in df.columns:
                ax1.plot(df['datetime'], df['memory_percent'], 
                        color=QUANTUM_THEME['quantum_yellow'], linewidth=2, alpha=0.8)
                ax1.fill_between(df['datetime'], df['memory_percent'], 
                               color=QUANTUM_THEME['quantum_yellow'], alpha=0.3)
                
                # 임계값 선
                ax1.axhline(y=80, color=QUANTUM_THEME['quantum_orange'], linestyle='--', alpha=0.7)
                ax1.axhline(y=90, color=QUANTUM_THEME['quantum_red'], linestyle='--', alpha=0.7)
                
                ax1.set_title('Memory Usage Over Time', color=QUANTUM_THEME['text_primary'])
                ax1.set_ylabel('Memory Usage (%)')
                ax1.set_ylim(0, 100)
                ax1.grid(True, alpha=0.3)
            
            # 메모리 사용률 박스플롯
            if 'memory_percent' in df.columns:
                box_plot = ax2.boxplot([df['memory_percent']], patch_artist=True, 
                                     labels=['Memory Usage'])
                
                # 박스플롯 색상 설정
                for patch in box_plot['boxes']:
                    patch.set_facecolor(QUANTUM_THEME['quantum_yellow'])
                    patch.set_alpha(0.7)
                
                ax2.set_title('Memory Usage Statistics', color=QUANTUM_THEME['text_primary'])
                ax2.set_ylabel('Memory Usage (%)')
                ax2.grid(True, alpha=0.3)
            
            # 스타일 적용
            for ax in [ax1, ax2]:
                ax.set_facecolor(QUANTUM_THEME['dark_matter'])
                ax.tick_params(colors=QUANTUM_THEME['text_secondary'])
            
            plt.tight_layout()
            
            # 파일 저장
            chart_file = self.charts_directory / f"{report_id}_memory_analysis.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight', 
                       facecolor=QUANTUM_THEME['void_black'], edgecolor='none')
            plt.close()
            
            return str(chart_file)
        
        except Exception as e:
            self.analytics_logger.error(f"Memory analysis chart creation failed: {e}")
            return None
    
    def create_disk_io_chart(self, df: pd.DataFrame, report_id: str) -> Optional[str]:
        """디스크 I/O 차트"""
        try:
            fig, ax = plt.subplots(figsize=(16, 8))
            fig.suptitle('Disk I/O Performance', fontsize=20, color=QUANTUM_THEME['quantum_cyan'])
            fig.patch.set_facecolor(QUANTUM_THEME['void_black'])
            
            if 'disk_read' in df.columns and 'disk_write' in df.columns and 'datetime' in df.columns:
                # 읽기/쓰기 영역 차트
                ax.fill_between(df['datetime'], df['disk_read'], 
                               color=QUANTUM_THEME['quantum_green'], alpha=0.6, label='Read')
                ax.fill_between(df['datetime'], -df['disk_write'], 
                               color=QUANTUM_THEME['quantum_blue'], alpha=0.6, label='Write')
                
                ax.set_title('Disk I/O Activity (MB/s)', color=QUANTUM_THEME['text_primary'])
                ax.set_ylabel('I/O Rate (MB/s)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # X축 0선
                ax.axhline(y=0, color=QUANTUM_THEME['text_primary'], linewidth=1, alpha=0.5)
            
            ax.set_facecolor(QUANTUM_THEME['dark_matter'])
            ax.tick_params(colors=QUANTUM_THEME['text_secondary'])
            
            plt.tight_layout()
            
            # 파일 저장
            chart_file = self.charts_directory / f"{report_id}_disk_io.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight', 
                       facecolor=QUANTUM_THEME['void_black'], edgecolor='none')
            plt.close()
            
            return str(chart_file)
        
        except Exception as e:
            self.analytics_logger.error(f"Disk I/O chart creation failed: {e}")
            return None
    
    def create_performance_heatmap(self, df: pd.DataFrame, report_id: str) -> Optional[str]:
        """성능 히트맵"""
        try:
            # 시간별 성능 히트맵 생성
            if 'datetime' not in df.columns:
                return None
            
            # 시간별 집계를 위한 데이터 준비
            df_hourly = df.copy()
            df_hourly['hour'] = df_hourly['datetime'].dt.hour
            df_hourly['day'] = df_hourly['datetime'].dt.day
            
            # 각 메트릭별 시간별 평균 계산
            metrics_cols = ['cpu_avg', 'memory_percent', 'disk_read', 'disk_write']
            available_metrics = [col for col in metrics_cols if col in df_hourly.columns]
            
            if not available_metrics:
                return None
            
            # 피벗 테이블 생성
            pivot_data = []
            for metric in available_metrics:
                hourly_avg = df_hourly.groupby('hour')[metric].mean()
                pivot_data.append(hourly_avg)
            
            heatmap_data = pd.DataFrame(pivot_data, index=available_metrics)
            
            # 히트맵 생성
            fig, ax = plt.subplots(figsize=(16, 8))
            fig.patch.set_facecolor(QUANTUM_THEME['void_black'])
            
            # 커스텀 컬러맵
            colors = ['#000000', QUANTUM_THEME['quantum_blue'], 
                     QUANTUM_THEME['quantum_cyan'], QUANTUM_THEME['quantum_green'],
                     QUANTUM_THEME['quantum_yellow'], QUANTUM_THEME['quantum_red']]
            n_bins = 100
            cmap = mcolors.LinearSegmentedColormap.from_list('quantum', colors, N=n_bins)
            
            im = ax.imshow(heatmap_data.values, cmap=cmap, aspect='auto', interpolation='nearest')
            
            # 축 설정
            ax.set_xticks(range(24))
            ax.set_xticklabels([f"{h:02d}:00" for h in range(24)])
            ax.set_yticks(range(len(available_metrics)))
            ax.set_yticklabels(available_metrics)
            
            ax.set_title('Performance Heatmap by Hour', fontsize=20, color=QUANTUM_THEME['quantum_cyan'])
            ax.set_xlabel('Hour of Day', color=QUANTUM_THEME['text_primary'])
            ax.set_ylabel('Metrics', color=QUANTUM_THEME['text_primary'])
            
            # 컬러바
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Performance Level', color=QUANTUM_THEME['text_primary'])
            cbar.ax.yaxis.set_tick_params(color=QUANTUM_THEME['text_secondary'])
            
            # 스타일 적용
            ax.set_facecolor(QUANTUM_THEME['dark_matter'])
            ax.tick_params(colors=QUANTUM_THEME['text_secondary'])
            
            plt.tight_layout()
            
            # 파일 저장
            chart_file = self.charts_directory / f"{report_id}_performance_heatmap.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight', 
                       facecolor=QUANTUM_THEME['void_black'], edgecolor='none')
            plt.close()
            
            return str(chart_file)
        
        except Exception as e:
            self.analytics_logger.error(f"Performance heatmap creation failed: {e}")
            return None
    
    def export_report(self, report: AnalyticsReport, raw_data: Dict[str, Any], format_type: str) -> Optional[str]:
        """보고서 내보내기"""
        try:
            if format_type == "html":
                return self.export_html_report(report, raw_data)
            elif format_type == "pdf":
                return self.export_pdf_report(report, raw_data)
            elif format_type == "excel" and HAS_EXCEL:
                return self.export_excel_report(report, raw_data)
            else:
                self.analytics_logger.warning(f"Unsupported format: {format_type}")
                return None
        
        except Exception as e:
            self.analytics_logger.error(f"Report export failed for format {format_type}: {e}")
            return None
    
    def export_html_report(self, report: AnalyticsReport, raw_data: Dict[str, Any]) -> Optional[str]:
        """HTML 보고서 내보내기"""
        try:
            # HTML 템플릿
            html_template = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{report.title}}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 100%);
            color: #ffffff;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(22, 33, 62, 0.9);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0, 245, 255, 0.3);
        }
        .header {
            text-align: center;
            border-bottom: 2px solid #00f5ff;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #00f5ff;
            font-size: 2.5em;
            margin: 0;
            text-shadow: 0 0 10px rgba(0, 245, 255, 0.5);
        }
        .header .subtitle {
            color: #b8c5d1;
            font-size: 1.2em;
            margin-top: 10px;
        }
        .section {
            margin: 30px 0;
            padding: 20px;
            background: rgba(15, 15, 26, 0.8);
            border-radius: 10px;
            border-left: 4px solid #00f5ff;
        }
        .section h2 {
            color: #00ff80;
            font-size: 1.8em;
            margin-top: 0;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background: rgba(26, 26, 46, 0.9);
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #8000ff;
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #ffff00;
            margin: 10px 0;
        }
        .metric-label {
            color: #b8c5d1;
            font-size: 0.9em;
        }
        .recommendations {
            background: rgba(255, 128, 0, 0.1);
            border: 1px solid #ff8000;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }
        .recommendation-item {
            margin: 10px 0;
            padding: 10px;
            background: rgba(255, 128, 0, 0.1);
            border-radius: 5px;
        }
        .chart-container {
            text-align: center;
            margin: 20px 0;
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #333;
            color: #888;
        }
        .status-critical { color: #ff073a; }
        .status-warning { color: #ff8c00; }
        .status-good { color: #39ff14; }
        .status-info { color: #00f5ff; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{report.title}}</h1>
            <div class="subtitle">
                Generated: {{generation_time}}<br>
                Period: {{period_start}} - {{period_end}}<br>
                Processing Time: {{processing_time}}s
            </div>
        </div>

        <div class="section">
            <h2>📊 Executive Summary</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Data Points Analyzed</div>
                    <div class="metric-value">{{data_points}}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">System Health Score</div>
                    <div class="metric-value status-{{health_status}}">{{health_score}}/100</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Security Score</div>
                    <div class="metric-value status-{{security_status}}">{{security_score}}/100</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Critical Issues</div>
                    <div class="metric-value status-{{issues_status}}">{{critical_issues}}</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>⚡ Performance Analysis</h2>
            <div class="metrics-grid">
                {% for metric, data in performance_metrics.items() %}
                <div class="metric-card">
                    <div class="metric-label">{{metric}} (Average)</div>
                    <div class="metric-value">{{data.average}}%</div>
                    <div class="metric-label">Peak: {{data.peak}}% | Trend: {{data.trend}}</div>
                </div>
                {% endfor %}
            </div>
        </div>

        {% if charts %}
        <div class="section">
            <h2>📈 Performance Charts</h2>
            {% for chart in charts %}
            <div class="chart-container">
                <img src="{{chart}}" alt="Performance Chart">
            </div>
            {% endfor %}
        </div>
        {% endif %}

        <div class="section">
            <h2>🔍 Key Insights</h2>
            {% for insight in insights %}
            <div class="recommendation-item">
                <strong>{{insight.title}}</strong><br>
                {{insight.description}}<br>
                <em>Impact: {{insight.impact_analysis}}</em>
            </div>
            {% endfor %}
        </div>

        <div class="section">
            <h2>💡 Recommendations</h2>
            <div class="recommendations">
                {% for recommendation in recommendations %}
                <div class="recommendation-item">
                    {{recommendation}}
                </div>
                {% endfor %}
            </div>
        </div>

        <div class="footer">
            <p>SysWatch Pro Quantum Analytics Engine</p>
            <p>© 2025 SysWatch Technologies Ltd. - Enterprise Edition</p>
        </div>
    </div>
</body>
</html>
"""

            # 템플릿 데이터 준비
            template_data = {
                'report': report,
                'generation_time': datetime.fromtimestamp(report.generation_time).strftime('%Y-%m-%d %H:%M:%S'),
                'period_start': datetime.fromtimestamp(report.period_start).strftime('%Y-%m-%d %H:%M:%S'),
                'period_end': datetime.fromtimestamp(report.period_end).strftime('%Y-%m-%d %H:%M:%S'),
                'processing_time': f"{report.processing_time:.2f}",
                'data_points': len(raw_data.get('metrics', [])),
                'health_score': report.performance_analysis.get('overall_health', {}).get('score', 0),
                'health_status': 'good' if report.performance_analysis.get('overall_health', {}).get('score', 0) > 80 else 'warning',
                'security_score': report.security_analysis.get('security_score', 100),
                'security_status': 'good' if report.security_analysis.get('security_score', 100) > 80 else 'warning',
                'critical_issues': len([i for i in report.security_analysis.get('security_recommendations', []) if 'URGENT' in str(i)]),
                'issues_status': 'critical' if len([i for i in report.security_analysis.get('security_recommendations', []) if 'URGENT' in str(i)]) > 0 else 'good',
                'performance_metrics': {
                    'CPU': report.performance_analysis.get('cpu_analysis', {}),
                    'Memory': report.performance_analysis.get('memory_analysis', {}),
                    'Disk': report.performance_analysis.get('disk_analysis', {}),
                    'Network': report.performance_analysis.get('network_analysis', {})
                },
                'charts': [os.path.basename(chart) for chart in report.charts],
                'insights': [],  # 인사이트 데이터는 별도 구현 필요
                'recommendations': report.recommendations
            }

            # Jinja2 템플릿 렌더링 (간단한 문자열 치환으로 대체)
            html_content = html_template
            
            # 간단한 템플릿 변수 치환
            for key, value in template_data.items():
                if isinstance(value, str):
                    html_content = html_content.replace(f"{{{{{key}}}}}", value)
                elif isinstance(value, (int, float)):
                    html_content = html_content.replace(f"{{{{{key}}}}}", str(value))
            
            # 파일 저장
            html_file = self.reports_directory / f"{report.id}.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return str(html_file)
        
        except Exception as e:
            self.analytics_logger.error(f"HTML report export failed: {e}")
            return None
    
    def export_pdf_report(self, report: AnalyticsReport, raw_data: Dict[str, Any]) -> Optional[str]:
        """PDF 보고서 내보내기"""
        try:
            # 먼저 HTML 생성
            html_file = self.export_html_report(report, raw_data)
            if not html_file:
                return None
            
            # PDF 변환 (간단한 matplotlib 기반 PDF)
            pdf_file = self.reports_directory / f"{report.id}.pdf"
            
            with PdfPages(pdf_file) as pdf:
                # 제목 페이지
                fig, ax = plt.subplots(figsize=(8.5, 11))
                fig.patch.set_facecolor('white')
                ax.axis('off')
                
                # 제목
                ax.text(0.5, 0.8, report.title, transform=ax.transAxes,
                       fontsize=24, fontweight='bold', ha='center')
                
                # 생성 정보
                generation_info = f"""
Generated: {datetime.fromtimestamp(report.generation_time).strftime('%Y-%m-%d %H:%M:%S')}
Period: {datetime.fromtimestamp(report.period_start).strftime('%Y-%m-%d %H:%M:%S')} - {datetime.fromtimestamp(report.period_end).strftime('%Y-%m-%d %H:%M:%S')}
Processing Time: {report.processing_time:.2f} seconds
                """
                
                ax.text(0.5, 0.6, generation_info, transform=ax.transAxes,
                       fontsize=12, ha='center', va='center')
                
                # 요약 정보
                summary_info = f"""
Data Points: {len(raw_data.get('metrics', []))}
Health Score: {report.performance_analysis.get('overall_health', {}).get('score', 0):.1f}/100
Security Score: {report.security_analysis.get('security_score', 100):.1f}/100
                """
                
                ax.text(0.5, 0.4, summary_info, transform=ax.transAxes,
                       fontsize=14, ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                
                # 차트들을 PDF에 추가
                for chart_file in report.charts:
                    if os.path.exists(chart_file):
                        img = plt.imread(chart_file)
                        fig, ax = plt.subplots(figsize=(8.5, 11))
                        ax.imshow(img)
                        ax.axis('off')
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close()
            
            return str(pdf_file)
        
        except Exception as e:
            self.analytics_logger.error(f"PDF report export failed: {e}")
            return None
    
    def export_excel_report(self, report: AnalyticsReport, raw_data: Dict[str, Any]) -> Optional[str]:
        """Excel 보고서 내보내기"""
        try:
            if not HAS_EXCEL:
                return None
            
            excel_file = self.reports_directory / f"{report.id}.xlsx"
            
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # 요약 시트
                summary_data = {
                    'Report Information': [
                        'Report ID', 'Title', 'Type', 'Generation Time',
                        'Period Start', 'Period End', 'Processing Time'
                    ],
                    'Values': [
                        report.id, report.title, report.report_type,
                        datetime.fromtimestamp(report.generation_time).strftime('%Y-%m-%d %H:%M:%S'),
                        datetime.fromtimestamp(report.period_start).strftime('%Y-%m-%d %H:%M:%S'),
                        datetime.fromtimestamp(report.period_end).strftime('%Y-%m-%d %H:%M:%S'),
                        f"{report.processing_time:.2f}s"
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # 원시 데이터 시트
                if raw_data.get('metrics'):
                    metrics_df = pd.DataFrame(raw_data['metrics'])
                    metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                
                # 성능 분석 시트
                if report.performance_analysis:
                    perf_data = []
                    for component, analysis in report.performance_analysis.items():
                        if isinstance(analysis, dict):
                            for metric, value in analysis.items():
                                perf_data.append({
                                    'Component': component,
                                    'Metric': metric,
                                    'Value': value
                                })
                    
                    if perf_data:
                        perf_df = pd.DataFrame(perf_data)
                        perf_df.to_excel(writer, sheet_name='Performance Analysis', index=False)
                
                # 권장사항 시트
                if report.recommendations:
                    recommendations_df = pd.DataFrame({
                        'Recommendation': report.recommendations
                    })
                    recommendations_df.to_excel(writer, sheet_name='Recommendations', index=False)
            
            return str(excel_file)
        
        except Exception as e:
            self.analytics_logger.error(f"Excel report export failed: {e}")
            return None
    
    def save_report_metadata(self, report: AnalyticsReport, report_files: List[str]):
        """보고서 메타데이터 저장"""
        try:
            conn = sqlite3.connect(self.analytics_db_path)
            cursor = conn.cursor()
            
            for file_path in report_files:
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                
                cursor.execute('''
                    INSERT OR REPLACE INTO reports (
                        id, title, report_type, generation_time, period_start,
                        period_end, file_path, file_size, processing_time, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    f"{report.id}_{os.path.splitext(os.path.basename(file_path))[1][1:]}",
                    report.title, report.report_type, report.generation_time,
                    report.period_start, report.period_end, file_path,
                    file_size, report.processing_time, 'completed'
                ))
            
            conn.commit()
            conn.close()
        
        except Exception as e:
            self.analytics_logger.error(f"Report metadata save failed: {e}")

# 전역 분석 엔진 인스턴스
quantum_analytics = QuantumAnalyticsEngine()

def main():
    """메인 함수"""
    print(f"""
{QUANTUM_THEME['quantum_purple']}╔══════════════════════════════════════════════════════════════╗
{QUANTUM_THEME['quantum_cyan']}║              QUANTUM ANALYTICS ENGINE                       ║
{QUANTUM_THEME['quantum_green']}║           🧊 엔터프라이즈급 데이터 분석                        ║
{QUANTUM_THEME['quantum_yellow']}║                                                              ║
{QUANTUM_THEME['quantum_orange']}║  📊 빅데이터 처리 및 분석                                     ║
{QUANTUM_THEME['quantum_blue']}║  📈 고급 통계 분석 및 예측                                    ║
{QUANTUM_THEME['quantum_red']}║  📋 자동 보고서 생성                                          ║
{QUANTUM_THEME['quantum_purple']}╚══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # 샘플 보고서 생성
        print(f"{QUANTUM_THEME['quantum_cyan']}📊 Generating comprehensive analytics report...")
        
        report = quantum_analytics.generate_comprehensive_report(
            report_type="daily",
            period_days=1,
            formats=['html', 'pdf']
        )
        
        print(f"{QUANTUM_THEME['quantum_green']}✅ Report generated successfully!")
        print(f"   Report ID: {report.id}")
        print(f"   Processing Time: {report.processing_time:.2f}s")
        print(f"   Charts Generated: {len(report.charts)}")
        print(f"   Recommendations: {len(report.recommendations)}")
        
        # 권장사항 출력
        if report.recommendations:
            print(f"\n{QUANTUM_THEME['quantum_yellow']}💡 Key Recommendations:")
            for i, rec in enumerate(report.recommendations[:5], 1):
                print(f"   {i}. {rec}")
        
        # 보고서 파일 위치
        print(f"\n{QUANTUM_THEME['quantum_blue']}📁 Report files saved to: {quantum_analytics.reports_directory}")
        print(f"📊 Charts saved to: {quantum_analytics.charts_directory}")
    
    except Exception as e:
        print(f"{QUANTUM_THEME['quantum_red']}❌ Analytics engine error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()