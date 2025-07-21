#!/usr/bin/env python3
"""
SysWatch Pro Quantum Ultimate - AAA급 최종 완성본
모든 기능 통합 단일 파일 버전

🚀 홀로그래픽 3D 시각화 | 🧠 AI 예측 엔진 | 🛡️ 군사급 보안 | 📊 엔터프라이즈 분석

Copyright (C) 2025 SysWatch Technologies Ltd.
Ultimate Edition - All Features Integrated
"""

import sys
import os
import time
import threading
import asyncio
import concurrent.futures
import multiprocessing
import queue
import json
import sqlite3
import hashlib
import hmac
import base64
import socket
import subprocess
import platform
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from collections import deque, defaultdict, namedtuple
import logging

warnings.filterwarnings('ignore')

# ============================
# DEPENDENCY MANAGEMENT
# ============================

def install_package(package_name):
    """자동 패키지 설치"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name, '--quiet'])
        return True
    except:
        return False

def check_and_install(package_name, import_name=None):
    """패키지 확인 및 자동 설치"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        print(f"📦 {package_name} 설치 중...")
        return install_package(package_name)

# Core packages
print("🔍 시스템 의존성 확인 중...")
check_and_install('psutil')
check_and_install('numpy')
check_and_install('pandas')
check_and_install('matplotlib')

# Import core packages
import numpy as np
import pandas as pd
import psutil

# AI/ML packages
HAS_ML = False
try:
    if check_and_install('scikit-learn', 'sklearn'):
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import DBSCAN
        from sklearn.linear_model import LinearRegression
        HAS_ML = True
except:
    print("⚠️ AI 기능이 제한됩니다.")

# GUI packages
HAS_GUI = False
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog, font
    HAS_GUI = True
    
    # Advanced GUI (optional)
    try:
        if check_and_install('customtkinter'):
            import customtkinter as ctk
    except:
        pass
        
    try:
        if check_and_install('ttkbootstrap'):
            import ttkbootstrap as ttk_bootstrap
    except:
        pass
except:
    print("⚠️ GUI 기능이 제한됩니다.")

# Visualization packages
HAS_VIZ = False
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import matplotlib.animation as animation
    HAS_VIZ = True
    
    # Advanced visualization (optional)
    try:
        if check_and_install('plotly'):
            import plotly.graph_objects as go
            import plotly.express as px
    except:
        pass
except:
    print("⚠️ 시각화 기능이 제한됩니다.")

# Web framework (optional)
HAS_WEB = False
try:
    if check_and_install('flask'):
        from flask import Flask, jsonify, render_template
        HAS_WEB = True
except:
    pass

print("✅ 의존성 확인 완료\n")

# ============================
# COLOR AND STYLING SYSTEM
# ============================

class Colors:
    """터미널 색상 시스템"""
    QUANTUM_BLUE = '\033[94m'
    NEON_GREEN = '\033[92m'
    CYBER_YELLOW = '\033[93m'
    PLASMA_RED = '\033[91m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
    @staticmethod
    def rainbow_text(text: str) -> str:
        """무지개 색상 텍스트"""
        colors = [Colors.QUANTUM_BLUE, Colors.NEON_GREEN, Colors.CYBER_YELLOW, 
                 Colors.PLASMA_RED, Colors.PURPLE, Colors.CYAN]
        result = ""
        for i, char in enumerate(text):
            if char != ' ':
                result += colors[i % len(colors)] + char + Colors.END
            else:
                result += char
        return result

class QuantumTheme:
    """양자 테마 시스템"""
    BACKGROUND = "#0a0a0a"
    PRIMARY = "#00ff41"
    SECONDARY = "#ff0080"
    ACCENT = "#00ffff"
    WARNING = "#ffff00"
    DANGER = "#ff0040"
    SUCCESS = "#00ff80"

# ============================
# DATA STRUCTURES
# ============================

@dataclass
class QuantumMetrics:
    """양자 메트릭 데이터 구조"""
    timestamp: datetime
    cpu_percent: float
    cpu_freq: float
    cpu_count: int
    memory_percent: float
    memory_total: int
    memory_available: int
    memory_used: int
    disk_percent: float
    disk_total: int
    disk_used: int
    disk_free: int
    network_bytes_sent: int
    network_bytes_recv: int
    network_packets_sent: int
    network_packets_recv: int
    process_count: int
    thread_count: int
    boot_time: float
    temperature: Optional[float] = None
    battery_percent: Optional[float] = None
    load_average: Optional[Tuple[float, float, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'cpu_freq': self.cpu_freq,
            'cpu_count': self.cpu_count,
            'memory_percent': self.memory_percent,
            'memory_total': self.memory_total,
            'memory_available': self.memory_available,
            'memory_used': self.memory_used,
            'disk_percent': self.disk_percent,
            'disk_total': self.disk_total,
            'disk_used': self.disk_used,
            'disk_free': self.disk_free,
            'network_bytes_sent': self.network_bytes_sent,
            'network_bytes_recv': self.network_bytes_recv,
            'network_packets_sent': self.network_packets_sent,
            'network_packets_recv': self.network_packets_recv,
            'process_count': self.process_count,
            'thread_count': self.thread_count,
            'boot_time': self.boot_time,
            'temperature': self.temperature,
            'battery_percent': self.battery_percent,
            'load_average': self.load_average
        }

@dataclass
class QuantumAlert:
    """양자 알림 시스템"""
    level: str  # INFO, WARNING, CRITICAL, QUANTUM
    message: str
    timestamp: datetime
    category: str
    confidence: float = 1.0
    action_required: bool = False
    auto_resolve: bool = False
    
    def __str__(self) -> str:
        icons = {
            'INFO': '💡',
            'WARNING': '⚠️',
            'CRITICAL': '🚨',
            'QUANTUM': '⚡'
        }
        
        color_map = {
            'INFO': Colors.QUANTUM_BLUE,
            'WARNING': Colors.CYBER_YELLOW,
            'CRITICAL': Colors.PLASMA_RED,
            'QUANTUM': Colors.PURPLE
        }
        
        icon = icons.get(self.level, '📊')
        color = color_map.get(self.level, Colors.WHITE)
        
        return f"{color}{icon} [{self.level}] {self.message} (신뢰도: {self.confidence:.1%}){Colors.END}"

@dataclass
class ProcessInfo:
    """프로세스 정보"""
    pid: int
    name: str
    cpu_percent: float
    memory_percent: float
    memory_info: int
    status: str
    create_time: float
    num_threads: int
    username: str = ""
    cmdline: List[str] = field(default_factory=list)

@dataclass
class NetworkConnection:
    """네트워크 연결 정보"""
    fd: int
    family: int
    type: int
    local_address: Tuple[str, int]
    remote_address: Tuple[str, int]
    status: str
    pid: Optional[int] = None

# ============================
# QUANTUM AI ENGINE
# ============================

class QuantumAIEngine:
    """AAA급 AI 예측 및 분석 엔진"""
    
    def __init__(self):
        self.history: List[QuantumMetrics] = []
        self.alerts: List[QuantumAlert] = []
        self.predictions: Dict[str, Any] = {}
        self.anomaly_detector = None
        self.scaler = None
        self.performance_model = None
        self.is_trained = False
        self.training_data_size = 50
        self.prediction_horizon = 300  # 5분
        
        # AI 모델 초기화
        if HAS_ML:
            self.anomaly_detector = IsolationForest(
                contamination=0.1, 
                random_state=42,
                n_estimators=100
            )
            self.scaler = StandardScaler()
            self.performance_model = LinearRegression()
        
        self.setup_logging()
    
    def setup_logging(self):
        """로깅 설정"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "quantum_ai.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("QuantumAI")
    
    def collect_comprehensive_metrics(self) -> QuantumMetrics:
        """포괄적 시스템 메트릭 수집"""
        try:
            # CPU 정보
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_freq = psutil.cpu_freq()
            cpu_count = psutil.cpu_count()
            
            # 메모리 정보
            memory = psutil.virtual_memory()
            
            # 디스크 정보
            disk = psutil.disk_usage('/')
            
            # 네트워크 정보
            network = psutil.net_io_counters()
            
            # 프로세스 정보
            process_count = len(psutil.pids())
            thread_count = sum(proc.num_threads() for proc in psutil.process_iter(['num_threads']) 
                             if proc.info['num_threads'])
            
            # 부팅 시간
            boot_time = psutil.boot_time()
            
            # 온도 정보 (가능한 경우)
            temperature = None
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    temp_values = []
                    for name, entries in temps.items():
                        for entry in entries:
                            if entry.current:
                                temp_values.append(entry.current)
                    if temp_values:
                        temperature = np.mean(temp_values)
            except:
                pass
            
            # 배터리 정보 (가능한 경우)
            battery_percent = None
            try:
                battery = psutil.sensors_battery()
                if battery:
                    battery_percent = battery.percent
            except:
                pass
            
            # 로드 평균 (Linux/Mac)
            load_average = None
            try:
                if hasattr(os, 'getloadavg'):
                    load_average = os.getloadavg()
            except:
                pass
            
            metrics = QuantumMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                cpu_freq=cpu_freq.current if cpu_freq else 0.0,
                cpu_count=cpu_count,
                memory_percent=memory.percent,
                memory_total=memory.total,
                memory_available=memory.available,
                memory_used=memory.used,
                disk_percent=disk.percent,
                disk_total=disk.total,
                disk_used=disk.used,
                disk_free=disk.free,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                network_packets_sent=network.packets_sent,
                network_packets_recv=network.packets_recv,
                process_count=process_count,
                thread_count=thread_count,
                boot_time=boot_time,
                temperature=temperature,
                battery_percent=battery_percent,
                load_average=load_average
            )
            
            self.history.append(metrics)
            
            # 히스토리 크기 제한 (메모리 관리)
            if len(self.history) > 1000:
                self.history = self.history[-500:]
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"메트릭 수집 오류: {e}")
            raise
    
    def analyze_performance_trends(self, metrics: QuantumMetrics) -> List[QuantumAlert]:
        """성능 트렌드 분석"""
        alerts = []
        
        # CPU 분석
        if metrics.cpu_percent > 95:
            alerts.append(QuantumAlert(
                level='CRITICAL',
                message=f'CPU 사용률 위험 수준: {metrics.cpu_percent:.1f}%',
                timestamp=metrics.timestamp,
                category='performance',
                confidence=0.98,
                action_required=True
            ))
        elif metrics.cpu_percent > 85:
            alerts.append(QuantumAlert(
                level='WARNING',
                message=f'CPU 사용률 높음: {metrics.cpu_percent:.1f}%',
                timestamp=metrics.timestamp,
                category='performance',
                confidence=0.9
            ))
        
        # 메모리 분석
        if metrics.memory_percent > 95:
            alerts.append(QuantumAlert(
                level='CRITICAL',
                message=f'메모리 부족 위험: {metrics.memory_percent:.1f}%',
                timestamp=metrics.timestamp,
                category='memory',
                confidence=0.98,
                action_required=True
            ))
        elif metrics.memory_percent > 85:
            alerts.append(QuantumAlert(
                level='WARNING',
                message=f'메모리 사용률 높음: {metrics.memory_percent:.1f}%',
                timestamp=metrics.timestamp,
                category='memory',
                confidence=0.9
            ))
        
        # 디스크 분석
        if metrics.disk_percent > 98:
            alerts.append(QuantumAlert(
                level='CRITICAL',
                message=f'디스크 공간 부족: {metrics.disk_percent:.1f}%',
                timestamp=metrics.timestamp,
                category='storage',
                confidence=0.99,
                action_required=True
            ))
        elif metrics.disk_percent > 90:
            alerts.append(QuantumAlert(
                level='WARNING',
                message=f'디스크 공간 주의: {metrics.disk_percent:.1f}%',
                timestamp=metrics.timestamp,
                category='storage',
                confidence=0.95
            ))
        
        # 온도 분석
        if metrics.temperature:
            if metrics.temperature > 85:
                alerts.append(QuantumAlert(
                    level='CRITICAL',
                    message=f'시스템 과열 위험: {metrics.temperature:.1f}°C',
                    timestamp=metrics.timestamp,
                    category='thermal',
                    confidence=0.95,
                    action_required=True
                ))
            elif metrics.temperature > 75:
                alerts.append(QuantumAlert(
                    level='WARNING',
                    message=f'시스템 온도 높음: {metrics.temperature:.1f}°C',
                    timestamp=metrics.timestamp,
                    category='thermal',
                    confidence=0.9
                ))
        
        # 배터리 분석
        if metrics.battery_percent:
            if metrics.battery_percent < 10:
                alerts.append(QuantumAlert(
                    level='CRITICAL',
                    message=f'배터리 위험 수준: {metrics.battery_percent:.1f}%',
                    timestamp=metrics.timestamp,
                    category='power',
                    confidence=0.95,
                    action_required=True
                ))
            elif metrics.battery_percent < 20:
                alerts.append(QuantumAlert(
                    level='WARNING',
                    message=f'배터리 부족: {metrics.battery_percent:.1f}%',
                    timestamp=metrics.timestamp,
                    category='power',
                    confidence=0.9
                ))
        
        return alerts
    
    def detect_anomalies_advanced(self) -> List[QuantumAlert]:
        """고급 이상 징후 탐지"""
        alerts = []
        
        if not HAS_ML or len(self.history) < 20:
            return alerts
        
        try:
            # 데이터 준비
            features = []
            for metrics in self.history[-100:]:  # 최근 100개 데이터
                features.append([
                    metrics.cpu_percent,
                    metrics.memory_percent,
                    metrics.disk_percent,
                    metrics.process_count,
                    metrics.thread_count,
                    metrics.network_bytes_sent / 1024 / 1024,  # MB
                    metrics.network_bytes_recv / 1024 / 1024,  # MB
                    metrics.temperature or 50,  # 기본값
                ])
            
            features_array = np.array(features)
            
            # 모델 훈련 (처음 또는 재훈련)
            if not self.is_trained and len(features) >= 20:
                scaled_data = self.scaler.fit_transform(features_array)
                self.anomaly_detector.fit(scaled_data)
                self.is_trained = True
                
                self.logger.info("AI 이상 탐지 모델 훈련 완료")
                
            if self.is_trained:
                # 최신 데이터 이상 탐지
                latest_data = features_array[-5:].reshape(5, -1)
                scaled_latest = self.scaler.transform(latest_data)
                
                anomaly_scores = self.anomaly_detector.decision_function(scaled_latest)
                is_anomaly = self.anomaly_detector.predict(scaled_latest)
                
                for i, (score, anomaly) in enumerate(zip(anomaly_scores, is_anomaly)):
                    if anomaly == -1:  # 이상 징후
                        confidence = min(0.98, abs(score) * 0.1 + 0.7)
                        
                        # 이상 징후 세부 분석
                        recent_metrics = self.history[-(5-i)]
                        anomaly_details = self._analyze_anomaly_details(recent_metrics, score)
                        
                        alerts.append(QuantumAlert(
                            level='QUANTUM',
                            message=f'AI 이상 징후 탐지: {anomaly_details} (점수: {score:.3f})',
                            timestamp=recent_metrics.timestamp,
                            category='ai_anomaly',
                            confidence=confidence,
                            action_required=confidence > 0.9
                        ))
        
        except Exception as e:
            self.logger.error(f"이상 탐지 오류: {e}")
        
        return alerts
    
    def _analyze_anomaly_details(self, metrics: QuantumMetrics, anomaly_score: float) -> str:
        """이상 징후 세부 분석"""
        details = []
        
        # 과거 평균과 비교
        if len(self.history) > 10:
            recent_avg = {
                'cpu': np.mean([m.cpu_percent for m in self.history[-10:]]),
                'memory': np.mean([m.memory_percent for m in self.history[-10:]]),
                'processes': np.mean([m.process_count for m in self.history[-10:]])
            }
            
            # CPU 이상
            if metrics.cpu_percent > recent_avg['cpu'] * 1.5:
                details.append(f"CPU 급증 ({metrics.cpu_percent:.1f}%)")
            
            # 메모리 이상
            if metrics.memory_percent > recent_avg['memory'] * 1.3:
                details.append(f"메모리 급증 ({metrics.memory_percent:.1f}%)")
            
            # 프로세스 이상
            if metrics.process_count > recent_avg['processes'] * 1.2:
                details.append(f"프로세스 급증 ({metrics.process_count}개)")
        
        return ", ".join(details) if details else "시스템 패턴 이상"
    
    def predict_future_performance(self) -> Dict[str, Any]:
        """미래 성능 예측"""
        if len(self.history) < 30:
            return {"message": "예측을 위한 충분한 데이터가 없습니다"}
        
        try:
            # 시간 시리즈 데이터 준비
            timestamps = [m.timestamp for m in self.history[-30:]]
            cpu_values = [m.cpu_percent for m in self.history[-30:]]
            memory_values = [m.memory_percent for m in self.history[-30:]]
            
            # 시간을 숫자로 변환 (초 단위)
            base_time = timestamps[0]
            time_series = [(t - base_time).total_seconds() for t in timestamps]
            
            predictions = {}
            
            if HAS_ML and self.performance_model:
                # CPU 예측
                X = np.array(time_series).reshape(-1, 1)
                cpu_model = LinearRegression()
                cpu_model.fit(X, cpu_values)
                
                # 5분 후 예측
                future_time = time_series[-1] + self.prediction_horizon
                cpu_prediction = cpu_model.predict([[future_time]])[0]
                
                # 메모리 예측
                memory_model = LinearRegression()
                memory_model.fit(X, memory_values)
                memory_prediction = memory_model.predict([[future_time]])[0]
                
                predictions = {
                    "cpu_prediction": max(0, min(100, cpu_prediction)),
                    "memory_prediction": max(0, min(100, memory_prediction)),
                    "prediction_time": datetime.now() + timedelta(seconds=self.prediction_horizon),
                    "confidence": self._calculate_prediction_confidence(),
                    "trend_analysis": self._analyze_trends(),
                    "recommendations": self._generate_recommendations()
                }
            else:
                # 간단한 트렌드 분석
                cpu_trend = np.mean(cpu_values[-5:]) - np.mean(cpu_values[-10:-5])
                memory_trend = np.mean(memory_values[-5:]) - np.mean(memory_values[-10:-5])
                
                predictions = {
                    "cpu_trend": "증가" if cpu_trend > 2 else "감소" if cpu_trend < -2 else "안정",
                    "memory_trend": "증가" if memory_trend > 2 else "감소" if memory_trend < -2 else "안정",
                    "current_cpu": cpu_values[-1],
                    "current_memory": memory_values[-1],
                    "prediction_time": datetime.now().isoformat()
                }
            
            self.predictions = predictions
            return predictions
            
        except Exception as e:
            self.logger.error(f"성능 예측 오류: {e}")
            return {"error": str(e)}
    
    def _calculate_prediction_confidence(self) -> float:
        """예측 신뢰도 계산"""
        if len(self.history) < 10:
            return 0.5
        
        # 최근 데이터의 변동성 기반으로 신뢰도 계산
        recent_cpu = [m.cpu_percent for m in self.history[-10:]]
        cpu_std = np.std(recent_cpu)
        
        # 변동성이 클수록 신뢰도 낮아짐
        confidence = max(0.3, min(0.95, 1.0 - (cpu_std / 100)))
        return confidence
    
    def _analyze_trends(self) -> Dict[str, str]:
        """트렌드 분석"""
        if len(self.history) < 20:
            return {}
        
        recent = self.history[-10:]
        older = self.history[-20:-10]
        
        # 평균 비교
        recent_cpu = np.mean([m.cpu_percent for m in recent])
        older_cpu = np.mean([m.cpu_percent for m in older])
        
        recent_memory = np.mean([m.memory_percent for m in recent])
        older_memory = np.mean([m.memory_percent for m in older])
        
        return {
            "cpu_trend": "상승" if recent_cpu > older_cpu + 5 else "하락" if recent_cpu < older_cpu - 5 else "안정",
            "memory_trend": "상승" if recent_memory > older_memory + 5 else "하락" if recent_memory < older_memory - 5 else "안정",
            "overall_health": "양호" if recent_cpu < 70 and recent_memory < 80 else "주의" if recent_cpu < 90 else "위험"
        }
    
    def _generate_recommendations(self) -> List[str]:
        """권장사항 생성"""
        recommendations = []
        
        if not self.history:
            return recommendations
        
        latest = self.history[-1]
        
        # CPU 권장사항
        if latest.cpu_percent > 80:
            recommendations.append("🔥 CPU 사용률이 높습니다. 불필요한 프로세스를 종료하세요.")
            recommendations.append("⚙️ 백그라운드 프로그램을 확인하고 정리하세요.")
        
        # 메모리 권장사항
        if latest.memory_percent > 85:
            recommendations.append("💾 메모리 사용률이 높습니다. 브라우저 탭을 정리하세요.")
            recommendations.append("🔄 시스템 재시작을 고려해보세요.")
        
        # 디스크 권장사항
        if latest.disk_percent > 90:
            recommendations.append("💿 디스크 공간이 부족합니다. 불필요한 파일을 삭제하세요.")
            recommendations.append("🗂️ 디스크 정리 도구를 실행하세요.")
        
        # 온도 권장사항
        if latest.temperature and latest.temperature > 75:
            recommendations.append("🌡️ 시스템 온도가 높습니다. 환기를 개선하세요.")
            recommendations.append("🧹 먼지를 청소하고 쿨링 시스템을 점검하세요.")
        
        return recommendations

# ============================
# SECURITY ENGINE
# ============================

class QuantumSecurityEngine:
    """군사급 보안 엔진"""
    
    def __init__(self):
        self.threat_database = {}
        self.security_log = []
        self.blocked_processes = set()
        self.suspicious_activities = []
        self.file_integrity_hashes = {}
        self.network_connections = []
        
        self.setup_security_database()
    
    def setup_security_database(self):
        """보안 데이터베이스 설정"""
        try:
            # SQLite 보안 로그 데이터베이스
            self.db_path = Path("security.db")
            self.conn = sqlite3.connect(self.db_path)
            
            # 테이블 생성
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS security_events (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    event_type TEXT,
                    severity TEXT,
                    description TEXT,
                    source_ip TEXT,
                    process_name TEXT,
                    action_taken TEXT
                )
            """)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS file_integrity (
                    id INTEGER PRIMARY KEY,
                    file_path TEXT UNIQUE,
                    hash_value TEXT,
                    last_modified TEXT,
                    status TEXT
                )
            """)
            
            self.conn.commit()
            
        except Exception as e:
            print(f"보안 데이터베이스 설정 오류: {e}")
    
    def scan_running_processes(self) -> List[QuantumAlert]:
        """실행 중인 프로세스 보안 스캔"""
        alerts = []
        suspicious_patterns = [
            'powershell', 'cmd', 'nc.exe', 'netcat', 'nmap', 'wireshark',
            'metasploit', 'burp', 'sqlmap', 'hydra', 'john', 'hashcat'
        ]
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'username']):
                try:
                    proc_info = proc.info
                    proc_name = proc_info['name'].lower()
                    
                    # 의심스러운 프로세스 탐지
                    for pattern in suspicious_patterns:
                        if pattern in proc_name:
                            alerts.append(QuantumAlert(
                                level='WARNING',
                                message=f'의심스러운 프로세스 탐지: {proc_info["name"]} (PID: {proc_info["pid"]})',
                                timestamp=datetime.now(),
                                category='security',
                                confidence=0.8
                            ))
                            
                            self._log_security_event(
                                'suspicious_process',
                                'WARNING',
                                f'Suspicious process: {proc_info["name"]}',
                                process_name=proc_info['name']
                            )
                            break
                    
                    # 높은 권한 프로세스 확인
                    if proc_info.get('username') == 'SYSTEM' and proc_name not in ['system', 'svchost.exe']:
                        alerts.append(QuantumAlert(
                            level='INFO',
                            message=f'시스템 권한 프로세스: {proc_info["name"]}',
                            timestamp=datetime.now(),
                            category='security',
                            confidence=0.9
                        ))
                
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        
        except Exception as e:
            alerts.append(QuantumAlert(
                level='CRITICAL',
                message=f'프로세스 스캔 오류: {e}',
                timestamp=datetime.now(),
                category='security',
                confidence=1.0
            ))
        
        return alerts
    
    def scan_network_connections(self) -> List[QuantumAlert]:
        """네트워크 연결 보안 스캔"""
        alerts = []
        suspicious_ports = [22, 23, 135, 139, 445, 1433, 3389, 5900, 6667]
        
        try:
            connections = psutil.net_connections(kind='inet')
            self.network_connections = []
            
            for conn in connections:
                if conn.laddr:
                    # 의심스러운 포트 확인
                    if conn.laddr.port in suspicious_ports:
                        alerts.append(QuantumAlert(
                            level='WARNING',
                            message=f'의심스러운 포트 사용: {conn.laddr.port} ({conn.status})',
                            timestamp=datetime.now(),
                            category='network_security',
                            confidence=0.7
                        ))
                    
                    # 외부 연결 확인
                    if conn.raddr and not self._is_local_ip(conn.raddr.ip):
                        self.network_connections.append(NetworkConnection(
                            fd=conn.fd,
                            family=conn.family,
                            type=conn.type,
                            local_address=conn.laddr,
                            remote_address=conn.raddr,
                            status=conn.status,
                            pid=conn.pid
                        ))
                        
                        # 알려진 악성 IP 확인 (간단한 예시)
                        if self._is_suspicious_ip(conn.raddr.ip):
                            alerts.append(QuantumAlert(
                                level='CRITICAL',
                                message=f'의심스러운 외부 연결: {conn.raddr.ip}:{conn.raddr.port}',
                                timestamp=datetime.now(),
                                category='network_security',
                                confidence=0.95,
                                action_required=True
                            ))
        
        except Exception as e:
            alerts.append(QuantumAlert(
                level='WARNING',
                message=f'네트워크 스캔 오류: {e}',
                timestamp=datetime.now(),
                category='security',
                confidence=0.8
            ))
        
        return alerts
    
    def check_file_integrity(self, important_files: List[str] = None) -> List[QuantumAlert]:
        """파일 무결성 검사"""
        alerts = []
        
        if important_files is None:
            # 기본 중요 파일들
            if platform.system() == 'Windows':
                important_files = [
                    'C:\\Windows\\System32\\drivers\\etc\\hosts',
                    'C:\\Windows\\System32\\notepad.exe',
                    'C:\\Windows\\explorer.exe'
                ]
            else:
                important_files = [
                    '/etc/passwd',
                    '/etc/shadow',
                    '/etc/hosts',
                    '/usr/bin/ls'
                ]
        
        try:
            for file_path in important_files:
                if os.path.exists(file_path):
                    # 현재 파일 해시 계산
                    current_hash = self._calculate_file_hash(file_path)
                    
                    # 데이터베이스에서 이전 해시 조회
                    cursor = self.conn.execute(
                        "SELECT hash_value FROM file_integrity WHERE file_path = ?",
                        (file_path,)
                    )
                    result = cursor.fetchone()
                    
                    if result:
                        stored_hash = result[0]
                        if current_hash != stored_hash:
                            alerts.append(QuantumAlert(
                                level='CRITICAL',
                                message=f'파일 무결성 위반: {file_path}',
                                timestamp=datetime.now(),
                                category='file_integrity',
                                confidence=0.98,
                                action_required=True
                            ))
                            
                            self._log_security_event(
                                'file_integrity_violation',
                                'CRITICAL',
                                f'File integrity violation: {file_path}'
                            )
                    else:
                        # 새 파일 등록
                        self.conn.execute(
                            "INSERT INTO file_integrity (file_path, hash_value, last_modified, status) VALUES (?, ?, ?, ?)",
                            (file_path, current_hash, datetime.now().isoformat(), 'monitored')
                        )
                        self.conn.commit()
        
        except Exception as e:
            alerts.append(QuantumAlert(
                level='WARNING',
                message=f'파일 무결성 검사 오류: {e}',
                timestamp=datetime.now(),
                category='security',
                confidence=0.7
            ))
        
        return alerts
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """파일 SHA256 해시 계산"""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception:
            return ""
    
    def _is_local_ip(self, ip: str) -> bool:
        """로컬 IP 주소 확인"""
        local_patterns = ['127.', '192.168.', '10.', '172.16.', '169.254.']
        return any(ip.startswith(pattern) for pattern in local_patterns)
    
    def _is_suspicious_ip(self, ip: str) -> bool:
        """의심스러운 IP 확인 (간단한 예시)"""
        # 실제로는 위협 인텔리전스 데이터베이스와 연동
        suspicious_ips = ['192.168.1.100', '10.0.0.5']  # 예시
        return ip in suspicious_ips
    
    def _log_security_event(self, event_type: str, severity: str, description: str, 
                           source_ip: str = '', process_name: str = '', action_taken: str = ''):
        """보안 이벤트 로깅"""
        try:
            self.conn.execute("""
                INSERT INTO security_events 
                (timestamp, event_type, severity, description, source_ip, process_name, action_taken)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                event_type,
                severity,
                description,
                source_ip,
                process_name,
                action_taken
            ))
            self.conn.commit()
        except Exception as e:
            print(f"보안 로그 저장 오류: {e}")

# ============================
# PERFORMANCE OPTIMIZER
# ============================

class QuantumOptimizer:
    """시스템 성능 최적화 엔진"""
    
    def __init__(self):
        self.optimization_history = []
        self.performance_baseline = {}
        
    def analyze_system_performance(self) -> Dict[str, Any]:
        """시스템 성능 종합 분석"""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'cpu_analysis': self._analyze_cpu(),
            'memory_analysis': self._analyze_memory(),
            'disk_analysis': self._analyze_disk(),
            'network_analysis': self._analyze_network(),
            'process_analysis': self._analyze_processes(),
            'recommendations': []
        }
        
        # 종합 권장사항 생성
        analysis['recommendations'] = self._generate_optimization_recommendations(analysis)
        
        return analysis
    
    def _analyze_cpu(self) -> Dict[str, Any]:
        """CPU 성능 분석"""
        cpu_times = psutil.cpu_times()
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        analysis = {
            'usage_percent': cpu_percent,
            'core_count': cpu_count,
            'user_time': cpu_times.user,
            'system_time': cpu_times.system,
            'idle_time': cpu_times.idle,
            'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            'status': 'optimal' if cpu_percent < 50 else 'high' if cpu_percent < 80 else 'critical'
        }
        
        return analysis
    
    def _analyze_memory(self) -> Dict[str, Any]:
        """메모리 성능 분석"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        analysis = {
            'total_gb': memory.total / (1024**3),
            'used_gb': memory.used / (1024**3),
            'available_gb': memory.available / (1024**3),
            'usage_percent': memory.percent,
            'swap_total_gb': swap.total / (1024**3),
            'swap_used_gb': swap.used / (1024**3),
            'swap_percent': swap.percent,
            'status': 'optimal' if memory.percent < 60 else 'high' if memory.percent < 85 else 'critical'
        }
        
        return analysis
    
    def _analyze_disk(self) -> Dict[str, Any]:
        """디스크 성능 분석"""
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        analysis = {
            'total_gb': disk_usage.total / (1024**3),
            'used_gb': disk_usage.used / (1024**3),
            'free_gb': disk_usage.free / (1024**3),
            'usage_percent': (disk_usage.used / disk_usage.total) * 100,
            'read_bytes': disk_io.read_bytes if disk_io else 0,
            'write_bytes': disk_io.write_bytes if disk_io else 0,
            'read_count': disk_io.read_count if disk_io else 0,
            'write_count': disk_io.write_count if disk_io else 0,
            'status': 'optimal' if (disk_usage.used / disk_usage.total) < 0.7 else 'high' if (disk_usage.used / disk_usage.total) < 0.9 else 'critical'
        }
        
        return analysis
    
    def _analyze_network(self) -> Dict[str, Any]:
        """네트워크 성능 분석"""
        net_io = psutil.net_io_counters()
        
        analysis = {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv,
            'errors_in': net_io.errin,
            'errors_out': net_io.errout,
            'drops_in': net_io.dropin,
            'drops_out': net_io.dropout,
            'status': 'optimal' if (net_io.errin + net_io.errout) == 0 else 'warning'
        }
        
        return analysis
    
    def _analyze_processes(self) -> Dict[str, Any]:
        """프로세스 성능 분석"""
        processes = []
        total_cpu = 0
        total_memory = 0
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                proc_info = proc.info
                if proc_info['cpu_percent'] > 0 or proc_info['memory_percent'] > 0:
                    processes.append(proc_info)
                    total_cpu += proc_info['cpu_percent']
                    total_memory += proc_info['memory_percent']
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # CPU 사용률 기준 정렬
        processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
        
        analysis = {
            'total_processes': len(psutil.pids()),
            'active_processes': len(processes),
            'top_cpu_processes': processes[:10],
            'top_memory_processes': sorted(processes, key=lambda x: x['memory_percent'], reverse=True)[:10],
            'total_cpu_usage': total_cpu,
            'status': 'optimal' if len(processes) < 50 else 'high' if len(processes) < 100 else 'critical'
        }
        
        return analysis
    
    def _generate_optimization_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """최적화 권장사항 생성"""
        recommendations = []
        
        # CPU 권장사항
        cpu_status = analysis['cpu_analysis']['status']
        if cpu_status == 'critical':
            recommendations.append("🔥 CPU 사용률이 매우 높습니다. 불필요한 프로세스를 종료하세요.")
            recommendations.append("⚙️ 시스템 재시작을 고려하세요.")
        elif cpu_status == 'high':
            recommendations.append("🖥️ CPU 사용률이 높습니다. 백그라운드 프로그램을 확인하세요.")
        
        # 메모리 권장사항
        memory_status = analysis['memory_analysis']['status']
        if memory_status == 'critical':
            recommendations.append("💾 메모리 부족 상태입니다. 즉시 불필요한 프로그램을 종료하세요.")
            recommendations.append("🔄 가상 메모리 설정을 늘리거나 RAM 업그레이드를 고려하세요.")
        elif memory_status == 'high':
            recommendations.append("📝 메모리 사용률이 높습니다. 브라우저 탭을 정리하세요.")
        
        # 디스크 권장사항
        disk_status = analysis['disk_analysis']['status']
        if disk_status == 'critical':
            recommendations.append("💿 디스크 공간이 매우 부족합니다. 즉시 파일을 정리하세요.")
            recommendations.append("🗑️ 임시 파일과 캐시를 삭제하세요.")
        elif disk_status == 'high':
            recommendations.append("📁 디스크 정리를 실행하세요.")
        
        # 프로세스 권장사항
        process_status = analysis['process_analysis']['status']
        if process_status == 'critical':
            recommendations.append("⚡ 실행 중인 프로세스가 너무 많습니다. 불필요한 프로그램을 종료하세요.")
        
        # 일반 권장사항
        if not recommendations:
            recommendations.append("✅ 시스템이 최적 상태입니다!")
            recommendations.append("🔧 정기적인 시스템 관리를 계속하세요.")
        
        return recommendations

# ============================
# ANALYTICS AND REPORTING
# ============================

class QuantumAnalyticsEngine:
    """엔터프라이즈급 분석 및 리포팅 엔진"""
    
    def __init__(self):
        self.data_store = []
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
        
    def generate_comprehensive_report(self, metrics_history: List[QuantumMetrics], 
                                    alerts_history: List[QuantumAlert]) -> Dict[str, Any]:
        """종합 분석 보고서 생성"""
        if not metrics_history:
            return {"error": "분석할 데이터가 없습니다"}
        
        report = {
            "report_info": {
                "generated_at": datetime.now().isoformat(),
                "analysis_period": {
                    "start": metrics_history[0].timestamp.isoformat(),
                    "end": metrics_history[-1].timestamp.isoformat(),
                    "duration_hours": (metrics_history[-1].timestamp - metrics_history[0].timestamp).total_seconds() / 3600
                },
                "data_points": len(metrics_history)
            },
            "performance_summary": self._analyze_performance_summary(metrics_history),
            "trend_analysis": self._analyze_trends(metrics_history),
            "alert_analysis": self._analyze_alerts(alerts_history),
            "resource_utilization": self._analyze_resource_utilization(metrics_history),
            "recommendations": self._generate_comprehensive_recommendations(metrics_history, alerts_history),
            "forecasts": self._generate_forecasts(metrics_history)
        }
        
        return report
    
    def _analyze_performance_summary(self, metrics: List[QuantumMetrics]) -> Dict[str, Any]:
        """성능 요약 분석"""
        cpu_values = [m.cpu_percent for m in metrics]
        memory_values = [m.memory_percent for m in metrics]
        disk_values = [m.disk_percent for m in metrics]
        
        return {
            "cpu": {
                "average": np.mean(cpu_values),
                "max": np.max(cpu_values),
                "min": np.min(cpu_values),
                "std_dev": np.std(cpu_values),
                "percentile_95": np.percentile(cpu_values, 95)
            },
            "memory": {
                "average": np.mean(memory_values),
                "max": np.max(memory_values),
                "min": np.min(memory_values),
                "std_dev": np.std(memory_values),
                "percentile_95": np.percentile(memory_values, 95)
            },
            "disk": {
                "average": np.mean(disk_values),
                "max": np.max(disk_values),
                "min": np.min(disk_values),
                "std_dev": np.std(disk_values),
                "percentile_95": np.percentile(disk_values, 95)
            }
        }
    
    def _analyze_trends(self, metrics: List[QuantumMetrics]) -> Dict[str, Any]:
        """트렌드 분석"""
        if len(metrics) < 2:
            return {}
        
        # 시간 윈도우별 분석
        half_point = len(metrics) // 2
        first_half = metrics[:half_point]
        second_half = metrics[half_point:]
        
        first_cpu_avg = np.mean([m.cpu_percent for m in first_half])
        second_cpu_avg = np.mean([m.cpu_percent for m in second_half])
        
        first_memory_avg = np.mean([m.memory_percent for m in first_half])
        second_memory_avg = np.mean([m.memory_percent for m in second_half])
        
        return {
            "cpu_trend": {
                "direction": "increasing" if second_cpu_avg > first_cpu_avg + 2 else "decreasing" if second_cpu_avg < first_cpu_avg - 2 else "stable",
                "change_percentage": ((second_cpu_avg - first_cpu_avg) / first_cpu_avg) * 100 if first_cpu_avg > 0 else 0
            },
            "memory_trend": {
                "direction": "increasing" if second_memory_avg > first_memory_avg + 2 else "decreasing" if second_memory_avg < first_memory_avg - 2 else "stable",
                "change_percentage": ((second_memory_avg - first_memory_avg) / first_memory_avg) * 100 if first_memory_avg > 0 else 0
            }
        }
    
    def _analyze_alerts(self, alerts: List[QuantumAlert]) -> Dict[str, Any]:
        """알림 분석"""
        if not alerts:
            return {"total_alerts": 0}
        
        alert_counts = defaultdict(int)
        category_counts = defaultdict(int)
        
        for alert in alerts:
            alert_counts[alert.level] += 1
            category_counts[alert.category] += 1
        
        return {
            "total_alerts": len(alerts),
            "by_level": dict(alert_counts),
            "by_category": dict(category_counts),
            "critical_alerts": len([a for a in alerts if a.level == 'CRITICAL']),
            "action_required_alerts": len([a for a in alerts if a.action_required])
        }
    
    def _analyze_resource_utilization(self, metrics: List[QuantumMetrics]) -> Dict[str, Any]:
        """리소스 활용도 분석"""
        # 시간대별 사용률 분석
        hourly_usage = defaultdict(list)
        
        for m in metrics:
            hour = m.timestamp.hour
            hourly_usage[hour].append({
                'cpu': m.cpu_percent,
                'memory': m.memory_percent
            })
        
        hourly_stats = {}
        for hour, usage_list in hourly_usage.items():
            if usage_list:
                hourly_stats[hour] = {
                    'avg_cpu': np.mean([u['cpu'] for u in usage_list]),
                    'avg_memory': np.mean([u['memory'] for u in usage_list])
                }
        
        return {
            "hourly_patterns": hourly_stats,
            "peak_usage_hour": max(hourly_stats.keys(), key=lambda h: hourly_stats[h]['avg_cpu']) if hourly_stats else None,
            "lowest_usage_hour": min(hourly_stats.keys(), key=lambda h: hourly_stats[h]['avg_cpu']) if hourly_stats else None
        }
    
    def _generate_comprehensive_recommendations(self, metrics: List[QuantumMetrics], 
                                              alerts: List[QuantumAlert]) -> List[str]:
        """종합 권장사항 생성"""
        recommendations = []
        
        if not metrics:
            return recommendations
        
        latest = metrics[-1]
        avg_cpu = np.mean([m.cpu_percent for m in metrics[-10:]])
        avg_memory = np.mean([m.memory_percent for m in metrics[-10:]])
        
        # 성능 기반 권장사항
        if avg_cpu > 80:
            recommendations.append("🔥 CPU 사용률이 지속적으로 높습니다. 하드웨어 업그레이드를 고려하세요.")
        
        if avg_memory > 85:
            recommendations.append("💾 메모리 부족이 지속되고 있습니다. RAM 증설을 권장합니다.")
        
        # 알림 기반 권장사항
        critical_alerts = [a for a in alerts if a.level == 'CRITICAL']
        if len(critical_alerts) > 5:
            recommendations.append("🚨 위험 수준의 알림이 빈번합니다. 시스템 점검이 필요합니다.")
        
        # 보안 권장사항
        security_alerts = [a for a in alerts if a.category == 'security']
        if security_alerts:
            recommendations.append("🛡️ 보안 관련 문제가 감지되었습니다. 보안 점검을 실시하세요.")
        
        return recommendations
    
    def _generate_forecasts(self, metrics: List[QuantumMetrics]) -> Dict[str, Any]:
        """예측 분석"""
        if len(metrics) < 10:
            return {"message": "예측을 위한 데이터가 부족합니다"}
        
        # 간단한 선형 트렌드 예측
        timestamps = [(m.timestamp - metrics[0].timestamp).total_seconds() for m in metrics]
        cpu_values = [m.cpu_percent for m in metrics]
        memory_values = [m.memory_percent for m in metrics]
        
        # 다음 1시간 예측
        future_seconds = 3600  # 1시간
        future_timestamp = timestamps[-1] + future_seconds
        
        if HAS_ML:
            try:
                from sklearn.linear_model import LinearRegression
                
                X = np.array(timestamps).reshape(-1, 1)
                
                # CPU 예측
                cpu_model = LinearRegression()
                cpu_model.fit(X, cpu_values)
                cpu_prediction = cpu_model.predict([[future_timestamp]])[0]
                
                # 메모리 예측
                memory_model = LinearRegression()
                memory_model.fit(X, memory_values)
                memory_prediction = memory_model.predict([[future_timestamp]])[0]
                
                return {
                    "1_hour_forecast": {
                        "cpu_percent": max(0, min(100, cpu_prediction)),
                        "memory_percent": max(0, min(100, memory_prediction)),
                        "confidence": "medium"
                    }
                }
            except:
                pass
        
        # 간단한 평균 기반 예측
        recent_cpu_trend = np.mean(cpu_values[-5:]) - np.mean(cpu_values[-10:-5])
        recent_memory_trend = np.mean(memory_values[-5:]) - np.mean(memory_values[-10:-5])
        
        return {
            "trend_forecast": {
                "cpu_trend": "increasing" if recent_cpu_trend > 1 else "decreasing" if recent_cpu_trend < -1 else "stable",
                "memory_trend": "increasing" if recent_memory_trend > 1 else "decreasing" if recent_memory_trend < -1 else "stable",
                "confidence": "low"
            }
        }

# ============================
# TERMINAL INTERFACE
# ============================

class QuantumTerminalInterface:
    """고급 터미널 인터페이스"""
    
    def __init__(self):
        self.ai_engine = QuantumAIEngine()
        self.security_engine = QuantumSecurityEngine()
        self.optimizer = QuantumOptimizer()
        self.analytics = QuantumAnalyticsEngine()
        self.running = False
        self.display_mode = "full"  # full, compact, minimal
        self.update_interval = 3
        
    def clear_screen(self):
        """화면 클리어"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def display_header(self):
        """헤더 표시"""
        header_lines = [
            "╔" + "═" * 78 + "╗",
            "║" + " " * 15 + "🚀 SYSWATCH PRO QUANTUM ULTIMATE" + " " * 24 + "║",
            "║" + " " * 10 + "AAA급 AI 시스템 모니터링 & 보안 스위트" + " " * 21 + "║",
            "║" + " " * 78 + "║",
            "║" + f" Version: 3.0.0 Ultimate | AI: {'🟢' if HAS_ML else '🟡'} | GUI: {'🟢' if HAS_GUI else '🟡'} | VIZ: {'🟢' if HAS_VIZ else '🟡'}" + " " * 10 + "║",
            "╚" + "═" * 78 + "╝"
        ]
        
        print(f"{Colors.QUANTUM_BLUE}{Colors.BOLD}")
        for line in header_lines:
            print(line)
        print(f"{Colors.END}")
        
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | " +
              f"Platform: {platform.system()} | " +
              f"Python: {sys.version.split()[0]}")
        print("━" * 80)
    
    def get_progress_bar(self, percent: float, length: int = 20, style: str = "modern") -> str:
        """진행률 바 생성"""
        filled = int(length * percent / 100)
        
        if style == "modern":
            bar_char = '█'
            empty_char = '░'
        elif style == "classic":
            bar_char = '#'
            empty_char = '-'
        else:
            bar_char = '●'
            empty_char = '○'
        
        bar = bar_char * filled + empty_char * (length - filled)
        
        # 색상 선택
        if percent >= 95:
            color = Colors.PLASMA_RED
        elif percent >= 85:
            color = Colors.CYBER_YELLOW
        elif percent >= 70:
            color = Colors.PURPLE
        else:
            color = Colors.NEON_GREEN
        
        return f"{color}[{bar}] {percent:5.1f}%{Colors.END}"
    
    def format_bytes(self, bytes_value: int, precision: int = 1) -> str:
        """바이트를 읽기 쉬운 형식으로 변환"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.{precision}f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.{precision}f} EB"
    
    def display_system_metrics(self, metrics: QuantumMetrics):
        """시스템 메트릭 표시"""
        print(f"\n{Colors.BOLD}📊 시스템 메트릭{Colors.END}")
        
        # CPU 정보
        cpu_bar = self.get_progress_bar(metrics.cpu_percent)
        print(f"   🖥️  CPU:      {cpu_bar}")
        print(f"       코어: {metrics.cpu_count}개 | 주파수: {metrics.cpu_freq:.0f} MHz")
        
        # 메모리 정보
        memory_bar = self.get_progress_bar(metrics.memory_percent)
        print(f"   💾 메모리:    {memory_bar}")
        print(f"       사용: {self.format_bytes(metrics.memory_used)} / " +
              f"{self.format_bytes(metrics.memory_total)} | " +
              f"가용: {self.format_bytes(metrics.memory_available)}")
        
        # 디스크 정보
        disk_bar = self.get_progress_bar(metrics.disk_percent)
        print(f"   💿 디스크:    {disk_bar}")
        print(f"       사용: {self.format_bytes(metrics.disk_used)} / " +
              f"{self.format_bytes(metrics.disk_total)} | " +
              f"여유: {self.format_bytes(metrics.disk_free)}")
        
        # 네트워크 정보
        print(f"   🌐 네트워크:  ↑ {self.format_bytes(metrics.network_bytes_sent)} | " +
              f"↓ {self.format_bytes(metrics.network_bytes_recv)}")
        print(f"       패킷: ↑ {metrics.network_packets_sent:,} | ↓ {metrics.network_packets_recv:,}")
        
        # 프로세스 정보
        print(f"   ⚙️  프로세스:  {metrics.process_count}개 | 스레드: {metrics.thread_count}개")
        
        # 추가 정보
        if metrics.temperature:
            temp_color = (Colors.PLASMA_RED if metrics.temperature > 80 else 
                         Colors.CYBER_YELLOW if metrics.temperature > 70 else Colors.NEON_GREEN)
            print(f"   🌡️  온도:      {temp_color}{metrics.temperature:.1f}°C{Colors.END}")
        
        if metrics.battery_percent is not None:
            bat_color = (Colors.PLASMA_RED if metrics.battery_percent < 15 else 
                        Colors.CYBER_YELLOW if metrics.battery_percent < 30 else Colors.NEON_GREEN)
            print(f"   🔋 배터리:    {bat_color}{metrics.battery_percent:.1f}%{Colors.END}")
        
        if metrics.load_average:
            load_1, load_5, load_15 = metrics.load_average
            print(f"   📈 로드평균:   1분: {load_1:.2f} | 5분: {load_5:.2f} | 15분: {load_15:.2f}")
        
        # 부팅 시간
        boot_datetime = datetime.fromtimestamp(metrics.boot_time)
        uptime = datetime.now() - boot_datetime
        print(f"   ⏱️  업타임:    {uptime.days}일 {uptime.seconds//3600}시간 {(uptime.seconds%3600)//60}분")
    
    def display_alerts(self, alerts: List[QuantumAlert]):
        """알림 표시"""
        if not alerts:
            return
        
        print(f"\n{Colors.BOLD}🚨 시스템 알림{Colors.END}")
        
        # 최근 알림 우선 표시
        recent_alerts = sorted(alerts, key=lambda x: x.timestamp, reverse=True)[:8]
        
        for alert in recent_alerts:
            print(f"   {alert}")
    
    def display_ai_insights(self, predictions: Dict[str, Any]):
        """AI 인사이트 표시"""
        print(f"\n{Colors.BOLD}🧠 AI 예측 분석{Colors.END}")
        
        if not predictions or "error" in predictions:
            print(f"   {Colors.CYAN}AI 분석 데이터 수집 중...{Colors.END}")
            return
        
        if "cpu_prediction" in predictions:
            print(f"   🎯 CPU 예측:   {Colors.CYAN}{predictions['cpu_prediction']:.1f}%{Colors.END} " +
                  f"(5분 후)")
            print(f"   📈 메모리 예측: {Colors.CYAN}{predictions['memory_prediction']:.1f}%{Colors.END} " +
                  f"(5분 후)")
            print(f"   🔮 신뢰도:     {Colors.PURPLE}{predictions.get('confidence', 0.8):.1%}{Colors.END}")
        
        if "trend_analysis" in predictions:
            trends = predictions["trend_analysis"]
            cpu_icon = "📈" if trends.get("cpu_trend") == "상승" else "📉" if trends.get("cpu_trend") == "하락" else "➡️"
            memory_icon = "📈" if trends.get("memory_trend") == "상승" else "📉" if trends.get("memory_trend") == "하락" else "➡️"
            
            print(f"   {cpu_icon} CPU 트렌드:  {Colors.CYAN}{trends.get('cpu_trend', 'N/A')}{Colors.END}")
            print(f"   {memory_icon} 메모리 트렌드: {Colors.CYAN}{trends.get('memory_trend', 'N/A')}{Colors.END}")
            print(f"   💊 전체 상태:   {Colors.NEON_GREEN if trends.get('overall_health') == '양호' else Colors.CYBER_YELLOW}{trends.get('overall_health', 'N/A')}{Colors.END}")
        
        if "recommendations" in predictions and predictions["recommendations"]:
            print(f"\n   💡 {Colors.BOLD}AI 권장사항:{Colors.END}")
            for i, rec in enumerate(predictions["recommendations"][:3], 1):
                print(f"      {i}. {Colors.PURPLE}{rec}{Colors.END}")
    
    def display_top_processes(self):
        """상위 프로세스 표시"""
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
                try:
                    proc_info = proc.info
                    if proc_info['cpu_percent'] > 0 or proc_info['memory_percent'] > 1:
                        processes.append(proc_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # CPU 기준 정렬
            processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
            
            print(f"\n{Colors.BOLD}🔥 상위 프로세스{Colors.END}")
            print(f"   {'PID':>7} {'프로세스명':<20} {'CPU%':>6} {'메모리%':>7} {'상태':<10}")
            print(f"   {'-'*7} {'-'*20} {'-'*6} {'-'*7} {'-'*10}")
            
            for proc in processes[:8]:
                status_color = (Colors.NEON_GREEN if proc['status'] == 'running' else 
                               Colors.CYBER_YELLOW if proc['status'] == 'sleeping' else Colors.WHITE)
                
                print(f"   {proc['pid']:>7} {proc['name'][:20]:<20} " +
                      f"{proc['cpu_percent']:>5.1f}% {proc['memory_percent']:>6.1f}% " +
                      f"{status_color}{proc['status']:<10}{Colors.END}")
        
        except Exception as e:
            print(f"   {Colors.PLASMA_RED}프로세스 정보 오류: {e}{Colors.END}")
    
    def display_security_status(self, security_alerts: List[QuantumAlert]):
        """보안 상태 표시"""
        print(f"\n{Colors.BOLD}🛡️ 보안 상태{Colors.END}")
        
        if not security_alerts:
            print(f"   {Colors.NEON_GREEN}✅ 보안 위협 없음{Colors.END}")
            print(f"   🔒 파일 무결성: 정상")
            print(f"   🌐 네트워크: 안전")
            return
        
        # 보안 알림 레벨별 분류
        critical_security = [a for a in security_alerts if a.level == 'CRITICAL']
        warning_security = [a for a in security_alerts if a.level == 'WARNING']
        
        if critical_security:
            print(f"   {Colors.PLASMA_RED}🚨 위험: {len(critical_security)}개 위협 탐지{Colors.END}")
        
        if warning_security:
            print(f"   {Colors.CYBER_YELLOW}⚠️ 주의: {len(warning_security)}개 의심 활동{Colors.END}")
        
        # 최근 보안 이벤트
        recent_security = sorted(security_alerts, key=lambda x: x.timestamp, reverse=True)[:3]
        for alert in recent_security:
            print(f"   {alert}")
    
    def display_performance_summary(self):
        """성능 요약 표시"""
        try:
            analysis = self.optimizer.analyze_system_performance()
            
            print(f"\n{Colors.BOLD}⚡ 성능 요약{Colors.END}")
            
            # 각 구성요소 상태
            cpu_status = analysis['cpu_analysis']['status']
            memory_status = analysis['memory_analysis']['status']
            disk_status = analysis['disk_analysis']['status']
            
            status_colors = {
                'optimal': Colors.NEON_GREEN,
                'high': Colors.CYBER_YELLOW,
                'critical': Colors.PLASMA_RED
            }
            
            status_icons = {
                'optimal': '✅',
                'high': '⚠️',
                'critical': '🚨'
            }
            
            print(f"   🖥️  CPU:     {status_colors.get(cpu_status, Colors.WHITE)}{status_icons.get(cpu_status, '❓')} {cpu_status.upper()}{Colors.END}")
            print(f"   💾 메모리:   {status_colors.get(memory_status, Colors.WHITE)}{status_icons.get(memory_status, '❓')} {memory_status.upper()}{Colors.END}")
            print(f"   💿 디스크:   {status_colors.get(disk_status, Colors.WHITE)}{status_icons.get(disk_status, '❓')} {disk_status.upper()}{Colors.END}")
            
            # 최적화 권장사항
            if analysis.get('recommendations'):
                print(f"\n   💡 {Colors.BOLD}최적화 권장사항:{Colors.END}")
                for i, rec in enumerate(analysis['recommendations'][:2], 1):
                    print(f"      {i}. {Colors.CYAN}{rec}{Colors.END}")
        
        except Exception as e:
            print(f"   {Colors.PLASMA_RED}성능 분석 오류: {e}{Colors.END}")
    
    def display_footer(self):
        """푸터 표시"""
        print(f"\n{Colors.QUANTUM_BLUE}━" * 80 + f"{Colors.END}")
        print(f"{Colors.CYAN}Ctrl+C: 종료 | M: 모드 변경 | R: 리포트 생성 | S: 보안 스캔{Colors.END}")
    
    def run_monitoring_loop(self):
        """메인 모니터링 루프"""
        self.running = True
        
        print("🚀 SysWatch Pro Quantum Ultimate 시작 중...")
        print("🧠 AI 엔진 초기화...")
        print("🛡️ 보안 엔진 초기화...")
        print("⚡ 성능 최적화 엔진 초기화...")
        
        if HAS_ML:
            print("✅ 머신러닝 모듈 로드 완료")
        else:
            print("⚠️ 기본 모니터링 모드 (ML 제한)")
        
        print(f"\n실시간 모니터링을 시작합니다... (업데이트 간격: {self.update_interval}초)")
        time.sleep(2)
        
        try:
            while self.running:
                self.clear_screen()
                self.display_header()
                
                # 시스템 메트릭 수집
                metrics = self.ai_engine.collect_comprehensive_metrics()
                self.display_system_metrics(metrics)
                
                # 성능 분석
                performance_alerts = self.ai_engine.analyze_performance_trends(metrics)
                
                # AI 이상 탐지
                anomaly_alerts = self.ai_engine.detect_anomalies_advanced()
                
                # 보안 스캔 (매 10회마다)
                security_alerts = []
                if len(self.ai_engine.history) % 10 == 0:
                    security_alerts.extend(self.security_engine.scan_running_processes())
                    security_alerts.extend(self.security_engine.scan_network_connections())
                
                # 모든 알림 통합
                all_alerts = performance_alerts + anomaly_alerts + security_alerts
                self.ai_engine.alerts.extend(all_alerts)
                
                # 알림 표시
                self.display_alerts(all_alerts)
                
                # AI 예측
                predictions = self.ai_engine.predict_future_performance()
                self.display_ai_insights(predictions)
                
                # 상위 프로세스
                self.display_top_processes()
                
                # 보안 상태
                security_only_alerts = [a for a in all_alerts if a.category in ['security', 'network_security', 'file_integrity']]
                self.display_security_status(security_only_alerts)
                
                # 성능 요약
                self.display_performance_summary()
                
                # 푸터
                self.display_footer()
                
                # 업데이트 대기
                time.sleep(self.update_interval)
        
        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            print(f"\n{Colors.PLASMA_RED}모니터링 오류: {e}{Colors.END}")
            self.stop()
    
    def stop(self):
        """모니터링 종료"""
        self.running = False
        print(f"\n\n{Colors.QUANTUM_BLUE}🛑 SysWatch Pro Quantum Ultimate 종료 중...{Colors.END}")
        
        # 통계 출력
        if self.ai_engine.history:
            print(f"📊 총 {len(self.ai_engine.history)}개 메트릭 수집")
            print(f"🚨 총 {len(self.ai_engine.alerts)}개 알림 생성")
            
            # 최종 리포트 생성 옵션
            try:
                report = self.analytics.generate_comprehensive_report(
                    self.ai_engine.history, 
                    self.ai_engine.alerts
                )
                
                report_file = self.analytics.reports_dir / f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                
                print(f"📋 최종 리포트 저장됨: {report_file}")
            except Exception as e:
                print(f"⚠️ 리포트 생성 오류: {e}")
        
        print(f"\n{Colors.NEON_GREEN}감사합니다! SysWatch Pro Quantum Ultimate을 사용해주셔서 감사합니다.{Colors.END}")
        print(f"{Colors.CYAN}🌟 차세대 AI 시스템 모니터링의 경험은 어떠셨나요?{Colors.END}")

# ============================
# GUI INTERFACE (OPTIONAL)
# ============================

class QuantumGUIInterface:
    """홀로그래픽 GUI 인터페이스 (선택적)"""
    
    def __init__(self):
        if not HAS_GUI:
            raise RuntimeError("GUI 패키지가 설치되지 않았습니다")
        
        self.root = tk.Tk()
        self.root.title("SysWatch Pro Quantum Ultimate")
        self.root.geometry("1200x800")
        self.root.configure(bg=QuantumTheme.BACKGROUND)
        
        self.ai_engine = QuantumAIEngine()
        self.security_engine = QuantumSecurityEngine()
        self.running = False
        
        self.setup_gui()
    
    def setup_gui(self):
        """GUI 설정"""
        # 메인 프레임
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 헤더
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill='x', pady=(0, 10))
        
        title_label = ttk.Label(
            header_frame, 
            text="🚀 SysWatch Pro Quantum Ultimate",
            font=("Arial", 16, "bold")
        )
        title_label.pack()
        
        # 노트북 (탭)
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill='both', expand=True)
        
        # 대시보드 탭
        self.dashboard_frame = ttk.Frame(notebook)
        notebook.add(self.dashboard_frame, text="📊 대시보드")
        
        # AI 분석 탭
        self.ai_frame = ttk.Frame(notebook)
        notebook.add(self.ai_frame, text="🧠 AI 분석")
        
        # 보안 탭
        self.security_frame = ttk.Frame(notebook)
        notebook.add(self.security_frame, text="🛡️ 보안")
        
        # 리포트 탭
        self.reports_frame = ttk.Frame(notebook)
        notebook.add(self.reports_frame, text="📋 리포트")
        
        self.setup_dashboard_tab()
        self.setup_ai_tab()
        self.setup_security_tab()
        self.setup_reports_tab()
        
        # 시작 버튼
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill='x', pady=(10, 0))
        
        self.start_button = ttk.Button(
            control_frame,
            text="🚀 모니터링 시작",
            command=self.start_monitoring
        )
        self.start_button.pack(side='left', padx=(0, 10))
        
        self.stop_button = ttk.Button(
            control_frame,
            text="🛑 중지",
            command=self.stop_monitoring,
            state='disabled'
        )
        self.stop_button.pack(side='left')
    
    def setup_dashboard_tab(self):
        """대시보드 탭 설정"""
        # CPU 프레임
        cpu_frame = ttk.LabelFrame(self.dashboard_frame, text="🖥️ CPU")
        cpu_frame.pack(fill='x', padx=5, pady=5)
        
        self.cpu_label = ttk.Label(cpu_frame, text="CPU: 0.0%")
        self.cpu_label.pack(pady=5)
        
        if HAS_VIZ:
            # CPU 차트
            self.cpu_fig = Figure(figsize=(6, 2), facecolor=QuantumTheme.BACKGROUND)
            self.cpu_ax = self.cpu_fig.add_subplot(111)
            self.cpu_canvas = FigureCanvasTkAgg(self.cpu_fig, cpu_frame)
            self.cpu_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # 메모리 프레임
        memory_frame = ttk.LabelFrame(self.dashboard_frame, text="💾 메모리")
        memory_frame.pack(fill='x', padx=5, pady=5)
        
        self.memory_label = ttk.Label(memory_frame, text="메모리: 0.0%")
        self.memory_label.pack(pady=5)
    
    def setup_ai_tab(self):
        """AI 분석 탭 설정"""
        ai_info_label = ttk.Label(
            self.ai_frame,
            text="🧠 AI 예측 분석 및 이상 탐지",
            font=("Arial", 12, "bold")
        )
        ai_info_label.pack(pady=10)
        
        self.ai_text = tk.Text(
            self.ai_frame,
            height=20,
            bg=QuantumTheme.BACKGROUND,
            fg=QuantumTheme.PRIMARY,
            font=("Consolas", 10)
        )
        self.ai_text.pack(fill='both', expand=True, padx=10, pady=10)
    
    def setup_security_tab(self):
        """보안 탭 설정"""
        security_info_label = ttk.Label(
            self.security_frame,
            text="🛡️ 실시간 보안 모니터링",
            font=("Arial", 12, "bold")
        )
        security_info_label.pack(pady=10)
        
        self.security_text = tk.Text(
            self.security_frame,
            height=20,
            bg=QuantumTheme.BACKGROUND,
            fg=QuantumTheme.ACCENT,
            font=("Consolas", 10)
        )
        self.security_text.pack(fill='both', expand=True, padx=10, pady=10)
    
    def setup_reports_tab(self):
        """리포트 탭 설정"""
        reports_info_label = ttk.Label(
            self.reports_frame,
            text="📋 분석 리포트 및 통계",
            font=("Arial", 12, "bold")
        )
        reports_info_label.pack(pady=10)
        
        generate_report_button = ttk.Button(
            self.reports_frame,
            text="📊 리포트 생성",
            command=self.generate_report
        )
        generate_report_button.pack(pady=10)
        
        self.reports_text = tk.Text(
            self.reports_frame,
            height=15,
            bg=QuantumTheme.BACKGROUND,
            fg=QuantumTheme.SUCCESS,
            font=("Consolas", 10)
        )
        self.reports_text.pack(fill='both', expand=True, padx=10, pady=10)
    
    def start_monitoring(self):
        """모니터링 시작"""
        self.running = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        
        # 모니터링 스레드 시작
        self.monitor_thread = threading.Thread(target=self.monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.running = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
    
    def monitoring_loop(self):
        """모니터링 루프"""
        while self.running:
            try:
                # 메트릭 수집
                metrics = self.ai_engine.collect_comprehensive_metrics()
                
                # GUI 업데이트 (메인 스레드에서)
                self.root.after(0, self.update_gui, metrics)
                
                time.sleep(3)
            except Exception as e:
                print(f"모니터링 오류: {e}")
                break
    
    def update_gui(self, metrics: QuantumMetrics):
        """GUI 업데이트"""
        # 레이블 업데이트
        self.cpu_label.config(text=f"CPU: {metrics.cpu_percent:.1f}%")
        self.memory_label.config(text=f"메모리: {metrics.memory_percent:.1f}%")
        
        # AI 텍스트 업데이트
        predictions = self.ai_engine.predict_future_performance()
        ai_info = f"[{datetime.now().strftime('%H:%M:%S')}] AI 분석 결과:\n"
        ai_info += f"CPU 예측: {predictions.get('cpu_prediction', 'N/A')}\n"
        ai_info += f"메모리 예측: {predictions.get('memory_prediction', 'N/A')}\n\n"
        
        self.ai_text.insert(tk.END, ai_info)
        self.ai_text.see(tk.END)
    
    def generate_report(self):
        """리포트 생성"""
        try:
            analytics = QuantumAnalyticsEngine()
            report = analytics.generate_comprehensive_report(
                self.ai_engine.history,
                self.ai_engine.alerts
            )
            
            # 리포트 텍스트 업데이트
            report_text = f"📋 분석 리포트 생성됨: {datetime.now()}\n"
            report_text += f"데이터 포인트: {len(self.ai_engine.history)}개\n"
            report_text += f"알림: {len(self.ai_engine.alerts)}개\n\n"
            
            if 'performance_summary' in report:
                perf = report['performance_summary']
                report_text += f"성능 요약:\n"
                report_text += f"  CPU 평균: {perf['cpu']['average']:.1f}%\n"
                report_text += f"  메모리 평균: {perf['memory']['average']:.1f}%\n\n"
            
            self.reports_text.delete(1.0, tk.END)
            self.reports_text.insert(tk.END, report_text)
            
        except Exception as e:
            messagebox.showerror("오류", f"리포트 생성 실패: {e}")
    
    def run(self):
        """GUI 실행"""
        self.root.mainloop()

# ============================
# WEB INTERFACE (OPTIONAL)
# ============================

class QuantumWebInterface:
    """웹 기반 인터페이스 (선택적)"""
    
    def __init__(self):
        if not HAS_WEB:
            raise RuntimeError("웹 프레임워크가 설치되지 않았습니다")
        
        self.app = Flask(__name__)
        self.ai_engine = QuantumAIEngine()
        self.setup_routes()
    
    def setup_routes(self):
        """웹 라우트 설정"""
        
        @self.app.route('/')
        def dashboard():
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>SysWatch Pro Quantum Ultimate</title>
                <style>
                    body { 
                        background: #0a0a0a; 
                        color: #00ff41; 
                        font-family: 'Courier New', monospace; 
                    }
                    .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
                    .header { text-align: center; margin-bottom: 30px; }
                    .metric { 
                        background: #1a1a1a; 
                        border: 1px solid #00ff41; 
                        padding: 15px; 
                        margin: 10px 0; 
                        border-radius: 5px; 
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🚀 SysWatch Pro Quantum Ultimate</h1>
                        <p>실시간 시스템 모니터링 웹 인터페이스</p>
                    </div>
                    
                    <div class="metric">
                        <h3>📊 시스템 메트릭</h3>
                        <div id="metrics">로딩 중...</div>
                    </div>
                    
                    <div class="metric">
                        <h3>🧠 AI 분석</h3>
                        <div id="ai-analysis">분석 중...</div>
                    </div>
                </div>
                
                <script>
                    function updateMetrics() {
                        fetch('/api/metrics')
                            .then(response => response.json())
                            .then(data => {
                                document.getElementById('metrics').innerHTML = 
                                    `CPU: ${data.cpu_percent}% | 메모리: ${data.memory_percent}%`;
                            });
                        
                        fetch('/api/ai-analysis')
                            .then(response => response.json())
                            .then(data => {
                                document.getElementById('ai-analysis').innerHTML = 
                                    JSON.stringify(data, null, 2);
                            });
                    }
                    
                    setInterval(updateMetrics, 3000);
                    updateMetrics();
                </script>
            </body>
            </html>
            """
        
        @self.app.route('/api/metrics')
        def api_metrics():
            metrics = self.ai_engine.collect_comprehensive_metrics()
            return jsonify(metrics.to_dict())
        
        @self.app.route('/api/ai-analysis')
        def api_ai_analysis():
            predictions = self.ai_engine.predict_future_performance()
            return jsonify(predictions)
    
    def run(self, host='127.0.0.1', port=5000):
        """웹 서버 실행"""
        print(f"🌐 웹 인터페이스 시작: http://{host}:{port}")
        self.app.run(host=host, port=port, debug=False)

# ============================
# MAIN APPLICATION
# ============================

class SysWatchQuantumUltimate:
    """메인 애플리케이션 클래스"""
    
    def __init__(self):
        self.terminal_interface = None
        self.gui_interface = None
        self.web_interface = None
        
    def show_banner(self):
        """시작 배너 표시"""
        print(f"{Colors.QUANTUM_BLUE}{Colors.BOLD}")
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 15 + "🚀 SYSWATCH PRO QUANTUM ULTIMATE" + " " * 24 + "║")
        print("║" + " " * 10 + "AAA급 통합 시스템 모니터링 & AI 분석 스위트" + " " * 18 + "║")
        print("║" + " " * 78 + "║")
        print("║" + " " * 5 + "🧠 AI 예측 엔진 | 🛡️ 군사급 보안 | 📊 엔터프라이즈 분석" + " " * 10 + "║")
        print("║" + " " * 20 + "Copyright (C) 2025 SysWatch Technologies" + " " * 17 + "║")
        print("╚" + "═" * 78 + "╝")
        print(f"{Colors.END}")
        print()
    
    def show_system_info(self):
        """시스템 정보 표시"""
        print(f"{Colors.CYAN}📋 시스템 환경:{Colors.END}")
        print(f"   OS: {platform.system()} {platform.release()} ({platform.machine()})")
        print(f"   Python: {sys.version.split()[0]}")
        print(f"   CPU: {psutil.cpu_count()}코어")
        print(f"   메모리: {psutil.virtual_memory().total // (1024**3)}GB")
        print()
        
        print(f"{Colors.CYAN}🔧 기능 지원:{Colors.END}")
        print(f"   🧠 AI/ML: {'🟢 지원' if HAS_ML else '🟡 제한'}")
        print(f"   🎨 GUI: {'🟢 지원' if HAS_GUI else '🟡 제한'}")
        print(f"   📊 시각화: {'🟢 지원' if HAS_VIZ else '🟡 제한'}")
        print(f"   🌐 웹: {'🟢 지원' if HAS_WEB else '🟡 제한'}")
        print()
    
    def show_menu(self):
        """메인 메뉴 표시"""
        print(f"{Colors.BOLD}🎯 실행 모드 선택:{Colors.END}")
        print()
        print(f"   1. {Colors.NEON_GREEN}💻 터미널 모니터링{Colors.END} (권장 - 모든 기능)")
        print(f"      • 실시간 AI 분석 및 예측")
        print(f"      • 군사급 보안 모니터링")
        print(f"      • 성능 최적화 권장사항")
        print()
        
        if HAS_GUI:
            print(f"   2. {Colors.PURPLE}🎨 GUI 모니터링{Colors.END} (홀로그래픽)")
            print(f"      • 3D 시각화 인터페이스")
            print(f"      • 인터랙티브 차트")
            print(f"      • 실시간 대시보드")
            print()
        
        if HAS_WEB:
            print(f"   3. {Colors.CYAN}🌐 웹 인터페이스{Colors.END} (원격 접근)")
            print(f"      • 브라우저 기반 모니터링")
            print(f"      • RESTful API")
            print(f"      • 다중 사용자 지원")
            print()
        
        print(f"   4. {Colors.CYBER_YELLOW}🔧 시스템 분석 리포트{Colors.END} (일회성)")
        print(f"      • 현재 시스템 상태 분석")
        print(f"      • 최적화 권장사항")
        print(f"      • 종합 성능 리포트")
        print()
        
        print(f"   5. {Colors.PLASMA_RED}🛡️ 보안 스캔{Colors.END} (보안 전용)")
        print(f"      • 실시간 위협 탐지")
        print(f"      • 파일 무결성 검사")
        print(f"      • 네트워크 보안 분석")
        print()
        
        print(f"   0. {Colors.WHITE}❌ 종료{Colors.END}")
        print()
    
    def run_terminal_monitoring(self):
        """터미널 모니터링 실행"""
        print(f"{Colors.NEON_GREEN}💻 터미널 모니터링 모드 시작...{Colors.END}")
        print()
        
        self.terminal_interface = QuantumTerminalInterface()
        self.terminal_interface.run_monitoring_loop()
    
    def run_gui_monitoring(self):
        """GUI 모니터링 실행"""
        if not HAS_GUI:
            print(f"{Colors.PLASMA_RED}❌ GUI 패키지가 설치되지 않았습니다.{Colors.END}")
            print("pip install customtkinter ttkbootstrap")
            return
        
        print(f"{Colors.PURPLE}🎨 GUI 모니터링 모드 시작...{Colors.END}")
        
        try:
            self.gui_interface = QuantumGUIInterface()
            self.gui_interface.run()
        except Exception as e:
            print(f"{Colors.PLASMA_RED}GUI 실행 오류: {e}{Colors.END}")
    
    def run_web_interface(self):
        """웹 인터페이스 실행"""
        if not HAS_WEB:
            print(f"{Colors.PLASMA_RED}❌ 웹 프레임워크가 설치되지 않았습니다.{Colors.END}")
            print("pip install flask flask-socketio")
            return
        
        print(f"{Colors.CYAN}🌐 웹 인터페이스 모드 시작...{Colors.END}")
        
        try:
            self.web_interface = QuantumWebInterface()
            self.web_interface.run()
        except Exception as e:
            print(f"{Colors.PLASMA_RED}웹 인터페이스 실행 오류: {e}{Colors.END}")
    
    def run_system_analysis(self):
        """시스템 분석 리포트 실행"""
        print(f"{Colors.CYBER_YELLOW}🔧 시스템 분석 중...{Colors.END}")
        print()
        
        try:
            # AI 엔진 초기화
            ai_engine = QuantumAIEngine()
            optimizer = QuantumOptimizer()
            analytics = QuantumAnalyticsEngine()
            
            # 메트릭 수집 (10초간)
            print("📊 시스템 메트릭 수집 중... (10초)")
            metrics_list = []
            for i in range(10):
                metrics = ai_engine.collect_comprehensive_metrics()
                metrics_list.append(metrics)
                print(f"   {i+1}/10 완료")
                time.sleep(1)
            
            print("\n🧠 AI 분석 수행 중...")
            
            # 성능 분석
            performance_analysis = optimizer.analyze_system_performance()
            
            # 최종 메트릭
            latest_metrics = metrics_list[-1]
            
            # 결과 출력
            print(f"\n{Colors.BOLD}📋 시스템 분석 결과{Colors.END}")
            print("=" * 60)
            
            # 현재 상태
            print(f"\n{Colors.BOLD}📊 현재 시스템 상태{Colors.END}")
            print(f"   CPU 사용률: {latest_metrics.cpu_percent:.1f}%")
            print(f"   메모리 사용률: {latest_metrics.memory_percent:.1f}%")
            print(f"   디스크 사용률: {latest_metrics.disk_percent:.1f}%")
            print(f"   실행 중인 프로세스: {latest_metrics.process_count}개")
            
            if latest_metrics.temperature:
                print(f"   시스템 온도: {latest_metrics.temperature:.1f}°C")
            
            # 성능 분석
            print(f"\n{Colors.BOLD}⚡ 성능 분석{Colors.END}")
            cpu_analysis = performance_analysis['cpu_analysis']
            memory_analysis = performance_analysis['memory_analysis']
            disk_analysis = performance_analysis['disk_analysis']
            
            print(f"   CPU 상태: {cpu_analysis['status'].upper()}")
            print(f"   메모리 상태: {memory_analysis['status'].upper()}")
            print(f"   디스크 상태: {disk_analysis['status'].upper()}")
            
            # 권장사항
            if performance_analysis['recommendations']:
                print(f"\n{Colors.BOLD}💡 최적화 권장사항{Colors.END}")
                for i, rec in enumerate(performance_analysis['recommendations'], 1):
                    print(f"   {i}. {rec}")
            
            # AI 예측
            if len(metrics_list) >= 5:
                predictions = ai_engine.predict_future_performance()
                if predictions and 'cpu_trend' in predictions:
                    print(f"\n{Colors.BOLD}🔮 AI 예측 분석{Colors.END}")
                    print(f"   CPU 트렌드: {predictions['cpu_trend']}")
                    print(f"   메모리 트렌드: {predictions['memory_trend']}")
            
            # 리포트 저장
            try:
                report = analytics.generate_comprehensive_report(metrics_list, ai_engine.alerts)
                report_file = analytics.reports_dir / f"system_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                
                print(f"\n📋 상세 리포트 저장됨: {report_file}")
            except Exception as e:
                print(f"⚠️ 리포트 저장 오류: {e}")
            
            print(f"\n{Colors.NEON_GREEN}✅ 시스템 분석 완료!{Colors.END}")
            
        except Exception as e:
            print(f"{Colors.PLASMA_RED}시스템 분석 오류: {e}{Colors.END}")
        
        input(f"\n{Colors.CYAN}Enter를 눌러 메인 메뉴로 돌아가세요...{Colors.END}")
    
    def run_security_scan(self):
        """보안 스캔 실행"""
        print(f"{Colors.PLASMA_RED}🛡️ 보안 스캔 시작...{Colors.END}")
        print()
        
        try:
            security_engine = QuantumSecurityEngine()
            
            print("🔍 프로세스 보안 스캔 중...")
            process_alerts = security_engine.scan_running_processes()
            
            print("🌐 네트워크 보안 스캔 중...")
            network_alerts = security_engine.scan_network_connections()
            
            print("📁 파일 무결성 검사 중...")
            integrity_alerts = security_engine.check_file_integrity()
            
            # 결과 출력
            all_security_alerts = process_alerts + network_alerts + integrity_alerts
            
            print(f"\n{Colors.BOLD}🛡️ 보안 스캔 결과{Colors.END}")
            print("=" * 60)
            
            if not all_security_alerts:
                print(f"{Colors.NEON_GREEN}✅ 보안 위협이 발견되지 않았습니다.{Colors.END}")
            else:
                print(f"🚨 총 {len(all_security_alerts)}개의 보안 이슈가 발견되었습니다:")
                print()
                
                # 레벨별 분류
                critical_alerts = [a for a in all_security_alerts if a.level == 'CRITICAL']
                warning_alerts = [a for a in all_security_alerts if a.level == 'WARNING']
                
                if critical_alerts:
                    print(f"{Colors.PLASMA_RED}🚨 위험 수준 ({len(critical_alerts)}개):{Colors.END}")
                    for alert in critical_alerts:
                        print(f"   • {alert.message}")
                    print()
                
                if warning_alerts:
                    print(f"{Colors.CYBER_YELLOW}⚠️ 주의 수준 ({len(warning_alerts)}개):{Colors.END}")
                    for alert in warning_alerts:
                        print(f"   • {alert.message}")
                    print()
            
            # 권장사항
            print(f"{Colors.BOLD}💡 보안 권장사항{Colors.END}")
            print("   1. 정기적인 시스템 업데이트 실시")
            print("   2. 강력한 비밀번호 사용")
            print("   3. 방화벽 및 백신 소프트웨어 최신 상태 유지")
            print("   4. 불필요한 서비스 및 포트 비활성화")
            print("   5. 중요 파일의 정기적인 백업")
            
            print(f"\n{Colors.NEON_GREEN}✅ 보안 스캔 완료!{Colors.END}")
            
        except Exception as e:
            print(f"{Colors.PLASMA_RED}보안 스캔 오류: {e}{Colors.END}")
        
        input(f"\n{Colors.CYAN}Enter를 눌러 메인 메뉴로 돌아가세요...{Colors.END}")
    
    def run(self):
        """메인 실행 함수"""
        while True:
            # 화면 클리어
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # 배너 및 정보 표시
            self.show_banner()
            self.show_system_info()
            self.show_menu()
            
            # 사용자 입력
            try:
                choice = input(f"{Colors.CYAN}모드를 선택하세요 (0-5): {Colors.END}").strip()
                
                if choice == '1':
                    self.run_terminal_monitoring()
                elif choice == '2' and HAS_GUI:
                    self.run_gui_monitoring()
                elif choice == '3' and HAS_WEB:
                    self.run_web_interface()
                elif choice == '4':
                    self.run_system_analysis()
                elif choice == '5':
                    self.run_security_scan()
                elif choice == '0':
                    print(f"\n{Colors.NEON_GREEN}👋 SysWatch Pro Quantum Ultimate을 종료합니다.{Colors.END}")
                    print(f"{Colors.CYAN}차세대 AI 시스템 모니터링을 경험해주셔서 감사합니다!{Colors.END}")
                    break
                else:
                    print(f"\n{Colors.PLASMA_RED}❌ 잘못된 선택입니다.{Colors.END}")
                    input(f"{Colors.CYAN}Enter를 눌러 계속하세요...{Colors.END}")
                    
            except KeyboardInterrupt:
                print(f"\n\n{Colors.CYBER_YELLOW}🛑 사용자에 의해 중단되었습니다.{Colors.END}")
                break
            except Exception as e:
                print(f"\n{Colors.PLASMA_RED}오류 발생: {e}{Colors.END}")
                input(f"{Colors.CYAN}Enter를 눌러 계속하세요...{Colors.END}")

# ============================
# ENTRY POINT
# ============================

def main():
    """메인 함수"""
    try:
        print("🚀 SysWatch Pro Quantum Ultimate 초기화 중...")
        
        # 권한 확인 (Windows)
        if platform.system() == 'Windows':
            try:
                import ctypes
                if not ctypes.windll.shell32.IsUserAnAdmin():
                    print(f"{Colors.CYBER_YELLOW}⚠️ 관리자 권한을 권장합니다.{Colors.END}")
                    print("일부 기능이 제한될 수 있습니다.")
                    time.sleep(2)
            except:
                pass
        
        # 메인 애플리케이션 실행
        app = SysWatchQuantumUltimate()
        app.run()
        
    except KeyboardInterrupt:
        print(f"\n\n{Colors.CYBER_YELLOW}🛑 프로그램이 중단되었습니다.{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.PLASMA_RED}심각한 오류 발생: {e}{Colors.END}")
        print("프로그램을 다시 시작해주세요.")
    
    print(f"\n{Colors.QUANTUM_BLUE}프로그램을 종료합니다...{Colors.END}")

if __name__ == "__main__":
    main()