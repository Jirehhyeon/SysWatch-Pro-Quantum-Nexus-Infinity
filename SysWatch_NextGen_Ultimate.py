#!/usr/bin/env python3
"""
SysWatch NextGen Ultimate - 차세대 통합 모니터링 시스템
모든 기능 통합 | 3D 시각화 | AI 예측 | 실시간 분석 | 홀로그래픽 인터페이스

🚀 차세대 AI 시스템 모니터링의 완성체
🧠 실시간 머신러닝 예측 | 🛡️ 군사급 보안 | 📊 엔터프라이즈 분석
💫 홀로그래픽 3D 시각화 | ⚡ 60fps 실시간 렌더링 | 🎮 인터랙티브 UI

Copyright (C) 2025 SysWatch Technologies Ltd.
NextGen Ultimate Edition - All-in-One Supreme Quality
"""

import sys
import os
import time
import threading
import asyncio
import math
import random
import json
import sqlite3
import hashlib
import hmac
import base64
import platform
import socket
import subprocess
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from collections import deque, defaultdict, namedtuple
import logging
import multiprocessing
import concurrent.futures

warnings.filterwarnings('ignore')

# ============================
# SMART PACKAGE MANAGER
# ============================

class SmartPackageManager:
    """지능형 패키지 관리 시스템"""
    
    def __init__(self):
        self.installed_packages = set()
        self.failed_packages = set()
        
    def install_package(self, package_name, import_name=None, version=None):
        """패키지 설치"""
        if package_name in self.installed_packages:
            return True
            
        if package_name in self.failed_packages:
            return False
            
        import_name = import_name or package_name
        
        try:
            __import__(import_name)
            self.installed_packages.add(package_name)
            return True
        except ImportError:
            try:
                print(f"📦 Installing {package_name}...")
                cmd = [sys.executable, '-m', 'pip', 'install', package_name, '--quiet', '--disable-pip-version-check']
                if version:
                    cmd[-2] = f"{package_name}=={version}"
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if result.returncode == 0:
                    __import__(import_name)
                    self.installed_packages.add(package_name)
                    print(f"✅ {package_name} installed successfully")
                    return True
                else:
                    self.failed_packages.add(package_name)
                    print(f"⚠️ {package_name} installation failed")
                    return False
            except Exception as e:
                self.failed_packages.add(package_name)
                print(f"❌ {package_name} installation error: {e}")
                return False

# 패키지 매니저 초기화
pkg_manager = SmartPackageManager()

# 필수 패키지 설치
essential_packages = [
    ('psutil', 'psutil'),
    ('numpy', 'numpy'),
    ('pandas', 'pandas'),
    ('matplotlib', 'matplotlib'),
    ('pygame', 'pygame'),
    ('pillow', 'PIL'),
    ('requests', 'requests'),
    ('flask', 'flask')
]

print("🚀 SysWatch NextGen Ultimate 초기화 중...")
print("📦 필수 패키지 확인 및 설치...")

for pkg_name, import_name in essential_packages:
    pkg_manager.install_package(pkg_name, import_name)

# 고급 패키지 (선택적)
advanced_packages = [
    ('scikit-learn', 'sklearn'),
    ('plotly', 'plotly'),
    ('opencv-python', 'cv2'),
    ('tensorflow-cpu', 'tensorflow'),
    ('torch', 'torch'),
    ('customtkinter', 'customtkinter'),
    ('ttkbootstrap', 'ttkbootstrap')
]

print("🧠 고급 AI/ML 패키지 설치 중...")
for pkg_name, import_name in advanced_packages:
    pkg_manager.install_package(pkg_name, import_name)

# 패키지 임포트
import numpy as np
import pandas as pd
import psutil
import pygame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import requests
import json
from flask import Flask, jsonify, render_template_string

# AI/ML 패키지
HAS_ML = False
try:
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score
    import sklearn.neural_network as nn
    HAS_ML = True
    print("✅ AI/ML 엔진 활성화")
except ImportError:
    print("⚠️ AI/ML 기능 제한됨")

# 고급 시각화
HAS_PLOTLY = False
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
    print("✅ 고급 시각화 활성화")
except ImportError:
    print("⚠️ 고급 시각화 제한됨")

# 컴퓨터 비전
HAS_CV2 = False
try:
    import cv2
    HAS_CV2 = True
    print("✅ 컴퓨터 비전 활성화")
except ImportError:
    print("⚠️ 컴퓨터 비전 제한됨")

# GUI 프레임워크
HAS_ADVANCED_GUI = False
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    import customtkinter as ctk
    import ttkbootstrap as tb
    HAS_ADVANCED_GUI = True
    print("✅ 고급 GUI 활성화")
except ImportError:
    try:
        import tkinter as tk
        from tkinter import ttk
        print("✅ 기본 GUI 활성화")
    except ImportError:
        print("⚠️ GUI 기능 제한됨")

print("🎯 모든 컴포넌트 로드 완료!\n")

# ============================
# PYGAME 초기화 및 설정
# ============================

pygame.init()
pygame.mixer.quit()  # 성능 최적화

# 디스플레이 설정
info = pygame.display.Info()
SCREEN_WIDTH = info.current_w
SCREEN_HEIGHT = info.current_h

# 고성능 디스플레이 모드
screen = pygame.display.set_mode(
    (SCREEN_WIDTH, SCREEN_HEIGHT), 
    pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE
)
pygame.display.set_caption("SysWatch NextGen Ultimate - 차세대 통합 모니터링")

# 폰트 로딩
def load_fonts():
    """폰트 로딩"""
    try:
        return {
            'title': pygame.font.Font(None, 72),
            'large': pygame.font.Font(None, 48),
            'medium': pygame.font.Font(None, 32),
            'small': pygame.font.Font(None, 24),
            'tiny': pygame.font.Font(None, 18)
        }
    except:
        return {
            'title': pygame.font.SysFont('arial', 72, bold=True),
            'large': pygame.font.SysFont('arial', 48, bold=True),
            'medium': pygame.font.SysFont('arial', 32),
            'small': pygame.font.SysFont('arial', 24),
            'tiny': pygame.font.SysFont('arial', 18)
        }

fonts = load_fonts()

# ============================
# 색상 시스템 - 차세대 홀로그래픽 테마
# ============================

class NextGenColors:
    """차세대 홀로그래픽 색상 시스템"""
    
    # 기본 색상
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    
    # 홀로그래픽 메인 색상
    HOLO_BLUE = (0, 180, 255)          # 홀로그래픽 블루
    NEON_CYAN = (0, 255, 255)          # 네온 시안
    PLASMA_GREEN = (0, 255, 100)       # 플라즈마 그린
    QUANTUM_PURPLE = (150, 0, 255)     # 양자 퍼플
    ENERGY_ORANGE = (255, 150, 0)      # 에너지 오렌지
    LASER_RED = (255, 0, 100)          # 레이저 레드
    CRYSTAL_PINK = (255, 100, 200)     # 크리스탈 핑크
    
    # 그라데이션 색상
    DEEP_SPACE = (5, 5, 15)            # 깊은 우주
    DARK_MATTER = (10, 10, 20)         # 암흑 물질
    NEBULA_BLUE = (20, 30, 60)         # 성운 블루
    COSMIC_PURPLE = (40, 20, 80)       # 우주 퍼플
    
    # 상태 색상
    HEALTH_GOOD = (0, 255, 100)        # 건강 - 좋음
    HEALTH_WARNING = (255, 200, 0)     # 건강 - 경고
    HEALTH_CRITICAL = (255, 50, 50)    # 건강 - 위험
    
    # 투명도 변형
    @staticmethod
    def with_alpha(color, alpha):
        """색상에 알파 채널 추가"""
        return (*color[:3], alpha)
    
    @staticmethod
    def mix_colors(color1, color2, ratio):
        """두 색상을 비율에 따라 혼합"""
        return tuple(int(c1 * (1 - ratio) + c2 * ratio) for c1, c2 in zip(color1, color2))
    
    @staticmethod
    def pulse_color(base_color, time, speed=2):
        """맥박 효과 색상"""
        pulse = (math.sin(time * speed) + 1) / 2
        return tuple(int(c * (0.5 + pulse * 0.5)) for c in base_color)

# ============================
# 데이터 구조체 - 고도화
# ============================

@dataclass
class ComprehensiveMetrics:
    """종합 시스템 메트릭"""
    timestamp: datetime
    
    # CPU 메트릭
    cpu_percent: float
    cpu_freq_current: float
    cpu_freq_min: float
    cpu_freq_max: float
    cpu_cores_physical: int
    cpu_cores_logical: int
    cpu_times_user: float
    cpu_times_system: float
    cpu_times_idle: float
    cpu_times_iowait: float
    cpu_per_core: List[float]
    
    # 메모리 메트릭
    memory_total: int
    memory_available: int
    memory_used: int
    memory_percent: float
    memory_cached: int
    memory_buffers: int
    swap_total: int
    swap_used: int
    swap_percent: float
    
    # 디스크 메트릭
    disk_total: int
    disk_used: int
    disk_free: int
    disk_percent: float
    disk_read_bytes: int
    disk_write_bytes: int
    disk_read_count: int
    disk_write_count: int
    disk_read_speed: float
    disk_write_speed: float
    
    # 네트워크 메트릭
    network_bytes_sent: int
    network_bytes_recv: int
    network_packets_sent: int
    network_packets_recv: int
    network_sent_speed: float
    network_recv_speed: float
    network_connections: int
    network_connections_established: int
    
    # 프로세스 메트릭
    process_count: int
    thread_count: int
    handle_count: int
    
    # 시스템 메트릭
    boot_time: float
    uptime_seconds: float
    load_average: Optional[Tuple[float, float, float]]
    
    # 하드웨어 메트릭
    temperature_cpu: Optional[float]
    temperature_gpu: Optional[float]
    temperature_system: Optional[float]
    battery_percent: Optional[float]
    battery_power_plugged: Optional[bool]
    battery_time_left: Optional[int]
    
    # GPU 메트릭 (추정/실제)
    gpu_percent: Optional[float]
    gpu_memory_used: Optional[int]
    gpu_memory_total: Optional[int]
    gpu_temperature: Optional[float]
    
    # 보안 메트릭
    security_threat_level: int
    security_active_scans: int
    security_blocked_connections: int
    
    # 성능 점수
    performance_score: float
    health_score: float
    efficiency_score: float

@dataclass
class AIAnalysisResult:
    """AI 분석 결과"""
    timestamp: datetime
    anomaly_score: float
    is_anomaly: bool
    prediction_cpu: float
    prediction_memory: float
    prediction_confidence: float
    performance_trend: str
    optimization_suggestions: List[str]
    risk_assessment: Dict[str, float]
    future_bottlenecks: List[str]

@dataclass
class SecurityThreat:
    """보안 위협 정보"""
    threat_id: str
    level: str  # LOW, MEDIUM, HIGH, CRITICAL
    category: str
    description: str
    timestamp: datetime
    source_ip: Optional[str]
    process_name: Optional[str]
    confidence: float
    mitigation_steps: List[str]

# ============================
# 고급 AI 엔진
# ============================

class NextGenAIEngine:
    """차세대 AI 분석 엔진"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=1000)
        self.analysis_results = deque(maxlen=100)
        
        # AI 모델들
        self.anomaly_detector = None
        self.performance_predictor = None
        self.pattern_analyzer = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # 학습 데이터
        self.training_features = []
        self.training_labels = []
        
        if HAS_ML:
            self.initialize_models()
        
        self.setup_logging()
    
    def initialize_models(self):
        """AI 모델 초기화"""
        try:
            # 이상 탐지 모델
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=200,
                max_samples='auto'
            )
            
            # 성능 예측 모델
            self.performance_predictor = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            
            # 패턴 분석 모델
            self.pattern_analyzer = KMeans(
                n_clusters=5,
                random_state=42,
                n_init=10
            )
            
            print("🧠 AI 모델 초기화 완료")
        except Exception as e:
            print(f"⚠️ AI 모델 초기화 오류: {e}")
    
    def setup_logging(self):
        """로깅 설정"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "nextgen_ai.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("NextGenAI")
    
    def extract_features(self, metrics: ComprehensiveMetrics) -> np.ndarray:
        """메트릭에서 특징 추출"""
        features = [
            metrics.cpu_percent,
            metrics.memory_percent,
            metrics.disk_percent,
            metrics.network_sent_speed / (1024 * 1024),  # MB/s
            metrics.network_recv_speed / (1024 * 1024),  # MB/s
            metrics.process_count,
            metrics.thread_count,
            len(metrics.cpu_per_core),
            np.mean(metrics.cpu_per_core) if metrics.cpu_per_core else 0,
            np.std(metrics.cpu_per_core) if metrics.cpu_per_core else 0,
            metrics.cpu_freq_current / 1000 if metrics.cpu_freq_current else 0,  # GHz
            metrics.disk_read_speed / (1024 * 1024),  # MB/s
            metrics.disk_write_speed / (1024 * 1024),  # MB/s
            metrics.uptime_seconds / 3600,  # hours
            metrics.temperature_cpu or 50,  # default temp
            metrics.network_connections,
            metrics.performance_score,
            metrics.health_score,
            metrics.efficiency_score
        ]
        
        return np.array(features).reshape(1, -1)
    
    def train_models(self):
        """모델 훈련"""
        if not HAS_ML or len(self.metrics_history) < 50:
            return False
        
        try:
            # 특징 데이터 준비
            features = []
            for metrics in list(self.metrics_history)[-100:]:
                feature_vector = self.extract_features(metrics)
                features.append(feature_vector.flatten())
            
            features_array = np.array(features)
            
            # 데이터 정규화
            scaled_features = self.scaler.fit_transform(features_array)
            
            # 이상 탐지 모델 훈련
            if self.anomaly_detector:
                self.anomaly_detector.fit(scaled_features)
            
            # 패턴 분석 모델 훈련
            if self.pattern_analyzer:
                self.pattern_analyzer.fit(scaled_features)
            
            # 성능 예측 모델 훈련 (CPU 사용률 예측)
            if self.performance_predictor and len(features) > 10:
                X = scaled_features[:-5]  # 이전 데이터
                y = [m.cpu_percent for m in list(self.metrics_history)[-95:-5]]  # 5스텝 후 CPU
                
                if len(X) == len(y) and len(y) > 0:
                    self.performance_predictor.fit(X, y)
            
            self.is_trained = True
            self.logger.info("AI 모델 훈련 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"모델 훈련 오류: {e}")
            return False
    
    def analyze_metrics(self, metrics: ComprehensiveMetrics) -> AIAnalysisResult:
        """메트릭 AI 분석"""
        self.metrics_history.append(metrics)
        
        # 주기적 모델 재훈련
        if len(self.metrics_history) % 20 == 0:
            self.train_models()
        
        # 기본 분석 결과
        result = AIAnalysisResult(
            timestamp=datetime.now(),
            anomaly_score=0.0,
            is_anomaly=False,
            prediction_cpu=metrics.cpu_percent,
            prediction_memory=metrics.memory_percent,
            prediction_confidence=0.5,
            performance_trend='stable',
            optimization_suggestions=[],
            risk_assessment={},
            future_bottlenecks=[]
        )
        
        if not HAS_ML or not self.is_trained:
            return self._basic_analysis(metrics, result)
        
        try:
            # 특징 추출
            features = self.extract_features(metrics)
            scaled_features = self.scaler.transform(features)
            
            # 이상 탐지
            if self.anomaly_detector:
                anomaly_score = self.anomaly_detector.decision_function(scaled_features)[0]
                is_anomaly = self.anomaly_detector.predict(scaled_features)[0] == -1
                
                result.anomaly_score = anomaly_score
                result.is_anomaly = is_anomaly
            
            # 성능 예측
            if self.performance_predictor:
                try:
                    cpu_prediction = self.performance_predictor.predict(scaled_features)[0]
                    result.prediction_cpu = max(0, min(100, cpu_prediction))
                    result.prediction_confidence = 0.8
                except:
                    pass
            
            # 트렌드 분석
            result.performance_trend = self._analyze_trend()
            
            # 최적화 제안
            result.optimization_suggestions = self._generate_optimization_suggestions(metrics)
            
            # 위험 평가
            result.risk_assessment = self._assess_risks(metrics)
            
            # 미래 병목 예측
            result.future_bottlenecks = self._predict_bottlenecks(metrics)
            
        except Exception as e:
            self.logger.error(f"AI 분석 오류: {e}")
        
        self.analysis_results.append(result)
        return result
    
    def _basic_analysis(self, metrics: ComprehensiveMetrics, result: AIAnalysisResult) -> AIAnalysisResult:
        """기본 분석 (ML 없이)"""
        # 간단한 이상 탐지
        if (metrics.cpu_percent > 90 or 
            metrics.memory_percent > 95 or 
            metrics.disk_percent > 98):
            result.is_anomaly = True
            result.anomaly_score = -0.5
        
        # 기본 예측 (현재값 기반)
        if len(self.metrics_history) > 5:
            recent_cpu = [m.cpu_percent for m in list(self.metrics_history)[-5:]]
            result.prediction_cpu = np.mean(recent_cpu)
        
        return result
    
    def _analyze_trend(self) -> str:
        """트렌드 분석"""
        if len(self.metrics_history) < 10:
            return 'stable'
        
        recent_cpu = [m.cpu_percent for m in list(self.metrics_history)[-10:]]
        recent_memory = [m.memory_percent for m in list(self.metrics_history)[-10:]]
        
        cpu_trend = np.polyfit(range(len(recent_cpu)), recent_cpu, 1)[0]
        memory_trend = np.polyfit(range(len(recent_memory)), recent_memory, 1)[0]
        
        if cpu_trend > 2 or memory_trend > 2:
            return 'increasing'
        elif cpu_trend < -2 or memory_trend < -2:
            return 'decreasing'
        else:
            return 'stable'
    
    def _generate_optimization_suggestions(self, metrics: ComprehensiveMetrics) -> List[str]:
        """최적화 제안 생성"""
        suggestions = []
        
        if metrics.cpu_percent > 80:
            suggestions.append("🔥 CPU 사용률이 높습니다. 불필요한 프로세스를 종료하세요.")
            
        if metrics.memory_percent > 85:
            suggestions.append("💾 메모리 부족입니다. 브라우저 탭을 정리하거나 프로그램을 종료하세요.")
            
        if metrics.disk_percent > 90:
            suggestions.append("💿 디스크 공간이 부족합니다. 파일을 정리하세요.")
            
        if metrics.network_sent_speed > 50 * 1024 * 1024:  # 50MB/s
            suggestions.append("🌐 네트워크 업로드가 높습니다. 대용량 파일 전송을 확인하세요.")
            
        if metrics.temperature_cpu and metrics.temperature_cpu > 75:
            suggestions.append("🌡️ CPU 온도가 높습니다. 냉각 시스템을 점검하세요.")
            
        if not suggestions:
            suggestions.append("✅ 시스템이 최적 상태입니다!")
            
        return suggestions
    
    def _assess_risks(self, metrics: ComprehensiveMetrics) -> Dict[str, float]:
        """위험 평가"""
        risks = {
            'performance': 0.0,
            'stability': 0.0,
            'security': 0.0,
            'hardware': 0.0
        }
        
        # 성능 위험
        if metrics.cpu_percent > 90:
            risks['performance'] += 0.5
        if metrics.memory_percent > 90:
            risks['performance'] += 0.4
        if metrics.disk_percent > 95:
            risks['performance'] += 0.3
            
        # 안정성 위험
        if metrics.uptime_seconds < 3600:  # 1시간 미만
            risks['stability'] += 0.2
        if metrics.process_count > 200:
            risks['stability'] += 0.3
            
        # 하드웨어 위험
        if metrics.temperature_cpu and metrics.temperature_cpu > 80:
            risks['hardware'] += 0.6
        if metrics.battery_percent and metrics.battery_percent < 15:
            risks['hardware'] += 0.4
            
        # 보안 위험
        if metrics.security_threat_level > 2:
            risks['security'] += 0.5
        if metrics.network_connections > 100:
            risks['security'] += 0.2
            
        return risks
    
    def _predict_bottlenecks(self, metrics: ComprehensiveMetrics) -> List[str]:
        """미래 병목 예측"""
        bottlenecks = []
        
        if len(self.metrics_history) < 5:
            return bottlenecks
            
        # CPU 병목 예측
        recent_cpu = [m.cpu_percent for m in list(self.metrics_history)[-5:]]
        if all(cpu > 70 for cpu in recent_cpu[-3:]):
            bottlenecks.append("CPU 병목 예상 (5분 이내)")
            
        # 메모리 병목 예측
        recent_memory = [m.memory_percent for m in list(self.metrics_history)[-5:]]
        memory_growth = (recent_memory[-1] - recent_memory[0]) / len(recent_memory)
        if memory_growth > 2:  # 2%씩 증가
            bottlenecks.append("메모리 고갈 예상 (10분 이내)")
            
        # 디스크 병목 예측
        if metrics.disk_write_speed > 100 * 1024 * 1024:  # 100MB/s
            bottlenecks.append("디스크 I/O 포화 가능성")
            
        return bottlenecks

# ============================
# 고급 보안 엔진
# ============================

class NextGenSecurityEngine:
    """차세대 보안 분석 엔진"""
    
    def __init__(self):
        self.threats = deque(maxlen=1000)
        self.blocked_connections = set()
        self.monitored_processes = {}
        self.file_integrity_db = {}
        
        # 위협 패턴 데이터베이스
        self.threat_patterns = {
            'suspicious_processes': [
                'nc.exe', 'netcat', 'nmap', 'wireshark', 'burp',
                'sqlmap', 'hydra', 'john', 'hashcat', 'metasploit',
                'powershell.exe', 'cmd.exe'
            ],
            'dangerous_ports': [22, 23, 135, 139, 445, 1433, 3389, 5900],
            'suspicious_network_patterns': [
                'excessive_connections',
                'unusual_data_transfer',
                'connection_to_tor',
                'connection_to_vpn'
            ]
        }
        
        self.setup_security_database()
    
    def setup_security_database(self):
        """보안 데이터베이스 설정"""
        try:
            db_path = Path("security_nextgen.db")
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            
            # 테이블 생성
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS security_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    threat_id TEXT NOT NULL,
                    level TEXT NOT NULL,
                    category TEXT NOT NULL,
                    description TEXT NOT NULL,
                    source_ip TEXT,
                    process_name TEXT,
                    confidence REAL,
                    mitigation_steps TEXT,
                    status TEXT DEFAULT 'active'
                )
            """)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS file_integrity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    hash_sha256 TEXT NOT NULL,
                    hash_md5 TEXT NOT NULL,
                    last_modified TEXT NOT NULL,
                    file_size INTEGER,
                    permissions TEXT,
                    status TEXT DEFAULT 'monitored'
                )
            """)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS network_monitoring (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    local_ip TEXT,
                    local_port INTEGER,
                    remote_ip TEXT,
                    remote_port INTEGER,
                    protocol TEXT,
                    status TEXT,
                    data_sent INTEGER,
                    data_recv INTEGER,
                    risk_level TEXT
                )
            """)
            
            self.conn.commit()
            
        except Exception as e:
            print(f"보안 데이터베이스 설정 오류: {e}")
    
    def comprehensive_security_scan(self) -> List[SecurityThreat]:
        """종합 보안 스캔"""
        threats = []
        
        # 프로세스 스캔
        threats.extend(self.scan_processes())
        
        # 네트워크 스캔
        threats.extend(self.scan_network())
        
        # 파일 무결성 검사
        threats.extend(self.check_file_integrity())
        
        # 시스템 취약점 스캔
        threats.extend(self.scan_vulnerabilities())
        
        # 행동 분석
        threats.extend(self.analyze_behavior())
        
        return threats
    
    def scan_processes(self) -> List[SecurityThreat]:
        """프로세스 보안 스캔"""
        threats = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'username', 'cpu_percent', 'memory_percent']):
                try:
                    proc_info = proc.info
                    proc_name = proc_info['name'].lower()
                    
                    # 의심스러운 프로세스 탐지
                    for suspicious in self.threat_patterns['suspicious_processes']:
                        if suspicious in proc_name:
                            threat = SecurityThreat(
                                threat_id=f"PROC_{proc_info['pid']}_{int(time.time())}",
                                level='MEDIUM',
                                category='suspicious_process',
                                description=f"의심스러운 프로세스 탐지: {proc_info['name']} (PID: {proc_info['pid']})",
                                timestamp=datetime.now(),
                                process_name=proc_info['name'],
                                confidence=0.7,
                                mitigation_steps=[
                                    "프로세스 세부 정보 확인",
                                    "필요시 프로세스 종료",
                                    "바이러스 스캔 실행"
                                ]
                            )
                            threats.append(threat)
                            break
                    
                    # 높은 리소스 사용 프로세스
                    if proc_info['cpu_percent'] > 90:
                        threat = SecurityThreat(
                            threat_id=f"HIGH_CPU_{proc_info['pid']}_{int(time.time())}",
                            level='LOW',
                            category='resource_abuse',
                            description=f"높은 CPU 사용률: {proc_info['name']} ({proc_info['cpu_percent']:.1f}%)",
                            timestamp=datetime.now(),
                            process_name=proc_info['name'],
                            confidence=0.6,
                            mitigation_steps=[
                                "프로세스 모니터링 계속",
                                "필요시 프로세스 우선순위 조정"
                            ]
                        )
                        threats.append(threat)
                
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        
        except Exception as e:
            print(f"프로세스 스캔 오류: {e}")
        
        return threats
    
    def scan_network(self) -> List[SecurityThreat]:
        """네트워크 보안 스캔"""
        threats = []
        
        try:
            connections = psutil.net_connections(kind='inet')
            
            # 연결 통계
            external_connections = 0
            suspicious_ports = 0
            
            for conn in connections:
                if conn.laddr:
                    # 의심스러운 포트 확인
                    if conn.laddr.port in self.threat_patterns['dangerous_ports']:
                        suspicious_ports += 1
                        
                        threat = SecurityThreat(
                            threat_id=f"PORT_{conn.laddr.port}_{int(time.time())}",
                            level='MEDIUM',
                            category='suspicious_port',
                            description=f"위험 포트 사용: {conn.laddr.port} ({conn.status})",
                            timestamp=datetime.now(),
                            source_ip=conn.laddr.ip,
                            confidence=0.8,
                            mitigation_steps=[
                                "포트 사용 목적 확인",
                                "불필요시 서비스 중지",
                                "방화벽 규칙 검토"
                            ]
                        )
                        threats.append(threat)
                
                # 외부 연결 카운트
                if conn.raddr and not self._is_local_ip(conn.raddr.ip):
                    external_connections += 1
            
            # 과도한 외부 연결
            if external_connections > 50:
                threat = SecurityThreat(
                    threat_id=f"CONN_FLOOD_{int(time.time())}",
                    level='MEDIUM',
                    category='connection_flood',
                    description=f"과도한 외부 연결: {external_connections}개",
                    timestamp=datetime.now(),
                    confidence=0.7,
                    mitigation_steps=[
                        "네트워크 트래픽 분석",
                        "불필요한 연결 차단",
                        "DDoS 공격 가능성 검토"
                    ]
                )
                threats.append(threat)
        
        except Exception as e:
            print(f"네트워크 스캔 오류: {e}")
        
        return threats
    
    def check_file_integrity(self) -> List[SecurityThreat]:
        """파일 무결성 검사"""
        threats = []
        
        # 중요 시스템 파일 목록
        if platform.system() == 'Windows':
            critical_files = [
                'C:\\Windows\\System32\\drivers\\etc\\hosts',
                'C:\\Windows\\System32\\kernel32.dll',
                'C:\\Windows\\System32\\ntdll.dll'
            ]
        else:
            critical_files = [
                '/etc/passwd',
                '/etc/shadow',
                '/etc/hosts',
                '/usr/bin/sudo'
            ]
        
        try:
            for file_path in critical_files:
                if os.path.exists(file_path):
                    current_hash = self._calculate_file_hash(file_path)
                    
                    # 데이터베이스에서 이전 해시 조회
                    cursor = self.conn.execute(
                        "SELECT hash_sha256 FROM file_integrity WHERE file_path = ?",
                        (file_path,)
                    )
                    result = cursor.fetchone()
                    
                    if result:
                        stored_hash = result[0]
                        if current_hash != stored_hash:
                            threat = SecurityThreat(
                                threat_id=f"INTEGRITY_{int(time.time())}",
                                level='CRITICAL',
                                category='file_integrity',
                                description=f"파일 무결성 위반: {file_path}",
                                timestamp=datetime.now(),
                                confidence=0.95,
                                mitigation_steps=[
                                    "파일 변경 사유 조사",
                                    "백업에서 복원 고려",
                                    "시스템 전체 스캔 실행"
                                ]
                            )
                            threats.append(threat)
                    else:
                        # 새 파일 등록
                        file_stat = os.stat(file_path)
                        self.conn.execute("""
                            INSERT INTO file_integrity 
                            (file_path, hash_sha256, hash_md5, last_modified, file_size, permissions)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            file_path,
                            current_hash,
                            hashlib.md5(open(file_path, 'rb').read()).hexdigest(),
                            datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                            file_stat.st_size,
                            oct(file_stat.st_mode)
                        ))
                        self.conn.commit()
        
        except Exception as e:
            print(f"파일 무결성 검사 오류: {e}")
        
        return threats
    
    def scan_vulnerabilities(self) -> List[SecurityThreat]:
        """시스템 취약점 스캔"""
        threats = []
        
        try:
            # 운영체제 정보
            os_info = platform.platform()
            
            # 패치 수준 확인 (간단한 예시)
            if platform.system() == 'Windows':
                # Windows 업데이트 상태 확인
                try:
                    result = subprocess.run(['powershell', 'Get-WmiObject -Class Win32_QuickFixEngineering | Measure-Object | Select-Object -ExpandProperty Count'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        patch_count = int(result.stdout.strip())
                        if patch_count < 10:  # 임의의 임계값
                            threat = SecurityThreat(
                                threat_id=f"PATCH_LOW_{int(time.time())}",
                                level='MEDIUM',
                                category='vulnerability',
                                description=f"패치 수준 낮음: {patch_count}개 업데이트만 설치됨",
                                timestamp=datetime.now(),
                                confidence=0.6,
                                mitigation_steps=[
                                    "Windows 업데이트 실행",
                                    "자동 업데이트 활성화",
                                    "보안 패치 우선 설치"
                                ]
                            )
                            threats.append(threat)
                except:
                    pass
            
            # 방화벽 상태 확인
            # (실제 구현에서는 시스템별 방화벽 상태를 확인)
            
            # 보안 소프트웨어 확인
            # (실제 구현에서는 설치된 보안 프로그램을 확인)
            
        except Exception as e:
            print(f"취약점 스캔 오류: {e}")
        
        return threats
    
    def analyze_behavior(self) -> List[SecurityThreat]:
        """행동 분석"""
        threats = []
        
        try:
            current_time = time.time()
            
            # 비정상적인 시스템 활동 패턴 분석
            # (예: 밤시간 높은 활동, 갑작스런 네트워크 트래픽 증가 등)
            
            hour = datetime.now().hour
            
            # 야간 시간대(22시-6시) 높은 활동
            if 22 <= hour or hour <= 6:
                cpu_usage = psutil.cpu_percent()
                if cpu_usage > 50:
                    threat = SecurityThreat(
                        threat_id=f"NIGHT_ACTIVITY_{int(time.time())}",
                        level='LOW',
                        category='behavioral_anomaly',
                        description=f"야간 시간대 높은 활동: CPU {cpu_usage:.1f}%",
                        timestamp=datetime.now(),
                        confidence=0.4,
                        mitigation_steps=[
                            "활동 프로세스 확인",
                            "스케줄된 작업 검토",
                            "악성코드 스캔 고려"
                        ]
                    )
                    threats.append(threat)
            
        except Exception as e:
            print(f"행동 분석 오류: {e}")
        
        return threats
    
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
        local_patterns = ['127.', '192.168.', '10.', '172.16.', '169.254.', '::1']
        return any(ip.startswith(pattern) for pattern in local_patterns)

# ============================
# 고성능 데이터 수집기
# ============================

class NextGenDataCollector:
    """차세대 데이터 수집 엔진"""
    
    def __init__(self):
        self.last_disk_io = psutil.disk_io_counters()
        self.last_network_io = psutil.net_io_counters()
        self.last_time = time.time()
        
        # 성능 캐시
        self.performance_cache = {}
        self.cache_timeout = 1.0  # 1초 캐시
        
    def collect_comprehensive_metrics(self) -> ComprehensiveMetrics:
        """종합 시스템 메트릭 수집"""
        current_time = time.time()
        time_delta = current_time - self.last_time
        
        # CPU 메트릭
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_freq = psutil.cpu_freq()
        cpu_times = psutil.cpu_times()
        cpu_per_core = psutil.cpu_percent(percpu=True, interval=None)
        
        # 메모리 메트릭
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # 디스크 메트릭
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # 디스크 속도 계산
        if time_delta > 0 and self.last_disk_io:
            disk_read_speed = (disk_io.read_bytes - self.last_disk_io.read_bytes) / time_delta
            disk_write_speed = (disk_io.write_bytes - self.last_disk_io.write_bytes) / time_delta
        else:
            disk_read_speed = disk_write_speed = 0
        
        self.last_disk_io = disk_io
        
        # 네트워크 메트릭
        network_io = psutil.net_io_counters()
        
        # 네트워크 속도 계산
        if time_delta > 0 and self.last_network_io:
            network_sent_speed = (network_io.bytes_sent - self.last_network_io.bytes_sent) / time_delta
            network_recv_speed = (network_io.bytes_recv - self.last_network_io.bytes_recv) / time_delta
        else:
            network_sent_speed = network_recv_speed = 0
        
        self.last_network_io = network_io
        self.last_time = current_time
        
        # 프로세스 및 시스템 메트릭
        process_count = len(psutil.pids())
        thread_count = sum(1 for _ in threading.enumerate())
        
        try:
            handle_count = sum(proc.num_handles() for proc in psutil.process_iter() 
                             if hasattr(proc, 'num_handles'))
        except:
            handle_count = 0
        
        # 시스템 메트릭
        boot_time = psutil.boot_time()
        uptime_seconds = current_time - boot_time
        
        # 로드 평균 (Unix 계열)
        load_average = None
        try:
            if hasattr(os, 'getloadavg'):
                load_average = os.getloadavg()
        except:
            pass
        
        # 온도 정보
        temperature_cpu = temperature_gpu = temperature_system = None
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                cpu_temps = []
                gpu_temps = []
                system_temps = []
                
                for name, entries in temps.items():
                    for entry in entries:
                        if entry.current:
                            if 'cpu' in name.lower() or 'core' in name.lower():
                                cpu_temps.append(entry.current)
                            elif 'gpu' in name.lower():
                                gpu_temps.append(entry.current)
                            else:
                                system_temps.append(entry.current)
                
                if cpu_temps:
                    temperature_cpu = sum(cpu_temps) / len(cpu_temps)
                if gpu_temps:
                    temperature_gpu = sum(gpu_temps) / len(gpu_temps)
                if system_temps:
                    temperature_system = sum(system_temps) / len(system_temps)
        except:
            pass
        
        # 배터리 정보
        battery_percent = battery_power_plugged = battery_time_left = None
        try:
            battery = psutil.sensors_battery()
            if battery:
                battery_percent = battery.percent
                battery_power_plugged = battery.power_plugged
                battery_time_left = battery.secsleft if battery.secsleft != psutil.POWER_TIME_UNLIMITED else None
        except:
            pass
        
        # GPU 메트릭 (추정치)
        gpu_percent = gpu_memory_used = gpu_memory_total = None
        try:
            # 실제 GPU 모니터링은 nvidia-ml-py, pynvml 등 필요
            # 여기서는 CPU 기반 추정
            gpu_percent = min(100, cpu_percent * 0.7 + random.uniform(-5, 5))
            gpu_memory_used = int(memory.used * 0.3)  # 추정
            gpu_memory_total = int(memory.total * 0.25)  # 추정
        except:
            pass
        
        # 네트워크 연결 분석
        network_connections = network_connections_established = 0
        try:
            connections = psutil.net_connections(kind='inet')
            network_connections = len(connections)
            network_connections_established = len([c for c in connections if c.status == 'ESTABLISHED'])
        except:
            pass
        
        # 보안 메트릭 (기본값)
        security_threat_level = 1
        security_active_scans = 0
        security_blocked_connections = 0
        
        # 성능 점수 계산
        performance_score = self._calculate_performance_score(
            cpu_percent, memory.percent, disk_usage.percent
        )
        health_score = self._calculate_health_score(
            cpu_percent, memory.percent, temperature_cpu, uptime_seconds
        )
        efficiency_score = self._calculate_efficiency_score(
            cpu_percent, memory.percent, process_count, thread_count
        )
        
        return ComprehensiveMetrics(
            timestamp=datetime.now(),
            
            # CPU 메트릭
            cpu_percent=cpu_percent,
            cpu_freq_current=cpu_freq.current if cpu_freq else 0,
            cpu_freq_min=cpu_freq.min if cpu_freq else 0,
            cpu_freq_max=cpu_freq.max if cpu_freq else 0,
            cpu_cores_physical=psutil.cpu_count(logical=False),
            cpu_cores_logical=psutil.cpu_count(logical=True),
            cpu_times_user=cpu_times.user,
            cpu_times_system=cpu_times.system,
            cpu_times_idle=cpu_times.idle,
            cpu_times_iowait=getattr(cpu_times, 'iowait', 0),
            cpu_per_core=cpu_per_core,
            
            # 메모리 메트릭
            memory_total=memory.total,
            memory_available=memory.available,
            memory_used=memory.used,
            memory_percent=memory.percent,
            memory_cached=getattr(memory, 'cached', 0),
            memory_buffers=getattr(memory, 'buffers', 0),
            swap_total=swap.total,
            swap_used=swap.used,
            swap_percent=swap.percent,
            
            # 디스크 메트릭
            disk_total=disk_usage.total,
            disk_used=disk_usage.used,
            disk_free=disk_usage.free,
            disk_percent=(disk_usage.used / disk_usage.total) * 100,
            disk_read_bytes=disk_io.read_bytes,
            disk_write_bytes=disk_io.write_bytes,
            disk_read_count=disk_io.read_count,
            disk_write_count=disk_io.write_count,
            disk_read_speed=disk_read_speed,
            disk_write_speed=disk_write_speed,
            
            # 네트워크 메트릭
            network_bytes_sent=network_io.bytes_sent,
            network_bytes_recv=network_io.bytes_recv,
            network_packets_sent=network_io.packets_sent,
            network_packets_recv=network_io.packets_recv,
            network_sent_speed=network_sent_speed,
            network_recv_speed=network_recv_speed,
            network_connections=network_connections,
            network_connections_established=network_connections_established,
            
            # 프로세스 메트릭
            process_count=process_count,
            thread_count=thread_count,
            handle_count=handle_count,
            
            # 시스템 메트릭
            boot_time=boot_time,
            uptime_seconds=uptime_seconds,
            load_average=load_average,
            
            # 하드웨어 메트릭
            temperature_cpu=temperature_cpu,
            temperature_gpu=temperature_gpu,
            temperature_system=temperature_system,
            battery_percent=battery_percent,
            battery_power_plugged=battery_power_plugged,
            battery_time_left=battery_time_left,
            
            # GPU 메트릭
            gpu_percent=gpu_percent,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            gpu_temperature=temperature_gpu,
            
            # 보안 메트릭
            security_threat_level=security_threat_level,
            security_active_scans=security_active_scans,
            security_blocked_connections=security_blocked_connections,
            
            # 성능 점수
            performance_score=performance_score,
            health_score=health_score,
            efficiency_score=efficiency_score
        )
    
    def _calculate_performance_score(self, cpu_percent: float, memory_percent: float, disk_percent: float) -> float:
        """성능 점수 계산"""
        score = 100
        
        # CPU 점수 차감
        if cpu_percent > 90:
            score -= 40
        elif cpu_percent > 70:
            score -= 20
        elif cpu_percent > 50:
            score -= 10
        
        # 메모리 점수 차감
        if memory_percent > 95:
            score -= 35
        elif memory_percent > 85:
            score -= 20
        elif memory_percent > 70:
            score -= 10
        
        # 디스크 점수 차감
        if disk_percent > 98:
            score -= 25
        elif disk_percent > 90:
            score -= 10
        elif disk_percent > 80:
            score -= 5
        
        return max(0, score)
    
    def _calculate_health_score(self, cpu_percent: float, memory_percent: float, 
                               temperature: Optional[float], uptime: float) -> float:
        """건강도 점수 계산"""
        score = 100
        
        # 온도 기반 점수
        if temperature:
            if temperature > 85:
                score -= 30
            elif temperature > 75:
                score -= 15
            elif temperature > 65:
                score -= 5
        
        # 시스템 부하 기반 점수
        load_factor = (cpu_percent + memory_percent) / 2
        if load_factor > 90:
            score -= 25
        elif load_factor > 70:
            score -= 15
        elif load_factor > 50:
            score -= 5
        
        # 업타임 기반 점수 (너무 짧거나 너무 길면 차감)
        hours = uptime / 3600
        if hours < 1:  # 1시간 미만 (불안정)
            score -= 10
        elif hours > 24 * 30:  # 30일 이상 (재시작 필요)
            score -= 5
        
        return max(0, score)
    
    def _calculate_efficiency_score(self, cpu_percent: float, memory_percent: float, 
                                   process_count: int, thread_count: int) -> float:
        """효율성 점수 계산"""
        score = 100
        
        # 리소스 대비 프로세스 수
        if process_count > 300:
            score -= 20
        elif process_count > 200:
            score -= 10
        
        # 스레드 수
        if thread_count > 1000:
            score -= 15
        elif thread_count > 500:
            score -= 5
        
        # CPU와 메모리 밸런스
        balance_diff = abs(cpu_percent - memory_percent)
        if balance_diff > 50:
            score -= 15
        elif balance_diff > 30:
            score -= 8
        
        return max(0, score)

# ============================
# 차세대 3D 렌더러
# ============================

class NextGen3DRenderer:
    """차세대 3D 렌더링 엔진"""
    
    def __init__(self, screen):
        self.screen = screen
        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT
        self.clock = pygame.time.Clock()
        
        # 3D 변환 매트릭스
        self.camera_pos = np.array([0, 0, -5])
        self.camera_rotation = np.array([0, 0, 0])
        
        # 애니메이션 상태
        self.time = 0
        self.pulse_phase = 0
        self.rotation_speed = 1.0
        
        # 홀로그래픽 효과
        self.hologram_intensity = 0.8
        self.scan_line_position = 0
        
        # 파티클 시스템
        self.particles = self._initialize_particles()
        
        # 3D 모델들
        self.cube_vertices = self._generate_cube_vertices()
        self.sphere_vertices = self._generate_sphere_vertices()
        
    def _initialize_particles(self) -> List[Dict]:
        """파티클 시스템 초기화"""
        particles = []
        for _ in range(100):
            particles.append({
                'pos': np.array([
                    random.uniform(-self.width//2, self.width//2),
                    random.uniform(-self.height//2, self.height//2),
                    random.uniform(-200, 200)
                ]),
                'vel': np.array([
                    random.uniform(-50, 50),
                    random.uniform(-50, 50),
                    random.uniform(-50, 50)
                ]),
                'life': random.uniform(0.5, 1.0),
                'size': random.uniform(1, 3),
                'color': random.choice([
                    NextGenColors.HOLO_BLUE,
                    NextGenColors.NEON_CYAN,
                    NextGenColors.PLASMA_GREEN,
                    NextGenColors.QUANTUM_PURPLE
                ])
            })
        return particles
    
    def _generate_cube_vertices(self) -> np.ndarray:
        """큐브 정점 생성"""
        return np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # 뒷면
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # 앞면
        ])
    
    def _generate_sphere_vertices(self) -> np.ndarray:
        """구 정점 생성"""
        vertices = []
        for i in range(20):
            for j in range(20):
                theta = (i / 20) * 2 * math.pi
                phi = (j / 20) * math.pi
                
                x = math.sin(phi) * math.cos(theta)
                y = math.sin(phi) * math.sin(theta)
                z = math.cos(phi)
                
                vertices.append([x, y, z])
        
        return np.array(vertices)
    
    def update(self, dt: float):
        """렌더러 업데이트"""
        self.time += dt
        self.pulse_phase = (self.pulse_phase + dt * 3) % (2 * math.pi)
        self.scan_line_position = (self.scan_line_position + dt * 200) % self.height
        
        # 파티클 업데이트
        for particle in self.particles:
            particle['pos'] += particle['vel'] * dt
            particle['life'] -= dt * 0.2
            
            # 경계 처리
            if (abs(particle['pos'][0]) > self.width//2 or 
                abs(particle['pos'][1]) > self.height//2 or
                particle['life'] <= 0):
                
                # 파티클 재생성
                particle['pos'] = np.array([
                    random.uniform(-self.width//2, self.width//2),
                    random.uniform(-self.height//2, self.height//2),
                    random.uniform(-200, 200)
                ])
                particle['life'] = random.uniform(0.5, 1.0)
    
    def project_3d_to_2d(self, point_3d: np.ndarray, scale: float = 100) -> Tuple[int, int]:
        """3D 점을 2D로 투영"""
        # 간단한 원근 투영
        x, y, z = point_3d
        
        # 카메라 거리
        camera_distance = 300
        
        # 투영
        if z + camera_distance != 0:
            screen_x = int((x * camera_distance) / (z + camera_distance) * scale + self.width // 2)
            screen_y = int((y * camera_distance) / (z + camera_distance) * scale + self.height // 2)
        else:
            screen_x, screen_y = self.width // 2, self.height // 2
        
        return screen_x, screen_y
    
    def rotate_point(self, point: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        """점 회전"""
        x, y, z = point
        rx, ry, rz = rotation
        
        # X축 회전
        cos_x, sin_x = math.cos(rx), math.sin(rx)
        y_new = y * cos_x - z * sin_x
        z_new = y * sin_x + z * cos_x
        y, z = y_new, z_new
        
        # Y축 회전
        cos_y, sin_y = math.cos(ry), math.sin(ry)
        x_new = x * cos_y + z * sin_y
        z_new = -x * sin_y + z * cos_y
        x, z = x_new, z_new
        
        # Z축 회전
        cos_z, sin_z = math.cos(rz), math.sin(rz)
        x_new = x * cos_z - y * sin_z
        y_new = x * sin_z + y * cos_z
        x, y = x_new, y_new
        
        return np.array([x, y, z])
    
    def draw_3d_cube(self, center: Tuple[int, int], size: float, rotation: np.ndarray, color: Tuple[int, int, int]):
        """3D 큐브 그리기"""
        # 정점 회전 및 투영
        projected_vertices = []
        for vertex in self.cube_vertices:
            rotated = self.rotate_point(vertex * size, rotation)
            screen_pos = self.project_3d_to_2d(rotated)
            projected_vertices.append(screen_pos)
        
        # 큐브 엣지 그리기
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 뒷면
            (4, 5), (5, 6), (6, 7), (7, 4),  # 앞면
            (0, 4), (1, 5), (2, 6), (3, 7)   # 연결선
        ]
        
        # 홀로그래픽 효과
        glow_color = NextGenColors.pulse_color(color, self.time)
        
        for start, end in edges:
            if (0 <= projected_vertices[start][0] < self.width and
                0 <= projected_vertices[start][1] < self.height and
                0 <= projected_vertices[end][0] < self.width and
                0 <= projected_vertices[end][1] < self.height):
                
                # 메인 선
                pygame.draw.line(self.screen, glow_color, 
                               projected_vertices[start], projected_vertices[end], 3)
                
                # 글로우 효과
                glow_alpha = int(100 * self.hologram_intensity)
                try:
                    pygame.draw.line(self.screen, (*glow_color[:3], glow_alpha),
                                   projected_vertices[start], projected_vertices[end], 6)
                except:
                    pass
    
    def draw_3d_sphere(self, center: Tuple[int, int], size: float, rotation: np.ndarray, color: Tuple[int, int, int]):
        """3D 구 그리기"""
        projected_points = []
        
        for vertex in self.sphere_vertices:
            rotated = self.rotate_point(vertex * size, rotation)
            screen_pos = self.project_3d_to_2d(rotated)
            projected_points.append(screen_pos)
        
        # 점들을 연결하여 구 표면 그리기
        glow_color = NextGenColors.pulse_color(color, self.time)
        
        for i, point in enumerate(projected_points):
            if (0 <= point[0] < self.width and 0 <= point[1] < self.height):
                pygame.draw.circle(self.screen, glow_color, point, 2)
    
    def draw_holographic_grid(self):
        """홀로그래픽 그리드 그리기"""
        grid_spacing = 50
        grid_color = NextGenColors.with_alpha(NextGenColors.HOLO_BLUE, 100)
        
        # 수직선
        for x in range(0, self.width, grid_spacing):
            alpha = int(50 + 30 * math.sin(self.time + x * 0.01))
            color = (*NextGenColors.HOLO_BLUE[:3], alpha)
            try:
                for y in range(0, self.height, 5):
                    pygame.draw.circle(self.screen, NextGenColors.HOLO_BLUE, (x, y), 1)
            except:
                pygame.draw.line(self.screen, NextGenColors.HOLO_BLUE, (x, 0), (x, self.height), 1)
        
        # 수평선
        for y in range(0, self.height, grid_spacing):
            alpha = int(50 + 30 * math.sin(self.time + y * 0.01))
            try:
                for x in range(0, self.width, 5):
                    pygame.draw.circle(self.screen, NextGenColors.HOLO_BLUE, (x, y), 1)
            except:
                pygame.draw.line(self.screen, NextGenColors.HOLO_BLUE, (0, y), (self.width, y), 1)
    
    def draw_scan_lines(self):
        """스캔라인 효과"""
        scan_color = NextGenColors.with_alpha(NextGenColors.NEON_CYAN, 150)
        
        # 수평 스캔라인
        for i in range(3):
            y = int(self.scan_line_position + i * 100) % self.height
            try:
                pygame.draw.line(self.screen, NextGenColors.NEON_CYAN, (0, y), (self.width, y), 2)
            except:
                pass
    
    def draw_particles(self):
        """파티클 시스템 렌더링"""
        for particle in self.particles:
            if particle['life'] > 0:
                screen_pos = self.project_3d_to_2d(particle['pos'])
                
                if (0 <= screen_pos[0] < self.width and 0 <= screen_pos[1] < self.height):
                    alpha = int(255 * particle['life'])
                    size = int(particle['size'] * particle['life'])
                    
                    # 파티클 그리기
                    color = particle['color']
                    pygame.draw.circle(self.screen, color, screen_pos, max(1, size))
                    
                    # 글로우 효과
                    if size > 1:
                        glow_color = (*color[:3], alpha // 2)
                        try:
                            pygame.draw.circle(self.screen, color, screen_pos, size + 2)
                        except:
                            pass

# ============================
# 통합 시각화 대시보드
# ============================

class NextGenDashboard:
    """차세대 통합 시각화 대시보드"""
    
    def __init__(self, screen):
        self.screen = screen
        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT
        self.renderer_3d = NextGen3DRenderer(screen)
        
        # 레이아웃 설정
        self.setup_layout()
        
        # 데이터 히스토리
        self.cpu_history = deque(maxlen=200)
        self.memory_history = deque(maxlen=200)
        self.network_history = deque(maxlen=200)
        self.temperature_history = deque(maxlen=200)
        
        # 차트 표면 캐시
        self.chart_cache = {}
        self.cache_timestamp = {}
        
    def setup_layout(self):
        """레이아웃 설정"""
        # 5x4 그리드 레이아웃
        grid_w = self.width // 5
        grid_h = self.height // 4
        
        self.layout = {
            # 첫 번째 행 - 주요 메트릭 게이지
            'cpu_gauge': (0, 0, grid_w, grid_h),
            'memory_gauge': (grid_w, 0, grid_w, grid_h),
            'disk_gauge': (grid_w * 2, 0, grid_w, grid_h),
            'gpu_gauge': (grid_w * 3, 0, grid_w, grid_h),
            'network_gauge': (grid_w * 4, 0, grid_w, grid_h),
            
            # 두 번째 행 - 실시간 그래프
            'cpu_graph': (0, grid_h, grid_w * 2, grid_h),
            'memory_graph': (grid_w * 2, grid_h, grid_w * 2, grid_h),
            'network_graph': (grid_w * 4, grid_h, grid_w, grid_h),
            
            # 세 번째 행 - 3D 시각화 및 분석
            '3d_visualization': (0, grid_h * 2, grid_w * 2, grid_h),
            'ai_analysis': (grid_w * 2, grid_h * 2, grid_w * 2, grid_h),
            'security_status': (grid_w * 4, grid_h * 2, grid_w, grid_h),
            
            # 네 번째 행 - 상세 정보
            'process_list': (0, grid_h * 3, grid_w * 2, grid_h),
            'system_info': (grid_w * 2, grid_h * 3, grid_w * 2, grid_h),
            'alerts_panel': (grid_w * 4, grid_h * 3, grid_w, grid_h)
        }
    
    def draw_holographic_gauge(self, rect: Tuple[int, int, int, int], value: float, max_value: float, 
                              label: str, color: Tuple[int, int, int], unit: str = ""):
        """홀로그래픽 원형 게이지"""
        x, y, w, h = rect
        center_x, center_y = x + w // 2, y + h // 2
        radius = min(w, h) // 3
        
        # 배경 원
        pygame.draw.circle(self.screen, NextGenColors.DARK_MATTER, (center_x, center_y), radius + 5, 2)
        
        # 값에 따른 호 그리기
        if value > 0:
            angle = (value / max_value) * 2 * math.pi
            
            # 세그먼트 그리기
            segments = max(1, int(angle * 30))
            for i in range(segments):
                segment_angle = (i / 30) * 2 * math.pi - math.pi / 2
                next_angle = ((i + 1) / 30) * 2 * math.pi - math.pi / 2
                
                start_x = center_x + (radius - 8) * math.cos(segment_angle)
                start_y = center_y + (radius - 8) * math.sin(segment_angle)
                end_x = center_x + (radius - 8) * math.cos(next_angle)
                end_y = center_y + (radius - 8) * math.sin(next_angle)
                
                # 세그먼트 색상 (값에 따라)
                segment_color = color
                if value > 80:
                    segment_color = NextGenColors.LASER_RED
                elif value > 60:
                    segment_color = NextGenColors.ENERGY_ORANGE
                
                # 홀로그래픽 효과
                pulse_intensity = 0.7 + 0.3 * math.sin(self.renderer_3d.time * 5 + i * 0.1)
                final_color = tuple(int(c * pulse_intensity) for c in segment_color)
                
                pygame.draw.line(self.screen, final_color, (start_x, start_y), (end_x, end_y), 6)
        
        # 중앙 값 표시
        value_text = fonts['medium'].render(f"{value:.1f}{unit}", True, NextGenColors.WHITE)
        value_rect = value_text.get_rect(center=(center_x, center_y - 10))
        self.screen.blit(value_text, value_rect)
        
        # 라벨
        label_text = fonts['small'].render(label, True, color)
        label_rect = label_text.get_rect(center=(center_x, center_y + 20))
        self.screen.blit(label_text, label_rect)
        
        # 최대값 표시
        max_text = fonts['tiny'].render(f"Max: {max_value:.0f}", True, NextGenColors.NEBULA_BLUE)
        max_rect = max_text.get_rect(center=(center_x, center_y + 35))
        self.screen.blit(max_text, max_rect)
    
    def draw_holographic_graph(self, rect: Tuple[int, int, int, int], data: List[float], 
                              color: Tuple[int, int, int], label: str, max_value: float = None):
        """홀로그래픽 선 그래프"""
        x, y, w, h = rect
        
        # 배경
        bg_color = NextGenColors.with_alpha(NextGenColors.DEEP_SPACE, 150)
        pygame.draw.rect(self.screen, NextGenColors.DEEP_SPACE, rect)
        pygame.draw.rect(self.screen, NextGenColors.HOLO_BLUE, rect, 2)
        
        if not data or len(data) < 2:
            # 데이터 없음 표시
            no_data_text = fonts['small'].render("수집 중...", True, NextGenColors.NEBULA_BLUE)
            text_rect = no_data_text.get_rect(center=(x + w // 2, y + h // 2))
            self.screen.blit(no_data_text, text_rect)
            return
        
        # 최대값 계산
        if max_value is None:
            max_value = max(max(data), 1)
        
        # 그리드 라인
        grid_lines = 5
        for i in range(grid_lines + 1):
            grid_y = y + (h * i // grid_lines)
            alpha = 50 + 20 * math.sin(self.renderer_3d.time + i)
            grid_color = (*NextGenColors.HOLO_BLUE[:3], int(alpha))
            
            pygame.draw.line(self.screen, NextGenColors.NEBULA_BLUE, (x, grid_y), (x + w, grid_y), 1)
            
            # Y축 라벨
            if i < grid_lines:
                value = max_value * (1 - i / grid_lines)
                label_text = fonts['tiny'].render(f"{value:.0f}", True, NextGenColors.NEON_CYAN)
                self.screen.blit(label_text, (x + 5, grid_y + 2))
        
        # 데이터 포인트 계산
        points = []
        for i, value in enumerate(data):
            point_x = x + (w * i // (len(data) - 1))
            point_y = y + h - int((value / max_value) * h)
            points.append((point_x, point_y))
        
        # 홀로그래픽 선 그리기
        if len(points) > 1:
            # 메인 선
            pygame.draw.lines(self.screen, color, False, points, 3)
            
            # 글로우 효과
            glow_color = NextGenColors.pulse_color(color, self.renderer_3d.time)
            for i in range(len(points) - 1):
                pygame.draw.line(self.screen, glow_color, points[i], points[i + 1], 5)
            
            # 데이터 포인트 강조
            for point in points[::5]:  # 5개마다 포인트 표시
                pygame.draw.circle(self.screen, NextGenColors.WHITE, point, 3)
                pygame.draw.circle(self.screen, color, point, 2)
        
        # 현재값 표시
        if data:
            current_value = data[-1]
            current_text = fonts['small'].render(f"{label}: {current_value:.1f}", True, color)
            self.screen.blit(current_text, (x + 10, y + 10))
    
    def draw_3d_system_visualization(self, rect: Tuple[int, int, int, int], metrics: ComprehensiveMetrics):
        """3D 시스템 시각화"""
        x, y, w, h = rect
        
        # 배경
        pygame.draw.rect(self.screen, NextGenColors.DEEP_SPACE, rect)
        pygame.draw.rect(self.screen, NextGenColors.QUANTUM_PURPLE, rect, 2)
        
        # 3D 시각화 영역 설정
        self.renderer_3d.width = w
        self.renderer_3d.height = h
        
        # 중앙 위치
        center_x, center_y = x + w // 2, y + h // 2
        
        # CPU 큐브 (중앙)
        cpu_rotation = np.array([
            self.renderer_3d.time * 0.5,
            self.renderer_3d.time * 0.3,
            self.renderer_3d.time * 0.7
        ])
        cpu_color = NextGenColors.HOLO_BLUE
        if metrics.cpu_percent > 80:
            cpu_color = NextGenColors.LASER_RED
        elif metrics.cpu_percent > 60:
            cpu_color = NextGenColors.ENERGY_ORANGE
        
        self.renderer_3d.draw_3d_cube(
            (center_x, center_y), 
            30 + metrics.cpu_percent * 0.5, 
            cpu_rotation, 
            cpu_color
        )
        
        # 메모리 구 (좌측)
        memory_rotation = np.array([
            self.renderer_3d.time * 0.3,
            self.renderer_3d.time * 0.8,
            self.renderer_3d.time * 0.4
        ])
        memory_color = NextGenColors.PLASMA_GREEN
        if metrics.memory_percent > 85:
            memory_color = NextGenColors.LASER_RED
        
        self.renderer_3d.draw_3d_sphere(
            (center_x - 100, center_y), 
            20 + metrics.memory_percent * 0.3, 
            memory_rotation, 
            memory_color
        )
        
        # 디스크 큐브 (우측)
        disk_rotation = np.array([
            self.renderer_3d.time * 0.2,
            self.renderer_3d.time * 0.6,
            self.renderer_3d.time * 0.9
        ])
        disk_color = NextGenColors.NEON_CYAN
        if metrics.disk_percent > 90:
            disk_color = NextGenColors.LASER_RED
        
        self.renderer_3d.draw_3d_cube(
            (center_x + 100, center_y), 
            25 + metrics.disk_percent * 0.2, 
            disk_rotation, 
            disk_color
        )
        
        # 3D 라벨
        cpu_label = fonts['tiny'].render(f"CPU: {metrics.cpu_percent:.1f}%", True, cpu_color)
        self.screen.blit(cpu_label, (center_x - 40, y + h - 60))
        
        memory_label = fonts['tiny'].render(f"RAM: {metrics.memory_percent:.1f}%", True, memory_color)
        self.screen.blit(memory_label, (center_x - 140, y + h - 40))
        
        disk_label = fonts['tiny'].render(f"DISK: {metrics.disk_percent:.1f}%", True, disk_color)
        self.screen.blit(disk_label, (center_x + 60, y + h - 40))
        
        # 성능 점수 표시
        score_color = NextGenColors.NEON_CYAN
        if metrics.performance_score < 50:
            score_color = NextGenColors.LASER_RED
        elif metrics.performance_score < 70:
            score_color = NextGenColors.ENERGY_ORANGE
        
        score_text = fonts['medium'].render(f"SCORE: {metrics.performance_score:.0f}", True, score_color)
        self.screen.blit(score_text, (x + 10, y + 10))
    
    def draw_ai_analysis_panel(self, rect: Tuple[int, int, int, int], ai_result: AIAnalysisResult):
        """AI 분석 패널"""
        x, y, w, h = rect
        
        # 배경
        pygame.draw.rect(self.screen, NextGenColors.DEEP_SPACE, rect)
        pygame.draw.rect(self.screen, NextGenColors.PLASMA_GREEN, rect, 2)
        
        # 타이틀
        title_text = fonts['medium'].render("🧠 AI ANALYSIS", True, NextGenColors.PLASMA_GREEN)
        self.screen.blit(title_text, (x + 10, y + 10))
        
        current_y = y + 45
        
        # 이상 탐지 결과
        anomaly_color = NextGenColors.LASER_RED if ai_result.is_anomaly else NextGenColors.NEON_CYAN
        anomaly_text = fonts['small'].render(
            f"Anomaly: {'DETECTED' if ai_result.is_anomaly else 'NORMAL'}", 
            True, anomaly_color
        )
        self.screen.blit(anomaly_text, (x + 10, current_y))
        current_y += 25
        
        # 예측 결과
        pred_text = fonts['small'].render(
            f"CPU Pred: {ai_result.prediction_cpu:.1f}%", 
            True, NextGenColors.HOLO_BLUE
        )
        self.screen.blit(pred_text, (x + 10, current_y))
        current_y += 20
        
        pred_mem_text = fonts['small'].render(
            f"RAM Pred: {ai_result.prediction_memory:.1f}%", 
            True, NextGenColors.QUANTUM_PURPLE
        )
        self.screen.blit(pred_mem_text, (x + 10, current_y))
        current_y += 25
        
        # 트렌드
        trend_color = (NextGenColors.LASER_RED if ai_result.performance_trend == 'increasing' 
                      else NextGenColors.NEON_CYAN if ai_result.performance_trend == 'decreasing'
                      else NextGenColors.PLASMA_GREEN)
        
        trend_text = fonts['small'].render(f"Trend: {ai_result.performance_trend.upper()}", True, trend_color)
        self.screen.blit(trend_text, (x + 10, current_y))
        current_y += 25
        
        # 최적화 제안
        if ai_result.optimization_suggestions:
            suggestions_text = fonts['tiny'].render("Optimization:", True, NextGenColors.ENERGY_ORANGE)
            self.screen.blit(suggestions_text, (x + 10, current_y))
            current_y += 18
            
            for i, suggestion in enumerate(ai_result.optimization_suggestions[:3]):
                # 텍스트가 너무 길면 자르기
                if len(suggestion) > 35:
                    suggestion = suggestion[:32] + "..."
                
                sugg_text = fonts['tiny'].render(f"• {suggestion}", True, NextGenColors.WHITE)
                self.screen.blit(sugg_text, (x + 15, current_y))
                current_y += 15
        
        # 신뢰도 표시
        confidence_bar_width = w - 40
        confidence_bar_height = 8
        confidence_x = x + 20
        confidence_y = y + h - 25
        
        # 신뢰도 바 배경
        pygame.draw.rect(self.screen, NextGenColors.DARK_MATTER, 
                        (confidence_x, confidence_y, confidence_bar_width, confidence_bar_height))
        
        # 신뢰도 바
        confidence_width = int(confidence_bar_width * ai_result.prediction_confidence)
        confidence_color = NextGenColors.NEON_CYAN
        pygame.draw.rect(self.screen, confidence_color,
                        (confidence_x, confidence_y, confidence_width, confidence_bar_height))
        
        # 신뢰도 텍스트
        conf_text = fonts['tiny'].render(f"Confidence: {ai_result.prediction_confidence:.1%}", 
                                        True, NextGenColors.WHITE)
        self.screen.blit(conf_text, (confidence_x, confidence_y - 15))
    
    def draw_security_status_panel(self, rect: Tuple[int, int, int, int], threats: List[SecurityThreat]):
        """보안 상태 패널"""
        x, y, w, h = rect
        
        # 배경
        pygame.draw.rect(self.screen, NextGenColors.DEEP_SPACE, rect)
        
        # 보안 레벨에 따른 테두리 색상
        threat_levels = [threat.level for threat in threats]
        if 'CRITICAL' in threat_levels:
            border_color = NextGenColors.LASER_RED
            status = "CRITICAL"
        elif 'HIGH' in threat_levels:
            border_color = NextGenColors.ENERGY_ORANGE
            status = "HIGH"
        elif 'MEDIUM' in threat_levels:
            border_color = NextGenColors.CYBER_YELLOW
            status = "MEDIUM"
        elif threats:
            border_color = NextGenColors.HOLO_BLUE
            status = "LOW"
        else:
            border_color = NextGenColors.NEON_CYAN
            status = "SECURE"
        
        pygame.draw.rect(self.screen, border_color, rect, 3)
        
        # 타이틀
        title_text = fonts['medium'].render("🛡️ SECURITY", True, border_color)
        self.screen.blit(title_text, (x + 10, y + 10))
        
        # 상태 표시
        status_text = fonts['small'].render(status, True, border_color)
        self.screen.blit(status_text, (x + 10, y + 40))
        
        # 위협 카운트
        threat_count_text = fonts['small'].render(f"Threats: {len(threats)}", True, NextGenColors.WHITE)
        self.screen.blit(threat_count_text, (x + 10, y + 65))
        
        # 최근 위협 표시
        if threats:
            recent_threats = sorted(threats, key=lambda t: t.timestamp, reverse=True)[:3]
            current_y = y + 90
            
            for threat in recent_threats:
                # 위협 레벨 아이콘
                level_icons = {
                    'CRITICAL': '🚨',
                    'HIGH': '⚠️',
                    'MEDIUM': '⚡',
                    'LOW': '💡'
                }
                icon = level_icons.get(threat.level, '📊')
                
                # 위협 설명 (축약)
                description = threat.description
                if len(description) > 25:
                    description = description[:22] + "..."
                
                threat_text = fonts['tiny'].render(f"{icon} {description}", True, NextGenColors.WHITE)
                self.screen.blit(threat_text, (x + 10, current_y))
                current_y += 15
        
        # 보안 스코어 (가상)
        security_score = max(0, 100 - len(threats) * 10)
        score_color = (NextGenColors.NEON_CYAN if security_score > 80 
                      else NextGenColors.ENERGY_ORANGE if security_score > 60 
                      else NextGenColors.LASER_RED)
        
        score_text = fonts['small'].render(f"Score: {security_score}", True, score_color)
        self.screen.blit(score_text, (x + 10, y + h - 25))
    
    def draw_process_list(self, rect: Tuple[int, int, int, int]):
        """프로세스 목록"""
        x, y, w, h = rect
        
        # 배경
        pygame.draw.rect(self.screen, NextGenColors.DEEP_SPACE, rect)
        pygame.draw.rect(self.screen, NextGenColors.HOLO_BLUE, rect, 2)
        
        # 타이틀
        title_text = fonts['medium'].render("⚙️ TOP PROCESSES", True, NextGenColors.HOLO_BLUE)
        self.screen.blit(title_text, (x + 10, y + 10))
        
        # 헤더
        header_y = y + 40
        header_text = fonts['tiny'].render("NAME              CPU%   RAM%", True, NextGenColors.NEON_CYAN)
        self.screen.blit(header_text, (x + 10, header_y))
        
        # 프로세스 정보 수집
        try:
            processes = []
            for proc in psutil.process_iter(['name', 'cpu_percent', 'memory_percent']):
                try:
                    proc_info = proc.info
                    if proc_info['cpu_percent'] > 0 or proc_info['memory_percent'] > 0:
                        processes.append(proc_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # CPU 사용률 기준 정렬
            processes.sort(key=lambda p: p['cpu_percent'], reverse=True)
            
            # 상위 프로세스 표시
            current_y = header_y + 20
            for i, proc in enumerate(processes[:8]):  # 상위 8개
                if current_y + 15 > y + h:
                    break
                
                # 프로세스 이름 축약
                name = proc['name'][:12]
                cpu_pct = proc['cpu_percent']
                mem_pct = proc['memory_percent']
                
                # 색상 결정
                if cpu_pct > 50:
                    color = NextGenColors.LASER_RED
                elif cpu_pct > 20:
                    color = NextGenColors.ENERGY_ORANGE
                else:
                    color = NextGenColors.WHITE
                
                proc_text = fonts['tiny'].render(
                    f"{name:<12} {cpu_pct:5.1f}% {mem_pct:5.1f}%", 
                    True, color
                )
                self.screen.blit(proc_text, (x + 10, current_y))
                current_y += 15
        
        except Exception as e:
            error_text = fonts['tiny'].render("Process scan error", True, NextGenColors.LASER_RED)
            self.screen.blit(error_text, (x + 10, header_y + 20))
    
    def draw_system_info_panel(self, rect: Tuple[int, int, int, int], metrics: ComprehensiveMetrics):
        """시스템 정보 패널"""
        x, y, w, h = rect
        
        # 배경
        pygame.draw.rect(self.screen, NextGenColors.DEEP_SPACE, rect)
        pygame.draw.rect(self.screen, NextGenColors.QUANTUM_PURPLE, rect, 2)
        
        # 타이틀
        title_text = fonts['medium'].render("💻 SYSTEM INFO", True, NextGenColors.QUANTUM_PURPLE)
        self.screen.blit(title_text, (x + 10, y + 10))
        
        current_y = y + 45
        line_height = 18
        
        # 시스템 정보
        info_items = [
            ("OS:", platform.system()),
            ("CPU Cores:", f"{metrics.cpu_cores_physical}P/{metrics.cpu_cores_logical}L"),
            ("CPU Freq:", f"{metrics.cpu_freq_current:.0f} MHz"),
            ("Memory:", f"{metrics.memory_total // (1024**3):.1f} GB"),
            ("Uptime:", f"{metrics.uptime_seconds / 3600:.1f} hours"),
            ("Processes:", f"{metrics.process_count}"),
            ("Threads:", f"{metrics.thread_count}"),
        ]
        
        if metrics.temperature_cpu:
            info_items.append(("CPU Temp:", f"{metrics.temperature_cpu:.1f}°C"))
        
        if metrics.battery_percent is not None:
            info_items.append(("Battery:", f"{metrics.battery_percent:.0f}%"))
        
        for label, value in info_items:
            if current_y + line_height > y + h:
                break
                
            label_text = fonts['tiny'].render(label, True, NextGenColors.NEON_CYAN)
            value_text = fonts['tiny'].render(str(value), True, NextGenColors.WHITE)
            
            self.screen.blit(label_text, (x + 10, current_y))
            self.screen.blit(value_text, (x + 80, current_y))
            current_y += line_height
        
        # 성능 점수들
        current_y += 10
        scores = [
            ("Performance:", f"{metrics.performance_score:.0f}/100"),
            ("Health:", f"{metrics.health_score:.0f}/100"),
            ("Efficiency:", f"{metrics.efficiency_score:.0f}/100")
        ]
        
        for label, value in scores:
            if current_y + line_height > y + h:
                break
                
            # 점수에 따른 색상
            score_val = float(value.split('/')[0])
            score_color = (NextGenColors.NEON_CYAN if score_val > 80 
                          else NextGenColors.ENERGY_ORANGE if score_val > 60 
                          else NextGenColors.LASER_RED)
            
            label_text = fonts['tiny'].render(label, True, NextGenColors.PLASMA_GREEN)
            value_text = fonts['tiny'].render(value, True, score_color)
            
            self.screen.blit(label_text, (x + 10, current_y))
            self.screen.blit(value_text, (x + 80, current_y))
            current_y += line_height
    
    def update_histories(self, metrics: ComprehensiveMetrics):
        """히스토리 데이터 업데이트"""
        self.cpu_history.append(metrics.cpu_percent)
        self.memory_history.append(metrics.memory_percent)
        self.network_history.append((metrics.network_sent_speed + metrics.network_recv_speed) / (1024 * 1024))  # MB/s
        
        if metrics.temperature_cpu:
            self.temperature_history.append(metrics.temperature_cpu)
    
    def render(self, metrics: ComprehensiveMetrics, ai_result: AIAnalysisResult, threats: List[SecurityThreat]):
        """메인 렌더링"""
        # 홀로그래픽 배경 효과
        self.renderer_3d.draw_holographic_grid()
        self.renderer_3d.draw_scan_lines()
        self.renderer_3d.draw_particles()
        
        # 히스토리 업데이트
        self.update_histories(metrics)
        
        # 게이지들
        self.draw_holographic_gauge(
            self.layout['cpu_gauge'], 
            metrics.cpu_percent, 100, 
            "CPU", NextGenColors.HOLO_BLUE, "%"
        )
        
        self.draw_holographic_gauge(
            self.layout['memory_gauge'], 
            metrics.memory_percent, 100, 
            "MEMORY", NextGenColors.QUANTUM_PURPLE, "%"
        )
        
        self.draw_holographic_gauge(
            self.layout['disk_gauge'], 
            metrics.disk_percent, 100, 
            "DISK", NextGenColors.NEON_CYAN, "%"
        )
        
        if metrics.gpu_percent:
            self.draw_holographic_gauge(
                self.layout['gpu_gauge'], 
                metrics.gpu_percent, 100, 
                "GPU", NextGenColors.ENERGY_ORANGE, "%"
            )
        
        # 네트워크 속도 게이지
        network_speed = (metrics.network_sent_speed + metrics.network_recv_speed) / (1024 * 1024)  # MB/s
        self.draw_holographic_gauge(
            self.layout['network_gauge'], 
            network_speed, 100, 
            "NET", NextGenColors.PLASMA_GREEN, "MB/s"
        )
        
        # 그래프들
        self.draw_holographic_graph(
            self.layout['cpu_graph'], 
            list(self.cpu_history), 
            NextGenColors.HOLO_BLUE, 
            "CPU Usage", 100
        )
        
        self.draw_holographic_graph(
            self.layout['memory_graph'], 
            list(self.memory_history), 
            NextGenColors.QUANTUM_PURPLE, 
            "Memory Usage", 100
        )
        
        self.draw_holographic_graph(
            self.layout['network_graph'], 
            list(self.network_history), 
            NextGenColors.PLASMA_GREEN, 
            "Network", 50
        )
        
        # 3D 시각화
        self.draw_3d_system_visualization(self.layout['3d_visualization'], metrics)
        
        # AI 분석 패널
        self.draw_ai_analysis_panel(self.layout['ai_analysis'], ai_result)
        
        # 보안 상태 패널
        self.draw_security_status_panel(self.layout['security_status'], threats)
        
        # 프로세스 목록
        self.draw_process_list(self.layout['process_list'])
        
        # 시스템 정보
        self.draw_system_info_panel(self.layout['system_info'], metrics)

# ============================
# 메인 애플리케이션
# ============================

class SysWatchNextGenUltimate:
    """차세대 통합 모니터링 시스템 메인 클래스"""
    
    def __init__(self):
        # 컴포넌트 초기화
        self.data_collector = NextGenDataCollector()
        self.ai_engine = NextGenAIEngine()
        self.security_engine = NextGenSecurityEngine()
        self.dashboard = NextGenDashboard(screen)
        
        # 실행 상태
        self.running = False
        self.paused = False
        
        # 성능 통계
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps_history = deque(maxlen=60)
        
        # 스레드 풀
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        # 설정
        self.target_fps = 60
        self.update_interval = 1.0 / self.target_fps
        
    def handle_events(self):
        """이벤트 처리"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_F11:
                    # 전체화면 토글 (이미 전체화면)
                    pass
                elif event.key == pygame.K_r:
                    # 리셋
                    self.ai_engine.metrics_history.clear()
                    self.ai_engine.analysis_results.clear()
                    self.security_engine.threats.clear()
                elif event.key == pygame.K_s:
                    # 스크린샷
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    pygame.image.save(screen, f"syswatch_screenshot_{timestamp}.png")
                    print(f"Screenshot saved: syswatch_screenshot_{timestamp}.png")
    
    def update_performance_stats(self):
        """성능 통계 업데이트"""
        current_time = time.time()
        
        # FPS 계산
        if current_time - self.last_fps_time >= 1.0:
            fps = self.frame_count / (current_time - self.last_fps_time)
            self.fps_history.append(fps)
            self.frame_count = 0
            self.last_fps_time = current_time
        
        self.frame_count += 1
    
    def display_controls_overlay(self):
        """컨트롤 오버레이 표시"""
        overlay_height = 100
        overlay_rect = (0, SCREEN_HEIGHT - overlay_height, SCREEN_WIDTH, overlay_height)
        
        # 반투명 배경
        overlay_surface = pygame.Surface((SCREEN_WIDTH, overlay_height))
        overlay_surface.set_alpha(200)
        overlay_surface.fill(NextGenColors.DEEP_SPACE)
        screen.blit(overlay_surface, (0, SCREEN_HEIGHT - overlay_height))
        
        # 테두리
        pygame.draw.rect(screen, NextGenColors.HOLO_BLUE, overlay_rect, 2)
        
        # 컨트롤 텍스트
        controls = [
            "ESC/Q: 종료",
            "SPACE: 일시정지",
            "R: 리셋",
            "S: 스크린샷"
        ]
        
        x_start = 20
        y_start = SCREEN_HEIGHT - 80
        
        for i, control in enumerate(controls):
            control_text = fonts['tiny'].render(control, True, NextGenColors.NEON_CYAN)
            screen.blit(control_text, (x_start + (i * 200), y_start))
        
        # FPS 표시
        if self.fps_history:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            fps_color = (NextGenColors.NEON_CYAN if avg_fps > 55 
                        else NextGenColors.ENERGY_ORANGE if avg_fps > 30 
                        else NextGenColors.LASER_RED)
            
            fps_text = fonts['small'].render(f"FPS: {avg_fps:.1f}", True, fps_color)
            screen.blit(fps_text, (SCREEN_WIDTH - 120, SCREEN_HEIGHT - 80))
        
        # 상태 표시
        status_text = "PAUSED" if self.paused else "RUNNING"
        status_color = NextGenColors.ENERGY_ORANGE if self.paused else NextGenColors.NEON_CYAN
        status_render = fonts['small'].render(status_text, True, status_color)
        screen.blit(status_render, (SCREEN_WIDTH - 120, SCREEN_HEIGHT - 50))
    
    def run(self):
        """메인 실행 루프"""
        print("🚀 SysWatch NextGen Ultimate 시작")
        print("🎮 컨트롤:")
        print("   ESC/Q: 종료")
        print("   SPACE: 일시정지/재개")
        print("   R: 데이터 리셋")
        print("   S: 스크린샷")
        print("🖥️ 전체화면 60fps 실시간 모니터링 시작...\n")
        
        self.running = True
        last_update_time = time.time()
        
        # 초기 데이터 수집
        try:
            initial_metrics = self.data_collector.collect_comprehensive_metrics()
            initial_ai_result = self.ai_engine.analyze_metrics(initial_metrics)
            initial_threats = []
        except Exception as e:
            print(f"초기화 오류: {e}")
            return
        
        while self.running:
            current_time = time.time()
            dt = current_time - last_update_time
            
            # 이벤트 처리
            self.handle_events()
            
            # 업데이트 (일시정지가 아닐 때만)
            if not self.paused:
                try:
                    # 시스템 메트릭 수집
                    metrics = self.data_collector.collect_comprehensive_metrics()
                    
                    # AI 분석 (비동기)
                    ai_result = self.ai_engine.analyze_metrics(metrics)
                    
                    # 보안 스캔 (5초마다)
                    if int(current_time) % 5 == 0:
                        threats_future = self.executor.submit(self.security_engine.comprehensive_security_scan)
                        try:
                            threats = threats_future.result(timeout=0.1)
                        except concurrent.futures.TimeoutError:
                            threats = list(self.security_engine.threats)
                    else:
                        threats = list(self.security_engine.threats)
                    
                except Exception as e:
                    print(f"업데이트 오류: {e}")
                    # 기본값 사용
                    metrics = initial_metrics
                    ai_result = initial_ai_result
                    threats = []
            else:
                # 일시정지 중에는 마지막 데이터 사용
                metrics = initial_metrics
                ai_result = initial_ai_result
                threats = []
            
            # 3D 렌더러 업데이트
            self.dashboard.renderer_3d.update(dt)
            
            # 화면 클리어
            screen.fill(NextGenColors.DEEP_SPACE)
            
            # 메인 대시보드 렌더링
            try:
                self.dashboard.render(metrics, ai_result, threats)
            except Exception as e:
                print(f"렌더링 오류: {e}")
                # 오류 메시지 표시
                error_text = fonts['large'].render("RENDERING ERROR", True, NextGenColors.LASER_RED)
                error_rect = error_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
                screen.blit(error_text, error_rect)
            
            # 컨트롤 오버레이
            self.display_controls_overlay()
            
            # 화면 업데이트
            pygame.display.flip()
            
            # 성능 통계 업데이트
            self.update_performance_stats()
            
            # FPS 제한
            self.dashboard.renderer_3d.clock.tick(self.target_fps)
            last_update_time = current_time
        
        # 정리
        self.cleanup()
    
    def cleanup(self):
        """정리 작업"""
        print("\n🛑 SysWatch NextGen Ultimate 종료 중...")
        
        try:
            # 스레드 풀 종료
            self.executor.shutdown(wait=True)
            
            # 데이터베이스 연결 종료
            if hasattr(self.security_engine, 'conn'):
                self.security_engine.conn.close()
            
            # 최종 통계
            if self.ai_engine.metrics_history:
                print(f"📊 총 {len(self.ai_engine.metrics_history)}개 메트릭 수집")
                print(f"🧠 총 {len(self.ai_engine.analysis_results)}개 AI 분석")
                print(f"🛡️ 총 {len(self.security_engine.threats)}개 보안 이벤트")
            
            if self.fps_history:
                avg_fps = sum(self.fps_history) / len(self.fps_history)
                print(f"🎮 평균 FPS: {avg_fps:.1f}")
            
        except Exception as e:
            print(f"정리 중 오류: {e}")
        
        finally:
            pygame.quit()
            print("\n🌟 SysWatch NextGen Ultimate 종료 완료")
            print("차세대 통합 모니터링을 경험해주셔서 감사합니다!")

# ============================
# 진입점
# ============================

def main():
    """메인 함수"""
    try:
        print("🎮 Pygame 초기화...")
        print(f"🖥️ 해상도: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
        print("🚀 SysWatch NextGen Ultimate 로딩...")
        
        # 시스템 요구사항 확인
        if SCREEN_WIDTH < 1024 or SCREEN_HEIGHT < 768:
            print("⚠️ 경고: 최소 해상도 1024x768을 권장합니다.")
        
        # 메인 애플리케이션 실행
        app = SysWatchNextGenUltimate()
        app.run()
        
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 중단됨")
    except Exception as e:
        print(f"\n❌ 치명적 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            pygame.quit()
        except:
            pass

if __name__ == "__main__":
    main()