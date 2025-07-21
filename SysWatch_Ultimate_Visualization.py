#!/usr/bin/env python3
"""
SysWatch Pro Quantum Ultimate - 60fps 전체화면 실시간 시각화 인터페이스
모든 데이터 시각화, 실시간 그래프, 원형 차트, 3D 렌더링

🚀 60fps 홀로그래픽 시각화 | 🧠 실시간 AI 분석 | 🛡️ 보안 모니터링 | 📊 종합 대시보드

Copyright (C) 2025 SysWatch Technologies Ltd.
Ultimate Visualization Edition
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
import platform
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import deque, defaultdict
import logging

warnings.filterwarnings('ignore')

# Auto-install required packages
def auto_install(package):
    try:
        __import__(package)
    except ImportError:
        print(f"📦 Installing {package}...")
        os.system(f"{sys.executable} -m pip install {package} --quiet")

# Install essential packages
packages = ['psutil', 'numpy', 'matplotlib', 'pygame', 'pillow', 'pandas']
for pkg in packages:
    auto_install(pkg)

import numpy as np
import pandas as pd
import psutil
import pygame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

# Optional packages
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    HAS_ML = True
except ImportError:
    HAS_ML = False

print("✅ All dependencies loaded successfully!")

# ============================
# PYGAME INITIALIZATION
# ============================

pygame.init()
pygame.mixer.quit()  # Disable sound to improve performance

# Get display info
info = pygame.display.Info()
SCREEN_WIDTH = info.current_w
SCREEN_HEIGHT = info.current_h

# Initialize display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE)
pygame.display.set_caption("SysWatch Pro Quantum Ultimate - 60fps Visualization")

# Fonts
try:
    font_large = pygame.font.Font(None, 48)
    font_medium = pygame.font.Font(None, 32)
    font_small = pygame.font.Font(None, 24)
    font_tiny = pygame.font.Font(None, 18)
except:
    font_large = pygame.font.SysFont('consolas', 48)
    font_medium = pygame.font.SysFont('consolas', 32)
    font_small = pygame.font.SysFont('consolas', 24)
    font_tiny = pygame.font.SysFont('consolas', 18)

# Colors - Quantum Theme
class Colors:
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    
    # Quantum Colors
    QUANTUM_BLUE = (0, 150, 255)
    NEON_GREEN = (0, 255, 65)
    CYBER_YELLOW = (255, 255, 0)
    PLASMA_RED = (255, 0, 64)
    PURPLE = (150, 0, 255)
    CYAN = (0, 255, 255)
    
    # Transparency variants
    BLUE_ALPHA = (0, 150, 255, 128)
    GREEN_ALPHA = (0, 255, 65, 128)
    RED_ALPHA = (255, 0, 64, 128)
    
    # Background gradients
    BG_DARK = (10, 10, 15)
    BG_DARKER = (5, 5, 10)
    
    # Grid colors
    GRID_COLOR = (30, 30, 40)
    GRID_BRIGHT = (50, 50, 60)

# ============================
# DATA STRUCTURES
# ============================

@dataclass
class SystemMetrics:
    """시스템 메트릭 데이터"""
    timestamp: datetime
    cpu_percent: float
    cpu_freq: float
    cpu_cores: int
    memory_percent: float
    memory_total: int
    memory_used: int
    memory_available: int
    disk_percent: float
    disk_total: int
    disk_used: int
    network_sent: int
    network_recv: int
    network_sent_speed: float
    network_recv_speed: float
    process_count: int
    thread_count: int
    temperature: Optional[float] = None
    battery_percent: Optional[float] = None
    gpu_percent: Optional[float] = None

@dataclass
class SecurityAlert:
    """보안 알림"""
    level: str
    message: str
    timestamp: datetime
    category: str
    confidence: float = 1.0

# ============================
# SYSTEM MONITOR
# ============================

class QuantumSystemMonitor:
    """실시간 시스템 모니터링 엔진"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=300)  # 5분간 데이터 (60fps * 300초)
        self.alerts = deque(maxlen=100)
        self.last_network_io = psutil.net_io_counters()
        self.last_time = time.time()
        
        # AI 엔진
        self.anomaly_detector = None
        self.scaler = None
        if HAS_ML:
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            self.scaler = StandardScaler()
            self.is_trained = False
        
        # 성능 예측
        self.predictions = {}
        
        # 보안 모니터링
        self.security_threats = 0
        self.last_security_scan = 0
        
    def collect_metrics(self) -> SystemMetrics:
        """시스템 메트릭 수집"""
        current_time = time.time()
        
        # CPU 정보
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_freq = psutil.cpu_freq()
        cpu_cores = psutil.cpu_count()
        
        # 메모리 정보
        memory = psutil.virtual_memory()
        
        # 디스크 정보
        disk = psutil.disk_usage('/')
        
        # 네트워크 정보
        network_io = psutil.net_io_counters()
        time_delta = current_time - self.last_time
        
        if time_delta > 0:
            sent_speed = (network_io.bytes_sent - self.last_network_io.bytes_sent) / time_delta
            recv_speed = (network_io.bytes_recv - self.last_network_io.bytes_recv) / time_delta
        else:
            sent_speed = recv_speed = 0
        
        self.last_network_io = network_io
        self.last_time = current_time
        
        # 프로세스 정보
        process_count = len(psutil.pids())
        thread_count = sum(1 for _ in threading.enumerate())
        
        # 온도 정보
        temperature = None
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                temp_values = []
                for sensor_list in temps.values():
                    for sensor in sensor_list:
                        if sensor.current:
                            temp_values.append(sensor.current)
                if temp_values:
                    temperature = sum(temp_values) / len(temp_values)
        except:
            pass
        
        # 배터리 정보
        battery_percent = None
        try:
            battery = psutil.sensors_battery()
            if battery:
                battery_percent = battery.percent
        except:
            pass
        
        # GPU 정보 (추정)
        gpu_percent = None
        try:
            # GPU 사용률은 실제로는 nvidia-ml-py 등이 필요하지만 여기서는 CPU 기반 추정
            gpu_percent = min(100, cpu_percent * 0.8 + random.uniform(-5, 5))
        except:
            pass
        
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            cpu_freq=cpu_freq.current if cpu_freq else 0,
            cpu_cores=cpu_cores,
            memory_percent=memory.percent,
            memory_total=memory.total,
            memory_used=memory.used,
            memory_available=memory.available,
            disk_percent=(disk.used / disk.total) * 100,
            disk_total=disk.total,
            disk_used=disk.used,
            network_sent=network_io.bytes_sent,
            network_recv=network_io.bytes_recv,
            network_sent_speed=sent_speed,
            network_recv_speed=recv_speed,
            process_count=process_count,
            thread_count=thread_count,
            temperature=temperature,
            battery_percent=battery_percent,
            gpu_percent=gpu_percent
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def analyze_anomalies(self) -> List[SecurityAlert]:
        """AI 기반 이상 징후 탐지"""
        alerts = []
        
        if not HAS_ML or len(self.metrics_history) < 20:
            return alerts
        
        try:
            # 데이터 준비
            features = []
            for m in list(self.metrics_history)[-50:]:
                features.append([
                    m.cpu_percent,
                    m.memory_percent,
                    m.disk_percent,
                    m.process_count,
                    m.network_sent_speed / 1024 / 1024,  # MB/s
                    m.network_recv_speed / 1024 / 1024   # MB/s
                ])
            
            if len(features) >= 20:
                features_array = np.array(features)
                
                # 모델 훈련
                if not self.is_trained:
                    scaled_data = self.scaler.fit_transform(features_array)
                    self.anomaly_detector.fit(scaled_data)
                    self.is_trained = True
                
                # 이상 탐지
                latest_data = features_array[-1:].reshape(1, -1)
                scaled_latest = self.scaler.transform(latest_data)
                anomaly_score = self.anomaly_detector.decision_function(scaled_latest)[0]
                is_anomaly = self.anomaly_detector.predict(scaled_latest)[0] == -1
                
                if is_anomaly:
                    alerts.append(SecurityAlert(
                        level='QUANTUM',
                        message=f'AI 이상 징후 탐지 (점수: {anomaly_score:.3f})',
                        timestamp=datetime.now(),
                        category='ai_anomaly',
                        confidence=min(0.95, abs(anomaly_score) * 0.1 + 0.7)
                    ))
        
        except Exception as e:
            pass
        
        return alerts
    
    def security_scan(self) -> List[SecurityAlert]:
        """보안 스캔"""
        alerts = []
        current_time = time.time()
        
        # 5초마다 보안 스캔
        if current_time - self.last_security_scan < 5:
            return alerts
        
        self.last_security_scan = current_time
        
        try:
            # 프로세스 스캔
            suspicious_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                try:
                    proc_info = proc.info
                    if proc_info['cpu_percent'] > 90:  # 높은 CPU 사용률
                        suspicious_processes.append(proc_info['name'])
                except:
                    continue
            
            if len(suspicious_processes) > 3:
                alerts.append(SecurityAlert(
                    level='WARNING',
                    message=f'{len(suspicious_processes)}개 고부하 프로세스 탐지',
                    timestamp=datetime.now(),
                    category='performance_security',
                    confidence=0.8
                ))
            
            # 네트워크 연결 스캔
            connections = psutil.net_connections(kind='inet')
            external_connections = 0
            for conn in connections:
                if conn.raddr and not self._is_local_ip(conn.raddr.ip):
                    external_connections += 1
            
            if external_connections > 20:
                alerts.append(SecurityAlert(
                    level='INFO',
                    message=f'{external_connections}개 외부 네트워크 연결',
                    timestamp=datetime.now(),
                    category='network',
                    confidence=0.7
                ))
        
        except Exception:
            pass
        
        self.alerts.extend(alerts)
        return alerts
    
    def _is_local_ip(self, ip: str) -> bool:
        """로컬 IP 확인"""
        local_prefixes = ['127.', '192.168.', '10.', '172.16.', '169.254.']
        return any(ip.startswith(prefix) for prefix in local_prefixes)
    
    def predict_performance(self) -> Dict[str, Any]:
        """성능 예측"""
        if len(self.metrics_history) < 30:
            return {}
        
        recent_metrics = list(self.metrics_history)[-30:]
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        
        # 간단한 트렌드 예측
        cpu_trend = np.mean(cpu_values[-10:]) - np.mean(cpu_values[-20:-10])
        memory_trend = np.mean(memory_values[-10:]) - np.mean(memory_values[-20:-10])
        
        predictions = {
            'cpu_trend': 'increasing' if cpu_trend > 2 else 'decreasing' if cpu_trend < -2 else 'stable',
            'memory_trend': 'increasing' if memory_trend > 2 else 'decreasing' if memory_trend < -2 else 'stable',
            'cpu_prediction': min(100, max(0, cpu_values[-1] + cpu_trend * 5)),
            'memory_prediction': min(100, max(0, memory_values[-1] + memory_trend * 5)),
            'health_score': self._calculate_health_score(recent_metrics[-1])
        }
        
        self.predictions = predictions
        return predictions
    
    def _calculate_health_score(self, metrics: SystemMetrics) -> float:
        """시스템 건강도 점수 계산"""
        score = 100
        
        # CPU 점수
        if metrics.cpu_percent > 90:
            score -= 30
        elif metrics.cpu_percent > 70:
            score -= 15
        
        # 메모리 점수
        if metrics.memory_percent > 90:
            score -= 25
        elif metrics.memory_percent > 80:
            score -= 10
        
        # 디스크 점수
        if metrics.disk_percent > 95:
            score -= 20
        elif metrics.disk_percent > 85:
            score -= 5
        
        # 온도 점수
        if metrics.temperature:
            if metrics.temperature > 80:
                score -= 15
            elif metrics.temperature > 70:
                score -= 5
        
        return max(0, score)

# ============================
# VISUALIZATION COMPONENTS
# ============================

class QuantumRenderer:
    """고성능 60fps 렌더링 엔진"""
    
    def __init__(self, screen):
        self.screen = screen
        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT
        self.clock = pygame.time.Clock()
        
        # 애니메이션 변수
        self.time = 0
        self.glow_phase = 0
        
        # 그래프 표면
        self.graph_surfaces = {}
        
        # 입자 효과
        self.particles = []
        for _ in range(50):
            self.particles.append({
                'x': random.uniform(0, self.width),
                'y': random.uniform(0, self.height),
                'vx': random.uniform(-1, 1),
                'vy': random.uniform(-1, 1),
                'life': random.uniform(0.5, 1.0)
            })
    
    def update_animations(self, dt):
        """애니메이션 업데이트"""
        self.time += dt
        self.glow_phase = (self.glow_phase + dt * 2) % (2 * math.pi)
        
        # 입자 업데이트
        for particle in self.particles:
            particle['x'] += particle['vx'] * dt * 20
            particle['y'] += particle['vy'] * dt * 20
            
            # 경계 체크
            if particle['x'] < 0 or particle['x'] > self.width:
                particle['vx'] *= -1
            if particle['y'] < 0 or particle['y'] > self.height:
                particle['vy'] *= -1
            
            particle['x'] = max(0, min(self.width, particle['x']))
            particle['y'] = max(0, min(self.height, particle['y']))
    
    def draw_background(self):
        """배경 그리기"""
        # 그라데이션 배경
        for y in range(self.height):
            ratio = y / self.height
            r = int(Colors.BG_DARK[0] * (1 - ratio) + Colors.BG_DARKER[0] * ratio)
            g = int(Colors.BG_DARK[1] * (1 - ratio) + Colors.BG_DARKER[1] * ratio)
            b = int(Colors.BG_DARK[2] * (1 - ratio) + Colors.BG_DARKER[2] * ratio)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.width, y))
        
        # 그리드 패턴
        grid_size = 50
        for x in range(0, self.width, grid_size):
            pygame.draw.line(self.screen, Colors.GRID_COLOR, (x, 0), (x, self.height), 1)
        for y in range(0, self.height, grid_size):
            pygame.draw.line(self.screen, Colors.GRID_COLOR, (0, y), (self.width, y), 1)
        
        # 입자 효과
        for particle in self.particles:
            alpha = int(particle['life'] * 100)
            color = (*Colors.CYAN[:3], alpha)
            try:
                # pygame의 gfxdraw 사용하거나 간단한 점으로 대체
                pygame.draw.circle(self.screen, Colors.CYAN, 
                                 (int(particle['x']), int(particle['y'])), 1)
            except:
                pass
    
    def draw_header(self):
        """헤더 그리기"""
        # 타이틀 박스
        header_height = 80
        pygame.draw.rect(self.screen, Colors.BG_DARKER, (0, 0, self.width, header_height))
        pygame.draw.rect(self.screen, Colors.QUANTUM_BLUE, (0, 0, self.width, header_height), 2)
        
        # 타이틀 텍스트
        title = font_large.render("🚀 SYSWATCH PRO QUANTUM ULTIMATE", True, Colors.NEON_GREEN)
        title_rect = title.get_rect(center=(self.width // 2, 30))
        self.screen.blit(title, title_rect)
        
        # 서브타이틀
        subtitle = font_small.render("60fps 실시간 AI 시스템 모니터링 & 시각화", True, Colors.CYAN)
        subtitle_rect = subtitle.get_rect(center=(self.width // 2, 55))
        self.screen.blit(subtitle, subtitle_rect)
        
        # 시간 표시
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_text = font_small.render(current_time, True, Colors.WHITE)
        self.screen.blit(time_text, (self.width - 200, 10))
        
        # FPS 표시
        fps = self.clock.get_fps()
        fps_text = font_small.render(f"FPS: {fps:.1f}", True, 
                                   Colors.NEON_GREEN if fps > 55 else Colors.CYBER_YELLOW if fps > 30 else Colors.PLASMA_RED)
        self.screen.blit(fps_text, (10, 10))
    
    def draw_circular_gauge(self, center, radius, value, max_value, color, label, unit=""):
        """원형 게이지 그리기"""
        x, y = center
        
        # 배경 원
        pygame.draw.circle(self.screen, Colors.GRID_COLOR, (x, y), radius, 3)
        
        # 값에 따른 호 그리기
        if value > 0:
            angle = (value / max_value) * 2 * math.pi
            points = []
            segments = int(angle * 20)  # 부드러운 호를 위해
            
            for i in range(segments + 1):
                segment_angle = (i / 20) * 2 * math.pi - math.pi / 2
                px = x + (radius - 5) * math.cos(segment_angle)
                py = y + (radius - 5) * math.sin(segment_angle)
                points.append((px, py))
            
            if len(points) > 1:
                for i in range(len(points) - 1):
                    pygame.draw.line(self.screen, color, points[i], points[i + 1], 6)
        
        # 중앙 텍스트
        value_text = font_medium.render(f"{value:.1f}{unit}", True, Colors.WHITE)
        value_rect = value_text.get_rect(center=(x, y - 10))
        self.screen.blit(value_text, value_rect)
        
        # 라벨
        label_text = font_small.render(label, True, color)
        label_rect = label_text.get_rect(center=(x, y + 15))
        self.screen.blit(label_text, label_rect)
        
        # 글로우 효과
        glow_intensity = int(50 + 30 * math.sin(self.glow_phase))
        glow_color = (*color[:3], glow_intensity)
        try:
            pygame.draw.circle(self.screen, color, (x, y), radius + 2, 1)
        except:
            pass
    
    def draw_bar_graph(self, rect, values, colors, labels, max_value=100):
        """막대 그래프 그리기"""
        x, y, width, height = rect
        
        # 배경
        pygame.draw.rect(self.screen, Colors.BG_DARKER, rect)
        pygame.draw.rect(self.screen, Colors.GRID_BRIGHT, rect, 2)
        
        if not values:
            return
        
        bar_width = width // len(values)
        
        for i, (value, color, label) in enumerate(zip(values, colors, labels)):
            bar_x = x + i * bar_width
            bar_height = int((value / max_value) * (height - 40))
            bar_y = y + height - bar_height - 20
            
            # 막대 그리기
            pygame.draw.rect(self.screen, color, (bar_x + 5, bar_y, bar_width - 10, bar_height))
            
            # 글로우 효과
            pygame.draw.rect(self.screen, color, (bar_x + 3, bar_y - 2, bar_width - 6, bar_height + 4), 1)
            
            # 값 표시
            value_text = font_tiny.render(f"{value:.1f}", True, Colors.WHITE)
            value_rect = value_text.get_rect(center=(bar_x + bar_width // 2, bar_y - 10))
            self.screen.blit(value_text, value_rect)
            
            # 라벨
            label_text = font_tiny.render(label, True, color)
            label_rect = label_text.get_rect(center=(bar_x + bar_width // 2, y + height - 10))
            self.screen.blit(label_text, label_rect)
    
    def draw_line_graph(self, rect, data_series, colors, labels, max_value=100):
        """선 그래프 그리기"""
        x, y, width, height = rect
        
        # 배경
        pygame.draw.rect(self.screen, Colors.BG_DARKER, rect)
        pygame.draw.rect(self.screen, Colors.GRID_BRIGHT, rect, 2)
        
        # 격자
        grid_lines = 5
        for i in range(grid_lines + 1):
            grid_y = y + (height * i // grid_lines)
            pygame.draw.line(self.screen, Colors.GRID_COLOR, (x, grid_y), (x + width, grid_y), 1)
            
            # Y축 라벨
            if i < grid_lines:
                value = max_value * (1 - i / grid_lines)
                label = font_tiny.render(f"{value:.0f}", True, Colors.WHITE)
                self.screen.blit(label, (x + 5, grid_y + 2))
        
        # 데이터 그리기
        for series_idx, (data, color, label) in enumerate(zip(data_series, colors, labels)):
            if len(data) < 2:
                continue
            
            points = []
            for i, value in enumerate(data):
                point_x = x + (width * i // (len(data) - 1))
                point_y = y + height - int((value / max_value) * height)
                points.append((point_x, point_y))
            
            # 선 그리기
            if len(points) > 1:
                pygame.draw.lines(self.screen, color, False, points, 2)
                
                # 점들 그리기
                for point in points:
                    pygame.draw.circle(self.screen, color, point, 3)
        
        # 범례
        legend_y = y + 10
        for i, (color, label) in enumerate(zip(colors, labels)):
            legend_x = x + width - 150
            legend_item_y = legend_y + i * 20
            
            pygame.draw.rect(self.screen, color, (legend_x, legend_item_y, 15, 10))
            legend_text = font_tiny.render(label, True, Colors.WHITE)
            self.screen.blit(legend_text, (legend_x + 20, legend_item_y))
    
    def draw_network_flow(self, rect, sent_speed, recv_speed):
        """네트워크 플로우 시각화"""
        x, y, width, height = rect
        
        # 배경
        pygame.draw.rect(self.screen, Colors.BG_DARKER, rect)
        pygame.draw.rect(self.screen, Colors.GRID_BRIGHT, rect, 2)
        
        # 중앙선
        center_y = y + height // 2
        pygame.draw.line(self.screen, Colors.GRID_COLOR, (x, center_y), (x + width, center_y), 2)
        
        # 업로드 (상단)
        if sent_speed > 0:
            upload_height = min(height // 2 - 10, int((sent_speed / (1024 * 1024)) * (height // 4)))
            pygame.draw.rect(self.screen, Colors.PLASMA_RED, 
                           (x + 20, center_y - upload_height, width - 40, upload_height))
        
        # 다운로드 (하단)
        if recv_speed > 0:
            download_height = min(height // 2 - 10, int((recv_speed / (1024 * 1024)) * (height // 4)))
            pygame.draw.rect(self.screen, Colors.NEON_GREEN, 
                           (x + 20, center_y, width - 40, download_height))
        
        # 라벨
        upload_text = font_small.render(f"↑ {self.format_bytes(sent_speed)}/s", True, Colors.PLASMA_RED)
        download_text = font_small.render(f"↓ {self.format_bytes(recv_speed)}/s", True, Colors.NEON_GREEN)
        
        self.screen.blit(upload_text, (x + 10, y + 10))
        self.screen.blit(download_text, (x + 10, y + height - 30))
    
    def draw_3d_cube(self, center, size, rotation, color):
        """3D 큐브 그리기 (와이어프레임)"""
        x, y = center
        
        # 3D 점들 정의
        vertices = [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # 뒷면
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # 앞면
        ]
        
        # 회전 적용
        cos_x, sin_x = math.cos(rotation[0]), math.sin(rotation[0])
        cos_y, sin_y = math.cos(rotation[1]), math.sin(rotation[1])
        cos_z, sin_z = math.cos(rotation[2]), math.sin(rotation[2])
        
        rotated_vertices = []
        for vx, vy, vz in vertices:
            # Y축 회전
            new_x = vx * cos_y - vz * sin_y
            new_z = vx * sin_y + vz * cos_y
            vx, vz = new_x, new_z
            
            # X축 회전
            new_y = vy * cos_x - vz * sin_x
            new_z = vy * sin_x + vz * cos_x
            vy, vz = new_y, new_z
            
            # Z축 회전
            new_x = vx * cos_z - vy * sin_z
            new_y = vx * sin_z + vy * cos_z
            vx, vy = new_x, new_y
            
            # 2D 투영
            screen_x = x + vx * size
            screen_y = y + vy * size
            rotated_vertices.append((screen_x, screen_y))
        
        # 엣지 그리기
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 뒷면
            (4, 5), (5, 6), (6, 7), (7, 4),  # 앞면
            (0, 4), (1, 5), (2, 6), (3, 7)   # 연결선
        ]
        
        for start, end in edges:
            pygame.draw.line(self.screen, color, rotated_vertices[start], rotated_vertices[end], 2)
    
    def draw_heatmap(self, rect, data, min_val, max_val):
        """히트맵 그리기"""
        x, y, width, height = rect
        
        # 배경
        pygame.draw.rect(self.screen, Colors.BG_DARKER, rect)
        pygame.draw.rect(self.screen, Colors.GRID_BRIGHT, rect, 2)
        
        if not data:
            return
        
        rows, cols = len(data), len(data[0]) if data else 0
        if rows == 0 or cols == 0:
            return
        
        cell_width = width // cols
        cell_height = height // rows
        
        for i in range(rows):
            for j in range(cols):
                if j < len(data[i]):
                    value = data[i][j]
                    # 값을 색상으로 변환
                    if max_val > min_val:
                        intensity = (value - min_val) / (max_val - min_val)
                    else:
                        intensity = 0.5
                    
                    # 색상 계산
                    r = int(255 * intensity)
                    g = int(255 * (1 - intensity))
                    b = 50
                    
                    cell_x = x + j * cell_width
                    cell_y = y + i * cell_height
                    
                    pygame.draw.rect(self.screen, (r, g, b), 
                                   (cell_x, cell_y, cell_width - 1, cell_height - 1))
    
    def format_bytes(self, bytes_value):
        """바이트를 읽기 쉬운 형식으로 변환"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"

# ============================
# MAIN VISUALIZATION ENGINE
# ============================

class QuantumVisualizationEngine:
    """메인 시각화 엔진"""
    
    def __init__(self):
        self.monitor = QuantumSystemMonitor()
        self.renderer = QuantumRenderer(screen)
        self.running = True
        
        # 레이아웃 정의
        self.setup_layout()
        
        # 3D 회전 상태
        self.cube_rotation = [0, 0, 0]
        
        # 데이터 히스토리
        self.cpu_history = deque(maxlen=100)
        self.memory_history = deque(maxlen=100)
        self.network_history = deque(maxlen=100)
        
    def setup_layout(self):
        """레이아웃 설정"""
        # 헤더 영역
        self.header_rect = (0, 0, self.renderer.width, 80)
        
        # 메인 영역을 4x3 그리드로 분할
        main_y = 80
        main_height = self.renderer.height - main_y
        
        grid_width = self.renderer.width // 4
        grid_height = main_height // 3
        
        # 각 섹션 정의
        self.sections = {
            # 첫 번째 행
            'cpu_gauge': (0, main_y, grid_width, grid_height),
            'memory_gauge': (grid_width, main_y, grid_width, grid_height),
            'disk_gauge': (grid_width * 2, main_y, grid_width, grid_height),
            'gpu_gauge': (grid_width * 3, main_y, grid_width, grid_height),
            
            # 두 번째 행
            'cpu_graph': (0, main_y + grid_height, grid_width * 2, grid_height),
            'memory_graph': (grid_width * 2, main_y + grid_height, grid_width * 2, grid_height),
            
            # 세 번째 행
            'network_flow': (0, main_y + grid_height * 2, grid_width, grid_height),
            'process_bar': (grid_width, main_y + grid_height * 2, grid_width, grid_height),
            'security_status': (grid_width * 2, main_y + grid_height * 2, grid_width, grid_height),
            '3d_visualization': (grid_width * 3, main_y + grid_height * 2, grid_width, grid_height),
        }
    
    def handle_events(self):
        """이벤트 처리"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    self.running = False
                elif event.key == pygame.K_F11:
                    # 전체화면 토글 (이미 전체화면이므로 무시)
                    pass
    
    def update(self, dt):
        """업데이트"""
        # 시스템 메트릭 수집
        metrics = self.monitor.collect_metrics()
        
        # 히스토리 업데이트
        self.cpu_history.append(metrics.cpu_percent)
        self.memory_history.append(metrics.memory_percent)
        self.network_history.append(metrics.network_sent_speed + metrics.network_recv_speed)
        
        # AI 분석
        self.monitor.analyze_anomalies()
        self.monitor.security_scan()
        self.monitor.predict_performance()
        
        # 애니메이션 업데이트
        self.renderer.update_animations(dt)
        self.cube_rotation[0] += dt * 0.5
        self.cube_rotation[1] += dt * 0.3
        self.cube_rotation[2] += dt * 0.7
        
        return metrics
    
    def render(self, metrics):
        """렌더링"""
        # 배경 그리기
        self.renderer.draw_background()
        
        # 헤더 그리기
        self.renderer.draw_header()
        
        # CPU 게이지
        cpu_color = (Colors.NEON_GREEN if metrics.cpu_percent < 50 else 
                    Colors.CYBER_YELLOW if metrics.cpu_percent < 80 else Colors.PLASMA_RED)
        gauge_center = (self.sections['cpu_gauge'][0] + self.sections['cpu_gauge'][2] // 2,
                       self.sections['cpu_gauge'][1] + self.sections['cpu_gauge'][3] // 2)
        self.renderer.draw_circular_gauge(
            gauge_center, 80, metrics.cpu_percent, 100, cpu_color, "CPU", "%"
        )
        
        # 메모리 게이지
        memory_color = (Colors.NEON_GREEN if metrics.memory_percent < 60 else 
                       Colors.CYBER_YELLOW if metrics.memory_percent < 85 else Colors.PLASMA_RED)
        memory_center = (self.sections['memory_gauge'][0] + self.sections['memory_gauge'][2] // 2,
                        self.sections['memory_gauge'][1] + self.sections['memory_gauge'][3] // 2)
        self.renderer.draw_circular_gauge(
            memory_center, 80, metrics.memory_percent, 100, memory_color, "RAM", "%"
        )
        
        # 디스크 게이지
        disk_color = (Colors.NEON_GREEN if metrics.disk_percent < 70 else 
                     Colors.CYBER_YELLOW if metrics.disk_percent < 90 else Colors.PLASMA_RED)
        disk_center = (self.sections['disk_gauge'][0] + self.sections['disk_gauge'][2] // 2,
                      self.sections['disk_gauge'][1] + self.sections['disk_gauge'][3] // 2)
        self.renderer.draw_circular_gauge(
            disk_center, 80, metrics.disk_percent, 100, disk_color, "DISK", "%"
        )
        
        # GPU 게이지 (추정값)
        gpu_percent = metrics.gpu_percent or 0
        gpu_color = (Colors.NEON_GREEN if gpu_percent < 50 else 
                    Colors.CYBER_YELLOW if gpu_percent < 80 else Colors.PLASMA_RED)
        gpu_center = (self.sections['gpu_gauge'][0] + self.sections['gpu_gauge'][2] // 2,
                     self.sections['gpu_gauge'][1] + self.sections['gpu_gauge'][3] // 2)
        self.renderer.draw_circular_gauge(
            gpu_center, 80, gpu_percent, 100, gpu_color, "GPU", "%"
        )
        
        # CPU 히스토리 그래프
        self.renderer.draw_line_graph(
            self.sections['cpu_graph'],
            [list(self.cpu_history)],
            [Colors.QUANTUM_BLUE],
            ["CPU Usage"],
            100
        )
        
        # 메모리 히스토리 그래프
        self.renderer.draw_line_graph(
            self.sections['memory_graph'],
            [list(self.memory_history)],
            [Colors.PURPLE],
            ["Memory Usage"],
            100
        )
        
        # 네트워크 플로우
        self.renderer.draw_network_flow(
            self.sections['network_flow'],
            metrics.network_sent_speed,
            metrics.network_recv_speed
        )
        
        # 프로세스 막대그래프
        try:
            top_processes = []
            process_cpu = []
            process_names = []
            
            for proc in psutil.process_iter(['name', 'cpu_percent']):
                try:
                    info = proc.info
                    if info['cpu_percent'] > 0:
                        top_processes.append((info['name'], info['cpu_percent']))
                except:
                    continue
            
            top_processes.sort(key=lambda x: x[1], reverse=True)
            top_processes = top_processes[:5]  # 상위 5개
            
            if top_processes:
                process_names = [p[0][:8] for p in top_processes]  # 이름 축약
                process_cpu = [p[1] for p in top_processes]
                colors = [Colors.NEON_GREEN, Colors.CYAN, Colors.CYBER_YELLOW, Colors.PURPLE, Colors.PLASMA_RED]
                
                self.renderer.draw_bar_graph(
                    self.sections['process_bar'],
                    process_cpu,
                    colors[:len(process_cpu)],
                    process_names,
                    max(process_cpu) if process_cpu else 100
                )
        except:
            pass
        
        # 보안 상태
        security_rect = self.sections['security_status']
        pygame.draw.rect(screen, Colors.BG_DARKER, security_rect)
        pygame.draw.rect(screen, Colors.GRID_BRIGHT, security_rect, 2)
        
        # 보안 상태 텍스트
        alert_count = len(self.monitor.alerts)
        security_color = (Colors.NEON_GREEN if alert_count == 0 else 
                         Colors.CYBER_YELLOW if alert_count < 5 else Colors.PLASMA_RED)
        
        security_title = font_medium.render("SECURITY", True, security_color)
        screen.blit(security_title, (security_rect[0] + 10, security_rect[1] + 10))
        
        status_text = "SECURE" if alert_count == 0 else f"{alert_count} ALERTS"
        status_render = font_small.render(status_text, True, security_color)
        screen.blit(status_render, (security_rect[0] + 10, security_rect[1] + 40))
        
        # AI 예측 표시
        if self.monitor.predictions:
            health_score = self.monitor.predictions.get('health_score', 100)
            health_text = font_small.render(f"Health: {health_score:.0f}%", True, Colors.CYAN)
            screen.blit(health_text, (security_rect[0] + 10, security_rect[1] + 70))
        
        # 3D 시각화
        viz_center = (self.sections['3d_visualization'][0] + self.sections['3d_visualization'][2] // 2,
                     self.sections['3d_visualization'][1] + self.sections['3d_visualization'][3] // 2)
        
        # CPU 로드에 따른 색상 변화
        cube_color = (
            int(255 * (metrics.cpu_percent / 100)),
            int(255 * (1 - metrics.cpu_percent / 100)),
            100
        )
        
        self.renderer.draw_3d_cube(viz_center, 50, self.cube_rotation, cube_color)
        
        # 온도 표시 (있는 경우)
        if metrics.temperature:
            temp_text = font_small.render(f"TEMP: {metrics.temperature:.1f}°C", True, Colors.CYBER_YELLOW)
            screen.blit(temp_text, (viz_center[0] - 50, viz_center[1] + 80))
        
        # 배터리 표시 (있는 경우)
        if metrics.battery_percent is not None:
            battery_color = (Colors.NEON_GREEN if metrics.battery_percent > 50 else 
                           Colors.CYBER_YELLOW if metrics.battery_percent > 20 else Colors.PLASMA_RED)
            battery_text = font_small.render(f"BAT: {metrics.battery_percent:.0f}%", True, battery_color)
            screen.blit(battery_text, (viz_center[0] - 50, viz_center[1] + 100))
    
    def run(self):
        """메인 실행 루프"""
        print("🚀 SysWatch Pro Quantum Ultimate 60fps 시각화 시작...")
        print("🎮 ESC 또는 Q 키로 종료")
        print("🖥️ 전체화면 모드에서 실행 중...")
        
        last_time = time.time()
        
        while self.running:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # 이벤트 처리
            self.handle_events()
            
            # 업데이트
            metrics = self.update(dt)
            
            # 렌더링
            self.render(metrics)
            
            # 화면 업데이트
            pygame.display.flip()
            
            # 60 FPS 유지
            self.renderer.clock.tick(60)
        
        # 정리
        pygame.quit()
        print("\n🌟 SysWatch Pro Quantum Ultimate 종료")
        print("차세대 60fps 시각화를 경험해주셔서 감사합니다!")

# ============================
# ENTRY POINT
# ============================

def main():
    """메인 함수"""
    try:
        print("🎮 Pygame 초기화 중...")
        print(f"🖥️ 해상도: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
        print("🚀 SysWatch Pro Quantum Ultimate 60fps 시각화 엔진 시작...")
        
        # 메인 시각화 엔진 실행
        engine = QuantumVisualizationEngine()
        engine.run()
        
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 중단됨")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            pygame.quit()
        except:
            pass

if __name__ == "__main__":
    main()