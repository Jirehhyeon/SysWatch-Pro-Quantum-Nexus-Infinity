#!/usr/bin/env python3
"""
SysWatch Pro Quantum GUI - 홀로그래픽 3D 인터페이스
AAA급 최첨단 시각화 및 AI 대시보드

Copyright (C) 2025 SysWatch Technologies Ltd.
"""

import sys
import os
import time
import threading
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import sqlite3

# Advanced GUI Framework
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, font
try:
    import customtkinter as ctk
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    HAS_CUSTOM_TK = True
except ImportError:
    HAS_CUSTOM_TK = False

# Professional visualization
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.animation as animation
import matplotlib.patheffects as path_effects
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Wedge
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

# Advanced plotting
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as py
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# OpenGL 3D graphics
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    import pygame
    from pygame.locals import *
    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False

# Core monitoring
from syswatch_quantum import (
    quantum_monitor, QuantumMetrics, QuantumAlert, QuantumPrediction,
    QUANTUM_THEME, VERSION, EDITION, CODENAME
)

class QuantumHUD:
    """홀로그래픽 HUD 오버레이"""
    
    def __init__(self, parent):
        self.parent = parent
        self.canvas = tk.Canvas(
            parent,
            bg=QUANTUM_THEME['void_black'],
            highlightthickness=0,
            height=200
        )
        self.canvas.pack(fill=tk.X, padx=10, pady=5)
        
        # HUD 요소들
        self.hud_elements = {}
        self.animation_frame = 0
        self.setup_hud()
        
        # 애니메이션 타이머
        self.animate_hud()
    
    def setup_hud(self):
        """HUD 요소 설정"""
        width = 800
        height = 200
        self.canvas.configure(width=width, height=height)
        
        # 중앙 원형 디스플레이
        center_x, center_y = width // 2, height // 2
        radius = 80
        
        # 외곽 링
        self.hud_elements['outer_ring'] = self.canvas.create_oval(
            center_x - radius, center_y - radius,
            center_x + radius, center_y + radius,
            outline=QUANTUM_THEME['quantum_cyan'],
            width=3,
            fill=""
        )
        
        # 내부 링
        inner_radius = radius - 20
        self.hud_elements['inner_ring'] = self.canvas.create_oval(
            center_x - inner_radius, center_y - inner_radius,
            center_x + inner_radius, center_y + inner_radius,
            outline=QUANTUM_THEME['quantum_purple'],
            width=2,
            fill=""
        )
        
        # 중앙 텍스트
        self.hud_elements['center_text'] = self.canvas.create_text(
            center_x, center_y,
            text="QUANTUM\nSYSTEM",
            fill=QUANTUM_THEME['quantum_green'],
            font=('Consolas', 12, 'bold'),
            justify=tk.CENTER
        )
        
        # 상태 표시기들
        positions = [
            (center_x - 150, center_y - 50, "CPU"),
            (center_x + 150, center_y - 50, "MEM"),
            (center_x - 150, center_y + 50, "DISK"),
            (center_x + 150, center_y + 50, "NET")
        ]
        
        for x, y, label in positions:
            # 상태 원
            circle = self.canvas.create_oval(
                x - 25, y - 25, x + 25, y + 25,
                outline=QUANTUM_THEME['quantum_blue'],
                width=2,
                fill=QUANTUM_THEME['glass_light']
            )
            
            # 라벨
            text = self.canvas.create_text(
                x, y - 40,
                text=label,
                fill=QUANTUM_THEME['text_primary'],
                font=('Consolas', 10, 'bold')
            )
            
            # 값
            value_text = self.canvas.create_text(
                x, y,
                text="0%",
                fill=QUANTUM_THEME['quantum_yellow'],
                font=('Consolas', 11, 'bold')
            )
            
            self.hud_elements[f'{label.lower()}_circle'] = circle
            self.hud_elements[f'{label.lower()}_text'] = text
            self.hud_elements[f'{label.lower()}_value'] = value_text
    
    def animate_hud(self):
        """HUD 애니메이션"""
        self.animation_frame += 1
        
        # 링 회전 효과 (CSS rotation 대신 색상 변화로 표현)
        colors = [
            QUANTUM_THEME['quantum_cyan'],
            QUANTUM_THEME['quantum_purple'],
            QUANTUM_THEME['quantum_green'],
            QUANTUM_THEME['quantum_yellow']
        ]
        
        color_index = (self.animation_frame // 10) % len(colors)
        self.canvas.itemconfig(self.hud_elements['outer_ring'], outline=colors[color_index])
        
        # 다음 프레임 스케줄
        self.parent.after(100, self.animate_hud)
    
    def update_status(self, metrics: QuantumMetrics):
        """상태 업데이트"""
        try:
            # CPU
            cpu_avg = np.mean(metrics.cpu_cores) if metrics.cpu_cores else 0
            self.canvas.itemconfig(
                self.hud_elements['cpu_value'],
                text=f"{cpu_avg:.0f}%",
                fill=self.get_status_color(cpu_avg)
            )
            
            # Memory
            self.canvas.itemconfig(
                self.hud_elements['mem_value'],
                text=f"{metrics.memory_percent:.0f}%",
                fill=self.get_status_color(metrics.memory_percent)
            )
            
            # Disk (사용률 대신 I/O로)
            disk_activity = min(100, (metrics.disk_read + metrics.disk_write) * 10)
            self.canvas.itemconfig(
                self.hud_elements['disk_value'],
                text=f"{disk_activity:.0f}%",
                fill=self.get_status_color(disk_activity)
            )
            
            # Network
            net_activity = min(100, (metrics.network_sent + metrics.network_recv) * 10)
            self.canvas.itemconfig(
                self.hud_elements['net_value'],
                text=f"{net_activity:.0f}%",
                fill=self.get_status_color(net_activity)
            )
            
        except Exception as e:
            print(f"HUD 업데이트 오류: {e}")
    
    def get_status_color(self, value: float) -> str:
        """상태에 따른 색상 반환"""
        if value < 30:
            return QUANTUM_THEME['quantum_green']
        elif value < 60:
            return QUANTUM_THEME['quantum_yellow']
        elif value < 80:
            return QUANTUM_THEME['quantum_orange']
        else:
            return QUANTUM_THEME['quantum_red']

class Quantum3DChart:
    """3D 홀로그래픽 차트"""
    
    def __init__(self, parent, title="3D Performance"):
        self.parent = parent
        self.title = title
        
        # Figure 생성
        self.fig = Figure(figsize=(8, 6), dpi=100, facecolor=QUANTUM_THEME['void_black'])
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 스타일 설정
        self.setup_3d_style()
        
        # 캔버스 생성
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 데이터 저장소
        self.data_history = {
            'cpu': deque(maxlen=100),
            'memory': deque(maxlen=100), 
            'disk': deque(maxlen=100),
            'network': deque(maxlen=100),
            'timestamps': deque(maxlen=100)
        }
        
        # 3D 객체들
        self.surface = None
        self.wireframe = None
        
    def setup_3d_style(self):
        """3D 스타일 설정"""
        self.ax.set_facecolor(QUANTUM_THEME['void_black'])
        self.fig.patch.set_facecolor(QUANTUM_THEME['void_black'])
        
        # 축 색상
        self.ax.xaxis.label.set_color(QUANTUM_THEME['text_primary'])
        self.ax.yaxis.label.set_color(QUANTUM_THEME['text_primary'])
        self.ax.zaxis.label.set_color(QUANTUM_THEME['text_primary'])
        
        self.ax.tick_params(axis='x', colors=QUANTUM_THEME['text_secondary'])
        self.ax.tick_params(axis='y', colors=QUANTUM_THEME['text_secondary'])
        self.ax.tick_params(axis='z', colors=QUANTUM_THEME['text_secondary'])
        
        # 격자
        self.ax.grid(True, alpha=0.3, color=QUANTUM_THEME['quantum_cyan'])
        
        # 제목
        self.ax.set_title(self.title, color=QUANTUM_THEME['quantum_green'], fontsize=14, pad=20)
        
        # 축 라벨
        self.ax.set_xlabel('Time', color=QUANTUM_THEME['text_primary'])
        self.ax.set_ylabel('Metric Type', color=QUANTUM_THEME['text_primary'])
        self.ax.set_zlabel('Usage (%)', color=QUANTUM_THEME['text_primary'])
    
    def update_data(self, metrics: QuantumMetrics):
        """데이터 업데이트"""
        try:
            cpu_avg = np.mean(metrics.cpu_cores) if metrics.cpu_cores else 0
            
            self.data_history['cpu'].append(cpu_avg)
            self.data_history['memory'].append(metrics.memory_percent)
            self.data_history['disk'].append(min(100, (metrics.disk_read + metrics.disk_write) * 10))
            self.data_history['network'].append(min(100, (metrics.network_sent + metrics.network_recv) * 10))
            self.data_history['timestamps'].append(time.time())
            
            self.render_3d_surface()
            
        except Exception as e:
            print(f"3D 차트 업데이트 오류: {e}")
    
    def render_3d_surface(self):
        """3D 표면 렌더링"""
        if len(self.data_history['cpu']) < 10:
            return
        
        try:
            self.ax.clear()
            self.setup_3d_style()
            
            # 데이터 준비
            data_length = len(self.data_history['cpu'])
            x = np.arange(data_length)
            y = np.arange(4)  # CPU, Memory, Disk, Network
            X, Y = np.meshgrid(x, y)
            
            # Z 데이터 (각 메트릭별)
            Z = np.array([
                list(self.data_history['cpu']),
                list(self.data_history['memory']),
                list(self.data_history['disk']),
                list(self.data_history['network'])
            ])
            
            # 홀로그래픽 효과를 위한 그라디언트 색상
            colors = plt.cm.plasma(Z / 100.0)
            
            # 와이어프레임 표면
            self.wireframe = self.ax.plot_wireframe(
                X, Y, Z,
                color=QUANTUM_THEME['quantum_cyan'],
                alpha=0.6,
                linewidth=1
            )
            
            # 표면 렌더링
            self.surface = self.ax.plot_surface(
                X, Y, Z,
                facecolors=colors,
                alpha=0.8,
                antialiased=True,
                shade=True
            )
            
            # 축 범위 설정
            self.ax.set_xlim(0, max(10, data_length))
            self.ax.set_ylim(0, 3)
            self.ax.set_zlim(0, 100)
            
            # Y축 라벨
            self.ax.set_yticks([0, 1, 2, 3])
            self.ax.set_yticklabels(['CPU', 'MEM', 'DISK', 'NET'])
            
            # 회전 애니메이션
            current_time = time.time()
            rotation_angle = (current_time * 10) % 360
            self.ax.view_init(elev=20, azim=rotation_angle)
            
            self.canvas.draw_idle()
            
        except Exception as e:
            print(f"3D 렌더링 오류: {e}")

class QuantumGaugePanel:
    """양자 게이지 패널"""
    
    def __init__(self, parent):
        self.parent = parent
        self.frame = tk.Frame(parent, bg=QUANTUM_THEME['void_black'])
        self.frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 게이지 목록
        self.gauges = {}
        self.setup_gauges()
    
    def setup_gauges(self):
        """게이지들 설정"""
        gauge_configs = [
            ("CPU", QUANTUM_THEME['quantum_red'], 0, 0),
            ("Memory", QUANTUM_THEME['quantum_yellow'], 0, 1),
            ("GPU", QUANTUM_THEME['quantum_green'], 1, 0),
            ("Network", QUANTUM_THEME['quantum_blue'], 1, 1)
        ]
        
        for name, color, row, col in gauge_configs:
            gauge_frame = tk.Frame(self.frame, bg=QUANTUM_THEME['void_black'])
            gauge_frame.grid(row=row, col=col, padx=10, pady=10, sticky="nsew")
            
            # Figure 생성
            fig = Figure(figsize=(3, 3), dpi=100, facecolor=QUANTUM_THEME['void_black'])
            ax = fig.add_subplot(111, polar=True)
            
            # 게이지 생성
            self.create_quantum_gauge(ax, name, color)
            
            # 캔버스
            canvas = FigureCanvasTkAgg(fig, gauge_frame)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.gauges[name.lower()] = {
                'ax': ax,
                'canvas': canvas,
                'fig': fig,
                'color': color
            }
        
        # 그리드 가중치 설정
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_rowconfigure(1, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_columnconfigure(1, weight=1)
    
    def create_quantum_gauge(self, ax, title, color):
        """양자 게이지 생성"""
        ax.set_facecolor(QUANTUM_THEME['void_black'])
        ax.set_ylim(0, 100)
        ax.set_xlim(0, 2 * np.pi)
        
        # 배경 호
        theta = np.linspace(0, 2 * np.pi, 100)
        r = np.full_like(theta, 90)
        ax.plot(theta, r, color=QUANTUM_THEME['glass_medium'], linewidth=8, alpha=0.3)
        
        # 제목
        ax.text(0, 110, title, ha='center', va='center', 
                fontsize=12, fontweight='bold', color=color,
                transform=ax.transData)
        
        # 눈금 제거
        ax.set_rticks([])
        ax.set_thetagrids([])
        ax.grid(False)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
    
    def update_gauge(self, gauge_name: str, value: float):
        """게이지 업데이트"""
        if gauge_name not in self.gauges:
            return
        
        try:
            gauge = self.gauges[gauge_name]
            ax = gauge['ax']
            color = gauge['color']
            
            # 게이지 클리어
            ax.clear()
            self.create_quantum_gauge(ax, gauge_name.upper(), color)
            
            # 값 제한
            value = max(0, min(100, value))
            
            # 게이지 호 그리기
            theta_range = (value / 100) * 2 * np.pi
            theta = np.linspace(0, theta_range, int(value) + 1)
            r = np.full_like(theta, 90)
            
            # 그라디언트 효과
            colors = [color] * len(theta)
            
            if len(theta) > 1:
                for i in range(len(theta) - 1):
                    ax.plot(theta[i:i+2], r[i:i+2], color=color, 
                           linewidth=12, alpha=0.8 + 0.2 * (i / len(theta)))
            
            # 중앙 값 표시
            ax.text(0, 50, f"{value:.0f}%", ha='center', va='center',
                   fontsize=16, fontweight='bold', color=color)
            
            # 상태 표시
            if value < 30:
                status_color = QUANTUM_THEME['quantum_green']
                status = "OPTIMAL"
            elif value < 60:
                status_color = QUANTUM_THEME['quantum_yellow']
                status = "NORMAL"
            elif value < 80:
                status_color = QUANTUM_THEME['quantum_orange']
                status = "HIGH"
            else:
                status_color = QUANTUM_THEME['quantum_red']
                status = "CRITICAL"
            
            ax.text(0, 20, status, ha='center', va='center',
                   fontsize=10, fontweight='bold', color=status_color)
            
            gauge['canvas'].draw_idle()
            
        except Exception as e:
            print(f"게이지 {gauge_name} 업데이트 오류: {e}")
    
    def update_all_gauges(self, metrics: QuantumMetrics):
        """모든 게이지 업데이트"""
        try:
            cpu_avg = np.mean(metrics.cpu_cores) if metrics.cpu_cores else 0
            self.update_gauge('cpu', cpu_avg)
            self.update_gauge('memory', metrics.memory_percent)
            self.update_gauge('gpu', metrics.gpu_usage)
            
            network_activity = min(100, (metrics.network_sent + metrics.network_recv) * 10)
            self.update_gauge('network', network_activity)
            
        except Exception as e:
            print(f"게이지 패널 업데이트 오류: {e}")

class QuantumAIPanel:
    """AI 예측 및 분석 패널"""
    
    def __init__(self, parent):
        self.parent = parent
        self.frame = tk.Frame(parent, bg=QUANTUM_THEME['void_black'])
        self.frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # AI 패널 구성
        self.setup_ai_panel()
        
        # 예측 데이터
        self.predictions = {}
        self.anomalies = []
    
    def setup_ai_panel(self):
        """AI 패널 설정"""
        # 제목
        title_label = tk.Label(
            self.frame,
            text="🧠 QUANTUM AI ANALYSIS",
            font=('Consolas', 16, 'bold'),
            fg=QUANTUM_THEME['quantum_purple'],
            bg=QUANTUM_THEME['void_black']
        )
        title_label.pack(pady=10)
        
        # 메인 프레임
        main_frame = tk.Frame(self.frame, bg=QUANTUM_THEME['void_black'])
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 좌측: 예측 차트
        left_frame = tk.Frame(main_frame, bg=QUANTUM_THEME['void_black'])
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # 예측 차트
        self.prediction_fig = Figure(figsize=(6, 4), dpi=100, facecolor=QUANTUM_THEME['void_black'])
        self.prediction_ax = self.prediction_fig.add_subplot(111)
        self.setup_prediction_chart()
        
        self.prediction_canvas = FigureCanvasTkAgg(self.prediction_fig, left_frame)
        self.prediction_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 우측: AI 인사이트
        right_frame = tk.Frame(main_frame, bg=QUANTUM_THEME['void_black'], width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_frame.pack_propagate(False)
        
        # AI 상태
        ai_status_label = tk.Label(
            right_frame,
            text="AI Engine Status",
            font=('Consolas', 12, 'bold'),
            fg=QUANTUM_THEME['quantum_cyan'],
            bg=QUANTUM_THEME['void_black']
        )
        ai_status_label.pack(pady=(0, 10))
        
        self.ai_status_text = tk.Text(
            right_frame,
            height=8,
            width=35,
            bg=QUANTUM_THEME['dark_matter'],
            fg=QUANTUM_THEME['text_primary'],
            font=('Consolas', 9),
            wrap=tk.WORD,
            relief=tk.FLAT,
            borderwidth=2,
            highlightbackground=QUANTUM_THEME['quantum_cyan'],
            highlightcolor=QUANTUM_THEME['quantum_purple'],
            insertbackground=QUANTUM_THEME['quantum_green']
        )
        self.ai_status_text.pack(fill=tk.X, pady=(0, 10))
        
        # 예측 목록
        prediction_label = tk.Label(
            right_frame,
            text="Performance Predictions",
            font=('Consolas', 12, 'bold'),
            fg=QUANTUM_THEME['quantum_green'],
            bg=QUANTUM_THEME['void_black']
        )
        prediction_label.pack(pady=(10, 5))
        
        self.prediction_listbox = tk.Listbox(
            right_frame,
            height=10,
            bg=QUANTUM_THEME['dark_matter'],
            fg=QUANTUM_THEME['text_primary'],
            font=('Consolas', 9),
            selectbackground=QUANTUM_THEME['quantum_purple'],
            selectforeground=QUANTUM_THEME['text_quantum'],
            relief=tk.FLAT,
            borderwidth=2,
            highlightbackground=QUANTUM_THEME['quantum_green'],
            highlightcolor=QUANTUM_THEME['quantum_yellow']
        )
        self.prediction_listbox.pack(fill=tk.BOTH, expand=True)
    
    def setup_prediction_chart(self):
        """예측 차트 설정"""
        self.prediction_ax.set_facecolor(QUANTUM_THEME['void_black'])
        self.prediction_fig.patch.set_facecolor(QUANTUM_THEME['void_black'])
        
        # 축 스타일
        self.prediction_ax.spines['bottom'].set_color(QUANTUM_THEME['quantum_cyan'])
        self.prediction_ax.spines['top'].set_color(QUANTUM_THEME['void_black'])
        self.prediction_ax.spines['right'].set_color(QUANTUM_THEME['void_black'])
        self.prediction_ax.spines['left'].set_color(QUANTUM_THEME['quantum_cyan'])
        
        self.prediction_ax.tick_params(axis='x', colors=QUANTUM_THEME['text_secondary'])
        self.prediction_ax.tick_params(axis='y', colors=QUANTUM_THEME['text_secondary'])
        
        self.prediction_ax.set_title(
            "AI Performance Predictions",
            color=QUANTUM_THEME['quantum_purple'],
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        self.prediction_ax.set_xlabel("Time (minutes ahead)", color=QUANTUM_THEME['text_primary'])
        self.prediction_ax.set_ylabel("Predicted Usage (%)", color=QUANTUM_THEME['text_primary'])
        
        self.prediction_ax.grid(True, alpha=0.3, color=QUANTUM_THEME['quantum_cyan'])
    
    def update_ai_panel(self, predictions: Dict[str, Any], metrics: QuantumMetrics):
        """AI 패널 업데이트"""
        try:
            self.predictions = predictions
            self.update_prediction_chart()
            self.update_ai_status(metrics)
            self.update_prediction_list()
            
        except Exception as e:
            print(f"AI 패널 업데이트 오류: {e}")
    
    def update_prediction_chart(self):
        """예측 차트 업데이트"""
        try:
            self.prediction_ax.clear()
            self.setup_prediction_chart()
            
            # 예측 데이터 필터링 (CPU만)
            cpu_predictions = {k: v for k, v in self.predictions.items() if k.startswith('cpu_')}
            
            if not cpu_predictions:
                self.prediction_ax.text(
                    0.5, 0.5, "Training AI Models...",
                    ha='center', va='center',
                    transform=self.prediction_ax.transAxes,
                    fontsize=12, color=QUANTUM_THEME['quantum_yellow']
                )
                self.prediction_canvas.draw_idle()
                return
            
            # 시간 축 생성
            time_horizons = []
            predicted_values = []
            confidences = []
            
            for key, prediction in cpu_predictions.items():
                time_horizon = int(key.split('_')[1].rstrip('m'))
                time_horizons.append(time_horizon)
                predicted_values.append(prediction.predicted_value)
                confidences.append(prediction.confidence)
            
            # 정렬
            sorted_data = sorted(zip(time_horizons, predicted_values, confidences))
            time_horizons, predicted_values, confidences = zip(*sorted_data)
            
            # 예측 라인
            self.prediction_ax.plot(
                time_horizons, predicted_values,
                color=QUANTUM_THEME['quantum_green'],
                linewidth=3,
                marker='o',
                markersize=6,
                label='CPU Prediction',
                alpha=0.9
            )
            
            # 신뢰도 영역
            confidence_upper = [p + (1-c)*20 for p, c in zip(predicted_values, confidences)]
            confidence_lower = [p - (1-c)*20 for p, c in zip(predicted_values, confidences)]
            
            self.prediction_ax.fill_between(
                time_horizons, confidence_lower, confidence_upper,
                color=QUANTUM_THEME['quantum_green'],
                alpha=0.2,
                label='Confidence Band'
            )
            
            # 임계값 선
            self.prediction_ax.axhline(
                y=80, color=QUANTUM_THEME['quantum_red'],
                linestyle='--', linewidth=2, alpha=0.7,
                label='Critical Threshold'
            )
            
            self.prediction_ax.axhline(
                y=60, color=QUANTUM_THEME['quantum_yellow'],
                linestyle='--', linewidth=2, alpha=0.7,
                label='Warning Threshold'
            )
            
            self.prediction_ax.set_ylim(0, 100)
            self.prediction_ax.legend(loc='upper left')
            
            self.prediction_canvas.draw_idle()
            
        except Exception as e:
            print(f"예측 차트 업데이트 오류: {e}")
    
    def update_ai_status(self, metrics: QuantumMetrics):
        """AI 상태 업데이트"""
        try:
            self.ai_status_text.delete(1.0, tk.END)
            
            status_text = f"""🤖 AI Engine: ACTIVE
📊 Data Points: {len(quantum_monitor.ai_engine.history['cpu'])}
🧠 Model Status: {'TRAINED' if quantum_monitor.ai_engine.is_trained else 'TRAINING'}
⚡ Quantum Mode: ENABLED

📈 Current Analysis:
CPU Trend: {self.get_trend_analysis(metrics.cpu_cores)}
Memory Load: {self.get_load_analysis(metrics.memory_percent)}
System Health: {self.get_health_score(metrics)}

🎯 Optimization Score: {self.calculate_optimization_score(metrics):.0f}/100

🔍 Anomaly Detection: {'MONITORING' if quantum_monitor.ai_engine.is_trained else 'CALIBRATING'}
"""
            
            self.ai_status_text.insert(tk.END, status_text)
            
            # 색상 하이라이트
            self.ai_status_text.tag_add("active", "1.13", "1.19")
            self.ai_status_text.tag_config("active", foreground=QUANTUM_THEME['quantum_green'])
            
        except Exception as e:
            print(f"AI 상태 업데이트 오류: {e}")
    
    def update_prediction_list(self):
        """예측 목록 업데이트"""
        try:
            self.prediction_listbox.delete(0, tk.END)
            
            for key, prediction in list(self.predictions.items())[:10]:
                confidence_icon = "🎯" if prediction.confidence > 0.7 else "⚠️" if prediction.confidence > 0.4 else "❓"
                risk_icon = {"low": "✅", "medium": "⚠️", "high": "🔥", "critical": "🚨"}.get(prediction.risk_level, "❓")
                
                list_item = f"{confidence_icon} {prediction.component.upper()}: {prediction.predicted_value:.0f}% ({prediction.time_horizon}m) {risk_icon}"
                self.prediction_listbox.insert(tk.END, list_item)
                
                # 색상 설정 (Listbox는 제한적이므로 간단히)
                if prediction.risk_level in ['high', 'critical']:
                    self.prediction_listbox.itemconfig(tk.END, bg=QUANTUM_THEME['status_critical'])
                elif prediction.risk_level == 'medium':
                    self.prediction_listbox.itemconfig(tk.END, bg=QUANTUM_THEME['status_warning'])
            
        except Exception as e:
            print(f"예측 목록 업데이트 오류: {e}")
    
    def get_trend_analysis(self, cpu_cores):
        """CPU 트렌드 분석"""
        if not cpu_cores:
            return "Unknown"
        
        avg_cpu = np.mean(cpu_cores)
        if avg_cpu < 30:
            return "Optimal"
        elif avg_cpu < 60:
            return "Stable"
        elif avg_cpu < 80:
            return "Elevated"
        else:
            return "Critical"
    
    def get_load_analysis(self, memory_percent):
        """메모리 로드 분석"""
        if memory_percent < 50:
            return "Light"
        elif memory_percent < 70:
            return "Moderate"
        elif memory_percent < 85:
            return "Heavy"
        else:
            return "Critical"
    
    def get_health_score(self, metrics):
        """시스템 건강도 점수"""
        cpu_avg = np.mean(metrics.cpu_cores) if metrics.cpu_cores else 0
        
        cpu_score = max(0, 100 - cpu_avg)
        memory_score = max(0, 100 - metrics.memory_percent)
        
        overall_score = (cpu_score + memory_score) / 2
        
        if overall_score > 80:
            return "Excellent"
        elif overall_score > 60:
            return "Good"
        elif overall_score > 40:
            return "Fair"
        else:
            return "Poor"
    
    def calculate_optimization_score(self, metrics):
        """최적화 점수 계산"""
        cpu_avg = np.mean(metrics.cpu_cores) if metrics.cpu_cores else 0
        
        # 각 메트릭의 최적 범위에서 벗어난 정도 계산
        cpu_penalty = max(0, cpu_avg - 60) * 2
        memory_penalty = max(0, metrics.memory_percent - 70) * 1.5
        
        base_score = 100
        total_penalty = cpu_penalty + memory_penalty
        
        return max(0, base_score - total_penalty)

class QuantumMainGUI:
    """양자 메인 GUI 애플리케이션"""
    
    def __init__(self):
        # 루트 윈도우 생성
        if HAS_CUSTOM_TK:
            self.root = ctk.CTk()
            self.root.configure(fg_color=QUANTUM_THEME['void_black'])
        else:
            self.root = tk.Tk()
            self.root.configure(bg=QUANTUM_THEME['void_black'])
        
        self.setup_main_window()
        self.create_quantum_interface()
        
        # 업데이트 타이머
        self.running = True
        self.update_gui()
    
    def setup_main_window(self):
        """메인 윈도우 설정"""
        self.root.title(f"SysWatch Pro Quantum {VERSION} - {EDITION}")
        self.root.geometry("1920x1080")
        self.root.state('zoomed')  # Windows에서 최대화
        
        # 아이콘 설정 (있는 경우)
        try:
            self.root.iconbitmap("quantum_icon.ico")
        except:
            pass
        
        # 윈도우 스타일
        self.root.configure(bg=QUANTUM_THEME['void_black'])
        
        # 폰트 설정
        self.fonts = {
            'title': ('Consolas', 24, 'bold'),
            'header': ('Consolas', 16, 'bold'),
            'body': ('Consolas', 12),
            'small': ('Consolas', 10),
            'mono': ('Courier New', 10)
        }
    
    def create_quantum_interface(self):
        """양자 인터페이스 생성"""
        # 메인 컨테이너
        main_container = tk.Frame(self.root, bg=QUANTUM_THEME['void_black'])
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # 상단 헤더
        self.create_quantum_header(main_container)
        
        # 중앙 HUD
        hud_frame = tk.Frame(main_container, bg=QUANTUM_THEME['void_black'])
        hud_frame.pack(fill=tk.X, pady=10)
        
        self.quantum_hud = QuantumHUD(hud_frame)
        
        # 메인 콘텐츠 (탭 인터페이스)
        self.create_quantum_tabs(main_container)
        
        # 하단 상태바
        self.create_quantum_statusbar(main_container)
    
    def create_quantum_header(self, parent):
        """양자 헤더 생성"""
        header_frame = tk.Frame(parent, bg=QUANTUM_THEME['deep_space'], height=120)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        # 좌측: 로고 및 제목
        left_frame = tk.Frame(header_frame, bg=QUANTUM_THEME['deep_space'])
        left_frame.pack(side=tk.LEFT, padx=30, pady=20)
        
        # 메인 타이틀
        title_label = tk.Label(
            left_frame,
            text="SYSWATCH PRO QUANTUM",
            font=self.fonts['title'],
            fg=QUANTUM_THEME['quantum_cyan'],
            bg=QUANTUM_THEME['deep_space']
        )
        title_label.pack(anchor='w')
        
        # 서브타이틀
        subtitle_label = tk.Label(
            left_frame,
            text=f"{VERSION} • {EDITION} • {CODENAME}",
            font=self.fonts['body'],
            fg=QUANTUM_THEME['text_secondary'],
            bg=QUANTUM_THEME['deep_space']
        )
        subtitle_label.pack(anchor='w')
        
        # 우측: 시스템 정보
        right_frame = tk.Frame(header_frame, bg=QUANTUM_THEME['deep_space'])
        right_frame.pack(side=tk.RIGHT, padx=30, pady=20)
        
        # 실시간 시계
        self.clock_label = tk.Label(
            right_frame,
            text="",
            font=('Digital-7', 20, 'bold'),
            fg=QUANTUM_THEME['quantum_green'],
            bg=QUANTUM_THEME['deep_space']
        )
        self.clock_label.pack(anchor='e')
        
        # 시스템 정보
        system_info = f"OS: {platform.system()} • CPU: {psutil.cpu_count()} cores • RAM: {psutil.virtual_memory().total/1024**3:.0f}GB"
        system_label = tk.Label(
            right_frame,
            text=system_info,
            font=self.fonts['small'],
            fg=QUANTUM_THEME['text_tertiary'],
            bg=QUANTUM_THEME['deep_space']
        )
        system_label.pack(anchor='e')
    
    def create_quantum_tabs(self, parent):
        """양자 탭 인터페이스 생성"""
        # 탭 컨테이너
        tab_container = tk.Frame(parent, bg=QUANTUM_THEME['void_black'])
        tab_container.pack(fill=tk.BOTH, expand=True, padx=10)
        
        # 커스텀 탭 버튼들
        tab_buttons_frame = tk.Frame(tab_container, bg=QUANTUM_THEME['void_black'], height=50)
        tab_buttons_frame.pack(fill=tk.X, pady=(0, 10))
        tab_buttons_frame.pack_propagate(False)
        
        # 탭 버튼들
        self.tab_buttons = {}
        self.tab_frames = {}
        
        tabs = [
            ("🏠 Overview", "overview"),
            ("📊 3D Analytics", "analytics"),
            ("🎛️ Gauges", "gauges"),
            ("🧠 AI Predictions", "ai"),
            ("⚙️ Settings", "settings")
        ]
        
        # 탭 콘텐츠 프레임
        self.tab_content_frame = tk.Frame(tab_container, bg=QUANTUM_THEME['void_black'])
        self.tab_content_frame.pack(fill=tk.BOTH, expand=True)
        
        for i, (tab_name, tab_key) in enumerate(tabs):
            # 탭 버튼
            btn = tk.Button(
                tab_buttons_frame,
                text=tab_name,
                font=self.fonts['header'],
                fg=QUANTUM_THEME['text_primary'],
                bg=QUANTUM_THEME['dark_matter'],
                activebackground=QUANTUM_THEME['quantum_cyan'],
                activeforeground=QUANTUM_THEME['void_black'],
                relief=tk.FLAT,
                padx=20,
                pady=10,
                command=lambda k=tab_key: self.switch_tab(k)
            )
            btn.pack(side=tk.LEFT, padx=5)
            self.tab_buttons[tab_key] = btn
            
            # 탭 프레임
            frame = tk.Frame(self.tab_content_frame, bg=QUANTUM_THEME['void_black'])
            if i == 0:  # 첫 번째 탭 활성화
                frame.pack(fill=tk.BOTH, expand=True)
            self.tab_frames[tab_key] = frame
        
        # 각 탭 콘텐츠 생성
        self.create_tab_contents()
        self.current_tab = "overview"
        self.update_tab_buttons()
    
    def create_tab_contents(self):
        """탭 콘텐츠 생성"""
        # Overview 탭
        overview_frame = self.tab_frames["overview"]
        self.create_overview_tab(overview_frame)
        
        # 3D Analytics 탭
        analytics_frame = self.tab_frames["analytics"]
        self.quantum_3d_chart = Quantum3DChart(analytics_frame, "Quantum Performance Matrix")
        
        # Gauges 탭
        gauges_frame = self.tab_frames["gauges"]
        self.quantum_gauges = QuantumGaugePanel(gauges_frame)
        
        # AI Predictions 탭
        ai_frame = self.tab_frames["ai"]
        self.quantum_ai_panel = QuantumAIPanel(ai_frame)
        
        # Settings 탭
        settings_frame = self.tab_frames["settings"]
        self.create_settings_tab(settings_frame)
    
    def create_overview_tab(self, parent):
        """개요 탭 생성"""
        # 좌측: 실시간 차트
        left_frame = tk.Frame(parent, bg=QUANTUM_THEME['void_black'])
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # 실시간 성능 차트
        self.overview_fig = Figure(figsize=(8, 6), dpi=100, facecolor=QUANTUM_THEME['void_black'])
        self.overview_ax = self.overview_fig.add_subplot(111)
        self.setup_overview_chart()
        
        self.overview_canvas = FigureCanvasTkAgg(self.overview_fig, left_frame)
        self.overview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 우측: 시스템 정보 및 알림
        right_frame = tk.Frame(parent, bg=QUANTUM_THEME['void_black'], width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_frame.pack_propagate(False)
        
        # 시스템 정보
        info_label = tk.Label(
            right_frame,
            text="📊 System Information",
            font=self.fonts['header'],
            fg=QUANTUM_THEME['quantum_blue'],
            bg=QUANTUM_THEME['void_black']
        )
        info_label.pack(pady=(0, 10))
        
        self.system_info_text = tk.Text(
            right_frame,
            height=12,
            width=45,
            bg=QUANTUM_THEME['dark_matter'],
            fg=QUANTUM_THEME['text_primary'],
            font=self.fonts['mono'],
            wrap=tk.WORD,
            relief=tk.FLAT,
            borderwidth=2,
            highlightbackground=QUANTUM_THEME['quantum_blue'],
            highlightcolor=QUANTUM_THEME['quantum_cyan']
        )
        self.system_info_text.pack(fill=tk.X, pady=(0, 15))
        
        # 알림 패널
        alerts_label = tk.Label(
            right_frame,
            text="🚨 Quantum Alerts",
            font=self.fonts['header'],
            fg=QUANTUM_THEME['quantum_red'],
            bg=QUANTUM_THEME['void_black']
        )
        alerts_label.pack(pady=(0, 5))
        
        self.alerts_listbox = tk.Listbox(
            right_frame,
            height=15,
            bg=QUANTUM_THEME['dark_matter'],
            fg=QUANTUM_THEME['text_primary'],
            font=self.fonts['small'],
            selectbackground=QUANTUM_THEME['quantum_red'],
            selectforeground=QUANTUM_THEME['text_quantum'],
            relief=tk.FLAT,
            borderwidth=2,
            highlightbackground=QUANTUM_THEME['quantum_red'],
            highlightcolor=QUANTUM_THEME['quantum_orange']
        )
        self.alerts_listbox.pack(fill=tk.BOTH, expand=True)
        
        # 데이터 히스토리
        self.overview_data = {
            'cpu': deque(maxlen=100),
            'memory': deque(maxlen=100),
            'disk': deque(maxlen=100),
            'network': deque(maxlen=100),
            'timestamps': deque(maxlen=100)
        }
    
    def setup_overview_chart(self):
        """개요 차트 설정"""
        self.overview_ax.set_facecolor(QUANTUM_THEME['void_black'])
        self.overview_fig.patch.set_facecolor(QUANTUM_THEME['void_black'])
        
        # 축 스타일
        for spine in self.overview_ax.spines.values():
            spine.set_color(QUANTUM_THEME['quantum_cyan'])
        
        self.overview_ax.tick_params(axis='x', colors=QUANTUM_THEME['text_secondary'])
        self.overview_ax.tick_params(axis='y', colors=QUANTUM_THEME['text_secondary'])
        
        self.overview_ax.set_title(
            "Real-time Performance Monitoring",
            color=QUANTUM_THEME['quantum_cyan'],
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        
        self.overview_ax.set_xlabel("Time", color=QUANTUM_THEME['text_primary'])
        self.overview_ax.set_ylabel("Usage (%)", color=QUANTUM_THEME['text_primary'])
        
        self.overview_ax.grid(True, alpha=0.3, color=QUANTUM_THEME['quantum_cyan'])
        self.overview_ax.set_ylim(0, 100)
    
    def create_settings_tab(self, parent):
        """설정 탭 생성"""
        # 설정 컨테이너
        settings_container = tk.Frame(parent, bg=QUANTUM_THEME['void_black'])
        settings_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 제목
        title_label = tk.Label(
            settings_container,
            text="⚙️ QUANTUM CONFIGURATION",
            font=self.fonts['title'],
            fg=QUANTUM_THEME['quantum_purple'],
            bg=QUANTUM_THEME['void_black']
        )
        title_label.pack(pady=(0, 30))
        
        # 설정 섹션들
        sections_frame = tk.Frame(settings_container, bg=QUANTUM_THEME['void_black'])
        sections_frame.pack(fill=tk.BOTH, expand=True)
        
        # 좌측: 모니터링 설정
        left_settings = tk.Frame(sections_frame, bg=QUANTUM_THEME['void_black'])
        left_settings.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))
        
        self.create_monitoring_settings(left_settings)
        
        # 우측: AI 설정
        right_settings = tk.Frame(sections_frame, bg=QUANTUM_THEME['void_black'])
        right_settings.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(20, 0))
        
        self.create_ai_settings(right_settings)
    
    def create_monitoring_settings(self, parent):
        """모니터링 설정 생성"""
        # 섹션 제목
        section_label = tk.Label(
            parent,
            text="📊 Monitoring Settings",
            font=self.fonts['header'],
            fg=QUANTUM_THEME['quantum_blue'],
            bg=QUANTUM_THEME['void_black']
        )
        section_label.pack(anchor='w', pady=(0, 15))
        
        # 업데이트 간격
        interval_frame = tk.Frame(parent, bg=QUANTUM_THEME['void_black'])
        interval_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(
            interval_frame,
            text="Update Interval (seconds):",
            font=self.fonts['body'],
            fg=QUANTUM_THEME['text_primary'],
            bg=QUANTUM_THEME['void_black']
        ).pack(side=tk.LEFT)
        
        self.interval_var = tk.DoubleVar(value=1.0)
        interval_scale = tk.Scale(
            interval_frame,
            from_=0.1, to=5.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.interval_var,
            bg=QUANTUM_THEME['dark_matter'],
            fg=QUANTUM_THEME['text_primary'],
            highlightbackground=QUANTUM_THEME['quantum_blue'],
            troughcolor=QUANTUM_THEME['cosmic_dust']
        )
        interval_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
        
        # 임계값 설정
        thresholds_label = tk.Label(
            parent,
            text="Alert Thresholds:",
            font=self.fonts['body'],
            fg=QUANTUM_THEME['text_primary'],
            bg=QUANTUM_THEME['void_black']
        )
        thresholds_label.pack(anchor='w', pady=(20, 10))
        
        # CPU 임계값
        cpu_frame = tk.Frame(parent, bg=QUANTUM_THEME['void_black'])
        cpu_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(cpu_frame, text="CPU:", font=self.fonts['body'], 
                fg=QUANTUM_THEME['text_primary'], bg=QUANTUM_THEME['void_black']).pack(side=tk.LEFT)
        
        self.cpu_threshold_var = tk.DoubleVar(value=80.0)
        cpu_scale = tk.Scale(
            cpu_frame, from_=50, to=95, orient=tk.HORIZONTAL,
            variable=self.cpu_threshold_var,
            bg=QUANTUM_THEME['dark_matter'], fg=QUANTUM_THEME['text_primary']
        )
        cpu_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
        
        # Memory 임계값
        mem_frame = tk.Frame(parent, bg=QUANTUM_THEME['void_black'])
        mem_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(mem_frame, text="Memory:", font=self.fonts['body'],
                fg=QUANTUM_THEME['text_primary'], bg=QUANTUM_THEME['void_black']).pack(side=tk.LEFT)
        
        self.memory_threshold_var = tk.DoubleVar(value=85.0)
        mem_scale = tk.Scale(
            mem_frame, from_=50, to=95, orient=tk.HORIZONTAL,
            variable=self.memory_threshold_var,
            bg=QUANTUM_THEME['dark_matter'], fg=QUANTUM_THEME['text_primary']
        )
        mem_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
    
    def create_ai_settings(self, parent):
        """AI 설정 생성"""
        # 섹션 제목
        section_label = tk.Label(
            parent,
            text="🧠 AI Engine Settings",
            font=self.fonts['header'],
            fg=QUANTUM_THEME['quantum_green'],
            bg=QUANTUM_THEME['void_black']
        )
        section_label.pack(anchor='w', pady=(0, 15))
        
        # AI 활성화
        self.ai_enabled_var = tk.BooleanVar(value=True)
        ai_check = tk.Checkbutton(
            parent,
            text="Enable AI Predictions",
            variable=self.ai_enabled_var,
            font=self.fonts['body'],
            fg=QUANTUM_THEME['text_primary'],
            bg=QUANTUM_THEME['void_black'],
            selectcolor=QUANTUM_THEME['dark_matter'],
            activebackground=QUANTUM_THEME['void_black'],
            activeforeground=QUANTUM_THEME['quantum_green']
        )
        ai_check.pack(anchor='w', pady=5)
        
        # 이상 탐지
        self.anomaly_detection_var = tk.BooleanVar(value=True)
        anomaly_check = tk.Checkbutton(
            parent,
            text="Enable Anomaly Detection",
            variable=self.anomaly_detection_var,
            font=self.fonts['body'],
            fg=QUANTUM_THEME['text_primary'],
            bg=QUANTUM_THEME['void_black'],
            selectcolor=QUANTUM_THEME['dark_matter'],
            activebackground=QUANTUM_THEME['void_black'],
            activeforeground=QUANTUM_THEME['quantum_green']
        )
        anomaly_check.pack(anchor='w', pady=5)
        
        # 예측 모델 훈련 버튼
        train_button = tk.Button(
            parent,
            text="🎯 Train AI Models",
            font=self.fonts['body'],
            fg=QUANTUM_THEME['void_black'],
            bg=QUANTUM_THEME['quantum_green'],
            activebackground=QUANTUM_THEME['quantum_yellow'],
            relief=tk.FLAT,
            padx=20,
            pady=10,
            command=self.train_ai_models
        )
        train_button.pack(pady=20)
        
        # AI 상태 표시
        self.ai_status_label = tk.Label(
            parent,
            text="AI Status: Ready",
            font=self.fonts['body'],
            fg=QUANTUM_THEME['quantum_cyan'],
            bg=QUANTUM_THEME['void_black']
        )
        self.ai_status_label.pack(pady=10)
    
    def create_quantum_statusbar(self, parent):
        """양자 상태바 생성"""
        statusbar_frame = tk.Frame(parent, bg=QUANTUM_THEME['cosmic_dust'], height=40)
        statusbar_frame.pack(fill=tk.X)
        statusbar_frame.pack_propagate(False)
        
        # 좌측: 상태 메시지
        self.status_label = tk.Label(
            statusbar_frame,
            text="🚀 Quantum monitoring system active",
            font=self.fonts['small'],
            fg=QUANTUM_THEME['quantum_green'],
            bg=QUANTUM_THEME['cosmic_dust']
        )
        self.status_label.pack(side=tk.LEFT, padx=15, pady=8)
        
        # 우측: 성능 지표
        performance_frame = tk.Frame(statusbar_frame, bg=QUANTUM_THEME['cosmic_dust'])
        performance_frame.pack(side=tk.RIGHT, padx=15, pady=8)
        
        self.perf_label = tk.Label(
            performance_frame,
            text="FPS: 60 | Data Rate: 100% | AI: Active",
            font=self.fonts['small'],
            fg=QUANTUM_THEME['text_secondary'],
            bg=QUANTUM_THEME['cosmic_dust']
        )
        self.perf_label.pack()
    
    def switch_tab(self, tab_key):
        """탭 전환"""
        # 모든 탭 숨기기
        for frame in self.tab_frames.values():
            frame.pack_forget()
        
        # 선택된 탭 표시
        self.tab_frames[tab_key].pack(fill=tk.BOTH, expand=True)
        self.current_tab = tab_key
        self.update_tab_buttons()
    
    def update_tab_buttons(self):
        """탭 버튼 스타일 업데이트"""
        for key, button in self.tab_buttons.items():
            if key == self.current_tab:
                button.configure(
                    bg=QUANTUM_THEME['quantum_cyan'],
                    fg=QUANTUM_THEME['void_black']
                )
            else:
                button.configure(
                    bg=QUANTUM_THEME['dark_matter'],
                    fg=QUANTUM_THEME['text_primary']
                )
    
    def train_ai_models(self):
        """AI 모델 훈련"""
        self.ai_status_label.configure(text="AI Status: Training...", fg=QUANTUM_THEME['quantum_yellow'])
        
        def train_in_background():
            success = quantum_monitor.ai_engine.train_models()
            
            if success:
                self.root.after(0, lambda: self.ai_status_label.configure(
                    text="AI Status: Trained ✅", fg=QUANTUM_THEME['quantum_green']))
            else:
                self.root.after(0, lambda: self.ai_status_label.configure(
                    text="AI Status: Training Failed ❌", fg=QUANTUM_THEME['quantum_red']))
        
        threading.Thread(target=train_in_background, daemon=True).start()
    
    def update_gui(self):
        """GUI 업데이트"""
        if not self.running:
            return
        
        try:
            # 현재 메트릭스 가져오기
            metrics = quantum_monitor.get_current_metrics()
            
            # 시계 업데이트
            current_time = datetime.now().strftime("%H:%M:%S")
            self.clock_label.configure(text=current_time)
            
            # HUD 업데이트
            self.quantum_hud.update_status(metrics)
            
            # 현재 탭에 따른 업데이트
            if self.current_tab == "overview":
                self.update_overview_tab(metrics)
            elif self.current_tab == "analytics":
                self.quantum_3d_chart.update_data(metrics)
            elif self.current_tab == "gauges":
                self.quantum_gauges.update_all_gauges(metrics)
            elif self.current_tab == "ai":
                predictions = quantum_monitor.get_predictions()
                self.quantum_ai_panel.update_ai_panel(predictions, metrics)
            
            # 상태바 업데이트
            ai_status = "Active" if quantum_monitor.ai_engine.is_trained else "Training"
            self.perf_label.configure(text=f"FPS: 30 | Data Rate: 100% | AI: {ai_status}")
            
        except Exception as e:
            print(f"GUI 업데이트 오류: {e}")
        
        # 다음 업데이트 스케줄 (30 FPS)
        self.root.after(33, self.update_gui)
    
    def update_overview_tab(self, metrics: QuantumMetrics):
        """개요 탭 업데이트"""
        try:
            # 데이터 추가
            cpu_avg = np.mean(metrics.cpu_cores) if metrics.cpu_cores else 0
            self.overview_data['cpu'].append(cpu_avg)
            self.overview_data['memory'].append(metrics.memory_percent)
            self.overview_data['disk'].append(min(100, (metrics.disk_read + metrics.disk_write) * 10))
            self.overview_data['network'].append(min(100, (metrics.network_sent + metrics.network_recv) * 10))
            self.overview_data['timestamps'].append(time.time())
            
            # 차트 업데이트
            if len(self.overview_data['cpu']) > 1:
                self.overview_ax.clear()
                self.setup_overview_chart()
                
                x_data = list(range(len(self.overview_data['cpu'])))
                
                # 성능 라인들
                self.overview_ax.plot(x_data, list(self.overview_data['cpu']), 
                                    color=QUANTUM_THEME['quantum_red'], linewidth=2, label='CPU', alpha=0.9)
                self.overview_ax.plot(x_data, list(self.overview_data['memory']), 
                                    color=QUANTUM_THEME['quantum_yellow'], linewidth=2, label='Memory', alpha=0.9)
                self.overview_ax.plot(x_data, list(self.overview_data['disk']), 
                                    color=QUANTUM_THEME['quantum_blue'], linewidth=2, label='Disk I/O', alpha=0.9)
                self.overview_ax.plot(x_data, list(self.overview_data['network']), 
                                    color=QUANTUM_THEME['quantum_green'], linewidth=2, label='Network', alpha=0.9)
                
                # 임계값 선
                self.overview_ax.axhline(y=80, color=QUANTUM_THEME['quantum_orange'], 
                                       linestyle='--', alpha=0.7, label='High Threshold')
                
                self.overview_ax.legend(loc='upper left')
                self.overview_ax.set_xlim(max(0, len(x_data) - 50), len(x_data))
                
                self.overview_canvas.draw_idle()
            
            # 시스템 정보 업데이트
            self.update_system_info(metrics)
            
            # 알림 업데이트
            self.update_alerts_list()
            
        except Exception as e:
            print(f"개요 탭 업데이트 오류: {e}")
    
    def update_system_info(self, metrics: QuantumMetrics):
        """시스템 정보 업데이트"""
        try:
            self.system_info_text.delete(1.0, tk.END)
            
            cpu_avg = np.mean(metrics.cpu_cores) if metrics.cpu_cores else 0
            memory_gb = metrics.memory_used / 1024**3
            memory_total_gb = (metrics.memory_used + metrics.memory_available) / 1024**3
            
            info_text = f"""🖥️  SYSTEM STATUS
CPU Usage:     {cpu_avg:6.1f}%
CPU Cores:     {len(metrics.cpu_cores) if metrics.cpu_cores else 0}
CPU Frequency: {metrics.cpu_freq:6.0f} MHz

💾 MEMORY STATUS  
Memory Usage:  {metrics.memory_percent:6.1f}%
Memory Used:   {memory_gb:6.1f} GB
Memory Total:  {memory_total_gb:6.1f} GB

💽 STORAGE I/O
Disk Read:     {metrics.disk_read:6.1f} MB/s
Disk Write:    {metrics.disk_write:6.1f} MB/s

🌐 NETWORK I/O
Network ↑:     {metrics.network_sent:6.1f} MB/s  
Network ↓:     {metrics.network_recv:6.1f} MB/s

🎮 GPU STATUS
GPU Usage:     {metrics.gpu_usage:6.1f}%

📊 PROCESSES
Active:        {metrics.process_count}
Threads:       {metrics.thread_count}

⏱️  UPTIME
System:        {metrics.uptime/3600:6.1f} hours
"""
            
            self.system_info_text.insert(tk.END, info_text)
            
        except Exception as e:
            print(f"시스템 정보 업데이트 오류: {e}")
    
    def update_alerts_list(self):
        """알림 목록 업데이트"""
        try:
            self.alerts_listbox.delete(0, tk.END)
            
            recent_alerts = quantum_monitor.get_recent_alerts(20)
            
            if not recent_alerts:
                self.alerts_listbox.insert(tk.END, "✅ No active alerts - System optimal")
                return
            
            for alert in reversed(recent_alerts[-10:]):  # 최근 10개만
                timestamp_str = datetime.fromtimestamp(alert.timestamp).strftime("%H:%M:%S")
                severity_icon = {
                    'critical': '🚨',
                    'warning': '⚠️',
                    'info': 'ℹ️'
                }.get(alert.severity, '❓')
                
                alert_text = f"{severity_icon} [{timestamp_str}] {alert.title}"
                self.alerts_listbox.insert(0, alert_text)  # 최신이 위로
                
                # 색상 설정
                color = {
                    'critical': QUANTUM_THEME['quantum_red'],
                    'warning': QUANTUM_THEME['quantum_orange'],
                    'info': QUANTUM_THEME['quantum_blue']
                }.get(alert.severity, QUANTUM_THEME['text_primary'])
                
                self.alerts_listbox.itemconfig(0, fg=color)
            
        except Exception as e:
            print(f"알림 목록 업데이트 오류: {e}")
    
    def on_closing(self):
        """종료 처리"""
        self.running = False
        quantum_monitor.stop_monitoring()
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """GUI 실행"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 모니터링 시작
        quantum_monitor.start_monitoring()
        
        print(f"{QUANTUM_THEME['quantum_green']}🌌 Quantum GUI interface launched!")
        print(f"{QUANTUM_THEME['quantum_cyan']}📡 Real-time monitoring active")
        print(f"{QUANTUM_THEME['quantum_purple']}🧠 AI engine initializing...")
        
        self.root.mainloop()

def main():
    """메인 함수"""
    try:
        # GUI 애플리케이션 생성 및 실행
        app = QuantumMainGUI()
        app.run()
        
    except KeyboardInterrupt:
        print(f"\n{QUANTUM_THEME['quantum_red']}🛑 GUI shutdown requested")
    except Exception as e:
        print(f"{QUANTUM_THEME['quantum_red']}❌ GUI Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"{QUANTUM_THEME['quantum_green']}✅ Quantum GUI shutdown complete")

if __name__ == "__main__":
    main()