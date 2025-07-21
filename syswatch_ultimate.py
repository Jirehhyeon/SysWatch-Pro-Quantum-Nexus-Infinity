#!/usr/bin/env python3
"""
SysWatch Pro Ultimate - 최고급 시각화 + 자동 프로세스 관리
고품질 그래프, 3D 시각화, 자동 시스템 최적화
"""

import sys
import time
import threading
import psutil
import os
import json
import queue
import signal
from datetime import datetime, timedelta
from collections import deque, defaultdict
import tkinter as tk
from tkinter import ttk, messagebox, font
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Wedge, Rectangle, FancyBboxPatch
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
import matplotlib.patheffects as path_effects
import numpy as np
import platform

VERSION = "7.0.0 Ultimate"
EDITION = "Ultimate Professional Edition"

# 프리미엄 시각 테마
VISUAL_THEME = {
    'bg_primary': '#0a0a0f',
    'bg_secondary': '#0f0f1a',
    'bg_tertiary': '#1a1a2e',
    'bg_card': '#16213e',
    'bg_gradient_start': '#0f3460',
    'bg_gradient_end': '#16537e',
    'text_primary': '#ffffff',
    'text_secondary': '#b8c5d1',
    'text_accent': '#e8f4fd',
    'neon_blue': '#00f5ff',
    'neon_cyan': '#00ffff',
    'neon_green': '#39ff14',
    'neon_yellow': '#ffff00',
    'neon_orange': '#ff8c00',
    'neon_red': '#ff073a',
    'neon_purple': '#bf00ff',
    'neon_pink': '#ff1493',
    'glow_blue': '#4d79a4',
    'glow_green': '#5cb85c',
    'glow_orange': '#f0ad4e',
    'glow_red': '#d9534f',
    'glass_effect': '#ffffff20',
    'shadow': '#000000aa'
}

# 자동 종료할 프로세스 목록 (시스템 리소스 절약)
AUTO_TERMINATE_PROCESSES = [
    # 브라우저 관련 (메모리 많이 사용)
    'chrome.exe', 'msedge.exe', 'firefox.exe', 'opera.exe',
    # 메신저/소셜
    'discord.exe', 'telegram.exe', 'kakaotalk.exe', 'slack.exe',
    # 게임 관련
    'steam.exe', 'epicgameslauncher.exe', 'origin.exe', 'uplay.exe',
    # 미디어
    'spotify.exe', 'vlc.exe', 'potplayer.exe',
    # 기타 리소스 집약적
    'obs64.exe', 'obs32.exe', 'streamlabs obs.exe',
    # 개발 도구 (선택적)
    'devenv.exe', 'code.exe', 'pycharm64.exe', 'intellij.exe'
]

class UltimateSystemMonitor:
    def __init__(self):
        self.root = tk.Tk()
        self.setup_ultimate_window()
        
        # 데이터 저장소 (더 많은 데이터 포인트)
        self.metrics = {
            'cpu_total': deque(maxlen=500),
            'cpu_cores': defaultdict(lambda: deque(maxlen=500)),
            'memory_percent': deque(maxlen=500),
            'memory_used': deque(maxlen=500),
            'memory_cached': deque(maxlen=500),
            'disk_read': deque(maxlen=500),
            'disk_write': deque(maxlen=500),
            'disk_usage': deque(maxlen=500),
            'net_sent': deque(maxlen=500),
            'net_recv': deque(maxlen=500),
            'gpu_usage': deque(maxlen=500),
            'gpu_memory': deque(maxlen=500),
            'temps': defaultdict(lambda: deque(maxlen=500)),
            'timestamps': deque(maxlen=500)
        }
        
        # 실시간 상태
        self.current_stats = {
            'cpu_total': 0,
            'cpu_freq': 0,
            'cpu_temp': 0,
            'memory_percent': 0,
            'memory_total': 0,
            'memory_used': 0,
            'memory_available': 0,
            'disk_total': 0,
            'disk_used': 0,
            'network_connections': 0,
            'processes': [],
            'high_cpu_processes': [],
            'high_memory_processes': [],
            'terminated_processes': []
        }
        
        # 자동 최적화 설정
        self.optimization_enabled = True
        self.cpu_threshold = 80
        self.memory_threshold = 85
        self.auto_terminate_enabled = True
        
        # 스레드 관리
        self.running = True
        self.data_lock = threading.Lock()
        
        # GUI 생성
        self.create_ultimate_ui()
        self.setup_premium_charts()
        
        # I/O 추적
        self.prev_disk_io = psutil.disk_io_counters()
        self.prev_net_io = psutil.net_io_counters()
        self.prev_time = time.time()

    def setup_ultimate_window(self):
        """최고급 윈도우 설정"""
        self.root.title(f"SysWatch Pro {VERSION} - Ultimate Performance Suite")
        self.root.geometry("3440x1440")  # 울트라와이드 지원
        self.root.configure(bg=VISUAL_THEME['bg_primary'])
        self.root.state('zoomed')
        
        # 고급 폰트
        self.fonts = {
            'title': font.Font(family='Segoe UI', size=24, weight='bold'),
            'header': font.Font(family='Segoe UI', size=14, weight='bold'),
            'body': font.Font(family='Segoe UI', size=10),
            'mono': font.Font(family='Consolas', size=10),
            'large_mono': font.Font(family='Consolas', size=16, weight='bold')
        }

    def create_ultimate_ui(self):
        """최고급 UI 생성"""
        # 메인 컨테이너
        main_container = tk.Frame(self.root, bg=VISUAL_THEME['bg_primary'])
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # 헤더
        self.create_premium_header(main_container)
        
        # 메인 콘텐츠 (4열 레이아웃)
        content_frame = tk.Frame(main_container, bg=VISUAL_THEME['bg_primary'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 좌측: 시스템 개요 & 컨트롤
        left_panel = tk.Frame(content_frame, bg=VISUAL_THEME['bg_primary'], width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_panel.pack_propagate(False)
        
        # 중앙 좌측: 주요 차트
        center_left = tk.Frame(content_frame, bg=VISUAL_THEME['bg_primary'])
        center_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # 중앙 우측: 보조 차트
        center_right = tk.Frame(content_frame, bg=VISUAL_THEME['bg_primary'])
        center_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # 우측: 프로세스 관리 & 최적화
        right_panel = tk.Frame(content_frame, bg=VISUAL_THEME['bg_primary'], width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_panel.pack_propagate(False)
        
        # 각 패널 구성
        self.create_system_control_panel(left_panel)
        self.create_main_charts(center_left)
        self.create_secondary_charts(center_right)
        self.create_process_management(right_panel)
        
        # 하단 상태바
        self.create_premium_status_bar(main_container)

    def create_premium_header(self, parent):
        """프리미엄 헤더"""
        header = tk.Frame(parent, bg=VISUAL_THEME['bg_secondary'], height=100)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        # 그라디언트 효과 (Canvas로 구현)
        header_canvas = tk.Canvas(header, bg=VISUAL_THEME['bg_secondary'], height=100, highlightthickness=0)
        header_canvas.pack(fill=tk.BOTH)
        
        # 제목 영역
        title_frame = tk.Frame(header_canvas, bg=VISUAL_THEME['bg_secondary'])
        header_canvas.create_window(20, 20, window=title_frame, anchor='nw')
        
        # 메인 타이틀 (네온 효과)
        main_title = tk.Label(
            title_frame,
            text="SYSWATCH PRO ULTIMATE",
            font=self.fonts['title'],
            fg=VISUAL_THEME['neon_cyan'],
            bg=VISUAL_THEME['bg_secondary']
        )
        main_title.pack(anchor='w')
        
        # 서브 타이틀
        sub_title = tk.Label(
            title_frame,
            text="Advanced Performance Suite with AI Optimization",
            font=('Segoe UI', 12),
            fg=VISUAL_THEME['text_secondary'],
            bg=VISUAL_THEME['bg_secondary']
        )
        sub_title.pack(anchor='w')
        
        # 실시간 정보 영역
        info_frame = tk.Frame(header_canvas, bg=VISUAL_THEME['bg_secondary'])
        header_canvas.create_window(header_canvas.winfo_reqwidth()-20, 20, window=info_frame, anchor='ne')
        
        # 시간 (대형 디지털 시계)
        self.digital_clock = tk.Label(
            info_frame,
            text="",
            font=self.fonts['large_mono'],
            fg=VISUAL_THEME['neon_green'],
            bg=VISUAL_THEME['bg_secondary']
        )
        self.digital_clock.pack(anchor='e')
        
        # 시스템 상태 표시기
        status_frame = tk.Frame(info_frame, bg=VISUAL_THEME['bg_secondary'])
        status_frame.pack(anchor='e', pady=(5, 0))
        
        self.status_indicators = {}
        indicators = [
            ('cpu', 'CPU', VISUAL_THEME['neon_blue']),
            ('memory', 'MEM', VISUAL_THEME['neon_yellow']),
            ('disk', 'DISK', VISUAL_THEME['neon_orange']),
            ('network', 'NET', VISUAL_THEME['neon_purple'])
        ]
        
        for key, label, color in indicators:
            indicator = tk.Label(
                status_frame,
                text=f"● {label}",
                font=('Segoe UI', 10, 'bold'),
                fg=color,
                bg=VISUAL_THEME['bg_secondary']
            )
            indicator.pack(side=tk.LEFT, padx=5)
            self.status_indicators[key] = indicator

    def create_glass_card(self, parent, title, width=None, height=None):
        """글래스 효과 카드 생성"""
        card = tk.Frame(parent, bg=VISUAL_THEME['bg_card'], relief=tk.FLAT, bd=1)
        if width:
            card.configure(width=width)
        if height:
            card.configure(height=height)
        
        # 헤더
        header = tk.Frame(card, bg=VISUAL_THEME['bg_tertiary'], height=40)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        # 타이틀 (글로우 효과)
        title_label = tk.Label(
            header,
            text=title,
            font=self.fonts['header'],
            fg=VISUAL_THEME['neon_cyan'],
            bg=VISUAL_THEME['bg_tertiary']
        )
        title_label.pack(side=tk.LEFT, padx=15, pady=8)
        
        return card

    def create_system_control_panel(self, parent):
        """시스템 제어 패널"""
        # 시스템 정보 카드
        sys_info_card = self.create_glass_card(parent, "🖥️ SYSTEM STATUS")
        sys_info_card.pack(fill=tk.X, padx=5, pady=5)
        
        # 원형 게이지들
        gauges_frame = tk.Frame(sys_info_card, bg=VISUAL_THEME['bg_card'])
        gauges_frame.pack(fill=tk.X, padx=15, pady=15)
        
        self.create_circular_gauges(gauges_frame)
        
        # 자동 최적화 제어
        optimization_card = self.create_glass_card(parent, "⚡ AUTO OPTIMIZATION")
        optimization_card.pack(fill=tk.X, padx=5, pady=5)
        
        self.create_optimization_controls(optimization_card)
        
        # 실시간 경고
        alerts_card = self.create_glass_card(parent, "⚠️ SYSTEM ALERTS")
        alerts_card.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.create_alerts_panel(alerts_card)

    def create_circular_gauges(self, parent):
        """원형 게이지 생성"""
        # CPU 게이지
        cpu_frame = tk.Frame(parent, bg=VISUAL_THEME['bg_card'])
        cpu_frame.pack(fill=tk.X, pady=5)
        
        self.cpu_gauge_fig = Figure(figsize=(4, 2), dpi=100, facecolor=VISUAL_THEME['bg_card'])
        self.cpu_gauge_ax = self.cpu_gauge_fig.add_subplot(111)
        self.setup_circular_gauge(self.cpu_gauge_ax, "CPU", VISUAL_THEME['neon_blue'])
        
        cpu_canvas = FigureCanvasTkAgg(self.cpu_gauge_fig, cpu_frame)
        cpu_canvas.get_tk_widget().pack()
        
        # 메모리 게이지
        mem_frame = tk.Frame(parent, bg=VISUAL_THEME['bg_card'])
        mem_frame.pack(fill=tk.X, pady=5)
        
        self.mem_gauge_fig = Figure(figsize=(4, 2), dpi=100, facecolor=VISUAL_THEME['bg_card'])
        self.mem_gauge_ax = self.mem_gauge_fig.add_subplot(111)
        self.setup_circular_gauge(self.mem_gauge_ax, "MEMORY", VISUAL_THEME['neon_yellow'])
        
        mem_canvas = FigureCanvasTkAgg(self.mem_gauge_fig, mem_frame)
        mem_canvas.get_tk_widget().pack()

    def setup_circular_gauge(self, ax, title, color):
        """원형 게이지 설정"""
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 배경 원
        background_circle = Circle((0, 0), 1, fill=False, linewidth=8, 
                                 edgecolor=VISUAL_THEME['bg_tertiary'])
        ax.add_patch(background_circle)
        
        # 제목
        ax.text(0, 1.4, title, ha='center', va='center', 
                fontsize=12, fontweight='bold', color=VISUAL_THEME['text_primary'])
        
        # 값 표시
        ax.text(0, 0, "0%", ha='center', va='center', 
                fontsize=16, fontweight='bold', color=color)

    def create_optimization_controls(self, parent):
        """최적화 제어 패널"""
        controls_frame = tk.Frame(parent, bg=VISUAL_THEME['bg_card'])
        controls_frame.pack(fill=tk.X, padx=15, pady=15)
        
        # 자동 최적화 토글
        self.auto_opt_var = tk.BooleanVar(value=True)
        auto_opt_check = tk.Checkbutton(
            controls_frame,
            text="Enable Auto-Optimization",
            variable=self.auto_opt_var,
            font=self.fonts['body'],
            fg=VISUAL_THEME['text_primary'],
            bg=VISUAL_THEME['bg_card'],
            selectcolor=VISUAL_THEME['bg_tertiary'],
            activebackground=VISUAL_THEME['bg_card'],
            activeforeground=VISUAL_THEME['neon_green'],
            command=self.toggle_optimization
        )
        auto_opt_check.pack(anchor='w', pady=2)
        
        # 프로세스 자동 종료 토글
        self.auto_term_var = tk.BooleanVar(value=True)
        auto_term_check = tk.Checkbutton(
            controls_frame,
            text="Auto-terminate Resource Hogs",
            variable=self.auto_term_var,
            font=self.fonts['body'],
            fg=VISUAL_THEME['text_primary'],
            bg=VISUAL_THEME['bg_card'],
            selectcolor=VISUAL_THEME['bg_tertiary'],
            activebackground=VISUAL_THEME['bg_card'],
            activeforeground=VISUAL_THEME['neon_green'],
            command=self.toggle_auto_terminate
        )
        auto_term_check.pack(anchor='w', pady=2)
        
        # 임계값 설정
        threshold_frame = tk.Frame(controls_frame, bg=VISUAL_THEME['bg_card'])
        threshold_frame.pack(fill=tk.X, pady=(10, 0))
        
        tk.Label(
            threshold_frame,
            text="CPU Threshold:",
            font=self.fonts['body'],
            fg=VISUAL_THEME['text_secondary'],
            bg=VISUAL_THEME['bg_card']
        ).pack(anchor='w')
        
        self.cpu_threshold_var = tk.IntVar(value=80)
        cpu_scale = tk.Scale(
            threshold_frame,
            from_=50, to=95,
            orient=tk.HORIZONTAL,
            variable=self.cpu_threshold_var,
            bg=VISUAL_THEME['bg_card'],
            fg=VISUAL_THEME['text_primary'],
            highlightthickness=0,
            troughcolor=VISUAL_THEME['bg_tertiary']
        )
        cpu_scale.pack(fill=tk.X, pady=2)
        
        # 수동 최적화 버튼
        optimize_btn = tk.Button(
            controls_frame,
            text="🚀 OPTIMIZE NOW",
            font=('Segoe UI', 11, 'bold'),
            fg=VISUAL_THEME['bg_primary'],
            bg=VISUAL_THEME['neon_green'],
            activebackground=VISUAL_THEME['glow_green'],
            relief=tk.FLAT,
            pady=8,
            command=self.manual_optimize
        )
        optimize_btn.pack(fill=tk.X, pady=(10, 0))

    def create_alerts_panel(self, parent):
        """경고 패널"""
        alerts_frame = tk.Frame(parent, bg=VISUAL_THEME['bg_card'])
        alerts_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # 스크롤 가능한 경고 리스트
        self.alerts_canvas = tk.Canvas(alerts_frame, bg=VISUAL_THEME['bg_card'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(alerts_frame, orient="vertical", command=self.alerts_canvas.yview)
        self.alerts_scrollable = tk.Frame(self.alerts_canvas, bg=VISUAL_THEME['bg_card'])
        
        self.alerts_scrollable.bind(
            "<Configure>",
            lambda e: self.alerts_canvas.configure(scrollregion=self.alerts_canvas.bbox("all"))
        )
        
        self.alerts_canvas.create_window((0, 0), window=self.alerts_scrollable, anchor="nw")
        self.alerts_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.alerts_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def create_main_charts(self, parent):
        """메인 차트 영역"""
        # CPU 멀티코어 차트
        cpu_card = self.create_glass_card(parent, "🔥 CPU PERFORMANCE (Multi-Core)")
        cpu_card.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.create_advanced_cpu_chart(cpu_card)
        
        # 메모리 상세 분석
        memory_card = self.create_glass_card(parent, "🧠 MEMORY ANALYSIS")
        memory_card.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.create_advanced_memory_chart(memory_card)

    def create_secondary_charts(self, parent):
        """보조 차트 영역"""
        # 디스크 I/O
        disk_card = self.create_glass_card(parent, "💾 DISK I/O PERFORMANCE")
        disk_card.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.create_advanced_disk_chart(disk_card)
        
        # 네트워크 트래픽
        network_card = self.create_glass_card(parent, "🌐 NETWORK TRAFFIC")
        network_card.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.create_advanced_network_chart(network_card)

    def create_process_management(self, parent):
        """프로세스 관리 패널"""
        # 프로세스 통계
        stats_card = self.create_glass_card(parent, "📊 PROCESS STATISTICS")
        stats_card.pack(fill=tk.X, padx=5, pady=5)
        self.create_process_stats(stats_card)
        
        # 상위 프로세스
        top_proc_card = self.create_glass_card(parent, "🎯 TOP PROCESSES")
        top_proc_card.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.create_process_table(top_proc_card)

    def setup_premium_charts(self):
        """프리미엄 차트 설정"""
        # matplotlib 고급 설정
        plt.style.use('dark_background')
        
        # 고품질 렌더링
        plt.rcParams.update({
            'figure.facecolor': VISUAL_THEME['bg_card'],
            'axes.facecolor': VISUAL_THEME['bg_primary'],
            'axes.edgecolor': VISUAL_THEME['neon_cyan'],
            'axes.labelcolor': VISUAL_THEME['text_primary'],
            'text.color': VISUAL_THEME['text_primary'],
            'xtick.color': VISUAL_THEME['text_secondary'],
            'ytick.color': VISUAL_THEME['text_secondary'],
            'grid.color': VISUAL_THEME['bg_tertiary'],
            'grid.alpha': 0.5,
            'lines.linewidth': 2.5,
            'lines.antialiased': True,
            'font.family': 'Segoe UI',
            'font.size': 9,
            'figure.dpi': 120,
            'savefig.dpi': 120,
            'figure.autolayout': True
        })

    def create_advanced_cpu_chart(self, parent):
        """고급 CPU 차트"""
        chart_frame = tk.Frame(parent, bg=VISUAL_THEME['bg_card'])
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.cpu_fig = Figure(figsize=(12, 6), dpi=120, facecolor=VISUAL_THEME['bg_card'])
        
        # 메인 차트 (멀티코어)
        self.cpu_main_ax = self.cpu_fig.add_subplot(211)
        self.cpu_main_ax.set_facecolor(VISUAL_THEME['bg_primary'])
        
        # CPU 코어별 라인 (네온 효과)
        self.cpu_core_lines = {}
        cpu_count = psutil.cpu_count()
        colors = plt.cm.rainbow(np.linspace(0, 1, cpu_count))
        
        for i in range(cpu_count):
            line, = self.cpu_main_ax.plot([], [], color=colors[i], 
                                        linewidth=2, alpha=0.8, 
                                        label=f'Core {i}',
                                        antialiased=True)
            # 글로우 효과
            line.set_path_effects([path_effects.Stroke(linewidth=4, foreground=colors[i], alpha=0.3),
                                 path_effects.Normal()])
            self.cpu_core_lines[i] = line
        
        # 전체 CPU 라인 (강조)
        self.cpu_total_line, = self.cpu_main_ax.plot([], [], 
                                                   color=VISUAL_THEME['neon_cyan'], 
                                                   linewidth=4, alpha=1.0,
                                                   label='Total CPU')
        self.cpu_total_line.set_path_effects([
            path_effects.Stroke(linewidth=6, foreground=VISUAL_THEME['neon_cyan'], alpha=0.5),
            path_effects.Normal()
        ])
        
        self.cpu_main_ax.set_ylim(0, 100)
        self.cpu_main_ax.set_ylabel('Usage (%)', color=VISUAL_THEME['text_primary'])
        self.cpu_main_ax.grid(True, alpha=0.3, color=VISUAL_THEME['bg_tertiary'])
        self.cpu_main_ax.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=4, fontsize=8)
        
        # 하위 차트 (주파수 & 온도)
        self.cpu_freq_ax = self.cpu_fig.add_subplot(212)
        self.cpu_freq_ax.set_facecolor(VISUAL_THEME['bg_primary'])
        
        self.cpu_freq_line, = self.cpu_freq_ax.plot([], [], 
                                                  color=VISUAL_THEME['neon_yellow'], 
                                                  linewidth=3, label='Frequency (GHz)')
        self.cpu_temp_line, = self.cpu_freq_ax.plot([], [], 
                                                  color=VISUAL_THEME['neon_red'], 
                                                  linewidth=3, label='Temperature (°C)')
        
        self.cpu_freq_ax.set_xlabel('Time', color=VISUAL_THEME['text_primary'])
        self.cpu_freq_ax.set_ylabel('Freq/Temp', color=VISUAL_THEME['text_primary'])
        self.cpu_freq_ax.grid(True, alpha=0.3)
        self.cpu_freq_ax.legend()
        
        self.cpu_canvas = FigureCanvasTkAgg(self.cpu_fig, chart_frame)
        self.cpu_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_advanced_memory_chart(self, parent):
        """고급 메모리 차트"""
        chart_frame = tk.Frame(parent, bg=VISUAL_THEME['bg_card'])
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.mem_fig = Figure(figsize=(12, 6), dpi=120, facecolor=VISUAL_THEME['bg_card'])
        
        # 좌측: 도넛 차트 (3D 효과)
        self.mem_donut_ax = self.mem_fig.add_subplot(121)
        self.mem_donut_ax.set_aspect('equal')
        
        # 우측: 메모리 히스토리 & 상세
        self.mem_history_ax = self.mem_fig.add_subplot(122)
        self.mem_history_ax.set_facecolor(VISUAL_THEME['bg_primary'])
        
        # 메모리 히스토리 라인들
        self.mem_used_line, = self.mem_history_ax.plot([], [], 
                                                     color=VISUAL_THEME['neon_red'], 
                                                     linewidth=3, label='Used')
        self.mem_cached_line, = self.mem_history_ax.plot([], [], 
                                                        color=VISUAL_THEME['neon_blue'], 
                                                        linewidth=3, label='Cached')
        self.mem_available_line, = self.mem_history_ax.plot([], [], 
                                                           color=VISUAL_THEME['neon_green'], 
                                                           linewidth=3, label='Available')
        
        self.mem_history_ax.set_ylabel('Memory (GB)', color=VISUAL_THEME['text_primary'])
        self.mem_history_ax.set_xlabel('Time', color=VISUAL_THEME['text_primary'])
        self.mem_history_ax.grid(True, alpha=0.3)
        self.mem_history_ax.legend()
        
        self.mem_canvas = FigureCanvasTkAgg(self.mem_fig, chart_frame)
        self.mem_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_advanced_disk_chart(self, parent):
        """고급 디스크 차트"""
        chart_frame = tk.Frame(parent, bg=VISUAL_THEME['bg_card'])
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.disk_fig = Figure(figsize=(12, 5), dpi=120, facecolor=VISUAL_THEME['bg_card'])
        self.disk_ax = self.disk_fig.add_subplot(111)
        self.disk_ax.set_facecolor(VISUAL_THEME['bg_primary'])
        
        # 읽기/쓰기 영역 차트
        self.disk_read_line, = self.disk_ax.plot([], [], 
                                                color=VISUAL_THEME['neon_green'], 
                                                linewidth=3, label='Read (MB/s)')
        self.disk_write_line, = self.disk_ax.plot([], [], 
                                                 color=VISUAL_THEME['neon_red'], 
                                                 linewidth=3, label='Write (MB/s)')
        
        # 영역 채우기
        self.disk_read_fill = None
        self.disk_write_fill = None
        
        self.disk_ax.set_ylabel('Speed (MB/s)', color=VISUAL_THEME['text_primary'])
        self.disk_ax.set_xlabel('Time', color=VISUAL_THEME['text_primary'])
        self.disk_ax.grid(True, alpha=0.3)
        self.disk_ax.legend()
        
        self.disk_canvas = FigureCanvasTkAgg(self.disk_fig, chart_frame)
        self.disk_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_advanced_network_chart(self, parent):
        """고급 네트워크 차트"""
        chart_frame = tk.Frame(parent, bg=VISUAL_THEME['bg_card'])
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.net_fig = Figure(figsize=(12, 5), dpi=120, facecolor=VISUAL_THEME['bg_card'])
        self.net_ax = self.net_fig.add_subplot(111)
        self.net_ax.set_facecolor(VISUAL_THEME['bg_primary'])
        
        # 업로드/다운로드 (영역 차트)
        self.net_upload_line, = self.net_ax.plot([], [], 
                                                color=VISUAL_THEME['neon_orange'], 
                                                linewidth=3, label='Upload (MB/s)')
        self.net_download_line, = self.net_ax.plot([], [], 
                                                  color=VISUAL_THEME['neon_purple'], 
                                                  linewidth=3, label='Download (MB/s)')
        
        self.net_ax.set_ylabel('Speed (MB/s)', color=VISUAL_THEME['text_primary'])
        self.net_ax.set_xlabel('Time', color=VISUAL_THEME['text_primary'])
        self.net_ax.grid(True, alpha=0.3)
        self.net_ax.legend()
        
        self.net_canvas = FigureCanvasTkAgg(self.net_fig, chart_frame)
        self.net_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_process_stats(self, parent):
        """프로세스 통계"""
        stats_frame = tk.Frame(parent, bg=VISUAL_THEME['bg_card'])
        stats_frame.pack(fill=tk.X, padx=15, pady=15)
        
        self.process_stats_labels = {}
        stats = [
            ('total', 'Total Processes', VISUAL_THEME['neon_cyan']),
            ('cpu_high', 'High CPU (>50%)', VISUAL_THEME['neon_red']),
            ('memory_high', 'High Memory (>50%)', VISUAL_THEME['neon_yellow']),
            ('terminated', 'Auto-Terminated', VISUAL_THEME['neon_green'])
        ]
        
        for key, label, color in stats:
            row = tk.Frame(stats_frame, bg=VISUAL_THEME['bg_card'])
            row.pack(fill=tk.X, pady=3)
            
            tk.Label(
                row, text=f"{label}:",
                font=self.fonts['body'],
                fg=VISUAL_THEME['text_secondary'],
                bg=VISUAL_THEME['bg_card']
            ).pack(side=tk.LEFT)
            
            value_label = tk.Label(
                row, text="0",
                font=('Consolas', 11, 'bold'),
                fg=color,
                bg=VISUAL_THEME['bg_card']
            )
            value_label.pack(side=tk.RIGHT)
            self.process_stats_labels[key] = value_label

    def create_process_table(self, parent):
        """프로세스 테이블"""
        table_frame = tk.Frame(parent, bg=VISUAL_THEME['bg_card'])
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 스타일 설정
        style = ttk.Style()
        style.configure("Custom.Treeview", 
                       background=VISUAL_THEME['bg_primary'],
                       foreground=VISUAL_THEME['text_primary'],
                       fieldbackground=VISUAL_THEME['bg_primary'])
        style.configure("Custom.Treeview.Heading",
                       background=VISUAL_THEME['bg_tertiary'],
                       foreground=VISUAL_THEME['neon_cyan'])
        
        # 트리뷰
        columns = ('PID', 'Name', 'CPU%', 'Memory%', 'Memory', 'Status', 'Action')
        self.process_tree = ttk.Treeview(
            table_frame, columns=columns, show='headings',
            style="Custom.Treeview", height=20
        )
        
        # 컬럼 설정
        column_widths = {'PID': 60, 'Name': 150, 'CPU%': 60, 'Memory%': 60, 
                        'Memory': 80, 'Status': 80, 'Action': 80}
        
        for col in columns:
            self.process_tree.heading(col, text=col)
            self.process_tree.column(col, width=column_widths.get(col, 100))
        
        # 스크롤바
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.process_tree.yview)
        self.process_tree.configure(yscrollcommand=scrollbar.set)
        
        self.process_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def create_premium_status_bar(self, parent):
        """프리미엄 상태바"""
        status_bar = tk.Frame(parent, bg=VISUAL_THEME['bg_tertiary'], height=40)
        status_bar.pack(fill=tk.X)
        status_bar.pack_propagate(False)
        
        # 왼쪽: 시스템 메시지
        left_frame = tk.Frame(status_bar, bg=VISUAL_THEME['bg_tertiary'])
        left_frame.pack(side=tk.LEFT, padx=15, pady=5)
        
        self.status_message = tk.Label(
            left_frame,
            text="🚀 Ultimate monitoring active - AI optimization enabled",
            font=('Segoe UI', 10, 'bold'),
            fg=VISUAL_THEME['neon_green'],
            bg=VISUAL_THEME['bg_tertiary']
        )
        self.status_message.pack(side=tk.LEFT)
        
        # 가운데: 성능 지표
        center_frame = tk.Frame(status_bar, bg=VISUAL_THEME['bg_tertiary'])
        center_frame.pack(side=tk.LEFT, expand=True)
        
        self.perf_labels = {}
        perf_items = [
            ('fps', 'FPS', VISUAL_THEME['neon_cyan']),
            ('data_rate', 'Data Rate', VISUAL_THEME['neon_yellow']),
            ('cpu_avg', 'CPU Avg', VISUAL_THEME['neon_blue']),
            ('mem_usage', 'Memory', VISUAL_THEME['neon_purple'])
        ]
        
        for key, label, color in perf_items:
            perf_label = tk.Label(
                center_frame,
                text=f"{label}: 0",
                font=('Consolas', 9, 'bold'),
                fg=color,
                bg=VISUAL_THEME['bg_tertiary']
            )
            perf_label.pack(side=tk.LEFT, padx=15)
            self.perf_labels[key] = perf_label
        
        # 오른쪽: 시간 & 업타임
        right_frame = tk.Frame(status_bar, bg=VISUAL_THEME['bg_tertiary'])
        right_frame.pack(side=tk.RIGHT, padx=15, pady=5)
        
        self.uptime_label = tk.Label(
            right_frame,
            text="Uptime: 0d 0h 0m",
            font=('Segoe UI', 9),
            fg=VISUAL_THEME['text_secondary'],
            bg=VISUAL_THEME['bg_tertiary']
        )
        self.uptime_label.pack(side=tk.RIGHT)

    def collect_ultimate_data(self):
        """최고급 데이터 수집"""
        while self.running:
            start_time = time.time()
            
            try:
                # CPU 데이터
                cpu_total = psutil.cpu_percent(interval=0.1)
                cpu_cores = psutil.cpu_percent(interval=0.1, percpu=True)
                cpu_freq = psutil.cpu_freq()
                
                # 메모리 데이터
                memory = psutil.virtual_memory()
                
                # 온도 (가능한 경우)
                cpu_temp = 0
                try:
                    if hasattr(psutil, 'sensors_temperatures'):
                        temps = psutil.sensors_temperatures()
                        if temps:
                            for name, entries in temps.items():
                                if 'cpu' in name.lower() and entries:
                                    cpu_temp = entries[0].current
                                    break
                except:
                    pass
                
                # 디스크 I/O
                disk_io = psutil.disk_io_counters()
                current_time = time.time()
                
                disk_read_speed = disk_write_speed = 0
                if self.prev_disk_io and self.prev_time:
                    time_delta = current_time - self.prev_time
                    if time_delta > 0:
                        disk_read_speed = (disk_io.read_bytes - self.prev_disk_io.read_bytes) / time_delta / 1024 / 1024
                        disk_write_speed = (disk_io.write_bytes - self.prev_disk_io.write_bytes) / time_delta / 1024 / 1024
                
                # 네트워크 I/O
                net_io = psutil.net_io_counters()
                net_sent_speed = net_recv_speed = 0
                if self.prev_net_io and self.prev_time:
                    time_delta = current_time - self.prev_time
                    if time_delta > 0:
                        net_sent_speed = (net_io.bytes_sent - self.prev_net_io.bytes_sent) / time_delta / 1024 / 1024
                        net_recv_speed = (net_io.bytes_recv - self.prev_net_io.bytes_recv) / time_delta / 1024 / 1024
                
                # 프로세스 데이터
                processes = []
                high_cpu_processes = []
                high_memory_processes = []
                
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 
                                               'memory_info', 'status', 'create_time']):
                    try:
                        pinfo = proc.info
                        if pinfo['cpu_percent'] and pinfo['cpu_percent'] > 0:
                            processes.append(pinfo)
                            
                            if pinfo['cpu_percent'] > 50:
                                high_cpu_processes.append(pinfo)
                            if pinfo['memory_percent'] and pinfo['memory_percent'] > 50:
                                high_memory_processes.append(pinfo)
                    except:
                        pass
                
                # 정렬
                processes.sort(key=lambda x: x.get('cpu_percent', 0), reverse=True)
                
                # 데이터 저장
                with self.data_lock:
                    self.metrics['cpu_total'].append(cpu_total)
                    for i, core_usage in enumerate(cpu_cores):
                        self.metrics['cpu_cores'][i].append(core_usage)
                    
                    self.metrics['memory_percent'].append(memory.percent)
                    self.metrics['memory_used'].append(memory.used / 1024 / 1024 / 1024)
                    self.metrics['memory_cached'].append(memory.cached / 1024 / 1024 / 1024 if hasattr(memory, 'cached') else 0)
                    self.metrics['disk_read'].append(max(0, disk_read_speed))
                    self.metrics['disk_write'].append(max(0, disk_write_speed))
                    self.metrics['net_sent'].append(max(0, net_sent_speed))
                    self.metrics['net_recv'].append(max(0, net_recv_speed))
                    self.metrics['timestamps'].append(current_time)
                    
                    # 현재 상태 업데이트
                    self.current_stats.update({
                        'cpu_total': cpu_total,
                        'cpu_freq': cpu_freq.current if cpu_freq else 0,
                        'cpu_temp': cpu_temp,
                        'memory_percent': memory.percent,
                        'memory_total': memory.total,
                        'memory_used': memory.used,
                        'memory_available': memory.available,
                        'processes': processes[:50],
                        'high_cpu_processes': high_cpu_processes,
                        'high_memory_processes': high_memory_processes
                    })
                
                # 자동 최적화 실행
                if self.optimization_enabled:
                    self.auto_optimize(cpu_total, memory.percent, high_cpu_processes)
                
                # 이전 값 저장
                self.prev_disk_io = disk_io
                self.prev_net_io = net_io
                self.prev_time = current_time
                
            except Exception as e:
                print(f"데이터 수집 오류: {e}")
            
            # 타이밍
            elapsed = time.time() - start_time
            sleep_time = max(0, 0.25 - elapsed)  # 4Hz 데이터 수집
            time.sleep(sleep_time)

    def auto_optimize(self, cpu_usage, memory_usage, high_cpu_processes):
        """자동 최적화 실행"""
        try:
            # CPU 임계값 체크
            if cpu_usage > self.cpu_threshold_var.get():
                self.add_alert('warning', f"High CPU usage: {cpu_usage:.1f}%")
                
                if self.auto_term_var.get():
                    # 높은 CPU 사용 프로세스 종료
                    for proc_info in high_cpu_processes[:3]:  # 상위 3개만
                        try:
                            if proc_info['name'] in AUTO_TERMINATE_PROCESSES:
                                pid = proc_info['pid']
                                proc = psutil.Process(pid)
                                proc.terminate()
                                self.current_stats['terminated_processes'].append(proc_info['name'])
                                self.add_alert('info', f"Terminated high CPU process: {proc_info['name']}")
                        except:
                            pass
            
            # 메모리 임계값 체크
            if memory_usage > 85:
                self.add_alert('critical', f"Critical memory usage: {memory_usage:.1f}%")
                
                if self.auto_term_var.get():
                    # 메모리 집약적 프로세스 정리
                    for proc_info in self.current_stats['high_memory_processes'][:2]:
                        try:
                            if proc_info['name'] in AUTO_TERMINATE_PROCESSES:
                                pid = proc_info['pid']
                                proc = psutil.Process(pid)
                                proc.terminate()
                                self.current_stats['terminated_processes'].append(proc_info['name'])
                                self.add_alert('info', f"Terminated high memory process: {proc_info['name']}")
                        except:
                            pass
                            
        except Exception as e:
            print(f"자동 최적화 오류: {e}")

    def manual_optimize(self):
        """수동 최적화"""
        try:
            terminated_count = 0
            
            # 리소스 집약적 프로세스 찾기 및 종료
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    pinfo = proc.info
                    if (pinfo['name'] in AUTO_TERMINATE_PROCESSES and 
                        (pinfo['cpu_percent'] > 30 or pinfo['memory_percent'] > 30)):
                        
                        psutil.Process(pinfo['pid']).terminate()
                        terminated_count += 1
                        self.add_alert('success', f"Manually terminated: {pinfo['name']}")
                        
                        if terminated_count >= 5:  # 최대 5개까지
                            break
                except:
                    pass
            
            # 시스템 캐시 정리 (Windows)
            if os.name == 'nt':
                try:
                    os.system('sfc /scannow >nul 2>&1')  # 시스템 파일 체크
                except:
                    pass
            
            self.add_alert('success', f"Manual optimization completed. Terminated {terminated_count} processes.")
            
        except Exception as e:
            self.add_alert('error', f"Optimization failed: {str(e)}")

    def add_alert(self, level, message):
        """알림 추가"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # 알림 색상
        colors = {
            'info': VISUAL_THEME['neon_blue'],
            'success': VISUAL_THEME['neon_green'], 
            'warning': VISUAL_THEME['neon_yellow'],
            'critical': VISUAL_THEME['neon_red'],
            'error': VISUAL_THEME['neon_red']
        }
        
        # 알림 아이콘
        icons = {
            'info': 'ℹ️',
            'success': '✅',
            'warning': '⚠️', 
            'critical': '🚨',
            'error': '❌'
        }
        
        # UI 업데이트는 메인 스레드에서
        self.root.after(0, lambda: self.display_alert(level, message, timestamp, colors, icons))

    def display_alert(self, level, message, timestamp, colors, icons):
        """알림 표시"""
        alert_frame = tk.Frame(self.alerts_scrollable, bg=VISUAL_THEME['bg_primary'])
        alert_frame.pack(fill=tk.X, pady=2)
        
        # 아이콘
        tk.Label(
            alert_frame,
            text=icons.get(level, 'ℹ️'),
            font=('Segoe UI', 12),
            fg=colors.get(level, VISUAL_THEME['text_primary']),
            bg=VISUAL_THEME['bg_primary']
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        # 메시지
        tk.Label(
            alert_frame,
            text=message,
            font=self.fonts['body'],
            fg=VISUAL_THEME['text_primary'],
            bg=VISUAL_THEME['bg_primary'],
            anchor='w'
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 시간
        tk.Label(
            alert_frame,
            text=timestamp,
            font=('Consolas', 8),
            fg=VISUAL_THEME['text_secondary'],
            bg=VISUAL_THEME['bg_primary']
        ).pack(side=tk.RIGHT)
        
        # 스크롤 업데이트
        self.alerts_canvas.update_idletasks()
        self.alerts_canvas.yview_moveto(1.0)

    def toggle_optimization(self):
        """최적화 토글"""
        self.optimization_enabled = self.auto_opt_var.get()
        status = "enabled" if self.optimization_enabled else "disabled"
        self.add_alert('info', f"Auto-optimization {status}")

    def toggle_auto_terminate(self):
        """자동 종료 토글"""
        self.auto_terminate_enabled = self.auto_term_var.get()
        status = "enabled" if self.auto_terminate_enabled else "disabled"
        self.add_alert('info', f"Auto-terminate {status}")

    def update_ultimate_ui(self):
        """최고급 UI 업데이트"""
        if not self.running:
            return
        
        # 디지털 시계
        self.digital_clock.config(text=datetime.now().strftime("%H:%M:%S"))
        
        # 상태 표시기 업데이트
        with self.data_lock:
            if self.metrics['cpu_total']:
                cpu_val = self.metrics['cpu_total'][-1]
                self.status_indicators['cpu'].config(
                    fg=VISUAL_THEME['neon_red'] if cpu_val > 80 else VISUAL_THEME['neon_green']
                )
            
            if self.metrics['memory_percent']:
                mem_val = self.metrics['memory_percent'][-1]
                self.status_indicators['memory'].config(
                    fg=VISUAL_THEME['neon_red'] if mem_val > 80 else VISUAL_THEME['neon_green']
                )
        
        # 원형 게이지 업데이트
        self.update_circular_gauges()
        
        # 프로세스 통계 업데이트
        self.update_process_statistics()
        
        # 성능 지표 업데이트
        self.update_performance_indicators()
        
        # 다음 프레임
        self.root.after(33, self.update_ultimate_ui)  # 30 FPS

    def update_circular_gauges(self):
        """원형 게이지 업데이트"""
        with self.data_lock:
            if self.metrics['cpu_total']:
                cpu_val = self.metrics['cpu_total'][-1]
                self.update_gauge(self.cpu_gauge_ax, cpu_val, "CPU", VISUAL_THEME['neon_blue'])
                self.cpu_gauge_fig.canvas.draw_idle()
            
            if self.metrics['memory_percent']:
                mem_val = self.metrics['memory_percent'][-1]
                self.update_gauge(self.mem_gauge_ax, mem_val, "MEMORY", VISUAL_THEME['neon_yellow'])
                self.mem_gauge_fig.canvas.draw_idle()

    def update_gauge(self, ax, value, title, color):
        """게이지 업데이트"""
        ax.clear()
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 배경 원
        background_circle = Circle((0, 0), 1, fill=False, linewidth=8, 
                                 edgecolor=VISUAL_THEME['bg_tertiary'])
        ax.add_patch(background_circle)
        
        # 진행률 호
        theta = (value / 100) * 2 * np.pi - np.pi/2
        arc = Wedge((0, 0), 1, -90, np.degrees(theta), width=0.1, 
                   facecolor=color, alpha=0.8)
        ax.add_patch(arc)
        
        # 글로우 효과
        glow_arc = Wedge((0, 0), 1.05, -90, np.degrees(theta), width=0.15, 
                        facecolor=color, alpha=0.3)
        ax.add_patch(glow_arc)
        
        # 제목
        ax.text(0, 1.4, title, ha='center', va='center', 
                fontsize=12, fontweight='bold', color=VISUAL_THEME['text_primary'])
        
        # 값 (네온 효과)
        text = ax.text(0, 0, f"{value:.1f}%", ha='center', va='center', 
                      fontsize=16, fontweight='bold', color=color)
        text.set_path_effects([path_effects.withStroke(linewidth=3, foreground=color, alpha=0.5)])

    def update_process_statistics(self):
        """프로세스 통계 업데이트"""
        with self.data_lock:
            self.process_stats_labels['total'].config(text=str(len(self.current_stats['processes'])))
            self.process_stats_labels['cpu_high'].config(text=str(len(self.current_stats['high_cpu_processes'])))
            self.process_stats_labels['memory_high'].config(text=str(len(self.current_stats['high_memory_processes'])))
            self.process_stats_labels['terminated'].config(text=str(len(self.current_stats['terminated_processes'])))

    def update_performance_indicators(self):
        """성능 지표 업데이트"""
        with self.data_lock:
            if self.metrics['cpu_total']:
                cpu_avg = sum(list(self.metrics['cpu_total'])[-60:]) / min(60, len(self.metrics['cpu_total']))
                self.perf_labels['cpu_avg'].config(text=f"CPU Avg: {cpu_avg:.1f}%")
            
            if self.metrics['memory_percent']:
                mem_current = self.metrics['memory_percent'][-1]
                self.perf_labels['mem_usage'].config(text=f"Memory: {mem_current:.1f}%")

    def update_charts(self):
        """차트 업데이트"""
        if not self.running:
            return
        
        with self.data_lock:
            self.update_cpu_charts()
            self.update_memory_charts()
            self.update_disk_charts()
            self.update_network_charts()
            self.update_process_table()
        
        self.root.after(100, self.update_charts)  # 10 FPS

    def update_cpu_charts(self):
        """CPU 차트 업데이트"""
        if len(self.metrics['cpu_total']) < 2:
            return
        
        x_data = list(range(len(self.metrics['cpu_total'])))
        
        # 코어별 차트
        for i, line in self.cpu_core_lines.items():
            if i in self.metrics['cpu_cores'] and self.metrics['cpu_cores'][i]:
                y_data = list(self.metrics['cpu_cores'][i])
                line.set_data(x_data[-len(y_data):], y_data)
        
        # 전체 CPU
        self.cpu_total_line.set_data(x_data, list(self.metrics['cpu_total']))
        
        # 주파수 & 온도
        if len(x_data) > 1:
            freq_data = [self.current_stats['cpu_freq'] / 1000] * len(x_data)  # GHz 변환
            temp_data = [self.current_stats['cpu_temp']] * len(x_data)
            
            self.cpu_freq_line.set_data(x_data, freq_data)
            self.cpu_temp_line.set_data(x_data, temp_data)
        
        # 축 범위 조정
        self.cpu_main_ax.set_xlim(max(0, len(x_data) - 300), len(x_data))
        self.cpu_freq_ax.set_xlim(max(0, len(x_data) - 300), len(x_data))
        
        self.cpu_canvas.draw_idle()

    def update_memory_charts(self):
        """메모리 차트 업데이트"""
        if not self.metrics['memory_percent']:
            return
        
        # 도넛 차트
        mem_percent = self.metrics['memory_percent'][-1]
        mem_used_gb = self.current_stats['memory_used'] / 1024 / 1024 / 1024
        mem_total_gb = self.current_stats['memory_total'] / 1024 / 1024 / 1024
        mem_free_gb = mem_total_gb - mem_used_gb
        
        self.mem_donut_ax.clear()
        
        # 3D 도넛 효과
        sizes = [mem_used_gb, mem_free_gb]
        colors = [VISUAL_THEME['neon_red'], VISUAL_THEME['neon_green']]
        explode = (0.05, 0)  # 살짝 분리
        
        wedges, texts = self.mem_donut_ax.pie(sizes, colors=colors, explode=explode,
                                             startangle=90, wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2))
        
        # 중앙 텍스트 (네온 효과)
        center_text = self.mem_donut_ax.text(0, 0, f"{mem_percent:.1f}%\nUSED", 
                                           ha='center', va='center', fontsize=14, 
                                           fontweight='bold', color=VISUAL_THEME['neon_cyan'])
        center_text.set_path_effects([path_effects.withStroke(linewidth=3, foreground=VISUAL_THEME['neon_cyan'], alpha=0.5)])
        
        # 히스토리 차트
        if len(self.metrics['memory_used']) > 1:
            x_data = list(range(len(self.metrics['memory_used'])))
            
            self.mem_used_line.set_data(x_data, list(self.metrics['memory_used']))
            self.mem_cached_line.set_data(x_data, list(self.metrics['memory_cached']))
            
            # Available 계산
            available_data = [self.current_stats['memory_available'] / 1024 / 1024 / 1024] * len(x_data)
            self.mem_available_line.set_data(x_data, available_data)
            
            self.mem_history_ax.set_xlim(max(0, len(x_data) - 300), len(x_data))
            self.mem_history_ax.set_ylim(0, mem_total_gb * 1.1)
        
        self.mem_canvas.draw_idle()

    def update_disk_charts(self):
        """디스크 차트 업데이트"""
        if len(self.metrics['disk_read']) < 2:
            return
        
        x_data = list(range(len(self.metrics['disk_read'])))
        read_data = list(self.metrics['disk_read'])
        write_data = list(self.metrics['disk_write'])
        
        self.disk_read_line.set_data(x_data, read_data)
        self.disk_write_line.set_data(x_data, write_data)
        
        # 영역 채우기 효과
        if self.disk_read_fill:
            self.disk_read_fill.remove()
        if self.disk_write_fill:
            self.disk_write_fill.remove()
        
        self.disk_read_fill = self.disk_ax.fill_between(x_data, read_data, alpha=0.3, color=VISUAL_THEME['neon_green'])
        self.disk_write_fill = self.disk_ax.fill_between(x_data, write_data, alpha=0.3, color=VISUAL_THEME['neon_red'])
        
        # 축 범위
        max_val = max(max(read_data), max(write_data), 1)
        self.disk_ax.set_xlim(max(0, len(x_data) - 300), len(x_data))
        self.disk_ax.set_ylim(0, max_val * 1.1)
        
        self.disk_canvas.draw_idle()

    def update_network_charts(self):
        """네트워크 차트 업데이트"""
        if len(self.metrics['net_sent']) < 2:
            return
        
        x_data = list(range(len(self.metrics['net_sent'])))
        sent_data = list(self.metrics['net_sent'])
        recv_data = list(self.metrics['net_recv'])
        
        self.net_upload_line.set_data(x_data, sent_data)
        self.net_download_line.set_data(x_data, recv_data)
        
        # 축 범위
        max_val = max(max(sent_data), max(recv_data), 0.1)
        self.net_ax.set_xlim(max(0, len(x_data) - 300), len(x_data))
        self.net_ax.set_ylim(0, max_val * 1.1)
        
        self.net_canvas.draw_idle()

    def update_process_table(self):
        """프로세스 테이블 업데이트"""
        # 기존 항목 삭제
        for item in self.process_tree.get_children():
            self.process_tree.delete(item)
        
        # 새 데이터 추가 (상위 20개)
        for i, proc in enumerate(self.current_stats['processes'][:20]):
            try:
                pid = proc.get('pid', 0)
                name = proc.get('name', 'Unknown')[:20]
                cpu_percent = proc.get('cpu_percent', 0) or 0
                memory_percent = proc.get('memory_percent', 0) or 0
                memory_info = proc.get('memory_info')
                memory_mb = memory_info.rss / 1024 / 1024 if memory_info else 0
                status = proc.get('status', 'unknown')
                
                # 액션 결정
                action = ""
                if name in AUTO_TERMINATE_PROCESSES:
                    if cpu_percent > 50 or memory_percent > 50:
                        action = "AUTO-KILL"
                    else:
                        action = "MONITOR"
                else:
                    action = "SAFE"
                
                # 색상 태그
                tag = ''
                if cpu_percent > 80:
                    tag = 'critical'
                elif cpu_percent > 50:
                    tag = 'warning'
                elif action == "AUTO-KILL":
                    tag = 'danger'
                
                self.process_tree.insert('', 'end', values=(
                    pid, name, f"{cpu_percent:.1f}%",
                    f"{memory_percent:.1f}%", f"{memory_mb:.0f} MB",
                    status, action
                ), tags=(tag,))
                
            except Exception as e:
                pass
        
        # 태그 색상 설정
        self.process_tree.tag_configure('critical', foreground=VISUAL_THEME['neon_red'])
        self.process_tree.tag_configure('warning', foreground=VISUAL_THEME['neon_yellow'])
        self.process_tree.tag_configure('danger', foreground=VISUAL_THEME['neon_orange'])

    def start_ultimate_monitoring(self):
        """최고급 모니터링 시작"""
        # 데이터 수집 스레드
        data_thread = threading.Thread(target=self.collect_ultimate_data, daemon=True)
        data_thread.start()
        
        # UI 업데이트
        self.root.after(100, self.update_ultimate_ui)
        self.root.after(200, self.update_charts)
        
        # 초기 알림
        self.add_alert('success', "Ultimate monitoring system activated")
        self.add_alert('info', f"Auto-optimization: {'enabled' if self.optimization_enabled else 'disabled'}")

    def on_closing(self):
        """종료 처리"""
        self.running = False
        self.root.quit()

    def run(self):
        """실행"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.start_ultimate_monitoring()
        self.root.mainloop()

def main():
    """메인 함수"""
    try:
        import psutil
        import matplotlib
        import numpy
    except ImportError as e:
        messagebox.showerror("Error", f"필수 모듈이 없습니다: {e}\n\npip install psutil matplotlib numpy")
        return 1
    
    app = UltimateSystemMonitor()
    app.run()
    return 0

if __name__ == "__main__":
    exit(main())