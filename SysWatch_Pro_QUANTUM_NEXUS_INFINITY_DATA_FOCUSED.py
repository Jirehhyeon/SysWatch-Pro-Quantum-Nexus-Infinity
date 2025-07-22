#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌌 SysWatch Pro QUANTUM NEXUS INFINITY DATA FOCUSED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 DATA-FOCUSED QUANTUM SYSTEM MONITOR
🔥 데이터 시각화에 완전 집중 | 모든 값이 명확히 보이는 버전
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Copyright (C) 2025 QUANTUM DATA FOCUS Corporation
"""

import os
import sys
import time
import math
import random
import threading
import warnings
import platform
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Tuple, Any, Optional
import colorsys
import subprocess

# Auto-install required packages
def ensure_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    try:
        __import__(import_name)
    except ImportError:
        print(f"📦 Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

# Ensure core packages
ensure_package("psutil")
ensure_package("numpy")
ensure_package("pygame-ce", "pygame")

import psutil
import numpy as np
import pygame
from pygame.locals import *

# Optional packages
try:
    import wmi
    WMI_AVAILABLE = True
except ImportError:
    WMI_AVAILABLE = False

try:
    import win32api
    import win32process
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

# Initialize Pygame
pygame.init()

# ============================================================================
# DATA-FOCUSED COLOR SYSTEM
# ============================================================================
class DataColors:
    # High contrast colors for data visibility
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    BRIGHT_GREEN = (0, 255, 0)
    BRIGHT_RED = (255, 0, 0)
    BRIGHT_BLUE = (0, 150, 255)
    BRIGHT_YELLOW = (255, 255, 0)
    BRIGHT_CYAN = (0, 255, 255)
    BRIGHT_MAGENTA = (255, 0, 255)
    BRIGHT_ORANGE = (255, 165, 0)
    
    # Performance status colors
    EXCELLENT = (0, 255, 0)      # Green
    GOOD = (150, 255, 0)         # Light Green
    WARNING = (255, 200, 0)      # Yellow
    DANGER = (255, 100, 0)       # Orange
    CRITICAL = (255, 0, 0)       # Red
    
    # UI colors
    PANEL_BG = (20, 20, 30)
    PANEL_BORDER = (100, 150, 255)
    TEXT_PRIMARY = (255, 255, 255)
    TEXT_SECONDARY = (200, 200, 200)
    TEXT_HIGHLIGHT = (255, 255, 100)
    
    @staticmethod
    def get_performance_color(value: float, max_value: float = 100) -> Tuple[int, int, int]:
        """Get color based on performance percentage"""
        ratio = min(value / max_value, 1.0)
        
        if ratio < 0.2:
            return DataColors.EXCELLENT
        elif ratio < 0.4:
            return DataColors.GOOD
        elif ratio < 0.6:
            return DataColors.WARNING
        elif ratio < 0.8:
            return DataColors.DANGER
        else:
            return DataColors.CRITICAL
    
    @staticmethod
    def get_gradient_color(start_color: Tuple[int, int, int], end_color: Tuple[int, int, int], ratio: float) -> Tuple[int, int, int]:
        """Get color between two colors"""
        ratio = max(0, min(1, ratio))
        r = int(start_color[0] * (1 - ratio) + end_color[0] * ratio)
        g = int(start_color[1] * (1 - ratio) + end_color[1] * ratio)
        b = int(start_color[2] * (1 - ratio) + end_color[2] * ratio)
        return (r, g, b)

# ============================================================================
# REAL-TIME DATA MONITOR
# ============================================================================
class RealTimeDataMonitor:
    def __init__(self):
        self.cpu_history = deque(maxlen=200)
        self.memory_history = deque(maxlen=200)
        self.network_history = deque(maxlen=200)
        self.disk_history = deque(maxlen=200)
        self.process_list = []
        
        # Current data
        self.current_data = {
            'cpu': {'percent': 0, 'per_core': [], 'frequency': 0, 'temperature': 0},
            'memory': {'percent': 0, 'used': 0, 'total': 0, 'available': 0},
            'network': {'upload': 0, 'download': 0, 'connections': 0},
            'disk': {'percent': 0, 'read_speed': 0, 'write_speed': 0},
            'processes': [],
            'system': {}
        }
        
        # Network/Disk tracking
        self.last_network = None
        self.last_disk = None
        self.last_time = time.time()
        
        # Monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        # System info
        self._get_system_info()
    
    def _get_system_info(self):
        """Get static system information"""
        self.current_data['system'] = {
            'os': f"{platform.system()} {platform.release()}",
            'machine': platform.machine(),
            'processor': platform.processor()[:50] + "..." if len(platform.processor()) > 50 else platform.processor(),
            'python': platform.python_version(),
            'boot_time': datetime.fromtimestamp(psutil.boot_time()),
            'cpu_count': psutil.cpu_count(),
            'cpu_logical': psutil.cpu_count(logical=True)
        }
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                current_time = time.time()
                dt = current_time - self.last_time
                self.last_time = current_time
                
                # CPU monitoring
                cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
                cpu_freq = psutil.cpu_freq()
                
                cpu_data = {
                    'timestamp': current_time,
                    'percent': sum(cpu_percent) / len(cpu_percent),
                    'per_core': cpu_percent,
                    'frequency': cpu_freq.current if cpu_freq else 0
                }
                
                # Get CPU temperature (Windows)
                if WMI_AVAILABLE:
                    try:
                        w = wmi.WMI(namespace="root\\wmi")
                        temp_info = w.MSAcpi_ThermalZoneTemperature()
                        if temp_info:
                            temp_kelvin = temp_info[0].CurrentTemperature / 10.0
                            cpu_data['temperature'] = temp_kelvin - 273.15
                    except:
                        cpu_data['temperature'] = 0
                else:
                    cpu_data['temperature'] = 0
                
                self.cpu_history.append(cpu_data)
                self.current_data['cpu'] = cpu_data
                
                # Memory monitoring
                memory = psutil.virtual_memory()
                memory_data = {
                    'timestamp': current_time,
                    'percent': memory.percent,
                    'used': memory.used,
                    'total': memory.total,
                    'available': memory.available
                }
                
                self.memory_history.append(memory_data)
                self.current_data['memory'] = memory_data
                
                # Network monitoring
                current_network = psutil.net_io_counters()
                if self.last_network and dt > 0:
                    upload_rate = (current_network.bytes_sent - self.last_network.bytes_sent) / dt
                    download_rate = (current_network.bytes_recv - self.last_network.bytes_recv) / dt
                    
                    network_data = {
                        'timestamp': current_time,
                        'upload': upload_rate,
                        'download': download_rate,
                        'connections': len(psutil.net_connections())
                    }
                    
                    self.network_history.append(network_data)
                    self.current_data['network'] = network_data
                
                self.last_network = current_network
                
                # Disk monitoring
                current_disk = psutil.disk_io_counters()
                if self.last_disk and dt > 0:
                    read_speed = (current_disk.read_bytes - self.last_disk.read_bytes) / dt
                    write_speed = (current_disk.write_bytes - self.last_disk.write_bytes) / dt
                    
                    disk_data = {
                        'timestamp': current_time,
                        'percent': psutil.disk_usage('/').percent,
                        'read_speed': read_speed,
                        'write_speed': write_speed
                    }
                    
                    self.disk_history.append(disk_data)
                    self.current_data['disk'] = disk_data
                
                self.last_disk = current_disk
                
                # Process monitoring
                processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
                    try:
                        info = proc.info
                        if info['cpu_percent'] is not None:
                            processes.append(info)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                # Sort by CPU usage
                processes.sort(key=lambda x: x.get('cpu_percent', 0), reverse=True)
                self.current_data['processes'] = processes[:30]  # Top 30
                
                time.sleep(0.2)  # Update every 200ms
                
            except Exception as e:
                time.sleep(1)
    
    def get_data(self) -> Dict[str, Any]:
        """Get current data"""
        return self.current_data.copy()
    
    def cleanup(self):
        """Stop monitoring"""
        self.monitoring = False

# ============================================================================
# DATA VISUALIZATION COMPONENTS
# ============================================================================
class DataChart:
    def __init__(self, x: int, y: int, width: int, height: int, title: str, max_value: float = 100):
        self.rect = pygame.Rect(x, y, width, height)
        self.title = title
        self.max_value = max_value
        self.data_points = deque(maxlen=width)
        self.chart_area = pygame.Rect(x + 40, y + 30, width - 60, height - 50)
    
    def add_data(self, value: float):
        """Add data point"""
        self.data_points.append(value)
    
    def render(self, screen, font):
        """Render chart"""
        # Background
        pygame.draw.rect(screen, DataColors.PANEL_BG, self.rect)
        pygame.draw.rect(screen, DataColors.PANEL_BORDER, self.rect, 2)
        
        # Title
        title_surface = font.render(self.title, True, DataColors.TEXT_PRIMARY)
        screen.blit(title_surface, (self.rect.x + 5, self.rect.y + 5))
        
        # Current value
        if self.data_points:
            current_value = self.data_points[-1]
            value_text = f"{current_value:.1f}%"
            value_color = DataColors.get_performance_color(current_value, self.max_value)
            value_surface = font.render(value_text, True, value_color)
            screen.blit(value_surface, (self.rect.x + self.rect.width - 80, self.rect.y + 5))
        
        # Chart area background
        pygame.draw.rect(screen, (10, 10, 20), self.chart_area)
        pygame.draw.rect(screen, DataColors.TEXT_SECONDARY, self.chart_area, 1)
        
        # Grid lines
        for i in range(5):
            y_pos = self.chart_area.y + (i * self.chart_area.height // 4)
            pygame.draw.line(screen, (40, 40, 60), 
                           (self.chart_area.x, y_pos), 
                           (self.chart_area.x + self.chart_area.width, y_pos))
            
            # Value labels
            value = self.max_value * (1 - i / 4)
            label = font.render(f"{value:.0f}", True, DataColors.TEXT_SECONDARY)
            screen.blit(label, (self.rect.x + 5, y_pos - 8))
        
        # Data line
        if len(self.data_points) > 1:
            points = []
            for i, value in enumerate(self.data_points):
                x = self.chart_area.x + (i * self.chart_area.width // max(1, len(self.data_points) - 1))
                y = self.chart_area.y + self.chart_area.height - (value / self.max_value * self.chart_area.height)
                points.append((x, max(self.chart_area.y, min(self.chart_area.y + self.chart_area.height, y))))
            
            if len(points) > 1:
                # Gradient fill
                for i in range(len(points) - 1):
                    x1, y1 = points[i]
                    x2, y2 = points[i + 1]
                    
                    # Fill area under curve
                    fill_points = [
                        (x1, y1),
                        (x2, y2),
                        (x2, self.chart_area.y + self.chart_area.height),
                        (x1, self.chart_area.y + self.chart_area.height)
                    ]
                    
                    avg_value = (self.data_points[i] + self.data_points[min(i + 1, len(self.data_points) - 1)]) / 2
                    fill_color = (*DataColors.get_performance_color(avg_value, self.max_value), 50)
                    
                    # Create surface for alpha blending
                    fill_surface = pygame.Surface((abs(x2 - x1) + 1, self.chart_area.height), pygame.SRCALPHA)
                    pygame.draw.polygon(fill_surface, fill_color, 
                                      [(0 if x1 < x2 else abs(x2 - x1), y1 - self.chart_area.y),
                                       (abs(x2 - x1) if x1 < x2 else 0, y2 - self.chart_area.y),
                                       (abs(x2 - x1) if x1 < x2 else 0, self.chart_area.height),
                                       (0 if x1 < x2 else abs(x2 - x1), self.chart_area.height)])
                    
                    screen.blit(fill_surface, (min(x1, x2), self.chart_area.y))
                
                # Main line
                pygame.draw.aalines(screen, DataColors.BRIGHT_CYAN, False, points, 2)
                
                # Data points
                for i, (x, y) in enumerate(points):
                    if i % 5 == 0:  # Show every 5th point
                        value = self.data_points[i]
                        color = DataColors.get_performance_color(value, self.max_value)
                        pygame.draw.circle(screen, color, (x, y), 3)

class DataBar:
    def __init__(self, x: int, y: int, width: int, height: int, title: str, max_value: float = 100):
        self.rect = pygame.Rect(x, y, width, height)
        self.title = title
        self.max_value = max_value
        self.current_value = 0
        self.target_value = 0
        self.animation_speed = 5.0
    
    def set_value(self, value: float):
        """Set target value with animation"""
        self.target_value = min(value, self.max_value)
    
    def update(self, dt: float):
        """Update animation"""
        diff = self.target_value - self.current_value
        self.current_value += diff * self.animation_speed * dt
    
    def render(self, screen, font):
        """Render bar"""
        # Background
        pygame.draw.rect(screen, DataColors.PANEL_BG, self.rect)
        pygame.draw.rect(screen, DataColors.PANEL_BORDER, self.rect, 2)
        
        # Title
        title_surface = font.render(self.title, True, DataColors.TEXT_PRIMARY)
        screen.blit(title_surface, (self.rect.x + 5, self.rect.y + 5))
        
        # Value text
        value_text = f"{self.current_value:.1f}%"
        value_color = DataColors.get_performance_color(self.current_value, self.max_value)
        value_surface = font.render(value_text, True, value_color)
        screen.blit(value_surface, (self.rect.x + self.rect.width - 80, self.rect.y + 5))
        
        # Bar area
        bar_area = pygame.Rect(self.rect.x + 10, self.rect.y + 25, self.rect.width - 20, self.rect.height - 35)
        pygame.draw.rect(screen, (20, 20, 30), bar_area)
        pygame.draw.rect(screen, DataColors.TEXT_SECONDARY, bar_area, 1)
        
        # Fill bar
        if self.current_value > 0:
            fill_width = (self.current_value / self.max_value) * (bar_area.width - 2)
            fill_rect = pygame.Rect(bar_area.x + 1, bar_area.y + 1, fill_width, bar_area.height - 2)
            
            # Gradient fill
            fill_color = DataColors.get_performance_color(self.current_value, self.max_value)
            
            # Create gradient
            for i in range(int(fill_width)):
                alpha = 0.3 + 0.7 * (i / max(1, fill_width))
                color = tuple(int(c * alpha) for c in fill_color)
                
                line_rect = pygame.Rect(bar_area.x + 1 + i, bar_area.y + 1, 1, bar_area.height - 2)
                pygame.draw.rect(screen, color, line_rect)
            
            # Border
            pygame.draw.rect(screen, fill_color, fill_rect, 2)
        
        # Percentage markers
        for i in range(5):
            marker_x = bar_area.x + (i * bar_area.width // 4)
            pygame.draw.line(screen, DataColors.TEXT_SECONDARY, 
                           (marker_x, bar_area.y + bar_area.height), 
                           (marker_x, bar_area.y + bar_area.height + 5))
            
            marker_value = (i / 4) * self.max_value
            marker_text = font.render(f"{marker_value:.0f}", True, DataColors.TEXT_SECONDARY)
            screen.blit(marker_text, (marker_x - 10, bar_area.y + bar_area.height + 8))

class DataTable:
    def __init__(self, x: int, y: int, width: int, height: int, title: str):
        self.rect = pygame.Rect(x, y, width, height)
        self.title = title
        self.headers = []
        self.rows = []
        self.row_height = 20
        self.header_height = 25
    
    def set_data(self, headers: List[str], rows: List[List[str]]):
        """Set table data"""
        self.headers = headers
        self.rows = rows
    
    def render(self, screen, font):
        """Render table"""
        # Background
        pygame.draw.rect(screen, DataColors.PANEL_BG, self.rect)
        pygame.draw.rect(screen, DataColors.PANEL_BORDER, self.rect, 2)
        
        # Title
        title_surface = font.render(self.title, True, DataColors.TEXT_PRIMARY)
        screen.blit(title_surface, (self.rect.x + 5, self.rect.y + 5))
        
        # Table area
        table_area = pygame.Rect(self.rect.x + 5, self.rect.y + 30, self.rect.width - 10, self.rect.height - 35)
        
        # Headers
        if self.headers:
            header_y = table_area.y
            col_width = table_area.width // max(1, len(self.headers))
            
            # Header background
            header_rect = pygame.Rect(table_area.x, header_y, table_area.width, self.header_height)
            pygame.draw.rect(screen, (30, 30, 50), header_rect)
            pygame.draw.rect(screen, DataColors.PANEL_BORDER, header_rect, 1)
            
            for i, header in enumerate(self.headers):
                header_x = table_area.x + i * col_width
                header_surface = font.render(header, True, DataColors.TEXT_HIGHLIGHT)
                screen.blit(header_surface, (header_x + 5, header_y + 3))
                
                # Column separator
                if i < len(self.headers) - 1:
                    sep_x = header_x + col_width
                    pygame.draw.line(screen, DataColors.PANEL_BORDER, 
                                   (sep_x, header_y), (sep_x, header_y + self.header_height))
        
        # Rows
        current_y = table_area.y + self.header_height
        visible_rows = (table_area.height - self.header_height) // self.row_height
        
        for row_idx, row in enumerate(self.rows[:visible_rows]):
            row_y = current_y + row_idx * self.row_height
            
            # Row background (alternating)
            if row_idx % 2 == 0:
                row_rect = pygame.Rect(table_area.x, row_y, table_area.width, self.row_height)
                pygame.draw.rect(screen, (15, 15, 25), row_rect)
            
            # Row border
            pygame.draw.line(screen, (40, 40, 60), 
                           (table_area.x, row_y + self.row_height), 
                           (table_area.x + table_area.width, row_y + self.row_height))
            
            # Cells
            col_width = table_area.width // max(1, len(self.headers))
            for col_idx, cell in enumerate(row[:len(self.headers)]):
                cell_x = table_area.x + col_idx * col_width
                
                # Color coding for certain columns
                if col_idx == 2 and '%' in str(cell):  # CPU column
                    try:
                        value = float(str(cell).replace('%', ''))
                        color = DataColors.get_performance_color(value)
                    except:
                        color = DataColors.TEXT_PRIMARY
                elif col_idx == 3 and '%' in str(cell):  # Memory column
                    try:
                        value = float(str(cell).replace('%', ''))
                        color = DataColors.get_performance_color(value)
                    except:
                        color = DataColors.TEXT_PRIMARY
                else:
                    color = DataColors.TEXT_PRIMARY
                
                cell_surface = font.render(str(cell)[:15], True, color)
                screen.blit(cell_surface, (cell_x + 5, row_y + 2))

# ============================================================================
# MAIN DATA-FOCUSED APPLICATION
# ============================================================================
class DataFocusedMonitor:
    def __init__(self):
        # Display setup
        self.screen_info = pygame.display.Info()
        self.width = self.screen_info.current_w
        self.height = self.screen_info.current_h
        
        # Create display
        self.screen = pygame.display.set_mode(
            (self.width, self.height),
            pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE
        )
        
        pygame.display.set_caption("🌌 QUANTUM NEXUS DATA FOCUSED - 실시간 시스템 데이터 모니터")
        
        # Performance tracking
        self.clock = pygame.time.Clock()
        self.target_fps = 60
        self.current_fps = 0
        self.running = True
        self.last_time = time.time()
        
        # Initialize subsystems
        self._initialize_fonts()
        self._initialize_data_monitor()
        self._initialize_charts()
        
        # Set high priority if possible
        self._set_high_priority()
    
    def _initialize_fonts(self):
        """Initialize fonts"""
        try:
            self.font_small = pygame.font.Font(None, 16)
            self.font_medium = pygame.font.Font(None, 20)
            self.font_large = pygame.font.Font(None, 24)
            self.font_huge = pygame.font.Font(None, 36)
        except:
            self.font_small = pygame.font.Font(None, 16)
            self.font_medium = pygame.font.Font(None, 20)
            self.font_large = pygame.font.Font(None, 24)
            self.font_huge = pygame.font.Font(None, 36)
    
    def _initialize_data_monitor(self):
        """Initialize data monitoring"""
        self.data_monitor = RealTimeDataMonitor()
    
    def _initialize_charts(self):
        """Initialize data visualization components"""
        # Calculate layout
        chart_width = 400
        chart_height = 200
        bar_height = 80
        table_width = 500
        table_height = 400
        
        margin = 20
        
        # Charts (top row)
        self.cpu_chart = DataChart(margin, margin, chart_width, chart_height, "🧠 CPU Usage (%)")
        self.memory_chart = DataChart(margin + chart_width + margin, margin, chart_width, chart_height, "💾 Memory Usage (%)")
        self.network_chart = DataChart(margin + (chart_width + margin) * 2, margin, chart_width, chart_height, "🌐 Network (MB/s)", max_value=10)
        
        # Bars (second row)
        bar_y = margin + chart_height + margin
        self.cpu_bar = DataBar(margin, bar_y, chart_width, bar_height, "🧠 CPU")
        self.memory_bar = DataBar(margin + chart_width + margin, bar_y, chart_width, bar_height, "💾 Memory")
        self.disk_bar = DataBar(margin + (chart_width + margin) * 2, bar_y, chart_width, bar_height, "💽 Disk")
        
        # System info panel
        info_y = bar_y + bar_height + margin
        self.system_info_rect = pygame.Rect(margin, info_y, chart_width * 2 + margin, 150)
        
        # Process table
        self.process_table = DataTable(margin + chart_width * 2 + margin * 2, info_y, table_width, table_height, "🔧 Top Processes")
        
        # Core usage bars (right side)
        self.core_bars = []
        cores_x = margin + chart_width * 3 + margin * 3
        for i in range(16):  # Support up to 16 cores
            core_y = margin + (i * 25)
            if core_y + 20 < self.height - 100:
                bar = DataBar(cores_x, core_y, 200, 20, f"Core {i}")
                self.core_bars.append(bar)
    
    def _set_high_priority(self):
        """Set high process priority"""
        if WIN32_AVAILABLE:
            try:
                handle = win32api.GetCurrentProcess()
                win32process.SetPriorityClass(handle, win32process.HIGH_PRIORITY_CLASS)
            except:
                pass
    
    def handle_events(self):
        """Handle events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    # Screenshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"data_monitor_screenshot_{timestamp}.png"
                    pygame.image.save(self.screen, filename)
                    print(f"📸 Screenshot saved: {filename}")
                elif event.key == pygame.K_r:
                    # Reset data
                    self.data_monitor.cpu_history.clear()
                    self.data_monitor.memory_history.clear()
                    self.data_monitor.network_history.clear()
                    self.data_monitor.disk_history.clear()
                    print("🔄 Data history reset")
    
    def update(self, dt: float):
        """Update all systems"""
        # Get latest data
        data = self.data_monitor.get_data()
        
        # Update charts
        if 'cpu' in data:
            self.cpu_chart.add_data(data['cpu']['percent'])
            self.cpu_bar.set_value(data['cpu']['percent'])
        
        if 'memory' in data:
            self.memory_chart.add_data(data['memory']['percent'])
            self.memory_bar.set_value(data['memory']['percent'])
        
        if 'network' in data:
            # Convert to MB/s
            total_network = (data['network']['upload'] + data['network']['download']) / (1024 * 1024)
            self.network_chart.add_data(total_network)
        
        if 'disk' in data:
            self.disk_bar.set_value(data['disk']['percent'])
        
        # Update core bars
        if 'cpu' in data and 'per_core' in data['cpu']:
            for i, usage in enumerate(data['cpu']['per_core']):
                if i < len(self.core_bars):
                    self.core_bars[i].set_value(usage)
        
        # Update animations
        self.cpu_bar.update(dt)
        self.memory_bar.update(dt)
        self.disk_bar.update(dt)
        
        for bar in self.core_bars:
            bar.update(dt)
        
        # Update process table
        if 'processes' in data:
            headers = ["PID", "Name", "CPU%", "Memory%", "Status"]
            rows = []
            
            for proc in data['processes'][:15]:  # Top 15 processes
                row = [
                    str(proc.get('pid', 'N/A')),
                    str(proc.get('name', 'Unknown'))[:20],
                    f"{proc.get('cpu_percent', 0):.1f}%",
                    f"{proc.get('memory_percent', 0):.1f}%",
                    str(proc.get('status', 'unknown'))[:10]
                ]
                rows.append(row)
            
            self.process_table.set_data(headers, rows)
    
    def render(self):
        """Render everything"""
        # Clear screen with dark background
        self.screen.fill((5, 5, 10))
        
        # Get current data
        data = self.data_monitor.get_data()
        
        # Render charts
        self.cpu_chart.render(self.screen, self.font_medium)
        self.memory_chart.render(self.screen, self.font_medium)
        self.network_chart.render(self.screen, self.font_medium)
        
        # Render bars
        self.cpu_bar.render(self.screen, self.font_medium)
        self.memory_bar.render(self.screen, self.font_medium)
        self.disk_bar.render(self.screen, self.font_medium)
        
        # Render core bars
        for bar in self.core_bars:
            bar.render(self.screen, self.font_small)
        
        # Render system info panel
        self._render_system_info(data)
        
        # Render process table
        self.process_table.render(self.screen, self.font_small)
        
        # Render header
        self._render_header()
        
        # Render footer with controls
        self._render_footer()
    
    def _render_system_info(self, data: Dict[str, Any]):
        """Render system information panel"""
        # Background
        pygame.draw.rect(self.screen, DataColors.PANEL_BG, self.system_info_rect)
        pygame.draw.rect(self.screen, DataColors.PANEL_BORDER, self.system_info_rect, 2)
        
        # Title
        title_surface = self.font_medium.render("🖥️ System Information", True, DataColors.TEXT_PRIMARY)
        self.screen.blit(title_surface, (self.system_info_rect.x + 5, self.system_info_rect.y + 5))
        
        # System info
        if 'system' in data:
            info_lines = [
                f"OS: {data['system'].get('os', 'Unknown')}",
                f"CPU: {data['system'].get('processor', 'Unknown')}",
                f"Cores: {data['system'].get('cpu_count', 'N/A')} / Threads: {data['system'].get('cpu_logical', 'N/A')}",
                f"Python: {data['system'].get('python', 'Unknown')}"
            ]
            
            # Boot time and uptime
            if 'boot_time' in data['system']:
                uptime = datetime.now() - data['system']['boot_time']
                days = uptime.days
                hours = uptime.seconds // 3600
                minutes = (uptime.seconds % 3600) // 60
                info_lines.append(f"Uptime: {days}d {hours}h {minutes}m")
        else:
            info_lines = ["Loading system information..."]
        
        # Current values
        if 'cpu' in data:
            cpu_freq = data['cpu'].get('frequency', 0)
            cpu_temp = data['cpu'].get('temperature', 0)
            info_lines.append(f"CPU Frequency: {cpu_freq:.0f} MHz")
            if cpu_temp > 0:
                temp_color = DataColors.get_performance_color(cpu_temp, 100)
                info_lines.append(f"CPU Temperature: {cpu_temp:.1f}°C")
        
        if 'memory' in data:
            used_gb = data['memory'].get('used', 0) / (1024**3)
            total_gb = data['memory'].get('total', 0) / (1024**3)
            info_lines.append(f"Memory: {used_gb:.1f} GB / {total_gb:.1f} GB")
        
        if 'network' in data:
            connections = data['network'].get('connections', 0)
            info_lines.append(f"Network Connections: {connections}")
        
        # Render info lines
        y_offset = 30
        for line in info_lines:
            # Check for temperature line for special coloring
            if "Temperature:" in line:
                try:
                    temp_value = float(line.split(":")[1].strip().replace("°C", ""))
                    color = DataColors.get_performance_color(temp_value, 100)
                except:
                    color = DataColors.TEXT_PRIMARY
            else:
                color = DataColors.TEXT_PRIMARY
            
            text_surface = self.font_small.render(line, True, color)
            self.screen.blit(text_surface, (self.system_info_rect.x + 10, self.system_info_rect.y + y_offset))
            y_offset += 18
    
    def _render_header(self):
        """Render header"""
        # Title
        title_text = "🌌 QUANTUM NEXUS DATA FOCUSED MONITOR"
        title_surface = self.font_huge.render(title_text, True, DataColors.BRIGHT_CYAN)
        title_rect = title_surface.get_rect(center=(self.width // 2, 30))
        self.screen.blit(title_surface, title_rect)
        
        # FPS counter
        fps_text = f"FPS: {self.current_fps:.0f}"
        fps_color = DataColors.BRIGHT_GREEN if self.current_fps > 50 else DataColors.BRIGHT_YELLOW
        fps_surface = self.font_medium.render(fps_text, True, fps_color)
        self.screen.blit(fps_surface, (self.width - 100, 10))
        
        # Real-time indicator
        current_time = datetime.now().strftime("%H:%M:%S")
        time_surface = self.font_medium.render(f"실시간: {current_time}", True, DataColors.TEXT_HIGHLIGHT)
        self.screen.blit(time_surface, (10, 10))
    
    def _render_footer(self):
        """Render footer with controls"""
        controls = [
            "🎮 조작법: ESC/Q: 종료 | SPACE: 스크린샷 | R: 데이터 리셋",
            "📊 모든 데이터가 실시간으로 업데이트됩니다 - 성능에 집중한 시각화"
        ]
        
        y_pos = self.height - 50
        for i, control in enumerate(controls):
            color = DataColors.TEXT_HIGHLIGHT if i == 0 else DataColors.TEXT_SECONDARY
            text_surface = self.font_small.render(control, True, color)
            text_rect = text_surface.get_rect(center=(self.width // 2, y_pos + i * 20))
            self.screen.blit(text_surface, text_rect)
    
    def run(self):
        """Main application loop"""
        print("🌌 QUANTUM NEXUS DATA FOCUSED MONITOR - Starting...")
        print(f"Display: {self.width}x{self.height}")
        print(f"Target FPS: {self.target_fps}")
        print("📊 데이터 시각화에 완전 집중한 버전입니다!")
        
        frame_times = deque(maxlen=30)
        
        try:
            while self.running:
                frame_start = time.time()
                
                # Calculate delta time
                current_time = time.time()
                dt = current_time - self.last_time
                self.last_time = current_time
                
                # Handle events
                self.handle_events()
                
                # Update
                self.update(dt)
                
                # Render
                self.render()
                
                # Update display
                pygame.display.flip()
                
                # Calculate FPS
                frame_time = time.time() - frame_start
                frame_times.append(frame_time)
                
                if len(frame_times) > 0:
                    avg_frame_time = sum(frame_times) / len(frame_times)
                    self.current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                
                # Control frame rate
                self.clock.tick(self.target_fps)
        
        except KeyboardInterrupt:
            print("\n⚡ 데이터 모니터 종료됨")
        except Exception as e:
            print(f"❌ 오류: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("🧹 리소스 정리 중...")
        
        if hasattr(self, 'data_monitor'):
            self.data_monitor.cleanup()
        
        pygame.quit()
        print("✅ 정리 완료")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():
    """Main entry point"""
    try:
        app = DataFocusedMonitor()
        app.run()
    except KeyboardInterrupt:
        print("\n⚡ 프로그램 종료")
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
    finally:
        try:
            pygame.quit()
        except:
            pass
        sys.exit(0)

if __name__ == "__main__":
    main()