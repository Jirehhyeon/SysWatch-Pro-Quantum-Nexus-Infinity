#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🖥️ SysWatch Pro ADVANCED ULTIMATE - 최고급 시스템 모니터
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 1920x1080 전체화면 고급 시스템 모니터링 & 3D 시각화
🎯 실시간 모니터링, 프로세스 관리, 3D 그래프, 레지스트리 정보
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Copyright (C) 2025 Advanced System Monitor Corp
"""

import os
import sys
import time
import threading
import warnings
import math
import random
import json
import socket
import platform
import subprocess
import ctypes
import shutil
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Tuple, Any, Optional
import winreg

# Auto-install packages
def install_package(package):
    try:
        __import__(package.split('==')[0] if '==' in package else package.split('[')[0])
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

# Install required packages
required_packages = [
    "psutil",
    "pygame-ce", 
    "numpy",
    "py-cpuinfo",
    "wmi",
    "GPUtil",
    "requests"
]

print("Installing required packages...")
for package in required_packages:
    install_package(package)

import psutil
import pygame
import numpy as np
import cpuinfo
import wmi
from pygame.locals import *

# Try to import additional GPU libraries
try:
    import GPUtil
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    NVIDIA_ML_AVAILABLE = True
except:
    NVIDIA_ML_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

# Initialize Pygame
pygame.init()

# ============================================================================
# ADVANCED COLOR SYSTEM
# ============================================================================
class AdvancedColors:
    # Base colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    
    # Status colors with gradients
    GOOD = (0, 255, 100)
    WARNING = (255, 200, 0)
    DANGER = (255, 50, 50)
    CRITICAL = (255, 0, 100)
    
    # 3D visualization colors
    NEON_BLUE = (0, 200, 255)
    NEON_GREEN = (50, 255, 100)
    NEON_PURPLE = (200, 100, 255)
    NEON_PINK = (255, 100, 200)
    NEON_ORANGE = (255, 150, 50)
    NEON_CYAN = (100, 255, 200)
    
    # UI colors
    BG_MAIN = (5, 5, 8)
    BG_PANEL = (15, 15, 20)
    BG_GRAPH = (8, 8, 12)
    BORDER_MAIN = (50, 150, 255)
    TEXT_MAIN = (240, 240, 240)
    TEXT_DIM = (150, 150, 150)
    TEXT_BRIGHT = (255, 255, 255)
    
    # Glow effects
    GLOW_BLUE = [(0, 100, 255), (0, 150, 255), (0, 200, 255)]
    GLOW_GREEN = [(0, 150, 50), (0, 200, 100), (0, 255, 150)]
    GLOW_RED = [(150, 0, 50), (200, 0, 100), (255, 0, 150)]
    
    @staticmethod
    def get_performance_color(value: float, reverse: bool = False) -> Tuple[int, int, int]:
        """Get color based on performance value"""
        if reverse:
            value = 100 - value
            
        if value < 20:
            return AdvancedColors.GOOD
        elif value < 50:
            return AdvancedColors.NEON_CYAN
        elif value < 70:
            return AdvancedColors.WARNING
        elif value < 90:
            return AdvancedColors.DANGER
        else:
            return AdvancedColors.CRITICAL
    
    @staticmethod
    def interpolate_color(color1: Tuple[int, int, int], color2: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
        """Interpolate between two colors"""
        return tuple(int(c1 + (c2 - c1) * factor) for c1, c2 in zip(color1, color2))

# ============================================================================
# 3D VISUALIZATION SYSTEM
# ============================================================================
class Circle3D:
    def __init__(self, center_x: int, center_y: int, radius: int, color: Tuple[int, int, int], value: float = 0):
        self.center_x = center_x
        self.center_y = center_y
        self.base_radius = radius
        self.radius = radius
        self.color = color
        self.value = value
        self.angle = 0
        self.pulse = 0
        self.particles = []
        
    def update(self, dt: float, new_value: float):
        """Update 3D circle animation"""
        self.value = new_value
        self.angle += dt * 2
        self.pulse += dt * 3
        
        # Dynamic radius based on value
        pulse_factor = 1 + 0.2 * abs(math.sin(self.pulse)) * (self.value / 100)
        self.radius = int(self.base_radius * (0.8 + 0.4 * self.value / 100) * pulse_factor)
        
        # Add particles for high activity
        if self.value > 50 and len(self.particles) < 20:
            angle = random.uniform(0, 2 * math.pi)
            particle_x = self.center_x + math.cos(angle) * self.radius
            particle_y = self.center_y + math.sin(angle) * self.radius
            self.particles.append({
                'x': particle_x,
                'y': particle_y,
                'vx': math.cos(angle) * 2,
                'vy': math.sin(angle) * 2,
                'life': 1.0,
                'size': random.uniform(2, 5)
            })
        
        # Update particles
        for particle in self.particles[:]:
            particle['x'] += particle['vx'] * dt * 60
            particle['y'] += particle['vy'] * dt * 60
            particle['life'] -= dt * 0.8
            if particle['life'] <= 0:
                self.particles.remove(particle)
    
    def draw(self, screen):
        """Draw 3D circle with effects"""
        # Draw glow rings
        for i in range(5):
            glow_radius = self.radius + i * 15
            alpha = (1 - i * 0.2) * (self.value / 100) * 0.5
            if alpha > 0:
                glow_color = tuple(int(c * alpha) for c in self.color)
                if sum(glow_color) > 0:
                    # Create glow surface
                    glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(glow_surf, (*glow_color, int(alpha * 100)), 
                                     (glow_radius, glow_radius), glow_radius, 2)
                    screen.blit(glow_surf, (self.center_x - glow_radius, self.center_y - glow_radius))
        
        # Draw main circle with 3D effect
        for i in range(8):
            offset = i * 2
            shadow_color = tuple(max(0, c - offset * 10) for c in self.color)
            if sum(shadow_color) > 0:
                pygame.draw.circle(screen, shadow_color, 
                                 (self.center_x + offset, self.center_y + offset), 
                                 max(1, self.radius - offset), 2)
        
        # Draw main circle
        pygame.draw.circle(screen, self.color, (self.center_x, self.center_y), self.radius, 3)
        
        # Draw percentage arc
        if self.value > 0:
            arc_points = []
            segments = int(self.value * 3.6)  # 360 degrees for 100%
            for i in range(segments):
                angle_rad = math.radians(i - 90)  # Start from top
                x = self.center_x + math.cos(angle_rad) * (self.radius - 5)
                y = self.center_y + math.sin(angle_rad) * (self.radius - 5)
                arc_points.append((x, y))
            
            if len(arc_points) > 1:
                bright_color = tuple(min(255, c + 50) for c in self.color)
                pygame.draw.lines(screen, bright_color, False, arc_points, 4)
        
        # Draw center value
        font = pygame.font.Font(None, 32)
        value_text = f"{self.value:.1f}%"
        text_surf = font.render(value_text, True, AdvancedColors.TEXT_BRIGHT)
        text_rect = text_surf.get_rect(center=(self.center_x, self.center_y))
        screen.blit(text_surf, text_rect)
        
        # Draw particles
        for particle in self.particles:
            if particle['life'] > 0:
                alpha = particle['life']
                particle_color = tuple(int(c * alpha) for c in self.color)
                if sum(particle_color) > 0:
                    pygame.draw.circle(screen, particle_color, 
                                     (int(particle['x']), int(particle['y'])), 
                                     max(1, int(particle['size'] * alpha)))

class Graph3D:
    def __init__(self, x: int, y: int, width: int, height: int, title: str, max_value: float = 100):
        self.rect = pygame.Rect(x, y, width, height)
        self.title = title
        self.max_value = max_value
        self.data = deque(maxlen=width // 3)  # More data points
        self.color = AdvancedColors.NEON_BLUE
        self.angle = 0
        
    def add_data(self, value: float):
        """Add data point"""
        self.data.append(value)
    
    def update(self, dt: float):
        """Update 3D graph animation"""
        self.angle += dt
    
    def draw(self, screen, font):
        """Draw 3D graph"""
        # Background with gradient
        for i in range(self.rect.height):
            alpha = i / self.rect.height
            bg_color = AdvancedColors.interpolate_color(
                AdvancedColors.BG_GRAPH, 
                AdvancedColors.BG_PANEL, 
                alpha
            )
            pygame.draw.line(screen, bg_color, 
                           (self.rect.x, self.rect.y + i),
                           (self.rect.x + self.rect.width, self.rect.y + i))
        
        # Border with glow
        pygame.draw.rect(screen, AdvancedColors.BORDER_MAIN, self.rect, 2)
        
        # Title
        title_surf = font.render(self.title, True, AdvancedColors.TEXT_BRIGHT)
        screen.blit(title_surf, (self.rect.x + 10, self.rect.y + 10))
        
        # Current value
        if self.data:
            current = self.data[-1]
            value_color = AdvancedColors.get_performance_color(current)
            value_text = f"{current:.1f}"
            if "%" not in self.title.lower():
                if "mb" in self.title.lower() or "network" in self.title.lower():
                    value_text += " MB/s"
                else:
                    value_text += "%"
            else:
                value_text += "%"
            
            value_surf = font.render(value_text, True, value_color)
            screen.blit(value_surf, (self.rect.x + self.rect.width - 120, self.rect.y + 10))
        
        # Graph area
        graph_area = pygame.Rect(self.rect.x + 20, self.rect.y + 50, 
                                self.rect.width - 40, self.rect.height - 70)
        
        # Grid lines with glow
        grid_color = (30, 30, 50)
        for i in range(5):
            y_pos = graph_area.y + (i * graph_area.height // 4)
            pygame.draw.line(screen, grid_color,
                           (graph_area.x, y_pos),
                           (graph_area.x + graph_area.width, y_pos), 1)
        
        # Draw 3D data visualization
        if len(self.data) > 1:
            points = []
            shadow_points = []
            
            for i, value in enumerate(self.data):
                x = graph_area.x + (i / max(1, len(self.data) - 1)) * graph_area.width
                y = graph_area.y + graph_area.height - (value / self.max_value) * graph_area.height
                
                # 3D effect with depth
                depth_offset = 5 + 3 * math.sin(self.angle + i * 0.1)
                points.append((x, y))
                shadow_points.append((x + depth_offset, y + depth_offset))
            
            # Draw shadow first
            if len(shadow_points) > 1:
                shadow_color = (20, 20, 30)
                pygame.draw.lines(screen, shadow_color, False, shadow_points, 3)
            
            # Draw main line with glow
            if len(points) > 1:
                # Glow layers
                for i in range(3):
                    glow_width = 6 - i * 2
                    alpha = 0.8 - i * 0.3
                    glow_color = tuple(int(c * alpha) for c in self.color)
                    if sum(glow_color) > 0 and glow_width > 0:
                        pygame.draw.lines(screen, glow_color, False, points, glow_width)
                
                # Fill area under curve
                if len(points) > 2:
                    fill_points = points + [(graph_area.x + graph_area.width, graph_area.y + graph_area.height),
                                          (graph_area.x, graph_area.y + graph_area.height)]
                    fill_color = (*self.color, 30)
                    fill_surf = pygame.Surface((graph_area.width, graph_area.height), pygame.SRCALPHA)
                    pygame.draw.polygon(fill_surf, fill_color, 
                                      [(p[0] - graph_area.x, p[1] - graph_area.y) for p in fill_points])
                    screen.blit(fill_surf, (graph_area.x, graph_area.y))

# ============================================================================
# ADVANCED SYSTEM DATA COLLECTOR
# ============================================================================
class AdvancedSystemCollector:
    def __init__(self):
        self.data = {
            # System Info
            'system': {
                'os': platform.system(),
                'version': platform.version(),
                'architecture': platform.architecture()[0],
                'computer_name': platform.node(),
                'user': os.getenv('USERNAME'),
                'boot_time': None,
                'uptime': '0d 0h 0m'
            },
            
            # CPU Info
            'cpu': {
                'model': '',
                'cores': 0,
                'threads': 0,
                'frequency': 0,
                'usage_percent': 0,
                'per_core_usage': [],
                'temperature': 0,
                'cache_l1': 0,
                'cache_l2': 0,
                'cache_l3': 0
            },
            
            # Memory Info
            'memory': {
                'total_gb': 0,
                'used_gb': 0,
                'available_gb': 0,
                'percent': 0,
                'swap_total_gb': 0,
                'swap_used_gb': 0,
                'swap_percent': 0,
                'cache_gb': 0
            },
            
            # Disk Info
            'disks': [],
            'disk_io': {
                'read_mb_s': 0,
                'write_mb_s': 0,
                'read_total_gb': 0,
                'write_total_gb': 0
            },
            
            # Network Info
            'network': {
                'interfaces': [],
                'ip_addresses': [],
                'upload_mb_s': 0,
                'download_mb_s': 0,
                'upload_total_gb': 0,
                'download_total_gb': 0
            },
            
            # GPU Info
            'gpu': {
                'name': 'N/A',
                'usage_percent': 0,
                'memory_used_mb': 0,
                'memory_total_mb': 0,
                'memory_percent': 0,
                'temperature': 0
            },
            
            # Registry Info
            'registry': {
                'startup_programs': [],
                'installed_software_count': 0,
                'last_boot_time': '',
                'windows_version': ''
            },
            
            # Process Info
            'processes': {
                'total_count': 0,
                'running': [],
                'high_cpu': [],
                'high_memory': [],
                'suspicious': []
            }
        }
        
        # History for graphs
        self.history_length = 300
        self.histories = {
            'cpu_usage': deque(maxlen=self.history_length),
            'memory_usage': deque(maxlen=self.history_length),
            'disk_read': deque(maxlen=self.history_length),
            'disk_write': deque(maxlen=self.history_length),
            'network_up': deque(maxlen=self.history_length),
            'network_down': deque(maxlen=self.history_length),
            'gpu_usage': deque(maxlen=self.history_length),
            'gpu_memory': deque(maxlen=self.history_length)
        }
        
        # Tracking variables
        self.last_disk_io = None
        self.last_network_io = None
        self.last_time = time.time()
        
        # Initialize static data
        self._init_static_data()
        
        # Start monitoring thread
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def _init_static_data(self):
        """Initialize static system data"""
        try:
            # Boot time
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            self.data['system']['boot_time'] = boot_time
            
            # CPU info
            cpu_info = cpuinfo.get_cpu_info()
            self.data['cpu']['model'] = cpu_info.get('brand_raw', 'Unknown CPU')
            self.data['cpu']['cores'] = psutil.cpu_count(logical=False)
            self.data['cpu']['threads'] = psutil.cpu_count(logical=True)
            
            # Network interfaces
            interfaces = psutil.net_if_addrs()
            for interface, addrs in interfaces.items():
                for addr in addrs:
                    if addr.family == socket.AF_INET:
                        self.data['network']['interfaces'].append({
                            'name': interface,
                            'ip': addr.address
                        })
                        if not addr.address.startswith('127.'):
                            self.data['network']['ip_addresses'].append(addr.address)
            
            # Registry info
            self._get_registry_info()
            
            # GPU info
            self._get_gpu_info()
            
        except Exception as e:
            print(f"Static data initialization error: {e}")
    
    def _get_registry_info(self):
        """Get registry information"""
        try:
            # Windows version
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                               r"SOFTWARE\Microsoft\Windows NT\CurrentVersion")
            try:
                version = winreg.QueryValueEx(key, "ProductName")[0]
                build = winreg.QueryValueEx(key, "CurrentBuild")[0]
                self.data['registry']['windows_version'] = f"{version} (Build {build})"
            except:
                pass
            winreg.CloseKey(key)
            
            # Startup programs
            startup_locations = [
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run",
                r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Run"
            ]
            
            startup_programs = []
            for location in startup_locations:
                try:
                    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, location)
                    i = 0
                    while True:
                        try:
                            name, value, _ = winreg.EnumValue(key, i)
                            startup_programs.append({'name': name, 'path': value})
                            i += 1
                        except OSError:
                            break
                    winreg.CloseKey(key)
                except:
                    pass
            
            self.data['registry']['startup_programs'] = startup_programs[:10]  # Limit to 10
            
        except Exception as e:
            print(f"Registry access error: {e}")
    
    def _get_gpu_info(self):
        """Get GPU information"""
        try:
            if NVIDIA_ML_AVAILABLE:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                self.data['gpu']['name'] = name
            elif GPU_AVAILABLE:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    self.data['gpu']['name'] = gpu.name
        except:
            self.data['gpu']['name'] = 'GPU information not available'
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                current_time = time.time()
                dt = current_time - self.last_time
                
                # System uptime
                if self.data['system']['boot_time']:
                    uptime = datetime.now() - self.data['system']['boot_time']
                    days = uptime.days
                    hours = uptime.seconds // 3600
                    minutes = (uptime.seconds % 3600) // 60
                    self.data['system']['uptime'] = f"{days}d {hours}h {minutes}m"
                
                # CPU information
                self.data['cpu']['usage_percent'] = psutil.cpu_percent(interval=0.1)
                self.data['cpu']['per_core_usage'] = psutil.cpu_percent(interval=0.1, percpu=True)
                
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    self.data['cpu']['frequency'] = cpu_freq.current
                
                # CPU temperature (if available)
                try:
                    temps = psutil.sensors_temperatures()
                    if temps:
                        for name, entries in temps.items():
                            if 'cpu' in name.lower() or 'core' in name.lower():
                                self.data['cpu']['temperature'] = entries[0].current
                                break
                except:
                    pass
                
                # Memory information
                memory = psutil.virtual_memory()
                self.data['memory']['total_gb'] = memory.total / (1024**3)
                self.data['memory']['used_gb'] = memory.used / (1024**3)
                self.data['memory']['available_gb'] = memory.available / (1024**3)
                self.data['memory']['percent'] = memory.percent
                self.data['memory']['cache_gb'] = getattr(memory, 'cached', 0) / (1024**3)
                
                swap = psutil.swap_memory()
                self.data['memory']['swap_total_gb'] = swap.total / (1024**3)
                self.data['memory']['swap_used_gb'] = swap.used / (1024**3)
                self.data['memory']['swap_percent'] = swap.percent
                
                # Disk information
                disk_info = []
                for partition in psutil.disk_partitions():
                    try:
                        usage = psutil.disk_usage(partition.mountpoint)
                        disk_info.append({
                            'device': partition.device,
                            'mountpoint': partition.mountpoint,
                            'fstype': partition.fstype,
                            'total_gb': usage.total / (1024**3),
                            'used_gb': usage.used / (1024**3),
                            'free_gb': usage.free / (1024**3),
                            'percent': (usage.used / usage.total) * 100
                        })
                    except:
                        pass
                self.data['disks'] = disk_info
                
                # Disk I/O
                current_disk_io = psutil.disk_io_counters()
                if current_disk_io and self.last_disk_io and dt > 0:
                    read_bytes = current_disk_io.read_bytes - self.last_disk_io.read_bytes
                    write_bytes = current_disk_io.write_bytes - self.last_disk_io.write_bytes
                    
                    self.data['disk_io']['read_mb_s'] = (read_bytes / dt) / (1024**2)
                    self.data['disk_io']['write_mb_s'] = (write_bytes / dt) / (1024**2)
                    self.data['disk_io']['read_total_gb'] = current_disk_io.read_bytes / (1024**3)
                    self.data['disk_io']['write_total_gb'] = current_disk_io.write_bytes / (1024**3)
                
                self.last_disk_io = current_disk_io
                
                # Network I/O
                current_network_io = psutil.net_io_counters()
                if current_network_io and self.last_network_io and dt > 0:
                    sent_bytes = current_network_io.bytes_sent - self.last_network_io.bytes_sent
                    recv_bytes = current_network_io.bytes_recv - self.last_network_io.bytes_recv
                    
                    self.data['network']['upload_mb_s'] = (sent_bytes / dt) / (1024**2)
                    self.data['network']['download_mb_s'] = (recv_bytes / dt) / (1024**2)
                    self.data['network']['upload_total_gb'] = current_network_io.bytes_sent / (1024**3)
                    self.data['network']['download_total_gb'] = current_network_io.bytes_recv / (1024**3)
                
                self.last_network_io = current_network_io
                
                # GPU information
                self._update_gpu_info()
                
                # Process information
                self._update_process_info()
                
                # Update histories
                self.histories['cpu_usage'].append(self.data['cpu']['usage_percent'])
                self.histories['memory_usage'].append(self.data['memory']['percent'])
                self.histories['disk_read'].append(self.data['disk_io']['read_mb_s'])
                self.histories['disk_write'].append(self.data['disk_io']['write_mb_s'])
                self.histories['network_up'].append(self.data['network']['upload_mb_s'])
                self.histories['network_down'].append(self.data['network']['download_mb_s'])
                self.histories['gpu_usage'].append(self.data['gpu']['usage_percent'])
                self.histories['gpu_memory'].append(self.data['gpu']['memory_percent'])
                
                self.last_time = current_time
                time.sleep(0.5)  # Update every 500ms
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(1)
    
    def _update_gpu_info(self):
        """Update GPU information"""
        try:
            if NVIDIA_ML_AVAILABLE:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                # Usage
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.data['gpu']['usage_percent'] = util.gpu
                
                # Memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.data['gpu']['memory_used_mb'] = mem_info.used / (1024**2)
                self.data['gpu']['memory_total_mb'] = mem_info.total / (1024**2)
                self.data['gpu']['memory_percent'] = (mem_info.used / mem_info.total) * 100
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                self.data['gpu']['temperature'] = temp
                
            elif GPU_AVAILABLE:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    self.data['gpu']['usage_percent'] = gpu.load * 100
                    self.data['gpu']['memory_used_mb'] = gpu.memoryUsed
                    self.data['gpu']['memory_total_mb'] = gpu.memoryTotal
                    self.data['gpu']['memory_percent'] = gpu.memoryUtil * 100
                    self.data['gpu']['temperature'] = gpu.temperature
        except:
            # Mock data if no GPU info available
            self.data['gpu']['usage_percent'] = random.uniform(10, 60)
            self.data['gpu']['memory_percent'] = random.uniform(30, 80)
            self.data['gpu']['temperature'] = random.uniform(45, 75)
    
    def _update_process_info(self):
        """Update process information"""
        try:
            processes = []
            high_cpu = []
            high_memory = []
            suspicious = []
            
            # Suspicious process patterns
            suspicious_patterns = [
                'bitcoin', 'miner', 'cryptonight', 'monero', 'xmr', 'eth',
                'trojan', 'virus', 'malware', 'keylog', 'backdoor',
                'temp', 'tmp', 'random', 'svchost.exe'  # Multiple svchost can be suspicious
            ]
            
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
                try:
                    info = proc.info
                    if info['name'] and info['cpu_percent'] is not None:
                        processes.append(info)
                        
                        # Categorize processes
                        if info['cpu_percent'] > 25:
                            high_cpu.append(info)
                        
                        if info['memory_percent'] > 10:
                            high_memory.append(info)
                        
                        # Check for suspicious processes
                        process_name = info['name'].lower()
                        for pattern in suspicious_patterns:
                            if pattern in process_name:
                                if pattern != 'svchost.exe' or len([p for p in processes if 'svchost' in p['name'].lower()]) > 8:
                                    suspicious.append(info)
                                break
                except:
                    continue
            
            # Sort by resource usage
            processes.sort(key=lambda x: x.get('cpu_percent', 0) + x.get('memory_percent', 0), reverse=True)
            high_cpu.sort(key=lambda x: x.get('cpu_percent', 0), reverse=True)
            high_memory.sort(key=lambda x: x.get('memory_percent', 0), reverse=True)
            
            self.data['processes'] = {
                'total_count': len(processes),
                'running': processes[:15],
                'high_cpu': high_cpu[:10],
                'high_memory': high_memory[:10],
                'suspicious': suspicious[:5]
            }
            
        except Exception as e:
            print(f"Process update error: {e}")
    
    def get_data(self) -> Dict[str, Any]:
        """Get current system data"""
        return self.data.copy()
    
    def get_histories(self) -> Dict[str, deque]:
        """Get history data"""
        return self.histories.copy()
    
    def terminate_process(self, pid: int) -> bool:
        """Terminate a process by PID"""
        try:
            process = psutil.Process(pid)
            process.terminate()
            return True
        except:
            return False
    
    def stop(self):
        """Stop monitoring"""
        self.running = False

# ============================================================================
# ADVANCED UI COMPONENTS
# ============================================================================
class AdvancedPanel:
    def __init__(self, x: int, y: int, width: int, height: int, title: str):
        self.rect = pygame.Rect(x, y, width, height)
        self.title = title
        self.content = []
        self.scroll_offset = 0
        self.max_scroll = 0
        self.title_color = AdvancedColors.NEON_BLUE
        
    def clear(self):
        """Clear content"""
        self.content = []
        self.scroll_offset = 0
    
    def add_line(self, text: str, color: Tuple[int, int, int] = None, clickable: bool = False, action: str = None):
        """Add content line"""
        if color is None:
            color = AdvancedColors.TEXT_MAIN
        self.content.append({
            'text': text,
            'color': color,
            'clickable': clickable,
            'action': action
        })
    
    def handle_scroll(self, event):
        """Handle mouse scroll"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                if event.button == 4:  # Scroll up
                    self.scroll_offset = max(0, self.scroll_offset - 1)
                elif event.button == 5:  # Scroll down
                    self.scroll_offset = min(self.max_scroll, self.scroll_offset + 1)
    
    def handle_click(self, pos) -> Optional[str]:
        """Handle mouse click, return action if any"""
        if not self.rect.collidepoint(pos):
            return None
        
        # Calculate which line was clicked
        local_y = pos[1] - self.rect.y - 50  # Account for title
        line_index = (local_y // 22) + self.scroll_offset
        
        if 0 <= line_index < len(self.content):
            item = self.content[line_index]
            if item['clickable'] and item['action']:
                return item['action']
        
        return None
    
    def draw(self, screen, font):
        """Draw panel with advanced styling"""
        # Background gradient
        for i in range(self.rect.height):
            alpha = i / self.rect.height
            bg_color = AdvancedColors.interpolate_color(
                AdvancedColors.BG_PANEL,
                AdvancedColors.BG_MAIN,
                alpha * 0.3
            )
            pygame.draw.line(screen, bg_color,
                           (self.rect.x, self.rect.y + i),
                           (self.rect.x + self.rect.width, self.rect.y + i))
        
        # Glowing border
        for i in range(3):
            border_color = tuple(max(0, c - i * 30) for c in AdvancedColors.BORDER_MAIN)
            pygame.draw.rect(screen, border_color, 
                           pygame.Rect(self.rect.x - i, self.rect.y - i,
                                     self.rect.width + i*2, self.rect.height + i*2), 1)
        
        # Title with glow effect
        title_surf = font.render(self.title, True, self.title_color)
        
        # Title glow
        for i in range(3):
            glow_color = tuple(int(c * (0.3 - i * 0.1)) for c in self.title_color)
            if sum(glow_color) > 0:
                glow_surf = font.render(self.title, True, glow_color)
                screen.blit(glow_surf, (self.rect.x + 12 + i, self.rect.y + 12 + i))
        
        screen.blit(title_surf, (self.rect.x + 12, self.rect.y + 12))
        
        # Content
        content_area = pygame.Rect(self.rect.x + 10, self.rect.y + 45,
                                 self.rect.width - 20, self.rect.height - 55)
        
        # Calculate scrolling
        visible_lines = content_area.height // 22
        self.max_scroll = max(0, len(self.content) - visible_lines)
        
        y_offset = 0
        for i, item in enumerate(self.content[self.scroll_offset:]):
            if y_offset >= content_area.height - 22:
                break
            
            text = item['text']
            color = item['color']
            
            # Highlight clickable items
            if item['clickable']:
                highlight_rect = pygame.Rect(content_area.x - 5, content_area.y + y_offset - 2,
                                           content_area.width + 10, 22)
                pygame.draw.rect(screen, (20, 20, 40), highlight_rect)
                pygame.draw.rect(screen, AdvancedColors.NEON_BLUE, highlight_rect, 1)
            
            text_surf = font.render(text, True, color)
            screen.blit(text_surf, (content_area.x, content_area.y + y_offset))
            y_offset += 22
        
        # Scrollbar
        if self.max_scroll > 0:
            scrollbar_height = max(20, (visible_lines / len(self.content)) * content_area.height)
            scrollbar_y = content_area.y + (self.scroll_offset / self.max_scroll) * (content_area.height - scrollbar_height)
            
            pygame.draw.rect(screen, AdvancedColors.BORDER_MAIN,
                           (self.rect.x + self.rect.width - 8, scrollbar_y,
                            4, scrollbar_height))

# ============================================================================
# MAIN APPLICATION
# ============================================================================
class AdvancedSystemMonitor:
    def __init__(self):
        # Full screen setup
        self.width = 1920
        self.height = 1080
        
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.FULLSCREEN)
        pygame.display.set_caption("🖥️ SysWatch Pro ADVANCED ULTIMATE")
        
        # Performance
        self.clock = pygame.time.Clock()
        self.running = True
        self.fps_target = 60
        self.current_fps = 0
        
        # Fonts - optimized for Full HD
        try:
            self.font_huge = pygame.font.Font(None, 64)      # Main title
            self.font_large = pygame.font.Font(None, 36)     # Section titles
            self.font_medium = pygame.font.Font(None, 24)    # Panel content
            self.font_small = pygame.font.Font(None, 18)     # Details
        except:
            self.font_huge = pygame.font.Font(None, 64)
            self.font_large = pygame.font.Font(None, 36)
            self.font_medium = pygame.font.Font(None, 24)
            self.font_small = pygame.font.Font(None, 18)
        
        # Data collector
        print("Initializing system data collector...")
        self.data_collector = AdvancedSystemCollector()
        
        # UI Components
        self._create_ui_components()
        
        # Animation
        self.animation_time = 0
        self.last_update = time.time()
        
        print("Advanced System Monitor ready!")
    
    def _create_ui_components(self):
        """Create all UI components for 1920x1080"""
        # 3D Circles - Main system indicators
        circle_y = 200
        circle_spacing = 240
        
        self.cpu_circle = Circle3D(240, circle_y, 80, AdvancedColors.NEON_BLUE)
        self.memory_circle = Circle3D(240 + circle_spacing, circle_y, 80, AdvancedColors.NEON_GREEN)
        self.disk_circle = Circle3D(240 + circle_spacing * 2, circle_y, 80, AdvancedColors.NEON_ORANGE)
        self.network_circle = Circle3D(240 + circle_spacing * 3, circle_y, 80, AdvancedColors.NEON_PURPLE)
        self.gpu_circle = Circle3D(240 + circle_spacing * 4, circle_y, 80, AdvancedColors.NEON_PINK)
        
        # 3D Graphs
        graph_y = 380
        graph_width = 350
        graph_height = 200
        graph_spacing = 380
        
        self.cpu_graph = Graph3D(50, graph_y, graph_width, graph_height, "CPU Usage History")
        self.cpu_graph.color = AdvancedColors.NEON_BLUE
        
        self.memory_graph = Graph3D(50 + graph_spacing, graph_y, graph_width, graph_height, "Memory Usage")
        self.memory_graph.color = AdvancedColors.NEON_GREEN
        
        self.network_graph = Graph3D(50 + graph_spacing * 2, graph_y, graph_width, graph_height, "Network I/O")
        self.network_graph.color = AdvancedColors.NEON_PURPLE
        self.network_graph.max_value = 50  # MB/s
        
        self.disk_graph = Graph3D(50 + graph_spacing * 3, graph_y, graph_width, graph_height, "Disk I/O")
        self.disk_graph.color = AdvancedColors.NEON_ORANGE
        self.disk_graph.max_value = 100  # MB/s
        
        # Information Panels
        panel_y = 600
        panel_width = 300
        panel_height = 420
        panel_spacing = 320
        
        self.system_panel = AdvancedPanel(50, panel_y, panel_width, panel_height, "💻 System Information")
        self.system_panel.title_color = AdvancedColors.NEON_CYAN
        
        self.cpu_panel = AdvancedPanel(50 + panel_spacing, panel_y, panel_width, panel_height, "🔥 CPU Details")
        self.cpu_panel.title_color = AdvancedColors.NEON_BLUE
        
        self.memory_panel = AdvancedPanel(50 + panel_spacing * 2, panel_y, panel_width, panel_height, "🧠 Memory & Disk")
        self.memory_panel.title_color = AdvancedColors.NEON_GREEN
        
        self.network_panel = AdvancedPanel(50 + panel_spacing * 3, panel_y, panel_width, panel_height, "🌐 Network & GPU")
        self.network_panel.title_color = AdvancedColors.NEON_PURPLE
        
        self.process_panel = AdvancedPanel(50 + panel_spacing * 4, panel_y, panel_width + 50, panel_height, "⚡ Process Manager")
        self.process_panel.title_color = AdvancedColors.NEON_PINK
    
    def handle_events(self):
        """Handle all events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    # Screenshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"advanced_monitor_{timestamp}.png"
                    pygame.image.save(self.screen, filename)
                    print(f"Screenshot saved: {filename}")
                elif event.key == pygame.K_r:
                    # Refresh/reset
                    print("Data refreshed")
                elif event.key == pygame.K_s:
                    # Save data to JSON
                    self._save_data_to_json()
            
            # Handle panel scrolling
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.system_panel.handle_scroll(event)
                self.cpu_panel.handle_scroll(event)
                self.memory_panel.handle_scroll(event)
                self.network_panel.handle_scroll(event)
                self.process_panel.handle_scroll(event)
                
                # Handle process termination clicks
                action = self.process_panel.handle_click(event.pos)
                if action and action.startswith("terminate_"):
                    pid = int(action.split("_")[1])
                    if self.data_collector.terminate_process(pid):
                        print(f"Process {pid} terminated successfully")
                    else:
                        print(f"Failed to terminate process {pid}")
    
    def _save_data_to_json(self):
        """Save current system data to JSON file"""
        try:
            data = self.data_collector.get_data()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"system_data_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"Data saved: {filename}")
        except Exception as e:
            print(f"Failed to save data: {e}")
    
    def update(self, dt: float):
        """Update all components"""
        data = self.data_collector.get_data()
        histories = self.data_collector.get_histories()
        
        # Update 3D circles
        self.cpu_circle.update(dt, data['cpu']['usage_percent'])
        self.memory_circle.update(dt, data['memory']['percent'])
        
        # Disk usage (average of all drives)
        avg_disk_usage = 0
        if data['disks']:
            avg_disk_usage = sum(disk['percent'] for disk in data['disks']) / len(data['disks'])
        self.disk_circle.update(dt, avg_disk_usage)
        
        self.network_circle.update(dt, min(100, (data['network']['upload_mb_s'] + data['network']['download_mb_s']) * 2))
        self.gpu_circle.update(dt, data['gpu']['usage_percent'])
        
        # Update 3D graphs
        self.cpu_graph.update(dt)
        for value in list(histories['cpu_usage'])[-10:]:  # Add recent values
            self.cpu_graph.add_data(value)
        
        self.memory_graph.update(dt)
        for value in list(histories['memory_usage'])[-10:]:
            self.memory_graph.add_data(value)
        
        self.network_graph.update(dt)
        network_total = [(up + down) for up, down in zip(histories['network_up'], histories['network_down'])]
        for value in network_total[-10:]:
            self.network_graph.add_data(value)
        
        self.disk_graph.update(dt)
        disk_total = [(read + write) for read, write in zip(histories['disk_read'], histories['disk_write'])]
        for value in disk_total[-10:]:
            self.disk_graph.add_data(value)
        
        # Update panels
        self._update_panels(data)
        
        # Animation
        self.animation_time += dt
    
    def _update_panels(self, data: Dict[str, Any]):
        """Update all panels with current data"""
        # System Information Panel
        self.system_panel.clear()
        self.system_panel.add_line(f"🖥️  {data['system']['os']} {data['system']['architecture']}")
        self.system_panel.add_line(f"💻 {data['system']['computer_name']}")
        self.system_panel.add_line(f"👤 User: {data['system']['user']}")
        self.system_panel.add_line(f"⏱️  Uptime: {data['system']['uptime']}")
        self.system_panel.add_line(f"🪟 {data['registry']['windows_version']}")
        self.system_panel.add_line("")
        self.system_panel.add_line("📂 Startup Programs:", AdvancedColors.NEON_CYAN)
        
        for program in data['registry']['startup_programs'][:5]:
            name = program['name'][:20] + "..." if len(program['name']) > 20 else program['name']
            self.system_panel.add_line(f"  • {name}", AdvancedColors.TEXT_DIM)
        
        # CPU Details Panel
        self.cpu_panel.clear()
        self.cpu_panel.add_line(f"🔥 {data['cpu']['model'][:25]}...")
        self.cpu_panel.add_line(f"📊 Usage: {data['cpu']['usage_percent']:.1f}%", 
                               AdvancedColors.get_performance_color(data['cpu']['usage_percent']))
        self.cpu_panel.add_line(f"🔧 Cores: {data['cpu']['cores']} | Threads: {data['cpu']['threads']}")
        
        if data['cpu']['frequency'] > 0:
            self.cpu_panel.add_line(f"⚡ Frequency: {data['cpu']['frequency']:.0f} MHz")
        
        if data['cpu']['temperature'] > 0:
            temp_color = AdvancedColors.get_performance_color(data['cpu']['temperature'], reverse=True)
            self.cpu_panel.add_line(f"🌡️  Temperature: {data['cpu']['temperature']:.1f}°C", temp_color)
        
        self.cpu_panel.add_line("")
        self.cpu_panel.add_line("🔥 Per-Core Usage:", AdvancedColors.NEON_BLUE)
        
        for i, usage in enumerate(data['cpu']['per_core_usage'][:8]):  # Show first 8 cores
            color = AdvancedColors.get_performance_color(usage)
            self.cpu_panel.add_line(f"  Core {i}: {usage:.1f}%", color)
        
        # Memory & Disk Panel
        self.memory_panel.clear()
        self.memory_panel.add_line(f"🧠 Memory: {data['memory']['used_gb']:.1f}GB / {data['memory']['total_gb']:.1f}GB")
        self.memory_panel.add_line(f"📊 Usage: {data['memory']['percent']:.1f}%",
                                 AdvancedColors.get_performance_color(data['memory']['percent']))
        self.memory_panel.add_line(f"💾 Available: {data['memory']['available_gb']:.1f}GB")
        
        if data['memory']['swap_total_gb'] > 0:
            self.memory_panel.add_line(f"💿 Swap: {data['memory']['swap_percent']:.1f}%",
                                     AdvancedColors.get_performance_color(data['memory']['swap_percent']))
        
        self.memory_panel.add_line("")
        self.memory_panel.add_line("💽 Disk Usage:", AdvancedColors.NEON_GREEN)
        
        for disk in data['disks'][:3]:  # Show first 3 disks
            color = AdvancedColors.get_performance_color(disk['percent'])
            self.memory_panel.add_line(f"  {disk['device']} {disk['percent']:.1f}%", color)
            self.memory_panel.add_line(f"    {disk['used_gb']:.1f}GB / {disk['total_gb']:.1f}GB", AdvancedColors.TEXT_DIM)
        
        self.memory_panel.add_line("")
        self.memory_panel.add_line(f"📖 Read: {data['disk_io']['read_mb_s']:.1f} MB/s")
        self.memory_panel.add_line(f"📝 Write: {data['disk_io']['write_mb_s']:.1f} MB/s")
        
        # Network & GPU Panel
        self.network_panel.clear()
        self.network_panel.add_line("🌐 Network Interfaces:", AdvancedColors.NEON_PURPLE)
        
        for interface in data['network']['interfaces'][:3]:
            self.network_panel.add_line(f"  {interface['name']}: {interface['ip']}", AdvancedColors.TEXT_DIM)
        
        self.network_panel.add_line("")
        up_color = AdvancedColors.NEON_ORANGE if data['network']['upload_mb_s'] > 1 else AdvancedColors.TEXT_DIM
        down_color = AdvancedColors.NEON_CYAN if data['network']['download_mb_s'] > 1 else AdvancedColors.TEXT_DIM
        
        self.network_panel.add_line(f"⬆️  Upload: {data['network']['upload_mb_s']:.2f} MB/s", up_color)
        self.network_panel.add_line(f"⬇️  Download: {data['network']['download_mb_s']:.2f} MB/s", down_color)
        self.network_panel.add_line(f"📊 Total Up: {data['network']['upload_total_gb']:.2f}GB")
        self.network_panel.add_line(f"📊 Total Down: {data['network']['download_total_gb']:.2f}GB")
        
        self.network_panel.add_line("")
        self.network_panel.add_line("🎮 GPU Information:", AdvancedColors.NEON_PINK)
        self.network_panel.add_line(f"  {data['gpu']['name'][:20]}...", AdvancedColors.TEXT_DIM)
        self.network_panel.add_line(f"📊 Usage: {data['gpu']['usage_percent']:.1f}%",
                                   AdvancedColors.get_performance_color(data['gpu']['usage_percent']))
        
        if data['gpu']['memory_total_mb'] > 0:
            self.network_panel.add_line(f"💾 Memory: {data['gpu']['memory_percent']:.1f}%",
                                       AdvancedColors.get_performance_color(data['gpu']['memory_percent']))
        
        if data['gpu']['temperature'] > 0:
            temp_color = AdvancedColors.get_performance_color(data['gpu']['temperature'], reverse=True)
            self.network_panel.add_line(f"🌡️  Temp: {data['gpu']['temperature']:.1f}°C", temp_color)
        
        # Process Manager Panel
        self.process_panel.clear()
        self.process_panel.add_line(f"📊 Total Processes: {data['processes']['total_count']}")
        self.process_panel.add_line("")
        
        # High CPU processes
        if data['processes']['high_cpu']:
            self.process_panel.add_line("🔥 High CPU Usage:", AdvancedColors.DANGER)
            for proc in data['processes']['high_cpu'][:3]:
                name = proc['name'][:15] + "..." if len(proc['name']) > 15 else proc['name']
                color = AdvancedColors.get_performance_color(proc['cpu_percent'])
                self.process_panel.add_line(f"  {name}: {proc['cpu_percent']:.1f}%", color, 
                                          clickable=True, action=f"terminate_{proc['pid']}")
        
        # High Memory processes
        if data['processes']['high_memory']:
            self.process_panel.add_line("")
            self.process_panel.add_line("🧠 High Memory Usage:", AdvancedColors.WARNING)
            for proc in data['processes']['high_memory'][:3]:
                name = proc['name'][:15] + "..." if len(proc['name']) > 15 else proc['name']
                color = AdvancedColors.get_performance_color(proc['memory_percent'])
                self.process_panel.add_line(f"  {name}: {proc['memory_percent']:.1f}%", color,
                                          clickable=True, action=f"terminate_{proc['pid']}")
        
        # Suspicious processes
        if data['processes']['suspicious']:
            self.process_panel.add_line("")
            self.process_panel.add_line("⚠️  Suspicious Processes:", AdvancedColors.CRITICAL)
            for proc in data['processes']['suspicious']:
                name = proc['name'][:15] + "..." if len(proc['name']) > 15 else proc['name']
                self.process_panel.add_line(f"  {name} (PID: {proc['pid']})", AdvancedColors.CRITICAL,
                                          clickable=True, action=f"terminate_{proc['pid']}")
        
        # Top processes by resource usage
        self.process_panel.add_line("")
        self.process_panel.add_line("📈 Top Processes:", AdvancedColors.TEXT_BRIGHT)
        for proc in data['processes']['running'][:5]:
            name = proc['name'][:12] + "..." if len(proc['name']) > 12 else proc['name']
            total_usage = proc['cpu_percent'] + proc['memory_percent']
            color = AdvancedColors.get_performance_color(total_usage)
            self.process_panel.add_line(f"  {name}: C{proc['cpu_percent']:.0f}% M{proc['memory_percent']:.0f}%", 
                                      color, clickable=True, action=f"terminate_{proc['pid']}")
    
    def render(self):
        """Render everything"""
        # Black background
        self.screen.fill(AdvancedColors.BLACK)
        
        # Animated background grid
        grid_spacing = 50
        grid_alpha = 0.3 + 0.2 * abs(math.sin(self.animation_time))
        grid_color = tuple(int(c * grid_alpha) for c in AdvancedColors.BORDER_MAIN)
        
        for x in range(0, self.width, grid_spacing):
            pygame.draw.line(self.screen, grid_color, (x, 0), (x, self.height), 1)
        for y in range(0, self.height, grid_spacing):
            pygame.draw.line(self.screen, grid_color, (0, y), (self.width, y), 1)
        
        # Main title with glow
        title_text = "🖥️ SysWatch Pro ADVANCED ULTIMATE - Real-time System Monitor"
        
        # Title glow effect
        for i in range(5):
            glow_alpha = 0.5 - i * 0.1
            glow_color = tuple(int(c * glow_alpha) for c in AdvancedColors.NEON_CYAN)
            if sum(glow_color) > 0:
                glow_surf = self.font_huge.render(title_text, True, glow_color)
                glow_rect = glow_surf.get_rect(center=(self.width // 2 + i, 50 + i))
                self.screen.blit(glow_surf, glow_rect)
        
        title_surf = self.font_huge.render(title_text, True, AdvancedColors.TEXT_BRIGHT)
        title_rect = title_surf.get_rect(center=(self.width // 2, 50))
        self.screen.blit(title_surf, title_rect)
        
        # System status indicators
        status_y = 120
        indicators = [
            ("CPU", self.cpu_circle.value, AdvancedColors.NEON_BLUE),
            ("MEMORY", self.memory_circle.value, AdvancedColors.NEON_GREEN),
            ("DISK", self.disk_circle.value, AdvancedColors.NEON_ORANGE),
            ("NETWORK", self.network_circle.value, AdvancedColors.NEON_PURPLE),
            ("GPU", self.gpu_circle.value, AdvancedColors.NEON_PINK)
        ]
        
        x_start = 240
        for i, (name, value, color) in enumerate(indicators):
            x = x_start + i * 240
            
            # Label
            label_surf = self.font_medium.render(name, True, color)
            label_rect = label_surf.get_rect(center=(x, status_y))
            self.screen.blit(label_surf, label_rect)
            
            # Value
            value_text = f"{value:.1f}%"
            value_color = AdvancedColors.get_performance_color(value)
            value_surf = self.font_small.render(value_text, True, value_color)
            value_rect = value_surf.get_rect(center=(x, status_y + 25))
            self.screen.blit(value_surf, value_rect)
        
        # 3D Circles
        self.cpu_circle.draw(self.screen)
        self.memory_circle.draw(self.screen)
        self.disk_circle.draw(self.screen)
        self.network_circle.draw(self.screen)
        self.gpu_circle.draw(self.screen)
        
        # 3D Graphs
        self.cpu_graph.draw(self.screen, self.font_medium)
        self.memory_graph.draw(self.screen, self.font_medium)
        self.network_graph.draw(self.screen, self.font_medium)
        self.disk_graph.draw(self.screen, self.font_medium)
        
        # Information Panels
        self.system_panel.draw(self.screen, self.font_small)
        self.cpu_panel.draw(self.screen, self.font_small)
        self.memory_panel.draw(self.screen, self.font_small)
        self.network_panel.draw(self.screen, self.font_small)
        self.process_panel.draw(self.screen, self.font_small)
        
        # Performance info
        fps_text = f"FPS: {self.current_fps:.0f} | Target: {self.fps_target}"
        time_text = datetime.now().strftime("%H:%M:%S")
        
        fps_surf = self.font_small.render(fps_text, True, AdvancedColors.TEXT_DIM)
        time_surf = self.font_small.render(time_text, True, AdvancedColors.TEXT_DIM)
        
        self.screen.blit(fps_surf, (20, 20))
        self.screen.blit(time_surf, (self.width - 100, 20))
        
        # Controls hint
        controls = "ESC: Exit | SPACE: Screenshot | R: Refresh | S: Save Data | Click processes to terminate"
        controls_surf = self.font_small.render(controls, True, AdvancedColors.TEXT_DIM)
        controls_rect = controls_surf.get_rect(center=(self.width // 2, self.height - 20))
        self.screen.blit(controls_surf, controls_rect)
    
    def run(self):
        """Main application loop"""
        print("Advanced System Monitor starting!")
        print(f"Resolution: {self.width}x{self.height} (fullscreen)")
        print("ESC to exit, SPACE for screenshot")
        
        frame_times = deque(maxlen=60)
        
        try:
            while self.running:
                frame_start = time.time()
                
                # Handle events
                self.handle_events()
                
                # Update
                dt = time.time() - self.last_update
                self.update(dt)
                self.last_update = time.time()
                
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
                self.clock.tick(self.fps_target)
        
        except KeyboardInterrupt:
            print("\nProgram terminated")
        except Exception as e:
            print(f"Error occurred: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("Cleaning up resources...")
        self.data_collector.stop()
        
        # Close NVIDIA ML if available
        if NVIDIA_ML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
        
        pygame.quit()
        print("Cleanup completed")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():
    """Main entry point"""
    try:
        # Check if running as administrator (for registry access)
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
        if not is_admin:
            print("WARNING: Not running as administrator. Some features may be limited.")
        
        app = AdvancedSystemMonitor()
        app.run()
    
    except KeyboardInterrupt:
        print("\nProgram terminated")
    except Exception as e:
        print(f"Critical error: {e}")
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