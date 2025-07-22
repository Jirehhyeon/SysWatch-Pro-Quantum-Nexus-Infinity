#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🖥️ SysWatch Pro SIMPLE CLEAN MONITOR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 SIMPLE & CLEAN SYSTEM MONITOR - 깔끔하고 단순한 실시간 모니터
🎯 모든 데이터가 명확하게 보이는 심플한 버전
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Copyright (C) 2025 Simple Monitor Corp
"""

import os
import sys
import time
import threading
import warnings
from datetime import datetime
from collections import deque
from typing import Dict, List, Tuple, Any
import subprocess

# Auto-install packages
def install_package(package):
    try:
        __import__(package.split('==')[0] if '==' in package else package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install required packages
install_package("psutil")
install_package("pygame-ce")

import psutil
import pygame
from pygame.locals import *

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

# Initialize Pygame
pygame.init()

# ============================================================================
# SIMPLE COLOR SCHEME
# ============================================================================
class SimpleColors:
    # Basic colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    DARK_GRAY = (40, 40, 40)
    LIGHT_GRAY = (120, 120, 120)
    
    # Status colors
    GREEN = (0, 200, 0)      # Good performance
    YELLOW = (255, 200, 0)   # Warning
    RED = (255, 50, 50)      # Critical
    BLUE = (50, 150, 255)    # Info
    CYAN = (0, 255, 255)     # Highlight
    
    # UI colors
    BACKGROUND = (15, 15, 20)
    PANEL_BG = (25, 25, 35)
    BORDER = (80, 80, 120)
    TEXT = (255, 255, 255)
    TEXT_DIM = (180, 180, 180)
    
    @staticmethod
    def get_status_color(value: float, max_val: float = 100) -> Tuple[int, int, int]:
        """Get color based on value percentage"""
        percentage = (value / max_val) * 100
        if percentage < 30:
            return SimpleColors.GREEN
        elif percentage < 70:
            return SimpleColors.YELLOW
        else:
            return SimpleColors.RED

# ============================================================================
# SYSTEM DATA COLLECTOR
# ============================================================================
class SystemDataCollector:
    def __init__(self):
        self.data = {
            'cpu_percent': 0,
            'cpu_cores': [],
            'cpu_freq': 0,
            'memory_percent': 0,
            'memory_used_gb': 0,
            'memory_total_gb': 0,
            'disk_percent': 0,
            'network_up_mbps': 0,
            'network_down_mbps': 0,
            'processes': [],
            'uptime': "0d 0h 0m"
        }
        
        # History for graphs
        self.cpu_history = deque(maxlen=100)
        self.memory_history = deque(maxlen=100)
        
        # Network tracking
        self.last_network = psutil.net_io_counters()
        self.last_time = time.time()
        
        # Start monitoring thread
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                current_time = time.time()
                dt = current_time - self.last_time
                
                # CPU data
                cpu_percents = psutil.cpu_percent(interval=0.1, percpu=True)
                self.data['cpu_percent'] = sum(cpu_percents) / len(cpu_percents)
                self.data['cpu_cores'] = cpu_percents
                
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    self.data['cpu_freq'] = cpu_freq.current
                
                # Memory data
                memory = psutil.virtual_memory()
                self.data['memory_percent'] = memory.percent
                self.data['memory_used_gb'] = memory.used / (1024**3)
                self.data['memory_total_gb'] = memory.total / (1024**3)
                
                # Disk data
                disk = psutil.disk_usage('/')
                self.data['disk_percent'] = disk.percent
                
                # Network data
                current_network = psutil.net_io_counters()
                if dt > 0:
                    bytes_sent = current_network.bytes_sent - self.last_network.bytes_sent
                    bytes_recv = current_network.bytes_recv - self.last_network.bytes_recv
                    
                    self.data['network_up_mbps'] = (bytes_sent / dt) / (1024**2)
                    self.data['network_down_mbps'] = (bytes_recv / dt) / (1024**2)
                
                self.last_network = current_network
                self.last_time = current_time
                
                # Process data
                processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                    try:
                        info = proc.info
                        if info['cpu_percent'] > 0:
                            processes.append(info)
                    except:
                        pass
                
                processes.sort(key=lambda x: x.get('cpu_percent', 0), reverse=True)
                self.data['processes'] = processes[:10]
                
                # Uptime
                boot_time = datetime.fromtimestamp(psutil.boot_time())
                uptime = datetime.now() - boot_time
                days = uptime.days
                hours = uptime.seconds // 3600
                minutes = (uptime.seconds % 3600) // 60
                self.data['uptime'] = f"{days}d {hours}h {minutes}m"
                
                # Add to history
                self.cpu_history.append(self.data['cpu_percent'])
                self.memory_history.append(self.data['memory_percent'])
                
                time.sleep(0.5)  # Update every 500ms
                
            except Exception as e:
                time.sleep(1)
    
    def get_data(self) -> Dict[str, Any]:
        """Get current system data"""
        return self.data.copy()
    
    def stop(self):
        """Stop monitoring"""
        self.running = False

# ============================================================================
# SIMPLE UI COMPONENTS
# ============================================================================
class SimplePanel:
    def __init__(self, x: int, y: int, width: int, height: int, title: str):
        self.rect = pygame.Rect(x, y, width, height)
        self.title = title
        self.content = []
    
    def clear(self):
        """Clear content"""
        self.content = []
    
    def add_line(self, text: str, color: Tuple[int, int, int] = None):
        """Add content line"""
        if color is None:
            color = SimpleColors.TEXT
        self.content.append((text, color))
    
    def draw(self, screen, font):
        """Draw panel"""
        # Background
        pygame.draw.rect(screen, SimpleColors.PANEL_BG, self.rect)
        pygame.draw.rect(screen, SimpleColors.BORDER, self.rect, 2)
        
        # Title
        title_surface = font.render(self.title, True, SimpleColors.CYAN)
        screen.blit(title_surface, (self.rect.x + 10, self.rect.y + 8))
        
        # Content
        y_offset = 35
        for text, color in self.content:
            if y_offset < self.rect.height - 20:
                text_surface = font.render(text, True, color)
                screen.blit(text_surface, (self.rect.x + 10, self.rect.y + y_offset))
                y_offset += 22

class SimpleGraph:
    def __init__(self, x: int, y: int, width: int, height: int, title: str):
        self.rect = pygame.Rect(x, y, width, height)
        self.title = title
        self.data = deque(maxlen=width - 40)
        self.max_value = 100
    
    def add_data(self, value: float):
        """Add data point"""
        self.data.append(value)
    
    def draw(self, screen, font):
        """Draw graph"""
        # Background
        pygame.draw.rect(screen, SimpleColors.PANEL_BG, self.rect)
        pygame.draw.rect(screen, SimpleColors.BORDER, self.rect, 2)
        
        # Title
        title_surface = font.render(self.title, True, SimpleColors.CYAN)
        screen.blit(title_surface, (self.rect.x + 10, self.rect.y + 8))
        
        # Current value
        if self.data:
            current = self.data[-1]
            value_text = f"{current:.1f}%"
            value_color = SimpleColors.get_status_color(current)
            value_surface = font.render(value_text, True, value_color)
            screen.blit(value_surface, (self.rect.x + self.rect.width - 80, self.rect.y + 8))
        
        # Graph area
        graph_area = pygame.Rect(self.rect.x + 20, self.rect.y + 35, 
                                self.rect.width - 40, self.rect.height - 45)
        pygame.draw.rect(screen, SimpleColors.BLACK, graph_area)
        pygame.draw.rect(screen, SimpleColors.LIGHT_GRAY, graph_area, 1)
        
        # Grid lines
        for i in range(5):
            y_pos = graph_area.y + (i * graph_area.height // 4)
            pygame.draw.line(screen, SimpleColors.DARK_GRAY,
                           (graph_area.x, y_pos),
                           (graph_area.x + graph_area.width, y_pos))
        
        # Data line
        if len(self.data) > 1:
            points = []
            for i, value in enumerate(self.data):
                x = graph_area.x + (i / max(1, len(self.data) - 1)) * graph_area.width
                y = graph_area.y + graph_area.height - (value / self.max_value) * graph_area.height
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.aalines(screen, SimpleColors.GREEN, False, points, 2)

class SimpleProgressBar:
    def __init__(self, x: int, y: int, width: int, height: int, label: str):
        self.rect = pygame.Rect(x, y, width, height)
        self.label = label
        self.value = 0
        self.max_value = 100
    
    def set_value(self, value: float, max_val: float = 100):
        """Set progress value"""
        self.value = value
        self.max_value = max_val
    
    def draw(self, screen, font):
        """Draw progress bar"""
        # Label
        label_surface = font.render(self.label, True, SimpleColors.TEXT)
        screen.blit(label_surface, (self.rect.x, self.rect.y - 20))
        
        # Background
        pygame.draw.rect(screen, SimpleColors.BLACK, self.rect)
        pygame.draw.rect(screen, SimpleColors.BORDER, self.rect, 1)
        
        # Fill
        if self.value > 0:
            fill_width = (self.value / self.max_value) * (self.rect.width - 2)
            fill_rect = pygame.Rect(self.rect.x + 1, self.rect.y + 1, 
                                  fill_width, self.rect.height - 2)
            
            fill_color = SimpleColors.get_status_color(self.value, self.max_value)
            pygame.draw.rect(screen, fill_color, fill_rect)
        
        # Value text
        value_text = f"{self.value:.1f}%"
        value_surface = font.render(value_text, True, SimpleColors.TEXT)
        text_rect = value_surface.get_rect(center=self.rect.center)
        screen.blit(value_surface, text_rect)

# ============================================================================
# MAIN APPLICATION
# ============================================================================
class SimpleMonitor:
    def __init__(self):
        # Display setup
        self.screen_info = pygame.display.Info()
        self.width = min(1400, self.screen_info.current_w)
        self.height = min(900, self.screen_info.current_h)
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("🖥️ SysWatch Pro - Simple Clean Monitor")
        
        # Performance
        self.clock = pygame.time.Clock()
        self.running = True
        self.fps = 0
        
        # Fonts
        try:
            self.font_large = pygame.font.Font(None, 32)
            self.font_medium = pygame.font.Font(None, 20)
            self.font_small = pygame.font.Font(None, 16)
        except:
            self.font_large = pygame.font.Font(None, 32)
            self.font_medium = pygame.font.Font(None, 20)
            self.font_small = pygame.font.Font(None, 16)
        
        # Data collector
        self.data_collector = SystemDataCollector()
        
        # UI Components
        self._create_ui_components()
    
    def _create_ui_components(self):
        """Create UI components"""
        margin = 20
        
        # Top row - Graphs
        graph_width = 400
        graph_height = 200
        
        self.cpu_graph = SimpleGraph(margin, 80, graph_width, graph_height, "CPU Usage")
        self.memory_graph = SimpleGraph(margin + graph_width + margin, 80, 
                                      graph_width, graph_height, "Memory Usage")
        
        # Second row - Progress bars
        bar_y = 80 + graph_height + 40
        bar_width = 250
        bar_height = 30
        bar_spacing = 300
        
        self.cpu_bar = SimpleProgressBar(margin, bar_y, bar_width, bar_height, "CPU")
        self.memory_bar = SimpleProgressBar(margin + bar_spacing, bar_y, 
                                          bar_width, bar_height, "Memory")
        self.disk_bar = SimpleProgressBar(margin + bar_spacing * 2, bar_y, 
                                        bar_width, bar_height, "Disk")
        
        # Core bars
        core_y = bar_y + 60
        self.core_bars = []
        cores_per_row = 8
        core_width = 80
        core_height = 20
        
        for i in range(16):  # Up to 16 cores
            row = i // cores_per_row
            col = i % cores_per_row
            x = margin + col * (core_width + 10)
            y = core_y + row * (core_height + 25)
            
            if y < self.height - 150:
                bar = SimpleProgressBar(x, y, core_width, core_height, f"C{i}")
                self.core_bars.append(bar)
        
        # Panels
        panel_width = 300
        panel_height = 180
        panel_y = self.height - panel_height - 20
        
        self.system_panel = SimplePanel(margin, panel_y, panel_width, panel_height, "System Info")
        self.network_panel = SimplePanel(margin + panel_width + margin, panel_y, 
                                       panel_width, panel_height, "Network")
        self.process_panel = SimplePanel(margin + (panel_width + margin) * 2, panel_y, 
                                       panel_width + 100, panel_height, "Top Processes")
    
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
                    filename = f"simple_monitor_{timestamp}.png"
                    pygame.image.save(self.screen, filename)
                    print(f"Screenshot saved: {filename}")
    
    def update(self):
        """Update data"""
        data = self.data_collector.get_data()
        
        # Update graphs
        self.cpu_graph.add_data(data['cpu_percent'])
        self.memory_graph.add_data(data['memory_percent'])
        
        # Update progress bars
        self.cpu_bar.set_value(data['cpu_percent'])
        self.memory_bar.set_value(data['memory_percent'])
        self.disk_bar.set_value(data['disk_percent'])
        
        # Update core bars
        for i, bar in enumerate(self.core_bars):
            if i < len(data['cpu_cores']):
                bar.set_value(data['cpu_cores'][i])
        
        # Update panels
        self._update_panels(data)
    
    def _update_panels(self, data: Dict[str, Any]):
        """Update panel content"""
        # System panel
        self.system_panel.clear()
        self.system_panel.add_line(f"Uptime: {data['uptime']}")
        self.system_panel.add_line(f"CPU Cores: {len(data['cpu_cores'])}")
        
        if data['cpu_freq'] > 0:
            self.system_panel.add_line(f"CPU Freq: {data['cpu_freq']:.0f} MHz")
        
        self.system_panel.add_line(f"Memory: {data['memory_used_gb']:.1f} GB")
        self.system_panel.add_line(f"Total: {data['memory_total_gb']:.1f} GB")
        self.system_panel.add_line(f"Disk Used: {data['disk_percent']:.1f}%")
        
        # Network panel
        self.network_panel.clear()
        up_color = SimpleColors.GREEN if data['network_up_mbps'] > 0.1 else SimpleColors.TEXT_DIM
        down_color = SimpleColors.GREEN if data['network_down_mbps'] > 0.1 else SimpleColors.TEXT_DIM
        
        self.network_panel.add_line(f"Upload: {data['network_up_mbps']:.2f} MB/s", up_color)
        self.network_panel.add_line(f"Download: {data['network_down_mbps']:.2f} MB/s", down_color)
        
        total_mb = data['network_up_mbps'] + data['network_down_mbps']
        self.network_panel.add_line(f"Total: {total_mb:.2f} MB/s")
        
        # Process panel
        self.process_panel.clear()
        for proc in data['processes'][:6]:
            name = proc['name'][:12] + "..." if len(proc['name']) > 12 else proc['name']
            cpu_percent = proc.get('cpu_percent', 0)
            
            color = SimpleColors.get_status_color(cpu_percent) if cpu_percent > 10 else SimpleColors.TEXT
            process_line = f"{name}: {cpu_percent:.1f}%"
            self.process_panel.add_line(process_line, color)
    
    def render(self):
        """Render everything"""
        # Clear screen
        self.screen.fill(SimpleColors.BACKGROUND)
        
        # Title
        title_text = "🖥️ SysWatch Pro - Simple Clean Monitor"
        title_surface = self.font_large.render(title_text, True, SimpleColors.CYAN)
        title_rect = title_surface.get_rect(center=(self.width // 2, 30))
        self.screen.blit(title_surface, title_rect)
        
        # FPS
        fps_text = f"FPS: {self.fps:.0f}"
        fps_surface = self.font_medium.render(fps_text, True, SimpleColors.TEXT_DIM)
        self.screen.blit(fps_surface, (self.width - 100, 10))
        
        # Time
        current_time = datetime.now().strftime("%H:%M:%S")
        time_surface = self.font_medium.render(current_time, True, SimpleColors.TEXT_DIM)
        self.screen.blit(time_surface, (20, 10))
        
        # Graphs
        self.cpu_graph.draw(self.screen, self.font_medium)
        self.memory_graph.draw(self.screen, self.font_medium)
        
        # Progress bars
        self.cpu_bar.draw(self.screen, self.font_medium)
        self.memory_bar.draw(self.screen, self.font_medium)
        self.disk_bar.draw(self.screen, self.font_medium)
        
        # Core bars
        for bar in self.core_bars:
            bar.draw(self.screen, self.font_small)
        
        # Panels
        self.system_panel.draw(self.screen, self.font_medium)
        self.network_panel.draw(self.screen, self.font_medium)
        self.process_panel.draw(self.screen, self.font_medium)
        
        # Controls
        controls_text = "ESC/Q: Exit  |  SPACE: Screenshot  |  Simple & Clean Data Visualization"
        controls_surface = self.font_small.render(controls_text, True, SimpleColors.TEXT_DIM)
        controls_rect = controls_surface.get_rect(center=(self.width // 2, self.height - 15))
        self.screen.blit(controls_surface, controls_rect)
    
    def run(self):
        """Main loop"""
        print("🖥️ Simple Clean Monitor - Starting...")
        print(f"Display: {self.width}x{self.height}")
        print("📊 Clean and simple data visualization focused!")
        
        frame_times = deque(maxlen=30)
        
        try:
            while self.running:
                frame_start = time.time()
                
                # Handle events
                self.handle_events()
                
                # Update
                self.update()
                
                # Render
                self.render()
                
                # Update display
                pygame.display.flip()
                
                # Calculate FPS
                frame_time = time.time() - frame_start
                frame_times.append(frame_time)
                
                if len(frame_times) > 0:
                    avg_frame_time = sum(frame_times) / len(frame_times)
                    self.fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                
                # Control frame rate
                self.clock.tick(60)
        
        except KeyboardInterrupt:
            print("\n⚡ Simple Monitor terminated")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup"""
        print("🧹 Cleaning up...")
        self.data_collector.stop()
        pygame.quit()
        print("✅ Cleanup complete")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():
    """Main entry point"""
    try:
        app = SimpleMonitor()
        app.run()
    except KeyboardInterrupt:
        print("\n⚡ Program terminated")
    except Exception as e:
        print(f"❌ Error: {e}")
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