# SysWatch Pro - System Monitoring Tool

A powerful, cross-platform system monitoring utility written in C, designed for real-time performance tracking and system analysis.

## Features

### Community Edition (Free)
- ✅ Real-time CPU, memory, and disk monitoring
- ✅ Process list with basic information
- ✅ Simple command-line interface
- ✅ Basic system information display
- ✅ Cross-platform support (Windows/Linux)

### Professional Edition
- 🚀 **Performance History**: Track system performance over time
- 🔔 **Smart Alerts**: Configurable threshold-based notifications
- 📊 **Data Export**: CSV, JSON, XML export formats
- ⚙️ **Background Service**: Continuous monitoring mode
- 🎯 **Process Management**: Kill, suspend, detailed process info
- 📈 **Advanced Analytics**: Trend analysis and reporting
- 🛠️ **Configuration Management**: Save/load custom settings
- 💼 **Technical Support**: Priority email support

## Quick Start

### Windows (MSVC)
```bash
# Build with Visual Studio
mkdir build
cd build
cl /I..\include ..\src\*.c /Fe:syswatch.exe psapi.lib pdh.lib

# Or use the provided Makefile with MinGW
make
```

### Windows (MinGW)
```bash
# Install dependencies
# MinGW with GCC

# Build
make
```

### Linux
```bash
# Install dependencies
sudo apt-get install build-essential

# Build
make
```

## Usage

### Basic Monitoring
```bash
# Start real-time monitoring
./syswatch

# Update every 5 seconds
./syswatch -i 5

# Run system test
./syswatch --test
```

### Professional Features
```bash
# Enable alerts and background monitoring
./syswatch -a -b

# Export performance data
./syswatch -o report.csv -f csv

# Load custom configuration
./syswatch -c config.conf

# Set custom thresholds
./syswatch --cpu-threshold 70 --memory-threshold 80
```

## Configuration

Create a `syswatch.conf` file:
```ini
# Update interval in seconds
update_interval=1

# Enable alerts (Pro only)
enable_alerts=true

# Alert thresholds (Pro only)
cpu_threshold=80.0
memory_threshold=85.0
disk_threshold=90.0

# Log file (Pro only)
log_file=syswatch.log
```

## System Requirements

### Windows
- Windows 7 or later
- Visual Studio 2019+ or MinGW-w64
- Administrator privileges for some features

### Linux
- Linux kernel 2.6+
- GCC 4.8+
- Root privileges for some features

## Professional Upgrade

Upgrade to Professional edition for advanced features:
- 📧 **Email**: pro@syswatch-tools.com
- 🌐 **Website**: https://syswatch-pro.com
- 💰 **Pricing**: $29.99 one-time purchase

### License Benefits
- ✅ Lifetime license (no subscription)
- ✅ Free updates for 1 year
- ✅ Priority technical support
- ✅ Commercial use allowed
- ✅ Source code access (additional fee)

## Architecture

```
SysWatch Pro/
├── src/
│   ├── main.c              # Main application and CLI
│   ├── system_info.c       # System information collection
│   ├── process_monitor.c   # Process monitoring and management
│   └── utils.c             # Utilities and formatting
├── include/
│   └── syswatch.h          # Public API and structures
├── build/                  # Build artifacts
├── docs/                   # Documentation
└── Makefile               # Build configuration
```

## API Documentation

### Core Functions
```c
// Initialize monitoring system
int init_monitor(monitor_config_t *config);

// Get current system information
int get_system_info(system_info_t *info);

// Get process list
int get_process_list(process_info_t *processes, int *count, int max_count);

// Cleanup resources
int cleanup_monitor(void);
```

### Professional Functions
```c
// Export performance data (Pro only)
int export_performance_data(const char *filename, const char *format);

// Set alert thresholds (Pro only)
int set_alert_thresholds(double cpu, double memory, double disk);

// Background monitoring (Pro only)
int start_background_monitoring(void);
int stop_background_monitoring(void);
```

## Performance

- **Memory Usage**: < 10MB resident memory
- **CPU Overhead**: < 1% on modern systems
- **Update Frequency**: 1-60 seconds configurable
- **Process Capacity**: Up to 1000 processes tracked
- **History Storage**: 60 data points (Pro edition)

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup
```bash
git clone https://github.com/syswatch-pro/syswatch.git
cd syswatch
make debug
./build/syswatch --test
```

## Support

### Community Support
- 📧 **Email**: support@syswatch-tools.com
- 💬 **Forum**: https://forum.syswatch-pro.com
- 🐛 **Issues**: https://github.com/syswatch-pro/syswatch/issues

### Professional Support
- 🎯 **Priority Email**: pro-support@syswatch-tools.com
- 📞 **Phone**: +1-555-SYSWATCH
- ⏰ **Response Time**: 24 hours guaranteed

## License

### Community Edition
GNU General Public License v3.0 - see LICENSE file

### Professional Edition
Commercial license - contact sales@syswatch-tools.com

## Roadmap

### Version 1.1 (Q2 2025)
- [ ] Web dashboard interface
- [ ] Network monitoring
- [ ] GPU usage tracking
- [ ] Docker container monitoring

### Version 1.2 (Q3 2025)
- [ ] Historical trending
- [ ] Email/SMS alerts
- [ ] REST API
- [ ] Plugin system

### Version 2.0 (Q4 2025)
- [ ] Machine learning anomaly detection
- [ ] Predictive alerts
- [ ] Multi-server monitoring
- [ ] Cloud deployment options

## Changelog

### v1.0.0 (2025-01-20)
- ✅ Initial release
- ✅ Basic system monitoring
- ✅ Process management
- ✅ Professional licensing system
- ✅ Cross-platform support
- ✅ Export functionality

---

**SysWatch Pro** - Professional System Monitoring Made Simple

Copyright (C) 2025 SysWatch Technologies. All rights reserved.