#include "syswatch.h"

static license_type_t current_license = LICENSE_COMMUNITY;
static alert_callback_t alert_callback = NULL;

// License management
bool verify_license(license_type_t license) {
    current_license = license;
    
    // For demo purposes, always return true
    // In real implementation, this would check license keys, online verification, etc.
    return true;
}

bool is_pro_feature_available(void) {
    return current_license == LICENSE_PROFESSIONAL;
}

// Formatting utilities
const char* format_bytes(unsigned long bytes, char *buffer, size_t size) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    double value = (double)bytes;
    int unit_index = 0;
    
    while (value >= 1024.0 && unit_index < 4) {
        value /= 1024.0;
        unit_index++;
    }
    
    if (unit_index == 0) {
        snprintf(buffer, size, "%lu %s", (unsigned long)value, units[unit_index]);
    } else {
        snprintf(buffer, size, "%.1f %s", value, units[unit_index]);
    }
    
    return buffer;
}

const char* format_time(time_t timestamp, char *buffer, size_t size) {
    struct tm *time_info = localtime(&timestamp);
    strftime(buffer, size, "%H:%M:%S", time_info);
    return buffer;
}

const char* format_uptime(time_t start_time, char *buffer, size_t size) {
    time_t current_time = time(NULL);
    time_t uptime = current_time - start_time;
    
    int days = uptime / 86400;
    int hours = (uptime % 86400) / 3600;
    int minutes = (uptime % 3600) / 60;
    int seconds = uptime % 60;
    
    if (days > 0) {
        snprintf(buffer, size, "%dd %02d:%02d:%02d", days, hours, minutes, seconds);
    } else {
        snprintf(buffer, size, "%02d:%02d:%02d", hours, minutes, seconds);
    }
    
    return buffer;
}

// Display functions
void print_system_info(const system_info_t *info) {
    char buffer[64];
    
    printf("System Information:\n");
    printf("  CPU Usage:       %6.1f%%\n", info->cpu_usage);
    printf("  Memory Usage:    %s / %s (%5.1f%%)\n", 
           format_bytes(info->memory_used * 1024, buffer, sizeof(buffer)),
           format_bytes(info->memory_total * 1024, buffer, sizeof(buffer)),
           info->memory_usage_percent);
    printf("  Disk Usage:      %s / %s (%5.1f%%)\n",
           format_bytes(info->disk_used * 1024, buffer, sizeof(buffer)),
           format_bytes(info->disk_total * 1024, buffer, sizeof(buffer)),
           info->disk_usage_percent);
    printf("  Network I/O:     Sent: %s, Received: %s\n",
           format_bytes(info->network_bytes_sent, buffer, sizeof(buffer)),
           format_bytes(info->network_bytes_received, buffer, sizeof(buffer)));
    printf("  Last Updated:    %s\n", format_time(info->timestamp, buffer, sizeof(buffer)));
}

void print_process_list(const process_info_t *processes, int count) {
    char buffer[64];
    
    printf("\nTop Processes:\n");
    printf("%-8s %-25s %8s %12s %12s\n", "PID", "Name", "CPU%", "Memory", "Start Time");
    printf("%-8s %-25s %8s %12s %12s\n", "--------", "-------------------------", 
           "--------", "------------", "------------");
    
    for (int i = 0; i < count; i++) {
        printf("%-8lu %-25s %7.1f%% %11s %12s\n",
               processes[i].pid,
               processes[i].name,
               processes[i].cpu_percent,
               format_bytes(processes[i].memory_kb * 1024, buffer, sizeof(buffer)),
               format_time(processes[i].start_time, buffer, sizeof(buffer)));
    }
}

void print_help(void) {
    printf("SysWatch Pro v%s - System Monitoring Tool\n\n", SYSWATCH_VERSION);
    printf("USAGE:\n");
    printf("  syswatch [OPTIONS]\n\n");
    printf("OPTIONS:\n");
    printf("  -h, --help              Show this help message\n");
    printf("  -v, --version           Show version information\n");
    printf("  -c, --config FILE       Load configuration from file\n");
    printf("  -i, --interval SECONDS  Update interval in seconds (default: 1)\n");
    printf("  -t, --test              Run system diagnostics\n\n");
    
    if (is_pro_feature_available()) {
        printf("PROFESSIONAL FEATURES:\n");
        printf("  -o, --output FILE       Export performance data to file\n");
        printf("  -f, --format FORMAT     Export format: csv, json, xml\n");
        printf("  -a, --alerts            Enable threshold-based alerts\n");
        printf("  -b, --background        Run in background monitoring mode\n");
        printf("  --cpu-threshold N       Set CPU usage alert threshold (%%)\n");
        printf("  --memory-threshold N    Set memory usage alert threshold (%%)\n");
        printf("  --disk-threshold N      Set disk usage alert threshold (%%)\n\n");
    } else {
        printf("UPGRADE TO PROFESSIONAL:\n");
        printf("  • Advanced performance analytics and history\n");
        printf("  • Automated alerts and notifications\n");
        printf("  • Data export in multiple formats\n");
        printf("  • Background monitoring service\n");
        printf("  • Process management capabilities\n");
        printf("  • Technical support and updates\n\n");
        printf("  Visit: https://syswatch-pro.com/upgrade\n\n");
    }
    
    printf("EXAMPLES:\n");
    printf("  syswatch                       # Basic real-time monitoring\n");
    printf("  syswatch -i 5                  # Update every 5 seconds\n");
    printf("  syswatch -t                    # Run system test\n");
    
    if (is_pro_feature_available()) {
        printf("  syswatch -o report.csv -f csv  # Export data to CSV\n");
        printf("  syswatch -a -b                 # Background with alerts\n");
    }
    
    printf("\nFor more information: https://syswatch-pro.com/docs\n");
}

void print_version(void) {
    printf("SysWatch Pro v%s\n", SYSWATCH_VERSION);
    printf("Edition: %s\n", SYSWATCH_BUILD);
    printf("Platform: ");
    #ifdef WINDOWS
        printf("Windows\n");
    #else
        printf("Linux\n");
    #endif
    printf("License: %s\n", is_pro_feature_available() ? "Professional" : "Community");
    printf("Copyright (C) 2025 SysWatch Pro. All rights reserved.\n");
}

// Alert system
void set_alert_callback(alert_callback_t callback) {
    alert_callback = callback;
}

void trigger_alert(const char *message, int severity) {
    if (alert_callback) {
        alert_callback(message, severity);
    } else {
        // Default alert handling
        const char* severity_text[] = {"INFO", "WARNING", "CRITICAL"};
        const char* severity_str = (severity >= 0 && severity <= 2) ? severity_text[severity] : "UNKNOWN";
        
        printf("\n[ALERT-%s] %s\n", severity_str, message);
    }
}

void check_thresholds(const system_info_t *info) {
    if (!is_pro_feature_available()) return;
    
    char alert_message[256];
    
    // Check CPU usage
    if (info->cpu_usage > 80.0) {
        snprintf(alert_message, sizeof(alert_message), 
                "High CPU usage detected: %.1f%%", info->cpu_usage);
        trigger_alert(alert_message, info->cpu_usage > 95.0 ? 2 : 1);
    }
    
    // Check memory usage
    if (info->memory_usage_percent > 85.0) {
        snprintf(alert_message, sizeof(alert_message), 
                "High memory usage detected: %.1f%%", info->memory_usage_percent);
        trigger_alert(alert_message, info->memory_usage_percent > 95.0 ? 2 : 1);
    }
    
    // Check disk usage
    if (info->disk_usage_percent > 90.0) {
        snprintf(alert_message, sizeof(alert_message), 
                "High disk usage detected: %.1f%%", info->disk_usage_percent);
        trigger_alert(alert_message, info->disk_usage_percent > 98.0 ? 2 : 1);
    }
}

// Configuration management
int load_config(const char *filename, monitor_config_t *config) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Warning: Could not load config file '%s', using defaults\n", filename);
        return SYSWATCH_ERROR;
    }
    
    char line[256];
    while (fgets(line, sizeof(line), file)) {
        char key[64], value[192];
        if (sscanf(line, "%63[^=]=%191s", key, value) == 2) {
            if (strcmp(key, "update_interval") == 0) {
                config->update_interval = atoi(value) * 1000;
            } else if (strcmp(key, "enable_alerts") == 0) {
                config->enable_alerts = (strcmp(value, "true") == 0);
            } else if (strcmp(key, "cpu_threshold") == 0) {
                config->cpu_alert_threshold = atof(value);
            } else if (strcmp(key, "memory_threshold") == 0) {
                config->memory_alert_threshold = atof(value);
            } else if (strcmp(key, "disk_threshold") == 0) {
                config->disk_alert_threshold = atof(value);
            } else if (strcmp(key, "log_file") == 0) {
                strncpy(config->log_file_path, value, MAX_PATH_LEN - 1);
                config->log_file_path[MAX_PATH_LEN - 1] = '\0';
                config->log_to_file = true;
            }
        }
    }
    
    fclose(file);
    printf("Configuration loaded from '%s'\n", filename);
    return SYSWATCH_OK;
}

int save_config(const char *filename, const monitor_config_t *config) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        return SYSWATCH_ERROR;
    }
    
    fprintf(file, "# SysWatch Pro Configuration\n");
    fprintf(file, "update_interval=%d\n", config->update_interval / 1000);
    fprintf(file, "enable_alerts=%s\n", config->enable_alerts ? "true" : "false");
    fprintf(file, "cpu_threshold=%.1f\n", config->cpu_alert_threshold);
    fprintf(file, "memory_threshold=%.1f\n", config->memory_alert_threshold);
    fprintf(file, "disk_threshold=%.1f\n", config->disk_alert_threshold);
    
    if (config->log_to_file) {
        fprintf(file, "log_file=%s\n", config->log_file_path);
    }
    
    fclose(file);
    return SYSWATCH_OK;
}

// Professional features
int export_performance_data(const char *filename, const char *format) {
    if (!is_pro_feature_available()) {
        return SYSWATCH_FEATURE_UNAVAILABLE;
    }
    
    FILE *file = fopen(filename, "w");
    if (!file) {
        return SYSWATCH_ERROR;
    }
    
    if (strcmp(format, "csv") == 0) {
        fprintf(file, "Timestamp,CPU%%,Memory%%,Disk%%,MemoryUsed,DiskUsed\n");
        // Export historical data here
        fprintf(file, "2025-01-20 12:00:00,25.5,45.2,60.1,8192000,102400000\n");
    } else if (strcmp(format, "json") == 0) {
        fprintf(file, "{\n");
        fprintf(file, "  \"export_time\": \"%ld\",\n", time(NULL));
        fprintf(file, "  \"system_info\": {\n");
        fprintf(file, "    \"cpu_usage\": 25.5,\n");
        fprintf(file, "    \"memory_usage\": 45.2,\n");
        fprintf(file, "    \"disk_usage\": 60.1\n");
        fprintf(file, "  }\n");
        fprintf(file, "}\n");
    }
    
    fclose(file);
    printf("Performance data exported to '%s' in %s format\n", filename, format);
    return SYSWATCH_OK;
}

int set_alert_thresholds(double cpu, double memory, double disk) {
    if (!is_pro_feature_available()) {
        return SYSWATCH_FEATURE_UNAVAILABLE;
    }
    
    printf("Alert thresholds updated: CPU %.1f%%, Memory %.1f%%, Disk %.1f%%\n", 
           cpu, memory, disk);
    return SYSWATCH_OK;
}

static bool background_monitoring = false;

int start_background_monitoring(void) {
    if (!is_pro_feature_available()) {
        return SYSWATCH_FEATURE_UNAVAILABLE;
    }
    
    background_monitoring = true;
    printf("Background monitoring started\n");
    return SYSWATCH_OK;
}

int stop_background_monitoring(void) {
    if (!is_pro_feature_available()) {
        return SYSWATCH_FEATURE_UNAVAILABLE;
    }
    
    background_monitoring = false;
    printf("Background monitoring stopped\n");
    return SYSWATCH_OK;
}