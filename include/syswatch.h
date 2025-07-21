#ifndef SYSWATCH_H
#define SYSWATCH_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>

#ifdef WINDOWS
    #include <windows.h>
    #include <psapi.h>
    #include <pdh.h>
    #include <winternl.h>
    #include <ntstatus.h>
    #include <iphlpapi.h>
    #include <setupapi.h>
    #include <devguid.h>
    #include <wbemidl.h>
    #pragma comment(lib, "psapi.lib")
    #pragma comment(lib, "pdh.lib")
    #pragma comment(lib, "iphlpapi.lib")
    #pragma comment(lib, "setupapi.lib")
    #pragma comment(lib, "wbemuuid.lib")
    #pragma comment(lib, "ole32.lib")
    #pragma comment(lib, "oleaut32.lib")
#else
    #include <unistd.h>
    #include <sys/sysinfo.h>
    #include <sys/statvfs.h>
    #include <sys/types.h>
    #include <sys/stat.h>
    #include <sys/utsname.h>
    #include <sys/resource.h>
    #include <sys/ioctl.h>
    #include <sys/socket.h>
    #include <net/if.h>
    #include <ifaddrs.h>
    #include <pthread.h>
    #include <dirent.h>
    #include <fcntl.h>
    #include <linux/netlink.h>
    #include <linux/rtnetlink.h>
#endif

// Version information - AAA Edition
#define SYSWATCH_VERSION "2.0.0"
#define SYSWATCH_BUILD "AAA Enterprise"
#define SYSWATCH_CODENAME "Prometheus"
#define SYSWATCH_COPYRIGHT "© 2025 SysWatch Technologies Ltd."

// Configuration constants - AAA Grade
#define MAX_PROCESSES 10000
#define MAX_CPU_CORES 256
#define MAX_DRIVES 64
#define MAX_NETWORK_INTERFACES 32
#define MAX_GPU_DEVICES 8
#define MAX_PATH_LEN 2048
#define MAX_NAME_LEN 512
#define MAX_DESCRIPTION_LEN 1024
#define UPDATE_INTERVAL 100   // milliseconds for ultra-responsiveness
#define HISTORY_SIZE 3600     // 1 hour of history
#define ALERT_HISTORY_SIZE 1000
#define MAX_CONCURRENT_THREADS 32
#define PERFORMANCE_BUFFER_SIZE 65536

// License types - Enhanced
typedef enum {
    LICENSE_COMMUNITY = 0,
    LICENSE_PROFESSIONAL = 1,
    LICENSE_ENTERPRISE = 2,
    LICENSE_DEVELOPER = 3,
    LICENSE_OEM = 4
} license_type_t;

// Performance monitoring levels
typedef enum {
    MONITORING_BASIC = 0,
    MONITORING_STANDARD = 1,
    MONITORING_ADVANCED = 2,
    MONITORING_ENTERPRISE = 3,
    MONITORING_REALTIME = 4
} monitoring_level_t;

// Alert severity levels
typedef enum {
    ALERT_INFO = 0,
    ALERT_WARNING = 1,
    ALERT_CRITICAL = 2,
    ALERT_EMERGENCY = 3
} alert_severity_t;

// System component types
typedef enum {
    COMPONENT_CPU = 0,
    COMPONENT_MEMORY = 1,
    COMPONENT_DISK = 2,
    COMPONENT_NETWORK = 3,
    COMPONENT_GPU = 4,
    COMPONENT_THERMAL = 5,
    COMPONENT_POWER = 6,
    COMPONENT_PROCESS = 7
} component_type_t;

// Advanced CPU information
typedef struct {
    double total_usage;
    double per_core_usage[MAX_CPU_CORES];
    uint32_t core_count;
    uint32_t logical_count;
    uint32_t thread_count;
    double base_frequency;    // MHz
    double current_frequency; // MHz
    double max_frequency;     // MHz
    double temperature;       // Celsius
    double voltage;          // Volts
    uint64_t cache_l1_size;
    uint64_t cache_l2_size;
    uint64_t cache_l3_size;
    char vendor[64];
    char model[128];
    char architecture[32];
    uint32_t feature_flags;
    double idle_time_percent;
    double kernel_time_percent;
    double user_time_percent;
    uint64_t context_switches;
    uint64_t interrupts;
} cpu_info_t;

// Advanced Memory information
typedef struct {
    uint64_t total_physical;
    uint64_t available_physical;
    uint64_t used_physical;
    uint64_t total_virtual;
    uint64_t available_virtual;
    uint64_t used_virtual;
    uint64_t total_swap;
    uint64_t used_swap;
    uint64_t cached;
    uint64_t buffers;
    uint64_t shared;
    uint64_t committed;
    uint64_t commit_limit;
    uint64_t pool_paged;
    uint64_t pool_nonpaged;
    uint64_t system_cache;
    uint64_t kernel_total;
    uint64_t kernel_paged;
    uint64_t kernel_nonpaged;
    double compression_ratio;
    uint32_t page_faults_per_sec;
    uint32_t memory_speed;    // MHz
    char memory_type[32];     // DDR4, DDR5, etc.
} memory_info_t;

// Advanced Disk information
typedef struct {
    char device_name[MAX_NAME_LEN];
    char mount_point[MAX_PATH_LEN];
    char filesystem[64];
    char model[128];
    char serial_number[64];
    uint64_t total_size;
    uint64_t used_size;
    uint64_t available_size;
    double usage_percent;
    uint64_t read_bytes;
    uint64_t write_bytes;
    uint32_t read_iops;
    uint32_t write_iops;
    double read_latency;      // ms
    double write_latency;     // ms
    double queue_depth;
    double temperature;       // Celsius
    uint32_t power_on_hours;
    uint32_t write_cycles;
    bool is_ssd;
    bool is_nvme;
    bool is_removable;
    uint32_t sector_size;
    uint64_t sectors_total;
} disk_info_t;

// Advanced Network information
typedef struct {
    char interface_name[MAX_NAME_LEN];
    char description[MAX_DESCRIPTION_LEN];
    char mac_address[18];
    char ip_address[46];      // IPv6 compatible
    uint64_t bytes_sent;
    uint64_t bytes_received;
    uint64_t packets_sent;
    uint64_t packets_received;
    uint64_t errors_in;
    uint64_t errors_out;
    uint64_t drops_in;
    uint64_t drops_out;
    uint32_t speed;           // Mbps
    double utilization_percent;
    double latency;           // ms
    double packet_loss;
    bool is_connected;
    bool is_wireless;
    int signal_strength;      // dBm for wireless
    char connection_type[32]; // Ethernet, WiFi, etc.
} network_interface_t;

// GPU information
typedef struct {
    char name[128];
    char vendor[64];
    char driver_version[64];
    uint64_t memory_total;
    uint64_t memory_used;
    uint64_t memory_free;
    double gpu_usage;
    double memory_usage;
    double temperature;
    double power_draw;        // Watts
    uint32_t core_clock;      // MHz
    uint32_t memory_clock;    // MHz
    uint32_t fan_speed;       // RPM
    bool is_dedicated;
} gpu_info_t;

// Thermal sensor information
typedef struct {
    char sensor_name[MAX_NAME_LEN];
    char sensor_type[32];
    double temperature;
    double critical_temp;
    double max_temp;
    double min_temp;
    bool is_critical;
} thermal_sensor_t;

// Power information
typedef struct {
    bool on_battery;
    double battery_percent;
    uint32_t battery_time_remaining; // minutes
    double power_consumption;        // Watts
    char power_plan[64];
    bool power_saver_mode;
} power_info_t;

// Comprehensive System information structure
typedef struct {
    cpu_info_t cpu;
    memory_info_t memory;
    disk_info_t disks[MAX_DRIVES];
    uint32_t disk_count;
    network_interface_t networks[MAX_NETWORK_INTERFACES];
    uint32_t network_count;
    gpu_info_t gpus[MAX_GPU_DEVICES];
    uint32_t gpu_count;
    thermal_sensor_t thermal_sensors[32];
    uint32_t thermal_count;
    power_info_t power;
    char hostname[MAX_NAME_LEN];
    char os_name[64];
    char os_version[64];
    char kernel_version[64];
    uint64_t uptime_seconds;
    uint32_t process_count;
    uint32_t thread_count;
    uint32_t handle_count;
    time_t timestamp;
    uint64_t collection_time_us; // Microseconds for performance tracking
} system_info_t;

// Enhanced Process information structure
typedef struct {
    uint32_t pid;
    uint32_t ppid;                     // Parent PID
    char name[MAX_NAME_LEN];
    char command_line[MAX_PATH_LEN];
    char executable_path[MAX_PATH_LEN];
    char working_directory[MAX_PATH_LEN];
    char username[128];
    char session_name[64];
    uint32_t session_id;
    uint32_t priority;
    uint32_t nice_value;
    double cpu_percent;
    double cpu_time_total;
    double cpu_time_user;
    double cpu_time_kernel;
    uint64_t memory_rss;               // Resident Set Size
    uint64_t memory_vms;               // Virtual Memory Size
    uint64_t memory_shared;
    uint64_t memory_private;
    uint64_t memory_working_set;
    uint64_t memory_peak_working_set;
    uint64_t memory_commit;
    uint64_t memory_peak_commit;
    uint32_t thread_count;
    uint32_t handle_count;
    uint32_t gdi_objects;
    uint32_t user_objects;
    uint64_t io_read_bytes;
    uint64_t io_write_bytes;
    uint64_t io_read_ops;
    uint64_t io_write_ops;
    uint64_t network_bytes_sent;
    uint64_t network_bytes_received;
    time_t start_time;
    uint64_t runtime_seconds;
    char status[32];                   // Running, Sleeping, etc.
    int exit_code;
    bool is_64bit;
    bool is_elevated;
    bool is_system_process;
    bool is_service;
    bool has_window;
    char integrity_level[32];
    char security_context[256];
    uint32_t affinity_mask;
    double environmental_impact;       // Power consumption estimation
} process_info_t;

// Advanced Monitoring configuration
typedef struct {
    license_type_t license;
    monitoring_level_t monitoring_level;
    
    // Performance settings
    uint32_t update_interval_ms;
    uint32_t history_retention_hours;
    uint32_t max_processes_tracked;
    bool enable_realtime_mode;
    bool enable_high_precision;
    
    // Alert configuration
    bool enable_alerts;
    alert_severity_t min_alert_level;
    double cpu_alert_threshold;
    double memory_alert_threshold;
    double disk_alert_threshold;
    double network_alert_threshold;
    double temperature_alert_threshold;
    double power_alert_threshold;
    
    // Advanced alert settings
    bool enable_predictive_alerts;
    bool enable_smart_thresholds;
    uint32_t alert_cooldown_seconds;
    bool enable_alert_aggregation;
    
    // Logging configuration
    bool log_to_file;
    bool log_to_syslog;
    bool log_to_eventlog;
    char log_file_path[MAX_PATH_LEN];
    char log_level[16];
    uint64_t max_log_size;
    uint32_t log_rotation_count;
    
    // Data collection
    bool collect_cpu_detailed;
    bool collect_memory_detailed;
    bool collect_disk_detailed;
    bool collect_network_detailed;
    bool collect_gpu_info;
    bool collect_thermal_info;
    bool collect_power_info;
    bool collect_process_detailed;
    
    // Security settings
    bool enable_integrity_monitoring;
    bool enable_anomaly_detection;
    bool enable_security_scanning;
    char trusted_processes[1024];
    
    // Export settings
    bool auto_export_enabled;
    char export_formats[256];          // csv,json,xml
    char export_directory[MAX_PATH_LEN];
    uint32_t export_interval_hours;
    
    // Network settings
    bool enable_remote_monitoring;
    uint16_t remote_port;
    char remote_interface[64];
    bool enable_encryption;
    char api_key[256];
    
    // Performance tuning
    uint32_t thread_pool_size;
    uint32_t memory_pool_size;
    bool enable_memory_compression;
    bool enable_cpu_optimization;
    
    // UI settings
    bool enable_gui;
    bool enable_web_interface;
    bool enable_notifications;
    char theme[32];
    char language[8];
} monitor_config_t;

// Advanced Performance history with ring buffer
typedef struct {
    system_info_t *data;
    uint32_t capacity;
    uint32_t size;
    uint32_t head;
    uint32_t tail;
    uint64_t total_samples;
    time_t start_time;
    time_t last_update;
    bool is_full;
    pthread_mutex_t mutex;
} performance_history_t;

// Performance analytics
typedef struct {
    double cpu_avg;
    double cpu_min;
    double cpu_max;
    double cpu_std_dev;
    double memory_avg;
    double memory_min;
    double memory_max;
    double memory_std_dev;
    double disk_avg_usage;
    double disk_avg_iops;
    double network_avg_utilization;
    uint32_t alert_count;
    uint32_t critical_alert_count;
    time_t analysis_period;
    char bottleneck_component[64];
    char performance_grade[8];         // A+, A, B, C, D, F
    double efficiency_score;           // 0-100
} performance_analytics_t;

// Alert information
typedef struct {
    uint32_t alert_id;
    time_t timestamp;
    alert_severity_t severity;
    component_type_t component;
    char title[256];
    char description[1024];
    double trigger_value;
    double threshold_value;
    char suggested_action[512];
    bool is_acknowledged;
    bool is_resolved;
    uint32_t occurrence_count;
} alert_info_t;

// System health status
typedef struct {
    char overall_status[32];           // Excellent, Good, Fair, Poor, Critical
    double health_score;               // 0-100
    double cpu_health;
    double memory_health;
    double disk_health;
    double network_health;
    double thermal_health;
    uint32_t active_alerts;
    uint32_t warning_count;
    uint32_t critical_count;
    time_t last_assessment;
    char recommendations[2048];
} system_health_t;

// Core monitoring functions - Enhanced
int init_monitor(monitor_config_t *config);
int cleanup_monitor(void);
int get_system_info(system_info_t *info);
int get_cpu_info(cpu_info_t *cpu);
int get_memory_info(memory_info_t *memory);
int get_disk_info(disk_info_t *disks, uint32_t *count, uint32_t max_count);
int get_network_info(network_interface_t *networks, uint32_t *count, uint32_t max_count);
int get_gpu_info(gpu_info_t *gpus, uint32_t *count, uint32_t max_count);
int get_thermal_info(thermal_sensor_t *sensors, uint32_t *count, uint32_t max_count);
int get_power_info(power_info_t *power);
int get_process_list(process_info_t *processes, uint32_t *count, uint32_t max_count);
int get_process_detailed(uint32_t pid, process_info_t *process);
int get_system_health(system_health_t *health);

// Professional & Enterprise features
int export_performance_data(const char *filename, const char *format);
int export_system_report(const char *filename, const char *format);
int set_alert_thresholds(double cpu, double memory, double disk, double network, double temperature);
int start_background_monitoring(void);
int stop_background_monitoring(void);
int start_realtime_monitoring(void);
int stop_realtime_monitoring(void);
int enable_predictive_analytics(bool enable);
int configure_auto_optimization(bool enable);
int schedule_maintenance_tasks(void);
int backup_configuration(const char *filename);
int restore_configuration(const char *filename);

// Performance history management
int init_performance_history(performance_history_t *history, uint32_t capacity);
int add_performance_sample(performance_history_t *history, const system_info_t *sample);
int get_performance_analytics(const performance_history_t *history, performance_analytics_t *analytics);
int cleanup_performance_history(performance_history_t *history);

// Alert system - Enhanced
int init_alert_system(void);
int add_alert(alert_severity_t severity, component_type_t component, const char *title, const char *description, double value, double threshold);
int get_active_alerts(alert_info_t *alerts, uint32_t *count, uint32_t max_count);
int acknowledge_alert(uint32_t alert_id);
int resolve_alert(uint32_t alert_id);
int clear_all_alerts(void);
int cleanup_alert_system(void);

// Security & Integrity
int verify_system_integrity(void);
int scan_for_anomalies(void);
int check_process_whitelist(const process_info_t *process);
int enable_real_time_protection(bool enable);
int get_security_status(char *status, size_t size);

// Optimization & Auto-tuning
int analyze_performance_bottlenecks(char *analysis, size_t size);
int optimize_system_performance(void);
int defragment_memory(void);
int clean_temporary_files(void);
int optimize_process_priorities(void);
int balance_cpu_load(void);

// Remote monitoring & API
int start_web_server(uint16_t port);
int stop_web_server(void);
int enable_remote_access(const char *api_key);
int disable_remote_access(void);
int send_telemetry_data(const char *endpoint);
int receive_remote_commands(void);

// Machine Learning & Prediction
int init_ml_engine(void);
int train_performance_model(void);
int predict_system_behavior(uint32_t minutes_ahead, system_info_t *prediction);
int detect_performance_patterns(void);
int classify_system_workload(char *workload_type, size_t size);
int cleanup_ml_engine(void);

// Enhanced Utility functions
void print_system_info(const system_info_t *info);
void print_cpu_info(const cpu_info_t *cpu);
void print_memory_info(const memory_info_t *memory);
void print_disk_info(const disk_info_t *disks, uint32_t count);
void print_network_info(const network_interface_t *networks, uint32_t count);
void print_gpu_info(const gpu_info_t *gpus, uint32_t count);
void print_process_list(const process_info_t *processes, uint32_t count);
void print_performance_analytics(const performance_analytics_t *analytics);
void print_system_health(const system_health_t *health);
void print_help(void);
void print_version(void);
void print_license_info(void);
void print_system_banner(void);

// Enhanced formatting functions
const char* format_bytes(uint64_t bytes, char *buffer, size_t size, bool use_binary);
const char* format_bytes_per_second(uint64_t bytes_per_sec, char *buffer, size_t size);
const char* format_frequency(double freq_mhz, char *buffer, size_t size);
const char* format_temperature(double temp_celsius, char *buffer, size_t size, bool use_fahrenheit);
const char* format_percentage(double percent, char *buffer, size_t size, int precision);
const char* format_time_duration(uint64_t seconds, char *buffer, size_t size);
const char* format_time(time_t timestamp, char *buffer, size_t size);
const char* format_timestamp_precise(uint64_t microseconds, char *buffer, size_t size);
const char* format_iops(uint32_t iops, char *buffer, size_t size);
const char* format_latency(double latency_ms, char *buffer, size_t size);

// Color and UI utilities
const char* get_usage_color(double percentage);
const char* get_health_color(double health_score);
const char* get_alert_color(alert_severity_t severity);
void print_progress_bar(double percentage, int width, const char* label);
void print_horizontal_line(int width, char character);
void clear_screen(void);
void move_cursor(int row, int col);
void set_console_color(int color_code);
void reset_console_color(void);

// String and data utilities
int safe_string_copy(char *dest, const char *src, size_t dest_size);
int safe_string_append(char *dest, const char *src, size_t dest_size);
uint64_t get_current_time_microseconds(void);
double calculate_standard_deviation(const double *values, uint32_t count);
double calculate_moving_average(const double *values, uint32_t count, uint32_t window_size);
int parse_size_string(const char *size_str, uint64_t *result);
int parse_time_string(const char *time_str, uint32_t *seconds);
bool is_valid_ip_address(const char *ip);
bool is_valid_mac_address(const char *mac);

// Enhanced License verification & Management
bool verify_license(license_type_t license);
bool is_pro_feature_available(void);
bool is_enterprise_feature_available(void);
bool is_developer_feature_available(void);
int validate_license_key(const char *license_key, license_type_t *license_type);
int activate_license(const char *license_key, const char *activation_code);
int deactivate_license(void);
int get_license_info(char *info, size_t size);
int check_license_expiration(time_t *expiry_date);
bool is_trial_version(void);
int get_trial_days_remaining(void);
int register_installation(const char *user_info);
int send_license_telemetry(void);

// Enhanced Alert system & Callbacks
typedef void (*alert_callback_t)(const alert_info_t *alert);
typedef void (*performance_callback_t)(const system_info_t *info);
typedef void (*anomaly_callback_t)(const char *anomaly_description, double severity_score);
typedef void (*security_callback_t)(const char *security_event, const char *details);

void set_alert_callback(alert_callback_t callback);
void set_performance_callback(performance_callback_t callback);
void set_anomaly_callback(anomaly_callback_t callback);
void set_security_callback(security_callback_t callback);

void check_all_thresholds(const system_info_t *info);
void check_cpu_thresholds(const cpu_info_t *cpu);
void check_memory_thresholds(const memory_info_t *memory);
void check_disk_thresholds(const disk_info_t *disks, uint32_t count);
void check_network_thresholds(const network_interface_t *networks, uint32_t count);
void check_thermal_thresholds(const thermal_sensor_t *sensors, uint32_t count);
void check_power_thresholds(const power_info_t *power);
void check_process_thresholds(const process_info_t *processes, uint32_t count);

// Smart threshold management
int enable_adaptive_thresholds(bool enable);
int calculate_optimal_thresholds(void);
int apply_learned_thresholds(void);
int reset_thresholds_to_default(void);

// Enhanced Configuration management
int load_config(const char *filename, monitor_config_t *config);
int save_config(const char *filename, const monitor_config_t *config);
int load_default_config(monitor_config_t *config);
int validate_config(const monitor_config_t *config);
int merge_config(monitor_config_t *base, const monitor_config_t *override);
int export_config_template(const char *filename);
int import_config_from_json(const char *json_str, monitor_config_t *config);
int export_config_to_json(const monitor_config_t *config, char *json_str, size_t size);
int auto_detect_optimal_config(monitor_config_t *config);
int backup_current_config(void);
int restore_config_backup(void);

// Registry/Settings management (Windows) or Config files (Linux)
int save_config_to_registry(const monitor_config_t *config);
int load_config_from_registry(monitor_config_t *config);
int save_user_preferences(const char *key, const char *value);
int load_user_preferences(const char *key, char *value, size_t size);
int clear_user_preferences(void);

// Enhanced Error codes & Status definitions
#define SYSWATCH_OK                    0
#define SYSWATCH_ERROR                -1
#define SYSWATCH_PERMISSION_DENIED    -2
#define SYSWATCH_INVALID_LICENSE      -3
#define SYSWATCH_FEATURE_UNAVAILABLE  -4
#define SYSWATCH_INSUFFICIENT_MEMORY  -5
#define SYSWATCH_INVALID_PARAMETER    -6
#define SYSWATCH_TIMEOUT              -7
#define SYSWATCH_NOT_INITIALIZED      -8
#define SYSWATCH_ALREADY_RUNNING      -9
#define SYSWATCH_NOT_RUNNING          -10
#define SYSWATCH_NETWORK_ERROR        -11
#define SYSWATCH_FILE_ERROR           -12
#define SYSWATCH_PARSE_ERROR          -13
#define SYSWATCH_AUTHENTICATION_FAILED -14
#define SYSWATCH_QUOTA_EXCEEDED       -15
#define SYSWATCH_VERSION_MISMATCH     -16
#define SYSWATCH_CORRUPTED_DATA       -17
#define SYSWATCH_HARDWARE_ERROR       -18
#define SYSWATCH_DRIVER_ERROR         -19
#define SYSWATCH_SERVICE_UNAVAILABLE  -20

// Performance status codes
#define PERFORMANCE_EXCELLENT  0
#define PERFORMANCE_GOOD       1
#define PERFORMANCE_FAIR       2
#define PERFORMANCE_POOR       3
#define PERFORMANCE_CRITICAL   4

// Color codes for console output
#define COLOR_RESET     "\033[0m"
#define COLOR_BLACK     "\033[30m"
#define COLOR_RED       "\033[31m"
#define COLOR_GREEN     "\033[32m"
#define COLOR_YELLOW    "\033[33m"
#define COLOR_BLUE      "\033[34m"
#define COLOR_MAGENTA   "\033[35m"
#define COLOR_CYAN      "\033[36m"
#define COLOR_WHITE     "\033[37m"
#define COLOR_BRIGHT_RED     "\033[91m"
#define COLOR_BRIGHT_GREEN   "\033[92m"
#define COLOR_BRIGHT_YELLOW  "\033[93m"
#define COLOR_BRIGHT_BLUE    "\033[94m"
#define COLOR_BRIGHT_MAGENTA "\033[95m"
#define COLOR_BRIGHT_CYAN    "\033[96m"
#define COLOR_BRIGHT_WHITE   "\033[97m"

// Background colors
#define BG_BLACK        "\033[40m"
#define BG_RED          "\033[41m"
#define BG_GREEN        "\033[42m"
#define BG_YELLOW       "\033[43m"
#define BG_BLUE         "\033[44m"
#define BG_MAGENTA      "\033[45m"
#define BG_CYAN         "\033[46m"
#define BG_WHITE        "\033[47m"

// Text formatting
#define FORMAT_BOLD      "\033[1m"
#define FORMAT_DIM       "\033[2m"
#define FORMAT_ITALIC    "\033[3m"
#define FORMAT_UNDERLINE "\033[4m"
#define FORMAT_BLINK     "\033[5m"
#define FORMAT_REVERSE   "\033[7m"
#define FORMAT_STRIKETHROUGH "\033[9m"

// Function return status helpers
#define SYSWATCH_SUCCESS(code) ((code) >= 0)
#define SYSWATCH_FAILED(code)  ((code) < 0)

// Error description function
const char* syswatch_error_string(int error_code);
void syswatch_print_error(int error_code, const char *context);
int syswatch_log_error(int error_code, const char *context, const char *details);

// Debug and logging macros
#ifdef DEBUG
    #define SYSWATCH_DEBUG(fmt, ...) printf("[DEBUG] " fmt "\n", ##__VA_ARGS__)
    #define SYSWATCH_TRACE(fmt, ...) printf("[TRACE] " fmt "\n", ##__VA_ARGS__)
#else
    #define SYSWATCH_DEBUG(fmt, ...)
    #define SYSWATCH_TRACE(fmt, ...)
#endif

#define SYSWATCH_INFO(fmt, ...)     printf("[INFO] " fmt "\n", ##__VA_ARGS__)
#define SYSWATCH_WARNING(fmt, ...)  printf("[WARNING] " fmt "\n", ##__VA_ARGS__)
#define SYSWATCH_ERROR_MSG(fmt, ...)   printf("[ERROR] " fmt "\n", ##__VA_ARGS__)
#define SYSWATCH_CRITICAL(fmt, ...) printf("[CRITICAL] " fmt "\n", ##__VA_ARGS__)

#endif // SYSWATCH_H