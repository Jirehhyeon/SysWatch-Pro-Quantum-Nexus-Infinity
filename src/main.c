#include "syswatch.h"
#include <signal.h>

static volatile bool running = true;
static monitor_config_t config;
static performance_history_t history = {0};

void signal_handler(int sig) {
    if (sig == SIGINT || sig == SIGTERM) {
        running = false;
        printf("\nShutting down SysWatch Pro...\n");
    }
}

void print_banner(void) {
    printf("╔══════════════════════════════════════╗\n");
    printf("║          SysWatch Pro v%s          ║\n", SYSWATCH_VERSION);
    printf("║      System Monitoring Tool          ║\n");
    printf("║                                      ║\n");
    printf("║  Edition: %-26s ║\n", SYSWATCH_BUILD);
    printf("╚══════════════════════════════════════╝\n\n");
}

void print_usage(void) {
    printf("Usage: syswatch [OPTIONS]\n\n");
    printf("Options:\n");
    printf("  -h, --help              Show this help message\n");
    printf("  -v, --version           Show version information\n");
    printf("  -c, --config FILE       Load configuration from file\n");
    printf("  -i, --interval SECONDS  Update interval (default: 1)\n");
    printf("  -o, --output FILE       Export data to file (Pro only)\n");
    printf("  -f, --format FORMAT     Export format: csv, json (Pro only)\n");
    printf("  -a, --alerts            Enable threshold alerts (Pro only)\n");
    printf("  -b, --background        Run in background mode (Pro only)\n");
    printf("  -t, --test              Run system test\n");
    printf("\nExamples:\n");
    printf("  syswatch                       # Basic monitoring\n");
    printf("  syswatch -i 5                  # Update every 5 seconds\n");
    printf("  syswatch -o report.csv -f csv  # Export to CSV (Pro)\n");
    printf("  syswatch -a -b                 # Background with alerts (Pro)\n");
}

void display_real_time_info(void) {
    system_info_t info;
    process_info_t processes[20];  // Show top 20 processes
    int process_count = 20;
    char buffer[64];
    
    // Clear screen (works on both Windows and Linux)
    #ifdef WINDOWS
        system("cls");
    #else
        system("clear");
    #endif
    
    print_banner();
    
    if (get_system_info(&info) != SYSWATCH_OK) {
        printf("Error: Failed to get system information\n");
        return;
    }
    
    printf("┌─ System Performance ────────────────────────────────┐\n");
    printf("│ CPU Usage:    %6.1f%%                              │\n", info.cpu_usage);
    printf("│ Memory:       %s / %s (%5.1f%%)     │\n", 
           format_bytes(info.memory_used, buffer, sizeof(buffer)),
           format_bytes(info.memory_total, buffer, sizeof(buffer)),
           info.memory_usage_percent);
    printf("│ Disk:         %s / %s (%5.1f%%)     │\n",
           format_bytes(info.disk_used, buffer, sizeof(buffer)),
           format_bytes(info.disk_total, buffer, sizeof(buffer)),
           info.disk_usage_percent);
    printf("│ Network I/O:  ↑%s ↓%s           │\n",
           format_bytes(info.network_bytes_sent, buffer, sizeof(buffer)),
           format_bytes(info.network_bytes_received, buffer, sizeof(buffer)));
    printf("│ Last Update:  %s                     │\n",
           format_time(info.timestamp, buffer, sizeof(buffer)));
    printf("└──────────────────────────────────────────────────────┘\n\n");
    
    // Get top processes
    if (get_process_list(processes, &process_count, 20) == SYSWATCH_OK) {
        printf("┌─ Top Processes ──────────────────────────────────────┐\n");
        printf("│ PID     Name                    CPU%%    Memory       │\n");
        printf("├──────────────────────────────────────────────────────┤\n");
        
        for (int i = 0; i < process_count && i < 10; i++) {
            printf("│ %-7lu %-22s %5.1f%% %10s   │\n",
                   processes[i].pid,
                   processes[i].name,
                   processes[i].cpu_percent,
                   format_bytes(processes[i].memory_kb * 1024, buffer, sizeof(buffer)));
        }
        printf("└──────────────────────────────────────────────────────┘\n");
    }
    
    // Check for alerts if enabled
    if (config.enable_alerts) {
        check_thresholds(&info);
    }
    
    // Store in history for pro features
    if (is_pro_feature_available()) {
        history.history[history.current_index] = info;
        history.current_index = (history.current_index + 1) % HISTORY_SIZE;
        if (history.count < HISTORY_SIZE) {
            history.count++;
        }
    }
}

void run_system_test(void) {
    printf("Running SysWatch Pro system test...\n\n");
    
    printf("1. Testing system information retrieval...\n");
    system_info_t info;
    if (get_system_info(&info) == SYSWATCH_OK) {
        printf("   ✓ System info: OK\n");
        print_system_info(&info);
    } else {
        printf("   ✗ System info: FAILED\n");
        return;
    }
    
    printf("\n2. Testing process enumeration...\n");
    process_info_t processes[10];
    int count = 10;
    if (get_process_list(processes, &count, 10) == SYSWATCH_OK) {
        printf("   ✓ Process list: OK (%d processes found)\n", count);
        printf("   Top 3 processes:\n");
        for (int i = 0; i < count && i < 3; i++) {
            printf("     - %s (PID: %lu)\n", processes[i].name, processes[i].pid);
        }
    } else {
        printf("   ✗ Process list: FAILED\n");
    }
    
    printf("\n3. Testing license verification...\n");
    if (verify_license(config.license)) {
        printf("   ✓ License: Valid (%s)\n", 
               config.license == LICENSE_PROFESSIONAL ? "Professional" : "Community");
    } else {
        printf("   ✗ License: Invalid\n");
    }
    
    printf("\n4. Testing configuration...\n");
    printf("   - Update interval: %d ms\n", config.update_interval);
    printf("   - Alerts enabled: %s\n", config.enable_alerts ? "Yes" : "No");
    printf("   - Log to file: %s\n", config.log_to_file ? "Yes" : "No");
    
    printf("\nSystem test completed successfully!\n");
}

int main(int argc, char *argv[]) {
    // Initialize default configuration
    config.enable_alerts = false;
    config.cpu_alert_threshold = 80.0;
    config.memory_alert_threshold = 85.0;
    config.disk_alert_threshold = 90.0;
    config.update_interval = UPDATE_INTERVAL;
    config.log_to_file = false;
    config.license = LICENSE_COMMUNITY;
    strcpy(config.log_file_path, "syswatch.log");
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage();
            return 0;
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--version") == 0) {
            print_version();
            return 0;
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--test") == 0) {
            if (init_monitor(&config) != SYSWATCH_OK) {
                printf("Error: Failed to initialize monitoring\n");
                return 1;
            }
            run_system_test();
            cleanup_monitor();
            return 0;
        } else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--interval") == 0) {
            if (i + 1 < argc) {
                config.update_interval = atoi(argv[++i]) * 1000;
            }
        } else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--alerts") == 0) {
            if (is_pro_feature_available()) {
                config.enable_alerts = true;
            } else {
                printf("Warning: Alerts feature requires Professional license\n");
            }
        } else if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--config") == 0) {
            if (i + 1 < argc) {
                load_config(argv[++i], &config);
            }
        }
    }
    
    // Initialize monitoring system
    if (init_monitor(&config) != SYSWATCH_OK) {
        printf("Error: Failed to initialize monitoring system\n");
        return 1;
    }
    
    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Main monitoring loop
    printf("Press Ctrl+C to exit\n\n");
    
    while (running) {
        display_real_time_info();
        
        // Sleep for update interval
        #ifdef WINDOWS
            Sleep(config.update_interval);
        #else
            usleep(config.update_interval * 1000);
        #endif
    }
    
    cleanup_monitor();
    printf("SysWatch Pro terminated.\n");
    return 0;
}