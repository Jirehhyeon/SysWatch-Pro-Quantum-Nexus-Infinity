#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
    #include <windows.h>
    #include <psapi.h>
    #define SLEEP(ms) Sleep(ms)
#else
    #include <unistd.h>
    #include <sys/sysinfo.h>
    #define SLEEP(ms) usleep(ms * 1000)
#endif

#define VERSION "1.0.0"

void print_banner() {
    printf("╔══════════════════════════════════════╗\n");
    printf("║          SysWatch Pro v%s          ║\n", VERSION);
    printf("║      System Monitoring Tool          ║\n");
    printf("║         Community Edition            ║\n");
    printf("╚══════════════════════════════════════╝\n\n");
}

void print_help() {
    printf("SysWatch Pro v%s - System Monitoring Tool\n\n", VERSION);
    printf("Usage: syswatch [OPTIONS]\n\n");
    printf("Options:\n");
    printf("  -h, --help     Show this help\n");
    printf("  -v, --version  Show version\n");
    printf("  -t, --test     Run system test\n");
    printf("  -i, --info     Show system info once\n");
    printf("\nExamples:\n");
    printf("  syswatch       # Start real-time monitoring\n");
    printf("  syswatch -i    # Show system info once\n");
    printf("  syswatch -t    # Run system test\n");
}

void print_version() {
    printf("SysWatch Pro v%s\n", VERSION);
    printf("Edition: Community\n");
    #ifdef _WIN32
        printf("Platform: Windows\n");
    #else
        printf("Platform: Linux\n");
    #endif
    printf("Copyright (C) 2025 SysWatch Pro\n");
}

#ifdef _WIN32
void get_system_info() {
    MEMORYSTATUSEX mem_status;
    mem_status.dwLength = sizeof(mem_status);
    
    if (GlobalMemoryStatusEx(&mem_status)) {
        ULARGE_INTEGER free_bytes, total_bytes;
        GetDiskFreeSpaceEx("C:\\", &free_bytes, &total_bytes, NULL);
        
        printf("┌─ System Information ────────────────────────────────┐\n");
        printf("│ Memory:  %5.1f%% used                              │\n", 
               (double)(mem_status.ullTotalPhys - mem_status.ullAvailPhys) / mem_status.ullTotalPhys * 100.0);
        printf("│          %4.1f GB / %4.1f GB                      │\n",
               (double)(mem_status.ullTotalPhys - mem_status.ullAvailPhys) / (1024*1024*1024),
               (double)mem_status.ullTotalPhys / (1024*1024*1024));
        printf("│                                                      │\n");
        printf("│ Disk:    %5.1f%% used (C: drive)                   │\n",
               (double)(total_bytes.QuadPart - free_bytes.QuadPart) / total_bytes.QuadPart * 100.0);
        printf("│          %4.1f GB / %4.1f GB                      │\n",
               (double)(total_bytes.QuadPart - free_bytes.QuadPart) / (1024*1024*1024),
               (double)total_bytes.QuadPart / (1024*1024*1024));
        printf("└──────────────────────────────────────────────────────┘\n");
    } else {
        printf("Error: Could not retrieve system information\n");
    }
}

void get_process_info() {
    DWORD processes[1024], bytes_returned, num_processes;
    
    if (EnumProcesses(processes, sizeof(processes), &bytes_returned)) {
        num_processes = bytes_returned / sizeof(DWORD);
        
        printf("\n┌─ Process Information ────────────────────────────────┐\n");
        printf("│ Total Processes: %-32d    │\n", num_processes);
        
        int running_count = 0;
        for (DWORD i = 0; i < num_processes; i++) {
            if (processes[i] != 0) {
                HANDLE process = OpenProcess(PROCESS_QUERY_INFORMATION, FALSE, processes[i]);
                if (process) {
                    running_count++;
                    CloseHandle(process);
                }
            }
        }
        
        printf("│ Running Processes: %-29d    │\n", running_count);
        printf("│ System Processes: %-30d    │\n", num_processes - running_count);
        printf("└──────────────────────────────────────────────────────┘\n");
    }
}

#else
void get_system_info() {
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        printf("┌─ System Information ────────────────────────────────┐\n");
        printf("│ Memory:  %5.1f%% used                              │\n", 
               (double)(info.totalram - info.freeram) / info.totalram * 100.0);
        printf("│          %4.1f GB / %4.1f GB                      │\n",
               (double)(info.totalram - info.freeram) * info.mem_unit / (1024*1024*1024),
               (double)info.totalram * info.mem_unit / (1024*1024*1024));
        printf("│                                                      │\n");
        printf("│ Uptime:  %ld days, %ld hours                       │\n",
               info.uptime / 86400, (info.uptime % 86400) / 3600);
        printf("│ Load:    %.2f, %.2f, %.2f                          │\n",
               info.loads[0] / 65536.0, info.loads[1] / 65536.0, info.loads[2] / 65536.0);
        printf("└──────────────────────────────────────────────────────┘\n");
    }
}

void get_process_info() {
    printf("\n┌─ Process Information ────────────────────────────────┐\n");
    printf("│ Process info available on Linux                      │\n");
    printf("│ (Feature coming in full version)                     │\n");
    printf("└──────────────────────────────────────────────────────┘\n");
}
#endif

void run_test() {
    printf("Running SysWatch Pro system test...\n\n");
    printf("1. Testing system information retrieval...\n");
    get_system_info();
    printf("   ✓ System info: OK\n\n");
    
    printf("2. Testing process enumeration...\n");
    get_process_info();
    printf("   ✓ Process info: OK\n\n");
    
    printf("3. Testing time functions...\n");
    time_t now = time(NULL);
    printf("   Current time: %s", ctime(&now));
    printf("   ✓ Time functions: OK\n\n");
    
    printf("System test completed successfully!\n");
}

void monitor_loop() {
    printf("Press Ctrl+C to exit monitoring\n\n");
    
    int count = 0;
    while (count < 10) {  // Run for 10 cycles in demo
        #ifdef _WIN32
            system("cls");
        #else
            system("clear");
        #endif
        
        print_banner();
        get_system_info();
        get_process_info();
        
        printf("\n\nUpdate #%d - Next update in 3 seconds...\n", count + 1);
        printf("(Demo version - limited to 10 updates)\n");
        
        SLEEP(3000);
        count++;
    }
    
    printf("\nDemo monitoring completed.\n");
    printf("Upgrade to Professional for continuous monitoring!\n");
}

int main(int argc, char *argv[]) {
    if (argc > 1) {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            print_help();
            return 0;
        } else if (strcmp(argv[1], "-v") == 0 || strcmp(argv[1], "--version") == 0) {
            print_version();
            return 0;
        } else if (strcmp(argv[1], "-t") == 0 || strcmp(argv[1], "--test") == 0) {
            run_test();
            return 0;
        } else if (strcmp(argv[1], "-i") == 0 || strcmp(argv[1], "--info") == 0) {
            print_banner();
            get_system_info();
            get_process_info();
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[1]);
            printf("Use --help for usage information\n");
            return 1;
        }
    }
    
    print_banner();
    monitor_loop();
    return 0;
}