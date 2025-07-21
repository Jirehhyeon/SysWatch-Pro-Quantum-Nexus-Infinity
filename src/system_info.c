#include "syswatch.h"

#ifdef WINDOWS

static PDH_HQUERY cpu_query = NULL;
static PDH_HCOUNTER cpu_counter = NULL;
static bool pdh_initialized = false;

int init_monitor(monitor_config_t *config) {
    // Initialize PDH for CPU monitoring
    if (PdhOpenQuery(NULL, 0, &cpu_query) != ERROR_SUCCESS) {
        return SYSWATCH_ERROR;
    }
    
    if (PdhAddCounter(cpu_query, "\\Processor(_Total)\\% Processor Time", 0, &cpu_counter) != ERROR_SUCCESS) {
        PdhCloseQuery(cpu_query);
        return SYSWATCH_ERROR;
    }
    
    // First call to initialize counters
    PdhCollectQueryData(cpu_query);
    pdh_initialized = true;
    
    return SYSWATCH_OK;
}

int cleanup_monitor(void) {
    if (pdh_initialized) {
        PdhCloseQuery(cpu_query);
        pdh_initialized = false;
    }
    return SYSWATCH_OK;
}

double get_cpu_usage(void) {
    if (!pdh_initialized) return 0.0;
    
    PDH_FMT_COUNTERVALUE counter_value;
    
    // Collect current data
    if (PdhCollectQueryData(cpu_query) != ERROR_SUCCESS) {
        return 0.0;
    }
    
    // Wait a bit for accurate reading
    Sleep(100);
    
    // Collect again for calculation
    if (PdhCollectQueryData(cpu_query) != ERROR_SUCCESS) {
        return 0.0;
    }
    
    if (PdhGetFormattedCounterValue(cpu_counter, PDH_FMT_DOUBLE, NULL, &counter_value) == ERROR_SUCCESS) {
        return counter_value.doubleValue;
    }
    
    return 0.0;
}

int get_memory_info(unsigned long *total, unsigned long *available, unsigned long *used) {
    MEMORYSTATUSEX mem_status;
    mem_status.dwLength = sizeof(mem_status);
    
    if (!GlobalMemoryStatusEx(&mem_status)) {
        return SYSWATCH_ERROR;
    }
    
    *total = (unsigned long)(mem_status.ullTotalPhys / 1024);  // Convert to KB
    *available = (unsigned long)(mem_status.ullAvailPhys / 1024);
    *used = *total - *available;
    
    return SYSWATCH_OK;
}

int get_disk_info(unsigned long *total, unsigned long *available, unsigned long *used) {
    ULARGE_INTEGER free_bytes, total_bytes;
    
    if (!GetDiskFreeSpaceEx("C:\\", &free_bytes, &total_bytes, NULL)) {
        return SYSWATCH_ERROR;
    }
    
    *total = (unsigned long)(total_bytes.QuadPart / 1024);  // Convert to KB
    *available = (unsigned long)(free_bytes.QuadPart / 1024);
    *used = *total - *available;
    
    return SYSWATCH_OK;
}

int get_network_info(unsigned long *bytes_sent, unsigned long *bytes_received) {
    // Simple implementation - in real version would use WMI or registry
    // For now, return dummy values
    *bytes_sent = 1024 * 1024;      // 1MB
    *bytes_received = 2048 * 1024;  // 2MB
    
    return SYSWATCH_OK;
}

int get_system_info(system_info_t *info) {
    if (!info) return SYSWATCH_ERROR;
    
    // Get CPU usage
    info->cpu_usage = get_cpu_usage();
    
    // Get memory information
    if (get_memory_info(&info->memory_total, &info->memory_available, &info->memory_used) != SYSWATCH_OK) {
        return SYSWATCH_ERROR;
    }
    info->memory_usage_percent = (double)info->memory_used / info->memory_total * 100.0;
    
    // Get disk information
    if (get_disk_info(&info->disk_total, &info->disk_available, &info->disk_used) != SYSWATCH_OK) {
        return SYSWATCH_ERROR;
    }
    info->disk_usage_percent = (double)info->disk_used / info->disk_total * 100.0;
    
    // Get network information
    get_network_info(&info->network_bytes_sent, &info->network_bytes_received);
    
    // Set timestamp
    info->timestamp = time(NULL);
    
    return SYSWATCH_OK;
}

#else
// Linux implementation placeholder
int init_monitor(monitor_config_t *config) {
    return SYSWATCH_OK;
}

int cleanup_monitor(void) {
    return SYSWATCH_OK;
}

int get_system_info(system_info_t *info) {
    // Linux implementation would go here
    // Read from /proc/stat, /proc/meminfo, etc.
    return SYSWATCH_ERROR;
}

#endif