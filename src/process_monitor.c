#include "syswatch.h"

#ifdef WINDOWS

typedef struct {
    DWORD process_id;
    ULONGLONG last_cpu_time;
    ULONGLONG last_system_time;
} process_cpu_data_t;

static process_cpu_data_t cpu_data[MAX_PROCESSES];
static int cpu_data_count = 0;

double calculate_process_cpu_usage(DWORD process_id, HANDLE process_handle) {
    FILETIME creation_time, exit_time, kernel_time, user_time;
    FILETIME current_time;
    ULONGLONG current_cpu_time, current_system_time;
    double cpu_percent = 0.0;
    
    // Get process times
    if (!GetProcessTimes(process_handle, &creation_time, &exit_time, &kernel_time, &user_time)) {
        return 0.0;
    }
    
    // Get current system time
    GetSystemTimeAsFileTime(&current_time);
    
    // Convert to 64-bit values
    current_cpu_time = ((ULONGLONG)kernel_time.dwHighDateTime << 32) + kernel_time.dwLowDateTime +
                       ((ULONGLONG)user_time.dwHighDateTime << 32) + user_time.dwLowDateTime;
    current_system_time = ((ULONGLONG)current_time.dwHighDateTime << 32) + current_time.dwLowDateTime;
    
    // Find existing CPU data for this process
    int data_index = -1;
    for (int i = 0; i < cpu_data_count; i++) {
        if (cpu_data[i].process_id == process_id) {
            data_index = i;
            break;
        }
    }
    
    if (data_index >= 0) {
        // Calculate CPU usage since last measurement
        ULONGLONG cpu_delta = current_cpu_time - cpu_data[data_index].last_cpu_time;
        ULONGLONG system_delta = current_system_time - cpu_data[data_index].last_system_time;
        
        if (system_delta > 0) {
            cpu_percent = (double)cpu_delta / system_delta * 100.0;
        }
        
        // Update stored values
        cpu_data[data_index].last_cpu_time = current_cpu_time;
        cpu_data[data_index].last_system_time = current_system_time;
    } else {
        // First time seeing this process, add to tracking
        if (cpu_data_count < MAX_PROCESSES) {
            cpu_data[cpu_data_count].process_id = process_id;
            cpu_data[cpu_data_count].last_cpu_time = current_cpu_time;
            cpu_data[cpu_data_count].last_system_time = current_system_time;
            cpu_data_count++;
        }
    }
    
    return cpu_percent;
}

int get_process_memory_info(HANDLE process_handle, unsigned long *memory_kb) {
    PROCESS_MEMORY_COUNTERS pmc;
    
    if (GetProcessMemoryInfo(process_handle, &pmc, sizeof(pmc))) {
        *memory_kb = (unsigned long)(pmc.WorkingSetSize / 1024);
        return SYSWATCH_OK;
    }
    
    *memory_kb = 0;
    return SYSWATCH_ERROR;
}

int get_process_list(process_info_t *processes, int *count, int max_count) {
    DWORD process_ids[MAX_PROCESSES];
    DWORD bytes_returned;
    DWORD num_processes;
    int valid_processes = 0;
    
    if (!processes || !count || max_count <= 0) {
        return SYSWATCH_ERROR;
    }
    
    // Enumerate all processes
    if (!EnumProcesses(process_ids, sizeof(process_ids), &bytes_returned)) {
        return SYSWATCH_ERROR;
    }
    
    num_processes = bytes_returned / sizeof(DWORD);
    
    for (DWORD i = 0; i < num_processes && valid_processes < max_count; i++) {
        HANDLE process_handle;
        HMODULE module_handle;
        DWORD module_bytes;
        char process_name[MAX_NAME_LEN];
        unsigned long memory_kb;
        
        // Skip system idle process
        if (process_ids[i] == 0) continue;
        
        // Open process with required permissions
        process_handle = OpenProcess(
            PROCESS_QUERY_INFORMATION | PROCESS_VM_READ,
            FALSE, 
            process_ids[i]
        );
        
        if (process_handle == NULL) continue;
        
        // Get process name
        if (EnumProcessModules(process_handle, &module_handle, sizeof(module_handle), &module_bytes)) {
            if (GetModuleBaseName(process_handle, module_handle, process_name, sizeof(process_name)) == 0) {
                strcpy(process_name, "Unknown");
            }
        } else {
            strcpy(process_name, "System Process");
        }
        
        // Get memory information
        if (get_process_memory_info(process_handle, &memory_kb) != SYSWATCH_OK) {
            memory_kb = 0;
        }
        
        // Fill process information
        processes[valid_processes].pid = process_ids[i];
        strncpy(processes[valid_processes].name, process_name, MAX_NAME_LEN - 1);
        processes[valid_processes].name[MAX_NAME_LEN - 1] = '\0';
        processes[valid_processes].memory_kb = memory_kb;
        processes[valid_processes].cpu_percent = calculate_process_cpu_usage(process_ids[i], process_handle);
        processes[valid_processes].is_running = true;
        processes[valid_processes].start_time = time(NULL); // Simplified
        processes[valid_processes].handles = 0; // Would need additional API calls
        
        CloseHandle(process_handle);
        valid_processes++;
    }
    
    // Sort processes by CPU usage (descending)
    for (int i = 0; i < valid_processes - 1; i++) {
        for (int j = i + 1; j < valid_processes; j++) {
            if (processes[i].cpu_percent < processes[j].cpu_percent) {
                process_info_t temp = processes[i];
                processes[i] = processes[j];
                processes[j] = temp;
            }
        }
    }
    
    *count = valid_processes;
    return SYSWATCH_OK;
}

// Professional features
int kill_process(unsigned long pid) {
    if (!is_pro_feature_available()) {
        return SYSWATCH_FEATURE_UNAVAILABLE;
    }
    
    HANDLE process_handle = OpenProcess(PROCESS_TERMINATE, FALSE, (DWORD)pid);
    if (process_handle == NULL) {
        return SYSWATCH_PERMISSION_DENIED;
    }
    
    BOOL result = TerminateProcess(process_handle, 1);
    CloseHandle(process_handle);
    
    return result ? SYSWATCH_OK : SYSWATCH_ERROR;
}

int suspend_process(unsigned long pid) {
    if (!is_pro_feature_available()) {
        return SYSWATCH_FEATURE_UNAVAILABLE;
    }
    
    // This would require additional implementation for process suspension
    // For now, return feature unavailable
    return SYSWATCH_FEATURE_UNAVAILABLE;
}

int get_process_details(unsigned long pid, process_info_t *details) {
    HANDLE process_handle = OpenProcess(
        PROCESS_QUERY_INFORMATION | PROCESS_VM_READ,
        FALSE, 
        (DWORD)pid
    );
    
    if (process_handle == NULL) {
        return SYSWATCH_ERROR;
    }
    
    HMODULE module_handle;
    DWORD module_bytes;
    char process_name[MAX_NAME_LEN];
    unsigned long memory_kb;
    
    // Get process name
    if (EnumProcessModules(process_handle, &module_handle, sizeof(module_handle), &module_bytes)) {
        if (GetModuleBaseName(process_handle, module_handle, process_name, sizeof(process_name)) == 0) {
            strcpy(process_name, "Unknown");
        }
    } else {
        strcpy(process_name, "System Process");
    }
    
    // Get memory information
    if (get_process_memory_info(process_handle, &memory_kb) != SYSWATCH_OK) {
        memory_kb = 0;
    }
    
    // Fill details
    details->pid = pid;
    strncpy(details->name, process_name, MAX_NAME_LEN - 1);
    details->name[MAX_NAME_LEN - 1] = '\0';
    details->memory_kb = memory_kb;
    details->cpu_percent = calculate_process_cpu_usage((DWORD)pid, process_handle);
    details->is_running = true;
    
    // Get process creation time
    FILETIME creation_time, exit_time, kernel_time, user_time;
    if (GetProcessTimes(process_handle, &creation_time, &exit_time, &kernel_time, &user_time)) {
        // Convert FILETIME to time_t (simplified)
        ULONGLONG time_64 = ((ULONGLONG)creation_time.dwHighDateTime << 32) + creation_time.dwLowDateTime;
        details->start_time = (time_t)((time_64 - 116444736000000000ULL) / 10000000ULL);
    } else {
        details->start_time = 0;
    }
    
    CloseHandle(process_handle);
    return SYSWATCH_OK;
}

#else
// Linux implementation placeholder
int get_process_list(process_info_t *processes, int *count, int max_count) {
    // Linux implementation would read from /proc
    return SYSWATCH_ERROR;
}

int kill_process(unsigned long pid) {
    return SYSWATCH_ERROR;
}

int suspend_process(unsigned long pid) {
    return SYSWATCH_ERROR;
}

int get_process_details(unsigned long pid, process_info_t *details) {
    return SYSWATCH_ERROR;
}

#endif