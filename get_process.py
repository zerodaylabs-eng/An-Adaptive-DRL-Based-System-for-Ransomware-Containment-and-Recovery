#!/usr/bin/env python3
#get_process.py
"""
Enhanced Process Monitor with Ransomware Detection Features
Reads directly from /proc and calculates advanced metrics for ML/RL training
Can be imported as a module or run standalone
"""

import os
import pwd
import time
import csv
import glob
import math
import statistics
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

class EnhancedProcessMonitor:
    def __init__(self, window_seconds=5):
        self.previous_stats = {}
        self.previous_cpu_total = 0
        self.boot_time = self._get_boot_time()
        self.total_memory = self._get_total_memory()
        self.clock_ticks = os.sysconf(os.sysconf_names['SC_CLK_TCK'])
        self.window_seconds = window_seconds
        
        # Historical data for calculations
        self.process_history = defaultdict(lambda: {
            'io_history': deque(maxlen=10),
            'fd_history': deque(maxlen=10),
            'path_history': deque(maxlen=20),
            'file_ops': deque(maxlen=50),
            'baseline_metrics': deque(maxlen=100)
        })
        
        # System baseline for z-score calculations
        self.system_baseline = {
            'cpu_values': deque(maxlen=1000),
            'memory_values': deque(maxlen=1000),
            'io_values': deque(maxlen=1000)
        }
        
    def _get_boot_time(self):
        """Get system boot time from /proc/stat"""
        try:
            with open('/proc/stat', 'r') as f:
                for line in f:
                    if line.startswith('btime'):
                        return int(line.split()[1])
        except:
            return int(time.time())
    
    def _get_total_memory(self):
        """Get total memory from /proc/meminfo"""
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        return int(line.split()[1]) * 1024  # Convert KB to bytes
        except:
            return 1024 * 1024 * 1024  # Default 1GB
    
    def _get_cpu_total_time(self):
        """Get total CPU time from /proc/stat"""
        try:
            with open('/proc/stat', 'r') as f:
                line = f.readline().strip()
                if line.startswith('cpu '):
                    times = list(map(int, line.split()[1:8]))
                    return sum(times)
        except:
            return 0
    
    def _read_process_stat(self, pid):
        """Read process stat file"""
        try:
            with open(f'/proc/{pid}/stat', 'r') as f:
                fields = f.read().strip().split()
                return {
                    'pid': int(fields[0]),
                    'comm': fields[1].strip('()'),
                    'state': fields[2],
                    'utime': int(fields[13]),
                    'stime': int(fields[14]),
                    'cutime': int(fields[15]),
                    'cstime': int(fields[16]),
                    'starttime': int(fields[21])
                }
        except:
            return None
    
    def _read_process_status(self, pid):
        """Read process status file"""
        try:
            status_info = {}
            with open(f'/proc/{pid}/status', 'r') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        status_info['memory'] = int(line.split()[1]) * 1024
                    elif line.startswith('Uid:'):
                        status_info['uid'] = int(line.split()[1])
                    elif line.startswith('State:'):
                        status_info['state_detailed'] = line.split()[1]
            return status_info
        except:
            return {}
    
    def _read_process_io(self, pid):
        """Read process I/O statistics"""
        try:
            io_info = {}
            with open(f'/proc/{pid}/io', 'r') as f:
                for line in f:
                    if line.startswith('read_bytes:'):
                        io_info['read_bytes'] = int(line.split()[1])
                    elif line.startswith('write_bytes:'):
                        io_info['write_bytes'] = int(line.split()[1])
                    elif line.startswith('syscr:'):
                        io_info['read_calls'] = int(line.split()[1])
                    elif line.startswith('syscw:'):
                        io_info['write_calls'] = int(line.split()[1])
            return io_info
        except:
            return {}
    
    def _get_open_fds(self, pid):
        """Count open file descriptors and analyze paths"""
        try:
            fd_path = f'/proc/{pid}/fd'
            if not os.path.exists(fd_path):
                return 0, set(), []
            
            fd_count = 0
            unique_paths = set()
            suspicious_files = []
            
            for fd_link in os.listdir(fd_path):
                try:
                    target = os.readlink(os.path.join(fd_path, fd_link))
                    fd_count += 1
                    
                    # Track unique paths
                    if target.startswith('/'):
                        unique_paths.add(os.path.dirname(target))
                    
                    # Check for suspicious patterns
                    suspicious_patterns = ['.enc', '.encrypted', '.locked', '.crypto', '.ransom']
                    if any(pattern in target.lower() for pattern in suspicious_patterns):
                        suspicious_files.append(target)
                        
                except:
                    continue
                    
            return fd_count, unique_paths, suspicious_files
        except:
            return 0, set(), []
    
    def _analyze_file_operations(self, pid):
        """Analyze file operations for rename/delete patterns"""
        rename_count = 0
        delete_count = 0
        
        try:
            fd_path = f'/proc/{pid}/fd'
            if os.path.exists(fd_path):
                for fd_link in os.listdir(fd_path):
                    try:
                        target = os.readlink(os.path.join(fd_path, fd_link))
                        
                        # Estimate renames based on encrypted file patterns
                        if any(ext in target.lower() for ext in ['.locked', '.encrypted', '.crypto']):
                            rename_count += 1
                            
                        # Estimate deletes based on temp file access
                        if '/tmp/' in target and 'deleted' in target:
                            delete_count += 1
                            
                    except:
                        continue
        except:
            pass
            
        return rename_count, delete_count
    
    def _calculate_entropy_sample(self, pid):
        """Calculate entropy sample from accessible process memory (sandbox only)"""
        try:
            cmdline_path = f'/proc/{pid}/cmdline'
            if os.path.exists(cmdline_path):
                with open(cmdline_path, 'rb') as f:
                    data = f.read()[:1024]  # Sample first 1KB
                    if data:
                        return self._calculate_entropy(data)
        except:
            pass
        return 0.0
    
    def _calculate_entropy(self, data):
        """Calculate Shannon entropy of data"""
        if not data:
            return 0.0
        
        # Count byte frequencies
        byte_counts = [0] * 256
        for byte in data:
            byte_counts[byte] += 1
        
        # Calculate entropy
        entropy = 0.0
        data_len = len(data)
        for count in byte_counts:
            if count > 0:
                probability = count / data_len
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _check_suspicious_filename_patterns(self, pid):
        """Check for suspicious filename patterns in file descriptors"""
        try:
            fd_path = f'/proc/{pid}/fd'
            if not os.path.exists(fd_path):
                return False
            
            suspicious_patterns = [
                '.locked', '.encrypted', '.crypto', '.ransom', '.enc',
                'readme_for_decrypt', 'how_to_decrypt', 'ransom_note'
            ]
            
            for fd_link in os.listdir(fd_path):
                try:
                    target = os.readlink(os.path.join(fd_path, fd_link))
                    target_lower = target.lower()
                    
                    if any(pattern in target_lower for pattern in suspicious_patterns):
                        return True
                except:
                    continue
                    
        except:
            pass
        return False
    
    def _check_cmdline_keywords(self, pid):
        """Check command line for encryption-related keywords"""
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                cmdline = f.read().decode('utf-8', errors='ignore').lower()
                
                encryption_keywords = [
                    'encrypt', 'decrypt', 'cipher', 'crypto', 'aes', 'rsa',
                    'ransom', 'bitcoin', 'payment', 'unlock', 'restore'
                ]
                
                return any(keyword in cmdline for keyword in encryption_keywords)
        except:
            return False
    
    def _calculate_enc_heuristic_score(self, metrics):
        """Calculate composite encryption heuristic score"""
        score = 0.0
        
        # High I/O activity
        if metrics['write_bytes_per_s'] > 10000:  # 10KB/s
            score += 2.0
        
        # High write/read ratio
        if metrics['write_read_ratio'] > 0.8:
            score += 1.5
        
        # Many file renames
        if metrics['rename_count'] > 0:
            score += 3.0
        
        # Suspicious filenames
        if metrics['suspicious_filename_flag']:
            score += 4.0
        
        # Command line keywords
        if metrics['cmdline_keyword_flag']:
            score += 2.0
        
        # High entropy
        if metrics['entropy_sample'] > 6.0:
            score += 1.0
        
        # Many directories accessed
        if metrics['num_dirs_touched'] > 10:
            score += 1.0
        
        # Large write sizes
        if metrics['avg_write_size_estimate'] > 50000:  # 50KB average
            score += 1.0
        
        return min(score, 10.0)  # Cap at 10
    
    def _calculate_zscore(self, value, baseline_values):
        """Calculate z-score against historical baseline"""
        if len(baseline_values) < 10:
            return 0.0
        
        try:
            mean = statistics.mean(baseline_values)
            stdev = statistics.stdev(baseline_values)
            if stdev == 0:
                return 0.0
            return abs(value - mean) / stdev
        except:
            return 0.0
    
    def get_enhanced_process_info(self):
        """Get enhanced process information with ransomware detection features"""
        processes = []
        current_cpu_total = self._get_cpu_total_time()
        current_time = time.time()
        timestamp = datetime.now().isoformat()
        
        # Get all PIDs
        pids = []
        for pid_dir in glob.glob('/proc/[0-9]*'):
            try:
                pid = int(os.path.basename(pid_dir))
                pids.append(pid)
            except:
                continue
        
        for pid in pids:
            try:
                # Read basic process info
                stat_info = self._read_process_stat(pid)
                if not stat_info:
                    continue
                
                status_info = self._read_process_status(pid)
                if not status_info:
                    continue
                
                io_info = self._read_process_io(pid)
                
                # Calculate basic metrics
                total_time = stat_info['utime'] + stat_info['stime']
                cpu_percent = 0.0
                
                if pid in self.previous_stats and self.previous_cpu_total > 0:
                    prev_total = self.previous_stats[pid]['total_time']
                    time_diff = total_time - prev_total
                    cpu_total_diff = current_cpu_total - self.previous_cpu_total
                    
                    if cpu_total_diff > 0:
                        cpu_percent = (time_diff / cpu_total_diff) * 100.0
                        cpu_percent = max(0.0, min(100.0, cpu_percent))
                
                # Memory percentage
                memory_bytes = status_info.get('memory', 0)
                memory_percent = (memory_bytes / self.total_memory) * 100.0 if self.total_memory > 0 else 0.0
                
                # I/O calculations
                write_bytes_per_s = 0.0
                read_bytes_per_s = 0.0
                write_read_ratio = 0.0
                avg_write_size_estimate = 0.0
                
                if pid in self.previous_stats:
                    time_delta = current_time - self.previous_stats[pid]['timestamp']
                    if time_delta > 0:
                        prev_io = self.previous_stats[pid].get('io_info', {})
                        
                        if 'write_bytes' in io_info and 'write_bytes' in prev_io:
                            write_diff = io_info['write_bytes'] - prev_io['write_bytes']
                            write_bytes_per_s = max(0, write_diff / time_delta)
                        
                        if 'read_bytes' in io_info and 'read_bytes' in prev_io:
                            read_diff = io_info['read_bytes'] - prev_io['read_bytes']
                            read_bytes_per_s = max(0, read_diff / time_delta)
                        
                        # Calculate write/read ratio
                        if read_bytes_per_s > 0:
                            write_read_ratio = write_bytes_per_s / read_bytes_per_s
                        elif write_bytes_per_s > 0:
                            write_read_ratio = 10.0  # High ratio when only writing
                        
                        # Estimate average write size
                        if 'write_calls' in io_info and 'write_calls' in prev_io:
                            call_diff = io_info['write_calls'] - prev_io['write_calls']
                            if call_diff > 0 and write_bytes_per_s > 0:
                                avg_write_size_estimate = (write_bytes_per_s * time_delta) / call_diff
                
                # File descriptor analysis
                fd_count, unique_paths, suspicious_files = self._get_open_fds(pid)
                
                # Calculate FD delta per second
                open_fds_delta_per_s = 0.0
                if pid in self.previous_stats:
                    prev_fd_count = self.previous_stats[pid].get('fd_count', 0)
                    time_delta = current_time - self.previous_stats[pid]['timestamp']
                    if time_delta > 0:
                        open_fds_delta_per_s = (fd_count - prev_fd_count) / time_delta
                
                # File operations analysis
                rename_count, delete_count = self._analyze_file_operations(pid)
                
                # Get username
                uid = status_info.get('uid', 0)
                username = self._get_username(uid)
                
                # Additional analysis
                suspicious_filename_flag = self._check_suspicious_filename_patterns(pid)
                cmdline_keyword_flag = self._check_cmdline_keywords(pid)
                entropy_sample = self._calculate_entropy_sample(pid)
                num_dirs_touched = len(unique_paths)
                
                # Calculate metrics for this process
                current_metrics = {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'write_bytes_per_s': write_bytes_per_s,
                    'read_bytes_per_s': read_bytes_per_s,
                    'write_read_ratio': write_read_ratio,
                    'avg_write_size_estimate': avg_write_size_estimate,
                    'rename_count': rename_count,
                    'suspicious_filename_flag': suspicious_filename_flag,
                    'cmdline_keyword_flag': cmdline_keyword_flag,
                    'entropy_sample': entropy_sample,
                    'num_dirs_touched': num_dirs_touched
                }
                
                # Calculate encryption heuristic score
                enc_heuristic_score = self._calculate_enc_heuristic_score(current_metrics)
                
                # Update system baseline
                self.system_baseline['cpu_values'].append(cpu_percent)
                self.system_baseline['memory_values'].append(memory_percent)
                self.system_baseline['io_values'].append(write_bytes_per_s)
                
                # Calculate z-scores
                cpu_zscore = self._calculate_zscore(cpu_percent, self.system_baseline['cpu_values'])
                memory_zscore = self._calculate_zscore(memory_percent, self.system_baseline['memory_values'])
                io_zscore = self._calculate_zscore(write_bytes_per_s, self.system_baseline['io_values'])
                host_baseline_zscore = max(cpu_zscore, memory_zscore, io_zscore)
                
                # Store current stats for next iteration
                self.previous_stats[pid] = {
                    'total_time': total_time,
                    'timestamp': current_time,
                    'io_info': io_info,
                    'fd_count': fd_count
                }
                
                # Runtime calculation
                start_time_seconds = stat_info['starttime'] / self.clock_ticks
                process_start_time = self.boot_time + start_time_seconds
                runtime_seconds = max(0, int(current_time - process_start_time))
                
                # Process state
                state = status_info.get('state_detailed', stat_info['state'])
                if enc_heuristic_score > 5.0:
                    state = 'E'  # Mark as encrypting
                
                processes.append({
                    'PID': pid,
                    'timestamp': timestamp,
                    'Name': stat_info['comm'][:15],
                    'Username': username,
                    'write_bytes_per_s': round(write_bytes_per_s, 2),
                    'read_bytes_per_s': round(read_bytes_per_s, 2),
                    'open_fds_delta_per_s': round(open_fds_delta_per_s, 2),
                    'unique_paths_accessed_count': num_dirs_touched,
                    'avg_write_size_estimate': round(avg_write_size_estimate, 2),
                    'write_read_ratio': round(write_read_ratio, 3),
                    'entropy_sample': round(entropy_sample, 3),
                    'rename_count': rename_count,
                    'delete_count': delete_count,
                    'cpu_percent': round(cpu_percent, 1),
                    'mem_percent': round(memory_percent, 1),
                    'num_dirs_touched': num_dirs_touched,
                    'suspicious_filename_flag': int(suspicious_filename_flag),
                    'cmdline_keyword_flag': int(cmdline_keyword_flag),
                    'enc_heuristic_score': round(enc_heuristic_score, 2),
                    'host_baseline_zscore': round(host_baseline_zscore, 2),
                    'State_ID': state,
                    'Runtime': self._format_runtime(runtime_seconds),
                    'Encrypting': 1 if enc_heuristic_score > 5.0 else 0
                })
                
            except (OSError, IOError, ValueError, KeyError):
                continue
        
        # Update previous CPU total
        self.previous_cpu_total = current_cpu_total
        
        # Sort by encryption heuristic score
        processes.sort(key=lambda x: x['enc_heuristic_score'], reverse=True)
        
        return processes
    
    def _get_username(self, uid):
        """Get username from UID"""
        try:
            return pwd.getpwuid(uid).pw_name
        except:
            return str(uid)
    
    def _format_runtime(self, seconds):
        """Format runtime in human-readable format"""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m{seconds % 60}s"
        elif seconds < 86400:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h{minutes}m"
        else:
            days = seconds // 86400
            hours = (seconds % 86400) // 3600
            return f"{days}d{hours}h"
    
    def export_to_csv(self, filename='enhances_process_monitor.csv'):
        """Export enhanced process information to CSV"""
        processes = self.get_enhanced_process_info()
        
        fieldnames = [
            'PID', 'timestamp', 'Name', 'Username', 'write_bytes_per_s', 'read_bytes_per_s',
            'open_fds_delta_per_s', 'unique_paths_accessed_count', 'avg_write_size_estimate',
            'write_read_ratio', 'entropy_sample', 'rename_count', 'delete_count',
            'cpu_percent', 'mem_percent', 'num_dirs_touched', 'suspicious_filename_flag',
            'cmdline_keyword_flag', 'enc_heuristic_score', 'host_baseline_zscore',
            'State_ID', 'Runtime', 'Encrypting'
        ]
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(processes)
        
        return processes


# ============== GLOBAL MONITOR INSTANCE ==============
# Maintains state across multiple calls
_global_monitor = None

def get_processes(csv_filename='enhances_process_monitor.csv', verbose=False):
    """
    Main function to update CSV with current process information.
    This maintains state across calls for accurate delta calculations.
    
    Args:
        csv_filename: Name of CSV file to write to
        verbose: Print summary statistics
    
    Returns:
        List of process dictionaries
    """
    global _global_monitor
    
    # Initialize monitor on first call
    if _global_monitor is None:
        _global_monitor = EnhancedProcessMonitor()
        # Take initial baseline measurement
        _global_monitor.get_enhanced_process_info()
        time.sleep(0.5)  # Short wait for better delta calculations
    
    # Get current process data and export to CSV
    processes = _global_monitor.export_to_csv(csv_filename)
    
    if verbose:
        print(f"✓ Updated {csv_filename} with {len(processes)} processes")
        high_risk = sum(1 for p in processes if p['enc_heuristic_score'] > 5)
        if high_risk > 0:
            print(f"⚠ High-risk processes detected: {high_risk}")
    
    return processes


def reset_monitor():
    """Reset the global monitor (useful for testing)"""
    global _global_monitor
    _global_monitor = None


def main():
    """Main function for standalone execution"""
    monitor = EnhancedProcessMonitor()
    
    print("Enhanced Process Monitor with Ransomware Detection")
    print("Collecting baseline metrics...")
    
    # Take initial measurements
    monitor.get_enhanced_process_info()
    time.sleep(2)  # Wait for more accurate calculations
    
    print("Collecting enhanced process information...")
    processes = monitor.export_to_csv()
    
    print(f"Exported {len(processes)} processes to enhances_process_monitor.csv")
    
    # Display top suspicious processes
    print("\nTop 10 processes by encryption heuristic score:")
    print(f"{'PID':<8} {'Name':<15} {'EncScore':<9} {'CPU%':<6} {'WriteB/s':<10} {'Ratio':<8} {'Dirs':<5} {'Flags':<6}")
    print("-" * 80)
    
    for proc in processes[:10]:
        flags = f"{'S' if proc['suspicious_filename_flag'] else ''}{'K' if proc['cmdline_keyword_flag'] else ''}"
        print(f"{proc['PID']:<8} {proc['Name']:<15} {proc['enc_heuristic_score']:<9} "
              f"{proc['cpu_percent']:<6} {proc['write_bytes_per_s']:<10.0f} "
              f"{proc['write_read_ratio']:<8.1f} {proc['num_dirs_touched']:<5} {flags:<6}")
    
    print(f"\nTotal processes: {len(processes)}")
    print(f"High-risk processes (score > 5): {sum(1 for p in processes if p['enc_heuristic_score'] > 5)}")
    print(f"Processes with encryption indicators: {sum(1 for p in processes if p['Encrypting'])}")


if __name__ == "__main__":
    main()