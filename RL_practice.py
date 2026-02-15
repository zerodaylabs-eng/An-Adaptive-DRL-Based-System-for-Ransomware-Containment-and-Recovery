import os
import time
import pwd
import csv
import datetime

HZ = os.sysconf(os.sysconf_names['SC_CLK_TCK'])

def get_boot_time():
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("btime"):
                return int(line.split()[1])
    return 0

BOOT_TIME = get_boot_time()

def get_total_memory():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    return 1

TOTAL_MEM = get_total_memory()

def read_cpu_times():
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu "):
                return list(map(int, line.strip().split()[1:]))
    return []

def get_process_state(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("State:"):
                    return line.split()[1]
    except:
        return None
    return None

def get_username(uid):
    try:
        return pwd.getpwuid(uid).pw_name
    except:
        return str(uid)

def get_process_info(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().split()
        comm = fields[1].strip("()")
        utime = int(fields[13])
        stime = int(fields[14])
        cutime = int(fields[15])
        cstime = int(fields[16])
        total_time = utime + stime + cutime + cstime
        start_time_ticks = int(fields[21])
        start_time = BOOT_TIME + (start_time_ticks // HZ)
        runtime = int(time.time()) - start_time
        return comm, total_time, start_time, runtime
    except:
        return None, 0, 0, 0

def get_process_mem(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except:
        return 0
    return 0

def is_encrypting(pid):
    try:
        fd_dir = f"/proc/{pid}/fd"
        if os.path.exists(fd_dir):
            write_count = 0
            for fd in os.listdir(fd_dir):
                try:
                    path = os.readlink(os.path.join(fd_dir, fd))
                    if "tmp" in path or ".enc" in path:
                        write_count += 1
                except:
                    continue
            return 1 if write_count > 5 else 0
    except:
        return 0
    return 0

def collect_process_info(interval=1.0):
    cpu_times_1 = read_cpu_times()
    proc_times_1 = {}
    for pid in filter(str.isdigit, os.listdir("/proc")):
        _, total_time, _, _ = get_process_info(pid)
        proc_times_1[pid] = total_time
    time.sleep(interval)

    cpu_times_2 = read_cpu_times()
    total_diff = sum(cpu_times_2) - sum(cpu_times_1)
    if total_diff <= 0: total_diff = 1

    process_data = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            comm, total_time, start_time, runtime = get_process_info(pid)
            prev_time = proc_times_1.get(pid, 0)
            cpu_usage = 100.0 * (total_time - prev_time) / total_diff

            mem_bytes = get_process_mem(pid)
            mem_percent = (mem_bytes / TOTAL_MEM) * 100

            with open(f"/proc/{pid}/status") as f:
                lines = f.readlines()
            uid = int([x.split()[1] for x in lines if x.startswith("Uid:")][0])
            username = get_username(uid)
            state = get_process_state(pid)

            enc = is_encrypting(pid)
            if enc:
                state = "E"

            process_data.append({
                "pid": int(pid),
                "name": comm,
                "username": username,
                "cpu_percent": round(cpu_usage, 2),
                "memory_percent": round(mem_percent, 2),
                "runtime": str(datetime.timedelta(seconds=runtime)),
                "state_id": state,
                "status": state,
                "encrypting": enc
            })
        except:
            continue

    return process_data

def write_to_csv(process_data, filename="processes.csv"):
    if not process_data:
        return
    keys = process_data[0].keys()
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(process_data)

if __name__ == "__main__":
    data = collect_process_info()
    write_to_csv(data)
    print(f"[✔] Wrote {len(data)} processes to processes.csv")
