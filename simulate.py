#!/usr/bin/env python3
"""
ransom_simulator.py - Continuous Ransomware Simulation for RL Training

This script runs continuously, simulating ransomware behavior until stopped.
Designed for training reinforcement learning agents to detect and stop ransomware.

USAGE:
  python3 simulate.py --continuous                           # Run until Ctrl+C or '1' pressed
  python3 simulate.py --continuous --interval 5              # New attack every 5 seconds
  python3 simulate.py --test                                 # Quick self-test
"""

import argparse
import json
import csv
import os
import sys
import time
import uuid
import random
import threading
import select
from pathlib import Path
from datetime import datetime, timezone

class ContinuousRansomSimulator:
    def __init__(self, args):
        self.args = args
        self.sandbox_path = None
        self.events = []
        self.running = True
        self.attack_count = 0
        self.total_files_created = 0
        self.total_files_encrypted = 0
        
    def setup_sandbox(self):
        """Setup main sandbox directory"""
        sandbox_str = self.args.sandbox or f"/tmp/ransom_continuous_{int(time.time())}"
        self.sandbox_path = Path(sandbox_str).resolve()
        
        try:
            self.sandbox_path.mkdir(parents=True, exist_ok=True)
            print(f"Continuous simulation sandbox: {self.sandbox_path}")
        except Exception as e:
            print(f"ERROR: Cannot create sandbox {self.sandbox_path}: {e}")
            sys.exit(1)
            
    def create_attack_directory(self):
        """Create a new directory for each simulated attack"""
        timestamp = datetime.now().strftime("%H%M%S")
        attack_dir = self.sandbox_path / f"attack_{self.attack_count:03d}_{timestamp}"
        
        if not self.args.dry_run:
            attack_dir.mkdir(exist_ok=True)
            
        return attack_dir
        
    def generate_realistic_files(self, attack_dir, count=None):
        """Generate realistic file types that ransomware typically targets"""
        if count is None:
            count = random.randint(3, 8)
            
        file_templates = [
            ("Financial_Report_{}.xlsx", b"Excel financial data simulation - " * 50),
            ("Contract_{}.pdf", b"PDF contract document simulation - " * 40),
            ("Presentation_{}.pptx", b"PowerPoint presentation data - " * 45),
            ("Database_backup_{}.db", b"Database backup simulation - " * 60),
            ("Image_{}.jpg", b"JPEG image binary data simulation - " * 35),
            ("Document_{}.docx", b"Word document simulation content - " * 42),
            ("Archive_{}.zip", b"ZIP archive simulation data - " * 38),
            ("Spreadsheet_{}.csv", b"CSV data simulation content - " * 30),
        ]
        
        created_files = []
        for i in range(count):
            template_name, content = random.choice(file_templates)
            filename = template_name.format(random.randint(1000, 9999))
            
            self.log_event("file_create", filename, len(content), f"target file #{i+1}")
            
            if not self.args.dry_run:
                file_path = attack_dir / filename
                with open(file_path, 'wb') as f:
                    f.write(content)
                    
            created_files.append(filename)
            self.total_files_created += 1
            
        return created_files
        
    def simulate_encryption_process(self, attack_dir, target_files):
        """Simulate the encryption process with realistic timing"""
        sim_pid = random.randint(1000, 9999)
        process_names = ["crypto_locker.exe", "ransom_encrypt.exe", "file_locker.exe", "data_encrypt.exe"]
        process_name = random.choice(process_names)
        
        print(f"  🦠 Simulating ransomware process: {process_name} (PID: {sim_pid})")
        
        # Phase 1: Discovery and scanning
        self.log_event("process_start", process_name, 0, f"PID {sim_pid} started")
        time.sleep(random.uniform(0.2, 0.5))
        
        self.log_event("directory_scan", str(attack_dir), 0, "scanning for target files")
        time.sleep(random.uniform(0.1, 0.3))
        
        # Phase 2: File encryption simulation
        for i, filename in enumerate(target_files):
            # Simulate reading original file
            read_size = random.randint(1000, 50000)
            self.log_event("file_read", filename, read_size, "reading for encryption")
            time.sleep(random.uniform(0.1, 0.4))
            
            # Simulate encryption
            encrypted_name = f"{filename}.LOCKED"
            encrypt_size = read_size + random.randint(-100, 200)  # Encrypted size varies
            self.log_event("file_encrypt", encrypted_name, encrypt_size, f"encrypted from {filename}")
            
            # Actually rename file if not dry run
            if not self.args.dry_run:
                original_path = attack_dir / filename
                encrypted_path = attack_dir / encrypted_name
                if original_path.exists():
                    original_path.rename(encrypted_path)
                    
            self.total_files_encrypted += 1
            
            # Random delay between encryptions
            time.sleep(random.uniform(0.2, 0.8))
            
            # Show progress
            print(f"    📁 Encrypted {i+1}/{len(target_files)}: {filename} -> {encrypted_name}")
            
        # Phase 3: Create ransom note
        self.create_ransom_note(attack_dir, process_name)
        
        # Phase 4: Network activity simulation (logged only)
        self.log_event("network_connect", "192.168.1.100:8080", 0, "contacting C&C server")
        time.sleep(random.uniform(0.1, 0.3))
        
        self.log_event("process_end", process_name, 0, f"PID {sim_pid} completed encryption")
        
    def create_ransom_note(self, attack_dir, process_name):
        """Create ransom note with attack-specific info"""
        ransom_note = f"""
🔒 YOUR FILES HAVE BEEN ENCRYPTED! 🔒

Attack ID: {self.attack_count:03d}
Process: {process_name}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

⚠️  THIS IS A SIMULATION FOR RL TRAINING ⚠️

Real ransomware would demand payment here.
This simulation is for training AI agents to detect and stop such attacks.

Files encrypted: {len([f for f in os.listdir(attack_dir) if f.endswith('.LOCKED')]) if not self.args.dry_run else 'simulated'}

To "decrypt" files in this simulation: remove .LOCKED extension
"""
        
        self.log_event("file_create", "🔒_RANSOM_NOTE.txt", len(ransom_note), "ransom demand")
        
        if not self.args.dry_run:
            note_path = attack_dir / "🔒_RANSOM_NOTE.txt"
            with open(note_path, 'w', encoding='utf-8') as f:
                f.write(ransom_note)
                
    def log_event(self, event_type, filename, size_bytes, note):
        """Log simulation event"""
        event = {
            "timestamp": time.time(),
            "iso": datetime.now(timezone.utc).isoformat(),
            "attack_id": self.attack_count,
            "event_type": event_type,
            "filename": filename,
            "size_bytes": size_bytes,
            "note": note
        }
        self.events.append(event)
        
        # Print real-time log
        time_str = event["iso"][11:19]
        print(f"[{time_str}] {event_type.upper()}: {filename} ({size_bytes}B) - {note}")
        
    def run_single_attack(self):
        """Run one complete ransomware simulation"""
        self.attack_count += 1
        print(f"\n🚨 ATTACK #{self.attack_count} STARTED 🚨")
        
        # Create attack directory
        attack_dir = self.create_attack_directory()
        print(f"  📂 Attack directory: {attack_dir.name}")
        
        # Generate target files
        print(f"  📄 Generating target files...")
        target_files = self.generate_realistic_files(attack_dir)
        
        # Small delay before encryption starts
        time.sleep(random.uniform(0.5, 1.5))
        
        # Simulate encryption process
        print(f"  🔐 Starting encryption simulation...")
        self.simulate_encryption_process(attack_dir, target_files)
        
        print(f"✅ ATTACK #{self.attack_count} COMPLETED")
        
    def save_continuous_log(self):
        """Save events to log file continuously"""
        if self.args.dry_run or not self.events:
            return
            
        log_path = self.sandbox_path / "continuous_simulation.jsonl"
        
        # Append new events to log file
        with open(log_path, 'a') as f:
            for event in self.events:
                json.dump(event, f)
                f.write('\n')
                
        self.events.clear()  # Clear events after saving
        
    def check_for_stop_command(self):
        """Check if user pressed '1' to stop"""
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline().strip()
            if line == '1':
                print("\n🛑 STOP COMMAND RECEIVED (user pressed '1')")
                self.running = False
                return True
        return False
        
    def print_status(self):
        """Print current simulation status"""
        print(f"\n📊 SIMULATION STATUS:")
        print(f"   Attacks completed: {self.attack_count}")
        print(f"   Files created: {self.total_files_created}")
        print(f"   Files encrypted: {self.total_files_encrypted}")
        print(f"   Mode: {'DRY RUN' if self.args.dry_run else 'FILE CREATION'}")
        print(f"   Press '1' + Enter to stop simulation")
        
    def run_continuous(self):
        """Run continuous simulation"""
        print("🎯 CONTINUOUS RANSOMWARE SIMULATION STARTED")
        print("=" * 60)
        print(f"Interval between attacks: {self.args.interval} seconds")
        print(f"Mode: {'DRY RUN' if self.args.dry_run else 'CREATING REAL FILES'}")
        print("\n🔴 TO STOP: Press '1' followed by Enter")
        print("=" * 60)
        
        self.setup_sandbox()
        
        try:
            while self.running:
                # Run single attack
                self.run_single_attack()
                
                # Save logs
                self.save_continuous_log()
                
                # Status update
                self.print_status()
                
                # Wait for next attack with stop check
                print(f"\n⏳ Waiting {self.args.interval}s for next attack...")
                
                for _ in range(int(self.args.interval * 10)):  # Check every 0.1s
                    if self.check_for_stop_command():
                        break
                    time.sleep(0.1)
                    
                if not self.running:
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 SIMULATION STOPPED (Ctrl+C)")
            self.running = False
            
        # Final summary
        print(f"\n🏁 SIMULATION COMPLETED")
        print(f"   Total attacks: {self.attack_count}")
        print(f"   Total files created: {self.total_files_created}")
        print(f"   Total files encrypted: {self.total_files_encrypted}")
        print(f"   Sandbox location: {self.sandbox_path}")
        
        if not self.args.dry_run:
            print(f"\n💾 Check {self.sandbox_path} for all simulation data")
            
    def run_test(self):
        """Quick self-test"""
        print("🧪 Running self-test...")
        
        old_dry_run = self.args.dry_run
        self.args.dry_run = True
        
        self.setup_sandbox()
        
        # Run one quick attack
        self.run_single_attack()
        
        if self.attack_count == 1 and len(self.events) > 0:
            print("✅ Self-test passed - simulation working correctly")
        else:
            print("❌ Self-test failed")
            
        self.args.dry_run = old_dry_run

def main():
    parser = argparse.ArgumentParser(
        description="Continuous ransomware simulation for RL agent training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  python3 simulate.py --continuous                    # Run continuous simulation
  python3 simulate.py --continuous --interval 3       # Attack every 3 seconds  
  python3 simulate.py --continuous --allow-writes     # Create real files
  python3 simulate.py --test                          # Self-test
        """
    )
    
    parser.add_argument("--continuous", action="store_true",
                       help="Run continuous simulation until stopped")
    parser.add_argument("--sandbox", default=None,
                       help="Sandbox directory (default: auto-create)")
    parser.add_argument("--interval", type=float, default=5.0,
                       help="Seconds between attacks (default: 5)")
    parser.add_argument("--dry-run", action="store_true", default=True,
                       help="Dry run - no files created (default)")
    parser.add_argument("--allow-writes", action="store_true",
                       help="Allow file creation")
    parser.add_argument("--test", action="store_true",
                       help="Run self-test")
                       
    args = parser.parse_args()
    
    if args.allow_writes:
        args.dry_run = False
        
    if args.interval < 0.5:
        print("ERROR: Minimum interval is 0.5 seconds")
        sys.exit(1)
        
    try:
        simulator = ContinuousRansomSimulator(args)
        
        if args.test:
            simulator.run_test()
        elif args.continuous:
            simulator.run_continuous()
        else:
            print("Use --continuous to run simulation or --test for self-test")
            parser.print_help()
            
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()