#!/usr/bin/env python3
"""
refer_env.py
Complete DRL System with CHECKPOINT & RESUME functionality
Agent never loses learning - automatically saves and resumes training
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import time
import os
import subprocess
import json
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
from datetime import datetime
import csv
import shutil
import pickle

# Import stable-baselines3 for PPO
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.logger import configure
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False
    print("⚠ Warning: stable-baselines3 not installed. Install with: pip install stable-baselines3")

# Import process monitoring
try:
    from get_process import get_processes
    PROCESS_MONITOR_AVAILABLE = True
except ImportError:
    PROCESS_MONITOR_AVAILABLE = False
    print("⚠ Warning: get_process.py not found. Using CSV-only mode.")


# ============== CHECKPOINT MANAGER ==============

class CheckpointManager:
    """
    Manages training checkpoints and resume functionality
    Ensures agent never loses learning progress
    """
    
    def __init__(self, checkpoint_dir='checkpoints', max_checkpoints=5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.metadata_file = self.checkpoint_dir / 'training_metadata.json'
        self.best_model_path = self.checkpoint_dir / 'best_model.zip'
        
    def save_checkpoint(self, model, episode, total_timesteps, mean_reward, 
                       additional_info=None):
        """
        Save a training checkpoint
        
        Args:
            model: PPO model to save
            episode: Current episode number
            total_timesteps: Total timesteps trained
            mean_reward: Mean reward for this checkpoint
            additional_info: Additional metadata to save
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_name = f'checkpoint_ep{episode}_ts{total_timesteps}_{timestamp}'
        checkpoint_path = self.checkpoint_dir / f'{checkpoint_name}.zip'
        
        # Save model
        model.save(checkpoint_path)
        
        # Load existing metadata
        metadata = self.load_metadata()
        
        # Add new checkpoint
        checkpoint_info = {
            'checkpoint_name': checkpoint_name,
            'checkpoint_path': str(checkpoint_path),
            'episode': episode,
            'total_timesteps': total_timesteps,
            'mean_reward': mean_reward,
            'timestamp': timestamp,
            'datetime': datetime.now().isoformat(),
        }
        
        if additional_info:
            checkpoint_info.update(additional_info)
        
        metadata['checkpoints'].append(checkpoint_info)
        metadata['last_checkpoint'] = checkpoint_info
        metadata['total_episodes'] = episode
        metadata['total_timesteps'] = total_timesteps
        
        # Update best model if this is better
        if mean_reward > metadata.get('best_reward', float('-inf')):
            metadata['best_reward'] = mean_reward
            metadata['best_checkpoint'] = checkpoint_info
            shutil.copy(checkpoint_path, self.best_model_path)
            print(f"🏆 New best model! Reward: {mean_reward:.2f}")
        
        # Save metadata
        self.save_metadata(metadata)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        print(f"💾 Checkpoint saved: {checkpoint_name}")
        return checkpoint_path
    
    def load_latest_checkpoint(self):
        """
        Load the most recent checkpoint
        
        Returns:
            tuple: (model_path, metadata) or (None, None) if no checkpoint exists
        """
        metadata = self.load_metadata()
        
        if not metadata.get('last_checkpoint'):
            return None, None
        
        last_checkpoint = metadata['last_checkpoint']
        checkpoint_path = last_checkpoint['checkpoint_path']
        
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Checkpoint file not found: {checkpoint_path}")
            return None, None
        
        return checkpoint_path, last_checkpoint
    
    def load_best_checkpoint(self):
        """Load the best performing checkpoint"""
        if self.best_model_path.exists():
            metadata = self.load_metadata()
            return str(self.best_model_path), metadata.get('best_checkpoint')
        return None, None
    
    def load_metadata(self):
        """Load training metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        
        # Default metadata
        return {
            'checkpoints': [],
            'last_checkpoint': None,
            'best_checkpoint': None,
            'best_reward': float('-inf'),
            'total_episodes': 0,
            'total_timesteps': 0,
            'training_sessions': []
        }
    
    def save_metadata(self, metadata):
        """Save training metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _cleanup_old_checkpoints(self):
        """Keep only the most recent N checkpoints"""
        metadata = self.load_metadata()
        checkpoints = metadata.get('checkpoints', [])
        
        if len(checkpoints) > self.max_checkpoints:
            # Sort by timestamp
            checkpoints.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            # Remove old checkpoints
            for old_checkpoint in checkpoints[self.max_checkpoints:]:
                old_path = Path(old_checkpoint['checkpoint_path'])
                if old_path.exists() and old_path != self.best_model_path:
                    try:
                        old_path.unlink()
                        print(f"🗑️  Removed old checkpoint: {old_checkpoint['checkpoint_name']}")
                    except Exception as e:
                        print(f"⚠️  Failed to remove {old_path}: {e}")
            
            # Update metadata
            metadata['checkpoints'] = checkpoints[:self.max_checkpoints]
            self.save_metadata(metadata)
    
    def get_training_summary(self):
        """Get a summary of training progress"""
        metadata = self.load_metadata()
        
        print(f"\n{'='*80}")
        print("📊 TRAINING PROGRESS SUMMARY")
        print(f"{'='*80}")
        print(f"Total Episodes Trained:    {metadata.get('total_episodes', 0)}")
        print(f"Total Timesteps:           {metadata.get('total_timesteps', 0):,}")
        print(f"Best Reward Achieved:      {metadata.get('best_reward', 0):.3f}")
        print(f"Number of Checkpoints:     {len(metadata.get('checkpoints', []))}")
        
        if metadata.get('last_checkpoint'):
            last = metadata['last_checkpoint']
            print(f"\n📍 Last Checkpoint:")
            print(f"   Episode:                {last.get('episode', 0)}")
            print(f"   Timesteps:              {last.get('total_timesteps', 0):,}")
            print(f"   Mean Reward:            {last.get('mean_reward', 0):.3f}")
            print(f"   Date:                   {last.get('datetime', 'N/A')}")
        
        if metadata.get('best_checkpoint'):
            best = metadata['best_checkpoint']
            print(f"\n🏆 Best Checkpoint:")
            print(f"   Episode:                {best.get('episode', 0)}")
            print(f"   Mean Reward:            {best.get('mean_reward', 0):.3f}")
            print(f"   Date:                   {best.get('datetime', 'N/A')}")
        
        print(f"{'='*80}\n")
        
        return metadata


# ============== ENVIRONMENT (Same as before) ==============

class OSProcessMonitorEnv(gym.Env):
    """Custom OpenAI Gym Environment for OS Process Monitoring"""

    metadata = {'render.modes': ['human']}

    CRITICAL_PROCESSES = {
        'init', 'kthreadd', 'ksoftirqd', 'rcu_gp', 'rcu_par_gp', 'kworker',
        'systemd', 'kernel', 'swapper', 'migration', 'watchdog', 'systemd-logind',
        'systemd-networkd', 'systemd-resolved', 'dbus', 'NetworkManager',
        'sshd', 'cron', 'rsyslog', 'udev', 'polkitd', 'chronyd',
        'gdm', 'x11', 'lightdm', 'pulseaudio', 'cups', 'bluetooth', 'avahi'
    }

    def __init__(self,
                 csv_file='enhances_process_monitor.csv',
                 max_processes=50,
                 snapshot_base=None,
                 snapshot_log_file='rsync_snapshots.json',
                 update_csv_on_step=True):

        super(OSProcessMonitorEnv, self).__init__()

        self.csv_file = csv_file
        self.max_processes = max_processes
        self.update_csv_on_step = update_csv_on_step
        
        if snapshot_base is None:
            home_dir = os.path.expanduser('~')
            self.snapshot_base = os.path.join(home_dir, 'ransomware_snapshots')
        else:
            self.snapshot_base = snapshot_base
            
        self.snapshot_log_file = snapshot_log_file

        logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger(__name__)

        try:
            Path(self.snapshot_base).mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise PermissionError(
                f"Cannot create snapshot directory at {self.snapshot_base}."
            ) from e

        if not os.path.exists(self.csv_file) and PROCESS_MONITOR_AVAILABLE:
            get_processes(self.csv_file, verbose=False)
        elif not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"CSV file {self.csv_file} not found.")

        self.action_space = spaces.Discrete(4)
        obs_shape = (self.max_processes, 19)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

        self.current_processes = {}
        self.step_count = 0
        self.total_reward = 0
        self.killed_processes = set()
        self.current_data_index = 0
        self.snapshots_taken = []
        self.snapshot_count = 0
        
        self.episode_kills = 0
        self.episode_snapshots = 0
        self.episode_critical_kills = 0
        self.episode_ransomware_detected = 0

    def _create_rsync_snapshot(self, snapshot_name: str, directories: List[str]) -> Tuple[bool, str]:
        try:
            snapshot_dir = os.path.join(self.snapshot_base, snapshot_name)
            Path(snapshot_dir).mkdir(parents=True, exist_ok=True)

            for dir_path in directories:
                if not os.access(dir_path, os.R_OK):
                    continue
                    
                if os.path.exists(dir_path):
                    target_dir = os.path.join(snapshot_dir, os.path.basename(dir_path))
                    cmd = ['rsync', '-a', '--delete', dir_path + '/', target_dir + '/']
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    if result.returncode != 0:
                        return False, snapshot_dir

            return True, snapshot_dir
        except Exception:
            return False, ""

    def _save_snapshot_log(self, snapshot_info: Dict):
        try:
            if os.path.exists(self.snapshot_log_file):
                with open(self.snapshot_log_file, 'r') as f:
                    snapshots = json.load(f)
            else:
                snapshots = []

            snapshots.append(snapshot_info)
            with open(self.snapshot_log_file, 'w') as f:
                json.dump(snapshots, f, indent=2)
        except Exception:
            pass

    def _take_rsync_snapshot(self, directories: List[str], process_info: Dict) -> bool:
        try:
            snapshot_name = f"ransomware_snap_{self.snapshot_count}_{int(time.time())}"
            self.snapshot_count += 1

            snapshot_info = {
                'snapshot_name': snapshot_name,
                'timestamp': time.time(),
                'timestamp_readable': time.strftime('%Y-%m-%d %H:%M:%S'),
                'pid': process_info['pid'],
                'process_name': process_info['process_name'],
                'directories': directories,
                'threat_score': process_info['threat_score'],
                'ransomware_confidence': process_info.get('ransomware_confidence', 0.0),
                'snapshot_path': None,
                'status': 'failed'
            }

            success, snapshot_path = self._create_rsync_snapshot(snapshot_name, directories)
            if success:
                snapshot_info['snapshot_path'] = snapshot_path
                snapshot_info['status'] = 'created'
                self._save_snapshot_log(snapshot_info)
                self.snapshots_taken.append(snapshot_info)
                self.episode_snapshots += 1
                return True
            return False
        except Exception:
            return False

    def _update_csv_data(self):
        if PROCESS_MONITOR_AVAILABLE and self.update_csv_on_step:
            try:
                get_processes(self.csv_file, verbose=False)
            except Exception:
                pass

    def _is_critical_process(self, process_name: str) -> bool:
        if pd.isna(process_name) or process_name == 'unknown':
            return False
        return any(c.lower() in process_name.lower() for c in self.CRITICAL_PROCESSES)

    def _load_latest_csv_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.csv_file)
            if df.empty:
                return pd.DataFrame()

            if 'pid' not in df.columns:
                if 'PID' in df.columns:
                    df['pid'] = df['PID']
                else:
                    df['pid'] = np.arange(len(df))

            if 'process_name' not in df.columns:
                df['process_name'] = df.get('Name', 'unknown')

            return df
        except Exception:
            return pd.DataFrame()

    def _detect_ransomware_behavior(self, row) -> Tuple[bool, float, List[str]]:
        indicators = 0
        confidence = 0.0
        suspicious_dirs = []
        try:
            if float(row.get('entropy_sample', 0)) > 7.0:
                indicators += 3; confidence += 0.3
            if float(row.get('rename_count', 0)) > 5:
                indicators += 2; confidence += 0.2
            if float(row.get('delete_count', 0)) > 3:
                indicators += 1; confidence += 0.1
            if float(row.get('write_bytes_per_s', 0)) > 5000000:
                indicators += 2; confidence += 0.2
            if float(row.get('num_dirs_touched', 0)) > 5:
                indicators += 1; confidence += 0.1
            if float(row.get('enc_heuristic_score', 0)) > 0.7:
                indicators += 2; confidence += 0.2

            is_ransomware = indicators >= 5
            if is_ransomware:
                home = os.path.expanduser('~')
                suspicious_dirs = [
                    os.path.join(home, 'Documents'),
                    os.path.join(home, 'Desktop'),
                    os.path.join(home, 'Downloads')
                ]
                self.episode_ransomware_detected += 1
        except Exception:
            pass
        return is_ransomware, min(confidence, 1.0), suspicious_dirs

    def _calculate_threat_score(self, row) -> float:
        score = 0.0
        try:
            score += min(float(row.get('entropy_sample', 0)) / 8.0, 1.0) * 0.3
            score += min(float(row.get('enc_heuristic_score', 0)), 1.0) * 0.25
            score += min(abs(float(row.get('host_baseline_zscore', 0)) / 5.0), 1.0) * 0.2
            score += (float(row.get('suspicious_filename_flag', 0)) +
                      float(row.get('cmdline_keyword_flag', 0))) * 0.125
        except Exception:
            pass
        return min(max(score, 0.0), 1.0)

    def _get_current_processes(self) -> pd.DataFrame:
        df = self._load_latest_csv_data()
        if df.empty:
            return df
        if self.killed_processes:
            df = df[~df['pid'].isin(self.killed_processes)]
        start, end = self.current_data_index, min(len(df), self.current_data_index + self.max_processes)
        self.current_data_index = (end) % len(df) if len(df) > 0 else 0
        return df.iloc[start:end].copy()

    def _get_observation(self) -> np.ndarray:
        if self.update_csv_on_step:
            self._update_csv_data()
        
        df = self._get_current_processes()
        obs = np.zeros((self.max_processes, 19), dtype=np.float32)
        self.current_processes = {}

        for i, (_, row) in enumerate(df.iterrows()):
            if i >= self.max_processes:
                break
            name = str(row.get('process_name', 'unknown'))
            is_critical = self._is_critical_process(name)
            threat = self._calculate_threat_score(row)
            is_ransom, conf, dirs = self._detect_ransomware_behavior(row)

            obs[i] = np.array([
                float(row.get('cpu_percent', 0)),
                float(row.get('mem_percent', 0)),
                float(row.get('write_bytes_per_s', 0)) / 1000000.0,
                float(row.get('read_bytes_per_s', 0)) / 1000000.0,
                float(row.get('write_read_ratio', 0)),
                float(row.get('entropy_sample', 0)) / 8.0,
                float(row.get('enc_heuristic_score', 0)) / 10.0,
                float(row.get('num_dirs_touched', 0)) / 100.0,
                float(row.get('suspicious_filename_flag', 0)),
                float(row.get('cmdline_keyword_flag', 0)),
                float(row.get('rename_count', 0)) / 10.0,
                float(row.get('delete_count', 0)) / 10.0,
                float(row.get('open_fds_delta_per_s', 0)),
                float(row.get('avg_write_size_estimate', 0)) / 100000.0,
                float(row.get('host_baseline_zscore', 0)) / 5.0,
                float(is_critical),
                threat,
                is_ransom,
                conf
            ], dtype=np.float32)
            
            self.current_processes[i] = {
                'pid': int(row.get('pid', 0)),
                'process_name': name,
                'is_critical': is_critical,
                'threat_score': threat,
                'is_ransomware': is_ransom,
                'ransomware_confidence': conf,
                'suspicious_dirs': dirs
            }

        return obs

    def _calculate_reward(self, action: int, idx: int = None) -> float:
        if action == 0:
            return 0.1
        if action == 1:
            if idx is not None and idx in self.current_processes:
                p = self.current_processes[idx]
                if p['threat_score'] > 0.8:
                    return -2.0
                elif p['is_ransomware']:
                    return -5.0
            return -0.05
        if action == 2:
            if idx is not None and idx in self.current_processes:
                p = self.current_processes[idx]
                if p['is_critical']:
                    return -20.0
                elif p['is_ransomware']:
                    return 15.0
                elif p['threat_score'] > 0.8:
                    return 8.0
                elif p['threat_score'] > 0.5:
                    return 3.0
                else:
                    return -2.0
        if action == 3:
            if idx is not None and idx in self.current_processes:
                p = self.current_processes[idx]
                if p['is_ransomware']:
                    return 10.0
                elif p['threat_score'] > 0.7:
                    return 5.0
                elif p['threat_score'] > 0.4:
                    return 1.0
                else:
                    return -1.0
        return 0.0

    def step(self, action: int):
        self.step_count += 1
        obs = self._get_observation()
        idx = None
        
        if self.current_processes:
            idx = max(self.current_processes.keys(), 
                     key=lambda x: self.current_processes[x]['threat_score'])
        
        reward = self._calculate_reward(action, idx)

        if action == 2 and idx is not None:
            p = self.current_processes[idx]
            if p['is_critical']:
                self.episode_critical_kills += 1
            else:
                self.killed_processes.add(p['pid'])
                self.episode_kills += 1

        elif action == 3 and idx is not None:
            p = self.current_processes[idx]
            if p['suspicious_dirs']:
                self._take_rsync_snapshot(p['suspicious_dirs'], p)

        done = self.step_count >= 200
        info = {
            'step': self.step_count,
            'reward_total': self.total_reward,
            'snapshots': self.episode_snapshots,
            'kills': self.episode_kills,
            'critical_kills': self.episode_critical_kills,
            'ransomware_detected': self.episode_ransomware_detected
        }
        self.total_reward += reward
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.total_reward = 0
        self.killed_processes = set()
        self.snapshots_taken = []
        self.snapshot_count = 0
        self.current_processes = {}
        self.current_data_index = 0
        self.episode_kills = 0
        self.episode_snapshots = 0
        self.episode_critical_kills = 0
        self.episode_ransomware_detected = 0
        
        if self.update_csv_on_step:
            self._update_csv_data()
        
        return self._get_observation(), {}


# ============== ENHANCED TRAINING CALLBACK WITH CHECKPOINTS ==============

class DetailedTrainingCallback(BaseCallback):
    """Enhanced callback with checkpoint support"""
    
    def __init__(self, checkpoint_manager, log_dir='training_logs', 
                 checkpoint_freq=50, check_freq=10, verbose=1):
        super(DetailedTrainingCallback, self).__init__(verbose)
        self.checkpoint_manager = checkpoint_manager
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.checkpoint_freq = checkpoint_freq  # Save checkpoint every N episodes
        self.check_freq = check_freq
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_file = self.log_dir / f'training_log_{timestamp}.csv'
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Episode', 'Total_Steps', 'Episode_Reward', 'Episode_Length',
                'Mean_Reward_100', 'Kills', 'Snapshots', 'Critical_Kills', 'Ransomware_Detected'
            ])
        
        print(f"📊 Training logs: {self.csv_file}")
    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        if len(self.model.ep_info_buffer) > 0:
            for ep_info in self.model.ep_info_buffer[-self.check_freq:]:
                if ep_info not in self.episode_rewards:
                    self.episode_count += 1
                    
                    reward = ep_info.get('r', 0)
                    length = ep_info.get('l', 0)
                    
                    self.episode_rewards.append(ep_info)
                    self.episode_lengths.append(length)
                    
                    mean_reward_100 = np.mean([e['r'] for e in self.model.ep_info_buffer[-100:]]) if len(self.model.ep_info_buffer) > 0 else 0
                    
                    with open(self.csv_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            self.episode_count,
                            self.num_timesteps,
                            f"{reward:.3f}",
                            length,
                            f"{mean_reward_100:.3f}",
                            0, 0, 0, 0
                        ])
                    
                    # ===== AUTO-CHECKPOINT EVERY N EPISODES =====
                    if self.episode_count % self.checkpoint_freq == 0:
                        self.checkpoint_manager.save_checkpoint(
                            model=self.model,
                            episode=self.episode_count,
                            total_timesteps=self.num_timesteps,
                            mean_reward=mean_reward_100,
                            additional_info={
                                'recent_reward': reward,
                                'episode_length': length
                            }
                        )
                    
                    if self.episode_count % 10 == 0 and self.verbose:
                        print(f"\n[Episode {self.episode_count:4d}] "
                              f"Steps: {self.num_timesteps:7d} | "
                              f"Reward: {reward:7.2f} | "
                              f"Mean(100): {mean_reward_100:7.2f}")


# ============== ENHANCED TRAIN FUNCTION WITH RESUME ==============

def train(episodes=1000, 
          timesteps_per_episode=200, 
          save_path='models/ppo_ransomware_agent',
          log_dir='training_logs',
          max_processes=10,
          csv_file='enhances_process_monitor.csv',
          verbose=True,
          checkpoint_freq=50,
          resume=True):  # ⬅️ NEW: Auto-resume from checkpoint
    """
    🔄 ENHANCED TRAINING WITH AUTO-CHECKPOINT & RESUME
    
    The agent NEVER loses learning! Features:
    - ✅ Auto-saves checkpoints every N episodes
    - ✅ Auto-resumes from last checkpoint
    - ✅ Keeps training history across sessions
    - ✅ Saves best model automatically
    
    Usage:
        # First training session
        train(episodes=100)
        
        # Later, resume training (picks up where it left off!)
        train(episodes=200, resume=True)  # Continues from episode 100
        
        # Train more
        train(episodes=100, resume=True)  # Continues from episode 200
    
    Args:
        episodes: NEW episodes to train (adds to existing)
        timesteps_per_episode: Max steps per episode
        save_path: Path to save final model
        log_dir: Directory for logs
        max_processes: Max processes to monitor
        csv_file: CSV file for process data
        verbose: Print progress
        checkpoint_freq: Save checkpoint every N episodes
        resume: If True, resumes from last checkpoint
    
    Returns:
        dict with training results
    """
    
    if not PPO_AVAILABLE:
        print("❌ ERROR: stable-baselines3 not installed!")
        return None
    
    # ===== CHECKPOINT MANAGER =====
    checkpoint_manager = CheckpointManager(
        checkpoint_dir='checkpoints',
        max_checkpoints=5
    )
    
    # ===== CHECK FOR EXISTING CHECKPOINT =====
    model = None
    start_episode = 0
    start_timesteps = 0
    
    if resume:
        checkpoint_path, checkpoint_info = checkpoint_manager.load_latest_checkpoint()
        
        if checkpoint_path:
            print(f"\n{'='*80}")
            print("🔄 RESUMING FROM CHECKPOINT")
            print(f"{'='*80}")
            print(f"📍 Checkpoint found: {Path(checkpoint_path).name}")
            print(f"   Previous Episodes:      {checkpoint_info.get('episode', 0)}")
            print(f"   Previous Timesteps:     {checkpoint_info.get('total_timesteps', 0):,}")
            print(f"   Previous Mean Reward:   {checkpoint_info.get('mean_reward', 0):.3f}")
            print(f"   Date:                   {checkpoint_info.get('datetime', 'N/A')}")
            print(f"{'='*80}\n")
            
            try:
                # ===== LOAD MODEL FROM CHECKPOINT =====
                print("📂 Loading model from checkpoint...")
                model = PPO.load(checkpoint_path)
                start_episode = checkpoint_info.get('episode', 0)
                start_timesteps = checkpoint_info.get('total_timesteps', 0)
                print("✅ Model loaded successfully!")
                print(f"🎯 Will continue training from episode {start_episode + 1}")
                
            except Exception as e:
                print(f"⚠️  Failed to load checkpoint: {e}")
                print("🔄 Starting fresh training instead...")
                model = None
        else:
            print("\n📝 No checkpoint found. Starting fresh training...")
    else:
        print("\n🆕 Starting fresh training (resume=False)")
    
    # ===== CALCULATE TOTAL TRAINING =====
    target_episodes = start_episode + episodes
    total_timesteps = episodes * timesteps_per_episode
    
    if verbose:
        print(f"\n{'='*80}")
        print("🚀 TRAINING CONFIGURATION")
        print(f"{'='*80}")
        print(f"📊 Training Plan:")
        if start_episode > 0:
            print(f"   Already Completed:     {start_episode} episodes")
        print(f"   New Episodes:          {episodes}")
        print(f"   Target Total:          {target_episodes} episodes")
        print(f"   Steps per Episode:     {timesteps_per_episode}")
        print(f"   New Timesteps:         {total_timesteps:,}")
        print(f"\n💾 Checkpoint Settings:")
        print(f"   Checkpoint Frequency:  Every {checkpoint_freq} episodes")
        print(f"   Max Checkpoints Kept:  5")
        print(f"   Auto-Resume:           {'✅ ON' if resume else '⚠️ OFF'}")
        print(f"\n📁 Paths:")
        print(f"   Final Model:           {save_path}")
        print(f"   Checkpoints:           checkpoints/")
        print(f"   Logs:                  {log_dir}/")
        print(f"{'='*80}\n")
    
    # ===== CREATE ENVIRONMENT =====
    try:
        env = OSProcessMonitorEnv(
            csv_file=csv_file,
            max_processes=max_processes,
            update_csv_on_step=True
        )
        print("✅ Environment created")
    except Exception as e:
        print(f"❌ Environment error: {e}")
        return None
    
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    # ===== CREATE OR UPDATE MODEL =====
    if model is None:
        print("\n🤖 Creating new PPO agent...")
        try:
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1 if verbose else 0,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                tensorboard_log="./tensorboard_logs/"
            )
            print("✅ PPO agent created")
        except Exception as e:
            print(f"❌ Failed to create agent: {e}")
            return None
    else:
        # Update environment for loaded model
        model.set_env(env)
        print("✅ Checkpoint model ready to continue training")
    
    # ===== CREATE CALLBACK =====
    callback = DetailedTrainingCallback(
        checkpoint_manager=checkpoint_manager,
        log_dir=log_dir,
        checkpoint_freq=checkpoint_freq,
        verbose=1 if verbose else 0
    )
    
    # Update episode count if resuming
    callback.episode_count = start_episode
    
    # ===== START TRAINING =====
    print(f"\n{'='*80}")
    print(f"🎯 TRAINING START")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=verbose,
            reset_num_timesteps=False  # ⬅️ Don't reset timesteps (accumulate)
        )
        
        training_time = time.time() - start_time
        
        # ===== SAVE FINAL MODEL =====
        print(f"\n💾 Saving final model...")
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        print(f"✅ Final model saved: {save_path}.zip")
        
        # ===== SAVE FINAL CHECKPOINT =====
        episode_rewards = [ep['r'] for ep in model.ep_info_buffer] if len(model.ep_info_buffer) > 0 else [0]
        mean_reward = np.mean(episode_rewards)
        
        checkpoint_manager.save_checkpoint(
            model=model,
            episode=callback.episode_count,
            total_timesteps=callback.num_timesteps,
            mean_reward=mean_reward,
            additional_info={'final_checkpoint': True}
        )
        
        # ===== SHOW SUMMARY =====
        print(f"\n{'='*80}")
        print("🎉 TRAINING COMPLETE!")
        print(f"{'='*80}")
        print(f"⏱️  Session Time:           {training_time:.1f}s ({training_time/60:.1f} min)")
        print(f"📊 Episodes This Session:  {episodes}")
        print(f"📊 Total Episodes:         {callback.episode_count}")
        print(f"📊 Total Timesteps:        {callback.num_timesteps:,}")
        print(f"📊 Mean Reward:            {mean_reward:.3f}")
        print(f"{'='*80}\n")
        
        # ===== SHOW TRAINING HISTORY =====
        checkpoint_manager.get_training_summary()
        
        return {
            'model': model,
            'session_episodes': episodes,
            'total_episodes': callback.episode_count,
            'total_timesteps': callback.num_timesteps,
            'training_time': training_time,
            'mean_reward': float(mean_reward),
            'save_path': save_path,
            'checkpoint_dir': 'checkpoints/'
        }
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted!")
        print("💾 Saving emergency checkpoint...")
        
        episode_rewards = [ep['r'] for ep in model.ep_info_buffer] if len(model.ep_info_buffer) > 0 else [0]
        checkpoint_manager.save_checkpoint(
            model=model,
            episode=callback.episode_count,
            total_timesteps=callback.num_timesteps,
            mean_reward=np.mean(episode_rewards),
            additional_info={'interrupted': True}
        )
        
        print("✅ Checkpoint saved! You can resume training later.")
        return None
        
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ===== NEW FUNCTION: VIEW CHECKPOINTS =====

def checkpoint_info():
    """
    📊 Display checkpoint information
    
    Usage:
        from new_refer_env import checkpoint_info
        checkpoint_info()
    """
    manager = CheckpointManager()
    manager.get_training_summary()


# ===== NEW FUNCTION: RESET TRAINING =====

def reset_training(confirm=False):
    """
    🗑️  Delete all checkpoints and start fresh
    
    Usage:
        from new_refer_env import reset_training
        reset_training(confirm=True)  # Must set confirm=True
    
    Args:
        confirm: Must be True to actually delete
    """
    if not confirm:
        print("⚠️  To reset training, call: reset_training(confirm=True)")
        return
    
    checkpoint_dir = Path('checkpoints')
    if checkpoint_dir.exists():
        import shutil
        shutil.rmtree(checkpoint_dir)
        print("✅ All checkpoints deleted. Training will start fresh.")
    else:
        print("📝 No checkpoints found. Already clean!")


# ============== EVALUATION (Same as before) ==============

def evaluate(model_path='models/ppo_ransomware_agent', 
             n_episodes=10, 
             csv_file='enhances_process_monitor.csv',
             max_processes=10,
             render=True,
             use_best=False):  # ⬅️ NEW: Option to use best checkpoint
    """
    Evaluate a trained agent
    
    Args:
        model_path: Path to model (or use_best=True for best checkpoint)
        n_episodes: Episodes to evaluate
        csv_file: CSV file
        max_processes: Max processes
        render: Show details
        use_best: If True, uses best checkpoint instead of model_path
    """
    
    if not PPO_AVAILABLE:
        print("❌ stable-baselines3 not installed!")
        return None
    
    # ===== OPTION TO USE BEST CHECKPOINT =====
    if use_best:
        manager = CheckpointManager()
        best_path, best_info = manager.load_best_checkpoint()
        if best_path:
            model_path = best_path
            print(f"\n🏆 Using BEST checkpoint:")
            print(f"   Reward: {best_info.get('mean_reward', 0):.3f}")
            print(f"   Episode: {best_info.get('episode', 0)}")
        else:
            print("⚠️  No best checkpoint found, using specified path")
    
    print(f"\n{'='*80}")
    print("🔍 EVALUATION")
    print(f"{'='*80}")
    print(f"   Model: {Path(model_path).name}")
    print(f"   Episodes: {n_episodes}")
    print(f"{'='*80}\n")
    
    try:
        model = PPO.load(model_path)
        print("✅ Model loaded\n")
    except Exception as e:
        print(f"❌ Load error: {e}")
        return None
    
    try:
        env = OSProcessMonitorEnv(
            csv_file=csv_file,
            max_processes=max_processes,
            update_csv_on_step=True
        )
    except Exception as e:
        print(f"❌ Environment error: {e}")
        return None
    
    action_names = ['Monitor', 'Ignore', 'Kill', 'Snapshot']
    episode_rewards = []
    episode_lengths = []
    total_kills = 0
    total_snapshots = 0
    total_critical_kills = 0
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step = 0
        
        if render:
            print(f"{'─'*80}")
            print(f"Episode {episode + 1}/{n_episodes}")
            print(f"{'─'*80}")
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            
            if render and step % 50 == 0:
                print(f"  Step {step:3d}: {action_names[action]:10s} | Reward: {reward:6.2f} | Total: {episode_reward:7.2f}")
            
            if done or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        total_kills += info.get('kills', 0)
        total_snapshots += info.get('snapshots', 0)
        total_critical_kills += info.get('critical_kills', 0)
        
        if render:
            print(f"  Result: Reward={episode_reward:.2f}, Steps={step}\n")
    
    print(f"\n{'='*80}")
    print("📊 EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Mean Reward:               {np.mean(episode_rewards):.3f}")
    print(f"Std Reward:                {np.std(episode_rewards):.3f}")
    print(f"Best Reward:               {max(episode_rewards):.3f}")
    print(f"Worst Reward:              {min(episode_rewards):.3f}")
    print(f"Mean Length:               {np.mean(episode_lengths):.1f} steps")
    print(f"Total Kills:               {total_kills}")
    print(f"Total Snapshots:           {total_snapshots}")
    print(f"Critical Kills:            {total_critical_kills} {'⚠️' if total_critical_kills > 0 else '✅'}")
    print(f"{'='*80}\n")
    
    return {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'max_reward': float(max(episode_rewards)),
        'min_reward': float(min(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths))
    }


# ============== MAIN ==============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PPO Agent with Checkpoint System')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'info', 'reset'], default='train')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Episodes to train (adds to existing if resuming)')
    parser.add_argument('--model-path', type=str, default='models/ppo_ransomware_agent')
    parser.add_argument('--eval-episodes', type=int, default=10)
    parser.add_argument('--checkpoint-freq', type=int, default=50,
                       help='Save checkpoint every N episodes')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh (ignore checkpoints)')
    parser.add_argument('--use-best', action='store_true',
                       help='Use best checkpoint for evaluation')
    parser.add_argument('--reset-confirm', action='store_true',
                       help='Confirm reset of all checkpoints')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print(f"\n🎓 Training Mode\n")
        train(
            episodes=args.episodes,
            save_path=args.model_path,
            checkpoint_freq=args.checkpoint_freq,
            resume=not args.no_resume
        )
    
    elif args.mode == 'evaluate':
        print(f"\n🔍 Evaluation Mode\n")
        evaluate(
            model_path=args.model_path,
            n_episodes=args.eval_episodes,
            use_best=args.use_best
        )
    
    elif args.mode == 'info':
        checkpoint_info()
    
    elif args.mode == 'reset':
        reset_training(confirm=args.reset_confirm)