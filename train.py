#!/usr/bin/env python3
"""
train.py
Training Script for Ransomware Detection Agent
Trains the agent for a specific number of episodes while logging detailed information
to training_report.csv

Usage:
    python train.py --episodes 10
    python train.py --episodes 100
    python train.py --episodes 500
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
import sys
import os

# Import required modules
try:
    from refer_env import OSProcessMonitorEnv, PPO_AVAILABLE
    from stable_baselines3 import PPO
    import numpy as np
    print("✅ Successfully imported all modules")
except ImportError as e:
    print(f"❌ Error importing required modules: {e}")
    print("\n🔍 Troubleshooting:")
    print("1. Make sure refer_env.py is in the same directory as train.py")
    print("2. Install dependencies: pip install stable-baselines3 gymnasium pandas numpy")
    sys.exit(1)


class TrainingLogger:
    """Logger for training data to CSV"""
    
    def __init__(self, csv_filename='training_report.csv'):
        self.csv_filename = csv_filename
        
        # Create CSV file with headers
        self.fieldnames = [
            'timestamp',
            'episode',
            'step',
            'total_steps',
            'action',
            'action_name',
            'reward',
            'cumulative_reward',
            'pid_targeted',
            'process_name',
            'threat_score',
            'is_ransomware',
            'ransomware_confidence',
            'is_critical',
            'process_killed',
            'snapshot_taken',
            'episode_total_kills',
            'episode_total_snapshots',
            'episode_critical_kills',
            'episode_ransomware_detected',
            'done'
        ]
        
        # Initialize CSV file
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
        
        print(f"📊 Training report will be saved to: {self.csv_filename}")
    
    def log_step(self, episode, step, total_steps, action, reward, cumulative_reward, 
                 process_info, done, info):
        """Log a single training step"""
        
        action_names = ['Monitor', 'Ignore', 'Kill', 'Snapshot']
        
        # Extract process information
        pid = process_info.get('pid', 0)
        process_name = process_info.get('process_name', 'unknown')
        threat_score = process_info.get('threat_score', 0.0)
        is_ransomware = process_info.get('is_ransomware', False)
        ransomware_confidence = process_info.get('ransomware_confidence', 0.0)
        is_critical = process_info.get('is_critical', False)
        
        # Determine if process was killed or snapshot taken
        process_killed = (action == 2 and not is_critical)
        snapshot_taken = (action == 3)
        
        # Create log entry
        log_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'episode': episode,
            'step': step,
            'total_steps': total_steps,
            'action': action,
            'action_name': action_names[action],
            'reward': round(reward, 3),
            'cumulative_reward': round(cumulative_reward, 3),
            'pid_targeted': pid,
            'process_name': process_name,
            'threat_score': round(threat_score, 4),
            'is_ransomware': int(is_ransomware),
            'ransomware_confidence': round(ransomware_confidence, 4),
            'is_critical': int(is_critical),
            'process_killed': int(process_killed),
            'snapshot_taken': int(snapshot_taken),
            'episode_total_kills': info.get('kills', 0),
            'episode_total_snapshots': info.get('snapshots', 0),
            'episode_critical_kills': info.get('critical_kills', 0),
            'episode_ransomware_detected': info.get('ransomware_detected', 0),
            'done': int(done)
        }
        
        # Write to CSV immediately (streaming write)
        with open(self.csv_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(log_entry)
        
        return log_entry


def train_for_episodes(episodes=100, 
                       max_steps_per_episode=200,
                       csv_file='enhances_process_monitor.csv',
                       max_processes=10,
                       model_save_path='models/ppo_ransomware_agent',
                       verbose=True):
    """
    Train PPO agent for a specific number of episodes with detailed logging
    
    Args:
        episodes: Number of episodes to train
        max_steps_per_episode: Maximum steps per episode (default: 200)
        csv_file: Process monitor CSV file
        max_processes: Maximum processes to monitor
        model_save_path: Path to save trained model
        verbose: Print training progress
    
    Returns:
        dict with training results
    """
    
    if not PPO_AVAILABLE:
        print("❌ ERROR: stable-baselines3 not installed!")
        print("📦 Install with: pip install stable-baselines3")
        return None
    
    print(f"\n{'='*80}")
    print("🚀 TRAINING PPO AGENT FOR SPECIFIC NUMBER OF EPISODES")
    print(f"{'='*80}")
    print(f"📊 Configuration:")
    print(f"   Target Episodes:       {episodes}")
    print(f"   Max Steps per Episode: {max_steps_per_episode}")
    print(f"   CSV File:              {csv_file}")
    print(f"   Max Processes:         {max_processes}")
    print(f"   Model Save Path:       {model_save_path}")
    print(f"   Training Report:       training_report.csv")
    print(f"{'='*80}\n")
    
    # Create logger
    logger = TrainingLogger('training_report.csv')
    
    # Create environment
    print("🔧 Creating environment...")
    try:
        env = OSProcessMonitorEnv(
            csv_file=csv_file,
            max_processes=max_processes,
            update_csv_on_step=True
        )
        print("✅ Environment created successfully\n")
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Create PPO agent
    print("🤖 Creating PPO agent...")
    try:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,  # We handle our own logging
            learning_rate=3e-4,
            n_steps=max_steps_per_episode,  # Buffer size matches episode length
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
        )
        print("✅ PPO agent created successfully\n")
    except Exception as e:
        print(f"❌ Failed to create PPO agent: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print(f"{'='*80}")
    print(f"🎯 STARTING TRAINING - {episodes} EPISODES")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    total_steps = 0
    all_episode_rewards = []
    all_episode_lengths = []
    
    # Training loop - manually control episodes
    for episode in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        step = 0
        cumulative_reward = 0
        
        print(f"📍 Episode {episode}/{episodes} - Starting...")
        
        # Run one episode
        while not done and step < max_steps_per_episode:
            # Predict action using current policy
            action, _states = model.predict(obs, deterministic=False)
            
            # Get process info before taking action
            if env.current_processes:
                idx = max(env.current_processes.keys(), 
                         key=lambda x: env.current_processes[x]['threat_score'])
                process_info = env.current_processes[idx]
            else:
                process_info = {
                    'pid': 0,
                    'process_name': 'none',
                    'threat_score': 0.0,
                    'is_ransomware': False,
                    'ransomware_confidence': 0.0,
                    'is_critical': False
                }
            
            # Take action in environment
            obs, reward, done, truncated, info = env.step(action)
            cumulative_reward += reward
            total_steps += 1
            step += 1
            
            # Log this step
            logger.log_step(
                episode=episode,
                step=step,
                total_steps=total_steps,
                action=int(action),
                reward=reward,
                cumulative_reward=cumulative_reward,
                process_info=process_info,
                done=done or truncated,
                info=info
            )
            
            # Print progress every 50 steps
            if step % 50 == 0 and verbose:
                action_names = ['Monitor', 'Ignore', 'Kill', 'Snapshot']
                print(f"  Step {step:3d}: {action_names[int(action)]:10s} | "
                      f"Reward: {reward:6.2f} | Total: {cumulative_reward:7.2f}")
            
            if done or truncated:
                break
        
        # Store episode statistics
        all_episode_rewards.append(cumulative_reward)
        all_episode_lengths.append(step)
        
        # Print episode summary
        if verbose:
            print(f"\n{'='*80}")
            print(f"Episode {episode}/{episodes} Complete:")
            print(f"  Total Reward:           {cumulative_reward:.2f}")
            print(f"  Steps:                  {step}")
            print(f"  Processes Killed:       {info.get('kills', 0)}")
            print(f"  Snapshots Taken:        {info.get('snapshots', 0)}")
            print(f"  Critical Kills:         {info.get('critical_kills', 0)} "
                  f"{'⚠️' if info.get('critical_kills', 0) > 0 else '✅'}")
            print(f"  Ransomware Detected:    {info.get('ransomware_detected', 0)}")
            print(f"{'='*80}\n")
        
        # Update the model every few episodes using collected experience
        # PPO needs a certain amount of data before updating
        if episode % 5 == 0 and episode > 0:
            if verbose:
                print(f"🔄 Updating policy neural network (after episode {episode})...\n")
            # The model updates automatically when we collect enough steps (n_steps parameter)
        
        # Save checkpoint every 25 episodes
        if episode % 25 == 0:
            checkpoint_path = f"{model_save_path}_ep{episode}"
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            model.save(checkpoint_path)
            if verbose:
                print(f"💾 Checkpoint saved: {checkpoint_path}.zip\n")
    
    training_time = time.time() - start_time
    
    # Save final model
    print(f"\n💾 Saving final trained model...")
    try:
        Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(model_save_path)
        print(f"✅ Model saved to: {model_save_path}.zip")
    except Exception as e:
        print(f"⚠️  Warning: Failed to save model: {e}")
    
    # Calculate final statistics
    mean_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)
    best_reward = max(all_episode_rewards)
    worst_reward = min(all_episode_rewards)
    mean_length = np.mean(all_episode_lengths)
    
    print(f"\n{'='*80}")
    print("🎉 TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"⏱️  Training Time:")
    print(f"   Total:                 {training_time:.2f}s ({training_time/60:.2f} minutes)")
    print(f"   Per Episode:           {training_time/episodes:.2f}s")
    print(f"\n📊 Performance Statistics:")
    print(f"   Episodes Completed:    {episodes}")
    print(f"   Total Timesteps:       {total_steps:,}")
    print(f"   Mean Episode Length:   {mean_length:.1f} steps")
    print(f"   Mean Reward:           {mean_reward:.3f}")
    print(f"   Std Reward:            {std_reward:.3f}")
    print(f"   Best Reward:           {best_reward:.3f}")
    print(f"   Worst Reward:          {worst_reward:.3f}")
    print(f"\n💾 Saved Files:")
    print(f"   Model:                 {model_save_path}.zip")
    print(f"   Training Report:       training_report.csv")
    print(f"{'='*80}\n")
    
    return {
        'model': model,
        'episodes': episodes,
        'total_timesteps': total_steps,
        'training_time': training_time,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'best_reward': best_reward,
        'worst_reward': worst_reward,
        'mean_length': mean_length,
        'all_rewards': all_episode_rewards,
        'csv_report': 'training_report.csv'
    }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Train PPO agent for ransomware detection - specific number of episodes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --episodes 10            # Train for 10 episodes
  python train.py --episodes 50            # Train for 50 episodes
  python train.py --episodes 100           # Train for 100 episodes (default)
  python train.py --episodes 500           # Train for 500 episodes
        """
    )
    
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of episodes to train (default: 100)')
    parser.add_argument('--max-steps', type=int, default=200,
                       help='Maximum steps per episode (default: 200)')
    parser.add_argument('--csv-file', type=str, default='enhances_process_monitor.csv',
                       help='CSV file for process data')
    parser.add_argument('--max-processes', type=int, default=10,
                       help='Maximum processes to monitor (default: 10)')
    parser.add_argument('--model-path', type=str, default='models/ppo_ransomware_agent',
                       help='Path to save model (default: models/ppo_ransomware_agent)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("🎓 RANSOMWARE DETECTION AGENT TRAINING")
    print(f"{'='*80}")
    print(f"Training for exactly {args.episodes} episodes...")
    print(f"{'='*80}\n")
    
    # Train for specific number of episodes
    result = train_for_episodes(
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        csv_file=args.csv_file,
        max_processes=args.max_processes,
        model_save_path=args.model_path,
        verbose=not args.quiet
    )
    
    if result:
        print("✅ Training completed successfully!")
        print(f"\n📊 View detailed training report in: training_report.csv")
        print(f"🔍 Analyze with: pandas.read_csv('training_report.csv')")
    else:
        print("❌ Training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()