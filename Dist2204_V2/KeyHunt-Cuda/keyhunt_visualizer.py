#!/usr/bin/env python3
"""
KeyHunt Visualizer v3.0 - Interactive Mining Control & Visualization
=====================================================================
Features:
- Grid visualization with drill-down zoom
- Start/stop mining from any grid cell
- Filter visualization (skipped blocks shown in red)
- Queue system for rescanning filtered blocks
- Real-time process monitoring
- Database integration for all tracking
"""

import sqlite3
import json
import os
import subprocess
import signal
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import webbrowser

# Configuration
DB_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PORT = 8080
KEYHUNT_PATH = os.path.join(DB_DIR, 'KeyHunt')

# Multi-GPU Configuration
GPU_CONFIG = {
    'available_gpus': [],  # Will be detected at startup
    'gpu_assignments': {},  # {process_id: gpu_id}
}

# Parallel Processing Configuration - allows 1 process per GPU
MAX_CONCURRENT_PROCESSES = 10  # Max processes (limited by number of GPUs available)

# Session Statistics
SESSION_STATS = {
    'start_time': time.time(),
    'total_keys_checked': 0,
    'blocks_completed': 0,
    'keys_per_second': 0,
    'gpu_speeds': {},  # {gpu_id: speed}
}

# Solved Puzzle Private Keys (puzzles 1-70) for pattern analysis
# These are the actual solved private keys from Bitcoin Puzzle challenges
# Data sourced from BTC32 repository - verified correct values
SOLVED_PUZZLES = {
    1: '1',
    2: '3',
    3: '7',
    4: '8',  # Fixed: was 'f'
    5: '15',
    6: '31',
    7: '4c',
    8: 'e0',
    9: '1d3',
    10: '202',
    11: '483',
    12: 'a7b',
    13: '1460',
    14: '2930',
    15: '68f3',
    16: 'c936',
    17: '1764f',
    18: '3080d',
    19: '5749f',
    20: 'd2c55',  # Fixed: was 'cce82'
    21: '1ba534',  # Fixed: was '1fa34f'
    22: '2de40f',  # Fixed: was '340326'
    23: '556e52',  # Fixed: was '6ac3875'
    24: 'dc2a04',  # Fixed: was 'd916ce8'
    25: '1fa5ee5',  # Fixed: was '17e2551e'
    26: '340326e',  # Fixed: was '3d94cd64'
    27: '6ac3875',  # Fixed: was '7d4fe747'
    28: 'd916ce8',  # Fixed: was 'b862a62e'
    29: '17e2551e',  # Fixed: was '1a96ca8d8'
    30: '3d94cd64',  # Fixed: was '34a65911d'
    31: '7d4fe747',  # Fixed: was '4aed21170'
    32: 'b862a62e',  # Fixed: was '9de820a7c'
    33: '1a96ca8d8',  # Fixed: was '1757756a93'
    34: '34a65911d',  # Fixed: was '22382facd0'
    35: '4aed21170',  # Fixed: was '4b5f8303e9'
    36: '9de820a7c',  # Fixed: was 'e9ae4933d6'
    37: '1757756a93',  # Fixed: was '153869acc5b'
    38: '22382facd0',  # Fixed: was '2a221c58d8f'
    39: '4b5f8303e9',  # Fixed: was '6bd3b27c591'
    40: 'e9ae4933d6',  # Fixed: was 'e02b35a358f'
    41: '153869acc5b',  # Fixed: was '122fca143c05'
    42: '2a221c58d8f',  # Fixed: was '2ec18388d544'
    43: '6bd3b27c591',  # Fixed: was '6cd610b53cba'
    44: 'e02b35a358f',  # Fixed: was 'ade6d7ce3b9b'
    45: '122fca143c05',  # Fixed: was '174176b015f4d'
    46: '2ec18388d544',  # Fixed: was '22bd43c2e9354'
    47: '6cd610b53cba',  # Fixed: was '75070a1a009d4'
    48: 'ade6d7ce3b9b',  # Fixed: was 'efae164cb9e3c'
    49: '174176b015f4d',  # Fixed: was '180788e47e326c'
    50: '22bd43c2e9354',  # Fixed: was '236fb6d5ad1f44'
    51: '75070a1a009d4',  # Fixed: was '6abe1f9b67e114'
    52: 'efae164cb9e3c',  # Fixed: was '9d18b63ac4ffdf'
    53: '180788e47e326c',  # Fixed: was '1eb25c90795d61c'
    54: '236fb6d5ad1f43',  # Fixed: was '2c675b852189a21'
    55: '6abe1f9b67e114',  # Fixed: was '7496cbb87cab44f'
    56: '9d18b63ac4ffdf',  # Fixed: was 'fc07a1825367bbe'
    57: '1eb25c90795d61c',  # Fixed: was '13c96a3742f64906'
    58: '2c675b852189a21',  # Fixed: was '363d541eb611abee'
    59: '7496cbb87cab44f',  # Fixed: was '7cce5efdaccf6808'
    60: 'fc07a1825367bbe',  # Fixed: was 'f7051f27b09112d4'
    61: '13c96a3742f64906',  # Fixed: was '1a838b13505b26867'
    62: '363d541eb611abee',  # Fixed: was '329ebc0e07d3d9a4c'
    63: '7cce5efdaccf6808',  # Fixed: was '7c822f51117ec0e6f'
    64: 'f7051f27b09112d4',  # Fixed: was 'f6a8f89546024e2d4'
    65: '1a838b13505b26867',  # Fixed: was '1a04b591f7a4eaccc4'
    66: '2832ed74f2b5e35ee',  # Fixed: was '340326be5f517f5624' - Solved 2024-10-12
    # Note: Puzzles 67-69, 71-74, 76-79, 81-84, 86-89, 91-94, 96-99 are UNSOLVED
    70: '349b84b6431a6c4ef1',  # Solved 2019-06-09
    # Additional solved puzzles from BTC32 (every 5th puzzle after 70)
    75: '4c5ce114686a1336e07',  # Solved 2019-06-10
    80: 'ea1a5c66dcc11b5ad180',  # Solved 2019-06-11
    85: '11720c4f018d51b8cebba8',  # Solved 2019-06-17
    90: '2ce00bb2136a445c71e85bf',  # Solved 2019-07-01
    95: '527a792b183c7f64a0e8b1f4',  # Solved 2019-07-06
}

# Scan Scheduler Configuration
SCHEDULER_CONFIG = {
    'enabled': False,
    'schedules': [],  # List of {start_time, end_time, days, action}
    'current_job': None,
}

# Cloud Sync Configuration
SYNC_CONFIG = {
    'enabled': False,
    'server_url': '',  # e.g., 'https://yoursite.com/wp-json/keyhunt/v1'
    'api_key': '',
    'client_id': '',
    'client_name': '',
    'auto_sync_interval': 90,  # 30 seconds for real-time team coordination
    'last_sync': None,
    'sync_in_progress': False,
    'is_master': False,  # Team members only upload to server; master sees full key details
}

# Puzzle presets with target addresses
PUZZLE_PRESETS = {
    71: {
        'name': 'Puzzle #71',
        'address': '1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU',
        'bits': 71,
        'reward': '7.1 BTC'
    },
    72: {
        'name': 'Puzzle #72',
        'address': '1JTK7s9YVYywfm5XUH7RNhHJH1LshCaRFR',
        'bits': 72,
        'reward': '7.2 BTC'
    },
    73: {
        'name': 'Puzzle #73',
        'address': '12VVRNPi4SJqUTsp6FmqDqY5sGosDtysn4',
        'bits': 73,
        'reward': '7.3 BTC'
    },
    74: {
        'name': 'Puzzle #74',
        'address': '1FWGcVDK3JGzCC3WtkYetULPszMaK2Jksv',
        'bits': 74,
        'reward': '7.4 BTC'
    },
    75: {
        'name': 'Puzzle #75',
        'address': '1DJh2eHFYQfACPmrvpyWc8MSTYKh7w9eRF',
        'bits': 75,
        'reward': '7.5 BTC'
    }
}

def get_puzzle_range(puzzle_num):
    """Calculate keyspace range for a puzzle"""
    start = 2 ** (puzzle_num - 1)
    end = 2 ** puzzle_num - 1
    return (start, end)

PUZZLE_RANGES = {n: get_puzzle_range(n) for n in range(66, 130)}


def detect_gpus():
    """Detect available NVIDIA GPUs using nvidia-smi"""
    gpus = []
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        gpus.append({
                            'id': int(parts[0]),
                            'name': parts[1].strip(),
                            'memory_mb': int(parts[2]),
                            'in_use': False
                        })
    except Exception as e:
        print(f"GPU detection error: {e}")

    GPU_CONFIG['available_gpus'] = gpus
    return gpus


def get_available_gpu():
    """Get the first GPU that's not in use (for multi-GPU parallel scanning)"""
    for gpu in GPU_CONFIG['available_gpus']:
        if not gpu['in_use']:
            return gpu['id']
    return None  # No GPU available


def get_gpu_count():
    """Get total number of GPUs"""
    return len(GPU_CONFIG['available_gpus'])


def get_available_gpu_count():
    """Get number of GPUs not currently in use"""
    return sum(1 for gpu in GPU_CONFIG['available_gpus'] if not gpu['in_use'])


def mark_gpu_in_use(gpu_id, in_use=True):
    """Mark a GPU as in use or available"""
    for gpu in GPU_CONFIG['available_gpus']:
        if gpu['id'] == gpu_id:
            gpu['in_use'] = in_use
            break


def analyze_solved_puzzles():
    """Analyze patterns in solved puzzle private keys"""
    analysis = {
        'hex_digit_frequency': {},
        'position_patterns': {},
        'bit_distribution': [],
        'first_digit_freq': {},
        'recommendations': []
    }

    # Analyze hex digit frequency across all solved puzzles
    all_digits = '0123456789abcdef'
    for d in all_digits:
        analysis['hex_digit_frequency'][d] = 0

    for puzzle_num, key in SOLVED_PUZZLES.items():
        key_lower = key.lower()
        for digit in key_lower:
            if digit in all_digits:
                analysis['hex_digit_frequency'][digit] += 1

        # Track first digit frequency (most significant)
        if key_lower:
            first = key_lower[0]
            analysis['first_digit_freq'][first] = analysis['first_digit_freq'].get(first, 0) + 1

    # Calculate total for percentages
    total_digits = sum(analysis['hex_digit_frequency'].values())
    if total_digits > 0:
        for d in all_digits:
            analysis['hex_digit_frequency'][d] = round(
                analysis['hex_digit_frequency'][d] / total_digits * 100, 2
            )

    # Analyze bit positions for high puzzles (60+)
    high_puzzles = {k: v for k, v in SOLVED_PUZZLES.items() if k >= 60}
    for puzzle_num, key in high_puzzles.items():
        key_int = int(key, 16)
        key_bits = bin(key_int)[2:]

        # Calculate position within the puzzle's keyspace
        range_start = 2 ** (puzzle_num - 1)
        range_end = 2 ** puzzle_num - 1
        range_size = range_end - range_start
        position_pct = ((key_int - range_start) / range_size) * 100 if range_size > 0 else 50

        analysis['bit_distribution'].append({
            'puzzle': puzzle_num,
            'position_pct': round(position_pct, 2),
            'key_hex': key
        })

    # Generate recommendations based on analysis
    avg_position = sum(b['position_pct'] for b in analysis['bit_distribution']) / len(analysis['bit_distribution']) if analysis['bit_distribution'] else 50

    if avg_position < 40:
        analysis['recommendations'].append("Historical keys tend toward lower ranges - prioritize first 40% of keyspace")
    elif avg_position > 60:
        analysis['recommendations'].append("Historical keys tend toward higher ranges - prioritize last 40% of keyspace")
    else:
        analysis['recommendations'].append("Historical keys are fairly distributed - scan middle ranges first")

    # Check for common starting patterns
    common_starts = sorted(analysis['first_digit_freq'].items(), key=lambda x: x[1], reverse=True)[:3]
    analysis['recommendations'].append(f"Most common first hex digits: {', '.join([f'{d}({c})' for d, c in common_starts])}")

    return analysis


def calculate_range_priority(start_hex, end_hex, puzzle_num=71):
    """Calculate priority score for a range based on solved puzzle patterns"""
    try:
        start_int = int(start_hex, 16)
        end_int = int(end_hex, 16)
        mid_point = (start_int + end_int) // 2

        # Get puzzle range
        range_start = 2 ** (puzzle_num - 1)
        range_end = 2 ** puzzle_num - 1
        range_size = range_end - range_start

        # Calculate position within keyspace (0-100%)
        position_pct = ((mid_point - range_start) / range_size) * 100 if range_size > 0 else 50

        # Base priority from position (prefer middle ranges slightly)
        # Bell curve centered at ~45% based on historical data
        position_score = 100 - abs(position_pct - 45) * 1.5

        # Check hex patterns
        mid_hex = hex(mid_point)[2:].lower()

        # Penalize vanity patterns
        pattern_penalty = 0

        # Repeated characters (AAA, FFF, etc.)
        for i in range(len(mid_hex) - 2):
            if mid_hex[i] == mid_hex[i+1] == mid_hex[i+2]:
                if mid_hex[i] != '0':  # Don't penalize zeros
                    pattern_penalty += 10

        # All same type (all letters or all numbers)
        check_portion = mid_hex[-8:] if len(mid_hex) > 8 else mid_hex
        if check_portion.isalpha() or check_portion.isdigit():
            pattern_penalty += 15

        # Bonus for matching common first digits from solved puzzles
        first_digit_bonus = 0
        if mid_hex and mid_hex[0] in ['1', '3', '6', '7', 'f', 'e']:
            first_digit_bonus = 5

        final_score = max(0, min(100, position_score - pattern_penalty + first_digit_bonus))

        return {
            'score': round(final_score, 1),
            'position_pct': round(position_pct, 1),
            'pattern_penalty': pattern_penalty,
            'recommendation': 'high' if final_score > 70 else 'medium' if final_score > 40 else 'low'
        }
    except Exception as e:
        return {'score': 50, 'error': str(e)}


class ProcessManager:
    """Manages KeyHunt processes"""

    def __init__(self):
        self.processes = {}  # {process_id: {process, puzzle, start, end, status, output}}
        self.lock = threading.Lock()
        self.next_id = 1
        self.found_keys = []  # Store any found keys/matches
        self.key_found_callbacks = []  # Callbacks to notify when key found

    def start_mining(self, puzzle_num, range_start, range_end, use_gpu=True, use_filter=False, gpu_id=None):
        """Start a KeyHunt mining process with optional GPU selection"""
        # Check concurrent process limit
        with self.lock:
            running_count = sum(1 for p in self.processes.values() if p['status'] == 'running')
            if running_count >= MAX_CONCURRENT_PROCESSES:
                return {'error': f'Maximum concurrent processes ({MAX_CONCURRENT_PROCESSES}) reached. Stop a process first.'}

            process_id = self.next_id
            self.next_id += 1

        if puzzle_num not in PUZZLE_PRESETS:
            return {'error': f'Unknown puzzle {puzzle_num}'}

        target_address = PUZZLE_PRESETS[puzzle_num]['address']

        # Select GPU - each GPU can only run 1 process at a time
        if use_gpu:
            if gpu_id is None:
                gpu_id = get_available_gpu()
            if gpu_id is None:
                return {'error': f'All GPUs are busy. You have {get_gpu_count()} GPU(s). Wait for a process to finish or use CPU mode.'}
            mark_gpu_in_use(gpu_id, True)
            GPU_CONFIG['gpu_assignments'][process_id] = gpu_id

        # Build command
        cmd = [KEYHUNT_PATH]
        if use_gpu:
            cmd.extend(['-g', '--gpui', str(gpu_id)])
        cmd.extend([
            '-m', 'ADDRESS',
            '--coin', 'BTC',
            '--range', f'{range_start}:{range_end}',
            target_address
        ])

        try:
            # Use unbuffered binary mode for real-time output capture
            # This is needed because KeyHunt uses \r for progress updates
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,  # Unbuffered
                cwd=DB_DIR
            )

            with self.lock:
                self.processes[process_id] = {
                    'process': process,
                    'puzzle': puzzle_num,
                    'start': range_start,
                    'end': range_end,
                    'status': 'running',
                    'output': [],
                    'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'use_filter': use_filter,
                    'keys_checked': 0,
                    'speed': '0 Mk/s',
                    'completion': 0.0,
                    'gpu_id': gpu_id if use_gpu else None,
                    'last_checkpoint': range_start,  # For checkpoint/resume
                }

            # Start output reader thread
            thread = threading.Thread(
                target=self._read_output,
                args=(process_id,),
                daemon=True
            )
            thread.start()

            return {
                'process_id': process_id,
                'status': 'started',
                'mode': 'GPU' if use_gpu else 'CPU',
                'command': ' '.join(cmd)
            }

        except Exception as e:
            return {'error': str(e)}

    def _read_output(self, process_id):
        """Read process output in background - handles both \\r and \\n line endings"""
        import re
        import io

        with self.lock:
            if process_id not in self.processes:
                return
            proc_info = self.processes[process_id]
            process = proc_info['process']

        try:
            # Use a buffer to accumulate characters and split on \r or \n
            # This handles KeyHunt's \r-based progress updates
            buffer = ""

            # Read character by character for real-time updates
            while True:
                byte = process.stdout.read(1)
                if not byte:
                    # Process ended
                    break

                # Decode byte to character
                char = byte.decode('utf-8', errors='replace')

                if char in ('\r', '\n'):
                    # Line complete - process it
                    line = buffer.strip()
                    buffer = ""

                    if not line:
                        continue

                    with self.lock:
                        if process_id not in self.processes:
                            break

                        self.processes[process_id]['output'].append(line)
                        # Keep only last 100 lines
                        if len(self.processes[process_id]['output']) > 100:
                            self.processes[process_id]['output'] = \
                                self.processes[process_id]['output'][-100:]

                        # Parse speed from KeyHunt output format: [GPU: 298.78 Mk/s]
                        # Matches both Mk/s and Gk/s formats
                        speed_match = re.search(r'\[GPU:\s*(\d+\.?\d*)\s*(Mk/s|Gk/s|MKey/s|GKey/s)\]', line)
                        if speed_match:
                            self.processes[process_id]['speed'] = f"{speed_match.group(1)} {speed_match.group(2)}"
                        else:
                            # Also try CPU+GPU format
                            speed_match = re.search(r'\[CPU\+GPU:\s*(\d+\.?\d*)\s*(Mk/s|Gk/s)\]', line)
                            if speed_match:
                                self.processes[process_id]['speed'] = f"{speed_match.group(1)} {speed_match.group(2)}"

                        # Parse keys checked from [T: 597,688,320 (30 bit)] format
                        keys_match = re.search(r'\[T:\s*([\d,]+)', line)
                        if keys_match:
                            try:
                                keys_str = keys_match.group(1).replace(',', '')
                                self.processes[process_id]['keys_checked'] = int(keys_str)
                            except:
                                pass

                        # Parse completion percentage [C: 13.916016 %]
                        completion_match = re.search(r'\[C:\s*(\d+\.?\d*)\s*%\]', line)
                        if completion_match:
                            new_completion = float(completion_match.group(1))
                            old_completion = self.processes[process_id].get('completion', 0)
                            self.processes[process_id]['completion'] = new_completion

                            # Save progress at 5% intervals for cloud sync
                            last_saved_pct = self.processes[process_id].get('last_saved_pct', 0)
                            if new_completion >= last_saved_pct + 5:
                                self.processes[process_id]['last_saved_pct'] = int(new_completion / 5) * 5
                                # Save progress to database (will be synced to cloud)
                                self._save_scan_progress(
                                    self.processes[process_id]['puzzle'],
                                    self.processes[process_id]['start'],
                                    self.processes[process_id]['end'],
                                    self.processes[process_id]['keys_checked'],
                                    new_completion
                                )

                        # KEY FOUND DETECTION - Check for actual key match indicators
                        # Be specific to avoid false positives from "MAX FOUND" or "OUTPUT FILE"
                        is_key_found = (
                            'PubAddress:' in line or
                            'Priv (HEX):' in line or
                            'Priv (WIF):' in line or
                            ('Address:' in line and 'Priv' in line)
                        )
                        if is_key_found:
                            self.processes[process_id]['key_found'] = True
                            self.processes[process_id]['found_line'] = line
                            self._handle_key_found(process_id, line)
                else:
                    buffer += char

            # Process any remaining buffer content
            if buffer.strip():
                with self.lock:
                    if process_id in self.processes:
                        self.processes[process_id]['output'].append(buffer.strip())

        except Exception as e:
            pass
        finally:
            # Capture process info while holding lock
            puzzle = None
            start = None
            end = None
            keys_checked = 0
            completion_pct = 0

            with self.lock:
                if process_id in self.processes:
                    proc_info = self.processes[process_id]
                    proc_info['status'] = 'completed'

                    # Save values for use outside lock
                    puzzle = proc_info['puzzle']
                    start = proc_info['start']
                    end = proc_info['end']
                    keys_checked = proc_info['keys_checked']
                    completion_pct = proc_info.get('completion', 0)

            # Save progress outside the lock to avoid deadlocks
            # Only save if we captured valid process info
            if puzzle is not None and start is not None:
                self._save_scan_progress(puzzle, start, end, keys_checked, completion_pct)

                # If this was a skipped block, remove it from skipped_blocks table
                self._clear_skipped_block(puzzle, start)

    def pause_mining(self, process_id):
        """Pause a mining process and save checkpoint for later resume"""
        with self.lock:
            if process_id not in self.processes:
                return {'error': 'Process not found'}

            proc_info = self.processes[process_id]
            process = proc_info['process']

            if proc_info['status'] == 'running':
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                except:
                    pass

                proc_info['status'] = 'paused'

                # Release GPU
                gpu_id = proc_info.get('gpu_id')
                if gpu_id is not None:
                    mark_gpu_in_use(gpu_id, False)
                    if process_id in GPU_CONFIG['gpu_assignments']:
                        del GPU_CONFIG['gpu_assignments'][process_id]

                # Save progress to database (partial completion)
                self._save_scan_progress(
                    proc_info['puzzle'],
                    proc_info['start'],
                    proc_info['end'],
                    proc_info['keys_checked'],
                    proc_info.get('completion', 0)
                )

                # ALWAYS save checkpoint when pausing
                self._save_checkpoint(
                    proc_info['puzzle'],
                    proc_info['start'],
                    proc_info['end'],
                    proc_info['keys_checked'],
                    proc_info.get('completion', 0)
                )

                print(f"[ProcessManager] Process {process_id} paused - checkpoint saved at {proc_info.get('completion', 0):.2f}%")

            return {
                'status': 'paused',
                'process_id': process_id,
                'completion': proc_info.get('completion', 0),
                'keys_checked': proc_info.get('keys_checked', 0)
            }

    def stop_mining(self, process_id, skip_checkpoint=False):
        """Stop a mining process, optionally skipping checkpoint save"""
        with self.lock:
            if process_id not in self.processes:
                return {'error': 'Process not found'}

            proc_info = self.processes[process_id]
            process = proc_info['process']

            if proc_info['status'] == 'running':
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                except:
                    pass

                proc_info['status'] = 'stopped'

                # Release GPU
                gpu_id = proc_info.get('gpu_id')
                if gpu_id is not None:
                    mark_gpu_in_use(gpu_id, False)
                    if process_id in GPU_CONFIG['gpu_assignments']:
                        del GPU_CONFIG['gpu_assignments'][process_id]

                # Update session stats
                SESSION_STATS['blocks_completed'] += 1
                SESSION_STATS['total_keys_checked'] += proc_info['keys_checked']

                # Save progress to database (with completion percentage)
                self._save_scan_progress(
                    proc_info['puzzle'],
                    proc_info['start'],
                    proc_info['end'],
                    proc_info['keys_checked'],
                    proc_info.get('completion', 0)
                )

                # Save checkpoint for resume (unless skip_checkpoint is True)
                if not skip_checkpoint:
                    self._save_checkpoint(
                        proc_info['puzzle'],
                        proc_info['start'],
                        proc_info['end'],
                        proc_info['keys_checked'],
                        proc_info.get('completion', 0)
                    )

                # Clear from skipped blocks if this was a queued rescan
                self._clear_skipped_block(
                    proc_info['puzzle'],
                    proc_info['start']
                )

            return {'status': 'stopped', 'process_id': process_id}

    def _save_scan_progress(self, puzzle_num, start, end, keys_checked, completion_pct=None):
        """Save scan progress to database - only saves actual progress made"""
        try:
            # Don't save if no work was done at all
            if keys_checked < 1:
                print(f"Skipping save - no keys checked")
                return

            db_path = os.path.join(DB_DIR, f'scan_data_puzzle_{puzzle_num}.db')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Ensure table has completion_pct column
            try:
                cursor.execute('ALTER TABLE my_scanned ADD COLUMN completion_pct REAL DEFAULT 100.0')
            except sqlite3.OperationalError:
                pass  # Column already exists

            # Calculate actual end based on keys checked
            try:
                start_int = int(start, 16)
                end_int = int(end, 16)
                total_range = end_int - start_int + 1

                # Use completion percentage if available, otherwise estimate from keys
                if completion_pct is not None and completion_pct > 0:
                    actual_pct = min(100.0, completion_pct)
                    actual_keys_scanned = int(total_range * (actual_pct / 100.0))
                    actual_end_int = start_int + actual_keys_scanned - 1
                else:
                    # Estimate based on keys checked
                    actual_end_int = start_int + keys_checked - 1
                    actual_pct = (keys_checked / total_range * 100) if total_range > 0 else 0

                # Don't exceed the original end
                actual_end_int = min(actual_end_int, end_int)
                actual_end = hex(actual_end_int)[2:].upper()

                # Store the actual scanned range, not the target range
                cursor.execute('''
                    INSERT OR REPLACE INTO my_scanned
                    (block_start, block_end, scanned_at, keys_checked, source, completion_pct)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (start, actual_end, time.strftime('%Y-%m-%dT%H:%M:%S'),
                      keys_checked, 'visualizer', round(actual_pct, 4)))

                print(f"Saved progress: {start[:12]}... -> {actual_end[:12]}... ({actual_pct:.2f}% of target range)")

            except (ValueError, TypeError) as e:
                # Fallback to original behavior if calculation fails
                cursor.execute('''
                    INSERT OR REPLACE INTO my_scanned
                    (block_start, block_end, scanned_at, keys_checked, source, completion_pct)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (start, end, time.strftime('%Y-%m-%dT%H:%M:%S'),
                      keys_checked, 'visualizer', completion_pct or 0))

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving progress: {e}")

    def _save_checkpoint(self, puzzle_num, start, end, keys_checked, completion_pct):
        """Save checkpoint for exact resume capability"""
        try:
            if keys_checked < 100:
                return  # Don't save trivial checkpoints

            db_path = os.path.join(DB_DIR, f'scan_data_puzzle_{puzzle_num}.db')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Create checkpoints table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    block_start TEXT,
                    block_end TEXT,
                    resume_position TEXT,
                    keys_checked INTEGER,
                    completion_pct REAL,
                    saved_at TEXT,
                    resumed INTEGER DEFAULT 0
                )
            ''')

            # Calculate exact resume position
            try:
                start_int = int(start, 16)
                resume_pos_int = start_int + keys_checked
                resume_position = hex(resume_pos_int)[2:].upper()
            except:
                resume_position = start

            # Insert checkpoint
            cursor.execute('''
                INSERT INTO checkpoints (block_start, block_end, resume_position, keys_checked, completion_pct, saved_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (start, end, resume_position, keys_checked, completion_pct or 0, time.strftime('%Y-%m-%dT%H:%M:%S')))

            conn.commit()
            conn.close()
            print(f"Checkpoint saved: {start[:12]}... -> resume at {resume_position[:12]}... ({completion_pct:.2f}%)")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def _clear_skipped_block(self, puzzle_num, start):
        """Remove a block from skipped_blocks table after it's been mined"""
        try:
            db_path = os.path.join(DB_DIR, f'scan_data_puzzle_{puzzle_num}.db')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check if skipped_blocks table exists
            cursor.execute('''
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='skipped_blocks'
            ''')
            if cursor.fetchone():
                cursor.execute('DELETE FROM skipped_blocks WHERE block_start = ?', (start,))
                if cursor.rowcount > 0:
                    print(f"Cleared skipped block {start} from queue after mining")

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error clearing skipped block: {e}")

    def _handle_key_found(self, process_id, line):
        """Handle when a private key is found - JACKPOT!"""
        import re
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        proc_info = self.processes.get(process_id, {})
        puzzle = proc_info.get('puzzle', 'unknown')
        start = proc_info.get('start', 'unknown')
        end = proc_info.get('end', 'unknown')

        # Extract key details from output
        priv_hex = None
        priv_wif = None
        pub_address = None

        # Get all recent output to find key details
        recent_output = proc_info.get('output', [])[-20:]
        all_text = '\n'.join(recent_output) + '\n' + line

        # Extract private key (HEX)
        hex_match = re.search(r'Priv \(HEX\):\s*([0-9A-Fa-f]+)', all_text)
        if hex_match:
            priv_hex = hex_match.group(1)

        # Extract private key (WIF)
        wif_match = re.search(r'Priv \(WIF\):\s*([A-Za-z0-9]+)', all_text)
        if wif_match:
            priv_wif = wif_match.group(1)

        # Extract public address
        addr_match = re.search(r'PubAddress:\s*([A-Za-z0-9]+)', all_text)
        if not addr_match:
            addr_match = re.search(r'Address:\s*([A-Za-z0-9]+)', all_text)
        if addr_match:
            pub_address = addr_match.group(1)

        # Check if this is master or team instance
        is_master = SYNC_CONFIG.get('is_master', False)

        # Create found key record (for master, include full details; for team, hide private key)
        found_record = {
            'timestamp': timestamp,
            'puzzle': puzzle,
            'range_start': start,
            'range_end': end,
            'line': line if is_master else "KEY FOUND - Details uploaded to server",
            'process_id': process_id,
            'priv_hex': priv_hex if is_master else None,  # Hide from team
            'priv_wif': priv_wif if is_master else None,  # Hide from team
            'pub_address': pub_address
        }
        self.found_keys.append(found_record)

        # CRITICAL: Always upload to sync server FIRST
        try:
            if SYNC_CONFIG.get('enabled') and SYNC_CONFIG.get('server_url'):
                print(f"Uploading found key to sync server...")
                upload_result = sync_upload_found_key(
                    puzzle=puzzle,
                    priv_hex=priv_hex,
                    priv_wif=priv_wif,
                    pub_address=pub_address,
                    block_start=start,
                    block_end=end,
                    raw_output=line
                )
                if upload_result.get('status') == 'ok':
                    print(f"Found key uploaded to server successfully!")
                else:
                    print(f"Warning: Failed to upload found key: {upload_result.get('message')}")
        except Exception as e:
            print(f"Error uploading found key to server: {e}")

        # Only save to Found.txt if this is the MASTER instance
        if is_master:
            try:
                found_file = os.path.join(DB_DIR, 'Found.txt')
                with open(found_file, 'a') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"KEY FOUND! - {timestamp}\n")
                    f.write(f"{'='*60}\n")
                    f.write(f"Puzzle: {puzzle}\n")
                    f.write(f"Range: {start} - {end}\n")
                    if pub_address:
                        f.write(f"Address: {pub_address}\n")
                    if priv_hex:
                        f.write(f"Private Key (HEX): {priv_hex}\n")
                    if priv_wif:
                        f.write(f"Private Key (WIF): {priv_wif}\n")
                    f.write(f"Raw Output: {line}\n")
                    f.write(f"{'='*60}\n\n")
                print(f"KEY FOUND! Saved to Found.txt")
                if priv_hex:
                    print(f"  Private Key (HEX): {priv_hex}")
                if priv_wif:
                    print(f"  Private Key (WIF): {priv_wif}")
            except Exception as e:
                print(f"Error saving found key: {e}")
        else:
            # Team member - just acknowledge the find
            print(f"KEY FOUND! Details uploaded to master server.")

        # Save to database
        try:
            if puzzle != 'unknown':
                db_path = os.path.join(DB_DIR, f'scan_data_puzzle_{puzzle}.db')
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS found_keys (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        range_start TEXT,
                        range_end TEXT,
                        details TEXT,
                        priv_hex TEXT,
                        priv_wif TEXT,
                        pub_address TEXT
                    )
                ''')

                cursor.execute('''
                    INSERT INTO found_keys (timestamp, range_start, range_end, details, priv_hex, priv_wif, pub_address)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (timestamp, start, end, line, priv_hex, priv_wif, pub_address))

                conn.commit()
                conn.close()
        except Exception as e:
            print(f"Error saving to database: {e}")

        # Upload to sync server - CRITICAL for distributed mining!
        try:
            if SYNC_CONFIG.get('enabled') and SYNC_CONFIG.get('server_url'):
                print(f"Uploading found key to sync server...")
                upload_result = sync_upload_found_key(
                    puzzle=puzzle,
                    priv_hex=priv_hex,
                    priv_wif=priv_wif,
                    pub_address=pub_address,
                    block_start=start,
                    block_end=end,
                    raw_output=line
                )
                if upload_result.get('status') == 'ok':
                    print(f"SUCCESS: Found key uploaded to server!")
                else:
                    print(f"WARNING: Failed to upload to server: {upload_result.get('message', 'Unknown error')}")
        except Exception as e:
            print(f"Error uploading found key to server: {e}")

        # Stop the mining process - we found it!
        try:
            process = proc_info.get('process')
            if process:
                process.terminate()
        except:
            pass

    def get_status(self, process_id=None):
        """Get status of one or all processes"""
        with self.lock:
            if process_id is not None:
                if process_id not in self.processes:
                    return {'error': 'Process not found'}
                proc = self.processes[process_id]
                return {
                    'process_id': process_id,
                    'puzzle': proc['puzzle'],
                    'start': proc['start'],
                    'end': proc['end'],
                    'status': proc['status'],
                    'started_at': proc['started_at'],
                    'speed': proc['speed'],
                    'keys_checked': proc['keys_checked'],
                    'completion': proc.get('completion', 0),
                    'output': proc['output'][-20:]  # Last 20 lines
                }
            else:
                return {
                    pid: {
                        'puzzle': p['puzzle'],
                        'start': p['start'],
                        'end': p['end'],
                        'status': p['status'],
                        'started_at': p['started_at'],
                        'speed': p['speed'],
                        'keys_checked': p['keys_checked'],
                        'completion': p.get('completion', 0)
                    }
                    for pid, p in self.processes.items()
                }

    def get_active_count(self):
        """Get count of active processes"""
        with self.lock:
            return sum(1 for p in self.processes.values() if p['status'] == 'running')

    def get_found_keys(self):
        """Get list of found keys"""
        return self.found_keys.copy()

    def has_key_found(self):
        """Check if any keys have been found"""
        return len(self.found_keys) > 0

    def clear_completed(self):
        """Remove completed/stopped processes from tracking"""
        with self.lock:
            to_remove = [
                pid for pid, proc in self.processes.items()
                if proc['status'] in ('completed', 'stopped')
            ]
            for pid in to_remove:
                del self.processes[pid]
            return {'status': 'success', 'cleared': len(to_remove)}


class FilterManager:
    """Manages filter/skip tracking"""

    def __init__(self):
        self.ensure_tables()

    def ensure_tables(self):
        """Ensure skipped_blocks table exists in all databases"""
        for f in os.listdir(DB_DIR):
            if f.startswith('scan_data_puzzle_') and f.endswith('.db'):
                try:
                    db_path = os.path.join(DB_DIR, f)
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()

                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS skipped_blocks (
                            block_start TEXT PRIMARY KEY,
                            block_end TEXT,
                            reason TEXT,
                            skipped_at TEXT,
                            queued_for_rescan INTEGER DEFAULT 0
                        )
                    ''')

                    conn.commit()
                    conn.close()
                except:
                    pass

    def add_skipped_block(self, puzzle_num, start, end, reason='filter'):
        """Mark a block as skipped/filtered"""
        try:
            db_path = os.path.join(DB_DIR, f'scan_data_puzzle_{puzzle_num}.db')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO skipped_blocks
                (block_start, block_end, reason, skipped_at, queued_for_rescan)
                VALUES (?, ?, ?, ?, 0)
            ''', (start, end, reason, time.strftime('%Y-%m-%dT%H:%M:%S')))

            conn.commit()
            conn.close()
            return {'status': 'added'}
        except Exception as e:
            return {'error': str(e)}

    def get_skipped_blocks(self, puzzle_num):
        """Get all skipped blocks for a puzzle"""
        try:
            db_path = os.path.join(DB_DIR, f'scan_data_puzzle_{puzzle_num}.db')
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            cursor = conn.cursor()

            cursor.execute('SELECT block_start, block_end, reason, skipped_at, queued_for_rescan FROM skipped_blocks')
            blocks = [
                {'start': row[0], 'end': row[1], 'reason': row[2], 'time': row[3], 'queued': row[4]}
                for row in cursor.fetchall()
            ]

            conn.close()
            return blocks
        except Exception as e:
            return []

    def queue_for_rescan(self, puzzle_num, start):
        """Queue a skipped block for rescanning"""
        try:
            db_path = os.path.join(DB_DIR, f'scan_data_puzzle_{puzzle_num}.db')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute('''
                UPDATE skipped_blocks SET queued_for_rescan = 1
                WHERE block_start = ?
            ''', (start,))

            conn.commit()
            conn.close()
            return {'status': 'queued'}
        except Exception as e:
            return {'error': str(e)}

    def remove_skipped(self, puzzle_num, start):
        """Remove a block from skipped list (after rescanning)"""
        try:
            db_path = os.path.join(DB_DIR, f'scan_data_puzzle_{puzzle_num}.db')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute('DELETE FROM skipped_blocks WHERE block_start = ?', (start,))

            conn.commit()
            conn.close()
            return {'status': 'removed'}
        except Exception as e:
            return {'error': str(e)}


class PoolScraper:
    """Scrapes btcpuzzle.info for already-scanned ranges"""

    def __init__(self):
        self.last_scrape = {}  # {puzzle_num: timestamp}
        self.scrape_results = {}  # {puzzle_num: {ranges, challenges, timestamp}}

    def scrape_puzzle(self, puzzle_num):
        """Scrape scanned ranges for a specific puzzle"""
        import re
        try:
            # Try to import requests, fallback to urllib if not available
            try:
                import requests
                from bs4 import BeautifulSoup
                use_requests = True
            except ImportError:
                from urllib.request import urlopen
                from html.parser import HTMLParser
                use_requests = False

            url = f"https://btcpuzzle.info/puzzle/{puzzle_num}"
            print(f"Scraping {url}...")

            if use_requests:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                html_content = response.text
            else:
                with urlopen(url, timeout=30) as response:
                    html_content = response.read().decode('utf-8')

            # Extract range IDs (format: 50XA1EC - 4 to 5 hex digits after X)
            range_pattern = re.compile(r'[0-9A-F]{2}X[0-9A-F]{4,5}')
            matches = range_pattern.findall(html_content)
            unique_ranges = list(set(matches))

            # Extract completed challenges (7FXXXXX format)
            challenge_pattern = re.compile(r'✅([0-9A-F]{2})XXXXX')
            challenge_matches = challenge_pattern.findall(html_content)
            unique_challenges = list(set(challenge_matches))

            # Decode ranges to actual blocks
            scanned_blocks = []

            for range_id in unique_ranges:
                blocks = self._decode_range_id(range_id)
                scanned_blocks.extend(blocks)

            for challenge_prefix in unique_challenges:
                blocks = self._decode_challenge(challenge_prefix)
                scanned_blocks.extend(blocks)

            # Store results
            self.last_scrape[puzzle_num] = time.strftime('%Y-%m-%d %H:%M:%S')
            self.scrape_results[puzzle_num] = {
                'ranges': len(unique_ranges),
                'challenges': len(unique_challenges),
                'total_blocks': len(scanned_blocks),
                'timestamp': self.last_scrape[puzzle_num]
            }

            return {
                'status': 'success',
                'ranges_found': len(unique_ranges),
                'challenges_found': len(unique_challenges),
                'blocks_decoded': len(scanned_blocks),
                'blocks': scanned_blocks
            }

        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    def _decode_range_id(self, range_id):
        """Decode pool range ID to actual hex blocks"""
        hex_prefix = range_id[0:2]
        hex_suffix = range_id[3:]

        blocks = []
        for x in '0123456789ABCDEF':
            block_start = int(hex_prefix + x + hex_suffix + '3000000000', 16)
            block_end = int(hex_prefix + x + hex_suffix + '3FFFFFFFFF', 16)
            blocks.append((block_start, block_end))

        return blocks

    def _decode_challenge(self, challenge_prefix):
        """Decode completed challenge to blocks"""
        blocks = []
        for x1 in '0123456789ABCDEF':
            for x2 in '0123456789ABCDEF':
                block_start = int(challenge_prefix + x1 + x2 + '0003000000000', 16)
                block_end = int(challenge_prefix + x1 + x2 + '0003FFFFFFFFF', 16)
                blocks.append((block_start, block_end))
        return blocks

    def save_to_database(self, puzzle_num, blocks):
        """Save scraped blocks to database"""
        try:
            db_path = os.path.join(DB_DIR, f'scan_data_puzzle_{puzzle_num}.db')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            timestamp = time.strftime('%Y-%m-%dT%H:%M:%S')
            added = 0
            existing = 0

            for block_start, block_end in blocks:
                block_start_hex = hex(block_start)[2:].upper()
                block_end_hex = hex(block_end)[2:].upper()

                cursor.execute('SELECT COUNT(*) FROM pool_scanned WHERE block_start = ?', (block_start_hex,))
                if cursor.fetchone()[0] == 0:
                    cursor.execute('''
                        INSERT INTO pool_scanned (block_start, block_end, scraped_at)
                        VALUES (?, ?, ?)
                    ''', (block_start_hex, block_end_hex, timestamp))
                    added += 1
                else:
                    existing += 1

            conn.commit()
            conn.close()

            return {'added': added, 'existing': existing}
        except Exception as e:
            return {'error': str(e)}

    def get_last_scrape(self, puzzle_num):
        """Get info about last scrape for a puzzle"""
        return self.scrape_results.get(puzzle_num, None)


class SmartBatchFilter:
    """
    Pre-filter large blocks into clean sub-ranges by detecting pattern addresses.

    Filters out addresses with:
    - 3+ repeated characters (AAA, FFF, 111, etc.)
    - 4+ repeated characters (AAAA, FFFF, 1111, etc.)
    - All-letters (ABCDEF...) or all-numbers (123456...)

    This dramatically reduces search space for vanity address research.
    """

    def __init__(self, block_start, block_end, subrange_size=10_000_000,
                 exclude_iter3=True, exclude_iter4=False, exclude_alphanum=True):
        """
        Initialize filter.

        Args:
            block_start: Starting address (int)
            block_end: Ending address (int)
            subrange_size: Keys per sub-range (default 10M)
            exclude_iter3: Exclude 3+ repeated chars
            exclude_iter4: Exclude 4+ repeated chars
            exclude_alphanum: Exclude all-letters/numbers
        """
        self.block_start = block_start
        self.block_end = block_end
        self.subrange_size = subrange_size
        self.exclude_iter3 = exclude_iter3
        self.exclude_iter4 = exclude_iter4
        self.exclude_alphanum = exclude_alphanum

    def generate_clean_subranges(self):
        """
        Break block into sub-ranges and filter out patterns.

        Returns:
            List of (start, end) tuples for clean sub-ranges
        """
        clean_ranges = []
        skipped_ranges = []
        current = self.block_start

        while current <= self.block_end:
            subrange_end = min(current + self.subrange_size - 1, self.block_end)

            # Sample 16 points in this sub-range to detect patterns
            sample_points = [current + (self.subrange_size * i) // 16
                           for i in range(17)]
            sample_points = [p for p in sample_points if p <= subrange_end]

            # Check if ANY sample has patterns
            has_pattern = False
            pattern_reason = None
            for sample_addr in sample_points:
                if sample_addr > self.block_end:
                    continue

                skip, reason = self.check_address_for_patterns(sample_addr)
                if skip:
                    has_pattern = True
                    pattern_reason = reason
                    break

            if has_pattern:
                skipped_ranges.append((current, subrange_end, pattern_reason))
            else:
                clean_ranges.append((current, subrange_end))

            current = subrange_end + 1

        return clean_ranges, skipped_ranges

    def check_address_for_patterns(self, address):
        """
        Check if address has patterns.

        Args:
            address: Address to check (int)

        Returns:
            (should_skip, reason) tuple
        """
        hex_str = hex(address)[2:].upper()

        # Check iter3 (3+ repeated chars)
        if self.exclude_iter3 and self.has_repeated_chars(hex_str, 3):
            return True, "repeated-3+"

        # Check iter4 (4+ repeated chars)
        if self.exclude_iter4 and self.has_repeated_chars(hex_str, 4):
            return True, "repeated-4+"

        # Check alphanum (all letters or all numbers)
        if self.exclude_alphanum and self.is_all_alpha_or_numeric(hex_str):
            return True, "all-alpha/numeric"

        return False, None

    def has_repeated_chars(self, hex_string, min_repeats=3):
        """
        Check if hex string has N or more repeated characters.

        Args:
            hex_string: Hex string to check
            min_repeats: Minimum repeats to trigger (default 3)

        Returns:
            True if has N+ repeated chars (ignoring zeros)
        """
        count = 1
        prev_char = ''

        for char in hex_string:
            if char == prev_char:
                count += 1
                # Ignore repeated zeros - they're common in valid ranges
                if count >= min_repeats and char != '0':
                    return True
            else:
                count = 1
                prev_char = char

        return False

    def is_all_alpha_or_numeric(self, hex_string):
        """
        Check if hex string is entirely letters (A-F) or numbers (0-9).

        Args:
            hex_string: Hex string to check

        Returns:
            True if all letters OR all numbers
        """
        # Only check last 8 characters (most significant for patterns)
        check_str = hex_string[-8:] if len(hex_string) > 8 else hex_string

        has_alpha = any(c in 'ABCDEF' for c in check_str)
        has_numeric = any(c in '0123456789' for c in check_str)

        # XOR: true if only one type is present
        return has_alpha ^ has_numeric

    def get_statistics(self):
        """Get filtering statistics"""
        clean_ranges, skipped_ranges = self.generate_clean_subranges()

        total_keys = self.block_end - self.block_start + 1
        clean_keys = sum(end - start + 1 for start, end in clean_ranges)
        skipped_keys = sum(end - start + 1 for start, end, _ in skipped_ranges)

        return {
            'total_subranges': len(clean_ranges) + len(skipped_ranges),
            'clean_subranges': len(clean_ranges),
            'skipped_subranges': len(skipped_ranges),
            'total_keys': total_keys,
            'clean_keys': clean_keys,
            'skipped_keys': skipped_keys,
            'reduction_pct': (skipped_keys / total_keys * 100) if total_keys > 0 else 0
        }


# Global managers
process_manager = ProcessManager()
filter_manager = FilterManager()
pool_scraper = PoolScraper()


def get_partial_blocks(puzzle_num):
    """Get partially completed blocks that can be resumed"""
    try:
        db_path = os.path.join(DB_DIR, f'scan_data_puzzle_{puzzle_num}.db')
        if not os.path.exists(db_path):
            return []

        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()

        # Get blocks where completion_pct < 100 (if column exists)
        try:
            cursor.execute('''
                SELECT block_start, block_end, keys_checked, completion_pct
                FROM my_scanned
                WHERE completion_pct < 99.0
                ORDER BY completion_pct ASC
                LIMIT 50
            ''')
            rows = cursor.fetchall()
        except sqlite3.OperationalError:
            # completion_pct column doesn't exist
            rows = []

        conn.close()

        partials = []
        for row in rows:
            start = row[0]
            original_end = row[1]
            keys_checked = row[2]
            completion_pct = row[3]

            # Calculate resume position
            try:
                start_int = int(start, 16)
                end_int = int(original_end, 16)
                # Resume from where we left off
                resume_start_int = start_int + keys_checked
                resume_start = hex(resume_start_int)[2:].upper()

                partials.append({
                    'start': start,
                    'end': original_end,
                    'resume_start': resume_start,
                    'keys_checked': keys_checked,
                    'completion_pct': completion_pct
                })
            except (ValueError, TypeError):
                continue

        return partials
    except Exception as e:
        print(f"Error getting partial blocks: {e}")
        return []


def analyze_pool_patterns(puzzle_num):
    """Analyze pool scanning patterns to predict next blocks"""
    try:
        db_path = os.path.join(DB_DIR, f'scan_data_puzzle_{puzzle_num}.db')
        if not os.path.exists(db_path):
            return {'error': 'No database found'}

        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()

        # Get all pool-scanned blocks
        cursor.execute('SELECT block_start, block_end, scraped_at FROM pool_scanned ORDER BY block_start')
        pool_blocks = cursor.fetchall()
        conn.close()

        if len(pool_blocks) < 10:
            return {'error': 'Not enough pool data for pattern analysis (need at least 10 blocks)'}

        # Convert to integers for analysis
        blocks = []
        for row in pool_blocks:
            try:
                start = int(row[0], 16)
                end = int(row[1], 16)
                blocks.append((start, end))
            except:
                continue

        # Sort by start position
        blocks.sort(key=lambda x: x[0])

        total_blocks = len(blocks)
        total_keyspace_blocks = 33554432  # 2^25 blocks for puzzle 71

        # Calculate statistics
        block_starts = [b[0] for b in blocks]
        block_sizes = [b[1] - b[0] + 1 for b in blocks]

        # Find gaps between blocks
        gaps = []
        for i in range(1, len(blocks)):
            gap = blocks[i][0] - blocks[i-1][1] - 1
            if gap > 0:
                gaps.append({
                    'start': hex(blocks[i-1][1] + 1)[2:].upper(),
                    'end': hex(blocks[i][0] - 1)[2:].upper(),
                    'size': gap
                })

        # Sort gaps by size (largest first)
        gaps.sort(key=lambda x: x['size'], reverse=True)

        # Analyze block distribution by prefix
        prefix_counts = {}
        for start, end in blocks:
            # Get first 2 hex digits as prefix
            prefix = hex(start)[2:4].upper()
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

        # Find most active prefixes
        sorted_prefixes = sorted(prefix_counts.items(), key=lambda x: x[1], reverse=True)

        # Predict next blocks based on sequential pattern
        # Find the highest block scanned
        highest_block = max(blocks, key=lambda x: x[0])
        lowest_block = min(blocks, key=lambda x: x[0])

        # Estimate block stride (common increment)
        if len(blocks) > 1:
            strides = []
            for i in range(1, min(100, len(blocks))):
                stride = blocks[i][0] - blocks[i-1][0]
                if stride > 0:
                    strides.append(stride)
            avg_stride = sum(strides) / len(strides) if strides else 0
        else:
            avg_stride = 0

        # Generate predictions for next blocks
        predictions = []
        if avg_stride > 0:
            # Predict next sequential blocks
            next_start = highest_block[0] + int(avg_stride)
            for i in range(10):
                pred_start = next_start + (i * int(avg_stride))
                pred_end = pred_start + (highest_block[1] - highest_block[0])
                predictions.append({
                    'start': hex(pred_start)[2:].upper(),
                    'end': hex(pred_end)[2:].upper(),
                    'confidence': 'high' if i < 3 else 'medium'
                })

        # Find unscanned regions before pool started
        pre_scan_blocks = []
        if lowest_block[0] > 0:
            # Get 10 blocks before the lowest scanned block
            block_size = lowest_block[1] - lowest_block[0] + 1
            for i in range(10, 0, -1):
                pred_start = lowest_block[0] - (i * block_size)
                if pred_start > 0:
                    pred_end = pred_start + block_size - 1
                    pre_scan_blocks.append({
                        'start': hex(pred_start)[2:].upper(),
                        'end': hex(pred_end)[2:].upper(),
                        'reason': 'before_pool_start'
                    })

        return {
            'status': 'success',
            'total_pool_blocks': total_blocks,
            'total_keyspace_blocks': total_keyspace_blocks,
            'coverage_pct': (total_blocks / total_keyspace_blocks * 100) if total_keyspace_blocks > 0 else 0,
            'avg_block_size': sum(block_sizes) / len(block_sizes) if block_sizes else 0,
            'avg_stride': avg_stride,
            'top_prefixes': sorted_prefixes[:10],
            'largest_gaps': gaps[:10],
            'predictions': predictions,
            'pre_scan_opportunities': pre_scan_blocks,
            'highest_scanned': hex(highest_block[0])[2:].upper(),
            'lowest_scanned': hex(lowest_block[0])[2:].upper()
        }
    except Exception as e:
        return {'error': str(e)}


def kill_all_gpu_processes():
    """Kill all KeyHunt GPU processes running on the system"""
    try:
        killed = 0

        # First, stop all tracked processes
        for pid in list(process_manager.processes.keys()):
            try:
                proc_info = process_manager.processes.get(pid)
                if proc_info and proc_info.get('status') == 'running':
                    process = proc_info.get('process')
                    if process:
                        process.kill()
                        killed += 1
            except:
                pass

        # Then use pkill to kill any orphaned KeyHunt processes
        try:
            result = subprocess.run(
                ['pkill', '-9', '-f', 'KeyHunt'],
                capture_output=True,
                timeout=5
            )
            # pkill returns 0 if processes killed, 1 if none found
            if result.returncode == 0:
                # Count how many were killed (approximate from output)
                killed += 1  # At least one killed
        except:
            pass

        # Also try to kill any keyhunt-cuda processes
        try:
            result = subprocess.run(
                ['pkill', '-9', '-f', 'keyhunt-cuda'],
                capture_output=True,
                timeout=5
            )
        except:
            pass

        # Clear our tracking
        with process_manager.lock:
            for pid in list(process_manager.processes.keys()):
                process_manager.processes[pid]['status'] = 'stopped'

        return {'status': 'success', 'killed': killed}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def get_checkpoints(puzzle_num):
    """Get all checkpoints for a puzzle"""
    try:
        db_path = os.path.join(DB_DIR, f'scan_data_puzzle_{puzzle_num}.db')
        if not os.path.exists(db_path):
            return []

        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                SELECT block_start, block_end, resume_position, keys_checked, completion_pct, saved_at, resumed
                FROM checkpoints
                WHERE resumed = 0
                ORDER BY saved_at DESC
                LIMIT 50
            ''')
            rows = cursor.fetchall()
        except sqlite3.OperationalError:
            rows = []

        conn.close()

        checkpoints = []
        for row in rows:
            checkpoints.append({
                'block_start': row[0],
                'block_end': row[1],
                'resume_position': row[2],
                'keys_checked': row[3],
                'completion_pct': row[4],
                'saved_at': row[5],
                'resumed': row[6]
            })

        return checkpoints
    except Exception as e:
        return {'error': str(e)}


def delete_checkpoint(puzzle_num, resume_position):
    """Delete a checkpoint from the database"""
    try:
        db_path = os.path.join(DB_DIR, f'scan_data_puzzle_{puzzle_num}.db')
        if not os.path.exists(db_path):
            return {'status': 'error', 'message': 'Database not found'}

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute('DELETE FROM checkpoints WHERE resume_position = ?', (resume_position,))
        deleted = cursor.rowcount

        conn.commit()
        conn.close()

        if deleted > 0:
            print(f"Deleted checkpoint at {resume_position[:12]}...")
            return {'status': 'ok', 'deleted': deleted}
        else:
            return {'status': 'error', 'message': 'Checkpoint not found'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def delete_partial_scan(puzzle_num, block_start):
    """Delete a partial scan from the database (my_scanned table)"""
    try:
        db_path = os.path.join(DB_DIR, f'scan_data_puzzle_{puzzle_num}.db')
        if not os.path.exists(db_path):
            return {'status': 'error', 'message': 'Database not found'}

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Delete from my_scanned table
        cursor.execute('DELETE FROM my_scanned WHERE block_start = ?', (block_start,))
        deleted_scans = cursor.rowcount

        # Also delete any associated checkpoints
        cursor.execute('DELETE FROM checkpoints WHERE block_start = ?', (block_start,))
        deleted_checkpoints = cursor.rowcount

        conn.commit()
        conn.close()

        print(f"Deleted partial scan at {block_start[:12]}... ({deleted_scans} scan records, {deleted_checkpoints} checkpoints)")
        return {'status': 'ok', 'deleted_scans': deleted_scans, 'deleted_checkpoints': deleted_checkpoints}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def generate_heatmap_data(puzzle_num, resolution=100):
    """Generate heatmap data showing scan density across keyspace"""
    try:
        db_path = os.path.join(DB_DIR, f'scan_data_puzzle_{puzzle_num}.db')
        if not os.path.exists(db_path):
            return {'error': 'No database found'}

        # Get puzzle range
        if puzzle_num in PUZZLE_RANGES:
            range_start, range_end = PUZZLE_RANGES[puzzle_num]
        else:
            range_start = 2 ** (puzzle_num - 1)
            range_end = 2 ** puzzle_num - 1

        range_size = range_end - range_start
        cell_size = range_size // resolution

        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()

        # Get all scanned blocks
        cursor.execute('SELECT block_start, block_end, keys_checked FROM my_scanned')
        my_blocks = cursor.fetchall()

        cursor.execute('SELECT block_start, block_end FROM pool_scanned')
        pool_blocks = cursor.fetchall()

        try:
            cursor.execute('SELECT block_start, block_end FROM skipped_blocks')
            skipped_blocks = cursor.fetchall()
        except:
            skipped_blocks = []

        conn.close()

        # Initialize heatmap
        heatmap = []
        for i in range(resolution):
            cell_start = range_start + (i * cell_size)
            cell_end = cell_start + cell_size - 1

            my_coverage = 0
            pool_coverage = 0
            skipped_coverage = 0
            total_keys_checked = 0

            # Calculate coverage for this cell
            for block in my_blocks:
                try:
                    b_start = int(block[0], 16)
                    b_end = int(block[1], 16)
                    keys = block[2] or 0

                    # Calculate overlap
                    overlap_start = max(cell_start, b_start)
                    overlap_end = min(cell_end, b_end)
                    if overlap_end > overlap_start:
                        overlap_size = overlap_end - overlap_start
                        my_coverage += overlap_size
                        total_keys_checked += keys * (overlap_size / (b_end - b_start + 1)) if b_end > b_start else 0
                except:
                    pass

            for block in pool_blocks:
                try:
                    b_start = int(block[0], 16)
                    b_end = int(block[1], 16)

                    overlap_start = max(cell_start, b_start)
                    overlap_end = min(cell_end, b_end)
                    if overlap_end > overlap_start:
                        pool_coverage += overlap_end - overlap_start
                except:
                    pass

            for block in skipped_blocks:
                try:
                    b_start = int(block[0], 16)
                    b_end = int(block[1], 16)

                    overlap_start = max(cell_start, b_start)
                    overlap_end = min(cell_end, b_end)
                    if overlap_end > overlap_start:
                        skipped_coverage += overlap_end - overlap_start
                except:
                    pass

            # Calculate percentages
            my_pct = (my_coverage / cell_size * 100) if cell_size > 0 else 0
            pool_pct = (pool_coverage / cell_size * 100) if cell_size > 0 else 0
            skipped_pct = (skipped_coverage / cell_size * 100) if cell_size > 0 else 0

            # Calculate priority score
            priority = calculate_range_priority(
                hex(cell_start)[2:].upper(),
                hex(cell_end)[2:].upper(),
                puzzle_num
            )

            heatmap.append({
                'index': i,
                'start': hex(cell_start)[2:].upper(),
                'end': hex(cell_end)[2:].upper(),
                'my_pct': round(my_pct, 2),
                'pool_pct': round(pool_pct, 2),
                'skipped_pct': round(skipped_pct, 2),
                'total_pct': round(my_pct + pool_pct, 2),
                'keys_checked': int(total_keys_checked),
                'priority_score': priority.get('score', 50),
                'heat': round(max(0, 100 - my_pct - pool_pct - skipped_pct), 1)  # Heat = uncovered area
            })

        return {
            'status': 'success',
            'puzzle': puzzle_num,
            'resolution': resolution,
            'range_start': hex(range_start)[2:].upper(),
            'range_end': hex(range_end)[2:].upper(),
            'cells': heatmap
        }
    except Exception as e:
        return {'error': str(e)}


def get_available_databases():
    """Find all scan_data_puzzle_*.db files"""
    databases = {}
    for f in os.listdir(DB_DIR):
        if f.startswith('scan_data_puzzle_') and f.endswith('.db'):
            try:
                puzzle_num = int(f.replace('scan_data_puzzle_', '').replace('.db', ''))
                databases[puzzle_num] = os.path.join(DB_DIR, f)
            except ValueError:
                continue
    return databases


def ensure_database_exists(puzzle_num):
    """Create database with proper schema if it doesn't exist"""
    db_path = os.path.join(DB_DIR, f'scan_data_puzzle_{puzzle_num}.db')
    if not os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Create all required tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS my_scanned (
                block_start TEXT PRIMARY KEY,
                block_end TEXT,
                scanned_at TEXT,
                keys_checked INTEGER,
                source TEXT DEFAULT 'local',
                completion_pct REAL DEFAULT 100.0
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pool_scanned (
                block_start TEXT,
                block_end TEXT,
                scraped_at TEXT,
                PRIMARY KEY (block_start, block_end)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS skipped_blocks (
                block_start TEXT PRIMARY KEY,
                block_end TEXT,
                reason TEXT,
                skipped_at TEXT
            )
        ''')
        conn.commit()
        conn.close()
    return db_path


def get_db_connection(puzzle_num, read_only=True):
    """Get database connection for a puzzle, creating it if needed"""
    db_path = ensure_database_exists(puzzle_num)
    if read_only:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5)
    else:
        conn = sqlite3.connect(db_path, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def get_puzzle_stats(puzzle_num):
    """Get statistics for a puzzle"""
    try:
        conn = get_db_connection(puzzle_num)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM pool_scanned")
        pool_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM my_scanned")
        my_count = cursor.fetchone()[0]

        cursor.execute("SELECT COALESCE(SUM(keys_checked), 0) FROM my_scanned")
        total_keys = cursor.fetchone()[0]

        # Get skipped count
        try:
            cursor.execute("SELECT COUNT(*) FROM skipped_blocks")
            skipped_count = cursor.fetchone()[0]
        except:
            skipped_count = 0

        conn.close()

        if puzzle_num in PUZZLE_RANGES:
            start, end = PUZZLE_RANGES[puzzle_num]
            keyspace_size = end - start + 1
        else:
            keyspace_size = 2 ** puzzle_num

        return {
            'puzzle': puzzle_num,
            'pool_blocks': pool_count,
            'my_blocks': my_count,
            'skipped_blocks': skipped_count,
            'total_blocks': pool_count + my_count,
            'keys_checked': total_keys,
            'keyspace_size': keyspace_size,
            'keyspace_hex': f"{keyspace_size:X}",
            'address': PUZZLE_PRESETS.get(puzzle_num, {}).get('address', 'Unknown'),
            'reward': PUZZLE_PRESETS.get(puzzle_num, {}).get('reward', 'Unknown')
        }
    except Exception as e:
        return {'error': str(e)}


def get_scanned_blocks(puzzle_num):
    """Get all scanned blocks for a puzzle"""
    try:
        conn = get_db_connection(puzzle_num)
        cursor = conn.cursor()

        cursor.execute("SELECT block_start, block_end, scraped_at FROM pool_scanned")
        pool_blocks = [{'start': row[0], 'end': row[1], 'time': row[2]} for row in cursor.fetchall()]

        cursor.execute("SELECT block_start, block_end, scanned_at, keys_checked, source FROM my_scanned")
        my_blocks = [{'start': row[0], 'end': row[1], 'time': row[2], 'keys': row[3], 'source': row[4]} for row in cursor.fetchall()]

        # Get skipped blocks
        try:
            cursor.execute("SELECT block_start, block_end, reason, skipped_at, queued_for_rescan FROM skipped_blocks")
            skipped_blocks = [
                {'start': row[0], 'end': row[1], 'reason': row[2], 'time': row[3], 'queued': row[4]}
                for row in cursor.fetchall()
            ]
        except:
            skipped_blocks = []

        # Get browser miner blocks
        browser_blocks = []
        try:
            cursor.execute("SELECT block_start, block_end, scanned_at, keys_checked, session_id FROM browser_scanned")
            browser_blocks = [
                {'start': row[0], 'end': row[1], 'time': row[2], 'keys': row[3], 'session': row[4]}
                for row in cursor.fetchall()
            ]
        except:
            pass  # Table may not exist yet

        conn.close()

        return {'pool': pool_blocks, 'mine': my_blocks, 'skipped': skipped_blocks, 'browser': browser_blocks}
    except Exception as e:
        return {'error': str(e), 'pool': [], 'mine': [], 'skipped': []}


def calculate_grid_coverage(puzzle_num, grid_start, grid_end, divisions=100):
    """Calculate coverage for a grid section"""
    try:
        blocks = get_scanned_blocks(puzzle_num)
        if 'error' in blocks:
            return {'error': blocks['error']}

        # Add currently running processes to the mine blocks (for real-time display)
        running_blocks = []
        for proc in process_manager.processes.values():
            if proc['status'] == 'running' and proc['puzzle'] == puzzle_num:
                keys_checked = proc.get('keys_checked', 0)
                completion = proc.get('completion', 0)
                if keys_checked > 0:
                    try:
                        start_int = int(proc['start'], 16)
                        end_int = int(proc['end'], 16)
                        total_range = end_int - start_int + 1
                        # Calculate actual scanned portion
                        if completion > 0:
                            actual_keys = int(total_range * completion / 100)
                        else:
                            actual_keys = keys_checked
                        actual_end = min(start_int + actual_keys, end_int)
                        running_blocks.append({
                            'start': proc['start'],
                            'end': hex(actual_end)[2:].upper(),
                            'time': 'running',
                            'keys': keys_checked,
                            'source': 'running',
                            'is_running': True
                        })
                    except (ValueError, TypeError):
                        pass

        # Combine saved blocks with running blocks
        all_my_blocks = blocks['mine'] + running_blocks

        grid_size = grid_end - grid_start
        cell_size = grid_size // divisions

        cells = []

        for i in range(divisions):
            cell_start = grid_start + (i * cell_size)
            cell_end = cell_start + cell_size - 1

            pool_coverage = 0
            my_coverage = 0
            skipped_coverage = 0
            browser_coverage = 0
            pool_block_count = 0
            running_coverage = 0  # Track running process coverage separately
            my_block_count = 0
            skipped_block_count = 0
            browser_block_count = 0
            queued_count = 0

            # Check pool blocks
            for block in blocks['pool']:
                try:
                    b_start = int(block['start'], 16)
                    b_end = int(block['end'], 16)

                    overlap_start = max(cell_start, b_start)
                    overlap_end = min(cell_end, b_end)

                    if overlap_start <= overlap_end:
                        overlap_size = overlap_end - overlap_start + 1
                        pool_coverage += overlap_size
                        pool_block_count += 1
                except (ValueError, TypeError):
                    continue

            # Check my blocks (including running processes)
            has_running_block = False
            for block in all_my_blocks:
                try:
                    b_start = int(block['start'], 16)
                    b_end = int(block['end'], 16)

                    overlap_start = max(cell_start, b_start)
                    overlap_end = min(cell_end, b_end)

                    if overlap_start <= overlap_end:
                        overlap_size = overlap_end - overlap_start + 1
                        my_coverage += overlap_size
                        my_block_count += 1
                        if block.get('is_running'):
                            has_running_block = True
                            running_coverage += overlap_size
                except (ValueError, TypeError):
                    continue

            # Check skipped blocks
            for block in blocks['skipped']:
                try:
                    b_start = int(block['start'], 16)
                    b_end = int(block['end'], 16)

                    overlap_start = max(cell_start, b_start)
                    overlap_end = min(cell_end, b_end)

                    if overlap_start <= overlap_end:
                        overlap_size = overlap_end - overlap_start + 1
                        skipped_coverage += overlap_size
                        skipped_block_count += 1
                        if block.get('queued'):
                            queued_count += 1
                except (ValueError, TypeError):
                    continue

            # Check browser miner blocks
            for block in blocks.get('browser', []):
                try:
                    b_start = int(block['start'], 16)
                    b_end = int(block['end'], 16)

                    overlap_start = max(cell_start, b_start)
                    overlap_end = min(cell_end, b_end)

                    if overlap_start <= overlap_end:
                        overlap_size = overlap_end - overlap_start + 1
                        browser_coverage += overlap_size
                        browser_block_count += 1
                except (ValueError, TypeError):
                    continue

            pool_pct = min(100, (pool_coverage / cell_size) * 100) if cell_size > 0 else 0
            my_pct = min(100, (my_coverage / cell_size) * 100) if cell_size > 0 else 0
            skipped_pct = min(100, (skipped_coverage / cell_size) * 100) if cell_size > 0 else 0
            browser_pct = min(100, (browser_coverage / cell_size) * 100) if cell_size > 0 else 0
            total_pct = min(100, pool_pct + my_pct + browser_pct)

            # For tiny coverage, use a minimum display value if blocks exist
            # This ensures scanned blocks are visible even at large scale views
            display_my_pct = my_pct if my_pct >= 0.0001 else (0.0001 if my_block_count > 0 else 0)
            display_pool_pct = pool_pct if pool_pct >= 0.0001 else (0.0001 if pool_block_count > 0 else 0)
            display_browser_pct = browser_pct if browser_pct >= 0.0001 else (0.0001 if browser_block_count > 0 else 0)
            display_total_pct = display_my_pct + display_pool_pct + display_browser_pct

            cells.append({
                'index': i,
                'start_hex': f"{cell_start:X}",
                'end_hex': f"{cell_end:X}",
                'key_count': cell_size,  # Number of keys in this cell
                'pool_pct': round(display_pool_pct, 6),
                'my_pct': round(display_my_pct, 6),
                'skipped_pct': round(skipped_pct, 6),
                'browser_pct': round(display_browser_pct, 6),
                'total_pct': round(display_total_pct, 6),
                'pool_blocks': pool_block_count,
                'my_blocks': my_block_count,
                'skipped_blocks': skipped_block_count,
                'browser_blocks': browser_block_count,
                'queued_blocks': queued_count,
                'has_my_scans': my_block_count > 0,  # Boolean flag for any scans
                'has_pool_scans': pool_block_count > 0,
                'has_browser_scans': browser_block_count > 0,  # Browser miner scans
                'has_running': has_running_block,  # Currently being mined
                'running_pct': round(min(100, (running_coverage / cell_size) * 100), 4) if cell_size > 0 else 0
            })

        return {
            'grid_start': f"{grid_start:X}",
            'grid_end': f"{grid_end:X}",
            'cell_size': cell_size,
            'cell_size_hex': f"{cell_size:X}",
            'divisions': divisions,
            'cells': cells
        }
    except Exception as e:
        return {'error': str(e)}


# HTML Template with mining controls
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KeyHunt Visualizer v3.0</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }

        .container { max-width: 1600px; margin: 0 auto; }

        h1 {
            text-align: center;
            margin-bottom: 20px;
            color: #00d4ff;
            text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        }

        .controls {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
            align-items: center;
        }

        .control-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        label { font-weight: bold; color: #aaa; font-size: 13px; }

        select, button, input[type="text"] {
            padding: 8px 15px;
            border: none;
            border-radius: 5px;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        select, input[type="text"] {
            background: #2a2a4a;
            color: #fff;
            border: 1px solid #444;
        }

        select:hover, input[type="text"]:hover { border-color: #00d4ff; }

        button {
            background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
            color: #000;
            font-weight: bold;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0, 212, 255, 0.4);
        }

        button:disabled {
            background: #444;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        button.danger { background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%); color: #fff; }
        button.success { background: linear-gradient(135deg, #00ff88 0%, #00cc66 100%); color: #000; }
        button.warning { background: linear-gradient(135deg, #ffaa00 0%, #cc8800 100%); color: #000; }
        button.small { padding: 5px 10px; font-size: 11px; }

        /* KEY FOUND ALERT - JACKPOT! */
        .key-found-alert {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.95);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 9999;
            animation: alertPulse 0.5s ease infinite;
        }
        .key-found-alert.show { display: flex; }
        .key-found-content {
            background: linear-gradient(135deg, #1a5f1a 0%, #2a8a2a 100%);
            border: 4px solid #00ff88;
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            max-width: 600px;
            box-shadow: 0 0 100px rgba(0, 255, 136, 0.8);
        }
        .key-found-content h1 { font-size: 48px; color: #00ff88; margin-bottom: 20px; animation: bounce 0.5s ease infinite; }
        .key-found-content .details { background: #000; padding: 20px; border-radius: 10px; margin: 20px 0; font-family: monospace; word-break: break-all; color: #ffaa00; }
        .key-found-content button { font-size: 18px; padding: 15px 40px; margin-top: 20px; }
        @keyframes alertPulse { 0%, 100% { background: rgba(0, 0, 0, 0.95); } 50% { background: rgba(0, 50, 0, 0.95); } }
        @keyframes bounce { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.1); } }

        /* Pool scraper panel */
        .pool-panel {
            background: rgba(74, 158, 255, 0.1);
            border: 1px solid #4a9eff;
            border-radius: 5px;
            padding: 10px;
            margin-top: 8px;
        }
        .pool-panel h4 { color: #4a9eff; font-size: 11px; margin-bottom: 8px; }
        .pool-status { font-size: 10px; color: #888; margin-bottom: 5px; }

        .main-layout {
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 20px;
        }

        @media (max-width: 1200px) { .main-layout { grid-template-columns: 1fr; } }

        .stats-panel {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 10px;
            margin-bottom: 15px;
        }

        .stat-card {
            background: rgba(42, 42, 74, 0.8);
            border-radius: 10px;
            padding: 10px;
            text-align: center;
            border: 1px solid #444;
        }

        .stat-card h3 { color: #888; font-size: 10px; text-transform: uppercase; margin-bottom: 3px; }
        .stat-card .value { font-size: 18px; font-weight: bold; color: #00d4ff; }
        .stat-card.pool .value { color: #4a9eff; }
        .stat-card.mine .value { color: #00ff88; }
        .stat-card.skipped .value { color: #ff4444; }
        .stat-card.total .value { color: #ffaa00; }

        .breadcrumb {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
            padding: 8px 12px;
            background: rgba(42, 42, 74, 0.5);
            border-radius: 5px;
            flex-wrap: wrap;
            font-family: monospace;
            font-size: 11px;
        }

        .breadcrumb span { color: #888; }
        .breadcrumb a { color: #00d4ff; text-decoration: none; cursor: pointer; }
        .breadcrumb a:hover { text-decoration: underline; }

        .grid-container {
            background: rgba(42, 42, 74, 0.5);
            border-radius: 10px;
            padding: 15px;
            border: 1px solid #444;
        }

        .grid-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            flex-wrap: wrap;
            gap: 10px;
        }

        .grid-header h2 { color: #00d4ff; font-size: 16px; }
        .grid-info { font-family: monospace; color: #888; font-size: 10px; }

        .grid {
            display: grid;
            grid-template-columns: repeat(10, 1fr);
            gap: 3px;
            margin-bottom: 12px;
        }

        .cell {
            aspect-ratio: 1;
            border-radius: 4px;
            cursor: pointer;
            position: relative;
            transition: all 0.2s ease;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            font-size: 8px;
            font-weight: bold;
            color: rgba(255,255,255,0.9);
            border: 1px solid rgba(255,255,255,0.1);
            overflow: hidden;
        }

        .cell:hover {
            transform: scale(1.15);
            z-index: 10;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        }

        .cell .key-count {
            font-size: 7px;
            color: rgba(255,255,255,0.6);
            margin-top: 1px;
        }

        .cell .mine-btn {
            position: absolute;
            top: 2px;
            right: 2px;
            width: 14px;
            height: 14px;
            border-radius: 3px;
            background: rgba(0, 212, 255, 0.9);
            border: none;
            cursor: pointer;
            font-size: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0;
            transition: opacity 0.2s;
            color: #000;
        }

        .cell:hover .mine-btn { opacity: 1; }
        .cell .mine-btn:hover { background: #00ff88; transform: scale(1.2); }
        .cell .mine-btn.running { background: #ff4444; opacity: 1; animation: pulse 1s infinite; }

        .cell .skip-btn {
            position: absolute;
            bottom: 2px;
            right: 2px;
            width: 14px;
            height: 14px;
            border-radius: 3px;
            background: rgba(255, 68, 68, 0.9);
            border: none;
            cursor: pointer;
            font-size: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0;
            transition: opacity 0.2s;
            color: #fff;
        }

        .cell:hover .skip-btn { opacity: 1; }
        .cell .skip-btn:hover { background: #ff6666; transform: scale(1.2); }

        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

        .cell.empty { background: #1a1a2e; }
        .cell.pool-only { background: linear-gradient(135deg, #1e3a5f 0%, #4a9eff 100%); }
        .cell.mine-only { background: linear-gradient(135deg, #1e5f3a 0%, #00ff88 100%); }
        .cell.browser-only { background: linear-gradient(135deg, #4a1e5f 0%, #aa44ff 100%); }
        .cell.both { background: linear-gradient(135deg, #5f4a1e 0%, #ffaa00 100%); }
        .cell.skipped { background: linear-gradient(135deg, #5f1e1e 0%, #ff4444 100%); }
        .cell.queued { background: linear-gradient(135deg, #5f1e5f 0%, #ff44ff 100%); }
        .cell.partial { background: linear-gradient(135deg, #2a2a4a 0%, #3a3a5a 100%); }
        .cell.filtered { background: linear-gradient(135deg, #5f3a1e 0%, #ff9900 100%); }
        .cell.has-filter-matches { box-shadow: inset 0 0 0 2px #ff9900; }
        .cell.running { animation: mining 2s infinite; border: 2px solid #00ff88; }
        .cell.has-running-child { animation: mining-parent 3s infinite; border: 2px dashed #00ff88; }
        .cell.synced { background: linear-gradient(135deg, #1e5f3a 0%, #00cc77 100%); }  /* Synced blocks same as mine */

        @keyframes mining { 0%, 100% { box-shadow: 0 0 5px #00ff88; } 50% { box-shadow: 0 0 20px #00ff88; } }
        @keyframes mining-parent { 0%, 100% { box-shadow: 0 0 3px #00ff88; opacity: 0.9; } 50% { box-shadow: 0 0 15px #00ff88; opacity: 1; } }

        /* Now Scanning Status Bar */
        .scanning-status {
            background: linear-gradient(135deg, #0a2a1a 0%, #1a4a2a 100%);
            border: 2px solid #00ff88;
            border-radius: 10px;
            padding: 15px 20px;
            margin-bottom: 15px;
            display: none;
            animation: pulse-border 2s infinite;
        }
        .scanning-status.active { display: block; }
        .scanning-status h3 { color: #00ff88; margin-bottom: 10px; display: flex; align-items: center; gap: 10px; }
        .scanning-status h3::before { content: ''; width: 12px; height: 12px; background: #00ff88; border-radius: 50%; animation: blink 1s infinite; }
        .scanning-status .scan-range { font-family: monospace; font-size: 12px; color: #aaffaa; margin-bottom: 8px; word-break: break-all; }
        .scanning-status .scan-stats { display: flex; gap: 30px; flex-wrap: wrap; }
        .scanning-status .scan-stat { display: flex; flex-direction: column; }
        .scanning-status .scan-stat .label { font-size: 10px; color: #888; }
        .scanning-status .scan-stat .value { font-size: 18px; font-weight: bold; color: #00ff88; }
        .scanning-status .scan-stat .value.speed { color: #00d4ff; }
        .scanning-status .progress-container { margin-top: 10px; }
        .scanning-status .progress-bar { height: 8px; background: #1a1a2e; border-radius: 4px; overflow: hidden; }
        .scanning-status .progress-bar .fill { height: 100%; background: linear-gradient(90deg, #00ff88, #00d4ff); transition: width 0.3s; }
        @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
        @keyframes pulse-border { 0%, 100% { border-color: #00ff88; } 50% { border-color: #00aa66; } }

        .legend {
            display: flex;
            justify-content: center;
            gap: 15px;
            flex-wrap: wrap;
            font-size: 11px;
        }

        .legend-item { display: flex; align-items: center; gap: 5px; }
        .legend-color { width: 14px; height: 14px; border-radius: 3px; }
        .legend-color.empty { background: #1a1a2e; border: 1px solid #444; }
        .legend-color.pool { background: linear-gradient(135deg, #1e3a5f 0%, #4a9eff 100%); }
        .legend-color.browser { background: linear-gradient(135deg, #4a1e5f 0%, #aa44ff 100%); }
        .legend-color.mine { background: linear-gradient(135deg, #1e5f3a 0%, #00ff88 100%); }
        .legend-color.both { background: linear-gradient(135deg, #5f4a1e 0%, #ffaa00 100%); }
        .legend-color.skipped { background: linear-gradient(135deg, #5f1e1e 0%, #ff4444 100%); }

        .sidebar { display: flex; flex-direction: column; gap: 12px; }

        .panel {
            background: rgba(42, 42, 74, 0.8);
            border-radius: 10px;
            padding: 12px;
            border: 1px solid #444;
        }

        .panel h3 {
            color: #00d4ff;
            margin-bottom: 8px;
            font-size: 13px;
            border-bottom: 1px solid #444;
            padding-bottom: 6px;
        }

        .process-list { max-height: 180px; overflow-y: auto; }

        .process-item {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 5px;
            padding: 8px;
            margin-bottom: 6px;
            font-size: 10px;
        }

        .process-item .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 4px;
        }

        .process-item .status { padding: 2px 6px; border-radius: 3px; font-size: 9px; }
        .process-item .status.running { background: #00ff88; color: #000; }
        .process-item .status.stopped { background: #ff4444; color: #fff; }
        .process-item .status.completed { background: #888; color: #fff; }
        .process-item .range { font-family: monospace; color: #888; font-size: 9px; word-break: break-all; }
        .process-item .speed { color: #00d4ff; font-weight: bold; font-size: 12px; }
        .process-item .keys { color: #00ff88; font-size: 9px; }
        .process-item .progress-bar { height: 4px; background: #333; border-radius: 2px; margin-top: 4px; overflow: hidden; }
        .process-item .progress-bar .fill { height: 100%; background: #00d4ff; transition: width 0.3s; }

        .queue-list { max-height: 120px; overflow-y: auto; }

        .queue-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 5px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 3px;
            margin-bottom: 4px;
            font-size: 9px;
        }

        .queue-item .range { font-family: monospace; color: #ff4444; }

        .tooltip {
            position: fixed;
            background: rgba(0, 0, 0, 0.95);
            border: 1px solid #00d4ff;
            border-radius: 8px;
            padding: 10px;
            font-size: 10px;
            pointer-events: none;
            z-index: 1000;
            max-width: 380px;
            font-family: monospace;
            display: none;
        }

        .tooltip .row { display: flex; justify-content: space-between; margin: 2px 0; gap: 15px; }
        .tooltip .label { color: #888; }
        .tooltip .value { color: #00d4ff; text-align: right; }
        .tooltip hr { border-color: #444; margin: 5px 0; }

        .coverage-bar {
            height: 22px;
            background: #1a1a2e;
            border-radius: 5px;
            margin-bottom: 12px;
            overflow: hidden;
            display: flex;
            position: relative;
        }

        .coverage-bar .mine-bar { background: linear-gradient(90deg, #00ff88 0%, #00cc6a 100%); height: 100%; transition: width 0.5s ease; }
        .coverage-bar .pool-bar { background: linear-gradient(90deg, #4a9eff 0%, #0066cc 100%); height: 100%; transition: width 0.5s ease; }
        .coverage-bar .browser-bar { background: linear-gradient(90deg, #aa44ff 0%, #7700cc 100%); height: 100%; transition: width 0.5s ease; }
        .coverage-bar .skipped-bar { background: linear-gradient(90deg, #ff4444 0%, #cc0000 100%); height: 100%; transition: width 0.5s ease; }
        .coverage-bar .label { position: absolute; right: 10px; top: 50%; transform: translateY(-50%); font-weight: bold; color: #fff; text-shadow: 0 0 5px #000; font-size: 11px; }

        .loading { display: flex; justify-content: center; align-items: center; height: 200px; font-size: 14px; color: #888; }
        .loading::after { content: ''; width: 20px; height: 20px; border: 3px solid #444; border-top-color: #00d4ff; border-radius: 50%; margin-left: 10px; animation: spin 1s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }

        .target-info {
            background: rgba(0, 212, 255, 0.1);
            border: 1px solid #00d4ff;
            border-radius: 5px;
            padding: 8px 12px;
            margin-bottom: 12px;
            font-size: 11px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }

        .target-info .address { font-family: monospace; color: #ffaa00; }

        .filter-panel {
            background: rgba(255, 68, 68, 0.1);
            border: 1px solid #ff4444;
            border-radius: 5px;
            padding: 10px;
            margin-top: 8px;
        }

        .filter-panel h4 { color: #ff4444; font-size: 11px; margin-bottom: 8px; }

        .filter-options { display: flex; flex-direction: column; gap: 6px; }

        .filter-options label {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
            cursor: pointer;
            color: #ccc;
        }

        .filter-options input[type="checkbox"] { width: 14px; height: 14px; }

        input[type="text"].small { padding: 5px 8px; font-size: 11px; width: 100px; }

        .checkbox-label {
            display: flex;
            align-items: center;
            gap: 5px;
            font-size: 12px;
            cursor: pointer;
        }

        .checkbox-label input { width: 14px; height: 14px; }
    </style>
</head>
<body>
    <!-- KEY FOUND ALERT - JACKPOT NOTIFICATION -->
    <div class="key-found-alert" id="key-found-alert">
        <div class="key-found-content">
            <h1>JACKPOT! KEY FOUND!</h1>
            <p style="font-size: 24px; margin-bottom: 20px;">You found a private key!</p>
            <div class="details" id="key-found-details">Loading details...</div>
            <p style="color: #00ff88;">Check Found.txt for full details!</p>
            <button class="success" onclick="dismissKeyFoundAlert()">AWESOME! Close</button>
        </div>
    </div>

    <div class="container">
        <h1>KeyHunt Visualizer v3.0</h1>

        <div class="controls">
            <div class="control-group">
                <label>Puzzle:</label>
                <select id="puzzle-select"></select>
            </div>
            <div class="control-group">
                <label>Refresh:</label>
                <select id="refresh-rate">
                    <option value="5000">5 sec</option>
                    <option value="10000" selected>10 sec</option>
                    <option value="30000">30 sec</option>
                    <option value="60000">1 min</option>
                    <option value="120000">2 min</option>
                    <option value="0">Manual</option>
                </select>
            </div>
            <div class="control-group">
                <button onclick="refreshData()">Refresh</button>
                <button id="zoom-out-btn" onclick="zoomOut()" disabled>Zoom Out</button>
                <button onclick="resetView()">Reset</button>
            </div>
            <div class="control-group">
                <label class="checkbox-label">
                    <input type="checkbox" id="show-filters" checked onchange="loadGrid()"> Show Skipped
                </label>
                <label class="checkbox-label">
                    <input type="checkbox" id="use-gpu" checked> GPU
                </label>
            </div>
        </div>

        <div class="target-info">
            <div>
                <strong>Target:</strong> <span class="address" id="target-address">Loading...</span>
            </div>
            <div>
                <span style="color: #00ff88; font-weight: bold;" id="target-reward"></span>
                <span style="color: #888; margin-left: 10px;">Cell Size: <span id="cell-size-display">-</span> keys</span>
            </div>
        </div>

        <!-- Now Scanning Status Bar -->
        <div class="scanning-status" id="scanning-status">
            <h3>Now Scanning</h3>
            <div class="scan-range" id="scan-range">-</div>
            <div class="scan-stats">
                <div class="scan-stat">
                    <span class="label">Speed</span>
                    <span class="value speed" id="scan-speed">0 Mk/s</span>
                </div>
                <div class="scan-stat">
                    <span class="label">Keys Checked</span>
                    <span class="value" id="scan-keys">0</span>
                </div>
                <div class="scan-stat">
                    <span class="label">Progress</span>
                    <span class="value" id="scan-progress">0%</span>
                </div>
                <div class="scan-stat">
                    <span class="label">GPU</span>
                    <span class="value" id="scan-gpu">-</span>
                </div>
            </div>
            <div class="progress-container">
                <div class="progress-bar"><div class="fill" id="scan-progress-bar" style="width: 0%"></div></div>
            </div>
        </div>

        <div class="stats-panel">
            <div class="stat-card mine">
                <h3>My Blocks + Synced</h3>
                <div class="value" id="my-blocks">-</div>
            </div>
            <div class="stat-card pool">
                <h3>Pool Blocks</h3>
                <div class="value" id="pool-blocks">-</div>
            </div>
            <div class="stat-card skipped">
                <h3>Skipped</h3>
                <div class="value" id="skipped-blocks">-</div>
            </div>
            <div class="stat-card total">
                <h3>Keys Checked</h3>
                <div class="value" id="keys-checked">-</div>
            </div>
            <div class="stat-card">
                <h3>Active Mining</h3>
                <div class="value" id="active-processes">0</div>
            </div>
        </div>

        <div class="main-layout">
            <div class="grid-section">
                <div class="breadcrumb" id="breadcrumb">
                    <span>Path:</span>
                    <a onclick="resetView()">Full Keyspace</a>
                </div>

                <div class="grid-container">
                    <div class="grid-header">
                        <h2>Keyspace Coverage Grid (10x10 = 100 cells)</h2>
                        <div class="grid-info" id="grid-info"></div>
                    </div>

                    <div class="coverage-bar">
                        <div class="mine-bar" id="mine-bar"></div>
                        <div class="pool-bar" id="pool-bar"></div>
                        <div class="browser-bar" id="browser-bar"></div>
                        <div class="skipped-bar" id="skipped-bar"></div>
                        <span class="label" id="coverage-label">0%</span>
                    </div>

                    <div class="grid" id="grid">
                        <div class="loading">Loading grid data</div>
                    </div>

                    <div class="legend">
                        <div class="legend-item"><div class="legend-color empty"></div><span>Unscanned</span></div>
                        <div class="legend-item"><div class="legend-color mine"></div><span>My Scans</span></div>
                        <div class="legend-item"><div class="legend-color pool"></div><span>Pool</span></div>
                        <div class="legend-item"><div class="legend-color browser"></div><span>Browser</span></div>
                        <div class="legend-item"><div class="legend-color both"></div><span>Both</span></div>
                        <div class="legend-item"><div class="legend-color skipped"></div><span>Skipped</span></div>
                        <div class="legend-item"><div class="legend-color" style="background: linear-gradient(135deg, #5f3a1e 0%, #ff9900 100%);"></div><span>Filter Match</span></div>
                    </div>
                </div>
            </div>

            <div class="sidebar">
                <div class="panel">
                    <h3>Active Mining Processes</h3>
                    <div class="process-list" id="process-list">
                        <div style="color: #888; text-align: center; padding: 15px;">No active processes</div>
                    </div>
                    <div style="display: flex; gap: 5px; margin-top: 8px;">
                        <button class="danger" onclick="stopAllMining()" style="flex: 1;">Stop All</button>
                        <button onclick="clearCompletedProcesses()" style="flex: 1;">Clear Done</button>
                    </div>
                </div>

                <div class="panel">
                    <h3 style="color: #ff9900;">Batch Mining Controls</h3>
                    <div style="display: flex; flex-direction: column; gap: 8px;">
                        <button class="success" onclick="scanSmartPriority()" style="width: 100%; background: linear-gradient(90deg, #00d4ff, #00ff88);">Smart Priority Scan</button>
                        <button onclick="scanAllVisibleBlocks()" style="width: 100%;">Scan All Unscanned Blocks</button>
                        <div style="display: flex; gap: 5px; align-items: center;">
                            <select id="random-block-count" style="flex: 1;">
                                <option value="10">10 blocks</option>
                                <option value="20">20 blocks</option>
                                <option value="30">30 blocks</option>
                                <option value="50" selected>50 blocks</option>
                                <option value="75">75 blocks</option>
                                <option value="100">100 blocks</option>
                            </select>
                            <button onclick="scanRandomBlocks()" style="flex: 1;">Scan Random</button>
                        </div>
                        <button onclick="resumePartialBlocks()" style="width: 100%;">Resume Partial Blocks</button>
                        <button class="danger" onclick="killAllGPUProcesses()" style="width: 100%;">Kill All GPU Services</button>
                    </div>
                    <div style="font-size: 9px; color: #666; margin-top: 8px; text-align: center;">
                        Smart Priority uses solved puzzle patterns
                    </div>
                </div>

                <div class="panel">
                    <h3>Quick Mine Range</h3>
                    <div style="font-size: 11px; margin-bottom: 8px;">
                        <label>Start (hex):</label>
                        <input type="text" id="quick-start" placeholder="400000000000000000" style="width: 100%; margin: 3px 0;">
                        <label>End (hex):</label>
                        <input type="text" id="quick-end" placeholder="400000100000000000" style="width: 100%; margin: 3px 0;">
                    </div>
                    <button class="success" onclick="quickMine()" style="width: 100%;">Start Mining</button>
                </div>

                <div class="panel">
                    <h3>Mark Block as Skipped</h3>
                    <div style="font-size: 11px; margin-bottom: 8px;">
                        <label>Start (hex):</label>
                        <input type="text" id="skip-start" placeholder="400000000000000000" style="width: 100%; margin: 3px 0;">
                        <label>End (hex):</label>
                        <input type="text" id="skip-end" placeholder="400000100000000000" style="width: 100%; margin: 3px 0;">
                        <label>Reason:</label>
                        <select id="skip-reason" style="width: 100%; margin: 3px 0;">
                            <option value="filter">Filter Pattern</option>
                            <option value="manual">Manual Skip</option>
                            <option value="low_priority">Low Priority</option>
                        </select>
                    </div>
                    <button class="danger" onclick="manualSkip()" style="width: 100%;">Mark as Skipped</button>
                </div>

                <div class="panel">
                    <h3>Skipped Blocks Queue</h3>
                    <div class="queue-list" id="queue-list">
                        <div style="color: #888; text-align: center; padding: 10px; font-size: 10px;">No queued blocks</div>
                    </div>
                    <div style="display: flex; gap: 5px; margin-top: 8px;">
                        <button class="warning" onclick="mineNextQueued()" style="flex: 1;">Mine Next</button>
                        <button onclick="loadQueueList()" style="flex: 1;">Refresh</button>
                    </div>
                </div>

                <div class="panel">
                    <h3>Pool Scraper (btcpuzzle.info)</h3>
                    <div class="pool-status" id="pool-status">Last scrape: Never</div>
                    <div id="pool-results" style="font-size: 10px; color: #888; margin-bottom: 8px;"></div>
                    <div style="display: flex; gap: 5px; margin-bottom: 8px;">
                        <button class="success" onclick="scrapePool()" style="flex: 2;" id="scrape-btn">Scrape Now</button>
                        <button id="auto-scrape-btn" onclick="toggleAutoScrape()" style="flex: 1; font-size: 9px;">Auto: OFF</button>
                    </div>
                    <div id="auto-scrape-status" style="font-size: 9px; color: #666; text-align: center;">
                        Auto-scrape every 5 minutes when enabled
                    </div>
                </div>

                <div class="panel">
                    <h3 style="color: #00d4ff;">Pool Pattern Analysis</h3>
                    <div id="pool-analysis" style="font-size: 10px; color: #888; margin-bottom: 8px;">
                        Click to analyze pool scanning patterns
                    </div>
                    <button onclick="analyzePoolPatterns()" style="width: 100%;">Analyze Patterns</button>
                    <div id="pool-predictions" style="max-height: 150px; overflow-y: auto; margin-top: 8px; display: none;"></div>
                </div>

                <div class="panel">
                    <h3 style="color: #ff9900;">Smart Batch Filter</h3>
                    <div style="font-size: 10px; color: #888; margin-bottom: 8px;">
                        Skip pattern addresses (AAA, FFF, etc.) to reduce search space
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 5px; margin-bottom: 8px;">
                        <label class="checkbox-label" style="font-size: 10px;">
                            <input type="checkbox" id="filter-iter3" checked> Skip 3+ repeated (AAA, 111)
                        </label>
                        <label class="checkbox-label" style="font-size: 10px;">
                            <input type="checkbox" id="filter-iter4"> Skip 4+ repeated (AAAA, 1111)
                        </label>
                        <label class="checkbox-label" style="font-size: 10px;">
                            <input type="checkbox" id="filter-alphanum" checked> Skip all-letters/numbers
                        </label>
                    </div>
                    <div style="margin-bottom: 8px;">
                        <label style="font-size: 10px; color: #888;">Sub-range size:</label>
                        <input type="text" id="filter-subrange" value="10000000" style="width: 100%; font-size: 10px; padding: 4px;">
                    </div>
                    <div id="filter-results" style="font-size: 10px; color: #888; margin-bottom: 8px; display: none;"></div>
                    <div style="display: flex; gap: 5px;">
                        <button class="warning" onclick="previewFilter()" style="flex: 1; font-size: 10px;">Preview</button>
                        <button onclick="applyFilter()" style="flex: 1; font-size: 10px;">Apply & Save</button>
                    </div>
                </div>

                <div class="panel" id="found-keys-panel" style="display: none;">
                    <h3 style="color: #00ff88;">KEYS FOUND!</h3>
                    <div id="found-keys-list" style="max-height: 250px; overflow-y: auto;"></div>
                </div>

                <div class="panel">
                    <h3 style="color: #00ff88;">Session Statistics</h3>
                    <div id="session-stats" style="font-size: 10px;">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px;">
                            <div>Session Time:</div><div id="stat-time">-</div>
                            <div>Keys Checked:</div><div id="stat-keys" style="color: #00ff88;">-</div>
                            <div>Keys/Hour:</div><div id="stat-rate">-</div>
                            <div>Current Speed:</div><div id="stat-speed" style="color: #ffaa00;">-</div>
                            <div>Blocks Done:</div><div id="stat-blocks">-</div>
                            <div>Active GPUs:</div><div id="stat-gpus" style="color: #00d4ff;">-</div>
                        </div>
                    </div>
                    <button onclick="testAlarm()" style="margin-top: 8px; padding: 4px 8px; font-size: 10px; background: #444; color: #fff; border: 1px solid #666; border-radius: 3px; cursor: pointer;">Test Alarm</button>
                </div>

                <div class="panel">
                    <h3 style="color: #ff9900;">GPU Status</h3>
                    <div id="gpu-list" style="font-size: 10px; max-height: 120px; overflow-y: auto;">
                        Loading GPUs...
                    </div>
                    <button onclick="refreshGPUs()" style="width: 100%; margin-top: 5px; font-size: 10px;">Refresh GPUs</button>
                </div>

                <div class="panel">
                    <h3 style="color: #00d4ff;">Pattern Analysis</h3>
                    <div id="pattern-analysis" style="font-size: 10px; max-height: 300px; overflow-y: auto;">
                        <button onclick="runPatternAnalysis()" style="width: 100%;">Analyze Solved Puzzles</button>
                    </div>
                    <div id="pattern-recommendations" style="margin-top: 8px; display: none;">
                        <div style="font-size: 10px; font-weight: bold; margin-bottom: 5px; color: #00d4ff;">Quick Scan Recommendations:</div>
                        <div style="display: flex; flex-direction: column; gap: 4px;">
                            <button onclick="scanRecommendedRange('lower')" style="font-size: 9px; padding: 6px;">Scan Lower 40% (Historical Bias)</button>
                            <button onclick="scanRecommendedRange('middle')" style="font-size: 9px; padding: 6px;">Scan Middle 20% (Balanced)</button>
                            <button onclick="scanRecommendedRange('upper')" style="font-size: 9px; padding: 6px;">Scan Upper 40% (Alternative)</button>
                            <button onclick="scanRecommendedRange('hot_digits')" style="font-size: 9px; padding: 6px; background: linear-gradient(90deg, #ff6600, #ffaa00);">Scan Hot Digit Ranges</button>
                        </div>
                    </div>
                </div>

                <div class="panel">
                    <h3 style="color: #ff6600;">Heat Map</h3>
                    <div id="heatmap-container" style="height: 100px; background: #1a1a2e; border-radius: 4px; overflow: hidden;">
                        <div id="heatmap" style="display: flex; height: 100%;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 8px; color: #666; margin-top: 4px;">
                        <span>0%</span>
                        <span style="color: #888;">Uncovered = Hot (Red)</span>
                        <span>100%</span>
                    </div>
                    <button onclick="loadHeatmap()" style="width: 100%; margin-top: 5px; font-size: 10px;">Refresh Heat Map</button>
                </div>

                <div class="panel">
                    <h3 style="color: #aa88ff;">Scan Scheduler</h3>
                    <div style="font-size: 10px;">
                        <div style="margin-bottom: 8px;">
                            <label><input type="checkbox" id="scheduler-enabled" onchange="toggleScheduler()"> Enable Scheduler</label>
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px; margin-bottom: 8px;">
                            <div>
                                <label style="font-size: 9px;">Start Time</label>
                                <input type="time" id="schedule-start" style="width: 100%;" value="22:00">
                            </div>
                            <div>
                                <label style="font-size: 9px;">End Time</label>
                                <input type="time" id="schedule-end" style="width: 100%;" value="06:00">
                            </div>
                        </div>
                        <div style="margin-bottom: 8px;">
                            <label style="font-size: 9px;">Blocks per Session</label>
                            <select id="schedule-blocks" style="width: 100%;">
                                <option value="10">10 blocks</option>
                                <option value="25">25 blocks</option>
                                <option value="50" selected>50 blocks</option>
                                <option value="100">100 blocks</option>
                                <option value="0">Until end time</option>
                            </select>
                        </div>
                        <button onclick="saveSchedule()" style="width: 100%;">Save Schedule</button>
                        <div id="scheduler-status" style="margin-top: 5px; font-size: 9px; color: #888;"></div>
                    </div>
                </div>

                <div class="panel">
                    <h3 style="color: #ffaa00;">⏸ Paused Scans</h3>
                    <div id="checkpoint-list" style="font-size: 9px; max-height: 180px; overflow-y: auto;">
                        <div style="color: #888; text-align: center; padding: 10px;">Click refresh to load</div>
                    </div>
                    <button onclick="loadCheckpoints()" style="width: 100%; margin-top: 5px; font-size: 10px;">🔄 Refresh Paused Scans</button>
                </div>

                <div class="panel" id="sync-panel">
                    <h3 style="color: #00d4ff;">
                        <span style="display: inline-flex; align-items: center; gap: 6px;">
                            Team Sync
                            <span id="sync-status-dot" style="width: 8px; height: 8px; border-radius: 50%; background: #666;" title="Connecting..."></span>
                        </span>
                    </h3>
                    <div id="sync-status" style="font-size: 10px; color: #888; margin-bottom: 8px;">
                        Status: <span id="sync-status-text">Connecting...</span>
                    </div>
                    <div style="font-size: 10px; margin-bottom: 8px; color: #888;">
                        <div style="margin-bottom: 5px;">
                            <span style="color: #666;">Client ID:</span>
                            <span id="sync-client-id" style="color: #00d4ff; font-family: monospace;">-</span>
                        </div>
                        <div style="margin-bottom: 5px;">
                            <span style="color: #666;">Last Sync:</span>
                            <span id="sync-last-time" style="color: #aaa;">Never</span>
                        </div>
                        <div>
                            <span style="color: #666;">Auto-sync:</span>
                            <span id="sync-interval-display" style="color: #00ff88;">Every 90s</span>
                        </div>
                    </div>
                    <div style="display: flex; gap: 5px;">
                        <button onclick="syncNow()" class="warning" style="flex: 1; font-size: 10px;" id="sync-now-btn">Sync Now</button>
                        <button onclick="loadSyncStats()" style="flex: 1; font-size: 10px;">Stats</button>
                    </div>
                    <div id="sync-results" style="font-size: 9px; color: #888; margin-top: 8px; max-height: 100px; overflow-y: auto;"></div>
                    <!-- Hidden inputs for compatibility with existing JS -->
                    <input type="hidden" id="sync-server-url">
                    <input type="hidden" id="sync-api-key">
                    <input type="hidden" id="sync-client-name">
                    <input type="hidden" id="sync-enabled" value="true">
                    <input type="hidden" id="sync-interval" value="90">
                </div>
            </div>
        </div>
    </div>

    <div class="tooltip" id="tooltip"></div>

    <script>
        let currentPuzzle = null;
        let currentGridStart = null;
        let currentGridEnd = null;
        let zoomHistory = [];
        let runningCells = new Set();  // For exact matches
        let runningProcesses = [];  // Array of {start: BigInt, end: BigInt} for overlap detection
        let refreshInterval = null;
        let processRefreshInterval = null;
        let previousProcessStatuses = {};  // Track process states to detect completions

        async function init() {
            const response = await fetch('/api/puzzles');
            const puzzles = await response.json();

            const select = document.getElementById('puzzle-select');
            puzzles.sort((a, b) => a - b).forEach(p => {
                const option = document.createElement('option');
                option.value = p;
                option.textContent = `Puzzle #${p}`;
                select.appendChild(option);
            });

            select.addEventListener('change', () => {
                currentPuzzle = parseInt(select.value);
                resetView();
            });

            if (puzzles.includes(71)) {
                currentPuzzle = 71;
            } else if (puzzles.length > 0) {
                currentPuzzle = puzzles[0];
            }

            if (currentPuzzle) {
                select.value = currentPuzzle;
                resetView();
            }

            // Setup refresh rate
            document.getElementById('refresh-rate').addEventListener('change', setupRefreshInterval);
            setupRefreshInterval();

            // Process list updates more frequently
            processRefreshInterval = setInterval(updateProcessList, 2000);

            // Check for found keys every 3 seconds
            setInterval(checkForFoundKeys, 3000);

            // Load pool status on startup
            loadPoolStatus();

            // Load new features
            refreshGPUs();
            loadSessionStats();
            setInterval(loadSessionStats, 5000);  // Update session stats every 5 seconds
            loadHeatmap();

            // Check scheduler every minute
            setInterval(checkScheduler, 60000);

            // Load sync configuration
            loadSyncConfig();
        }

        // ============= HELPER FUNCTIONS =============

        // Check if a cell (by hex range) is within any running process
        function isCellWithinRunningProcess(cellStartHex, cellEndHex) {
            if (!runningProcesses || runningProcesses.length === 0) return false;

            try {
                const cellStart = BigInt('0x' + cellStartHex);
                const cellEnd = BigInt('0x' + cellEndHex);

                for (const proc of runningProcesses) {
                    // Check if cell overlaps with process range
                    // Overlap exists if: cellStart <= procEnd AND cellEnd >= procStart
                    if (cellStart <= proc.end && cellEnd >= proc.start) {
                        return true;
                    }
                }
            } catch (e) {
                console.error('Error checking cell overlap:', e);
            }
            return false;
        }

        // Get running process info for a cell (returns process if cell is within it)
        function getRunningProcessForCell(cellStartHex, cellEndHex) {
            if (!runningProcesses || runningProcesses.length === 0) return null;

            try {
                const cellStart = BigInt('0x' + cellStartHex);
                const cellEnd = BigInt('0x' + cellEndHex);

                for (const proc of runningProcesses) {
                    if (cellStart <= proc.end && cellEnd >= proc.start) {
                        return proc;
                    }
                }
            } catch (e) {
                console.error('Error getting process for cell:', e);
            }
            return null;
        }

        // ============= NEW FEATURE FUNCTIONS =============

        // GPU Management
        async function refreshGPUs() {
            try {
                const response = await fetch('/api/gpus');
                const data = await response.json();

                const listEl = document.getElementById('gpu-list');
                if (!data.gpus || data.gpus.length === 0) {
                    listEl.innerHTML = '<div style="color: #888;">No GPUs detected</div>';
                    return;
                }

                listEl.innerHTML = data.gpus.map(gpu => `
                    <div style="background: ${gpu.in_use ? 'rgba(0,255,136,0.1)' : 'rgba(255,255,255,0.05)'}; padding: 6px; margin-bottom: 4px; border-radius: 3px; border-left: 3px solid ${gpu.in_use ? '#00ff88' : '#444'};">
                        <div style="font-weight: bold;">GPU ${gpu.id}: ${gpu.name}</div>
                        <div style="color: #888; font-size: 9px;">${gpu.memory_mb} MB | ${gpu.in_use ? '<span style="color:#00ff88">Mining</span>' : '<span style="color:#666">Idle</span>'}</div>
                    </div>
                `).join('');
            } catch (e) {
                document.getElementById('gpu-list').innerHTML = '<div style="color: #ff4444;">Error loading GPUs</div>';
            }
        }

        // Session Statistics
        async function loadSessionStats() {
            try {
                const response = await fetch('/api/session-stats');
                const stats = await response.json();

                document.getElementById('stat-time').textContent = `${stats.elapsed_hours}h`;
                document.getElementById('stat-keys').textContent = formatNumber(stats.total_keys_checked);
                document.getElementById('stat-rate').textContent = formatNumber(stats.keys_per_hour) + '/h';
                document.getElementById('stat-speed').textContent = stats.current_speed_formatted;
                document.getElementById('stat-blocks').textContent = stats.blocks_completed;
                document.getElementById('stat-gpus').textContent = stats.active_gpus;
            } catch (e) {
                console.error('Error loading session stats:', e);
            }
        }

        // Pattern Analysis
        let patternAnalysisData = null;

        async function runPatternAnalysis() {
            const container = document.getElementById('pattern-analysis');
            container.innerHTML = '<div style="color: #ffaa00;">Analyzing patterns...</div>';

            try {
                const response = await fetch('/api/pattern-analysis');
                const data = await response.json();
                patternAnalysisData = data;  // Store for later use

                let html = '<div style="margin-bottom: 8px;"><strong>Hex Digit Frequency:</strong></div>';
                html += '<div style="display: flex; flex-wrap: wrap; gap: 2px; margin-bottom: 8px;">';

                // Sort by frequency to identify hot digits
                const sortedDigits = Object.entries(data.hex_digit_frequency || {}).sort((a, b) => b[1] - a[1]);
                window.hotDigits = sortedDigits.slice(0, 4).map(d => d[0]);  // Store top 4 hot digits

                for (const [digit, pct] of Object.entries(data.hex_digit_frequency || {})) {
                    const isHot = window.hotDigits.includes(digit);
                    const color = pct > 7 ? '#00ff88' : pct > 5 ? '#ffaa00' : '#666';
                    html += `<div style="background: ${color}22; padding: 2px 4px; border-radius: 2px; font-size: 8px; ${isHot ? 'border: 1px solid #ff6600;' : ''}"><span style="color:${color}">${digit.toUpperCase()}</span>: ${pct}%</div>`;
                }
                html += '</div>';

                html += '<div style="margin-bottom: 4px;"><strong>Key Positions (Puzzles 60-70):</strong></div>';
                html += '<div style="background: #1a1a2e; padding: 4px; border-radius: 3px; margin-bottom: 8px;">';

                // Calculate average position
                let totalPos = 0;
                let count = 0;
                for (const dist of (data.bit_distribution || []).slice(-5)) {
                    const pos = dist.position_pct;
                    totalPos += pos;
                    count++;
                    const barWidth = pos;
                    html += `<div style="margin-bottom: 2px;">
                        <span style="font-size: 8px; color: #888;">P${dist.puzzle}:</span>
                        <div style="background: #333; height: 8px; border-radius: 2px; position: relative;">
                            <div style="background: linear-gradient(90deg, #00ff88, #ffaa00); width: ${barWidth}%; height: 100%; border-radius: 2px;"></div>
                            <span style="position: absolute; right: 2px; top: -1px; font-size: 7px; color: #fff;">${pos.toFixed(1)}%</span>
                        </div>
                    </div>`;
                }
                const avgPos = count > 0 ? totalPos / count : 50;
                window.avgKeyPosition = avgPos;
                html += `<div style="margin-top: 4px; font-size: 9px; color: #ffaa00;">Average position: ${avgPos.toFixed(1)}%</div>`;
                html += '</div>';

                html += '<div style="margin-bottom: 4px;"><strong>Recommendations:</strong></div>';
                html += '<div style="font-size: 9px; color: #00d4ff;">';
                for (const rec of data.recommendations || []) {
                    html += `<div style="margin-bottom: 4px;">• ${rec}</div>`;
                }
                html += '</div>';

                container.innerHTML = html;

                // Show the recommendations panel
                document.getElementById('pattern-recommendations').style.display = 'block';

            } catch (e) {
                container.innerHTML = `<div style="color: #ff4444;">Error: ${e.message}</div>`;
            }
        }

        async function scanRecommendedRange(rangeType) {
            // Get puzzle range
            const puzzleNum = currentPuzzle || 71;
            const rangeStart = BigInt(2) ** BigInt(puzzleNum - 1);
            const rangeEnd = BigInt(2) ** BigInt(puzzleNum) - BigInt(1);
            const rangeSize = rangeEnd - rangeStart;

            let startPct, endPct, description;

            switch (rangeType) {
                case 'lower':
                    startPct = 0;
                    endPct = 40;
                    description = 'Lower 40% - Based on historical key positions';
                    break;
                case 'middle':
                    startPct = 40;
                    endPct = 60;
                    description = 'Middle 20% - Balanced approach';
                    break;
                case 'upper':
                    startPct = 60;
                    endPct = 100;
                    description = 'Upper 40% - Alternative coverage';
                    break;
                case 'hot_digits':
                    // Scan ranges that start with hot digits
                    await scanHotDigitRanges();
                    return;
                default:
                    alert('Unknown range type');
                    return;
            }

            // Calculate actual range values
            const scanStart = rangeStart + (rangeSize * BigInt(startPct)) / BigInt(100);
            const scanEnd = rangeStart + (rangeSize * BigInt(endPct)) / BigInt(100);

            const startHex = scanStart.toString(16).toUpperCase();
            const endHex = scanEnd.toString(16).toUpperCase();

            if (!confirm(`Scan ${description}?\n\nRange: 0x${startHex.substring(0, 12)}... to 0x${endHex.substring(0, 12)}...\n\nThis will zoom into this range and you can select specific blocks to mine.`)) {
                return;
            }

            // Zoom into this range
            zoomInto(startHex, endHex);
        }

        async function scanHotDigitRanges() {
            if (!window.hotDigits || window.hotDigits.length === 0) {
                alert('Please run Pattern Analysis first to identify hot digits');
                return;
            }

            const hotDigits = window.hotDigits;
            const description = `Ranges starting with hot digits: ${hotDigits.map(d => d.toUpperCase()).join(', ')}`;

            if (!confirm(`${description}\n\nThis will scan blocks in the current view that start with these frequent hex digits.`)) {
                return;
            }

            // Get all visible cells and filter by hot digits
            const cells = document.querySelectorAll('.cell');
            const hotCells = [];

            cells.forEach(cell => {
                const startHex = cell.dataset.startHex?.toLowerCase();
                const poolPct = parseFloat(cell.dataset.poolPct) || 0;
                const myPct = parseFloat(cell.dataset.myPct) || 0;
                const skippedPct = parseFloat(cell.dataset.skippedPct) || 0;

                // Check if starts with a hot digit (after the leading digits based on puzzle)
                if (startHex && poolPct < 50 && myPct < 50 && skippedPct < 50) {
                    // Check various positions for hot digits
                    for (const digit of hotDigits) {
                        if (startHex.includes(digit.toLowerCase())) {
                            hotCells.push({
                                start: cell.dataset.startHex,
                                end: cell.dataset.endHex,
                                hotCount: (startHex.match(new RegExp(digit, 'gi')) || []).length
                            });
                            break;
                        }
                    }
                }
            });

            if (hotCells.length === 0) {
                alert('No unscanned blocks with hot digits found in current view. Try zooming out or selecting a different area.');
                return;
            }

            // Sort by hot digit frequency in the range
            hotCells.sort((a, b) => b.hotCount - a.hotCount);

            // Take top 50
            const selected = hotCells.slice(0, 50);

            if (!confirm(`Found ${selected.length} blocks with hot digit patterns. Start mining?`)) {
                return;
            }

            window.miningQueue = selected;
            window.miningQueueIndex = 0;
            startNextQueuedBlock();
        }

        // Heat Map
        async function loadHeatmap() {
            const container = document.getElementById('heatmap');
            container.innerHTML = '';

            try {
                const response = await fetch(`/api/heatmap/${currentPuzzle}?resolution=50`);
                const data = await response.json();

                if (data.error) {
                    container.innerHTML = `<div style="padding: 10px; color: #ff4444;">${data.error}</div>`;
                    return;
                }

                for (const cell of data.cells) {
                    const heat = cell.heat;  // 0-100, higher = more uncovered
                    let color;
                    if (heat > 80) color = '#ff0000';  // Hot - uncovered
                    else if (heat > 60) color = '#ff6600';
                    else if (heat > 40) color = '#ffaa00';
                    else if (heat > 20) color = '#88ff00';
                    else color = '#00ff88';  // Cool - covered

                    const div = document.createElement('div');
                    div.style.cssText = `flex: 1; background: ${color}; cursor: pointer; min-width: 2px;`;
                    div.title = `Range: 0x${cell.start.substring(0, 8)}...\\nCoverage: ${(100 - heat).toFixed(1)}%\\nPriority: ${cell.priority_score}`;
                    div.onclick = () => zoomInto(cell.start, cell.end);
                    container.appendChild(div);
                }
            } catch (e) {
                container.innerHTML = `<div style="padding: 10px; color: #ff4444;">Error loading heatmap</div>`;
            }
        }

        // Checkpoints
        async function loadCheckpoints() {
            const container = document.getElementById('checkpoint-list');
            container.innerHTML = '<div style="color: #ffaa00;">Loading...</div>';

            try {
                const response = await fetch(`/api/checkpoints/${currentPuzzle}`);
                const data = await response.json();

                if (!data || data.length === 0) {
                    container.innerHTML = '<div style="color: #888;">No paused scans</div>';
                    return;
                }

                container.innerHTML = data.slice(0, 10).map(cp => `
                    <div style="background: rgba(255,170,0,0.1); padding: 6px; margin-bottom: 4px; border-radius: 3px; border-left: 2px solid #ffaa00;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
                            <span style="font-family: monospace; font-size: 8px; color: #00d4ff;">0x${cp.resume_position?.substring(0, 10)}...</span>
                            <span style="color: #ffaa00; font-size: 9px; font-weight: bold;">${cp.completion_pct?.toFixed(1)}%</span>
                        </div>
                        <div style="font-size: 7px; color: #888; margin-bottom: 3px;">
                            ${formatNumber(cp.keys_checked)} keys | ${cp.saved_at?.substring(5, 16) || 'N/A'}
                        </div>
                        <div style="display: flex; gap: 3px;">
                            <button onclick="resumeFromCheckpoint('${cp.resume_position}', '${cp.block_end}')" class="success" style="flex: 1; font-size: 8px; padding: 2px 4px;">▶ Resume</button>
                            <button onclick="deleteCheckpoint('${cp.resume_position}')" class="danger" style="font-size: 8px; padding: 2px 4px;">🗑</button>
                        </div>
                    </div>
                `).join('');
            } catch (e) {
                container.innerHTML = `<div style="color: #ff4444;">Error: ${e.message}</div>`;
            }
        }

        async function resumeFromCheckpoint(resumePos, endPos) {
            await startCellMining(resumePos, endPos);
        }

        async function deleteCheckpoint(resumePos) {
            if (!confirm('Delete this checkpoint? This cannot be undone.')) return;

            try {
                const response = await fetch('/api/checkpoint/delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ puzzle: currentPuzzle, resume_position: resumePos })
                });
                const result = await response.json();

                if (result.status === 'ok') {
                    showToast('Checkpoint deleted', 'success');
                    loadCheckpoints();
                } else {
                    showToast(result.message || 'Failed to delete', 'error');
                }
            } catch (e) {
                showToast('Error: ' + e.message, 'error');
            }
        }

        async function deletePartialScan(blockStart) {
            if (!confirm('Delete this partial scan and all associated checkpoints? This cannot be undone.')) return;

            try {
                const response = await fetch('/api/partial-scan/delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ puzzle: currentPuzzle, block_start: blockStart })
                });
                const result = await response.json();

                if (result.status === 'ok') {
                    showToast('Partial scan deleted', 'success');
                    loadCheckpoints();
                    loadGrid();
                } else {
                    showToast(result.message || 'Failed to delete', 'error');
                }
            } catch (e) {
                showToast('Error: ' + e.message, 'error');
            }
        }

        // Scheduler
        let schedulerInterval = null;

        function toggleScheduler() {
            const enabled = document.getElementById('scheduler-enabled').checked;
            const statusEl = document.getElementById('scheduler-status');

            if (enabled) {
                statusEl.textContent = 'Scheduler enabled - will check at specified times';
                statusEl.style.color = '#00ff88';
            } else {
                statusEl.textContent = 'Scheduler disabled';
                statusEl.style.color = '#888';
            }
        }

        function saveSchedule() {
            const startTime = document.getElementById('schedule-start').value;
            const endTime = document.getElementById('schedule-end').value;
            const blocks = document.getElementById('schedule-blocks').value;

            localStorage.setItem('schedule', JSON.stringify({ startTime, endTime, blocks }));

            document.getElementById('scheduler-status').textContent = `Schedule saved: ${startTime} - ${endTime}, ${blocks} blocks`;
            document.getElementById('scheduler-status').style.color = '#00ff88';
        }

        function checkScheduler() {
            if (!document.getElementById('scheduler-enabled').checked) return;

            const schedule = JSON.parse(localStorage.getItem('schedule') || '{}');
            if (!schedule.startTime || !schedule.endTime) return;

            const now = new Date();
            const currentTime = now.getHours() * 60 + now.getMinutes();
            const [startH, startM] = schedule.startTime.split(':').map(Number);
            const [endH, endM] = schedule.endTime.split(':').map(Number);
            const startMinutes = startH * 60 + startM;
            const endMinutes = endH * 60 + endM;

            let inSchedule = false;
            if (endMinutes > startMinutes) {
                // Same day schedule
                inSchedule = currentTime >= startMinutes && currentTime < endMinutes;
            } else {
                // Overnight schedule
                inSchedule = currentTime >= startMinutes || currentTime < endMinutes;
            }

            const statusEl = document.getElementById('scheduler-status');
            if (inSchedule) {
                // Check if we need to start mining
                const activeCount = document.querySelectorAll('.process-item .status.running').length;
                if (activeCount === 0) {
                    statusEl.textContent = 'In schedule window - starting mining...';
                    statusEl.style.color = '#00ff88';
                    // Start random blocks
                    scanRandomBlocks();
                } else {
                    statusEl.textContent = `In schedule - ${activeCount} processes running`;
                    statusEl.style.color = '#00ff88';
                }
            } else {
                statusEl.textContent = `Outside schedule (${schedule.startTime}-${schedule.endTime})`;
                statusEl.style.color = '#888';
            }
        }

        // KEY FOUND ALERT FUNCTIONS
        async function checkForFoundKeys() {
            try {
                const response = await fetch('/api/found');
                const data = await response.json();

                if (data.has_found && data.found.length > 0) {
                    // Show the found keys panel
                    const panel = document.getElementById('found-keys-panel');
                    panel.style.display = 'block';

                    // Update the list
                    const list = document.getElementById('found-keys-list');
                    list.innerHTML = data.found.map(k => `
                        <div style="background: rgba(0,255,136,0.1); padding: 8px; margin-bottom: 8px; border-radius: 5px; font-size: 9px; border: 1px solid #00ff88;">
                            <div style="color: #00ff88; font-weight: bold; font-size: 11px;">Puzzle #${k.puzzle}</div>
                            ${k.pub_address ? `<div style="color: #ffaa00; margin-top: 4px;"><strong>Address:</strong> ${k.pub_address}</div>` : ''}
                            ${k.priv_hex ? `<div style="color: #ff6600; word-break: break-all; margin-top: 4px;"><strong>Private Key (HEX):</strong><br><code style="background: #000; padding: 2px 4px; border-radius: 2px;">${k.priv_hex}</code></div>` : ''}
                            ${k.priv_wif ? `<div style="color: #ff6600; word-break: break-all; margin-top: 4px;"><strong>Private Key (WIF):</strong><br><code style="background: #000; padding: 2px 4px; border-radius: 2px;">${k.priv_wif}</code></div>` : ''}
                            <div style="color: #888; margin-top: 4px; font-size: 8px;">${k.timestamp}</div>
                        </div>
                    `).join('');

                    // Show big alert for any new keys (check if we haven't shown it yet)
                    const lastShown = localStorage.getItem('lastKeyFoundShown') || '0';
                    const latestKey = data.found[data.found.length - 1];
                    if (latestKey && latestKey.timestamp !== lastShown) {
                        showKeyFoundAlert(latestKey);
                        localStorage.setItem('lastKeyFoundShown', latestKey.timestamp);
                    }
                }
            } catch (e) {
                console.error('Error checking for found keys:', e);
            }
        }

        // Global alarm state
        let alarmInterval = null;
        let alarmAudioContext = null;
        let alarmOscillator = null;
        let alarmGainNode = null;

        function startContinuousAlarm() {
            // Stop any existing alarm first
            stopAlarm();

            try {
                alarmAudioContext = new (window.AudioContext || window.webkitAudioContext)();

                // Create oscillator for continuous alarm
                function playAlarmCycle() {
                    if (!alarmAudioContext) return;

                    const osc = alarmAudioContext.createOscillator();
                    const gain = alarmAudioContext.createGain();
                    osc.connect(gain);
                    gain.connect(alarmAudioContext.destination);

                    // Alarm pattern: high-low siren effect
                    osc.type = 'square';
                    gain.gain.setValueAtTime(0.25, alarmAudioContext.currentTime);

                    // Siren sweep
                    osc.frequency.setValueAtTime(800, alarmAudioContext.currentTime);
                    osc.frequency.linearRampToValueAtTime(1200, alarmAudioContext.currentTime + 0.3);
                    osc.frequency.linearRampToValueAtTime(800, alarmAudioContext.currentTime + 0.6);

                    osc.start(alarmAudioContext.currentTime);
                    osc.stop(alarmAudioContext.currentTime + 0.6);
                }

                // Play immediately and repeat
                playAlarmCycle();
                alarmInterval = setInterval(playAlarmCycle, 800);

            } catch (e) {
                console.log('Could not start alarm:', e);
            }
        }

        function stopAlarm() {
            if (alarmInterval) {
                clearInterval(alarmInterval);
                alarmInterval = null;
            }
            if (alarmAudioContext) {
                try {
                    alarmAudioContext.close();
                } catch (e) {}
                alarmAudioContext = null;
            }
        }

        function testAlarm() {
            // Play alarm for 3 seconds then stop
            startContinuousAlarm();
            setTimeout(() => {
                stopAlarm();
            }, 3000);
        }

        function showKeyFoundAlert(keyData) {
            const alert = document.getElementById('key-found-alert');
            const details = document.getElementById('key-found-details');

            details.innerHTML = `
                <div style="font-size: 16px;">Puzzle: #${keyData.puzzle}</div>
                ${keyData.pub_address ? `<div style="color: #ffaa00; margin-top: 10px;"><strong>Address:</strong><br>${keyData.pub_address}</div>` : ''}
                ${keyData.priv_hex ? `<div style="color: #ff6600; margin-top: 15px;"><strong>Private Key (HEX):</strong><br><code style="font-size: 12px; word-break: break-all;">${keyData.priv_hex}</code></div>` : ''}
                ${keyData.priv_wif ? `<div style="color: #ff6600; margin-top: 10px;"><strong>Private Key (WIF):</strong><br><code style="font-size: 12px; word-break: break-all;">${keyData.priv_wif}</code></div>` : ''}
                <div style="color: #888; margin-top: 15px; font-size: 11px;">Range: ${keyData.range_start?.substring(0, 12)}... - ${keyData.range_end?.substring(0, 12)}...</div>
            `;

            alert.classList.add('show');

            // Start continuous alarm - will play until dismissed
            startContinuousAlarm();

            // Try browser notification
            if ('Notification' in window && Notification.permission === 'granted') {
                new Notification('KEY FOUND!', {
                    body: `Puzzle #${keyData.puzzle} - Check Found.txt!`,
                    icon: 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">🎉</text></svg>'
                });
            } else if ('Notification' in window && Notification.permission !== 'denied') {
                Notification.requestPermission();
            }
        }

        function dismissKeyFoundAlert() {
            // Stop the alarm when dismissed
            stopAlarm();
            document.getElementById('key-found-alert').classList.remove('show');
        }

        // POOL SCRAPER FUNCTIONS
        async function scrapePool() {
            const btn = document.getElementById('scrape-btn');
            const statusEl = document.getElementById('pool-status');
            const resultsEl = document.getElementById('pool-results');

            btn.disabled = true;
            btn.textContent = 'Scraping...';
            statusEl.textContent = 'Fetching data from btcpuzzle.info...';

            try {
                const response = await fetch('/api/pool/scrape', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ puzzle: currentPuzzle })
                });

                const result = await response.json();

                if (result.status === 'success') {
                    statusEl.textContent = `Last scrape: ${new Date().toLocaleString()}`;
                    resultsEl.innerHTML = `
                        <div style="color: #00ff88;">Found ${result.ranges_found} ranges, ${result.challenges_found} challenges</div>
                        <div>Decoded ${result.blocks_decoded} blocks</div>
                        ${result.saved ? `<div>Added ${result.saved.added} new, ${result.saved.existing} existing</div>` : ''}
                    `;

                    // Refresh the grid to show new pool blocks
                    loadStats();
                    loadGrid();
                } else {
                    statusEl.textContent = 'Scrape failed';
                    resultsEl.innerHTML = `<div style="color: #ff4444;">Error: ${result.error || 'Unknown error'}</div>`;
                }
            } catch (e) {
                statusEl.textContent = 'Scrape failed';
                resultsEl.innerHTML = `<div style="color: #ff4444;">Error: ${e.message}</div>`;
            }

            btn.disabled = false;
            btn.textContent = 'Scrape Pool Data';
        }

        // Auto-scrape variables
        let autoScrapeEnabled = false;
        let autoScrapeInterval = null;

        function toggleAutoScrape() {
            autoScrapeEnabled = !autoScrapeEnabled;
            const btn = document.getElementById('auto-scrape-btn');
            const statusEl = document.getElementById('auto-scrape-status');

            if (autoScrapeEnabled) {
                btn.textContent = 'Auto: ON';
                btn.style.background = '#00ff88';
                btn.style.color = '#000';
                statusEl.textContent = 'Auto-scraping every 5 minutes...';
                statusEl.style.color = '#00ff88';

                // Start interval (5 minutes = 300000ms)
                autoScrapeInterval = setInterval(() => {
                    console.log('Auto-scraping pool data...');
                    scrapePool();
                }, 300000);

                // Also scrape immediately
                scrapePool();
            } else {
                btn.textContent = 'Auto: OFF';
                btn.style.background = '';
                btn.style.color = '';
                statusEl.textContent = 'Auto-scrape disabled';
                statusEl.style.color = '#666';

                if (autoScrapeInterval) {
                    clearInterval(autoScrapeInterval);
                    autoScrapeInterval = null;
                }
            }
        }

        async function loadPoolStatus() {
            try {
                const response = await fetch(`/api/pool/status/${currentPuzzle}`);
                const data = await response.json();

                const statusEl = document.getElementById('pool-status');
                if (data.timestamp) {
                    statusEl.textContent = `Last scrape: ${data.timestamp}`;
                    document.getElementById('pool-results').innerHTML = `
                        <div>Ranges: ${data.ranges}, Challenges: ${data.challenges}</div>
                        <div>Total blocks: ${data.total_blocks}</div>
                    `;
                }
            } catch (e) {
                console.error('Error loading pool status:', e);
            }
        }

        async function analyzePoolPatterns() {
            const analysisEl = document.getElementById('pool-analysis');
            const predictionsEl = document.getElementById('pool-predictions');

            analysisEl.innerHTML = '<div style="color: #ffaa00;">Analyzing pool patterns...</div>';

            try {
                const response = await fetch(`/api/pool/analyze/${currentPuzzle}`);
                const data = await response.json();

                if (data.error) {
                    analysisEl.innerHTML = `<div style="color: #ff4444;">Error: ${data.error}</div>`;
                    return;
                }

                analysisEl.innerHTML = `
                    <div><strong>Pool Statistics:</strong></div>
                    <div>Total blocks scanned: ${data.total_pool_blocks?.toLocaleString() || 0}</div>
                    <div>Coverage: ${data.coverage_pct?.toFixed(4) || 0}%</div>
                    <div>Avg block size: ${formatNumber(data.avg_block_size || 0)} keys</div>
                    <div style="margin-top: 5px;"><strong>Range:</strong></div>
                    <div style="font-family: monospace; font-size: 9px;">
                        Low: 0x${data.lowest_scanned || '?'}<br>
                        High: 0x${data.highest_scanned || '?'}
                    </div>
                `;

                // Show predictions
                if (data.predictions && data.predictions.length > 0) {
                    predictionsEl.style.display = 'block';
                    predictionsEl.innerHTML = `
                        <div style="color: #00d4ff; font-weight: bold; margin-bottom: 5px;">Predicted Next Blocks:</div>
                        ${data.predictions.slice(0, 5).map(p => `
                            <div style="background: rgba(0,212,255,0.1); padding: 4px; margin-bottom: 3px; border-radius: 3px; font-size: 9px;">
                                <span style="color: ${p.confidence === 'high' ? '#00ff88' : '#ffaa00'};">[${p.confidence}]</span>
                                <span style="font-family: monospace;">0x${p.start?.substring(0, 12)}...</span>
                                <button class="small success" onclick="startCellMining('${p.start}', '${p.end}')" style="float: right; padding: 2px 6px;">Mine</button>
                            </div>
                        `).join('')}
                        ${data.largest_gaps && data.largest_gaps.length > 0 ? `
                            <div style="color: #ff9900; font-weight: bold; margin: 8px 0 5px 0;">Gaps (Pre-Pool):</div>
                            ${data.largest_gaps.slice(0, 3).map(g => `
                                <div style="background: rgba(255,153,0,0.1); padding: 4px; margin-bottom: 3px; border-radius: 3px; font-size: 9px;">
                                    <span style="font-family: monospace;">0x${g.start?.substring(0, 10)}...</span>
                                    <button class="small warning" onclick="startCellMining('${g.start}', '${g.end}')" style="float: right; padding: 2px 6px;">Mine</button>
                                </div>
                            `).join('')}
                        ` : ''}
                    `;
                }
            } catch (e) {
                analysisEl.innerHTML = `<div style="color: #ff4444;">Error: ${e.message}</div>`;
            }
        }

        // Check if a hex address has filter patterns
        function hasFilterPatterns(hexStr) {
            if (!hexStr) return false;
            const settings = getFilterSettings();

            // Check for repeated characters (3+)
            if (settings.iter3) {
                let count = 1;
                let prev = '';
                for (const char of hexStr) {
                    if (char === prev && char !== '0') {
                        count++;
                        if (count >= 3) return true;
                    } else {
                        count = 1;
                        prev = char;
                    }
                }
            }

            // Check for repeated characters (4+)
            if (settings.iter4) {
                let count = 1;
                let prev = '';
                for (const char of hexStr) {
                    if (char === prev && char !== '0') {
                        count++;
                        if (count >= 4) return true;
                    } else {
                        count = 1;
                        prev = char;
                    }
                }
            }

            // Check for all alpha or all numeric (last 8 chars)
            if (settings.alphanum) {
                const checkStr = hexStr.slice(-8);
                const hasAlpha = /[A-F]/i.test(checkStr);
                const hasNumeric = /[0-9]/.test(checkStr);
                if (hasAlpha !== hasNumeric) return true; // XOR
            }

            return false;
        }

        // SMART BATCH FILTER FUNCTIONS
        function getFilterSettings() {
            return {
                iter3: document.getElementById('filter-iter3').checked,
                iter4: document.getElementById('filter-iter4').checked,
                alphanum: document.getElementById('filter-alphanum').checked,
                subrange_size: parseInt(document.getElementById('filter-subrange').value) || 10000000
            };
        }

        async function previewFilter() {
            if (!currentGridStart || !currentGridEnd) {
                alert('Please select a range first by zooming into the grid');
                return;
            }

            const resultsEl = document.getElementById('filter-results');
            resultsEl.style.display = 'block';
            resultsEl.innerHTML = '<div style="color: #ffaa00;">Analyzing range...</div>';

            try {
                const settings = getFilterSettings();
                const response = await fetch('/api/filter/preview', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        start: currentGridStart,
                        end: currentGridEnd,
                        ...settings
                    })
                });

                const result = await response.json();

                if (result.status === 'success') {
                    resultsEl.innerHTML = `
                        <div style="color: #00ff88; font-weight: bold;">Filter Preview:</div>
                        <div>Total sub-ranges: ${result.total_subranges.toLocaleString()}</div>
                        <div style="color: #00ff88;">Clean: ${result.clean_subranges.toLocaleString()}</div>
                        <div style="color: #ff4444;">Skipped: ${result.skipped_subranges.toLocaleString()}</div>
                        <div style="color: #ffaa00; font-weight: bold;">
                            Reduction: ${result.reduction_pct.toFixed(1)}%
                        </div>
                        <div style="font-size: 9px; color: #888;">
                            Keys to skip: ${formatNumber(result.skipped_keys)}
                        </div>
                    `;
                } else {
                    resultsEl.innerHTML = `<div style="color: #ff4444;">Error: ${result.error}</div>`;
                }
            } catch (e) {
                resultsEl.innerHTML = `<div style="color: #ff4444;">Error: ${e.message}</div>`;
            }
        }

        async function applyFilter() {
            if (!currentGridStart || !currentGridEnd) {
                alert('Please select a range first by zooming into the grid');
                return;
            }

            if (!confirm('Apply filter and save skipped ranges to database?')) {
                return;
            }

            const resultsEl = document.getElementById('filter-results');
            resultsEl.style.display = 'block';
            resultsEl.innerHTML = '<div style="color: #ffaa00;">Applying filter...</div>';

            try {
                const settings = getFilterSettings();
                const response = await fetch('/api/filter/apply', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        puzzle: currentPuzzle,
                        start: currentGridStart,
                        end: currentGridEnd,
                        ...settings
                    })
                });

                const result = await response.json();

                if (result.status === 'success') {
                    resultsEl.innerHTML = `
                        <div style="color: #00ff88; font-weight: bold;">Filter Applied!</div>
                        <div>Saved ${result.skipped_saved} skipped ranges</div>
                        <div>Clean ranges: ${result.clean_ranges}</div>
                        <div style="color: #ffaa00;">Reduction: ${result.reduction_pct.toFixed(1)}%</div>
                    `;

                    // Refresh the grid and stats
                    loadStats();
                    loadGrid();
                    loadQueueList();
                } else {
                    resultsEl.innerHTML = `<div style="color: #ff4444;">Error: ${result.error}</div>`;
                }
            } catch (e) {
                resultsEl.innerHTML = `<div style="color: #ff4444;">Error: ${e.message}</div>`;
            }
        }

        async function startFilteredMining(startHex, endHex) {
            // Start mining with smart filtering enabled
            const settings = getFilterSettings();

            try {
                const response = await fetch('/api/mine/start-filtered', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        puzzle: currentPuzzle,
                        start: startHex,
                        end: endHex,
                        gpu: document.getElementById('use-gpu').checked,
                        ...settings
                    })
                });

                const result = await response.json();

                if (result.filter_info) {
                    console.log(`Started filtered mining: ${result.filter_info.total_clean_ranges} clean ranges, ${result.filter_info.reduction_pct.toFixed(1)}% reduction`);
                }

                updateProcessList();
                loadGrid();
            } catch (e) {
                console.error('Error starting filtered mining:', e);
            }
        }

        function setupRefreshInterval() {
            if (refreshInterval) clearInterval(refreshInterval);

            const rate = parseInt(document.getElementById('refresh-rate').value);
            if (rate > 0) {
                refreshInterval = setInterval(refreshData, rate);
            }
        }

        async function loadStats() {
            if (!currentPuzzle) return;

            const response = await fetch(`/api/stats/${currentPuzzle}`);
            const stats = await response.json();

            document.getElementById('my-blocks').textContent = stats.my_blocks?.toLocaleString() || '-';
            document.getElementById('pool-blocks').textContent = stats.pool_blocks?.toLocaleString() || '-';
            document.getElementById('skipped-blocks').textContent = stats.skipped_blocks?.toLocaleString() || '0';
            document.getElementById('keys-checked').textContent = formatNumber(stats.keys_checked) || '-';
            document.getElementById('target-address').textContent = stats.address || 'Unknown';
            document.getElementById('target-reward').textContent = stats.reward || '';
        }

        async function loadGrid() {
            if (!currentPuzzle) return;

            const gridEl = document.getElementById('grid');
            gridEl.innerHTML = '<div class="loading">Loading grid data</div>';

            let url = `/api/grid/${currentPuzzle}`;
            if (currentGridStart !== null && currentGridEnd !== null) {
                url += `?start=${currentGridStart}&end=${currentGridEnd}`;
            }

            const response = await fetch(url);
            const data = await response.json();

            if (data.error) {
                gridEl.innerHTML = `<div class="loading" style="color: #ff4444;">Error: ${data.error}</div>`;
                return;
            }

            document.getElementById('grid-info').textContent = `0x${data.grid_start} - 0x${data.grid_end}`;

            // Display cell size
            if (data.cells && data.cells.length > 0) {
                document.getElementById('cell-size-display').textContent = formatNumber(data.cells[0].key_count);
            }

            let totalPoolPct = 0, totalMyPct = 0, totalBrowserPct = 0, totalSkippedPct = 0;
            data.cells.forEach(cell => {
                totalPoolPct += cell.pool_pct;
                totalMyPct += cell.my_pct;
                totalBrowserPct += cell.browser_pct || 0;
                totalSkippedPct += cell.skipped_pct;
            });
            totalPoolPct /= data.cells.length;
            totalMyPct /= data.cells.length;
            totalBrowserPct /= data.cells.length;
            totalSkippedPct /= data.cells.length;

            document.getElementById('mine-bar').style.width = `${totalMyPct}%`;
            document.getElementById('pool-bar').style.width = `${totalPoolPct}%`;
            document.getElementById('browser-bar').style.width = `${totalBrowserPct}%`;
            document.getElementById('skipped-bar').style.width = `${totalSkippedPct}%`;
            document.getElementById('coverage-label').textContent = `${(totalPoolPct + totalMyPct + totalBrowserPct).toFixed(4)}% scanned`;

            const showFilters = document.getElementById('show-filters').checked;

            gridEl.innerHTML = '';
            data.cells.forEach((cell, index) => {
                const cellEl = document.createElement('div');
                cellEl.className = 'cell';

                // Determine cell class - use has_my_scans/has_pool_scans/has_browser_scans for better detection
                const hasMyScans = cell.has_my_scans || cell.my_pct > 0 || cell.my_blocks > 0;
                const hasPoolScans = cell.has_pool_scans || cell.pool_pct > 0 || cell.pool_blocks > 0;
                const hasBrowserScans = cell.has_browser_scans || cell.browser_pct > 0 || cell.browser_blocks > 0;

                if (cell.skipped_pct > 0 && showFilters) {
                    cellEl.classList.add(cell.queued_blocks > 0 ? 'queued' : 'skipped');
                } else if ((hasMyScans && hasPoolScans) || (hasMyScans && hasBrowserScans) || (hasPoolScans && hasBrowserScans)) {
                    cellEl.classList.add('both');
                } else if (hasMyScans) {
                    cellEl.classList.add('mine-only');
                } else if (hasPoolScans) {
                    cellEl.classList.add('pool-only');
                } else if (hasBrowserScans) {
                    cellEl.classList.add('browser-only');
                } else if (cell.total_pct > 0) {
                    cellEl.classList.add('partial');
                } else {
                    cellEl.classList.add('empty');
                }

                // Check if cell has filter pattern matches (real-time filter highlighting)
                const showFilterHighlight = document.getElementById('show-filters')?.checked;
                if (showFilterHighlight && (hasFilterPatterns(cell.start_hex) || hasFilterPatterns(cell.end_hex))) {
                    cellEl.classList.add('has-filter-matches');
                }

                const cellKey = `${cell.start_hex}-${cell.end_hex}`;
                // Check if cell is running - server-side detection, exact match, or within running process range
                const isRunning = cell.has_running || runningCells.has(cellKey) || isCellWithinRunningProcess(cell.start_hex, cell.end_hex);
                if (isRunning) {
                    cellEl.classList.add('running');
                }

                // Show percentage and key count
                const pctDisplay = document.createElement('span');
                if (cell.total_pct > 0 || cell.skipped_pct > 0) {
                    const pct = Math.max(cell.total_pct, cell.skipped_pct);
                    pctDisplay.textContent = pct >= 1 ? Math.round(pct) + '%' : '<1%';
                } else {
                    pctDisplay.textContent = index;
                }
                cellEl.appendChild(pctDisplay);

                // Key count display
                const keyCount = document.createElement('span');
                keyCount.className = 'key-count';
                keyCount.textContent = formatNumberShort(cell.key_count);
                cellEl.appendChild(keyCount);

                // Mine button
                const mineBtn = document.createElement('button');
                mineBtn.className = 'mine-btn' + (isRunning ? ' running' : '');
                mineBtn.innerHTML = isRunning ? '■' : '▶';
                mineBtn.title = isRunning ? 'Stop mining' : 'Start mining this cell';
                mineBtn.onclick = (e) => {
                    e.stopPropagation();
                    if (isRunning) {
                        // If within a running process, stop that process
                        const runningProc = getRunningProcessForCell(cell.start_hex, cell.end_hex);
                        if (runningProc) {
                            stopCellMining(runningProc.startHex, runningProc.endHex);
                        } else {
                            stopCellMining(cell.start_hex, cell.end_hex);
                        }
                    } else {
                        startCellMining(cell.start_hex, cell.end_hex);
                    }
                };
                cellEl.appendChild(mineBtn);

                // Skip button (only show if not already skipped)
                if (cell.skipped_pct === 0) {
                    const skipBtn = document.createElement('button');
                    skipBtn.className = 'skip-btn';
                    skipBtn.innerHTML = '✕';
                    skipBtn.title = 'Mark as skipped';
                    skipBtn.onclick = (e) => {
                        e.stopPropagation();
                        markAsSkipped(cell.start_hex, cell.end_hex, 'manual');
                    };
                    cellEl.appendChild(skipBtn);
                }

                // Store cell data
                cellEl.dataset.startHex = cell.start_hex;
                cellEl.dataset.endHex = cell.end_hex;
                cellEl.dataset.keyCount = cell.key_count;
                cellEl.dataset.poolPct = cell.pool_pct;
                cellEl.dataset.myPct = cell.my_pct;
                cellEl.dataset.skippedPct = cell.skipped_pct;
                cellEl.dataset.poolBlocks = cell.pool_blocks;
                cellEl.dataset.myBlocks = cell.my_blocks;
                cellEl.dataset.skippedBlocks = cell.skipped_blocks;
                cellEl.dataset.hasMyScans = hasMyScans ? '1' : '0';
                cellEl.dataset.hasPoolScans = hasPoolScans ? '1' : '0';
                cellEl.dataset.hasBrowserScans = hasBrowserScans ? '1' : '0';
                cellEl.dataset.browserPct = cell.browser_pct || 0;
                cellEl.dataset.browserBlocks = cell.browser_blocks || 0;
                cellEl.dataset.hasRunning = cell.has_running ? '1' : '0';
                cellEl.dataset.runningPct = cell.running_pct || 0;

                cellEl.addEventListener('mouseenter', showTooltip);
                cellEl.addEventListener('mousemove', moveTooltip);
                cellEl.addEventListener('mouseleave', hideTooltip);
                cellEl.addEventListener('click', () => zoomInto(cell.start_hex, cell.end_hex));

                gridEl.appendChild(cellEl);
            });
        }

        async function startCellMining(startHex, endHex) {
            const useGpu = document.getElementById('use-gpu').checked;

            const response = await fetch('/api/mine/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    puzzle: currentPuzzle,
                    start: startHex,
                    end: endHex,
                    gpu: useGpu
                })
            });

            const result = await response.json();
            if (result.process_id) {
                runningCells.add(`${startHex}-${endHex}`);
                loadGrid();
                updateProcessList();
            } else if (result.error) {
                alert('Error: ' + result.error);
            }
        }

        async function stopCellMining(startHex, endHex) {
            const response = await fetch('/api/processes');
            const processes = await response.json();

            for (const [pid, proc] of Object.entries(processes)) {
                if (proc.start === startHex && proc.end === endHex && proc.status === 'running') {
                    await fetch(`/api/mine/stop/${pid}`, {method: 'POST'});
                    runningCells.delete(`${startHex}-${endHex}`);
                    break;
                }
            }

            loadGrid();
            updateProcessList();
        }

        async function stopAllMining() {
            const response = await fetch('/api/processes');
            const processes = await response.json();

            for (const [pid, proc] of Object.entries(processes)) {
                if (proc.status === 'running') {
                    await fetch(`/api/mine/stop/${pid}`, {method: 'POST'});
                }
            }

            runningCells.clear();
            loadGrid();
            updateProcessList();
        }

        async function clearCompletedProcesses() {
            // Clear completed/stopped processes from the list
            const response = await fetch('/api/processes/clear-completed', {method: 'POST'});
            updateProcessList();
        }

        async function scanAllVisibleBlocks() {
            // Get all visible cells and start mining unscanned ones
            const cells = document.querySelectorAll('.cell');
            const unscannedCells = [];

            cells.forEach(cell => {
                const poolPct = parseFloat(cell.dataset.poolPct) || 0;
                const myPct = parseFloat(cell.dataset.myPct) || 0;
                const skippedPct = parseFloat(cell.dataset.skippedPct) || 0;

                // Only add cells that are mostly unscanned
                if (poolPct < 50 && myPct < 50 && skippedPct < 50) {
                    unscannedCells.push({
                        start: cell.dataset.startHex,
                        end: cell.dataset.endHex
                    });
                }
            });

            if (unscannedCells.length === 0) {
                alert('No unscanned blocks found in current view');
                return;
            }

            if (!confirm(`Start mining ${unscannedCells.length} unscanned blocks?\\nThey will be queued and started sequentially.`)) {
                return;
            }

            // Add all to a queue and start the first one
            window.miningQueue = unscannedCells;
            window.miningQueueIndex = 0;
            startNextQueuedBlock();
        }

        async function scanRandomBlocks() {
            const count = parseInt(document.getElementById('random-block-count').value);
            const cells = document.querySelectorAll('.cell');
            const unscannedCells = [];

            cells.forEach(cell => {
                const poolPct = parseFloat(cell.dataset.poolPct) || 0;
                const myPct = parseFloat(cell.dataset.myPct) || 0;
                const skippedPct = parseFloat(cell.dataset.skippedPct) || 0;

                if (poolPct < 50 && myPct < 50 && skippedPct < 50) {
                    unscannedCells.push({
                        start: cell.dataset.startHex,
                        end: cell.dataset.endHex
                    });
                }
            });

            if (unscannedCells.length === 0) {
                alert('No unscanned blocks found in current view');
                return;
            }

            // Shuffle and take requested count
            const shuffled = unscannedCells.sort(() => Math.random() - 0.5);
            const selected = shuffled.slice(0, Math.min(count, shuffled.length));

            if (!confirm(`Start mining ${selected.length} random blocks?`)) {
                return;
            }

            window.miningQueue = selected;
            window.miningQueueIndex = 0;
            startNextQueuedBlock();
        }

        async function scanSmartPriority() {
            // Scan blocks prioritized by pattern analysis
            const cells = document.querySelectorAll('.cell');
            const unscannedCells = [];

            cells.forEach(cell => {
                const poolPct = parseFloat(cell.dataset.poolPct) || 0;
                const myPct = parseFloat(cell.dataset.myPct) || 0;
                const skippedPct = parseFloat(cell.dataset.skippedPct) || 0;

                if (poolPct < 50 && myPct < 50 && skippedPct < 50) {
                    unscannedCells.push({
                        start: cell.dataset.startHex,
                        end: cell.dataset.endHex
                    });
                }
            });

            if (unscannedCells.length === 0) {
                alert('No unscanned blocks found in current view');
                return;
            }

            // Get priority scores for each cell
            const scoredCells = [];
            for (const cell of unscannedCells) {
                try {
                    const response = await fetch(`/api/range-priority/${currentPuzzle}?start=${cell.start}&end=${cell.end}`);
                    const priority = await response.json();
                    scoredCells.push({
                        ...cell,
                        score: priority.score || 50
                    });
                } catch (e) {
                    scoredCells.push({ ...cell, score: 50 });
                }
            }

            // Sort by priority (highest first)
            scoredCells.sort((a, b) => b.score - a.score);

            // Take top 50
            const selected = scoredCells.slice(0, 50);

            if (!confirm(`Start mining ${selected.length} high-priority blocks?\\nThese are prioritized based on solved puzzle patterns.`)) {
                return;
            }

            window.miningQueue = selected;
            window.miningQueueIndex = 0;
            startNextQueuedBlock();
        }

        async function startNextQueuedBlock() {
            if (!window.miningQueue || window.miningQueueIndex >= window.miningQueue.length) {
                console.log('Mining queue complete');
                return;
            }

            const block = window.miningQueue[window.miningQueueIndex];
            window.miningQueueIndex++;

            await startCellMining(block.start, block.end);

            // When current mining completes, start next (handled by process completion detection)
            console.log(`Started block ${window.miningQueueIndex}/${window.miningQueue.length}`);
        }

        async function resumePartialBlocks() {
            // Get partially completed blocks from the database and resume them
            try {
                const response = await fetch(`/api/partial-blocks/${currentPuzzle}`);
                const partials = await response.json();

                if (!partials || partials.length === 0) {
                    alert('No partial blocks found to resume');
                    return;
                }

                if (!confirm(`Found ${partials.length} partial blocks. Resume mining?`)) {
                    return;
                }

                // Start mining each partial block from where it left off
                for (const block of partials) {
                    await startCellMining(block.resume_start, block.end);
                }
            } catch (e) {
                alert('Error getting partial blocks: ' + e.message);
            }
        }

        async function killAllGPUProcesses() {
            if (!confirm('This will forcefully kill ALL KeyHunt GPU processes on the system. Continue?')) {
                return;
            }

            try {
                const response = await fetch('/api/kill-all-gpu', {method: 'POST'});
                const result = await response.json();

                if (result.status === 'success') {
                    alert(`Killed ${result.killed} GPU processes`);
                } else {
                    alert('Error: ' + (result.error || 'Unknown error'));
                }

                runningCells.clear();
                updateProcessList();
                loadGrid();
            } catch (e) {
                alert('Error killing GPU processes: ' + e.message);
            }
        }

        async function quickMine() {
            const start = document.getElementById('quick-start').value.trim().toUpperCase();
            const end = document.getElementById('quick-end').value.trim().toUpperCase();

            if (!start || !end) {
                alert('Please enter both start and end range');
                return;
            }

            const useGpu = document.getElementById('use-gpu').checked;

            const response = await fetch('/api/mine/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ puzzle: currentPuzzle, start, end, gpu: useGpu })
            });

            const result = await response.json();
            if (result.error) {
                alert('Error: ' + result.error);
            } else {
                document.getElementById('quick-start').value = '';
                document.getElementById('quick-end').value = '';
                updateProcessList();
            }
        }

        async function manualSkip() {
            const start = document.getElementById('skip-start').value.trim().toUpperCase();
            const end = document.getElementById('skip-end').value.trim().toUpperCase();
            const reason = document.getElementById('skip-reason').value;

            if (!start || !end) {
                alert('Please enter both start and end range');
                return;
            }

            await markAsSkipped(start, end, reason);
            document.getElementById('skip-start').value = '';
            document.getElementById('skip-end').value = '';
        }

        async function updateProcessList() {
            const response = await fetch('/api/processes');
            const processes = await response.json();

            const listEl = document.getElementById('process-list');
            const entries = Object.entries(processes);

            const activeCount = entries.filter(([_, p]) => p.status === 'running').length;
            document.getElementById('active-processes').textContent = activeCount;

            // Update the prominent scanning status bar
            const scanningStatus = document.getElementById('scanning-status');
            const runningProcs = entries.filter(([_, p]) => p.status === 'running');

            if (runningProcs.length > 0) {
                const [pid, proc] = runningProcs[0];
                scanningStatus.classList.add('active');
                document.getElementById('scan-range').innerHTML = `
                    <strong>Range:</strong> 0x${proc.start} → 0x${proc.end}<br>
                    <strong>Puzzle:</strong> #${proc.puzzle} | <strong>Process ID:</strong> ${pid}
                `;
                document.getElementById('scan-speed').textContent = proc.speed || '0 Mk/s';
                document.getElementById('scan-keys').textContent = formatNumber(proc.keys_checked || 0);
                document.getElementById('scan-progress').textContent = (proc.completion || 0).toFixed(2) + '%';
                document.getElementById('scan-gpu').textContent = proc.gpu_id !== null ? `GPU #${proc.gpu_id}` : 'CPU';
                document.getElementById('scan-progress-bar').style.width = (proc.completion || 0) + '%';
            } else {
                scanningStatus.classList.remove('active');
            }

            // Detect process completions - if a process was running and is now completed/stopped
            let processCompleted = false;
            entries.forEach(([pid, proc]) => {
                const prevStatus = previousProcessStatuses[pid];
                if (prevStatus === 'running' && (proc.status === 'completed' || proc.status === 'stopped')) {
                    processCompleted = true;
                    console.log(`Process ${pid} completed - will refresh grid and queue`);
                }
                previousProcessStatuses[pid] = proc.status;
            });

            // If any process completed, refresh the grid and queue list
            // This ensures completed blocks show as scanned and skipped blocks are cleared
            if (processCompleted) {
                setTimeout(() => {
                    loadStats();
                    loadGrid();
                    loadQueueList();

                    // Check if there are more blocks in the queue to mine
                    if (window.miningQueue && window.miningQueueIndex < window.miningQueue.length) {
                        console.log('Process completed, starting next queued block...');
                        startNextQueuedBlock();
                    }
                }, 500);  // Small delay to ensure DB writes complete
            }

            runningCells.clear();
            runningProcesses = [];  // Clear and rebuild running processes array
            entries.forEach(([_, p]) => {
                if (p.status === 'running') {
                    runningCells.add(`${p.start}-${p.end}`);
                    // Also add to runningProcesses array for overlap detection
                    try {
                        runningProcesses.push({
                            start: BigInt('0x' + p.start),
                            end: BigInt('0x' + p.end),
                            startHex: p.start,
                            endHex: p.end
                        });
                    } catch (e) {
                        console.error('Error parsing process range:', e);
                    }
                }
            });

            if (entries.length === 0) {
                listEl.innerHTML = '<div style="color: #888; text-align: center; padding: 15px;">No active processes</div>';
                previousProcessStatuses = {};  // Clear status tracking when no processes
                return;
            }

            listEl.innerHTML = entries.map(([pid, proc]) => `
                <div class="process-item">
                    <div class="header" style="cursor: pointer;" onclick="loadProcessBlock('${proc.start}', '${proc.end}')" title="Click to view this block in grid">
                        <span>Puzzle #${proc.puzzle}</span>
                        <span class="status ${proc.status}">${proc.status}</span>
                    </div>
                    <div class="range" style="cursor: pointer;" onclick="toggleSubBlockView(${pid})" title="Click to show sub-block progress">0x${proc.start?.substring(0, 12)}...</div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="speed">${proc.speed || 'Starting...'}</span>
                        <span class="keys">${formatNumber(proc.keys_checked || 0)} keys</span>
                    </div>
                    ${proc.status === 'running' ? `
                        <div class="progress-bar">
                            <div class="fill" style="width: ${proc.completion || 0}%"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 4px;">
                            <span style="font-size: 9px; color: #888;">${(proc.completion || 0).toFixed(2)}%</span>
                            <button class="small" onclick="event.stopPropagation(); toggleSubBlockView(${pid})" title="Show sub-blocks">📊</button>
                            <button class="warning small" onclick="event.stopPropagation(); pauseProcess(${pid})" title="Pause and save checkpoint for later resume">⏸ Pause</button>
                            <button class="danger small" onclick="event.stopPropagation(); stopProcess(${pid}, true)" title="Stop without saving checkpoint">⏹ Stop</button>
                        </div>
                        <div id="subblock-view-${pid}" style="display: none; margin-top: 8px;">
                            ${generateSubBlockGrid(proc.completion || 0)}
                        </div>
                    ` : ''}
                </div>
            `).join('');
        }

        async function pauseProcess(pid) {
            // Pause saves checkpoint for later resume
            const response = await fetch(`/api/mine/pause/${pid}`, {method: 'POST'});
            const result = await response.json();
            if (result.status === 'paused') {
                showToast('Process paused - checkpoint saved for resume', 'success');
                loadCheckpoints();  // Refresh checkpoint list
            }
            updateProcessList();
            loadGrid();
        }

        async function stopProcess(pid, skipCheckpoint = false) {
            // Stop with option to skip checkpoint
            const url = skipCheckpoint ? `/api/mine/stop/${pid}?skip_checkpoint=1` : `/api/mine/stop/${pid}`;
            await fetch(url, {method: 'POST'});
            updateProcessList();
            loadGrid();
        }

        function showToast(message, type = 'info') {
            const toast = document.createElement('div');
            toast.style.cssText = `
                position: fixed; bottom: 20px; right: 20px; padding: 12px 20px;
                background: ${type === 'success' ? '#00ff88' : type === 'error' ? '#ff4444' : '#00d4ff'};
                color: #000; border-radius: 8px; font-size: 12px; z-index: 10000;
                animation: slideIn 0.3s ease;
            `;
            toast.textContent = message;
            document.body.appendChild(toast);
            setTimeout(() => toast.remove(), 3000);
        }

        function loadProcessBlock(startHex, endHex) {
            // Zoom into the block being mined
            if (startHex && endHex) {
                zoomInto(startHex, endHex);
                console.log(`Zoomed to block: ${startHex} - ${endHex}`);
            }
        }

        function toggleSubBlockView(pid) {
            const view = document.getElementById(`subblock-view-${pid}`);
            if (view) {
                view.style.display = view.style.display === 'none' ? 'block' : 'none';
            }
        }

        function generateSubBlockGrid(completionPct) {
            // Generate a 10x10 grid showing sub-block completion
            const totalSubBlocks = 100;
            const completedSubBlocks = Math.floor(completionPct);

            let html = '<div style="display: grid; grid-template-columns: repeat(10, 1fr); gap: 2px;">';
            for (let i = 0; i < totalSubBlocks; i++) {
                const isComplete = i < completedSubBlocks;
                const isInProgress = i === completedSubBlocks;
                let bgColor = '#333';
                if (isComplete) bgColor = '#00ff88';
                else if (isInProgress) bgColor = '#ffaa00';

                html += `<div style="
                    width: 100%;
                    aspect-ratio: 1;
                    background: ${bgColor};
                    border-radius: 2px;
                    font-size: 6px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: ${isComplete ? '#000' : '#666'};
                " title="Sub-block ${i + 1}: ${isComplete ? 'Complete' : isInProgress ? 'In Progress' : 'Pending'}">${i + 1}</div>`;
            }
            html += '</div>';
            html += `<div style="text-align: center; font-size: 8px; color: #888; margin-top: 4px;">${completedSubBlocks}/100 sub-blocks complete</div>`;
            return html;
        }

        async function loadQueueList() {
            const response = await fetch(`/api/skipped/${currentPuzzle}`);
            const skipped = await response.json();

            const queued = skipped.filter(s => s.queued);
            const listEl = document.getElementById('queue-list');

            if (queued.length === 0) {
                // Show non-queued skipped blocks that could be queued
                const unqueued = skipped.filter(s => !s.queued).slice(0, 5);
                if (unqueued.length > 0) {
                    listEl.innerHTML = '<div style="color: #888; font-size: 9px; margin-bottom: 5px;">Click to queue:</div>' +
                        unqueued.map(s => `
                            <div class="queue-item" style="cursor: pointer;" onclick="queueForRescan('${s.start}')">
                                <span class="range">0x${s.start?.substring(0, 10)}...</span>
                                <span style="color: #888; font-size: 8px;">${s.reason}</span>
                            </div>
                        `).join('');
                } else {
                    listEl.innerHTML = '<div style="color: #888; text-align: center; padding: 10px; font-size: 10px;">No skipped blocks</div>';
                }
                return;
            }

            listEl.innerHTML = queued.slice(0, 8).map(q => `
                <div class="queue-item">
                    <span class="range">0x${q.start?.substring(0, 10)}...</span>
                    <button class="small success" onclick="mineSkipped('${q.start}', '${q.end}')">Mine</button>
                </div>
            `).join('');
        }

        async function mineSkipped(start, end) {
            await startCellMining(start, end);
        }

        async function mineNextQueued() {
            const response = await fetch(`/api/skipped/${currentPuzzle}`);
            const skipped = await response.json();
            const queued = skipped.filter(s => s.queued);

            if (queued.length > 0) {
                const next = queued[0];
                await startCellMining(next.start, next.end);
            } else {
                alert('No blocks in queue. Click on skipped blocks in the list to queue them.');
            }
        }

        async function markAsSkipped(startHex, endHex, reason = 'manual') {
            await fetch('/api/skip/add', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ puzzle: currentPuzzle, start: startHex, end: endHex, reason })
            });
            loadGrid();
            loadQueueList();
        }

        async function queueForRescan(startHex) {
            await fetch('/api/skip/queue', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ puzzle: currentPuzzle, start: startHex })
            });
            loadGrid();
            loadQueueList();
        }

        function zoomInto(startHex, endHex) {
            zoomHistory.push({ start: currentGridStart, end: currentGridEnd });
            currentGridStart = startHex;
            currentGridEnd = endHex;
            updateBreadcrumb();
            loadGrid();
            document.getElementById('zoom-out-btn').disabled = false;
        }

        function zoomOut() {
            if (zoomHistory.length === 0) return;
            const prev = zoomHistory.pop();
            currentGridStart = prev.start;
            currentGridEnd = prev.end;
            updateBreadcrumb();
            loadGrid();
            if (zoomHistory.length === 0) {
                document.getElementById('zoom-out-btn').disabled = true;
            }
        }

        function resetView() {
            currentGridStart = null;
            currentGridEnd = null;
            zoomHistory = [];
            updateBreadcrumb();
            loadStats();
            loadGrid();
            loadQueueList();
            updateProcessList();
            document.getElementById('zoom-out-btn').disabled = true;
        }

        function updateBreadcrumb() {
            const breadcrumb = document.getElementById('breadcrumb');
            breadcrumb.innerHTML = '<span>Path:</span><a onclick="resetView()">Full Keyspace</a>';

            zoomHistory.forEach((h, i) => {
                if (h.start && h.end) {
                    const a = document.createElement('a');
                    a.textContent = `0x${h.start.substring(0, 8)}...`;
                    a.onclick = () => {
                        zoomHistory = zoomHistory.slice(0, i);
                        currentGridStart = h.start;
                        currentGridEnd = h.end;
                        updateBreadcrumb();
                        loadGrid();
                    };
                    breadcrumb.appendChild(document.createTextNode(' > '));
                    breadcrumb.appendChild(a);
                }
            });

            if (currentGridStart && currentGridEnd) {
                breadcrumb.appendChild(document.createTextNode(' > '));
                const current = document.createElement('span');
                current.textContent = `0x${currentGridStart.substring(0, 8)}...`;
                current.style.color = '#00d4ff';
                breadcrumb.appendChild(current);
            }
        }

        function refreshData() {
            loadStats();
            loadGrid();
            loadQueueList();
        }

        function showTooltip(e) {
            const tooltip = document.getElementById('tooltip');
            const cell = e.target.closest('.cell');
            if (!cell) return;

            const isSkipped = parseFloat(cell.dataset.skippedPct) > 0;
            const keyCount = parseInt(cell.dataset.keyCount) || 0;
            const hasMyScans = cell.dataset.hasMyScans === '1';
            const hasPoolScans = cell.dataset.hasPoolScans === '1';
            const hasBrowserScans = cell.dataset.hasBrowserScans === '1';
            const hasRunning = cell.dataset.hasRunning === '1';
            const myPct = parseFloat(cell.dataset.myPct);
            const poolPct = parseFloat(cell.dataset.poolPct);
            const browserPct = parseFloat(cell.dataset.browserPct) || 0;
            const runningPct = parseFloat(cell.dataset.runningPct) || 0;

            // Format percentage - show "<0.01%" for very tiny values
            const formatPct = (pct, hasScans) => {
                if (pct >= 0.01) return pct.toFixed(4) + '%';
                if (pct > 0 || hasScans) return '<0.01% (scanned)';
                return '0%';
            };

            // Running status indicator
            const runningStatus = hasRunning ?
                `<div class="row" style="background: rgba(0,255,136,0.2); padding: 4px; margin: 4px 0; border-radius: 3px;">
                    <span class="label" style="color: #00ff88;">⚡ MINING:</span>
                    <span class="value" style="color: #00ff88;">${runningPct.toFixed(2)}% complete</span>
                </div>` : '';

            tooltip.innerHTML = `
                <div class="row"><span class="label">Start:</span><span class="value">0x${cell.dataset.startHex}</span></div>
                <div class="row"><span class="label">End:</span><span class="value">0x${cell.dataset.endHex}</span></div>
                <div class="row"><span class="label">Keys in cell:</span><span class="value" style="color: #ffaa00;">${keyCount.toLocaleString()}</span></div>
                ${runningStatus}
                <hr>
                <div class="row"><span class="label">My Coverage:</span><span class="value" style="color: #00ff88;">${formatPct(myPct, hasMyScans)}</span></div>
                <div class="row"><span class="label">Pool Coverage:</span><span class="value" style="color: #4a9eff;">${formatPct(poolPct, hasPoolScans)}</span></div>
                <div class="row"><span class="label">Browser Coverage:</span><span class="value" style="color: #aa44ff;">${formatPct(browserPct, hasBrowserScans)}</span></div>
                <div class="row"><span class="label">Skipped:</span><span class="value" style="color: #ff4444;">${parseFloat(cell.dataset.skippedPct).toFixed(4)}%</span></div>
                <hr>
                <div class="row"><span class="label">My Blocks:</span><span class="value">${cell.dataset.myBlocks}${hasMyScans ? ' ✓' : ''}</span></div>
                <div class="row"><span class="label">Pool Blocks:</span><span class="value">${cell.dataset.poolBlocks}${hasPoolScans ? ' ✓' : ''}</span></div>
                <div class="row"><span class="label">Browser Blocks:</span><span class="value">${cell.dataset.browserBlocks || 0}${hasBrowserScans ? ' ✓' : ''}</span></div>
                <div class="row"><span class="label">Skipped Blocks:</span><span class="value">${cell.dataset.skippedBlocks}</span></div>
                <hr>
                <div style="text-align: center; color: #888; font-size: 9px;">
                    Click to zoom | ▶ mine | ✕ skip
                </div>
            `;
            tooltip.style.display = 'block';
            moveTooltip(e);
        }

        function moveTooltip(e) {
            const tooltip = document.getElementById('tooltip');
            let x = e.clientX + 15;
            let y = e.clientY + 15;

            // Keep tooltip in viewport
            const rect = tooltip.getBoundingClientRect();
            if (x + 380 > window.innerWidth) x = e.clientX - 390;
            if (y + 300 > window.innerHeight) y = e.clientY - 310;

            tooltip.style.left = x + 'px';
            tooltip.style.top = y + 'px';
        }

        function hideTooltip() {
            document.getElementById('tooltip').style.display = 'none';
        }

        function formatNumber(num) {
            if (!num) return '0';
            if (num >= 1e15) return (num / 1e15).toFixed(2) + 'P';
            if (num >= 1e12) return (num / 1e12).toFixed(2) + 'T';
            if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
            if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
            if (num >= 1e3) return (num / 1e3).toFixed(2) + 'K';
            return num.toLocaleString();
        }

        function formatNumberShort(num) {
            if (!num) return '0';
            if (num >= 1e18) return (num / 1e18).toFixed(0) + 'E';
            if (num >= 1e15) return (num / 1e15).toFixed(0) + 'P';
            if (num >= 1e12) return (num / 1e12).toFixed(0) + 'T';
            if (num >= 1e9) return (num / 1e9).toFixed(0) + 'G';
            if (num >= 1e6) return (num / 1e6).toFixed(0) + 'M';
            if (num >= 1e3) return (num / 1e3).toFixed(0) + 'K';
            return num.toString();
        }

        // Context menu for cells
        document.addEventListener('contextmenu', async (e) => {
            const cell = e.target.closest('.cell');
            if (cell) {
                e.preventDefault();
                const isSkipped = parseFloat(cell.dataset.skippedPct) > 0;
                if (isSkipped) {
                    if (confirm('Queue this block for rescanning?')) {
                        await queueForRescan(cell.dataset.startHex);
                    }
                } else {
                    if (confirm('Mark this block as skipped?')) {
                        await markAsSkipped(cell.dataset.startHex, cell.dataset.endHex, 'manual');
                    }
                }
            }
        });

        // ============= SYNC FUNCTIONS =============
        let syncInterval = null;
        let syncConfig = {
            enabled: false,
            server_url: '',
            api_key: '',
            client_name: '',
            auto_sync_interval: 90
        };

        async function loadSyncConfig() {
            try {
                const response = await fetch('/api/sync/config');
                const data = await response.json();
                if (data.status === 'ok') {
                    syncConfig = data;

                    // Update Team Sync panel display (read-only for team version)
                    const clientIdEl = document.getElementById('sync-client-id');
                    const lastTimeEl = document.getElementById('sync-last-time');
                    const intervalEl = document.getElementById('sync-interval-display');

                    if (clientIdEl) clientIdEl.textContent = data.client_id || 'Generating...';
                    if (lastTimeEl) lastTimeEl.textContent = data.last_sync ? new Date(data.last_sync).toLocaleTimeString() : 'Never';
                    if (intervalEl) intervalEl.textContent = 'Every ' + (data.auto_sync_interval || 90) + 's';

                    // Hidden inputs for compatibility
                    const serverUrlEl = document.getElementById('sync-server-url');
                    const apiKeyEl = document.getElementById('sync-api-key');
                    const clientNameEl = document.getElementById('sync-client-name');
                    const enabledEl = document.getElementById('sync-enabled');
                    const intervalInputEl = document.getElementById('sync-interval');

                    if (serverUrlEl) serverUrlEl.value = data.server_url || '';
                    if (apiKeyEl) apiKeyEl.value = data.api_key || '';
                    if (clientNameEl) clientNameEl.value = data.client_name || '';
                    if (enabledEl) enabledEl.value = data.enabled ? 'true' : 'false';
                    if (intervalInputEl) intervalInputEl.value = data.auto_sync_interval || 90;

                    updateSyncStatusUI(data);

                    // Start auto-sync if enabled
                    if (data.enabled && data.auto_sync_interval > 0) {
                        startAutoSync(data.auto_sync_interval);
                    }
                }
            } catch (e) {
                console.error('Failed to load sync config:', e);
            }
        }

        function updateSyncStatusUI(config) {
            const dot = document.getElementById('sync-status-dot');
            const text = document.getElementById('sync-status-text');
            const lastTimeEl = document.getElementById('sync-last-time');

            if (!config.server_url) {
                dot.style.background = '#ff9900';
                dot.title = 'Connecting...';
                text.textContent = 'Connecting...';
                text.style.color = '#ff9900';
            } else if (config.sync_in_progress) {
                dot.style.background = '#00d4ff';
                dot.title = 'Syncing...';
                text.textContent = 'Syncing...';
                text.style.color = '#00d4ff';
            } else if (config.enabled) {
                dot.style.background = '#00ff88';
                dot.title = 'Connected';
                text.textContent = 'Connected';
                text.style.color = '#00ff88';
            } else {
                dot.style.background = '#ff9900';
                dot.title = 'Reconnecting...';
                text.textContent = 'Reconnecting...';
                text.style.color = '#ff9900';
            }

            // Update last sync time
            if (lastTimeEl && config.last_sync) {
                lastTimeEl.textContent = new Date(config.last_sync).toLocaleTimeString();
            }
        }

        async function testSyncConnection() {
            const serverUrl = document.getElementById('sync-server-url').value;
            const apiKey = document.getElementById('sync-api-key').value;
            const resultsDiv = document.getElementById('sync-results');

            if (!serverUrl) {
                resultsDiv.innerHTML = '<span style="color: #ff4444;">Please enter a server URL</span>';
                return;
            }

            resultsDiv.innerHTML = '<span style="color: #00d4ff;">Testing connection...</span>';

            try {
                const response = await fetch('/api/sync/test', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        server_url: serverUrl,
                        api_key: apiKey !== '********' ? apiKey : null
                    })
                });
                const data = await response.json();

                if (data.status === 'ok') {
                    resultsDiv.innerHTML = '<span style="color: #00ff88;">Connection successful! Server: ' + (data.server || 'OK') + '</span>';
                } else {
                    resultsDiv.innerHTML = '<span style="color: #ff4444;">Failed: ' + (data.message || 'Unknown error') + '</span>';
                }
            } catch (e) {
                resultsDiv.innerHTML = '<span style="color: #ff4444;">Error: ' + e.message + '</span>';
            }
        }

        async function saveSyncConfig() {
            // Team version: Sync settings are managed automatically
            const resultsDiv = document.getElementById('sync-results');
            if (resultsDiv) {
                resultsDiv.innerHTML = '<span style="color: #ff9900;">Sync settings are managed automatically for team members.</span>';
            }
            console.log('Team version: Sync settings cannot be modified manually.');
        }

        function startAutoSync(intervalSeconds) {
            if (syncInterval) clearInterval(syncInterval);
            syncInterval = setInterval(() => {
                syncNow(true);  // Silent sync
            }, intervalSeconds * 1000);
            console.log('Auto-sync started: every ' + intervalSeconds + ' seconds');
        }

        async function syncNow(silent = false) {
            const resultsDiv = document.getElementById('sync-results');
            const btn = document.getElementById('sync-now-btn');

            // Ensure a puzzle is selected
            if (!currentPuzzle) {
                resultsDiv.innerHTML = '<span style="color: #ff6b6b;">Please select a puzzle first</span>';
                return;
            }

            if (!silent) {
                resultsDiv.innerHTML = '<span style="color: #00d4ff;">Syncing...</span>';
            }
            btn.disabled = true;
            btn.textContent = 'Syncing...';

            const dot = document.getElementById('sync-status-dot');
            const text = document.getElementById('sync-status-text');
            dot.style.background = '#00d4ff';
            text.textContent = 'Syncing...';

            try {
                const response = await fetch('/api/sync/now', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ puzzle: currentPuzzle })
                });
                const data = await response.json();

                if (data.status === 'ok') {
                    const msg = 'Uploaded: ' + (data.uploaded || 0) + ' | Downloaded: ' + (data.downloaded || 0);
                    if (!silent) {
                        resultsDiv.innerHTML = '<span style="color: #00ff88;">' + msg + '</span>';
                    }
                    dot.style.background = '#00ff88';
                    text.textContent = 'Synced ' + new Date().toLocaleTimeString();

                    // Refresh grid if we downloaded new blocks
                    if (data.downloaded > 0) {
                        loadGrid();
                    }
                } else {
                    if (!silent) {
                        resultsDiv.innerHTML = '<span style="color: #ff4444;">' + (data.message || 'Sync failed') + '</span>';
                    }
                    dot.style.background = '#ff4444';
                    text.textContent = 'Sync failed';
                }
            } catch (e) {
                if (!silent) {
                    resultsDiv.innerHTML = '<span style="color: #ff4444;">Error: ' + e.message + '</span>';
                }
                dot.style.background = '#ff4444';
                text.textContent = 'Error';
            }

            btn.disabled = false;
            btn.textContent = 'Sync Now';
        }

        async function loadSyncStats() {
            const resultsDiv = document.getElementById('sync-results');
            resultsDiv.innerHTML = '<span style="color: #00d4ff;">Loading stats...</span>';

            try {
                const response = await fetch('/api/sync/stats/' + currentPuzzle);
                const data = await response.json();

                if (data.status === 'ok') {
                    let html = '<div style="color: #00ff88; margin-bottom: 4px; font-weight: bold;">Server Stats (Puzzle #' + currentPuzzle + ')</div>';
                    html += '<div style="display: grid; grid-template-columns: auto auto; gap: 2px 8px;">';
                    html += '<div>Active Clients:</div><div style="color: #00ff88;">' + (data.total_clients || 0) + '</div>';
                    html += '<div>My Blocks:</div><div style="color: #00d4ff;">' + formatNumber(data.my_blocks || data.total_blocks || 0) + '</div>';
                    html += '<div>My Keys:</div><div style="color: #00ff88;">' + formatNumber(data.total_keys || 0) + '</div>';
                    html += '<div>Pool Blocks:</div><div style="color: #ff9900;">' + formatNumber(data.pool_blocks || 0) + '</div>';
                    html += '</div>';
                    if (data.last_activity) {
                        html += '<div style="margin-top: 4px; color: #888;">Last: ' + new Date(data.last_activity).toLocaleString() + '</div>';
                    }
                    // Show client breakdown if available
                    if (data.client_breakdown && data.client_breakdown.length > 0) {
                        html += '<div style="margin-top: 6px; border-top: 1px solid #333; padding-top: 4px;">';
                        html += '<div style="color: #888; margin-bottom: 2px;">Per-Client:</div>';
                        data.client_breakdown.forEach(c => {
                            html += '<div style="font-size: 8px;">' + c.client_id + ': ' + formatNumber(c.keys_checked) + ' (' + c.blocks + ' blocks)</div>';
                        });
                        html += '</div>';
                    }
                    resultsDiv.innerHTML = html;
                } else {
                    resultsDiv.innerHTML = '<span style="color: #ff4444;">' + (data.message || 'Failed to load stats') + '</span>';
                }
            } catch (e) {
                resultsDiv.innerHTML = '<span style="color: #ff4444;">Error: ' + e.message + '</span>';
            }
        }

        document.addEventListener('DOMContentLoaded', init);
    </script>
</body>
</html>
'''


# ============= CLOUD SYNC FUNCTIONS =============

def sync_load_config():
    """Load sync configuration from file - Auto-setup for team version"""
    import secrets
    import socket

    config_path = os.path.join(DB_DIR, 'sync_config.json')
    config_changed = False

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                SYNC_CONFIG.update(config)
        except Exception as e:
            print(f"Error loading sync config: {e}")

    # TEAM VERSION AUTO-SETUP: Always ensure sync is properly configured
    # Hardcoded server URL and API key for team members
    TEAM_SERVER_URL = "https://client.intercomserviceslondon.co.uk/wp-json/keyhunt/v1"
    TEAM_API_KEY = "gtHtVP99sNkrveHjfXgLJWijv7nmVd9j"

    # Auto-set server URL if not set or different
    if not SYNC_CONFIG.get('server_url') or SYNC_CONFIG['server_url'] != TEAM_SERVER_URL:
        SYNC_CONFIG['server_url'] = TEAM_SERVER_URL
        config_changed = True

    # Auto-set API key if not set
    if not SYNC_CONFIG.get('api_key') or SYNC_CONFIG['api_key'] != TEAM_API_KEY:
        SYNC_CONFIG['api_key'] = TEAM_API_KEY
        config_changed = True

    # Auto-generate client_id if not set
    if not SYNC_CONFIG.get('client_id'):
        SYNC_CONFIG['client_id'] = secrets.token_hex(4)  # 8 character random ID
        config_changed = True
        print(f"[TEAM SYNC] Generated new client ID: {SYNC_CONFIG['client_id']}")

    # Auto-set client_name based on hostname if not set
    if not SYNC_CONFIG.get('client_name'):
        try:
            hostname = socket.gethostname()
            SYNC_CONFIG['client_name'] = f"Team-{hostname[:20]}"
        except:
            SYNC_CONFIG['client_name'] = f"Team-{SYNC_CONFIG['client_id']}"
        config_changed = True

    # Always enable sync for team version
    if not SYNC_CONFIG.get('enabled'):
        SYNC_CONFIG['enabled'] = True
        config_changed = True

    # Set auto-sync interval to 90 seconds
    if SYNC_CONFIG.get('auto_sync_interval') != 90:
        SYNC_CONFIG['auto_sync_interval'] = 90
        config_changed = True

    # Ensure is_master is False for team version
    if SYNC_CONFIG.get('is_master') != False:
        SYNC_CONFIG['is_master'] = False
        config_changed = True

    # Save config if anything changed
    if config_changed:
        sync_save_config()
        print(f"[TEAM SYNC] Auto-configured sync settings for team member")
        print(f"[TEAM SYNC] Client ID: {SYNC_CONFIG['client_id']}, Name: {SYNC_CONFIG['client_name']}")

    return True

def sync_save_config():
    """Save sync configuration to file"""
    config_path = os.path.join(DB_DIR, 'sync_config.json')
    try:
        with open(config_path, 'w') as f:
            json.dump({
                'enabled': SYNC_CONFIG['enabled'],
                'server_url': SYNC_CONFIG['server_url'],
                'api_key': SYNC_CONFIG['api_key'],
                'client_id': SYNC_CONFIG['client_id'],
                'client_name': SYNC_CONFIG['client_name'],
                'auto_sync_interval': SYNC_CONFIG['auto_sync_interval'],
                'is_master': SYNC_CONFIG.get('is_master', False),
            }, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving sync config: {e}")
        return False

def sync_test_connection(server_url=None, api_key=None):
    """Test connection to sync server. Can use provided values or SYNC_CONFIG."""
    test_url = server_url if server_url else SYNC_CONFIG['server_url']
    test_key = api_key if api_key else SYNC_CONFIG['api_key']

    if not test_url or not test_key:
        return {'status': 'error', 'message': 'Server URL and API key required'}

    # Ensure URL doesn't have trailing slash
    test_url = test_url.rstrip('/')

    try:
        import urllib.request
        import urllib.error

        url = f"{test_url}/ping"
        req = urllib.request.Request(url)
        req.add_header('X-KeyHunt-API-Key', test_key)

        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            return {'status': 'ok', 'server': data}
    except urllib.error.HTTPError as e:
        return {'status': 'error', 'message': f'HTTP {e.code}: {e.reason}'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

def sync_authenticate():
    """Authenticate with sync server and register client"""
    if not SYNC_CONFIG['server_url'] or not SYNC_CONFIG['api_key']:
        return {'status': 'error', 'message': 'Server URL and API key required'}

    # Generate client_id if not set
    if not SYNC_CONFIG['client_id']:
        import uuid
        SYNC_CONFIG['client_id'] = str(uuid.uuid4())[:8]
        sync_save_config()

    if not SYNC_CONFIG['client_name']:
        import socket
        SYNC_CONFIG['client_name'] = socket.gethostname()
        sync_save_config()

    try:
        import urllib.request

        url = f"{SYNC_CONFIG['server_url']}/auth"
        data = json.dumps({
            'client_id': SYNC_CONFIG['client_id'],
            'client_name': SYNC_CONFIG['client_name']
        }).encode()

        req = urllib.request.Request(url, data=data, method='POST')
        req.add_header('Content-Type', 'application/json')
        req.add_header('X-KeyHunt-API-Key', SYNC_CONFIG['api_key'])

        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

def sync_upload_blocks(puzzle_num):
    """Upload local scanned blocks to sync server"""
    if not SYNC_CONFIG['enabled'] or not SYNC_CONFIG['server_url']:
        return {'status': 'error', 'message': 'Sync not configured'}

    try:
        # Get local blocks
        blocks = get_scanned_blocks(puzzle_num)
        if 'error' in blocks:
            return blocks

        my_blocks = []
        for block in blocks['mine']:
            my_blocks.append({
                'start': block['start'],
                'end': block['end'],
                'keys_checked': block.get('keys', 0),
                'completion_pct': 100,
                'scanned_at': block.get('time', '')
            })

        if not my_blocks:
            return {'status': 'ok', 'message': 'No blocks to upload', 'uploaded': 0}

        import urllib.request

        url = f"{SYNC_CONFIG['server_url']}/blocks/upload"
        data = json.dumps({
            'client_id': SYNC_CONFIG['client_id'],
            'puzzle': puzzle_num,
            'blocks': my_blocks
        }).encode()

        req = urllib.request.Request(url, data=data, method='POST')
        req.add_header('Content-Type', 'application/json')
        req.add_header('X-KeyHunt-API-Key', SYNC_CONFIG['api_key'])

        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode())
            return result
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

def sync_download_blocks(puzzle_num):
    """Download blocks from sync server and merge into local database"""
    if not SYNC_CONFIG['enabled'] or not SYNC_CONFIG['server_url']:
        return {'status': 'error', 'message': 'Sync not configured'}

    try:
        import urllib.request

        url = f"{SYNC_CONFIG['server_url']}/blocks/download/{puzzle_num}"
        req = urllib.request.Request(url)
        req.add_header('X-KeyHunt-API-Key', SYNC_CONFIG['api_key'])

        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())

        if data.get('status') != 'ok':
            return data

        # Merge into local database
        db_path = os.path.join(DB_DIR, f'scan_data_puzzle_{puzzle_num}.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Ensure table exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS my_scanned (
                block_start TEXT PRIMARY KEY,
                block_end TEXT,
                scanned_at TEXT,
                keys_checked INTEGER,
                source TEXT DEFAULT 'sync',
                completion_pct REAL DEFAULT 100.0
            )
        ''')

        imported = 0
        for block in data.get('blocks', []):
            # Only import if from other clients or if more complete
            if block.get('client_id') != SYNC_CONFIG['client_id']:
                cursor.execute('''
                    INSERT OR REPLACE INTO my_scanned
                    (block_start, block_end, scanned_at, keys_checked, source, completion_pct)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    block['start'],
                    block['end'],
                    block.get('scanned_at', ''),
                    block.get('keys_checked', 0),
                    f"sync:{block.get('client_id', 'unknown')}",
                    block.get('completion_pct', 100)
                ))
                imported += 1

        conn.commit()
        conn.close()

        return {'status': 'ok', 'imported': imported, 'total': len(data.get('blocks', []))}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

def sync_full_sync(puzzle_num):
    """Perform full bidirectional sync"""
    if not SYNC_CONFIG['enabled'] or not SYNC_CONFIG['server_url']:
        return {'status': 'error', 'message': 'Sync not configured'}

    SYNC_CONFIG['sync_in_progress'] = True

    try:
        # Get local blocks
        blocks = get_scanned_blocks(puzzle_num)
        my_blocks = []
        pool_blocks = []

        for block in blocks.get('mine', []):
            keys_checked = block.get('keys', 0) or 0
            my_blocks.append({
                'start': block['start'],
                'end': block['end'],
                'keys_checked': keys_checked,
                'keys_in_block': keys_checked,  # Use actual keys checked, not full range
                'completion_pct': 100,
                'scanned_at': block.get('time', '')
            })

        for block in blocks.get('pool', []):
            pool_blocks.append({
                'start': block['start'],
                'end': block['end']
            })

        import urllib.request

        url = f"{SYNC_CONFIG['server_url']}/sync/{puzzle_num}"
        data = json.dumps({
            'client_id': SYNC_CONFIG['client_id'],
            'my_blocks': my_blocks,
            'pool_blocks': pool_blocks,
            'last_sync': SYNC_CONFIG['last_sync']
        }).encode()

        req = urllib.request.Request(url, data=data, method='POST')
        req.add_header('Content-Type', 'application/json')
        req.add_header('X-KeyHunt-API-Key', SYNC_CONFIG['api_key'])

        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode())

        if result.get('status') != 'ok':
            return result

        # Import server blocks into local database
        db_path = os.path.join(DB_DIR, f'scan_data_puzzle_{puzzle_num}.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        imported = 0
        for block in result.get('server_blocks', []):
            if block.get('client_id') != SYNC_CONFIG['client_id']:
                cursor.execute('''
                    INSERT OR IGNORE INTO my_scanned
                    (block_start, block_end, scanned_at, keys_checked, source, completion_pct)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    block['start'],
                    block['end'],
                    block.get('scanned_at', ''),
                    block.get('keys_checked', 0),
                    f"sync:{block.get('client_id', 'unknown')}",
                    block.get('completion_pct', 100)
                ))
                if cursor.rowcount > 0:
                    imported += 1

        # Import pool blocks
        pool_imported = 0
        for block in result.get('server_pool', []):
            cursor.execute('''
                INSERT OR IGNORE INTO pool_scanned (block_start, block_end, scraped_at)
                VALUES (?, ?, ?)
            ''', (block['start'], block['end'], time.strftime('%Y-%m-%dT%H:%M:%S')))
            if cursor.rowcount > 0:
                pool_imported += 1

        # Import browser miner blocks
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS browser_scanned (
                block_start TEXT PRIMARY KEY,
                block_end TEXT,
                scanned_at TEXT,
                keys_checked INTEGER DEFAULT 0,
                session_id TEXT
            )
        ''')

        browser_imported = 0
        for block in result.get('browser_blocks', []):
            cursor.execute('''
                INSERT OR REPLACE INTO browser_scanned
                (block_start, block_end, scanned_at, keys_checked, session_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                block['start'],
                block['end'],
                block.get('scanned_at', ''),
                block.get('keys_checked', 0),
                block.get('session_id', '')
            ))
            if cursor.rowcount > 0:
                browser_imported += 1

        conn.commit()
        conn.close()

        SYNC_CONFIG['last_sync'] = result.get('sync_time')

        return {
            'status': 'ok',
            'uploaded': result.get('uploaded', 0),
            'downloaded': imported + pool_imported + browser_imported,
            'imported': imported,
            'pool_imported': pool_imported,
            'browser_imported': browser_imported,
            'sync_time': result.get('sync_time')
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}
    finally:
        SYNC_CONFIG['sync_in_progress'] = False

def sync_upload_found_key(puzzle, priv_hex, priv_wif, pub_address, block_start, block_end, raw_output):
    """Upload found key to sync server - CRITICAL!"""
    if not SYNC_CONFIG['server_url'] or not SYNC_CONFIG['api_key']:
        return {'status': 'error', 'message': 'Sync not configured'}

    try:
        import urllib.request

        url = f"{SYNC_CONFIG['server_url']}/found/upload"
        data = json.dumps({
            'puzzle': puzzle,
            'client_id': SYNC_CONFIG['client_id'],
            'priv_hex': priv_hex,
            'priv_wif': priv_wif,
            'pub_address': pub_address,
            'block_start': block_start,
            'block_end': block_end,
            'raw_output': raw_output
        }).encode()

        req = urllib.request.Request(url, data=data, method='POST')
        req.add_header('Content-Type', 'application/json')
        req.add_header('X-KeyHunt-API-Key', SYNC_CONFIG['api_key'])

        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"Error uploading found key to sync server: {e}")
        return {'status': 'error', 'message': str(e)}

def sync_get_stats(puzzle_num):
    """Get sync server stats for a puzzle"""
    if not SYNC_CONFIG['server_url'] or not SYNC_CONFIG['api_key']:
        return {'status': 'error', 'message': 'Sync not configured'}

    try:
        import urllib.request

        url = f"{SYNC_CONFIG['server_url']}/blocks/stats/{puzzle_num}"
        req = urllib.request.Request(url)
        req.add_header('X-KeyHunt-API-Key', SYNC_CONFIG['api_key'])

        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

# Load sync config on startup
sync_load_config()


class VisualizerHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for the visualizer"""

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == '/' or path == '/index.html':
            self.send_html()
        elif path == '/api/puzzles':
            # Include default puzzles (71-80) plus any with existing databases
            available = set(get_available_databases().keys())
            available.update(range(71, 81))  # Always show puzzles 71-80
            self.send_json(sorted(list(available)))
        elif path.startswith('/api/stats/'):
            try:
                puzzle = int(path.split('/')[-1])
                self.send_json(get_puzzle_stats(puzzle))
            except ValueError:
                self.send_json({'error': 'Invalid puzzle number'})
        elif path.startswith('/api/grid/'):
            try:
                puzzle = int(path.split('/')[-1])

                if puzzle in PUZZLE_RANGES:
                    default_start, default_end = PUZZLE_RANGES[puzzle]
                else:
                    default_start = 2 ** (puzzle - 1)
                    default_end = 2 ** puzzle - 1

                start_hex = query.get('start', [None])[0]
                end_hex = query.get('end', [None])[0]

                if start_hex and end_hex:
                    grid_start = int(start_hex, 16)
                    grid_end = int(end_hex, 16)
                else:
                    grid_start = default_start
                    grid_end = default_end

                self.send_json(calculate_grid_coverage(puzzle, grid_start, grid_end))
            except ValueError as e:
                self.send_json({'error': f'Invalid parameters: {e}'})
        elif path.startswith('/api/blocks/'):
            try:
                puzzle = int(path.split('/')[-1])
                self.send_json(get_scanned_blocks(puzzle))
            except ValueError:
                self.send_json({'error': 'Invalid puzzle number'})
        elif path.startswith('/api/skipped/'):
            try:
                puzzle = int(path.split('/')[-1])
                self.send_json(filter_manager.get_skipped_blocks(puzzle))
            except ValueError:
                self.send_json({'error': 'Invalid puzzle number'})
        elif path == '/api/processes':
            self.send_json(process_manager.get_status())
        elif path.startswith('/api/process/'):
            try:
                pid = int(path.split('/')[-1])
                self.send_json(process_manager.get_status(pid))
            except ValueError:
                self.send_json({'error': 'Invalid process ID'})
        elif path == '/api/found':
            # Get found keys
            self.send_json({
                'found': process_manager.get_found_keys(),
                'has_found': process_manager.has_key_found()
            })
        elif path == '/api/gpus':
            # Get available GPUs
            gpus = detect_gpus()
            self.send_json({
                'gpus': gpus,
                'assignments': GPU_CONFIG['gpu_assignments']
            })
        elif path == '/api/settings':
            # Get current settings
            running_count = sum(1 for p in process_manager.processes.values() if p['status'] == 'running')
            self.send_json({
                'max_concurrent_processes': MAX_CONCURRENT_PROCESSES,
                'running_processes': running_count
            })
        elif path == '/api/session-stats':
            # Get session statistics
            elapsed = time.time() - SESSION_STATS['start_time']
            hours = elapsed / 3600

            # Calculate current total from running processes
            current_keys = SESSION_STATS['total_keys_checked']
            current_speed = 0
            for proc in process_manager.processes.values():
                if proc['status'] == 'running':
                    current_keys += proc.get('keys_checked', 0)
                    # Parse speed
                    speed_str = proc.get('speed', '0')
                    try:
                        if 'Gk/s' in speed_str:
                            current_speed += float(speed_str.replace('Gk/s', '').strip()) * 1e9
                        elif 'Mk/s' in speed_str:
                            current_speed += float(speed_str.replace('Mk/s', '').strip()) * 1e6
                        elif 'k/s' in speed_str:
                            current_speed += float(speed_str.replace('k/s', '').strip()) * 1e3
                    except:
                        pass

            self.send_json({
                'session_start': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(SESSION_STATS['start_time'])),
                'elapsed_hours': round(hours, 2),
                'total_keys_checked': current_keys,
                'blocks_completed': SESSION_STATS['blocks_completed'],
                'keys_per_hour': round(current_keys / hours if hours > 0 else 0, 0),
                'current_speed': current_speed,
                'current_speed_formatted': f"{current_speed/1e6:.2f} Mk/s" if current_speed > 0 else "0 Mk/s",
                'active_gpus': len([g for g in GPU_CONFIG['available_gpus'] if g['in_use']])
            })
        elif path == '/api/pattern-analysis':
            # Get solved puzzle pattern analysis
            self.send_json(analyze_solved_puzzles())
        elif path.startswith('/api/range-priority/'):
            # Get priority score for a range
            try:
                parts = path.split('/')
                puzzle = int(parts[-1])
                start = query.get('start', [None])[0]
                end = query.get('end', [None])[0]
                if start and end:
                    self.send_json(calculate_range_priority(start, end, puzzle))
                else:
                    self.send_json({'error': 'Missing start or end parameter'})
            except ValueError:
                self.send_json({'error': 'Invalid puzzle number'})
        elif path.startswith('/api/checkpoints/'):
            # Get checkpoints for a puzzle
            try:
                puzzle = int(path.split('/')[-1])
                self.send_json(get_checkpoints(puzzle))
            except ValueError:
                self.send_json({'error': 'Invalid puzzle number'})
        elif path.startswith('/api/heatmap/'):
            # Get heatmap data for visualization
            try:
                puzzle = int(path.split('/')[-1])
                resolution = int(query.get('resolution', ['100'])[0])
                self.send_json(generate_heatmap_data(puzzle, resolution))
            except ValueError:
                self.send_json({'error': 'Invalid parameters'})
        elif path.startswith('/api/pool/status/'):
            # Get pool scraper status
            try:
                puzzle = int(path.split('/')[-1])
                result = pool_scraper.get_last_scrape(puzzle)
                self.send_json(result if result else {'status': 'never_scraped'})
            except ValueError:
                self.send_json({'error': 'Invalid puzzle number'})
        elif path.startswith('/api/partial-blocks/'):
            # Get partial blocks for resuming
            try:
                puzzle = int(path.split('/')[-1])
                partials = get_partial_blocks(puzzle)
                self.send_json(partials)
            except ValueError:
                self.send_json({'error': 'Invalid puzzle number'})
        elif path.startswith('/api/pool/analyze/'):
            # Analyze pool patterns
            try:
                puzzle = int(path.split('/')[-1])
                analysis = analyze_pool_patterns(puzzle)
                self.send_json(analysis)
            except ValueError:
                self.send_json({'error': 'Invalid puzzle number'})
        # ============= SYNC API ENDPOINTS =============
        elif path == '/api/sync/config':
            # Get sync configuration
            self.send_json({
                'status': 'ok',
                'enabled': SYNC_CONFIG['enabled'],
                'server_url': SYNC_CONFIG['server_url'],
                'api_key': SYNC_CONFIG['api_key'][:8] + '...' if SYNC_CONFIG['api_key'] else '',
                'client_id': SYNC_CONFIG['client_id'],
                'client_name': SYNC_CONFIG['client_name'],
                'auto_sync_interval': SYNC_CONFIG['auto_sync_interval'],
                'last_sync': SYNC_CONFIG['last_sync'],
                'sync_in_progress': SYNC_CONFIG['sync_in_progress']
            })
        elif path == '/api/sync/test':
            # Test sync server connection
            self.send_json(sync_test_connection())
        elif path.startswith('/api/sync/stats/'):
            # Get sync server stats
            try:
                puzzle = int(path.split('/')[-1])
                self.send_json(sync_get_stats(puzzle))
            except ValueError:
                self.send_json({'error': 'Invalid puzzle number'})
        else:
            self.send_error(404, 'Not found')

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else '{}'

        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}

        if path == '/api/mine/start':
            puzzle = data.get('puzzle')
            start = data.get('start')
            end = data.get('end')
            use_gpu = data.get('gpu', True)
            use_filter = data.get('filter', False)

            if not all([puzzle, start, end]):
                self.send_json({'error': 'Missing required parameters'})
                return

            result = process_manager.start_mining(puzzle, start, end, use_gpu, use_filter)
            self.send_json(result)

        elif path.startswith('/api/mine/pause/'):
            # Pause process and save checkpoint for later resume
            try:
                pid = int(path.split('/')[-1])
                result = process_manager.pause_mining(pid)
                self.send_json(result)
            except ValueError:
                self.send_json({'error': 'Invalid process ID'})

        elif path.startswith('/api/mine/stop/'):
            try:
                pid = int(path.split('/')[-1])
                # Check if skip_checkpoint is requested
                skip_checkpoint = 'skip_checkpoint=1' in (self.path.split('?')[1] if '?' in self.path else '')
                result = process_manager.stop_mining(pid, skip_checkpoint=skip_checkpoint)
                self.send_json(result)
            except ValueError:
                self.send_json({'error': 'Invalid process ID'})

        elif path == '/api/processes/clear-completed':
            # Clear completed/stopped processes from tracking
            result = process_manager.clear_completed()
            self.send_json(result)

        elif path == '/api/kill-all-gpu':
            # Kill all KeyHunt GPU processes on the system
            result = kill_all_gpu_processes()
            self.send_json(result)

        elif path == '/api/skip/add':
            puzzle = data.get('puzzle')
            start = data.get('start')
            end = data.get('end')
            reason = data.get('reason', 'manual')

            result = filter_manager.add_skipped_block(puzzle, start, end, reason)
            self.send_json(result)

        elif path == '/api/skip/queue':
            puzzle = data.get('puzzle')
            start = data.get('start')

            result = filter_manager.queue_for_rescan(puzzle, start)
            self.send_json(result)

        elif path == '/api/skip/remove':
            puzzle = data.get('puzzle')
            start = data.get('start')

            result = filter_manager.remove_skipped(puzzle, start)
            self.send_json(result)

        elif path == '/api/pool/scrape':
            # Trigger pool scraping
            puzzle = data.get('puzzle')
            if not puzzle:
                self.send_json({'error': 'Missing puzzle number'})
                return

            # Scrape in background thread to avoid blocking
            def do_scrape():
                result = pool_scraper.scrape_puzzle(puzzle)
                if result.get('status') == 'success' and result.get('blocks'):
                    save_result = pool_scraper.save_to_database(puzzle, result['blocks'])
                    result['saved'] = save_result
                return result

            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(do_scrape)
                try:
                    result = future.result(timeout=60)
                    self.send_json(result)
                except concurrent.futures.TimeoutError:
                    self.send_json({'error': 'Scraping timed out'})

        elif path == '/api/filter/preview':
            # Preview filter results for a range
            start = data.get('start')
            end = data.get('end')
            subrange_size = data.get('subrange_size', 10_000_000)
            exclude_iter3 = data.get('iter3', True)
            exclude_iter4 = data.get('iter4', False)
            exclude_alphanum = data.get('alphanum', True)

            if not start or not end:
                self.send_json({'error': 'Missing start or end'})
                return

            try:
                start_int = int(start, 16)
                end_int = int(end, 16)

                filter_obj = SmartBatchFilter(
                    start_int, end_int, subrange_size,
                    exclude_iter3, exclude_iter4, exclude_alphanum
                )

                stats = filter_obj.get_statistics()
                self.send_json({
                    'status': 'success',
                    **stats
                })
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/filter/apply':
            # Apply filter and save skipped ranges to database
            puzzle = data.get('puzzle')
            start = data.get('start')
            end = data.get('end')
            subrange_size = data.get('subrange_size', 10_000_000)
            exclude_iter3 = data.get('iter3', True)
            exclude_iter4 = data.get('iter4', False)
            exclude_alphanum = data.get('alphanum', True)

            if not all([puzzle, start, end]):
                self.send_json({'error': 'Missing required parameters'})
                return

            try:
                start_int = int(start, 16)
                end_int = int(end, 16)

                filter_obj = SmartBatchFilter(
                    start_int, end_int, subrange_size,
                    exclude_iter3, exclude_iter4, exclude_alphanum
                )

                clean_ranges, skipped_ranges = filter_obj.generate_clean_subranges()

                # Save skipped ranges to database
                saved_count = 0
                for skip_start, skip_end, reason in skipped_ranges:
                    skip_start_hex = hex(skip_start)[2:].upper()
                    skip_end_hex = hex(skip_end)[2:].upper()
                    filter_manager.add_skipped_block(puzzle, skip_start_hex, skip_end_hex, f'filter:{reason}')
                    saved_count += 1

                stats = filter_obj.get_statistics()
                self.send_json({
                    'status': 'success',
                    'skipped_saved': saved_count,
                    'clean_ranges': len(clean_ranges),
                    **stats
                })
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/mine/start-filtered':
            # Start mining with smart filtering - mines only clean sub-ranges
            puzzle = data.get('puzzle')
            start = data.get('start')
            end = data.get('end')
            use_gpu = data.get('gpu', True)
            subrange_size = data.get('subrange_size', 10_000_000)
            exclude_iter3 = data.get('iter3', True)
            exclude_iter4 = data.get('iter4', False)
            exclude_alphanum = data.get('alphanum', True)

            if not all([puzzle, start, end]):
                self.send_json({'error': 'Missing required parameters'})
                return

            try:
                start_int = int(start, 16)
                end_int = int(end, 16)

                filter_obj = SmartBatchFilter(
                    start_int, end_int, subrange_size,
                    exclude_iter3, exclude_iter4, exclude_alphanum
                )

                clean_ranges, skipped_ranges = filter_obj.generate_clean_subranges()

                if not clean_ranges:
                    self.send_json({'error': 'No clean ranges found after filtering'})
                    return

                # Start mining the first clean range
                first_clean = clean_ranges[0]
                first_start = hex(first_clean[0])[2:].upper()
                first_end = hex(first_clean[1])[2:].upper()

                result = process_manager.start_mining(puzzle, first_start, first_end, use_gpu, False)

                # Add info about remaining ranges
                result['filter_info'] = {
                    'total_clean_ranges': len(clean_ranges),
                    'total_skipped_ranges': len(skipped_ranges),
                    'current_range': 1,
                    'reduction_pct': filter_obj.get_statistics()['reduction_pct']
                }

                self.send_json(result)
            except Exception as e:
                self.send_json({'error': str(e)})

        # ============= CHECKPOINT MANAGEMENT ENDPOINTS =============
        elif path == '/api/checkpoint/delete':
            # Delete a checkpoint
            puzzle = data.get('puzzle')
            resume_position = data.get('resume_position')

            if not puzzle or not resume_position:
                self.send_json({'error': 'Missing puzzle or resume_position'})
                return

            result = delete_checkpoint(puzzle, resume_position)
            self.send_json(result)

        elif path == '/api/partial-scan/delete':
            # Delete a partial scan record (from my_scanned table)
            puzzle = data.get('puzzle')
            block_start = data.get('block_start')

            if not puzzle or not block_start:
                self.send_json({'error': 'Missing puzzle or block_start'})
                return

            result = delete_partial_scan(puzzle, block_start)
            self.send_json(result)

        # ============= SYNC POST ENDPOINTS =============
        elif path == '/api/sync/test':
            # Test connection with provided values (before saving)
            server_url = data.get('server_url', '').rstrip('/')
            api_key = data.get('api_key')

            # If api_key is null/None, use existing config
            if not api_key:
                api_key = SYNC_CONFIG['api_key']

            result = sync_test_connection(server_url, api_key)
            self.send_json(result)

        elif path == '/api/sync/config':
            # Update sync configuration
            server_url = data.get('server_url', '').rstrip('/')
            api_key = data.get('api_key')
            client_name = data.get('client_name', '')
            enabled = data.get('enabled', False)
            auto_sync_interval = int(data.get('auto_sync_interval', 300))

            SYNC_CONFIG['server_url'] = server_url
            # Only update API key if a new one is provided (not None or placeholder)
            if api_key and api_key != '********':
                SYNC_CONFIG['api_key'] = api_key
            SYNC_CONFIG['enabled'] = enabled
            SYNC_CONFIG['auto_sync_interval'] = auto_sync_interval
            if client_name:
                SYNC_CONFIG['client_name'] = client_name

            sync_save_config()

            # Build response with full config data
            response_data = {
                'status': 'ok',
                'message': 'Configuration saved',
                'enabled': SYNC_CONFIG['enabled'],
                'server_url': SYNC_CONFIG['server_url'],
                'client_id': SYNC_CONFIG['client_id'],
                'client_name': SYNC_CONFIG['client_name'],
                'auto_sync_interval': SYNC_CONFIG['auto_sync_interval'],
                'last_sync': SYNC_CONFIG['last_sync'],
                'sync_in_progress': SYNC_CONFIG['sync_in_progress']
            }

            # If enabled, try to authenticate
            if enabled and server_url and SYNC_CONFIG['api_key']:
                auth_result = sync_authenticate()
                response_data['auth_result'] = auth_result

            self.send_json(response_data)

        elif path == '/api/sync/now':
            # Trigger immediate sync
            puzzle = data.get('puzzle')
            if not puzzle:
                self.send_json({'error': 'Puzzle number required'})
                return

            if SYNC_CONFIG['sync_in_progress']:
                self.send_json({'status': 'busy', 'message': 'Sync already in progress'})
                return

            result = sync_full_sync(puzzle)
            self.send_json(result)

        elif path == '/api/sync/upload':
            # Upload blocks to server
            puzzle = data.get('puzzle')
            if not puzzle:
                self.send_json({'error': 'Puzzle number required'})
                return

            result = sync_upload_blocks(puzzle)
            self.send_json(result)

        elif path == '/api/sync/download':
            # Download blocks from server
            puzzle = data.get('puzzle')
            if not puzzle:
                self.send_json({'error': 'Puzzle number required'})
                return

            result = sync_download_blocks(puzzle)
            self.send_json(result)

        else:
            self.send_error(404, 'Not found')

    def send_html(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(HTML_TEMPLATE.encode())

    def send_json(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        pass


def run_server(port=DEFAULT_PORT, open_browser=True):
    """Run the HTTP server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, VisualizerHandler)

    # Detect GPUs at startup
    gpus = detect_gpus()

    print(f"\n{'='*60}")
    print(f"  KeyHunt Visualizer v3.0 - Advanced Mining Control Center")
    print(f"{'='*60}")
    print(f"\n  GPUs Detected: {len(gpus)}")
    for gpu in gpus:
        print(f"    - GPU {gpu['id']}: {gpu['name']} ({gpu['memory_mb']} MB)")
    if not gpus:
        print(f"    - No GPUs detected (CPU mode only)")
    print(f"\n  Server running at: http://localhost:{port}")
    print(f"  Press Ctrl+C to stop\n")
    print(f"  Features:")
    print(f"    - Click grid cells to zoom in")
    print(f"    - Click ▶ button to start mining a cell")
    print(f"    - Red cells = skipped/filtered blocks")
    print(f"    - Right-click red cells to queue for rescan")
    print(f"{'='*60}\n")

    if open_browser:
        def open_browser_delayed():
            time.sleep(0.5)
            webbrowser.open(f'http://localhost:{port}')
        threading.Thread(target=open_browser_delayed, daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        # Stop all mining processes
        for pid in list(process_manager.processes.keys()):
            process_manager.stop_mining(pid)
        httpd.shutdown()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='KeyHunt Visualizer v3.0')
    parser.add_argument('-p', '--port', type=int, default=DEFAULT_PORT,
                        help=f'Port to run server on (default: {DEFAULT_PORT})')
    parser.add_argument('--no-browser', action='store_true',
                        help='Do not open browser automatically')

    args = parser.parse_args()
    run_server(port=args.port, open_browser=not args.no_browser)
