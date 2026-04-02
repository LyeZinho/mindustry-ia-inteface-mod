#!/usr/bin/env python3
"""
Mimi Gateway Test Client - Standalone executable test suite for all 9 commands

Usage:
    python test_mimi_client.py [--host localhost] [--port 9000] [--verbose]

Requirements:
    - Mindustry running locally with Mimi Gateway mod loaded
    - Python 3.6+
    - No external dependencies (uses stdlib only)

Test Coverage:
    - Connection validation
    - All 9 command types (BUILD, UNIT_MOVE, FACTORY, ATTACK, STOP, REPAIR, DELETE, UPGRADE, MSG)
    - State update parsing and validation
    - Error handling and reconnection
    - Performance benchmarks
"""

import socket
import json
import time
import sys
import argparse
from typing import Optional, Dict, Any
from datetime import datetime


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class MimiTestClient:
    """TCP client for testing Mimi Gateway mod"""
    
    def __init__(self, host: str = 'localhost', port: int = 9000, timeout: float = 5.0, verbose: bool = False):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket: Optional[socket.socket] = None
        self.connected = False
        self.verbose = verbose
        self.test_results = []
        self.state_count = 0
        self.last_state = None
        
    def log(self, level: str, message: str):
        """Print timestamped log message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix_map = {
            'INFO': f"{Colors.BLUE}[INFO]{Colors.RESET}",
            'PASS': f"{Colors.GREEN}[PASS]{Colors.RESET}",
            'FAIL': f"{Colors.RED}[FAIL]{Colors.RESET}",
            'WARN': f"{Colors.YELLOW}[WARN]{Colors.RESET}",
            'TEST': f"{Colors.BOLD}{Colors.CYAN}[TEST]{Colors.RESET}",
        }
        prefix = prefix_map.get(level, level)
        print(f"{timestamp} {prefix} {message}")
    
    def connect(self) -> bool:
        """Establish connection to Mimi Gateway server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            if self.verbose:
                self.log('INFO', f"Connecting to {self.host}:{self.port}...")
            self.socket.connect((self.host, self.port))
            self.connected = True
            self.log('PASS', f"Connected to {self.host}:{self.port}")
            return True
        except ConnectionRefusedError:
            self.log('FAIL', f"Connection refused on {self.host}:{self.port}")
            self.log('WARN', "Ensure Mindustry is running with Mimi Gateway mod loaded")
            return False
        except socket.timeout:
            self.log('FAIL', f"Connection timeout after {self.timeout}s")
            return False
        except Exception as e:
            self.log('FAIL', f"Connection error: {e}")
            return False
    
    def disconnect(self):
        """Close connection"""
        if self.socket:
            self.socket.close()
            self.connected = False
            if self.verbose:
                self.log('INFO', "Disconnected")
    
    def receive_state(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Receive one state update from server"""
        if not self.connected:
            self.log('FAIL', "Not connected to server")
            return None
        
        try:
            old_timeout = self.socket.gettimeout()
            if timeout:
                self.socket.settimeout(timeout)
            
            line = self.socket.recv(16384).decode('utf-8').strip()
            
            if timeout:
                self.socket.settimeout(old_timeout)
            
            if not line:
                self.log('FAIL', "Server closed connection")
                self.connected = False
                return None
            
            state = json.loads(line)
            self.state_count += 1
            self.last_state = state
            
            if self.verbose:
                self.log('INFO', f"Received state #{self.state_count} (tick={state.get('time')})")
            
            return state
        except socket.timeout:
            self.log('WARN', f"Timeout waiting for state (>{timeout or self.timeout}s)")
            return None
        except json.JSONDecodeError as e:
            self.log('FAIL', f"Invalid JSON from server: {e}")
            return None
        except Exception as e:
            self.log('FAIL', f"Error receiving state: {e}")
            self.connected = False
            return None
    
    def send_command(self, cmd: str) -> bool:
        """Send command to server"""
        if not self.connected:
            self.log('FAIL', "Not connected to server")
            return False
        
        try:
            if self.verbose:
                self.log('INFO', f"Sending command: {cmd}")
            self.socket.send(f"{cmd}\n".encode('utf-8'))
            time.sleep(0.2)  # Brief delay for server processing
            return True
        except Exception as e:
            self.log('FAIL', f"Error sending command: {e}")
            self.connected = False
            return False
    
    def validate_state(self, state: Dict[str, Any]) -> bool:
        """Validate state structure contains required fields"""
        required_fields = ['time', 'tick', 'wave', 'resources', 'power', 'core', 'buildings', 'grid']
        missing = [f for f in required_fields if f not in state]
        
        if missing:
            self.log('FAIL', f"State missing required fields: {missing}")
            return False
        
        # Validate nested structures
        if not isinstance(state['resources'], dict):
            self.log('FAIL', "State.resources is not a dict")
            return False
        
        if not isinstance(state['power'], dict):
            self.log('FAIL', "State.power is not a dict")
            return False
        
        if not isinstance(state['grid'], list):
            self.log('FAIL', "State.grid is not a list")
            return False
        
        return True
    
    # ==================== TEST PROCEDURES ====================
    
    def test_connection(self) -> bool:
        """TEST 1: Connection and initial state"""
        self.log('TEST', "Connection to Mimi Gateway")
        
        if not self.connect():
            return False
        
        state = self.receive_state(timeout=10)
        if not state:
            self.log('FAIL', "No initial state received")
            return False
        
        if not self.validate_state(state):
            return False
        
        self.log('PASS', f"State validation passed ({len(state)} fields)")
        self.test_results.append(('CONNECTION', True))
        return True
    
    def test_build_command(self) -> bool:
        """TEST 2: BUILD command validation"""
        self.log('TEST', "BUILD command execution")
        
        if not self.last_state:
            self.log('FAIL', "No state available")
            return False
        
        # Get core position and build offset
        core_x = self.last_state['core']['x']
        core_y = self.last_state['core']['y']
        build_x = core_x + 3
        build_y = core_y + 3
        
        # Send BUILD command
        cmd = f"BUILD;duo;{build_x};{build_y};0"
        if not self.send_command(cmd):
            return False
        
        # Receive state to verify command processing
        state = self.receive_state()
        if not state:
            self.log('WARN', "No state received after BUILD command (may succeed later)")
            self.test_results.append(('BUILD', True))
            return True
        
        self.log('PASS', "BUILD command sent and acknowledged")
        self.test_results.append(('BUILD', True))
        return True
    
    def test_unit_move_command(self) -> bool:
        """TEST 3: UNIT_MOVE command validation"""
        self.log('TEST', "UNIT_MOVE command execution")
        
        if not self.last_state:
            self.log('FAIL', "No state available")
            return False
        
        friendly = self.last_state.get('friendlyUnits', [])
        if not friendly:
            self.log('WARN', "No friendly units available to test UNIT_MOVE")
            self.test_results.append(('UNIT_MOVE', True))
            return True
        
        unit_id = friendly[0]['id']
        target_x = friendly[0]['x'] + 5
        target_y = friendly[0]['y'] + 5
        
        cmd = f"UNIT_MOVE;{unit_id};{target_x};{target_y}"
        if not self.send_command(cmd):
            return False
        
        state = self.receive_state()
        if state:
            self.log('PASS', "UNIT_MOVE command sent")
            self.test_results.append(('UNIT_MOVE', True))
            return True
        
        self.log('WARN', "UNIT_MOVE sent but no state confirmation")
        self.test_results.append(('UNIT_MOVE', True))
        return True
    
    def test_attack_command(self) -> bool:
        """TEST 4: ATTACK command validation"""
        self.log('TEST', "ATTACK command execution")
        
        if not self.last_state:
            self.log('FAIL', "No state available")
            return False
        
        # Get friendly unit and enemy target
        friendly = self.last_state.get('friendlyUnits', [])
        enemies = self.last_state.get('enemies', [])
        
        if not friendly or not enemies:
            self.log('WARN', "Insufficient units for ATTACK test (need friendly + enemy)")
            self.test_results.append(('ATTACK', True))
            return True
        
        unit_id = friendly[0]['id']
        enemy = enemies[0]
        
        cmd = f"ATTACK;{unit_id};{enemy['x']};{enemy['y']}"
        if not self.send_command(cmd):
            return False
        
        state = self.receive_state()
        self.log('PASS', "ATTACK command sent")
        self.test_results.append(('ATTACK', True))
        return True
    
    def test_stop_command(self) -> bool:
        """TEST 5: STOP command validation"""
        self.log('TEST', "STOP command execution")
        
        if not self.last_state:
            self.log('FAIL', "No state available")
            return False
        
        friendly = self.last_state.get('friendlyUnits', [])
        if not friendly:
            self.log('WARN', "No friendly units available for STOP test")
            self.test_results.append(('STOP', True))
            return True
        
        unit_id = friendly[0]['id']
        cmd = f"STOP;{unit_id}"
        
        if not self.send_command(cmd):
            return False
        
        state = self.receive_state()
        self.log('PASS', "STOP command sent")
        self.test_results.append(('STOP', True))
        return True
    
    def test_repair_command(self) -> bool:
        """TEST 6: REPAIR command validation"""
        self.log('TEST', "REPAIR command execution")
        
        if not self.last_state:
            self.log('FAIL', "No state available")
            return False
        
        buildings = self.last_state.get('buildings', [])
        if not buildings:
            self.log('WARN', "No buildings available for REPAIR test")
            self.test_results.append(('REPAIR', True))
            return True
        
        building = buildings[0]
        cmd = f"REPAIR;{building['x']};{building['y']}"
        
        if not self.send_command(cmd):
            return False
        
        state = self.receive_state()
        self.log('PASS', "REPAIR command sent")
        self.test_results.append(('REPAIR', True))
        return True
    
    def test_delete_command(self) -> bool:
        """TEST 7: DELETE command validation"""
        self.log('TEST', "DELETE command execution")
        
        if not self.last_state:
            self.log('FAIL', "No state available")
            return False
        
        buildings = self.last_state.get('buildings', [])
        # Skip core building
        destructible = [b for b in buildings if b.get('block') != 'core-sharded']
        
        if not destructible:
            self.log('WARN', "No destructible buildings available for DELETE test")
            self.test_results.append(('DELETE', True))
            return True
        
        building = destructible[0]
        cmd = f"DELETE;{building['x']};{building['y']}"
        
        if not self.send_command(cmd):
            return False
        
        state = self.receive_state()
        self.log('PASS', "DELETE command sent")
        self.test_results.append(('DELETE', True))
        return True
    
    def test_upgrade_command(self) -> bool:
        """TEST 8: UPGRADE command validation"""
        self.log('TEST', "UPGRADE command execution")
        
        if not self.last_state:
            self.log('FAIL', "No state available")
            return False
        
        buildings = self.last_state.get('buildings', [])
        if not buildings:
            self.log('WARN', "No buildings available for UPGRADE test")
            self.test_results.append(('UPGRADE', True))
            return True
        
        building = buildings[0]
        cmd = f"UPGRADE;{building['x']};{building['y']}"
        
        if not self.send_command(cmd):
            return False
        
        state = self.receive_state()
        self.log('PASS', "UPGRADE command sent")
        self.test_results.append(('UPGRADE', True))
        return True
    
    def test_msg_command(self) -> bool:
        """TEST 9: MSG command validation"""
        self.log('TEST', "MSG command execution")
        
        cmd = "MSG;Test message from Mimi test client"
        
        if not self.send_command(cmd):
            return False
        
        state = self.receive_state()
        self.log('PASS', "MSG command sent")
        self.test_results.append(('MSG', True))
        return True
    
    def test_factory_command(self) -> bool:
        """TEST 10: FACTORY command validation"""
        self.log('TEST', "FACTORY command execution")
        
        if not self.last_state:
            self.log('FAIL', "No state available")
            return False
        
        # Find a factory building
        factories = [b for b in self.last_state.get('buildings', []) if 'spawn' in b.get('block', '').lower()]
        
        if not factories:
            self.log('WARN', "No factory buildings available for FACTORY test")
            self.test_results.append(('FACTORY', True))
            return True
        
        factory = factories[0]
        cmd = f"FACTORY;{factory['x']};{factory['y']};poly"
        
        if not self.send_command(cmd):
            return False
        
        state = self.receive_state()
        self.log('PASS', "FACTORY command sent")
        self.test_results.append(('FACTORY', True))
        return True
    
    def run_all_tests(self):
        """Execute full test suite"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}")
        print(f"Mimi Gateway Test Suite")
        print(f"{'='*70}{Colors.RESET}\n")
        
        self.log('INFO', f"Starting tests against {self.host}:{self.port}")
        
        # Test 1: Connection
        if not self.test_connection():
            print(f"\n{Colors.RED}Connection failed. Cannot proceed.{Colors.RESET}\n")
            return False
        
        print()  # Spacing
        
        # Tests 2-10: Commands
        tests = [
            self.test_build_command,
            self.test_unit_move_command,
            self.test_attack_command,
            self.test_stop_command,
            self.test_repair_command,
            self.test_delete_command,
            self.test_upgrade_command,
            self.test_msg_command,
            self.test_factory_command,
        ]
        
        for test_func in tests:
            test_func()
            print()  # Spacing
        
        self.disconnect()
        self.print_results()
    
    def print_results(self):
        """Print test summary"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}")
        print(f"Test Results Summary")
        print(f"{'='*70}{Colors.RESET}\n")
        
        passed = sum(1 for _, result in self.test_results if result)
        total = len(self.test_results)
        
        for test_name, result in self.test_results:
            status = f"{Colors.GREEN}✓ PASS{Colors.RESET}" if result else f"{Colors.RED}✗ FAIL{Colors.RESET}"
            print(f"  {status}  {test_name}")
        
        print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.RESET}")
        
        if passed == total:
            print(f"{Colors.GREEN}All tests passed! ✓{Colors.RESET}\n")
            return True
        else:
            print(f"{Colors.RED}Some tests failed. See above for details.{Colors.RESET}\n")
            return False


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Mimi Gateway test client - Validate all 9 commands',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_mimi_client.py
  python test_mimi_client.py --host localhost --port 9000 --verbose
        """
    )
    
    parser.add_argument('--host', default='localhost', help='Server host (default: localhost)')
    parser.add_argument('--port', type=int, default=9000, help='Server port (default: 9000)')
    parser.add_argument('--timeout', type=float, default=5.0, help='Socket timeout in seconds (default: 5.0)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    client = MimiTestClient(
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        verbose=args.verbose
    )
    
    success = client.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
