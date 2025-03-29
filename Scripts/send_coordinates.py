#!/usr/bin/env python3
import requests
import time
import zmq
from pymavlink import mavutil
import json
import signal
import sys
from threading import Thread, Lock, Event
from queue import Queue
import subprocess
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

# Configuration
CONFIG = {
    # Telemetry Sources
    'pixhawk': {
        'port': '/dev/ttyUSB0',
        'baud': 57600,
        'priority': 1
    },
    'radio': {
        'port': 'udpin:0.0.0.0:14550',
        'priority': 0
    },
    
    # Data Distribution
    'zmq': {
        'pub_port': 5556,  # For telemetry
        'router_port': 5557  # For command/response
    },
    
    # External Services
    'http_endpoints': [
        'http://127.0.0.1:5000/update_data',
        'http://127.0.0.1:5002/update_data'
    ],
    
    # System Control
    'update_rate_hz': 10,
    'max_stale_seconds': 2.0,
    'log_file': 'telemetry.log',
    
    # External Processor
    'processor_script': 'data_processor.py',
    'processor_restart_interval': 3600  # Restart every hour
}

# Global control
running = True
processor_process = None

@dataclass
class TelemetrySource:
    name: str
    connection: Any
    priority: int
    last_update: float = 0.0
    data: Dict = None
    lock: Lock = Lock()

def signal_handler(sig, frame):
    global running
    print("\nShutting down gracefully...")
    running = False
    if processor_process:
        processor_process.terminate()
    sys.exit(0)

class DataProcessorManager:
    def __init__(self):
        self.process = None
        self.last_restart = 0
        self.lock = Lock()
    
    def start(self):
        with self.lock:
            if self.process and self.process.poll() is None:
                return  # Already running
            
            try:
                self.process = subprocess.Popen(
                    ['python3', CONFIG['processor_script']],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setpgrp
                )
                self.last_restart = time.time()
                print(f"Started data processor (PID: {self.process.pid})")
            except Exception as e:
                print(f"Failed to start processor: {e}")
    
    def check_restart(self):
        if time.time() - self.last_restart > CONFIG['processor_restart_interval']:
            print("Scheduled processor restart...")
            self.restart()
    
    def restart(self):
        with self.lock:
            if self.process:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=5)
                except:
                    self.process.kill()
            self.start()
    
    def stop(self):
        with self.lock:
            if self.process:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=3)
                except:
                    self.process.kill()

class CommandHandler:
    def __init__(self, context):
        self.router = context.socket(zmq.ROUTER)
        self.router.bind(f"tcp://*:{CONFIG['zmq']['router_port']}")
        self.poller = zmq.Poller()
        self.poller.register(self.router, zmq.POLLIN)
        self.commands = {
            'status': self.handle_status,
            'restart': self.handle_restart,
            'config': self.handle_config
        }
    
    def handle_commands(self):
        socks = dict(self.poller.poll(100))  # 100ms timeout
        
        if self.router in socks:
            identity, _, command = self.router.recv_multipart()
            try:
                cmd = json.loads(command.decode())
                handler = self.commands.get(cmd.get('command'))
                if handler:
                    response = handler(cmd)
                    self.router.send_multipart([identity, b'', json.dumps(response).encode()])
            except Exception as e:
                response = {'error': str(e)}
                self.router.send_multipart([identity, b'', json.dumps(response).encode()])
    
    def handle_status(self, cmd):
        return {'status': 'running', 'timestamp': time.time()}
    
    def handle_restart(self, cmd):
        global processor_process
        if processor_process:
            processor_process.terminate()
            processor_process = None
        return {'restarted': True}
    
    def handle_config(self, cmd):
        return {'config': CONFIG}

class TelemetryAggregator:
    def __init__(self):
        self.context = zmq.Context()
        self.zmq_publisher = self.context.socket(zmq.PUB)
        self.zmq_publisher.bind(f"tcp://*:{CONFIG['zmq']['pub_port']}")
        self.sources = []
        self.processor_manager = DataProcessorManager()
        self.command_handler = CommandHandler(self.context)
        self.setup_logging()
        self.connect_sources()
    
    def setup_logging(self):
        import logging
        logging.basicConfig(
            filename=CONFIG['log_file'],
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger()
    
    def connect_sources(self):
        """Initialize all telemetry sources"""
        # Pixhawk direct connection
        try:
            pixhawk_conn = mavutil.mavlink_connection(
                CONFIG['pixhawk']['port'],
                baud=CONFIG['pixhawk']['baud']
            )
            pixhawk_conn.wait_heartbeat()
            self.sources.append(TelemetrySource(
                name='pixhawk',
                connection=pixhawk_conn,
                priority=CONFIG['pixhawk']['priority']
            ))
            self.logger.info(f"Connected to Pixhawk on {CONFIG['pixhawk']['port']}")
        except Exception as e:
            self.logger.error(f"Pixhawk connection failed: {e}")

        # Telemetry radio connection
        try:
            radio_conn = mavutil.mavlink_connection(CONFIG['radio']['port'])
            self.sources.append(TelemetrySource(
                name='radio',
                connection=radio_conn,
                priority=CONFIG['radio']['priority']
            ))
            self.logger.info(f"Listening for radio telemetry on {CONFIG['radio']['port']}")
        except Exception as e:
            self.logger.error(f"Radio connection failed: {e}")
    
    def start_source_readers(self):
        """Start a thread for each telemetry source"""
        for source in self.sources:
            Thread(target=self.read_source, args=(source,), daemon=True).start()
    
    def read_source(self, source: TelemetrySource):
        """Continuously read data from a single source"""
        while running:
            try:
                msg = source.connection.recv_match(blocking=True, timeout=1)
                if msg:
                    with source.lock:
                        self.process_message(source, msg)
                        source.last_update = time.time()
            except Exception as e:
                self.logger.error(f"Error reading from {source.name}: {e}")
                time.sleep(1)
    
    def process_message(self, source: TelemetrySource, msg):
        """Parse MAVLink message and update source data"""
        data = {}
        
        if msg.get_type() == 'GLOBAL_POSITION_INT':
            data.update({
                'lat': msg.lat / 1e7,
                'lng': msg.lon / 1e7,
                'altitude': msg.relative_alt / 1000.0,
                'vertical_speed': msg.vz / 100.0,
                'horizontal_speed': ((msg.vx**2 + msg.vy**2)**0.5) / 100.0
            })
        
        elif msg.get_type() == 'ATTITUDE':
            data.update({
                'yaw': msg.yaw * 180.0 / 3.14159,
                'pitch': msg.pitch * 180.0 / 3.14159,
                'roll': msg.roll * 180.0 / 3.14159
            })
        
        elif msg.get_type() == 'SYS_STATUS':
            voltage = msg.voltage_battery / 1000.0
            data['battery'] = min((voltage - 10.0) / (16.8 - 10.0) * 100, 100)
        
        # Merge new data with existing source data
        if data:
            source.data = {**(source.data or {}), **data, 'source': source.name}
    
    def get_best_telemetry(self) -> Optional[Dict]:
        """Get merged telemetry from all sources with priority"""
        merged_data = {}
        now = time.time()
        
        # Sort sources by priority (highest first)
        active_sources = sorted(
            [s for s in self.sources if s.data and (now - s.last_update) < CONFIG['max_stale_seconds']],
            key=lambda x: x.priority,
            reverse=True
        )

        # Merge data from all active sources, higher priority overwrites
        for source in active_sources:
            with source.lock:
                if source.data:
                    merged_data.update(source.data)
        
        if not merged_data:
            return None
            
        # Add system timestamp
        merged_data['timestamp'] = now
        
        return merged_data
    
    def send_to_http(self, endpoint: str, data: Dict):
        """Send data to HTTP endpoint"""
        try:
            response = requests.post(endpoint, json=data, timeout=0.5)
            if response.status_code != 200:
                self.logger.warning(f"HTTP {endpoint} returned {response.status_code}")
            return response.ok
        except Exception as e:
            self.logger.warning(f"HTTP send to {endpoint} failed: {e}")
            return False
    
    def publish_telemetry(self, data: Dict):
        """Publish data to all outputs"""
        # Publish via ZMQ
        self.zmq_publisher.send_json(data)
        
        # Send to HTTP endpoints in parallel
        for endpoint in CONFIG['http_endpoints']:
            Thread(target=self.send_to_http, args=(endpoint, data)).start()
        
        # Log the sent data
        self.logger.debug(f"Published: {json.dumps(data)}")
        print(f"\rTelemetry: {json.dumps({k: v for k, v in data.items() if k != 'source'})}", end='')
    
    def run(self):
        """Main processing loop"""
        self.start_source_readers()
        self.processor_manager.start()
        sleep_time = 1.0 / CONFIG['update_rate_hz']
        
        while running:
            try:
                # Handle incoming commands
                self.command_handler.handle_commands()
                
                # Check processor status
                self.processor_manager.check_restart()
                
                # Process telemetry
                data = self.get_best_telemetry()
                if data:
                    self.publish_telemetry(data)
                
                time.sleep(sleep_time)
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Main loop error: {e}")
                time.sleep(1)
    
    def cleanup(self):
        """Clean up resources"""
        self.processor_manager.stop()
        self.zmq_publisher.close()
        self.command_handler.router.close()
        self.context.term()
        self.logger.info("Telemetry aggregator shutdown complete")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    aggregator = TelemetryAggregator()
    try:
        aggregator.run()
    finally:
        aggregator.cleanup()
