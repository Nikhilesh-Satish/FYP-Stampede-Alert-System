"""
Camera Worker using HF Space APIs instead of local model
Fetches both counts AND annotated video frames
"""
import cv2
import time
import threading
import requests
from io import BytesIO
from hf_api_client import hf_client


class CameraWorkerHF:
    """Process camera frames via HF API"""
    
    def __init__(
        self,
        camera_id,
        video_path,
        count_axis="x",
        in_direction="positive",
        shot_type="ground",
        processing_preset="balanced",
        counting_enabled=True,
        density_enabled=False,
        monitored_area_sqm=None,
        shared_dict=None,
        shared_totals=None,
        shared_count_ts=None,
        shared_frames=None,
        shared_density_stats=None,
        running_status=None,
        lock=None,
    ):
        self.camera_id = camera_id
        self.video_path = video_path
        self.count_axis = count_axis
        self.in_direction = in_direction
        self.shot_type = shot_type
        self.processing_preset = processing_preset
        self.counting_enabled = counting_enabled
        self.density_enabled = density_enabled
        self.monitored_area_sqm = monitored_area_sqm
        
        # Shared state (for aggregation on central server)
        self.shared_dict = shared_dict or {}
        self.shared_totals = shared_totals or {}
        self.shared_count_ts = shared_count_ts or {}
        self.shared_frames = shared_frames or {}
        self.shared_density_stats = shared_density_stats or {}
        self.running_status = running_status or {}
        self.lock = lock or threading.Lock()
        
        # Local state
        self.net_count = 0
        self.interval_in = 0
        self.interval_out = 0
        self.process_failed = False
    
    def get_hf_api_url(self):
        """Get the HF API URL assigned to this camera"""
        return hf_client.get_camera_api(self.camera_id)
    
    def fetch_frame(self):
        """Fetch annotated frame from HF API"""
        try:
            api_url = self.get_hf_api_url()
            if not api_url:
                return None
            
            # Fetch frame from HF API stream endpoint
            stream_url = f"{api_url}/cameras/stream/{self.camera_id}?overlay=true"
            
            # Get one frame from MJPEG stream
            resp = requests.get(stream_url, timeout=10, stream=True)
            resp.raise_for_status()

            # Read raw content and extract JPEG via JPEG start/end markers
            content = resp.content

            # Look for JPEG start/end markers for robustness
            start = content.find(b'\xff\xd8')
            end = content.find(b'\xff\xd9', start + 2) if start != -1 else -1
            if start != -1 and end != -1:
                jpg = content[start:end + 2]
                return jpg

            # Fallback: try to parse MJPEG boundary
            boundary = b'--frame'
            if boundary in content:
                parts = content.split(boundary)
                if len(parts) > 1:
                    frame_data = parts[1]
                    if b'\r\n\r\n' in frame_data:
                        jpg_data = frame_data.split(b'\r\n\r\n', 1)[1]
                        # attempt to find end marker
                        end2 = jpg_data.find(b'\xff\xd9')
                        if end2 != -1:
                            return jpg_data[: end2 + 2]

            return None
        
        except Exception as e:
            print(f"[{self.camera_id}] Error fetching frame: {e}")
            return None
    
    def run(self):
        """Main worker loop"""
        print(f"[{self.camera_id}] Starting camera worker via HF API...")
        
        try:
            # Register camera on assigned HF API
            result = hf_client.add_camera(
                self.camera_id,
                self.video_path,
                self.count_axis,
                self.in_direction,
                self.shot_type,
                self.processing_preset,
            )
            # Log registration response for debugging
            print(f"[{self.camera_id}] add_camera response: {result}")
            if isinstance(result, dict) and "error" in result:
                print(f"[{self.camera_id}] Failed to register: {result.get('error')}")
                self.process_failed = True
                return

            print(f"[{self.camera_id}] Registered on HF API successfully")
            print(f"[{self.camera_id}] Using API: {self.get_hf_api_url()}")
            
            # Poll for counts and frames
            last_poll = 0
            last_frame_fetch = 0
            while self.running_status.get(self.camera_id, True):
                now = time.time()
                
                # Poll API counts every 2 seconds
                if now - last_poll > 2:
                    try:
                        counts = hf_client.get_camera_counts(self.camera_id)

                        # If API returned error, log it so we can debug
                        if isinstance(counts, dict) and "error" in counts:
                            print(f"[{self.camera_id}] HF counts error: {counts.get('error')}")
                        else:
                            # Update shared state
                            with self.lock:
                                self.shared_dict[self.camera_id] = counts.get("count", 0)
                                self.shared_totals[self.camera_id] = {
                                    "in_count": counts.get("in_count", 0),
                                    "out_count": counts.get("out_count", 0),
                                }
                                self.shared_count_ts[self.camera_id] = int(now * 1000)

                            print(f"[{self.camera_id}] Count: {counts.get('count', 0)} " +
                                  f"(in: {counts.get('in_count', 0)}, out: {counts.get('out_count', 0)})")

                    except Exception as e:
                        print(f"[{self.camera_id}] Error polling counts: {e}")
                    
                    last_poll = now
                
                # Fetch frame every 1 second (for display)
                if now - last_frame_fetch > 1:
                    try:
                        jpg_bytes = self.fetch_frame()
                        if jpg_bytes:
                            with self.lock:
                                self.shared_frames[self.camera_id] = {
                                    "raw": jpg_bytes,
                                    "overlay": jpg_bytes,
                                    "timestamp": int(now * 1000),
                                }
                            print(f"[{self.camera_id}] Frame updated ({len(jpg_bytes)} bytes)")
                    
                    except Exception as e:
                        print(f"[{self.camera_id}] Error fetching frame: {e}")
                    
                    last_frame_fetch = now
                
                time.sleep(0.5)
        
        except Exception as e:
            print(f"[{self.camera_id}] Worker error: {e}")
            self.process_failed = True
        
        finally:
            # Cleanup
            try:
                hf_client.remove_camera(self.camera_id)
                print(f"[{self.camera_id}] Cleaned up")
            except:
                pass
    
    def reset_local_count(self):
        """Reset counts for this camera"""
        self.net_count = 0
        self.interval_in = 0
        self.interval_out = 0
        
        with self.lock:
            self.shared_dict[self.camera_id] = 0
            self.shared_totals[self.camera_id] = {"in_count": 0, "out_count": 0}
            self.shared_count_ts[self.camera_id] = int(time.time() * 1000)
