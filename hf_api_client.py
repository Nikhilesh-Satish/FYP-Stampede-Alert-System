"""
Hugging Face API Client with round-robin load distribution
"""
import requests
import json
from typing import Dict, List, Optional
import os
import itertools

# List of HF Space URLs (these will be converted to API URLs)
HF_SPACES = [
    "https://huggingface.co/spaces/Niks1904/stampede-alert-system-backend-fyp",
    "https://huggingface.co/spaces/Niks1904/stampede-alert-system-backend-fyp-2",
    "https://huggingface.co/spaces/Niks1904/stampede-alert-system-backend-fyp-3",
    "https://huggingface.co/spaces/Niks1904/stampede-alert-system-backend-fyp-4",
    "https://huggingface.co/spaces/Niks1904/stampede-alert-system-backend-fyp-5",
    "https://huggingface.co/spaces/Niks1904/stampede-alert-system-backend-fyp-6",
    "https://huggingface.co/spaces/Niks001904/stampede-alert-system-backend-fyp-7",
    "https://huggingface.co/spaces/Niks001904/stampede-alert-system-backend-fyp-8",
    "https://huggingface.co/spaces/Niks001904/stampede-alert-system-backend-fyp-9",
    "https://huggingface.co/spaces/Niks001904/stampede-alert-system-backend-fyp-10",
    "https://huggingface.co/spaces/Niks001904/stampede-alert-system-backend-fyp-11",
    "https://huggingface.co/spaces/Niks001904/stampede-alert-system-backend-fyp-12",
]

# Allow quick local testing by setting HF_TEST_API to a single URL, e.g.
# export HF_TEST_API=http://localhost:8001
if os.environ.get("HF_TEST_API"):
    HF_SPACES = [os.environ.get("HF_TEST_API")]


def space_url_to_api_url(space_url: Optional[str]) -> str:
    """Convert HF space URL to API endpoint

    Accepts Optional[str] for better type-safety when values may come
    from environment variables. Raises ValueError if input is missing.
    """
    if not space_url:
        raise ValueError("space_url is required")
    # Extract owner and space name from URL
    # https://huggingface.co/spaces/Niks1904/stampede-alert-system-backend-fyp
    # -> https://Niks1904-stampede-alert-system-backend-fyp.hf.space
    parts = space_url.split("/spaces/")[-1].split("/")
    owner = parts[0]
    space_name = parts[1]
    return f"https://{owner}-{space_name}.hf.space"


class HFAPIClient:
    """Round-robin client for HF Space APIs"""
    
    def __init__(self, space_urls: Optional[List[str]] = None):
        self.space_urls = space_urls or HF_SPACES
        self.api_urls = [space_url_to_api_url(url) for url in self.space_urls]
        # Map api_url -> camera_id (or None when free)
        self.api_to_camera: Dict[str, Optional[str]] = {u: None for u in self.api_urls}
        self.camera_to_api: Dict[str, str] = {}  # Track which API each camera uses
        
    def get_next_api(self) -> str:
        """Get next API in round-robin fashion (fallback)"""
        # fallback: return the first api
        return self.api_urls[0]
    
    def assign_camera_api(self, camera_id: str) -> str:
        """Assign a camera to an API and return the API URL"""
        # If already assigned, return existing mapping
        if camera_id in self.camera_to_api:
            return self.camera_to_api[camera_id]

        # Find a free API (api_to_camera value is None)
        for api_url, owner in self.api_to_camera.items():
            if owner is None:
                self.api_to_camera[api_url] = camera_id
                self.camera_to_api[camera_id] = api_url
                return api_url

        # No free API available
        return ""  # caller should treat empty string as error
    
    def get_camera_api(self, camera_id: str) -> Optional[str]:
        """Get the API assigned to a camera"""
        return self.camera_to_api.get(camera_id)
    
    def call_api(
        self, 
        camera_id: str, 
        endpoint: str, 
        method: str = "GET", 
        data: Optional[Dict] = None,
        timeout: int = 30
    ) -> Dict:
        """
        Call the HF API assigned to this camera
        Example: call_api("camera1", "/cameras/counts/camera1", "GET")
        """
        api_url = self.get_camera_api(camera_id)
        if not api_url:
            raise ValueError(f"Camera {camera_id} not assigned to any API")
        
        full_url = api_url + endpoint
        
        try:
            if method == "GET":
                resp = requests.get(full_url, timeout=timeout)
            elif method == "POST":
                resp = requests.post(full_url, json=data, timeout=timeout)
            elif method == "DELETE":
                resp = requests.delete(full_url, timeout=timeout)
            else:
                raise ValueError(f"Unsupported method: {method}")

            # Provide richer debug information on failure
            try:
                resp.raise_for_status()
            except requests.exceptions.RequestException as e:
                body = None
                try:
                    body = resp.text
                except Exception:
                    body = '<unreadable body>'
                print(f"API call failed (status {resp.status_code}): {full_url} - {str(e)}\nResponse body: {body}")
                return {"error": f"status={resp.status_code}, body={body}"}

            # Try to parse JSON, but if not JSON return raw text for debugging
            try:
                return resp.json()
            except ValueError:
                text = resp.text
                print(f"API call returned non-JSON response: {full_url} - {text}")
                return {"error": f"non-json response: {text}"}

        except requests.exceptions.RequestException as e:
            print(f"API call failed: {full_url} - {str(e)}")
            return {"error": str(e)}
    
    def add_camera(
        self,
        camera_id: str,
        stream_path: str,
        count_axis: str = "x",
        in_direction: str = "positive",
        shot_type: str = "ground",
        processing_preset: str = "balanced"
    ) -> Dict:
        """Add camera to assigned API"""
        api_url = self.assign_camera_api(camera_id)

        # Quick health-check to ensure the assigned API is reachable
        health_url = api_url + "/health"
        try:
            hresp = requests.get(health_url, timeout=10)
            hresp.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Assigned HF API not reachable for camera {camera_id}: {api_url} - {e}")
            return {"error": f"api_unreachable: {e}"}

        endpoint = "/cameras/add"
        full_url = api_url + endpoint

        data = {
            "name": camera_id,
            "stream_path": stream_path,
            "count_axis": count_axis,
            "in_direction": in_direction,
            "shot_type": shot_type,
            "processing_preset": processing_preset,
            "counting_enabled": True,
            "density_enabled": False,
        }
        
        try:
            resp = requests.post(full_url, json=data, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to add camera {camera_id} to {api_url}: {str(e)}")
            return {"error": str(e)}
    
    def get_camera_counts(self, camera_id: str) -> Dict:
        """Get counts for a camera from its assigned API"""
        return self.call_api(camera_id, f"/cameras/counts/{camera_id}", "GET")
    
    def remove_camera(self, camera_id: str) -> Dict:
        """Remove camera from its assigned API"""
        resp = self.call_api(camera_id, f"/cameras/{camera_id}", "DELETE")
        if "error" not in resp:
            api = self.camera_to_api.pop(camera_id, None)
            if api:
                # mark api as free
                self.api_to_camera[api] = None
        return resp


# Global client instance
hf_client = HFAPIClient()
