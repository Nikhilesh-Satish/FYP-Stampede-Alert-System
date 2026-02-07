import threading
import requests
import time

CAM_ID = "gate_1"

SERVER_URL = "http://127.0.0.1:5000/update_count"

# Interval counters
interval_in = 0
interval_out = 0
lock = threading.Lock()


def detection_thread():
    """
    Simulates YOLO detection + tracking.
    Replace this logic with real in/out counting.
    """
    global interval_in, interval_out

    while True:
        # Example: detected 2 entering, 1 exiting
        people_in = 2
        people_out = 1

        with lock:
            interval_in += people_in
            interval_out += people_out

        time.sleep(1)


def sender_thread():
    """
    Sends net count every 10 seconds to central server.
    """
    global interval_in, interval_out

    SEND_INTERVAL = 10

    while True:
        time.sleep(SEND_INTERVAL)

        with lock:
            net_count = interval_in - interval_out
            interval_in = 0
            interval_out = 0

        payload = {
            "camera_id": CAM_ID,
            "net_count": net_count,
            "timestamp": int(time.time())
        }

        try:
            requests.post(SERVER_URL, json=payload, timeout=2)
            print("✅ Sent:", payload)

        except Exception as e:
            print("❌ Send failed:", e)


def main():
    t1 = threading.Thread(target=detection_thread, daemon=True)
    t2 = threading.Thread(target=sender_thread, daemon=True)

    t1.start()
    t2.start()

    t1.join()
    t2.join()


if __name__ == "__main__":
    main()
