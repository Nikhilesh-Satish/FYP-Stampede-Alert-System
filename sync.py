import threading
import requests
import time
import queue

count_queue=queue.Queue()

net_count=0
count_lock=threading.Lock()

CAM_ID="your_camera_id"
SERVER_URL=" "


def detection_thread():
    while True:
        ## Model will count the crowd here and store results in people_in and people_out variables
        people_in=5
        people_out=3
        if people_in>0:
            count_queue.put(people_in)
        if people_out>0:
            count_queue.put(-people_out)
        time.sleep(2)


def sender_thread():
    global net_count
    SEND_INTERVAL=1
    while True:
        start=time.time()
        while True:
            try:
                change=count_queue.get_nowait()
            except queue.Empty:
                break
            with count_lock:
                net_count+=change
            if time.time()-start>=SEND_INTERVAL:
                with count_lock:
                    to_send=net_count
                    net_count=0
                if to_send!=0:
                    try:
                        payload = {
                            "camera_id": CAM_ID,
                            "count": to_send
                        }
                        requests.post(SERVER_URL, json=payload, timeout=0.5)
                        print(f"Sent delta: {to_send}")
                    except Exception as e:
                        print("Send failed, restoring:", e)
                        # restore if failed
                        with count_lock:
                            net_delta += to_send

                # sleep remainder of interval
                elapsed = time.time() - start
                time.sleep(max(0, SEND_INTERVAL - elapsed))


def main():
    t1 = threading.Thread(target=detection_thread, daemon=True)
    t2 = threading.Thread(target=sender_thread, daemon=True)

    t1.start()
    t2.start()

    t1.join()
    t2.join()


if __name__ == "__main__":
    main()
