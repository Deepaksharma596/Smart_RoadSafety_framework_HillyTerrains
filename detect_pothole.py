import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, Label, Button
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

POTHOLE_WEIGHTS_PATH = r'D:/road_safety/models/pothole_yolov8.pt'
YOLO_INPUT_SIZE = 2048
CONF_THRESHOLD = 0.4
GOOGLE_API_KEY = 'AIzaSyBDO_XrYnJv5jVh7tc87E0YKbH7JU3pY7w'

# Email configuration
AUTH_EMAIL = 'deepakkumar.sharma2022@vitstudent.ac.in'
AUTH_PASSWORD = 'djqnmfncobixrtyn'
RECIPIENT_EMAIL = 'shilpa.3569@gmail.com'
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587

def test_connectivity():
    """Test Google API and SMTP connectivity."""
    logging.info("Testing Google Maps API connectivity...")
    try:
        url = f"https://maps.googleapis.com/maps/api/geocode/json?address=test&key={GOOGLE_API_KEY}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            logging.info("Google Maps API is accessible")
        else:
            logging.error(f"Google Maps API failed: Status {response.status_code}")
    except Exception as e:
        logging.error(f"Google Maps API connectivity test failed: {e}")

    logging.info("Testing SMTP connectivity...")
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=5)
        server.starttls()
        server.login(AUTH_EMAIL, AUTH_PASSWORD)
        server.quit()
        logging.info("SMTP server login successful")
    except Exception as e:
        logging.error(f"SMTP connectivity test failed: {e}")

    logging.info("Testing geolocation API connectivity...")
    try:
        response = requests.get("http://ip-api.com/json/", timeout=5)
        if response.status_code == 200:
            logging.info("Geolocation API (ip-api.com) is accessible")
        else:
            logging.error(f"Geolocation API failed: Status {response.status_code}")
    except Exception as e:
        logging.error(f"Geolocation API connectivity test failed: {e}")

def get_live_location():
    """Fetch live location using ip-api.com."""
    try:
        response = requests.get("http://ip-api.com/json/", timeout=5)
        data = response.json()
        if data['status'] == 'success':
            lat = data['lat']
            lng = data['lon']
            formatted_address = f"{data.get('city', 'Unknown')}, {data.get('regionName', 'Unknown')}, {data.get('country', 'Unknown')}"
            logging.info(f"Live location fetched: {formatted_address}, Lat: {lat}, Lng: {lng}")
            return lat, lng, formatted_address
        else:
            logging.error(f"Geolocation failed: {data.get('message', 'Unknown error')}")
            return None, None, "Unknown"
    except Exception as e:
        logging.error(f"Geolocation request failed: {e}")
        return None, None, "Unknown"

def send_alert_email(num_detections, lat, lng, formatted_address):
    """Send email alert with pothole detection and live location coordinates."""
    msg = MIMEMultipart()
    msg['From'] = AUTH_EMAIL
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = 'Road Safety Alert: Pothole Detected'
    
    coordinates = f"Latitude: {lat:.6f}, Longitude: {lng:.6f}" if lat and lng else "Coordinates: Unknown"
    map_url = f"https://www.google.com/maps?q={lat},{lng}" if lat and lng else "Map not available"
    body = f"""
    Potholes have been detected on the road.

    Details:
    - Type: Pothole
    - Number of Potholes: {num_detections}
    - Location: {formatted_address}
    - {coordinates}
    - Map Link: {map_url}
    - Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    Please take appropriate action.
    """
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=5)
        server.starttls()
        server.login(AUTH_EMAIL, AUTH_PASSWORD)
        text = msg.as_string()
        server.sendmail(AUTH_EMAIL, RECIPIENT_EMAIL, text)
        server.quit()
        logging.info("Alert email sent successfully for pothole")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")

def select_video_file_gui():
    """GUI to select video file."""
    root = tk.Tk()
    root.title("Pothole Detection")
    root.geometry("400x200")
    video_path = {'value': None}

    def open_video():
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv"), ("All Files", "*.*")]
        )
        if path:
            video_path['value'] = path
            root.destroy()

    tk.Label(root, text="Select Video for Pothole Detection", font=("Arial", 16)).pack(pady=20)
    btn_video = tk.Button(root, text="Open Video File", command=open_video, font=("Arial", 12), width=20)
    btn_video.pack(pady=20)
    root.mainloop()
    return video_path['value']

def rescale_frame(frame, width=1280):
    """Rescale video frame to specified width while maintaining aspect ratio."""
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (int(w * scale), int(h * scale)))

def draw_detection_boxes(frame, detections, class_names, color=(0, 165, 255)):
    """Draw bounding boxes and labels for detected objects."""
    font_scale = min(frame.shape[0], frame.shape[1]) / 1000
    for det in detections:
        x1, y1, x2, y2 = map(int, det['box'])
        cls_id = det['class_id']
        if cls_id >= len(class_names):
            continue
        conf = det['confidence']
        label = f"{class_names[cls_id]} ({conf:.2f})"
        thickness = 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * font_scale, 2)
        cv2.rectangle(frame, (x1, y1 - int(30 * font_scale)), (x1 + text_size[0] + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - int(10 * font_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * font_scale, (0, 0, 0), 2)

def process_video(video_path, model, class_names):
    """Process video for pothole detection, display results, and send alerts."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video: {video_path}")
        return
    win_name = "Pothole Detection"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Get live location
    lat, lng, formatted_address = get_live_location()
    coordinates = f"Lat: {lat:.6f}, Lng: {lng:.6f}" if lat and lng else "Coordinates: Unknown"
    
    alert_message = ""
    email_sent = False
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 25
    frame_duration = 1.0 / fps
    
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame = rescale_frame(frame, width=800)
        resized = cv2.resize(frame[..., ::-1], (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))
        scale_x, scale_y = frame.shape[1] / YOLO_INPUT_SIZE, frame.shape[0] / YOLO_INPUT_SIZE
        results = model(resized, conf=CONF_THRESHOLD)
        detections = []
        detected_classes = set()
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf.item()
                cls = int(box.cls.item())
                class_name = class_names.get(cls, "Unknown")
                detected_classes.add(class_name)
                if class_name.strip().lower() == 'potholes' and conf >= CONF_THRESHOLD:
                    x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
                    detections.append({'box': [x1, y1, x2, y2], 'confidence': conf, 'class_id': cls})
        logging.info(f"Detected classes this frame (pothole): {detected_classes}")
        
        if detections:
            alert_message = "Pothole detected, slow down!"
            if not email_sent:
                send_alert_email(len(detections), lat, lng, formatted_address)
                email_sent = True
        else:
            alert_message = ""
            email_sent = False
        
        video_disp = frame.copy()
        draw_detection_boxes(video_disp, detections, list(class_names.values()), color=(0, 165, 255))
        
        panel_w = 400
        panel_h = video_disp.shape[0]
        data_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8) + 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 60
        cv2.putText(data_panel, f"Potholes: {len(detections)}", (30, y), font, 1.0, (0,200,0) if len(detections)<=8 else (0,0,255), 2)
        y += 40
        cv2.putText(data_panel, f"Time: {datetime.now().strftime('%H:%M:%S')}", (30, y), font, 0.7, (200,200,200), 1)
        y += 30
        cv2.putText(data_panel, f"Location: {formatted_address}", (30, y), font, 0.7, (200,200,200), 1)
        y += 30
        cv2.putText(data_panel, coordinates, (30, y), font, 0.7, (200,200,200), 1)
        
        if alert_message:
            cv2.putText(video_disp, alert_message, (30, 50), font, 1.2, (0,0,255), 3, cv2.LINE_AA)
        
        combined = np.hstack([video_disp, data_panel])
        cv2.imshow(win_name, combined)
        key = cv2.waitKey(1) & 0xFF
        elapsed = time.time() - start_time
        sleep_time = frame_duration - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main function to run pothole detection."""
    test_connectivity()
    model = YOLO(POTHOLE_WEIGHTS_PATH)
    class_names = model.names if hasattr(model, 'names') else {}
    video_path = select_video_file_gui()
    if video_path:
        process_video(video_path, model, class_names)
    else:
        logging.info("No video selected. Please select a video to proceed.")

if __name__ == "__main__":
    main()