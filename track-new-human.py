import cv2
import numpy as np
import time
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

print("OpenCV version:", cv2.__version__)

# Initialize InsightFace
try:
    # Create a FaceAnalysis app with specified models
    face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
    # Initialize with default parameters or specify detection size
    face_app.prepare(ctx_id=0, det_size=(320, 320))
    print("InsightFace initialized successfully")
    use_insightface = True
except Exception as e:
    print(f"Failed to initialize InsightFace: {e}")
    print("Falling back to OpenCV HOG detector")
    use_insightface = False

# Initialize HOG detector as fallback
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Try to enable OpenCL for Intel GPU acceleration
try:
    cv2.ocl.setUseOpenCL(True)
    if cv2.ocl.useOpenCL():
        print("Using OpenCL acceleration")
    else:
        print("OpenCL is not available, using CPU")
except:
    print("Failed to set OpenCL, using CPU")

# Create tracker based on OpenCV version
def create_tracker():
    # Check OpenCV version and create appropriate tracker
    major_ver, minor_ver, _ = cv2.__version__.split('.')
    
    try:
        if int(major_ver) < 4:
            tracker = cv2.Tracker_create('KCF')  # OpenCV 3.x syntax
        else:
            # OpenCV 4.x syntax - try different methods
            try:
                tracker = cv2.TrackerKCF_create()  # Try direct method
            except:
                try:
                    tracker = cv2.legacy.TrackerKCF_create()  # Try legacy namespace
                except:
                    print("KCF tracker not available. Using basic tracking instead.")
                    return None  # Return None if no tracking available
    except:
        print("Tracker creation failed. Using detection only.")
        return None
        
    return tracker

# Video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

# Initialize tracking variables
tracker = create_tracker()
tracking_initialized = False
tracking_box = None

# Face tracking variables
face_embeddings = {}
next_face_id = 1
tracked_faces = {}

# Frame processing parameters
detect_interval = 3  # Run detector every 15 frames
frame_count = 0
last_detected_boxes = []

print("Starting human tracking with InsightFace...")
print("Press 'q' to quit")

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Make a copy for display
    display_frame = frame.copy()
    
    # Update tracker if initialized and tracker is available
    if tracking_initialized and tracker is not None:
        try:
            track_success, tracking_box = tracker.update(frame)
            
            if track_success:
                # Draw tracking box
                x, y, w, h = [int(v) for v in tracking_box]
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(display_frame, "Tracking", (x, y - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                # Lost tracking, reset
                tracking_initialized = False
                cv2.putText(display_frame, "Lost tracking", (10, 80), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        except:
            # Error in tracking, reset
            tracking_initialized = False
            print("Tracking error, resetting")
    
    # Run detector periodically or when tracking is lost
    if frame_count % detect_interval == 0 or not tracking_initialized:
        # Clear previously detected faces
        tracked_faces = {}
        
        # Detect using InsightFace if available
        if use_insightface:
            start_time = time.time()
            
            # Face detection and analysis with InsightFace
            faces = face_app.get(frame)
            
            end_time = time.time()
            
            # Store detected faces
            last_detected_boxes = []
            
            for i, face in enumerate(faces):
                # Get bounding box
                bbox = face.bbox.astype(int)
                x, y, x2, y2 = bbox
                w, h = x2 - x, y2 - y
                
                # Store face embedding for recognition
                face_id = None
                if hasattr(face, 'embedding') and face.embedding is not None:
                    # Check if this face matches any known face
                    for known_id, known_embedding in face_embeddings.items():
                        similarity = np.dot(face.embedding, known_embedding) / (
                                np.linalg.norm(face.embedding) * np.linalg.norm(known_embedding))
                        if similarity > 0.6:  # Similarity threshold
                            face_id = known_id
                            break
                    
                    # If no match, add as new face
                    if face_id is None:
                        face_id = next_face_id
                        face_embeddings[face_id] = face.embedding
                        next_face_id += 1
                else:
                    face_id = f"Unknown-{i}"
                
                # Store for display
                last_detected_boxes.append((x, y, w, h))
                tracked_faces[i] = {"bbox": (x, y, w, h), "id": face_id}
                
                # Draw face rectangle
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add labels - face ID, gender, age
                label = f"Face #{face_id}"
                if hasattr(face, 'gender') and face.gender is not None:
                    gender = 'M' if face.gender == 1 else 'F'
                    label += f" ({gender})"
                if hasattr(face, 'age') and face.age is not None:
                    label += f", Age: {int(face.age)}"
                
                cv2.putText(display_frame, label, (x, y - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # If no faces detected, fall back to HOG for general person detection
            if len(faces) == 0:
                # Resize for faster detection
                small_frame = cv2.resize(frame, (320, 240))
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                
                # Run HOG detector
                boxes, weights = hog.detectMultiScale(
                    gray, 
                    winStride=(8, 8),
                    padding=(4, 4), 
                    scale=1.05
                )
                
                if len(boxes) > 0:
                    # Scale boxes to original size
                    scale_x = frame.shape[1] / gray.shape[1]
                    scale_y = frame.shape[0] / gray.shape[0]
                    
                    for i, (x, y, w, h) in enumerate(boxes):
                        # Scale box to original size
                        x_orig = int(x * scale_x)
                        y_orig = int(y * scale_y)
                        w_orig = int(w * scale_x)
                        h_orig = int(h * scale_y)
                        
                        last_detected_boxes.append((x_orig, y_orig, w_orig, h_orig))
                        
                        # Draw detection rectangles
                        cv2.rectangle(display_frame, (x_orig, y_orig), 
                                    (x_orig + w_orig, y_orig + h_orig), (0, 100, 255), 2)
                        cv2.putText(display_frame, "Person", (x_orig, y_orig - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)
            
            print(f"InsightFace detection time: {(end_time - start_time):.3f}s, found {len(faces)} faces")
                
        else:
            # Fallback to HOG detector
            # Resize for faster detection
            small_frame = cv2.resize(frame, (320, 240))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            # Run detector
            start_time = time.time()
            boxes, weights = hog.detectMultiScale(
                gray, 
                winStride=(8, 8),
                padding=(4, 4), 
                scale=1.05
            )
            end_time = time.time()
            
            # Store detected boxes for display
            last_detected_boxes = []
            
            if len(boxes) > 0:
                # Scale boxes to original size
                scale_x = frame.shape[1] / gray.shape[1]
                scale_y = frame.shape[0] / gray.shape[0]
                
                for i, (x, y, w, h) in enumerate(boxes):
                    # Scale box to original size
                    x_orig = int(x * scale_x)
                    y_orig = int(y * scale_y)
                    w_orig = int(w * scale_x)
                    h_orig = int(h * scale_y)
                    
                    last_detected_boxes.append((x_orig, y_orig, w_orig, h_orig))
                    
                    # Draw detection rectangles
                    cv2.rectangle(display_frame, (x_orig, y_orig), 
                                (x_orig + w_orig, y_orig + h_orig), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Person", (x_orig, y_orig - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                print(f"HOG detection time: {(end_time - start_time):.3f}s, found {len(boxes)} people")
        
        # Set up tracking for first detected person if available
        if len(last_detected_boxes) > 0 and tracker is not None:
            # Use the first detection (typically the largest/closest)
            x, y, w, h = last_detected_boxes[0]
            
            try:
                # Re-initialize tracker
                tracker = create_tracker()
                if tracker is not None:
                    track_success = tracker.init(frame, (x, y, w, h))
                    if track_success:
                        tracking_initialized = True
                        tracking_box = (x, y, w, h)
            except:
                print("Failed to initialize tracker")
    
    # Display previously detected boxes if no new detections in this frame
    elif not tracking_initialized:
        if tracked_faces:
            # Display previously detected faces
            for face_idx, face_data in tracked_faces.items():
                x, y, w, h = face_data["bbox"]
                face_id = face_data["id"]
                
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_frame, f"Face #{face_id}", (x, y - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        elif len(last_detected_boxes) > 0:
            # Display previously detected people
            for (x, y, w, h) in last_detected_boxes:
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_frame, "Person", (x, y - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display status
    if use_insightface:
        status_text = "InsightFace: "
    else:
        status_text = "HOG Detector: "
        
    if tracking_initialized and tracker is not None:
        cv2.putText(display_frame, status_text + "Tracking", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(display_frame, status_text + "Detecting", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display people/face count
    if use_insightface:
        face_count = len(tracked_faces)
        cv2.putText(display_frame, f"Faces: {face_count}", (10, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        people_count = len(last_detected_boxes)
        cv2.putText(display_frame, f"People: {people_count}", (10, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Show frame
    cv2.imshow("Face Detection & Tracking", display_frame)
    
    # Increment frame counter
    frame_count += 1
    
    # Break loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Tracking stopped")