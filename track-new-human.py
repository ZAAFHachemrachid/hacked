import cv2
import numpy as np
import time
import os
import pickle
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from PIL import Image, ImageTk
import threading
import insightface
from insightface.app import FaceAnalysis

print("OpenCV version:", cv2.__version__)

# Create directories for saving face data
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_data")
os.makedirs(DATA_DIR, exist_ok=True)
FACE_DB_PATH = os.path.join(DATA_DIR, "face_database.pkl")

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

# Load face database if exists
face_database = {}
if os.path.exists(FACE_DB_PATH):
    try:
        with open(FACE_DB_PATH, 'rb') as f:
            face_database = pickle.load(f)
        print(f"Loaded {len(face_database)} faces from database")
    except Exception as e:
        print(f"Error loading face database: {e}")

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

# Function to save face database
def save_face_database():
    try:
        with open(FACE_DB_PATH, 'wb') as f:
            pickle.dump(face_database, f)
        print(f"Saved {len(face_database)} faces to database")
    except Exception as e:
        print(f"Error saving face database: {e}")

# Function to recognize face
def recognize_face(face_embedding):
    if not face_database:
        return "Unknown", 0.0
    
    max_similarity = 0.0
    best_match_name = "Unknown"
    
    for name, embeddings in face_database.items():
        for embedding in embeddings:
            similarity = np.dot(face_embedding, embedding) / (
                np.linalg.norm(face_embedding) * np.linalg.norm(embedding))
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_name = name
    
    # If similarity is below threshold, consider it unknown
    if max_similarity < 0.6:
        return "Unknown", max_similarity
    
    return best_match_name, max_similarity

# Main application class
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("1200x700")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Set up the UI
        self.setup_ui()
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        
        # Initialize tracking variables
        self.tracker = create_tracker()
        self.tracking_initialized = False
        self.tracking_box = None
        
        # Face tracking variables
        self.tracked_faces = {}
        
        # Frame processing parameters
        self.detect_interval = 3  # Run detector every 3 frames
        self.frame_count = 0
        self.last_detected_boxes = []
        
        # Control flags
        self.running = True
        self.registering_face = False
        self.current_frame = None
        
        # Start video processing thread
        self.thread = threading.Thread(target=self.video_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def setup_ui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create left panel for video display
        self.video_frame = ttk.LabelFrame(main_frame, text="Video Feed")
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create video label
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create right panel for controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls")
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # Status display
        status_frame = ttk.LabelFrame(control_frame, text="Status")
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Initializing...")
        self.status_label.pack(padx=5, pady=5)
        
        self.face_count_label = ttk.Label(status_frame, text="Faces: 0")
        self.face_count_label.pack(padx=5, pady=5)
        
        # Control buttons
        buttons_frame = ttk.LabelFrame(control_frame, text="Actions")
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.register_button = ttk.Button(buttons_frame, text="Register New Face", 
                                         command=self.register_new_face)
        self.register_button.pack(fill=tk.X, padx=5, pady=5)
        
        self.view_faces_button = ttk.Button(buttons_frame, text="View Saved Faces", 
                                           command=self.view_saved_faces)
        self.view_faces_button.pack(fill=tk.X, padx=5, pady=5)
        
        # Face list
        faces_frame = ttk.LabelFrame(control_frame, text="Saved Faces")
        faces_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.face_listbox = tk.Listbox(faces_frame)
        self.face_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Populate face list
        self.update_face_list()
        
        # Delete face button
        self.delete_face_button = ttk.Button(faces_frame, text="Delete Selected Face", 
                                           command=self.delete_face)
        self.delete_face_button.pack(fill=tk.X, padx=5, pady=5)
    
    def update_face_list(self):
        self.face_listbox.delete(0, tk.END)
        for name in face_database.keys():
            self.face_listbox.insert(tk.END, name)
    
    def register_new_face(self):
        if not use_insightface:
            messagebox.showerror("Error", "InsightFace is required for face registration")
            return
        
        if self.current_frame is None:
            messagebox.showerror("Error", "No video frame available")
            return
        
        # Check if any face is detected in current frame
        frame = self.current_frame.copy()
        faces = face_app.get(frame)
        
        if not faces:
            messagebox.showerror("Error", "No face detected in current frame")
            return
        
        # Ask for person's name
        name = simpledialog.askstring("Register Face", "Enter person's name:")
        if not name:
            return
        
        # Get face with highest confidence
        best_face = max(faces, key=lambda x: x.det_score)
        
        # Check if face has embedding
        if not hasattr(best_face, 'embedding') or best_face.embedding is None:
            messagebox.showerror("Error", "Failed to extract face features")
            return
        
        # Add to database
        if name in face_database:
            face_database[name].append(best_face.embedding)
        else:
            face_database[name] = [best_face.embedding]
        
        # Save database
        save_face_database()
        
        # Update face list
        self.update_face_list()
        
        messagebox.showinfo("Success", f"Face registered for {name}")
    
    def view_saved_faces(self):
        if not face_database:
            messagebox.showinfo("Info", "No faces saved in database")
            return
        
        # Show list of saved faces
        names = "\n".join(face_database.keys())
        messagebox.showinfo("Saved Faces", f"Saved faces:\n{names}")
    
    def delete_face(self):
        selected = self.face_listbox.curselection()
        if not selected:
            messagebox.showerror("Error", "No face selected")
            return
        
        name = self.face_listbox.get(selected[0])
        confirm = messagebox.askyesno("Confirm", f"Delete face data for {name}?")
        
        if confirm:
            if name in face_database:
                del face_database[name]
                save_face_database()
                self.update_face_list()
                messagebox.showinfo("Success", f"Face data for {name} deleted")
    
    def video_loop(self):
        while self.running:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                time.sleep(0.1)
                continue
            
            # Store current frame for registration
            self.current_frame = frame.copy()
            
            # Make a copy for display
            display_frame = frame.copy()
            
            # Update tracker if initialized and tracker is available
            if self.tracking_initialized and self.tracker is not None:
                try:
                    track_success, tracking_box = self.tracker.update(frame)
                    
                    if track_success:
                        # Draw tracking box
                        x, y, w, h = [int(v) for v in tracking_box]
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                        cv2.putText(display_frame, "Tracking", (x, y - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    else:
                        # Lost tracking, reset
                        self.tracking_initialized = False
                except:
                    # Error in tracking, reset
                    self.tracking_initialized = False
                    print("Tracking error, resetting")
            
            # Run detector periodically or when tracking is lost
            if self.frame_count % self.detect_interval == 0 or not self.tracking_initialized:
                # Clear previously detected faces
                self.tracked_faces = {}
                
                # Detect using InsightFace if available
                if use_insightface:
                    start_time = time.time()
                    
                    # Face detection and analysis with InsightFace
                    faces = face_app.get(frame)
                    
                    end_time = time.time()
                    
                    # Store detected faces
                    self.last_detected_boxes = []
                    
                    for i, face in enumerate(faces):
                        # Get bounding box
                        bbox = face.bbox.astype(int)
                        x, y, x2, y2 = bbox
                        w, h = x2 - x, y2 - y
                        
                        # Recognize face if it has embedding
                        person_name = "Unknown"
                        confidence = 0.0
                        
                        if hasattr(face, 'embedding') and face.embedding is not None:
                            person_name, confidence = recognize_face(face.embedding)
                        
                        # Store for display
                        self.last_detected_boxes.append((x, y, w, h))
                        self.tracked_faces[i] = {
                            "bbox": (x, y, w, h), 
                            "name": person_name,
                            "confidence": confidence
                        }
                        
                        # Draw face rectangle
                        color = (0, 255, 0) if person_name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                        
                        # Add labels - name, gender, age
                        label = f"{person_name}"
                        if confidence > 0:
                            label += f" ({confidence:.2f})"
                        if hasattr(face, 'gender') and face.gender is not None:
                            gender = 'M' if face.gender == 1 else 'F'
                            label += f" ({gender})"
                        if hasattr(face, 'age') and face.age is not None:
                            label += f", Age: {int(face.age)}"
                        
                        cv2.putText(display_frame, label, (x, y - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                     
                     # Update face count label
                        self.face_count_label.config(text=f"Faces: {len(faces)}")
                     
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
                                 
                                 self.last_detected_boxes.append((x_orig, y_orig, w_orig, h_orig))
                                 
                                 # Draw detection rectangles
                                 cv2.rectangle(display_frame, (x_orig, y_orig), 
                                             (x_orig + w_orig, y_orig + h_orig), (0, 100, 255), 2)
                                 cv2.putText(display_frame, "Person", (x_orig, y_orig - 10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)
                     
                     # Update status
                        if use_insightface:
                         self.status_label.config(text=f"InsightFace: {len(faces)} faces detected")
                    else:
                     # Fallback to HOG detector
                     # Resize for faster detection
                     small_frame = cv2.resize(frame, (320, 240))
                     gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                     
                     # Run detector
                     boxes, weights = hog.detectMultiScale(
                         gray, 
                         winStride=(8, 8),
                         padding=(4, 4), 
                         scale=1.05
                     )
                     
                     # Store detected boxes for display
                     self.last_detected_boxes = []
                     
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
                             
                             self.last_detected_boxes.append((x_orig, y_orig, w_orig, h_orig))
                             
                             # Draw detection rectangles
                             cv2.rectangle(display_frame, (x_orig, y_orig), 
                                         (x_orig + w_orig, y_orig + h_orig), (0, 255, 0), 2)
                             cv2.putText(display_frame, f"Person", (x_orig, y_orig - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                     
                     # Update status and face count
                     self.status_label.config(text=f"HOG Detector: {len(boxes)} people detected")
                     self.face_count_label.config(text=f"People: {len(boxes)}")
                 
                 # Set up tracking for first detected person if available
                    if len(self.last_detected_boxes) > 0 and self.tracker is not None:
                     # Use the first detection (typically the largest/closest)
                     x, y, w, h = self.last_detected_boxes[0]
                     
                     try:
                         # Re-initialize tracker
                         self.tracker = create_tracker()
                         if self.tracker is not None:
                             track_success = self.tracker.init(frame, (x, y, w, h))
                             if track_success:
                                 self.tracking_initialized = True
                                 self.tracking_box = (x, y, w, h)
                     except:
                         print("Failed to initialize tracker")
             
             # Display previously detected boxes if no new detections in this frame
                elif not self.tracking_initialized:
                 if self.tracked_faces:
                     # Display previously detected faces
                     for face_idx, face_data in self.tracked_faces.items():
                         x, y, w, h = face_data["bbox"]
                         name = face_data["name"]
                         
                         color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                         cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                         cv2.putText(display_frame, name, (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                 elif len(self.last_detected_boxes) > 0:
                     # Display previously detected people
                     for (x, y, w, h) in self.last_detected_boxes:
                         cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                         cv2.putText(display_frame, "Person", (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
              
              # Increment frame counter
                self.frame_count += 1
              
              # Convert OpenCV image to Tkinter format
                cv_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(cv_image)
                tk_image = ImageTk.PhotoImage(image=pil_image)
              
              # Update video label
                self.video_label.configure(image=tk_image)
                self.video_label.image = tk_image
              
              # Sleep to reduce CPU usage
                time.sleep(0.01)
      
    def on_closing(self):
          self.running = False
          if self.thread.is_alive():
              self.thread.join(timeout=1.0)
          if self.cap.isOpened():
              self.cap.release()
          self.root.destroy()
          print("Application closed")

# Main entry point
if __name__ == "__main__":
    print("Starting Face Recognition System...")
    print("Press 'Register New Face' to add faces to the database")
    
    # Create Tkinter root
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    
    # Start Tkinter main loop
    root.mainloop()
    
    print("Application exited")