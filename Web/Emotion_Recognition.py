import cv2
import numpy as np
from tensorflow.keras.models import load_model
import threading

class EmotionRecognition:
    def __init__(self):
        try:
            # Update this path to your actual model path
            model_path = "C:\\Users\\ngshu\\Downloads\\best_custom_resnet50.h5"
            print(f"Loading model from: {model_path}")
            self.model = load_model(model_path)
            print("Model loaded successfully")
            
            # Load face cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            print(f"Loading cascade from: {cascade_path}")
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                raise Exception("Error loading face cascade classifier")
            
            self.emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
            
            # Optimization settings
            self.skip_frames = 5  # Increased frame skip
            self.frame_count = 0
            self.last_prediction = None
            self.processing_thread = None
            self.is_processing = False
            
        except Exception as e:
            print(f"Error in initialization: {e}")
            raise

    def process_frame(self, frame):
        try:
            # Reduce frame size for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            
            # Face detection on smaller frame
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=3,
                minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            predictions = []
            for (x, y, w, h) in faces:
                # Scale coordinates back to original size
                x, y, w, h = [coord * 2 for coord in (x, y, w, h)]
                
                face_roi = frame[y:y+h, x:x+w]
                # Preprocess face
                face = cv2.resize(face_roi, (224, 224))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = face.astype('float32') / 255.0
                face = np.expand_dims(face, axis=0)
                
                # Make prediction
                prediction = self.model.predict(face, verbose=0)
                emotion_idx = np.argmax(prediction[0])
                confidence = prediction[0][emotion_idx] * 100
                emotion = self.emotions[emotion_idx]
                
                predictions.append(((x, y, w, h), (emotion, confidence)))
            
            return predictions
            
        except Exception as e:
            print(f"Error in processing: {e}")
            return None

    def generate_frames(self):
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Error: Could not open camera")
            return

        # Optimize camera settings
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        try:
            while True:
                success, frame = camera.read()
                if not success:
                    break

                self.frame_count += 1
                
                # Process every nth frame
                if self.frame_count % self.skip_frames == 0 and not self.is_processing:
                    self.is_processing = True
                    # Start processing in a separate thread
                    self.processing_thread = threading.Thread(
                        target=lambda: self.process_and_update(frame.copy())
                    )
                    self.processing_thread.start()

                # Draw last known predictions
                if self.last_prediction:
                    for (x, y, w, h), (emotion, confidence) in self.last_prediction:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        label = f"{emotion} ({confidence:.1f}%)"
                        cv2.putText(frame, label, (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Optimize frame encoding
                ret, buffer = cv2.imencode(
                    '.jpg', 
                    frame, 
                    [cv2.IMWRITE_JPEG_QUALITY, 65]  # Reduced quality for better performance
                )
                
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            print(f"Error in generate_frames: {e}")
        finally:
            camera.release()

    def process_and_update(self, frame):
        predictions = self.process_frame(frame)
        if predictions:
            self.last_prediction = predictions
        self.is_processing = False

    def analyze_image(self, frame):
        try:
            # Debug print
            print("Starting image analysis")
            
            # Convert to RGB (since model expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            print(f"Found {len(faces)} faces")
            
            if len(faces) > 0:
                (x, y, w, h) = faces[0]  # Get the first face
                face_roi = rgb_frame[y:y+h, x:x+w]
                
                # Preprocess face
                face = cv2.resize(face_roi, (224, 224))
                face = face.astype('float32') / 255.0
                face = np.expand_dims(face, axis=0)
                
                # Make prediction
                predictions = self.model.predict(face, verbose=0)[0]
                print("Raw predictions:", predictions)
                
                # Normalize predictions to percentages
                predictions = predictions * 100
                
                results = {
                    'anger': float(predictions[0]),
                    'disgust': float(predictions[1]),
                    'fear': float(predictions[2]),
                    'joy': float(predictions[3]),
                    'neutral': float(predictions[4]),
                    'sadness': float(predictions[5]),
                    'surprise': float(predictions[6])
                }
                
                print("Processed results:", results)
                return results
                
            else:
                print("No face detected in the image")
                return {emotion: 0.0 for emotion in self.emotions}
                
        except Exception as e:
            print(f"Error in analyze_image: {e}")
            import traceback
            traceback.print_exc()
            return {emotion: 0.0 for emotion in self.emotions}