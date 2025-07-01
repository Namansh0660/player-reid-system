import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import time
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize
import argparse

class PlayerReID:
    def __init__(self, model_path, video_path, conf_threshold=0.5, similarity_threshold=0.7):
        """
        Initialize Player Re-Identification system
        
        Args:
            model_path (str): Path to the YOLO model (best.pt)
            video_path (str): Path to input video
            conf_threshold (float): Confidence threshold for detections
            similarity_threshold (float): Similarity threshold for re-identification
        """
        # Load YOLO model
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.conf_threshold = conf_threshold
        self.similarity_threshold = similarity_threshold
        
        # Re-ID tracking variables
        self.player_gallery = {}  # Store player features and info
        self.next_id = 1
        self.max_frames_lost = 10  # Max frames a player can be lost before considered gone
        self.lost_players = {}  # Track temporarily lost players
        
        # Video processing variables
        self.cap = None
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0
        
    def extract_features(self, frame, bbox):
        """
        Extract appearance features from player bounding box
        
        Args:
            frame: Input frame
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Feature vector for the player
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure coordinates are within frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        # Extract player crop
        player_crop = frame[y1:y2, x1:x2]
        
        if player_crop.size == 0:
            return np.zeros(512)  # Return zero vector if crop is empty
        
        # Resize to standard size
        player_crop = cv2.resize(player_crop, (64, 128))
        
        # Extract color histogram features
        hist_b = cv2.calcHist([player_crop], [0], None, [16], [0, 256])
        hist_g = cv2.calcHist([player_crop], [1], None, [16], [0, 256])
        hist_r = cv2.calcHist([player_crop], [2], None, [16], [0, 256])
        
        # Extract texture features using LBP-like approach
        gray = cv2.cvtColor(player_crop, cv2.COLOR_BGR2GRAY)
        texture_features = self.extract_texture_features(gray)
        
        # Combine features
        color_features = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
        all_features = np.concatenate([color_features, texture_features])
        
        # Normalize features
        all_features = normalize([all_features])[0]
        
        return all_features
    
    def extract_texture_features(self, gray_image):
        """
        Extract simple texture features using gradients
        """
        # Compute gradients
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Compute histogram of gradient magnitudes
        hist_texture = cv2.calcHist([magnitude.astype(np.uint8)], [0], None, [16], [0, 256])
        
        return hist_texture.flatten()
    
    def compute_similarity(self, features1, features2):
        """
        Compute cosine similarity between two feature vectors
        """
        if len(features1) == 0 or len(features2) == 0:
            return 0.0
        
        similarity = 1 - cosine(features1, features2)
        return max(0.0, similarity)  # Ensure non-negative
    
    def assign_player_id(self, frame, bbox, frame_number):
        """
        Assign player ID based on similarity to existing players
        
        Args:
            frame: Current frame
            bbox: Player bounding box
            frame_number: Current frame number
            
        Returns:
            Player ID
        """
        features = self.extract_features(frame, bbox)
        
        best_match_id = None
        best_similarity = 0.0
        
        # Compare with existing players in gallery
        for player_id, player_info in self.player_gallery.items():
            similarity = self.compute_similarity(features, player_info['features'])
            
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_match_id = player_id
        
        # Check lost players for potential matches
        for player_id, lost_info in list(self.lost_players.items()):
            similarity = self.compute_similarity(features, lost_info['features'])
            
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_match_id = player_id
                
                # Move back to active gallery
                self.player_gallery[player_id] = lost_info
                del self.lost_players[player_id]
        
        if best_match_id is not None:
            # Update existing player
            self.player_gallery[best_match_id]['features'] = features
            self.player_gallery[best_match_id]['last_seen'] = frame_number
            self.player_gallery[best_match_id]['bbox'] = bbox
            return best_match_id
        else:
            # Create new player
            new_id = self.next_id
            self.next_id += 1
            
            self.player_gallery[new_id] = {
                'features': features,
                'first_seen': frame_number,
                'last_seen': frame_number,
                'bbox': bbox
            }
            
            return new_id
    
    def update_lost_players(self, frame_number):
        """
        Update lost players and remove those lost for too long
        """
        # Move players not seen in current frame to lost players
        for player_id, player_info in list(self.player_gallery.items()):
            if player_info['last_seen'] < frame_number:
                frames_lost = frame_number - player_info['last_seen']
                
                if frames_lost <= self.max_frames_lost:
                    # Move to lost players
                    self.lost_players[player_id] = player_info
                
                # Remove from active gallery
                del self.player_gallery[player_id]
        
        # Remove players lost for too long
        for player_id, lost_info in list(self.lost_players.items()):
            frames_lost = frame_number - lost_info['last_seen']
            if frames_lost > self.max_frames_lost:
                del self.lost_players[player_id]
    
    def process_video(self, output_path=None):
        """
        Process the entire video and perform player re-identification
        
        Args:
            output_path (str): Path to save output video (optional)
        """
        # Open video
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {self.frame_width}x{self.frame_height} @ {self.fps} fps")
        print(f"Total frames: {total_frames}")
        
        # Setup output video writer if path provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps, 
                                (self.frame_width, self.frame_height))
        
        frame_number = 0
        processing_times = []
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                start_time = time.time()
                
                # Run YOLO detection
                results = self.model(frame, conf=self.conf_threshold, verbose=False)
                
                # Process detections
                current_frame_players = set()
                
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            
                            # Only process person class (class 0 in COCO)
                            if int(box.cls[0]) == 0 and conf > self.conf_threshold:
                                # Assign player ID
                                player_id = self.assign_player_id(frame, [x1, y1, x2, y2], frame_number)
                                current_frame_players.add(player_id)
                                
                                # Draw bounding box and ID
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                
                                # Add player ID label
                                label = f"Player {player_id}"
                                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                                cv2.rectangle(frame, (int(x1), int(y1) - label_size[1] - 10), 
                                            (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
                                cv2.putText(frame, label, (int(x1), int(y1) - 5), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Update lost players
                self.update_lost_players(frame_number)
                
                # Add frame info
                info_text = f"Frame: {frame_number + 1}/{total_frames} | Active Players: {len(current_frame_players)} | Lost: {len(self.lost_players)}"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Calculate processing time
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                fps_text = f"Processing FPS: {1.0/processing_time:.1f}"
                cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Write frame to output video
                if out:
                    out.write(frame)
                
                # Display frame (comment out for headless processing)
                cv2.imshow('Player Re-Identification', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                frame_number += 1
                
                # Print progress
                if frame_number % 30 == 0:
                    print(f"Processed {frame_number}/{total_frames} frames")
        
        finally:
            # Cleanup
            self.cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
        
        # Print final statistics
        avg_processing_time = np.mean(processing_times)
        print(f"\nProcessing completed!")
        print(f"Average processing time per frame: {avg_processing_time:.3f}s")
        print(f"Average processing FPS: {1.0/avg_processing_time:.1f}")
        print(f"Total unique players detected: {self.next_id - 1}")

def main():
    parser = argparse.ArgumentParser(description='Player Re-Identification System')
    parser.add_argument('--model', type=str, default='best.pt', 
                       help='Path to YOLO model file')
    parser.add_argument('--video', type=str, default='15sec_input_720p.mp4', 
                       help='Path to input video file')
    parser.add_argument('--output', type=str, default='output_with_tracking.mp4', 
                       help='Path to output video file')
    parser.add_argument('--conf', type=float, default=0.5, 
                       help='Confidence threshold for detection')
    parser.add_argument('--similarity', type=float, default=0.7, 
                       help='Similarity threshold for re-identification')
    
    args = parser.parse_args()
    
    # Initialize and run player re-identification
    player_reid = PlayerReID(
        model_path=args.model,
        video_path=args.video,
        conf_threshold=args.conf,
        similarity_threshold=args.similarity
    )
    
    print("Starting Player Re-Identification...")
    player_reid.process_video(output_path=args.output)

if __name__ == "__main__":
    main()
