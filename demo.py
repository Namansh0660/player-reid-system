import numpy as np
import cv2
import random

# --- 1. Define Constants and Helper Functions ---

# Simulate a small video frame
FRAME_WIDTH, FRAME_HEIGHT = 640, 360
NUM_PLAYERS = 4
MAX_FRAMES = 30
MAX_FRAMES_LOST = 5

def generate_random_bbox():
    """Generate a random bounding box within frame limits"""
    w, h = 60, 120
    x1 = random.randint(0, FRAME_WIDTH - w)
    y1 = random.randint(0, FRAME_HEIGHT - h)
    x2, y2 = x1 + w, y1 + h
    return [x1, y1, x2, y2]

def extract_features(bbox):
    """Simulate feature extraction (just use the bbox center as a feature for demo)"""
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return [center_x, center_y]

def compute_similarity(feat1, feat2):
    """Simulate similarity computation (Euclidean distance for demo)"""
    dist = np.linalg.norm(np.array(feat1) - np.array(feat2))
    return 1.0 / (1.0 + dist)  # Inverse distance as similarity

# --- 2. Define Player Re-ID Class ---

class PlayerReID:
    def __init__(self):
        self.next_id = 1
        self.player_gallery = {}  # {id: {'features': [...], 'last_seen': frame}}
        self.lost_players = {}    # {id: {'features': [...], 'frames_lost': int}}

    def assign_player_id(self, features, frame_number):
        best_match_id = None
        best_similarity = 0.0

        # Check active players
        for pid, info in self.player_gallery.items():
            similarity = compute_similarity(features, info['features'])
            if similarity > best_similarity and similarity > 0.5:  # Demo threshold
                best_similarity = similarity
                best_match_id = pid

        # Check lost players
        for pid, info in list(self.lost_players.items()):
            similarity = compute_similarity(features, info['features'])
            if similarity > best_similarity and similarity > 0.5:
                best_similarity = similarity
                best_match_id = pid
                # Move back to gallery
                self.player_gallery[pid] = info
                del self.lost_players[pid]

        if best_match_id is not None:
            # Update existing player
            self.player_gallery[best_match_id]['features'] = features
            self.player_gallery[best_match_id]['last_seen'] = frame_number
            return best_match_id
        else:
            # New player
            new_id = self.next_id
            self.next_id += 1
            self.player_gallery[new_id] = {'features': features, 'last_seen': frame_number}
            return new_id

    def update_lost_players(self, frame_number):
        # Move players not seen to lost_players
        for pid, info in list(self.player_gallery.items()):
            if info['last_seen'] < frame_number:
                frames_lost = frame_number - info['last_seen']
                if frames_lost <= MAX_FRAMES_LOST:
                    self.lost_players[pid] = info
                del self.player_gallery[pid]

        # Remove players lost for too long
        for pid, info in list(self.lost_players.items()):
            frames_lost = frame_number - info['last_seen']
            if frames_lost > MAX_FRAMES_LOST:
                del self.lost_players[pid]

# --- 3. Simulate Video Processing ---

def main():
    print("=== Player Re-ID System Explanation ===")
    print("This demo simulates tracking and re-identifying players in a video.")
    print("It uses a simple feature (bbox center) and similarity (inverse distance).")
    print("Players can disappear and reappear, and the system keeps their IDs consistent.")
    print("\n--- Step 1: Initialize the Re-ID System ---")
    reid = PlayerReID()
    print("Re-ID system initialized. Next available ID: 1")

    print("\n--- Step 2: Simulate Video Frames ---")
    for frame in range(1, MAX_FRAMES + 1):
        print(f"\nFrame {frame}:")
        # Simulate detections
        detections = []
        for _ in range(NUM_PLAYERS):
            if random.random() > 0.2:  # 80% chance of being detected
                bbox = generate_random_bbox()
                features = extract_features(bbox)
                detections.append(features)
                print(f"  Detected player at {bbox} (features: {features})")

        # Assign IDs
        player_ids = []
        for feat in detections:
            pid = reid.assign_player_id(feat, frame)
            player_ids.append(pid)
            print(f"  Assigned ID: {pid}")

        # Update lost players
        reid.update_lost_players(frame)
        print(f"  Active players: {len(reid.player_gallery)}")
        print(f"  Lost players: {len(reid.lost_players)}")

    print("\n--- Step 3: Summary ---")
    print(f"Total unique players detected: {reid.next_id - 1}")
    print("System keeps track of players even if they are temporarily lost.")
    print("Players are re-identified based on their features when they reappear.")

    print("\n=== Key Concepts Printed ===\n")
    print("1. **Detection:** Players are detected in each frame (simulated with random bboxes).")
    print("2. **Feature Extraction:** Features are extracted from detections (here: bbox center).")
    print("3. **Similarity Matching:** New detections are matched to existing players using similarity.")
    print("4. **ID Assignment:** Each player gets a consistent ID across frames.")
    print("5. **Occlusion Handling:** Players can be temporarily lost and re-identified when they reappear.")
    print("6. **Gallery Management:** Active and lost player galleries are maintained for efficient matching.")

if __name__ == "__main__":
    main()
