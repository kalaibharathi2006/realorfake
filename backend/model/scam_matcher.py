import os
from PIL import Image
import numpy as np

class ScamMatcher:
    def __init__(self, patterns_dir=None):
        if patterns_dir is None:
            # Default to ../data/scam_patterns relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.patterns_dir = os.path.join(base_dir, 'data', 'scam_patterns')
        else:
            self.patterns_dir = patterns_dir
            
        self.hashes = []
        self._load_patterns()
        
    def _dhash(self, image, hash_size=8):
        """
        Calculate difference hash (dHash) for an image.
        1. Resize to (width=hash_size+1, height=hash_size)
        2. Convert to grayscale
        3. Compare adjacent pixels
        """
        try:
            # Resize
            image = image.convert('L').resize(
                (hash_size + 1, hash_size), 
                Image.Resampling.LANCZOS
            )
            
            pixels = np.array(image)
            # Compare adjacent pixels
            diff = pixels[:, 1:] > pixels[:, :-1]
            # Convert to hex string
            return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
        except Exception as e:
            print(f"Error calculating hash: {e}")
            return None

    def _load_patterns(self):
        """Load and hash all scam pattern images."""
        if not os.path.exists(self.patterns_dir):
            print(f"Patterns directory not found: {self.patterns_dir}")
            os.makedirs(self.patterns_dir, exist_ok=True)
            return

        print(f"Loading scam patterns from {self.patterns_dir}...")
        count = 0
        try:
            files = os.listdir(self.patterns_dir)
            print(f"DEBUG: Found {len(files)} files in directory.")
            for filename in files:
                print(f"DEBUG: Processing {filename}")
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    try:
                        path = os.path.join(self.patterns_dir, filename)
                        print(f"DEBUG: Loading image {path}")
                        img = Image.open(path)
                        img_hash = self._dhash(img)
                        print(f"DEBUG: Hash result: {img_hash}")
                        if img_hash is not None:
                            self.hashes.append((img_hash, filename))
                            count += 1
                    except Exception as e:
                        print(f"Failed to load pattern {filename}: {e}")
        except Exception as e:
            print(f"Error listing directory: {e}")

        print(f"Loaded {count} scam patterns.")

    def match(self, image_path, threshold=5):
        """
        Check if the image at image_path matches any known scam patterns.
        Returns: (is_match, matched_filename)
        """
        try:
            print(f"DEBUG: Matching {image_path} against {len(self.hashes)} patterns")
            target_img = Image.open(image_path)
            target_hash = self._dhash(target_img)
            
            if target_hash is None:
                return False, None
            
            print(f"DEBUG: Target hash {target_hash}")
                
            for pattern_hash, filename in self.hashes:
                # Hamming distance
                distance = bin(target_hash ^ pattern_hash).count('1')
                print(f"DEBUG: Distance to {filename}: {distance}")
                if distance <= threshold:
                    return True, filename
                    
            return False, None
        except Exception as e:
            print(f"Error matching image: {e}")
            return False, None
