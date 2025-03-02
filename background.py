import cv2
import numpy as np
import os

def create_background_model(video_path, num_frames=50):
    """Creates a background model by averaging multiple frames from the given video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    sum_saturation = None
    sum_value = None
    hue_frames = []
    count = 0
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(frame_hsv)
        
        hue_frames.append(hue)
        
        if sum_saturation is None:
            sum_saturation = np.float32(sat)
            sum_value = np.float32(val)
        else:
            sum_saturation += np.float32(sat)
            sum_value += np.float32(val)
        count += 1
    
    cap.release()
    
    if count == 0:
        print("Error: No valid frames read.")
        return None
    
    # Compute median hue to prevent hue wrapping issues
    hue_stack = np.stack(hue_frames, axis=0)
    median_hue = np.median(hue_stack, axis=0).astype(np.uint8)
    
    background_model = cv2.merge([
        median_hue,  
        (sum_saturation / count).astype(np.uint8),  
        (sum_value / count).astype(np.uint8) 
    ])
    
    # Apply slight Gaussian blur to smooth artifacts
    background_model = cv2.GaussianBlur(background_model, (5, 5), 0)
    
    return background_model

def subtract_background(video_path, background_model):
    """Performs background subtraction on the given video using adaptive HSV thresholding."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        diff = cv2.absdiff(frame_hsv, background_model)
        
        # Adaptive thresholding per channel
        h_diff, s_diff, v_diff = cv2.split(diff)
        
        h_mask = cv2.threshold(h_diff, 15, 255, cv2.THRESH_BINARY)[1]
        s_mask = cv2.threshold(s_diff, 40, 255, cv2.THRESH_BINARY)[1]
        v_mask = cv2.threshold(v_diff, 40, 255, cv2.THRESH_BINARY)[1]
        
        # Combine masks
        mask = cv2.bitwise_or(h_mask, cv2.bitwise_or(s_mask, v_mask))
        
        # Post-processing: Remove noise & fill gaps
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Fill small holes
        
        # Overlay mask on original frame for visualization
        overlay = cv2.bitwise_and(frame, frame, mask=mask)
        
        cv2.imshow("Foreground Mask", mask)
        cv2.imshow("Foreground Overlay", overlay)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    background_video = "data/cam4/background.avi"  # Change for each camera
    video_to_process = "data/cam4/video.avi"  # Change for each camera
    
    background_model = create_background_model(background_video)
    if background_model is not None:
        background_bgr = cv2.cvtColor(background_model, cv2.COLOR_HSV2BGR)  # Convert to BGR for proper display
        cv2.imshow("Background Model", background_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save background model for later use
        cv2.imwrite("data/cam3/background_model.png", background_bgr)
        print("Background model saved successfully.")
        
        # Perform background subtraction
        subtract_background(video_to_process, background_model)

if __name__ == "__main__":
    main()
