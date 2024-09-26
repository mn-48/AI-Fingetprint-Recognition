import os
import cv2

def match_fingerprints(reference_image_path, directory_path, ratio_test_threshold=0.75):
    # Load the reference fingerprint image
    reference_image = cv2.imread(reference_image_path)
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors of the reference image
    keypoints_ref, descriptors_ref = sift.detectAndCompute(reference_image, None)
    
    if descriptors_ref is None:
        print("No descriptors found in the reference image.")
        return

    # List all images in the specified directory
    # files = sorted([file for file in os.listdir(directory_path)][:1000])
    # files = sorted([file for file in os.listdir(directory_path) if file.endswith(".BMP") or file.endswith(".jpg") or file.endswith(".png")])[:1000]

    files = [file for file in os.listdir(directory_path) if file.endswith(".BMP") or file.endswith(".jpg") or file.endswith(".png")][:1000]  
    # Initialize FlannBasedMatcher
    flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {})
    
    best_score = 0
    best_file = None
    best_matches = None
    best_keypoints_f = None
    best_descriptors_f = None
    best_fingerprint_image = None
    
    for file in files:
        if best_score >= 100:
            break
        # Load each fingerprint image from the directory
        fingerprint_image_path = os.path.join(directory_path, file)
        fingerprint_image = cv2.imread(fingerprint_image_path)
        
        if fingerprint_image is None:
            print(f"Could not read image: {file}")
            continue
        
        # Detect keypoints and descriptors of the fingerprint image
        keypoints_f, descriptors_f = sift.detectAndCompute(fingerprint_image, None)
        
        if descriptors_f is None:
            print(f"No descriptors found in image: {file}")
            continue

        # Match descriptors using FlannBasedMatcher with KNN
        matches = flann.knnMatch(descriptors_ref, descriptors_f, k=2)
        
        # Apply ratio test to filter good matches
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_test_threshold * n.distance:
                good_matches.append(m)
        
        # Calculate matching percentage
        keypoints = min(len(keypoints_ref), len(keypoints_f))
        if keypoints > 0:
            match_percentage = (len(good_matches) / keypoints) * 100
            print(f"Image: {file}, Match Percentage: {match_percentage:.2f}%")
            
            # Update best match
            if match_percentage > best_score:
                best_score = match_percentage
                best_file = file
                best_matches = good_matches
                best_keypoints_f = keypoints_f
                best_fingerprint_image = fingerprint_image
        else:
            print(f"Image: {file}, No valid keypoints detected.")
    
    # Print the best matching file and score
    if best_file:
        print(f"\nBest Match: {best_file}, Score: {best_score:.2f}%")

        # Draw matches using cv2.drawMatches
        result = cv2.drawMatches(reference_image, keypoints_ref, best_fingerprint_image, best_keypoints_f, best_matches, None)
        
        # Resize for better visibility
        result = cv2.resize(result, None, fx=2, fy=2)
        
        # Display the result
        cv2.imshow(f"Best Match: {best_file}", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    else:
        print("No valid matches found.")

# Define paths
# reference_image_path = "archive/SOCOFing/Real/150__M_Right_index_finger.BMP"
reference_image_path = "archive/SOCOFing/Real/54__M_Left_index_finger.BMP"
directory_path = "archive/SOCOFing/Altered/Altered-Easy/"
# directory_path = "archive/SOCOFing/Real/"

# Call the function
match_fingerprints(reference_image_path, directory_path)
