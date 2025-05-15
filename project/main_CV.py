import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import os
import skimage as ski
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error

    
class classifier:
    def __init__(self, reference_dir):
        # Load reference images
        self.references = self._load_references(reference_dir)
  
    def _load_references(self, reference_dir):
        references = []
        for img_path in Path(reference_dir).glob('*.jpg'):
            img = cv2.imread(str(img_path))
            # Convert to grayscale
            image = cv2.resize(img, (1080, 720))      
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.medianBlur(gray, 5)#cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply threshold
            #_, binary = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,11,2)
            # cv2.imshow("result",binary)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()   
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(4,4))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(20,20))
            binary = cv2.dilate(binary,kernel,iterations = 1)

            # cv2.imshow("result",binary)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()        

            # Find contours
            x, y, w, h = 0, 0, 0, 0
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                ref = image[y:y+h, x:x+w] 

            # cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)
            # cv2.imshow("result",image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            if ref is not None:
                references.append({
                    'image': cv2.resize(ref, (1080, 720)), #cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                    'name': img_path.stem.replace('_', ' ')

                })
        return references

    def _generate_proposals(self, image):
        """Generate region proposals using contour detection"""
        # Convert to grayscale        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.medianBlur(gray, 5)
        
        # Apply threshold
        binary = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,9,2)
        #_, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # cv2.imshow("result",binary)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()   
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(4,4))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(20,20))
        binary = cv2.dilate(binary,kernel,iterations = 1)

        # cv2.imshow("result",binary)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()                 
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        proposals = []
        x, y, w, h = 0, 0, 0, 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            proposals.append((x, y, w, h))
            # cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)
            # cv2.imshow("result",image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        return proposals

    def _compare_with_references(self, region):
        """Compare region with reference images using feature matching"""
        orb = cv2.ORB_create(nfeatures=9000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Get region features
        region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, region_des = orb.detectAndCompute(region, None)
        if region_des is None:
            return 0.0, None
        
        max_similarity = 0.0
        max_class = None
        for ref in self.references:
            ref_im = cv2.cvtColor(ref["image"], cv2.COLOR_BGR2GRAY)
            _, ref_des = orb.detectAndCompute(ref_im, None)
            if ref_des is not None:
                matches = bf.match(region_des, ref_des)
                similarity = len(matches) / max(len(region_des), len(ref_des))
                if(similarity > max_similarity):
                    max_similarity = similarity
                    max_class = ref['name']
                    max_ref = ref["image"]
 
        # cv2.imshow("region_final",region)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imshow("final",max_ref)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return max_similarity, max_class

    def detect(self, image_path, similarity_threshold=0.2):
        image = cv2.imread(str(image_path))
        image_rgb = cv2.resize(image, (1080, 720))# cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Generate proposals
        proposals = self._generate_proposals(image_rgb)
        detections = []
        for x, y, w, h in proposals:
            region = image_rgb[y:y+h, x:x+w]
            region = cv2.resize(region, (1080, 720))

            # Check similarity with references
            similarity, class_name = self._compare_with_references(region)
            if similarity > similarity_threshold:
                # cv2.rectangle(image_rgb,(x,y),(x+w,y+h),(255,0,0),3)
                # cv2.imshow("result",image_rgb)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                detections.append({
                    'bbox': (x, y, w, h),
                    'confidence': similarity,
                    'class_name': class_name#class_id
                })
        
        return detections
    
def predict_to_csv(detections, col, num_classes=13):
    """Convert detections to row format matching train.csv"""
    # Initialize counts for each class
    class_counts = [0] * num_classes

    # Count occurrences of each class in detections

    for ind, class_ref in enumerate(col[1:]):
        for det in detections:
            if(det['class_name'] == class_ref):
                class_counts[ind] += 1
    return class_counts
def print_first_line(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Get the first row
    first_row = df.iloc[0]
    
    # Print each value in the first row
    print(','.join(str(value) for value in first_row.values))
def calculate_similarity(csv1_path, csv2_path, metric='f1'):
    """
    Calculate similarity between two CSV files containing classification results
    Args:
        csv1_path: Path to first CSV file
        csv2_path: Path to second CSV file
        metric: Similarity metric to use ('f1', 'accuracy', 'precision', 'recall', 'mse')
    Returns:
        overall_similarity, class_scores
    """
    # Read CSV files
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)
    
    # Sort both DataFrames by ID to ensure alignment
    df1 = df1.sort_values('id').reset_index(drop=True)
    df2 = df2.sort_values('id').reset_index(drop=True)
    
    # Get class columns (all except 'id')
    class_columns = [col for col in df1.columns if col != 'id']
    
    # Calculate per-class scores
    class_scores = {}
    for col in class_columns:
        try:
            if metric == 'f1':
                score = f1_score(df2[col], df1[col], average='macro')
            elif metric == 'accuracy':
                score = accuracy_score(df2[col], df1[col])
            elif metric == 'precision':
                score = precision_score(df2[col], df1[col], average='macro', zero_division=0)
            elif metric == 'recall':
                score = recall_score(df2[col], df1[col], average='macro', zero_division=0)
            elif metric == 'mse':
                score = -mean_squared_error(df2[col], df1[col])  # Negative so higher is better
            else:
                raise ValueError(f"Unknown metric: {metric}")
            class_scores[col] = score
        except Exception as e:
            print(f"Error calculating {metric} for {col}: {str(e)}")
            class_scores[col] = 0
    
    # Calculate overall similarity
    overall_similarity = np.mean(list(class_scores.values()))
    
    return overall_similarity, class_scores

def main():
    # Set paths using Path for better cross-platform compatibility
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    
    train_csv = data_dir / "train.csv"
    train_img_dir = data_dir / "train"
    reference_dir = data_dir / "references"
    test_dir = data_dir / "test"
    
        
    # print(f"Loading data from: {data_dir}")
    
    # Initialize RCNN with both training data and references
    classi = classifier(str(reference_dir))
    
    
    # Test on all images in test directory
    test_images = list(train_img_dir.glob('*.jpg'))
    if not test_images:
        print(f"No test images found in {test_dir}")
        return
        
    train_df = pd.read_csv(train_csv)
    results_df = pd.DataFrame(columns=train_df.columns)
    
    # Process test images and save results
    for test_image in test_images:
        # print(f"\nProcessing: {test_image.name}")
        try:
            detections = classi.detect(str(test_image))
            # print(f"Found {len(detections)} objects")
            
            # Convert image filename to ID (remove 'L' and '.jpg')
            img_id = test_image.stem
            if img_id.startswith('L'):
                img_id = img_id[1:]
            
            # Get counts for each class
            class_counts = predict_to_csv(detections, train_df.columns[0:])
            
            # Create row with image ID and class counts
            row_data = {'id': img_id}
            for col, count in zip(train_df.columns[1:], class_counts):
                row_data[col] = count
                
            # Add row to results DataFrame
            results_df = pd.concat([results_df, pd.DataFrame([row_data])], ignore_index=True)
            
        except Exception as e:
            print(f"Error processing {test_image.name}: {str(e)}")
    
    # Save results to CSV
    output_path = data_dir / "predictions.csv"
    results_df.to_csv(output_path, index=False)
    # print(f"\nResults saved to: {output_path}")
    # print("pred: ")
    # print_first_line(data_dir/"predictions.csv")
    # print("truth: ")
    # print_first_line(train_csv)
    
    # Print results
    #metrics = ['f1', 'accuracy', 'precision', 'recall', 'mse'] 
    metrics = ['f1', 'accuracy', 'precision']
    for metric in metrics:
        # print(f"\n{metric.upper()} Scores:")
        # print("-" * 40)
        overall_sim, class_sims = calculate_similarity(
            data_dir / "predictions.csv", 
            train_csv,
            metric=metric
        )
        # for class_name, score in class_sims.items():
        #     print(f"{class_name:20s}: {score:.4f}")
        print(f"\nOverall {metric.upper()} Score: {overall_sim:.4f}")
    


if __name__ == "__main__":
    main()