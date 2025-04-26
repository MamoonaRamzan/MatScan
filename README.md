# MatScan

An industrial analysis tool for detecting and classifying defects in material surfaces using machine learning.

---

## 📝 Overview

MatScan: Material Surface Analyzer System is a desktop application that helps manufacturing and quality control professionals identify and classify defects in material surfaces. Using computer vision and deep learning, the system can detect six common types of industrial material defects with high accuracy.

---

## Dataset

The model was trained on the NEU Surface Defect Database, which contains six types of typical surface defects of hot-rolled steel strips. Each class contains 300 grayscale images (200×200 pixels).

---

## Features

- **User-Friendly GUI** – Clean, tabbed interface built for ease of use
- **Multi-Class Defect Detection** – Supports classification of:
  - Crazing (fine network of surface cracks)
  - Inclusions (embedded foreign particles)
  - Patches (irregular surface areas)
  - Pitted Surfaces (small depressions or holes)
  - Rolled-in Scale (pressed scale particles)
  - Scratches (linear marks or grooves)
- **Real-Time Image Analysis** – Fast and responsive defect prediction
- **Grad-CAM Heatmap Visualization** – Understand what influenced the model’s decision
- **Confidence Scores** – Displayed for primary and alternative predictions

---

## 💻 Usage 

- **Run the application:**
python material_defect_analyzer.py

1. The application will load the pre-trained model automatically. Else load the model manually.
2. Click "Select Material Image" to choose an image file to analyze.
3. Click "Analyze Defect" to process the image and see the classification results.
4. View the detailed results in the "Analysis Results" tab, including:

-Primary defect type
-Confidence score
-Alternative possible defect types


5. Switch to the "Defect Visualization" tab to see a heatmap highlighting areas that influenced the model's decision.

---
## Preview
<img width="960" alt="Screenshot 2025-04-26 125844" src="https://github.com/user-attachments/assets/24be5fc1-e36a-4cde-b3c8-cff55a2e6824" />
<img width="960" alt="Screenshot 2025-04-26 125931" src="https://github.com/user-attachments/assets/852302c2-96cd-4a77-8339-58258e684a04" />
<img width="960" alt="Screenshot 2025-04-26 130014" src="https://github.com/user-attachments/assets/d83460b8-a6fa-415c-862e-75c832edffb9" />





