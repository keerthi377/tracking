# tracking
Tracking is a critical phenomenon that occurs in high-voltage transmission and distribution systems. This happens due to the presence of contaminants such as dust, dirt, salt, or any conductive particles on the surface of insulators. This results in the formation of carbon paths and can lead to critical degradation and insulation failure. Failure due to tracking is a major concern when using organic insulating materials due to which several standard analyzing methods were proposed to evaluate tracking. Traditional tracking methods primarily rely on the material’s resistance characteristics, electrical parameters, and visual inspection for carbon track. In the tracking process, the temperature rise due to discharge at the interface can be detected using a thermal camera. It eliminates direct physical interaction with the material.

METHODOLOGY:

When a high-voltage insulator undergoes degradation due to tracking, it releases energy as heat, leading to a non-uniform temperature distribution across its surface. This distribution can be captured and displayed using a thermal camera. By analysing the heat distribution in the thermal image, we can obtain the stage and severity of the tracking phenomenon. The study we propose involves two approaches, namely, filter-based and feature-based approaches towards classifying the stages and severity of tracking in silicone rubber insulators. The 4 stages of tracking are as follows:
1. surface discharge: 80-90 degrees
2. dry band formation: 100-130 degrees
3. localised bright spot: 600-660 degrees
4. track path: 1200-1500  degrees
In the following part we will describe each of the approaches in detail.


TEST SETUP

The study aims to classify the severity levels of tracking in SiR insulators. To simulate tracking, the IEC587 inclined plane tracking test was conducted in the laboratory under controlled environment conditions. A contaminant solution of ammonium chloride (NH₄Cl) was made to flow down the underside of a flat inclined test specimen. The contaminant solution had a conductivity of 2500S/cm.  The test specimen with the dimensions of 50 × 120 × 6 mm³ was mounted at 45° to the horizontal between electrodes. The two electrodes, namely, the high voltage electrode and the ground electrode were separated by a distance of 50 mm, and tests were performed at room temperature. The rate of the contaminant release was done at 0.6 ml/min using a peristaltic pump. The test was carried out at 4.5kV which was supplied constantly for 6 hours or until insulator failure.
A high-resolution FLIR thermal camera was used to conduct the thermal imaging analysis. Images were collected at 15-minute intervals for 2 minutes to monitor the thermal behavior.

DATA AQUISITION AND DATASET 

Dataset Size: 250 high-quality infrared thermogram images
Image Distribution: 50 images per classification stage (4 stages total) and further augmented to get 250images per stage.
Capture Technology: FLIR thermal camera system
Data Formats:
16-bit TIFF images for raw pixel data analysis
32-bit floating TIFF images for lossless radiometric data
Training Split: 80% training / 20% testing for optimal model performance

IMAGE PREPROCESSING TECHNIQUE

The following 5 filters were used for image enhancement 

Bilateral Filter - Noise reduction while preserving edges
CLAHE (Contrast Limited Adaptive Histogram Equalization) - Enhanced contrast optimization
Gamma Transformation - Brightness and contrast adjustment
Thresholding - Binary image segmentation
Canny Edge Detection - Precise boundary identification

Each filter was optimized for bright spot localization and underwent rigorous accuracy comparison testing to ensure optimal classification performance.

MACHINE LEARNING ARCHITECTURE

UMAP (Uniform Manifold Approximation and Projection)
Purpose: Dimensionality reduction and feature optimization
Benefits:
Prevents overfitting by reducing high-dimensional pixel data
Preserves topological structure in lower dimensions
Eliminates noise while extracting critical features
Methodology: Constructs nearest-neighbor graphs to maintain data relationships in reduced dimensions

XGBoost (Extreme Gradient Boosting)
Architecture: Ensemble of decision trees with sequential error correction
Optimization: Utilizes first and second-order gradients for optimal leaf values and tree structure
Performance: Gradient-based loss function minimizes prediction errors iteratively
Advantage: Progressive model improvement through ensemble learning


