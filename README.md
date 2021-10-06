# Speed-Estimation-of-Vehicles-with-Plate-Detection
The main objective of this project is to identify overspeed vehicles, using Deep Learning and Machine Learning Algorithms. After acquisition of series of images from the video, trucks are detected using Haar Cascade Classifier. The model for the classifier is trained using lots of positive and negative images to make an XML file. This is followed by tracking down the vehicles and estimating their speeds with the help of their respective locations, ppm (pixels per meter) and fps (frames per second). Now, the cropped images of the identified trucks are sent for License Plate detection. The CCA (Connected Component Analysis) assists in Number Plate detection and Characters Segmentation. The SVC model is trained using characters images (20X20) and to increase the accuracy, 4 cross fold validation (Machine Learning) is also done. This model aids in recognizing the segmented characters. After recognition, the calculated speed of the trucks is fed into an excel sheet along with their license plate numbers. These trucks are also assigned some IDs to generate a systematized database.

To run Speed_Detection_&_License_Plate_Detection.py, follow these steps below:
  1. Download IDLE python from this site: 'https://www.python.org/downloads/'
  2. Install OpenCV and dlib libraries from: 'https://www.learnopencv.com/install-dlib-on-windows/'
  3. Install these other libraries as well: skimage, sklearn, openpyxl, threading, time, joblib, numpy, matplotlib, datetime, & os.
  4. Through Command Prompt, run this code. (Make sure to change the directories)
 
Methodology: 
  1. Image Acquisition: Extracting series of images from a video one by one, then reading them using cv2 (Open Source Computer Vision)        library.
  
  2. Trucks Detection: Using Haar Cascade CLassifier
     This Algorithm includes 4 stages:
      a. Haar Feature Selction
      b. Creating Integral Images
      c. Adaboost (Adaptive boosting) Training
      d. Cascading Classifiers
      Functions Used: CascadeClassifier and detectMultiScale
      
  3. Training a machine to make an XML file():
     I WAY: Command Prompt Method
      a. Collection of Image Database
      b. Augmentation: for boosting image database
      c. Crop and mark positive images using Objectmarker or Image Clipper
      d. Haar Training
      Refernce: 'www.cs.auckland.ac.nz/~m.rezaei/Downloads.html'
     II WAY: Cascade Trainer (Graphical User Interface)
      Install GUI software and specify the values of parameters.
      Reference: 'http://amin-ahmadi.com/cascade-trainer-gui/'
      
  4. Tracking the detected trucks using dlib library:
     Functions Used:
      a. dlib.correlation_tracker()
      b. dlib.correlation_tracker().start_track()
      c. dlib.rectangle
      
  5. Speed Estimation: 
     Speed (in km/hr) = Distance Travelled by the detected trucks (in meters) * fps * 3.6 
     
  6. CCA (Connected Component Analysis): for detecting License Plate and segmenting Characters
     Functions Used: skimage.measure.label and skimage.measure.regionprops 
     
     Assumptions used in this code: (change it accordingly)
     height of the license plate = 6% - 18% of the height of the cropped truck image 
     width of the license plate = 8.5% - 20% of the width of the cropped truck image 
     height of the characters to be segmented = 18.75% - 37.5% of the height of the license plate 
     width of the characters to be segmented = 5% - 40% of the width of the license plate
     
  7. Support Vector Classifier and Cross Validation (4-fold): for predicting Characters
  
  8. Working with Excel Sheet: Using 'openpyxl' library, the calculated speed of trucks can be fed into the excel sheet along with their      license plate numbers. These trucks are assigned some IDs to generate a systemized database. 
  
  9. Overspeed Vehicle Enumeration
  
Limitations:
  1. Sometimes, the dlib correlation tracker fails when the scale of the object keeps on changing.
  2. The estimated speed is not so authentic because of the expensive scanning and processing time.
  3. A good resolution camera ought to be used for predicting non-erroneous license plate characters. Neural Enhance – Super Resolution      of images (Deep Learning) can also be used, instead. However it increases the processing time.
  4. The license plate, occasionally, are covered with dust or are veiled by a rod in the front or are not even there, thereby not            letting the detection possible.
  5. Old trucks cannot be identified because the machine is trained using only new models of trucks.
  
 Inspired by https://github.com/apoorva-dave/LicensePlateDetector/blob/master/DetectPlate.py
 Reference: https://github.com/kraten/vehicle-speed-check
            https://github.com/kraten/vehicle-speed-check/blob/master/myhaar.xml
 
 
 
#Note: The technique (image processing) used to solve this Machine Learning problem is outdated now. You would be able to achieve better results if you use YOLOv3 (https://pjreddie.com/yolo/) object detector for truck detection and character recognition and fine-tuning of VGG16 model for character segmentation.
