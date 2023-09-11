# Abnormal-Xray-Detection
Abnormal X-Ray Detection System using Convolution Neural Network

More than 1.7 billion people worldwide [BMU, 2017], affected with musculoskeletal conditions which causes severe, long- term pain and disability. 
The abnormality detection task is a critical radiological task i.e. it determines whether a radiographic study is normal or abnormal. A study interpreted as normal
rules out disease and thus eliminates the requirement of patients to endure further diagnostic procedures. We develop an abnormality detection model on MURA which
takes as input one or more views for a study of an upper extremity. The 169-layer convolution neural network predicts the probability of abnormality on each view.
Then the per-view probabilities are averaged to determine the probability of abnormality for the study.

The main objectives of the system is to:

• Design a model which detects and highlights the abnormalities in X-ray images.
• Train the model over 50,000 training images avoiding over-fitting and under- fitting problems.
• Build the abnormality detection process efficient so that it is much faster than manual detection.
• Formulate the model by achieving good amount of accuracy in abnormalities detection.
