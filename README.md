# Pupil-Labs-Detect-Saccades-And-Saccades-Mean-Velocity
A tutorial to detect saccades and saccades mean velocity from eye tracking fixations data collected from Pupil Labs.

A script in python to detect saccades and saccades mean velocity from eye tracking fixations data collected from Pupil Labs. <br>

The python code created for saccade detection follows the tutorial from Engbert et al. (2016): http://read.psych.uni-potsdam.de/attachments/article/156/TechRep-16-1-Engbert.pdf and Pupil Labs gaze velocity tutorial: https://github.com/pupil-labs/pupil-tutorials/blob/master/05_visualize_gaze_velocity.ipynb  

Requirements:<br>
1- Python Development Environment. <br>
2- Fixations data collected from Pupil Labs Eye Tracker in CSV format. <br>
3- Timestamps included in your data. <br> <br> 

Setup:<br> 
Download first Anaconda: https://www.anaconda.com/<br> 
After the download has been completed, you can run Anaconda and lunch Spyder. If the launch button does not show but an Install button shows instead, make sure to install Spyder and then run it. After you open Spyder, you can drag and drop the file saccadeDetection.py included here into Spyder and save the file in the same directory where your data is located. <br> <br> 

Run the python script:<br>
Once you run the script you will see printed the Number of Saccades (NOS) and Saccade Mean Velocity (SMV) in Spyder console. You will also see some plots generated. 


