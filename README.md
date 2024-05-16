# Digitalization of Entity Relationship Diagrams
Bachelor Thesis, FEI STU Bratislava

Detection and classification of shapes (rectangles, triangles and diamonds) in images using Python and OpenCV library. Application is also able to detect interconnections between the shapes
and searching whitch shapes are connected by the line (can be more than 2 based on the line). For detection and recognition of text is used Keras-OCR library. 

## Run application
Required: Python 3.10 

### Windows:
1. Create virtual environment: `py -3.10 -m venv venv`
2. Activate the environment: `venv\Scripts\activate.bat`
3. Install requirements: `pip install -r requirements.txt`
4. Run the application: `py digitalization_of_erd.py`
     - optional arguments:  
       --help: show help message  
       --demo: run demo of the application  
       --imgs-path: set path to input images  
       --resized-imgs-path: set path to the location where resized images will be stored  
       --shapes-path: set path to the location where images with detected shapes will be stored  
       --lines-path: set path to the location where images with detected lines will be stored  
       --json-path: set path to the location where output JSON data will be stored
5. Generated HTML file is located in root folder of the project
