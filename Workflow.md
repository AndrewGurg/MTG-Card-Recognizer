# Card Detection Workflow

### [Receive Image from Webcam](https://www.geeksforgeeks.org/real-time-edge-detection-using-opencv-python/)

### Pre-process image
- Greyscale
- Blur
- [Perspective Transformation](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html)

### Find Bounding Rectangle
- [Canny-Edge](https://www.geeksforgeeks.org/real-time-edge-detection-using-opencv-python/)
- [Contour](https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html)
- Find Card using Hierarchy
- [Rotation](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html)
- [Draw Rectangle](https://docs.opencv.org/3.4/d6/d6e/group__imgproc__draw.html#ga746c0625f1781f1ffc9056259103edbc)

### Find Card Name / Set
- Segment rectangle to isolate nameplate/set code
- [OCR](https://www.geeksforgeeks.org/text-detection-and-extraction-using-opencv-and-ocr/) on nameplate
- [OCR](https://www.geeksforgeeks.org/text-detection-and-extraction-using-opencv-and-ocr/) on set code and number

### Find Card
- Lookup name and set code/number in database

