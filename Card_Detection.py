import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt
import easyocr
import argparse

# PARAMETERS:
# ksize, sigmaX, upper and lower thresholds


def canny_edge_detection(frame):
    # Greyscale the image
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
      
    # Apply Gaussian blur to reduce noise and smoothen edges 
    # Can adjust ksize and sigmaX to increase blur
    blurred = cv.GaussianBlur(src=gray, ksize=(3, 5), sigmaX=0.9) 

    # Perform Canny edge detection 
    # Can adjust thresholds to determine valid edges
    edges = cv.Canny(blurred, 70, 135) 

    return blurred, edges


def find_contours(img, edges):
    # Find the contours
    ret, thresh_img = cv.threshold(edges,91,255,cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    img_cnt = img.copy()
    cv.drawContours(img_cnt, contours, -1, (0,255,0), -1)

    return img_cnt, contours


def find_boxes(img, contours):
    approx = -1
    hasBox = False
    for cnt in contours:
        # Draw the boxes around the edges
        if cv.contourArea(cnt) > 10**4.5:
            epsilon = 0.1*cv.arcLength(cnt,True)
            approx = cv.approxPolyDP(cnt,epsilon,True)
            if len(approx) >=4:
                cv.drawContours(img, [approx], 0, (0,255,0), 2)
                hasBox = True
            break
    return img, hasBox, approx


def arrange_points(ptarray):

    ptarray = np.squeeze(ptarray)
    # Arrange the points in ptarray in order of TL, TR, BL, BR
    # Find topmost point
    top = 0
    for i in range(len(ptarray)):
        if i != top:
            if ptarray[i][1] < ptarray[top][1]:
                top = i
    
    topval = ptarray[top]
    ptarray = np.delete(ptarray, top, 0)

    close_dist = np.inf
    closest = 0
    # if next closest is to the right => top is TL
    # Next closest left => top is TR
    for i in range(len(ptarray)):
        dist = np.linalg.norm(ptarray[i] - topval)
        if dist < close_dist:
            closest = i
            close_dist = dist

    TL = topval if ptarray[closest][0] > topval[0] else ptarray[closest]
    TR = ptarray[closest] if ptarray[closest][0] > topval[0] else topval
    
    ptarray = np.delete(ptarray, closest, 0)

    # Find leftmost point on the bottom
    left = 0
    for i in range(len(ptarray)):
        if ptarray[i][0] < ptarray[left][0]:
            left = i
    
    BL = ptarray[left]
    BR = ptarray[1 - left]

    return np.stack((TL, TR, BL, BR))

def get_perspective(image, box):
    # Source points
    # Top left, top right, bottom left, bottom right
    pts1 = np.float32(box)
    pts2 = np.float32([[0,0],[500,0],[0,700],[500,700]])
    
    M = cv.getPerspectiveTransform(pts1,pts2)
    
    card_persp = cv.warpPerspective(image,M,(500,700))

    return card_persp


def get_nameplate(image):
    height = image.shape[0]
    width = image.shape[1]
    nameplate = image[0:height//9, 0:width]
    return nameplate

def get_setnum(image):
    height = image.shape[0]
    width = image.shape[1]
    nameplate = image[(height - height//9):height, 0:width//2]
    return nameplate


def find_card_outline(image):
    # Perform Canny edge detection on the frame 
    blurred, edges = canny_edge_detection(image) 
    
    # Find the contours from the edges
    img_cnt, contours = find_contours(image, edges)
    
    # Make boxes around cards on image
    img_w_boxes, hasBox, boxed_card = find_boxes(image, contours)

    # Display the original frame and the edge-detected frame 
    #cv.imshow("Original", image) 
    #cv.imshow("Blurred", blurred) 
    #cv.imshow("Edges", edges) 
    #cv.imshow("Contours", img_cnt)
    cv.imshow("Boxes", img_w_boxes)
    return boxed_card, hasBox

def read_text(reader, image):
    text = reader.readtext(image, allowlist="abcdefghigjlmnopqrstuvwxyz0123456789", text_threshold=0.8)
    # for t in text:
    #      print("Text: ", t[1])
    return text[0][1] if len(text) >= 1 else ""

def read_card_from_box(reader, image, boxed_card):
    fixedbox = arrange_points(boxed_card)
    card_persp = get_perspective(image, fixedbox)
    cv.imshow("Card", card_persp)
    # Find the nameplate
    nameplate = get_nameplate(card_persp)
    cv.imshow("Nameplate", nameplate)
    # Read the nameplate
    cardname = read_text(reader, nameplate)
    # Find the set information
    setinfo = get_setnum(card_persp)
    cv.imshow("Set Info", setinfo)
    # Read the setinfo
    #settext= read_text(reader, setinfo)

    return cardname


def VideoStream(webcam_no=0): 
    # Open the default webcam  
    cap = cv.VideoCapture(webcam_no)
    fps = cap.get(cv.CAP_PROP_FPS)

    # Initialize easyOCR Reader
    reader = easyocr.Reader(['en'], gpu=False)

    # Keep track of the frame number
    frameno = 0
    reads_per_sec = 2
    while True: 
        # Read a frame from the webcam 
        ret, frame = cap.read() 
        if not ret: 
            print('Image not captured') 
            break
        
        # Find the card on the frame and isolate it
        boxed_card, hasBox = find_card_outline(frame)
        # Only read a few times every second
        if(frameno % (fps // reads_per_sec) == 0):
            # Read the isolated card's name from the nameplate
            if(hasBox):
                cardname = read_card_from_box(reader, frame, boxed_card)
                print("Card Name: %s            \r"%cardname, end="")

        # Increment or reset frame count
        frameno = frameno + 1 if frameno < fps-1 else 0

        # Exit the loop when 'q' key is pressed 
        if cv.waitKey(1) & 0xFF == ord('q'): 
            break
      
    # Release the webcam and close the windows 
    cap.release() 
    cv.destroyAllWindows()
    return

def SingleImage(filename):
    # Read the image given filename
    im = cv.imread('images/' + filename)
    assert im is not None, "file could not be read, check with os.path.exists()"
    
    # Initialize easyOCR Reader
    reader = easyocr.Reader(['en'], gpu=False)

    # Find the card on the frame and isolate it
    boxed_card, hasBox = find_card_outline(im)
    # Read the isolated card's name from the nameplate
    if(hasBox):
        cardname = read_card_from_box(reader, im, boxed_card)
        print("Card Name: %s            \r"%cardname, end="")

    if cv.waitKey(0) & 0xff == 27:  
        cv.destroyAllWindows()  
    return


if __name__ == "__main__":
    modes = {'video': VideoStream,'image': SingleImage}
    parser = argparse.ArgumentParser()

    # Options for video/image mode and filename input
    parser.add_argument('-m', '--mode',
                        choices=modes, 
                        help='Video stream or single image mode',
                        required=True, type=str)
    
    parser.add_argument('-f', '--filename', 
                        help='Filename for single image mode',
                        required=False, type=str)
    
    parser.add_argument('-w', '--webcam', 
                        help='Webcam number selected (defult 0)',
                        required=False, type=int, default=0)
    
    # Runs specified mode with optional argument
    args = parser.parse_args()
    modes[args.mode](args.filename if args.mode=='image' else args.webcam)
