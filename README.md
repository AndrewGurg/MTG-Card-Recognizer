# MTG-Card-Recognizer

A program for finding and reading a magic card from a video stream or image in front of a clear background.

## How to Run

For a video stream:

> `python Card_Detection.py -m video` or
> 
>  `python Card_Detection.py --mode video`

Add `-w 1` or `--webcam 1` to change which webcam is used (0 is default webcam).

For a single image: 

> `python Card_Decection.py -m image -f filename.jpg` or
> 
> `python Card_Decection.py --mode image --filename filename.jpg`
