# Playing Set with Computer Vision

Uses OpenCV to find SETs.

## How does it work?

### Card Segmentation
The first step is to segment the cards. We do this by thresholding the image,
then running a contour detection algorithm which gives the outline of each "region".
Then we find the largest twelve rectangular contours, which finds the cards relatively
reliably. Finally, we run a perspective transformation to unwarp the cards.

### Number Detection
To find how many shapes there are on one card, we run the contour detection again
on that single card. After filtering out contours that are too small (noise), we
simply count the number of contours.

### Color Detection
We find the average color of a segmented contour, and compare it to prerecorded values
for red, green and purple. Color detection sometimes errors, because red and purple
can be mistaken for one another under different lighting conditions.

### Shape Detection
We run OpenCV's `moments` function on one of the segmented contours, which gives various 
characteristics of the shape. We then compare the result to prerecorded values to determine
whether the card is a squiggle, diamond or ellipse.

### Shading Detection
This is done by finding how much of the card is colored. A solid-shaded card will have
more colored pixels than a striped or empty-shaded card. We also run a Sobel edge-detection
algorithm; a large number of edge pixels indicates that the card is striped.

### SET Determination
This is the easiest part - just a complete search on all possible SETs.