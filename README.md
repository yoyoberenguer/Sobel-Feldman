# Sobel-Feldman, Prewitt, Canny 
Sobel–Feldman filter

Python implementation of Sobel Feldman algorithm also known as edge detection algorithm.
The program contains 4 differents algorithms (4 different methods Gx and Gy decomposed as the products of an averaging and a differentiation kernel etc) 

WIKIPEDIA

The Sobel operator, sometimes called the Sobel–Feldman operator or Sobel filter, is used in image processing and computer vision, particularly within edge detection algorithms where it creates an image emphasising edges. It is named after Irwin Sobel and Gary Feldman, colleagues at the Stanford Artificial Intelligence Laboratory (SAIL). Sobel and Feldman presented the idea of an "Isotropic 3x3 Image Gradient Operator" at a talk at SAIL in 1968.
Technically, it is a discrete differentiation operator, computing an approximation of the gradient of the image intensity function. At each point in the image, the result of the Sobel–Feldman operator is either the corresponding gradient vector or the norm of this vector. The Sobel–Feldman operator is based on convolving the image with a small, separable, and integer-valued filter in the horizontal and vertical directions and is therefore relatively inexpensive in terms of computations. On the other hand, the gradient approximation that it produces is relatively crude, in particular for high-frequency variations in the image.

# Formulation
The operator uses two 3×3 kernels which are convolved with the original image to calculate approximations of the derivatives – one for horizontal changes, and one for vertical. If we define A as the source image, and Gx and Gy are two images which at each point contain the horizontal and vertical derivative approximations respectively, the computations are as follows:

![alt text](https://github.com/yoyoberenguer/Sobel-Feldman/blob/master/Assets/Graphics/sobel1.png)

where * here denotes the 2-dimensional signal processing convolution operation.
Since the Sobel kernels can be decomposed as the products of an averaging and a differentiation kernel, they compute the gradient with smoothing. For example, 
Gx can be written as

![alt text](https://github.com/yoyoberenguer/Sobel-Feldman/blob/master/Assets/Graphics/sobel2.png)

The x-coordinate is defined here as increasing in the "right"-direction, and the y-coordinate is defined as increasing in the "down"-direction. At each point in the image, the resulting gradient approximations can be combined to give the gradient magnitude, using:

![alt text](https://github.com/yoyoberenguer/Sobel-Feldman/blob/master/Assets/Graphics/sobel3.png)

Using this information, we can also calculate the gradient's direction:

![alt text](https://github.com/yoyoberenguer/Sobel-Feldman/blob/master/Assets/Graphics/sobel4.png)

where, for example, Θ is 0 for a vertical edge which is lighter on the right side.


![alt text](https://github.com/yoyoberenguer/Sobel-Feldman/blob/master/Sobel.png)

# Prewitt 

The Prewitt operator is used in image processing, particularly within edge detection algorithms. Technically, it is a discrete differentiation operator, computing an approximation of the gradient of the image intensity function. At each point in the image, the result of the Prewitt operator is either the corresponding gradient vector or the norm of this vector. The Prewitt operator is based on convolving the image with a small, separable, and integer valued filter in horizontal and vertical directions and is therefore relatively inexpensive in terms of computations like Sobel and Kayyali operators.
On the other hand, the gradient approximation which it produces is relatively crude, in particular for high frequency variations in the image. The Prewitt operator was developed by Judith M. S. Prewitt.

# Formulation
Mathematically, the operator uses two 3×3 kernels which are convolved with the original image to calculate approximations of the derivatives - one for horizontal changes, and one for vertical. If we define 
A as the source image, and Gx and Gy. 
are two images which at each point contain the horizontal and vertical derivative approximations, the latter are computed as:

![alt text](https://github.com/yoyoberenguer/Sobel-Feldman/blob/master/Assets/Graphics/Prewitt1.png)

The x-coordinate is defined here as increasing in the "right"-direction, and the y-coordinate is defined as increasing in the "down"-direction. At each point in the image, the resulting gradient approximations can be combined to give the gradient magnitude, using:

![alt text](https://github.com/yoyoberenguer/Sobel-Feldman/blob/master/Assets/Graphics/Prewitt2.png)

Using this information, we can also calculate the gradient's direction:

![alt text](https://github.com/yoyoberenguer/Sobel-Feldman/blob/master/Assets/Graphics/Prewitt3.png)


![alt text](https://github.com/yoyoberenguer/Sobel-Feldman/blob/master/Prewitt.png)


# Canny 

The Canny edge detector is an edge detection operator that uses a multi-stage algorithm to detect a wide range of edges in images. It was developed by John F. Canny in 1986. Canny also produced a computational theory of edge detection explaining why the technique works. 

# Process of Canny edge detection algorithm
The Process of Canny edge detection algorithm can be broken down to 5 different steps: 
Apply Gaussian filter to smooth the image in order to remove the noise
Find the intensity gradients of the image
Apply non-maximum suppression to get rid of spurious response to edge detection
Apply double threshold to determine potential edges
Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.
# Gaussian filter
Since all edge detection results are easily affected by image noise, it is essential to filter out the noise to prevent false detection caused by noise. To smooth the image, a Gaussian filter is applied to convolve with the image. This step will slightly smooth the image to reduce the effects of obvious noise on the edge detector. 
The equation for a Gaussian filter kernel of size (2k+1)×(2k+1) is given by

![alt text](https://github.com/yoyoberenguer/Sobel-Feldman/blob/master/Assets/Graphics/Canny1.png)

Here is an example of a 5×5 Gaussian filter, used to create the adjacent image, with 
σ = 1.4. (The asterisk denotes a convolution operation.) 

![alt text](https://github.com/yoyoberenguer/Sobel-Feldman/blob/master/Assets/Graphics/Canny2.png)

It is important to understand that the selection of the size of the Gaussian kernel will affect the performance of the detector. The larger the size is, the lower the detector’s sensitivity to noise. Additionally, the localization error to detect the edge will slightly increase with the increase of the Gaussian filter kernel size. A 5×5 is a good size for most cases, but this will also vary depending on specific situations. 

![alt text](https://github.com/yoyoberenguer/Sobel-Feldman/blob/master/Canny.png)

