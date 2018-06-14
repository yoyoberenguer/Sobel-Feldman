import pygame
import numpy
import time
import math

__author__ = "Yoann Berenguer"
__copyright__ = "Copyright 2007."
__credits__ = ["Yoann Berenguer"]
__license__ = "MIT License"
__version__ = "1.0.0"
__maintainer__ = "Yoann Berenguer"
__email__ = "yoyoberenguer@hotmail.com"
__status__ = "Demo"


class Sobel4:
    """
    Sobel algorithm version 4
    """

    def __init__(self, surface_, array_):

        self.gx = numpy.array(([-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]))
        self.gy = numpy.array(([-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]))
        self.kernel_half = 1
        self.surface = surface_
        self.shape = array_.shape
        self.array = array_
        self.source_array = numpy.zeros((self.shape[0], self.shape[1], 3))
        self.threshold = 0

    def run(self):

        for y in range(0, self.shape[1]):

            for x in range(0, self.shape[0]):
                r_gx, r_gy = 0, 0
                for kernel_offset_y in range(-self.kernel_half, self.kernel_half + 1):

                    for kernel_offset_x in range(-self.kernel_half, self.kernel_half + 1):
                        xx = x + kernel_offset_x
                        yy = y + kernel_offset_y

                        try:
                            color = self.surface.get_at((xx, yy))
                            k = self.gx[kernel_offset_x + self.kernel_half,
                                        kernel_offset_y + self.kernel_half]
                            r_gx += color[0] * k
                            k = self.gy[kernel_offset_x + self.kernel_half,
                                        kernel_offset_y + self.kernel_half]
                            r_gy += color[0] * k

                        except IndexError:

                            k = self.gx[kernel_offset_x + self.kernel_half,
                                        kernel_offset_y + self.kernel_half]
                            r_gx += 128 * k
                            k = self.gy[kernel_offset_x + self.kernel_half,
                                        kernel_offset_y + self.kernel_half]
                            r_gy += 128 * k

                magnitude = math.sqrt(r_gx ** 2 + r_gy ** 2)
                # update the pixel if the magnitude is above threshold else black pixel
                self.source_array[x, y] = magnitude if magnitude > self.threshold else 0
        # cap the values
        numpy.putmask(self.source_array, self.source_array > 255, 255)
        numpy.putmask(self.source_array, self.source_array < 0, 0)
        return self.source_array


class Sobel3:
    # Sobel algoritm with Gx and Gy decomposed as the products.
    # This algorithm is slower than the version 2

    def __init__(self, surface_, array_):

        # Gx vertical / horizontal
        self.gx_v = numpy.array(([1, 2, 1]))
        self.gx_h = numpy.array(([1, 0, -1]))
        # Gy vertical/ horizontal
        self.gy_v = numpy.array(([1, 0, -1]))
        self.gy_h = numpy.array(([1, 2, 1]))

        self.array = array_
        self.surface = surface_
        self.shape = array_.shape
        self.threshold = 0

    def horizontal(self):
        self.source_array = numpy.zeros((self.shape[0], self.shape[1], 3))
        self.array_gxh = numpy.zeros((self.shape[0], self.shape[1], 3))
        self.array_gyh = numpy.zeros((self.shape[0], self.shape[1], 3))
        for y in range(0, self.shape[1]):

            for x in range(0, self.shape[0]):

                try:
                    # Horizontal kernel Gx_h and Gy_h
                    data = self.array[x - 1:x + 2, y, :1].reshape(3)
                    h12 = sum(numpy.multiply(data, self.gx_h))
                    h13 = sum(numpy.multiply(data, self.gy_h))
                except (ValueError, IndexError) as e:
                    h12 = 0
                    h13 = 0
                self.array_gxh[x, y] = h12
                self.array_gyh[x, y] = h13

    def vertical(self):
        self.source_array = numpy.zeros((self.shape[0], self.shape[1], 3))
        for y in range(0, self.shape[1]):

            for x in range(0, self.shape[0]):
                # Apply both kernels at once for each pixels
                try:
                    # Vertical kernel Gx_v and gy_v
                    v1 = sum(numpy.multiply(self.array_gxh[x, y - 1: y + 2, :1].reshape(3), self.gx_v))
                    v2 = sum(numpy.multiply(self.array_gyh[x, y - 1: y + 2, :1].reshape(3), self.gy_v))
                except (ValueError, IndexError) as e:
                    v1, v2 = 0, 0
                magnitude = math.sqrt(v1 ** 2 + v2 ** 2)
                # update the pixel if the magnitude is above threshold else black pixel
                self.source_array[x, y] = magnitude if magnitude > self.threshold else 0

        # cap the values
        numpy.putmask(self.source_array, self.source_array > 255, 255)
        numpy.putmask(self.source_array, self.source_array < 0, 0)
        return self.source_array

    def run(self):
        self.horizontal()
        return self.vertical()


class Sobel2:
    """

    Sobel algorithm version 2

    WIKIPEDIA
    The Sobel operator, sometimes called the Sobel–Feldman operator or Sobel filter,
    is used in image processing and computer vision, particularly within edge detection
     algorithms where it creates an image emphasising edges.
     The operator uses two 3×3 kernels which are convolved with the original image to
     calculate approximations of the derivatives – one for horizontal changes, and one for vertical.
     If we define A as the source image, and Gx and Gy are two images which at each point contain
     the horizontal and vertical derivative approximations respectively, the computations are as follows:

            |+1 0 -1|               |+1 +2 +1|
     Gx =   |+2 0 -2| * A  and Gy = | 0  0  0| * A
            |+1 0 -1|               |-1 -2 -1|

     Where * here denotes the 2-dimensional signal processing convolution operation
     Since the Sobel kernels can be decomposed as the products of an averaging and a differentiation kernel,
     they compute the gradient with smoothing. For example,Gx can be written as
        |+1 0 -1|     |1|
        |+2 0 -2|  =  |2| [+1 0 -1]
        |+1 0 -1|     |1|

     G = sqrt(Gx ** 2 + Gy **2)
     gradient direction
     alpha = atan(Gy/Gx
     """

    def __init__(self, surface_, array_):

        self.gx = numpy.array(([-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]))
        self.gy = numpy.array(([-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]))
        self.kernel_half = 1
        self.surface = surface_
        self.shape = array_.shape
        self.array = array_
        self.source_array = numpy.zeros((self.shape[0], self.shape[1], 3))
        # Threshold at 50%
        self.threshold = 0

    def run(self):

        # Starting at row 1, finishing at shape[0] - 1 due to the size of the kernel
        # and to avoid IndexError
        for y in range(0, self.shape[1]):

            for x in range(0, self.shape[0]):
                # Apply both kernels at once for each pixels
                try:
                    # Horizontal kernel Gx
                    s1 = sum(sum(numpy.multiply(self.array[x - 1:x + 2, y - 1:y + 2][0], self.gx)))
                    # Vertical kernel Gy
                    s2 = sum(sum(numpy.multiply(self.array[x - 1:x + 2, y - 1:y + 2][0], self.gy)))
                except:
                    s1, s2 = 0, 0
                magnitude = math.sqrt(s1 ** 2 + s2 ** 2)
                # update the pixel if the magnitude is above threshold else black pixel
                self.source_array[x, y] = magnitude if magnitude > self.threshold else 0
        # cap the values
        numpy.putmask(self.source_array, self.source_array > 255, 255)
        numpy.putmask(self.source_array, self.source_array < 0, 0)
        return self.source_array



class Sobel:
    """

    Sobel algorithm version 1

    WIKIPEDIA
    The Sobel operator, sometimes called the Sobel–Feldman operator or Sobel filter,
    is used in image processing and computer vision, particularly within edge detection
     algorithms where it creates an image emphasising edges.
     The operator uses two 3×3 kernels which are convolved with the original image to
     calculate approximations of the derivatives – one for horizontal changes, and one for vertical.
     If we define A as the source image, and Gx and Gy are two images which at each point contain
     the horizontal and vertical derivative approximations respectively, the computations are as follows:

            |+1 0 -1|               |+1 +2 +1|
     Gx =   |+2 0 -2| * A  and Gy = | 0  0  0| * A
            |+1 0 -1|               |-1 -2 -1|

     Where * here denotes the 2-dimensional signal processing convolution operation
     Since the Sobel kernels can be decomposed as the products of an averaging and a differentiation kernel,
     they compute the gradient with smoothing. For example,Gx can be written as
        |+1 0 -1|     |1|
        |+2 0 -2|  =  |2| [+1 0 -1]
        |+1 0 -1|     |1|

     G = sqrt(Gx ** 2 + Gy **2)
     gradient direction
     alpha = atan(Gy/Gx
     """

    def __init__(self, surface_, array_):

        self.sobel_h = numpy.array(([-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]))
        self.sobel_v = numpy.array(([-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]))
        self.kernel_half = 1
        self.array = array_
        self.surface = surface_
        self.shape = array_.shape
        self.source_array = numpy.zeros((self.shape[0], self.shape[1], 3))
        self.kernel_length = len(self.sobel_h)
        self.kernel_weight = numpy.sum(self.sobel_h)

    def horizontal(self):
        self.source_array = numpy.zeros((self.shape[0], self.shape[1], 3))
        for y in range(0, self.shape[1]):

            for x in range(0, self.shape[0]):

                r = 0

                for kernel_offset_y in range(-self.kernel_half, self.kernel_half + 1):

                    for kernel_offset_x in range(-self.kernel_half, self.kernel_half + 1):

                        xx = x + kernel_offset_x
                        yy = y + kernel_offset_y

                        try:
                            color = self.surface.get_at((xx, yy))
                            k = self.sobel_h[kernel_offset_x + self.kernel_half,
                                             kernel_offset_y + self.kernel_half]
                            r += color[0] * k

                        except IndexError:

                            k = self.sobel_h[kernel_offset_x + self.kernel_half,
                                             kernel_offset_y + self.kernel_half]
                            r += 128 * k
                            pass

                self.source_array[x][y] = r, r, r

        return self.source_array

    def vertical(self):
        self.source_array = numpy.zeros((self.shape[0], self.shape[1], 3))
        for y in range(0, self.shape[1]):

            for x in range(0, self.shape[0]):

                r = 0

                for kernel_offset_y in range(-self.kernel_half, self.kernel_half + 1):

                    for kernel_offset_x in range(-self.kernel_half, self.kernel_half + 1):

                        xx = x + kernel_offset_x
                        yy = y + kernel_offset_y

                        try:
                            color = self.surface.get_at((xx, yy))
                            k = self.sobel_v[kernel_offset_x + self.kernel_half,
                                             kernel_offset_y + self.kernel_half]

                            r += color[0] * k

                        except IndexError:

                            k = self.sobel_v[kernel_offset_y + self.kernel_half,
                                             kernel_offset_x + self.kernel_half]
                            r += 128 * k
                            pass
                self.source_array[x][y] = r, r, r

        return self.source_array

    def magnitude(self, horizontal, vertical):
        magn = numpy.zeros(vertical.shape)
        for y in range(0, self.shape[1]):

            for x in range(0, self.shape[0]):
                h_r, h_g, h_b = horizontal[x, y]
                v_r, v_g, v_b = vertical[x, y]
                g_r = math.sqrt(h_r ** 2 + v_r ** 2)
                magn[x, y] = g_r, g_r, g_r
        numpy.putmask(magn, magn > 255, 255)

        return magn

    def run(self):
        horizontal = self.horizontal()
        vertical = self.vertical()
        return self.magnitude(horizontal, vertical)


if __name__ == '__main__':
    numpy.set_printoptions(threshold=numpy.nan)

    SIZE = (800, 600)
    SCREENRECT = pygame.Rect((0, 0), SIZE)
    pygame.init()
    SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.RESIZABLE, 32)
    TEXTURE1 = pygame.image.load("Assets\\Graphics\\seychelles_gray.jpg").convert()
    TEXTURE1 = pygame.transform.smoothscale(TEXTURE1, (SIZE[0], SIZE[1] >> 1))
    # Texture re-scale to create extra data (padding) on each sides
    PADDING = pygame.transform.smoothscale(TEXTURE1, (SIZE[0] + 8, (SIZE[1] >> 1) + 8))

    # 13 seconds for 800x300
    # Sob = Sobel(TEXTURE1, pygame.surfarray.array3d(TEXTURE1))
    # 8 seconds for 800x300
    # Sob = Sobel2(TEXTURE1, pygame.surfarray.array3d(TEXTURE1))
    # 10.9 seconds for 800x300
    # Sob = Sobel3(TEXTURE1, pygame.surfarray.array3d(TEXTURE1))
    # 7.5 seconds for 800x300
    Sob = Sobel4(TEXTURE1, pygame.surfarray.array3d(TEXTURE1))

    FRAME = 0
    clock = pygame.time.Clock()
    STOP_GAME = False
    PAUSE = False

    while not STOP_GAME:

        pygame.event.pump()

        while PAUSE:
            event = pygame.event.wait()
            keys = pygame.key.get_pressed()
            if keys[pygame.K_PAUSE]:
                PAUSE = False
                pygame.event.clear()
                keys = None
            break

        for event in pygame.event.get():

            keys = pygame.key.get_pressed()

            if event.type == pygame.QUIT or keys[pygame.K_ESCAPE]:
                print('Quitting')
                STOP_GAME = True

            elif event.type == pygame.MOUSEMOTION:
                MOUSE_POS = event.pos

            elif keys[pygame.K_PAUSE]:
                PAUSE = True
                print('Paused')

        t = time.time()
        array = Sob.run()
        print(time.time() - t)

        surface = pygame.surfarray.make_surface(array)

        SCREEN.fill((0, 0, 0, 0))
        SCREEN.blit(TEXTURE1, (0, 0))
        SCREEN.blit(surface, (0, SIZE[1] // 2))

        pygame.display.flip()
        TIME_PASSED_SECONDS = clock.tick(120)
        FRAME += 1

    pygame.quit()
