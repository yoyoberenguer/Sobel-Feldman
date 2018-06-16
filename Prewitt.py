
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


class Prewitt:
    """
    WIKIPEDIA
    The Prewitt operator is used in image processing,
    particularly within edge detection algorithms.
    Technically, it is a discrete differentiation operator,
    computing an approximation of the gradient of the image intensity function.
    At each point in the image, the result of the Prewitt operator is
    either the corresponding gradient vector or the norm of this vector.
    The Prewitt operator is based on convolving the image with a small,
    separable, and integer valued filter in horizontal and vertical
    directions and is therefore relatively inexpensive in terms of computations
    like Sobel and Kayyali[1] operators.
    On the other hand, the gradient approximation which it produces
    is relatively crude, in particular for high frequency variations in the image.
    The Prewitt operator was developed by Judith M. S. Prewitt.
    """

    def __init__(self, surface_, array_):

        self.gx = numpy.array(([-1, 0, 1],
                               [-1, 0, 1],
                               [-1, 0, 1]))
        self.gy = numpy.array(([-1, -1, -1],
                               [0, 0, 0],
                               [1, 1, 1]))
        self.kernel_half = 1
        self.surface = surface_
        self.shape = array_.shape
        self.array = array_
        self.source_array = numpy.zeros((self.shape[0], self.shape[1], 3))
        self.threshold = 0

    def run(self):

        for y in range(2, self.shape[1]-2):

            for x in range(2, self.shape[0]-2):
                r_gx, r_gy = 0, 0
                for kernel_offset_y in range(-self.kernel_half, self.kernel_half + 1):

                    for kernel_offset_x in range(-self.kernel_half, self.kernel_half + 1):

                        xx = x + kernel_offset_x
                        yy = y + kernel_offset_y
                        color = self.surface.get_at((xx, yy))
                        # print(kernel_offset_y, kernel_offset_x)
                        if kernel_offset_x != 0:

                            k = self.gx[kernel_offset_x + self.kernel_half,
                                        kernel_offset_y + self.kernel_half]
                            r_gx += color[0] * k

                        if kernel_offset_y != 0:
                            k = self.gy[kernel_offset_x + self.kernel_half,
                                        kernel_offset_y + self.kernel_half]
                            r_gy += color[0] * k

                magnitude = math.sqrt(r_gx ** 2 + r_gy ** 2)
                # update the pixel if the magnitude is above threshold else black pixel
                self.source_array[x, y] = magnitude if magnitude > self.threshold else 0
        # cap the values
        numpy.putmask(self.source_array, self.source_array > 255, 255)
        numpy.putmask(self.source_array, self.source_array < 0, 0)
        return self.source_array


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

    pygame.display.set_caption('Prewitt algorithm')
    Prw = Prewitt(TEXTURE1, pygame.surfarray.array3d(TEXTURE1))

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
        array = Prw.run()
        print(time.time() - t)

        surface = pygame.surfarray.make_surface(array)

        SCREEN.fill((0, 0, 0, 0))
        SCREEN.blit(TEXTURE1, (0, 0))
        SCREEN.blit(surface, (0, SIZE[1] // 2))

        pygame.display.flip()
        TIME_PASSED_SECONDS = clock.tick(120)
        FRAME += 1

    pygame.quit()
