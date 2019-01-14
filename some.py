import numpy
import cv2
import pbcvt # your module, also the name of your compiled dynamic library file w/o the extensiona = numpy.array([[1., 2., 3.]])
b = numpy.array([[1.],
                [2.],
                [3.]])
background_subtractor = pbcvt.apply(a)
print(background_subtractor)
print(pbcvt.dot(a, b)) # should print [[14.]]
print(pbcvt.dot2(a, b)) # should also print [[14.]]
