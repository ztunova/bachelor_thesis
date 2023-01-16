# contour
# hierarchy
# type rectangle, ellipse, triangle, diamond
# bounding rectangle

class Shape:
    def __init__(self, contour, hierarchy, shape_name, bounding_rectangle):
        self.contour = contour
        self.hierarchy = hierarchy
        self.shape_name = shape_name
        self.bounding_rectangle = bounding_rectangle
