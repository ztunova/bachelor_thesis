# contour
# enlarged contour - for line detection (convex hull of contour)
# hierarchy
# type rectangle, ellipse, triangle, diamond
# bounding rectangle


class Shape:
    def __init__(self, contour, hierarchy, shape_name, bounding_rectangle):
        self.contour = contour
        self.enlarged_contour = None
        self.hierarchy = hierarchy
        self.shape_name = shape_name
        self.bounding_rectangle = bounding_rectangle


class Line:
    def __init__(self, contour, edge_points, color):
        self.contour = contour
        self.edge_points = edge_points
        self.color = color
        self.connecting_shapes = []

    def set_contour(self, new_contour):
        self.contour = new_contour

    def set_edge_points(self, new_edge_points):
        self.edge_points = new_edge_points
