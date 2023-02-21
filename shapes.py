# contour
# enlarged contour - for line detection (convex hull of contour)
# hierarchy
# type rectangle, ellipse, triangle, diamond
# bounding rectangle


class Shape:
    def __init__(self, contour, hierarchy, shape_name, bounding_rectangle):
        self.text = ""
        self.contour = contour
        self.enlarged_contour = None
        self.hierarchy = hierarchy
        self.shape_name = shape_name
        self.bounding_rectangle = bounding_rectangle
        self.bounding_ellipse = None
        self.convex_hull = None

    def set_text(self, text):
        self.text = text

    def set_enlarged_contour(self, enlarged_contour):
        self.enlarged_contour = enlarged_contour

    def set_bounding_ellipse(self, ellipse):
        self.bounding_ellipse = ellipse

    def set_convex_hull(self, hull):
        self.convex_hull = hull


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
