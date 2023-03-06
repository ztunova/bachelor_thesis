# contour
# enlarged contour - for line detection (convex hull of contour)
# hierarchy
# type rectangle, ellipse, triangle, diamond
# bounding rectangle
import json


class ImageResult:
    # zoznam tvarov
    # zoznam ciar

    def __init__(self):
        self.objects = []
        self.connections = []

    def to_json(self):
        return self.__dict__

    def add_object(self, new_object):
        self.objects.append(new_object)

    def add_connection(self, new_connection):
        self.connections.append(new_connection)


class Dto:
    # ID
    # typ
    # popis
    # stred

    def __init__(self):
        pass

    def to_json(self):
        # return json.dumps(self.__dict__, indent=4)
        return self.__dict__


class Shape:
    def __init__(self, contour, hierarchy, shape_name, bounding_rectangle):
        self.id = None
        self.text = ""
        self.contour = contour
        self.enlarged_contour = None
        self.hierarchy = hierarchy

        # rectangle, ellipse, diamont, triangle
        self.shape_name = shape_name
        self.bounding_rectangle = bounding_rectangle
        self.bounding_ellipse = None
        self.convex_hull = None
        self.shape_centre = None

    def to_dto(self, id):
        dto = Dto()
        dto.__setattr__("ID", id)
        dto.__setattr__("type", self.shape_name)
        dto.__setattr__("name", self.text)
        dto.__setattr__("centre", self.shape_centre)

        return dto

    def set_id(self, id):
        self.id = id

    def set_text(self, text):
        self.text = text

    def set_shape_centre(self, centre):
        self.shape_centre = centre

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
