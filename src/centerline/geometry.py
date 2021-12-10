# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import os
import pickle

from numpy import array
from scipy.spatial import Voronoi
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.ops import unary_union
import time
from tqdm import tqdm
from multiprocess import Pool
import multiprocessing
from numba import jit

from . import exceptions


import warnings
warnings.filterwarnings("ignore")


@jit(nopython=True, cache=True)
def _ridge_is_finite(ridge):
    return -1 not in ridge


@jit(nopython=True, cache=True)
def _create_point_with_restored_coordinates(x, y, min_x, min_y):
    return (x + min_x, y + min_y)


@jit(cache=True)
def _process_ridge(data):
    ridge = data[0]
    vertices = data[1]
    input_geometry = data[2]
    min_x = data[3][0]
    min_y = data[3][1]

    if _ridge_is_finite(ridge):
        starting_point = _create_point_with_restored_coordinates(
            x=vertices[ridge[0]][0], y=vertices[ridge[0]][1], min_x=min_x, min_y=min_y)
        ending_point = _create_point_with_restored_coordinates(
            x=vertices[ridge[1]][0], y=vertices[ridge[1]][1], min_x=min_x, min_y=min_y)
        linestring = LineString((starting_point, ending_point))

        if linestring.within(input_geometry) and len(linestring.coords[0]) > 1:
            return linestring
        else:
            return None


class Centerline(MultiLineString):
    """Create a centerline object.

    The ``attributes`` are copied and set as the centerline's
    attributes.

    :param input_geometry: input geometry
    :type input_geometry: :py:class:`shapely.geometry.Polygon` or
        :py:class:`shapely.geometry.MultiPolygon`
    :param interpolation_distance: densify the input geometry's
        border by placing additional points at this distance,
        defaults to 0.5 [meter]
    :type interpolation_distance: float, optional
    :raises exceptions.InvalidInputTypeError: input geometry is not
        of type :py:class:`shapely.geometry.Polygon` or
        :py:class:`shapely.geometry.MultiPolygon`
    """

    def __init__(
        self, input_geometry=None, vertices_and_ridges=None, interpolation_distance=0.5, **attributes
    ):
        if input_geometry == None:
            return

        if isinstance(input_geometry, str):
            with open(input_geometry + '.pkl', 'rb') as f:
                self._input_geometry = pickle.load(f)
        else:
            self._input_geometry = input_geometry

        if isinstance(vertices_and_ridges, str):
            with open(vertices_and_ridges + '.pkl', 'rb') as f:
                self.vertices_and_ridges = pickle.load(f)
        else:
            self._vertices_and_ridges = vertices_and_ridges

        self._interpolation_distance = abs(interpolation_distance)

        if not self.input_geometry_is_valid():
            raise exceptions.InvalidInputTypeError

        self._min_x, self._min_y = self._get_reduced_coordinates()
        self.assign_attributes_to_instance(attributes)

        if not self.run:
            return

        if not self.multiprocess:
            super(Centerline, self).__init__(
                lines=self._construct_centerline())
        else:
            super(Centerline, self).__init__(
                lines=self._construct_centerline_multiprocess())

    @jit(cache=True)
    def input_geometry_is_valid(self):
        """Input geometry is of a :py:class:`shapely.geometry.Polygon`
        or a :py:class:`shapely.geometry.MultiPolygon`.

        :return: geometry is valid
        :rtype: bool
        """
        if isinstance(self._input_geometry, Polygon) or isinstance(
            self._input_geometry, MultiPolygon
        ):
            return True
        else:
            return False

    @jit(cache=True)
    def _get_reduced_coordinates(self):
        min_x = int(min(self._input_geometry.envelope.exterior.xy[0]))
        min_y = int(min(self._input_geometry.envelope.exterior.xy[1]))
        return min_x, min_y

    @jit(cache=True)
    def assign_attributes_to_instance(self, attributes):
        """Assign the ``attributes`` to the :py:class:`Centerline` object.

        :param attributes: polygon's attributes
        :type attributes: dict
        """
        for key in attributes:
            setattr(self, key, attributes.get(key))

    @jit(cache=True)
    def _construct_centerline_multiprocess(self):
        p = Pool(multiprocessing.cpu_count())

        vertices, ridges = self._get_voronoi_vertices_and_ridges()
        num_ridges = len(ridges)
        num_vertices = len(vertices)
        print(f"ridges={num_ridges} vertices={num_vertices}")
        linestrings = []
        shared_data = []

        for ridge in ridges:
            shared_data.append(
                (
                    ridge, vertices, self._input_geometry, (
                        self._min_x, self._min_y)
                )
            )

        # result = p.map(_process_ridge, shared_data)

        result = list(tqdm(p.imap(_process_ridge, shared_data)))

        for linestring in result:
            if linestring:
                linestrings.append(linestring)

        if len(linestrings) < 2:
            raise exceptions.TooFewRidgesError

        result = unary_union(linestrings)

        if self.save_to_file:
            self._save_to_file(result, self.save_to_file)

        return result

    @jit(cache=True)
    def _construct_centerline(self):
        if not self._vertices_and_ridges:
            vertices, ridges = self._get_voronoi_vertices_and_ridges()
        else:
            vertices, ridges = self._vertices_and_ridges

        num_ridges = len(ridges)
        num_vertices = len(vertices)
        print(f"ridges={num_ridges} vertices={num_vertices}")
        linestrings = []
        #  for ridge in tqdm(ridges, desc=f"Process: {os.getpid()}", colour="#00ff00"):
        for ridge in ridges:
            # print(f"_construct_centerline [{os.getpid()}]")
            if self._ridge_is_finite(ridge):
                starting_point = self._create_point_with_restored_coordinates(
                    x=vertices[ridge[0]][0], y=vertices[ridge[0]][1]
                )
                ending_point = self._create_point_with_restored_coordinates(
                    x=vertices[ridge[1]][0], y=vertices[ridge[1]][1]
                )
                linestring = LineString((starting_point, ending_point))

                if self._linestring_is_within_input_geometry(linestring):
                    linestrings.append(linestring)

        if len(linestrings) < 2:
            raise exceptions.TooFewRidgesError

        result = unary_union(linestrings)

        if self.save_to_file:
            self._save_to_file(result, self.save_to_file)

        # print("_construct_centerline finished!")

        return result

    def _dump_voronoi_vertices_and_ridges(self, filename):
        result = self._get_voronoi_vertices_and_ridges()
        with open(filename + '_vr.pkl', 'wb') as f:
            pickle.dump(result, f)
        return result

    @jit(cache=True)
    def _save_to_file(self, data, filename):
        with open(filename + '_centerline.pkl', 'wb') as f:
            pickle.dump(data, f)

    @jit(cache=True)
    def _get_voronoi_vertices_and_ridges(self):
        borders = self._get_densified_borders()

        voronoi_diagram = Voronoi(borders)
        vertices = voronoi_diagram.vertices
        ridges = voronoi_diagram.ridge_vertices

        return vertices, ridges

    def _ridge_is_finite(self, ridge):
        return -1 not in ridge

    def _create_point_with_restored_coordinates(self, x, y):
        return (x + self._min_x, y + self._min_y)

    @jit(cache=True)
    def _linestring_is_within_input_geometry(self, linestring):
        return (
            linestring.within(self._input_geometry)
            and len(linestring.coords[0]) > 1
        )

    @jit(cache=True)
    def _get_densified_borders(self):
        polygons = self._extract_polygons_from_input_geometry()
        points = []
        for polygon in polygons:
            points += self._get_interpolated_boundary(polygon.exterior)
            if self._polygon_has_interior_rings(polygon):
                for interior in polygon.interiors:
                    points += self._get_interpolated_boundary(interior)

        return array(points)

    # @jit(forceobj=True)
    def _extract_polygons_from_input_geometry(self):
        if isinstance(self._input_geometry, MultiPolygon):
            return (polygon for polygon in self._input_geometry)
        else:
            return (self._input_geometry,)

    @jit(cache=True)
    def _polygon_has_interior_rings(self, polygon):
        return len(polygon.interiors) > 0

    @jit(cache=True)
    def _get_interpolated_boundary(self, boundary):
        line = LineString(boundary)

        first_point = self._get_coordinates_of_first_point(line)
        last_point = self._get_coordinates_of_last_point(line)

        intermediate_points = self._get_coordinates_of_interpolated_points(
            line
        )

        return [first_point] + intermediate_points + [last_point]

    @jit(cache=True)
    def _get_coordinates_of_first_point(self, linestring):
        return self._create_point_with_reduced_coordinates(
            x=linestring.xy[0][0], y=linestring.xy[1][0]
        )

    @jit(cache=True)
    def _get_coordinates_of_last_point(self, linestring):
        return self._create_point_with_reduced_coordinates(
            x=linestring.xy[0][-1], y=linestring.xy[1][-1]
        )

    @jit(cache=True)
    def _get_coordinates_of_interpolated_points(self, linestring):
        intermediate_points = []
        interpolation_distance = self._interpolation_distance
        line_length = linestring.length
        while interpolation_distance < line_length:
            point = linestring.interpolate(interpolation_distance)
            reduced_point = self._create_point_with_reduced_coordinates(
                x=point.x, y=point.y
            )
            intermediate_points.append(reduced_point)
            interpolation_distance += self._interpolation_distance

        return intermediate_points

    @jit(cache=True)
    def _create_point_with_reduced_coordinates(self, x, y):
        return (x - self._min_x, y - self._min_y)
