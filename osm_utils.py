#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OSM utilities.

This file is part of the Truck Detection Algorithm.

EDC consortium / H. Fisser
"""
import os
import subprocess
import sys
import time
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from edc import setup_environment_variables
from OSMPythonTools.overpass import Overpass, overpassQueryBuilder
from shapely.geometry import shape
from tqdm import tqdm
from xcube_geodb.core.geodb import GeoDBClient


logging.basicConfig(filename='truck_detector.log')
LOGGER = logging.getLogger(__name__)

class Osm(object):
    """OSM roads class."""

    def __init__(
        self,
        tiles_info,
        osm_values,
        roads_buffer=25,
        offset=2,
        osm_key="highway",
        element_type=["way", "relation"],
        geodb_collection="osm_europe_roads",
        geodb_database="geodb_a2b85af8-6b99-4fa2-acf6-e87d74e40431",
    ):
        """Constructs the necessary attributes for the Osm object.

        Args:
            tiles_info (GeoDataFrame): A GeoDataFrame containing the tiles' id, geometry, shape, and transform.
            osm_values (list): A list of osm road types to be queried.
            roads_buffer (int, optional): The value used to buffer the motorway from linestrings to polygons. Defaults to 25 (meters).
            offset (int, optional): The value subtracted from roads_buffer to buffer the primary and the truck . Defaults to 2 (meters).
            osm_key (str, optional): Key to query in the Osm. Defaults to "highway".
            element_type (list, optional): Variables to query in the Osm. Defaults to ["way", "relation"].
            geodb_collection (str, optional): Name of the GeoDB collection to query.
            geodb_database (str, optional): Name of the GeoDB database to query.
        """
        self.tiles = tiles_info
        self.osm_values = osm_values
        self.roads_buffer = roads_buffer
        self.offset = offset
        self.osm_key = osm_key
        self.element_type = element_type
        self.geodb_collection = geodb_collection
        self.geodb_database = geodb_database

    def _update_roads(self, geometry, original_crs, buffer_dict, sleep_time=5, backoff_coefficient=3):
        """Query osm data via OSMPythonTools.

        Args:
            geometry (sentinelhub.geometry.Geometry): The geometry of the tile for querying.
            original_crs (str): A string showing the original crs (UTM) of the tile.
            buffer_dict (dict): A dictionary containing osm_values as keys and buffering values as values.

        Returns:
            GeoSeries: A GeoSeries containing all queried osm data as polygons.
        """

        # Initialize a list to hold osm roads Geoseries of each bbox (tile)
        osm_roads_geoseries = []

        # get bbox  from the geometry of each row in gdf  / .bbox returns BBox object representing bounding box around the geometry
        bbox = list(geometry.bbox)

        # take bbox and convert it to  bbox_osm
        bbox_osm = self._convert_to_bbox_osm(bbox)

        # initialize an empty list to populate with Geoseries of each road type as per 'osm_value'
        osm_roads_buf = []

        # initialize an empty list to populate with instances of invalid geometry
        has_errors = []

        # loop through the list of osm_values
        for osm_value in self.osm_values:

            # for each osm_value, create a list of osm_values to be used in  query selector parameter
            select = f'"{self.osm_key}"="{osm_value}"'
            select_link = select.replace(
                osm_value, f"{osm_value}_link"
            )  # query for link roads too
            select_junction = select.replace(
                osm_value, f"{osm_value}_junction"
            )  # query for  junctions too

            # initialize an empty list to populate with geometry of fetched roads
            geoms = []

            # loop through each osm value in the selector list
            for selector in [select, select_link, select_junction]:

                # wait 5 seconds before starting a new query
                time.sleep(5)

                # build osm query to fetch roads
                query = overpassQueryBuilder(
                    bbox=bbox_osm,
                    elementType=self.element_type,
                    selector=selector,
                    out="body",
                    includeGeometry=True,
                )

                for attemp_num in range(3):
                        try:
                            # get elements of the query result
                            elements = Overpass().query(query, timeout=120).elements()
                            break
                        except BaseException as exception:
                            if attemp_num == 2:
                                LOGGER.error(
                                    f"Failed to download query: bbox={bbox_osm}, elementType={self.element_type}, selector={selector}",
                                    exc_info=True
                                )
                                elements = []
                            else:
                                time.sleep(sleep_time)
                                sleep_time *= backoff_coefficient

                for element in elements:
                    geoms.append(element.geometry())

                # use shapely.geometry.shape to create shapely geometry objects from geoms list
                geom = [shape(k) for k in geoms]

                # create a GeoSeries from geom list and re-project to EPSG:3857  and buffer
                roads = gpd.GeoSeries(geom, crs="EPSG:4326").to_crs(original_crs)

                # buffer
                roads_buf = roads.buffer(buffer_dict[osm_value])

                # append GeoSeries of each road type to a list initialized above
                osm_roads_buf.append(roads_buf)

            # Merge all GeoSeries of all the road types in osm_roads_buf  list
            roads_merge = gpd.GeoSeries(
                pd.concat(osm_roads_buf, ignore_index=True), crs=osm_roads_buf[0].crs
            )

            # append geoseries of each bbox to osm_roads_geoseries list
            osm_roads_geoseries.append(roads_merge)

        return osm_roads_geoseries

    def _fetch_from_GeoDB(
        self, tiles_gdf, row, osm_crs, original_crs, where_stmt, buffer_dict, road_dict
    ):
        """Fetch osm data from GeoDB European osm collection.

        Args:
            tiles_gdf (GeoDataFrame): A GeoDataFrame of tiles info whose crs is coverted to osm_crs.
            row (int): The integer indicating the row of tiles_gdf.
            osm_crs (str): The crs of the osm data.
            original_crs (str): The original crs (UTM) of the tile.
            where_stmt (str): The string for querying data from GeoDB.
            buffer_dict (dict): A dictionary containing osm_values as keys and buffering values as values.
            road_dict (dict): A dictionary containing the subclasses of the osm_values, e.g., motorway, motorway_link, and motorway_junction.

        Returns:
            GeoSeries: A GeoSeries containing all queried osm data as polygons.
        """

        # get tile bbox
        bbox = list(tiles_gdf["geometry"][row].bbox)

        # generate token
        geodb = GeoDBClient()

        # query data from GeoDB with bbox. we made a public collection of european osm data, osm_europe_roads,
        #  under the user geodb_a2b85af8-6b99-4fa2-acf6-e87d74e40431. by passing the bbox, bbox_crs,
        #  and the where statement (e.g. "highway='primary'") of each tile a GeoDataFrame containing geometries
        # within the tile will be returned.
        osm_roads = geodb.get_collection_by_bbox(
            collection=self.geodb_collection,
            database=self.geodb_database,
            bbox=bbox,
            bbox_crs=int(osm_crs[5:]),  # exclude EPSG:
            where=where_stmt,
            comparison_mode="contains",
        )

        # check if data is empty
        if len(osm_roads) == 0:

            # if there's no data, return a None
            gs_osm = None

        else:
            # covert to original crs (UTM)
            osm_roads = osm_roads.to_crs(original_crs)

            # # loop through osm_values (motroway, primary, and trunk)
            for i in range(len(self.osm_values)):
                # loop through all keys in road_dict (val, link, and junction)
                for key in road_dict:

                    # buffer linestring to polygon
                    osm_roads.geometry[
                        osm_roads.highway == road_dict[key][i]
                    ] = osm_roads.geometry[
                        osm_roads.highway == road_dict[key][i]
                    ].buffer(
                        buffer_dict[self.osm_values[i]]
                    )

            # store data as GeoSeries in original crs (UTM)
            gs_osm = gpd.GeoSeries(osm_roads.geometry, crs=original_crs)

        return gs_osm

    def get_osm(self, mode, osm_crs="EPSG:4326"):
        """Obtain osm data for each tile.

        Args:
            mode (int): The flag for switching between fetching osm data from GeoDB and downloading it ad hoc.
            osm_crs (str, optional): The crs of the osm data. Defaults to "EPSG:4326".
        """
        # get origin crs
        self.tiles["CRS"] = [x.crs.ogc_string() for x in self.tiles.geometry]

        # keep original tile geometry
        original_geom = [x for x in self.tiles.geometry]

        # convert tiles_gdf crs to EPSG:4326
        tiles_gdf = self._convert_crs(self.tiles, crs=osm_crs)

        # buffer dictionary. for motorway, primary, and trunk we buffer 25, 23, and 21 meters.
        buffers_distance_dict = {
            key: buffer
            for (key, buffer) in zip(
                self.osm_values,
                [
                    self.roads_buffer,
                    self.roads_buffer - self.offset,
                    self.roads_buffer - (2 * self.offset),
                ],
            )
        }

        # road dictionary
        road_dict = {
            "val": [
                x for x in self.osm_values
            ],  # for the key "val" there are 3 values: motorway, primary, and trunk
            "link": [
                f"{x}_link" for x in self.osm_values
            ],  # for the key "link" there are 3 values: motorway_link, primary_link, and trunk_link
            "junction": [
                f"{x}_junction" for x in self.osm_values
            ],  # for the key "junction" there are 3 values: motorway_junction, primary_junction, and trunk_junction
        }

        # get where statement
        where_stmt = self._get_where_statement(road_dict)

        # mode 0 for fetching from GeoDB
        if mode == 0:

            # initialize a list to contain Geoseries
            gs_li = []
            
            for row in tqdm(range(len(tiles_gdf))):
                for attemp_num in range(4):
                    try:
                        osm_roads = self._fetch_from_GeoDB(
                            tiles_gdf, 
                            row, 
                            osm_crs,
                            self.tiles["CRS"][row],
                            where_stmt,
                            buffers_distance_dict,
                            road_dict
                        )
                        
                        gs_li.append(osm_roads)
                        break
                    except BaseException as exception:
                        if attemp_num == 3:
                            LOGGER.error(
                                f"Failed to fetch osm for tile {tiles_gdf.tile.iloc[row]}.",
                                exc_info=True
                            )
                        else:
                            time.sleep(5)

            # assign osm geoseries list to the geodataframe as a new column OSM_roads
            self.tiles["OSM_roads"] = gs_li

        # mode 1 for downloading data
        elif mode == 1:
            osm_roads = tiles_gdf.apply(
                lambda x: self._update_roads(
                    x["geometry"], x["CRS"], buffers_distance_dict
                ),
                axis=1,
            )

            # assign osm geoseries list to the geodataframe as a new column OSM_roads
            self.tiles["OSM_roads"] = osm_roads

        # assign original geometries to the geometry column
        self.tiles["geometry"] = original_geom

    def _convert_crs(self, tiles_info, crs):
        """Convert tiles' crs to match osm data.

        Args:
            tiles_info (GeoDataFrame): A GeoDataFrame containing the tiles' id, geometry, shape, and transform.
            crs (str): The crs of the osm data.

        Returns:
            GeoDataFrame: A GeoDataFrame whose crs is coverted to the crs of osm data.
        """

        # loop through each row, and check geometry crs
        for i in range(len(tiles_info)):

            # if crs is not  EPSG:4326, geometry is Sentinel Hub Geometry object
            if tiles_info.geometry[i].crs != crs:

                # transform crs to target crs
                tiles_info.geometry[i] = tiles_info.geometry[i].transform(crs)
            else:
                tiles_info.geometry[i] = tiles_info.geometry[i]

        # return gdf with new target crs
        return tiles_info

    def _convert_to_bbox_osm(self, bbox):
        """Takes a list of bbox coordinates of a tile and reorders bbox coordinates to OSM order

        Args:
            bbox (list): list of 4 bbox coordinates

        Returns:
            list: OSM ordered list
        """
        # return a new list of bbox coordinates from [minx, miny, maxx, maxy] order to [miny, minx, maxy, maxx] order
        return [bbox[1], bbox[0], bbox[3], bbox[2]]

    def _get_where_statement(self, road_dict):
        """Create a statement for fetching osm data from GeoDB.

        Args:
            road_dict (dict): A dictionary containing the subclasses of the osm_values, e.g., motorway, motorway_link, and motorway_junction.

        Returns:
            str: A string used to set as a variable (where) for the method of GeoDBClient, get_collection_by_bbox.
        """

        road_str = []

        # loop through osm_values
        for i in range(len(self.osm_values)):
            # loop through key in road_dirc, i.e., val, link, and junction
            for key in road_dict:
                road_str.append(
                    f"{self.osm_key}='{road_dict[key][i]}'"
                )

        # join string to make where statement
        where_stmt = " or ".join(road_str)
        return where_stmt

    def get_total_bounds(self):
        """Fetch the extents of the geoDB from the geoDB collection."""

        # order osm values by extent
        osm_order = {
            "trunk": 0,
            "primary": 1,
            "motorway": 2
        }

        osm_sorted = sorted(self.osm_values, key=lambda osm_value: osm_order[osm_value])

        for attemp_num in range(4):
            try:
                # generate token
                geodb = GeoDBClient()

                # load primary from osm_europe_roads
                gdf = geodb.get_collection_pg(
                    collection=self.geodb_collection,
                    database=self.geodb_database,
                    where=f"{self.osm_key}='{osm_sorted[0]}'",
                    select="geometry",
                )

                # get the bbox of dataset
                bbox = list(gdf.total_bounds)
                break
            except BaseException as exception:
                if attemp_num == 3:
                    LOGGER.error(
                        "Failed to fetech the extents of the european osm road from geodb.", 
                        exc_info=True
                    )
                else:
                    time.sleep(5)

        return bbox
