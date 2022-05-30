#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Truck detection classes.

This file is part of the Truck Detection Algorithm.

EDC consortium / H. Fisser
"""
import warnings
from datetime import datetime
from pathlib import Path
import logging

import boto3
import geopandas as gpd
import numpy as np
import rasterio
from sentinelhub import CRS, BBox, Geometry, SHConfig
from tqdm import tqdm
from xcube_geodb.core.geodb import GeoDBClient
from shapely.wkt import loads
from shapely.geometry import Polygon

from batch import Batch
from osm_utils import Osm
from process_utils import Process

LOGGER = logging.getLogger(__name__)


class TruckDetector(object):
    """Truck detection class.

    Attributes:
        aoi (str): An area of interest as a polygon or multipolygon in wkt format.
        proj (int): CRS (EPSG code) of the coordinates in the bbox.
        time_period (list): A list with start date and end date as a string in the format of YYYY-MM-DD
        output_folder (str, optional): The output folder for the summary log. Defaults to "./truck_detection_results".
        config (Object, optional): The sentinel hub configuration SHConfig(). Defaults to None.

    Methods:
        set_sh_credentials: Set the credentials for Sentinel Hub services.
        set_aws_credentials: Set the credentials for access to AWS bucket service.
        set_td_thresholds: Set thresholds for truck detection.
        set_cm_thresholds: Set thresholds for cloud masking.
        set_weekdays: Set the days of the week to be considered.
        get_sentinel_data: Fetch Sentinel Data using SH services and write to bucket.
        batch_status: Retrieve the status of a previously run Batch request.
        get_tile_info: Fetch the information about Batch tiles located in a AWS bucket.
        get_osm_data: Fetch OSM data.
        _create_geodb_collection: Create a geoDB collection based on EDC credentials.
        process_tiles: Main Truck Detection process.
    """

    def __init__(
            self,
            aoi,
            proj,
            time_period,
            output_folder="./truck_detection_results",
            config=SHConfig(),
    ):
        """Constructs the necessary attributes for the TruckDetector object.

        Args:
            aoi (str): An area of interest as a polygon or multipolygon in wkt format.
            proj (int): CRS (EPSG code) of the coordinates in the bbox.
            time_period (str): A list with start date and end date as a string in the format of YYYY-MM-DD
            output_folder (str, optional): The output folder for the summary log.
            Defaults to "./truck_detection_results".
            config (Object, optional): The sentinel hub configuration SHConfig(). Defaults to None.
        """
        self.config = config
        self.aws_bucket = None

        def _get_geom_and_coords(wkt):
            geom = loads(wkt)
            if geom.geom_type == 'Polygon':
                coords = [[[
                    [coord[0], coord[1]] for coord in list(geom.exterior.coords)
                ]]]
            elif geom.geom_type == 'MultiPolygon':
                coords = []
                for polygon in geom:
                    poly_coords = [[
                        [coord[0], coord[1]] for coord in list(polygon.exterior.coords)
                    ]]
                    coords.append(poly_coords)
            else:
                raise TypeError("Input aoi should be Polygon or MultiPolygon.")
            return geom, coords

        self.aoi_geom, self.aoi_coords = _get_geom_and_coords(aoi)
        self.crs = proj

        def _get_start_and_end_date(date_range):
            start_date = date_range.split("/")[0]
            end_date = date_range.split("/")[1]
            return start_date, end_date

        self.start_date, self.end_date = _get_start_and_end_date(time_period)
        self.tile_info = None
        self.td_thresholds = {}
        self.cm_thresholds = {}
        self.filter = {"MaxCC": None}
        self.batch_id = None
        self.tile_keys = None
        self.output_folder = output_folder
        self.geodb_collection = None
        self.mode = None
        self.out_format = ["GEODB", "GPKG", "SHP", "GEOJSON"]

    def set_sh_credentials(self, client_secret, client_id):
        """Set the credentials for Sentinel Hub services.

        Args:
            client_secret (str): your sentinel hub client secret
            client_id (str): your sentinel hub client id
        """
        # Set credentials if found
        if client_id and client_secret:
            self.config.sh_client_id = client_id
            self.config.sh_client_secret = client_secret

        if self.config.sh_client_id == "" or self.config.sh_client_secret == "":
            print(
                "Warning! To use Sentinel Hub services, please provide the credentials (client ID and client secret)."
            )

    def set_aws_credentials(self, aws_client, aws_secret, aws_bucket):
        """Set the credentials for access to AWS bucket service.

        Args:
            aws_client (str): your AWS client id
            aws_secret (str): your AWS client secret
            aws_bucket (str): the name of your S3 bucket on AWS
        """
        self.config.aws_access_key_id = aws_client
        self.config.aws_secret_access_key = aws_secret
        self.aws_bucket = aws_bucket

    def set_td_thresholds(
            self,
            min_blue=0.06,
            min_green=0.04,
            min_red=0.04,
            max_blue=0.2,
            max_green=0.15,
            max_red=0.15,
            max_ndvi=0.5,
            max_ndwi=0.0001,
            max_ndsi=0.0001,
            min_blue_green_ratio=0.03,
            min_blue_red_ratio=0.05,
            max_blue_green_ratio=0.17,
            max_blue_red_ratio=0.2,
    ):
        """Set thresholds for truck detection.

        Args:
            min_blue (float, optional): minimum blue band threshold. Defaults to 0.06.
            min_green (float, optional): minimum green band threshold. Defaults to 0.04.
            min_red (float, optional): minimum red band threshold. Defaults to 0.04.
            max_blue (float, optional): maximum blue band threshold. Defaults to 0.2.
            max_green (float, optional): maximum green band threshold. Defaults to 0.15.
            max_red (float, optional): maximum red band threshold. Defaults to 0.15.
            max_ndvi (float, optional): maximum ndvi threshold. Defaults to 0.5.
            max_ndwi (float, optional): maximum ndwi threshold. Defaults to 0.0001.
            max_ndsi (float, optional): maximum ndsi threshold. Defaults to 0.0001.
            min_blue_green_ratio (float, optional): minimum blue/green ratio threshold. Defaults to 0.03.
            min_blue_red_ratio (float, optional): minimum blue/red ratio threshold. Defaults to 0.05.
            max_blue_green_ratio (float, optional): maximum blue/green ratio threshold. Defaults to 0.17.
            max_blue_red_ratio (float, optional): maximum blue/red ratio threshold. Defaults to 0.2.
        """
        # Build a dictionary from the input thresholds
        self.td_thresholds = {
            "min_blue": min_blue,
            "min_green": min_green,
            "min_red": min_red,
            "max_blue": max_blue,
            "max_green": max_green,
            "max_red": max_red,
            "max_ndvi": max_ndvi,
            "max_ndwi": max_ndwi,
            "max_ndsi": max_ndsi,
            "min_blue_green_ratio": min_blue_green_ratio,
            "min_blue_red_ratio": min_blue_red_ratio,
            "max_blue_green_ratio": max_blue_green_ratio,
            "max_blue_red_ratio": max_blue_red_ratio,
        }

    def set_cm_thresholds(self, rgb=0.25, blue_green=0.2, blue_red=0.2):
        """Set thresholds for cloud masking.

        Args:
            rgb (float, optional): RGB band threshold. Defaults to 0.25.
            blue_green (float, optional): Blue/Green ratio threshold. Defaults to 0.2.
            blue_red (float, optional): Blue/Red ratio threshold. Defaults to 0.2.
        """
        # Build a dictionary from the input thresholds
        self.cm_thresholds = {
            "rgb": rgb,
            "blue_green": blue_green,
            "blue_red": blue_red,
        }

    def set_weekdays(self, days_sel="all"):
        """Set the days of the week to be considered.

        Args:
            days_sel (str or list, optional): Days of the week selected. Defaults to "all".
        """

        # If default all days is used convert to list containing all days
        if days_sel == "all":
            days_sel = [
                "Sunday",
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
            ]

        # Initialise the list of integers
        days_int = []

        # Convert days' string to integer
        for day in days_sel:
            if day == "Sunday":
                day = 7
                days_int.append(day)
            elif day == "Monday":
                day = 1
                days_int.append(day)
            elif day == "Tuesday":
                day = 2
                days_int.append(day)
            elif day == "Wednesday":
                day = 3
                days_int.append(day)
            elif day == "Thursday":
                day = 4
                days_int.append(day)
            elif day == "Friday":
                day = 5
                days_int.append(day)
            elif day == "Saturday":
                day = 6
                days_int.append(day)

        self.filter["days_sel"] = days_int

    def get_sentinel_data(self):
        """Fetch Sentinel Data using SH services and write to bucket."""
        # Initialise a batch class
        batch_request = Batch(
            self.config,
            self.aws_bucket,
            self.aoi_coords,
            self.crs,
            self.start_date,
            self.end_date,
        )

        # If not already set, put default thresholds for truck and cloud detection
        if not self.td_thresholds:
            self.set_td_thresholds()

        if not self.cm_thresholds:
            self.set_cm_thresholds()

        # Run the Batch request
        batch_request.create_request(self.td_thresholds, self.cm_thresholds)

        batch_request.analyse_request()

        batch_request.data_availability()

        if batch_request.availability.check_data_availability():
            batch_request.cancel_request()
            tile_keys = batch_request.availability.get_available_tiles_keys()
            new_request_obj_key = batch_request.availability.get_new_request_obj_key()
            batch_request.availability.delete_new_request_obj(new_request_obj_key)
        else:
            batch_request.start_batch()
            batch_request.monitor_batch()

            # Get batchID
            self.batch_id = batch_request.batch_id
            tile_keys = batch_request.get_tile_keys()

        self.tile_keys = tile_keys

    def get_tile_info(self):
        """Fetch the information about Batch tiles located in a AWS bucket.

        To use a previously executed Batch request, manually set the `request_id` parameter in the class attributes.
        """
        # Create an Amazon boto session
        aws_session = boto3.Session(
            aws_access_key_id=self.config.aws_access_key_id,
            aws_secret_access_key=self.config.aws_secret_access_key,
        )

        # Create a dictionary to contain tiles' info. Geometry is used for querying osm data;
        #  shape and transform are used for rasterization
        d = {"tile": [], "geometry": [], "shape": [], "transform": [], "key": []}

        # Retrieve tiles' name, geometry, shape, and transform from f_cloud.tif
        for key in self.tile_keys:
            # Append tiles' name to "tile" column
            d["tile"].append(key.split("/")[1])
            d["key"].append(key)

            # Open f_cloud.tif with rasterio to collect tiles' info
            with rasterio.Env(rasterio.session.AWSSession(aws_session)) as env:
                s3_url = f"s3://{self.aws_bucket}/{key}f_cloud.tif"
                with rasterio.open(s3_url) as source:
                    # Retrieve bounding box and crs
                    bbox = BBox(source.bounds, CRS(source.crs.to_epsg()))

                    # Store geometry and crs info in sentinelhub Geometry object
                    geometry = Geometry(bbox.geometry, CRS(source.crs.to_epsg()))

                    # Append geometry to "geometry" column
                    d["geometry"].append(geometry)

                    # Append shape to "shape" column
                    d["shape"].append(source.shape)

                    # Append transform to "transform" column
                    d["transform"].append(source.transform)

        # Store tiles' info in a GeoDataFrame
        self.tile_info = gpd.GeoDataFrame(d)

    def get_osm_data(self, values, download=False):
        """Fetch OSM data.

        Retrieve the roads in the configuration of the tiles.
        If download is set as True, the script will download OSM data no matter what.
        If download is set as False, the script will fetch OSM data from the internal GeoDB collection if the area of
         interest is covered by the dataset, or download it if not covered.

        Args:
            values (list): OSM key values to retrieve.
            download (bool, optional): Flag to download OSM data. Defaults to False.
        """

        # Initialize a osm class
        osm_roads = Osm(self.tile_info, values)

        # fetch osm data from GeoDB if the AOI is covered by our collection and download osm data if not covered.
        if not download:

            # set the bbox of our GeoDB
            bbox_geodb = osm_roads.get_total_bounds()

            geom_geodb = Polygon(
                [
                    (bbox_geodb[0], bbox_geodb[1]),
                    (bbox_geodb[2], bbox_geodb[1]),
                    (bbox_geodb[2], bbox_geodb[3]),
                    (bbox_geodb[0], bbox_geodb[3]),
                    (bbox_geodb[0], bbox_geodb[1])
                ]
            )

            # if the area of interest is covered by our osm collection, fetch osm data from GeoDB
            if geom_geodb.contains(self.aoi_geom):

                # set to fetching mode
                self.mode = 0

                # silence warnings of geometry column contains no geometry
                warnings.filterwarnings("ignore")

                # get OSM data from the GeoDB
                osm_roads.get_osm(mode=self.mode)

            # if the area of interest is not covered by our osm collection, download the osm data.
            else:

                # set to downloading mode
                self.mode = 1

                # Silence the multiple OSM warnings and download data from the API
                warnings.filterwarnings("ignore")
                osm_roads.get_osm(mode=self.mode)

        # if download is set as True
        elif download:

            # set to downloading mode
            self.mode = 1

            # Silence the multiple OSM warnings and download data from the API
            warnings.filterwarnings("ignore")
            osm_roads.get_osm(mode=self.mode)

    def _create_geodb_collection(self, crs_out):
        """Create a geoDB collection based on EDC credentials.

        Args:
            crs_out (str): Output CRS (eg "EPSG:4326") of the geoDB

        Returns:
            xcube_geodb.core.geodb.GeoDBClient(): The geoDB client object.
        """

        # Obtain current time as the identifier of collection
        dt_string = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        # Connect to GeoDB
        geodb = GeoDBClient()

        # Set collection's crs and properties
        collections = {
            f"{dt_string}": {
                "crs": int(crs_out[5:]),
                "properties": {
                    "Date": "date",
                    "roads_covered": "float"
                }
            }
        }

        # Create collections
        geodb.create_collections(collections)

        # assign geodb collection's identifier
        self.geodb_collection = dt_string

        return geodb

    def process_tiles(self, out_format, crs_out="EPSG:4326"):
        """Main Truck Detection process.

        Process tiles' data retrieved from batch API.
        Filter data, build Xarray dataset, and output detected trucks as geometry points.

        Args:
            out_format (str): Should be one of "GEODB", "GPKG", "SHP", "GEOJSON".
            crs_out (str, optional): Output projection of detected truck layers. Defaults to "EPSG:4326".
        """

        # If output folder doesn't exist, create it
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)

        # Set output path
        out_path = Path(self.output_folder)

        # Initialise a Process class
        process = Process(
            self.config,
            self.aws_bucket,
            self.tile_info
        )

        # format warning
        warnings.formatwarning = lambda msg, *args, **kwargs: f"{msg}\n"

        # If weekday not already set, put all days in a week
        if not self.filter["days_sel"]:
            self.set_weekdays()

        # If maximum cloud coverage not already set, put 1 for MaxCC
        if not self.filter["MaxCC"]:
            self.filter["MaxCC"] = 100

        # If out_format is set to GeoDB
        if out_format.upper() == self.out_format[0]:

            # Create GeoDB collection
            geodb = self._create_geodb_collection(crs_out)

        # If out_format is set to GPKG, SHP, or GeoJSON
        elif out_format.upper() in self.out_format[1:]:

            # Initialize a aoi_gdf as NoneType object
            aoi_gdf = None

        # If out_format is set to others
        else:

            # Raise KeyError
            raise KeyError(f"Please select from {self.out_format}.")

        # Loop through all tiles in one batch
        for row in tqdm(range(len(self.tile_info))):

            # Open a text file for processing log
            with open(out_path.joinpath("Summary.txt"), "a") as log:
                log.write(f"Tile {self.tile_info['tile'].iloc[row]}\n")

            # Rasterize osm data
            osm_raster = process.osm_raster(row=row, mode=self.mode)

            # If there's no road in the tile
            if osm_raster is None:

                # Write no roads in tile to summary log
                with open(out_path.joinpath("Summary.txt"), "a") as log:
                    log.write(f"No road in tile {self.tile_info['tile'].iloc[row]}\n")

            # If there are roads in the tile
            else:

                # Build xarray dataset and extract timestamps
                xrds, time_obj = process.get_xrds(
                    key=self.tile_info['key'].iloc[row],
                    days_sel=self.filter["days_sel"],
                    osm_raster=osm_raster
                )

                # If there is no available data
                if xrds is None and time_obj is None:

                    # Write no available data for tile to summary log
                    with open(out_path.joinpath("Summary.txt"), "a") as log:
                        log.write(
                            f"No available data for tile {self.tile_info['tile'].iloc[row]}\n"
                        )

                # If there is available data
                else:

                    # Mask out cloud-covered and non-road area
                    xrds["trucks"] = (
                            xrds["trucks"] * np.abs(1 - xrds["f_cloud"]) * xrds["osm"]
                    )

                    # Drop cloudy data based on the MaxCC threshold
                    xrds = process.drop_cloudy(
                        xrds=xrds, time_obj=time_obj, MaxCC=self.filter["MaxCC"]
                    )

                    # Loop through timestamps and filter out invalid truck points
                    for t in range(len(xrds["time"])):
                        # Filter duplicated trucks
                        trucks_filtered = process.trucks_filter(xrds["trucks"][t, :, :])

                        # Find valid trucks yx
                        trucks_yx = np.argwhere(trucks_filtered == 1)

                        # If there is no truck detected
                        if len(trucks_yx) == 0:

                            # Write No trucks detected on date to summary log
                            with open(out_path.joinpath("Summary.txt"), "a") as log:
                                log.write(
                                    f'No trucks detected on {xrds.time[t].dt.year.values}-{"%02d" % xrds.time[t].dt.month.values}-{"%02d" % xrds.time[t].dt.day.values}\n'
                                )

                        # If there are trucks detected
                        else:

                            # Write Trucks detected on date to summary log
                            with open(out_path.joinpath("Summary.txt"), "a") as log:
                                log.write(
                                    f'Trucks detected on {xrds.time[t].dt.year.values}-{"%02d" % xrds.time[t].dt.month.values}-{"%02d" % xrds.time[t].dt.day.values}\n'
                                )

                            # Calculate roads covered in percentage
                            roads_covered = float(
                                np.sum(np.abs(1 - xrds["f_cloud"][t, :, :]) * xrds["osm"]) / np.sum(xrds["osm"]) * 100)

                            # Retrieve detected trucks' lon and lat
                            lon = [xrds["lon"][ind] for ind in trucks_yx[:, 1]]
                            lat = [xrds["lat"][ind] for ind in trucks_yx[:, 0]]

                            # Covert lon and lat to geometry points
                            geometry = gpd.points_from_xy(
                                lon, lat, crs=self.tile_info['CRS'].iloc[row]
                            )

                            # Create a dictionary to store geometry points, date, and roads covered (%)
                            d = {
                                "date": f'{xrds.time[t].dt.year.values}-{"%02d" % xrds.time[t].dt.month.values}-{"%02d" % xrds.time[t].dt.day.values}',
                                "clear_road_percentage": round(roads_covered, 2),
                                "geometry": geometry,
                            }

                            # Build a GeoDataFrame as a format to insert the results to GeoDB
                            gdf = gpd.GeoDataFrame(d, crs=self.tile_info['CRS'].iloc[row])

                            # Reproject the tile from UTM to EPSG:4326
                            gdf = gdf.to_crs(crs_out)

                            # If out_format is set to GeoDB
                            if out_format.upper() == self.out_format[0]:

                                # Insert to geodb
                                geodb.insert_into_collection(f"{self.geodb_collection}", gdf)

                            # If out_format is set to GPKG, SHP, GeoJSON
                            else:

                                # If aoi_gdf not yet assigned
                                if aoi_gdf is None:

                                    # Assign gdf to aoi_gdf
                                    aoi_gdf = gdf

                                # If aoi_gdf has values
                                else:

                                    # Append gdf to aoi_gdf
                                    aoi_gdf = aoi_gdf.append(gdf)

        # If out_format is set to GPKG
        if out_format.upper() == self.out_format[1]:

            # Save as a layered GPKG file by dates
            for date in set(aoi_gdf.date):
                aoi_gdf[aoi_gdf["date"] == date].to_file(out_path.joinpath(f"result.{self.out_format[1].lower()}"),
                                                         layer=date, driver=self.out_format[1])

        # If out_format is set to SHP
        elif out_format.upper() == self.out_format[2]:

            # Save as shapefile
            aoi_gdf.to_file(out_path.joinpath(f"result.{self.out_format[2].lower()}"))

        # If out_format is set to SHP
        elif out_format.upper() == self.out_format[3]:

            # Save as geojson
            aoi_gdf.to_file(out_path.joinpath(f"result.{self.out_format[3].lower()}"), driver="GeoJSON")
