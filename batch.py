#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Batch processing.

This file is part of the Truck Detection Algorithm.

EDC consortium / H. Fisser
"""
import time
from collections import Counter

import boto3
import geopandas as gpd
import rasterio
import logging
from edc import setup_environment_variables
from fs import open_fs
from fs_s3fs import S3FS
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from sentinelhub import CRS, BBox, Geometry, SHConfig
from tqdm import tqdm

import evalscripts

LOGGER = logging.getLogger(__name__)

class Batch(object):
    """Batch Processing class.

    Attributes:
        config (Object): the sentinel hub configuration SHConfig()
        bucket_name (str): the name of your S3 bucket on AWS
        aoi (list): list of coordinates forming the bounding box (AOI).
        crs (str):  CRS (EPSG code) of the coordinates in the AOI.
        start (str):  Start date of the period to fetch data for in the formt of YYYY-MM-DD
        end (str): End date of the period to fetch data for in the formt of YYYY-MM-DD
        grid_id (int, optional): Predefined tiling grid ID (https://docs.sentinel-hub.com/api/latest/api/batch/#tiling-grids). Defaults to 1.
        grid_res (int, optional): Predefined tiling grid resolution (https://docs.sentinel-hub.com/api/latest/api/batch/#tiling-grids). Defaults to 10.
        data (str, optional): Satellite data to query. Defaults to "S2L2A".
        mosaicking_order (str, optional): SH mosaicking order (https://docs.sentinel-hub.com/api/latest/evalscript/v3/#mosaicking). Defaults to "mostRecent".
        max_cloud_coverage (int, optional): Maximum cloud cover in percentage. Defaults to 100.
        upsampling (str, optional): Upsampling method, depends on sensor (e.g. https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/#processing-options). Defaults to "NEAREST".
        downsampling (str, optional): Downsampling method, depends on sensor (e.g. https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/#processing-options).. Defaults to "NEAREST".
        descr (str, optional): User description for the Batch run. Defaults to "default".

    Methods:
        _authentication: Authenticate with SH services.
        run: Run the Batch Processing API.
        get_tiles_info: Retrieve information about the Batch API results stored as tiles on a Bucket.
        get_status: Fetch Batch request status.

    """

    def __init__(
        self,
        config,
        bucket_name,
        aoi,
        crs,
        start,
        end,
        grid_id=1,
        grid_res=10,
        data="sentinel-2-l2a",
        mosaicking_order="mostRecent",
        max_cloud_coverage=100,
        upsampling="NEAREST",
        downsampling="NEAREST",
        descr="default",
    ):
        """Constructs the necessary attributes for the Batch object.

        Args:
            config (Object): the sentinel hub configuration SHConfig()
            bucket_name (str): the name of your S3 bucket on AWS
            aoi (list): list of coordinates forming the bounding box (AOI).
            crs (str):  CRS (EPSG code) of the coordinates in the AOI.
            start (str):  Start date of the period to fetch data for in the formt of YYYY-MM-DD
            end (str): End date of the period to fetch data for in the formt of YYYY-MM-DD
            grid_id (int, optional): Predefined tiling grid ID (https://docs.sentinel-hub.com/api/latest/api/batch/#tiling-grids). Defaults to 1.
            grid_res (int, optional): Predefined tiling grid resolution (https://docs.sentinel-hub.com/api/latest/api/batch/#tiling-grids). Defaults to 10.
            data (str, optional): Satellite data to query. Defaults to "S2L2A".
            mosaicking_order (str, optional): SH mosaicking order (https://docs.sentinel-hub.com/api/latest/evalscript/v3/#mosaicking). Defaults to "mostRecent".
            max_cloud_coverage (int, optional): Maximum cloud cover in percentage. Defaults to 100.
            upsampling (str, optional): Upsampling method, depends on sensor (e.g. https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/#processing-options). Defaults to "NEAREST".
            downsampling (str, optional): Downsampling method, depends on sensor (e.g. https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/#processing-options).. Defaults to "NEAREST".
            descr (str, optional): User description for the Batch run. Defaults to "default".
        """

        self.config = config
        self.bucket_name = bucket_name
        self.aoi = aoi
        self.crs = crs
        self.start = start
        self.end = end
        self.grid_id = grid_id
        self.grid_res = grid_res
        self.data = data
        self.mosaicking_order = mosaicking_order
        self.max_cloud_coverage = max_cloud_coverage
        self.upsampling = upsampling
        self.downsampling = downsampling
        self.descr = descr
        self.batchID = None

    def _authentication(self):
        """Authenticate with SH services

        Returns:
            OAuth token: a bearer token for SH services.
        """
        # Create a clinet
        client = BackendApplicationClient(client_id=self.config.sh_client_id)
        oauth = OAuth2Session(client=client)

        # Fetch token based on credentials
        token = oauth.fetch_token(
            token_url="https://services.sentinel-hub.com/oauth/token",
            client_id=self.config.sh_client_id,
            client_secret=self.config.sh_client_secret,
        )
        return oauth

    def run(self, td_thresholds, cm_threshold):
        """Run the Batch Processing API based on inputs stored as class attributes.

        Args:
            td_thresholds (dict): Thresholds for Truck Detection.
            cm_threshold ([type]): Thresholds for Cloud masking.

        Raises:
            RuntimeError: All tiles failed.
            RuntimeError: Batch failed after a fixed number of tries.
        """
        # Authentication
        oauth = self._authentication()

        # Set evalscript
        evalscript = evalscripts.create_evalscript(td_thresholds, cm_threshold)

        # Build payload based on class attributes
        payload = {
            "processRequest": {
                "input": {
                    "bounds": {
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                self.aoi
                            ]
                        },
                        "properties": {
                            "crs": f"http://www.opengis.net/def/crs/EPSG/0/{self.crs}"
                        },
                    },
                    "data": [
                        {
                            "type": self.data,
                            "dataFilter": {
                                "timeRange": {
                                    "from": f"{self.start}T00:00:00Z",
                                    "to": f"{self.end}T23:59:59Z",
                                },
                                "mosaickingOrder": self.mosaicking_order,
                                "maxCloudCoverage": self.max_cloud_coverage,
                            },
                            "processing": {
                                "upsampling": self.upsampling,
                                "downsampling": self.downsampling,
                            },
                        }
                    ],
                },
                "output": {
                    "responses": [
                        {"identifier": "cloud_mask", "format": {"type": "image/tiff"}},
                        {"identifier": "trucks", "format": {"type": "image/tiff"}},
                        {"identifier": "f_cloud", "format": {"type": "image/tiff"}},
                        {
                            "identifier": "userdata",
                            "format": {"type": "application/json"},
                        },
                    ]
                },
                "evalscript": evalscript,
            },
            "tilingGrid": {"id": self.grid_id, "resolution": self.grid_res},
            "bucketName": self.bucket_name,
            "description": self.descr,
        }
        
        for attemp_num in range(3):
            try:
                # Create request
                response = oauth.request(
                    "POST",
                    "https://services.sentinel-hub.com/api/v1/batch/process",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                )

                # Raise for status if the request is erroneous
                response.raise_for_status()

                # Get requests id
                batch_request_id = response.json()["id"]

                # Start
                response = oauth.request(
                    "POST",
                    f"https://services.sentinel-hub.com/api/v1/batch/process/{batch_request_id}/start",
                )

                # Status check
                status = oauth.request(
                    "GET",
                    f"https://services.sentinel-hub.com/api/v1/batch/process/{batch_request_id}",
                ).json()["status"]

                print(f"Batch: {batch_request_id} is {status}.")

                # A bit of delay to have time to start tile count
                time.sleep(20)

                # Get tiles
                tiles_url = f"https://services.sentinel-hub.com/api/v1/batch/process/{batch_request_id}/tiles"
                tiles = oauth.request("GET", tiles_url)
                status_tiles = [x["status"] for x in tiles.json()["data"]]

                # Build a progress bar
                pbar = tqdm(total=len(status_tiles))

                # Set counter for while loop
                count = 0

                # Start looping on status of the Batch request
                while status != "DONE":
                    oauth = self._authentication()
                    if status == "FAILED":
                        LOGGER.error("All tiles failed processing")
                        raise RuntimeError("All tiles failed processing")
                    elif status == "PARTIAL":
                        if count < 3:
                            print(
                                f"Batch: {batch_request_id} is {status}LY failed. Start re-processing all failed tiles."
                            )
                            LOGGER.warning(f"Batch: {batch_request_id} is {status}LY failed. Start re-processing all failed tiles.")
                            response = oauth.request(
                                "POST",
                                f"https://services.sentinel-hub.com/api/v1/batch/process/{batch_request_id}/restartpartial",
                            )
                            count += 1
                            time.sleep(30)
                            status = oauth.request(
                                "GET",
                                f"https://services.sentinel-hub.com/api/v1/batch/process/{batch_request_id}",
                            ).json()["status"]
                        else:
                            LOGGER.error(f"Stop re-processing Batch: {batch_request_id} after 3 tries")
                            raise RuntimeError(
                                f"Stop re-processing Batch: {batch_request_id} after {count} tries."
                            )
                    else:
                        # Get tiles
                        tiles_url = f"https://services.sentinel-hub.com/api/v1/batch/process/{batch_request_id}/tiles"
                        tiles = oauth.request("GET", tiles_url)
                        status_tiles = [x["status"] for x in tiles.json()["data"]]
                        pbar.update(Counter(status_tiles)["PROCESSED"])

                        time.sleep(5)

                        status = oauth.request(
                            "GET",
                            f"https://services.sentinel-hub.com/api/v1/batch/process/{batch_request_id}",
                        ).json()["status"]

                pbar.close()
                print(f"Batch: {batch_request_id} is {status}.")
                self.batchID = batch_request_id
                break
                
            except BaseException as exception:
                if attemp_num < 2:
                    time.sleep(30)
                else:
                    reason = oauth.request(
                        "GET",
                        f"https://services.sentinel-hub.com/api/v1/batch/process/{batch_request_id}",
                    ).json()["error"]
                    LOGGER.error(exception, exc_info=True)
                    raise RuntimeError(f"Stop running batch due to server issues. Reason: {reason}")

    def get_tiles_info(self):
        """Get information about tile from a Batch API request.

        Returns:
            geopandas.GeoDataframe: table containing information about Batch tiles.
        """
        # Create an Amazon boto session
        AWS_session = boto3.Session(
            aws_access_key_id=self.config.aws_access_key_id,
            aws_secret_access_key=self.config.aws_secret_access_key,
        )

        # Setup S3 filesytem
        s3fs = S3FS(
            self.bucket_name,
            dir_path=self.batchID,
            aws_access_key_id=self.config.aws_access_key_id,
            aws_secret_access_key=self.config.aws_secret_access_key,
        )

        # Go through the files
        folders = [x for x in s3fs.listdir("/") if not x.endswith("json")]

        # Initialise a dictionnary with information to be stored
        d = {"tile": [], "geometry": [], "shape": [], "transform": []}

        # Retrieve tiles' name, geometry, shape, and transform from cloud_mask.tif
        for tile in folders:
            d["tile"].append(tile)
            # Open cloud mask band with Rasterio
            with rasterio.Env(rasterio.session.AWSSession(AWS_session)) as env:
                s3_url = f"s3://{self.bucket_name}/{self.batchID}/{tile}/cloud_mask.tif"
                # Fetch information about the raster
                with rasterio.open(s3_url) as source:
                    bbox = BBox(source.bounds, CRS(source.crs.to_epsg()))
                    geometry = Geometry(bbox.geometry, CRS(source.crs.to_epsg()))
                    d["geometry"].append(geometry)
                    d["shape"].append(source.shape)
                    d["transform"].append(source.transform)

        return gpd.GeoDataFrame(d)

    def get_status(self, batchID):
        """Fetch Batch request status.

        Args:
            batchID (str): The ID of the Batch request.

        Returns:
            json: Status of the request.
        """
        # Authenticate with SH services
        oauth = self._authentication()

        # Get the status of a batch request
        status = oauth.request(
            "GET", f"https://services.sentinel-hub.com/api/v1/batch/process/{batchID}"
        ).json()["status"]

        return status
