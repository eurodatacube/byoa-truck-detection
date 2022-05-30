#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Batch processing.

This file is part of the Truck Detection Algorithm.

EDC consortium / H. Fisser
"""
import time
from collections import Counter

import boto3
import logging
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from tqdm import tqdm

from evalscripts import create_evalscript
from data_availability import DataAvailability

LOGGER = logging.getLogger(__name__)


class Batch:
    """Batch Processing class.

    Attributes:
        config (Object): the sentinel hub configuration SHConfig()
        bucket_name (str): the name of your S3 bucket on AWS
        aoi (list): list of coordinates forming the bounding box (AOI).
        crs (str):  CRS (EPSG code) of the coordinates in the AOI.
        start (str):  Start date of the period to fetch data for in the format of YYYY-MM-DD
        end (str): End date of the period to fetch data for in the format of YYYY-MM-DD
        grid_id (int, optional): Predefined tiling grid ID
        (https://docs.sentinel-hub.com/api/latest/api/batch/#tiling-grids). Defaults to 1.
        grid_res (int, optional): Predefined tiling grid resolution
        (https://docs.sentinel-hub.com/api/latest/api/batch/#tiling-grids). Defaults to 10.
        data (str, optional): Satellite data to query. Defaults to "S2L2A".
        mosaicking_order (str, optional): SH mosaicking order
        (https://docs.sentinel-hub.com/api/latest/evalscript/v3/#mosaicking). Defaults to "mostRecent".
        max_cloud_coverage (int, optional): Maximum cloud cover in percentage. Defaults to 100.
        upsampling (str, optional): Upsampling method, depends on sensor
        (e.g. https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/#processing-options). Defaults to "NEAREST".
        downsampling (str, optional): Downsampling method, depends on sensor
        (e.g. https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/#processing-options). Defaults to "NEAREST".
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
            batch_url="https://services.sentinel-hub.com/api/v1/batch/process",
            token_url="https://services.sentinel-hub.com/oauth/token"
    ):
        """Constructs the necessary attributes for the Batch object.

        Args:
            config (Object): the sentinel hub configuration SHConfig()
            bucket_name (str): the name of your S3 bucket on AWS
            aoi (list): list of coordinates forming the bounding box (AOI).
            crs (int):  CRS (EPSG code) of the coordinates in the AOI.
            start (str):  Start date of the period to fetch data for in the format of YYYY-MM-DD
            end (str): End date of the period to fetch data for in the format of YYYY-MM-DD
            grid_id (int, optional): Predefined tiling grid ID
            (https://docs.sentinel-hub.com/api/latest/api/batch/#tiling-grids). Defaults to 1.
            grid_res (int, optional): Predefined tiling grid resolution
            (https://docs.sentinel-hub.com/api/latest/api/batch/#tiling-grids). Defaults to 10.
            data (str, optional): Satellite data to query. Defaults to "S2L2A".
            mosaicking_order (str, optional): SH mosaicking order
            (https://docs.sentinel-hub.com/api/latest/evalscript/v3/#mosaicking). Defaults to "mostRecent".
            max_cloud_coverage (int, optional): Maximum cloud cover in percentage. Defaults to 100.
            upsampling (str, optional): Upsampling method, depends on sensor
            (e.g. https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/#processing-options).
            Defaults to "NEAREST".
            downsampling (str, optional): Downsampling method, depends on sensor
            (e.g. https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/#processing-options).
            Defaults to "NEAREST".
            descr (str, optional): User description for the Batch run. Defaults to "default".
            batch_url (str, optional): The url of Batch Processing API endpoint.
            token_url (str, optional): The url to fetch token for Sentinel Hub services.
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
        self.token_url = token_url
        self.batch_url = batch_url

        def _authenticate(sh_client_id, sh_client_secret, sh_token_url):
            """Authenticate with SH services

            Returns:
                OAuth token: a bearer token for SH services.
            """
            client = BackendApplicationClient(client_id=sh_client_id)
            oauth = OAuth2Session(client=client)
            for retry in range(3):
                try:
                    token = oauth.fetch_token(
                        token_url=sh_token_url,
                        client_secret=sh_client_secret
                    )
                except BaseException as exception:
                    if retry == 2:
                        LOGGER.error(
                            f"Failed to fetch the token. Reason: {exception}"
                        )
                        raise RuntimeError("Failed to fetch the token.")
                    else:
                        time.sleep(30)

            return oauth, token

        self.oauth, self.token = _authenticate(
            config.sh_client_id, config.sh_client_secret, token_url
        )
        self.batch_id = None
        self.availability = None

    def _refresh_token_if_expires(self):
        """Refresh token if it expires.
        """
        if time.time() > self.token['expires_at'] - 60:
            for retry in range(3):
                try:
                    self.token = self.oauth.fetch_token(
                        token_url=self.token_url,
                        client_secret=self.config.sh_client_secret
                    )
                except BaseException as exception:
                    if retry == 2:
                        LOGGER.error(
                            f"Failed to refresh the token. Reason: {exception}"
                        )
                        raise RuntimeError("Failed to refresh the token.")
                    else:
                        time.sleep(30)

    def _get_request_status(self):
        """Get request status.

        Returns:
            str: The batch request status.
        """
        request_url = f"{self.batch_url}/{self.batch_id}"
        self._refresh_token_if_expires()
        response = self.oauth.request("GET", request_url)
        for retry in range(3):
            try:
                response.raise_for_status()
                break
            except BaseException as exception:
                if retry == 2:
                    LOGGER.error(
                        f"Failed to get the status. Reason: {exception}"
                    )
                    raise RuntimeError("Failed to get request status.")
                else:
                    time.sleep(30)
                    self._refresh_token_if_expires()
                    response = self.oauth.request("GET", request_url)
        return response.json()['status']

    def _get_tiles_status(self):
        """Get tile status.

        Returns:
            List: _description_
        """
        tiles_url = f"{self.batch_url}/{self.batch_id}/tiles"
        self._refresh_token_if_expires()
        response = self.oauth.request("GET", tiles_url)
        for retry in range(3):
            try:
                response.raise_for_status()
                break
            except BaseException as exception:
                if retry == 2:
                    LOGGER.error(
                        f"Failed to get the tiles. Reason: {exception}"
                    )
                    raise RuntimeError("Failed to get tiles status.")
                else:
                    time.sleep(30)
                    self._refresh_token_if_expires()
                    response = self.oauth.request("GET", tiles_url)
        tiles_status = [tile["status"] for tile in response.json()["data"]]
        return tiles_status

    def _restart_partially_processed_request(self):
        restart_url = f"{self.batch_url}/{self.batch_id}/restartpartial"
        self._refresh_token_if_expires()
        response = self.oauth.request("POST", restart_url)
        for retry in range(3):
            try:
                response.raise_for_status()
                break
            except BaseException as exception:
                if retry == 2:
                    LOGGER.error(
                        f"Failed to restart the partially processed request. Reason: {exception}"
                    )
                    raise RuntimeError("Failed to restart partially processed request.")
                else:
                    time.sleep(30)
                    self._refresh_token_if_expires()
                    response = self.oauth.request("POST", restart_url)
        print(f"Batch: {self.batch_id} restarts processing the failed tiles.")

    def create_request(self, td_thresholds, cm_thresholds):
        evalscript = create_evalscript(
            td_thresholds, cm_thresholds
        )
        payload = {
            "processRequest": {
                "input": {
                    "bounds": {
                        "geometry": {
                            "type": "MultiPolygon",
                            "coordinates": self.aoi
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
        self._refresh_token_if_expires()
        response = self.oauth.request(
            "POST",
            self.batch_url,
            headers={"Content-Type": "application/json"},
            json=payload,
        )
        for retry in range(3):
            try:
                response.raise_for_status()
                break
            except BaseException as exception:
                if retry == 2:
                    LOGGER.error(
                        f"Failed to create the request. Reason: {exception}"
                    )
                    raise RuntimeError("Failed to create request.")
                else:
                    time.sleep(30)
                    self._refresh_token_if_expires()
                    response = self.oauth.request(
                        "POST",
                        self.batch_url,
                        headers={"Content-Type": "application/json"},
                        json=payload,
                    )
        self.batch_id = response.json()["id"]
        print(f"Batch: {self.batch_id} is CREATED.")

    def analyse_request(self):
        analyse_url = f"{self.batch_url}/{self.batch_id}/analyse"
        self._refresh_token_if_expires()
        response = self.oauth.request("POST", analyse_url)
        for retry in range(3):
            try:
                response.raise_for_status()
                break
            except BaseException as exception:
                if retry == 2:
                    LOGGER.error(f"ANALYSIS FAILED. Reason: {exception}")
                    raise RuntimeError("Failed to analyse request.")
                else:
                    time.sleep(30)
                    self._refresh_token_if_expires()
                    response = self.oauth.request("POST", analyse_url)
        status = self._get_request_status()
        while status not in ['ANALYSIS_DONE', 'FAILED']:
            time.sleep(30)
            status = self._get_request_status()
        if status == 'FAILED':
            raise RuntimeError(f"Batch: {self.batch_id} is FAILED analysis.")
        else:
            print(f"Batch: {self.batch_id} is DONE ANALYSIS.")

    def cancel_request(self):
        cancel_url = f"{self.batch_url}/{self.batch_id}/cancel"
        self._refresh_token_if_expires()
        response = self.oauth.request("POST", cancel_url)
        for retry in range(3):
            try:
                response.raise_for_status()
                break
            except BaseException as exception:
                if retry == 2:
                    LOGGER.error(f"CANCELLATION FAILED. Reason: {exception}")
                    raise RuntimeError("Failed to cancel request.")
                else:
                    time.sleep(30)
                    self._refresh_token_if_expires()
                    response = self.oauth.request("POST", cancel_url)
        print(f"Batch: {self.batch_id} is CANCELED.")

    def data_availability(self):
        self.availability = DataAvailability(
            f"{self.start}/{self.end}",
            self.batch_id,
            self.config,
            self.bucket_name
        )

    def start_batch(self):
        """Run the Batch Processing API based on inputs stored as class attributes.

        Raises:
            RuntimeError: All tiles failed.
            RuntimeError: Batch failed after a fixed number of tries.
        """
        start_url = f"{self.batch_url}/{self.batch_id}/start"
        self._refresh_token_if_expires()
        response = self.oauth.request("POST", start_url)
        for retry in range(3):
            try:
                response.raise_for_status()
                break
            except BaseException as exception:
                if retry == 2:
                    LOGGER.error(
                        f"Failed to start the request. Reason: {exception}"
                    )
                    raise RuntimeError("Failed to start batch request.")
                else:
                    time.sleep(30)
                    self._refresh_token_if_expires()
                    response = self.oauth.request("POST", start_url)

    def monitor_batch(self):
        tiles_status = self._get_tiles_status()

        # Build a progressbar
        pbar = tqdm(total=len(tiles_status))

        # Set counter for while loop
        retry = 0

        # Start looping on status of the Batch request
        request_status = self._get_request_status()
        while request_status != "DONE":
            if request_status == "FAILED":
                LOGGER.error("All tiles failed to be processed.")
                raise RuntimeError(f"Batch: {self.batch_id} is FAILED.")
            elif request_status == "PARTIAL":
                if retry < 3:
                    LOGGER.warning(f"Batch: {self.batch_id} is PARTIALLY DONE. Start re-processing all failed tiles.")
                    self._restart_partially_processed_request()
                    retry += 1
                    time.sleep(30)
                    request_status = self._get_request_status()
                else:
                    LOGGER.error(f"Stop re-processing Batch: {self.batch_id} after 3 tries.")
                    raise RuntimeError(
                        f"Batch: {self.batch_id} is PARTIALLY DONE and stop re-processing after 3 tries."
                    )
            else:
                # Get tiles status
                tiles_status = self._get_tiles_status()
                pbar.update(Counter(tiles_status)["PROCESSED"])
                time.sleep(10)
                request_status = self._get_request_status()

        pbar.close()
        print(f"Batch: {self.batch_id} is DONE.")

    def get_tile_keys(self):
        for retry in range(3):
            try:
                s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=self.config.aws_access_key_id,
                    aws_secret_access_key=self.config.aws_secret_access_key
                )
                common_prefixes = s3_client.list_objects(
                    Bucket=self.bucket_name,
                    Prefix=f"{self.batch_id}/",
                    Delimiter="/"
                )['CommonPrefixes']
            except BaseException as exception:
                if retry == 2:
                    LOGGER.error(
                        f"Failed to get tile keys from s3 bucket. Reason: {exception}"
                    )
                    raise RuntimeError("Failed to get tile keys from s3 bucket.")
                else:
                    time.sleep(30)
        tile_keys = [prefix['Prefix'] for prefix in common_prefixes]
        return tile_keys
