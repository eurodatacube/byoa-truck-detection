import datetime as dt
import json
from collections import Counter

import boto3
import geopandas as gpd
import numpy as np
import rasterio as rio
import xarray as xr
from fs import open_fs
from fs_s3fs import S3FS
from rasterio.features import rasterize


class Process(object):
    """Class for processing tiles' data."""

    def __init__(
            self,
            config,
            bucket_name,
            tiles_info,
    ):
        """Constructs the necessary attributes for the Process object.

        Args:
            config (Object): the sentinel hub configuration SHConfig()
            bucket_name (str): the name of your S3 bucket on AWS
            tiles_info (GeoDataFrame): A GeoDataFrame containing tiles' id, geometry, shape, transform, crs, and osm road polygons.
        """
        self.config = config
        self.bucket_name = bucket_name
        self.tiles_info = tiles_info

    def osm_raster(self, row, mode):
        """Rasterise OSM data.

        Args:
            row (int): The integer indicating the row of tiles_info.
            mode (int): A integer indicating the current mode to get osm data.

        Returns:
            numpy ndarray: The raster data of osm roads.
        """

        # Get tile's shape
        shape = self.tiles_info["shape"][row]

        # Get tile's transform
        transform = self.tiles_info["transform"][row]

        # If set to fetching mode
        if mode == 0:

            # If there's no road in the tile
            if self.tiles_info["OSM_roads"][row] is None:

                # Set raset as a NoneType object
                raster = None

            # If there are roads in the tile
            else:

                # Loop through geometries and get iteritems
                iter_obj = [
                    (x, 1) for j, x in self.tiles_info["OSM_roads"][row].iteritems()
                ]

                # Rasterize
                raster = rasterize(
                    iter_obj,
                    out_shape=shape,
                    transform=transform,
                    fill=0,
                    dtype=rio.uint8,
                )

        # If set as downloading mode
        elif mode == 1:

            # Loop through 3 types of roads and get iteritems
            iter_obj = [x.iteritems() for x in self.tiles_info["OSM_roads"][row]]

            # Put all iteritems into one list
            iter_ras = list(
                [(x, 1) for j, x in iter_obj[0]]
                + [(x, 1) for j, x in iter_obj[1]]
                + [(x, 1) for j, x in iter_obj[2]]
            )

            # If there's no road in the tile
            if len(iter_ras) == 0:

                # Set raset as a NoneType object
                raster = None

            # If there are roads in the tile
            else:
                # Rasterize
                raster = rasterize(
                    iter_ras,
                    out_shape=shape,
                    transform=transform,
                    fill=0,
                    dtype=rio.uint8,
                )

        return raster

    def get_xrds(
            self,
            key,
            days_sel,
            osm_raster,
            bands_name=["f_cloud.tif", "trucks.tif"],
    ):
        """Build xarray dataset.

        Walk through all bands in a tile and build a cube for a tile.

        Args:
            key (int): The object key to the tile in S3 bucket.
            days_sel (list): A list of selected days in a week.
            osm_raster (numpy ndarray): The raster data of osm roads.
            bands_name (list, optional): A list of bands' name required to build the xarray dataset. Defaults to ["f_cloud.tif", "trucks.tif"].

        Returns:
            xarray Dataset: A xarray dataset containing required data to output the result of turck detection.
            list: A list containing datetime objects of each timestamp in a tile.
        """

        # Create AWS sesssion
        AWS_session = boto3.Session(
            aws_access_key_id=self.config.aws_access_key_id,
            aws_secret_access_key=self.config.aws_secret_access_key,
        )

        # Open f_cloud.tif with rasterio to get basic tile info
        with rio.Env(rio.session.AWSSession(AWS_session)) as env:
            with rio.open(
                    f"s3://{self.bucket_name}/{key}f_cloud.tif"
            ) as src:

                # Obtain tiles' height and width
                height = src.height
                width = src.width

                # Create zero-array for coordinates' storage
                lon = np.zeros(width)
                lat = np.zeros(height)

                # Obtain the location of the center of each pixel on x and y axis to calculate lon and lat.
                x_li = [(x + 0.5) for x in range(width)]
                y_li = [(y + 0.5) for y in range(height)]

                # Get lon range
                for x in x_li:
                    coords = src.transform * (x, 0.5)
                    lon[int(x - 0.5)] = coords[0]

                # Get lat range
                for y in y_li:
                    coords = src.transform * (0.5, y)
                    lat[int(y - 0.5)] = coords[1]

            # Get timestamps
            AWS_resourse = AWS_session.resource("s3")
            content_obj = AWS_resourse.Object(
                self.bucket_name, f"{key}userdata.json"
            )
            file_content = content_obj.get()["Body"].read().decode("utf-8")
            json_content = json.loads(file_content)
            date_li = json_content["dates"][2:-2].split('","')
            time_obj = [
                dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ") for x in date_li
            ]

            # If selected days are less than 7 in a week
            if len(days_sel) < 7:

                # Convert datetime object to integer, i.e., Monday to 1, Tuesday to 2, ..., and Sunday to 7
                weekday_li = [date.isoweekday() for date in time_obj]

                # Count the times that the selected days appears in weekday_li, e.g., Monday appears 5 times; Tuesday apperas 7 times
                count_dict = Counter(weekday_li)

                # Calculate the amount of selected days
                time_size = 0
                for day in days_sel:
                    time_size = time_size + count_dict[day]

                # If there is no data for selected days
                if time_size == 0:
                    trucks = None

                # If there is data for selected days
                else:

                    # Create an empty list to contain the indices of selected days
                    day_ind = []

                    # Get days' indices as a list
                    for day in days_sel:
                        ind = [
                            index
                            for index, element in enumerate(weekday_li)
                            if element == day
                        ]

                        # Concatenate the index lists of each selected day
                        day_ind = day_ind + ind

                    # Sort the list by index
                    day_ind = sorted(day_ind)

                    # Get selected days' timestamps
                    time_obj = [time_obj[index] for index in day_ind]

                    # Create zero-arryas for cloud mask, fisser's cloud mask, and trucks data storage
                    fc = np.zeros((time_size, height, width))
                    trucks = np.zeros((time_size, height, width))

                    # Walk through all bands
                    for (band, arr) in zip(bands_name, [fc, trucks]):

                        # Set the route of files
                        s3_url = f"s3://{self.bucket_name}/{key}{band}"

                        # Open the file with rasterio
                        with rio.open(s3_url) as src:

                            # Read data in tif
                            data = src.read()

                            # Write data to corresponding array, e.g., fisser's cloud mask
                            #  data to fc array; trucks data to trucks array
                            for band_posn, day_posn in zip(
                                    range(len(day_ind)), day_ind
                            ):
                                arr[band_posn, :, :] = data[day_posn, :, :]

            # Not doing filtering if all days are selected
            elif len(days_sel) == 7:

                # Create zero-arryas for cloud mask, fisser's cloud mask, and trucks data storage
                fc = np.zeros((len(time_obj), height, width))
                trucks = np.zeros((len(time_obj), height, width))

                # Walk through all bands
                for (band, arr) in zip(bands_name, [fc, trucks]):

                    # Set the route of files
                    s3_url = f"s3://{self.bucket_name}/{key}{band}"

                    # Open the file with rasterio
                    with rio.open(s3_url) as src:

                        # Read data in tif
                        data = src.read()

                        # bug temp fix: if bands > time_obj
                        if data.shape[0] > len(time_obj):

                            dif = int(len(time_obj) - data.shape[0])

                            arr[:, :, :] = data[:dif, :, :]

                        else:

                            # Write data to corresponding array
                            arr[:, :, :] = data[:, :, :]

        # if there is no available data
        if trucks is None:

            # Set xarray and time_obj as None
            xrds = None
            time_obj = None

        # if there is available data
        else:

            # Build xarray dataset
            xrds = xr.Dataset(
                {
                    "f_cloud": (["time", "lat", "lon"], fc),
                    "trucks": (["time", "lat", "lon"], trucks),
                    "osm": (["lat", "lon"], osm_raster),
                },
                coords={"lon": lon, "lat": lat, "time": time_obj},
            )
        return xrds, time_obj

    def drop_cloudy(self, xrds, time_obj, MaxCC):
        """Filter out cloudy days.

        Drop data if roads are covered by clouds over MaxCC.

        Args:
            xrds (xarray Dataset): The xarray dataset output from get_xrds.
            time_obj (list): The list of tile's timestamps output from get_xrds.
            MaxCC (int): The maximum value of data's cloud coverage to keep.

        Returns:
            xarray Dataset: A xarray dataset without data whose cloud coverage is greater than MaxCC.
        """

        # Create an empty list to store indices to be dropped
        drop_ind = []

        # Loop through entire timestamps
        for t in range(len(xrds["time"])):

            # If the percentage of cloudy area over roads is higher than MaxCC
            if (
                    np.sum(xrds["f_cloud"][t, :, :] * xrds["osm"]) / np.sum(xrds["osm"])
            ) > (MaxCC / 100):

                # Append index to the drop_list
                drop_ind.append(t)

            # If the percentage of cloudy area over roads is lower/equal to MacCC
            else:
                # Continue to the next timestamp
                continue

        # Make a list of time objects to be dropped
        drop_time = [time_obj[ind] for ind in drop_ind]

        # Drop cloudy data
        xrds = xrds.drop_sel(time=drop_time)
        return xrds

    def trucks_filter(self, detected_trucks):
        """Filter out duplicate trucks.

        Trucks appear in 1 pixel around and 2 pixels horizontally and vertically are considered duplicated.

        Args:
            detected_trucks (numpy ndarray): A numpy ndarray showing detected trucks as 1.

        Returns:
            numpy ndarray: A numpy ndarray without duplicated trucks.
        """

        # Duplicate detected_trucks
        filtered_trucks = np.array(detected_trucks)

        # Find all pixels detected as trucks
        valid = np.where(detected_trucks == 1)

        # Walk through each pixel and check if there are duplicated trucks in the following positions: left, left above, above, right above, left next, above next
        for y, x in zip(valid[0], valid[1]):
            y_above = y - 1
            y_above_next = y - 2
            x_left = x - 1
            x_right = x + 1
            x_left_next = x - 2
            space_left = x_left >= 0
            space_right = x_right >= 0 and x_right < filtered_trucks.shape[1]
            space_above = y_above >= 0
            val_left_above = (
                filtered_trucks[y_above, x_left] if space_left and space_above else 0
            )
            val_right_above = (
                filtered_trucks[y_above, x_right] if space_right and space_above else 0
            )
            val_left = filtered_trucks[y, x_left] if space_left else 0
            val_above = filtered_trucks[y_above, x] if space_above else 0
            val_left_next = filtered_trucks[y, x_left_next] if x_left_next >= 0 else 0
            val_above_next = (
                filtered_trucks[y_above_next, x] if y_above_next >= 0 else 0
            )
            if (
                    val_left_above
                    + val_right_above
                    + val_left
                    + val_above
                    + val_left_next
                    + val_above_next
            ) >= 1:
                filtered_trucks[y, x] = 0
        return filtered_trucks
