"""Module to check if data is available in the S3 bucket"""

import itertools as it
import logging
from typing import Dict, List, Tuple

import boto3

from sentinelhub import BatchRequestStatus, SentinelHubBatch

LOGGER = logging.getLogger(__name__)


class DataAvailability:
    """Class to check data availability in S3 bucket"""

    def __init__(
        self,
        new_request_time_range,
        new_request_id,
        sh_config,
        aws_s3_bucket,
    ) -> None:
        """Necessary inputs to check data existence

        Args:
            new_request_time_range (str: "YYYY-MM-DD/YYYY-MM-DD"): timeRange of the new request
            new_request_id (str): The new request id
            sh_config (sentinelhub.config.SHConfig): Sentinel Hub Configurations class
            aws_s3_bucket (str): AWS s3 bucket name
        """
        self.new_request_time_range = new_request_time_range
        self.new_request_id = new_request_id
        self.batch_client = SentinelHubBatch(sh_config)
        self.aws_s3_bucket = aws_s3_bucket
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=sh_config.aws_access_key_id,
            aws_secret_access_key=sh_config.aws_secret_access_key,
        )
        self.ids_with_same_time_range = None
        self.new_tiles = None

    def check_data_availability(self) -> bool:
        """Main data availability checking function

        Returns:
            bool: True for data available and False for data unavailable
        """
        all_existed_request_ids = self._get_all_existed_request_ids()
        pairs_of_id_and_time_range = self._get_time_range_from_request_id(all_existed_request_ids)
        ids_with_same_time_range = self._get_all_request_ids_with_new_request_time_range(pairs_of_id_and_time_range)
        new_tiles = self._get_new_tiles()
        return self._check_existed_tiles_coverage(new_tiles, ids_with_same_time_range)

    def delete_new_request_obj(self, obj_key) -> None:
        """Delete new request JSON file

        Args:
            obj_key (str): Object key to the request JSON file
        """
        try:
            self.s3_client.head_object(Bucket=self.aws_s3_bucket, Key=obj_key)
            self.s3_client.delete_object(Bucket=self.aws_s3_bucket, Key=obj_key)
            print(f"Deleted {obj_key}.")
        except BaseException as exception:
            print(exception)

    def get_new_request_obj_key(self) -> str:
        """Get the key of the new request JSON file

        Returns:
            str: Object key to the JSON
        """
        return f"{self.new_request_id}/request-{self.new_request_id}.json"

    def get_available_tiles_keys(self) -> List[str]:
        """Get keys to available tiles

        Returns:
            List[str]: A list of keys to available tiles
        """
        available_tiles_keys = []
        for tile in self.new_tiles:
            for request_id in self.ids_with_same_time_range:
                tile_key = f"{request_id}{tile}/"
                metadata_keys = self.s3_client.list_objects(Bucket=self.aws_s3_bucket, Prefix=tile_key).keys()
                if "Contents" in metadata_keys:
                    available_tiles_keys.append(tile_key)
                    break
        return available_tiles_keys

    def _get_common_prefixes(self, prefix="") -> List[Dict[str, str]]:
        """Get common prefixes

        Args:
            prefix (str, optional): Insert request id to get common prefixes in a sub folder.
            Defaults to ''.

        Returns:
            List[Dict[str, str]]: A list of common prefixes indicating request ids and tiles
        """
        metadata = self.s3_client.list_objects(Bucket=self.aws_s3_bucket, Prefix=prefix, Delimiter="/")
        common_prefixes = metadata["CommonPrefixes"]
        return common_prefixes

    def _get_all_existed_request_ids(self) -> List[str]:
        """Get all existed request ids in the bucket except the new request

        Returns:
            List[str]: A list of request ids already exists in the bucket
        """
        common_prefixes = self._get_common_prefixes()
        all_existed_request_ids = [
            prefix_dict["Prefix"]
            for prefix_dict in common_prefixes
            if prefix_dict["Prefix"] != f"{self.new_request_id}/"
        ]
        return all_existed_request_ids

    def _get_time_range_from_request_id(self, all_existed_request_ids: List[str]) -> List[Tuple[str, str]]:
        """Get all time ranges for the existed requests

        Args:
            all_existed_request_ids (List): A list of request ids in the bucket

        Returns:
            List[Tuple[str, str]]: A list of tuples containing request id and time range
        """
        pairs_of_id_and_time_range = []
        for request_id in all_existed_request_ids:
            batch_request = self.batch_client.get_request(request_id)
            if batch_request.status == BatchRequestStatus.DONE:
                time_from = batch_request.process_request["input"]["data"][0]["dataFilter"]["timeRange"]["from"].split(
                    "T"
                )[0]
                time_to = batch_request.process_request["input"]["data"][0]["dataFilter"]["timeRange"]["to"].split("T")[
                    0
                ]
                time_range = f"{time_from}/{time_to}"
                pairs_of_id_and_time_range.append((request_id, time_range))
        return pairs_of_id_and_time_range

    def _get_all_request_ids_with_new_request_time_range(self, pairs_of_id_and_time_range: List[tuple]) -> List[str]:
        """Get a list of request ids having the same time range as the new request

        Args:
            pairs_of_id_and_time_range (list[tuple]): A list of tuples containing request id and time range

        Returns:
            List[str]: A list of request ids having the same time range as the new request
        """
        request_ids = []
        for request_id, time_range in pairs_of_id_and_time_range:
            if time_range == self.new_request_time_range and request_id != f"{self.new_request_id}/":
                request_ids.append(request_id)
        self.ids_with_same_time_range = request_ids
        return request_ids

    def _get_existed_tiles_from_request_id(self, request_id: str) -> List[str]:
        """Get all tiles existed in the bucket with a request id

        Args:
            request_id (str): The batch request id

        Returns:
            List[str]: A list of tiles name existed in the bucket created by a specific batch request
        """
        common_prefixes = self._get_common_prefixes(prefix=request_id)
        tiles_list = [prefix["Prefix"].split("/")[-2] for prefix in common_prefixes]
        return tiles_list

    def _get_new_tiles(self) -> List[str]:
        """Get a list of tiles that will be created from the new batch request

        Returns:
            List[str]: A list of tile name
        """
        new_tiles = []
        for tile in self.batch_client.iter_tiles(self.new_request_id):
            new_tiles.append(tile["name"])
        self.new_tiles = new_tiles
        return new_tiles

    def _check_tile_existence(self, request_id: str, tile: str) -> bool:
        """Check if a tile exists in the S3 bucket.

        Args:
            request_id (str): A batch request id already in the S3 bucket.
            tile (str): A tile that will be created by the new batch request.

        Returns:
            bool: True for tile available in the S3 bucket and False for unavailable.
        """
        metadata_keys = self.s3_client.list_objects(Bucket=self.aws_s3_bucket, Prefix=f"{request_id}{tile}/")
        return "Contents" in metadata_keys

    def _check_single_tile_existence_in_single_request_id(self, request_ids: List[str], tile: str) -> bool:
        return any([*map(self._check_tile_existence, request_ids, it.repeat(tile))])

    def _check_existed_tiles_coverage(self, new_tiles: List[str], existed_ids: List[str]) -> bool:
        """Check the data availability

        Args:
            new_tiles (List[str]): A list of tiles that will be created from the new batch request
            existed_ids (Set[str]): A set of tiles already exists in the bucket

        Returns:
            bool: True for data available and False for data unavailable
        """
        return all(self._check_single_tile_existence_in_single_request_id(existed_ids, tile) for tile in new_tiles)
