#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evalscripts.

This file is part of the Truck Detection Algorithm.

EDC consortium / H. Fisser
"""


def create_evalscript(td_thresholds, cm_thresholds):
    """Create an evalscript based on input parameters.

    Args:
        td_thresholds (dict): Thresholds for Truck Detection.
        cm_threshold ([type]): Thresholds for Cloud masking.

    Returns:
        str: Evalscript with inserted inputs.
    """
    # Parse the options into the evalscript
    evalscript = f"""
      //VERSION=3

      function setup() {{
        return {{
          input: [{{
            bands: ["B02", "B03", "B04", "B08", "B11", "CLM", "SCL"],
            units: ["REFLECTANCE", "REFLECTANCE", "REFLECTANCE", "REFLECTANCE", "REFLECTANCE", "DN", "DN"]
          }}],
          output: [
            {{
              id: "cloud_mask",
              bands: 1,
              sampleType: SampleType.UINT8
            }},
            {{
              id: "trucks",
              bands: 1,
              sampleType: SampleType.UINT8
            }},
            {{
              id: "f_cloud",
              bands: 1,
              sampleType: SampleType.UINT8
            }}
          ],
          mosaicking: "ORBIT"
        }}
      }}

      function updateOutput(outputs, collection) {{
          Object.values(outputs).forEach((output) => {{
              output.bands = collection.scenes.length;
          }});
      }}

      function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {{
        var all_dates = [];
        for (var i = 0; i < scenes.length; i++){{
          all_dates.push(scenes[i].date)
        }}
        outputMetadata.userData = {{
          "nb_dates": JSON.stringify(scenes.length),
          "dates": JSON.stringify(all_dates)
        }}
      }}

      function evaluatePixel(samples) {{
        var n_observations = samples.length;
        let band_b = new Array(n_observations).fill(0);
        let band_g = new Array(n_observations).fill(0);
        let band_r = new Array(n_observations).fill(0);
        let band_clm = new Array(n_observations).fill(0);
        let band_ndvi_mask = new Array(n_observations).fill(0);
        let band_ndwi_mask = new Array(n_observations).fill(0);
        let band_ndsi_mask = new Array(n_observations).fill(0);
        let band_low_rgb_mask = new Array(n_observations).fill(0);
        let band_high_rgb_mask = new Array(n_observations).fill(0);
        let band_no_truck_mask = new Array(n_observations).fill(0);
        let band_bg_min = new Array(n_observations).fill(0);
        let band_br_min = new Array(n_observations).fill(0);
        let band_bg_max = new Array(n_observations).fill(0);
        let band_br_max = new Array(n_observations).fill(0);
        let band_trucks = new Array(n_observations).fill(0);
        let band_med_prob = new Array(n_observations).fill(0);
        let band_high_prob = new Array(n_observations).fill(0);
        let band_cirrus = new Array(n_observations).fill(0);
        let band_no_data = new Array(n_observations).fill(0);
        let band_clouds_rgb = new Array(n_observations).fill(0);
        let band_clouds_bg = new Array(n_observations).fill(0);
        let band_clouds_br = new Array(n_observations).fill(0);
        let band_clouds = new Array(n_observations).fill(0);
        let band_f_cloud = new Array(n_observations).fill(0);

        // Set a multiplier of 10000 for saving to integers
        var f = 10000;

        samples.forEach((sample, index) => {{
          band_b[index] = sample.B02 * f;
          band_g[index] = sample.B03 * f;
          band_r[index] = sample.B04 * f;
          band_clm[index] = sample.CLM;
          band_ndvi_mask[index] = (sample.B08-sample.B04)/(sample.B08+sample.B04) < {td_thresholds["max_ndvi"]};
          band_ndwi_mask[index] = (sample.B02-sample.B11)/(sample.B02+sample.B11) < {td_thresholds["max_ndwi"]};
          band_ndsi_mask[index] = (sample.B03-sample.B11)/(sample.B03+sample.B11) < {td_thresholds["max_ndsi"]};
          band_low_rgb_mask[index] = (sample.B02 > {td_thresholds["min_blue"]}) * (sample.B03 > {td_thresholds["min_green"]}) * (sample.B04 > {td_thresholds["min_red"]});
          band_high_rgb_mask[index] = (sample.B02 < {td_thresholds["max_blue"]}) * (sample.B03 < {td_thresholds["max_green"]}) * (sample.B04 < {td_thresholds["max_red"]});
          band_no_truck_mask[index] = band_ndvi_mask[index] * band_ndwi_mask[index] * band_ndsi_mask[index] * band_low_rgb_mask[index] * band_high_rgb_mask[index];
          band_bg_min[index] = (sample.B02-sample.B03)/(sample.B02+sample.B03) > {td_thresholds["min_blue_green_ratio"]};
          band_br_min[index] = (sample.B02-sample.B04)/(sample.B02+sample.B04) > {td_thresholds["min_blue_red_ratio"]};
          band_bg_max[index] = (sample.B02-sample.B03)/(sample.B02+sample.B03) < {td_thresholds["max_blue_green_ratio"]};
          band_br_max[index] = (sample.B02-sample.B04)/(sample.B02+sample.B04) < {td_thresholds["max_blue_red_ratio"]};
          band_trucks[index] = (band_bg_min[index] * band_br_min[index] * band_bg_max[index] * band_br_max[index]) * band_no_truck_mask[index];
          band_med_prob[index] = sample.SCL == 8;
          band_high_prob[index] = sample.SCL == 9;
          band_cirrus[index] = sample.SCL == 10;
          band_no_data[index] = sample.SCL == 0;
          band_clouds_rgb[index] = (sample.B02 > {cm_thresholds["rgb"]}) + (sample.B03 > {cm_thresholds["rgb"]}) + (sample.B04 > {cm_thresholds["rgb"]}) >= 1;
          band_clouds_bg[index] = (sample.B02-sample.B03)/(sample.B02+sample.B03) > {cm_thresholds["blue_green"]};
          band_clouds_br[index] = (sample.B02-sample.B04)/(sample.B02+sample.B04) > {cm_thresholds["blue_red"]};
          band_clouds[index] = band_clouds_rgb[index] + band_clouds_bg[index] + band_clouds_br[index] >= 1;
          band_f_cloud[index] = band_med_prob[index] + band_high_prob[index] + band_cirrus[index] + band_no_data[index] + band_clouds[index] == 0 ? 0 : 1;
        }});

        return {{
          cloud_mask: band_clm,
          trucks: band_trucks,
          f_cloud: band_f_cloud
        }};
      }}
    """

    return evalscript
