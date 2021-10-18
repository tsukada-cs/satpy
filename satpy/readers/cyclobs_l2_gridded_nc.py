#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Interface to CyclObs L2 gridded NetCDF Reader.
IFREMER team is working on algorithms to retrieve geophysical parameters 
over TC. Recent work has been devoted to the use of C-band Synthetic 
Aperture Radar (SAR) for providing an estimate of the ocean surface wind 
speed. CyclObs aims to present this unique catalogue of measurements.

Level 2 data products are available via the CyclObs API.
For more information visit the follng URL:
https://cyclobs.ifremer.fr/app/docs/
"""

import logging
from datetime import datetime

import numpy as np
import xarray as xr

from satpy.readers.netcdf_utils import NetCDF4FileHandler
from pyresample.geometry import AreaDefinition

logger = logging.getLogger(__name__)


class CyclObsL2GriddedNCFileHandler(NetCDF4FileHandler):
    """Measurement file reader."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init the file reader."""
        super(CyclObsL2GriddedNCFileHandler, self).__init__(filename, filename_info,
                                                            filetype_info,
                                                            xarray_kwargs={
                                                                "mask_and_scale": True,
                                                                "chunks": "auto"
                                                            })
        self.platforms = {"rs2": "RADARSAT-2", "s1a": "SENTINEL-1A", "s1b": "SENTINEL-1B"}

    def get_dataset(self, ds_id, info):
        """Load dataset designated by the given key from file."""
        logger.debug("Getting data for: %s", ds_id['name'])
        file_key = info.get('file_key', ds_id['name'])
        data = self[file_key]
        time = datetime.utcfromtimestamp(data["time"].values[0].astype(int) * 1e-9)
        data = np.flipud(data.squeeze())
        data = xr.DataArray(data, dims=['y', 'x'])
        data.attrs.update({"time": time})
        data.attrs = self.get_metadata(data, info)

        if 'lon' in data.dims:
            data.rename({'lon': 'x'})
        if 'lat' in data.dims:
            data.rename({'lat': 'y'})
        return data.chunk("auto")

    def get_area_def(self, dsid):
        """Flip data up/down and define equirectangular AreaDefintion."""
        flip_lat = np.flipud(self['lat'])
        latlon = np.meshgrid(self['lon'], flip_lat)

        width = self['lon/shape'][0]
        height = self['lat/shape'][0]

        lower_left_x = latlon[0][height-1][0]
        lower_left_y = latlon[1][height-1][0]

        upper_right_y = latlon[1][0][width-1]
        upper_right_x = latlon[0][0][width-1]

        area_extent = (lower_left_x, lower_left_y, upper_right_x, upper_right_y)
        description = "CyclObs L2 WGS84"
        area_id = 'cyclobs'
        proj_id = 'World Geodetic System 1984'
        projection = 'EPSG:4326'
        area_def = AreaDefinition(area_id, description, proj_id, projection, width, height, area_extent)
        return area_def

    def get_metadata(self, data, info):
        """Get general metadata for file."""
        metadata = {}
        metadata.update(data.attrs)
        metadata.update(info)
        metadata.update({
            'platform_shortname': self.platform_shortname,
            'platform_name': self.platforms[self.platform_shortname],
            'sensor': 'sar-c',
            'start_time': self.start_time,
            'end_time': self.end_time,
        })
        metadata.update(self[info.get('file_key')].variable.attrs)
        metadata.update({'global_attributes': self['/attrs']})

        return metadata

    @property
    def start_time(self):
        """Start timestamp of the dataset determined from yaml."""
        return self.filename_info['start_time']

    @property
    def end_time(self):
        """End timestamp of the dataset same as start_time."""
        return self.filename_info.get('end_time', self.start_time)

    @property
    def sensor_name(self):
        """Sensor name."""
        return "sar-c"

    @property
    def platform_shortname(self):
        """platform shortname."""
        return self.filename_info['platform_shortname']