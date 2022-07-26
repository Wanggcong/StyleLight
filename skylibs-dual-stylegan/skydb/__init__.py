import os
from os import listdir
from os.path import abspath, isdir, join
import fnmatch
import datetime

import numpy as np

from envmap import EnvironmentMap
from hdrtools import sunutils


class SkyDB:
    def __init__(self, path):
        """Creates a SkyDB.
        The path should contain folders named by YYYYMMDD (ie. 20130619 for June 19th 2013).
        These folders should contain folders named by HHMMSS (ie. 102639 for 10h26 39s).
        Inside these folders should a file named envmap.exr be located.
        """
        p = abspath(path)
        self.intervals_dates = [join(p, f) for f in listdir(p) if isdir(join(p, f))]
        self.intervals = list(map(SkyInterval, self.intervals_dates))


class SkyInterval:
    def __init__(self, path):
        """Represent an interval, usually a day.
        The path should contain folders named by HHMMSS (ie. 102639 for 10h26 39s).
        """
        self.path =  path
        matches = []
        for root, dirnames, filenames in os.walk(path):
            for filename in fnmatch.filter(filenames, 'envmap.exr'):
                matches.append(join(root, filename))

        self.probes = list(map(SkyProbe, matches))
        self.reftimes = [x.datetime for x in self.probes]

    @property
    def sun_visibility(self):
        """
        Return sun_visibility of the interval
        """
        if len(self.probes) > 0:
            sun_visibility = sum(1 for x in self.probes if x.sun_visible) / len(self.probes)
        else:
            sun_visibility = 0
        return sun_visibility

    @property
    def date(self):
        """
        :returns: datetime.date object
        """
        date = os.path.normpath(self.path).split(os.sep)[-1]
        infos = {
            "day": int(date[-2:]),
            "month": int(date[4:6]),
            "year": int(date[:4]),
        }
        return datetime.date(**infos)

    def closestProbe(self, hours, minutes=0, seconds=0):
        """
        Return the SkyProbe object closest to the requested time.
        TODO : check for day change (if we ask for 6:00 AM and the probe sequence
            only begins at 7:00 PM and ends at 9:00 PM, then 9:00 PM is actually
            closer than 7:00 PM and will be wrongly selected; not a big deal but...)
        TODO : Take the code from skymangler.
        """
        cmpdate = datetime.datetime(year=1, month=1, day=1, hour=hours, minute=minutes, second=seconds)
        idx = np.argmin([np.abs((cmpdate - t).total_seconds()) for t in self.reftimes])
        return self.probes[idx]


class SkyProbe:
    def __init__(self, path, format_=None):
        """Represent an environment map among an interval."""
        self.path = path
        self.format_ = format_

    def init_properties(self):
        """
        Cache properties that are resource intensive to generate.
        """
        if not hasattr(self, '_envmap'):
            self._envmap = self.environment_map

    def remove_envmap(self):
        """
        Delete probe's envmap from memory.
        """
        del self._envmap

    @property
    def sun_visible(self):
        """
        :returns: boolean, True if the sun is visible, False otherwise.
        """
        self.init_properties()
        return self._envmap.data.max() > 5000

    @property
    def datetime(self):
        """Datetime of the capture.
        :returns: datetime object.
        """
        time_ = os.path.normpath(self.path).split(os.sep)[-2]
        date = os.path.normpath(self.path).split(os.sep)[-3]
        infos = {
            "second": int(time_[-2:]),
            "minute": int(time_[2:4]),
            "hour": int(time_[:2]),
            "day": int(date[-2:]),
            "month": int(date[4:6]),
            "year": int(date[:4]),
        }

        if infos["second"] >= 60:
            infos["second"] = 59

        try:
            datetime_ = datetime.datetime(**infos)
        except ValueError:
            print('error on path:', self.path)
            raise

        return datetime_

    @property
    def environment_map(self):
        """
        :returns: EnvironmentMap object.
        """
        if self.format_:
            return EnvironmentMap(self.path, self.format_)
        else:
            return EnvironmentMap(self.path)

    def sun_position(self, method="coords"):
        """
        :returns: (elevation, azimuth)
        """
        if method == "intensity":
            self.init_properties()
            return sunutils.sunPosFromEnvmap(self._envmap)
        elif method == "coords":
            latitude = 46.778969
            longitude = -71.274914
            elevation = 125

            if self.datetime < datetime.datetime(2013, 12, 25, 10, 10, 10):
                latitude, longitude = 40.442794, -79.944115
                elevation = 300

            d = self.datetime
            if self.datetime.tzinfo is None:
                #TODO get timezone from latitude and longitude
                # d = pytz.timezone('US/Eastern').localize(self.datetime, is_dst=False)
                d += datetime.timedelta(hours=+4)


            return sunutils.sunPosFromCoord(latitude, longitude, d, elevation=elevation)
