# Built-In Python
import json
import logging
from pathlib import Path

# Third-Party
import pandas as pd
import numpy as np


class Dataset:
    class Preset:
        presets = {
            'all': {},
            'north_america': {'continents': ['North America']},
            'south_america': {'continents': ['South America']},
            'europe': {'continents': ['Europe']},
            'africa': {'continents': ['Africa']},
            'oceania': {'continents': ['Oceania']},
            'asia': {'continents': ['Asia']},
            'temperate_north': {'minLat': 30, 'maxLat': 60, 'useAbsoluteLats': False},
            'temperate_south': {'minLat': -60, 'maxLat': -30, 'useAbsoluteLats': False},
            'temperate': {'minLat': 30, 'maxLat': 60, 'useAbsoluteLats': True},
            'most_strict': {'minStringency': 70},
            'least_strict': {'maxStringency': 31.5},
            'most_dense': {'minDensity': 600},
            'least_dense': {'maxDensity': 5}
        }

        @classmethod
        def getBasePresets(cls):
            return cls.presets

        @classmethod
        def getCustomPresets(cls, path):
            custom_presets_path = Path(path) if path else None
            custom_presets = {}
            if custom_presets_path and custom_presets_path.exists():
                with open(custom_presets_path) as jf:
                    custom_presets = json.load(jf)
            return custom_presets

        @classmethod
        def getPresets(cls, customPresetsPath=None):
            base_presets = cls.getBasePresets()
            custom_presets = cls.getCustomPresets(customPresetsPath)
            return {**base_presets, **custom_presets}

    class DynamicFields:
        fields = ['days_from_winter_solstice', 'days_from_mid_winter']

    def __init__(self, locations, startDate, endDate, xField, yFields, filters):
        self.locations = locations
        self.startDate = startDate
        self.endDate = endDate
        self.xField = xField
        self.yFields = yFields
        self.filters = filters

    def build(self, fullDf, coordinatesDf, config=None):
        logging.info(f"Building Dataset...")
        all_countries = fullDf.location.unique()

        location_presets = self.Preset.getPresets(config.get('custom_location_presets_path') if config else None)

        errors = [l for l in self.locations if l not in location_presets and l not in all_countries]
        if not errors:

            # Store params which come from presets (found in Preset class)
            preset_kwargs = [location_presets[location] for location in self.locations
                             if location in location_presets]
            # Store non-preset Country names
            country_names = [location for location in self.locations if location in all_countries]

            dfs = []
            for preset in preset_kwargs:
                preset_df = DatasetFilterer(fullDf, coordinatesDf).byDate(self.startDate, self.endDate).filter(**preset).df
                dfs.append(preset_df)
            if country_names:
                country_df = DatasetFilterer(fullDf, coordinatesDf).byDate(self.startDate, self.endDate).byLocation(country_names).df
                dfs.append(country_df)

            if dfs:
                filtered_df = dfs[0]
                for df in dfs[1:]:
                    filtered_df = pd.merge(filtered_df, df, how='outer')

                if self.filters:
                    filtered_df = DatasetFilterer(filtered_df, coordinatesDf).byDate(self.startDate, self.endDate).filter(**self.filters).df

                column_adder = ColumnAdder(filtered_df)
                if coordinatesDf is not None:
                    column_adder.addCoordinates(latitudeName='latitude', longitudeName='longitude',
                                                coordinateData=coordinatesDf)
                    column_adder.addDaysFromWinterSolstice(name='days_from_winter_solstice',
                                                           midWinterName='days_from_mid_winter')
                    filtered_df = column_adder.df
                return filtered_df
        else:
            raise Exception(f"{errors} are not valid Countries or Presets")


class DatasetFilterer:
    def __init__(self, df, coordinatesDf=None):
        self.df = df
        self.coordinatesDf = coordinatesDf

    @classmethod
    def getFilterNames(cls, removeMainFilters=True):
        filter_names = cls.filter.__code__.co_varnames
        remove = ['self']
        if removeMainFilters:
            remove.extend(['countries', 'startDate', 'endDate'])
        filter_names = list(filter(lambda f: f not in remove, filter_names))
        return filter_names

    def filter(self, countries=None, startDate=None, endDate=None, minLat=None, maxLat=None,
               useAbsoluteLats=False, minStringency=None, maxStringency=None, minDensity=None, maxDensity=None,
               continents=None):
        self.byLocation(countries)
        self.byLatitude(minLat, maxLat, absolute=useAbsoluteLats)
        self.byDensity(minDensity, maxDensity)
        self.byStringency(minStringency, maxStringency)
        self.byContinent(continents=continents)
        self.byDate(startDate, endDate)
        return self

    def byDate(self, startDate=None, endDate=None):
        if startDate or endDate:
            self.df['date'] = pd.to_datetime(self.df['date'])
            if startDate:
                mask = self.df['date'] >= startDate
                self.df = self.df.loc[mask]
            if endDate:
                mask = self.df['date'] < endDate
                self.df = self.df.loc[mask]
        return self

    def byLocation(self, locations):
        if locations:
            locations = locations if isinstance(locations, (list, np.ndarray)) else [locations]
            self.df = self.df[self.df['location'].isin(locations)]
        return self

    def byLatitude(self, minLat, maxLat, absolute=False):
        if self.coordinatesDf is not None:
            if minLat or maxLat:
                self.coordinatesDf.loc[:] = self.coordinatesDf[self.coordinatesDf['Country'].isin(
                    self.df.location.values)]
                self.coordinatesDf['abs_latitude'] = self.coordinatesDf['latitude'].abs()
                latitude_to_use = 'abs_latitude' if absolute else 'latitude'
                if minLat:
                    self.coordinatesDf = self.coordinatesDf.query(f'{latitude_to_use} >= @minLat')
                if maxLat:
                    self.coordinatesDf = self.coordinatesDf.query(f'{latitude_to_use} <= @maxLat')

                self.df = self.df[self.df['location'].isin(self.coordinatesDf.Country.values)]
            return self
        else:
            raise Exception(f"{self} must have a coordinatesDf to filter by latitude")

    def byContinent(self, continents=None):
        # There are some "locations" that are not countries (i.e. World, European Union etc)
        # We filter them out by only keeping locations with a valid "continent" field.
        # These could be kept by passing in something like: continents=['World']
        real_continents = continents or ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']
        self.df = self.df.loc[self.df['continent'].isin(real_continents)]
        return self

    def byStringency(self, minStringency, maxStringency):
        if minStringency or maxStringency:
            countries = self.df.location.unique()
            correct_stringency = []
            for country in countries:
                if minStringency is None or self.df.loc[self.df['location'] == country]['stringency_index'].mean() >= minStringency:
                    if maxStringency is None or self.df.loc[self.df['location'] == country]['stringency_index'].mean() <= maxStringency:
                        correct_stringency.append(country)

            self.df = self.df.loc[self.df['location'].isin(correct_stringency)]
        return self

    def byDensity(self, minDensity, maxDensity):
        if minDensity:
            self.df = self.df.loc[self.df['population_density'] >= minDensity]
        if maxDensity:
            self.df = self.df.loc[self.df['population_density'] <= maxDensity]
        return self


class ColumnAdder:
    def __init__(self, df):
        self.df = df

    def getColumns(self):
        return self.df.columns.values

    def addCoordinates(self, latitudeName='latitude', longitudeName='longitude', coordinateData=None):
        coordinateData = coordinateData.loc[coordinateData.owid_name.isin(self.df.location.values)]
        if not coordinateData.empty:
            self.df = pd.merge(self.df, coordinateData, left_on=['location'], right_on=['owid_name'], how='outer')
            self.df.drop(['ISO 3166 Country Code', 'Country', 'owid_name'], axis=1, inplace=True)

            if latitudeName:
                self.df.rename({'latitude': latitudeName}, axis=1, inplace=True)
            if longitudeName:
                self.df.rename({'longitude': longitudeName}, axis=1, inplace=True)
        else:
            self.df[latitudeName] = np.nan
            self.df[longitudeName] = np.nan

        return self.df

    def addDaysFromWinterSolstice(self, name='days_from_winter_solstice', midWinterName='days_from_mid_winter'):
        northern_solstice_date = '12-21'
        southern_solstice_date = '06-20'
        get_solstices = lambda isNorthern, offset=0: [
            pd.to_datetime(d) + pd.DateOffset(offset)
            for d in [f'{year}-{northern_solstice_date if isNorthern else southern_solstice_date}'
                      for year in range(2019, 2023)]
        ]
        northern_solstices = get_solstices(isNorthern=True, offset=0)
        southern_solstices = get_solstices(isNorthern=False, offset=0)
        is_northern = lambda x: x.latitude >= 0
        filt = lambda x: (min([abs(pd.to_datetime(x.date) - solstice) for solstice in
                               (northern_solstices if is_northern(x) else southern_solstices)])).days
        self.df[name] = self.df.apply(filt, axis=1)
        if midWinterName:
            northern_midwinters = get_solstices(isNorthern=True, offset=int(365 / 8))
            southern_midwinters = get_solstices(isNorthern=False, offset=int(365 / 8))

            filt = lambda x: (min([abs(pd.to_datetime(x.date) - solstice) for solstice in
                                   (northern_midwinters if is_northern(x) else southern_midwinters)])).days
            self.df[midWinterName] = self.df.apply(filt, axis=1)
        return self.df
