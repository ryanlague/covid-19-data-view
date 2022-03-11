# Built-In Python
from pathlib import Path
import logging
import copy
import warnings
import json

# Third-Party
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Custom
from dataset import Dataset, DatasetFilterer
from labelLines import labelLines


class CovidDataProcessor:
    def __init__(self):
        self.dfs = {}
        self._config = {}

    @classmethod
    def getConfig(cls, path='config.json'):
        # Load the Config file
        with open(path) as jf:
            config = json.load(jf)
        return config

    def loadData(self, path, name='main'):
        if Path(path).exists():
            df = pd.read_csv(path)
            self.dfs[name] = df
        else:
            raise Exception(f"{path} does not exist")

    def getData(self, name='main'):
        return self.dfs[name]

    def getColumnDescription(self, columnName):
        df = self.getData('codes')
        if columnName in df['column'].values:
            info = df.query('column == @columnName')
            return info.description.values[0]
        else:
            return ""

    def getAllLocations(self, dataName='main'):
        df = self.getData(dataName)
        return df.location.unique().tolist()

    def getAllColumns(self, dataNames=None):
        dataNames = dataNames or ['main', 'coordinates']
        dataNames = dataNames if isinstance(dataNames, list) else [dataNames]
        columns = []
        for name in dataNames:
            df = self.getData(name)
            columns.extend(df.columns.values)
        columns.extend(Dataset.DynamicFields.fields)
        return columns

    def getDate(self, eventName):
        return self.getData('dates').query('name == @eventName').date.values[0]

    @classmethod
    def getPresetNames(cls):
        base_presets = list(Dataset.Preset.presets.keys())
        custom_presets_path = Path(cls.getConfig('custom_presets_path'))
        custom_presets = Dataset.Preset.getCustomPresets(custom_presets_path)
        return base_presets + custom_presets

    def getDateNames(self):
        return self.getData('dates').name.values

    @classmethod
    def getFilterNames(cls):
        return DatasetFilterer.getFilterNames()

    @classmethod
    def getDataset(cls, locations, startDate, endDate, xField, yFields, filters):
        d = Dataset(locations, startDate=startDate, endDate=endDate, xField=xField, yFields=yFields, filters=filters)
        return d

    @classmethod
    def buildCurves(cls, df, dataset, normalize=False, fillNans=True):
        curve_filter = CurveFilter()

        # All unique countries in our data
        countries = df.location.unique()

        # Initialize a list to store PlottableCurve objects
        curves = []

        # If the same X comes up multiple times, only keep one (composite the Ys below)
        x = df[dataset.xField].unique()
        x = np.sort(x)

        # Make sure X includes the whole range of values
        # i.e. if X is a date, make sure every date is represented (even with no value)
        # so the X-axis is linear
        if curve_filter.shouldFilter(dataset.xField, x):
            # If we filter X, we must also filter y.index (below)
            # TODO: Maybe loop through the datasets here to keep all the filtering together
            x = curve_filter.filter(x, dataset.xField)
        x = curve_filter.fillGaps(x, dataset.xField)

        # Build PlottableCurve objects from the data
        for yfield in dataset.yFields:
            for country in countries:
                this_country_df = DatasetFilterer(df).byLocation(country).df
                # Make sure yfield is a valid column
                if yfield in df.columns.values:
                    # If Y contains multiple entries for the same X, average them
                    y = this_country_df.groupby(dataset.xField)[yfield].mean()
                    # If X is atypical (i.e. not dates), make sure it is still linear
                    # (we are filtering y.index, which are values in x)
                    y.index = curve_filter.filter(y.index, dataset.xField)
                    # If Y contains multiple entries for the same X, average them again after filtering
                    y = curve_filter.aggregateDoubles(y, 'mean')

                    if not np.isnan(y).all():
                        # If there are missing values for X (ie no data on the weekends), add nan values so X is linear
                        missing_idx = [_x for _x in x if _x not in y.index.values]

                        if missing_idx:
                            missing_idx_nans = pd.Series([np.nan] * len(missing_idx), index=missing_idx)
                            y = pd.concat([y, missing_idx_nans]).sort_index()
                            # Show a list of missing indices, but don't show the whole thing if it's too long
                            MAX_LEN = 100
                            missing_idx_str = f"{str(missing_idx):.{MAX_LEN}s}{'...]' if len(str(missing_idx)) >= MAX_LEN else ''}"
                            logging.warning(f"Nan values were added to {country}-{yfield} "
                                            f"because it was missing some values:\n{missing_idx_str}")

                        # If X and Y are valid, create a PlottableCurve object. This represents a single curve on a plot
                        if len(x) == len(y):
                            label = f"{country} - {yfield}" if len(dataset.yFields) > 1 else f"{country}"
                            curve = PlottableCurve(
                                x, y,
                                xField=dataset.xField, yField=yfield,
                                country=country,
                                label=label,
                                xLabel=dataset.xField, yLabel=yfield
                            )
                            curves.append(curve)
                        else:
                            raise Exception(f"X and Y have different lengths for {country} - {yfield}.\n"
                                            f"len(x) = {len(x)} | len(y) = {len(y)}")
                    else:
                        logging.warning(f"{country} - {dataset.xField} by {yfield} was not added because "
                                        f"there is no available data in the requested date range")
                elif yfield.startswith('mean'):
                    # If a valid Y field name is suffixed with 'mean_'
                    # create a mean of the suffixed Y field over all Countries in dataset
                    if yfield not in [curve.yField for curve in curves]:
                        real_yfield_name = yfield.replace('mean_', '')
                        dataset_for_mean = copy.copy(dataset)
                        dataset_for_mean.yFields = [real_yfield_name]
                        curves_to_mean = cls.buildCurves(df, dataset=dataset_for_mean, normalize=normalize,
                                                         fillNans=fillNans)
                        valid_curves = [curve for curve in curves_to_mean if not np.isnan(curve.y).all()]
                        countries = [curve.country for curve in valid_curves]
                        ys = [curve.y for curve in valid_curves]
                        if ys:
                            # Numpy has some annoying warnings about nan values when using nanmean
                            # (despite the fact that nan is in the name).
                            # Silence these
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", category=RuntimeWarning)
                                Y = np.nanmean(ys, axis=0)

                            # A plottable curve represent the mean of several curves
                            mean_curve = PlottableCurve(
                                x, Y, xField=dataset_for_mean.xField, yField=yfield, country=', '.join(countries),
                                label=f'Mean {real_yfield_name} ({len(curves_to_mean)} countries)',
                                xLabel=dataset_for_mean.xField, yLabel=real_yfield_name
                            )
                            curves.append(mean_curve)
                        else:
                            logging.warning(f"{yfield} was removed because there is not available data")

                else:
                    raise Exception(f"{yfield} is not a valid column.\nColumns:\n{df.columns.values}")

        if normalize:
            def NormalizeData(data):
                return (data - np.min(data)) / (np.max(data) - np.min(data))

            for curve in curves:
                curve.y = NormalizeData(curve.y)

        if fillNans:
            for curve in curves:
                if np.isnan(curve.y).any() and not np.isnan(curve.y).all():
                    curve.y.fillna(method='ffill', inplace=True)  # Fill with prev val
                    curve.y.fillna(method='backfill', inplace=True)  # Fill starting-nans with first non-nan val
        return curves

    @classmethod
    def plotCurves(cls, curves, plotStyles=None, saveDir=None, show=True, legendStyle='standard'):

        # Sanitize
        plotStyles = plotStyles or ['all_at_once']
        plotStyles = plotStyles if isinstance(plotStyles, list) else [plotStyles]
        PLOT_TYPES = {'all_at_once', 'one_at_a_time'}
        assert all([plot in PLOT_TYPES for plot in plotStyles])

        # Ensure the data_curves is a list
        curves = curves if isinstance(curves, list) else [curves]

        # SaveDir is a directory to save Plot images
        saveDir = Path(saveDir) if saveDir else None

        # All Xs should be the same
        x = curves[0].x
        x_field = curves[0].xField

        # Format X-Tick labels
        x_tick_filt = (lambda l: l.strftime('%y-%m')) if x_field == 'date' else None

        # Do the plotting
        for plotStyle in plotStyles:
            if plotStyle == 'all_at_once':
                y_label = LabelMaker.aggregateLabels([curve.yLabel for curve in curves])
                countries = LabelMaker.aggregateLabels([curve.country for curve in curves])
                title = f"{y_label} by {x_field}"
                save_path = saveDir.joinpath(f"{countries} - {y_label} by {x_field}.png") if saveDir else None
                plotter = Plotter(x, title=title, xLabel=x_field, yLabel=y_label, xTickFilt=x_tick_filt)
                plotter.plotCurves(curves, savePath=save_path, show=show, separate=False, legendStyle=legendStyle)

            elif plotStyle == 'one_at_a_time':
                for curve in curves:
                    if len(curve.x) and len(curve.y):
                        title = f"{curve.country}\n{curve.yLabel} by {curve.xLabel}"
                        save_path = saveDir.joinpath(f"{title}.png") if saveDir else None
                        plotter = Plotter(x, title=title, xLabel=curve.xLabel, yLabel=curve.yLabel,
                                          xTickFilt=x_tick_filt)
                        plotter.plotCurve(curve, savePath=save_path, show=show)
                    else:
                        raise Exception(f"There is no data for {curve.country}")
            else:
                raise Exception(f'Unknown plotStyle: {plotStyle}')


class CurveFilter:
    def __init__(self):
        self.fieldFilterFuncs = {
            'stringency_index': round
        }
        self.dtypeFilterFuncs = {
            np.dtype(np.float64): round
        }
        self.extendFuncs = {
            'date': self.getDateRange,
            'stringency_index': lambda x: list(range(0, 101))  # Ignore X. Return range from 0 to 100
        }

    @classmethod
    def getDateRange(cls, dates):
        date_range = pd.date_range(dates.min(), dates.max()).values
        return [pd.to_datetime(str(dt)) for dt in date_range]

    def shouldFilter(self, name, x):
        is_filtered_field = name in list(self.fieldFilterFuncs)
        is_filtered_dtype = x.dtype in list(self.dtypeFilterFuncs)
        return is_filtered_field or is_filtered_dtype

    def filter(self, x, name):
        if self.shouldFilter(name, x):
            if name in list(self.fieldFilterFuncs):
                return [self.fieldFilterFuncs[name](el) if not np.isnan(el) else el for el in x]
            elif x.dtype in list(self.dtypeFilterFuncs):
                return [self.dtypeFilterFuncs[x.dtype](el) if not np.isnan(el) else el for el in x]
            else:
                return x
        else:
            return x

    def fillGaps(self, x, name):
        if name in self.extendFuncs:
            return self.extendFuncs[name](x)
        else:
            return list(range(int(min(x)), int(max(x) + 1)))

    @classmethod
    def aggregateDoubles(cls, x, aggrType='mean'):
        group = x.groupby(x.index)
        if aggrType == 'mean':
            return group.mean()
        else:
            raise Exception(f"Unknown aggregation type {aggrType}")


class PlottableCurve:
    def __init__(self, x, y, xField, yField, country, label, xLabel, yLabel):
        self.x = x
        self.y = y
        self.xField = xField
        self.yField = yField
        self.country = country
        self.label = label
        self.xLabel = xLabel
        self.yLabel = yLabel

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.label}>"

    @property
    def correlationCoefficient(self):
        return np.corrcoef(np.arange(len(self.x)), self.y)[1, 0]


class LabelMaker:
    @classmethod
    def aggregateLabels(cls, labels):
        unique = list(set(labels))
        aggregated_labels = unique
        substring_counts = {}
        if len(unique) > 1:
            for label in unique:
                for n in range(2, len(label) + 1):
                    if n not in substring_counts:
                        substring_counts[n] = {}
                    for i in range(len(label) - n + 1):
                        substring = label[i:i + n]
                        if substring not in substring_counts[n]:
                            substring_counts[n][substring] = 1
                        else:
                            substring_counts[n][substring] += 1
            possible_substrings = [k for k, v in substring_counts.items() if max(v.values()) > 1]
            if possible_substrings:
                longest_substrings = substring_counts[max([k for k, v in substring_counts.items() if max(v.values()) > 1])]
                best_substrings = list(
                    {k: v for k, v in longest_substrings.items() if v == max(longest_substrings.values())})
                best_substring = best_substrings[0] if best_substrings else None

                aggregated_labels = list(
                    set([label if best_substring not in label else best_substring for label in labels]))
        return ', '.join(aggregated_labels)


class Plotter:
    def __init__(self, x, title, xLabel, yLabel, numXTicks=10, xTickFilt=None):
        self.x = x
        self.title = title
        self.xLabel = xLabel
        self.yLabel = yLabel
        self.numXTicks = numXTicks
        self.xTickFilt = xTickFilt

        # When we create a new Plotter, clear any old plots
        plt.clf()

    def decorate(self, legendStyle='standard', prettyTitle=True, prettyCoordinateLabels=True, prettyLegend=True):
        def makePretty(string):
            words = string.split('_')
            return ' '.join([w.title() for w in words])

        title = makePretty(self.title) if prettyTitle else self.title
        x_label = makePretty(self.xLabel) if prettyCoordinateLabels else self.xLabel
        y_label = makePretty(self.yLabel) if prettyCoordinateLabels else self.yLabel

        plt.title(title)

        if legendStyle:
            if legendStyle == 'standard':
                legend = plt.legend()
                if prettyLegend:
                    for t in legend.get_texts():
                        t.set_text(makePretty(t.get_text()))
            elif legendStyle == 'on_curve':
                labelLines(plt.gca().get_lines(), align=False, ha='center', va='center')
            else:
                raise Exception(f"{legendStyle} is not a valid legendStyle")

        ticks = self.x[::len(self.x) // self.numXTicks] if self.numXTicks else self.x
        labels = [self.xTickFilt(tick) for tick in ticks] if self.xTickFilt else ticks

        plt.xticks(ticks, labels=labels)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

    def plotCurve(self, curve, savePath=None, show=False, legendStyle='standard'):
        return self.plotCurves([curve], savePath=savePath, show=show, legendStyle=legendStyle)

    def plotCurves(self, curves, savePath=None, show=False, separate=False, legendStyle='standard'):
        if separate:
            for curve in curves:
                self._plotCurves([curve], savePath=savePath, show=show, legendStyle=legendStyle)
        else:
            self._plotCurves(curves, savePath=savePath, show=show, legendStyle=legendStyle)

    def _plotCurves(self, curves, savePath=None, show=False, legendStyle='standard'):
        for curve in curves:
            if len(curve.x) and len(curve.y):
                plt.plot(curve.x, curve.y, label=curve.label)
            else:
                raise Exception(f"There is no data for {curve.country}")

        self.decorate(legendStyle=legendStyle)

        if savePath:
            savePath = Path(savePath)
            if not savePath.parent.exists():
                savePath.parent.mkdir(parents=True)
            logging.info(f"Saved {savePath}")
            plt.savefig(savePath)
        if show:
            plt.show()
