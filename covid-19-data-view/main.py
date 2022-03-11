# Built-In Python
import json
from argparse import ArgumentParser
import re
import logging

# Custom
from pathlib import Path

from dataProcessor import CovidDataProcessor

if __name__ == '__main__':

    CONFIG_PATH = Path('config.json')
    if not CONFIG_PATH.exists():
        raise Exception(f"{CONFIG_PATH} does not exist")

    # Load the data
    processor = CovidDataProcessor()
    CONFIG = processor.getConfig(CONFIG_PATH)
    processor.loadData(CONFIG['main_data_path'], 'main')
    processor.loadData(CONFIG['codebook_path'], 'codes')
    processor.loadData(CONFIG['coordinates_path'], 'coordinates')
    processor.loadData(CONFIG['dates_path'], 'dates')

    # Parse command line args (load data first so we can show good help)
    parser = ArgumentParser(description="A tool to visualize Covid-19 Data mainly obtained from Our World In Data")
    parser.add_argument('-x', default=['date'], nargs='*',
                        help='X fields to analyze. Use "-show fields" to see a full list with descriptions.')
    parser.add_argument('-y', default=['new_cases_smoothed_per_million'], nargs='*',
                        help='Y fields to analyze. Use "-show fields" to see a full list with descriptions.'
                             'Prefix a field with "mean_" to average over all locations.')
    parser.add_argument('-locations', default=['Canada'], nargs='*',
                        help='Which countries to analyze. Use "-show locations" to see a full list.')
    parser.add_argument('-filters', nargs='*',
                        help="Filters to narrow down which locations are shown. "
                             "Use format: filterName1=value1 filterName2=value2 ")
    parser.add_argument('-startDate', default='2020-01-01', help='Analyze the data beginning on this date')
    parser.add_argument('-endDate', default=None, help='Analyze the data before this date')
    parser.add_argument('-show', default=[], nargs='*', choices={'locations', 'fields', 'plot', 'shortcuts'},
                        help='Things to show on screen')
    parser.add_argument('-plotStyles', default=['all_at_once'], nargs='*', choices={'one_at_a_time', 'all_at_once'},
                        help='How the list of -y fields should be shown')
    parser.add_argument('-outdir', help='Save all plots to this directory')
    parser.add_argument('-verbosity', default='INFO', type=str.upper,
                        choices={'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'},
                        help='Verbosity level')
    parser.add_argument('-saveLocations', help='Save the locations used as a json for future use')
    parser.add_argument('-legendStyle', default='standard', choices={'standard', 'on_curve'},
                        help='How the legend should be displayed')
    parser.add_argument('-normalize', action='store_true', help='All values will be normalized between 0 and 1')
    parser.add_argument('-fillNans', action='store_true',
                        help='Fill gaps in plot with most recent value. i.e. [1, 2, nan, nan, 3, 4] '
                             'becomes [1, 2, 2, 2, 3, 4]')

    args = parser.parse_args()

    # Setup Logger
    level = logging.getLevelName(args.verbosity)
    logging.getLogger().setLevel(logging.INFO)

    if args.show:
        # Print some info
        if 'locations' in args.show:
            locations_per_line = 10
            all_locations = processor.getAllLocations()
            location_string = '\n\t'.join(
                [', '.join(all_locations[i:i + locations_per_line])
                 for i in range(0, len(all_locations), locations_per_line)]
            )
            print(f"Locations:\n\t{location_string}\n")

        if 'fields' in args.show:
            all_fields = '\n\t'.join([f"{col:<50}: {processor.getColumnDescription(col)}"
                                      for col in processor.getAllColumns()])
            print(f"Fields:\n{all_fields}\n")

        if 'shortcuts' in args.show:
            print(
                f"Shortcuts:\n"
                f"\tLocations: {', '.join(processor.getPresetNames())}\n"
                f"\tDates: {', '.join(processor.getDateNames())}\n"
                f"\tFilters: {', '.join(processor.getFilterNames())}"
            )

    # Parse filters
    filters = {}
    if args.filters:
        for f in args.filters:
            key, val = f.split('=')
            key = key.strip()
            val = val.strip()
            if val.isdigit():
                filters[key] = int(val)
            elif re.match(r'^-?\d+(?:\.\d+)$', val) is not None:
                filters[key] = float(val)
            else:
                filters[key] = val

    # If args.startDate or args.endDates use shortcuts (like Variant names), convert to a date string
    start_date = processor.getDate(args.startDate) if args.startDate in processor.getDateNames() else args.startDate
    end_date = processor.getDate(args.endDate) if args.endDate in processor.getDateNames() else args.endDate

    # Get a Dataset object for each X Field
    DATA = [processor.getDataset(args.locations, start_date, end_date, x_field, args.y, filters) for x_field in args.x]

    main_df = processor.getData('main')
    coordinates_df = processor.getData('coordinates')
    config = processor.getConfig(CONFIG_PATH)

    for d_set in DATA:
        dataframe = d_set.build(main_df, coordinates_df, config)
        if d_set.xField in dataframe:
            data_curves = processor.buildCurves(dataframe, d_set, normalize=args.normalize, fillNans=args.fillNans)

            if data_curves:
                # Show and/or Save Plot(s)
                if 'plot' in args.show or args.outdir:
                    processor.plotCurves(data_curves, plotStyles=args.plotStyles, saveDir=args.outdir, show='plot' in args.show,
                                         legendStyle=args.legendStyle)

                if args.saveLocations:
                    presets_path = Path(CONFIG['custom_location_presets_path'])
                    if not presets_path.parent.exists():
                        presets_path.parent.mkdir(parents=True)

                    presets = {}
                    if presets_path.exists():
                        with open(presets_path, 'r') as jf:
                            old_presets = json.load(jf)
                            presets = old_presets

                    new_preset = {args.saveLocations: {'countries': dataframe.location.unique().tolist()}}
                    presets.update(new_preset)
                    with open(presets_path, 'w') as jf:
                        json.dump(presets, jf, indent=4)
            else:
                logging.error(f"There was no available data for the given parameters")
        else:
            logging.error(f"{d_set.xField} is not a valid field. Valid fields:\n{dataframe.columns.values}")
