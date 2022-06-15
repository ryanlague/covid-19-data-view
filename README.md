# Covid-19 Data View

View international data on Covid-19 as collected by https://ourworldindata.org/ in Python/

## Installation

### PyPI (for use as a module)
```bash
    pip install covid-19-data-view
```
### GitHub
```bash
clone https://github.com/ryanlague/covid-19-data-view.git
cd covid-19-data-view
pip install .
```

## Usage

### CLI
```bash
  # Show a plot for New Cases (smoothed over 7 days) over time for Canada, the United States and Mexico
  python3 main.py -x date -y new_cases_smoothed -location Canada "United States" Mexico -show plot
  
  # Show all possible locations and exit
  python3 main.py -show locations
  
  # Show all possible fields for the X and Y axes
  # WARNING: Not all of them work yet.
  python3 main.py -show fields
  
  # Show Vaccinations over time for all of europe
  python3 main.py -x date -y total_vaccinations -location europe -show plot
  
  # Show Total Cases over time for the year 2021 in locations with the highest population densities
  python3 main.py -x date -y total_cases -startDate 2021-01-01 -endDate 2022-01-01 -location most_dense -show plot

  # Show New Deaths (smoothed over 7 days) in Australia from the date Delta was declared a VOI to the date Omicron was declared a VUM
  python3 main.py -x date -y new_deaths_smoothed -startDate Delta -endDate Omicron -location Australia -show plot
  
  # Show shortcuts for groups of locations, date ranges, filters etc.
  python3 main.py -show shortcuts
  
  # Show a full list of parameters and exit
  python3 main.py --help
  
```

### As a module (Coming Soon)

```python

```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
MIT License

Copyright (c) 2022 Ryan Lague

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.