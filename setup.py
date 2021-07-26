from setuptools import setup

setup(
        name="geotools",
        version="0.0.1",
        author="Alberto Cattaneo",
        packages=["geo-tools", "streetview-scraping"],
        package_dir={"geo-tools":"geo-tools", "streetview-scraping":"streetview-scraping"},
        url="https://github.com/AlCatt91/Europe-Geoguesser/",
)