import setuptools

setuptools.setup(
        name="europegeo",
        author="Alberto Cattaneo",
        url="https://github.com/AlCatt91/Europe-Geoguesser/",
        packages=setuptools.find_packages(where='src'),
        package_dir={"": "src"}
)