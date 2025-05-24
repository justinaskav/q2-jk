# ----------------------------------------------------------------------------
# Copyright (c) 2024, Justinas Kavoliūnas.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from setuptools import find_packages, setup

import versioneer

description = (
    "Misc. tools."
)

setup(
    name="q2-jk",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license="BSD-3-Clause",
    packages=find_packages(),
    author="Justinas Kavoliūnas",
    author_email="justinaskav@gmail.com",
    description=description,
    url="https://github.com/justinaskav/q2-jk",
    entry_points={
        "qiime2.plugins": [
            "q2_jk="
            "q2_jk"
            ".plugin_setup:plugin"]
    },
    package_data={
        "q2_jk": ["citations.bib", "templates/*"],
        "q2_jk.tests": ["data/*"],
    },
    zip_safe=False,
)
