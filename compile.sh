#!/usr/bin/env bash
cd packages/resample2d_package/
python setup.py develop
cd ../correlation_package
python setup.py develop
cd ../channelnorm_package
python setup.py develop
cd ..
