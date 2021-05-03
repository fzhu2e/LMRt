#!/usr/bin/env bash

cd tutorial && make && cd -
make html
cp -r _build/html/* ../docs


