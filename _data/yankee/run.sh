#!/bin/bash

hadoop fs -cat /user/bjohnson/data/instagram/trickle/yankee-3/* | gzip -cd > yankee.jl

cat yankee.jl | jq -rc '[
    ._source.location.latitude,
    ._source.location.longitude,
    ._source.created_time
] | @tsv' > yankee.tsv

python filter-date.py

# --


