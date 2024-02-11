#!/bin/bash

# check clang-format
if ! command -v clang-format &> /dev/null
then
    echo "clang-format could not be found. Please install it before running this script."
    exit
fi

find . -not -path "./build/*" \
    -not -path "./third_party/*"\
    -type f -name "*.cpp" -o -name "*.hpp" \
    -exec clang-format -i {} \;

echo "Formatting complete."

