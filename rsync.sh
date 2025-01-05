#!/bin/bash

# Define the source and destination
SOURCE_DIR="./"
DESTINATION="cluster-ai:/home1/lee1jun/develop/spch"

# Use rsync with --exclude-from to exclude files listed in .gitignore
rsync -azvP --exclude-from="${SOURCE_DIR}.rsyncignore" "$SOURCE_DIR" "$DESTINATION"