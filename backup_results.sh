#!/bin/bash

# chmod +x backup_results.sh
# ./backup_results.sh

# The name of the folder to backup
SOURCE_FOLDER="."

# The location where the backup will be stored
BACKUP_FOLDER="results_backup"

# The date to tag the backup with
DATE=$(date +%Y%m%d_%H%M%S)

# The final backup directory with the date tag
FINAL_BACKUP_FOLDER="${BACKUP_FOLDER}_${DATE}"

# Create the backup directory
mkdir -p "$FINAL_BACKUP_FOLDER"

# Use rsync to copy the models folder to the backup location
# --exclude patterns ensure that inputs.npy, outputs.npy, and model.pth files in subdirectories are not copied
rsync -av --exclude '*/inputs.npy' --exclude '*/outputs.npy' --exclude '*/model.pth' "$SOURCE_FOLDER"/ "$FINAL_BACKUP_FOLDER"/

echo "Backup of $SOURCE_FOLDER completed to $FINAL_BACKUP_FOLDER"
