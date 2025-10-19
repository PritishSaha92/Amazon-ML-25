#!/bin/bash

# ===================================================================================
# VLM DATA PREPARATION PIPELINE
# ===================================================================================
# This script runs the entire offline data preparation process.



# # Run with default 10% validation split
# ./run.sh

# # Run with custom validation split (20%)
# ./run.sh 0.2

# # Run with no validation split
# ./run.sh 0.0




# Accept validation ratio as command-line argument, default to 0.1
VALIDATION_SPLIT_RATIO=${1:-0.1}

echo "======================================================================"
echo "STARTING VLM DATA PIPELINE"
echo "Validation Split Ratio: ${VALIDATION_SPLIT_RATIO}"
echo "======================================================================"

# # Step 1: Download images (30-60 min)
# echo -e "\n---> STEP 1: DOWNLOADING IMAGES..."
# python 1_download.py
# if [ $? -ne 0 ]; then
#     echo "❌ Error in download step. Exiting."
#     exit 1
# fi

# # Step 2: Resize images (10-15 min)
# echo -e "\n---> STEP 2: RESIZING IMAGES..."
# python 2_resize.py
# if [ $? -ne 0 ]; then
#     echo "❌ Error in resize step. Exiting."
#     exit 1
# fi

# # Step 3: Create JSONL files with validation split (1-2 min)
# echo -e "\n---> STEP 3: CREATING JSONL FILES..."
# python 3_create_jsonl.py --validation-split-ratio $VALIDATION_SPLIT_RATIO
# if [ $? -ne 0 ]; then
#     echo "❌ Error in JSONL creation step. Exiting."
#     exit 1
# fi

# # Step 4: Preprocess to tensors (30-60 min per split)
# echo -e "\n---> STEP 4: PREPROCESSING TO TENSORS..."
# python 4_preprocess.py --split all
# if [ $? -ne 0 ]; then
#     echo "❌ Error in preprocessing step. Exiting."
#     exit 1
# fi

# # Step 5: Convert to WebDataset (5-10 min per split)
# echo -e "\n---> STEP 5: CONVERTING TO WEBDATASET..."
# python 5_convert.py --split all
# if [ $? -ne 0 ]; then
#     echo "❌ Error in WebDataset conversion step. Exiting."
#     exit 1
# fi

# Delete old (broken) preprocessed data
rm -rf ./preprocessed_train ./preprocessed_validation ./preprocessed_test
rm -rf ./webdataset_train ./webdataset_validation ./webdataset_test

# Reprocess with ACTUAL tensors (will take 2-3 hours!)
python 4_preprocess.py --split all

# Convert to WebDataset
python 5_convert.py --split all --samples-per-shard 1000


echo -e "\n======================================================================"
echo "✅ PIPELINE COMPLETE!"
echo "======================================================================"
echo "Training WebDataset:   ./webdataset_train/"
echo "Validation WebDataset: ./webdataset_validation/"
echo "Test WebDataset:       ./webdataset_test/"
echo ""
echo "Next: Open 6_main.ipynb and update the WebDataset URL patterns"
echo "======================================================================"
