#!/bin/bash

echo "Running build_cluster_x_y_and_draw_summary.C ..."
root build_cluster_x_y_and_draw_summary.C
if [ $? -ne 0 ]; then
    echo "Error: build_cluster_x_y_and_draw_summary.C failed."
    exit 1
fi

echo "Running track_reco_eventDisplay.C ..."
root track_reco_eventDisplay.C
if [ $? -ne 0 ]; then
    echo "Error: track_reco_eventDisplay.C failed."
    exit 1
fi

echo "All jobs finished successfully."
