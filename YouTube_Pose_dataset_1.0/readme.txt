YouTube Pose Dataset
--------------------

The videos were downloaded from YouTube and automatically scaled and cropped (using an upper body detector) so that the shoulder width is 100 pixels wide (on average across the video). 100 frames from each of the scaled and cropped videos were randomly chosen and manually annotated with upper body joints. The Head, Right wrist, Left wrist, Right elbow, Left elbow, Right shoulder and Left shoulder were annotated.

Annotated frames are contained within the GT_frames.zip package.

The matlab file YouTube_Pose_dataset.mat contains a structure array called 'data'. There are 50 elements in the array, one for each video of the dataset. Each element is structured as follows:

data(i).url - string for the youtube weblink for video i
data(i).videoname - string for the code name of the youtube video
data(i).locs - 2 by 7 by 100 array containing 2D locations for the ground truth upper body joints. Row 1 are x values and Row 2 are y values. Columns are formatted from left to right as: Head, Right wrist, Left wrist, Right elbow, Left elbow, Right shoulder and Left shoulder (Person centric).
data(i).frameids = 1 by 100 array containing the frame indicies which were annotated.
data(i).label_names - cell array of strings for corresponding body joint labels
data(i).crop - 1 by 4 array giving the crop bounding box [topx topy botx boty] from the original video
data(i).scale - value the video should be scaled by
data(i).imgPath - cell array containing paths to the pre scaled and cropped annotated frames 
data(i).origRes - 1 by 2 array [height,width] resolution of original video
data(i).isYouTubeSubset - boolean, true if video belongs to the YouTube Subset dataset

e.g. data(i).imgPath{f} refers to video i and frame f, with frame id data(i).frameids(f) and joint locations data(i).locs(:,:,f)

Note: ground truth body joint locations correspond to cropped and scaled videos (original videos are first cropped, then scaled using the appropriate values).

 
