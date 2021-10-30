#!/bin/bash
# execute at the folder where the file is located


function download_depth_files(){
    wget 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip'
    wget 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_velodyne.zip'
    wget 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip'

    unzip 'data_depth_annotated.zip'
    unzip 'data_depth_velodyne.zip'
    unzip 'data_depth_selection.zip'

    rm -f 'data_depth_annotated.zip'
    rm -f 'data_depth_velodyne.zip'
    rm -f 'data_depth_selection.zip'
}

function Download_files(){
	files=($@)
	for i in ${files[@]}; do
		if [ ${i:(-3)} != "zip" ]; then
				date="${i:0:10}"       # 2011_09_26
				name=$(basename $i /)  # 2011_09_26_drive_0009_sync
				shortname=$name'.zip'  # 2011_09_26_drive_0009_sync.zip
				fullname=$(basename $i _sync)'/'$name'.zip'  # 2011_09_26_drive_0009/2011_09_26_drive_0009_sync.zip
				echo 'shortname: '$shortname
		else
			echo 'Something went wrong. Input array names are probably not correct! Check this manually!'
		fi
        echo "Downloading: "$shortname
		wget 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/'$fullname
        unzip -o $shortname
		rm -f $shortname
		mv $i'proj_depth' $date'/'$name

        # Remove first 5 and last 5 files of camera images
        # cd $date'/'$name'/image_02/data' 
        # ls | sort | (head -n 5) | xargs rm -f
        # ls | sort | (tail -n 5) | xargs rm -f
        # cd '../../image_03/data'
        # ls | sort | (head -n 5) | xargs rm -f
        # ls | sort | (tail -n 5) | xargs rm -f
        # cd ../../../../

		rm -rf $name
    done
}

download_depth_files

cd 'train/'

#train_files=($(ls -d */ | sed 's#/##'))
#train_files=('2011_09_26_drive_0009_sync')
train_files=($(ls -d */))
Download_files ${train_files[@]}

cd '../val/'
valid_files=($(ls -d */))
Download_files ${valid_files[@]}

cd '..'
