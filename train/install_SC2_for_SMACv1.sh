#!/bin/bash
# Install SC2 and add the custom maps

apt install -y unzip vim

export SC2PATH=$HOME/StarCraftII
cd ~
if [ ! -d $SC2PATH ]; then
	wget -c http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.6.2.69232.zip
	unzip -P iagreetotheeula SC2.4.6.2.69232.zip
	rm -rf SC2.4.6.2.69232.zip
fi

cd $SC2PATH

MAP_DIR=$SC2PATH/Maps/

if [ ! -d $MAP_DIR ]; then
        mkdir -p $MAP_DIR
fi

wget https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip
unzip SMAC_Maps.zip
mv SMAC_Maps $MAP_DIR
rm -rf SMAC_Maps.zip

