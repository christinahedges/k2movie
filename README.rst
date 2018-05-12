k2movie: Create movies of TPF super stamps
==========================================

**Put Target Pixel Files (TPFs) obtained by NASA's Kepler/K2 missions into superstamps and generate images and movies.**

k2movie is currently under construction.

Example Work Flow
-----------------

To work with the TPF files quickly we need a library of files that are easy to access quickly. This is done by creating single files which contain all pixels values, saved as binary HDF5 format tables. These can then be queried quickly using *dask* or *pandas*. The database building is expensive in time, but only happens once. Once built, it allows movies and images of the focal plane to be created in seconds to minutes, allowing users to quickly investigate any targets they like.

To build the database use:
``k2movie build -c CAMPAIGN_NO -ch CHANNEL_NO``

where CAMPAIGN_NO and CHANNEL_NO are campaign and channel numbers that you want to process. The default is to process all campaigns and channels.

This will create a database directory, download all TPFs from MAST in stages and create a file for each K2 campaign and channel. However, there are >2.5 Tb of data and this may take weeks. If you have a local copy of the TPFs and wish to build the database using it use

``k2movie build -in INPUT_DIRECTORY -c CAMPAIGN_NO -ch CHANNEL_NO``

The following command will show you the completeness of your database:

``k2movie list DIRECTORY``

Coming Soon
-----------

``k2movie movie NAME``


To Do
-----

* Automatically find campaign and channel with K2fov
* Tutorial
* Add some unit tests
