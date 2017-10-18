k2movie: Create movies of TPF super stamps
==========================================

**Put Target Pixel Files (TPFs) obtained by NASA's Kepler/K2 missions into superstamps and generate images and movies.**



k2movie is currently under construction.

Example Work Flow
-----------------

To work with the TPF files quickly we need a library of files that are easy to access quickly. This is done by creating single files which contain all pixels values, saved as binary HDF5 format tables. These can then be queried quickly using *dask* or *pandas*. The database building is expensive in time, but only happens once. Once built, it allows movies and images of the focal plane to be created in seconds to minutes.

To build the database use:
``k2movie database -c CAMPAIGN_NO -ch CHANNEL_NO``

where CAMPAIGN_NO and CHANNEL_NO are campaign and channel numbers that you want to process. The default is to process all campaigns and channels.

This will create a database directory, download all TPFs from MAST in stages and create a file for each K2 campaign and channel. However, there are >2.5 Tb of data and this may take weeks. If you have a local copy of the TPFs and wish to build the database using it use

``k2movie database -in INPUT_DIRECTORY -c CAMPAIGN_NO -ch CHANNEL_NO``

However, this assumes that the structure of the directories containing the TPFs is the same as the MAST structure.

*Note: This will also build a set of WCS coordinates to use in the package directory.*


Coming Soon
-----------


``k2movie inspect NAME``
``k2movie movie NAME``



Attribution
-----------
Created by Christina Hedges at the NASA Kepler/K2 Guest Observer Office.
