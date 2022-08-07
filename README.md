# Datasets

## Ready

- Allen 2021, Natural Scenes
- Bonner 2021, Object2Vec
- Stringer 2019, Mouse 10K

## In progress

- BOLD 5000

## Environment variables

`BONNER_DATASETS_CACHE`

## Things to do

- TODO add default voxel selection to `bonner2021.object2vec` that matches what Mick was using
- TODO add localizer info to `allen2021.natural-scenes`
- TODO add localizer T-values and other voxel metadata (https://cvnlab.slite.com/p/channel/CPyFRAyDYpxdkPK6YbB5R1/notes/G5dUBGBxMo)

## Notes

- to use the NSD, you will need to set the `AWS_SHARED_CREDENTIALS_FILE` environment variable, typically `~/.aws/credentials`
