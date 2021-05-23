## Data Pre-processing

In this repository, we designed the program to sub-frame the database video and extract face information. The program is able to perform face recognition on skewed video and save the detection results in a TXT file.

### Usage

- ###### to get the video list path

```python
python creat_data_list.py
```

- ###### to extract the landmark of the face from the sub-frame

```python
python write_landmark.py
```

### TXT File

```
./output/FMFCC/train/train_video/video/10000002/0001.jpg,None,1,502,965,348,810,370,651,371,704,378,755,387,805,404,852,430,894,465,932,508,963,558,974,610,967,660,939,702,904,735,864,758,820,773,771,786,721,793,669,401,593,426,562,464,549,505,551,543,565,608,564,648,555,692,557,730,574,755,607,573,621,571,646,569,671,566,697,529,741,547,744,566,748,585,747,604,744,445,633,467,617,495,618,516,638,492,643,464,641,633,643,654,624,682,626,705,643,683,651,654,650,494,827,517,798,546,781,564,786,581,783,611,802,639,833,612,864,583,877,563,879,543,876,517,862,508,827,546,807,563,808,580,809,624,832,582,845,564,846,546,844
```

|     1      |     1     |   1   |    4     |    68    |
| :--------: | :-------: | :---: | :------: | :------: |
| image path | mask path | label | boundary | landmark |

