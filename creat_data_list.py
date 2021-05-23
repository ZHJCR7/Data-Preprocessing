"""
Creat txt file to list the set of the dataset

Author: Jeffrey Chao
Data: 2021.05.20
"""
import os
import argparse
import json

def get_video_file_list(dirname, outputpath):
    postfix = set(['mp4'])  # 设置要保存的文件格式
    # json_file = open(json_path, "r")  # 读取json文件读取label
    # json_dict = json.load(json_file)
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            if True:        # 保存全部文件名。若要保留指定文件格式的文件名则注释该句
                if apath.split('.')[-1] in postfix:   # 匹配后缀，只保存所选的文件格式。若要保存全部文件，则注释该句
                    try:
                        apath = apath.replace("\\", "/")
                        # video_name = apath.split("/")[-1]
                        # label = json_dict[video_name]
                        with open(outputpath, 'a+') as fo:
                            fo.writelines(apath.replace("\\", "/"))
                            fo.write('\n')
                    except:
                        pass    # 所有异常全部忽略即可

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_source_path', type=str, default='./FMFCC', help='Source Video Path')
    parser.add_argument('--video_pathname_list', type=str, default='./data_list', help='Save the list of video path')

    args = parser.parse_args()

    ## Parameters parsing
    video_source_path = args.video_source_path
    video_pathname_list = args.video_pathname_list

    video_path_list_filename = video_pathname_list + "/video_path_list.txt"

    if not os.path.exists(video_pathname_list):
        os.makedirs(video_pathname_list)

    if os.path.exists(video_path_list_filename):
       os.remove(video_path_list_filename)

    get_video_file_list(video_source_path, video_path_list_filename)

if __name__ == '__main__':
    main()