"""
preprocess filename not to duplicate filename
"""
import os
import argparse

import shutil


def main(original_dir, target_dir):
    if os.path.exists(os.path.dirname(target_dir)) is False:
        os.mkdir(os.path.dirname(target_dir))
    if os.path.exists(target_dir) is False:
        os.mkdir(target_dir)
    
    for root, dirs, files in os.walk(original_dir):
        for f in files:
            org_filepath = os.path.join(root, f)
            filename = os.path.basename(org_filepath)
            dirname = os.path.basename(os.path.dirname(org_filepath))
            # join dirname and filename not to duplicate filename
            copy_filename =  dirname + '_' + filename
            copy_filepath = os.path.join(target_dir, copy_filename)
            # copy files to the target directory
            shutil.copy2(org_filepath, copy_filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_dir',
                                 dest='original_dir',
                                 type=str,
                                 default=None,
                                 help='please input directory name'
                                      'which you want to copy directories')
    parser.add_argument('--target_dir',
                                 dest='target_dir',
                                 type=str,
                                 default=None,
                                 help='please input directory name'
                                      'which you want to join'
                                      'filename and dirname')
    argv = parser.parse_args()
    main(argv.original_dir, argv.target_dir)


