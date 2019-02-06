import os
import argparse


def parse_diff_txt(txt_file):
    """
    find files that have changed from diff command output.

    Input  : .txt file diff command output (diff -r <dir1> <dir2>)
    Output : set(filepath)
      filepath : Files that have changed
    """
    diff_files = set()
    with open(txt_file, 'r') as r:
        lines = r.readlines()
        
    for line in lines:
        print(line)
        line = line.strip()
        filepath = line.split(' ')
        filepath = (filepath[0], filepath[2])
        diff_files.add(filepath)

    return diff_files


def delete_xml(filepath, filenames, root_dir):
    """
    delete xml file linked png file that have changed
    
    Input: directory path that exists xml files
    Output: nothing
    """
    for f in filenames:
        print(f)
        if f[1].find('png') >= 0:
            target_file = os.path.join(filepath, f)
            target_file = target_file.replace('png', 'xml')
            if os.path.exists(target_file) is True:
                print('True')
                print('delete {}'.format(target_file))
                try:
                    os.remove(target_file)
                except FileNotFoundError:
                    print('file is already removed.')
                    pass
            else:
                print('False')
        elif f[1].find('png') < 0:
            target_dir = os.path.join(root_dir, f[0])
            target_dir = os.path.join(target_dir, f[1])
            print(target_dir)
            if os.path.isdir(target_dir) is False:
                print('target is not found')
            else:
                print('ok')
                target_files = os.listdir(target_dir)
                if len(target_files) == 0:
                    print('file is not found')
                    continue
                for t in target_files:
                    target_path = os.path.join(target_dir, t)
                    print(target_path)
                    if os.path.exists(target_path) is True:
                        target_path = target_path.replace('png', 'xml')
                        fname = os.path.basename(target_path)
                        target_xml = os.path.join(filepath, fname)
                        print('True')
                        print('delete {}'.format(target_xml))
                        try:
                            os.remove(target_xml)
                        except FileNotFoundError:
                            print('file is already removed')
                            pass
                    else:
                        print('False')

    return


def main(diff_file, xml_dir):
    parsed_fname = parse_diff_txt(diff_file)
    print(parsed_fname)
    print(len(parsed_fname))

    root_dir = os.path.dirname(diff_file)

    delete_xml(xml_dir, parsed_fname, root_dir)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parse diff annotation directory from txt file which generate diff command.')
    parser.add_argument(
        '--diff_file',
        dest='diff_file',
        type=str,
        default=None,
        help='please input text file path'
        ' which is genereated diff command.')
    parser.add_argument(
        '--xml_dir',
        dest='xml_dir',
        type=str,
        default=None,
        help='please input xml file directory'
        ' containing the xml file you need to modify.')
    argv = parser.parse_args()
    main(argv.diff_file, argv.xml_dir)
